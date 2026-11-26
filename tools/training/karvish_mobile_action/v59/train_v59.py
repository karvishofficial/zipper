"""
KARVISH UNIVERSAL V59 — UNIVERSAL NLU & ACTION POLICY TRAINER
Teacher: Karvish V58 (v58_best.pt)
Student: Karvish V59 (warm-started from V58)
Dataset: 5,019 Authoritative Trajectories (v59_authoritative_dataset.pt)

Multi-Task Loss Objectives:
  1. Action Selection & Candidate Ranking (CE + Margin loss).
  2. Invariance Shield (KD KL-Div beta=2.0 on V58 replay items).
  3. Action Type & Dual-Slot Span Pointer Heads.
  4. Evidence-Gated Calibration (P(complete | evidence=0) < 0.01).
  5. Token-level BIO Slot Classification (15 slot tags, masked CE loss).
  6. Contrastive Goal Invariance Loss (Cosine alignment between noisy & canonical goals).
"""

from __future__ import annotations

import copy
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
V58_DIR = SCRIPT_DIR.parent / "v58"
V57_DIR = SCRIPT_DIR.parent / "v57"
V56_DIR = SCRIPT_DIR.parent / "v56"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(V58_DIR))
sys.path.insert(0, str(V57_DIR))
sys.path.insert(0, str(V56_DIR))

from model_v58 import KarvishModelV58
from model_v59 import KarvishModelV59
from schema_v59 import ACTION_TO_IDX, ROLE_TO_IDX, NUM_BIO_TAGS, BIO_TAG_TO_IDX

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

V58_TEACHER_CKPT = V58_DIR / "checkpoints" / "v58_best.pt"
V59_DATASET_FILE = SCRIPT_DIR / "checkpoints" / "v59_authoritative_dataset.pt"
V59_BEST_CKPT = SCRIPT_DIR / "checkpoints" / "v59_best.pt"


def collate_v59_batch(batch_items: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    B = len(batch_items)
    goals = [it["goal"] for it in batch_items]
    g_enc = TOKENIZER(goals, max_length=32, padding="max_length", truncation=True, return_tensors="pt")

    canon_goals = [it.get("canonical_goal", it["goal"]) for it in batch_items]
    canon_enc = TOKENIZER(canon_goals, max_length=32, padding="max_length", truncation=True, return_tensors="pt")

    max_cands = max(len(it["candidates"]) for it in batch_items)
    cand_label_ids = torch.zeros(B, max_cands, 16, dtype=torch.long)
    cand_label_mask = torch.zeros(B, max_cands, 16, dtype=torch.float32)
    cand_roles = torch.zeros(B, max_cands, dtype=torch.long)
    cand_flags = torch.zeros(B, max_cands, 16, dtype=torch.float32)
    cand_pad_mask = torch.zeros(B, max_cands, dtype=torch.float32)
    cand_bounds = torch.zeros(B, max_cands, 10, dtype=torch.float32)
    cand_num_features = torch.zeros(B, max_cands, 4, dtype=torch.float32)

    target_indices = torch.tensor([it["target_idx"] for it in batch_items], dtype=torch.long)
    target_actions = torch.tensor([it.get("action_idx", 0) for it in batch_items], dtype=torch.long)

    bio_slot_labels = torch.zeros(B, 32, dtype=torch.long)
    span_starts = []
    span_ends = []
    slot_indices = []

    for b_idx, it in enumerate(batch_items):
        sp = it.get("target_span", (0, 0))
        span_starts.append(sp[0])
        span_ends.append(sp[1])
        slot_indices.append(it.get("slot_type", 0))

        # BIO tags
        raw_bio = it.get("bio_slot_tags", [])
        for tok_i in range(min(32, len(raw_bio))):
            bio_slot_labels[b_idx, tok_i] = raw_bio[tok_i]

        cands = it["candidates"]
        for c_idx, c in enumerate(cands):
            cand_pad_mask[b_idx, c_idx] = 1.0
            lbl = c.get("label", "")
            lbl_enc = TOKENIZER(lbl, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
            cand_label_ids[b_idx, c_idx] = lbl_enc["input_ids"][0]
            cand_label_mask[b_idx, c_idx] = lbl_enc["attention_mask"][0].float()

            r = c.get("role", "BUTTON")
            if isinstance(r, int):
                cand_roles[b_idx, c_idx] = r
            elif isinstance(r, str):
                r_upper = r.upper().strip()
                cand_roles[b_idx, c_idx] = ROLE_TO_IDX.get(r_upper, 1)
            else:
                cand_roles[b_idx, c_idx] = 1

            # Flags
            if c.get("is_editable", False):
                cand_flags[b_idx, c_idx, 0] = 1.0
            if c.get("is_focused", False):
                cand_flags[b_idx, c_idx, 1] = 1.0
            if c.get("is_clickable", True):
                cand_flags[b_idx, c_idx, 2] = 1.0
            if c.get("is_selected", False):
                cand_flags[b_idx, c_idx, 5] = 1.0

            # 2D Bounds
            b = c.get("bounds", [0, 0, 100, 100])
            cand_bounds[b_idx, c_idx, :4] = torch.tensor(b, dtype=torch.float32)

    has_diff_canon = torch.tensor([it["goal"] != it.get("canonical_goal", it["goal"]) for it in batch_items], dtype=torch.bool)

    return {
        "goal_ids": g_enc["input_ids"],
        "goal_mask": g_enc["attention_mask"].float(),
        "canon_ids": canon_enc["input_ids"],
        "canon_mask": canon_enc["attention_mask"].float(),
        "candidate_label_ids": cand_label_ids,
        "candidate_label_mask": cand_label_mask,
        "candidate_roles": cand_roles,
        "candidate_flags": cand_flags,
        "candidate_pad_mask": cand_pad_mask,
        "candidate_bounds": cand_bounds,
        "candidate_num_features": cand_num_features,
        "target_indices": target_indices,
        "target_actions": target_actions,
        "bio_slot_labels": bio_slot_labels,
        "span_starts": torch.tensor(span_starts, dtype=torch.long),
        "span_ends": torch.tensor(span_ends, dtype=torch.long),
        "slot_indices": torch.tensor(slot_indices, dtype=torch.long),
        "evidence_observed": torch.tensor([it.get("verification_evidence", 0.0) for it in batch_items], dtype=torch.float32),
        "is_replay": torch.tensor([not str(it.get("dataset_class", "")).startswith("v59_") for it in batch_items], dtype=torch.bool),
        "has_diff_canon": has_diff_canon,
    }


def compute_margin_loss(scores: torch.Tensor, target_idx: torch.Tensor, pad_mask: torch.Tensor, margin: float = 3.0) -> torch.Tensor:
    gold_scores = scores.gather(1, target_idx.unsqueeze(1))
    diffs = scores - gold_scores + margin
    diffs = diffs * pad_mask
    mask_gold = torch.ones_like(scores, dtype=torch.bool)
    mask_gold.scatter_(1, target_idx.unsqueeze(1), False)
    hard_diffs = F.relu(diffs) * mask_gold.float()
    return hard_diffs.sum() / (mask_gold.float() * pad_mask).sum().clamp(min=1.0)


def train_v59():
    print("=" * 80)
    print("KARVISH V59: TRAINING UNIVERSAL NLU & ACTION POLICY")
    print(f"Device: {DEVICE}")
    print("=" * 80)

    # 1. Load Dataset
    if not V59_DATASET_FILE.exists():
        print(f">> Dataset file not found at {V59_DATASET_FILE}. Generating now...")
        from build_v59_dataset import generate_v59_authoritative_dataset
        dataset = generate_v59_authoritative_dataset()
        V59_DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dataset, V59_DATASET_FILE)
    else:
        dataset = torch.load(V59_DATASET_FILE, weights_only=False)
        print(f">> Loaded V59 dataset: {len(dataset)} items from {V59_DATASET_FILE}")

    # 2. Load V58 Teacher Model (Frozen Invariance Shield)
    print(f">> Initializing V58 Teacher from {V58_TEACHER_CKPT}...")
    teacher = KarvishModelV58().to(DEVICE)
    if V58_TEACHER_CKPT.exists():
        t_data = torch.load(V58_TEACHER_CKPT, map_location=DEVICE, weights_only=False)
        teacher.load_state_dict(t_data["model_state_dict"], strict=False)
        print(">> V58 Teacher weights successfully loaded.")
    else:
        print(">> WARNING: V58 teacher checkpoint not found. Training without KD.")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # 3. Initialize V59 Student (Warm-started from V58)
    student = KarvishModelV59().to(DEVICE)
    if V58_TEACHER_CKPT.exists():
        s_data = torch.load(V58_TEACHER_CKPT, map_location=DEVICE, weights_only=False)
        student.load_state_dict(s_data["model_state_dict"], strict=False)
        print(">> V59 Student warm-started from V58 Teacher (base weights preserved, bio_slot_head initialized).")

    # 4. Training Hyperparameters
    # Give higher LR to bio_slot_head for fast convergence on slot tags
    base_params = [p for n, p in student.named_parameters() if "bio_slot_head" not in n]
    head_params = [p for n, p in student.named_parameters() if "bio_slot_head" in n]

    optimizer = AdamW([
        {"params": base_params, "lr": 5e-6},
        {"params": head_params, "lr": 2e-4},
    ], weight_decay=0.01)

    batch_size = 16
    epochs = 5
    TEMP = 2.0

    print(f">> Commencing V59 Multi-Task Fine-Tuning ({epochs} epochs, batch_size={batch_size})...")
    student.train()

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        random.shuffle(dataset)
        total_loss = 0.0
        total_bio_loss = 0.0
        total_inv_loss = 0.0
        num_batches = 0

        for b_start in range(0, len(dataset), batch_size):
            batch = dataset[b_start : b_start + batch_size]
            t = collate_v59_batch(batch)
            t = {k: v.to(DEVICE) for k, v in t.items()}

            optimizer.zero_grad()

            with torch.no_grad():
                t_out = teacher(
                    goal_ids=t["goal_ids"],
                    goal_mask=t["goal_mask"],
                    candidate_label_ids=t["candidate_label_ids"],
                    candidate_label_mask=t["candidate_label_mask"],
                    candidate_bounds=t["candidate_bounds"],
                    candidate_roles=t["candidate_roles"],
                    candidate_flags=t["candidate_flags"],
                    candidate_pad_mask=t["candidate_pad_mask"],
                    candidate_num_features=t["candidate_num_features"],
                    slot_idx=t["slot_indices"],
                    evidence_observed=t["evidence_observed"],
                )

            s_out = student(
                goal_ids=t["goal_ids"],
                goal_mask=t["goal_mask"],
                candidate_label_ids=t["candidate_label_ids"],
                candidate_label_mask=t["candidate_label_mask"],
                candidate_bounds=t["candidate_bounds"],
                candidate_roles=t["candidate_roles"],
                candidate_flags=t["candidate_flags"],
                candidate_pad_mask=t["candidate_pad_mask"],
                candidate_num_features=t["candidate_num_features"],
                slot_idx=t["slot_indices"],
                evidence_observed=t["evidence_observed"],
            )

            # 1. Ranking Loss on Gold Target
            loss_ce = F.cross_entropy(s_out["ranking_scores"], t["target_indices"])
            loss_margin = compute_margin_loss(
                s_out["ranking_scores"],
                t["target_indices"],
                t["candidate_pad_mask"],
                margin=3.0,
            )

            # 2. Knowledge Distillation Loss on Replay Items (Preserves V58/V57/V52 Anchors)
            if t["is_replay"].any():
                s_log = F.log_softmax(s_out["ranking_scores"][t["is_replay"]] / TEMP, dim=-1)
                t_p = F.softmax(t_out["ranking_scores"][t["is_replay"]] / TEMP, dim=-1)
                kd_term = F.kl_div(s_log, t_p, reduction="batchmean") * (TEMP * TEMP)
            else:
                kd_term = torch.tensor(0.0, device=DEVICE)

            # 3. Action Type Cross-Entropy Loss
            action_loss = F.cross_entropy(s_out["action_logits"], t["target_actions"])

            # 4. Span Loss for typing actions
            is_type = (t["target_actions"] == ACTION_TO_IDX["TYPE_TEXT"])
            if is_type.any():
                loss_st = F.cross_entropy(s_out["span_start_logits"][is_type], t["span_starts"][is_type])
                loss_en = F.cross_entropy(s_out["span_end_logits"][is_type], t["span_ends"][is_type])
                loss_span = (loss_st + loss_en) * 0.5
            else:
                loss_span = torch.tensor(0.0, device=DEVICE)

            # 5. Evidence-Gated Completion Calibration Loss
            loss_ev = F.binary_cross_entropy(s_out["termination_prob"].view(-1), t["evidence_observed"].view(-1))

            # 6. Token-Level BIO Slot Extraction Loss (Supervised on NLU annotated trajectories)
            has_nlu = (~t["is_replay"])
            if has_nlu.any():
                nlu_bio_logits = s_out["bio_slot_logits"][has_nlu] # [B_nlu, 32, NUM_BIO_TAGS]
                nlu_labels = t["bio_slot_labels"][has_nlu]         # [B_nlu, 32]
                nlu_mask = (t["goal_mask"][has_nlu] > 0)           # [B_nlu, 32]

                flat_logits = nlu_bio_logits.reshape(-1, NUM_BIO_TAGS)
                flat_labels = nlu_labels.reshape(-1)
                flat_m = nlu_mask.reshape(-1)

                class_weights = torch.ones(NUM_BIO_TAGS, device=DEVICE) * 2.0
                class_weights[0] = 0.25 # Down-weight background O tokens

                loss_bio = F.cross_entropy(flat_logits[flat_m], flat_labels[flat_m], weight=class_weights)
            else:
                loss_bio = torch.tensor(0.0, device=DEVICE)

            # 7. Contrastive Goal Invariance Loss (Aligns noisy slang representation with canonical anchor)
            if t["has_diff_canon"].any():
                canon_sub_ids = t["canon_ids"][t["has_diff_canon"]]
                canon_sub_mask = t["canon_mask"][t["has_diff_canon"]]
                canon_goal_out = student.text_encoder(canon_sub_ids, attention_mask=canon_sub_mask)
                canon_u = canon_goal_out["pooled_output"] # [B_sub, hidden_dim]

                noisy_u = s_out["u_active"][t["has_diff_canon"]] # [B_sub, hidden_dim]
                cos_sim = F.cosine_similarity(noisy_u, canon_u, dim=-1) # [B_sub]
                loss_inv = (1.0 - cos_sim).mean()
            else:
                loss_inv = torch.tensor(0.0, device=DEVICE)

            # Total Multi-Task Loss
            loss = (
                0.6 * loss_ce
                + 2.0 * kd_term
                + 0.5 * loss_margin
                + 0.5 * action_loss
                + 0.4 * loss_span
                + 0.3 * loss_ev
                + 0.6 * loss_bio
                + 0.6 * loss_inv
            )

            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_bio_loss += loss_bio.item()
            total_inv_loss += loss_inv.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        avg_bio = total_bio_loss / max(1, num_batches)
        avg_inv = total_inv_loss / max(1, num_batches)
        print(f"Epoch {epoch}/{epochs} | Avg Loss: {avg_loss:.4f} | Slot CE: {avg_bio:.4f} | Invariance: {avg_inv:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            V59_BEST_CKPT.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "architecture": "KarvishModelV59",
                "dataset_version": "V59",
                "num_bio_tags": NUM_BIO_TAGS,
            }, V59_BEST_CKPT)
            print(f">> Saved new best V59 checkpoint to {V59_BEST_CKPT}")

    print("=" * 80)
    print("V59 TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Authoritative Artifact: {V59_BEST_CKPT}")
    print("=" * 80)


if __name__ == "__main__":
    train_v59()
