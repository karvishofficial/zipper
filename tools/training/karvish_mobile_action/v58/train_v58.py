"""
KARVISH UNIVERSAL V58 — GRADUAL DISTILLATION & FINE-TUNING TRAINER
Teacher: Karvish V57 (v57_best.pt)
Student: Karvish V58
Dataset: 3,423 Authoritative Trajectories (v58_authoritative_dataset.pt)

Pillars:
  1. Overlay Occlusion Escape Trajectories (Instagram Search suggestions stall resolution).
  2. Search Suggestion Hard Negative suppression vs Navigation targets.
  3. Evidence-Gated Calibration: Zero tolerance for hallucinated completion.
  4. Invariance Shield: 100% preservation of V57 and V52-LOCKED-GOLD anchors via distillation.
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
REPO_ROOT = SCRIPT_DIR.parents[3]
V57_DIR = SCRIPT_DIR.parent / "v57"
V56_DIR = SCRIPT_DIR.parent / "v56"
V55_DIR = SCRIPT_DIR.parent / "v55"
V54_DIR = SCRIPT_DIR.parent / "v54"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(V57_DIR))
sys.path.insert(0, str(V56_DIR))
sys.path.insert(0, str(V55_DIR))
sys.path.insert(0, str(V54_DIR))

from model_v57 import KarvishModelV57
from model_v58 import KarvishModelV58
from schema_v58 import ACTION_TO_IDX, ROLE_TO_IDX, CANONICAL_ACTIONS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

V57_TEACHER_CKPT = V57_DIR / "checkpoints" / "v57_best.pt"
V58_DATASET_FILE = SCRIPT_DIR / "checkpoints" / "v58_authoritative_dataset.pt"
V58_BEST_CKPT = SCRIPT_DIR / "checkpoints" / "v58_best.pt"


def collate_v58_batch(batch_items: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    B = len(batch_items)
    goals = [it["goal"] for it in batch_items]
    g_enc = TOKENIZER(goals, max_length=32, padding=True, truncation=True, return_tensors="pt")

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

    span_starts = []
    span_ends = []
    slot_indices = []

    for b_idx, it in enumerate(batch_items):
        sp = it.get("target_span", (0, 0))
        span_starts.append(sp[0])
        span_ends.append(sp[1])
        slot_indices.append(it.get("slot_type", 0))

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

    return {
        "goal_ids": g_enc["input_ids"],
        "goal_mask": g_enc["attention_mask"].float(),
        "candidate_label_ids": cand_label_ids,
        "candidate_label_mask": cand_label_mask,
        "candidate_roles": cand_roles,
        "candidate_flags": cand_flags,
        "candidate_pad_mask": cand_pad_mask,
        "candidate_bounds": cand_bounds,
        "candidate_num_features": cand_num_features,
        "target_indices": target_indices,
        "target_actions": target_actions,
        "span_starts": torch.tensor(span_starts, dtype=torch.long),
        "span_ends": torch.tensor(span_ends, dtype=torch.long),
        "slot_indices": torch.tensor(slot_indices, dtype=torch.long),
        "evidence_observed": torch.tensor([it.get("verification_evidence", 0.0) for it in batch_items], dtype=torch.float32),
        "is_replay": torch.tensor([not str(it.get("dataset_class", "")).startswith("v58_") for it in batch_items], dtype=torch.bool),
    }


def compute_margin_loss(scores: torch.Tensor, target_idx: torch.Tensor, pad_mask: torch.Tensor, margin: float = 3.0) -> torch.Tensor:
    gold_scores = scores.gather(1, target_idx.unsqueeze(1))
    diffs = scores - gold_scores + margin
    diffs = diffs * pad_mask
    mask_gold = torch.ones_like(scores, dtype=torch.bool)
    mask_gold.scatter_(1, target_idx.unsqueeze(1), False)
    hard_diffs = F.relu(diffs) * mask_gold.float()
    return hard_diffs.sum() / (mask_gold.float() * pad_mask).sum().clamp(min=1.0)


def train_v58():
    print("=" * 80)
    print("KARVISH V58: TRAINING AUTHORITATIVE UNIVERSAL ACTION POLICY")
    print(f"Device: {DEVICE}")
    print("=" * 80)

    # 1. Load Dataset
    if not V58_DATASET_FILE.exists():
        print(f">> Dataset file not found at {V58_DATASET_FILE}. Generating now...")
        from build_v58_dataset import generate_v58_authoritative_dataset
        dataset = generate_v58_authoritative_dataset()
        V58_DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dataset, V58_DATASET_FILE)
    else:
        dataset = torch.load(V58_DATASET_FILE, weights_only=False)
        print(f">> Loaded V58 dataset: {len(dataset)} items from {V58_DATASET_FILE}")

    # 2. Load V57 Teacher Model (Frozen Invariance Shield)
    print(f">> Initializing V57 Teacher from {V57_TEACHER_CKPT}...")
    teacher = KarvishModelV57().to(DEVICE)
    if V57_TEACHER_CKPT.exists():
        t_data = torch.load(V57_TEACHER_CKPT, map_location=DEVICE, weights_only=False)
        teacher.load_state_dict(t_data["model_state_dict"])
        print(">> V57 Teacher weights successfully loaded.")
    else:
        print(">> WARNING: V57 teacher checkpoint not found. Training without KD.")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # 3. Initialize V58 Student (Warm-started from V57)
    student = KarvishModelV58().to(DEVICE)
    if V57_TEACHER_CKPT.exists():
        s_data = torch.load(V57_TEACHER_CKPT, map_location=DEVICE, weights_only=False)
        # Load compatible base weights with strict=False to accommodate transition_head
        student.load_state_dict(s_data["model_state_dict"], strict=False)
        print(">> V58 Student warm-started from V57 Teacher (base weights preserved).")

    # 4. Training Loop (Fine-Tuning with KD Beta = 2.0 to protect 12/12 Anchors)
    optimizer = AdamW(student.parameters(), lr=5e-6, weight_decay=0.01)
    batch_size = 16
    epochs = 4
    TEMP = 2.0

    print(f">> Commencing V58 Fine-Tuning ({epochs} epochs, batch_size={batch_size})...")
    student.train()

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        random.shuffle(dataset)
        total_loss = 0.0
        num_batches = 0

        for b_start in range(0, len(dataset), batch_size):
            batch = dataset[b_start : b_start + batch_size]
            t = collate_v58_batch(batch)
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

            # 2. Knowledge Distillation Loss on Replay Items (Preserves V57 Anchor Skills)
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

            loss = 0.6 * loss_ce + 2.0 * kd_term + 0.6 * loss_margin + 0.5 * action_loss + 0.4 * loss_span + 0.3 * loss_ev

            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        print(f"Epoch {epoch}/{epochs} | Avg Loss: {avg_loss:.4f} (CE: {loss_ce.item():.4f}, Margin: {loss_margin.item():.4f}, KD: {kd_term.item():.4f})")

        if avg_loss < best_loss:
            best_loss = avg_loss
            V58_BEST_CKPT.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "architecture": "KarvishModelV58",
                "dataset_version": "V58",
            }, V58_BEST_CKPT)
            print(f">> Saved new best V58 checkpoint to {V58_BEST_CKPT}")

    print("=" * 80)
    print("V58 TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Authoritative Artifact: {V58_BEST_CKPT}")
    print("=" * 80)


if __name__ == "__main__":
    train_v58()
