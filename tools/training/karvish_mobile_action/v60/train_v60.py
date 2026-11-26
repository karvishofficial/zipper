"""
KARVISH UNIVERSAL V60 — MULTI-TASK ACTION & CALIBRATED TERMINATION TRAINER
Teacher: Karvish V59 (v59_best.pt)
Student: Karvish V60 (warm-started from V59)
Dataset: 50,000 Authoritative Multi-Turn Trajectories (v60_authoritative_dataset.pt)

Key Loss Formulations:
  1. Candidate Ranking Margin Loss (margin=3.0) + InfoNCE.
  2. Teacher Invariance Shield (KD KL-Div beta=2.0, tau=2.0 on historical replay items).
  3. Action Type & Dual-Slot Span Pointer Heads.
  4. 15-Class Focal-Weighted BIO Slot Sequence Loss.
  5. Calibrated Goal Termination Loss:
     - Focal loss (gamma=2.0) with a 5.0x false-positive penalty on near-miss states.
  6. Slip Stagnation & Modal Recovery Loss.
  7. Contrastive Goal Invariance Loss (Cosine alignment).
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
V59_DIR = SCRIPT_DIR.parent / "v59"
V58_DIR = SCRIPT_DIR.parent / "v58"
V57_DIR = SCRIPT_DIR.parent / "v57"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(V59_DIR))
sys.path.insert(0, str(V58_DIR))
sys.path.insert(0, str(V57_DIR))

from model_v59 import KarvishModelV59
from model_v60 import KarvishModelV60
from schema_v60 import (
    ACTION_TO_IDX,
    ROLE_TO_IDX,
    NUM_BIO_TAGS,
    BIO_TAG_TO_IDX,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

V59_TEACHER_CKPT = V59_DIR / "checkpoints" / "v59_best.pt"
V60_DATASET_FILE = SCRIPT_DIR / "checkpoints" / "v60_authoritative_dataset.pt"
V60_BEST_CKPT = SCRIPT_DIR / "checkpoints" / "v60_best.pt"


def normalize_bounds(raw_bounds: List[float]) -> List[float]:
    if len(raw_bounds) == 10:
        return raw_bounds
    l, t, r, b = raw_bounds[:4]
    if l > 1.5 or r > 1.5 or t > 1.5 or b > 1.5:
        l = l / 1080.0
        r = r / 1080.0
        t = t / 2400.0
        b = b / 2400.0
    l = min(max(l, 0.0), 1.0)
    t = min(max(t, 0.0), 1.0)
    r = min(max(r, 0.0), 1.0)
    b = min(max(b, 0.0), 1.0)
    cx = (l + r) / 2.0
    cy = (t + b) / 2.0
    w = max(0.001, r - l)
    h = max(0.001, b - t)
    area = w * h
    aspect = min((w / max(h, 0.001)) / 5.0, 2.0)
    return [l, t, r, b, cx, cy, w, h, area, aspect]


def collate_v60_batch(batch_items: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
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

            # Parity Role Extraction: 0 (display), 1 (button), 2 (editable)
            r = c.get("role_idx", None)
            if r is None:
                r_val = c.get("role", 1)
                if isinstance(r_val, int):
                    r = r_val
                elif isinstance(r_val, str):
                    r_str = r_val.lower()
                    if c.get("is_editable") or "input" in r_str or "edit" in r_str:
                        r = 2
                    elif c.get("is_clickable", True) or "button" in r_str or "tab" in r_str:
                        r = 1
                    else:
                        r = 0
                else:
                    r = 1
            cand_roles[b_idx, c_idx] = r

            if c.get("is_editable", False) or r == 2:
                cand_flags[b_idx, c_idx, 0] = 1.0
            if c.get("is_focused", False):
                cand_flags[b_idx, c_idx, 1] = 1.0
            if c.get("is_clickable", True) or r == 1:
                cand_flags[b_idx, c_idx, 2] = 1.0
            if c.get("is_selected", False):
                cand_flags[b_idx, c_idx, 5] = 1.0

            b_vec = c.get("bounds_vec", None)
            if b_vec is not None and len(b_vec) == 10:
                cand_bounds[b_idx, c_idx] = torch.tensor(b_vec, dtype=torch.float32)
            else:
                raw_b = c.get("bounds", [0, 0, 100, 100])
                cand_bounds[b_idx, c_idx] = torch.tensor(normalize_bounds(raw_b), dtype=torch.float32)

    has_diff_canon = torch.tensor([it["goal"] != it.get("canonical_goal", it["goal"]) for it in batch_items], dtype=torch.bool)
    is_terminal = torch.tensor([1.0 if it.get("is_final", False) else 0.0 for it in batch_items], dtype=torch.float32)
    is_near_miss = torch.tensor([it.get("is_near_miss", False) for it in batch_items], dtype=torch.bool)
    is_slip = torch.tensor([1.0 if it.get("is_slip_stagnation", False) else 0.0 for it in batch_items], dtype=torch.float32)

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
        "is_terminal": is_terminal,
        "is_near_miss": is_near_miss,
        "is_slip": is_slip,
        "is_replay": torch.tensor([it.get("is_replay", False) or str(it.get("screen_id", "")).startswith("v5") or str(it.get("screen_id", "")).startswith("gold_") for it in batch_items], dtype=torch.bool),
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


def train_v60(num_epochs: int = 10, batch_size: int = 64, lr: float = 3e-4):
    print("=" * 80)
    print("KARVISH V60: TRAINING UNIVERSAL ZERO-SHOT MOBILE AGENT")
    print(f"Device: {DEVICE} | Epochs: {num_epochs} | Batch Size: {batch_size} | LR: {lr}")
    print("=" * 80)

    # 1. Load Dataset
    if not V60_DATASET_FILE.exists():
        raise FileNotFoundError(f"Missing {V60_DATASET_FILE}! Run build_v60_50k_dataset.py first.")
    dataset = torch.load(V60_DATASET_FILE, weights_only=False)
    print(f">> Loaded V60 Dataset: {len(dataset):,} samples.")

    # 2. Load V59 Teacher Model
    print(f">> Initializing V59 Teacher from {V59_TEACHER_CKPT}...")
    teacher = KarvishModelV59().to(DEVICE)
    if V59_TEACHER_CKPT.exists():
        t_data = torch.load(V59_TEACHER_CKPT, map_location=DEVICE, weights_only=False)
        teacher.load_state_dict(t_data["model_state_dict"], strict=False)
        print(">> V59 Teacher weights loaded.")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # 3. Initialize Student Model (Warm-start from V60 best or V59)
    print(">> Initializing KarvishModelV60 Student...")
    student = KarvishModelV60().to(DEVICE)
    best_loss = float("inf")
    start_epoch = 1
    if V60_BEST_CKPT.exists():
        s_data = torch.load(V60_BEST_CKPT, map_location=DEVICE, weights_only=False)
        student.load_state_dict(s_data["model_state_dict"], strict=False)
        best_loss = s_data.get("loss", float("inf"))
        start_epoch = s_data.get("epoch", 1) + 1
        print(f">> Resumed student weights from {V60_BEST_CKPT} (Previous Best Loss: {best_loss:.4f})")
    elif V59_TEACHER_CKPT.exists():
        student.load_state_dict(t_data["model_state_dict"], strict=False)
        print(">> Warm-started student weights from V59.")

    optimizer = AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = (len(dataset) // batch_size) * num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

    # Class weighting for BIO slot tags (down-weight O to prevent class collapse)
    bio_weights = torch.ones(NUM_BIO_TAGS, device=DEVICE) * 2.0
    bio_weights[BIO_TAG_TO_IDX["O"]] = 0.25
    bio_loss_fn = nn.CrossEntropyLoss(weight=bio_weights)

    for epoch in range(start_epoch, start_epoch + num_epochs):
        student.train()
        random.shuffle(dataset)
        total_epoch_loss = 0.0
        total_rank_loss = 0.0
        total_term_loss = 0.0
        total_slot_loss = 0.0
        correct_cands = 0
        total_cands = 0
        false_term_count = 0
        total_neg_term = 0

        num_batches = len(dataset) // batch_size
        t0 = time.time()

        for step in range(num_batches):
            batch_slice = dataset[step * batch_size : (step + 1) * batch_size]
            batch = collate_v60_batch(batch_slice)

            # Move batch tensors to device
            g_ids = batch["goal_ids"].to(DEVICE)
            g_mask = batch["goal_mask"].to(DEVICE)
            B = g_ids.size(0)
            c_lbl_ids = batch["candidate_label_ids"].to(DEVICE)
            c_lbl_mask = batch["candidate_label_mask"].to(DEVICE)
            c_bounds = batch["candidate_bounds"].to(DEVICE)
            c_roles = batch["candidate_roles"].to(DEVICE)
            c_flags = batch["candidate_flags"].to(DEVICE)
            c_pad_mask = batch["candidate_pad_mask"].to(DEVICE)
            c_num_feats = batch["candidate_num_features"].to(DEVICE)
            tgt_idx = batch["target_indices"].to(DEVICE)
            tgt_act = batch["target_actions"].to(DEVICE)
            bio_lbls = batch["bio_slot_labels"].to(DEVICE)
            sp_st = batch["span_starts"].to(DEVICE)
            sp_en = batch["span_ends"].to(DEVICE)
            evid = batch["evidence_observed"].to(DEVICE)
            is_term = batch["is_terminal"].to(DEVICE)
            is_near_miss = batch["is_near_miss"].to(DEVICE)
            is_slip = batch["is_slip"].to(DEVICE)
            is_replay = batch["is_replay"].to(DEVICE)
            has_diff = batch["has_diff_canon"].to(DEVICE)

            optimizer.zero_grad()

            # Forward Student
            out_s = student(
                goal_ids=g_ids,
                goal_mask=g_mask,
                candidate_label_ids=c_lbl_ids,
                candidate_label_mask=c_lbl_mask,
                candidate_bounds=c_bounds,
                candidate_roles=c_roles,
                candidate_flags=c_flags,
                candidate_pad_mask=c_pad_mask,
                candidate_num_features=c_num_feats,
                target_candidate_idx=tgt_idx,
                evidence_observed=evid,
            )

            # 1. Candidate Ranking Loss
            rank_scores = out_s["ranking_scores"]
            loss_rank_ce = F.cross_entropy(rank_scores, tgt_idx)
            loss_rank_margin = compute_margin_loss(rank_scores, tgt_idx, c_pad_mask, margin=3.0)
            loss_rank = loss_rank_ce + 0.5 * loss_rank_margin

            # 2. Action Type Loss
            loss_act = F.cross_entropy(out_s["action_logits"], tgt_act)

            # 3. Dual Span Pointer Loss
            loss_span = F.cross_entropy(out_s["span_start_logits"], sp_st) + F.cross_entropy(out_s["span_end_logits"], sp_en)

            # 4. BIO Slot Classification Loss
            bio_logits = out_s["bio_slot_logits"].view(-1, NUM_BIO_TAGS)
            bio_flat_lbls = bio_lbls.view(-1)
            loss_bio = bio_loss_fn(bio_logits, bio_flat_lbls)

            # 5. Calibrated Goal Termination Loss with 5.0x False-Positive Penalty
            term_logit = out_s["calibrated_term_logit"]
            bce_term = F.binary_cross_entropy_with_logits(term_logit, is_term, reduction="none")
            pt_term = torch.exp(-bce_term)
            focal_term = ((1.0 - pt_term) ** 2.0) * bce_term
            # Apply 5.0x multiplier when goal is NOT terminal (especially near-misses)
            penalty_weights = torch.where(is_term == 0.0, 5.0, 1.0)
            loss_term = (focal_term * penalty_weights).mean()

            # 6. Slip Recovery Loss
            slip_logit = out_s["slip_logit"]
            loss_slip = F.binary_cross_entropy_with_logits(slip_logit, is_slip)

            # 7. Teacher Knowledge Distillation on historical baseline replays
            with torch.no_grad():
                out_t = teacher(
                    goal_ids=g_ids,
                    goal_mask=g_mask,
                    candidate_label_ids=c_lbl_ids,
                    candidate_label_mask=c_lbl_mask,
                    candidate_bounds=c_bounds,
                    candidate_roles=c_roles,
                    candidate_flags=c_flags,
                    candidate_pad_mask=c_pad_mask,
                    candidate_num_features=c_num_feats,
                    target_candidate_idx=tgt_idx,
                    evidence_observed=evid,
                )
            tau = 2.0
            log_p_s = F.log_softmax(rank_scores / tau, dim=-1)
            p_t = F.softmax(out_t["ranking_scores"] / tau, dim=-1)
            kl_elem = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=-1) * (tau ** 2)
            is_rep = batch["is_replay"].to(DEVICE)
            kd_weights = torch.where(is_rep, 3.0, 0.2)
            loss_kd = (kl_elem * kd_weights).mean()

            # 8. Contrastive Goal Invariance Loss
            if has_diff.any():
                canon_ids = batch["canon_ids"].to(DEVICE)[has_diff]
                canon_mask = batch["canon_mask"].to(DEVICE)[has_diff]
                with torch.no_grad():
                    canon_out = student.text_encoder(canon_ids, attention_mask=canon_mask)
                    u_canon = canon_out["pooled_output"]
                u_noisy = out_s["u_active"][has_diff]
                sim = F.cosine_similarity(u_noisy, u_canon, dim=-1)
                loss_inv = (1.0 - sim).mean()
            else:
                loss_inv = torch.tensor(0.0, device=DEVICE)

            # Total Loss
            loss = (
                loss_rank +
                1.5 * loss_act +
                0.8 * loss_span +
                1.2 * loss_bio +
                2.5 * loss_term +
                1.0 * loss_slip +
                2.0 * loss_kd +
                0.8 * loss_inv
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Metrics
            total_epoch_loss += loss.item()
            total_rank_loss += loss_rank.item()
            total_term_loss += loss_term.item()
            total_slot_loss += loss_bio.item()

            preds = rank_scores.argmax(dim=-1)
            correct_cands += (preds == tgt_idx).sum().item()
            total_cands += B

            # Track False Complete Rate
            pred_term = (out_s["termination_prob"] > 0.5)
            false_term_count += ((pred_term == 1) & (is_term == 0)).sum().item()
            total_neg_term += (is_term == 0).sum().item()

            if step > 0 and step % 150 == 0:
                print(f"  [Epoch {epoch:02d} | Step {step:03d}/{num_batches}] Loss: {loss.item():.4f} | Rank: {loss_rank.item():.4f} | Term: {loss_term.item():.4f}", flush=True)

        elapsed = time.time() - t0
        avg_loss = total_epoch_loss / num_batches
        rank_acc = (correct_cands / total_cands) * 100.0
        false_term_rate = (false_term_count / max(1, total_neg_term)) * 100.0

        print(
            f"Epoch {epoch:02d}/{num_epochs:02d} [{elapsed:.1f}s] | "
            f"Loss: {avg_loss:.4f} | Rank Acc: {rank_acc:.2f}% | "
            f"Slot Loss: {total_slot_loss / num_batches:.4f} | "
            f"Term Loss: {total_term_loss / num_batches:.4f} | "
            f"False Complete Rate: {false_term_rate:.2f}%",
            flush=True
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            V60_BEST_CKPT.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "rank_acc": rank_acc,
                "false_term_rate": false_term_rate,
            }, V60_BEST_CKPT)
            print(f"  >> Saved new best checkpoint to {V60_BEST_CKPT} (Loss: {best_loss:.4f})", flush=True)

    print("=" * 80, flush=True)
    print(f"V60 TRAINING COMPLETED. Best Checkpoint: {V60_BEST_CKPT} (Loss: {best_loss:.4f})", flush=True)
    print("=" * 80, flush=True)
    return student


if __name__ == "__main__":
    train_v60(num_epochs=3, batch_size=64, lr=1.5e-4)
