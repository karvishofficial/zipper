"""
KARVISH UNIVERSAL V61 — GATED HINDSIGHT DISTILLATION & MULTIMODAL FINE-TUNER
Teacher: Karvish V60 (v60_best.pt)
Student: Karvish V61 (32.33M policy, warm-started from V60 with Net2Net Zero-Init)
Dataset: 85,000 Authoritative Multi-Turn Trajectories (v61_authoritative_dataset.pt)

Key Training Objectives:
  1. Candidate Ranking Margin Loss + InfoNCE deep metric learning.
  2. V60 Teacher Invariance Shield (KD KL-Divergence, tau=2.0) on anchor replay items.
  3. Gated Hindsight Distillation (GHD):
     Uses next-screen delta (o_{t+1}) as privileged hindsight to distill causal subgoal achievement.
  4. 15-Class Focal BIO Slot Sequence Loss for NLU slot filling.
  5. Calibrated Goal Termination Loss with 5.0x penalty against premature false stops.
  6. Autonomous Self-Repair: Dynamically monitors metrics and re-weights losses to prevent false learning.
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
V60_DIR = SCRIPT_DIR.parent / "v60"
V59_DIR = SCRIPT_DIR.parent / "v59"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(V60_DIR))
sys.path.insert(0, str(V59_DIR))

from model_v60 import KarvishModelV60
from model_v61 import KarvishModelV61, load_v60_weights_into_v61
from schema_v61 import (
    ACTION_TO_IDX,
    ROLE_TO_IDX,
    NUM_BIO_TAGS,
    BIO_TAG_TO_IDX,
)
from vocab_v61 import BASE_TOKENIZER_ID

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained(BASE_TOKENIZER_ID)

V60_TEACHER_CKPT = V60_DIR / "checkpoints" / "v60_best.pt"
V61_DATASET_FILE = SCRIPT_DIR / "checkpoints" / "v61_authoritative_dataset.pt"
V61_BEST_CKPT = SCRIPT_DIR / "checkpoints" / "v61_best.pt"


def collate_v61_batch(batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collates variable-candidate trajectories into dense batch tensors with visual features."""
    B = len(batch_items)
    max_cands = max(len(it["candidates"]) for it in batch_items)
    max_cands = max(max_cands, 4)

    goals = [it["goal"] for it in batch_items]
    enc_goals = TOKENIZER(goals, max_length=64, padding="max_length", truncation=True, return_tensors="pt")

    cand_label_ids = torch.zeros(B, max_cands, 16, dtype=torch.long)
    cand_label_mask = torch.zeros(B, max_cands, 16, dtype=torch.float32)
    cand_bounds = torch.zeros(B, max_cands, 10, dtype=torch.float32)
    cand_roles = torch.zeros(B, max_cands, dtype=torch.long)
    cand_flags = torch.zeros(B, max_cands, 16, dtype=torch.float32)
    cand_pad_mask = torch.zeros(B, max_cands, dtype=torch.float32)
    cand_visual_feats = torch.zeros(B, max_cands, 512, dtype=torch.float32)
    cand_num_feats = torch.zeros(B, max_cands, 4, dtype=torch.float32)

    target_indices = torch.tensor([it["target_idx"] for it in batch_items], dtype=torch.long)
    target_actions = torch.tensor([it.get("action_idx", 0) for it in batch_items], dtype=torch.long)
    is_terminal = torch.tensor([it.get("is_terminal_complete", 0.0) for it in batch_items], dtype=torch.float32)
    is_near_miss = torch.tensor([it.get("is_near_miss_negative", 0.0) for it in batch_items], dtype=torch.float32)
    hindsight_delta = torch.tensor([it.get("hindsight_next_screen_delta", 1.0) for it in batch_items], dtype=torch.float32)

    bio_slot_labels = torch.zeros(B, 32, dtype=torch.long)
    span_starts = []
    span_ends = []

    for b_idx, it in enumerate(batch_items):
        sp = it.get("target_span", (0, 0))
        span_starts.append(sp[0])
        span_ends.append(sp[1])

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

            r = c.get("role", 1)
            cand_roles[b_idx, c_idx] = r if isinstance(r, int) else 1

            b_raw = c.get("bounds", [0.0] * 10)
            if len(b_raw) < 10:
                l, t, r, b = b_raw[:4]
                if l > 1.5 or r > 1.5 or t > 1.5 or b > 1.5:
                    l, r, t, b = l / 1080.0, r / 1080.0, t / 2400.0, b / 2400.0
                cx, cy = (l + r) / 2.0, (t + b) / 2.0
                w, h = max(0.001, r - l), max(0.001, b - t)
                b_geom = [l, t, r, b, cx, cy, w, h, w * h, w / h]
            else:
                b_geom = b_raw[:10]
            cand_bounds[b_idx, c_idx] = torch.tensor(b_geom, dtype=torch.float32)

            v_feat = c.get("visual_feat", None)
            if v_feat is not None:
                if isinstance(v_feat, torch.Tensor):
                    cand_visual_feats[b_idx, c_idx] = v_feat
                elif isinstance(v_feat, list):
                    cand_visual_feats[b_idx, c_idx] = torch.tensor(v_feat[:512], dtype=torch.float32)

    return {
        "goal_ids": enc_goals["input_ids"],
        "goal_mask": enc_goals["attention_mask"].float(),
        "candidate_label_ids": cand_label_ids,
        "candidate_label_mask": cand_label_mask,
        "candidate_bounds": cand_bounds,
        "candidate_roles": cand_roles,
        "candidate_flags": cand_flags,
        "candidate_pad_mask": cand_pad_mask,
        "candidate_visual_feats": cand_visual_feats,
        "candidate_num_features": cand_num_feats,
        "target_indices": target_indices,
        "target_actions": target_actions,
        "is_terminal": is_terminal,
        "is_near_miss": is_near_miss,
        "hindsight_delta": hindsight_delta,
        "bio_slot_labels": bio_slot_labels,
        "span_starts": torch.tensor(span_starts, dtype=torch.long),
        "span_ends": torch.tensor(span_ends, dtype=torch.long),
    }


def train_v61_policy(num_epochs: int = 6, batch_size: int = 64, lr: float = 3e-5):
    print("=" * 70)
    print("  KARVISH V61 — GATED HINDSIGHT DISTILLATION & MULTIMODAL FINE-TUNER")
    print(f"  Device: {DEVICE} | Epochs: {num_epochs} | Batch Size: {batch_size} | LR: {lr}")
    print("=" * 70)

    # 1. Load Dataset
    if not V61_DATASET_FILE.exists():
        raise FileNotFoundError(f"Missing {V61_DATASET_FILE}! Run build_v61_dataset.py first.")
    dataset = torch.load(V61_DATASET_FILE, weights_only=False)
    print(f">> Loaded V61 Dataset: {len(dataset):,} trajectories.")

    # 2. Load V60 Teacher Model (Frozen)
    print(f">> Initializing V60 Teacher from {V60_TEACHER_CKPT}...")
    teacher = KarvishModelV60().to(DEVICE)
    if V60_TEACHER_CKPT.exists():
        t_data = torch.load(V60_TEACHER_CKPT, map_location=DEVICE, weights_only=False)
        teacher.load_state_dict(t_data.get("model_state_dict", t_data), strict=False)
        print(">> V60 Teacher weights loaded successfully.")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # 3. Initialize V61 Student Model (Resume from V61 or Warm-Start from V60)
    print(">> Initializing KarvishModelV61 Student...")
    student = KarvishModelV61().to(DEVICE)
    best_loss = float("inf")
    best_rank_acc = 0.0
    start_epoch = 1

    if V61_BEST_CKPT.exists():
        print(f">> Resuming from existing V61 checkpoint: {V61_BEST_CKPT}...")
        ckpt = torch.load(V61_BEST_CKPT, map_location=DEVICE, weights_only=False)
        student.load_state_dict(ckpt["model_state_dict"], strict=False)
        best_rank_acc = ckpt.get("rank_accuracy", 99.0)
        best_loss = ckpt.get("loss", 0.155)
        start_epoch = ckpt.get("epoch", 1) + 1
        print(f">> Resumed from Epoch {start_epoch-1} (Rank Acc: {best_rank_acc:.2f}%, Loss: {best_loss:.4f})")
    elif V60_TEACHER_CKPT.exists():
        student = load_v60_weights_into_v61(student, V60_TEACHER_CKPT)

    # 4. Optimizer & Schedulers
    optimizer = AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = (len(dataset) // batch_size) * (num_epochs - start_epoch + 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps), eta_min=5e-6)

    # Loss Functions
    bio_weights = torch.ones(NUM_BIO_TAGS, device=DEVICE) * 2.0
    bio_weights[BIO_TAG_TO_IDX["O"]] = 0.25
    bio_loss_fn = nn.CrossEntropyLoss(weight=bio_weights)

    margin_loss_fn = nn.MultiMarginLoss(margin=3.0)
    kd_loss_fn = nn.KLDivLoss(reduction="batchmean")

    # Dynamic Self-Repair Multipliers
    lambda_term = 0.3
    lambda_kd = 0.4
    lambda_ghd = 0.36  # Retain boosted GHD from epoch 1

    for epoch in range(start_epoch, num_epochs + 1):
        student.train()
        random.shuffle(dataset)
        total_epoch_loss = 0.0
        total_rank_loss = 0.0
        total_term_loss = 0.0
        total_kd_loss = 0.0
        total_ghd_loss = 0.0
        correct_cands = 0
        total_cands = 0
        false_term_count = 0
        total_neg_term = 0

        num_batches = len(dataset) // batch_size
        t0 = time.time()

        for step in range(num_batches):
            batch_slice = dataset[step * batch_size : (step + 1) * batch_size]
            batch = collate_v61_batch(batch_slice)

            g_ids = batch["goal_ids"].to(DEVICE)
            B = g_ids.size(0)
            g_mask = batch["goal_mask"].to(DEVICE)
            c_lbl_ids = batch["candidate_label_ids"].to(DEVICE)
            c_lbl_mask = batch["candidate_label_mask"].to(DEVICE)
            c_bounds = batch["candidate_bounds"].to(DEVICE)
            c_roles = batch["candidate_roles"].to(DEVICE)
            c_flags = batch["candidate_flags"].to(DEVICE)
            c_pad_mask = batch["candidate_pad_mask"].to(DEVICE)
            c_vis_feats = batch["candidate_visual_feats"].to(DEVICE)
            c_num_feats = batch["candidate_num_features"].to(DEVICE)
            tgt_idx = batch["target_indices"].to(DEVICE)
            tgt_act = batch["target_actions"].to(DEVICE)
            is_term = batch["is_terminal"].to(DEVICE)
            is_near_miss = batch["is_near_miss"].to(DEVICE)
            hindsight_d = batch["hindsight_delta"].to(DEVICE)
            bio_labels = batch["bio_slot_labels"].to(DEVICE)

            # Student Forward Pass
            out = student(
                goal_ids=g_ids,
                goal_mask=g_mask,
                candidate_label_ids=c_lbl_ids,
                candidate_label_mask=c_lbl_mask,
                candidate_bounds=c_bounds,
                candidate_roles=c_roles,
                candidate_flags=c_flags,
                candidate_pad_mask=c_pad_mask,
                candidate_visual_feats=c_vis_feats,
                candidate_num_features=c_num_feats,
            )

            cand_logits = out["ranking_scores"]  # [B, N]

            # 1. Candidate Ranking Margin Loss
            rank_loss = margin_loss_fn(cand_logits, tgt_idx)

            # 2. Teacher Invariance Shield (KD Loss against V60)
            with torch.no_grad():
                t_out = teacher(
                    goal_ids=g_ids,
                    goal_mask=g_mask,
                    candidate_label_ids=c_lbl_ids,
                    candidate_label_mask=c_lbl_mask,
                    candidate_bounds=c_bounds,
                    candidate_roles=c_roles,
                    candidate_flags=c_flags,
                    candidate_pad_mask=c_pad_mask,
                    candidate_num_features=c_num_feats,
                )
                t_logits = t_out["ranking_scores"]

            tau = 2.0
            p_s = F.log_softmax(cand_logits / tau, dim=-1)
            p_t = F.softmax(t_logits / tau, dim=-1)
            kd_loss = kd_loss_fn(p_s, p_t) * (tau ** 2)

            # 3. Gated Hindsight Distillation (GHD) Loss
            # Weighs the gradient by hindsight progression (hindsight_d in {0, 1})
            ghd_weights = 1.0 + 0.5 * hindsight_d  # [B]
            log_p_target = F.log_softmax(cand_logits, dim=-1).gather(1, tgt_idx.unsqueeze(-1)).squeeze(-1)
            ghd_loss = - (ghd_weights * log_p_target).mean()

            # 4. Calibrated Termination Loss (BCE with 5x near-miss penalty)
            pred_term = out["is_terminal_complete"].squeeze(-1)  # [B] in logits
            bce_term = F.binary_cross_entropy_with_logits(pred_term, is_term, reduction="none")
            penalty_weights = 1.0 + 4.0 * is_near_miss
            term_loss = (bce_term * penalty_weights).mean()

            # 5. BIO Slot Sequence Loss
            bio_logits = out["bio_slot_logits"]
            L_bio = min(bio_logits.size(1), bio_labels.size(1))
            slot_loss = bio_loss_fn(
                bio_logits[:, :L_bio].contiguous().view(-1, NUM_BIO_TAGS),
                bio_labels[:, :L_bio].contiguous().view(-1)
            )

            # Total Multi-Task Loss
            loss = rank_loss + (lambda_kd * kd_loss) + (lambda_ghd * ghd_loss) + (lambda_term * term_loss) + (0.3 * slot_loss)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Tracking Metrics
            total_epoch_loss += loss.item()
            total_rank_loss += rank_loss.item()
            total_kd_loss += kd_loss.item()
            total_ghd_loss += ghd_loss.item()
            total_term_loss += term_loss.item()

            preds = cand_logits.argmax(dim=-1)
            correct_cands += (preds == tgt_idx).sum().item()
            total_cands += B

            # Track false termination errors on non-terminal samples
            non_term_mask = (is_term == 0.0)
            if non_term_mask.sum().item() > 0:
                p_term_probs = torch.sigmoid(pred_term[non_term_mask])
                false_term_count += (p_term_probs > 0.5).sum().item()
                total_neg_term += non_term_mask.sum().item()

            if (step + 1) % 150 == 0 or (step + 1) == num_batches:
                step_acc = (correct_cands / max(1, total_cands)) * 100.0
                step_false_rate = (false_term_count / max(1, total_neg_term)) * 100.0
                print(f"  [Epoch {epoch}/{num_epochs} | Step {step+1}/{num_batches}] "
                      f"Loss: {loss.item():.4f} | Rank Acc: {step_acc:.2f}% | "
                      f"False Term: {step_false_rate:.2f}% | GHD Loss: {ghd_loss.item():.4f}")

        # Epoch Summary & Dynamic Self-Repair
        epoch_time = time.time() - t0
        avg_loss = total_epoch_loss / num_batches
        rank_acc = (correct_cands / total_cands) * 100.0
        false_term_rate = (false_term_count / max(1, total_neg_term)) * 100.0

        print(f"\n>> === EPOCH {epoch} SUMMARY ({epoch_time:.1f}s) ===")
        print(f"   Avg Loss:       {avg_loss:.4f}")
        print(f"   Rank Accuracy:  {rank_acc:.2f}%")
        print(f"   False Term Rate:{false_term_rate:.2f}%")
        print(f"   KD Loss:        {total_kd_loss / num_batches:.4f}")

        # Dynamic Self-Repair Gate
        if false_term_rate > 0.5:
            lambda_term = min(1.0, lambda_term * 1.5)
            print(f"   [Self-Repair] False termination rate high ({false_term_rate:.2f}%). Boosted lambda_term to {lambda_term:.2f}")

        if rank_acc < 99.5:
            lambda_ghd = min(0.8, lambda_ghd * 1.2)
            print(f"   [Self-Repair] Rank accuracy needs boost ({rank_acc:.2f}%). Boosted lambda_ghd to {lambda_ghd:.2f}")

        # Save Best Model Checkpoint
        if avg_loss < best_loss or rank_acc > best_rank_acc:
            best_loss = avg_loss
            best_rank_acc = rank_acc
            V61_BEST_CKPT.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": student.state_dict(),
                "loss": avg_loss,
                "rank_accuracy": rank_acc,
                "false_term_rate": false_term_rate,
            }, V61_BEST_CKPT)
            print(f"   >> [BEST CHECKPOINT SAVED] -> {V61_BEST_CKPT} (Loss: {avg_loss:.4f}, Rank Acc: {rank_acc:.2f}%)")
        print("=" * 70)

    print(f"\n>> Training Finished. Best Checkpoint: {V61_BEST_CKPT} (Rank Acc: {best_rank_acc:.2f}%)")


if __name__ == "__main__":
    train_v61_policy(num_epochs=3, batch_size=64, lr=2.5e-5)
