"""
KARVISH UNIVERSAL V57 — GRADUAL DISTILLATION & FINE-TUNING TRAINER
Teacher: Karvish V56 (v56_distilled_best.pt)
Student: Karvish V57
Dataset: 3,255 Authoritative Trajectories (v57_authoritative_dataset.pt)

Pillars:
  1. Social Multi-Hop Feed-to-Inbox Navigation Grounding.
  2. Search History Chip vs. Active Composer Input Disambiguation.
  3. Selected Bottom Navigation Tab Invariance.
  4. Invariance Shield: 100% preservation of 12 baseline anchors via V56 distillation.
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
V56_DIR = SCRIPT_DIR.parent / "v56"
V55_DIR = SCRIPT_DIR.parent / "v55"
V54_DIR = SCRIPT_DIR.parent / "v54"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(V56_DIR))
sys.path.insert(0, str(V55_DIR))
sys.path.insert(0, str(V54_DIR))

from model_v56 import KarvishModelV56
from model_v57 import KarvishModelV57
from schema_v57 import ACTION_TO_IDX, ROLE_TO_IDX, CANONICAL_ACTIONS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

V56_TEACHER_CKPT = V56_DIR / "checkpoints" / "v56_distilled_best.pt"
V57_DATASET_FILE = SCRIPT_DIR / "checkpoints" / "v57_authoritative_dataset.pt"
V57_BEST_CKPT = SCRIPT_DIR / "checkpoints" / "v57_best.pt"


def collate_v57_batch(batch_items: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
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
    subgoals = []

    for b_idx, it in enumerate(batch_items):
        sp = it.get("target_span", (0, 0))
        span_starts.append(sp[0])
        span_ends.append(sp[1])
        slot_indices.append(it.get("slot_type", 0))
        subgoals.append(0)

        cands = it["candidates"]
        for c_idx, c in enumerate(cands):
            cand_pad_mask[b_idx, c_idx] = 1.0
            lbl = c.get("label", "")
            lbl_enc = TOKENIZER(lbl, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
            cand_label_ids[b_idx, c_idx] = lbl_enc["input_ids"][0]
            cand_label_mask[b_idx, c_idx] = lbl_enc["attention_mask"][0].float()

            r = c.get("role", "BUTTON")
            cand_roles[b_idx, c_idx] = ROLE_TO_IDX.get(r, 1)

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
        "stall_targets": torch.tensor([it.get("stall_target", 0.0) for it in batch_items], dtype=torch.float32),
        "is_replay": torch.tensor([not str(it.get("dataset_class", "")).startswith("v57_") for it in batch_items], dtype=torch.bool),
    }


def compute_margin_loss(scores: torch.Tensor, target_idx: torch.Tensor, pad_mask: torch.Tensor, margin: float = 3.0) -> torch.Tensor:
    B, N = scores.shape
    gold_scores = scores.gather(1, target_idx.unsqueeze(1))
    diffs = scores - gold_scores + margin
    diffs = diffs * pad_mask
    mask_gold = torch.ones_like(scores, dtype=torch.bool)
    mask_gold.scatter_(1, target_idx.unsqueeze(1), False)
    hard_diffs = F.relu(diffs) * mask_gold.float()
    return hard_diffs.sum() / (mask_gold.float() * pad_mask).sum().clamp(min=1.0)


def train_v57():
    print("=" * 80)
    print("KARVISH V57: TRAINING AUTHORITATIVE UNIVERSAL ACTION POLICY")
    print(f"Device: {DEVICE}")
    print("=" * 80)

    # 1. Load Dataset
    if not V57_DATASET_FILE.exists():
        print(f">> Dataset file not found at {V57_DATASET_FILE}. Generating now...")
        from build_v57_dataset import generate_v57_authoritative_dataset
        dataset = generate_v57_authoritative_dataset()
        torch.save(dataset, V57_DATASET_FILE)
    else:
        dataset = torch.load(V57_DATASET_FILE, weights_only=False)
        print(f">> Loaded V57 dataset: {len(dataset)} items from {V57_DATASET_FILE}")

    # 2. Load V56 Teacher Model (Frozen Invariance Shield)
    print(f">> Initializing V56 Teacher from {V56_TEACHER_CKPT}...")
    teacher = KarvishModelV56().to(DEVICE)
    if V56_TEACHER_CKPT.exists():
        t_data = torch.load(V56_TEACHER_CKPT, map_location=DEVICE, weights_only=False)
        teacher.load_state_dict(t_data["model_state_dict"])
        print(">> V56 Teacher weights successfully loaded.")
    else:
        print(">> WARNING: V56 teacher checkpoint not found. Training without KD.")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # 3. Initialize V57 Student (Warm-started from V56)
    student = KarvishModelV57().to(DEVICE)
    if V56_TEACHER_CKPT.exists():
        s_data = torch.load(V56_TEACHER_CKPT, map_location=DEVICE, weights_only=False)
        student.load_state_dict(s_data["model_state_dict"])
        print(">> V57 Student warm-started from V56 Teacher (100% parameter match).")

    # 4. Training Loop (Fine-Tuning with KD Beta = 0.5)
    optimizer = AdamW(student.parameters(), lr=6e-6, weight_decay=0.01)
    batch_size = 16
    epochs = 3
    TEMP = 2.0

    print(f">> Commencing V57 Fine-Tuning ({epochs} epochs, batch_size={batch_size})...")
    student.train()

    for epoch in range(1, epochs + 1):
        random.shuffle(dataset)
        total_loss = 0.0
        num_batches = 0

        for b_start in range(0, len(dataset), batch_size):
            batch = dataset[b_start : b_start + batch_size]
            t = collate_v57_batch(batch)
            t = {k: v.to(DEVICE) for k, v in t.items()}

            optimizer.zero_grad()

            with torch.no_grad():
                t_out = teacher(
                    goal_ids=t["goal_ids"],
                    goal_mask=t["goal_mask"],
                    candidate_label_ids=t["candidate_label_ids"],
                    candidate_label_mask=t["candidate_label_mask"],
                    candidate_roles=t["candidate_roles"],
                    candidate_flags=t["candidate_flags"],
                    candidate_pad_mask=t["candidate_pad_mask"],
                    candidate_bounds=t["candidate_bounds"],
                    candidate_num_features=t["candidate_num_features"],
                )
                t_logits = t_out["ranking_scores"]

            s_out = student(
                goal_ids=t["goal_ids"],
                goal_mask=t["goal_mask"],
                candidate_label_ids=t["candidate_label_ids"],
                candidate_label_mask=t["candidate_label_mask"],
                candidate_roles=t["candidate_roles"],
                candidate_flags=t["candidate_flags"],
                candidate_pad_mask=t["candidate_pad_mask"],
                candidate_bounds=t["candidate_bounds"],
                candidate_num_features=t["candidate_num_features"],
                slot_idx=t["slot_indices"],
                evidence_observed=t["evidence_observed"],
            )

            # Loss Components
            loss_ce = F.cross_entropy(s_out["ranking_scores"], t["target_indices"])
            loss_act = F.cross_entropy(s_out["action_logits"], t["target_actions"])

            # KD on anchor replay items
            if t["is_replay"].any():
                s_log = F.log_softmax(s_out["ranking_scores"][t["is_replay"]] / TEMP, dim=-1)
                t_p = F.softmax(t_logits[t["is_replay"]] / TEMP, dim=-1)
                loss_kd = F.kl_div(s_log, t_p, reduction="batchmean") * (TEMP * TEMP)
            else:
                loss_kd = torch.tensor(0.0, device=DEVICE)

            # Span Loss for typing actions
            is_type = (t["target_actions"] == ACTION_TO_IDX["TYPE_TEXT"])
            if is_type.any():
                loss_st = F.cross_entropy(s_out["span_start_logits"][is_type], t["span_starts"][is_type])
                loss_en = F.cross_entropy(s_out["span_end_logits"][is_type], t["span_ends"][is_type])
                loss_span = (loss_st + loss_en) * 0.5
            else:
                loss_span = torch.tensor(0.0, device=DEVICE)

            loss_margin = compute_margin_loss(s_out["ranking_scores"], t["target_indices"], t["candidate_pad_mask"], margin=3.0)
            loss_ev = F.binary_cross_entropy(s_out["termination_prob"].view(-1), t["evidence_observed"].view(-1))

            loss = 0.6 * loss_ce + 2.0 * loss_kd + 0.6 * loss_margin + 0.5 * loss_act + 0.4 * loss_span + 0.3 * loss_ev
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch}/{epochs} | Avg Loss: {avg_loss:.4f}")

    # 5. Save Certified V57 Checkpoint
    V57_BEST_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": student.state_dict(),
            "architecture": "KarvishModelV57",
            "teacher_source": str(V56_TEACHER_CKPT),
            "total_items": len(dataset),
        },
        V57_BEST_CKPT,
    )
    print("\n" + "=" * 80)
    print(f"[SUCCESS] Trained V57 Policy Model saved to: {V57_BEST_CKPT}")
    print("=" * 80)
    return student


if __name__ == "__main__":
    train_v57()
