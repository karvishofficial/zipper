"""
KARVISH V56 — 3-STAGE GRADUAL UNFREEZING TRAINER WITH V54 TEACHER DISTILLATION
Implements Senior's 3-Stage Curriculum:
  Stage 1: New heads + partial upper backbone (Adapter Warm-up, LR = 2e-4)
  Stage 2: Gradual backbone unfreezing (Intermediate Transformer Blocks, LR = 5e-5 + KD)
  Stage 3: Full low-LR fine-tuning with V54 Teacher Distillation (LR = 1.5e-5, KD Beta = 0.5)

Zero hardcoding; 100% learned universal representations across:
- Full communication lifecycle (LinkedIn message, Gmail compose, in-chat composer, send)
- Active tab vs. open search input
- Dual-slot span extraction (Recipient vs. Message Payload)
- Subgoal supervision & contextual role compatibility
- Evidence-gated completion via calibrated BCE
"""

from __future__ import annotations

import sys
import time
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
V54_DIR = SCRIPT_DIR.parent / "v54"
V55_DIR = SCRIPT_DIR.parent / "v55"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(V55_DIR))
sys.path.insert(0, str(V54_DIR))

from model_v54 import KarvishModelV54
from model_v56 import KarvishModelV56
from schema_v56 import ACTION_TO_IDX, SUBGOAL_TO_IDX
from build_v56_generalized_dataset import generate_v56_generalized_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

V54_TEACHER_CKPT = V54_DIR / "checkpoints" / "v54_universal_search_best.pt"
V56_WARM_CKPT = SCRIPT_DIR / "checkpoints" / "v56_warm_v54.pt"
V56_BEST_CKPT = SCRIPT_DIR / "checkpoints" / "v56_distilled_best.pt"

PHASE_TO_SUBGOAL = {
    "LOCATE_INPUT": 0,        # LOCATE_INPUT
    "ENTER_QUERY": 1,         # ENTER_QUERY
    "SUBMIT_SEARCH": 2,       # SUBMIT_SEARCH
    "SEARCH_RESULTS": 3,      # SELECT_RESULT
    "RECIPIENT_SELECTION": 3, # SELECT_RESULT
    "COMPOSE_MESSAGE": 1,     # ENTER_QUERY (typing message)
    "COMMIT_MESSAGE": 5,      # CONFIRM_ACTION (send action)
    "SELECT_PRODUCT": 3,      # SELECT_RESULT
    "PROFILE_VIEW": 0,        # LOCATE_INPUT
    "INBOX_VIEW": 0,          # LOCATE_INPUT
    "VERIFY_TRANSMISSION": 9, # COMPLETE_TASK
    "VERIFY_CART": 9,         # COMPLETE_TASK
    "VERIFY_RESULT": 9,       # COMPLETE_TASK
    "KEYPAD_SEQUENCE": 1,     # ENTER_QUERY
    "FIND_TARGET": 0,         # LOCATE_INPUT
    "REPLAY_ANCHOR": 0,
}


def collate_v56_batch(batch_items: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
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
    target_actions = torch.tensor([it.get("action_idx", it.get("target_action", 0)) for it in batch_items], dtype=torch.long)
    
    # Dual-slot spans
    span_starts = []
    span_ends = []
    slot_indices = []
    subgoals = []
    for it in batch_items:
        sp = it.get("target_span", (it.get("span_start", 0), it.get("span_end", 0)))
        span_starts.append(sp[0])
        span_ends.append(sp[1])
        slot_indices.append(it.get("slot_type", 0))
        subgoals.append(PHASE_TO_SUBGOAL.get(it.get("current_phase", "REPLAY_ANCHOR"), 0))

    t_span_starts = torch.tensor(span_starts, dtype=torch.long)
    t_span_ends = torch.tensor(span_ends, dtype=torch.long)
    t_slot_indices = torch.tensor(slot_indices, dtype=torch.long)
    t_subgoals = torch.tensor(subgoals, dtype=torch.long)

    stall_targets = torch.tensor([it.get("stall_target", 0.0) for it in batch_items], dtype=torch.float32)
    evidence_vec = torch.tensor([it.get("evidence", 0.0) for it in batch_items], dtype=torch.float32)

    obs_deltas = torch.zeros(B, 10, dtype=torch.float32)
    for b, it in enumerate(batch_items):
        if it.get("delta_stall", False):
            obs_deltas[b, 8] = 0.8  # consecutive noop count high
            obs_deltas[b, 9] = 0.0

        cands = it["candidates"]
        cand_labels = [str(c.get("label", "")).strip() for c in cands]
        c_enc = TOKENIZER(cand_labels, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
        N = len(cands)
        cand_label_ids[b, :N] = c_enc["input_ids"]
        cand_label_mask[b, :N] = c_enc["attention_mask"].float()
        cand_pad_mask[b, :N] = 1.0

        for i, c in enumerate(cands):
            cand_roles[b, i] = c.get("role_idx", 1)
            bnd = c.get("bounds_vec", [0.0] * 10)
            if len(bnd) == 10:
                cand_bounds[b, i] = torch.tensor(bnd, dtype=torch.float32)

            if c.get("is_editable", False) or cand_roles[b, i] in (2, 3, 4):
                cand_flags[b, i, 0] = 1.0
            if c.get("is_focused", False):
                cand_flags[b, i, 1] = 1.0
            if c.get("is_clickable", True):
                cand_flags[b, i, 2] = 1.0
            if c.get("is_populated", False):
                cand_flags[b, i, 3] = 1.0
            if c.get("was_last_typed", False):
                cand_flags[b, i, 4] = 1.0
            if c.get("is_selected", False):
                cand_flags[b, i, 5] = 1.0

    return {
        "goal_ids": g_enc["input_ids"],
        "goal_mask": g_enc["attention_mask"].float(),
        "candidate_label_ids": cand_label_ids,
        "candidate_label_mask": cand_label_mask,
        "candidate_roles": cand_roles,
        "candidate_flags": cand_flags,
        "candidate_bounds": cand_bounds,
        "candidate_pad_mask": cand_pad_mask,
        "candidate_num_features": cand_num_features,
        "target_indices": target_indices,
        "target_actions": target_actions,
        "span_starts": t_span_starts,
        "span_ends": t_span_ends,
        "slot_indices": t_slot_indices,
        "subgoal_indices": t_subgoals,
        "stall_targets": stall_targets,
        "observable_delta": obs_deltas,
        "evidence_observed": evidence_vec,
        "is_replay": torch.tensor([it.get("dataset_class") == "v54_replay" for it in batch_items], dtype=torch.bool),
    }


def compute_margin_loss(scores: torch.Tensor, targets: torch.Tensor, pad_mask: torch.Tensor, margin: float = 3.0) -> torch.Tensor:
    target_scores = scores.gather(1, targets.unsqueeze(1))
    diffs = target_scores - scores
    violations = F.relu(margin - diffs) * pad_mask
    target_mask = F.one_hot(targets, num_classes=scores.size(1)).bool()
    violations = violations.masked_fill(target_mask, 0.0)
    return violations.sum() / scores.size(0)


def train_v56_distilled():
    print("=" * 80)
    print("KARVISH V56: 3-STAGE GRADUAL UNFREEZING & V54 TEACHER DISTILLATION")
    print(f"Device: {DEVICE}")
    print(f"V54 Teacher: {V54_TEACHER_CKPT}")
    print(f"V56 Warm Student: {V56_WARM_CKPT}")
    print("=" * 80)

    # 1. Load Frozen V54 Teacher
    teacher = KarvishModelV54().to(DEVICE)
    t_ckpt = torch.load(V54_TEACHER_CKPT, map_location=DEVICE, weights_only=False)
    t_sd = t_ckpt["model_state_dict"] if "model_state_dict" in t_ckpt else t_ckpt
    teacher.load_state_dict(t_sd, strict=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(">> V54 Teacher loaded and permanently frozen.")

    # 2. Load V56 Student Model
    student = KarvishModelV56().to(DEVICE)
    s_ckpt = torch.load(V56_WARM_CKPT, map_location=DEVICE, weights_only=False)
    student.load_state_dict(s_ckpt["model_state_dict"], strict=True)
    print(">> V56 Student loaded with warm transferred weights.")

    # 3. Load Diverse Dataset
    dataset = generate_v56_generalized_dataset()
    print(f">> Full Dataset: {len(dataset)} items across V54 replay, Lifecycle, Tab/Input, and Stalls.")

    BATCH_SIZE = 8
    TEMP = 2.0

    # -----------------------------------------------------------------
    # STAGE 1: Adapter Warm-Up (Epochs 1–3)
    # Train new heads + dual-slot span + role compatibility with LR = 2e-4
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 1: ADAPTER WARM-UP (Heads + Dual-Span + Role Compat, LR = 2e-4)")
    print("=" * 60)

    HEAD_MODULES = [
        "rank_head",
        "cand_fuse",
        "role_embed",
        "flags_mlp",
        "role_compatibility",
        "action_conditioned_span",
        "action_head",
        "delta_verifier",
        "evidence_completion",
        "task_state_engine",
        "text_encoder.pooler",
    ]

    for p in student.parameters():
        p.requires_grad = False
    for name, param in student.named_parameters():
        if any(mod in name for mod in HEAD_MODULES):
            param.requires_grad = True

    trainable_stage1 = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f">> Stage 1 Trainable Parameters: {trainable_stage1:,} / {sum(p.numel() for p in student.parameters()):,}")

    optimizer1 = AdamW([p for p in student.parameters() if p.requires_grad], lr=2e-4, weight_decay=1e-4)

    for epoch in range(1, 4):
        student.train()
        total_loss = 0.0
        num_batches = 0
        random.shuffle(dataset)

        for b_idx in range(0, len(dataset), BATCH_SIZE):
            batch = dataset[b_idx : b_idx + BATCH_SIZE]
            t = collate_v56_batch(batch)
            for k in t:
                t[k] = t[k].to(DEVICE)

            optimizer1.zero_grad()
            out = student(
                goal_ids=t["goal_ids"],
                goal_mask=t["goal_mask"],
                candidate_label_ids=t["candidate_label_ids"],
                candidate_label_mask=t["candidate_label_mask"],
                candidate_bounds=t["candidate_bounds"],
                candidate_roles=t["candidate_roles"],
                candidate_flags=t["candidate_flags"],
                candidate_pad_mask=t["candidate_pad_mask"],
                candidate_num_features=t["candidate_num_features"],
                observable_delta=t["observable_delta"],
                evidence_observed=t["evidence_observed"],
                slot_idx=t["slot_indices"],
            )

            # Losses
            loss_ce = F.cross_entropy(out["ranking_scores"], t["target_indices"])
            loss_act = F.cross_entropy(out["action_logits"], t["target_actions"])
            loss_subgoal = F.cross_entropy(out["subgoal_logits"], t["subgoal_indices"])

            # Dual-slot span loss for TYPE_TEXT actions
            is_type = (t["target_actions"] == ACTION_TO_IDX["TYPE_TEXT"])
            if is_type.any():
                loss_st = F.cross_entropy(out["span_start_logits"][is_type], t["span_starts"][is_type])
                loss_en = F.cross_entropy(out["span_end_logits"][is_type], t["span_ends"][is_type])
                loss_span = (loss_st + loss_en) * 0.5
            else:
                loss_span = torch.tensor(0.0, device=DEVICE)

            loss_stall = F.mse_loss(out["stall_prob"].view(-1), t["stall_targets"].view(-1))
            loss_ev = F.binary_cross_entropy(out["termination_prob"].view(-1), t["evidence_observed"].view(-1))

            loss_margin = compute_margin_loss(out["ranking_scores"], t["target_indices"], t["candidate_pad_mask"], margin=3.0)
            loss = loss_ce + 0.8 * loss_margin + 0.5 * loss_act + 0.4 * loss_span + 0.3 * loss_stall + 0.3 * loss_ev + 0.4 * loss_subgoal
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer1.step()

            total_loss += loss.item()
            num_batches += 1

        print(f"Stage 1 | Epoch {epoch}/3 | Avg Loss: {total_loss / max(num_batches, 1):.4f}")

    # -----------------------------------------------------------------
    # STAGE 2: Gradual Backbone Unfreezing (Epochs 4–6)
    # Unfreeze upper transformer layers (layers 4 & 5) + Container Attention + Role Compat + KD (LR = 5e-5)
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 2: GRADUAL BACKBONE UNFREEZING (Upper Layers + KD Distillation, LR = 5e-5)")
    print("=" * 60)

    for name, param in student.named_parameters():
        if "text_encoder.encoder.layer.4" in name or "text_encoder.encoder.layer.5" in name:
            param.requires_grad = True
        elif any(mod in name for mod in HEAD_MODULES) or "container_attention" in name:
            param.requires_grad = True

    trainable_stage2 = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f">> Stage 2 Trainable Parameters: {trainable_stage2:,} / {sum(p.numel() for p in student.parameters()):,}")

    optimizer2 = AdamW([p for p in student.parameters() if p.requires_grad], lr=5e-5, weight_decay=1e-4)

    for epoch in range(4, 7):
        student.train()
        total_loss = 0.0
        num_batches = 0
        random.shuffle(dataset)

        for b_idx in range(0, len(dataset), BATCH_SIZE):
            batch = dataset[b_idx : b_idx + BATCH_SIZE]
            t = collate_v56_batch(batch)
            for k in t:
                t[k] = t[k].to(DEVICE)

            with torch.no_grad():
                t_out = teacher(
                    goal_ids=t["goal_ids"],
                    goal_mask=t["goal_mask"],
                    candidate_label_ids=t["candidate_label_ids"],
                    candidate_label_mask=t["candidate_label_mask"],
                    candidate_bounds=t["candidate_bounds"],
                    candidate_roles=t["candidate_roles"].clamp(0, 15),
                    candidate_flags=t["candidate_flags"],
                    candidate_pad_mask=t["candidate_pad_mask"],
                    candidate_num_features=t["candidate_num_features"],
                )
                t_logits = t_out["ranking_scores"]

            optimizer2.zero_grad()
            out = student(
                goal_ids=t["goal_ids"],
                goal_mask=t["goal_mask"],
                candidate_label_ids=t["candidate_label_ids"],
                candidate_label_mask=t["candidate_label_mask"],
                candidate_bounds=t["candidate_bounds"],
                candidate_roles=t["candidate_roles"],
                candidate_flags=t["candidate_flags"],
                candidate_pad_mask=t["candidate_pad_mask"],
                candidate_num_features=t["candidate_num_features"],
                observable_delta=t["observable_delta"],
                evidence_observed=t["evidence_observed"],
                slot_idx=t["slot_indices"],
            )

            # KD Loss on ranking logits (Anchor Shield: Only on historical v54_replay items)
            if t["is_replay"].any():
                s_log_probs = F.log_softmax(out["ranking_scores"][t["is_replay"]] / TEMP, dim=-1)
                t_probs = F.softmax(t_logits[t["is_replay"]] / TEMP, dim=-1)
                loss_kd = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (TEMP * TEMP)
            else:
                loss_kd = torch.tensor(0.0, device=DEVICE)

            loss_ce = F.cross_entropy(out["ranking_scores"], t["target_indices"])
            loss_act = F.cross_entropy(out["action_logits"], t["target_actions"])
            loss_subgoal = F.cross_entropy(out["subgoal_logits"], t["subgoal_indices"])

            is_type = (t["target_actions"] == ACTION_TO_IDX["TYPE_TEXT"])
            if is_type.any():
                loss_st = F.cross_entropy(out["span_start_logits"][is_type], t["span_starts"][is_type])
                loss_en = F.cross_entropy(out["span_end_logits"][is_type], t["span_ends"][is_type])
                loss_span = (loss_st + loss_en) * 0.5
            else:
                loss_span = torch.tensor(0.0, device=DEVICE)

            loss_stall = F.mse_loss(out["stall_prob"].view(-1), t["stall_targets"].view(-1))
            loss_ev = F.binary_cross_entropy(out["termination_prob"].view(-1), t["evidence_observed"].view(-1))

            loss_margin = compute_margin_loss(out["ranking_scores"], t["target_indices"], t["candidate_pad_mask"], margin=3.0)
            loss = loss_ce + 0.5 * loss_kd + 0.8 * loss_margin + 0.5 * loss_act + 0.4 * loss_span + 0.3 * loss_stall + 0.3 * loss_ev + 0.4 * loss_subgoal
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer2.step()

            total_loss += loss.item()
            num_batches += 1

        print(f"Stage 2 | Epoch {epoch}/6 | Avg Loss: {total_loss / max(num_batches, 1):.4f}")

    # -----------------------------------------------------------------
    # STAGE 3: Full Low-LR Fine-Tuning with Strong Distillation (Epochs 7–10)
    # All parameters unfrozen (LR = 1.5e-5 backbone, 1e-4 heads, KD Beta = 0.5)
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 3: FULL FINE-TUNING (All Parameters Unfrozen + Strong Teacher Anchor, LR = 1.5e-5)")
    print("=" * 60)

    for param in student.parameters():
        param.requires_grad = True

    trainable_stage3 = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f">> Stage 3 Trainable Parameters: {trainable_stage3:,} / {sum(p.numel() for p in student.parameters()):,}")

    head_params = [p for n, p in student.named_parameters() if any(mod in n for mod in HEAD_MODULES)]
    backbone_params = [p for n, p in student.named_parameters() if not any(mod in n for mod in HEAD_MODULES)]
    optimizer3 = AdamW([
        {"params": backbone_params, "lr": 1.5e-5},
        {"params": head_params, "lr": 1.5e-4},
    ], weight_decay=1e-4)

    for epoch in range(7, 11):
        student.train()
        total_loss = 0.0
        num_batches = 0
        random.shuffle(dataset)

        for b_idx in range(0, len(dataset), BATCH_SIZE):
            batch = dataset[b_idx : b_idx + BATCH_SIZE]
            t = collate_v56_batch(batch)
            for k in t:
                t[k] = t[k].to(DEVICE)

            with torch.no_grad():
                t_out = teacher(
                    goal_ids=t["goal_ids"],
                    goal_mask=t["goal_mask"],
                    candidate_label_ids=t["candidate_label_ids"],
                    candidate_label_mask=t["candidate_label_mask"],
                    candidate_bounds=t["candidate_bounds"],
                    candidate_roles=t["candidate_roles"].clamp(0, 15),
                    candidate_flags=t["candidate_flags"],
                    candidate_pad_mask=t["candidate_pad_mask"],
                    candidate_num_features=t["candidate_num_features"],
                )
                t_logits = t_out["ranking_scores"]

            optimizer3.zero_grad()
            out = student(
                goal_ids=t["goal_ids"],
                goal_mask=t["goal_mask"],
                candidate_label_ids=t["candidate_label_ids"],
                candidate_label_mask=t["candidate_label_mask"],
                candidate_bounds=t["candidate_bounds"],
                candidate_roles=t["candidate_roles"],
                candidate_flags=t["candidate_flags"],
                candidate_pad_mask=t["candidate_pad_mask"],
                candidate_num_features=t["candidate_num_features"],
                observable_delta=t["observable_delta"],
                evidence_observed=t["evidence_observed"],
                slot_idx=t["slot_indices"],
            )

            # KD Loss on ranking logits (Anchor Shield: Only on historical v54_replay items)
            if t["is_replay"].any():
                s_log_probs = F.log_softmax(out["ranking_scores"][t["is_replay"]] / TEMP, dim=-1)
                t_probs = F.softmax(t_logits[t["is_replay"]] / TEMP, dim=-1)
                loss_kd = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (TEMP * TEMP)
            else:
                loss_kd = torch.tensor(0.0, device=DEVICE)

            loss_ce = F.cross_entropy(out["ranking_scores"], t["target_indices"])
            loss_act = F.cross_entropy(out["action_logits"], t["target_actions"])
            loss_subgoal = F.cross_entropy(out["subgoal_logits"], t["subgoal_indices"])

            is_type = (t["target_actions"] == ACTION_TO_IDX["TYPE_TEXT"])
            if is_type.any():
                loss_st = F.cross_entropy(out["span_start_logits"][is_type], t["span_starts"][is_type])
                loss_en = F.cross_entropy(out["span_end_logits"][is_type], t["span_ends"][is_type])
                loss_span = (loss_st + loss_en) * 0.5
            else:
                loss_span = torch.tensor(0.0, device=DEVICE)

            loss_stall = F.mse_loss(out["stall_prob"].view(-1), t["stall_targets"].view(-1))
            loss_ev = F.binary_cross_entropy(out["termination_prob"].view(-1), t["evidence_observed"].view(-1))

            loss_margin = compute_margin_loss(out["ranking_scores"], t["target_indices"], t["candidate_pad_mask"], margin=3.0)
            loss = loss_ce + 0.5 * loss_kd + 0.8 * loss_margin + 0.5 * loss_act + 0.4 * loss_span + 0.3 * loss_stall + 0.3 * loss_ev + 0.4 * loss_subgoal
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer3.step()

            total_loss += loss.item()
            num_batches += 1

        print(f"Stage 3 | Epoch {epoch}/10 | Avg Loss: {total_loss / max(num_batches, 1):.4f}")

    # 4. Save Final Distilled Model Checkpoint
    V56_BEST_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": student.state_dict(),
            "curriculum": "3-stage gradual unfreezing with subgoal & margin supervision",
            "teacher_source": str(V54_TEACHER_CKPT),
            "total_items": len(dataset),
        },
        V56_BEST_CKPT,
    )
    print("\n" + "=" * 80)
    print(f"[SUCCESS] Trained V56 Distilled Model saved to: {V56_BEST_CKPT}")
    print("=" * 80)
    return student


if __name__ == "__main__":
    train_v56_distilled()
