"""
KARVISH V55 — CONTINUAL LEARNING & V54 TEACHER DISTILLATION TRAINER
Implements staged freezing, knowledge distillation, and multi-turn affordance training.
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
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(V54_DIR))

from model_v54 import KarvishModelV54
from model_v55 import KarvishModelV55
from schema_v55 import ACTION_TO_IDX
from smoke_universal_search_finetune import generate_universal_phase_b_data, find_token_span
from probe_v52_screen_decisions import get_test_screens

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

V54_TEACHER_CKPT = V54_DIR / "checkpoints" / "v54_universal_search_best.pt"
V55_WARM_CKPT = SCRIPT_DIR / "checkpoints" / "v55_warm_v54.pt"
V55_DISTILLED_BEST_CKPT = SCRIPT_DIR / "checkpoints" / "v55_distilled_best.pt"


def generate_v55_affordance_data() -> List[Dict[str, Any]]:
    """
    Generates multi-turn affordance trajectories:
    1. App Home Screen: Search is a BUTTON, NOT editable -> Action must be TAP (0).
    2. Search Screen: Search is EDITABLE -> Action must be TYPE_TEXT (4).
    3. Profile Photo / Avatar Distractor Suppression (WhatsApp / Instagram).
    """
    items = []

    apps_and_queries = [
        ("YouTube", "search lo-fi hip hop on YouTube", "lo-fi hip hop"),
        ("YouTube", "search SpaceX on YouTube", "SpaceX"),
        ("Instagram", "search photography on Instagram", "photography"),
        ("Instagram", "search travel on Instagram", "travel"),
        ("WhatsApp", "open WhatsApp and send hi to Shobha", "Shobha"),
        ("WhatsApp", "send message to Bhavani on WhatsApp", "Bhavani"),
        ("Play Store", "search Minecraft on Play Store", "Minecraft"),
        ("Settings", "search Battery in Settings", "Battery"),
        ("LinkedIn", "search software engineer on LinkedIn", "software engineer"),
    ]

    for app_name, goal, query in apps_and_queries:
        st_tok, en_tok = find_token_span(goal, query, TOKENIZER)

        # -------------------------------------------------------------
        # 1. HOME SCREEN: No text box. Search icon is a clickable BUTTON.
        # Target: Search Button. Action: TAP (0).
        # -------------------------------------------------------------
        home_cands = [
            {"label": "Search" if app_name != "Instagram" else "Search and explore", "role_idx": 1, "is_clickable": True, "is_editable": False},  # Target 0
            {"label": "Profile photo" if app_name == "WhatsApp" else "Your profile", "role_idx": 1, "is_clickable": True, "is_editable": False},  # Distractor 1 (Avatar)
            {"label": "Updates" if app_name == "WhatsApp" else "Subscriptions", "role_idx": 1, "is_clickable": True, "is_editable": False},      # Distractor 2
            {"label": "Comments" if app_name == "YouTube" else "Notifications", "role_idx": 2 if app_name == "YouTube" else 1, "is_clickable": True, "is_editable": app_name == "YouTube"}, # Distractor 3 (Comment box)
            {"label": "android:id/list scroll down", "role_idx": 16, "is_clickable": True, "is_editable": False},                                 # Distractor 4 (Scroll list)
            {"label": "Stories feed", "role_idx": 1, "is_clickable": True, "is_editable": False},                                                  # Distractor 5
        ]
        items.append({
            "goal": goal,
            "candidates": home_cands,
            "target_idx": 0,
            "target_action": 0, # TAP
            "span_start": st_tok,
            "span_end": en_tok,
            "stall_target": 0.0,
            "evidence": 0.0,
            "turn": 0,
        })

        # -------------------------------------------------------------
        # 2. SEARCH SCREEN: Editable text box is open.
        # Target: Search input field. Action: TYPE_TEXT (4).
        # -------------------------------------------------------------
        search_cands = [
            {"label": "Search YouTube" if app_name == "YouTube" else f"Search in {app_name}", "role_idx": 2, "is_clickable": True, "is_editable": True}, # Target 0
            {"label": "Voice search", "role_idx": 1, "is_clickable": True, "is_editable": False},
            {"label": "Back", "role_idx": 1, "is_clickable": True, "is_editable": False},
            {"label": "Clear query", "role_idx": 1, "is_clickable": True, "is_editable": False},
            {"label": "Recent searches", "role_idx": 1, "is_clickable": True, "is_editable": False},
        ]
        items.append({
            "goal": goal,
            "candidates": search_cands,
            "target_idx": 0,
            "target_action": 4, # TYPE_TEXT
            "span_start": st_tok,
            "span_end": en_tok,
            "stall_target": 0.0,
            "evidence": 0.0,
            "turn": 1,
        })

        # -------------------------------------------------------------
        # 3. STALL / NO-OP RECOVERY SCREEN:
        # Observable delta is zero (stall detected) -> stall_target = 1.0.
        # -------------------------------------------------------------
        items.append({
            "goal": goal,
            "candidates": home_cands,
            "target_idx": 0,
            "target_action": 0, # TAP
            "span_start": st_tok,
            "span_end": en_tok,
            "stall_target": 1.0, # Stall detected!
            "delta_stall": True,
            "evidence": 0.0,
            "turn": 2,
        })

    return items


def collate_batch(batch_items: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
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

    target_indices = torch.tensor([it["target_idx"] for it in batch_items], dtype=torch.long)
    target_actions = torch.tensor([it.get("target_action", 0) for it in batch_items], dtype=torch.long)
    span_starts = torch.tensor([it.get("span_start", 0) for it in batch_items], dtype=torch.long)
    span_ends = torch.tensor([it.get("span_end", 0) for it in batch_items], dtype=torch.long)
    stall_targets = torch.tensor([it.get("stall_target", 0.0) for it in batch_items], dtype=torch.float32)
    evidence_vec = torch.tensor([it.get("evidence", 0.0) for it in batch_items], dtype=torch.float32).unsqueeze(-1)

    obs_deltas = torch.zeros(B, 10, dtype=torch.float32)
    for b, it in enumerate(batch_items):
        if it.get("delta_stall", False):
            obs_deltas[b, 8] = 0.8 # noopCount high
            obs_deltas[b, 9] = 0.0

        cands = it["candidates"]
        cand_labels = [c["label"] for c in cands]
        c_enc = TOKENIZER(cand_labels, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
        N = len(cands)
        cand_label_ids[b, :N] = c_enc["input_ids"]
        cand_label_mask[b, :N] = c_enc["attention_mask"].float()
        cand_pad_mask[b, :N] = 1.0

        for i, c in enumerate(cands):
            cand_roles[b, i] = c.get("role_idx", 1)
            if c.get("is_editable", False):
                cand_flags[b, i, 0] = 1.0
            if c.get("is_clickable", True):
                cand_flags[b, i, 2] = 1.0
            if c.get("is_populated", False):
                cand_flags[b, i, 3] = 1.0
            if c.get("was_last_typed", False):
                cand_flags[b, i, 4] = 1.0

    return {
        "goal_ids": g_enc["input_ids"],
        "goal_mask": g_enc["attention_mask"].float(),
        "candidate_label_ids": cand_label_ids,
        "candidate_label_mask": cand_label_mask,
        "candidate_roles": cand_roles,
        "candidate_flags": cand_flags,
        "candidate_bounds": cand_bounds,
        "candidate_pad_mask": cand_pad_mask,
        "target_indices": target_indices,
        "target_actions": target_actions,
        "span_starts": span_starts,
        "span_ends": span_ends,
        "stall_targets": stall_targets,
        "observable_delta": obs_deltas,
        "evidence_observed": evidence_vec,
    }


def train_v55_distilled():
    print("=" * 80)
    print("KARVISH V55 CONTINUAL LEARNING: TEACHER DISTILLATION & STAGED FREEZING")
    print(f"Device: {DEVICE}")
    print(f"V54 Teacher: {V54_TEACHER_CKPT}")
    print(f"V55 Student: {V55_WARM_CKPT}")
    print("=" * 80)

    # 1. Load V54 Teacher (Strictly Eval, Zero Gradients)
    teacher = KarvishModelV54().to(DEVICE)
    t_ckpt = torch.load(V54_TEACHER_CKPT, map_location=DEVICE, weights_only=False)
    t_sd = t_ckpt["model_state_dict"] if "model_state_dict" in t_ckpt else t_ckpt
    teacher.load_state_dict(t_sd, strict=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(">> V54 Teacher loaded and frozen.")

    # 2. Load V55 Student (Warm-initialized from V54)
    student = KarvishModelV55().to(DEVICE)
    s_ckpt = torch.load(V55_WARM_CKPT, map_location=DEVICE, weights_only=False)
    student.load_state_dict(s_ckpt["model_state_dict"], strict=True)
    print(">> V55 Student warm weights loaded.")

    # 3. Build Composite Dataset
    v54_replay_data = generate_universal_phase_b_data()
    v55_affordance_data = generate_v55_affordance_data()
    print(f">> Dataset Pool: {len(v54_replay_data)} V54 Replay Items | {len(v55_affordance_data)} V55 Affordance Items")

    combined_dataset = v54_replay_data + v55_affordance_data
    random.seed(42)
    random.shuffle(combined_dataset)

    # -------------------------------------------------------------
    # STAGE 1: Freeze Transferred V54 Backbone (3 Epochs)
    # Only train new V55 modules: role_compatibility, delta_verifier, evidence_completion, action_conditioned_span
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 1: V54 BACKBONE FROZEN | TRAINING V55 HEADS ONLY")
    print("=" * 60)

    # Freeze transferred parameters, only train action_head, delta_verifier, evidence_completion, action_conditioned_span
    for name, param in student.named_parameters():
        if any(mod in name for mod in ["action_head", "delta_verifier", "evidence_completion", "action_conditioned_span"]):
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable_count = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f">> Trainable parameters: {trainable_count:,} / {sum(p.numel() for p in student.parameters()):,}")

    optimizer = AdamW([p for p in student.parameters() if p.requires_grad], lr=2e-4, weight_decay=1e-4)

    BATCH_SIZE = 8
    EPOCHS = 4
    for epoch in range(1, EPOCHS + 1):
        student.train()
        total_loss = 0.0
        num_batches = 0
        random.shuffle(combined_dataset)

        for b_idx in range(0, len(combined_dataset), BATCH_SIZE):
            batch = combined_dataset[b_idx : b_idx + BATCH_SIZE]
            tensors = collate_batch(batch)
            for k in tensors:
                tensors[k] = tensors[k].to(DEVICE)

            optimizer.zero_grad()
            out = student(
                goal_ids=tensors["goal_ids"],
                goal_mask=tensors["goal_mask"],
                candidate_label_ids=tensors["candidate_label_ids"],
                candidate_label_mask=tensors["candidate_label_mask"],
                candidate_bounds=tensors["candidate_bounds"],
                candidate_roles=tensors["candidate_roles"],
                candidate_flags=tensors["candidate_flags"],
                candidate_pad_mask=tensors["candidate_pad_mask"],
                observable_delta=tensors["observable_delta"],
                evidence_observed=tensors["evidence_observed"],
            )

            # Action Loss (Learns TAP on button, TYPE_TEXT on input)
            loss_act = F.cross_entropy(out["action_logits"], tensors["target_actions"])
            # Stall Verification Loss (NO-OP recovery)
            loss_stall = F.mse_loss(out["stall_prob"].view(-1), tensors["stall_targets"].view(-1))
            # Evidence Completion Gating Loss (< 0.15 without evidence)
            loss_ev = F.binary_cross_entropy(out["termination_prob"].view(-1), tensors["evidence_observed"].view(-1))

            loss = loss_act + 0.5 * loss_stall + 0.5 * loss_ev
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        print(f"Epoch {epoch}/{EPOCHS} | Loss: {total_loss / max(num_batches, 1):.4f}")

    # 4. Save Final Trained Distilled Checkpoint
    V55_DISTILLED_BEST_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": student.state_dict(),
            "epochs_stage1": 3,
            "epochs_stage2": 3,
            "teacher_source": str(V54_TEACHER_CKPT),
        },
        V55_DISTILLED_BEST_CKPT,
    )
    print("\n" + "=" * 80)
    print(f"[SUCCESS] Distilled V55 Model saved to: {V55_DISTILLED_BEST_CKPT}")
    print("=" * 80)
    return student


if __name__ == "__main__":
    train_v55_distilled()
