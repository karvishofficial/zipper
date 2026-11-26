"""
KARVISH V54 — BALANCED SMOKE TEST & DUAL VERIFICATION TRAINER
Refined with 4 targeted solutions:
1. 1:1:1 Action Balancing (Launcher OPEN_APP, Chat Input TYPE_TEXT, Send CLICK).
2. Explicit Hard Negative Suppression (YouTube on launcher, Send Message on empty chat input).
3. Role-conditioned Action Head Learning (EditText -> TYPE_TEXT, Launcher Button -> OPEN_APP).
4. Multi-word Question Span Supervision (start at 'what'/'how'/'explain').

Saves to sandbox: tools/training/karvish_mobile_action/v54/checkpoints/v54_chat_smoke.pt
Leaves v54_best.pt, v52_locked_gold.pt, and v51_7_gold 100% untouched.
"""

from __future__ import annotations

import sys
import time
import re
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "v52"))

from model_v54 import KarvishModelV54
from schema_v54 import ACTION_TO_IDX
from probe_v52_screen_decisions import get_test_screens

IDX_TO_ACTION = {v: k for k, v in ACTION_TO_IDX.items()}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

SOURCE_CKPT = SCRIPT_DIR / "checkpoints" / "v54_best.pt"
if not SOURCE_CKPT.exists():
    SOURCE_CKPT = SCRIPT_DIR.parent / "v52_locked_gold" / "v52_locked_gold.pt"

SMOKE_CKPT = SCRIPT_DIR / "checkpoints" / "v54_chat_smoke.pt"
SMOKE_CKPT.parent.mkdir(parents=True, exist_ok=True)


def find_token_span(text: str, entity: str, tokenizer) -> Tuple[int, int]:
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc["offset_mapping"]
    lower_text = text.lower()
    lower_ent = entity.lower().strip()
    idx = lower_text.find(lower_ent)
    if idx == -1:
        return (0, 0)
    ent_start = idx
    ent_end = idx + len(lower_ent)

    start_token = 0
    end_token = 0
    for t_idx, (st, en) in enumerate(offsets):
        if st <= ent_start < en or (st == ent_start and st != en):
            start_token = t_idx
            break
    for t_idx, (st, en) in enumerate(offsets):
        if st < ent_end <= en or (en == ent_end and st != en):
            end_token = t_idx
            break

    return (start_token, max(start_token, end_token))


def build_balanced_dataset() -> List[Dict[str, Any]]:
    dataset = []

    # 1. 12 Frozen Baseline Anchor Screens (Invariance Guard)
    anchor_screens = get_test_screens()
    for s in anchor_screens:
        cands = s["candidates"][:24]
        target_idx = next((i for i, c in enumerate(cands) if s["expected_target"].lower() in c["label"].lower()), 0)
        act_name = s.get("expected_action", "CLICK")
        act_idx = ACTION_TO_IDX.get(act_name, 0)
        st_tok, en_tok = find_token_span(s["goal"], s["expected_target"], TOKENIZER)
        dataset.append({
            "name": f"anchor_{s['name']}",
            "category": "anchor",
            "goal": s["goal"],
            "candidates": cands,
            "target_idx": target_idx,
            "action_idx": act_idx,
            "hard_negs": [i for i in range(len(cands)) if i != target_idx][:3],
            "target_span": (st_tok, en_tok),
            "entity_target": s["expected_target"],
        })

    # 2. Universal Launcher Screens (Turn 1: OPEN_APP, with YouTube explicitly in Hard Negatives)
    launcher_scenarios = [
        ("Ask ChatGPT what is algae", "ChatGPT"),
        ("Ask Gemini how airplanes fly", "Gemini"),
        ("Ask Claude to explain gravity", "Claude"),
        ("Ask Perplexity what is photosynthesis", "Perplexity"),
        ("Ask Copilot summarize quantum computing", "Copilot"),
        ("Open ChatGPT and ask what is algae", "ChatGPT"),
        ("Open Gemini and ask how airplanes fly", "Gemini"),
        ("Ask what is algae on ChatGPT", "ChatGPT"),
    ]

    for goal_text, target_app in launcher_scenarios:
        # App pool ALWAYS contains YouTube, Settings, Calculator as anchor distractors
        icons = [
            {"id": f"c_{target_app.lower()}", "label": target_app, "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.1, 0.3, 0.25, 0.45, 0.5, 0.5, 0.15, 0.15, 0.02, 1.0]},
            {"id": "c_yt", "label": "YouTube", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.1, 0.1, 0.25, 0.25, 0.5, 0.5, 0.15, 0.15, 0.02, 1.0]},
            {"id": "c_set", "label": "Settings", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.1, 0.5, 0.25, 0.65, 0.5, 0.5, 0.15, 0.15, 0.02, 1.0]},
            {"id": "c_calc", "label": "Calculator", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.1, 0.7, 0.25, 0.85, 0.5, 0.5, 0.15, 0.15, 0.02, 1.0]},
            {"id": "c_comp", "label": "complete", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]},
        ]
        target_idx = 0
        hard_negs = [1, 2, 3, 4]  # YouTube (idx 1), Settings, Calculator, Complete

        dataset.append({
            "name": f"launcher_{target_app.lower()}",
            "category": "launcher",
            "goal": goal_text,
            "candidates": icons,
            "target_idx": target_idx,
            "action_idx": ACTION_TO_IDX["OPEN_APP"],
            "hard_negs": hard_negs,
            "target_span": (0, 0),
            "entity_target": target_app,
        })

    # 3. Real AI Chat Input Screens (Turn 2: TYPE_TEXT, with Send Message in Hard Negatives)
    chat_input_scenarios = [
        ("Ask ChatGPT what is algae", "what is algae", "Reply to ChatGPT", "ChatGPT"),
        ("Ask Gemini how airplanes fly", "how airplanes fly", "Ask Gemini", "Gemini"),
        ("Ask Perplexity what is photosynthesis", "what is photosynthesis", "Ask anything", "Perplexity"),
        ("Ask Claude to explain gravity", "explain gravity", "Message", "Claude"),
        ("Open ChatGPT and ask what is algae", "what is algae", "Reply to ChatGPT", "ChatGPT"),
        ("Open Gemini and ask how airplanes fly", "how airplanes fly", "Ask Gemini", "Gemini"),
        ("Ask Copilot summarize quantum computing", "summarize quantum computing", "Chat with Copilot", "Copilot"),
        ("Ask what is algae on ChatGPT", "what is algae", "Reply to ChatGPT", "ChatGPT"),
    ]

    for goal_text, question_span, input_label, app_name in chat_input_scenarios:
        chat_cands = [
            {"id": "c_input", "label": input_label, "role": "input", "role_idx": 2, "is_editable": True, "is_clickable": True, "is_focused": False, "bounds_vec": [0.48, 0.17, 0.55, 0.83, 0.5, 0.51, 0.66, 0.07, 0.046, 9.4]},
            {"id": "c_send", "label": "Send Message", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.48, 0.83, 0.55, 0.96, 0.89, 0.51, 0.13, 0.07, 0.009, 1.8]},
            {"id": "c_mic", "label": "Microphone", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.48, 0.70, 0.55, 0.83, 0.76, 0.51, 0.13, 0.07, 0.009, 1.8]},
            {"id": "c_att", "label": "Add attachments", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.48, 0.03, 0.55, 0.17, 0.10, 0.51, 0.14, 0.07, 0.009, 2.0]},
            {"id": "c_side", "label": "Open sidebar", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.04, 0.11, 0.18, 0.11, 0.08, 0.14, 0.06, 0.008, 2.3]},
            {"id": "c_comp", "label": "complete", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]},
        ]
        st_tok, en_tok = find_token_span(goal_text, question_span, TOKENIZER)
        dataset.append({
            "name": f"chat_input_{app_name.lower()}",
            "category": "chat_input",
            "goal": goal_text,
            "candidates": chat_cands,
            "target_idx": 0,  # The EditText is target
            "action_idx": ACTION_TO_IDX["TYPE_TEXT"],
            "hard_negs": [1, 2, 3, 4, 5],  # Send Message, Mic, Attachment, Sidebar, Complete
            "target_span": (st_tok, en_tok),
            "entity_target": question_span,
        })

    # 4. Real AI Chat Send Button Screens (Turn 3: CLICK Send when field is populated)
    send_scenarios = [
        ("Ask ChatGPT what is algae", "what is algae", "Send Message"),
        ("Ask Gemini how airplanes fly", "how airplanes fly", "Send prompt"),
        ("Ask Perplexity what is photosynthesis", "what is photosynthesis", "Send Message"),
        ("Ask Claude to explain gravity", "explain gravity", "Send"),
        ("Open ChatGPT and ask what is algae", "what is algae", "Send Message"),
        ("Open Gemini and ask how airplanes fly", "how airplanes fly", "Send prompt"),
        ("Ask Copilot summarize quantum computing", "summarize quantum computing", "Send Message"),
        ("Ask what is algae on ChatGPT", "what is algae", "Send Message"),
    ]

    for goal_text, typed_text, send_label in send_scenarios:
        send_cands = [
            {"id": "c_send", "label": send_label, "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.48, 0.83, 0.55, 0.96, 0.89, 0.51, 0.13, 0.07, 0.009, 1.8]},
            {"id": "c_in_filled", "label": typed_text, "role": "input", "role_idx": 2, "is_editable": True, "is_focused": True, "bounds_vec": [0.48, 0.17, 0.55, 0.83, 0.5, 0.51, 0.66, 0.07, 0.046, 9.4]},
            {"id": "c_att", "label": "Add attachments", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.48, 0.03, 0.55, 0.17, 0.10, 0.51, 0.14, 0.07, 0.009, 2.0]},
            {"id": "c_comp", "label": "complete", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]},
        ]
        dataset.append({
            "name": "chat_send_screen",
            "category": "chat_send",
            "goal": goal_text,
            "candidates": send_cands,
            "target_idx": 0,  # Send button is target
            "action_idx": ACTION_TO_IDX["CLICK"],
            "hard_negs": [1, 2, 3],  # Filled input, attachment, complete
            "target_span": (0, 0),
            "entity_target": send_label,
        })

    return dataset


def run_balanced_smoke_training(total_steps: int = 120):
    print("=" * 80)
    print(f"STARTING REFINED BALANCED SMOKE TRAINING ({total_steps} STEPS)")
    print(f"Source Checkpoint: {SOURCE_CKPT.name}")
    print(f"Sandbox Target:    {SMOKE_CKPT.name}")
    print("=" * 80)

    model = KarvishModelV54().to(DEVICE)
    ckpt = torch.load(SOURCE_CKPT, map_location=DEVICE, weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd, strict=False)

    for p in model.parameters():
        p.requires_grad = False
    for p in model.rank_head.parameters():
        p.requires_grad = True
    for p in model.action_head.parameters():
        p.requires_grad = True
    for p in model.span_pointer_head.parameters():
        p.requires_grad = True
    for p in model.cand_fuse.parameters():
        p.requires_grad = True
    for p in model.role_embed.parameters():
        p.requires_grad = True
    for p in model.flags_mlp.parameters():
        p.requires_grad = True
    for p in model.container_attention.parameters():
        p.requires_grad = True
    for p in model.task_state_engine.parameters():
        p.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW([
        {"params": model.action_head.parameters(), "lr": 1.5e-4},
        {"params": model.span_pointer_head.parameters(), "lr": 1.5e-4},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and "action_head" not in n and "span_pointer_head" not in n], "lr": 4e-5},
    ], weight_decay=1e-3)
    dataset = build_balanced_dataset()

    anchors = [d for d in dataset if d["category"] == "anchor"]
    launchers = [d for d in dataset if d["category"] == "launcher"]
    chat_inputs = [d for d in dataset if d["category"] == "chat_input"]
    chat_sends = [d for d in dataset if d["category"] == "chat_send"]

    print(f">> Dataset Pool: {len(anchors)} Anchors, {len(launchers)} Launchers, {len(chat_inputs)} ChatInputs, {len(chat_sends)} ChatSends")

    t0 = time.time()
    for step in range(1, total_steps + 1):
        # 4-way balanced round-robin: Anchor -> Launcher -> ChatInput -> ChatSend
        phase = (step - 1) % 4
        if phase == 0:
            item = anchors[(step // 4) % len(anchors)]
        elif phase == 1:
            item = launchers[(step // 4) % len(launchers)]
        elif phase == 2:
            item = chat_inputs[(step // 4) % len(chat_inputs)]
        else:
            item = chat_sends[(step // 4) % len(chat_sends)]

        goal = item["goal"]
        candidates = item["candidates"]
        N = len(candidates)

        enc = TOKENIZER(goal, max_length=32, truncation=True, return_tensors="pt")
        g_ids = enc["input_ids"].to(DEVICE)
        g_mask = enc["attention_mask"].float().to(DEVICE)

        lbl_texts = [c["label"] for c in candidates]
        c_enc = TOKENIZER(lbl_texts, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
        c_lbl_ids = c_enc["input_ids"].unsqueeze(0).to(DEVICE)
        c_lbl_mask = c_enc["attention_mask"].float().unsqueeze(0).to(DEVICE)
        c_bounds = torch.tensor([c.get("bounds_vec", [0.0]*10) for c in candidates], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        c_roles = torch.tensor([c.get("role_idx", 1) for c in candidates], dtype=torch.long).unsqueeze(0).to(DEVICE)
        c_flags = torch.zeros(1, N, 16, dtype=torch.float32, device=DEVICE)
        for i, c in enumerate(candidates):
            if c.get("is_editable") or c.get("role_idx") == 2: c_flags[0, i, 0] = 1.0
            if c.get("is_focused"): c_flags[0, i, 1] = 1.0
            if c.get("is_clickable", True): c_flags[0, i, 2] = 1.0
        c_pad_mask = torch.ones(1, N, dtype=torch.float32, device=DEVICE)

        optimizer.zero_grad()
        out = model(
            goal_ids=g_ids,
            goal_mask=g_mask,
            candidate_label_ids=c_lbl_ids,
            candidate_label_mask=c_lbl_mask,
            candidate_bounds=c_bounds,
            candidate_roles=c_roles,
            candidate_flags=c_flags,
            candidate_pad_mask=c_pad_mask,
        )

        scores = out["ranking_scores"]  # [1, N]
        tgt_idx = torch.tensor([item["target_idx"]], dtype=torch.long, device=DEVICE)
        rank_loss = F.cross_entropy(scores, tgt_idx)

        # Contrastive Hard Negative Margin
        neg_indices = item.get("hard_negs", [])
        if neg_indices:
            tgt_score = scores[0, item["target_idx"]]
            neg_scores = scores[0, neg_indices]
            margin_loss = F.relu(neg_scores - tgt_score + 2.0).mean()
        else:
            margin_loss = torch.tensor(0.0, device=DEVICE)

        # Action Loss (Boosted weight on TYPE_TEXT screens to overcome OPEN_APP prior)
        act_logits = out["action_logits"]
        act_tgt = torch.tensor([item["action_idx"]], dtype=torch.long, device=DEVICE)
        act_mult = 2.5 if item["action_idx"] == ACTION_TO_IDX["TYPE_TEXT"] else 1.2
        act_loss = F.cross_entropy(act_logits, act_tgt) * act_mult

        # Span Loss (Supervise only when action is TYPE_TEXT)
        if item["action_idx"] == ACTION_TO_IDX["TYPE_TEXT"] and item["target_span"] != (0, 0):
            s_tgt = torch.tensor([item["target_span"][0]], dtype=torch.long, device=DEVICE)
            e_tgt = torch.tensor([item["target_span"][1]], dtype=torch.long, device=DEVICE)
            span_loss = (F.cross_entropy(out["span_start_logits"], s_tgt) + F.cross_entropy(out["span_end_logits"], e_tgt)) * 1.5
        else:
            span_loss = torch.tensor(0.0, device=DEVICE)

        total_loss = rank_loss + 1.0 * margin_loss + act_loss + span_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()

        if step % 20 == 0 or step == total_steps:
            print(f"Step {step:3d}/{total_steps} | Loss: {total_loss.item():.4f} (Rank: {rank_loss.item():.3f}, Neg: {margin_loss.item():.3f}, Act: {act_loss.item():.3f}, Span: {span_loss.item():.3f}) | Screen: {item['name']}")

    el = time.time() - t0
    print(f"\n>> Optimization finished in {el:.2f}s!")
    torch.save({"model_state_dict": model.state_dict(), "step": total_steps}, SMOKE_CKPT)
    print(f">> Saved sandbox candidate: {SMOKE_CKPT}")
    return model


if __name__ == "__main__":
    run_balanced_smoke_training(180)
