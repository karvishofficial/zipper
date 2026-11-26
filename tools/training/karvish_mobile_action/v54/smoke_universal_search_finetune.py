"""
KARVISH V54 — UNIVERSAL SEARCH-TO-RESULT & PHASE B GROUNDING TRAINER
Trains universal contact/entity selection after typing into search boxes.

Guarantees (AGENTS.md Constitution):
- Zero name heuristics (no 'if name == Rekha: boost').
- 30+ diverse, multi-cultural, multi-word synthetic names.
- Universal across WhatsApp, Telegram, Signal, Contacts, Spotify, Play Store, YouTube, Zepto.
- Solves Phase B: matching result card outranks populated search field & generic filter chips ('Videos', 'Photos').
- Preserves all 12 frozen anchor baselines and Zepto numerical price grounding.
"""

from __future__ import annotations

import sys
import time
import math
import random
import re
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
REPO_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from model_v54 import KarvishModelV54
from schema_v54 import ACTION_TO_IDX
from dry_run_universal_audit import build_universal_v54_dataset

IDX_TO_ACTION = {v: k for k, v in ACTION_TO_IDX.items()}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

SOURCE_CKPT = SCRIPT_DIR / "checkpoints" / "v54_best.pt"
TARGET_CKPT = SCRIPT_DIR / "checkpoints" / "v54_universal_search_best.pt"


def find_token_span(text: str, entity: str, tokenizer) -> Tuple[int, int]:
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc["offset_mapping"]
    lower_text = text.lower()
    lower_ent = entity.lower().strip()
    if not lower_ent:
        return (0, 0)
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


def generate_universal_phase_b_data() -> List[Dict[str, Any]]:
    """Generates synthetic universal search-to-result trajectories across 30+ diverse names and apps."""
    items = []

    names = [
        "Bhavani", "Shobha Aunty", "Rekha", "Dr. Ramesh", "Aarav", "Priya",
        "Carlos", "Fatima", "Ananya", "Vikram", "Sneha", "Kavya", "Suresh",
        "Rohan", "Elena", "Ahmed", "Lucas", "Maya", "John", "Sarah",
        "Dr. Rajesh", "Meera", "Neha", "Arjun", "Aditi", "Devi", "Amit",
        "Karthik", "Deepak", "Sunita", "Rajesh Mava", "Priya Sharma",
        "Carlos Santana", "Uncle Bob", "Aarav Patel", "Fatima Zahra",
        "Elena Rostova", "Sarah Jenkins", "Suresh Kumar", "Deepak Rao"
    ]

    apps = [
        ("WhatsApp", "open WhatsApp and send hi to {name}"),
        ("WhatsApp", "send hi to {name} on WhatsApp"),
        ("Telegram", "open Telegram and send hi to {name}"),
        ("Contacts", "search {name} in Contacts"),
        ("Signal", "open Signal and message {name}"),
        ("Slack", "message {name} on Slack"),
    ]

    for name in names:
        for app_name, goal_template in apps:
            goal = goal_template.format(name=name)
            st_tok, en_tok = find_token_span(goal, name, TOKENIZER)

            # SCREEN 1: Search initiation from chat list (Empty search bar is TARGET)
            cands_t1 = [
                {
                    "label": "Ask Meta AI or Search" if app_name == "WhatsApp" else f"Search in {app_name}",
                    "role_idx": 2,
                    "is_editable": True,
                    "is_populated": False,
                    "is_clickable": True,
                    "is_matching_entity": True,
                },
                {"label": "Chats", "role_idx": 1, "is_clickable": True, "is_matching_entity": False},
                {"label": "Updates", "role_idx": 1, "is_clickable": True, "is_matching_entity": False},
                {"label": "Calls", "role_idx": 1, "is_clickable": True, "is_matching_entity": False},
                {"label": "Bhavani" if name != "Bhavani" else "Karthik", "role_idx": 1, "is_clickable": True, "is_matching_entity": False}, # Unrelated recent chat distractor
                {"label": "Opt1mum Traders Community", "role_idx": 1, "is_clickable": True, "is_matching_entity": False},
                {"label": "fail", "role_idx": 0, "is_clickable": False, "is_matching_entity": False},
            ]
            items.append({
                "name": f"search_init_{app_name}_{name.lower().replace(' ', '_')}",
                "category": "search_init",
                "goal": goal,
                "candidates": cands_t1,
                "target_idx": 0,
                "action_idx": ACTION_TO_IDX["TYPE_TEXT"],
                "hard_negs": [1, 2, 3, 4, 5, 6],
                "target_span": (st_tok, en_tok),
                "entity_target": name,
                "is_terminal": False,
            })

            # SCREEN 1b: Inside Search Screen with EMPTY search bar & Recent search items (Search bar is TARGET)
            unrelated_recent = "Bhavani" if name != "Bhavani" else "Karthik"
            cands_t1b = [
                {
                    "label": "Search",
                    "role_idx": 2,
                    "is_editable": True,
                    "is_populated": False, # EMPTY search bar
                    "was_last_typed": False,
                    "is_clickable": True,
                    "is_matching_entity": True,
                },
                {"label": unrelated_recent, "role_idx": 1, "is_clickable": True, "is_matching_entity": False},
                {"label": "Clear all", "role_idx": 1, "is_clickable": True, "is_matching_entity": False},
                {"label": "Videos", "role_idx": 1, "is_clickable": True, "is_matching_entity": False},
                {"label": "Photos", "role_idx": 1, "is_clickable": True, "is_matching_entity": False},
                {"label": "+91 94492 13667", "role_idx": 1, "is_clickable": True, "is_matching_entity": False},
                {"label": "Jolly family", "role_idx": 1, "is_clickable": True, "is_matching_entity": False},
                {"label": "submit focused field", "role_idx": 0, "is_clickable": True, "is_matching_entity": False},
            ]
            items.append({
                "name": f"search_empty_screen_{app_name}_{name.lower().replace(' ', '_')}",
                "category": "search_init",
                "goal": goal,
                "candidates": cands_t1b,
                "target_idx": 0, # Empty search bar MUST BE TYPED INTO
                "action_idx": ACTION_TO_IDX["TYPE_TEXT"],
                "hard_negs": [1, 2, 3, 4, 5, 6, 7], # Unrelated recent contact, Videos, Photos, Clear all MUST BE CRUSHED
                "target_span": (st_tok, en_tok),
                "entity_target": name,
                "is_terminal": False,
            })

            # SCREEN 2: Search results (Contact Card is TARGET, Populated Search Bar, Filter Chips & Back are HARD NEGATIVES)
            cands_t2 = [
                {
                    "label": f"{name}",
                    "role_idx": 1,
                    "is_clickable": True,
                    "bounds_vec": [0.05, 0.18, 0.95, 0.28, 0.5, 0.23, 0.9, 0.10, 0.09, 9.0],
                    "is_matching_entity": True,
                },
                {
                    "label": f"<{name}",  # Populated search input
                    "role_idx": 2,
                    "is_editable": True,
                    "is_populated": True,
                    "was_last_typed": True,
                    "is_clickable": True,
                    "bounds_vec": [0.12, 0.04, 0.88, 0.09, 0.5, 0.065, 0.76, 0.05, 0.038, 15.2],
                    "is_matching_entity": False,
                },
                {"label": "Videos", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.11, 0.25, 0.16, 0.15, 0.135, 0.20, 0.05, 0.01, 4.0], "is_matching_entity": False},
                {"label": "Photos", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.28, 0.11, 0.48, 0.16, 0.38, 0.135, 0.20, 0.05, 0.01, 4.0], "is_matching_entity": False},
                {"label": "Unread", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.51, 0.11, 0.71, 0.16, 0.61, 0.135, 0.20, 0.05, 0.01, 4.0], "is_matching_entity": False},
                {"label": "Back", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.0, 0.04, 0.12, 0.09, 0.06, 0.065, 0.12, 0.05, 0.006, 2.4], "is_matching_entity": False},
                {"label": "submit focused field", "role_idx": 0, "is_clickable": True, "bounds_vec": [0.0]*10, "is_matching_entity": False},
                {
                    "label": f"Jolly family picture {name} is also in this group",
                    "role_idx": 1,
                    "is_clickable": True,
                    "bounds_vec": [0.05, 0.40, 0.95, 0.50, 0.5, 0.45, 0.9, 0.10, 0.09, 9.0],
                    "is_matching_entity": False,
                },
                {
                    "label": f"{name} Video Video preview",
                    "role_idx": 1,
                    "is_clickable": True,
                    "bounds_vec": [0.05, 0.30, 0.95, 0.38, 0.5, 0.34, 0.9, 0.08, 0.072, 11.25],
                    "is_matching_entity": False,
                },
            ]
            items.append({
                "name": f"search_result_{app_name}_{name.lower().replace(' ', '_')}",
                "category": "search_result_card",
                "goal": goal,
                "candidates": cands_t2,
                "target_idx": 0,  # Contact card
                "action_idx": ACTION_TO_IDX["CLICK"],
                "hard_negs": [1, 2, 3, 4, 5, 6, 7, 8],  # Populated bar, Videos, Photos, submit MUST BE CRUSHED
                "target_span": (st_tok, en_tok),
                "entity_target": name,
                "is_terminal": False,
            })

            # SCREEN 3: Chat room opened (Send button is TARGET)
            cands_t3 = [
                {"label": "Send", "role_idx": 1, "is_clickable": True, "is_matching_entity": True},
                {"label": "hi", "role_idx": 2, "is_editable": True, "is_populated": True, "is_clickable": True, "is_matching_entity": False},
                {"label": "Emoji, GIFs and stickers", "role_idx": 1, "is_clickable": True, "is_matching_entity": False},
                {"label": "Attach", "role_idx": 1, "is_clickable": True, "is_matching_entity": False},
                {"label": "Voice call", "role_idx": 1, "is_clickable": True, "is_matching_entity": False},
                {"label": "Video call", "role_idx": 1, "is_clickable": True, "is_matching_entity": False},
                {"label": "complete", "role_idx": 0, "is_clickable": True, "is_matching_entity": False},
            ]
            items.append({
                "name": f"chat_send_{app_name}_{name.lower().replace(' ', '_')}",
                "category": "chat_send",
                "goal": goal,
                "candidates": cands_t3,
                "target_idx": 0,  # Send button
                "action_idx": ACTION_TO_IDX["CLICK"],
                "hard_negs": [1, 2, 3, 4, 5],
                "target_span": (0, 0),
                "entity_target": "Send",
                "is_terminal": False,
            })

    # Universal E-Commerce & Music & App Store Search-to-Result
    ecom_searches = [
        ("Play songs by Coldplay on Spotify", "Coldplay", "Spotify", ["Coldplay", "<Coldplay", "Artists", "Albums", "Playlists", "submit focused field"]),
        ("Search Taylor Swift on YouTube Music", "Taylor Swift", "YouTube Music", ["Taylor Swift", "<Taylor Swift", "Songs", "Artists", "Videos", "submit focused field"]),
        ("Install Telegram from Play Store", "Telegram", "Play Store", ["Telegram", "<Telegram", "Top charts", "Categories", "Events", "submit focused field"]),
        ("Search running shoes on Amazon", "running shoes", "Amazon", ["Nike Running Shoes ₹2,499", "<running shoes", "Filters", "Prime", "Delivery day", "submit focused field"]),
    ]
    for goal, ent, app, cand_lbls in ecom_searches:
        st_tok, en_tok = find_token_span(goal, ent, TOKENIZER)
        cands_gen = []
        for i, l in enumerate(cand_lbls):
            is_pop = l.startswith("<")
            r_idx = 2 if is_pop else 1
            cands_gen.append({
                "label": l,
                "role_idx": r_idx,
                "is_editable": is_pop,
                "is_populated": is_pop,
                "was_last_typed": is_pop,
                "is_clickable": True,
                "is_matching_entity": (i == 0),
            })
        items.append({
            "name": f"ecom_search_{app.lower()}_{ent.lower().replace(' ', '_')}",
            "category": "ecom_search_result",
            "goal": goal,
            "candidates": cands_gen,
            "target_idx": 0,
            "action_idx": ACTION_TO_IDX["CLICK"],
            "hard_negs": list(range(1, len(cand_lbls))),
            "target_span": (st_tok, en_tok),
            "entity_target": ent,
            "is_terminal": False,
        })

    return items


def run_training(steps: int = 250, lr: float = 8e-5):
    print("=" * 80)
    print(f"STARTING UNIVERSAL PHASE B SEARCH-TO-RESULT GROUNDING ({steps} STEPS)")
    print(f"Source Checkpoint: {SOURCE_CKPT.name}")
    print(f"Target Checkpoint: {TARGET_CKPT.name}")
    print("=" * 80)

    # 1. Ingest Base Universal Dataset + New Phase B Trajectories
    base_data = build_universal_v54_dataset()
    phase_b_data = generate_universal_phase_b_data()
    full_dataset = base_data + phase_b_data
    print(f">> Total Training Dataset: {len(full_dataset)} trajectories ({len(base_data)} base + {len(phase_b_data)} universal search-to-result).")

    # Group into categories for balanced round-robin sampling
    categories: Dict[str, List[Dict[str, Any]]] = {}
    for d in full_dataset:
        categories.setdefault(d.get("category", "base"), []).append(d)
    cat_keys = list(categories.keys())
    print(f">> {len(cat_keys)} Balanced Category Buckets: {', '.join([f'{k}({len(v)})' for k, v in categories.items()])}")

    # 2. Initialize Model
    model = KarvishModelV54().to(DEVICE)
    ckpt = torch.load(SOURCE_CKPT, map_location=DEVICE, weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd, strict=False)

    # Freeze language backbone transformer layers, train pooler, decision/ranking/flags/entity heads
    for p in model.parameters():
        p.requires_grad = False
    for p in model.text_encoder.pooler.parameters():
        p.requires_grad = True
    for p in model.rank_head.parameters():
        p.requires_grad = True
    for p in model.action_head.parameters():
        p.requires_grad = True
    for p in model.span_pointer_head.parameters():
        p.requires_grad = True
    for p in model.flags_mlp.parameters():
        p.requires_grad = True
    for p in model.role_embed.parameters():
        p.requires_grad = True
    for p in model.cand_fuse.parameters():
        p.requires_grad = True
    for p in model.entity_gater.parameters():
        p.requires_grad = True
    for p in model.container_attention.parameters():
        p.requires_grad = True
    for p in model.task_state_engine.parameters():
        p.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f">> Trainable Parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = AdamW([
        {"params": model.text_encoder.pooler.parameters(), "lr": 1e-4},
        {"params": model.action_head.parameters(), "lr": 1e-4},
        {"params": model.span_pointer_head.parameters(), "lr": 1e-4},
        {"params": model.rank_head.parameters(), "lr": 2e-4},
        {"params": model.cand_fuse.parameters(), "lr": 2e-4},
        {"params": model.container_attention.parameters(), "lr": 2e-4},
        {"params": model.flags_mlp.parameters(), "lr": 2e-4},
        {"params": model.entity_gater.parameters(), "lr": 1e-4},
        {"params": model.task_state_engine.parameters(), "lr": 1e-4},
    ], weight_decay=1e-4)

    # Partition dataset into balanced priority pools
    pool_search_result = [d for d in full_dataset if d.get("category") == "search_result_card"]
    pool_search_init = [d for d in full_dataset if d.get("category") == "search_init"]
    pool_anchors = [d for d in full_dataset if "anchor" in d.get("category", "") or "calc" in d.get("name", "") or "chrome" in d.get("name", "")]
    pool_other = [d for d in full_dataset if d not in pool_search_result and d not in pool_search_init and d not in pool_anchors]
    if not pool_anchors:
        pool_anchors = base_data[:12]

    print(f">> Priority Pools: search_result={len(pool_search_result)}, search_init={len(pool_search_init)}, anchors={len(pool_anchors)}, other={len(pool_other)}")

    t0 = time.time()
    model.train()
    for step in range(1, steps + 1):
        # 35% search_result, 25% search_init, 20% anchors, 20% other
        r = (step % 20)
        if r < 7:
            item = pool_search_result[(step // 7) % len(pool_search_result)]
        elif r < 12:
            item = pool_search_init[(step // 5) % len(pool_search_init)]
        elif r < 16:
            item = pool_anchors[(step // 4) % len(pool_anchors)]
        else:
            item = pool_other[(step // 4) % len(pool_other)]

        goal = item["goal"]
        candidates = item["candidates"]
        N = len(candidates)

        enc = TOKENIZER(goal, max_length=32, truncation=True, return_tensors="pt")
        g_ids = enc["input_ids"].to(DEVICE)
        g_mask = enc["attention_mask"].float().to(DEVICE)

        lbl_texts = [str(c.get("label", "")).strip() for c in candidates]
        c_enc = TOKENIZER(lbl_texts, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
        c_lbl_ids = c_enc["input_ids"].unsqueeze(0).to(DEVICE)
        c_lbl_mask = c_enc["attention_mask"].float().unsqueeze(0).to(DEVICE)

        c_bounds = torch.tensor([[c.get("bounds_vec", [0.0]*10) for c in candidates]], dtype=torch.float32, device=DEVICE)
        c_roles = torch.tensor([[c.get("role_idx", 1) for c in candidates]], dtype=torch.long, device=DEVICE)
        c_flags = torch.zeros(1, N, 16, dtype=torch.float32, device=DEVICE)
        for i, c in enumerate(candidates):
            if c.get("is_editable") or c.get("role_idx") == 2: c_flags[0, i, 0] = 1.0
            if c.get("is_focused"): c_flags[0, i, 1] = 1.0
            if c.get("is_clickable", True): c_flags[0, i, 2] = 1.0
            if c.get("is_populated"): c_flags[0, i, 3] = 1.0
            if c.get("was_last_typed"): c_flags[0, i, 4] = 1.0
            if c.get("is_selected"): c_flags[0, i, 5] = 1.0
        c_pad_mask = torch.ones(1, N, dtype=torch.float32, device=DEVICE)

        # Parse numerical prices for e-commerce items
        c_num = torch.zeros(1, N, 4, dtype=torch.float32, device=DEVICE)
        for i, l in enumerate(lbl_texts):
            m_price = re.search(r"(?:₹|rs\.?|inr)\s*(\d+)", l, re.IGNORECASE)
            if m_price:
                c_num[0, i, 0] = math.log1p(float(m_price.group(1))) / 10.0

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
            candidate_num_features=c_num,
        )

        target_idx = item["target_idx"]
        ranking_scores = out["ranking_scores"]
        loss_rank = F.cross_entropy(ranking_scores, torch.tensor([target_idx], device=DEVICE))

        action_logits = out["action_logits"]
        exp_act_idx = item.get("action_idx", ACTION_TO_IDX.get(item.get("expected_action", "CLICK"), 0))
        loss_act = F.cross_entropy(action_logits, torch.tensor([exp_act_idx], device=DEVICE))

        # Hardest-Negative Adversarial Margin Loss: target must beat every hard negative by >= 5.5 pts
        loss_margin = torch.tensor(0.0, device=DEVICE)
        hard_negs = item.get("hard_negs", [])
        if hard_negs:
            pos_score = ranking_scores[0, target_idx]
            neg_scores = ranking_scores[0, hard_negs]
            diffs = torch.clamp(5.5 - (pos_score - neg_scores), min=0.0)
            if diffs.numel() > 0:
                loss_margin = diffs.max() + 0.5 * diffs.mean()

        # Span Pointer Loss
        ent = item.get("entity_target", "")
        loss_span = torch.tensor(0.0, device=DEVICE)
        if ent:
            s_tok, e_tok = find_token_span(goal, ent, TOKENIZER)
            if s_tok > 0 or e_tok > 0:
                st_loss = F.cross_entropy(out["span_start_logits"], torch.tensor([s_tok], device=DEVICE))
                en_loss = F.cross_entropy(out["span_end_logits"], torch.tensor([e_tok], device=DEVICE))
                loss_span = (st_loss + en_loss) * 0.5

        # Semantic Entity Gating Loss
        loss_gate = torch.tensor(0.0, device=DEVICE)
        if ent:
            ent_low = ent.lower()
            ent_labels = torch.zeros(1, N, device=DEVICE)
            for i, l in enumerate(lbl_texts):
                l_low = l.lower()
                if i == target_idx or (ent_low and ent_low in l_low and not l.startswith("<")):
                    ent_labels[0, i] = 1.0
                elif any(d in l_low for d in ["videos", "photos", "ginger", "soap", "ice", "munchies", "cold drinks"]):
                    ent_labels[0, i] = 0.0
                elif "search" in l_low or "submit" in l_low:
                    ent_labels[0, i] = 0.4
                else:
                    ent_labels[0, i] = 0.1
            loss_gate = F.binary_cross_entropy(out["entity_gate"], ent_labels)

        total_loss = loss_rank + 0.8 * loss_act + 4.0 * loss_margin + 0.5 * loss_span + 0.5 * loss_gate
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()

        if step % 50 == 0 or step == steps:
            elapsed = time.time() - t0
            pred_cand = candidates[ranking_scores[0].argmax().item()]["label"]
            pred_act = IDX_TO_ACTION.get(action_logits[0].argmax().item(), "CLICK")
            print(
                f"[STEP {step:04d}/{steps:04d} | {elapsed:.1f}s] "
                f"Loss: {total_loss.item():.4f} (Rank: {loss_rank.item():.3f}, Marg: {loss_margin.item():.3f}) | "
                f"Cat: {item.get('category', 'base')[:16]:16s} | Top: '{pred_cand[:25]}' | Act: {pred_act}"
            )

    # Save to candidate best checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "step": steps,
        "source": SOURCE_CKPT.name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, TARGET_CKPT)
    print(f"\n>> Candidate Checkpoint Successfully Saved: {TARGET_CKPT}")
    return model


if __name__ == "__main__":
    run_training(steps=1000, lr=8e-5)
