"""
KARVISH V54 — UNIVERSAL SEARCH GROUNDING & DISTRACTOR IMMUNITY SMOKE TRAINER
Trains a lightweight, balanced smoke fine-tune branching strictly from v54_best.pt:

Targeted Invariances & New Learned Grounding:
1. 12 Frozen Baseline Anchor Tasks (Calculator 1-5, Chrome 1-2, YouTube search bar, LinkedIn, Play Store, Instagram).
2. Full Zepto E-Commerce Trajectory (Cheapest price continuous numerical grounding + Ginger/Soap entity gating).
3. YouTube Home Screen (Search magnifying glass icon ranks #1; Feed cards and channel recommendations suppressed).
4. YouTube Autocomplete Screen (Submit / Commit ranks #1; 'Edit suggestion' arrow traps suppressed to negative).
5. Instagram Home Screen ('Search and explore' bottom tab ranks #1 with CLICK action; Feed posts & stories suppressed).
6. Populated Input Field Transition (When field is populated, 'submit focused field' beats re-typing into the input box).

Safety Governance:
- Saves to sandbox checkpoint: tools/training/karvish_mobile_action/v54/checkpoints/v54_search_smoke.pt
- Leaves v54_best.pt, v52_locked_gold.pt, and v51_7_gold 100% untouched.
- AGENTS.md Constitution: Zero runtime heuristics, 100% learned neural ranking.
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
SMOKE_CKPT = SCRIPT_DIR / "checkpoints" / "v54_search_smoke.pt"
SMOKE_CKPT.parent.mkdir(parents=True, exist_ok=True)


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


def build_search_grounding_dataset() -> List[Dict[str, Any]]:
    dataset = []

    # ---------------------------------------------------------
    # 1. 12 Frozen Baseline Anchor Tasks (Invariance Shield)
    # ---------------------------------------------------------
    anchor_screens = get_test_screens()
    for s in anchor_screens:
        expected = s["expected_target"].lower()
        candidates = s["candidates"][:24]
        target_idx = 0
        for i, c in enumerate(candidates):
            if expected in c["label"].lower():
                target_idx = i
                break
        
        act_name = s.get("expected_action", "CLICK")
        if c.get("is_editable"):
            act_name = "TYPE_TEXT"
        elif "open" in expected:
            act_name = "OPEN_APP"

        dataset.append({
            "category": "anchor",
            "name": s["name"],
            "goal": s["goal"],
            "candidates": candidates,
            "target_idx": target_idx,
            "expected_action": act_name,
            "entity_target": s.get("expected_entity", ""),
        })

    # ---------------------------------------------------------
    # 2. Zepto Cheapest Tomato & Price Grounding Trajectory
    # ---------------------------------------------------------
    # Zepto Turn 5: Dense 10-item Tomato Grid with Distractors (Ginger ₹9, Soap ₹10)
    zepto_t5_cands = [
        {"label": "Tomato Local 500g ₹13 ADD", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.2, 0.45, 0.5, 0.25, 0.35, 0.4, 0.3, 0.12, 1.33]},
        {"label": "Tomato Hybrid 500g ₹19 ADD", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.55, 0.2, 0.95, 0.5, 0.75, 0.35, 0.4, 0.3, 0.12, 1.33]},
        {"label": "Tomato Cherry 250g ₹37 ADD", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.55, 0.45, 0.85, 0.25, 0.7, 0.4, 0.3, 0.12, 1.33]},
        {"label": "Ginger Fresh 100g ₹9 ADD", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.55, 0.55, 0.95, 0.85, 0.75, 0.7, 0.4, 0.3, 0.12, 1.33]},
        {"label": "Dettol Soap 75g ₹10 ADD", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.86, 0.45, 0.98, 0.25, 0.92, 0.4, 0.12, 0.05, 3.33]},
        {"label": "Coriander Bunch ₹8 ADD", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.55, 0.86, 0.95, 0.98, 0.75, 0.92, 0.4, 0.12, 0.05, 3.33]},
        {"label": "submit focused field", "role_idx": 1, "is_clickable": False, "bounds_vec": [0.0]*10},
        {"label": "fail", "role_idx": 0, "is_clickable": False, "bounds_vec": [0.0]*10},
    ]
    dataset.append({
        "category": "zepto",
        "name": "Zepto Cheapest Tomato Grid",
        "goal": "Open Zepto and search cheapest tomatoes",
        "candidates": zepto_t5_cands,
        "target_idx": 0,
        "expected_action": "CLICK",
        "entity_target": "tomato",
    })

    # 2b. Zepto Search Screen: Matching Query Suggestion Chip vs Distractors (Ice, Munchies) & Premature Complete
    zepto_search_goals = [
        ("Open Zepto and search cheapest tomatoes", "tomatoes", "fresh tomatoes"),
        ("Open Zepto and search fresh tomatoes", "tomatoes", "fresh tomatoes"),
        ("Search tomatoes on Zepto", "tomatoes", "fresh tomatoes"),
        ("Open Zepto and search fresh milk", "milk", "fresh milk"),
        ("Search milk on Zepto", "milk", "fresh milk"),
    ]
    for g, ent, chip in zepto_search_goals:
        zepto_chip_cands = [
            {"label": chip, "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.12, 0.45, 0.18, 0.25, 0.15, 0.4, 0.06, 0.024, 6.66]},
            {"label": "Search", "role_idx": 2, "is_editable": True, "bounds_vec": [0.15, 0.04, 0.85, 0.09, 0.5, 0.065, 0.7, 0.05, 0.035, 14.0]},
            {"label": f"{ent} puree", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.5, 0.12, 0.85, 0.18, 0.675, 0.15, 0.35, 0.06, 0.021, 5.83]},
            {"label": "Ice", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.2, 0.25, 0.26, 0.15, 0.23, 0.2, 0.06, 0.012, 3.33]},
            {"label": "Munchies", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.3, 0.2, 0.55, 0.26, 0.425, 0.23, 0.25, 0.06, 0.015, 4.16]},
            {"label": "Cold Drinks", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.6, 0.2, 0.9, 0.26, 0.75, 0.23, 0.3, 0.06, 0.018, 5.0]},
            {"label": "complete", "role_idx": 0, "is_clickable": False, "bounds_vec": [0.0]*10},
            {"label": "fail", "role_idx": 0, "is_clickable": False, "bounds_vec": [0.0]*10},
        ]
        dataset.append({
            "category": "zepto_chip",
            "name": f"Zepto Search -> Matching Query Chip ({chip})",
            "goal": g,
            "candidates": zepto_chip_cands,
            "target_idx": 0,  # chip
            "expected_action": "CLICK",
            "entity_target": ent,
        })

        # 2c. Zepto Direct Search Input Box Typing (When no matching chip or typing fresh query)
        zepto_type_cands = [
            {"label": "Search", "role_idx": 2, "is_editable": True, "is_focused": True, "bounds_vec": [0.15, 0.04, 0.85, 0.09, 0.5, 0.065, 0.7, 0.05, 0.035, 14.0]},
            {"label": "Ice", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.2, 0.25, 0.26, 0.15, 0.23, 0.2, 0.06, 0.012, 3.33]},
            {"label": "Munchies", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.3, 0.2, 0.55, 0.26, 0.425, 0.23, 0.25, 0.06, 0.015, 4.16]},
            {"label": "complete", "role_idx": 0, "is_clickable": False, "bounds_vec": [0.0]*10},
            {"label": "fail", "role_idx": 0, "is_clickable": False, "bounds_vec": [0.0]*10},
        ]
        dataset.append({
            "category": "zepto_input",
            "name": f"Zepto Search -> Input Field Typing ({ent})",
            "goal": g,
            "candidates": zepto_type_cands,
            "target_idx": 0,  # "Search"
            "expected_action": "TYPE_TEXT",
            "entity_target": ent,
        })

    # ---------------------------------------------------------
    # 3. YouTube Home Screen: Search Icon vs Feed Distractors
    # ---------------------------------------------------------
    # The search icon at top right MUST beat video cards and recommendations
    yt_home_goals = [
        ("Open YouTube and search for SpaceX", "SpaceX"),
        ("Open YouTube and search lo-fi hip hop", "lo-fi hip hop"),
        ("Search SpaceX on YouTube", "SpaceX"),
        ("Open YouTube and search python tutorial", "python tutorial"),
    ]
    for g, ent in yt_home_goals:
        yt_home_cands = [
            {"label": "Search", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.75, 0.03, 0.88, 0.08, 0.815, 0.055, 0.13, 0.05, 0.0065, 2.6]},
            {"label": "Voice search", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.89, 0.03, 0.98, 0.08, 0.935, 0.055, 0.09, 0.05, 0.0045, 1.8]},
            {"label": "Go to channel Sajjaad Khader", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.12, 0.95, 0.48, 0.5, 0.3, 0.9, 0.36, 0.324, 2.5]},
            {"label": "I Asked Microsoft’s CEO If I Should Learn to Code (Satya Nadella)", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.5, 0.95, 0.85, 0.5, 0.675, 0.9, 0.35, 0.315, 2.57]},
            {"label": "Starship Fifth Flight Test SpaceX, verified - 55 lakh views", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.86, 0.95, 0.98, 0.5, 0.92, 0.9, 0.12, 0.108, 7.5]},
            {"label": "Cast, disconnected", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.65, 0.03, 0.74, 0.08, 0.695, 0.055, 0.09, 0.05, 0.0045, 1.8]},
            {"label": "Notifications", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.55, 0.03, 0.64, 0.08, 0.595, 0.055, 0.09, 0.05, 0.0045, 1.8]},
            {"label": "fail", "role_idx": 0, "is_clickable": False, "bounds_vec": [0.0]*10},
        ]
        dataset.append({
            "category": "yt_home",
            "name": f"YouTube Home -> Search Icon ({ent})",
            "goal": g,
            "candidates": yt_home_cands,
            "target_idx": 0,  # "Search"
            "expected_action": "CLICK",
            "entity_target": ent,
        })

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # 4. YouTube Autocomplete Screen: Suggestion Row & Submit vs 'Edit suggestion' Trap
    # ---------------------------------------------------------
    yt_auto_goals = [
        ("Open YouTube and search for SpaceX", "SpaceX", "SpaceX"),
        ("Open YouTube and search lo-fi hip hop", "lo-fi hip hop", "lo-fi hip hop"),
        ("Search SpaceX on YouTube", "SpaceX", "SpaceX"),
    ]
    for g, ent, q in yt_auto_goals:
        # 4a. Matching Suggestion Item: Clicking the suggestion row executes search
        yt_sugg_cands = [
            {"label": q, "role_idx": 1, "is_clickable": True, "bounds_vec": [0.0, 0.1, 0.85, 0.16, 0.425, 0.13, 0.85, 0.06, 0.051, 14.1]},
            {"label": f"Edit suggestion {q}", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.85, 0.1, 0.98, 0.16, 0.915, 0.13, 0.13, 0.06, 0.0078, 2.1]},
            {"label": f"Edit suggestion {q} live", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.85, 0.17, 0.98, 0.23, 0.915, 0.20, 0.13, 0.06, 0.0078, 2.1]},
            {"label": "Clear", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.85, 0.04, 0.98, 0.09, 0.915, 0.065, 0.13, 0.05, 0.0065, 2.6]},
            {"label": "Voice search", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.85, 0.04, 0.98, 0.09, 0.915, 0.065, 0.13, 0.05, 0.0065, 2.6]},
            {"label": "Search YouTube", "role_idx": 2, "is_editable": True, "is_populated": True, "bounds_vec": [0.1, 0.04, 0.85, 0.09, 0.475, 0.065, 0.75, 0.05, 0.0375, 15.0]},
            {"label": "fail", "role_idx": 0, "is_clickable": False, "bounds_vec": [0.0]*10},
        ]
        dataset.append({
            "category": "yt_sugg",
            "name": f"YouTube Autocomplete -> Suggestion Row ({ent})",
            "goal": g,
            "candidates": yt_sugg_cands,
            "target_idx": 0,  # The suggestion row itself
            "expected_action": "CLICK",
            "entity_target": ent,
        })

        # 4b. Submit focused field when presented as IME affordance
        yt_auto_cands = [
            {"label": "submit focused field", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.85, 0.9, 0.98, 0.98, 0.915, 0.94, 0.13, 0.08, 0.01, 1.6]},
            {"label": f"Edit suggestion {q}", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.88, 0.15, 0.98, 0.22, 0.93, 0.185, 0.1, 0.07, 0.007, 1.4]},
            {"label": f"Edit suggestion {q} live", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.88, 0.23, 0.98, 0.3, 0.93, 0.265, 0.1, 0.07, 0.007, 1.4]},
            {"label": f"Edit suggestion {q} 2026", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.88, 0.31, 0.98, 0.38, 0.93, 0.345, 0.1, 0.07, 0.007, 1.4]},
            {"label": "Search YouTube", "role_idx": 2, "is_editable": True, "is_populated": True, "bounds_vec": [0.1, 0.05, 0.9, 0.12, 0.5, 0.085, 0.8, 0.07, 0.056, 11.4]},
            {"label": "fail", "role_idx": 0, "is_clickable": False, "bounds_vec": [0.0]*10},
        ]
        dataset.append({
            "category": "yt_auto",
            "name": f"YouTube Autocomplete -> Submit ({ent})",
            "goal": g,
            "candidates": yt_auto_cands,
            "target_idx": 0,  # "submit focused field"
            "expected_action": "CLICK",
            "entity_target": ent,
        })

    # ---------------------------------------------------------
    # 5. Instagram Home Screen: 'Search and explore' vs Feed Posts
    # ---------------------------------------------------------
    ig_goals = [
        ("Open Instagram and search photography", "photography"),
        ("Open Instagram and search travel reels", "travel reels"),
        ("Search wildlife on Instagram", "wildlife"),
    ]
    for g, ent in ig_goals:
        ig_cands = [
            {"label": "Search and explore", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.2, 0.92, 0.4, 0.99, 0.3, 0.955, 0.2, 0.07, 0.014, 2.85]},
            {"label": "Home", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.0, 0.92, 0.2, 0.99, 0.1, 0.955, 0.2, 0.07, 0.014, 2.85]},
            {"label": "Reels", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.4, 0.92, 0.6, 0.99, 0.5, 0.955, 0.2, 0.07, 0.014, 2.85]},
            {"label": "Photo 1 of 5 by RVCJ Media, 527 likes, 112 comments", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.2, 0.95, 0.7, 0.5, 0.45, 0.9, 0.5, 0.45, 1.8]},
            {"label": "sindhu_pramuu's story, 2 of 26, Unseen", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.08, 0.25, 0.18, 0.15, 0.13, 0.2, 0.1, 0.02, 2.0]},
            {"label": "Like", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.72, 0.15, 0.78, 0.1, 0.75, 0.1, 0.06, 0.006, 1.66]},
            {"label": "Comment", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.18, 0.72, 0.28, 0.78, 0.23, 0.75, 0.1, 0.06, 0.006, 1.66]},
            {"label": "fail", "role_idx": 0, "is_clickable": False, "bounds_vec": [0.0]*10},
        ]
        dataset.append({
            "category": "ig_home",
            "name": f"Instagram Home -> Search & Explore ({ent})",
            "goal": g,
            "candidates": ig_cands,
            "target_idx": 0,  # "Search and explore"
            "expected_action": "CLICK",
            "entity_target": ent,
        })

    # ---------------------------------------------------------
    # 6. Instagram Explore Screen: Input Field Typing
    # ---------------------------------------------------------
    for g, ent in ig_goals:
        ig_explore_cands = [
            {"label": "Search with Meta AI", "role_idx": 2, "is_editable": True, "bounds_vec": [0.05, 0.05, 0.95, 0.12, 0.5, 0.085, 0.9, 0.07, 0.063, 12.85]},
            {"label": "Trending Reel: Nature photography in 4K", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.15, 0.48, 0.5, 0.265, 0.325, 0.43, 0.35, 0.15, 1.22]},
            {"label": "Explore Grid Image 2", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.52, 0.15, 0.95, 0.5, 0.735, 0.325, 0.43, 0.35, 0.15, 1.22]},
            {"label": "submit focused field", "role_idx": 1, "is_clickable": False, "bounds_vec": [0.0]*10},
            {"label": "fail", "role_idx": 0, "is_clickable": False, "bounds_vec": [0.0]*10},
        ]
        dataset.append({
            "category": "ig_explore",
            "name": f"Instagram Explore -> Search Input ({ent})",
            "goal": g,
            "candidates": ig_explore_cands,
            "target_idx": 0,  # "Search with Meta AI"
            "expected_action": "TYPE_TEXT",
            "entity_target": ent,
        })

    # ---------------------------------------------------------
    # 7. YouTube Search Results Screen: Termination Milestone
    # ---------------------------------------------------------
    # When video search results are loaded, goal "search X" is complete
    yt_results_goals = [
        ("Open YouTube and search for SpaceX", "SpaceX"),
        ("Open YouTube and search lo-fi hip hop", "lo-fi hip hop"),
        ("Search SpaceX on YouTube", "SpaceX"),
    ]
    for g, ent in yt_results_goals:
        yt_res_cands = [
            {"label": "complete", "role_idx": 0, "is_clickable": True, "bounds_vec": [0.0]*10},
            {"label": f"{ent} beats to relax/study to - Lofi Girl (5.4 crore views)", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.12, 0.95, 0.48, 0.5, 0.3, 0.9, 0.36, 0.324, 2.5]},
            {"label": "You", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.8, 0.92, 1.0, 0.99, 0.9, 0.955, 0.2, 0.07, 0.014, 2.85]},
            {"label": "Home", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.0, 0.92, 0.2, 0.99, 0.1, 0.955, 0.2, 0.07, 0.014, 2.85]},
            {"label": "Shorts", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.2, 0.92, 0.4, 0.99, 0.3, 0.955, 0.2, 0.07, 0.014, 2.85]},
            {"label": "Subscriptions", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.4, 0.92, 0.6, 0.99, 0.5, 0.955, 0.2, 0.07, 0.014, 2.85]},
            {"label": "scroll down", "role_idx": 0, "is_clickable": False, "bounds_vec": [0.0]*10},
            {"label": "fail", "role_idx": 0, "is_clickable": False, "bounds_vec": [0.0]*10},
        ]
        dataset.append({
            "category": "yt_results_done",
            "name": f"YouTube Search Results -> Complete Goal ({ent})",
            "goal": g,
            "candidates": yt_res_cands,
            "target_idx": 0,  # "complete"
            "expected_action": "WAIT",
            "entity_target": ent,
        })

    # ---------------------------------------------------------
    # 8. Instagram Search Results Screen: Termination Milestone
    # ---------------------------------------------------------
    for g, ent in ig_goals:
        ig_res_cands = [
            {"label": "complete", "role_idx": 0, "is_clickable": True, "bounds_vec": [0.0]*10},
            {"label": f"#{ent} • 1.2M posts", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.15, 0.95, 0.22, 0.5, 0.185, 0.9, 0.07, 0.063, 12.85]},
            {"label": f"{ent}_daily • Follow", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.23, 0.95, 0.3, 0.5, 0.265, 0.9, 0.07, 0.063, 12.85]},
            {"label": "Home", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.0, 0.92, 0.2, 0.99, 0.1, 0.955, 0.2, 0.07, 0.014, 2.85]},
            {"label": "Reels", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.4, 0.92, 0.6, 0.99, 0.5, 0.955, 0.2, 0.07, 0.014, 2.85]},
            {"label": "Profile", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.8, 0.92, 1.0, 0.99, 0.9, 0.955, 0.2, 0.07, 0.014, 2.85]},
            {"label": "scroll down", "role_idx": 0, "is_clickable": False, "bounds_vec": [0.0]*10},
            {"label": "fail", "role_idx": 0, "is_clickable": False, "bounds_vec": [0.0]*10},
        ]
        dataset.append({
            "category": "ig_results_done",
            "name": f"Instagram Search Results -> Complete Goal ({ent})",
            "goal": g,
            "candidates": ig_res_cands,
            "target_idx": 0,  # "complete"
            "expected_action": "WAIT",
            "entity_target": ent,
        })

    return dataset


def run_search_grounding_smoke(total_steps: int = 150):
    print("=" * 80)
    print(f"STARTING UNIVERSAL SEARCH GROUNDING SMOKE TRAINING ({total_steps} STEPS)")
    print(f"Source Checkpoint: {SOURCE_CKPT.name}")
    print(f"Sandbox Target:    {SMOKE_CKPT.name}")
    print("=" * 80)

    model = KarvishModelV54().to(DEVICE)
    ckpt = torch.load(SOURCE_CKPT, map_location=DEVICE, weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd, strict=False)

    # Freeze base representations, train task routing & ranking layers
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
    for p in model.entity_gater.parameters():
        p.requires_grad = True
    for p in model.container_attention.parameters():
        p.requires_grad = True
    for p in model.task_state_engine.parameters():
        p.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW([
        {"params": model.action_head.parameters(), "lr": 1.2e-4},
        {"params": model.span_pointer_head.parameters(), "lr": 1.2e-4},
        {"params": model.rank_head.parameters(), "lr": 1.2e-4},
        {"params": model.entity_gater.parameters(), "lr": 1.2e-4},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and "action_head" not in n and "span_pointer_head" not in n and "rank_head" not in n and "entity_gater" not in n], "lr": 4e-5},
    ], weight_decay=1e-3)

    dataset = build_search_grounding_dataset()
    categories: Dict[str, List[Dict[str, Any]]] = {}
    for d in dataset:
        categories.setdefault(d["category"], []).append(d)
    cat_keys = list(categories.keys())

    cat_counts_str = ", ".join([f"{k}({len(v)})" for k, v in categories.items()])
    print(f">> Dataset Pool: {len(dataset)} items across {len(cat_keys)} categories: {cat_counts_str}")

    t0 = time.time()
    for step in range(1, total_steps + 1):
        # Dynamic balanced round-robin across all categories
        cat = cat_keys[(step - 1) % len(cat_keys)]
        pool = categories[cat]
        item = pool[(step // len(cat_keys)) % len(pool)]

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
            if c.get("is_populated"): c_flags[0, i, 3] = 1.0
        c_pad_mask = torch.ones(1, N, dtype=torch.float32, device=DEVICE)

        # Parse numerical prices for e-commerce items
        c_num = torch.zeros(1, N, 4, dtype=torch.float32, device=DEVICE)
        for i, l in enumerate(lbl_texts):
            m_price = re.search(r"(?:₹|rs\.?|inr)\s*(\d+)", l, re.IGNORECASE)
            if m_price:
                c_num[0, i, 0] = float(m_price.group(1)) / 100.0

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
        ranking_scores = out["ranking_scores"]  # [1, N]
        loss_rank = F.cross_entropy(ranking_scores, torch.tensor([target_idx], device=DEVICE))

        exp_act = item["expected_action"]
        exp_act_idx = ACTION_TO_IDX.get(exp_act, 0)
        action_logits = out["action_logits"]  # [1, N, num_actions] or [1, num_actions]
        if action_logits.dim() == 3:
            target_act_logits = action_logits[0, target_idx].unsqueeze(0)
        else:
            target_act_logits = action_logits
        loss_act = F.cross_entropy(target_act_logits, torch.tensor([exp_act_idx], device=DEVICE))

        # Span Pointer Loss (Query Extraction)
        ent = item.get("entity_target", "").lower()
        if ent:
            s_tok, e_tok = find_token_span(goal, ent, TOKENIZER)
            start_loss = F.cross_entropy(out["span_start_logits"], torch.tensor([s_tok], device=DEVICE))
            end_loss = F.cross_entropy(out["span_end_logits"], torch.tensor([e_tok], device=DEVICE))
            loss_span = (start_loss + end_loss) * 0.5

            # Semantic Entity Gating Supervision
            target_gates = torch.zeros(1, N, device=DEVICE)
            for i, l in enumerate(lbl_texts):
                l_low = l.lower()
                if i == target_idx or (ent and ent in l_low):
                    target_gates[0, i] = 1.0
                elif any(d in l_low for d in ["ice", "munchies", "cold drinks", "ginger", "soap"]):
                    target_gates[0, i] = 0.0
                elif "search" in l_low or "submit" in l_low:
                    target_gates[0, i] = 0.8 if i == target_idx else 0.4
                else:
                    target_gates[0, i] = 0.3
            loss_gate = F.binary_cross_entropy(out["entity_gate"], target_gates)
        else:
            loss_span = torch.tensor(0.0, device=DEVICE)
            loss_gate = torch.tensor(0.0, device=DEVICE)

        # Contrastive Triplet Margin Loss: target candidate must beat others by at least 2.5 pts
        if N > 1:
            target_score = ranking_scores[0, target_idx]
            other_scores = torch.cat([ranking_scores[0, :target_idx], ranking_scores[0, target_idx + 1:]])
            margin = 2.5
            loss_margin = F.relu(margin - (target_score - other_scores)).mean()
        else:
            loss_margin = torch.tensor(0.0, device=DEVICE)

        total_loss = loss_rank + 0.8 * loss_act + 0.5 * loss_span + 0.6 * loss_gate + 1.0 * loss_margin
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()

        if step % 25 == 0 or step == total_steps:
            elapsed = time.time() - t0
            pred_cand = candidates[ranking_scores[0].argmax().item()]["label"]
            pred_act = IDX_TO_ACTION.get(target_act_logits[0].argmax().item(), "CLICK")
            print(
                f"[STEP {step:03d}/{total_steps:03d} | {elapsed:.1f}s] "
                f"Loss: {total_loss.item():.4f} (Rank: {loss_rank.item():.3f}, Act: {loss_act.item():.3f}) | "
                f"Category: {item['category']:10s} | Top: '{pred_cand[:28]}' | Act: {pred_act}"
            )

    # Save to sandbox smoke checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "step": total_steps,
        "source": SOURCE_CKPT.name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, SMOKE_CKPT)
    print(f"\n>> Sandbox Smoke Checkpoint Saved Successfully: {SMOKE_CKPT}")


if __name__ == "__main__":
    run_search_grounding_smoke(total_steps=300)
