"""
KARVISH V54 — FULL PRODUCTION TRAINING & COMPREHENSIVE CERTIFICATION
Trains 250 full optimization steps on:
1. All 12 Frozen Anchor Screens (Calculator 1-5, Chrome, YouTube, LinkedIn, Play Store, Instagram).
2. Live Zepto Home Screen with Floating Cart Bar (Search box ranked #1).
3. Above-the-fold Tomato Results Grid (Explore for cheapest: "scroll down" candidate).
4. Scrolled Tomato Results Grid with Cross-Sell Distractors (Ginger ₹9, Soap ₹10 rejected by Entity Gate).
5. Numerical Comparative Grounding (Continuous monotonic magnitude: ₹13 vs ₹28).
6. Programmatic Contrastive Numerical Permutations (Generalizes monotonic price ordering to unseen numbers).
7. Terminal Milestone State Transition (Cart counter = 1 -> "complete" candidate with WAIT action).

Saves strictly to: tools/training/karvish_mobile_action/v54/checkpoints/v54_best.pt
Leaves v52_locked_gold.pt and v51_7_gold completely untouched.
"""

from __future__ import annotations

import sys
import time
import re
import math
import random
import xml.etree.ElementTree as ET
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

from model_v54 import KarvishModelV54
from schema_v54 import ACTION_TO_IDX
IDX_TO_ACTION = {v: k for k, v in ACTION_TO_IDX.items()}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

LOCKED_GOLD_CKPT = SCRIPT_DIR.parent / "v52_locked_gold" / "v52_locked_gold.pt"
BEST_V54_CKPT = SCRIPT_DIR / "checkpoints" / "v54_best.pt"

ZEPTO_HOME_XML = REPO_ROOT / "tools" / "live_device" / "zepto_home_live.xml"
TOMATO_ABOVE_XML = REPO_ROOT / "tools" / "live_device" / "current_tomato.xml"
TOMATO_SCROLLED_XML = REPO_ROOT / "tools" / "live_device" / "scrolled_tomato.xml"
CART_XML = REPO_ROOT / "tools" / "live_device" / "cart.xml"

# Import anchor helpers from v52
sys.path.insert(0, str(SCRIPT_DIR.parent / "v52"))
from probe_v52_screen_decisions import get_test_screens, evaluate_screen


def find_token_span(goal: str, target_text: str, tokenizer: Any) -> Tuple[int, int]:
    goal_lower = goal.lower()
    t_lower = target_text.lower()
    char_start = goal_lower.find(t_lower)
    if char_start < 0:
        return 0, 0
    char_end = char_start + len(t_lower)
    enc = tokenizer(goal, max_length=32, truncation=True, return_offsets_mapping=True)
    offsets = enc.get("offset_mapping", [])
    start_tok, end_tok = None, None
    for tok_idx, (st, en) in enumerate(offsets):
        if st == en == 0:
            continue
        if start_tok is None and en > char_start:
            start_tok = tok_idx
        if start_tok is not None and st < char_end:
            end_tok = tok_idx
    if start_tok is None or end_tok is None:
        return 0, 0
    return start_tok, end_tok


def extract_candidate_price(c: Dict[str, Any]) -> float:
    if "price" in c and c["price"] is not None:
        return float(c["price"])
    lbl = c.get("label", "")
    # 1. Clean commas in currency: ₹1,299 -> ₹1299
    cleaned = re.sub(r'([₹]|rs\.?|inr)\s*([\d]+),([\d]+)', r'\1\2\3', lbl, flags=re.I)
    matches = re.findall(r"(?:₹|rs\.?|inr)\s*([\d]+(?:\.\d+)?)", cleaned, re.IGNORECASE)
    if matches:
        valid_nums = [float(x) for x in matches if 0.0 < float(x) < 100000.0]
        if valid_nums:
            # Payable selling price is the minimum price on the card (MRP is always higher)
            return min(valid_nums)
    return 0.0


def parse_xml_candidates(xml_path: Path) -> List[Dict[str, Any]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    parent_map = {c: p for p in root.iter() for c in p}
    cands = []
    seen = set()
    screen_w, screen_h = 1080.0, 2340.0

    for idx, el in enumerate(root.iter("node")):
        text = el.get("text", "").strip()
        desc = el.get("content-desc", "").strip()
        res = el.get("resource-id", "").strip()
        cls = el.get("class", "").strip()
        clickable = el.get("clickable") == "true"
        editable = el.get("editable") == "true" or "EditText" in cls or "input" in cls.lower()
        bounds_str = el.get("bounds", "")

        if not clickable and not editable:
            continue
        lbl = text if text else desc
        if not lbl:
            continue

        res_short = res.split("/")[-1] if res else ""

        # Contextualize generic add buttons with parent card details (Title, Price, Unit)
        if ("add-button" in res or lbl in ["Add to Cart", "ADD", "Add", "add"]) and not res.endswith("-minus") and not res.endswith("-plus"):
            curr = el
            card_desc = ""
            while curr in parent_map:
                curr = parent_map[curr]
                p_desc = curr.get("content-desc", "").strip()
                if p_desc and any(k in p_desc for k in ["₹", "Tomato", "Milk", "Bread", "Paneer", "OFF"]):
                    card_desc = p_desc
                    break
            if card_desc:
                clean_desc = re.sub(r'[\u200a-\u200f\ufeff]', '', card_desc)
                lbl = f"{clean_desc} Add to Cart"

        key = (lbl, bounds_str)
        if key in seen:
            continue
        seen.add(key)

        if not lbl and clickable:
            cls_short = cls.split(".")[-1]
            lbl = f"clickable_{cls_short}"

        m = re.findall(r"\[(\d+),(\d+)\]", bounds_str)
        if len(m) == 2:
            x1, y1 = float(m[0][0]), float(m[0][1])
            x2, y2 = float(m[1][0]), float(m[1][1])
            ymin = max(0.0, min(1.0, y1 / screen_h))
            xmin = max(0.0, min(1.0, x1 / screen_w))
            ymax = max(0.0, min(1.0, y2 / screen_h))
            xmax = max(0.0, min(1.0, x2 / screen_w))
            w = max(0.001, xmax - xmin)
            h = max(0.001, ymax - ymin)
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            area = w * h
            ar = w / h
            b_vec = [xmin, ymin, xmax, ymax, cx, cy, w, h, area, ar]
        else:
            b_vec = [0.0] * 10

        role_idx = 2 if editable else (1 if clickable else 0)
        cands.append({
            "id": f"c_{idx}_{res_short or 'node'}",
            "label": lbl,
            "res": res_short,
            "role": "input" if editable else ("button" if clickable else "text"),
            "role_idx": role_idx,
            "is_clickable": clickable,
            "is_editable": editable,
            "bounds_vec": b_vec,
        })
    return cands


def make_system_cand(cid: str, label: str, role: str = "button") -> Dict[str, Any]:
    return {
        "id": cid,
        "label": label,
        "res": cid,
        "role": role,
        "role_idx": 1 if role == "button" else 0,
        "is_clickable": True,
        "is_editable": False,
        "bounds_vec": [0.0, 0.0, 1.0, 0.05, 0.5, 0.025, 1.0, 0.05, 0.05, 20.0],
    }


def create_dense_30_candidate_screen() -> List[Dict[str, Any]]:
    # 1. Target: The cheapest tomato Local ₹13
    target = {
        "id": "c_tomato_13",
        "label": "Fresh Tomato Local 500g ₹13 Add to Cart",
        "res": "btn_add_13",
        "role": "button",
        "role_idx": 1,
        "is_clickable": True,
        "is_editable": False,
        "bounds_vec": [0.05, 0.20, 0.45, 0.35, 0.25, 0.275, 0.4, 0.15, 0.06, 2.6],
        "price": 13.0,
        "is_matching_entity": True,
    }

    # 2. Competing Tomato options with higher prices (11 tomatoes)
    competing_tomatoes = [
        {"id": "c_tom_18", "label": "Hybrid Tomato 500g ₹18 Add to Cart", "price": 18.0, "is_matching_entity": True},
        {"id": "c_tom_24", "label": "Farm Fresh Tomato 1kg ₹24 Add to Cart", "price": 24.0, "is_matching_entity": True},
        {"id": "c_tom_28", "label": "Organic Country Tomato 500g ₹28 Add to Cart", "price": 28.0, "is_matching_entity": True},
        {"id": "c_tom_32", "label": "Salad Tomato 250g ₹32 Add to Cart", "price": 32.0, "is_matching_entity": True},
        {"id": "c_tom_39", "label": "Cherry Tomato 250g ₹39 Add to Cart", "price": 39.0, "is_matching_entity": True},
        {"id": "c_tom_45", "label": "Hydroponic Vine Tomato 500g ₹45 Add to Cart", "price": 45.0, "is_matching_entity": True},
        {"id": "c_tom_52", "label": "Italian Plum Tomato 500g ₹52 Add to Cart", "price": 52.0, "is_matching_entity": True},
        {"id": "c_tom_68", "label": "Heritage Black Tomato 250g ₹68 Add to Cart", "price": 68.0, "is_matching_entity": True},
        {"id": "c_tom_75", "label": "Organic Roma Tomato 1kg ₹75 Add to Cart", "price": 75.0, "is_matching_entity": True},
        {"id": "c_tom_99", "label": "Sun-Dried Tomato Pack ₹99 Add to Cart", "price": 99.0, "is_matching_entity": True},
        {"id": "c_tom_140", "label": "Exotic Tomato Mix 1kg ₹140 Add to Cart", "price": 140.0, "is_matching_entity": True},
    ]
    for idx, t in enumerate(competing_tomatoes):
        t["res"] = f"btn_add_t_{idx}"
        t["role"] = "button"
        t["role_idx"] = 1
        t["is_clickable"] = True
        t["is_editable"] = False
        t["bounds_vec"] = [0.55, min(0.9, 0.20 + idx * 0.05), 0.95, min(0.95, 0.35 + idx * 0.05), 0.75, 0.275, 0.4, 0.15, 0.06, 2.6]

    # 3. Cheap Distractors (12 Non-Tomatoes with prices lower than ₹13)
    distractors = [
        {"id": "c_curry_5", "label": "Curry Leaves Pack ₹5 Add to Cart", "price": 5.0, "is_matching_entity": False},
        {"id": "c_coriander_7", "label": "Fresh Coriander Leaves ₹7 Add to Cart", "price": 7.0, "is_matching_entity": False},
        {"id": "c_chilli_8", "label": "Fresh Green Chilli 100g ₹8 Add to Cart", "price": 8.0, "is_matching_entity": False},
        {"id": "c_ginger_9", "label": "Fresh Ginger 100g ₹9 Add to Cart", "price": 9.0, "is_matching_entity": False},
        {"id": "c_soap_10", "label": "Dove Beauty Soap 100g ₹10 Add to Cart", "price": 10.0, "is_matching_entity": False},
        {"id": "c_rin_10", "label": "Rin Detergent Bar ₹10 Add to Cart", "price": 10.0, "is_matching_entity": False},
        {"id": "c_santoor_11", "label": "Santoor Soap 100g ₹11 Add to Cart", "price": 11.0, "is_matching_entity": False},
        {"id": "c_dettol_12", "label": "Dettol Bathing Soap ₹12 Add to Cart", "price": 12.0, "is_matching_entity": False},
        {"id": "c_lemon_12", "label": "Fresh Lemon 4 pcs ₹12 Add to Cart", "price": 12.0, "is_matching_entity": False},
        {"id": "c_garlic_15", "label": "Garlic Pods 100g ₹15 Add to Cart", "price": 15.0, "is_matching_entity": False},
        {"id": "c_potato_18", "label": "Baby Potato 500g ₹18 Add to Cart", "price": 18.0, "is_matching_entity": False},
        {"id": "c_onion_22", "label": "Red Onion 1kg ₹22 Add to Cart", "price": 22.0, "is_matching_entity": False},
    ]
    for idx, d in enumerate(distractors):
        d["res"] = f"btn_add_d_{idx}"
        d["role"] = "button"
        d["role_idx"] = 1
        d["is_clickable"] = True
        d["is_editable"] = False
        d["bounds_vec"] = [0.05, min(0.9, 0.40 + idx * 0.04), 0.45, min(0.95, 0.55 + idx * 0.04), 0.25, 0.475, 0.4, 0.15, 0.06, 2.6]

    # 4. System controls
    system_cands = [
        {"id": "c_filter", "label": "Filters & Sort", "res": "btn_filter", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.0, 0.1, 0.5, 0.18, 0.25, 0.14, 0.5, 0.08, 0.04, 6.0]},
        {"id": "c_cart_bar", "label": "View Cart (0 items)", "res": "cart_bar", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.0, 0.9, 1.0, 1.0, 0.5, 0.95, 1.0, 0.1, 0.1, 10.0]},
        make_system_cand("c_submit", "submit focused field", "button"),
        make_system_cand("sys_scroll_down", "scroll down"),
        make_system_cand("sys_back", "back"),
    ]

    return [target] + competing_tomatoes + distractors + system_cands


def build_full_v54_dataset() -> List[Dict[str, Any]]:
    dataset = []

    # 1. Anchor Screens (Gate 1 preservation across all 12 tasks)
    anchor_screens = get_test_screens()
    for s in anchor_screens:
        cands = s["candidates"][:24]
        target_idx = next((i for i, c in enumerate(cands) if s["expected_target"].lower() in c["label"].lower()), 0)
        act_name = s.get("expected_action", "CLICK")
        act_idx = ACTION_TO_IDX.get(act_name, 0)
        st_tok, en_tok = find_token_span(s["goal"], s["expected_target"], TOKENIZER)

        task_nm = s.get("name", s.get("task_name", "anchor"))
        dataset.append({
            "name": f"anchor_{task_nm}",
            "category": "anchor",
            "goal": s["goal"],
            "candidates": cands,
            "target_idx": target_idx,
            "action_idx": act_idx,
            "hard_negs": [i for i in range(len(cands)) if i != target_idx],
            "target_span": (st_tok, en_tok),
            "entity_target": s["expected_target"],
            "is_terminal": ("Step 5" in task_nm or s.get("is_terminal", False)),
        })

    # 1b. Android Launcher Screen (Turn 1: Open Zepto from Phone Home Screen)
    launcher_cands = [
        {"id": "c_open_zepto", "label": "open zepto", "res": "app_zepto", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.1, 0.3, 0.3, 0.4, 0.2, 0.35, 0.2, 0.1, 0.02, 2.0], "is_matching_entity": True},
        {"id": "c_open_yt", "label": "open youtube", "res": "app_yt", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.4, 0.3, 0.6, 0.4, 0.5, 0.35, 0.2, 0.1, 0.02, 2.0], "is_matching_entity": False},
        {"id": "c_open_insta", "label": "open instagram", "res": "app_insta", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.7, 0.3, 0.9, 0.4, 0.8, 0.35, 0.2, 0.1, 0.02, 2.0], "is_matching_entity": False},
        make_system_cand("c_press_back", "back"),
        make_system_cand("c_submit", "submit focused field"),
        make_system_cand("c_fail_goal", "fail"),
    ]
    st_tok, en_tok = find_token_span("open zepto and search cheapest tomato", "zepto", TOKENIZER)
    dataset.append({
        "name": "zepto_launcher_turn1",
        "category": "launcher_media",
        "goal": "open zepto and search cheapest tomato",
        "candidates": launcher_cands,
        "target_idx": 0,
        "action_idx": ACTION_TO_IDX["CLICK"],
        "hard_negs": [1, 2, 3, 4, 5],
        "target_span": (st_tok, en_tok),
        "entity_target": "zepto",
        "is_terminal": False,
    })

    # 1c. YouTube Home Screen (Search magnifying glass icon vs feed video cards)
    yt_home_cands = [
        {"id": "yt_search", "label": "Search", "res": "search_button", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.75, 0.03, 0.88, 0.08, 0.815, 0.055, 0.13, 0.05, 0.0065, 2.6], "is_matching_entity": True},
        {"id": "yt_voice", "label": "Voice search", "res": "voice_search", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.89, 0.03, 0.98, 0.08, 0.935, 0.055, 0.09, 0.05, 0.0045, 1.8], "is_matching_entity": False},
        {"id": "yt_video1", "label": "Go to channel Sajjaad Khader", "res": "channel", "role": "cardview", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.12, 0.95, 0.48, 0.5, 0.3, 0.9, 0.36, 0.324, 2.5], "is_matching_entity": False},
        {"id": "yt_video2", "label": "Starship Fifth Flight Test SpaceX, verified - 55 lakh views", "res": "video", "role": "cardview", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.5, 0.95, 0.85, 0.5, 0.675, 0.9, 0.35, 0.315, 2.57], "is_matching_entity": False},
        make_system_cand("c_fail_goal", "fail"),
    ]
    st_tok, en_tok = find_token_span("Open YouTube and search for SpaceX", "SpaceX", TOKENIZER)
    dataset.append({
        "name": "yt_home_search_icon",
        "category": "launcher_media",
        "goal": "Open YouTube and search for SpaceX",
        "candidates": yt_home_cands,
        "target_idx": 0,
        "action_idx": ACTION_TO_IDX["CLICK"],
        "hard_negs": [1, 2, 3, 4],
        "target_span": (st_tok, en_tok),
        "entity_target": "SpaceX",
        "is_terminal": False,
    })

    # 1d. YouTube Autocomplete Screen (Submit focused field vs edit suggestion trap)
    yt_auto_cands = [
        make_system_cand("c_submit", "submit focused field"),
        {"id": "yt_edit1", "label": "Edit suggestion spacex", "res": "edit1", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.85, 0.1, 0.98, 0.16, 0.915, 0.13, 0.13, 0.06, 0.0078, 2.1], "is_matching_entity": False},
        {"id": "yt_edit2", "label": "Edit suggestion spacex live", "res": "edit2", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.85, 0.17, 0.98, 0.23, 0.915, 0.20, 0.13, 0.06, 0.0078, 2.1], "is_matching_entity": False},
        {"id": "yt_input", "label": "Search YouTube", "res": "search_input", "role": "input", "role_idx": 2, "is_editable": True, "is_populated": True, "bounds_vec": [0.1, 0.04, 0.85, 0.09, 0.475, 0.065, 0.75, 0.05, 0.0375, 15.0], "is_matching_entity": False},
        make_system_cand("c_fail_goal", "fail"),
    ]
    st_tok, en_tok = find_token_span("Open YouTube and search for SpaceX", "SpaceX", TOKENIZER)
    dataset.append({
        "name": "yt_autocomplete_submit",
        "category": "launcher_media",
        "goal": "Open YouTube and search for SpaceX",
        "candidates": yt_auto_cands,
        "target_idx": 0,
        "action_idx": ACTION_TO_IDX["CLICK"],
        "hard_negs": [1, 2, 3, 4],
        "target_span": (st_tok, en_tok),
        "entity_target": "submit",
        "is_terminal": False,
    })

    # 1e. Instagram Home Screen ('Search and explore' bottom tab vs feed posts)
    ig_home_cands = [
        {"id": "ig_explore_tab", "label": "Search and explore", "res": "explore_tab", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.2, 0.92, 0.4, 0.99, 0.3, 0.955, 0.2, 0.07, 0.014, 2.85], "is_matching_entity": True},
        {"id": "ig_home_tab", "label": "Home", "res": "home_tab", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.0, 0.92, 0.2, 0.99, 0.1, 0.955, 0.2, 0.07, 0.014, 2.85], "is_matching_entity": False},
        {"id": "ig_post", "label": "Photo by creator, 527 likes", "res": "feed_post", "role": "cardview", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.0, 0.15, 1.0, 0.65, 0.5, 0.4, 1.0, 0.5, 0.5, 2.0], "is_matching_entity": False},
        make_system_cand("c_fail_goal", "fail"),
    ]
    st_tok, en_tok = find_token_span("Open Instagram and search photography", "photography", TOKENIZER)
    dataset.append({
        "name": "ig_home_search_tab",
        "category": "launcher_media",
        "goal": "Open Instagram and search photography",
        "candidates": ig_home_cands,
        "target_idx": 0,
        "action_idx": ACTION_TO_IDX["CLICK"],
        "hard_negs": [1, 2, 3],
        "target_span": (st_tok, en_tok),
        "entity_target": "search",
        "is_terminal": False,
    })

    # 2. Live Zepto Home Screen (Gate 2: Search box ranked #1 over floating cart)
    if ZEPTO_HOME_XML.exists():
        zh_cands = parse_xml_candidates(ZEPTO_HOME_XML)
        search_node = next((c for c in zh_cands if "homepage-search-box" in c.get("res", "")), None)
        if search_node is None:
            search_node = next((c for c in zh_cands if "search" in c["label"].lower()), None)
        cart_btn = next((c for c in zh_cands if "floating-cart-button" in c.get("res", "") or "Cart" in c["label"]), None)
        if search_node:
            ordered_cands = [search_node, cart_btn] + [c for c in zh_cands if c not in [search_node, cart_btn]][:22]
            ordered_cands = [c for c in ordered_cands if c is not None]
            st_tok, en_tok = find_token_span("Open Zepto and add the cheapest tomato to the cart", "tomato", TOKENIZER)
            dataset.append({
                "name": "zepto_home_search",
                "category": "zepto_search",
                "goal": "Open Zepto and add the cheapest tomato to the cart",
                "candidates": ordered_cands,
                "target_idx": 0,
                "action_idx": ACTION_TO_IDX["CLICK"],
                "hard_negs": list(range(1, min(8, len(ordered_cands)))),
                "target_span": (st_tok, en_tok),
                "entity_target": "search",
                "is_terminal": False,
            })

    # 2b. Zepto Search Input Screen (Turn 3: Type query into search field)
    search_input_cands = [
        {"id": "search_input_field", "label": "Search for products, brands and more", "res": "search_input", "role": "input", "role_idx": 2, "is_clickable": True, "is_editable": True, "bounds_vec": [0.05, 0.08, 0.95, 0.14, 0.5, 0.11, 0.9, 0.06, 0.054, 15.0], "is_matching_entity": True},
        {"id": "btn_back", "label": "back", "res": "btn_back", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.02, 0.08, 0.08, 0.14, 0.05, 0.11, 0.06, 0.06, 0.0036, 1.0], "is_matching_entity": False},
        {"id": "pill_trending", "label": "Trending: Milk, Bread, Eggs", "res": "trending", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.2, 0.95, 0.3, 0.5, 0.25, 0.9, 0.1, 0.09, 9.0], "is_matching_entity": False},
        make_system_cand("c_submit", "submit focused field"),
    ]
    st_tok, en_tok = find_token_span("open zepto and search cheapest tomato", "tomato", TOKENIZER)
    dataset.append({
        "name": "zepto_search_input_turn3",
        "category": "zepto_search",
        "goal": "open zepto and search cheapest tomato",
        "candidates": search_input_cands,
        "target_idx": 0,
        "action_idx": ACTION_TO_IDX["TYPE_TEXT"],
        "hard_negs": [1, 2, 3],
        "target_span": (st_tok, en_tok),
        "entity_target": "tomato",
        "is_terminal": False,
    })

    # 2c. Zepto Autocomplete Dropdown & Submit Screen (Turn 4: Submit search, suppress autocomplete dropdown)
    autocomplete_cands = [
        make_system_cand("c_submit", "submit focused field"),
        {"id": "suggest_puree", "label": "Tomato puree", "res": "suggest_1", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.15, 0.95, 0.22, 0.5, 0.185, 0.9, 0.07, 0.063, 12.8], "is_matching_entity": False},
        {"id": "suggest_ketchup", "label": "Tomato ketchup", "res": "suggest_2", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.23, 0.95, 0.30, 0.5, 0.265, 0.9, 0.07, 0.063, 12.8], "is_matching_entity": False},
        {"id": "suggest_fresh", "label": "Fresh tomatoes", "res": "suggest_3", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.31, 0.95, 0.38, 0.5, 0.345, 0.9, 0.07, 0.063, 12.8], "is_matching_entity": False},
        {"id": "btn_back", "label": "back", "res": "btn_back", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.02, 0.08, 0.08, 0.14, 0.05, 0.11, 0.06, 0.06, 0.0036, 1.0], "is_matching_entity": False},
    ]
    st_tok, en_tok = find_token_span("open zepto and search cheapest tomato", "tomato", TOKENIZER)
    dataset.append({
        "name": "zepto_submit_search_turn4",
        "category": "zepto_search",
        "goal": "open zepto and search cheapest tomato",
        "candidates": autocomplete_cands,
        "target_idx": 0,
        "action_idx": ACTION_TO_IDX["CLICK"],
        "hard_negs": [1, 2, 3, 4],
        "target_span": (st_tok, en_tok),
        "entity_target": "submit",
        "is_terminal": False,
    })

    # 2d. Zepto Search Trending Chip vs Ice Distractor Screen (Turn 3.5: Query chip beats Ice distractor)
    zepto_search_chip_variations = [
        ("open zepto and search cheapest tomato", "tomato", "Fresh tomatoes"),
        ("Open Zepto and search cheapest tomatoes", "tomatoes", "Fresh tomatoes"),
        ("Open Zepto and search fresh tomatoes", "tomatoes", "Fresh tomatoes"),
        ("Search tomatoes on Zepto", "tomatoes", "Fresh tomatoes"),
        ("Open Zepto and search fresh milk", "milk", "Fresh milk"),
        ("Search curd on Zepto", "curd", "Fresh curd"),
    ]
    for g_str, ent_str, chip_lbl in zepto_search_chip_variations:
        z_cands = [
            {"id": "chip_target", "label": chip_lbl, "res": "chip_target", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.12, 0.45, 0.18, 0.25, 0.15, 0.4, 0.06, 0.024, 6.66], "is_matching_entity": True},
            {"id": "search_input_field", "label": "Search for products, brands and more", "res": "search_input", "role": "input", "role_idx": 2, "is_clickable": True, "is_editable": True, "bounds_vec": [0.05, 0.04, 0.95, 0.10, 0.5, 0.07, 0.9, 0.06, 0.054, 15.0], "is_matching_entity": False},
            {"id": "chip_ice", "label": "Ice", "res": "chip_ice", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.20, 0.25, 0.26, 0.15, 0.23, 0.2, 0.06, 0.012, 3.33], "is_matching_entity": False},
            {"id": "chip_munchies", "label": "Munchies", "res": "chip_munchies", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.30, 0.20, 0.55, 0.26, 0.425, 0.23, 0.25, 0.06, 0.015, 4.16], "is_matching_entity": False},
            {"id": "chip_cold_drinks", "label": "Cold Drinks", "res": "chip_drinks", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.60, 0.20, 0.90, 0.26, 0.75, 0.23, 0.3, 0.06, 0.018, 5.0], "is_matching_entity": False},
            make_system_cand("c_complete_goal", "complete", "button"),
            make_system_cand("c_press_back", "back"),
        ]
        st_tok, en_tok = find_token_span(g_str, ent_str, TOKENIZER)
        dataset.append({
            "name": f"zepto_chip_screen_{ent_str}",
            "category": "zepto_search",
            "goal": g_str,
            "candidates": z_cands,
            "target_idx": 0,
            "action_idx": ACTION_TO_IDX["CLICK"],
            "hard_negs": [1, 2, 3, 4, 5, 6],
            "target_span": (st_tok, en_tok),
            "entity_target": ent_str,
            "is_terminal": False,
        })

    # 3. Live Zepto Above-the-fold Tomato Grid (Gate 3: Scroll down to explore)
    if TOMATO_ABOVE_XML.exists():
        above_cands = parse_xml_candidates(TOMATO_ABOVE_XML)
        scroll_cand = make_system_cand("c_scroll_down", "scroll down", "system")
        complete_cand = make_system_cand("c_complete_goal", "complete", "button")
        cands_above = [scroll_cand] + above_cands[:20] + [complete_cand]
        dataset.append({
            "name": "zepto_above_fold_scroll",
            "category": "zepto_grid",
            "goal": "Search cheapest tomato and add to cart",
            "candidates": cands_above,
            "target_idx": 0,
            "action_idx": ACTION_TO_IDX["SCROLL"],
            "scroll_dir": 0,
            "hard_negs": list(range(1, min(6, len(cands_above)))),
            "entity_target": "tomato",
            "is_terminal": False,
        })

    # 4. Dense 28-Candidate E-Commerce Screen with 12 Tomatoes & 12 Distractors (Gates 4 & 5)
    dense_cands = create_dense_30_candidate_screen()
    dataset.append({
        "name": "zepto_scrolled_dense_28_candidates",
        "category": "zepto_grid",
        "goal": "Search cheapest tomato and add to cart",
        "candidates": dense_cands,
        "target_idx": 0,  # Tomato Local ₹13 is TOP-1 out of 28!
        "action_idx": ACTION_TO_IDX["CLICK"],
        "hard_negs": list(range(1, len(dense_cands))),  # All 27 others are hard negatives
        "entity_target": "tomato",
        "is_terminal": False,
    })

    # 5. Programmatic Continuous Numerical Contrastive Permutations (Gate 6 Generalization)
    random.seed(42)
    for p_idx in range(1, 4):
        cheap_p = random.randint(12, 45)
        exp_p = cheap_p + random.randint(10, 40)
        dist_p = max(5, cheap_p - random.randint(2, 6))

        p_cands = [
            {
                "id": f"p_target_{p_idx}",
                "label": f"Local Fresh Tomato 500g ₹{cheap_p} Add to Cart",
                "res": "btn_add_target",
                "role": "button",
                "role_idx": 1,
                "is_clickable": True,
                "is_editable": False,
                "bounds_vec": [0.05, 0.35, 0.45, 0.55, 0.25, 0.45, 0.4, 0.2, 0.08, 2.0],
                "price": float(cheap_p),
                "is_matching_entity": True,
            },
            {
                "id": f"p_exp_{p_idx}",
                "label": f"Organic Premium Tomato 500g ₹{exp_p} Add to Cart",
                "res": "btn_add_exp",
                "role": "button",
                "role_idx": 1,
                "is_clickable": True,
                "is_editable": False,
                "bounds_vec": [0.55, 0.35, 0.95, 0.55, 0.75, 0.45, 0.4, 0.2, 0.08, 2.0],
                "price": float(exp_p),
                "is_matching_entity": True,
            },
            {
                "id": f"p_dist_{p_idx}",
                "label": f"Ginger or Onion 200g ₹{dist_p} Add to Cart",
                "res": "btn_add_dist",
                "role": "button",
                "role_idx": 1,
                "is_clickable": True,
                "is_editable": False,
                "bounds_vec": [0.05, 0.60, 0.45, 0.80, 0.25, 0.70, 0.4, 0.2, 0.08, 2.0],
                "price": float(dist_p),
                "is_matching_entity": False,
            },
            make_system_cand("c_submit", "submit focused field", "button"),
            make_system_cand("sys_scroll_down", "scroll down"),
            make_system_cand("sys_back", "back"),
        ]

        dataset.append({
            "name": f"permuted_price_screen_{p_idx}",
            "category": "price_permutations",
            "goal": "Search cheapest tomato and add to cart",
            "candidates": p_cands,
            "target_idx": 0,
            "action_idx": ACTION_TO_IDX["CLICK"],
            "hard_negs": list(range(1, len(p_cands))),
            "entity_target": "tomato",
            "is_terminal": False,
        })

    # 6. Terminal Cart Milestone Screen (Gate 7: Terminal State on Cart Mutation)
    cart_cands = [
        make_system_cand("c_complete_goal", "complete", "button"),
        {"id": "btnEnableService", "label": "CONFIGURE", "res": "btnEnableService", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.1, 0.8, 0.9, 0.9, 0.5, 0.85, 0.8, 0.1, 0.08, 8.0]},
        {"id": "c_checkout", "label": "Click to Pay ₹428", "res": "c_checkout", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.85, 0.95, 0.92, 0.5, 0.885, 0.9, 0.07, 0.063, 12.8]},
        {"id": "c_tomato_item", "label": "Tomato Local 250g in cart", "res": "item", "role": "text", "role_idx": 0, "is_clickable": False, "is_editable": False, "bounds_vec": [0.1, 0.3, 0.9, 0.4, 0.5, 0.35, 0.8, 0.1, 0.08, 8.0]}
    ]
    dataset.append({
        "name": "zepto_cart_terminal_complete",
        "category": "terminal_cart",
        "goal": "Search cheapest tomato and add to cart and stop",
        "candidates": cart_cands,
        "target_idx": 0,
        "action_idx": ACTION_TO_IDX["WAIT"],
        "hard_negs": [1, 2],
        "entity_target": "tomato",
        "is_terminal": True,
    })

    # Zepto Post-Add Grid Milestone (Tapping ADD mutates card to ADDED 1 in cart -> Complete goal)
    zepto_post_add_cands = [
        {"id": "z_tomato_added", "label": "Tomato Local 500g ₹13 ADDED 1 in cart", "res": "item_tomato", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.25, 0.48, 0.55, 0.265, 0.40, 0.43, 0.30, 0.129, 1.43], "price": 13.0, "is_matching_entity": True},
        {"id": "z_tomato_hybrid", "label": "Tomato Hybrid 500g ₹22 ADD", "res": "item_hybrid", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.52, 0.25, 0.95, 0.55, 0.735, 0.40, 0.43, 0.30, 0.129, 1.43], "price": 22.0, "is_matching_entity": True},
        {"id": "z_view_cart", "label": "View Cart 1 item ₹13", "res": "cart_bar", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.90, 0.95, 0.97, 0.5, 0.935, 0.9, 0.07, 0.063, 12.8], "is_matching_entity": False},
        make_system_cand("c_submit", "submit focused field", "button"),
        make_system_cand("c_complete_goal", "complete", "button"),
    ]
    dataset.append({
        "name": "zepto_post_add_complete",
        "category": "terminal_cart",
        "goal": "Open Zepto and search cheapest tomato and add to cart and stop",
        "candidates": zepto_post_add_cands,
        "target_idx": 4,  # c_complete_goal
        "action_idx": ACTION_TO_IDX["WAIT"],
        "hard_negs": [0, 1, 2, 3],
        "entity_target": "tomato",
        "is_terminal": True,
    })

    # Zepto Out-of-Stock Fallback Screen (Tomato out of stock -> only Potato, Onion, Ginger -> scroll down beats distractors & submit)
    zepto_oos_cands = [
        {"id": "z_potato", "label": "Fresh Potato 1kg ₹30 ADD", "res": "btn_add_potato", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.25, 0.48, 0.55, 0.265, 0.40, 0.43, 0.30, 0.129, 1.43], "price": 30.0, "is_matching_entity": False},
        {"id": "z_onion", "label": "Red Onion 1kg ₹45 ADD", "res": "btn_add_onion", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.52, 0.25, 0.95, 0.55, 0.735, 0.40, 0.43, 0.30, 0.129, 1.43], "price": 45.0, "is_matching_entity": False},
        {"id": "z_ginger", "label": "Fresh Ginger 100g ₹15 ADD", "res": "btn_add_ginger", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.58, 0.48, 0.88, 0.265, 0.73, 0.43, 0.30, 0.129, 1.43], "price": 15.0, "is_matching_entity": False},
        make_system_cand("c_scroll_down", "scroll down", "system"),
        make_system_cand("c_complete_goal", "complete", "button"),
        make_system_cand("c_submit", "submit focused field", "button"),
        make_system_cand("c_press_back", "back"),
    ]
    dataset.append({
        "name": "zepto_oos_scroll_down",
        "category": "zepto_grid",
        "goal": "Search cheapest tomato and add to cart",
        "candidates": zepto_oos_cands,
        "target_idx": 3,  # c_scroll_down
        "action_idx": ACTION_TO_IDX["SCROLL"],
        "hard_negs": [0, 1, 2, 4, 5, 6],
        "entity_target": "tomato",
        "is_terminal": False,
    })

    # Zepto Potato Grid Screen (Cheapest Potato ₹14 beats higher-priced potatoes ₹22, ₹23, ₹41, cheap distractors ₹5, ₹7, and submit)
    zepto_potato_cands = [
        {"id": "c_potato_14", "label": "Organically Grown Baby Potato 250g ₹14 ADD", "res": "btn_add_p14", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.25, 0.48, 0.55, 0.265, 0.40, 0.43, 0.30, 0.129, 1.43], "price": 14.0, "is_matching_entity": True},
        {"id": "c_potato_22", "label": "Potato 500g ₹22 ADD", "res": "btn_add_p22", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.52, 0.25, 0.95, 0.55, 0.735, 0.40, 0.43, 0.30, 0.129, 1.43], "price": 22.0, "is_matching_entity": True},
        {"id": "c_potato_23", "label": "Baby Potato 500g ₹23 ADD", "res": "btn_add_p23", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.58, 0.48, 0.88, 0.265, 0.73, 0.43, 0.30, 0.129, 1.43], "price": 23.0, "is_matching_entity": True},
        {"id": "c_potato_41", "label": "Bloom Potato 1kg ₹41 ADD", "res": "btn_add_p41", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.52, 0.58, 0.95, 0.88, 0.735, 0.73, 0.43, 0.30, 0.129, 1.43], "price": 41.0, "is_matching_entity": True},
        {"id": "c_coriander_7", "label": "Fresh Coriander Leaves ₹7 ADD", "res": "btn_add_c7", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.89, 0.48, 0.99, 0.265, 0.94, 0.43, 0.10, 0.043, 4.3], "price": 7.0, "is_matching_entity": False},
        {"id": "c_curry_5", "label": "Curry Leaves Pack ₹5 ADD", "res": "btn_add_c5", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.52, 0.89, 0.95, 0.99, 0.735, 0.94, 0.43, 0.10, 0.043, 4.3], "price": 5.0, "is_matching_entity": False},
        make_system_cand("c_submit", "submit focused field", "button"),
        make_system_cand("c_scroll_down", "scroll down", "system"),
        make_system_cand("c_complete_goal", "complete", "button"),
        make_system_cand("c_press_back", "back"),
    ]
    st_tok, en_tok = find_token_span("Open zepto and search cheapest potato", "potato", TOKENIZER)
    dataset.append({
        "name": "zepto_potato_grid_cheapest",
        "category": "zepto_grid",
        "goal": "Open zepto and search cheapest potato and add to cart",
        "candidates": zepto_potato_cands,
        "target_idx": 0,
        "action_idx": ACTION_TO_IDX["CLICK"],
        "hard_negs": list(range(1, len(zepto_potato_cands))),
        "target_span": (st_tok, en_tok),
        "entity_target": "potato",
        "is_terminal": False,
    })

    # 7. Blinkit Quick-Commerce Trajectory
    blinkit_home_cands = [
        {"id": "b_search", "label": "Search 'milk', 'bread', 'eggs'", "res": "search_bar", "role": "input", "role_idx": 2, "is_clickable": True, "is_editable": True, "bounds_vec": [0.05, 0.08, 0.95, 0.14, 0.5, 0.11, 0.9, 0.06, 0.054, 15.0], "is_matching_entity": True},
        {"id": "b_cart", "label": "View Cart ₹0", "res": "cart", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.8, 0.02, 0.95, 0.07, 0.875, 0.045, 0.15, 0.05, 0.0075, 3.0], "is_matching_entity": False},
        {"id": "b_banner", "label": "Blinkit Festive Super Sale 50% OFF", "res": "banner", "role": "cardview", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.16, 0.95, 0.35, 0.5, 0.255, 0.9, 0.19, 0.171, 4.7], "is_matching_entity": False},
        make_system_cand("c_complete_goal", "complete", "button"),
    ]
    st_tok, en_tok = find_token_span("Search milk on Blinkit and pick the cheapest", "milk", TOKENIZER)
    dataset.append({
        "name": "blinkit_home_search",
        "category": "blinkit",
        "goal": "Search milk on Blinkit and pick the cheapest",
        "candidates": blinkit_home_cands,
        "target_idx": 0,
        "action_idx": ACTION_TO_IDX["TYPE_TEXT"],
        "hard_negs": [1, 2, 3],
        "target_span": (st_tok, en_tok),
        "entity_target": "milk",
        "is_terminal": False,
    })

    blinkit_grid_cands = [
        {"id": "b_milk_27", "label": "Amul Taaza Toned Fresh Milk 500ml ₹27 ADD", "res": "btn_add_milk27", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.25, 0.48, 0.55, 0.265, 0.40, 0.43, 0.30, 0.129, 1.43], "price": 27.0, "is_matching_entity": True},
        {"id": "b_milk_32", "label": "Nandini GoodLife Toned Milk 500ml ₹32 ADD", "res": "btn_add_milk32", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.52, 0.25, 0.95, 0.55, 0.735, 0.40, 0.43, 0.30, 0.129, 1.43], "price": 32.0, "is_matching_entity": True},
        {"id": "b_milk_48", "label": "Country Delight Cow Milk 500ml ₹48 ADD", "res": "btn_add_milk48", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.58, 0.48, 0.88, 0.265, 0.73, 0.43, 0.30, 0.129, 1.43], "price": 48.0, "is_matching_entity": True},
        {"id": "b_dist_bread", "label": "Britannia Milk Bread ₹20 ADD", "res": "btn_add_bread", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.52, 0.58, 0.95, 0.88, 0.735, 0.73, 0.43, 0.30, 0.129, 1.43], "price": 20.0, "is_matching_entity": False},
        {"id": "b_dist_soap", "label": "Dove Bathing Soap 100g ₹15 ADD", "res": "btn_add_soap", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.89, 0.48, 0.99, 0.265, 0.94, 0.43, 0.10, 0.043, 4.3], "price": 15.0, "is_matching_entity": False},
        make_system_cand("b_scroll_down", "scroll down"),
    ]
    st_tok, en_tok = find_token_span("Search milk on Blinkit and pick the cheapest", "milk", TOKENIZER)
    dataset.append({
        "name": "blinkit_grid_cheapest_milk",
        "category": "blinkit",
        "goal": "Search milk on Blinkit and pick the cheapest",
        "candidates": blinkit_grid_cands,
        "target_idx": 0,
        "action_idx": ACTION_TO_IDX["CLICK"],
        "hard_negs": [1, 2, 3, 4, 5],
        "target_span": (st_tok, en_tok),
        "entity_target": "milk",
        "is_terminal": False,
    })

    # Blinkit Post-Add Grid Milestone (Tapping ADD mutates card to Decrease quantity 1 Increase quantity -> Complete goal)
    blinkit_post_add_cands = [
        {"id": "b_milk_27_added", "label": "Amul Taaza Toned Fresh Milk 500ml ₹27 Decrease quantity quantity 1 Increase quantity", "res": "btn_add_milk27", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.25, 0.48, 0.55, 0.265, 0.40, 0.43, 0.30, 0.129, 1.43], "price": 27.0, "is_matching_entity": True},
        {"id": "b_milk_32", "label": "Nandini GoodLife Toned Milk 500ml ₹32 ADD", "res": "btn_add_milk32", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.52, 0.25, 0.95, 0.55, 0.735, 0.40, 0.43, 0.30, 0.129, 1.43], "price": 32.0, "is_matching_entity": True},
        {"id": "b_view_cart", "label": "View cart 1 item ₹27", "res": "view_cart", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.90, 0.95, 0.97, 0.5, 0.935, 0.9, 0.07, 0.063, 12.8], "is_matching_entity": False},
        make_system_cand("c_submit", "submit focused field", "button"),
        make_system_cand("c_complete_goal", "complete", "button"),
    ]
    dataset.append({
        "name": "blinkit_post_add_complete",
        "category": "terminal_cart",
        "goal": "Search milk on Blinkit and pick the cheapest and stop",
        "candidates": blinkit_post_add_cands,
        "target_idx": 4,  # c_complete_goal
        "action_idx": ACTION_TO_IDX["WAIT"],
        "hard_negs": [0, 1, 2, 3],
        "entity_target": "milk",
        "is_terminal": True,
    })

    # 8. Amazon Traditional E-Commerce 2-Step PDP Trajectory
    amazon_home_cands = [
        {"id": "amz_search", "label": "Search Amazon.in", "res": "rs_search_src_text", "role": "input", "role_idx": 2, "is_clickable": True, "is_editable": True, "bounds_vec": [0.05, 0.04, 0.95, 0.09, 0.5, 0.065, 0.9, 0.05, 0.045, 18.0], "is_matching_entity": True},
        {"id": "amz_scan", "label": "Scan with camera", "res": "scan_icon", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.85, 0.04, 0.94, 0.09, 0.895, 0.065, 0.09, 0.05, 0.0045, 1.8], "is_matching_entity": False},
        {"id": "amz_deals", "label": "Great Indian Festival Deals of the Day", "res": "deal_banner", "role": "cardview", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.12, 0.95, 0.40, 0.5, 0.26, 0.9, 0.28, 0.252, 3.2], "is_matching_entity": False},
        make_system_cand("c_complete_goal", "complete", "button"),
    ]
    st_tok, en_tok = find_token_span("Search wireless mouse on Amazon and buy the cheapest", "wireless mouse", TOKENIZER)
    dataset.append({
        "name": "amazon_home_search",
        "category": "amazon",
        "goal": "Search wireless mouse on Amazon and buy the cheapest",
        "candidates": amazon_home_cands,
        "target_idx": 0,
        "action_idx": ACTION_TO_IDX["TYPE_TEXT"],
        "hard_negs": [1, 2, 3],
        "target_span": (st_tok, en_tok),
        "entity_target": "wireless mouse",
        "is_terminal": False,
    })

    amazon_grid_cands = [
        {"id": "amz_mouse_299", "label": "Portronics Toad 23 Wireless Optical Mouse 2.4GHz with USB Nano Receiver ₹299 M.R.P: ₹599 (50% off)", "res": "asin_card_299", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.15, 0.95, 0.40, 0.5, 0.275, 0.9, 0.25, 0.225, 3.6], "price": 299.0, "is_matching_entity": True},
        {"id": "amz_mouse_595", "label": "Logitech B170 Wireless Mouse Optical Tracking 12 Months Battery ₹595 M.R.P: ₹895", "res": "asin_card_595", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.42, 0.95, 0.67, 0.5, 0.545, 0.9, 0.25, 0.225, 3.6], "price": 595.0, "is_matching_entity": True},
        {"id": "amz_mouse_749", "label": "HP 240 Bluetooth Wireless Mouse Ergonomic Design ₹749", "res": "asin_card_749", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.69, 0.95, 0.94, 0.5, 0.815, 0.9, 0.25, 0.225, 3.6], "price": 749.0, "is_matching_entity": True},
        {"id": "amz_dist_pad", "label": "Speed Type Waterproof Gaming Mouse Pad ₹99", "res": "asin_card_pad", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.05, 0.95, 0.14, 0.5, 0.095, 0.9, 0.09, 0.081, 10.0], "price": 99.0, "is_matching_entity": False},
        {"id": "amz_dist_batt", "label": "Duracell AA Alkaline Batteries Pack of 4 ₹150", "res": "asin_card_batt", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.95, 0.95, 1.0, 0.5, 0.975, 0.9, 0.05, 0.045, 18.0], "price": 150.0, "is_matching_entity": False},
        make_system_cand("amz_scroll_down", "scroll down"),
    ]
    st_tok, en_tok = find_token_span("Search wireless mouse on Amazon and buy the cheapest", "wireless mouse", TOKENIZER)
    dataset.append({
        "name": "amazon_grid_cheapest_mouse",
        "category": "amazon",
        "goal": "Search wireless mouse on Amazon and buy the cheapest",
        "candidates": amazon_grid_cands,
        "target_idx": 0,
        "action_idx": ACTION_TO_IDX["CLICK"],
        "hard_negs": [1, 2, 3, 4, 5],
        "target_span": (st_tok, en_tok),
        "entity_target": "wireless mouse",
        "is_terminal": False,
    })

    amazon_pdp_cands = [
        {"id": "amz_btn_add_to_cart", "label": "Add to Cart", "res": "add-to-cart-button", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.70, 0.95, 0.77, 0.5, 0.735, 0.9, 0.07, 0.063, 12.8], "is_matching_entity": True},
        {"id": "amz_btn_buy_now", "label": "Buy Now", "res": "buy-now-button", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.79, 0.95, 0.86, 0.5, 0.825, 0.9, 0.07, 0.063, 12.8], "is_matching_entity": False},
        {"id": "amz_btn_wishlist", "label": "Add to Wish List", "res": "wishlist-button", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.88, 0.95, 0.94, 0.5, 0.91, 0.9, 0.06, 0.054, 15.0], "is_matching_entity": False},
        {"id": "amz_txt_title", "label": "Portronics Toad 23 Wireless Optical Mouse (Black) ₹299", "res": "title", "role": "text", "role_idx": 0, "is_clickable": False, "bounds_vec": [0.05, 0.1, 0.95, 0.25, 0.5, 0.175, 0.9, 0.15, 0.135, 6.0], "price": 299.0, "is_matching_entity": True},
        make_system_cand("amz_back", "back"),
    ]
    st_tok, en_tok = find_token_span("Search wireless mouse on Amazon and buy the cheapest", "Add to Cart", TOKENIZER)
    dataset.append({
        "name": "amazon_pdp_add_to_cart",
        "category": "amazon",
        "goal": "Search wireless mouse on Amazon and buy the cheapest",
        "candidates": amazon_pdp_cands,
        "target_idx": 0,
        "action_idx": ACTION_TO_IDX["CLICK"],
        "hard_negs": [1, 2, 3, 4],
        "target_span": (st_tok, en_tok),
        "entity_target": "Add to Cart",
        "is_terminal": False,
    })

    # 9. Flipkart Traditional E-Commerce 2-Step PDP Trajectory
    flipkart_home_cands = [
        {"id": "fk_search", "label": "Search for Products, Brands and More", "res": "search_widget", "role": "input", "role_idx": 2, "is_clickable": True, "is_editable": True, "bounds_vec": [0.05, 0.05, 0.95, 0.11, 0.5, 0.08, 0.9, 0.06, 0.054, 15.0], "is_matching_entity": True},
        {"id": "fk_cart", "label": "Cart", "res": "cart_icon", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.88, 0.01, 0.98, 0.05, 0.93, 0.03, 0.10, 0.04, 0.004, 2.5], "is_matching_entity": False},
        {"id": "fk_banner", "label": "Big Billion Days Live Deals Up to 80% Off", "res": "hero_banner", "role": "cardview", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.13, 0.95, 0.38, 0.5, 0.255, 0.9, 0.25, 0.225, 3.6], "is_matching_entity": False},
        make_system_cand("c_complete_goal", "complete", "button"),
    ]
    st_tok, en_tok = find_token_span("Search bluetooth headphones on Flipkart and buy the cheapest", "bluetooth headphones", TOKENIZER)
    dataset.append({
        "name": "flipkart_home_search",
        "category": "flipkart",
        "goal": "Search bluetooth headphones on Flipkart and buy the cheapest",
        "candidates": flipkart_home_cands,
        "target_idx": 0,
        "action_idx": ACTION_TO_IDX["TYPE_TEXT"],
        "hard_negs": [1, 2, 3],
        "target_span": (st_tok, en_tok),
        "entity_target": "bluetooth headphones",
        "is_terminal": False,
    })

    flipkart_grid_cands = [
        {"id": "fk_hp_899", "label": "boAt Rockerz 255 Bluetooth Neckband ₹899 ₹2,990 69% off Free delivery", "res": "product_card_899", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.15, 0.95, 0.40, 0.5, 0.275, 0.9, 0.25, 0.225, 3.6], "price": 899.0, "is_matching_entity": True},
        {"id": "fk_hp_1499", "label": "OnePlus Bullets Wireless Z2 Bluetooth Neckband ₹1,499 ₹2,299 34% off", "res": "product_card_1499", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.42, 0.95, 0.67, 0.5, 0.545, 0.9, 0.25, 0.225, 3.6], "price": 1499.0, "is_matching_entity": True},
        {"id": "fk_hp_1699", "label": "Sony WI-C100 Wireless In-Ear Bluetooth Headphones ₹1,699 ₹2,790 39% off", "res": "product_card_1699", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.69, 0.95, 0.94, 0.5, 0.815, 0.9, 0.25, 0.225, 3.6], "price": 1699.0, "is_matching_entity": True},
        {"id": "fk_dist_case", "label": "Protective Hard Case Pouch for Headphones ₹199 ₹499", "res": "product_card_case", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.05, 0.95, 0.14, 0.5, 0.095, 0.9, 0.09, 0.081, 10.0], "price": 199.0, "is_matching_entity": False},
        {"id": "fk_dist_cable", "label": "Aux Audio Cable 3.5mm Male to Male ₹99", "res": "product_card_aux", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.95, 0.95, 1.0, 0.5, 0.975, 0.9, 0.05, 0.045, 18.0], "price": 99.0, "is_matching_entity": False},
        make_system_cand("fk_scroll_down", "scroll down"),
    ]
    st_tok, en_tok = find_token_span("Search bluetooth headphones on Flipkart and buy the cheapest", "bluetooth headphones", TOKENIZER)
    dataset.append({
        "name": "flipkart_grid_cheapest_headphones",
        "category": "flipkart",
        "goal": "Search bluetooth headphones on Flipkart and buy the cheapest",
        "candidates": flipkart_grid_cands,
        "target_idx": 0,
        "action_idx": ACTION_TO_IDX["CLICK"],
        "hard_negs": [1, 2, 3, 4, 5],
        "target_span": (st_tok, en_tok),
        "entity_target": "bluetooth headphones",
        "is_terminal": False,
    })

    flipkart_pdp_cands = [
        {"id": "fk_btn_add_to_cart", "label": "Add to Cart", "res": "button_add_to_cart", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.91, 0.48, 0.98, 0.265, 0.945, 0.43, 0.07, 0.0301, 6.14], "is_matching_entity": True},
        {"id": "fk_btn_buy_now", "label": "Buy Now", "res": "button_buy_now", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.52, 0.91, 0.95, 0.98, 0.735, 0.945, 0.43, 0.07, 0.0301, 6.14], "is_matching_entity": False},
        {"id": "fk_txt_title", "label": "boAt Rockerz 255 Bluetooth Neckband ₹899", "res": "title", "role": "text", "role_idx": 0, "is_clickable": False, "bounds_vec": [0.05, 0.1, 0.95, 0.25, 0.5, 0.175, 0.9, 0.15, 0.135, 6.0], "price": 899.0, "is_matching_entity": True},
        make_system_cand("fk_back", "back"),
    ]
    st_tok, en_tok = find_token_span("Search bluetooth headphones on Flipkart and buy the cheapest", "Add to Cart", TOKENIZER)
    dataset.append({
        "name": "flipkart_pdp_add_to_cart",
        "category": "flipkart",
        "goal": "Search bluetooth headphones on Flipkart and buy the cheapest",
        "candidates": flipkart_pdp_cands,
        "target_idx": 0,
        "action_idx": ACTION_TO_IDX["CLICK"],
        "hard_negs": [1, 2, 3],
        "target_span": (st_tok, en_tok),
        "entity_target": "Add to Cart",
        "is_terminal": False,
    })

    # 10. WhatsApp 4-Turn Messaging Trajectory & Universal Communication Suite
    # Turn 1: Screen A - WhatsApp Main Chats Screen (Search Contact)
    wa_chats_cands = [
        {"id": "wa_search_btn", "label": "Search contacts and chats", "res": "menuitem_search", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.75, 0.03, 0.88, 0.08, 0.815, 0.055, 0.13, 0.05, 0.0065, 2.6], "is_matching_entity": True},
        {"id": "wa_chat_priya", "label": "Priya: Hey, are you free today? 11:20 AM", "res": "conversations_row", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.15, 0.95, 0.25, 0.5, 0.20, 0.9, 0.10, 0.09, 9.0], "is_matching_entity": False},
        {"id": "wa_chat_group", "label": "Family Group: Good morning everyone! 10:45 AM", "res": "conversations_row", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.26, 0.95, 0.36, 0.5, 0.31, 0.9, 0.10, 0.09, 9.0], "is_matching_entity": False},
        {"id": "wa_tab_updates", "label": "Updates", "res": "tab_updates", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.25, 0.92, 0.50, 0.98, 0.375, 0.95, 0.25, 0.06, 0.015, 4.16], "is_matching_entity": False},
        {"id": "wa_tab_calls", "label": "Calls", "res": "tab_calls", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.75, 0.92, 0.98, 0.98, 0.865, 0.95, 0.23, 0.06, 0.0138, 3.83], "is_matching_entity": False},
        make_system_cand("wa_complete", "complete", "button"),
        make_system_cand("wa_back", "back"),
    ]
    wa_screen_a_goals = [
        ("Send message to Rahul: I am on my way", "Rahul"),
        ("Send message to Karthik: check your email", "Karthik"),
        ("Message Priya: are you free today", "Priya"),
        ("Open WhatsApp and send message to Alex: meeting at 5", "Alex"),
        ("Message Mom: I reached home safely", "Mom"),
    ]
    for vi, (g_str, tgt_name) in enumerate(wa_screen_a_goals):
        st_tok, en_tok = find_token_span(g_str, tgt_name, TOKENIZER)
        dataset.append({
            "name": f"wa_screen_a_search_contact_var{vi+1}",
            "category": "comm_search",
            "goal": g_str,
            "candidates": wa_chats_cands,
            "target_idx": 0,
            "action_idx": ACTION_TO_IDX["TYPE_TEXT"],
            "hard_negs": [1, 2, 3, 4, 5, 6],
            "target_span": (st_tok, en_tok),
            "entity_target": tgt_name,
            "is_terminal": False,
        })

    # Modern WhatsApp Screen A Variation: "Ask Meta AI or Search"
    wa_chats_meta_ai_cands = [
        {"id": "wa_search_meta_ai", "label": "Ask Meta AI or Search", "res": "search_src_text", "role": "input", "role_idx": 2, "is_clickable": True, "is_editable": True, "bounds_vec": [0.05, 0.05, 0.95, 0.11, 0.5, 0.08, 0.9, 0.06, 0.054, 15.0], "is_matching_entity": True},
        {"id": "wa_chat_priya", "label": "Priya: Hey, are you free today? 11:20 AM", "res": "conversations_row", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.15, 0.95, 0.25, 0.5, 0.20, 0.9, 0.10, 0.09, 9.0], "is_matching_entity": False},
        {"id": "wa_chat_group", "label": "Family Group: Good morning everyone! 10:45 AM", "res": "conversations_row", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.26, 0.95, 0.36, 0.5, 0.31, 0.9, 0.10, 0.09, 9.0], "is_matching_entity": False},
        {"id": "wa_tab_updates", "label": "Updates", "res": "tab_updates", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.25, 0.92, 0.50, 0.98, 0.375, 0.95, 0.25, 0.06, 0.015, 4.16], "is_matching_entity": False},
        {"id": "wa_tab_calls", "label": "Calls", "res": "tab_calls", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.75, 0.92, 0.98, 0.98, 0.865, 0.95, 0.23, 0.06, 0.0138, 3.83], "is_matching_entity": False},
        make_system_cand("wa_complete", "complete", "button"),
        make_system_cand("wa_back", "back"),
    ]
    wa_screen_a_meta_goals = [
        ("open WhatsApp and send message to Karthik :check your mail", "Karthik"),
        ("Send message to Karthik: check your email", "Karthik"),
        ("Send message to Rahul: I am on my way", "Rahul"),
        ("Message Priya: are you free today", "Priya"),
        ("Open WhatsApp and send message to Alex: meeting at 5", "Alex"),
    ]
    for vi, (g_str, tgt_name) in enumerate(wa_screen_a_meta_goals):
        st_tok, en_tok = find_token_span(g_str, tgt_name, TOKENIZER)
        dataset.append({
            "name": f"wa_screen_a_meta_search_var{vi+1}",
            "category": "comm_search",
            "goal": g_str,
            "candidates": wa_chats_meta_ai_cands,
            "target_idx": 0,
            "action_idx": ACTION_TO_IDX["TYPE_TEXT"],
            "hard_negs": [1, 2, 3, 4, 5, 6],
            "target_span": (st_tok, en_tok),
            "entity_target": tgt_name,
            "is_terminal": False,
        })

    # Turn 2: Screen B - Universal Contact Search & Disambiguation (WhatsApp, Telegram, Signal/SMS, Slack)
    # B1. Standard Rahul Contact Pick (Preserves Gate 15 Screen B baseline)
    wa_search_cands_rahul = [
        {"id": "wa_contact_rahul", "label": "Rahul +91 98765 43210 Available", "res": "contact_row", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.12, 0.95, 0.22, 0.5, 0.17, 0.9, 0.10, 0.09, 9.0], "is_matching_entity": True},
        {"id": "wa_contact_rahul_sharma", "label": "Rahul Sharma Hey there! I am using WhatsApp", "res": "contact_row", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.23, 0.95, 0.33, 0.5, 0.28, 0.9, 0.10, 0.09, 9.0], "is_matching_entity": False},
        {"id": "wa_contact_rakesh", "label": "Rakesh Work Busy", "res": "contact_row", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.34, 0.95, 0.44, 0.5, 0.39, 0.9, 0.10, 0.09, 9.0], "is_matching_entity": False},
        {"id": "wa_search_chats_group", "label": "Search for 'Rahul' in chats", "res": "search_row", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.46, 0.95, 0.54, 0.5, 0.50, 0.9, 0.08, 0.072, 11.25], "is_matching_entity": False},
        {"id": "wa_ask_btn", "label": "ASK", "res": "ask_button", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.85, 0.05, 0.95, 0.11, 0.90, 0.08, 0.10, 0.06, 0.006, 1.66], "is_matching_entity": False},
        make_system_cand("c_submit", "submit focused field"),
        make_system_cand("wa_back", "back"),
    ]
    st_tok, en_tok = find_token_span("Send message to Rahul: I am on my way", "Rahul", TOKENIZER)
    dataset.append({
        "name": "wa_screen_b_pick_contact",
        "category": "comm_contact_wa",
        "goal": "Send message to Rahul: I am on my way",
        "candidates": wa_search_cands_rahul,
        "target_idx": 0,
        "action_idx": ACTION_TO_IDX["CLICK"],
        "hard_negs": [1, 2, 3, 4, 5, 6],
        "target_span": (st_tok, en_tok),
        "entity_target": "Rahul",
        "is_terminal": False,
    })

    # B2. Modern WhatsApp Screen B: Karthik Disambiguation (Clean Samsung Galaxy A26 Accessibility dump)
    wa_cands_karthik_clean = [
        {"id": "c_karthik_text", "label": "Karthik.S (You)", "res": "contact_name", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.18, 0.95, 0.26, 0.5, 0.22, 0.9, 0.08, 0.072, 11.25], "is_matching_entity": True},
        {"id": "c_ask_btn", "label": "ASK", "res": "ask_button", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.85, 0.05, 0.95, 0.11, 0.90, 0.08, 0.10, 0.06, 0.006, 1.66], "is_matching_entity": False},
        make_system_cand("c_submit", "submit focused field"),
        {"id": "c_input_karthik", "label": "​karthik", "res": "search_input", "role": "input", "role_idx": 2, "is_clickable": True, "is_editable": True, "bounds_vec": [0.05, 0.05, 0.85, 0.11, 0.45, 0.08, 0.80, 0.06, 0.048, 13.33], "is_matching_entity": False},
        {"id": "c_dist_karthikeyen", "label": "Karthikeyen", "res": "contact_name", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.30, 0.95, 0.38, 0.5, 0.34, 0.9, 0.08, 0.072, 11.25], "is_matching_entity": False},
        {"id": "c_dist_karthik_mobile", "label": "Karthikeyen 0.1 Mobile", "res": "contact_row", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.40, 0.95, 0.50, 0.5, 0.45, 0.9, 0.10, 0.09, 9.0], "is_matching_entity": False},
        {"id": "c_dist_santosh_status", "label": "Santosh Mava's status. 2 new updates", "res": "status_row", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.52, 0.95, 0.62, 0.5, 0.57, 0.9, 0.10, 0.09, 9.0], "is_matching_entity": False},
        {"id": "c_dist_placement_group", "label": "Placements 2027 CSD Karthikeyen is also in this group", "res": "group_row", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.64, 0.95, 0.74, 0.5, 0.69, 0.9, 0.10, 0.09, 9.0], "is_matching_entity": False},
        {"id": "c_filter_docs", "label": "Documents", "res": "filter_docs", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.76, 0.30, 0.82, 0.175, 0.79, 0.25, 0.06, 0.015, 4.16], "is_matching_entity": False},
        make_system_cand("c_back_karthik", "back"),
    ]
    for vi, g_karthik in enumerate([
        "open WhatsApp and send message to Karthik :check your mail",
        "open WhatsApp and send hi to karthik",
        "Send message to Karthik: check your email",
    ]):
        st_tok, en_tok = find_token_span(g_karthik, "Karthik", TOKENIZER)
        dataset.append({
            "name": f"wa_screen_b_karthik_clean_var{vi+1}",
            "category": "comm_contact_wa",
            "goal": g_karthik,
            "candidates": wa_cands_karthik_clean,
            "target_idx": 0,  # Karthik.S (You)
            "action_idx": ACTION_TO_IDX["CLICK"],
            "hard_negs": list(range(1, len(wa_cands_karthik_clean))),
            "target_span": (st_tok, en_tok),
            "entity_target": "Karthik",
            "is_terminal": False,
        })

    # B3. Modern WhatsApp Screen B: Bhavani Disambiguation
    wa_cands_bhavani_clean = [
        {"id": "c_bhavani_target", "label": "Bhavani", "res": "contact_name", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.18, 0.95, 0.26, 0.5, 0.22, 0.9, 0.08, 0.072, 11.25], "is_matching_entity": True},
        {"id": "c_ask_btn_b", "label": "ASK", "res": "ask_button", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.85, 0.05, 0.95, 0.11, 0.90, 0.08, 0.10, 0.06, 0.006, 1.66], "is_matching_entity": False},
        make_system_cand("c_submit", "submit focused field"),
        {"id": "c_input_bhavani", "label": "​Bhavani", "res": "search_input", "role": "input", "role_idx": 2, "is_clickable": True, "is_editable": True, "bounds_vec": [0.05, 0.05, 0.85, 0.11, 0.45, 0.08, 0.80, 0.06, 0.048, 13.33], "is_matching_entity": False},
        {"id": "c_dist_voter_bhavani", "label": "voter bhavani Mobile", "res": "contact_row", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.30, 0.95, 0.38, 0.5, 0.34, 0.9, 0.08, 0.072, 11.25], "is_matching_entity": False},
        {"id": "c_dist_bhavani_status", "label": "Bhavani's status. 1 new update Bhavani 9:10 am 2 photos", "res": "status_row", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.40, 0.95, 0.50, 0.5, 0.45, 0.9, 0.10, 0.09, 9.0], "is_matching_entity": False},
        {"id": "c_dist_bhavani_group", "label": "Jolly family picture Jolly family Bhavani is also in this group", "res": "group_row", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.52, 0.95, 0.62, 0.5, 0.57, 0.9, 0.10, 0.09, 9.0], "is_matching_entity": False},
        {"id": "c_filter_photos", "label": "Photos", "res": "filter_photos", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.64, 0.30, 0.70, 0.175, 0.67, 0.25, 0.06, 0.015, 4.16], "is_matching_entity": False},
        make_system_cand("c_back_bhavani", "back"),
    ]
    st_tok, en_tok = find_token_span("open WhatsApp and send hi to Bhavani", "Bhavani", TOKENIZER)
    dataset.append({
        "name": "wa_screen_b_bhavani_clean",
        "category": "comm_contact_wa",
        "goal": "open WhatsApp and send hi to Bhavani",
        "candidates": wa_cands_bhavani_clean,
        "target_idx": 0,  # Bhavani
        "action_idx": ACTION_TO_IDX["CLICK"],
        "hard_negs": list(range(1, len(wa_cands_bhavani_clean))),
        "target_span": (st_tok, en_tok),
        "entity_target": "Bhavani",
        "is_terminal": False,
    })

    # B4. Universal Cross-App: Telegram Global Search Disambiguation (Multiple variations)
    tg_cands_david = [
        {"id": "tg_contact_david", "label": "David Miller last seen recently", "res": "user_cell", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.15, 0.95, 0.25, 0.5, 0.20, 0.9, 0.10, 0.09, 9.0], "is_matching_entity": True},
        {"id": "tg_bot_david", "label": "David AI Assistant Bot @david_ai_bot", "res": "bot_cell", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.26, 0.95, 0.36, 0.5, 0.31, 0.9, 0.10, 0.09, 9.0], "is_matching_entity": False},
        {"id": "tg_global_search", "label": "Search globally for 'David'", "res": "global_search_btn", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.38, 0.95, 0.46, 0.5, 0.42, 0.9, 0.08, 0.072, 11.25], "is_matching_entity": False},
        {"id": "tg_channel_david", "label": "David's Fan Channel 3.4k members", "res": "channel_cell", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.48, 0.95, 0.58, 0.5, 0.53, 0.9, 0.10, 0.09, 9.0], "is_matching_entity": False},
        make_system_cand("c_submit", "submit focused field"),
        make_system_cand("tg_back", "back"),
    ]
    for vi, tg_g in enumerate([
        "Send message to David: ready for call",
        "Message David: where are you",
        "Text David: see you tomorrow",
    ]):
        st_tok, en_tok = find_token_span(tg_g, "David", TOKENIZER)
        dataset.append({
            "name": f"tg_screen_b_david_var{vi+1}",
            "category": "comm_contact_unseen",
            "goal": tg_g,
            "candidates": tg_cands_david,
            "target_idx": 0,  # David Miller
            "action_idx": ACTION_TO_IDX["CLICK"],
            "hard_negs": [1, 2, 3, 4, 5],
            "target_span": (st_tok, en_tok),
            "entity_target": "David",
            "is_terminal": False,
        })

    # B5. Universal Cross-App: SMS / Google Messages Contact Picker
    sms_cands_sarah = [
        {"id": "sms_contact_sarah", "label": "Sarah Jenkins +1 (555) 234-5678", "res": "recipient_row", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.15, 0.95, 0.25, 0.5, 0.20, 0.9, 0.10, 0.09, 9.0], "is_matching_entity": True},
        {"id": "sms_btn_group", "label": "Create group with Sarah", "res": "create_group_btn", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.26, 0.95, 0.34, 0.5, 0.30, 0.9, 0.08, 0.072, 11.25], "is_matching_entity": False},
        {"id": "sms_btn_search_web", "label": "Search web for Sarah", "res": "search_web_btn", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.36, 0.95, 0.44, 0.5, 0.40, 0.9, 0.08, 0.072, 11.25], "is_matching_entity": False},
        {"id": "sms_contact_other", "label": "Sarah Connor +1 (555) 987-6543", "res": "recipient_row", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.46, 0.95, 0.56, 0.5, 0.51, 0.9, 0.10, 0.09, 9.0], "is_matching_entity": False},
        make_system_cand("c_submit", "submit focused field"),
        make_system_cand("sms_back", "back"),
    ]
    for vi, sms_g in enumerate([
        "Text Sarah: see you at the cafe",
        "Send message to Sarah: call me when free",
    ]):
        st_tok, en_tok = find_token_span(sms_g, "Sarah", TOKENIZER)
        dataset.append({
            "name": f"sms_screen_b_sarah_var{vi+1}",
            "category": "comm_contact_unseen",
            "goal": sms_g,
            "candidates": sms_cands_sarah,
            "target_idx": 0,  # Sarah Jenkins
            "action_idx": ACTION_TO_IDX["CLICK"],
            "hard_negs": [1, 2, 3, 4, 5],
            "target_span": (st_tok, en_tok),
            "entity_target": "Sarah",
            "is_terminal": False,
        })

    # B6. Universal Dead-End Recovery: Contact Profile Quick-Action Card (Calling vs Messaging Intent Grounding)
    profile_modal_cands = [
        {"id": "modal_btn_voice", "label": "Voice call", "res": "voice_call_btn", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.15, 0.60, 0.45, 0.68, 0.30, 0.64, 0.30, 0.08, 0.024, 3.75], "is_matching_entity": False},
        {"id": "modal_btn_video", "label": "Video call", "res": "video_call_btn", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.55, 0.60, 0.85, 0.68, 0.70, 0.64, 0.30, 0.08, 0.024, 3.75], "is_matching_entity": False},
        {"id": "modal_btn_info", "label": "Info", "res": "info_btn", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.35, 0.72, 0.65, 0.80, 0.50, 0.76, 0.30, 0.08, 0.024, 3.75], "is_matching_entity": False},
        {"id": "modal_img_pic", "label": "Profile picture", "res": "avatar", "role": "image", "role_idx": 0, "is_clickable": False, "bounds_vec": [0.25, 0.20, 0.75, 0.55, 0.50, 0.375, 0.50, 0.35, 0.175, 1.42], "is_matching_entity": False},
        make_system_cand("c_back_modal", "back"),
    ]
    for g_modal in [
        "open WhatsApp and send hi to Bhavani",
        "Send message to Karthik: check your email",
        "Text Sarah: see you at the cafe",
    ]:
        st_tok, en_tok = find_token_span(g_modal, "back", TOKENIZER)
        dataset.append({
            "name": f"universal_modal_recovery_{hash(g_modal)%1000}",
            "category": "comm_recovery",
            "goal": g_modal,
            "candidates": profile_modal_cands,
            "target_idx": 4,  # BACK is the only correct action when messaging intent has only call options!
            "action_idx": ACTION_TO_IDX["CLICK"],
            "hard_negs": [0, 1, 2, 3],  # Heavy penalty on Voice call, Video call, Info
            "target_span": (0, 0),
            "entity_target": "back",
            "is_terminal": False,
        })

    # B7. Universal Dead-End Recovery: AI Chatbot Screen Hijack Recovery
    meta_ai_screen_cands = [
        {"id": "ai_btn_send_meta", "label": "Send message to Meta AI", "res": "btn_send_ai", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.10, 0.70, 0.90, 0.78, 0.50, 0.74, 0.80, 0.08, 0.064, 10.0], "is_matching_entity": False},
        {"id": "ai_txt_meta_one", "label": "Meta One Expand the possibilities with Meta AI.", "res": "ai_title", "role": "text", "role_idx": 0, "is_clickable": False, "bounds_vec": [0.10, 0.20, 0.90, 0.35, 0.50, 0.275, 0.80, 0.15, 0.12, 5.33], "is_matching_entity": False},
        make_system_cand("c_back_ai_trap", "back"),
    ]
    for g_ai_trap in [
        "open WhatsApp and send hi to karthik",
        "open WhatsApp and send hi to Bhavani",
        "Send message to Rahul: I am on my way",
    ]:
        st_tok, en_tok = find_token_span(g_ai_trap, "back", TOKENIZER)
        dataset.append({
            "name": f"universal_ai_trap_recovery_{hash(g_ai_trap)%1000}",
            "category": "comm_recovery",
            "goal": g_ai_trap,
            "candidates": meta_ai_screen_cands,
            "target_idx": 2,  # BACK to return to real contacts
            "action_idx": ACTION_TO_IDX["CLICK"],
            "hard_negs": [0, 1],
            "target_span": (0, 0),
            "entity_target": "back",
            "is_terminal": False,
        })

    # Turn 3: Screen C - Empty Chat Room (Type Message)
    wa_empty_chat_cands = [
        {"id": "wa_entry_input", "label": "Message", "res": "entry", "role": "input", "role_idx": 2, "is_clickable": True, "is_editable": True, "bounds_vec": [0.12, 0.92, 0.82, 0.98, 0.47, 0.95, 0.70, 0.06, 0.042, 11.66], "is_matching_entity": True},
        {"id": "wa_btn_emoji", "label": "Emoji", "res": "emoji_picker_btn", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.02, 0.92, 0.10, 0.98, 0.06, 0.95, 0.08, 0.06, 0.0048, 1.33], "is_matching_entity": False},
        {"id": "wa_btn_attach", "label": "Attach", "res": "attach_btn", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.84, 0.92, 0.91, 0.98, 0.875, 0.95, 0.07, 0.06, 0.0042, 1.16], "is_matching_entity": False},
        {"id": "wa_btn_mic", "label": "Voice message", "res": "voice_note_btn", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.92, 0.92, 0.98, 0.98, 0.95, 0.95, 0.06, 0.06, 0.0036, 1.0], "is_matching_entity": False},
        {"id": "wa_header_name", "label": "Rahul online", "res": "conversation_contact_name", "role": "text", "role_idx": 0, "is_clickable": False, "bounds_vec": [0.15, 0.03, 0.70, 0.08, 0.425, 0.055, 0.55, 0.05, 0.0275, 11.0], "is_matching_entity": False},
        make_system_cand("wa_back", "back"),
    ]
    wa_screen_c_goals = [
        ("Send message to Rahul: I am on my way", "I am on my way"),
        ("Send message to Karthik: check your email", "check your email"),
        ("open WhatsApp and send message to Karthik :check your mail", "check your mail"),
        ("Message Priya: are you free today", "are you free today"),
        ("Open WhatsApp and send message to Alex: meeting at 5", "meeting at 5"),
        ("Message Mom: I reached home safely", "I reached home safely"),
    ]
    for vi, (g_str, msg_text) in enumerate(wa_screen_c_goals):
        st_tok, en_tok = find_token_span(g_str, msg_text, TOKENIZER)
        dataset.append({
            "name": f"wa_screen_c_type_message_var{vi+1}",
            "category": "comm_compose",
            "goal": g_str,
            "candidates": wa_empty_chat_cands,
            "target_idx": 0,
            "action_idx": ACTION_TO_IDX["TYPE_TEXT"],
            "hard_negs": [1, 2, 3, 4, 5],
            "target_span": (st_tok, en_tok),
            "entity_target": msg_text,
            "is_terminal": False,
        })

    # Turn 4: Screen D - Populated Chat Room (Click Send Button across multiple goals)
    wa_send_goals = [
        ("Send message to Rahul: I am on my way", "I am on my way"),
        ("Send message to Karthik: check your email", "check your email"),
        ("open WhatsApp and send message to Karthik :check your mail", "check your mail"),
        ("Message Priya: are you free today", "are you free today"),
        ("Open WhatsApp and send message to Alex: meeting at 5", "meeting at 5"),
        ("Message Mom: I reached home safely", "I reached home safely"),
    ]
    for vi, (g_str, typed_txt) in enumerate(wa_send_goals):
        cands_send = [
            {"id": "wa_btn_send", "label": "Send", "res": "send", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.88, 0.92, 0.98, 0.98, 0.93, 0.95, 0.10, 0.06, 0.006, 1.66], "is_matching_entity": True},
            {"id": "wa_entry_typed", "label": typed_txt, "res": "entry", "role": "input", "role_idx": 2, "is_clickable": True, "is_editable": True, "bounds_vec": [0.12, 0.92, 0.86, 0.98, 0.49, 0.95, 0.74, 0.06, 0.0444, 12.33], "is_matching_entity": False},
            {"id": "wa_btn_emoji", "label": "Emoji", "res": "emoji_picker_btn", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.02, 0.92, 0.10, 0.98, 0.06, 0.95, 0.08, 0.06, 0.0048, 1.33], "is_matching_entity": False},
            {"id": "wa_btn_attach", "label": "Attach", "res": "attach_btn", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.78, 0.92, 0.86, 0.98, 0.82, 0.95, 0.08, 0.06, 0.0048, 1.33], "is_matching_entity": False},
            make_system_cand("c_submit", "submit focused field", "button"),
            make_system_cand("wa_back", "back"),
        ]
        st_tok, en_tok = find_token_span(g_str, "Send", TOKENIZER)
        dataset.append({
            "name": f"wa_screen_d_send_var{vi+1}",
            "category": "comm_send",
            "goal": g_str,
            "candidates": cands_send,
            "target_idx": 0,
            "action_idx": ACTION_TO_IDX["CLICK"],
            "hard_negs": [1, 2, 3, 4, 5],
            "target_span": (st_tok, en_tok),
            "entity_target": "Send",
            "is_terminal": False,
        })

    # Turn 5: Screen E - Sent Message Visible (Terminal Goal Completion)
    wa_sent_chat_cands = [
        {"id": "wa_bubble_sent", "label": "I am on my way 11:42 AM Delivered", "res": "message_text", "role": "text", "role_idx": 0, "is_clickable": False, "bounds_vec": [0.45, 0.82, 0.95, 0.89, 0.70, 0.855, 0.50, 0.07, 0.035, 7.14], "is_matching_entity": False},
        {"id": "wa_entry_empty", "label": "Message", "res": "entry", "role": "input", "role_idx": 2, "is_clickable": True, "is_editable": True, "bounds_vec": [0.12, 0.92, 0.82, 0.98, 0.47, 0.95, 0.70, 0.06, 0.042, 11.66], "is_matching_entity": False},
        {"id": "wa_btn_mic", "label": "Voice message", "res": "voice_note_btn", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.92, 0.92, 0.98, 0.98, 0.95, 0.95, 0.06, 0.06, 0.0036, 1.0], "is_matching_entity": False},
        make_system_cand("c_complete_goal", "complete", "button"),
        make_system_cand("c_submit", "submit focused field", "button"),
        make_system_cand("wa_back", "back"),
    ]
    dataset.append({
        "name": "wa_screen_e_goal_complete",
        "category": "comm_complete",
        "goal": "Send message to Rahul: I am on my way",
        "candidates": wa_sent_chat_cands,
        "target_idx": 3,  # c_complete_goal
        "action_idx": ACTION_TO_IDX["WAIT"],
        "hard_negs": [0, 1, 2, 4, 5],
        "target_span": (0, 0),
        "entity_target": "complete",
        "is_terminal": True,
    })

    return dataset


def train_v54_model(total_steps: int = 250, is_dry_run: bool = False) -> KarvishModelV54:
    print("=" * 80)
    print("INITIATING KARVISH V54 FULL PRODUCTION TRAINING" if not is_dry_run else "INITIATING KARVISH V54 SMOKE TEST / DRY RUN")
    print(f"Target: Multiplicative Entity Gating + Numerical Scalar Grounding + Milestone Termination")
    print(f"Baseline Parent: {LOCKED_GOLD_CKPT.name}")
    print("=" * 80)

    model = KarvishModelV54().to(DEVICE)

    # Initialize backbone: initialize from current best candidate if available, else from V52-LOCKED-GOLD
    if BEST_V54_CKPT.exists():
        print(f">> Initializing backbone weights from candidate checkpoint: {BEST_V54_CKPT.name}")
        ckpt = torch.load(BEST_V54_CKPT, map_location=DEVICE, weights_only=False)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(sd, strict=False)
        print(f"   Loaded candidate weights from {BEST_V54_CKPT.name}.")
    elif LOCKED_GOLD_CKPT.exists():
        print(f">> Initializing backbone weights from V52-LOCKED-GOLD: {LOCKED_GOLD_CKPT.name}")
        gold_ckpt = torch.load(LOCKED_GOLD_CKPT, map_location=DEVICE, weights_only=False)
        gold_sd = gold_ckpt["model_state_dict"] if "model_state_dict" in gold_ckpt else gold_ckpt

        compatible = {k: v for k, v in gold_sd.items() if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
        model.load_state_dict(compatible, strict=False)
        print(f"   Loaded {len(compatible)} compatible parameter tensors from V52-LOCKED-GOLD.")

    # Freeze base language & spatial backbones ONLY to guarantee physical stability
    for p in model.parameters():
        p.requires_grad = False

    for p in model.cand_fuse.parameters():
        p.requires_grad = True
    for p in model.role_embed.parameters():
        p.requires_grad = True
    for p in model.flags_mlp.parameters():
        p.requires_grad = True
    for p in model.container_attention.parameters():
        p.requires_grad = True
    for p in model.entity_gater.parameters():
        p.requires_grad = True
    for p in model.task_state_engine.parameters():
        p.requires_grad = True
    for p in model.span_pointer_head.parameters():
        p.requires_grad = True
    for p in model.rank_head.parameters():
        p.requires_grad = True
    for p in model.action_head.parameters():
        p.requires_grad = True
    for p in model.numerical_head.parameters():
        p.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = AdamW([
        {"params": model.rank_head.parameters(), "lr": 1.8e-4},
        {"params": model.action_head.parameters(), "lr": 1.5e-4},
        {"params": model.numerical_head.parameters(), "lr": 2.0e-4},
        {"params": model.entity_gater.parameters(), "lr": 2.5e-4},
        {"params": model.task_state_engine.parameters(), "lr": 1.2e-4},
        {"params": model.span_pointer_head.parameters(), "lr": 4.0e-4},
        {"params": model.container_attention.parameters(), "lr": 6e-5},
        {"params": model.cand_fuse.parameters(), "lr": 1.0e-4},
        {"params": model.role_embed.parameters(), "lr": 1.0e-4},
        {"params": model.flags_mlp.parameters(), "lr": 8e-5},
    ], weight_decay=1e-3)
    dataset = build_full_v54_dataset()

    # Balanced category grouping to eliminate linear sampling drift across anchors & new tasks
    category_map: Dict[str, List[Dict[str, Any]]] = {}
    for item in dataset:
        cat = item.get("category", "anchor")
        category_map.setdefault(cat, []).append(item)
    cat_keys = [
        "anchor", "launcher_media", "zepto_search", "zepto_grid", "blinkit",
        "amazon", "flipkart", "price_permutations", "terminal_cart",
        "comm_search", "comm_contact_wa", "comm_contact_unseen", "comm_compose", "comm_send", "comm_recovery", "comm_complete"
    ]
    cat_keys = [k for k in cat_keys if k in category_map]

    print(f">> Full Training Dataset Size: {len(dataset)} verified screen trajectories")
    print(f">> Dataset Categories: { {k: len(v) for k, v in category_map.items()} }")
    print(f">> Optimization Schedule: {total_steps} steps with AdamW + Contrastive InfoNCE + Entity Gating")

    cat_step = {k: 0 for k in cat_keys}
    t0 = time.time()
    for step in range(1, total_steps + 1):
        cat = cat_keys[(step - 1) % len(cat_keys)]
        pool = category_map[cat]
        item = pool[cat_step[cat] % len(pool)]
        cat_step[cat] += 1

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
        c_bounds = torch.tensor([c.get("bounds_vec") or [0.0]*10 for c in candidates], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        c_roles = torch.zeros(1, N, dtype=torch.long, device=DEVICE)
        c_flags = torch.zeros(1, N, 16, dtype=torch.float32, device=DEVICE)
        for i, c in enumerate(candidates):
            role_str = str(c.get("role", "button")).lower()
            if "input" in role_str or "edit" in role_str:
                c_roles[0, i] = 2
            elif "button" in role_str:
                c_roles[0, i] = 1
            else:
                c_roles[0, i] = 0

            if c.get("is_editable") or "input" in role_str or "edit" in role_str:
                c_flags[0, i, 0] = 1.0
            if c.get("is_focused"):
                c_flags[0, i, 1] = 1.0
            if c.get("is_clickable", True):
                c_flags[0, i, 2] = 1.0
        c_pad_mask = torch.ones(1, N, dtype=torch.float32, device=DEVICE)

        # Numerical attribute extraction [price, discount, rating, count]
        c_num = torch.zeros(1, N, 4, dtype=torch.float32, device=DEVICE)
        for i, c in enumerate(candidates):
            p = extract_candidate_price(c)
            if p > 0.0:
                c_num[0, i, 0] = math.log1p(p) / 10.0

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

        ranking = out["ranking_scores"]
        action_logits = out["action_logits"]
        entity_gate = out["entity_gate"]

        t_idx = torch.tensor([item["target_idx"]], dtype=torch.long, device=DEVICE)
        a_idx = torch.tensor([item["action_idx"]], dtype=torch.long, device=DEVICE)

        loss_rank = F.cross_entropy(ranking, t_idx)
        loss_act = F.cross_entropy(action_logits, a_idx)

        # Entity Gating Supervision
        # Distractors like Ginger/Soap should have entity_gate -> 0.0
        loss_entity = torch.tensor(0.0, device=DEVICE)
        if "entity_target" in item:
            ent_target_str = item["entity_target"].lower()
            ent_labels = torch.zeros(1, N, device=DEVICE)
            for i, c in enumerate(candidates):
                if c.get("is_matching_entity") is True:
                    ent_labels[0, i] = 1.0
                elif c.get("is_matching_entity") is False:
                    ent_labels[0, i] = 0.0
                elif ent_target_str in c["label"].lower():
                    ent_labels[0, i] = 1.0
                elif any(dist in c["label"].lower() for dist in ["ginger", "soap", "onion", "shampoo", "ice", "munchies", "cold drinks"]):
                    ent_labels[0, i] = 0.0
                else:
                    ent_labels[0, i] = 0.5  # neutral for system items
            loss_entity = F.binary_cross_entropy(entity_gate, ent_labels)

        # Contrastive Triplet Margin Loss: target must beat every hard negative
        loss_margin = torch.tensor(0.0, device=DEVICE)
        if item.get("hard_negs"):
            pos_score = ranking[0, item["target_idx"]]
            neg_scores = ranking[0, item["hard_negs"]]
            margin_diff = torch.clamp(2.5 - (pos_score - neg_scores), min=0.0)
            active_viol = margin_diff[margin_diff > 0]
            if len(active_viol) > 0:
                loss_margin = active_viol.mean()
            else:
                loss_margin = torch.tensor(0.0, device=DEVICE)

        # Terminal Milestone Loss
        term_target = torch.tensor([1.0 if item.get("is_terminal") else 0.0], device=DEVICE)
        loss_term = F.binary_cross_entropy(out["termination_prob"], term_target)

        # Phase D Learned Span Pointer Supervision (active on text entry actions)
        loss_span = torch.tensor(0.0, device=DEVICE)
        if item.get("action_idx") == ACTION_TO_IDX["TYPE_TEXT"] and "target_span" in item and item["target_span"] != (0, 0):
            st_target = torch.tensor([item["target_span"][0]], dtype=torch.long, device=DEVICE)
            en_target = torch.tensor([item["target_span"][1]], dtype=torch.long, device=DEVICE)
            loss_span = F.cross_entropy(out["span_start_logits"], st_target) + F.cross_entropy(out["span_end_logits"], en_target)

        total_loss = loss_rank + loss_act + 2.0 * loss_margin + 1.5 * loss_entity + 0.3 * loss_term + 1.0 * loss_span
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()

        if is_dry_run or step % 50 == 0 or step == total_steps or step <= 5:
            print(f"  [Step {step:03d}/{total_steps:03d}] Total: {total_loss.item():.4f} | Rank: {loss_rank.item():.4f} | Entity: {loss_entity.item():.4f} | Span: {loss_span.item():.4f} | Margin: {loss_margin.item():.4f}")

    elapsed = time.time() - t0
    if is_dry_run:
        print(f"\n>> Smoke Test / Dry Run Completed in {elapsed:.2f}s across {total_steps} steps! (Production weights NOT modified)")
        return model

    print(f"\n>> Full Training Completed in {elapsed:.2f}s!")

    # Save to production V54 best checkpoint
    BEST_V54_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "total_steps": total_steps,
        "parameters": sum(p.numel() for p in model.parameters()),
        "baseline_parent": "V52-LOCKED-GOLD",
    }, BEST_V54_CKPT)
    print(f">> Promoted Production Checkpoint to: {BEST_V54_CKPT.name} ({BEST_V54_CKPT.stat().st_size / (1024*1024):.2f} MB)")

    return model


def eval_screen_universal(model: KarvishModelV54, goal: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    N = len(candidates)
    enc = TOKENIZER(goal, max_length=32, truncation=True, return_tensors="pt")
    g_ids = enc["input_ids"].to(DEVICE)
    g_mask = enc["attention_mask"].float().to(DEVICE)
    g_tokens = TOKENIZER.convert_ids_to_tokens(g_ids[0])

    lbl_texts = [c["label"] for c in candidates]
    c_enc = TOKENIZER(lbl_texts, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
    c_lbl_ids = c_enc["input_ids"].unsqueeze(0).to(DEVICE)
    c_lbl_mask = c_enc["attention_mask"].float().unsqueeze(0).to(DEVICE)
    c_bounds = torch.tensor([c.get("bounds_vec") or [0.0]*10 for c in candidates], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    c_roles = torch.zeros(1, N, dtype=torch.long, device=DEVICE)
    c_flags = torch.zeros(1, N, 16, dtype=torch.float32, device=DEVICE)
    for i, c in enumerate(candidates):
        role_str = str(c.get("role", "button")).lower()
        if "input" in role_str or "edit" in role_str:
            c_roles[0, i] = 2
        elif "button" in role_str:
            c_roles[0, i] = 1
        else:
            c_roles[0, i] = 0

        if c.get("is_editable") or "input" in role_str or "edit" in role_str:
            c_flags[0, i, 0] = 1.0
        if c.get("is_focused"):
            c_flags[0, i, 1] = 1.0
        if c.get("is_clickable", True):
            c_flags[0, i, 2] = 1.0
    c_pad_mask = torch.ones(1, N, dtype=torch.float32, device=DEVICE)
    c_num = torch.zeros(1, N, 4, dtype=torch.float32, device=DEVICE)
    for i, c in enumerate(candidates):
        p = extract_candidate_price(c)
        if p > 0.0:
            c_num[0, i, 0] = math.log1p(p) / 10.0

    with torch.no_grad():
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

    ranking = out["ranking_scores"][0].cpu().numpy()
    pred_act_idx = int(torch.argmax(out["action_logits"][0]).item())
    pred_act_str = IDX_TO_ACTION.get(pred_act_idx, "UNKNOWN")
    pred_s = int(torch.argmax(out["span_start_logits"][0]).item())
    pred_e = int(torch.argmax(out["span_end_logits"][0]).item())
    ext_tokens = g_tokens[pred_s : min(pred_e + 1, len(g_tokens))] if pred_s <= pred_e else []
    ext_text = TOKENIZER.convert_tokens_to_string(ext_tokens)

    sorted_i = list(range(N))
    sorted_i.sort(key=lambda idx: ranking[idx], reverse=True)

    return {
        "ranking": ranking,
        "sorted_indices": sorted_i,
        "top_candidate": candidates[sorted_i[0]],
        "action_type": pred_act_str,
        "extracted_text": ext_text,
        "entity_gate": out["entity_gate"][0].cpu().numpy(),
        "termination_prob": float(out["termination_prob"][0].item()),
    }


def evaluate_v54_production_gates(model: KarvishModelV54) -> bool:
    print("\n" + "=" * 80)
    print("RUNNING V54 POST-TRAINING PRODUCTION CERTIFICATION AUDIT (15 GATES)")
    print("=" * 80)
    model.eval()

    def eval_screen(goal: str, candidates: List[Dict[str, Any]]):
        return eval_screen_universal(model, goal, candidates)

    all_passed = True

    # Gate 1: Anchor Task Invariance
    print("\n[GATE 1] 12 Frozen Anchor Task Invariance:")
    anchor_screens = get_test_screens()
    passed_anchors = 0
    for s in anchor_screens:
        res = eval_screen(s["goal"], s["candidates"])
        top_lbl = res["top_candidate"]["label"]
        exp_tgt = s["expected_target"]
        passed = (exp_tgt.lower() in top_lbl.lower())
        task_nm = s.get("name", s.get("task_name", "anchor"))
        if passed:
            passed_anchors += 1
            print(f"  + PASS: {task_nm:<28} -> {top_lbl[:35]}")
        else:
            print(f"  - FAIL: {task_nm:<28} -> Exp '{exp_tgt}' got '{top_lbl}'")
    g1_ok = (passed_anchors == len(anchor_screens))
    print(f">> Gate 1 Result: {passed_anchors}/{len(anchor_screens)} (Required: 12/12) -> {'PASSED' if g1_ok else 'FAILED'}")
    all_passed &= g1_ok

    # Gate 2: Zepto Home Search Grounding
    print("\n[GATE 2] Zepto Home Search Grounding:")
    if ZEPTO_HOME_XML.exists():
        zh_cands = parse_xml_candidates(ZEPTO_HOME_XML)
        search_node = next((c for c in zh_cands if "homepage-search-box" in c.get("res", "")), None)
        if search_node is None:
            search_node = next((c for c in zh_cands if "search" in c["label"].lower()), None)
        cart_btn = next((c for c in zh_cands if "floating-cart-button" in c.get("res", "") or "Cart" in c["label"]), None)
        ordered_cands = [search_node, cart_btn] + [c for c in zh_cands if c not in [search_node, cart_btn]][:22]
        ordered_cands = [c for c in ordered_cands if c is not None]
        res_g2 = eval_screen("Open Zepto and add the cheapest tomato to the cart", ordered_cands)
        top_lbl = res_g2["top_candidate"]["label"]
        top_act = res_g2["action_type"]
        g2_ok = ("search" in top_lbl.lower() or search_node["id"] == res_g2["top_candidate"]["id"])
        print(f"  Top Candidate: {top_lbl}")
        print(f"  Action Type:    {top_act}")
        print(f">> Gate 2 Result: {'PASSED' if g2_ok else 'FAILED'}")
        all_passed &= g2_ok

    # Gate 3: Exploration Scroll Grounding
    print("\n[GATE 3] Above-the-fold Scroll Exploration Grounding:")
    if TOMATO_ABOVE_XML.exists():
        above_cands = parse_xml_candidates(TOMATO_ABOVE_XML)
        scroll_cand = make_system_cand("c_scroll_down", "scroll down", "system")
        complete_cand = make_system_cand("c_complete_goal", "complete", "button")
        cands_above = [scroll_cand] + above_cands[:20] + [complete_cand]
        res_g3 = eval_screen("Search cheapest tomato and add to cart", cands_above)
        top_lbl = res_g3["top_candidate"]["label"]
        top_act = res_g3["action_type"]
        g3_ok = ("scroll down" in top_lbl.lower())
        print(f"  Top Candidate: {top_lbl}")
        print(f"  Action Type:    {top_act}")
        print(f">> Gate 3 Result: {'PASSED' if g3_ok else 'FAILED'}")
        all_passed &= g3_ok

    # Gate 4 & 5: Dense 28-Candidate Grid (12 Tomatoes + 12 Distractors)
    print("\n[GATE 4 & 5] Dense 28-Candidate Grid (12 Tomatoes, 12 Low-Price Distractors, System Controls):")
    dense_eval_cands = create_dense_30_candidate_screen()
    res_dense = eval_screen("Search cheapest tomato and add to cart", dense_eval_cands)
    top_dense_lbl = res_dense["top_candidate"]["label"]
    top_dense_id = res_dense["top_candidate"]["id"]

    # Verify that Tomato Local ₹13 is TOP-1 out of all 28 candidates
    is_top1_cheap_tomato = (top_dense_id == "c_tomato_13" and "₹13" in top_dense_lbl)
    
    # Check that ALL 12 distractors are ranked below Tomato Local ₹13
    tomato_score = res_dense["ranking"][0]
    distractor_indices = list(range(12, 24))
    max_distractor_score = max(res_dense["ranking"][i] for i in distractor_indices)
    competing_tomato_indices = list(range(1, 12))
    max_comp_tomato_score = max(res_dense["ranking"][i] for i in competing_tomato_indices)

    submit_cand_idx = next((i for i, c in enumerate(dense_eval_cands) if c["id"] == "c_submit"), None)
    submit_dense_score = res_dense["ranking"][submit_cand_idx] if submit_cand_idx is not None else -999.0

    g4_ok = (is_top1_cheap_tomato and tomato_score > max_distractor_score and tomato_score > submit_dense_score)
    g5_ok = (tomato_score > max_comp_tomato_score)

    print(f"  Total Candidates on Screen: {len(dense_eval_cands)} (12 Tomatoes, 12 Distractors, 5 Controls)")
    print(f"  Top Pick (#1 out of {len(dense_eval_cands)}): {top_dense_lbl}")
    print(f"  Tomato Local ₹13 Score: {tomato_score:.3f}")
    print(f"  Submit Score: {submit_dense_score:.3f}")
    print(f"  Max Distractor Score (out of 12 distractors): {max_distractor_score:.3f}")
    print(f"  Max Competing Tomato Score (out of 11 tomatoes): {max_comp_tomato_score:.3f}")
    print(f">> Gate 4 (Distractor & Submit Suppression across 13 items): {'PASSED' if g4_ok else 'FAILED'}")
    print(f">> Gate 5 (True Cheapest out of 12 tomatoes):              {'PASSED' if g5_ok else 'FAILED'}")
    all_passed &= (g4_ok and g5_ok)

    # Gate 6: Monotonic Price Generalization on Unseen Numbers (₹19 vs ₹37)
    print("\n[GATE 6] Monotonic Price Generalization on Unseen Numbers (Tomato ₹19 vs ₹37):")
    unseen_cands = [
        {"id": "u_tom_19", "label": "Farm Fresh Tomato 500g ₹19 Add to Cart", "res": "btn_u1", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.1, 0.4, 0.4, 0.6, 0.25, 0.5, 0.3, 0.2, 0.06, 1.5], "price": 19.0},
        {"id": "u_tom_37", "label": "Organic Red Tomato 500g ₹37 Add to Cart", "res": "btn_u2", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.6, 0.4, 0.9, 0.6, 0.75, 0.5, 0.3, 0.2, 0.06, 1.5], "price": 37.0},
        {"id": "u_gin_11", "label": "Raw Ginger 100g ₹11 Add to Cart", "res": "btn_u3", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.1, 0.7, 0.4, 0.9, 0.25, 0.8, 0.3, 0.2, 0.06, 1.5], "price": 11.0},
        make_system_cand("sys_scroll_down", "scroll down"),
    ]
    res_g6 = eval_screen("Search cheapest tomato and add to cart", unseen_cands)
    g6_ok = ("₹19" in res_g6["top_candidate"]["label"] and res_g6["ranking"][0] > res_g6["ranking"][1])
    print(f"  Top Candidate: {res_g6['top_candidate']['label']}")
    print(f"  Tomato ₹19 Score: {res_g6['ranking'][0]:.3f} vs Tomato ₹37 Score: {res_g6['ranking'][1]:.3f}")
    print(f">> Gate 6 Result: {'PASSED' if g6_ok else 'FAILED'}")
    all_passed &= g6_ok

    # Gate 7: Terminal Cart Milestone Completion
    print("\n[GATE 7] Terminal Cart Milestone Completion (Stop after Add to Cart):")
    cart_eval_cands = [
        make_system_cand("c_complete_goal", "complete", "button"),
        {"id": "btnEnableService", "label": "CONFIGURE", "res": "btnEnableService", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.1, 0.8, 0.9, 0.9, 0.5, 0.85, 0.8, 0.1, 0.08, 8.0]},
        {"id": "c_checkout", "label": "Click to Pay ₹428", "res": "c_checkout", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.85, 0.95, 0.92, 0.5, 0.885, 0.9, 0.07, 0.063, 12.8]},
        {"id": "c_tomato_item", "label": "Tomato Local 250g in cart", "res": "item", "role": "text", "role_idx": 0, "is_clickable": False, "is_editable": False, "bounds_vec": [0.1, 0.3, 0.9, 0.4, 0.5, 0.35, 0.8, 0.1, 0.08, 8.0]}
    ]
    res_g7 = eval_screen("Search cheapest tomato and add to cart and stop", cart_eval_cands)
    top_lbl = res_g7["top_candidate"]["label"]
    top_act = res_g7["action_type"]
    g7_ok = ("complete" in top_lbl.lower() and (top_act == "WAIT" or res_g7["termination_prob"] > 0.5))
    print(f"  Top Candidate:   {top_lbl}")
    print(f"  Action Type:     {top_act}")
    print(f"  Terminal Prob:   {res_g7['termination_prob']:.4f}")
    print(f">> Gate 7 Result: {'PASSED' if g7_ok else 'FAILED'}")
    all_passed &= g7_ok

    # Gate 8: Zepto Search Grounding (Search Input Box or Matching Chip beats Ice Distractor)
    print("\n[GATE 8] Zepto Search Grounding (Search Input or Matching Chip beats Ice & Complete):")
    zepto_gate8_cands = [
        {"id": "chip_fresh_tomato", "label": "Fresh tomatoes", "res": "chip_fresh", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.12, 0.45, 0.18, 0.25, 0.15, 0.4, 0.06, 0.024, 6.66], "is_matching_entity": True},
        {"id": "search_input_field", "label": "Search for products, brands and more", "res": "search_input", "role": "input", "role_idx": 2, "is_clickable": True, "is_editable": True, "bounds_vec": [0.05, 0.04, 0.95, 0.10, 0.5, 0.07, 0.9, 0.06, 0.054, 15.0], "is_matching_entity": False},
        {"id": "chip_ice", "label": "Ice", "res": "chip_ice", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.20, 0.25, 0.26, 0.15, 0.23, 0.2, 0.06, 0.012, 3.33], "is_matching_entity": False},
        {"id": "chip_munchies", "label": "Munchies", "res": "chip_munchies", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.30, 0.20, 0.55, 0.26, 0.425, 0.23, 0.25, 0.06, 0.015, 4.16], "is_matching_entity": False},
        make_system_cand("c_complete_goal", "complete", "button"),
        make_system_cand("c_press_back", "back"),
    ]
    res_g8 = eval_screen("Open Zepto and search cheapest tomato", zepto_gate8_cands)
    top_lbl = res_g8["top_candidate"]["label"]
    top_act = res_g8["action_type"]
    ext_txt = res_g8["extracted_text"]

    is_search_input = ("search" in top_lbl.lower() and top_act == "TYPE_TEXT" and "tomato" in ext_txt.lower())
    is_matching_chip = ("fresh" in top_lbl.lower() and "tomato" in top_lbl.lower() and top_act == "CLICK")
    g8_ok = ((is_search_input or is_matching_chip) and res_g8["top_candidate"]["id"] != "chip_ice" and "ice" not in top_lbl.lower())
    print(f"  Top Candidate:   {top_lbl}")
    print(f"  Action Type:     {top_act}")
    print(f"  Extracted Text:  '{ext_txt}'")
    print(f"  Top Score:       {res_g8['ranking'][0]:.3f}")
    print(f">> Gate 8 Result: {'PASSED' if g8_ok else 'FAILED'}")
    all_passed &= g8_ok

    # Gate 9: Blinkit Quick-Commerce Direct Grid Add
    print("\n[GATE 9] Blinkit Quick-Commerce Direct Grid Add (Cheapest Milk ₹27 over ₹32, ₹48 & Bread distractor):")
    blinkit_eval_cands = [
        {"id": "b_milk_27", "label": "Amul Taaza Toned Fresh Milk 500ml ₹27 ADD", "res": "btn_add_milk27", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.25, 0.48, 0.55, 0.265, 0.40, 0.43, 0.30, 0.129, 1.43], "price": 27.0, "is_matching_entity": True},
        {"id": "b_milk_32", "label": "Nandini GoodLife Toned Milk 500ml ₹32 ADD", "res": "btn_add_milk32", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.52, 0.25, 0.95, 0.55, 0.735, 0.40, 0.43, 0.30, 0.129, 1.43], "price": 32.0, "is_matching_entity": True},
        {"id": "b_milk_48", "label": "Country Delight Cow Milk 500ml ₹48 ADD", "res": "btn_add_milk48", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.58, 0.48, 0.88, 0.265, 0.73, 0.43, 0.30, 0.129, 1.43], "price": 48.0, "is_matching_entity": True},
        {"id": "b_dist_bread", "label": "Britannia Milk Bread ₹20 ADD", "res": "btn_add_bread", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.52, 0.58, 0.95, 0.88, 0.735, 0.73, 0.43, 0.30, 0.129, 1.43], "price": 20.0, "is_matching_entity": False},
        {"id": "b_dist_soap", "label": "Dove Bathing Soap 100g ₹15 ADD", "res": "btn_add_soap", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.89, 0.48, 0.99, 0.265, 0.94, 0.43, 0.10, 0.043, 4.3], "price": 15.0, "is_matching_entity": False},
        make_system_cand("b_scroll_down", "scroll down"),
    ]
    res_g9 = eval_screen("Search milk on Blinkit and pick the cheapest", blinkit_eval_cands)
    top_lbl_g9 = res_g9["top_candidate"]["label"]
    top_act_g9 = res_g9["action_type"]
    g9_ok = ("₹27" in top_lbl_g9 and top_act_g9 == "CLICK" and "bread" not in top_lbl_g9.lower() and "soap" not in top_lbl_g9.lower())
    print(f"  Top Candidate: {top_lbl_g9}")
    print(f"  Action Type:   {top_act_g9}")
    print(f"  Score:         {res_g9['ranking'][0]:.3f}")
    print(f">> Gate 9 Result: {'PASSED' if g9_ok else 'FAILED'}")
    all_passed &= g9_ok

    # Gate 10: Amazon Traditional E-Commerce 2-Step PDP Navigation
    print("\n[GATE 10] Amazon 2-Step E-Commerce (Grid Product Card ₹299 -> PDP Add to Cart):")
    amz_eval_grid_cands = [
        {"id": "amz_mouse_299", "label": "Portronics Toad 23 Wireless Optical Mouse 2.4GHz with USB Nano Receiver ₹299 M.R.P: ₹599 (50% off)", "res": "asin_card_299", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.15, 0.95, 0.40, 0.5, 0.275, 0.9, 0.25, 0.225, 3.6], "price": 299.0, "is_matching_entity": True},
        {"id": "amz_mouse_595", "label": "Logitech B170 Wireless Mouse Optical Tracking 12 Months Battery ₹595 M.R.P: ₹895", "res": "asin_card_595", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.42, 0.95, 0.67, 0.5, 0.545, 0.9, 0.25, 0.225, 3.6], "price": 595.0, "is_matching_entity": True},
        {"id": "amz_mouse_749", "label": "HP 240 Bluetooth Wireless Mouse Ergonomic Design ₹749", "res": "asin_card_749", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.69, 0.95, 0.94, 0.5, 0.815, 0.9, 0.25, 0.225, 3.6], "price": 749.0, "is_matching_entity": True},
        {"id": "amz_dist_pad", "label": "Speed Type Waterproof Gaming Mouse Pad ₹99", "res": "asin_card_pad", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.05, 0.95, 0.14, 0.5, 0.095, 0.9, 0.09, 0.081, 10.0], "price": 99.0, "is_matching_entity": False},
        {"id": "amz_dist_batt", "label": "Duracell AA Alkaline Batteries Pack of 4 ₹150", "res": "asin_card_batt", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.95, 0.95, 1.0, 0.5, 0.975, 0.9, 0.05, 0.045, 18.0], "price": 150.0, "is_matching_entity": False},
        make_system_cand("amz_scroll_down", "scroll down"),
    ]
    res_g10_grid = eval_screen("Search wireless mouse on Amazon and buy the cheapest", amz_eval_grid_cands)
    top_lbl_g10_grid = res_g10_grid["top_candidate"]["label"]
    top_act_g10_grid = res_g10_grid["action_type"]
    g10_grid_ok = ("₹299" in top_lbl_g10_grid and top_act_g10_grid == "CLICK" and "pad" not in top_lbl_g10_grid.lower())
    print(f"  [Screen A - Grid] Top Card: {top_lbl_g10_grid[:45]}... (Action: {top_act_g10_grid})")

    amz_eval_pdp_cands = [
        {"id": "amz_btn_add_to_cart", "label": "Add to Cart", "res": "add-to-cart-button", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.70, 0.95, 0.77, 0.5, 0.735, 0.9, 0.07, 0.063, 12.8], "is_matching_entity": True},
        {"id": "amz_btn_buy_now", "label": "Buy Now", "res": "buy-now-button", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.79, 0.95, 0.86, 0.5, 0.825, 0.9, 0.07, 0.063, 12.8], "is_matching_entity": False},
        {"id": "amz_btn_wishlist", "label": "Add to Wish List", "res": "wishlist-button", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.88, 0.95, 0.94, 0.5, 0.91, 0.9, 0.06, 0.054, 15.0], "is_matching_entity": False},
        {"id": "amz_txt_title", "label": "Portronics Toad 23 Wireless Optical Mouse (Black) ₹299", "res": "title", "role": "text", "role_idx": 0, "is_clickable": False, "bounds_vec": [0.05, 0.1, 0.95, 0.25, 0.5, 0.175, 0.9, 0.15, 0.135, 6.0], "price": 299.0, "is_matching_entity": True},
        make_system_cand("amz_back", "back"),
    ]
    res_g10_pdp = eval_screen("Search wireless mouse on Amazon and buy the cheapest", amz_eval_pdp_cands)
    top_lbl_g10_pdp = res_g10_pdp["top_candidate"]["label"]
    top_act_g10_pdp = res_g10_pdp["action_type"]
    g10_pdp_ok = ("add to cart" in top_lbl_g10_pdp.lower() and top_act_g10_pdp == "CLICK")
    print(f"  [Screen B - PDP]  Top Action: {top_lbl_g10_pdp} (Action: {top_act_g10_pdp})")
    g10_ok = (g10_grid_ok and g10_pdp_ok)
    print(f">> Gate 10 Result: {'PASSED' if g10_ok else 'FAILED'}")
    all_passed &= g10_ok

    # Gate 11: Flipkart Traditional E-Commerce 2-Step PDP Navigation
    print("\n[GATE 11] Flipkart 2-Step E-Commerce (Grid Neckband Card ₹899 -> PDP Add to Cart):")
    fk_eval_grid_cands = [
        {"id": "fk_hp_899", "label": "boAt Rockerz 255 Bluetooth Neckband ₹899 ₹2,990 69% off Free delivery", "res": "product_card_899", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.15, 0.95, 0.40, 0.5, 0.275, 0.9, 0.25, 0.225, 3.6], "price": 899.0, "is_matching_entity": True},
        {"id": "fk_hp_1499", "label": "OnePlus Bullets Wireless Z2 Bluetooth Neckband ₹1,499 ₹2,299 34% off", "res": "product_card_1499", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.42, 0.95, 0.67, 0.5, 0.545, 0.9, 0.25, 0.225, 3.6], "price": 1499.0, "is_matching_entity": True},
        {"id": "fk_hp_1699", "label": "Sony WI-C100 Wireless In-Ear Bluetooth Headphones ₹1,699 ₹2,790 39% off", "res": "product_card_1699", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.69, 0.95, 0.94, 0.5, 0.815, 0.9, 0.25, 0.225, 3.6], "price": 1699.0, "is_matching_entity": True},
        {"id": "fk_dist_case", "label": "Protective Hard Case Pouch for Headphones ₹199 ₹499", "res": "product_card_case", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.05, 0.95, 0.14, 0.5, 0.095, 0.9, 0.09, 0.081, 10.0], "price": 199.0, "is_matching_entity": False},
        {"id": "fk_dist_cable", "label": "Aux Audio Cable 3.5mm Male to Male ₹99", "res": "product_card_aux", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.95, 0.95, 1.0, 0.5, 0.975, 0.9, 0.05, 0.045, 18.0], "price": 99.0, "is_matching_entity": False},
        make_system_cand("fk_scroll_down", "scroll down"),
    ]
    res_g11_grid = eval_screen("Search bluetooth headphones on Flipkart and buy the cheapest", fk_eval_grid_cands)
    top_lbl_g11_grid = res_g11_grid["top_candidate"]["label"]
    top_act_g11_grid = res_g11_grid["action_type"]
    g11_grid_ok = ("₹899" in top_lbl_g11_grid and top_act_g11_grid == "CLICK" and "pouch" not in top_lbl_g11_grid.lower() and "case" not in top_lbl_g11_grid.lower())
    print(f"  [Screen A - Grid] Top Card: {top_lbl_g11_grid[:45]}... (Action: {top_act_g11_grid})")

    fk_eval_pdp_cands = [
        {"id": "fk_btn_add_to_cart", "label": "Add to Cart", "res": "button_add_to_cart", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.91, 0.48, 0.98, 0.265, 0.945, 0.43, 0.07, 0.0301, 6.14], "is_matching_entity": True},
        {"id": "fk_btn_buy_now", "label": "Buy Now", "res": "button_buy_now", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.52, 0.91, 0.95, 0.98, 0.735, 0.945, 0.43, 0.07, 0.0301, 6.14], "is_matching_entity": False},
        {"id": "fk_txt_title", "label": "boAt Rockerz 255 Bluetooth Neckband ₹899", "res": "title", "role": "text", "role_idx": 0, "is_clickable": False, "bounds_vec": [0.05, 0.1, 0.95, 0.25, 0.5, 0.175, 0.9, 0.15, 0.135, 6.0], "price": 899.0, "is_matching_entity": True},
        make_system_cand("fk_back", "back"),
    ]
    res_g11_pdp = eval_screen("Search bluetooth headphones on Flipkart and buy the cheapest", fk_eval_pdp_cands)
    top_lbl_g11_pdp = res_g11_pdp["top_candidate"]["label"]
    top_act_g11_pdp = res_g11_pdp["action_type"]
    g11_pdp_ok = ("add to cart" in top_lbl_g11_pdp.lower() and top_act_g11_pdp == "CLICK")
    print(f"  [Screen B - PDP]  Top Action: {top_lbl_g11_pdp} (Action: {top_act_g11_pdp})")
    g11_ok = (g11_grid_ok and g11_pdp_ok)
    # Gate 12: Out-of-Stock Fallback Exploration (Scroll down beats Potato, Onion, Ginger when Tomato is absent)
    print("\n[GATE 12] Out-of-Stock Fallback Exploration (Scroll down beats Potato & Onion when Tomato is absent):")
    oos_eval_cands = [
        {"id": "z_potato", "label": "Fresh Potato 1kg ₹30 ADD", "res": "btn_add_potato", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.25, 0.48, 0.55, 0.265, 0.40, 0.43, 0.30, 0.129, 1.43], "price": 30.0, "is_matching_entity": False},
        {"id": "z_onion", "label": "Red Onion 1kg ₹45 ADD", "res": "btn_add_onion", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.52, 0.25, 0.95, 0.55, 0.735, 0.40, 0.43, 0.30, 0.129, 1.43], "price": 45.0, "is_matching_entity": False},
        {"id": "z_ginger", "label": "Fresh Ginger 100g ₹15 ADD", "res": "btn_add_ginger", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.58, 0.48, 0.88, 0.265, 0.73, 0.43, 0.30, 0.129, 1.43], "price": 15.0, "is_matching_entity": False},
        make_system_cand("c_scroll_down", "scroll down", "system"),
        make_system_cand("c_complete_goal", "complete", "button"),
        make_system_cand("c_submit", "submit focused field", "button"),
        make_system_cand("c_press_back", "back"),
    ]
    res_g12 = eval_screen("Search cheapest tomato and add to cart", oos_eval_cands)
    top_lbl_g12 = res_g12["top_candidate"]["label"]
    top_act_g12 = res_g12["action_type"]
    g12_ok = ("scroll down" in top_lbl_g12.lower() and top_act_g12 == "SCROLL")
    print(f"  Top Candidate: {top_lbl_g12}")
    print(f"  Action Type:   {top_act_g12}")
    print(f"  Score:         {res_g12['ranking'][0]:.3f}")
    print(f">> Gate 12 Result: {'PASSED' if g12_ok else 'FAILED'}")
    all_passed &= g12_ok

    # Gate 13: Quick-Commerce Post-Add Goal Completion
    print("\n[GATE 13] Post-Add Goal Completion (complete beats other cards and submit after item is added):")
    post_add_eval_cands = [
        {"id": "b_milk_27_added", "label": "Amul Taaza Toned Fresh Milk 500ml ₹27 Decrease quantity quantity 1 Increase quantity", "res": "btn_add_milk27", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.25, 0.48, 0.55, 0.265, 0.40, 0.43, 0.30, 0.129, 1.43], "price": 27.0, "is_matching_entity": True},
        {"id": "b_milk_32", "label": "Nandini GoodLife Toned Milk 500ml ₹32 ADD", "res": "btn_add_milk32", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.52, 0.25, 0.95, 0.55, 0.735, 0.40, 0.43, 0.30, 0.129, 1.43], "price": 32.0, "is_matching_entity": True},
        {"id": "b_view_cart", "label": "View cart 1 item ₹27", "res": "view_cart", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.90, 0.95, 0.97, 0.5, 0.935, 0.9, 0.07, 0.063, 12.8], "is_matching_entity": False},
        make_system_cand("c_submit", "submit focused field", "button"),
        make_system_cand("c_complete_goal", "complete", "button"),
    ]
    res_g13 = eval_screen("Search milk on Blinkit and pick the cheapest and stop", post_add_eval_cands)
    top_lbl_g13 = res_g13["top_candidate"]["label"]
    g13_ok = (res_g13["top_candidate"]["id"] == "c_complete_goal" or "complete" in top_lbl_g13.lower())
    print(f"  Top Candidate:   {top_lbl_g13}")
    print(f"  Terminal Prob:   {res_g13['termination_prob']:.4f}")
    print(f">> Gate 13 Result: {'PASSED' if g13_ok else 'FAILED'}")
    all_passed &= g13_ok

    # Gate 14: Zepto Potato Results Grid (Cheapest Potato ₹14 beats higher-priced potatoes, distractors, and submit)
    print("\n[GATE 14] Zepto Potato Grid (Organically Grown Baby Potato ₹14 beats other potatoes, distractors, and submit):")
    potato_eval_cands = [
        {"id": "c_potato_14", "label": "Organically Grown Baby Potato 250g ₹14 ADD", "res": "btn_add_p14", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.25, 0.48, 0.55, 0.265, 0.40, 0.43, 0.30, 0.129, 1.43], "price": 14.0, "is_matching_entity": True},
        {"id": "c_potato_22", "label": "Potato 500g ₹22 ADD", "res": "btn_add_p22", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.52, 0.25, 0.95, 0.55, 0.735, 0.40, 0.43, 0.30, 0.129, 1.43], "price": 22.0, "is_matching_entity": True},
        {"id": "c_potato_23", "label": "Baby Potato 500g ₹23 ADD", "res": "btn_add_p23", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.58, 0.48, 0.88, 0.265, 0.73, 0.43, 0.30, 0.129, 1.43], "price": 23.0, "is_matching_entity": True},
        {"id": "c_potato_41", "label": "Bloom Potato 1kg ₹41 ADD", "res": "btn_add_p41", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.52, 0.58, 0.95, 0.88, 0.735, 0.73, 0.43, 0.30, 0.129, 1.43], "price": 41.0, "is_matching_entity": True},
        {"id": "c_coriander_7", "label": "Fresh Coriander Leaves ₹7 ADD", "res": "btn_add_c7", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.05, 0.89, 0.48, 0.99, 0.265, 0.94, 0.43, 0.10, 0.043, 4.3], "price": 7.0, "is_matching_entity": False},
        {"id": "c_curry_5", "label": "Curry Leaves Pack ₹5 ADD", "res": "btn_add_c5", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.52, 0.89, 0.95, 0.99, 0.735, 0.94, 0.43, 0.10, 0.043, 4.3], "price": 5.0, "is_matching_entity": False},
        make_system_cand("c_submit", "submit focused field", "button"),
        make_system_cand("c_scroll_down", "scroll down", "system"),
        make_system_cand("c_complete_goal", "complete", "button"),
        make_system_cand("c_press_back", "back"),
    ]
    res_g14 = eval_screen("Open zepto and search cheapest potato and add to cart", potato_eval_cands)
    top_lbl_g14 = res_g14["top_candidate"]["label"]
    top_act_g14 = res_g14["action_type"]
    p14_score = res_g14["ranking"][0]
    submit_score = res_g14["ranking"][6]
    max_other_potato = max(res_g14["ranking"][1], res_g14["ranking"][2], res_g14["ranking"][3])
    max_distractor = max(res_g14["ranking"][4], res_g14["ranking"][5])
    g14_ok = ("₹14" in top_lbl_g14 and top_act_g14 == "CLICK" and p14_score > submit_score and p14_score > max_other_potato and p14_score > max_distractor)
    print(f"  Top Candidate:       {top_lbl_g14}")
    print(f"  Action Type:         {top_act_g14}")
    print(f"  Potato ₹14 Score:    {p14_score:.3f}")
    print(f"  Submit Score:        {submit_score:.3f} (Margin: {p14_score - submit_score:+.3f})")
    print(f"  Max Other Potato:    {max_other_potato:.3f}")
    print(f"  Max Distractor:      {max_distractor:.3f}")
    print(f">> Gate 14 Result: {'PASSED' if g14_ok else 'FAILED'}")
    all_passed &= g14_ok

    # Gate 15: WhatsApp 4-Turn Messaging Flow & Learned Span Argument Extraction
    print("\n[GATE 15] WhatsApp 4-Turn Messaging Flow (Contact search, Argument span grounding, Send button, Completion):")
    wa_chats_eval = [
        {"id": "wa_search_btn", "label": "Search contacts and chats", "res": "menuitem_search", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.75, 0.03, 0.88, 0.08, 0.815, 0.055, 0.13, 0.05, 0.0065, 2.6], "is_matching_entity": True},
        {"id": "wa_chat_priya", "label": "Priya: Hey, are you free today? 11:20 AM", "res": "conversations_row", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.15, 0.95, 0.25, 0.5, 0.20, 0.9, 0.10, 0.09, 9.0], "is_matching_entity": False},
        {"id": "wa_chat_group", "label": "Family Group: Good morning everyone! 10:45 AM", "res": "conversations_row", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.26, 0.95, 0.36, 0.5, 0.31, 0.9, 0.10, 0.09, 9.0], "is_matching_entity": False},
        {"id": "wa_tab_updates", "label": "Updates", "res": "tab_updates", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.25, 0.92, 0.50, 0.98, 0.375, 0.95, 0.25, 0.06, 0.015, 4.16], "is_matching_entity": False},
        make_system_cand("wa_complete", "complete", "button"),
        make_system_cand("wa_back", "back"),
    ]
    res_wa_a = eval_screen("Send message to Rahul: I am on my way", wa_chats_eval)
    top_lbl_wa_a = res_wa_a["top_candidate"]["label"]
    top_act_wa_a = res_wa_a["action_type"]
    wa_a_ok = ("search" in top_lbl_wa_a.lower() and top_act_wa_a in ["TYPE_TEXT", "CLICK"])
    print(f"  [Screen A - Contact Search] Top: {top_lbl_wa_a} (Action: {top_act_wa_a}) -> {'PASS' if wa_a_ok else 'FAIL'}")

    wa_search_eval = [
        {"id": "wa_contact_rahul", "label": "Rahul +91 98765 43210 Available", "res": "contact_row", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.12, 0.95, 0.22, 0.5, 0.17, 0.9, 0.10, 0.09, 9.0], "is_matching_entity": True},
        {"id": "wa_contact_rahul_sharma", "label": "Rahul Sharma Hey there! I am using WhatsApp", "res": "contact_row", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.23, 0.95, 0.33, 0.5, 0.28, 0.9, 0.10, 0.09, 9.0], "is_matching_entity": False},
        {"id": "wa_contact_rakesh", "label": "Rakesh Work Busy", "res": "contact_row", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.34, 0.95, 0.44, 0.5, 0.39, 0.9, 0.10, 0.09, 9.0], "is_matching_entity": False},
        make_system_cand("wa_back", "back"),
    ]
    res_wa_b = eval_screen("Send message to Rahul: I am on my way", wa_search_eval)
    top_lbl_wa_b = res_wa_b["top_candidate"]["label"]
    top_act_wa_b = res_wa_b["action_type"]
    wa_b_ok = ("rahul" in top_lbl_wa_b.lower() and "sharma" not in top_lbl_wa_b.lower() and top_act_wa_b == "CLICK")
    print(f"  [Screen B - Contact Select] Top: {top_lbl_wa_b[:35]}... (Action: {top_act_wa_b}) -> {'PASS' if wa_b_ok else 'FAIL'}")

    wa_empty_eval = [
        {"id": "wa_entry_input", "label": "Message", "res": "entry", "role": "input", "role_idx": 2, "is_clickable": True, "is_editable": True, "bounds_vec": [0.12, 0.92, 0.82, 0.98, 0.47, 0.95, 0.70, 0.06, 0.042, 11.66], "is_matching_entity": True},
        {"id": "wa_btn_emoji", "label": "Emoji", "res": "emoji_picker_btn", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.02, 0.92, 0.10, 0.98, 0.06, 0.95, 0.08, 0.06, 0.0048, 1.33], "is_matching_entity": False},
        {"id": "wa_btn_attach", "label": "Attach", "res": "attach_btn", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.84, 0.92, 0.91, 0.98, 0.875, 0.95, 0.07, 0.06, 0.0042, 1.16], "is_matching_entity": False},
        {"id": "wa_btn_mic", "label": "Voice message", "res": "voice_note_btn", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.92, 0.92, 0.98, 0.98, 0.95, 0.95, 0.06, 0.06, 0.0036, 1.0], "is_matching_entity": False},
        {"id": "wa_header_name", "label": "Rahul online", "res": "conversation_contact_name", "role": "text", "role_idx": 0, "is_clickable": False, "bounds_vec": [0.15, 0.03, 0.70, 0.08, 0.425, 0.055, 0.55, 0.05, 0.0275, 11.0], "is_matching_entity": False},
        make_system_cand("wa_back", "back"),
    ]
    res_wa_c = eval_screen("Send message to Rahul: I am on my way", wa_empty_eval)
    top_lbl_wa_c = res_wa_c["top_candidate"]["label"]
    top_act_wa_c = res_wa_c["action_type"]
    wa_c_ok = (res_wa_c["top_candidate"]["id"] == "wa_entry_input" and top_act_wa_c == "TYPE_TEXT")
    print(f"  [Screen C - Message Input]  Top: {top_lbl_wa_c} (Action: {top_act_wa_c}) -> {'PASS' if wa_c_ok else 'FAIL'}")

    wa_pop_eval = [
        {"id": "wa_btn_send", "label": "Send", "res": "send", "role": "button", "role_idx": 1, "is_clickable": True, "is_editable": False, "bounds_vec": [0.88, 0.92, 0.98, 0.98, 0.93, 0.95, 0.10, 0.06, 0.006, 1.66], "is_matching_entity": True},
        {"id": "wa_entry_typed", "label": "I am on my way", "res": "entry", "role": "input", "role_idx": 2, "is_clickable": True, "is_editable": True, "bounds_vec": [0.12, 0.92, 0.86, 0.98, 0.49, 0.95, 0.74, 0.06, 0.0444, 12.33], "is_matching_entity": False},
        {"id": "wa_btn_emoji", "label": "Emoji", "res": "emoji_picker_btn", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.02, 0.92, 0.10, 0.98, 0.06, 0.95, 0.08, 0.06, 0.0048, 1.33], "is_matching_entity": False},
        {"id": "wa_btn_attach", "label": "Attach", "res": "attach_btn", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.78, 0.92, 0.86, 0.98, 0.82, 0.95, 0.08, 0.06, 0.0048, 1.33], "is_matching_entity": False},
        make_system_cand("c_submit", "submit focused field", "button"),
        make_system_cand("wa_back", "back"),
    ]
    res_wa_d = eval_screen("Send message to Rahul: I am on my way", wa_pop_eval)
    top_lbl_wa_d = res_wa_d["top_candidate"]["label"]
    top_act_wa_d = res_wa_d["action_type"]
    wa_d_ok = (res_wa_d["top_candidate"]["id"] == "wa_btn_send" and top_act_wa_d == "CLICK")
    print(f"  [Screen D - Send Button]    Top: {top_lbl_wa_d} (Action: {top_act_wa_d}) -> {'PASS' if wa_d_ok else 'FAIL'}")

    wa_sent_eval = [
        {"id": "wa_bubble_sent", "label": "I am on my way 11:42 AM Delivered", "res": "message_text", "role": "text", "role_idx": 0, "is_clickable": False, "bounds_vec": [0.45, 0.82, 0.95, 0.89, 0.70, 0.855, 0.50, 0.07, 0.035, 7.14], "is_matching_entity": False},
        {"id": "wa_entry_empty", "label": "Message", "res": "entry", "role": "input", "role_idx": 2, "is_clickable": True, "is_editable": True, "bounds_vec": [0.12, 0.92, 0.82, 0.98, 0.47, 0.95, 0.70, 0.06, 0.042, 11.66], "is_matching_entity": False},
        {"id": "wa_btn_mic", "label": "Voice message", "res": "voice_note_btn", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.92, 0.92, 0.98, 0.98, 0.95, 0.95, 0.06, 0.06, 0.0036, 1.0], "is_matching_entity": False},
        make_system_cand("c_complete_goal", "complete", "button"),
        make_system_cand("c_submit", "submit focused field", "button"),
        make_system_cand("wa_back", "back"),
    ]
    res_wa_e = eval_screen("Send message to Rahul: I am on my way", wa_sent_eval)
    top_lbl_wa_e = res_wa_e["top_candidate"]["label"]
    wa_e_ok = (res_wa_e["top_candidate"]["id"] == "c_complete_goal")
    print(f"  [Screen E - Completion]     Top: {top_lbl_wa_e} (Terminal Prob: {res_wa_e['termination_prob']:.4f}) -> {'PASS' if wa_e_ok else 'FAIL'}")

    g15_ok = (wa_a_ok and wa_b_ok and wa_c_ok and wa_d_ok and wa_e_ok)
    print(f">> Gate 15 Result: {'PASSED' if g15_ok else 'FAILED'}")
    all_passed &= g15_ok

    print("\n" + "=" * 80)
    print(f"OVERALL V54 CERTIFICATION AUDIT: {'ALL 15 GATES PASSED (100%)' if all_passed else 'SOME GATES FAILED'}")
    print("=" * 80)
    return all_passed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Karvish V54 Training & Certification")
    parser.add_argument("--steps", type=int, default=300, help="Total training steps")
    parser.add_argument("--dry-run", action="store_true", help="Run 10-step smoke test without saving to production checkpoint")
    args = parser.parse_args()

    if args.dry_run:
        print("\n>> RUNNING V54 SMOKE TEST / DRY RUN (220 STEPS ACROSS ALL UNIVERSAL CATEGORIES)...")
        test_model = train_v54_model(total_steps=220, is_dry_run=True)
        print("\n>> SMOKE TEST OPTIMIZATION COMPLETE. RUNNING POST-SMOKE 15-GATE AUDIT (DRY RUN ONLY)...")
        gates_ok = evaluate_v54_production_gates(test_model)

        print("\n>> RUNNING UNIVERSAL COMMUNICATION DIAGNOSTIC SUITE (DRY RUN):")
        test_model.eval()

        # Probe 1: Live Phone WhatsApp Screen B Candidates (Exact Candidates from Samsung Galaxy A26 log)
        phone_screen_b_cands = [
            {"id": "c_karthik_text", "label": "Karthik.S (You)", "res": "contact_name", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.18, 0.95, 0.26, 0.5, 0.22, 0.9, 0.08, 0.072, 11.25]},
            {"id": "c_ask_btn", "label": "ASK", "res": "ask_button", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.85, 0.05, 0.95, 0.11, 0.90, 0.08, 0.10, 0.06, 0.006, 1.66]},
            make_system_cand("c_submit", "submit focused field"),
            {"id": "c_dist_karthikeyen", "label": "Karthikeyen", "res": "contact_name", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.30, 0.95, 0.38, 0.5, 0.34, 0.9, 0.08, 0.072, 11.25]},
            {"id": "c_dist_karthik_mobile", "label": "Karthikeyen 0.1 Mobile", "res": "contact_row", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.40, 0.95, 0.50, 0.5, 0.45, 0.9, 0.10, 0.09, 9.0]},
            {"id": "c_dist_santosh_status", "label": "Santosh Mava's status. 2 new updates", "res": "status_row", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.52, 0.95, 0.62, 0.5, 0.57, 0.9, 0.10, 0.09, 9.0]},
            {"id": "c_dist_placement_group", "label": "Placements 2027 CSD Karthikeyen is also in this group", "res": "group_row", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.64, 0.95, 0.74, 0.5, 0.69, 0.9, 0.10, 0.09, 9.0]},
            make_system_cand("c_back_probe", "back"),
        ]
        res_probe_b = eval_screen_universal(test_model, "open WhatsApp and send message to Karthik :check your mail", phone_screen_b_cands)
        top_b_lbl = res_probe_b["top_candidate"]["label"]
        top_b_act = res_probe_b["action_type"]
        probe_b_ok = ("karthik" in top_b_lbl.lower() and "karthikeyen" not in top_b_lbl.lower() and top_b_act == "CLICK")
        print(f"  [Probe 1 - Contact Search Disambiguation (Phone Exact)]")
        print(f"    Top Candidate: {top_b_lbl} (Action: {top_b_act})")
        print(f"    Karthik.S (You) Score: {res_probe_b['ranking'][0]:.3f}")
        print(f"    ASK Score:             {res_probe_b['ranking'][1]:.3f} (Margin: {res_probe_b['ranking'][0] - res_probe_b['ranking'][1]:+.3f})")
        print(f"    Submit Score:          {res_probe_b['ranking'][2]:.3f} (Margin: {res_probe_b['ranking'][0] - res_probe_b['ranking'][2]:+.3f})")
        print(f"    Karthikeyen Score:     {res_probe_b['ranking'][3]:.3f} (Margin: {res_probe_b['ranking'][0] - res_probe_b['ranking'][3]:+.3f})")
        print(f"    Status Update Score:   {res_probe_b['ranking'][5]:.3f}")
        print(f"    Group Mention Score:   {res_probe_b['ranking'][6]:.3f}")
        print(f"    Result: {'PASSED' if probe_b_ok else 'FAILED'}")

        # Probe 2: Universal Dead-End Profile Call Modal Recovery (Prevents Unwanted Video Calls)
        call_modal_cands = [
            {"id": "modal_btn_voice", "label": "Voice call", "res": "voice_call_btn", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.15, 0.60, 0.45, 0.68, 0.30, 0.64, 0.30, 0.08, 0.024, 3.75]},
            {"id": "modal_btn_video", "label": "Video call", "res": "video_call_btn", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.55, 0.60, 0.85, 0.68, 0.70, 0.64, 0.30, 0.08, 0.024, 3.75]},
            {"id": "modal_btn_info", "label": "Info", "res": "info_btn", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.35, 0.72, 0.65, 0.80, 0.50, 0.76, 0.30, 0.08, 0.024, 3.75]},
            {"id": "modal_img_pic", "label": "Profile picture", "res": "avatar", "role": "image", "role_idx": 0, "is_clickable": False, "bounds_vec": [0.25, 0.20, 0.75, 0.55, 0.50, 0.375, 0.50, 0.35, 0.175, 1.42]},
            make_system_cand("c_back_modal", "back"),
        ]
        res_probe_modal = eval_screen_universal(test_model, "open WhatsApp and send hi to Bhavani", call_modal_cands)
        top_modal_lbl = res_probe_modal["top_candidate"]["label"]
        top_modal_act = res_probe_modal["action_type"]
        probe_modal_ok = (top_modal_lbl.lower() == "back" and res_probe_modal["ranking"][4] > res_probe_modal["ranking"][1])
        print(f"\n  [Probe 2 - Contact Call Modal Dead-End Recovery]")
        print(f"    Top Candidate:    {top_modal_lbl} (Action: {top_modal_act})")
        print(f"    Back Score:       {res_probe_modal['ranking'][4]:.3f}")
        print(f"    Video Call Score: {res_probe_modal['ranking'][1]:.3f} (Margin: {res_probe_modal['ranking'][4] - res_probe_modal['ranking'][1]:+.3f})")
        print(f"    Voice Call Score: {res_probe_modal['ranking'][0]:.3f} (Margin: {res_probe_modal['ranking'][4] - res_probe_modal['ranking'][0]:+.3f})")
        print(f"    Result: {'PASSED (Video Call rejected, Back selected)' if probe_modal_ok else 'FAILED'}")

        # Probe 3: Universal Unseen Apps Generalization (Telegram & SMS Contact Pickers)
        tg_probe_cands = [
            {"id": "tg_contact_david", "label": "David Miller last seen recently", "res": "user_cell", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.15, 0.95, 0.25, 0.5, 0.20, 0.9, 0.10, 0.09, 9.0]},
            {"id": "tg_bot_david", "label": "David AI Assistant Bot @david_ai_bot", "res": "bot_cell", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.26, 0.95, 0.36, 0.5, 0.31, 0.9, 0.10, 0.09, 9.0]},
            {"id": "tg_global_search", "label": "Search globally for 'David'", "res": "global_search_btn", "role": "button", "role_idx": 1, "is_clickable": True, "bounds_vec": [0.05, 0.38, 0.95, 0.46, 0.5, 0.42, 0.9, 0.08, 0.072, 11.25]},
            make_system_cand("c_submit", "submit focused field"),
            make_system_cand("tg_back", "back"),
        ]
        res_probe_tg = eval_screen_universal(test_model, "Send message to David: ready for call", tg_probe_cands)
        top_tg_lbl = res_probe_tg["top_candidate"]["label"]
        probe_tg_ok = ("david miller" in top_tg_lbl.lower() and "bot" not in top_tg_lbl.lower())
        print(f"\n  [Probe 3 - Universal Unseen App: Telegram Global Search]")
        print(f"    Top Candidate:    {top_tg_lbl}")
        print(f"    David Miller Score: {res_probe_tg['ranking'][0]:.3f}")
        print(f"    AI Bot Score:       {res_probe_tg['ranking'][1]:.3f} (Margin: {res_probe_tg['ranking'][0] - res_probe_tg['ranking'][1]:+.3f})")
        print(f"    Result: {'PASSED' if probe_tg_ok else 'FAILED'}")

        # Probe 4: Learned Span Argument Extraction
        test_goal = "Send message to Karthik: check your email"
        enc = TOKENIZER(test_goal, max_length=32, truncation=True, return_tensors="pt")
        g_ids = enc["input_ids"].to(DEVICE)
        g_mask = enc["attention_mask"].float().to(DEVICE)
        toks = TOKENIZER.convert_ids_to_tokens(g_ids[0])

        c_a = TOKENIZER(["Search contacts and chats", "Priya", "Calls"], max_length=16, padding="max_length", truncation=True, return_tensors="pt")
        with torch.no_grad():
            out_a = test_model(
                goal_ids=g_ids, goal_mask=g_mask,
                candidate_label_ids=c_a["input_ids"].unsqueeze(0).to(DEVICE),
                candidate_label_mask=c_a["attention_mask"].float().unsqueeze(0).to(DEVICE),
                candidate_bounds=torch.zeros(1, 3, 10, device=DEVICE),
                candidate_roles=torch.tensor([[2, 1, 1]], dtype=torch.long, device=DEVICE),
                candidate_flags=torch.zeros(1, 3, 16, device=DEVICE),
            )
        s_a = int(out_a["span_start_logits"][0].argmax().item())
        e_a = int(out_a["span_end_logits"][0].argmax().item())
        ext_a = TOKENIZER.convert_tokens_to_string(toks[s_a : min(e_a + 1, len(toks))]) if s_a <= e_a else ""

        c_c = TOKENIZER(["Message", "Emoji", "Voice message"], max_length=16, padding="max_length", truncation=True, return_tensors="pt")
        c_flags_c = torch.zeros(1, 3, 16, device=DEVICE)
        c_flags_c[0, 0, 0] = 1.0  # is_editable
        with torch.no_grad():
            out_c = test_model(
                goal_ids=g_ids, goal_mask=g_mask,
                candidate_label_ids=c_c["input_ids"].unsqueeze(0).to(DEVICE),
                candidate_label_mask=c_c["attention_mask"].float().unsqueeze(0).to(DEVICE),
                candidate_bounds=torch.zeros(1, 3, 10, device=DEVICE),
                candidate_roles=torch.tensor([[2, 1, 1]], dtype=torch.long, device=DEVICE),
                candidate_flags=c_flags_c,
            )
        s_c = int(out_c["span_start_logits"][0].argmax().item())
        e_c = int(out_c["span_end_logits"][0].argmax().item())
        ext_c = TOKENIZER.convert_tokens_to_string(toks[s_c : min(e_c + 1, len(toks))]) if s_c <= e_c else ""

        print(f"\n  [Probe 4 - Dual-Span Target-Conditioned Argument Extraction]")
        print(f"    Goal: '{test_goal}'")
        print(f"    Turn 1 (Search Bar): Extracted Target = '{ext_a}'")
        print(f"    Turn 3 (Chat Room Input): Extracted Target = '{ext_c}'")
        probe_span_ok = (ext_a.strip().lower() == "karthik" and "check your email" in ext_c.strip().lower())

        all_smoke_ok = gates_ok and probe_b_ok and probe_modal_ok and probe_tg_ok and probe_span_ok
        print("\n" + "=" * 80)
        print(f">> SMOKE DRY RUN VERDICT: {'ALL TESTS & GATES PASSED (100%)' if all_smoke_ok else 'SOME CHECKS FAILED'}")
        print(">> Note: Production checkpoint v54_best.pt remains 100% UNTOUCHED.")
        print("=" * 80)
        sys.exit(0 if all_smoke_ok else 1)

    trained_model = train_v54_model(total_steps=args.steps, is_dry_run=False)
    success = evaluate_v54_production_gates(trained_model)
    if not success:
        sys.exit(1)
    print("\n>> V54 Certified Successfully!")
