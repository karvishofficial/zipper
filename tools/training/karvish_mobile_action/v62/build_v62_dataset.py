"""
KARVISH UNIVERSAL V62 — AUTHORITATIVE 125,000-SAMPLE BALANCED DATASET GENERATOR
Builds 125,000 authoritative multi-turn trajectories with:
  1. Unified (K + 1) Differentiable Action Space:
     - 25,000+ True Terminal Milestones (20.0%+ of dataset).
     - In terminal states, target_idx = N (the index of c_complete_goal).
     - Screens have realistic, variable candidate counts (4 to 16 UI elements).
  2. 20,000+ Hard Near-Terminal Negatives:
     - Crucial defense against false stopping:
       a) Search results loaded when goal requires playing/opening the item -> Action: CLICK video card (NOT stop!).
       b) Query typed in search box, search icon/enter visible -> Action: SUBMIT / CLICK search icon (NOT stop!).
       c) Dialog / Confirmation modal -> Action: CLICK Allow / Continue (NOT stop!).
  3. 20,000+ Anti-Wandering Distractor Screens:
     - Prominent feed cards (YouTube Shorts, Reels, sponsored banners) tagged with `is_distractor = 1.0`.
  4. Everyday Voice, Slang & Hinglish Dual-Stream NLU:
     - Conversational phrasing trajectories with 15-tag BIO token sequences.
  5. 10-D Continuous Observable Transition Deltas:
     - [textChanged, nodesAdded, nodesRemoved, scrollDelta, isErrorScreen, latencyMs, visualDelta, progressDelta, stallSignal, outcomeSuccess]
     - Differentiates true accomplishment (outcomeSuccess = 1.0, nodesAdded > 0) from intermediate states.
"""

from __future__ import annotations

import copy
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
V61_DIR = SCRIPT_DIR.parent / "v61"
V60_DIR = SCRIPT_DIR.parent / "v60"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(V61_DIR))
sys.path.insert(0, str(V60_DIR))

from schema_v62 import (
    APP_DOMAINS,
    BIO_SLOT_TAGS,
    BIO_TAG_TO_IDX,
    CANONICAL_ACTIONS,
    ACTION_TO_IDX,
    ROLE_TO_IDX,
    VISUAL_CLASS_TO_IDX,
    NUM_VISUAL_CLASSES,
    COMPLETE_CANDIDATE_ID,
    COMPLETE_SEMANTIC_LABEL,
)
from vocab_v62 import BASE_TOKENIZER_ID

TOKENIZER = AutoTokenizer.from_pretrained(BASE_TOKENIZER_ID)
V61_DATASET_FILE = V61_DIR / "checkpoints" / "v61_authoritative_dataset.pt"
OUTPUT_V62_DATASET = SCRIPT_DIR / "checkpoints" / "v62_authoritative_dataset.pt"

TARGET_DATASET_SIZE = 125000
TARGET_TERMINAL_COUNT = 25000  # 20.0% of 125,000

# 512-D prototype embeddings for MobileNetV4 icon classes
torch.manual_seed(42)
ICON_PROTOTYPES = {}
for cls_idx in range(NUM_VISUAL_CLASSES):
    if cls_idx == 0:
        ICON_PROTOTYPES[cls_idx] = torch.zeros(512)
    else:
        v = torch.randn(512)
        ICON_PROTOTYPES[cls_idx] = v / torch.norm(v)


def get_visual_embedding(visual_class_idx: int) -> torch.Tensor:
    """Returns realistic 512-D MobileNetV4 embedding with slight natural variance."""
    if visual_class_idx == 0:
        return torch.zeros(512, dtype=torch.float32)
    proto = ICON_PROTOTYPES.get(visual_class_idx, torch.zeros(512))
    noise = torch.randn(512) * 0.05
    vec = proto + noise
    return (vec / torch.norm(vec)).float()


def normalize_bounds(raw_bounds: List[float]) -> List[float]:
    """Converts 4-element or raw pixel bounds to 10-D normalized continuous geometry."""
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
    return [l, t, r, b, cx, cy, w, h, w * h, w / h]


VOICE_TEMPLATES = [
    # Quick commerce & Grocery
    ("{app} me {item} search karo", ["zepto", "blinkit", "instamart", "bigbasket"], ["amul doodh", "aashirvaad atta", "paneer 200g", "curd pouch", "maggi noodles", "chai patti", "refined oil"], "search"),
    ("order {item} jaldi on {app}", ["zepto", "blinkit"], ["bread and eggs", "amul milk", "haldiram bhujia", "dahi", "fresh paneer"], "order"),
    ("get {item} from {app}", ["blinkit", "zepto", "instamart"], ["lays chips", "cadbury dairy milk", "ketchup bottle", "atta 5kg", "sugar 1kg"], "order"),
    
    # Food Delivery
    ("swiggy pe {item} dekho", ["swiggy", "zomato"], ["chicken biryani", "masala dosa", "paneer butter masala", "veg thali", "gulab jamun", "burger and fries", "chicken roll"], "search"),
    ("find {item} on {app}", ["zomato", "swiggy", "eatsure"], ["dominos pizza", "mcdonalds burger", "chole bhature", "pav bhaji", "shawarma"], "search"),
    
    # Video & Media
    ("yt pe {item} chalao", ["youtube"], ["spacex starship launch", "lofi hip hop radio", "cricket highlights", "ind vs aus t20", "arjit singh songs", "python tutorial"], "play"),
    ("search {item} on youtube", ["youtube"], ["james webb telescope", "ai coding agents", "techno gamerz gta 5", "funny cat videos", "news live", "spacex"], "search"),
    ("play {item} on youtube", ["youtube"], ["spacex starship launch", "lofi music", "latest tech news", "quantum physics lecture"], "play"),
    
    # System Settings & Utilities
    ("turn on {item} in settings", ["settings"], ["bluetooth", "wi-fi", "hotspot", "dark mode", "airplane mode", "flashlight"], "toggle"),
    ("settings me {item} check karo", ["settings"], ["battery health", "storage space", "screen time", "software update", "display brightness"], "check"),
    
    # Messaging
    ("msg {item} reached home", ["whatsapp"], ["karthik", "rahul", "mom", "dad", "bhavani", "priya", "amit"], "message"),
    ("ping {item} call me back", ["whatsapp"], ["rohit", "ananya", "suresh", "boss", "deepak"], "message"),
    
    # Calculator
    ("calculate {item}", ["calculator"], ["45 plus 55", "1200 into 18 percent", "5000 divided by 4", "99 minus 45"], "calc"),
]

DISTRACTOR_LABELS = [
    ("Trending Shorts: Amazing Magic Trick", 0),
    ("Shorts: Crazy Physics Experiment", 0),
    ("Sponsored: 60% OFF Mega Sale", 0),
    ("Ad: Download Dream11 App Now", 0),
    ("Recommended Reel: Viral Dance Video", 0),
    ("You Might Like: Top 10 Richest People", 0),
    ("Trending in India: Cricket News", 0),
    ("Flash Sale: Flat Rs. 100 Cashback", 0),
    ("Promoted: Best Credit Cards 2026", 0),
    ("Shorts: Funny Dog Reaction", 0),
]


def generate_v62_synthetic_sample(
    sample_id: int,
    regime: str = "random",  # "terminal", "near_terminal_neg", "initial"
    force_distractor: bool = False,
) -> Dict[str, Any]:
    """
    Generates a realistic multimodal trajectory sample with:
    - 4 to 16 realistic screen UI elements.
    - Full chrome (headers, search bars, filters, result cards, bottom nav).
    - Clear distinction between initial navigation, near-terminal negatives, and true completion.
    """
    template, apps, items, task_intent = random.choice(VOICE_TEMPLATES)
    app = random.choice(apps)
    item = random.choice(items)
    goal = template.format(app=app, item=item)
    query = item

    if regime == "random":
        r_val = random.random()
        if r_val < 0.25:
            regime = "terminal"
        elif r_val < 0.50:
            regime = "near_terminal_neg"
        else:
            regime = "initial"

    has_distractors = force_distractor or (random.random() < 0.45)
    candidates: List[Dict[str, Any]] = []
    
    # Common app chrome
    app_title = app.title()
    nav_tabs = ["Home", "Explore", "Subscriptions" if app == "youtube" else "Cart", "Profile"]
    
    obs_delta = [0.0] * 10

    if regime == "terminal":
        # -------------------------------------------------------------
        # TRUE TERMINAL ACCOMPLISHED MILESTONE
        # -------------------------------------------------------------
        is_terminal = True
        obs_delta[0] = 0.6  # textChanged
        obs_delta[1] = 0.8  # nodesAdded
        obs_delta[6] = 0.5  # visualDelta
        obs_delta[7] = 1.0  # progressDelta
        obs_delta[9] = 1.0  # outcomeSuccess

        if task_intent == "play":
            # Video player actively playing the target video
            candidates.append({"label": app_title, "bounds": normalize_bounds([0.05, 0.04, 0.30, 0.08]), "role": 0, "visual_class": 0, "visual_feat": get_visual_embedding(0), "is_distractor": 0.0})
            candidates.append({"label": f"{item.title()} - Official Stream", "bounds": normalize_bounds([0.05, 0.10, 0.95, 0.40]), "role": 7, "visual_class": 0, "visual_feat": get_visual_embedding(0), "is_distractor": 0.0})
            candidates.append({"label": "Pause", "bounds": normalize_bounds([0.45, 0.22, 0.55, 0.28]), "role": 1, "visual_class": 0, "visual_feat": get_visual_embedding(0), "is_distractor": 0.0})
            candidates.append({"label": "1080p HD", "bounds": normalize_bounds([0.80, 0.12, 0.95, 0.16]), "role": 1, "visual_class": 0, "visual_feat": get_visual_embedding(0), "is_distractor": 0.0})
            candidates.append({"label": "Like (124k)", "bounds": normalize_bounds([0.05, 0.42, 0.25, 0.48]), "role": 1, "visual_class": 0, "visual_feat": get_visual_embedding(0), "is_distractor": 0.0})
            candidates.append({"label": "Share", "bounds": normalize_bounds([0.30, 0.42, 0.50, 0.48]), "role": 1, "visual_class": 0, "visual_feat": get_visual_embedding(0), "is_distractor": 0.0})
            candidates.append({"label": "Subscribe", "bounds": normalize_bounds([0.70, 0.42, 0.95, 0.48]), "role": 1, "visual_class": 0, "visual_feat": get_visual_embedding(0), "is_distractor": 0.0})
            candidates.append({"label": "Comments (3.2k)", "bounds": normalize_bounds([0.05, 0.50, 0.95, 0.58]), "role": 0, "visual_class": 0, "visual_feat": get_visual_embedding(0), "is_distractor": 0.0})
        elif task_intent == "search":
            # Search Results page for "Search {item} on {app}" (Goal is satisfied by displaying results!)
            candidates.append({"label": app_title, "bounds": normalize_bounds([0.05, 0.04, 0.25, 0.08]), "role": 0, "visual_class": 0, "visual_feat": get_visual_embedding(0), "is_distractor": 0.0})
            candidates.append({"label": f"{item}", "bounds": normalize_bounds([0.30, 0.04, 0.85, 0.08]), "role": 3, "visual_class": 1, "visual_feat": get_visual_embedding(1), "is_distractor": 0.0})
            candidates.append({"label": "Filter & Sort", "bounds": normalize_bounds([0.05, 0.10, 0.35, 0.15]), "role": 1, "visual_class": 14, "visual_feat": get_visual_embedding(14), "is_distractor": 0.0})
            candidates.append({"label": f"{item.title()} (Top Result)", "bounds": normalize_bounds([0.05, 0.18, 0.95, 0.35]), "role": 7, "visual_class": 0, "visual_feat": get_visual_embedding(0), "is_distractor": 0.0})
            candidates.append({"label": f"More options for {item}", "bounds": normalize_bounds([0.05, 0.38, 0.95, 0.55]), "role": 7, "visual_class": 0, "visual_feat": get_visual_embedding(0), "is_distractor": 0.0})
        elif task_intent == "toggle":
            # Settings page with toggle in ON state
            candidates.append({"label": "Settings", "bounds": normalize_bounds([0.05, 0.04, 0.30, 0.08]), "role": 0, "visual_class": 0, "visual_feat": get_visual_embedding(0), "is_distractor": 0.0})
            candidates.append({"label": f"{item.title()}", "bounds": normalize_bounds([0.05, 0.10, 0.60, 0.16]), "role": 0, "visual_class": 0, "visual_feat": get_visual_embedding(0), "is_distractor": 0.0})
            candidates.append({"label": f"{item.title()} [ON]", "bounds": normalize_bounds([0.75, 0.10, 0.95, 0.16]), "role": 1, "visual_class": 0, "visual_feat": get_visual_embedding(0), "is_distractor": 0.0})
            candidates.append({"label": f"Connected to {item} network", "bounds": normalize_bounds([0.05, 0.18, 0.95, 0.25]), "role": 0, "visual_class": 0, "visual_feat": get_visual_embedding(0), "is_distractor": 0.0})
        elif task_intent == "order":
            # Cart summary screen
            candidates.append({"label": f"{app_title} Cart", "bounds": normalize_bounds([0.05, 0.04, 0.40, 0.08]), "role": 0, "visual_class": 0, "visual_feat": get_visual_embedding(0), "is_distractor": 0.0})
            candidates.append({"label": f"1 item: {item.title()}", "bounds": normalize_bounds([0.05, 0.12, 0.95, 0.22]), "role": 7, "visual_class": 0, "visual_feat": get_visual_embedding(0), "is_distractor": 0.0})
            candidates.append({"label": "Bill Summary: Rs. 65", "bounds": normalize_bounds([0.05, 0.25, 0.95, 0.35]), "role": 0, "visual_class": 0, "visual_feat": get_visual_embedding(0), "is_distractor": 0.0})
            candidates.append({"label": "Proceed to Pay", "bounds": normalize_bounds([0.05, 0.88, 0.95, 0.95]), "role": 1, "visual_class": 0, "visual_feat": get_visual_embedding(0), "is_distractor": 0.0})
        elif task_intent == "message":
            # Chat bubble sent
            candidates.append({"label": f"Chat with {item.title()}", "bounds": normalize_bounds([0.05, 0.04, 0.50, 0.08]), "role": 0, "visual_class": 0, "visual_feat": get_visual_embedding(0), "is_distractor": 0.0})
            candidates.append({"label": "reached home", "bounds": normalize_bounds([0.40, 0.40, 0.90, 0.48]), "role": 0, "visual_class": 0, "visual_feat": get_visual_embedding(0), "is_distractor": 0.0})
            candidates.append({"label": "Delivered ✓✓", "bounds": normalize_bounds([0.75, 0.49, 0.90, 0.53]), "role": 0, "visual_class": 0, "visual_feat": get_visual_embedding(0), "is_distractor": 0.0})
        else:
            # Calculator result display
            candidates.append({"label": "Calculator", "bounds": normalize_bounds([0.05, 0.04, 0.40, 0.08]), "role": 0, "visual_class": 0, "visual_feat": get_visual_embedding(0), "is_distractor": 0.0})
            candidates.append({"label": f"{item} = 100", "bounds": normalize_bounds([0.05, 0.12, 0.95, 0.25]), "role": 0, "visual_class": 0, "visual_feat": get_visual_embedding(0), "is_distractor": 0.0})

        # Add standard navigation tabs to bottom
        for t_idx, tab_name in enumerate(nav_tabs):
            x1 = 0.05 + t_idx * 0.23
            candidates.append({"label": tab_name, "bounds": normalize_bounds([x1, 0.93, x1 + 0.20, 0.98]), "role": 1, "visual_class": 0, "visual_feat": get_visual_embedding(0), "is_distractor": 0.0})

        if has_distractors:
            d_lbl, d_cls = random.choice(DISTRACTOR_LABELS)
            candidates.append({"label": d_lbl, "bounds": normalize_bounds([0.05, 0.65, 0.95, 0.85]), "role": 7, "visual_class": d_cls, "visual_feat": get_visual_embedding(d_cls), "is_distractor": 1.0})

        # Terminal target is index N (c_complete_goal)
        target_idx = len(candidates)
        action_idx = 7  # COMPLETE

    elif regime == "near_terminal_neg":
        # -------------------------------------------------------------
        # HARD NEAR-TERMINAL NEGATIVE (DO NOT STOP! CLICK NEXT AFFORDANCE)
        # -------------------------------------------------------------
        is_terminal = False
        obs_delta[0] = 0.4
        obs_delta[1] = 0.2
        obs_delta[9] = 0.0  # NOT success yet!

        neg_type = random.choice(["results_click_card", "input_click_submit", "dialog_allow"])

        if neg_type == "results_click_card":
            # Search results are visible, but the goal is "play {item}" or "order {item}"
            # The model MUST click the video card or product, NOT stop!
            target_idx = 3  # The target card
            candidates.append({"label": app_title, "bounds": normalize_bounds([0.05, 0.04, 0.25, 0.08]), "role": 0, "visual_class": 0, "visual_feat": get_visual_embedding(0), "is_distractor": 0.0})
            candidates.append({"label": f"{item}", "bounds": normalize_bounds([0.30, 0.04, 0.85, 0.08]), "role": 3, "visual_class": 1, "visual_feat": get_visual_embedding(1), "is_distractor": 0.0})
            candidates.append({"label": "Filter", "bounds": normalize_bounds([0.05, 0.10, 0.25, 0.15]), "role": 1, "visual_class": 14, "visual_feat": get_visual_embedding(14), "is_distractor": 0.0})
            candidates.append({"label": f"{item.title()} (Watch Video)", "bounds": normalize_bounds([0.05, 0.18, 0.95, 0.35]), "role": 7, "visual_class": 0, "visual_feat": get_visual_embedding(0), "is_distractor": 0.0})
            candidates.append({"label": "Related Videos", "bounds": normalize_bounds([0.05, 0.38, 0.95, 0.55]), "role": 7, "visual_class": 0, "visual_feat": get_visual_embedding(0), "is_distractor": 0.0})
            action_idx = 0  # CLICK target card

        elif neg_type == "input_click_submit":
            # Query is typed in search bar, Search button / keyboard enter is visible
            # The model MUST click Search / Submit, NOT stop!
            target_idx = 2  # The search submit button
            candidates.append({"label": app_title, "bounds": normalize_bounds([0.05, 0.04, 0.25, 0.08]), "role": 0, "visual_class": 0, "visual_feat": get_visual_embedding(0), "is_distractor": 0.0})
            candidates.append({"label": f"{item}", "bounds": normalize_bounds([0.30, 0.04, 0.80, 0.08]), "role": 3, "visual_class": 0, "visual_feat": get_visual_embedding(0), "is_distractor": 0.0})
            candidates.append({"label": "Search 🔍", "bounds": normalize_bounds([0.82, 0.04, 0.95, 0.08]), "role": 1, "visual_class": 1, "visual_feat": get_visual_embedding(1), "is_distractor": 0.0})
            candidates.append({"label": f"Recent search: {item}", "bounds": normalize_bounds([0.05, 0.12, 0.95, 0.18]), "role": 0, "visual_class": 0, "visual_feat": get_visual_embedding(0), "is_distractor": 0.0})
            candidates.append({"label": "Clear all", "bounds": normalize_bounds([0.70, 0.20, 0.95, 0.25]), "role": 1, "visual_class": 0, "visual_feat": get_visual_embedding(0), "is_distractor": 0.0})
            action_idx = 0  # CLICK search

        else:
            # Confirmation dialog
            target_idx = 2  # Continue / Allow
            candidates.append({"label": "Notice", "bounds": normalize_bounds([0.20, 0.35, 0.80, 0.42]), "role": 0, "visual_class": 0, "visual_feat": get_visual_embedding(0), "is_distractor": 0.0})
            candidates.append({"label": "Cancel", "bounds": normalize_bounds([0.25, 0.55, 0.45, 0.62]), "role": 1, "visual_class": 0, "visual_feat": get_visual_embedding(0), "is_distractor": 0.0})
            candidates.append({"label": "Allow / Continue", "bounds": normalize_bounds([0.55, 0.55, 0.75, 0.62]), "role": 1, "visual_class": 0, "visual_feat": get_visual_embedding(0), "is_distractor": 0.0})
            action_idx = 0  # CLICK Allow

        # Bottom nav
        for t_idx, tab_name in enumerate(nav_tabs):
            x1 = 0.05 + t_idx * 0.23
            candidates.append({"label": tab_name, "bounds": normalize_bounds([x1, 0.93, x1 + 0.20, 0.98]), "role": 1, "visual_class": 0, "visual_feat": get_visual_embedding(0), "is_distractor": 0.0})

        if has_distractors:
            d_lbl, d_cls = random.choice(DISTRACTOR_LABELS)
            candidates.append({"label": d_lbl, "bounds": normalize_bounds([0.05, 0.65, 0.95, 0.85]), "role": 7, "visual_class": d_cls, "visual_feat": get_visual_embedding(d_cls), "is_distractor": 1.0})

    else:
        # -------------------------------------------------------------
        # INITIAL ACTIVE NAVIGATION SCREEN
        # -------------------------------------------------------------
        is_terminal = False
        target_idx = 1  # The Search bar
        candidates.append({"label": app_title, "bounds": normalize_bounds([0.05, 0.04, 0.30, 0.08]), "role": 0, "visual_class": 0, "visual_feat": get_visual_embedding(0), "is_distractor": 0.0})
        candidates.append({"label": f"Search '{item}'", "bounds": normalize_bounds([0.05, 0.10, 0.95, 0.18]), "role": 3, "visual_class": 1, "visual_feat": get_visual_embedding(1), "is_distractor": 0.0})
        candidates.append({"label": "Categories", "bounds": normalize_bounds([0.05, 0.22, 0.45, 0.30]), "role": 1, "visual_class": 0, "visual_feat": get_visual_embedding(0), "is_distractor": 0.0})
        candidates.append({"label": "Offers & Discounts", "bounds": normalize_bounds([0.55, 0.22, 0.95, 0.30]), "role": 1, "visual_class": 0, "visual_feat": get_visual_embedding(0), "is_distractor": 0.0})

        for t_idx, tab_name in enumerate(nav_tabs):
            x1 = 0.05 + t_idx * 0.23
            candidates.append({"label": tab_name, "bounds": normalize_bounds([x1, 0.93, x1 + 0.20, 0.98]), "role": 1, "visual_class": 0, "visual_feat": get_visual_embedding(0), "is_distractor": 0.0})

        if has_distractors:
            d_lbl, d_cls = random.choice(DISTRACTOR_LABELS)
            candidates.append({"label": d_lbl, "bounds": normalize_bounds([0.05, 0.35, 0.95, 0.65]), "role": 7, "visual_class": d_cls, "visual_feat": get_visual_embedding(d_cls), "is_distractor": 1.0})

        action_idx = 0  # CLICK search bar

    # Random permutation of screen candidates so the model doesn't overfit to position
    num_screen_cands = len(candidates)
    perm = list(range(num_screen_cands))
    random.shuffle(perm)
    permuted_candidates = [candidates[i] for i in perm]

    if is_terminal:
        # In unified (K + 1) action space, target is the completion candidate at index N
        new_target_idx = num_screen_cands
    else:
        new_target_idx = perm.index(target_idx)

    # Compute BIO slot tags for conversational / voice extraction
    tok_res = TOKENIZER(goal, return_offsets_mapping=True, truncation=True, max_length=64)
    offsets = tok_res["offset_mapping"]
    bio_tags = [0] * len(offsets)  # 0 = 'O'
    target_span = (0, 0)

    if query:
        q_clean = query.lower()
        g_clean = goal.lower()
        start_char = g_clean.find(q_clean)
        if start_char != -1:
            end_char = start_char + len(q_clean)
            tok_start = 0
            tok_end = 0
            for t_i, (s, e) in enumerate(offsets):
                if s <= start_char < e or (s == start_char and e > s):
                    tok_start = t_i
                if s < end_char <= e or (e == end_char and e > s):
                    tok_end = t_i
            if tok_end >= tok_start and tok_end > 0:
                target_span = (tok_start, tok_end)
                for t_i in range(tok_start, min(tok_end + 1, len(bio_tags))):
                    bio_tags[t_i] = 4 if t_i > tok_start else 3  # B-ITEM / I-ITEM

    return {
        "id": f"v62_syn_{sample_id}",
        "goal": goal,
        "target_idx": new_target_idx,
        "action_idx": action_idx,
        "target_span": target_span,
        "slot_type": 1 if query else 0,
        "bio_slot_tags": bio_tags,
        "is_terminal_complete": 1.0 if is_terminal else 0.0,
        "is_near_miss_negative": 1.0 if regime == "near_terminal_neg" else 0.0,
        "observable_delta": obs_delta,
        "hindsight_next_screen_delta": 1.0 if is_terminal else 0.0,
        "candidates": permuted_candidates,
    }


def main():
    print("=" * 80)
    print(">> [Phase 2] Building V62 Authoritative Balanced Dataset (Realistic Chrome & Hard Negatives)")
    print(f">> Target Dataset Size: {TARGET_DATASET_SIZE:,}")
    print(f">> Target Terminal Milestones: {TARGET_TERMINAL_COUNT:,} ({TARGET_TERMINAL_COUNT/TARGET_DATASET_SIZE*100:.1f}%)")
    print("=" * 80)

    v62_samples: List[Dict[str, Any]] = []

    # 1. Load and sanitize existing V61 dataset
    if V61_DATASET_FILE.exists():
        print(f">> Loading base trajectories from {V61_DATASET_FILE}...")
        v61_raw = torch.load(V61_DATASET_FILE, weights_only=False)
        print(f"   Loaded {len(v61_raw):,} base samples from V61.")

        for idx, s in enumerate(v61_raw):
            cands = s.get("candidates", [])
            # IMPORTANT: In V61, is_final meant the last action of a trajectory (like clicking Add to Cart).
            # We ONLY convert to c_complete_goal if it was explicitly a COMPLETE action (role 7 or action_idx 4/7).
            act_idx = s.get("action_idx", 0)
            is_term = s.get("is_terminal_complete", 0.0) == 1.0 or act_idx == 7

            for c in cands:
                if "is_distractor" not in c:
                    lbl = c.get("label", "").lower()
                    if any(w in lbl for w in ["shorts", "reel", "sponsored", "promoted", "trending"]):
                        c["is_distractor"] = 1.0
                    else:
                        c["is_distractor"] = 0.0
                if "visual_feat" not in c or c["visual_feat"] is None:
                    c["visual_feat"] = get_visual_embedding(c.get("visual_class", 0))

            if is_term:
                s["target_idx"] = len(cands)
                s["is_terminal_complete"] = 1.0
                s["action_idx"] = 7
                s["observable_delta"] = [0.6, 0.8, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.0, 1.0]
            else:
                s["is_terminal_complete"] = 0.0
                s["observable_delta"] = [0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.0, 0.0]

            v62_samples.append(s)

    current_terminals = sum(1 for s in v62_samples if s.get("is_terminal_complete", 0.0) == 1.0)
    print(f">> Base Terminal Milestone Samples: {current_terminals:,}")

    # 2. Generate new realistic Terminal Milestone trajectories
    needed_terminals = max(0, TARGET_TERMINAL_COUNT - current_terminals)
    print(f">> Generating {needed_terminals:,} new Realistic Terminal Milestone trajectories...")
    for i in range(needed_terminals):
        s = generate_v62_synthetic_sample(len(v62_samples) + i, regime="terminal")
        v62_samples.append(s)

    # 3. Generate Hard Near-Terminal Negatives (20,000 samples)
    target_near_term_negs = 20000
    print(f">> Generating {target_near_term_negs:,} Hard Near-Terminal Negatives (results loaded / input typed)...")
    for i in range(target_near_term_negs):
        s = generate_v62_synthetic_sample(len(v62_samples) + i, regime="near_terminal_neg")
        v62_samples.append(s)

    # 4. Generate initial active & distractor trajectories to reach TARGET_DATASET_SIZE
    needed_total = max(0, TARGET_DATASET_SIZE - len(v62_samples))
    print(f">> Generating {needed_total:,} new Active Navigation & Distractor trajectories...")
    for i in range(needed_total):
        force_dist = (i % 2 == 0)
        s = generate_v62_synthetic_sample(len(v62_samples) + i, regime="initial", force_distractor=force_dist)
        v62_samples.append(s)

    random.shuffle(v62_samples)
    final_terminals = sum(1 for s in v62_samples if s.get("is_terminal_complete", 0.0) == 1.0)
    final_near_negs = sum(1 for s in v62_samples if s.get("is_near_miss_negative", 0.0) == 1.0)
    final_distractors = sum(1 for s in v62_samples if any(c.get("is_distractor", 0.0) == 1.0 for c in s.get("candidates", [])))

    print("\n" + "=" * 80)
    print(">> Final V62 Authoritative Dataset Census:")
    print(f"   Total Trajectories:         {len(v62_samples):,}")
    print(f"   Terminal Milestones:        {final_terminals:,} ({final_terminals/len(v62_samples)*100:.1f}%)")
    print(f"   Hard Near-Terminal Negs:    {final_near_negs:,} ({final_near_negs/len(v62_samples)*100:.1f}%)")
    print(f"   Distractor Trajectories:    {final_distractors:,} ({final_distractors/len(v62_samples)*100:.1f}%)")
    print("=" * 80)

    OUTPUT_V62_DATASET.parent.mkdir(parents=True, exist_ok=True)
    print(f">> Saving authoritative dataset to {OUTPUT_V62_DATASET}...")
    torch.save(v62_samples, OUTPUT_V62_DATASET)
    file_size_mb = OUTPUT_V62_DATASET.stat().st_size / 1e6
    print(f">> Successfully saved V62 dataset ({file_size_mb:.2f} MB).")


if __name__ == "__main__":
    main()
