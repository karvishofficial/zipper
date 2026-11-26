"""
KARVISH UNIVERSAL V57 — AUTHORITATIVE FAILURE-CLASS DATASET GENERATOR
Conforming strictly to Karvish Constitution (AGENTS.md):
  1. Ingests all 12 frozen baseline anchors & V56 generalized failure classes.
  2. Multi-Hop Social Navigation (Instagram, Threads, Twitter Feed-to-Inbox routing).
  3. Search History Chips vs. Active Composer Input Disambiguation (YouTube, Play Store).
  4. Selected Bottom Tab vs. Top Search Input Disambiguation (Instagram Explore, Zepto).
  5. Dual-Slot Action-Conditioned Span Grounding (Slot 0: Recipient/Query vs Slot 1: Payload).
  6. Permutation Invariance & Evidence-Gated Completion (BCE Calibration).
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
V56_DIR = SCRIPT_DIR.parent / "v56"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(V56_DIR))

from build_v56_generalized_dataset import generate_v56_generalized_dataset, _normalize_item, find_token_span
from schema_v57 import CANONICAL_ACTIONS, ACTION_TO_IDX, ROLE_TO_IDX

TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


def build_social_multihop_trajectories() -> List[Dict[str, Any]]:
    """
    Generates multi-hop trajectories for social apps (Instagram, Threads, Twitter, Reddit)
    navigating from the public feed to the private direct messaging inbox and sending messages.
    Teaches the model to ignore stranger post authors and story circles when messaging.
    """
    items: List[Dict[str, Any]] = []

    apps = ["Instagram", "Threads", "Twitter", "Reddit"]
    recipients = ["Manoj", "Ajay", "Karthik", "Bhavani", "Sneha", "Vikram", "Priya", "Rahul", "Carlos", "Elena"]
    messages = ["hi", "hello", "where are you", "meeting at 5", "call me please", "check this out", "how are you"]
    stranger_authors = [
        "mahimagi09", "factbysciencex", "daily_memes_hub", "photography_daily",
        "tech_insider", "nature_vibes", "wanderlust_travel", "code_ninja"
    ]

    for app in apps:
        for rec in recipients:
            msg = random.choice(messages)
            goal = f"Open {app} and send {msg} to {rec}"
            toks = TOKENIZER(goal, return_offsets_mapping=True)
            offsets = toks["offset_mapping"]

            rec_span = find_token_span(goal, rec, TOKENIZER)
            msg_span = find_token_span(goal, msg, TOKENIZER)

            # -------------------------------------------------------------
            # HOP 1: Home Feed Screen -> Must click Top-Right DM Icon
            # -------------------------------------------------------------
            cands_hop1 = [
                # Top Header Elements
                {"label": f"{app}", "role": "HEADER", "bounds": [48, 120, 300, 200], "is_clickable": False, "is_editable": False, "is_selected": False},
                {"label": "Notifications", "role": "BUTTON", "bounds": [800, 120, 900, 200], "is_clickable": True, "is_editable": False, "is_selected": False},
                {"label": "Message", "role": "BUTTON", "bounds": [936, 120, 1056, 200], "is_clickable": True, "is_editable": False, "is_selected": False}, # <--- TARGET
                # Distractor Story Circles
                {"label": "Your story", "role": "BUTTON", "bounds": [36, 240, 180, 380], "is_clickable": True, "is_editable": False, "is_selected": False},
                {"label": f"{random.choice(stranger_authors)}", "role": "BUTTON", "bounds": [200, 240, 350, 380], "is_clickable": True, "is_editable": False, "is_selected": False},
                # Distractor Feed Posts
                {"label": f"Post by {random.choice(stranger_authors)}", "role": "BUTTON", "bounds": [48, 420, 1032, 1200], "is_clickable": True, "is_editable": False, "is_selected": False},
                {"label": "Like", "role": "BUTTON", "bounds": [48, 1220, 140, 1300], "is_clickable": True, "is_editable": False, "is_selected": False},
                {"label": "Comment", "role": "BUTTON", "bounds": [160, 1220, 260, 1300], "is_clickable": True, "is_editable": False, "is_selected": False},
                {"label": "Share", "role": "BUTTON", "bounds": [280, 1220, 380, 1300], "is_clickable": True, "is_editable": False, "is_selected": False},
                # Bottom Navigation Bar
                {"label": "Home", "role": "TAB_ITEM", "bounds": [0, 2100, 216, 2240], "is_clickable": True, "is_editable": False, "is_selected": True},
                {"label": "Search and explore", "role": "TAB_ITEM", "bounds": [216, 2100, 432, 2240], "is_clickable": True, "is_editable": False, "is_selected": False},
                {"label": "Reels", "role": "TAB_ITEM", "bounds": [648, 2100, 864, 2240], "is_clickable": True, "is_editable": False, "is_selected": False},
                {"label": "Profile", "role": "TAB_ITEM", "bounds": [864, 2100, 1080, 2240], "is_clickable": True, "is_editable": False, "is_selected": False},
            ]
            items.append(_normalize_item({
                "goal": goal,
                "dataset_class": "v57_social_multihop_feed",
                "current_phase": "NAV_TO_MESSAGING",
                "expected_state": "INBOX_OPENED",
                "candidates": cands_hop1,
                "target_idx": 2, # "Message" icon
                "action_idx": 0, # CLICK
                "slot_type": 0,
                "target_span": (0, 0),
                "is_final": False,
                "verification_evidence": 0.0,
            }))

            # -------------------------------------------------------------
            # HOP 2: Direct Messages Inbox -> Must search recipient
            # -------------------------------------------------------------
            cands_hop2 = [
                {"label": "Back", "role": "BACK_BUTTON", "bounds": [24, 100, 120, 200], "is_clickable": True, "is_editable": False, "is_selected": False},
                {"label": "Messages", "role": "HEADER", "bounds": [140, 100, 400, 200], "is_clickable": False, "is_editable": False, "is_selected": False},
                {"label": "Search", "role": "SEARCH_INPUT", "bounds": [48, 240, 1032, 360], "is_clickable": True, "is_editable": True, "is_selected": False}, # <--- TARGET
                {"label": "Primary", "role": "TAB_ITEM", "bounds": [48, 380, 250, 460], "is_clickable": True, "is_editable": False, "is_selected": True},
                {"label": "General", "role": "TAB_ITEM", "bounds": [260, 380, 460, 460], "is_clickable": True, "is_editable": False, "is_selected": False},
                {"label": "Recent Chat 1", "role": "CHAT_ITEM", "bounds": [48, 480, 1032, 600], "is_clickable": True, "is_editable": False, "is_selected": False},
                {"label": "Recent Chat 2", "role": "CHAT_ITEM", "bounds": [48, 620, 1032, 740], "is_clickable": True, "is_editable": False, "is_selected": False},
            ]
            items.append(_normalize_item({
                "goal": goal,
                "dataset_class": "v57_social_multihop_inbox",
                "current_phase": "SEARCH_RECIPIENT",
                "expected_state": "RECIPIENT_FILTERED",
                "candidates": cands_hop2,
                "target_idx": 2, # "Search" input
                "action_idx": 4, # TYPE_TEXT
                "slot_type": 0, # Slot 0: Recipient
                "target_span": rec_span,
                "argument_value": rec,
                "is_final": False,
                "verification_evidence": 0.0,
            }))

            # -------------------------------------------------------------
            # HOP 3: Filtered Results -> Must Click Matching Recipient
            # -------------------------------------------------------------
            cands_hop3 = [
                {"label": f"{rec}", "role": "CONTACT_ITEM", "bounds": [48, 380, 1032, 520], "is_clickable": True, "is_editable": False, "is_selected": False}, # <--- TARGET
                {"label": f"{rec} Kumar", "role": "CONTACT_ITEM", "bounds": [48, 540, 1032, 680], "is_clickable": True, "is_editable": False, "is_selected": False},
                {"label": "Other Contact", "role": "CONTACT_ITEM", "bounds": [48, 700, 1032, 840], "is_clickable": True, "is_editable": False, "is_selected": False},
            ]
            items.append(_normalize_item({
                "goal": goal,
                "dataset_class": "v57_social_multihop_contact",
                "current_phase": "SELECT_CONTACT",
                "expected_state": "CONVERSATION_OPENED",
                "candidates": cands_hop3,
                "target_idx": 0, # Recipient card
                "action_idx": 0, # CLICK
                "slot_type": 0,
                "target_span": (0, 0),
                "is_final": False,
                "verification_evidence": 0.0,
            }))

            # -------------------------------------------------------------
            # HOP 4: In-Chat Screen -> Must Type Message in Composer
            # -------------------------------------------------------------
            cands_hop4 = [
                {"label": f"{rec}", "role": "HEADER", "bounds": [140, 100, 500, 200], "is_clickable": False, "is_editable": False, "is_selected": False},
                {"label": "Audio call", "role": "BUTTON", "bounds": [750, 100, 850, 200], "is_clickable": True, "is_editable": False, "is_selected": False},
                {"label": "Video call", "role": "BUTTON", "bounds": [870, 100, 970, 200], "is_clickable": True, "is_editable": False, "is_selected": False},
                {"label": "Conversation list", "role": "SCROLL_CONTAINER", "bounds": [0, 220, 1080, 1900], "is_clickable": False, "is_editable": False, "is_selected": False},
                {"label": "Message...", "role": "MESSAGE_INPUT", "bounds": [48, 1950, 900, 2100], "is_clickable": True, "is_editable": True, "is_selected": False}, # <--- TARGET
                {"label": "Send", "role": "SEND_BUTTON", "bounds": [920, 1950, 1040, 2100], "is_clickable": True, "is_editable": False, "is_selected": False},
            ]
            items.append(_normalize_item({
                "goal": goal,
                "dataset_class": "v57_social_multihop_composer",
                "current_phase": "TYPE_MESSAGE",
                "expected_state": "MESSAGE_POPULATED",
                "candidates": cands_hop4,
                "target_idx": 4, # "Message..." input
                "action_idx": 4, # TYPE_TEXT
                "slot_type": 1, # Slot 1: Payload
                "target_span": msg_span,
                "argument_value": msg,
                "is_final": False,
                "verification_evidence": 0.0,
            }))

            # -------------------------------------------------------------
            # HOP 5: Populated Composer -> Must Click Send
            # -------------------------------------------------------------
            cands_hop5 = copy.deepcopy(cands_hop4)
            cands_hop5[4]["label"] = msg
            items.append(_normalize_item({
                "goal": goal,
                "dataset_class": "v57_social_multihop_send",
                "current_phase": "SUBMIT_MESSAGE",
                "expected_state": "MESSAGE_DELIVERED",
                "candidates": cands_hop5,
                "target_idx": 5, # Send Button
                "action_idx": 0, # CLICK
                "slot_type": 0,
                "target_span": (0, 0),
                "is_final": False,
                "verification_evidence": 0.0,
            }))

    return items


def build_search_history_disambiguation_data() -> List[Dict[str, Any]]:
    """
    Teaches the model to select the active search input field for typing when a query is specified,
    strictly penalizing recent search history chips (e.g. 'doraemon hindi', 'doraemon hindi movie').
    """
    items: List[Dict[str, Any]] = []

    apps = ["YouTube", "Play Store", "Google", "Zepto", "Chrome", "Amazon"]
    queries = [
        "lo-fi hip hop", "SpaceX launch", "fresh tomatoes", "calculator",
        "wireless mouse", "python programming", "headphones", "cricket score",
        "weather update", "running shoes", "flight tickets", "organic milk"
    ]
    stale_history_chips = [
        "doraemon hindi", "doraemon hindi movie", "motu patlu cartoon",
        "latest hindi songs", "funny cat videos", "old bollywood movies",
        "news live stream", "gaming setup tour", "minecraft gameplay"
    ]

    for app in apps:
        for q in queries:
            goal = f"Open {app} and search {q}"
            toks = TOKENIZER(goal, return_offsets_mapping=True)
            offsets = toks["offset_mapping"]
            q_span = find_token_span(goal, q, TOKENIZER)

            chip1 = random.choice(stale_history_chips)
            chip2 = random.choice([c for c in stale_history_chips if c != chip1])
            chip3 = random.choice([c for c in stale_history_chips if c not in (chip1, chip2)])

            cands = [
                {"label": "Back", "role": "BACK_BUTTON", "bounds": [24, 100, 120, 200], "is_clickable": True, "is_editable": False, "is_selected": False},
                {"label": f"Search {app}", "role": "SEARCH_INPUT", "bounds": [140, 100, 900, 200], "is_clickable": True, "is_editable": True, "is_focused": True, "is_selected": False}, # <--- TARGET
                {"label": "Voice search", "role": "BUTTON", "bounds": [920, 100, 1040, 200], "is_clickable": True, "is_editable": False, "is_selected": False},
                # Stale Search History Chips (HARD NEGATIVES)
                {"label": chip1, "role": "BUTTON", "bounds": [48, 240, 1032, 340], "is_clickable": True, "is_editable": False, "is_selected": False},
                {"label": chip2, "role": "BUTTON", "bounds": [48, 360, 1032, 460], "is_clickable": True, "is_editable": False, "is_selected": False},
                {"label": chip3, "role": "BUTTON", "bounds": [48, 480, 1032, 580], "is_clickable": True, "is_editable": False, "is_selected": False},
            ]

            items.append(_normalize_item({
                "goal": goal,
                "dataset_class": "v57_search_history_disambiguation",
                "current_phase": "INPUT_SEARCH_QUERY",
                "expected_state": "SEARCH_SUBMITTED",
                "candidates": cands,
                "target_idx": 1, # Search input box
                "action_idx": 4, # TYPE_TEXT
                "slot_type": 0, # Slot 0: Query
                "target_span": q_span,
                "argument_value": q,
                "is_final": False,
                "verification_evidence": 0.0,
            }))

    return items


def build_selected_tab_invariance_data() -> List[Dict[str, Any]]:
    """
    Teaches the model that when a navigation tab is already selected (e.g. Explore tab),
    re-tapping it is a NO-OP, and it must target the search input at the top of the screen.
    """
    items: List[Dict[str, Any]] = []

    apps = ["Instagram", "Zepto", "Amazon", "Play Store", "LinkedIn"]
    queries = ["photography", "tomatoes", "shoes", "telegram", "software engineer"]

    for app, q in zip(apps, queries):
        goal = f"Open {app} and search {q}"
        toks = TOKENIZER(goal, return_offsets_mapping=True)
        offsets = toks["offset_mapping"]
        q_span = find_token_span(goal, q, TOKENIZER)

        cands = [
            # Top Search Input
            {"label": f"Search on {app}", "role": "SEARCH_INPUT", "bounds": [48, 120, 1032, 240], "is_clickable": True, "is_editable": True, "is_selected": False}, # <--- TARGET
            # Grid Elements
            {"label": "Item 1", "role": "PRODUCT_CARD", "bounds": [48, 280, 500, 700], "is_clickable": True, "is_editable": False, "is_selected": False},
            {"label": "Item 2", "role": "PRODUCT_CARD", "bounds": [540, 280, 1032, 700], "is_clickable": True, "is_editable": False, "is_selected": False},
            # Bottom Navigation Bar
            {"label": "Home", "role": "TAB_ITEM", "bounds": [0, 2100, 270, 2240], "is_clickable": True, "is_editable": False, "is_selected": False},
            {"label": "Search and explore", "role": "TAB_ITEM", "bounds": [270, 2100, 540, 2240], "is_clickable": True, "is_editable": False, "is_selected": True}, # <--- ALREADY SELECTED (NO-OP)
            {"label": "Orders", "role": "TAB_ITEM", "bounds": [540, 2100, 810, 2240], "is_clickable": True, "is_editable": False, "is_selected": False},
            {"label": "Account", "role": "TAB_ITEM", "bounds": [810, 2100, 1080, 2240], "is_clickable": True, "is_editable": False, "is_selected": False},
        ]

        items.append(_normalize_item({
            "goal": goal,
            "dataset_class": "v57_selected_tab_invariance",
            "current_phase": "INPUT_SEARCH_QUERY",
            "expected_state": "SEARCH_RESULTS",
            "candidates": cands,
            "target_idx": 0, # Top search input
            "action_idx": 4, # TYPE_TEXT
            "slot_type": 0,
            "target_span": q_span,
            "argument_value": q,
            "is_final": False,
            "verification_evidence": 0.0,
        }))

    return items


def generate_v57_authoritative_dataset() -> List[Dict[str, Any]]:
    """
    Compiles the authoritative V57 dataset:
    - All V56 baseline items (V54 anchors + Comm lifecycle + Permutations)
    - + Social Multi-Hop Navigation (Feed -> Inbox -> Chat)
    - + Search History Chip Disambiguation (Active input vs History chips)
    - + Selected Tab Invariance (Top search input vs Selected bottom tab)
    """
    # 1. Ingest V56 Baseline
    v56_items = generate_v56_generalized_dataset()
    print(f">> Ingested {len(v56_items)} verified V56 foundation items.")

    # 2. Generate New V57 Failure-Class Trajectories
    social_items = build_social_multihop_trajectories()
    print(f">> Synthesized {len(social_items)} Social Multi-Hop Navigation items.")

    history_items = build_search_history_disambiguation_data()
    print(f">> Synthesized {len(history_items)} Search History Disambiguation items.")

    tab_items = build_selected_tab_invariance_data()
    print(f">> Synthesized {len(tab_items)} Selected Tab Invariance items.")

    all_items = v56_items + social_items + history_items + tab_items
    random.seed(42)
    random.shuffle(all_items)

    print(f"============================================================")
    print(f"V57 AUTHORITATIVE DATASET COMPILED: {len(all_items)} TOTAL EXAMPLES")
    print(f"============================================================")
    return all_items


if __name__ == "__main__":
    out_dir = SCRIPT_DIR / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = generate_v57_authoritative_dataset()
    out_file = out_dir / "v57_authoritative_dataset.pt"
    torch.save(dataset, out_file)
    print(f">> Successfully saved V57 dataset to {out_file} ({len(dataset)} items).")
