"""
KARVISH UNIVERSAL V56 — CROSS-APP GENERALIZED FAILURE-CLASS DATASET GENERATOR
Authoritative implementation conforming to Karvish Constitution (AGENTS.md) and Senior's Mandate:
- Full Communication Lifecycle Grounding (LinkedIn Profile Message, Gmail Inbox Compose, Chat Composer, Send).
- Selected Navigation Tab vs. Open Editable Search Input (Instagram Explore & Store Landings).
- Dual-Slot Action-Conditioned Span Grounding (Slot 0: Recipient/Query vs. Slot 1: Payload).
- Evidence-Gated Completion via BCE Calibration (Observable Evidence 0.0 vs 1.0, is_final: False/True).
- Candidate Permutation Variants: 100% order-invariant training data.
- Zero app-specific branching; 100% learned universal neural affordance representations.
"""

from __future__ import annotations

import sys
import math
import random
import copy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
V54_DIR = SCRIPT_DIR.parent / "v54"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(V54_DIR))

from smoke_universal_search_finetune import generate_universal_phase_b_data, find_token_span
from schema_v56 import ACTION_TO_IDX, ROLE_TO_IDX, CANONICAL_ACTIONS

TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


def _normalize_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Ensures every dataset item adheres to the canonical V56 schema."""
    it = copy.deepcopy(item)
    if "candidates" not in it and "candidate_set" in it:
        it["candidates"] = it["candidate_set"]
    if "candidate_set" not in it and "candidates" in it:
        it["candidate_set"] = it["candidates"]

    cands = it["candidates"]
    target_idx = it.get("target_idx", 0)
    if 0 <= target_idx < len(cands):
        it["correct_candidate"] = str(cands[target_idx].get("label", ""))
    else:
        it["correct_candidate"] = ""

    action_idx = it.get("action_idx", 0)
    it["action_idx"] = action_idx
    it["action"] = CANONICAL_ACTIONS[action_idx] if 0 <= action_idx < len(CANONICAL_ACTIONS) else "CLICK"

    # Slot alignment
    slot_type = it.get("slot_type", it.get("argument_slot", 0))
    it["slot_type"] = slot_type
    it["argument_slot"] = slot_type if it["action"] == "TYPE_TEXT" else -1

    # Text value from span
    target_span = it.get("target_span", (0, 0))
    it["target_span"] = target_span
    goal = it.get("goal", "")
    it["goal"] = goal
    it["argument_value"] = it.get("argument_value", "")

    # Evidence and completion state
    is_terminal = it.get("is_final", it.get("is_terminal", False))
    it["is_final"] = bool(is_terminal)
    evidence = float(it.get("verification_evidence", it.get("evidence", 1.0 if is_terminal else 0.0)))
    it["verification_evidence"] = evidence
    it["evidence"] = evidence

    it["stall_target"] = float(it.get("stall_target", 0.0))
    it["current_phase"] = str(it.get("current_phase", "EXECUTE_ACTION"))
    it["expected_state"] = str(it.get("expected_state", "OBSERVABLE_TRANSITION"))

    return it


def generate_v56_generalized_dataset() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    # =========================================================================
    # 1. Base Universal V54 Replay Data (Invariance Shield Anchors + Price Bounds)
    # =========================================================================
    base_v54_data = generate_universal_phase_b_data()
    for item in base_v54_data:
        it = copy.deepcopy(item)
        it["dataset_class"] = "v54_replay"
        it["current_phase"] = "REPLAY_ANCHOR"
        it["expected_state"] = "OBSERVABLE_TRANSITION"
        it["is_final"] = it.get("is_terminal", False)
        it["verification_evidence"] = 1.0 if it["is_final"] else 0.0
        it["evidence"] = it["verification_evidence"]
        items.append(_normalize_item(it))

    print(f">> Ingested {len(base_v54_data)} frozen V54 replay anchor items.")

    # =========================================================================
    # 2. FAILURE CLASS A: Universal Communication Lifecycle Grounding
    # Solves physical failures on LinkedIn (Manoj loop) & Gmail (Karthik compose).
    # Lifecycle: SEARCH_RESULTS -> RECIPIENT_SELECTION -> PROFILE/INBOX -> COMPOSE/MESSAGE -> INPUT -> SEND -> VERIFY
    # =========================================================================
    comm_apps = [
        "LinkedIn", "Gmail", "WhatsApp", "Telegram", "Signal", "Slack",
        "Messages", "Discord", "Instagram", "WeChat", "Viber", "Outlook"
    ]

    recipients = [
        "Manoj", "Karthik", "Bhavani", "Basavaraj", "Ajay", "Shobha Aunty", "Dr. Ramesh",
        "Rekha", "Carlos", "Elena", "Fatima", "Lucas", "Maya", "Priya", "Vikram",
        "Sneha", "Kavya", "Suresh", "Rohan", "Ahmed", "Sarah", "Dr. Rajesh", "Meera",
        "Neha", "Arjun", "Aditi", "Devi", "Uncle Bob", "Aarav Patel", "Fatima Zahra",
        "Elena Rostova", "Deepak Rao"
    ]

    messages = [
        "hi", "hello", "where are you", "meeting at 5", "call me",
        "happy birthday", "good morning", "thanks", "on my way",
        "see you soon", "I will be late", "got the file", "can we talk",
        "check this out", "how are you", "sounds good", "please confirm",
        "yes sure", "let's do it", "sharing details now", "congratulations on the role",
        "project update attached", "please review", "arrived at venue", "great work"
    ]

    composer_hints = [
        "Type a message", "Message", "Text message", "Write a message",
        "Chat...", "Write a reply...", "Send a chat...", "Type message here",
        "Start a message", "Compose email"
    ]

    for name in recipients:
        for app in random.sample(comm_apps, 4):
            payload = random.choice(messages)
            composer_hint = random.choice(composer_hints)

            templates = [
                f"open {app} and message {name}: {payload}",
                f"open {app} and send {payload} to {name}",
                f"send {payload} to {name} on {app}",
                f"message {name} on {app}: {payload}",
                f"tell {name} {payload} on {app}",
                f"text {name} {payload} on {app}",
            ]
            goal = random.choice(templates)

            recip_st, recip_en = find_token_span(goal, name, TOKENIZER)
            msg_st, msg_en = find_token_span(goal, payload, TOKENIZER)

            # -----------------------------------------------------------------
            # Screen A1: Search Results -> Select Recipient Card
            # TARGET: Recipient contact card.
            # HARD NEGATIVE: Search input field (prevents re-searching loop).
            # -----------------------------------------------------------------
            cands_results = [
                {
                    "label": f"{name}",  # Recipient contact item
                    "role_idx": 5,        # CONTACT_ITEM
                    "is_editable": False,
                    "is_clickable": True,
                    "bounds_vec": [0.05, 0.15, 0.95, 0.25, 0.50, 0.20, 0.90, 0.10, 0.09, 30.0],
                },
                {
                    "label": f"Search {name}",  # Populated search input
                    "role_idx": 3,             # SEARCH_INPUT
                    "is_editable": True,
                    "is_clickable": True,
                    "is_populated": True,
                    "bounds_vec": [0.05, 0.04, 0.85, 0.10, 0.45, 0.07, 0.80, 0.06, 0.048, 15.0],
                },
                {
                    "label": "Clear query",
                    "role_idx": 30,            # CLEAR_BUTTON
                    "is_editable": False,
                    "is_clickable": True,
                    "bounds_vec": [0.86, 0.04, 0.95, 0.10, 0.90, 0.07, 0.09, 0.06, 0.005, 15.0],
                },
                {
                    "label": "Back",
                    "role_idx": 12,
                    "is_editable": False,
                    "is_clickable": True,
                    "bounds_vec": [0.02, 0.04, 0.10, 0.10, 0.06, 0.07, 0.08, 0.06, 0.005, 15.0],
                },
                {
                    "label": "complete",
                    "role_idx": 1,
                    "is_editable": False,
                    "is_clickable": True,
                    "bounds_vec": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                }
            ]

            for perm_idx in range(2):
                c_list = copy.deepcopy(cands_results)
                if perm_idx > 0:
                    random.shuffle(c_list)
                t_idx = next(i for i, c in enumerate(c_list) if c["label"] == name)

                items.append(_normalize_item({
                    "name": f"comm_results_select_{app.lower()}_{name.lower().replace(' ', '_')}_p{perm_idx}",
                    "category": "communication_lifecycle",
                    "dataset_class": "failure_class_a",
                    "goal": goal,
                    "candidates": c_list,
                    "target_idx": t_idx,
                    "action_idx": ACTION_TO_IDX["CLICK"],
                    "target_span": (recip_st, recip_en),
                    "slot_type": 0,
                    "argument_value": name,
                    "stall_target": 0.0,
                    "evidence": 0.0,
                    "is_final": False,
                    "verification_evidence": 0.0,
                    "current_phase": "SEARCH_RESULTS",
                    "expected_state": "RECIPIENT_SELECTED",
                }))

            # -----------------------------------------------------------------
            # Screen A2: Profile Context -> Select Message Affordance (LinkedIn Fix!)
            # TARGET: "Message" action button.
            # HARD NEGATIVE: Search bar, Connect, Follow.
            # -----------------------------------------------------------------
            msg_btn_label = "Message" if app != "Gmail" else "Send message"
            cands_profile = [
                {
                    "label": msg_btn_label,    # Target action button
                    "role_idx": 1,             # BUTTON
                    "is_editable": False,
                    "is_clickable": True,
                    "bounds_vec": [0.05, 0.40, 0.45, 0.48, 0.25, 0.44, 0.40, 0.08, 0.032, 25.0],
                },
                {
                    "label": f"Search",        # Search header
                    "role_idx": 3,             # SEARCH_INPUT
                    "is_editable": True,
                    "is_clickable": True,
                    "bounds_vec": [0.05, 0.04, 0.95, 0.10, 0.50, 0.07, 0.90, 0.06, 0.054, 15.0],
                },
                {
                    "label": f"{name}",        # Profile header
                    "role_idx": 22,            # PROFILE_HEADER
                    "is_editable": False,
                    "is_clickable": False,
                    "bounds_vec": [0.05, 0.20, 0.70, 0.30, 0.37, 0.25, 0.65, 0.10, 0.065, 20.0],
                },
                {
                    "label": "Connect",
                    "role_idx": 1,
                    "is_editable": False,
                    "is_clickable": True,
                    "bounds_vec": [0.50, 0.40, 0.75, 0.48, 0.62, 0.44, 0.25, 0.08, 0.02, 25.0],
                },
                {
                    "label": "More actions",
                    "role_idx": 1,
                    "is_editable": False,
                    "is_clickable": True,
                    "bounds_vec": [0.80, 0.40, 0.95, 0.48, 0.87, 0.44, 0.15, 0.08, 0.012, 25.0],
                },
                {
                    "label": "complete",
                    "role_idx": 1,
                    "is_editable": False,
                    "is_clickable": True,
                    "bounds_vec": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                }
            ]

            for perm_idx in range(2):
                c_list = copy.deepcopy(cands_profile)
                if perm_idx > 0:
                    random.shuffle(c_list)
                t_idx = next(i for i, c in enumerate(c_list) if c["label"] == msg_btn_label)

                items.append(_normalize_item({
                    "name": f"comm_profile_message_{app.lower()}_{name.lower().replace(' ', '_')}_p{perm_idx}",
                    "category": "communication_lifecycle",
                    "dataset_class": "failure_class_a",
                    "goal": goal,
                    "candidates": c_list,
                    "target_idx": t_idx,
                    "action_idx": ACTION_TO_IDX["CLICK"],
                    "target_span": (0, 0),
                    "slot_type": 0,
                    "argument_value": "",
                    "stall_target": 0.0,
                    "evidence": 0.0,
                    "is_final": False,
                    "verification_evidence": 0.0,
                    "current_phase": "PROFILE_VIEW",
                    "expected_state": "COMPOSER_OPENED",
                }))

            # -----------------------------------------------------------------
            # Screen A3: Inbox Context -> Select Floating Compose (Gmail Fix!)
            # TARGET: Floating "Compose" button.
            # HARD NEGATIVE: "Search in emails", email threads, complete.
            # -----------------------------------------------------------------
            cands_inbox = [
                {
                    "label": "Compose",        # Floating Action Button
                    "role_idx": 1,             # BUTTON
                    "is_editable": False,
                    "is_clickable": True,
                    "bounds_vec": [0.65, 0.80, 0.95, 0.88, 0.80, 0.84, 0.30, 0.08, 0.024, 45.0],
                },
                {
                    "label": "Search in emails", # Top search bar
                    "role_idx": 3,               # SEARCH_INPUT
                    "is_editable": True,
                    "is_clickable": True,
                    "bounds_vec": [0.05, 0.04, 0.95, 0.10, 0.50, 0.07, 0.90, 0.06, 0.054, 15.0],
                },
                {
                    "label": f"{name} - Meeting notes", # Recent email thread
                    "role_idx": 15,                     # LIST_ITEM
                    "is_editable": False,
                    "is_clickable": True,
                    "bounds_vec": [0.05, 0.15, 0.95, 0.28, 0.50, 0.21, 0.90, 0.13, 0.117, 20.0],
                },
                {
                    "label": "Primary inbox",
                    "role_idx": 14,
                    "is_editable": False,
                    "is_clickable": True,
                    "is_selected": True,
                    "bounds_vec": [0.05, 0.92, 0.35, 0.98, 0.20, 0.95, 0.30, 0.06, 0.018, 55.0],
                },
                {
                    "label": "complete",
                    "role_idx": 1,
                    "is_editable": False,
                    "is_clickable": True,
                    "bounds_vec": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                }
            ]

            for perm_idx in range(2):
                c_list = copy.deepcopy(cands_inbox)
                if perm_idx > 0:
                    random.shuffle(c_list)
                t_idx = next(i for i, c in enumerate(c_list) if c["label"] == "Compose")

                items.append(_normalize_item({
                    "name": f"comm_inbox_compose_{app.lower()}_{name.lower().replace(' ', '_')}_p{perm_idx}",
                    "category": "communication_lifecycle",
                    "dataset_class": "failure_class_a",
                    "goal": goal,
                    "candidates": c_list,
                    "target_idx": t_idx,
                    "action_idx": ACTION_TO_IDX["CLICK"],
                    "target_span": (0, 0),
                    "slot_type": 0,
                    "argument_value": "",
                    "stall_target": 0.0,
                    "evidence": 0.0,
                    "is_final": False,
                    "verification_evidence": 0.0,
                    "current_phase": "INBOX_VIEW",
                    "expected_state": "COMPOSER_OPENED",
                }))

            # -----------------------------------------------------------------
            # Screen A4: Inside Chat Conversation -> Focus Message Composer
            # TARGET: Message input field (Dual-slot: Payload).
            # HARD NEGATIVE: Contact title header.
            # -----------------------------------------------------------------
            cands_chat = [
                {
                    "label": composer_hint,
                    "role_idx": 4,  # MESSAGE_INPUT
                    "is_editable": True,
                    "is_clickable": True,
                    "is_populated": False,
                    "bounds_vec": [0.05, 0.90, 0.85, 0.96, 0.45, 0.93, 0.80, 0.06, 0.048, 50.0],
                },
                {
                    "label": f"{name}",  # Top contact header
                    "role_idx": 11,      # HEADER / PROFILE_HEADER
                    "is_editable": False,
                    "is_clickable": True,
                    "bounds_vec": [0.15, 0.04, 0.65, 0.09, 0.40, 0.065, 0.50, 0.05, 0.025, 20.0],
                },
                {
                    "label": "Video call",
                    "role_idx": 1,
                    "is_editable": False,
                    "is_clickable": True,
                    "bounds_vec": [0.70, 0.04, 0.80, 0.09, 0.75, 0.065, 0.10, 0.05, 0.005, 20.0],
                },
                {
                    "label": "Attach media",
                    "role_idx": 1,
                    "is_editable": False,
                    "is_clickable": True,
                    "bounds_vec": [0.87, 0.90, 0.95, 0.96, 0.91, 0.93, 0.08, 0.06, 0.005, 50.0],
                },
                {
                    "label": "Back",
                    "role_idx": 12,
                    "is_editable": False,
                    "is_clickable": True,
                    "bounds_vec": [0.02, 0.04, 0.12, 0.09, 0.07, 0.065, 0.10, 0.05, 0.005, 20.0],
                },
                {
                    "label": "complete",
                    "role_idx": 1,
                    "is_editable": False,
                    "is_clickable": True,
                    "bounds_vec": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                }
            ]

            for perm_idx in range(2):
                c_list = copy.deepcopy(cands_chat)
                if perm_idx > 0:
                    random.shuffle(c_list)
                t_idx = next(i for i, c in enumerate(c_list) if c["label"] == composer_hint)

                items.append(_normalize_item({
                    "name": f"comm_chat_composer_{app.lower()}_{name.lower().replace(' ', '_')}_p{perm_idx}",
                    "category": "communication_lifecycle",
                    "dataset_class": "failure_class_a",
                    "goal": goal,
                    "candidates": c_list,
                    "target_idx": t_idx,
                    "action_idx": ACTION_TO_IDX["TYPE_TEXT"],
                    "target_span": (msg_st, msg_en),  # DUAL-SLOT: PAYLOAD SPAN!
                    "slot_type": 1,                   # 0 = Recipient, 1 = Payload
                    "argument_value": payload,
                    "stall_target": 0.0,
                    "evidence": 0.0,
                    "is_final": False,
                    "verification_evidence": 0.0,
                    "current_phase": "COMPOSE_MESSAGE",
                    "expected_state": "MESSAGE_TYPED",
                }))

            # -----------------------------------------------------------------
            # Screen A5: Message Typed -> Send Button Active
            # TARGET: Send button.
            # HARD NEGATIVE: Populated message input, Header.
            # -----------------------------------------------------------------
            cands_send = [
                {
                    "label": "Send",
                    "role_idx": 28, # SEND_BUTTON
                    "is_editable": False,
                    "is_clickable": True,
                    "bounds_vec": [0.88, 0.90, 0.98, 0.96, 0.93, 0.93, 0.10, 0.06, 0.006, 50.0],
                },
                {
                    "label": f"<{payload}",  # Populated composer
                    "role_idx": 4,           # MESSAGE_INPUT
                    "is_editable": True,
                    "is_clickable": True,
                    "is_populated": True,
                    "was_last_typed": True,
                    "bounds_vec": [0.05, 0.90, 0.85, 0.96, 0.45, 0.93, 0.80, 0.06, 0.048, 50.0],
                },
                {
                    "label": f"{name}",      # Top title header
                    "role_idx": 11,
                    "is_editable": False,
                    "is_clickable": True,
                    "bounds_vec": [0.15, 0.04, 0.65, 0.09, 0.40, 0.065, 0.50, 0.05, 0.025, 20.0],
                },
                {
                    "label": "complete",
                    "role_idx": 1,
                    "is_editable": False,
                    "is_clickable": True,
                    "bounds_vec": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                }
            ]

            for perm_idx in range(2):
                c_list = copy.deepcopy(cands_send)
                if perm_idx > 0:
                    random.shuffle(c_list)
                t_idx = next(i for i, c in enumerate(c_list) if c["label"] == "Send")

                items.append(_normalize_item({
                    "name": f"comm_chat_send_{app.lower()}_{name.lower().replace(' ', '_')}_p{perm_idx}",
                    "category": "communication_lifecycle",
                    "dataset_class": "failure_class_a",
                    "goal": goal,
                    "candidates": c_list,
                    "target_idx": t_idx,
                    "action_idx": ACTION_TO_IDX["CLICK"],
                    "target_span": (0, 0),
                    "slot_type": 0,
                    "argument_value": "",
                    "stall_target": 0.0,
                    "evidence": 0.0,
                    "is_final": False,
                    "verification_evidence": 0.0,
                    "current_phase": "COMMIT_MESSAGE",
                    "expected_state": "SEND_DISPATCHED",
                }))

            # -----------------------------------------------------------------
            # Screen A6: Send Verified Screen -> Task Complete
            # TARGET: complete candidate (Evidence is 1.0, is_final: True).
            # -----------------------------------------------------------------
            cands_verified = [
                {
                    "label": "complete",
                    "role_idx": 1,
                    "is_editable": False,
                    "is_clickable": True,
                    "bounds_vec": [0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0],
                },
                {
                    "label": f"{payload}",  # Sent message bubble
                    "role_idx": 6,          # CHAT_ITEM
                    "is_editable": False,
                    "is_clickable": False,
                    "bounds_vec": [0.40, 0.70, 0.95, 0.78, 0.67, 0.74, 0.55, 0.08, 0.044, 40.0],
                },
                {
                    "label": "Delivered 5:30 PM",  # Verified delivery timestamp
                    "role_idx": 23,                # STATUS_BAR / META
                    "is_editable": False,
                    "is_clickable": False,
                    "bounds_vec": [0.65, 0.79, 0.95, 0.82, 0.80, 0.805, 0.30, 0.03, 0.009, 40.0],
                },
                {
                    "label": composer_hint,
                    "role_idx": 4,
                    "is_editable": True,
                    "is_clickable": True,
                    "bounds_vec": [0.05, 0.90, 0.85, 0.96, 0.45, 0.93, 0.80, 0.06, 0.048, 50.0],
                },
            ]

            for perm_idx in range(2):
                c_list = copy.deepcopy(cands_verified)
                if perm_idx > 0:
                    random.shuffle(c_list)
                t_idx = next(i for i, c in enumerate(c_list) if c["label"] == "complete")

                items.append(_normalize_item({
                    "name": f"comm_send_verified_{app.lower()}_{name.lower().replace(' ', '_')}_p{perm_idx}",
                    "category": "communication_lifecycle",
                    "dataset_class": "failure_class_a",
                    "goal": goal,
                    "candidates": c_list,
                    "target_idx": t_idx,
                    "action_idx": ACTION_TO_IDX["WAIT"],
                    "target_span": (0, 0),
                    "slot_type": 0,
                    "argument_value": "",
                    "stall_target": 0.0,
                    "evidence": 1.0,
                    "is_final": True,
                    "verification_evidence": 1.0,
                    "current_phase": "VERIFY_TRANSMISSION",
                    "expected_state": "OUTCOME_VERIFIED",
                }))

    print(f">> Added Failure Class A (Full Communication Lifecycle): Total dataset items now = {len(items)}")

    # =========================================================================
    # 3. FAILURE CLASS B: Selected Tab vs. Open Input (Across 8 Discovery Apps)
    # Selected Tab is_selected=True is a NO-OP Trap; Top Input is the true TARGET.
    # =========================================================================
    discovery_apps = [
        ("Instagram", "search {query} on Instagram", "Search and explore", "Search"),
        ("Play Store", "search {query} on Play Store", "Search", "Search Google Play"),
        ("YouTube", "search {query} on YouTube", "Explore", "Search YouTube"),
        ("Twitter", "search {query} on Twitter", "Explore", "Search Twitter"),
        ("Reddit", "search {query} on Reddit", "Discover", "Search Reddit"),
        ("TikTok", "search {query} on TikTok", "Discover", "Search"),
        ("Pinterest", "search {query} on Pinterest", "Search", "Search for ideas"),
        ("Maps", "search {query} on Google Maps", "Explore", "Search here"),
    ]

    discovery_queries = [
        "photography", "lo-fi hip hop", "Minecraft", "software engineer",
        "astronomy", "machine learning", "cooking recipes", "SpaceX",
        "travel vlogs", "mechanical keyboards", "digital art", "fitness workout"
    ]

    for app, goal_tpl, tab_name, input_hint in discovery_apps:
        for query in discovery_queries:
            goal = goal_tpl.format(query=query)
            q_st, q_en = find_token_span(goal, query, TOKENIZER)

            cands_explore = [
                {
                    "label": input_hint,
                    "role_idx": 3,  # SEARCH_INPUT
                    "is_editable": True,
                    "is_clickable": True,
                    "is_populated": False,
                    "bounds_vec": [0.05, 0.04, 0.95, 0.10, 0.50, 0.07, 0.90, 0.06, 0.054, 15.0],
                },
                {
                    "label": tab_name,  # Bottom navigation tab
                    "role_idx": 14,     # TAB_ITEM
                    "is_editable": False,
                    "is_clickable": True,
                    "is_selected": True, # CRITICAL: ALREADY SELECTED NO-OP!
                    "bounds_vec": [0.25, 0.93, 0.50, 0.99, 0.37, 0.96, 0.25, 0.06, 0.015, 60.0],
                },
                {
                    "label": "Home",
                    "role_idx": 14,
                    "is_editable": False,
                    "is_clickable": True,
                    "is_selected": False,
                    "bounds_vec": [0.0, 0.93, 0.25, 0.99, 0.12, 0.96, 0.25, 0.06, 0.015, 60.0],
                },
                {
                    "label": "Profile",
                    "role_idx": 14,
                    "is_editable": False,
                    "is_clickable": True,
                    "is_selected": False,
                    "bounds_vec": [0.75, 0.93, 1.0, 0.99, 0.87, 0.96, 0.25, 0.06, 0.015, 60.0],
                },
                {
                    "label": "Reshare number is 15002",
                    "role_idx": 24, # MEDIA_ITEM
                    "is_editable": False,
                    "is_clickable": True,
                    "bounds_vec": [0.05, 0.20, 0.48, 0.45, 0.26, 0.32, 0.43, 0.25, 0.107, 30.0],
                },
                {
                    "label": "complete",
                    "role_idx": 1,
                    "is_editable": False,
                    "is_clickable": True,
                    "bounds_vec": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                },
            ]

            for perm_idx in range(2):
                c_list = copy.deepcopy(cands_explore)
                if perm_idx > 0:
                    random.shuffle(c_list)
                t_idx = next(i for i, c in enumerate(c_list) if c["label"] == input_hint)

                items.append(_normalize_item({
                    "name": f"explore_open_input_{app.lower()}_{query.lower().replace(' ', '_')}_p{perm_idx}",
                    "category": "explore_open_input",
                    "dataset_class": "failure_class_b",
                    "goal": goal,
                    "candidates": c_list,
                    "target_idx": t_idx,
                    "action_idx": ACTION_TO_IDX["TYPE_TEXT"],
                    "target_span": (q_st, q_en),
                    "slot_type": 0,  # Recipient / Query
                    "argument_value": query,
                    "stall_target": 0.0,
                    "evidence": 0.0,
                    "is_final": False,
                    "verification_evidence": 0.0,
                    "current_phase": "LOCATE_INPUT",
                    "expected_state": "SEARCH_INPUT_FOCUSED",
                }))

    print(f">> Added Failure Class B (Selected Tab vs Open Input): Total dataset items now = {len(items)}")

    # =========================================================================
    # 4. FAILURE CLASS D: Evidence-Gated Completion (Learned BCE Calibration)
    # Intermediate non-final turns (evidence=0.0, is_final=False) vs.
    # Verified terminal turns (evidence=1.0, is_final=True).
    # =========================================================================
    ecom_apps = ["Zepto", "Blinkit", "Instamart", "Amazon"]
    ecom_products = ["fresh tomatoes", "potato", "onion 1kg", "organic milk", "green tea", "bread"]

    for e_app in ecom_apps:
        for prod in ecom_products:
            goal_e = f"buy {prod} on {e_app}"
            p_st, p_en = find_token_span(goal_e, prod, TOKENIZER)

            # Screen D1: Search results displayed -> intermediate turn, complete is DISTRACTOR
            cands_ecom_res = [
                {
                    "label": f"{prod.title()} ₹25 ADD",
                    "role_idx": 8,  # ADD_TO_CART_BUTTON / PRODUCT_CARD
                    "is_editable": False,
                    "is_clickable": True,
                    "bounds_vec": [0.05, 0.20, 0.95, 0.35, 0.50, 0.27, 0.90, 0.15, 0.135, 25.0],
                },
                {
                    "label": "Search for items",
                    "role_idx": 3,
                    "is_editable": True,
                    "is_clickable": True,
                    "bounds_vec": [0.05, 0.04, 0.95, 0.10, 0.50, 0.07, 0.90, 0.06, 0.054, 15.0],
                },
                {
                    "label": "complete",  # Distractor! Task is NOT complete yet
                    "role_idx": 1,
                    "is_editable": False,
                    "is_clickable": True,
                    "bounds_vec": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                }
            ]

            for perm_idx in range(2):
                c_list = copy.deepcopy(cands_ecom_res)
                if perm_idx > 0:
                    random.shuffle(c_list)
                t_idx = next(i for i, c in enumerate(c_list) if "ADD" in c["label"])

                items.append(_normalize_item({
                    "name": f"ecom_intermediate_{e_app.lower()}_{prod.replace(' ', '_')}_p{perm_idx}",
                    "category": "evidence_gated_completion",
                    "dataset_class": "failure_class_d",
                    "goal": goal_e,
                    "candidates": c_list,
                    "target_idx": t_idx,
                    "action_idx": ACTION_TO_IDX["CLICK"],
                    "target_span": (0, 0),
                    "slot_type": 0,
                    "argument_value": "",
                    "stall_target": 0.0,
                    "evidence": 0.0,
                    "is_final": False,
                    "verification_evidence": 0.0,
                    "current_phase": "SELECT_PRODUCT",
                    "expected_state": "CART_UPDATED",
                }))

            # Screen D2: Cart badge updated with evidence -> verified terminal turn, complete is TARGET
            cands_ecom_verified = [
                {
                    "label": "complete",  # TARGET! Outcome verified
                    "role_idx": 1,
                    "is_editable": False,
                    "is_clickable": True,
                    "bounds_vec": [0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0],
                },
                {
                    "label": "Cart: 1 item ₹25",  # Verified observable cart evidence
                    "role_idx": 23,
                    "is_editable": False,
                    "is_clickable": True,
                    "bounds_vec": [0.10, 0.90, 0.90, 0.96, 0.50, 0.93, 0.80, 0.06, 0.048, 50.0],
                },
                {
                    "label": f"{prod.title()} Added (1)",
                    "role_idx": 8,
                    "is_editable": False,
                    "is_clickable": False,
                    "bounds_vec": [0.05, 0.20, 0.95, 0.35, 0.50, 0.27, 0.90, 0.15, 0.135, 25.0],
                }
            ]

            for perm_idx in range(2):
                c_list = copy.deepcopy(cands_ecom_verified)
                if perm_idx > 0:
                    random.shuffle(c_list)
                t_idx = next(i for i, c in enumerate(c_list) if c["label"] == "complete")

                items.append(_normalize_item({
                    "name": f"ecom_terminal_verified_{e_app.lower()}_{prod.replace(' ', '_')}_p{perm_idx}",
                    "category": "evidence_gated_completion",
                    "dataset_class": "failure_class_d",
                    "goal": goal_e,
                    "candidates": c_list,
                    "target_idx": t_idx,
                    "action_idx": ACTION_TO_IDX["WAIT"],
                    "target_span": (0, 0),
                    "slot_type": 0,
                    "argument_value": "",
                    "stall_target": 0.0,
                    "evidence": 1.0,
                    "is_final": True,
                    "verification_evidence": 1.0,
                    "current_phase": "VERIFY_CART",
                    "expected_state": "OUTCOME_VERIFIED",
                }))

    print(f">> Added Failure Class D (Evidence-Gated Calibration): Total dataset items now = {len(items)}")

    # =========================================================================
    # 5. FAILURE CLASS E: Absent Entity Search Fallback (Replay from Prior)
    # =========================================================================
    unlisted_contacts = ["Dr. Ramesh", "Rekha", "Uncle Bob", "Aarav Patel", "Fatima Zahra", "Deepak Rao"]
    for app in ["WhatsApp", "Telegram", "Signal"]:
        for name in unlisted_contacts:
            goal_e = f"open {app} and send hello to {name}"
            n_st, n_en = find_token_span(goal_e, name, TOKENIZER)

            cands_fallback = [
                {
                    "label": "Search",
                    "role_idx": 3,
                    "is_editable": True,
                    "is_clickable": True,
                    "bounds_vec": [0.80, 0.04, 0.90, 0.09, 0.85, 0.065, 0.10, 0.05, 0.005, 20.0],
                },
                {
                    "label": "Chats",
                    "role_idx": 14,
                    "is_editable": False,
                    "is_clickable": True,
                    "is_selected": True,
                    "bounds_vec": [0.05, 0.04, 0.30, 0.09, 0.17, 0.065, 0.25, 0.05, 0.012, 20.0],
                },
                {
                    "label": "Bhavani",
                    "role_idx": 5,
                    "is_editable": False,
                    "is_clickable": True,
                    "bounds_vec": [0.05, 0.15, 0.95, 0.25, 0.50, 0.20, 0.90, 0.10, 0.09, 30.0],
                },
                {
                    "label": "Manoj",
                    "role_idx": 5,
                    "is_editable": False,
                    "is_clickable": True,
                    "bounds_vec": [0.05, 0.26, 0.95, 0.36, 0.50, 0.31, 0.90, 0.10, 0.09, 30.0],
                },
                {
                    "label": "complete",
                    "role_idx": 1,
                    "is_editable": False,
                    "is_clickable": True,
                    "bounds_vec": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                },
            ]

            for perm_idx in range(2):
                c_list = copy.deepcopy(cands_fallback)
                if perm_idx > 0:
                    random.shuffle(c_list)
                t_idx = next(i for i, c in enumerate(c_list) if c["label"] == "Search")

                items.append(_normalize_item({
                    "name": f"fallback_search_{app.lower()}_{name.lower().replace(' ', '_')}_p{perm_idx}",
                    "category": "fallback_search",
                    "dataset_class": "failure_class_e",
                    "goal": goal_e,
                    "candidates": c_list,
                    "target_idx": t_idx,
                    "action_idx": ACTION_TO_IDX["CLICK"],
                    "target_span": (n_st, n_en),
                    "slot_type": 0,
                    "argument_value": name,
                    "stall_target": 0.0,
                    "evidence": 0.0,
                    "is_final": False,
                    "verification_evidence": 0.0,
                    "current_phase": "FIND_TARGET",
                    "expected_state": "SEARCH_OPENED",
                }))

    print(f">> Added Failure Class E (Absent Entity Search Fallback): Total dataset items now = {len(items)}")

    # =========================================================================
    # 6. FAILURE CLASS F: Multi-Turn Keypad & Form Sequencing (Samsung Calculator)
    # =========================================================================
    calc_sequences = [
        ("open Calculator and calculate 50 - 20", ["5", "0", "- Minus", "2", "0", "= Equal"]),
        ("calculate 25 + 37", ["2", "5", "+ Plus", "3", "7", "= Equal"]),
        ("calculate 100 / 4", ["1", "0", "0", "÷ Divide", "4", "= Equal"]),
        ("calculate 45 * 2", ["4", "5", "× Multiply", "2", "= Equal"]),
        ("open Calculator and calculate 80 - 15", ["8", "0", "- Minus", "1", "5", "= Equal"]),
    ]

    calc_keypad_pool = [
        ("0", 0.35, 0.85), ("1", 0.15, 0.72), ("2", 0.40, 0.72), ("3", 0.65, 0.72),
        ("4", 0.15, 0.59), ("5", 0.40, 0.59), ("6", 0.65, 0.59), ("7", 0.15, 0.46),
        ("8", 0.40, 0.46), ("9", 0.65, 0.46), ("+ Plus", 0.85, 0.72), ("- Minus", 0.85, 0.59),
        ("× Multiply", 0.85, 0.46), ("÷ Divide", 0.85, 0.33), ("= Equal", 0.85, 0.85),
        ("C", 0.15, 0.33), ("() Parentheses", 0.40, 0.33), ("% Percentage", 0.65, 0.33)
    ]

    for goal_calc, seq in calc_sequences:
        for step_idx, key in enumerate(seq):
            cands_calc = []
            for klbl, cx, cy in calc_keypad_pool:
                cands_calc.append({
                    "label": klbl,
                    "role_idx": 1,
                    "is_editable": False,
                    "is_clickable": True,
                    "bounds_vec": [cx - 0.10, cy - 0.05, cx + 0.10, cy + 0.05, cx, cy, 0.20, 0.10, 0.02, 35.0],
                })
            disp_text = ''.join(k.split()[0] for k in seq[:step_idx])
            cands_calc.append({
                "label": f"Calculator input field {disp_text}" if disp_text else "Calculator input field",
                "role_idx": 3,
                "is_editable": True,
                "is_clickable": False,
                "bounds_vec": [0.19, 0.83, 0.26, 0.96, 0.9, 0.22, 0.13, 0.07, 0.009, 1.8],
            })
            cands_calc.append({
                "label": "complete",
                "role_idx": 1,
                "is_editable": False,
                "is_clickable": True,
                "bounds_vec": [0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0],
            })

            target_idx = next(i for i, c in enumerate(cands_calc) if c["label"] == key)

            items.append(_normalize_item({
                "name": f"calc_seq_{goal_calc.split()[-3]}_{key.split()[0]}_step{step_idx}",
                "category": "calc_seq",
                "dataset_class": "failure_class_f",
                "goal": goal_calc,
                "candidates": cands_calc,
                "target_idx": target_idx,
                "action_idx": ACTION_TO_IDX["CLICK"],
                "target_span": (0, 0),
                "slot_type": 0,
                "argument_value": key,
                "stall_target": 0.0,
                "evidence": 0.0,
                "is_final": False,
                "verification_evidence": 0.0,
                "current_phase": "KEYPAD_SEQUENCE",
                "expected_state": "KEYPAD_UPDATED",
            }))

        # Final result display state: complete IS the target (is_final: True, evidence: 1.0)
        cands_final = []
        for klbl, cx, cy in calc_keypad_pool:
            cands_final.append({
                "label": klbl,
                "role_idx": 1,
                "is_editable": False,
                "is_clickable": True,
                "bounds_vec": [cx - 0.10, cy - 0.05, cx + 0.10, cy + 0.05, cx, cy, 0.20, 0.10, 0.02, 35.0],
            })
        cands_final.append({
            "label": "Calculator display 30",
            "role_idx": 0,
            "is_editable": False,
            "is_clickable": False,
            "bounds_vec": [0.1, 0.05, 0.25, 0.95, 0.5, 0.175, 0.9, 0.15, 0.135, 6.0],
        })
        cands_final.append({
            "label": "complete",
            "role_idx": 1,
            "is_editable": False,
            "is_clickable": True,
            "bounds_vec": [0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0],
        })
        target_final_idx = next(i for i, c in enumerate(cands_final) if c["label"] == "complete")

        items.append(_normalize_item({
            "name": f"calc_seq_{goal_calc.split()[-3]}_complete_final",
            "category": "calc_seq",
            "dataset_class": "failure_class_f",
            "goal": goal_calc,
            "candidates": cands_final,
            "target_idx": target_final_idx,
            "action_idx": ACTION_TO_IDX["WAIT"],
            "target_span": (0, 0),
            "slot_type": 0,
            "argument_value": "",
            "stall_target": 0.0,
            "evidence": 1.0,
            "is_final": True,
            "verification_evidence": 1.0,
            "current_phase": "VERIFY_RESULT",
            "expected_state": "OUTCOME_VERIFIED",
        }))

    print(f">> Added Failure Class F (Calculator Keypad Sequencing): Total dataset items now = {len(items)}")

    # =========================================================================
    # 7. Multimodal Delta NO-OP & Recovery Screen Perturbations
    # =========================================================================
    stall_samples = random.sample(items, min(len(items), 120))
    for it in stall_samples:
        stall_it = copy.deepcopy(it)
        stall_it["name"] += "_stall_perturbation"
        stall_it["stall_target"] = 1.0
        stall_it["delta_stall"] = True
        items.append(_normalize_item(stall_it))

    print(f">> Added Stall & NO-OP Perturbations: Final dataset pool size = {len(items)}")
    return items


if __name__ == "__main__":
    dataset = generate_v56_generalized_dataset()
    print("=" * 80)
    print(f"KARVISH V56 GENERALIZED DATASET SYNTHESIS COMPLETE: {len(dataset)} ITEMS")
    print("Sample Item 0 keys:", list(dataset[0].keys()))
    print("Sample Comm Item keys:", list(next(it for it in dataset if it.get("dataset_class") == "failure_class_a").keys()))
    print("=" * 80)
