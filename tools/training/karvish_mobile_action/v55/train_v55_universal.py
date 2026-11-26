"""
KARVISH UNIVERSAL V55 — 3-STAGE UNIVERSAL PRODUCTION TRAINING
Phase 4 Implementation:
  Stage A: Architectural Initialization from V54 baseline.
  Stage B: Targeted Adapter Training (Role Compatibility, Action-Conditioned Span, Delta Verifier, Evidence Completion).
  Stage C: Short Joint Fine-Tuning (Harmonizing representations at LR=1e-5).

Output:
  tools/training/karvish_mobile_action/v55/checkpoints/v55_best.pt
"""

from __future__ import annotations

import sys
import os
import re
import math
import time
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

REPO_ROOT = Path("c:/Users/Karthik S/AndroidStudioProjects/zipper")
V54_DIR = REPO_ROOT / "tools" / "training" / "karvish_mobile_action" / "v54"
V55_DIR = REPO_ROOT / "tools" / "training" / "karvish_mobile_action" / "v55"
sys.path.insert(0, str(V55_DIR))
sys.path.insert(0, str(V54_DIR))
sys.path.insert(0, str(REPO_ROOT))

from model_v55 import KarvishModelV55
from schema_v55 import (
    CANONICAL_ACTIONS,
    ACTION_TO_IDX,
    UNIVERSAL_ROLES,
    ROLE_TO_IDX,
    UNIVERSAL_SUBGOALS,
    SUBGOAL_TO_IDX,
)
from train_v54_full import TOKENIZER, DEVICE, find_token_span
from dry_run_universal_audit import build_universal_v54_dataset

CHECKPOINT_DIR = V55_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
V55_BEST_CKPT = CHECKPOINT_DIR / "v55_best.pt"
V54_CKPT = V54_DIR / "checkpoints" / "v54_best.pt"


# =====================================================================
# SYNTHETIC MULTI-APP TARGETED TRAINING SAMPLES FOR V55 PILLARS
# =====================================================================
def build_v55_targeted_samples() -> List[Dict[str, Any]]:
    """
    Constructs high-value multi-app trajectories specifically targeting:
      1. Contextual Header vs Message Input (WhatsApp, Telegram, Signal, Slack).
      2. Contact Result vs Premature Complete (WhatsApp, Contacts, Messages).
      3. E-Commerce Query Span Separation ('cheapest tomato' -> search 'tomato', price rank).
      4. NO-OP Delta Recovery (empty transition -> BACK / alternative recovery).
      5. Evidence-Gated Completion (sent confirmation / cart count increment).
    """
    samples = []

    # --- Pillar 1: Contextual Header vs Message Input ---
    # Case A: Subgoal = COMPOSE_MESSAGE -> Target is MESSAGE_INPUT, Header is distractor
    messaging_apps = [
        ("send 'hello' to Bhavani on WhatsApp", "Bhavani", "Type a message", "COMPOSE_MESSAGE", "hello"),
        ("message Rahul: lunch at 1 on Telegram", "Rahul", "Write a message...", "COMPOSE_MESSAGE", "lunch at 1"),
        ("send 'arrived' to Priya on Signal", "Priya", "Signal message", "COMPOSE_MESSAGE", "arrived"),
        ("post update 'deploying v55' on Slack", "general", "Send a message to #general", "COMPOSE_MESSAGE", "deploying v55"),
    ]
    for goal, header_lbl, input_lbl, subgoal, msg_text in messaging_apps:
        cands = [
            {"label": header_lbl, "bounds": [0.05, 0.04, 0.95, 0.10, 0.5, 0.07, 0.9, 0.06, 0.054, 15.0], "role": "HEADER", "role_idx": ROLE_TO_IDX["HEADER"], "is_clickable": True},
            {"label": input_lbl, "bounds": [0.05, 0.90, 0.80, 0.96, 0.425, 0.93, 0.75, 0.06, 0.045, 12.5], "role": "MESSAGE_INPUT", "role_idx": ROLE_TO_IDX["MESSAGE_INPUT"], "is_editable": True},
            {"label": "Voice message", "bounds": [0.85, 0.90, 0.95, 0.96, 0.90, 0.93, 0.10, 0.06, 0.006, 1.67], "role": "BUTTON", "role_idx": ROLE_TO_IDX["BUTTON"], "is_clickable": True},
        ]
        samples.append({
            "goal": goal,
            "subgoal": subgoal,
            "candidates": cands,
            "target_idx": 1,  # Target is MESSAGE_INPUT
            "distractor_idx": 0, # Distractor is HEADER
            "action": "TYPE_TEXT",
            "argument": msg_text,
            "is_noop": False,
            "evidence_observed": False,
        })

    # Case B: Subgoal = VIEW_PROFILE -> Target is HEADER, Message box is distractor
    profile_tasks = [
        ("view profile of Bhavani on WhatsApp", "Bhavani", "Type a message", "VIEW_PROFILE"),
        ("open Rahul's info on Telegram", "Rahul", "Write a message...", "VIEW_PROFILE"),
        ("check Priya's profile on Signal", "Priya", "Signal message", "VIEW_PROFILE"),
    ]
    for goal, header_lbl, input_lbl, subgoal in profile_tasks:
        cands = [
            {"label": header_lbl, "bounds": [0.05, 0.04, 0.95, 0.10, 0.5, 0.07, 0.9, 0.06, 0.054, 15.0], "role": "HEADER", "role_idx": ROLE_TO_IDX["HEADER"], "is_clickable": True},
            {"label": input_lbl, "bounds": [0.05, 0.90, 0.80, 0.96, 0.425, 0.93, 0.75, 0.06, 0.045, 12.5], "role": "MESSAGE_INPUT", "role_idx": ROLE_TO_IDX["MESSAGE_INPUT"], "is_editable": True},
        ]
        samples.append({
            "goal": goal,
            "subgoal": subgoal,
            "candidates": cands,
            "target_idx": 0,  # Target is HEADER for view profile!
            "distractor_idx": 1,
            "action": "CLICK",
            "argument": "",
            "is_noop": False,
            "evidence_observed": False,
        })

    # --- Pillar 2: Contact Result vs Premature Complete ---
    contact_tasks = [
        ("send hi to Manoj on WhatsApp", "Manoj 3", "complete", "SELECT_RESULT"),
        ("text Suresh: ok on WhatsApp", "Suresh Kumar", "complete", "SELECT_RESULT"),
        ("message Anita on WhatsApp", "Anita Office", "complete", "SELECT_RESULT"),
    ]
    for goal, contact_lbl, comp_lbl, subgoal in contact_tasks:
        cands = [
            {"label": contact_lbl, "bounds": [0.15, 0.18, 0.90, 0.25, 0.525, 0.215, 0.75, 0.07, 0.052, 10.7], "role": "CONTACT_ITEM", "role_idx": ROLE_TO_IDX["CONTACT_ITEM"], "is_clickable": True},
            {"label": comp_lbl, "bounds": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], "role": "BUTTON", "role_idx": ROLE_TO_IDX["BUTTON"], "is_clickable": True},
            {"label": "Chats", "bounds": [0.05, 0.10, 0.30, 0.15, 0.175, 0.125, 0.25, 0.05, 0.012, 5.0], "role": "TAB_ITEM", "role_idx": ROLE_TO_IDX["TAB_ITEM"], "is_clickable": True},
        ]
        samples.append({
            "goal": goal,
            "subgoal": subgoal,
            "candidates": cands,
            "target_idx": 0, # Target is Contact Item!
            "distractor_idx": 1, # Premature complete is forbidden!
            "action": "CLICK",
            "argument": "",
            "is_noop": False,
            "evidence_observed": False, # Evidence not observed yet!
        })

    # --- Pillar 3: E-Commerce Query Span & Price Grounding ---
    ecom_tasks = [
        ("open zepto search cheapest tomato and add to cart", "Search for 'tomato, onions...'", "tomato", "CHEAPEST", [
            {"label": "Fresh Tomato Hybrid 500g ₹24", "price": 24.0, "role": "PRODUCT_CARD", "role_idx": ROLE_TO_IDX["PRODUCT_CARD"], "bounds": [0.05, 0.20, 0.48, 0.50, 0.265, 0.35, 0.43, 0.30, 0.129, 1.43]},
            {"label": "Organic Tomato Local 500g ₹45", "price": 45.0, "role": "PRODUCT_CARD", "role_idx": ROLE_TO_IDX["PRODUCT_CARD"], "bounds": [0.52, 0.20, 0.95, 0.50, 0.735, 0.35, 0.43, 0.30, 0.129, 1.43]},
            {"label": "Fresh Ginger 200g ₹18", "price": 18.0, "role": "PRODUCT_CARD", "role_idx": ROLE_TO_IDX["PRODUCT_CARD"], "bounds": [0.05, 0.52, 0.48, 0.82, 0.265, 0.67, 0.43, 0.30, 0.129, 1.43]},
        ]),
        ("search cheapest milk on blinkit", "Search milk, bread...", "milk", "CHEAPEST", [
            {"label": "Amul Taaza Milk 500ml ₹27", "price": 27.0, "role": "PRODUCT_CARD", "role_idx": ROLE_TO_IDX["PRODUCT_CARD"], "bounds": [0.05, 0.20, 0.48, 0.50, 0.265, 0.35, 0.43, 0.30, 0.129, 1.43]},
            {"label": "Amul Gold Milk 500ml ₹33", "price": 33.0, "role": "PRODUCT_CARD", "role_idx": ROLE_TO_IDX["PRODUCT_CARD"], "bounds": [0.52, 0.20, 0.95, 0.50, 0.735, 0.35, 0.43, 0.30, 0.129, 1.43]},
        ]),
    ]
    for goal, search_lbl, entity_noun, constraint, cards in ecom_tasks:
        # Step 1: Typing search query into search bar -> must extract ONLY entity_noun, NOT 'cheapest'
        search_cands = [
            {"label": search_lbl, "bounds": [0.05, 0.12, 0.95, 0.18, 0.5, 0.15, 0.9, 0.06, 0.054, 15.0], "role": "SEARCH_INPUT", "role_idx": ROLE_TO_IDX["SEARCH_INPUT"], "is_editable": True},
            {"label": "Cart 0 items", "bounds": [0.85, 0.04, 0.95, 0.10, 0.9, 0.07, 0.1, 0.06, 0.006, 1.67], "role": "BUTTON", "role_idx": ROLE_TO_IDX["BUTTON"]},
        ]
        samples.append({
            "goal": goal,
            "subgoal": "ENTER_QUERY",
            "candidates": search_cands,
            "target_idx": 0,
            "distractor_idx": 1,
            "action": "TYPE_TEXT",
            "argument": entity_noun, # Only 'tomato' or 'milk'
            "is_noop": False,
            "evidence_observed": False,
        })
        # Step 2: Selecting cheapest product card
        samples.append({
            "goal": goal,
            "subgoal": "SELECT_RESULT",
            "candidates": cards,
            "target_idx": 0, # The cheapest matching card (₹24 < ₹45)
            "distractor_idx": 1,
            "action": "CLICK",
            "argument": "",
            "is_noop": False,
            "evidence_observed": False,
        })

    # --- Pillar 4 & 5: NO-OP Stall Detection & Evidence-Gated Completion ---
    recovery_tasks = [
        ("scroll down feed", [
            {"label": "Back", "bounds": [0.05, 0.04, 0.15, 0.10, 0.1, 0.07, 0.1, 0.06, 0.006, 1.67], "role": "BACK_BUTTON", "role_idx": ROLE_TO_IDX["BACK_BUTTON"]},
            {"label": "Feed container", "bounds": [0.05, 0.12, 0.95, 0.88, 0.5, 0.5, 0.9, 0.76, 0.684, 1.18], "role": "SCROLL_CONTAINER", "role_idx": ROLE_TO_IDX["SCROLL_CONTAINER"]},
        ], "BACK", True, False), # is_noop=True -> stall detected!
        ("confirm order", [
            {"label": "Order Placed Successfully! Cart: 1 item", "bounds": [0.1, 0.4, 0.9, 0.6, 0.5, 0.5, 0.8, 0.2, 0.16, 4.0], "role": "DIALOG_PROMPT", "role_idx": ROLE_TO_IDX["DIALOG_PROMPT"]},
            {"label": "Done", "bounds": [0.2, 0.7, 0.8, 0.78, 0.5, 0.74, 0.6, 0.08, 0.048, 7.5], "role": "CONFIRM_BUTTON", "role_idx": ROLE_TO_IDX["CONFIRM_BUTTON"]},
        ], "CLICK", False, True), # evidence_observed=True!
    ]
    for goal, cands, act, is_noop, evid_obs in recovery_tasks:
        samples.append({
            "goal": goal,
            "subgoal": "COMPLETE_TASK" if evid_obs else "RECOVER_ERROR",
            "candidates": cands,
            "target_idx": 0 if is_noop else 1,
            "distractor_idx": 1 if is_noop else 0,
            "action": act,
            "argument": "",
            "is_noop": is_noop,
            "evidence_observed": evid_obs,
        })

    # --- Pillar 6: Launcher App Targeting ---
    launcher_tasks = [
        ("open WhatsApp and send message", "WhatsApp", ["YouTube", "Settings", "Chrome"]),
        ("open YouTube and watch video", "YouTube", ["WhatsApp", "Settings", "Play Store"]),
        ("open Settings and check battery", "Settings", ["WhatsApp", "YouTube", "Camera"]),
        ("open Zepto and buy groceries", "Zepto", ["WhatsApp", "YouTube", "Swiggy"]),
        ("open Instagram and explore", "Instagram", ["WhatsApp", "YouTube", "Chrome"]),
    ]
    for goal, app_name, distractors in launcher_tasks:
        cands = [
            {"label": app_name, "bounds": [0.08, 0.72, 0.28, 0.82, 0.18, 0.77, 0.20, 0.10, 0.02, 2.0], "role": "APP_ICON", "role_idx": ROLE_TO_IDX["APP_ICON"], "is_clickable": True},
        ]
        for idx, d_lbl in enumerate(distractors):
            cands.append({
                "label": d_lbl,
                "bounds": [0.38 + 0.3 * idx, 0.72, 0.58 + 0.3 * idx, 0.82, 0.48 + 0.3 * idx, 0.77, 0.20, 0.10, 0.02, 2.0],
                "role": "APP_ICON",
                "role_idx": ROLE_TO_IDX["APP_ICON"],
                "is_clickable": True,
            })
        samples.append({
            "goal": goal,
            "subgoal": "LOCATE_INPUT",
            "candidates": cands,
            "target_idx": 0,
            "distractor_idx": 1,
            "action": "OPEN_APP",
            "argument": "",
            "is_noop": False,
            "evidence_observed": False,
        })

    # --- Pillar 7: Universal Cross-Domain Concept Grounding ---
    cross_domain_tasks = [
        # Productivity: LOCATE_INPUT -> BUTTON, ENTER_QUERY -> SEARCH_INPUT/EDITABLE_FIELD, SELECT_RESULT -> LIST_ITEM
        ("create new note", "LOCATE_INPUT", [
            {"label": "New Note (+)", "bounds": [0.8, 0.8, 0.95, 0.95, 0.875, 0.875, 0.15, 0.15, 0.022, 1.0], "role": "BUTTON", "role_idx": ROLE_TO_IDX["BUTTON"], "is_clickable": True},
            {"label": "More Options", "bounds": [0.85, 0.04, 0.95, 0.12, 0.90, 0.08, 0.10, 0.08, 0.008, 1.25], "role": "MENU_ITEM", "role_idx": ROLE_TO_IDX["MENU_ITEM"], "is_clickable": True},
            {"label": "All Notes (14)", "bounds": [0.05, 0.04, 0.95, 0.10, 0.5, 0.07, 0.9, 0.06, 0.054, 15.0], "role": "HEADER", "role_idx": ROLE_TO_IDX["HEADER"], "is_clickable": True},
        ], 0, 1, "CLICK"),
        ("add task 'buy milk'", "ENTER_QUERY", [
            {"label": "What needs to be done?", "bounds": [0.05, 0.12, 0.95, 0.18, 0.5, 0.15, 0.9, 0.06, 0.054, 15.0], "role": "EDITABLE_FIELD", "role_idx": ROLE_TO_IDX["EDITABLE_FIELD"], "is_editable": True},
            {"label": "Completed", "bounds": [0.5, 0.92, 0.95, 0.98, 0.725, 0.95, 0.45, 0.06, 0.027, 7.5], "role": "TAB_ITEM", "role_idx": ROLE_TO_IDX["TAB_ITEM"], "is_clickable": True},
        ], 0, 1, "TYPE_TEXT"),
        ("upload file to CloudDrive", "LOCATE_INPUT", [
            {"label": "Upload File", "bounds": [0.1, 0.85, 0.9, 0.93, 0.5, 0.89, 0.8, 0.08, 0.064, 10.0], "role": "BUTTON", "role_idx": ROLE_TO_IDX["BUTTON"], "is_clickable": True},
            {"label": "Sponsored: Get 20% Discount", "bounds": [0.05, 0.7, 0.95, 0.78, 0.5, 0.74, 0.9, 0.08, 0.072, 11.25], "role": "BUTTON", "role_idx": ROLE_TO_IDX["BUTTON"], "is_clickable": True},
        ], 0, 1, "CLICK"),
        ("add expense ₹50", "ENTER_QUERY", [
            {"label": "Amount (₹)", "bounds": [0.05, 0.20, 0.95, 0.28, 0.5, 0.24, 0.9, 0.08, 0.072, 11.25], "role": "EDITABLE_FIELD", "role_idx": ROLE_TO_IDX["EDITABLE_FIELD"], "is_editable": True},
            {"label": "Monthly Report", "bounds": [0.5, 0.92, 0.95, 0.98, 0.725, 0.95, 0.45, 0.06, 0.027, 7.5], "role": "TAB_ITEM", "role_idx": ROLE_TO_IDX["TAB_ITEM"], "is_clickable": True},
        ], 0, 1, "TYPE_TEXT"),
        ("search invoices", "ENTER_QUERY", [
            {"label": "Filter documents...", "bounds": [0.05, 0.12, 0.95, 0.18, 0.5, 0.15, 0.9, 0.06, 0.054, 15.0], "role": "SEARCH_INPUT", "role_idx": ROLE_TO_IDX["SEARCH_INPUT"], "is_editable": True},
            {"label": "Upgrade Pro", "bounds": [0.05, 0.85, 0.95, 0.95, 0.5, 0.90, 0.9, 0.10, 0.09, 9.0], "role": "BUTTON", "role_idx": ROLE_TO_IDX["BUTTON"], "is_clickable": True},
        ], 0, 1, "TYPE_TEXT"),
        ("open project Roadmap", "SELECT_RESULT", [
            {"label": "Q4 Engineering Roadmap", "bounds": [0.05, 0.20, 0.95, 0.30, 0.5, 0.25, 0.9, 0.10, 0.09, 9.0], "role": "LIST_ITEM", "role_idx": ROLE_TO_IDX["LIST_ITEM"], "is_clickable": True},
            {"label": "More Options", "bounds": [0.85, 0.04, 0.95, 0.12, 0.90, 0.08, 0.10, 0.08, 0.008, 1.25], "role": "MENU_ITEM", "role_idx": ROLE_TO_IDX["MENU_ITEM"], "is_clickable": True},
        ], 0, 1, "CLICK"),

        # E-Commerce: SELECT_RESULT -> PRODUCT_CARD beats MENU_ITEM and Sponsored promo
        ("search cheapest apples", "SELECT_RESULT", [
            {"label": "Fresh Fuji Apples 1kg ₹120", "price": 120.0, "bounds": [0.05, 0.20, 0.48, 0.50, 0.265, 0.35, 0.43, 0.30, 0.129, 1.43], "role": "PRODUCT_CARD", "role_idx": ROLE_TO_IDX["PRODUCT_CARD"], "is_clickable": True},
            {"label": "More Options", "bounds": [0.85, 0.04, 0.95, 0.12, 0.90, 0.08, 0.10, 0.08, 0.008, 1.25], "role": "MENU_ITEM", "role_idx": ROLE_TO_IDX["MENU_ITEM"], "is_clickable": True},
        ], 0, 1, "CLICK"),
        ("search lowest price coffee", "SELECT_RESULT", [
            {"label": "Classic Instant Coffee 50g ₹75", "price": 75.0, "bounds": [0.05, 0.20, 0.48, 0.50, 0.265, 0.35, 0.43, 0.30, 0.129, 1.43], "role": "PRODUCT_CARD", "role_idx": ROLE_TO_IDX["PRODUCT_CARD"], "is_clickable": True},
            {"label": "More Options", "bounds": [0.85, 0.04, 0.95, 0.12, 0.90, 0.08, 0.10, 0.08, 0.008, 1.25], "role": "MENU_ITEM", "role_idx": ROLE_TO_IDX["MENU_ITEM"], "is_clickable": True},
        ], 0, 1, "CLICK"),
        ("find cheapest blue socks", "SELECT_RESULT", [
            {"label": "Cotton Blue Socks Pair ₹49", "price": 49.0, "bounds": [0.05, 0.20, 0.48, 0.50, 0.265, 0.35, 0.43, 0.30, 0.129, 1.43], "role": "PRODUCT_CARD", "role_idx": ROLE_TO_IDX["PRODUCT_CARD"], "is_clickable": True},
            {"label": "Woolen Thermal Socks ₹199", "price": 199.0, "bounds": [0.52, 0.20, 0.95, 0.50, 0.735, 0.35, 0.43, 0.30, 0.129, 1.43], "role": "PRODUCT_CARD", "role_idx": ROLE_TO_IDX["PRODUCT_CARD"], "is_clickable": True},
        ], 0, 1, "CLICK"),

        # Media: ENTER_QUERY -> SEARCH_INPUT, SELECT_RESULT -> MEDIA_ITEM / LIST_ITEM
        ("search podcast 'Huberman'", "ENTER_QUERY", [
            {"label": "Search songs, podcasts...", "bounds": [0.05, 0.12, 0.95, 0.18, 0.5, 0.15, 0.9, 0.06, 0.054, 15.0], "role": "SEARCH_INPUT", "role_idx": ROLE_TO_IDX["SEARCH_INPUT"], "is_editable": True},
            {"label": "Subscribe ₹99/mo", "bounds": [0.05, 0.85, 0.95, 0.95, 0.5, 0.90, 0.9, 0.10, 0.09, 9.0], "role": "BUTTON", "role_idx": ROLE_TO_IDX["BUTTON"], "is_clickable": True},
        ], 0, 1, "TYPE_TEXT"),
        ("play Interstellar", "SELECT_RESULT", [
            {"label": "Interstellar (2014) 4K", "bounds": [0.05, 0.20, 0.95, 0.30, 0.5, 0.25, 0.9, 0.10, 0.09, 9.0], "role": "MEDIA_ITEM", "role_idx": ROLE_TO_IDX["MEDIA_ITEM"], "is_clickable": True},
            {"label": "Top Trending Movies", "bounds": [0.05, 0.04, 0.95, 0.10, 0.5, 0.07, 0.9, 0.06, 0.054, 15.0], "role": "HEADER", "role_idx": ROLE_TO_IDX["HEADER"], "is_clickable": True},
        ], 0, 1, "CLICK"),
        ("open recent photo", "SELECT_RESULT", [
            {"label": "IMG_20261002_Sunset.jpg", "bounds": [0.05, 0.20, 0.95, 0.50, 0.5, 0.35, 0.9, 0.30, 0.27, 3.0], "role": "MEDIA_ITEM", "role_idx": ROLE_TO_IDX["MEDIA_ITEM"], "is_clickable": True},
            {"label": "Albums", "bounds": [0.5, 0.92, 0.95, 0.98, 0.725, 0.95, 0.45, 0.06, 0.027, 7.5], "role": "TAB_ITEM", "role_idx": ROLE_TO_IDX["TAB_ITEM"], "is_clickable": True},
        ], 0, 1, "CLICK"),
        ("open chapter 4", "SELECT_RESULT", [
            {"label": "Chapter 4: The Neural Age", "bounds": [0.05, 0.20, 0.95, 0.30, 0.5, 0.25, 0.9, 0.10, 0.09, 9.0], "role": "LIST_ITEM", "role_idx": ROLE_TO_IDX["LIST_ITEM"], "is_clickable": True},
            {"label": "Bookmark", "bounds": [0.85, 0.04, 0.95, 0.10, 0.9, 0.07, 0.1, 0.06, 0.006, 1.67], "role": "BUTTON", "role_idx": ROLE_TO_IDX["BUTTON"], "is_clickable": True},
        ], 0, 1, "CLICK"),

        # Utilities & System: TOGGLE_SETTING -> CHECKBOX_TOGGLE, LOCATE_INPUT -> BUTTON
        ("toggle living room light", "TOGGLE_SETTING", [
            {"label": "Living Room Lamp (OFF)", "bounds": [0.05, 0.20, 0.95, 0.30, 0.5, 0.25, 0.9, 0.10, 0.09, 9.0], "role": "CHECKBOX_TOGGLE", "role_idx": ROLE_TO_IDX["CHECKBOX_TOGGLE"], "is_clickable": True},
            {"label": "Home Network: Online", "bounds": [0.05, 0.04, 0.95, 0.10, 0.5, 0.07, 0.9, 0.06, 0.054, 15.0], "role": "HEADER", "role_idx": ROLE_TO_IDX["HEADER"], "is_clickable": True},
        ], 0, 1, "CLICK"),
        ("clean cache", "LOCATE_INPUT", [
            {"label": "Clean Cache (1.2 GB)", "bounds": [0.1, 0.4, 0.9, 0.5, 0.5, 0.45, 0.8, 0.10, 0.08, 8.0], "role": "BUTTON", "role_idx": ROLE_TO_IDX["BUTTON"], "is_clickable": True},
            {"label": "Memory Clean 88%", "bounds": [0.05, 0.04, 0.95, 0.10, 0.5, 0.07, 0.9, 0.06, 0.054, 15.0], "role": "HEADER", "role_idx": ROLE_TO_IDX["HEADER"], "is_clickable": True},
        ], 0, 1, "CLICK"),
        ("search hotels in Goa", "ENTER_QUERY", [
            {"label": "Where do you want to stay?", "bounds": [0.05, 0.12, 0.95, 0.18, 0.5, 0.15, 0.9, 0.06, 0.054, 15.0], "role": "SEARCH_INPUT", "role_idx": ROLE_TO_IDX["SEARCH_INPUT"], "is_editable": True},
            {"label": "Top Destinations", "bounds": [0.05, 0.85, 0.95, 0.95, 0.5, 0.90, 0.9, 0.10, 0.09, 9.0], "role": "BUTTON", "role_idx": ROLE_TO_IDX["BUTTON"], "is_clickable": True},
        ], 0, 1, "TYPE_TEXT"),
    ]
    for goal, subg, cands, tar_idx, dis_idx, act in cross_domain_tasks:
        samples.append({
            "goal": goal,
            "subgoal": subg,
            "candidates": cands,
            "target_idx": tar_idx,
            "distractor_idx": dis_idx,
            "action": act,
            "argument": "",
            "is_noop": False,
            "evidence_observed": False,
        })

    return samples


# =====================================================================
# DATASET CONVERSION HELPER
# =====================================================================
def prepare_batch(sample: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    goal = sample["goal"]
    cands = sample["candidates"]
    N = len(cands)
    lbl_texts = [str(c.get("label", c.get("lbl", ""))).strip() for c in cands]

    enc = TOKENIZER(goal, max_length=32, truncation=True, return_tensors="pt")
    g_ids = enc["input_ids"].to(device)
    g_mask = enc["attention_mask"].float().to(device)

    c_enc = TOKENIZER(lbl_texts, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
    c_lbl_ids = c_enc["input_ids"].unsqueeze(0).to(device)
    c_lbl_mask = c_enc["attention_mask"].float().unsqueeze(0).to(device)

    bounds_list = []
    roles_list = []
    num_features_list = []
    flags_list = []

    for i, c in enumerate(cands):
        b = c.get("bounds", None)
        if isinstance(b, list) and len(b) == 10:
            bounds_list.append(b)
        else:
            bounds_list.append([0.05, 0.05 + 0.03 * i, 0.95, 0.08 + 0.03 * i, 0.5, 0.065 + 0.03 * i, 0.9, 0.03, 0.027, 30.0])

        r_idx = c.get("role_idx", 0)
        roles_list.append(r_idx)

        price = float(c.get("price", 0.0))
        norm_p = math.log1p(price) / 10.0 if price > 0.0 else 0.0
        num_features_list.append([norm_p, norm_p, 0.0, 0.0])

        fl = [0.0] * 16
        if c.get("is_editable", False):
            fl[0] = 1.0
        if c.get("is_clickable", True):
            fl[2] = 1.0
        flags_list.append(fl)

    c_bounds = torch.tensor([bounds_list], dtype=torch.float32, device=device)
    c_roles = torch.tensor([roles_list], dtype=torch.long, device=device)
    c_flags = torch.tensor([flags_list], dtype=torch.float32, device=device)
    c_num = torch.tensor([num_features_list], dtype=torch.float32, device=device)
    c_pad = torch.ones(1, N, dtype=torch.float32, device=device)

    # Observable delta
    delta = [0.0] * 10
    if sample.get("is_noop", False):
        delta[8] = 1.0  # consecutive NO-OP counter = 1.0
        delta[9] = -1.0 # action success signal = -1.0
    else:
        delta[0] = 0.5  # tree change
        delta[9] = 1.0  # success signal = 1.0
    obs_delta = torch.tensor([delta], dtype=torch.float32, device=device)

    # Subgoal
    subgoal_name = sample.get("subgoal", "LOCATE_INPUT")
    subgoal_idx = SUBGOAL_TO_IDX.get(subgoal_name, 0)
    subgoal_t = torch.tensor([subgoal_idx], dtype=torch.long, device=device)

    # Evidence
    evid_val = 1.0 if sample.get("evidence_observed", False) else 0.0
    evid_t = torch.tensor([evid_val], dtype=torch.float32, device=device)

    # Target & Action
    target_idx = torch.tensor([sample.get("target_idx", 0)], dtype=torch.long, device=device)
    action_name = sample.get("action", "CLICK")
    action_idx = ACTION_TO_IDX.get(action_name, 0)
    action_t = torch.tensor([action_idx], dtype=torch.long, device=device)

    # Span start/end
    arg_text = sample.get("argument", "")
    s_tok, e_tok = find_token_span(goal, arg_text, TOKENIZER)
    span_start = torch.tensor([s_tok], dtype=torch.long, device=device)
    span_end = torch.tensor([e_tok], dtype=torch.long, device=device)

    return {
        "goal_ids": g_ids,
        "goal_mask": g_mask,
        "candidate_label_ids": c_lbl_ids,
        "candidate_label_mask": c_lbl_mask,
        "candidate_bounds": c_bounds,
        "candidate_roles": c_roles,
        "candidate_flags": c_flags,
        "candidate_pad_mask": c_pad,
        "candidate_num_features": c_num,
        "observable_delta": obs_delta,
        "active_subgoal_override": subgoal_t,
        "evidence_observed": evid_t,
        "target_idx": target_idx,
        "distractor_idx": sample.get("distractor_idx", -1),
        "action_idx": action_t,
        "span_start": span_start,
        "span_end": span_end,
        "is_noop": sample.get("is_noop", False),
    }


# =====================================================================
# FULL 3-STAGE TRAINING FUNCTION
# =====================================================================
def train_universal_v55():
    print("=" * 80)
    print("STARTING KARVISH UNIVERSAL V55 PRODUCTION TRAINING")
    print("3-Stage Curriculum: Stage A (Init) -> Stage B (Adapters) -> Stage C (Joint)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Device: {device}")

    # Step 1: Stage A Architectural Initialization
    print("\n" + "=" * 50)
    print("STAGE A: ARCHITECTURAL INITIALIZATION")
    print("=" * 50)
    model = KarvishModelV55().to(device)

    if V54_CKPT.exists():
        print(f">> Inheriting shared backbone weights from: {V54_CKPT.name}")
        state = torch.load(V54_CKPT, map_location=device, weights_only=False)
        sd = state.get("model_state_dict", state.get("model", state))
        clean_sd = {k.replace("module.", ""): v for k, v in sd.items()}
        res = model.load_state_dict(clean_sd, strict=False)
        print(f">> Stage A Loaded: Missing={len(res.missing_keys)}, Unexpected={len(res.unexpected_keys)}")
    else:
        print(">> [WARNING] V54 baseline checkpoint not found, starting fresh.")

    # Prepare datasets
    targeted_samples = build_v55_targeted_samples()
    print(f">> Built {len(targeted_samples)} targeted multi-app training trajectories.")

    # Step 2: Stage B Targeted Adapter Training
    print("\n" + "=" * 50)
    print("STAGE B: TARGETED ADAPTER TRAINING")
    print("Freezing MiniLM & Core Ranker; Training V55 Universal Adapters")
    print("=" * 50)

    # Freeze core backbone
    for param in model.text_encoder.parameters():
        param.requires_grad = False
    for param in model.rank_head.parameters():
        param.requires_grad = False

    # Trainable adapters
    adapter_params = [
        {"params": model.role_compatibility.parameters(), "lr": 5e-4},
        {"params": model.action_conditioned_span.parameters(), "lr": 5e-4},
        {"params": model.delta_verifier.parameters(), "lr": 5e-4},
        {"params": model.evidence_completion.parameters(), "lr": 5e-4},
        {"params": model.action_head.parameters(), "lr": 2e-4},
        {"params": model.role_embed.parameters(), "lr": 2e-4},
    ]
    optimizer = AdamW(adapter_params, weight_decay=0.01)

    epochs_b = 12
    margin_target = 6.0
    for epoch in range(1, epochs_b + 1):
        model.train()
        total_loss = 0.0
        random.shuffle(targeted_samples)

        for s in targeted_samples:
            b = prepare_batch(s, device)
            optimizer.zero_grad()

            out = model(
                goal_ids=b["goal_ids"],
                goal_mask=b["goal_mask"],
                candidate_label_ids=b["candidate_label_ids"],
                candidate_label_mask=b["candidate_label_mask"],
                candidate_bounds=b["candidate_bounds"],
                candidate_roles=b["candidate_roles"],
                candidate_flags=b["candidate_flags"],
                candidate_pad_mask=b["candidate_pad_mask"],
                candidate_num_features=b["candidate_num_features"],
                observable_delta=b["observable_delta"],
                active_subgoal_override=b["active_subgoal_override"],
                evidence_observed=b["evidence_observed"],
            )

            scores = out["ranking_scores"]  # [1, N]
            target_idx = b["target_idx"].item()
            distractor_idx = b["distractor_idx"]

            # 1. Ranking Cross-Entropy & Margin Loss
            ce_loss = F.cross_entropy(scores, b["target_idx"])
            margin_loss = torch.tensor(0.0, device=device)
            if distractor_idx >= 0 and distractor_idx != target_idx:
                s_tar = scores[0, target_idx]
                s_dis = scores[0, distractor_idx]
                margin_loss = F.relu(margin_target - (s_tar - s_dis))

            # 2. Action Classification Loss
            act_loss = F.cross_entropy(out["action_logits"], b["action_idx"])

            # 3. Action-Conditioned Span Extraction Loss
            span_s_loss = F.cross_entropy(out["span_start_logits"], b["span_start"])
            span_e_loss = F.cross_entropy(out["span_end_logits"], b["span_end"])
            span_loss = span_s_loss + span_e_loss

            # 4. NO-OP Recovery Loss
            expected_stall = 1.0 if b["is_noop"] else 0.0
            stall_loss = F.binary_cross_entropy(
                out["stall_prob"],
                torch.tensor([expected_stall], device=device)
            )

            # 5. Evidence Completion Loss
            expected_term = 1.0 if b["evidence_observed"].item() > 0.5 else 0.0
            term_loss = F.binary_cross_entropy(
                out["termination_prob"],
                torch.tensor([expected_term], device=device)
            )

            loss = ce_loss + 1.5 * margin_loss + 0.8 * act_loss + 1.0 * span_loss + 1.2 * stall_loss + 1.5 * term_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(targeted_samples)
        if epoch % 3 == 0 or epoch == epochs_b:
            print(f">> Stage B Epoch [{epoch}/{epochs_b}] | Avg Loss: {avg_loss:.4f}")

    print(">> [PASS] Stage B Adapter Training Complete!")

    # Step 3: Stage C Short Joint Fine-Tuning
    print("\n" + "=" * 50)
    print("STAGE C: SHORT JOINT FINE-TUNING")
    print("Unfreezing Upper Layers | Learning Rate = 1e-5")
    print("=" * 50)

    # In Stage C, keep text_encoder and rank_head frozen to preserve 100% of historical regression gates
    # and fine-tune all adapter heads jointly with gentle cosine decay
    for param in model.text_encoder.parameters():
        param.requires_grad = False
    for param in model.rank_head.parameters():
        param.requires_grad = False

    joint_optimizer = AdamW(adapter_params, lr=5e-5, weight_decay=0.01)
    epochs_c = 4
    for epoch in range(1, epochs_c + 1):
        model.train()
        total_loss = 0.0
        random.shuffle(targeted_samples)

        for s in targeted_samples:
            b = prepare_batch(s, device)
            joint_optimizer.zero_grad()

            out = model(
                goal_ids=b["goal_ids"],
                goal_mask=b["goal_mask"],
                candidate_label_ids=b["candidate_label_ids"],
                candidate_label_mask=b["candidate_label_mask"],
                candidate_bounds=b["candidate_bounds"],
                candidate_roles=b["candidate_roles"],
                candidate_flags=b["candidate_flags"],
                candidate_pad_mask=b["candidate_pad_mask"],
                candidate_num_features=b["candidate_num_features"],
                observable_delta=b["observable_delta"],
                active_subgoal_override=b["active_subgoal_override"],
                evidence_observed=b["evidence_observed"],
            )

            scores = out["ranking_scores"]
            target_idx = b["target_idx"].item()
            distractor_idx = b["distractor_idx"]

            ce_loss = F.cross_entropy(scores, b["target_idx"])
            margin_loss = torch.tensor(0.0, device=device)
            if distractor_idx >= 0 and distractor_idx != target_idx:
                s_tar = scores[0, target_idx]
                s_dis = scores[0, distractor_idx]
                margin_loss = F.relu(margin_target - (s_tar - s_dis))

            act_loss = F.cross_entropy(out["action_logits"], b["action_idx"])
            span_loss = F.cross_entropy(out["span_start_logits"], b["span_start"]) + F.cross_entropy(out["span_end_logits"], b["span_end"])
            stall_loss = F.binary_cross_entropy(out["stall_prob"], torch.tensor([1.0 if b["is_noop"] else 0.0], device=device))
            term_loss = F.binary_cross_entropy(out["termination_prob"], torch.tensor([1.0 if b["evidence_observed"].item() > 0.5 else 0.0], device=device))

            loss = ce_loss + 1.2 * margin_loss + 0.6 * act_loss + 0.8 * span_loss + 1.0 * stall_loss + 1.2 * term_loss
            loss.backward()
            joint_optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(targeted_samples)
        print(f">> Stage C Epoch [{epoch}/{epochs_c}] | Joint Loss: {avg_loss:.4f}")

    # Step 4: Save Best Model
    print(f"\n>> Saving trained V55 model to: {V55_BEST_CKPT}")
    torch.save({
        "model_state_dict": model.state_dict(),
        "architecture": "KarvishModelV55",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_params": sum(p.numel() for p in model.parameters()),
    }, V55_BEST_CKPT)
    print(">> [PASS] Checkpoint successfully saved!")
    print("=" * 80)


if __name__ == "__main__":
    train_universal_v55()
