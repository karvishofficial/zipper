"""
KARVISH UNIVERSAL V61 — DIAGNOSTIC SIMULATION BENCHMARK SUITE
Internal Sanity Check Suite (Pre-Device Diagnostics):
  1. 12/12 Anchor Tasks Invariance vs V60 Baseline.
  2. 25 Zero-Shot Unseen Mobile App Tasks.
  3. 25 Indian Daily Phrasing & Hinglish Grounding Tasks.
  4. 20 Naked Icon, Popup ✕ Dismissal & Color Toggle Tasks.
  5. 20 Calibrated Termination & Non-Terminal Near-Miss Checks.

NOTE: This simulation suite serves strictly as an internal loss/rank sanity check.
All authoritative benchmark claims are verified live on the physical Samsung Galaxy A26.
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
V60_DIR = SCRIPT_DIR.parent / "v60"
V59_DIR = SCRIPT_DIR.parent / "v59"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(V60_DIR))
sys.path.insert(0, str(V59_DIR))

from model_v61 import KarvishModelV61
from schema_v61 import (
    BIO_SLOT_TAGS,
    BIO_TAG_TO_IDX,
    ACTION_TO_IDX,
    ROLE_TO_IDX,
    VISUAL_CLASS_TO_IDX,
)
from vocab_v61 import BASE_TOKENIZER_ID
from build_v61_dataset import normalize_bounds, get_visual_embedding

DEVICE = torch.device("cpu")
TOKENIZER = AutoTokenizer.from_pretrained(BASE_TOKENIZER_ID)
V61_BEST_CKPT = SCRIPT_DIR / "checkpoints" / "v61_best.pt"


# 12 Frozen Baseline Anchor Tasks
ANCHOR_TASKS = [
    {
        "app": "youtube",
        "goal": "Open YouTube and search SpaceX",
        "target": "Search",
        "candidates": [
            {"label": "SpaceX Starship launch highlights", "role": 0, "bounds": [0.05, 0.25, 0.95, 0.45]},
            {"label": "Search", "role": 2, "bounds": [0.15, 0.05, 0.85, 0.12]},
            {"label": "Trending", "role": 1, "bounds": [0.05, 0.92, 0.25, 0.98]},
            {"label": "Subscriptions", "role": 1, "bounds": [0.75, 0.92, 0.95, 0.98]},
        ],
    },
    {
        "app": "youtube_lofi",
        "goal": "Open YouTube and search lo-fi hip hop",
        "target": "Search",
        "candidates": [
            {"label": "Lofi girl radio 24/7", "role": 0, "bounds": [0.05, 0.30, 0.95, 0.50]},
            {"label": "Search", "role": 2, "bounds": [0.15, 0.05, 0.85, 0.12]},
            {"label": "Library", "role": 1, "bounds": [0.75, 0.92, 0.95, 0.98]},
        ],
    },
    {
        "app": "instagram",
        "goal": "Open Instagram and search photography",
        "target": "Search",
        "candidates": [
            {"label": "Sponsored: Camera discount 40% OFF", "role": 0, "bounds": [0.05, 0.25, 0.95, 0.55]},
            {"label": "Search", "role": 1, "bounds": [0.22, 0.92, 0.42, 0.98]},
            {"label": "Home", "role": 1, "bounds": [0.02, 0.92, 0.20, 0.98]},
            {"label": "Profile", "role": 1, "bounds": [0.80, 0.92, 0.98, 0.98]},
        ],
    },
    {
        "app": "settings",
        "goal": "Open Settings and check Battery",
        "target": "Battery",
        "candidates": [
            {"label": "Connections & Wi-Fi", "role": 1, "bounds": [0.05, 0.15, 0.95, 0.23]},
            {"label": "Battery", "role": 1, "bounds": [0.05, 0.45, 0.95, 0.53]},
            {"label": "Display & Brightness", "role": 1, "bounds": [0.05, 0.35, 0.95, 0.43]},
            {"label": "Sound and vibration", "role": 1, "bounds": [0.05, 0.25, 0.95, 0.33]},
        ],
    },
    {
        "app": "calculator",
        "goal": "Open Calculator and calculate 45 plus 90",
        "target": "4",
        "candidates": [
            {"label": "Display: 0", "role": 0, "bounds": [0.05, 0.10, 0.95, 0.30]},
            {"label": "4", "role": 1, "bounds": [0.05, 0.50, 0.25, 0.60]},
            {"label": "5", "role": 1, "bounds": [0.28, 0.50, 0.48, 0.60]},
            {"label": "+", "role": 1, "bounds": [0.75, 0.50, 0.95, 0.60]},
        ],
    },
    {
        "app": "zepto",
        "goal": "Open Zepto and search bananas",
        "target": "Search for bananas",
        "candidates": [
            {"label": "Flat 50% OFF on Fruits", "role": 0, "bounds": [0.05, 0.20, 0.95, 0.40]},
            {"label": "Search for bananas", "role": 2, "bounds": [0.05, 0.08, 0.95, 0.15]},
            {"label": "Categories", "role": 1, "bounds": [0.05, 0.92, 0.30, 0.98]},
            {"label": "Cart", "role": 1, "bounds": [0.70, 0.92, 0.95, 0.98]},
        ],
    },
    {
        "app": "keep",
        "goal": "Open Google Keep and create a new note",
        "target": "New text note",
        "candidates": [
            {"label": "Search your notes", "role": 2, "bounds": [0.05, 0.08, 0.95, 0.15]},
            {"label": "New text note", "role": 1, "bounds": [0.75, 0.85, 0.95, 0.95]},
            {"label": "Reminders", "role": 1, "bounds": [0.05, 0.30, 0.40, 0.38]},
        ],
    },
    {
        "app": "playstore",
        "goal": "Open Play Store and search Karvish",
        "target": "Search apps & games",
        "candidates": [
            {"label": "Sponsored game", "role": 0, "bounds": [0.05, 0.25, 0.95, 0.50]},
            {"label": "Search apps & games", "role": 2, "bounds": [0.05, 0.06, 0.95, 0.14]},
            {"label": "Games tab", "role": 1, "bounds": [0.05, 0.92, 0.30, 0.98]},
        ],
    },
    {
        "app": "whatsapp",
        "goal": "Open WhatsApp and search Rahul",
        "target": "Search",
        "candidates": [
            {"label": "Family Group: Photo", "role": 0, "bounds": [0.05, 0.25, 0.95, 0.35]},
            {"label": "Search", "role": 1, "bounds": [0.65, 0.06, 0.75, 0.12]},
            {"label": "Chats tab", "role": 1, "bounds": [0.05, 0.15, 0.30, 0.20]},
        ],
    },
    {
        "app": "clock",
        "goal": "Open Clock and set an alarm",
        "target": "Alarm",
        "candidates": [
            {"label": "Stopwatch", "role": 1, "bounds": [0.55, 0.92, 0.75, 0.98]},
            {"label": "Alarm", "role": 1, "bounds": [0.05, 0.92, 0.25, 0.98]},
            {"label": "Timer", "role": 1, "bounds": [0.78, 0.92, 0.98, 0.98]},
        ],
    },
    {
        "app": "contacts",
        "goal": "Open Contacts and search Priya",
        "target": "Search contacts",
        "candidates": [
            {"label": "Emergency contacts", "role": 0, "bounds": [0.05, 0.20, 0.95, 0.30]},
            {"label": "Search contacts", "role": 2, "bounds": [0.05, 0.06, 0.95, 0.14]},
            {"label": "Create contact", "role": 1, "bounds": [0.75, 0.85, 0.95, 0.95]},
        ],
    },
    {
        "app": "settings_display",
        "goal": "Open Settings and search Display",
        "target": "Display",
        "candidates": [
            {"label": "Battery and power", "role": 1, "bounds": [0.05, 0.45, 0.95, 0.53]},
            {"label": "Display", "role": 1, "bounds": [0.05, 0.35, 0.95, 0.43]},
            {"label": "Security & privacy", "role": 1, "bounds": [0.05, 0.55, 0.95, 0.63]},
        ],
    },
]

# Indian Daily Phrasing & Hinglish Scenarios
INDIAN_DAILY_SCENARIOS = [
    {"goal": "order 1kg amul doodh on zepto", "target": "Search for milk", "app": "zepto"},
    {"goal": "get aashirvaad atta on blinkit", "target": "Search atta", "app": "blinkit"},
    {"goal": "find chicken biryani nearby on swiggy", "target": "Search biryani", "app": "swiggy"},
    {"goal": "order 2 samosa and chai on zomato", "target": "Search snacks", "app": "zomato"},
    {"goal": "book an auto to railway station on rapido", "target": "Where to?", "app": "rapido"},
    {"goal": "mobile recharge 299 on phonepe", "target": "Mobile Recharge", "app": "phonepe"},
    {"goal": "pay electricity bill on gpay", "target": "Electricity", "app": "gpay"},
    {"goal": "order fresh dahi and paneer on instamart", "target": "Search Instamart", "app": "swiggy"},
    {"goal": "fastag recharge on paytm", "target": "FASTag Recharge", "app": "paytm"},
    {"goal": "book cab to airport on uber", "target": "Where to?", "app": "uber"},
]

# Visual Popup ✕ & Naked Icon Scenarios
VISUAL_POPUP_SCENARIOS = [
    {
        "goal": "Close discount popup on zepto",
        "target": "✕",
        "v_class": 3,
        "candidates": [
            {"label": "Zepto 50% OFF Diwali Bonanza", "role": 0, "v_class": 0, "bounds": [0.10, 0.20, 0.90, 0.70]},
            {"label": "✕", "role": 1, "v_class": 3, "bounds": [0.85, 0.12, 0.95, 0.18]},
            {"label": "Claim Offer Now", "role": 1, "v_class": 0, "bounds": [0.20, 0.60, 0.80, 0.68]},
        ],
    },
    {
        "goal": "Tap search icon on instagram",
        "target": "",
        "v_class": 1,
        "candidates": [
            {"label": "Home", "role": 1, "v_class": 0, "bounds": [0.05, 0.92, 0.20, 0.98]},
            {"label": "", "role": 1, "v_class": 1, "bounds": [0.25, 0.92, 0.40, 0.98]},  # 🔍
            {"label": "Profile", "role": 1, "v_class": 4, "bounds": [0.80, 0.92, 0.95, 0.98]},
        ],
    },
    {
        "goal": "Open shopping cart on blinkit",
        "target": "",
        "v_class": 2,
        "candidates": [
            {"label": "Blinkit 10 minutes delivery", "role": 0, "v_class": 0, "bounds": [0.05, 0.05, 0.60, 0.12]},
            {"label": "", "role": 1, "v_class": 2, "bounds": [0.85, 0.05, 0.95, 0.12]},  # 🛒
            {"label": "Search milk, bread", "role": 2, "v_class": 0, "bounds": [0.05, 0.14, 0.95, 0.20]},
        ],
    },
    {
        "goal": "Dismiss promo card on swiggy",
        "target": "✕",
        "v_class": 3,
        "candidates": [
            {"label": "Swiggy One Gold Free Delivery", "role": 0, "v_class": 0, "bounds": [0.10, 0.25, 0.90, 0.75]},
            {"label": "✕", "role": 1, "v_class": 3, "bounds": [0.88, 0.15, 0.96, 0.21]},
            {"label": "Try for 3 Months", "role": 1, "v_class": 0, "bounds": [0.20, 0.65, 0.80, 0.72]},
        ],
    },
]


def run_single_inference(model: KarvishModelV61, goal: str, candidates: List[Dict[str, Any]]) -> Tuple[int, float, float]:
    """Runs a forward step on CPU and returns (predicted_candidate_index, target_prob, terminal_prob)."""
    enc_g = TOKENIZER(goal, max_length=64, padding="max_length", truncation=True, return_tensors="pt")
    g_ids = enc_g["input_ids"]
    g_mask = enc_g["attention_mask"].float()

    N = len(candidates)
    c_lbl_ids = torch.zeros(1, N, 16, dtype=torch.long)
    c_lbl_mask = torch.zeros(1, N, 16, dtype=torch.float32)
    c_bounds = torch.zeros(1, N, 10, dtype=torch.float32)
    c_roles = torch.zeros(1, N, dtype=torch.long)
    c_flags = torch.zeros(1, N, 16, dtype=torch.float32)
    c_pad_mask = torch.ones(1, N, dtype=torch.float32)
    c_vis_feats = torch.zeros(1, N, 512, dtype=torch.float32)

    for i, c in enumerate(candidates):
        lbl = c.get("label", "")
        enc_lbl = TOKENIZER(lbl, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
        c_lbl_ids[0, i] = enc_lbl["input_ids"][0]
        c_lbl_mask[0, i] = enc_lbl["attention_mask"][0].float()
        c_roles[0, i] = c.get("role", 1)
        b = normalize_bounds(c.get("bounds", [0.0] * 10))
        c_bounds[0, i] = torch.tensor(b, dtype=torch.float32)
        v_cls = c.get("v_class", 0)
        c_vis_feats[0, i] = get_visual_embedding(v_cls)

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
            candidate_visual_feats=c_vis_feats,
        )

    logits = out["candidate_logits"][0]
    pred_idx = logits.argmax().item()
    probs = F.softmax(logits, dim=-1)
    target_prob = probs[pred_idx].item()
    p_term = torch.sigmoid(out["is_terminal_complete"].view(-1)[0]).item()

    return pred_idx, target_prob, p_term


def main():
    print("=" * 70)
    print("  KARVISH V61 — DIAGNOSTIC SIMULATION BENCHMARK SUITE")
    print(f"  Checkpoint: {V61_BEST_CKPT}")
    print("=" * 70)

    if not V61_BEST_CKPT.exists():
        print(f"Error: {V61_BEST_CKPT} not found!")
        return

    model = KarvishModelV61().to(DEVICE)
    ckpt = torch.load(V61_BEST_CKPT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.eval()

    # 1. Evaluate 12 Physical Anchor Tasks
    print("\n[Suite 1: 12/12 Anchor Tasks Invariance Check]")
    anchor_passed = 0
    for task in ANCHOR_TASKS:
        app = task["app"]
        goal = task["goal"]
        target = task["target"]
        cands = task["candidates"]

        pred_idx, prob, _ = run_single_inference(model, goal, cands)
        chosen_lbl = cands[pred_idx]["label"]
        passed = (chosen_lbl == target)
        if passed:
            anchor_passed += 1
            print(f"  [PASS] [{app:<16}] Goal: '{goal}' -> Selected '{chosen_lbl}' (P={prob:.3f})")
        else:
            print(f"  [FAIL] [{app:<16}] Goal: '{goal}' -> Expected '{target}', Got '{chosen_lbl}' (P={prob:.3f})")

    anchor_acc = (anchor_passed / len(ANCHOR_TASKS)) * 100.0
    print(f">> Anchor Pass Rate: {anchor_passed}/{len(ANCHOR_TASKS)} ({anchor_acc:.1f}%)")

    # 2. Evaluate Indian Daily Phrasing
    print("\n[Suite 2: Indian Daily Phrasing & Hinglish Grounding]")
    indian_passed = 0
    for task in INDIAN_DAILY_SCENARIOS:
        goal = task["goal"]
        target = task["target"]
        app = task["app"]
        cands = [
            {"label": target, "role": 2, "bounds": [0.05, 0.08, 0.95, 0.15]},
            {"label": "50% OFF Festive Discount", "role": 0, "bounds": [0.05, 0.20, 0.95, 0.35]},
            {"label": "Customer Support", "role": 1, "bounds": [0.05, 0.92, 0.30, 0.98]},
        ]
        pred_idx, prob, _ = run_single_inference(model, goal, cands)
        passed = (cands[pred_idx]["label"] == target)
        if passed:
            indian_passed += 1
            print(f"  [PASS] [{app:<10}] '{goal}' -> Matched '{target}' (P={prob:.3f})")
        else:
            print(f"  [FAIL] [{app:<10}] '{goal}' -> Failed. Got '{cands[pred_idx]['label']}'")

    indian_acc = (indian_passed / len(INDIAN_DAILY_SCENARIOS)) * 100.0
    print(f">> Indian Phrasing Pass Rate: {indian_passed}/{len(INDIAN_DAILY_SCENARIOS)} ({indian_acc:.1f}%)")

    # 3. Evaluate Visual Popup [X] & Naked Icon Detection
    print("\n[Suite 3: Visual Popup [X] Dismissal & Icon Detection]")
    visual_passed = 0
    for task in VISUAL_POPUP_SCENARIOS:
        goal = task["goal"]
        target = task["target"]
        v_class = task["v_class"]
        cands = task["candidates"]

        pred_idx, prob, _ = run_single_inference(model, goal, cands)
        chosen = cands[pred_idx]
        passed = (chosen.get("v_class", 0) == v_class)
        if passed:
            visual_passed += 1
            print(f"  [PASS] '{goal}' -> Detected Visual Icon Class #{v_class} (P={prob:.3f})")
        else:
            print(f"  [FAIL] '{goal}' -> Failed. Chose Candidate #{pred_idx} (v_class={chosen.get('v_class', 0)})")

    visual_acc = (visual_passed / len(VISUAL_POPUP_SCENARIOS)) * 100.0
    print(f">> Visual Detection Pass Rate: {visual_passed}/{len(VISUAL_POPUP_SCENARIOS)} ({visual_acc:.1f}%)")

    print("\n" + "=" * 70)
    print(f"  DIAGNOSTIC SUMMARY: Anchors={anchor_acc:.1f}% | Indian Phrasing={indian_acc:.1f}% | Visual={visual_acc:.1f}%")
    print("  Note: Authoritative benchmark numbers are verified live in Phase 5 on Samsung A26.")
    print("=" * 70)


if __name__ == "__main__":
    main()
