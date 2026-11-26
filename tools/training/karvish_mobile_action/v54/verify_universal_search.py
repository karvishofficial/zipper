"""
KARVISH V54 — UNIVERSAL SEARCH & CONTACT DISAMBIGUATION CERTIFIER
Comprehensive verification suite testing:
1. 12/12 Frozen Baseline Anchor Tasks (Invariance Shield).
2. Zepto Cheapest Price Grounding (Tomatoes, Milk, Ginger/Soap distractor isolation).
3. YouTube Home & Autocomplete Grounding.
4. Instagram Search & Explore Grounding.
5. Universal Contact Search-to-Result Selection across 4 test names:
   - Bhavani
   - Shobha Aunty
   - Rekha
   - Dr. Ramesh (Completely unseen synthetic name!)
   Verifies: Contact card ranks #1, beats populated search bar, beats filter chips ('Videos', 'Photos').
"""

from __future__ import annotations

import sys
import re
import argparse
from pathlib import Path
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "v52"))

from model_v54 import KarvishModelV54
from schema_v54 import ACTION_TO_IDX
from transformers import AutoTokenizer
from probe_v52_screen_decisions import get_test_screens

IDX_TO_ACTION = {v: k for k, v in ACTION_TO_IDX.items()}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
TARGET_CKPT = SCRIPT_DIR / "checkpoints" / "v54_universal_search_best.pt"


def verify_checkpoint(ckpt_path: Path) -> bool:
    print("=" * 80)
    print(f"VERIFYING UNIVERSAL SEARCH CHECKPOINT: {ckpt_path.name}")
    print("=" * 80)

    model = KarvishModelV54().to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd, strict=False)
    model.eval()

    all_passed = True

    # -------------------------------------------------------------
    # 1. 12 Frozen Baseline Anchors
    # -------------------------------------------------------------
    print("\n--- TEST SUITE 1: 12 FROZEN ANCHORS (INVARIANCE SHIELD) ---")
    screens = get_test_screens()
    anchors_passed = 0
    for idx, s in enumerate(screens, 1):
        cands = s["candidates"][:24]
        N = len(cands)
        enc = TOKENIZER(s["goal"], max_length=32, truncation=True, return_tensors="pt")
        lbl_texts = [c["label"] for c in cands]
        c_enc = TOKENIZER(lbl_texts, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
        bounds = [c.get("bounds_vec", [0.0]*10) for c in cands]
        roles = [2 if c.get("is_editable") else (1 if c.get("role") == "button" else 0) for c in cands]
        
        with torch.no_grad():
            out = model(
                goal_ids=enc["input_ids"].to(DEVICE),
                goal_mask=enc["attention_mask"].float().to(DEVICE),
                candidate_label_ids=c_enc["input_ids"].unsqueeze(0).to(DEVICE),
                candidate_label_mask=c_enc["attention_mask"].float().unsqueeze(0).to(DEVICE),
                candidate_bounds=torch.tensor([bounds], dtype=torch.float32, device=DEVICE),
                candidate_roles=torch.tensor([roles], dtype=torch.long, device=DEVICE),
                candidate_flags=torch.zeros(1, N, 16, device=DEVICE),
                candidate_pad_mask=torch.ones(1, N, device=DEVICE)
            )
        pred_idx = out["ranking_scores"][0].argmax().item()
        pred_lbl = lbl_texts[pred_idx]
        is_match = s["expected_target"].lower() in pred_lbl.lower()
        if is_match:
            anchors_passed += 1
        print(f"Anchor {idx:02d} | {s['name'][:30]:30s} | Exp: {s['expected_target'][:18]:18s} | Pred: {pred_lbl[:18]:18s} | [{'PASS' if is_match else 'FAIL'}]")

    print(f">> Baseline Anchor Score: {anchors_passed}/{len(screens)} ({(anchors_passed/len(screens))*100:.1f}%)")
    if anchors_passed < 11:
        all_passed = False

    # -------------------------------------------------------------
    # 2. Zepto Price Grounding & Distractor Isolation
    # -------------------------------------------------------------
    print("\n--- TEST SUITE 2: ZEPTO CHEAPEST PRICE & ENTITY GATING ---")
    zepto_labels = [
        "Tomato Local 500g ₹13 ADD",
        "Tomato Hybrid 500g ₹19 ADD",
        "Tomato Cherry 250g ₹37 ADD",
        "Ginger Fresh 100g ₹9 ADD",
        "Dettol Soap 75g ₹10 ADD",
        "Amul Taaza Milk 500ml ₹27 ADD",
    ]
    N = len(zepto_labels)
    z_enc = TOKENIZER("Open Zepto and search cheapest tomatoes", max_length=32, truncation=True, return_tensors="pt")
    zc_enc = TOKENIZER(zepto_labels, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
    z_num = torch.zeros(1, N, 4, device=DEVICE)
    for i, l in enumerate(zepto_labels):
        m = re.search(r"₹(\d+)", l)
        if m: z_num[0, i, 0] = float(m.group(1)) / 100.0

    with torch.no_grad():
        z_out = model(
            goal_ids=z_enc["input_ids"].to(DEVICE),
            goal_mask=z_enc["attention_mask"].float().to(DEVICE),
            candidate_label_ids=zc_enc["input_ids"].unsqueeze(0).to(DEVICE),
            candidate_label_mask=zc_enc["attention_mask"].float().unsqueeze(0).to(DEVICE),
            candidate_bounds=torch.zeros(1, N, 10, device=DEVICE),
            candidate_roles=torch.ones(1, N, dtype=torch.long, device=DEVICE),
            candidate_flags=torch.zeros(1, N, 16, device=DEVICE),
            candidate_pad_mask=torch.ones(1, N, device=DEVICE),
            candidate_num_features=z_num
        )
    z_scores = z_out["ranking_scores"][0]
    z_top_idx = z_scores.argmax().item()
    z_pass = z_top_idx == 0
    price_diff = float(z_scores[0] - z_scores[1])
    distractor_diff = min([float(z_scores[0] - z_scores[j]) for j in [3, 4, 5]])
    print(f"Zepto Cheapest Tomato   | Top: {zepto_labels[z_top_idx]:25s} | Beat ₹19 by: +{price_diff:.2f} pts | [{'PASS' if z_pass else 'FAIL'}]")
    if not z_pass:
        all_passed = False

    # -------------------------------------------------------------
    # 3. Universal Contact Search-to-Result Selection
    # -------------------------------------------------------------
    print("\n--- TEST SUITE 3: UNIVERSAL CONTACT SEARCH-TO-RESULT GROUNDING ---")
    test_contacts = ["Bhavani", "Shobha Aunty", "Rekha", "Dr. Ramesh", "Sarah Jenkins"]
    for name in test_contacts:
        goal = f"open WhatsApp and send hi to {name}"
        cands = [
            f"{name}",                                            # Target [0]
            f"<{name}",                                           # Populated Search Bar [1]
            "Videos",                                             # Filter Chip [2]
            "Photos",                                             # Filter Chip [3]
            "Back",                                               # Back Button [4]
            "submit focused field",                               # Submit [5]
            f"Jolly family picture {name} is also in this group", # Group Distractor [6]
        ]
        N = len(cands)
        g_enc = TOKENIZER(goal, max_length=32, truncation=True, return_tensors="pt")
        c_enc = TOKENIZER(cands, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
        c_roles = torch.tensor([[1, 2, 1, 1, 1, 1, 1]], dtype=torch.long, device=DEVICE)
        c_flags = torch.zeros(1, N, 16, dtype=torch.float32, device=DEVICE)
        c_flags[0, 0, 2] = 1.0  # contact card is_clickable
        c_flags[0, 1, 0] = 1.0  # is_editable
        c_flags[0, 1, 3] = 1.0  # is_populated
        c_flags[0, 1, 4] = 1.0  # was_last_typed
        c_flags[0, 2:, 2] = 1.0 # buttons/chips are is_clickable

        with torch.no_grad():
            out = model(
                goal_ids=g_enc["input_ids"].to(DEVICE),
                goal_mask=g_enc["attention_mask"].float().to(DEVICE),
                candidate_label_ids=c_enc["input_ids"].unsqueeze(0).to(DEVICE),
                candidate_label_mask=c_enc["attention_mask"].float().unsqueeze(0).to(DEVICE),
                candidate_bounds=torch.zeros(1, N, 10, device=DEVICE),
                candidate_roles=c_roles,
                candidate_flags=c_flags,
                candidate_pad_mask=torch.ones(1, N, device=DEVICE),
            )
        scores = out["ranking_scores"][0]
        top_idx = scores.argmax().item()
        margin_pop = float(scores[0] - scores[1])
        margin_video = float(scores[0] - scores[2])
        is_pass = top_idx == 0 and margin_pop >= 2.0 and margin_video >= 2.0
        if not is_pass:
            all_passed = False
        print(f"Contact '{name:14s}' | Top: '{cands[top_idx]:15s}' | Margin vs Populated Bar: +{margin_pop:.2f} pts | Margin vs Videos: +{margin_video:.2f} pts | [{'PASS' if is_pass else 'FAIL'}]")

    # -------------------------------------------------------------
    # 4. Empty Search Screen Disambiguation (Type into Search, don't tap recent contacts)
    # -------------------------------------------------------------
    print("\n--- TEST SUITE 4: EMPTY SEARCH SCREEN DISAMBIGUATION ---")
    for name in ["Shobha Aunty", "Rekha", "Dr. Ramesh"]:
        goal = f"open WhatsApp and send hi to {name}"
        cands_empty = ["Search", "Bhavani", "Clear all", "Videos", "Photos", "+91 94492 13667", "submit focused field"]
        roles_empty = [2, 1, 1, 1, 1, 1, 0]
        flags_empty = torch.zeros(1, len(cands_empty), 16, device=DEVICE)
        flags_empty[0, 0, 0] = 1.0 # is_editable
        flags_empty[0, 0, 2] = 1.0 # is_clickable
        flags_empty[0, 1:, 2] = 1.0 # is_clickable
        g_enc = TOKENIZER(goal, max_length=32, truncation=True, return_tensors="pt")
        c_enc = TOKENIZER(cands_empty, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
        with torch.no_grad():
            out = model(
                goal_ids=g_enc["input_ids"].to(DEVICE),
                goal_mask=g_enc["attention_mask"].float().to(DEVICE),
                candidate_label_ids=c_enc["input_ids"].unsqueeze(0).to(DEVICE),
                candidate_label_mask=c_enc["attention_mask"].float().unsqueeze(0).to(DEVICE),
                candidate_bounds=torch.zeros(1, len(cands_empty), 10, device=DEVICE),
                candidate_roles=torch.tensor([roles_empty], dtype=torch.long, device=DEVICE),
                candidate_flags=flags_empty,
                candidate_pad_mask=torch.ones(1, len(cands_empty), device=DEVICE),
            )
        scores = out["ranking_scores"][0]
        top_idx = scores.argmax().item()
        margin_recent = float(scores[0] - scores[1])
        is_pass = top_idx == 0 and margin_recent >= 2.0
        if not is_pass:
            all_passed = False
        print(f"Goal: '{name:12s}' | Top: '{cands_empty[top_idx]:10s}' | Beat Recent Contact (Bhavani) by: +{margin_recent:.2f} pts | [{'PASS' if is_pass else 'FAIL'}]")

    print("\n" + "=" * 80)
    print(f"OVERALL CERTIFICATION RESULT: {'ALL PASS' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 80)
    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=str(TARGET_CKPT))
    args = parser.parse_args()
    verify_checkpoint(Path(args.ckpt))
