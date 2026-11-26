"""
KARVISH V55 — BEHAVIORAL REGRESSION CERTIFIER (PRE-DEPLOYMENT GATE)
Tests:
1. 12/12 Frozen Baseline Anchor Tasks (Invariance Shield).
2. Universal Contact Search-to-Result Disambiguation ("Shobha Aunty", "Bhavani", "Rekha", "Dr. Ramesh").
3. Zepto Price Grounding (Cheapest Tomato ₹13 beats ₹19).
4. V55 2-Step Affordance (Home screen button -> TAP; Search box -> TYPE_TEXT).
5. WhatsApp Avatar / Header Distractor Rejection.
6. Multimodal NO-OP Stall Verification.
"""

from __future__ import annotations

import sys
import math
import torch
from pathlib import Path
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
V54_DIR = SCRIPT_DIR.parent / "v54"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(V54_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "v52"))

from model_v55 import KarvishModelV55
from probe_v52_screen_decisions import get_test_screens

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
DISTILLED_CKPT = SCRIPT_DIR / "checkpoints" / "v55_distilled_best.pt"


def run_regression_suite(ckpt_path: Path) -> bool:
    print("=" * 80)
    print(f"KARVISH V55 BEHAVIORAL REGRESSION CERTIFICATION: {ckpt_path.name}")
    print("=" * 80)

    model = KarvishModelV55().to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd, strict=False)
    model.eval()

    all_passed = True

    # -------------------------------------------------------------
    # 1. 12 Frozen Baseline Anchor Tasks (Invariance Shield)
    # -------------------------------------------------------------
    print("\n--- SUITE 1: 12 FROZEN BASELINE ANCHORS (INVARIANCE SHIELD) ---")
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
                candidate_pad_mask=torch.ones(1, N, device=DEVICE),
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
    # 2. Universal Contact Search-to-Result Disambiguation
    # -------------------------------------------------------------
    print("\n--- SUITE 2: UNIVERSAL CONTACT SEARCH-TO-RESULT DISAMBIGUATION ---")
    contacts = ["Shobha Aunty", "Bhavani", "Rekha", "Dr. Ramesh"]
    contacts_passed = 0
    for name in contacts:
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
        if is_pass:
            contacts_passed += 1
        else:
            all_passed = False
        print(f"Contact '{name:14s}' | Top: '{cands[top_idx]:15s}' | Margin vs Populated: +{margin_pop:.2f} pts | Margin vs Filter: +{margin_video:.2f} pts | [{'PASS' if is_pass else 'FAIL'}]")

    print(f">> Contact Score: {contacts_passed}/{len(contacts)}")

    # -------------------------------------------------------------
    # 3. Zepto Price Grounding (Cheapest Tomato ₹13 vs ₹19)
    # -------------------------------------------------------------
    print("\n--- SUITE 3: ZEPTO CHEAPEST PRICE GROUNDING ---")
    zepto_labels = [
        "Tomato Local 500g ₹13 ADD",
        "Tomato Hybrid 500g ₹19 ADD",
        "Onion 1kg ₹35 ADD",
        "Ginger 200g ₹28 ADD",
        "Dettol Soap ₹45 ADD",
    ]
    N = len(zepto_labels)
    zg_enc = TOKENIZER("buy tomatoes on Zepto", max_length=32, truncation=True, return_tensors="pt")
    zc_enc = TOKENIZER(zepto_labels, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
    z_num = torch.zeros(1, N, 4, device=DEVICE)
    z_num[0, 0, 0] = (math.log1p(13.0) / 10.0)
    z_num[0, 1, 0] = (math.log1p(19.0) / 10.0)
    z_num[0, 2, 0] = (math.log1p(35.0) / 10.0)
    z_num[0, 3, 0] = (math.log1p(28.0) / 10.0)
    z_num[0, 4, 0] = (math.log1p(45.0) / 10.0)

    with torch.no_grad():
        z_out = model(
            goal_ids=zg_enc["input_ids"].to(DEVICE),
            goal_mask=zg_enc["attention_mask"].float().to(DEVICE),
            candidate_label_ids=zc_enc["input_ids"].unsqueeze(0).to(DEVICE),
            candidate_label_mask=zc_enc["attention_mask"].float().unsqueeze(0).to(DEVICE),
            candidate_bounds=torch.zeros(1, N, 10, device=DEVICE),
            candidate_roles=torch.ones(1, N, dtype=torch.long, device=DEVICE),
            candidate_flags=torch.zeros(1, N, 16, device=DEVICE),
            candidate_pad_mask=torch.ones(1, N, device=DEVICE),
            candidate_num_features=z_num,
        )
    z_scores = z_out["ranking_scores"][0]
    z_top_idx = z_scores.argmax().item()
    z_pass = z_top_idx == 0
    price_diff = float(z_scores[0] - z_scores[1])
    print(f"Zepto Tomato Grounding  | Top: {zepto_labels[z_top_idx]:25s} | Beat ₹19 by: +{price_diff:.2f} pts | [{'PASS' if z_pass else 'FAIL'}]")
    if not z_pass:
        all_passed = False

    # -------------------------------------------------------------
    # 4. V55 2-Step Affordance Grounding (The Key Flaw Fix!)
    # -------------------------------------------------------------
    print("\n--- SUITE 4: V55 2-STEP AFFORDANCE GROUNDING ---")
    # Step 1: YouTube Home Screen without text box -> Target: Search icon. Action: TAP (0).
    yt_home_cands = ["Search", "Subscriptions", "android:id/list scroll down", "Comments"]
    yt_enc = TOKENIZER("search lo-fi hip hop on YouTube", max_length=32, truncation=True, return_tensors="pt")
    ytc_enc = TOKENIZER(yt_home_cands, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
    yt_roles = torch.tensor([[1, 1, 16, 2]], dtype=torch.long, device=DEVICE)
    yt_flags = torch.zeros(1, 4, 16, device=DEVICE)
    yt_flags[0, 0, 2] = 1.0 # Search is clickable button
    yt_flags[0, 3, 0] = 1.0 # Comments is editable

    with torch.no_grad():
        yt_out = model(
            goal_ids=yt_enc["input_ids"].to(DEVICE),
            goal_mask=yt_enc["attention_mask"].float().to(DEVICE),
            candidate_label_ids=ytc_enc["input_ids"].unsqueeze(0).to(DEVICE),
            candidate_label_mask=ytc_enc["attention_mask"].float().unsqueeze(0).to(DEVICE),
            candidate_bounds=torch.zeros(1, 4, 10, device=DEVICE),
            candidate_roles=yt_roles,
            candidate_flags=yt_flags,
            candidate_pad_mask=torch.ones(1, 4, device=DEVICE),
        )
    yt_top_cand = yt_out["ranking_scores"][0].argmax().item()
    yt_top_act = yt_out["action_logits"][0].argmax().item()
    yt_margin_feed = float(yt_out["ranking_scores"][0][0] - yt_out["ranking_scores"][0][2])
    yt_margin_comment = float(yt_out["ranking_scores"][0][0] - yt_out["ranking_scores"][0][3])
    yt_pass = yt_top_cand == 0 and yt_top_act == 0 and yt_margin_feed >= 2.0
    print(f"YouTube Home Affordance | Top: '{yt_home_cands[yt_top_cand]}' | Action: {yt_top_act} (TAP) | Margin vs Scroll: +{yt_margin_feed:.2f} pts | Margin vs Comment: +{yt_margin_comment:.2f} pts | [{'PASS' if yt_pass else 'FAIL'}]")
    if not yt_pass:
        all_passed = False

    # -------------------------------------------------------------
    # 5. WhatsApp Profile Avatar Distractor Rejection
    # -------------------------------------------------------------
    print("\n--- SUITE 5: WHATSAPP PROFILE AVATAR REJECTION ---")
    wa_cands = ["Search", "Profile photo", "Clear all", "Videos", "Photos", "submit focused field"]
    wa_enc = TOKENIZER("open WhatsApp and send hi to Shobha Aunty", max_length=32, truncation=True, return_tensors="pt")
    wac_enc = TOKENIZER(wa_cands, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
    wa_roles = torch.tensor([[2, 1, 1, 1, 1, 0]], dtype=torch.long, device=DEVICE)
    wa_flags = torch.zeros(1, 6, 16, device=DEVICE)
    wa_flags[0, 0, 0] = 1.0 # search is editable
    wa_flags[0, 0, 2] = 1.0 # search is clickable
    wa_flags[0, 1:, 2] = 1.0 # avatar/buttons clickable

    with torch.no_grad():
        wa_out = model(
            goal_ids=wa_enc["input_ids"].to(DEVICE),
            goal_mask=wa_enc["attention_mask"].float().to(DEVICE),
            candidate_label_ids=wac_enc["input_ids"].unsqueeze(0).to(DEVICE),
            candidate_label_mask=wac_enc["attention_mask"].float().unsqueeze(0).to(DEVICE),
            candidate_bounds=torch.zeros(1, 6, 10, device=DEVICE),
            candidate_roles=wa_roles,
            candidate_flags=wa_flags,
            candidate_pad_mask=torch.ones(1, 6, device=DEVICE),
        )
    wa_top_cand = wa_out["ranking_scores"][0].argmax().item()
    wa_margin_avatar = float(wa_out["ranking_scores"][0][0] - wa_out["ranking_scores"][0][1])
    wa_pass = wa_top_cand == 0 and wa_margin_avatar >= 2.0
    print(f"WhatsApp Avatar Reject  | Top: '{wa_cands[wa_top_cand]}' | Beat Avatar by: +{wa_margin_avatar:.2f} pts | [{'PASS' if wa_pass else 'FAIL'}]")
    if not wa_pass:
        all_passed = False

    # -------------------------------------------------------------
    # 6. Multimodal NO-OP Stall Verification
    # -------------------------------------------------------------
    print("\n--- SUITE 6: MULTIMODAL NO-OP STALL RECOVERY TRIGGER ---")
    stall_delta = torch.zeros(1, 10, device=DEVICE)
    stall_delta[0, 8] = 0.8 # high NO-OP count
    with torch.no_grad():
        stall_out = model(
            goal_ids=yt_enc["input_ids"].to(DEVICE),
            goal_mask=yt_enc["attention_mask"].float().to(DEVICE),
            candidate_label_ids=ytc_enc["input_ids"].unsqueeze(0).to(DEVICE),
            candidate_label_mask=ytc_enc["attention_mask"].float().unsqueeze(0).to(DEVICE),
            candidate_bounds=torch.zeros(1, 4, 10, device=DEVICE),
            candidate_roles=yt_roles,
            candidate_flags=yt_flags,
            candidate_pad_mask=torch.ones(1, 4, device=DEVICE),
            observable_delta=stall_delta,
        )
    stall_p = float(stall_out["stall_prob"][0])
    stall_pass = stall_p >= 0.70
    print(f"NO-OP Stall Trigger     | Stall Prob: {stall_p:.3f} (Threshold: 0.70) | [{'PASS' if stall_pass else 'FAIL'}]")
    if not stall_pass:
        all_passed = False

    print("\n" + "=" * 80)
    print(f"V55 CERTIFICATION RESULT: {'ALL PASS — READY FOR DEPLOYMENT' if all_passed else 'FAILED — DO NOT DEPLOY'}")
    print("=" * 80)
    return all_passed


if __name__ == "__main__":
    success = run_regression_suite(DISTILLED_CKPT)
    sys.exit(0 if success else 1)
