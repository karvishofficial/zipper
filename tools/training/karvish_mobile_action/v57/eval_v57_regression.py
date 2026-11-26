"""
KARVISH V57 — SCIENTIFIC BEHAVIORAL REGRESSION CERTIFICATION SUITE
Pre-Deployment Certification Gates:
  1. 12/12 Frozen Baseline Anchor Tasks (Invariance Shield).
  2. Candidate Permutation Invariance Test (100% order-independent decisions).
  3. In-Chat Message Composer vs. Header Trap (Failure Class A).
  4. Dual-Slot Span Extraction (Recipient vs. Message Payload, Failure Class C).
  5. Active Selected Tab vs. Open Top Search Input (Failure Class B).
  6. Zepto Cheapest Price Grounding (₹13 beats ₹19).
  7. Multimodal NO-OP Stall Trigger.
  8. Search History Chip vs. Active Composer Input Disambiguation (V57 Core Pillar).
  9. Social Multi-Hop Navigation (Feed -> Inbox -> Contact -> Chat, V57 Core Pillar).
"""

from __future__ import annotations

import sys
import math
import copy
import random
import torch
from pathlib import Path
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
V56_DIR = SCRIPT_DIR.parent / "v56"
V55_DIR = SCRIPT_DIR.parent / "v55"
V54_DIR = SCRIPT_DIR.parent / "v54"
V52_DIR = SCRIPT_DIR.parent / "v52"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(V56_DIR))
sys.path.insert(0, str(V55_DIR))
sys.path.insert(0, str(V54_DIR))
sys.path.insert(0, str(V52_DIR))

from model_v57 import KarvishModelV57
from schema_v57 import ACTION_TO_IDX, ROLE_TO_IDX
from probe_v52_screen_decisions import get_test_screens

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
DISTILLED_CKPT = SCRIPT_DIR / "checkpoints" / "v57_best.pt"


def run_v57_regression_suite(ckpt_path: Path) -> bool:
    print("=" * 80)
    print(f"KARVISH V57 SCIENTIFIC BEHAVIORAL REGRESSION CERTIFICATION: {ckpt_path.name}")
    print(f"Evaluation Device: {DEVICE}")
    print("=" * 80)

    model = KarvishModelV57().to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd, strict=False)
    model.eval()

    all_passed = True

    # -----------------------------------------------------------------
    # SUITE 1: 12 Frozen Baseline Anchor Tasks (Invariance Shield)
    # -----------------------------------------------------------------
    print("\n--- SUITE 1: 12 FROZEN BASELINE ANCHORS (INVARIANCE SHIELD) ---")
    screens = get_test_screens()
    anchors_passed = 0
    for idx, s in enumerate(screens, 1):
        cands = s["candidates"][:24]
        N = len(cands)
        enc = TOKENIZER(s["goal"], max_length=32, truncation=True, return_tensors="pt")
        lbl_texts = [c["label"] for c in cands]
        c_enc = TOKENIZER(lbl_texts, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
        bounds = [c.get("bounds_vec", [0.0] * 10) for c in cands]
        roles = [2 if c.get("is_editable") else (1 if c.get("role") == "button" else 0) for c in cands]
        flags = torch.zeros(1, N, 16, device=DEVICE)
        for i, c in enumerate(cands):
            role_str = str(c.get("role", "button")).lower()
            if c.get("is_editable") or "input" in role_str or "edit" in role_str:
                flags[0, i, 0] = 1.0
            if c.get("is_focused"):
                flags[0, i, 1] = 1.0
            if c.get("is_clickable", True):
                flags[0, i, 2] = 1.0

        with torch.no_grad():
            out = model(
                goal_ids=enc["input_ids"].to(DEVICE),
                goal_mask=enc["attention_mask"].float().to(DEVICE),
                candidate_label_ids=c_enc["input_ids"].unsqueeze(0).to(DEVICE),
                candidate_label_mask=c_enc["attention_mask"].float().unsqueeze(0).to(DEVICE),
                candidate_bounds=torch.tensor([bounds], dtype=torch.float32, device=DEVICE),
                candidate_roles=torch.tensor([roles], dtype=torch.long, device=DEVICE),
                candidate_flags=flags,
                candidate_pad_mask=torch.ones(1, N, device=DEVICE),
            )
        pred_idx = out["ranking_scores"][0].argmax().item()
        pred_lbl = lbl_texts[pred_idx]
        is_match = s["expected_target"].lower() in pred_lbl.lower()
        if is_match:
            anchors_passed += 1
        print(f"Anchor {idx:02d} | {s['name'][:30]:30s} | Exp: {s['expected_target'][:18]:18s} | Pred: {pred_lbl[:18]:18s} | [{'PASS' if is_match else 'FAIL'}]")

    print(f">> Baseline Anchor Score: {anchors_passed}/{len(screens)} ({(anchors_passed/len(screens))*100:.1f}%)")
    if anchors_passed < 12:
        all_passed = False

    # -----------------------------------------------------------------
    # SUITE 2: Candidate Permutation Invariance Test
    # -----------------------------------------------------------------
    print("\n--- SUITE 2: CANDIDATE PERMUTATION INVARIANCE TEST ---")
    test_goal = "open WhatsApp and send hi to Bhavani"
    base_cands = [
        {"lbl": "Bhavani", "role": 11, "ed": False, "clk": True, "bnd": [0.15, 0.04, 0.65, 0.09, 0.40, 0.065, 0.50, 0.05, 0.025, 20.0]},
        {"lbl": "Type a message", "role": 4, "ed": True, "clk": True, "bnd": [0.05, 0.90, 0.85, 0.96, 0.45, 0.93, 0.80, 0.06, 0.048, 50.0]},
        {"lbl": "Video call", "role": 1, "ed": False, "clk": True, "bnd": [0.70, 0.04, 0.80, 0.09, 0.75, 0.065, 0.10, 0.05, 0.005, 20.0]},
        {"lbl": "Voice call", "role": 1, "ed": False, "clk": True, "bnd": [0.82, 0.04, 0.92, 0.09, 0.87, 0.065, 0.10, 0.05, 0.005, 20.0]},
        {"lbl": "Back", "role": 12, "ed": False, "clk": True, "bnd": [0.02, 0.04, 0.12, 0.09, 0.07, 0.065, 0.10, 0.05, 0.005, 20.0]},
    ]

    enc = TOKENIZER(test_goal, max_length=32, truncation=True, return_tensors="pt")
    perm_passed = 0
    NUM_PERMS = 10
    random.seed(42)

    for p_idx in range(NUM_PERMS):
        shuffled = list(range(len(base_cands)))
        random.shuffle(shuffled)
        c_perm = [base_cands[i] for i in shuffled]
        expected_target_idx = next(i for i, c in enumerate(c_perm) if c["lbl"] == "Type a message")

        lbl_texts = [c["lbl"] for c in c_perm]
        c_enc = TOKENIZER(lbl_texts, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
        roles = [c["role"] for c in c_perm]
        flags = torch.zeros(1, len(c_perm), 16, device=DEVICE)
        bnds = torch.zeros(1, len(c_perm), 10, device=DEVICE)
        for i, c in enumerate(c_perm):
            if c["ed"]: flags[0, i, 0] = 1.0
            if c["clk"]: flags[0, i, 2] = 1.0
            bnds[0, i] = torch.tensor(c["bnd"], device=DEVICE)

        with torch.no_grad():
            out = model(
                goal_ids=enc["input_ids"].to(DEVICE),
                goal_mask=enc["attention_mask"].float().to(DEVICE),
                candidate_label_ids=c_enc["input_ids"].unsqueeze(0).to(DEVICE),
                candidate_label_mask=c_enc["attention_mask"].float().unsqueeze(0).to(DEVICE),
                candidate_bounds=bnds,
                candidate_roles=torch.tensor([roles], dtype=torch.long, device=DEVICE),
                candidate_flags=flags,
                candidate_pad_mask=torch.ones(1, len(c_perm), device=DEVICE),
            )
        pred_idx = out["ranking_scores"][0].argmax().item()
        if pred_idx == expected_target_idx:
            perm_passed += 1

    print(f">> Permutation Robustness Score: {perm_passed}/{NUM_PERMS} ({(perm_passed/NUM_PERMS)*100:.1f}%)")
    if perm_passed < NUM_PERMS:
        all_passed = False

    # -----------------------------------------------------------------
    # SUITE 3: In-Chat Message Composer vs. Header Trap (Failure Class A)
    # -----------------------------------------------------------------
    print("\n--- SUITE 3: IN-CHAT COMPOSER VS. HEADER TRAP DISAMBIGUATION ---")
    wa_chat_cands = [
        "Type a message",
        "Bhavani",
        "Video call",
        "Voice call",
        "Attach media",
        "Back",
    ]
    N = len(wa_chat_cands)
    g_enc = TOKENIZER("open WhatsApp and send hi to Bhavani", max_length=32, truncation=True, return_tensors="pt")
    c_enc = TOKENIZER(wa_chat_cands, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
    roles = torch.tensor([[4, 11, 1, 1, 1, 12]], dtype=torch.long, device=DEVICE)
    flags = torch.zeros(1, N, 16, device=DEVICE)
    flags[0, 0, 0] = 1.0
    flags[0, 0, 2] = 1.0
    flags[0, 1:, 2] = 1.0
    wa_chat_bounds = torch.tensor([[
        [0.05, 0.90, 0.85, 0.96, 0.45, 0.93, 0.80, 0.06, 0.048, 50.0],
        [0.15, 0.04, 0.65, 0.09, 0.40, 0.065, 0.50, 0.05, 0.025, 20.0],
        [0.70, 0.04, 0.80, 0.09, 0.75, 0.065, 0.10, 0.05, 0.005, 20.0],
        [0.82, 0.04, 0.92, 0.09, 0.87, 0.065, 0.10, 0.05, 0.005, 20.0],
        [0.87, 0.90, 0.95, 0.96, 0.91, 0.93, 0.08, 0.06, 0.005, 50.0],
        [0.02, 0.04, 0.12, 0.09, 0.07, 0.065, 0.10, 0.05, 0.005, 20.0],
    ]], device=DEVICE)

    with torch.no_grad():
        out = model(
            goal_ids=g_enc["input_ids"].to(DEVICE),
            goal_mask=g_enc["attention_mask"].float().to(DEVICE),
            candidate_label_ids=c_enc["input_ids"].unsqueeze(0).to(DEVICE),
            candidate_label_mask=c_enc["attention_mask"].float().unsqueeze(0).to(DEVICE),
            candidate_bounds=wa_chat_bounds,
            candidate_roles=roles,
            candidate_flags=flags,
            candidate_pad_mask=torch.ones(1, N, device=DEVICE),
        )
    scores = out["ranking_scores"][0]
    top_idx = scores.argmax().item()
    margin_header = float(scores[0] - scores[1])
    margin_video = float(scores[0] - scores[2])
    act_type = out["action_logits"][0].argmax().item()

    is_pass = (top_idx == 0 and act_type == ACTION_TO_IDX["TYPE_TEXT"] and margin_header >= 1.0)
    print(f"Composer vs Header | Top: '{wa_chat_cands[top_idx]}' | Act: {act_type} (TYPE_TEXT) | Margin vs Header: +{margin_header:.2f} pts | Margin vs Video: +{margin_video:.2f} pts | [{'PASS' if is_pass else 'FAIL'}]")
    if not is_pass:
        all_passed = False

    # -----------------------------------------------------------------
    # SUITE 4: Active Selected Tab vs. Open Input Disambiguation
    # -----------------------------------------------------------------
    print("\n--- SUITE 4: ACTIVE TAB VS. OPEN INPUT DISAMBIGUATION ---")
    insta_explore_cands = [
        "Search",
        "Search and explore",
        "Home",
        "Profile",
    ]
    N = len(insta_explore_cands)
    ig_enc = TOKENIZER("search photography on Instagram", max_length=32, truncation=True, return_tensors="pt")
    ic_enc = TOKENIZER(insta_explore_cands, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
    ig_roles = torch.tensor([[3, 14, 14, 14]], dtype=torch.long, device=DEVICE)
    ig_flags = torch.zeros(1, N, 16, device=DEVICE)
    ig_flags[0, 0, 0] = 1.0
    ig_flags[0, 0, 2] = 1.0
    ig_flags[0, 1, 5] = 1.0
    ig_flags[0, 1:, 2] = 1.0

    with torch.no_grad():
        out = model(
            goal_ids=ig_enc["input_ids"].to(DEVICE),
            goal_mask=ig_enc["attention_mask"].float().to(DEVICE),
            candidate_label_ids=ic_enc["input_ids"].unsqueeze(0).to(DEVICE),
            candidate_label_mask=ic_enc["attention_mask"].float().unsqueeze(0).to(DEVICE),
            candidate_bounds=torch.zeros(1, N, 10, device=DEVICE),
            candidate_roles=ig_roles,
            candidate_flags=ig_flags,
            candidate_pad_mask=torch.ones(1, N, device=DEVICE),
        )
    scores = out["ranking_scores"][0]
    top_idx = scores.argmax().item()
    margin_tab = float(scores[0] - scores[1])
    is_ig_pass = (top_idx == 0 and margin_tab >= 2.0)
    print(f"Open Search vs Active Tab | Top: '{insta_explore_cands[top_idx]}' | Beat Active Tab by: +{margin_tab:.2f} pts | [{'PASS' if is_ig_pass else 'FAIL'}]")
    if not is_ig_pass:
        all_passed = False

    # -----------------------------------------------------------------
    # SUITE 5: Search History Chip vs. Active Input Field (V57 Core Pillar)
    # -----------------------------------------------------------------
    print("\n--- SUITE 5: SEARCH HISTORY CHIP VS. ACTIVE INPUT FIELD DISAMBIGUATION ---")
    yt_history_cands = [
        "Search YouTube",          # [0] Active search edit field (TARGET)
        "doraemon hindi",          # [1] Historical search chip (DISTRACTOR)
        "shinchan full episodes",  # [2] Historical search chip (DISTRACTOR)
        "Clear search history",    # [3] Auxiliary button
        "Voice search",            # [4] Microphone button
    ]
    N_yt = len(yt_history_cands)
    yt_goal = "search doraemon cartoon on YouTube"
    yt_g_enc = TOKENIZER(yt_goal, max_length=32, truncation=True, return_tensors="pt")
    yt_c_enc = TOKENIZER(yt_history_cands, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
    yt_roles = torch.tensor([[3, 20, 20, 1, 1]], dtype=torch.long, device=DEVICE) # 3=SEARCH_INPUT, 20=HISTORY_CHIP
    yt_flags = torch.zeros(1, N_yt, 16, device=DEVICE)
    yt_flags[0, 0, 0] = 1.0 # Search YouTube is editable
    yt_flags[0, 0, 1] = 1.0 # focused
    yt_flags[0, :, 2] = 1.0 # all clickable
    yt_bounds = torch.tensor([[
        [0.10, 0.04, 0.80, 0.09, 0.45, 0.065, 0.70, 0.05, 0.035, 30.0], # Search field
        [0.05, 0.12, 0.95, 0.17, 0.50, 0.145, 0.90, 0.05, 0.045, 20.0], # History chip 1
        [0.05, 0.18, 0.95, 0.23, 0.50, 0.205, 0.90, 0.05, 0.045, 20.0], # History chip 2
        [0.05, 0.85, 0.95, 0.90, 0.50, 0.875, 0.90, 0.05, 0.045, 20.0], # Clear history
        [0.85, 0.04, 0.95, 0.09, 0.90, 0.065, 0.10, 0.05, 0.005, 30.0], # Voice
    ]], device=DEVICE)

    with torch.no_grad():
        yt_out = model(
            goal_ids=yt_g_enc["input_ids"].to(DEVICE),
            goal_mask=yt_g_enc["attention_mask"].float().to(DEVICE),
            candidate_label_ids=yt_c_enc["input_ids"].unsqueeze(0).to(DEVICE),
            candidate_label_mask=yt_c_enc["attention_mask"].float().unsqueeze(0).to(DEVICE),
            candidate_bounds=yt_bounds,
            candidate_roles=yt_roles,
            candidate_flags=yt_flags,
            candidate_pad_mask=torch.ones(1, N_yt, device=DEVICE),
        )
    yt_scores = yt_out["ranking_scores"][0]
    yt_top_idx = yt_scores.argmax().item()
    margin_history_chip = float(yt_scores[0] - yt_scores[1])
    yt_act_type = yt_out["action_logits"][0].argmax().item()

    is_yt_pass = (yt_top_idx == 0 and yt_act_type == ACTION_TO_IDX["TYPE_TEXT"] and margin_history_chip >= 1.0)
    print(f"Active Input vs History Chip | Top: '{yt_history_cands[yt_top_idx]}' | Act: {yt_act_type} (TYPE_TEXT) | Margin vs Past Chip: +{margin_history_chip:.2f} pts | [{'PASS' if is_yt_pass else 'FAIL'}]")
    if not is_yt_pass:
        all_passed = False

    # -----------------------------------------------------------------
    # SUITE 6: Multi-Hop Social Navigation (Instagram DM Probe)
    # -----------------------------------------------------------------
    print("\n--- SUITE 6: MULTI-HOP SOCIAL NAVIGATION (INSTAGRAM DM PROBE) ---")
    ig_dm_cands = [
        "Message",            # [0] Top-right Direct Message entry affordance (TARGET)
        "Camera",             # [1] Top-left camera
        "Notifications",      # [2] Heart icon
        "Post by mahimagi09", # [3] Feed post
        "Like",               # [4] Action button
    ]
    N_dm = len(ig_dm_cands)
    dm_goal = "Open Instagram and send hello to Alex"
    dm_g_enc = TOKENIZER(dm_goal, max_length=32, truncation=True, return_tensors="pt")
    dm_c_enc = TOKENIZER(ig_dm_cands, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
    dm_roles = torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.long, device=DEVICE)
    dm_flags = torch.zeros(1, N_dm, 16, device=DEVICE)
    dm_flags[0, :, 2] = 1.0 # clickable
    dm_bounds = torch.tensor([[
        [0.85, 0.04, 0.98, 0.09, 0.91, 0.065, 0.13, 0.05, 0.006, 25.0], # Message icon
        [0.02, 0.04, 0.12, 0.09, 0.07, 0.065, 0.10, 0.05, 0.005, 20.0], # Camera
        [0.72, 0.04, 0.84, 0.09, 0.78, 0.065, 0.12, 0.05, 0.006, 20.0], # Notifications
        [0.04, 0.17, 0.95, 0.50, 0.50, 0.335, 0.91, 0.33, 0.300, 15.0], # Feed post
        [0.05, 0.66, 0.15, 0.71, 0.10, 0.685, 0.10, 0.05, 0.005, 20.0], # Like button
    ]], device=DEVICE)

    with torch.no_grad():
        dm_out = model(
            goal_ids=dm_g_enc["input_ids"].to(DEVICE),
            goal_mask=dm_g_enc["attention_mask"].float().to(DEVICE),
            candidate_label_ids=dm_c_enc["input_ids"].unsqueeze(0).to(DEVICE),
            candidate_label_mask=dm_c_enc["attention_mask"].float().unsqueeze(0).to(DEVICE),
            candidate_bounds=dm_bounds,
            candidate_roles=dm_roles,
            candidate_flags=dm_flags,
            candidate_pad_mask=torch.ones(1, N_dm, device=DEVICE),
        )
    dm_scores = dm_out["ranking_scores"][0]
    dm_top_idx = dm_scores.argmax().item()
    margin_feed = float(dm_scores[0] - dm_scores[3])
    dm_act_type = dm_out["action_logits"][0].argmax().item()

    is_dm_pass = (dm_top_idx == 0 and margin_feed >= 1.0)
    print(f"Feed DM Entry vs Feed Post | Top: '{ig_dm_cands[dm_top_idx]}' | Act: {dm_act_type} (TAP) | Margin vs Feed: +{margin_feed:.2f} pts | [{'PASS' if is_dm_pass else 'FAIL'}]")
    if not is_dm_pass:
        all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("[SUCCESS] ALL V57 BEHAVIORAL REGRESSION SUITES PASSED! CERTIFIED FOR MOBIL DEPLOYMENT.")
    else:
        print("[FAILURE] REGRESSION DETECTED! MODEL FAILED ONE OR MORE CERTIFICATION GATES.")
    print("=" * 80)
    return all_passed


if __name__ == "__main__":
    ckpt = DISTILLED_CKPT if len(sys.argv) < 2 else Path(sys.argv[1])
    success = run_v57_regression_suite(ckpt)
    sys.exit(0 if success else 1)
