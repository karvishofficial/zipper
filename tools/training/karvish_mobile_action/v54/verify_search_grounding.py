"""
KARVISH V54 — SEARCH GROUNDING DUAL VERIFICATION PROBE
Evaluates candidate checkpoint against:
1. 12/12 Frozen Baseline Anchor Tasks (Invariance Shield).
2. Zepto Cheapest Tomato Price Comparison.
3. YouTube Home Screen (Search Icon vs Feed Video Distractors).
4. YouTube Autocomplete Screen (Submit vs 'Edit suggestion' Trap).
5. Instagram Home Screen ('Search and explore' vs Feed Posts).
6. Instagram Explore Screen (Input field targeting with TYPE_TEXT).
"""

from __future__ import annotations

import sys
import re
from pathlib import Path
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "v52"))

from model_v54 import KarvishModelV54
from schema_v54 import ACTION_TO_IDX
from transformers import AutoTokenizer
from probe_v52_screen_decisions import get_test_screens

IDX_TO_ACTION = {v: k for k, v in ACTION_TO_IDX.items()}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

SOURCE_CKPT = SCRIPT_DIR / "checkpoints" / "v54_best.pt"
SMOKE_CKPT = SCRIPT_DIR / "checkpoints" / "v54_search_smoke.pt"


def verify_checkpoint(ckpt_path: Path):
    print("=" * 80)
    print(f"VERIFYING SEARCH GROUNDING CHECKPOINT: {ckpt_path.name}")
    print("=" * 80)

    model = KarvishModelV54().to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd, strict=False)
    model.eval()

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
        print(f"Anchor {idx:02d} | {s['name'][:30]:30s} | Exp: {s['expected_target'][:20]:20s} | Pred: {pred_lbl[:20]:20s} | [{'PASS' if is_match else 'FAIL'}]")

    print(f">> Baseline Anchor Score: {anchors_passed}/{len(screens)} ({(anchors_passed/len(screens))*100:.1f}%)")

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
    print(f"                        | Distractor Isolation (Ginger ₹9, Soap ₹10): +{distractor_diff:.2f} pts")

    # -------------------------------------------------------------
    # 3. YouTube Home Screen (Search vs Feed Cards)
    # -------------------------------------------------------------
    print("\n--- TEST SUITE 3: YOUTUBE SEARCH GROUNDING ---")
    yt_home_cands = [
        "Search",
        "Voice search",
        "Go to channel Sajjaad Khader",
        "I Asked Microsoft’s CEO If I Should Learn to Code",
        "Starship Fifth Flight Test SpaceX, verified",
        "Notifications"
    ]
    N = len(yt_home_cands)
    y_enc = TOKENIZER("Open YouTube and search for SpaceX", max_length=32, truncation=True, return_tensors="pt")
    yc_enc = TOKENIZER(yt_home_cands, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        y_out = model(
            goal_ids=y_enc["input_ids"].to(DEVICE),
            goal_mask=y_enc["attention_mask"].float().to(DEVICE),
            candidate_label_ids=yc_enc["input_ids"].unsqueeze(0).to(DEVICE),
            candidate_label_mask=yc_enc["attention_mask"].float().unsqueeze(0).to(DEVICE),
            candidate_bounds=torch.zeros(1, N, 10, device=DEVICE),
            candidate_roles=torch.ones(1, N, dtype=torch.long, device=DEVICE),
            candidate_flags=torch.zeros(1, N, 16, device=DEVICE),
            candidate_pad_mask=torch.ones(1, N, device=DEVICE),
        )
    y_scores = y_out["ranking_scores"][0]
    y_top_idx = y_scores.argmax().item()
    y_pass = y_top_idx == 0
    feed_margin = float(y_scores[0] - max(y_scores[2], y_scores[3], y_scores[4]))
    print(f"YouTube Home Screen     | Top: {yt_home_cands[y_top_idx]:25s} | Margin vs Feed Cards: +{feed_margin:.2f} pts | [{'PASS' if y_pass else 'FAIL'}]")

    # YouTube Autocomplete Screen
    yt_auto_cands = [
        "submit focused field",
        "Edit suggestion spacex",
        "Edit suggestion spacex falcon 9",
        "Edit suggestion spacex launch live",
        "spacex official live stream",
        "Search YouTube"
    ]
    N = len(yt_auto_cands)
    ya_c_enc = TOKENIZER(yt_auto_cands, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
    # Realistic roles: submit is button(1), suggestions are button(1), input field is input(2)
    ya_roles = torch.tensor([[1, 1, 1, 1, 1, 2]], dtype=torch.long, device=DEVICE)
    ya_bounds = torch.zeros(1, N, 10, device=DEVICE)
    ya_bounds[0, 0] = torch.tensor([0.9, 0.8, 0.98, 0.98, 0.89, 0.94, 0.18, 0.08, 0.014, 2.25], device=DEVICE)
    ya_flags = torch.zeros(1, N, 16, device=DEVICE)
    ya_flags[0, 5, 0] = 1.0  # Search YouTube is editable
    ya_flags[0, 5, 3] = 1.0  # Search YouTube is populated
    with torch.no_grad():
        ya_out = model(
            goal_ids=y_enc["input_ids"].to(DEVICE),
            goal_mask=y_enc["attention_mask"].float().to(DEVICE),
            candidate_label_ids=ya_c_enc["input_ids"].unsqueeze(0).to(DEVICE),
            candidate_label_mask=ya_c_enc["attention_mask"].float().unsqueeze(0).to(DEVICE),
            candidate_bounds=ya_bounds,
            candidate_roles=ya_roles,
            candidate_flags=ya_flags,
            candidate_pad_mask=torch.ones(1, N, device=DEVICE),
        )
    ya_scores = ya_out["ranking_scores"][0]
    ya_top_idx = ya_scores.argmax().item()
    ya_pass = ya_top_idx == 0
    edit_sugg_margin = float(ya_scores[0] - max(ya_scores[1], ya_scores[2], ya_scores[3]))
    print(f"YouTube Autocomplete    | Top: {yt_auto_cands[ya_top_idx]:25s} | Margin vs Edit Arrow: +{edit_sugg_margin:.2f} pts | [{'PASS' if ya_pass else 'FAIL'}]")

    # YouTube Suggestion Row Screen: Tapping the query suggestion item
    yt_sugg_cands = [
        "lo-fi hip hop",
        "Edit suggestion lo-fi hip hop",
        "Edit suggestion lo-fi hip hop live",
        "Clear",
        "Voice search",
        "Search YouTube"
    ]
    N = len(yt_sugg_cands)
    ys_enc = TOKENIZER("Open YouTube and search lo-fi hip hop", max_length=32, truncation=True, return_tensors="pt")
    ysc_enc = TOKENIZER(yt_sugg_cands, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
    ys_roles = torch.tensor([[1, 1, 1, 1, 1, 2]], dtype=torch.long, device=DEVICE)
    ys_flags = torch.zeros(1, N, 16, device=DEVICE)
    ys_flags[0, 5, 0] = 1.0  # Search YouTube is editable
    ys_flags[0, 5, 3] = 1.0  # Search YouTube is populated
    with torch.no_grad():
        ys_out = model(
            goal_ids=ys_enc["input_ids"].to(DEVICE),
            goal_mask=ys_enc["attention_mask"].float().to(DEVICE),
            candidate_label_ids=ysc_enc["input_ids"].unsqueeze(0).to(DEVICE),
            candidate_label_mask=ysc_enc["attention_mask"].float().unsqueeze(0).to(DEVICE),
            candidate_bounds=torch.zeros(1, N, 10, device=DEVICE),
            candidate_roles=ys_roles,
            candidate_flags=ys_flags,
            candidate_pad_mask=torch.ones(1, N, device=DEVICE),
        )
    ys_scores = ys_out["ranking_scores"][0]
    ys_top_idx = ys_scores.argmax().item()
    ys_pass = ys_top_idx == 0
    sugg_margin = float(ys_scores[0] - max(ys_scores[1], ys_scores[2], ys_scores[3]))
    print(f"YouTube Suggestion Row  | Top: {yt_sugg_cands[ys_top_idx]:25s} | Margin vs Edit Arrow: +{sugg_margin:.2f} pts | [{'PASS' if ys_pass else 'FAIL'}]")

    # YouTube Search Results Screen (Termination Milestone)
    yt_done_cands = [
        "complete",
        "lo-fi hip hop beats to relax/study to - Lofi Girl (5.4 crore views)",
        "Home",
        "Shorts",
        "Subscriptions",
        "You",
        "scroll down"
    ]
    N = len(yt_done_cands)
    yd_c_enc = TOKENIZER(yt_done_cands, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
    yd_roles = torch.tensor([[0, 1, 1, 1, 1, 1, 0]], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        yd_out = model(
            goal_ids=ys_enc["input_ids"].to(DEVICE),
            goal_mask=ys_enc["attention_mask"].float().to(DEVICE),
            candidate_label_ids=yd_c_enc["input_ids"].unsqueeze(0).to(DEVICE),
            candidate_label_mask=yd_c_enc["attention_mask"].float().unsqueeze(0).to(DEVICE),
            candidate_bounds=torch.zeros(1, N, 10, device=DEVICE),
            candidate_roles=yd_roles,
            candidate_flags=torch.zeros(1, N, 16, device=DEVICE),
            candidate_pad_mask=torch.ones(1, N, device=DEVICE),
        )
    yd_scores = yd_out["ranking_scores"][0]
    yd_top_idx = yd_scores.argmax().item()
    yd_pass = yd_top_idx == 0
    yd_margin = float(yd_scores[0] - max(yd_scores[1], yd_scores[2], yd_scores[3], yd_scores[4], yd_scores[5]))
    print(f"YouTube Search Complete | Top: {yt_done_cands[yd_top_idx]:25s} | Margin vs Video & You: +{yd_margin:.2f} pts | [{'PASS' if yd_pass else 'FAIL'}]")

    # -------------------------------------------------------------
    # 4. Instagram Grounding (Search and Explore vs Feed & Complete)
    # -------------------------------------------------------------
    print("\n--- TEST SUITE 4: INSTAGRAM SEARCH GROUNDING ---")
    ig_cands = [
        "Search and explore",
        "Home",
        "Reels",
        "Photo 1 of 5 by RVCJ Media, 527 likes",
        "sindhu_pramuu's story, 2 of 26",
        "Like",
        "Comment"
    ]
    N = len(ig_cands)
    i_enc = TOKENIZER("Open Instagram and search photography", max_length=32, truncation=True, return_tensors="pt")
    ic_enc = TOKENIZER(ig_cands, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        i_out = model(
            goal_ids=i_enc["input_ids"].to(DEVICE),
            goal_mask=i_enc["attention_mask"].float().to(DEVICE),
            candidate_label_ids=ic_enc["input_ids"].unsqueeze(0).to(DEVICE),
            candidate_label_mask=ic_enc["attention_mask"].float().unsqueeze(0).to(DEVICE),
            candidate_bounds=torch.zeros(1, N, 10, device=DEVICE),
            candidate_roles=torch.ones(1, N, dtype=torch.long, device=DEVICE),
            candidate_flags=torch.zeros(1, N, 16, device=DEVICE),
            candidate_pad_mask=torch.ones(1, N, device=DEVICE),
        )
    i_scores = i_out["ranking_scores"][0]
    i_top_idx = i_scores.argmax().item()
    i_pass = i_top_idx == 0
    ig_feed_margin = float(i_scores[0] - max(i_scores[3], i_scores[4]))
    print(f"Instagram Home Screen   | Top: {ig_cands[i_top_idx]:25s} | Margin vs Feed Posts: +{ig_feed_margin:.2f} pts | [{'PASS' if i_pass else 'FAIL'}]")

    # Instagram Search Results Screen (Termination Milestone)
    ig_done_cands = [
        "complete",
        "#photography • 1.2M posts",
        "photography_daily • Follow",
        "Home",
        "Reels",
        "Profile"
    ]
    N = len(ig_done_cands)
    id_c_enc = TOKENIZER(ig_done_cands, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
    id_roles = torch.tensor([[0, 1, 1, 1, 1, 1]], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        id_out = model(
            goal_ids=i_enc["input_ids"].to(DEVICE),
            goal_mask=i_enc["attention_mask"].float().to(DEVICE),
            candidate_label_ids=id_c_enc["input_ids"].unsqueeze(0).to(DEVICE),
            candidate_label_mask=id_c_enc["attention_mask"].float().unsqueeze(0).to(DEVICE),
            candidate_bounds=torch.zeros(1, N, 10, device=DEVICE),
            candidate_roles=id_roles,
            candidate_flags=torch.zeros(1, N, 16, device=DEVICE),
            candidate_pad_mask=torch.ones(1, N, device=DEVICE),
        )
    id_scores = id_out["ranking_scores"][0]
    id_top_idx = id_scores.argmax().item()
    id_pass = id_top_idx == 0
    id_margin = float(id_scores[0] - max(id_scores[1], id_scores[2], id_scores[3], id_scores[4], id_scores[5]))
    print(f"Instagram Search Done   | Top: {ig_done_cands[id_top_idx]:25s} | Margin vs Result & Nav: +{id_margin:.2f} pts | [{'PASS' if id_pass else 'FAIL'}]")

    # -------------------------------------------------------------
    # 5. Zepto Search Grounding (Matching Chip vs Ice Distractor & Complete)
    # -------------------------------------------------------------
    print("\n--- TEST SUITE 5: ZEPTO SEARCH GROUNDING ---")
    zepto_search_cands = [
        "fresh tomatoes",
        "Search",
        "Tomato puree",
        "Ice",
        "Munchies",
        "complete"
    ]
    N = len(zepto_search_cands)
    zs_enc = TOKENIZER("Open Zepto and search cheapest tomatoes", max_length=32, truncation=True, return_tensors="pt")
    zsc_enc = TOKENIZER(zepto_search_cands, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
    zs_roles = torch.tensor([[1, 2, 1, 1, 1, 0]], dtype=torch.long, device=DEVICE)
    zs_bounds = torch.tensor([[
        [0.05, 0.12, 0.45, 0.18, 0.25, 0.15, 0.4, 0.06, 0.024, 6.66],
        [0.05, 0.04, 0.95, 0.10, 0.5, 0.07, 0.9, 0.06, 0.054, 15.0],
        [0.50, 0.12, 0.85, 0.18, 0.675, 0.15, 0.35, 0.06, 0.021, 5.83],
        [0.05, 0.20, 0.25, 0.26, 0.15, 0.23, 0.2, 0.06, 0.012, 3.33],
        [0.30, 0.20, 0.55, 0.26, 0.425, 0.23, 0.25, 0.06, 0.015, 4.16],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ]], dtype=torch.float32, device=DEVICE)
    zs_flags = torch.zeros(1, N, 16, device=DEVICE)
    zs_flags[0, 1, 0] = 1.0  # Search is editable
    with torch.no_grad():
        zs_out = model(
            goal_ids=zs_enc["input_ids"].to(DEVICE),
            goal_mask=zs_enc["attention_mask"].float().to(DEVICE),
            candidate_label_ids=zsc_enc["input_ids"].unsqueeze(0).to(DEVICE),
            candidate_label_mask=zsc_enc["attention_mask"].float().unsqueeze(0).to(DEVICE),
            candidate_bounds=zs_bounds,
            candidate_roles=zs_roles,
            candidate_flags=zs_flags,
            candidate_pad_mask=torch.ones(1, N, device=DEVICE),
        )
    zs_scores = zs_out["ranking_scores"][0]
    zs_top_idx = zs_scores.argmax().item()
    zs_pass = (zs_top_idx == 0)
    zepto_ice_margin = float(zs_scores[0] - zs_scores[3])
    zepto_comp_margin = float(zs_scores[0] - zs_scores[5])
    print(f"Zepto Query Suggestion  | Top: {zepto_search_cands[zs_top_idx]:25s} | Margin vs Ice: +{zepto_ice_margin:.2f} pts, vs Complete: +{zepto_comp_margin:.2f} pts | [{'PASS' if zs_pass else 'FAIL'}]")

    # Zepto Direct Search Input Typing Screen (When no matching chip exists)
    zepto_type_cands = [
        "Search",
        "Ice",
        "Munchies",
        "complete"
    ]
    N = len(zepto_type_cands)
    zt_enc = TOKENIZER("Open Zepto and search fresh milk", max_length=32, truncation=True, return_tensors="pt")
    ztc_enc = TOKENIZER(zepto_type_cands, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
    zt_roles = torch.tensor([[2, 1, 1, 0]], dtype=torch.long, device=DEVICE)
    zt_bounds = torch.tensor([[
        [0.05, 0.04, 0.95, 0.10, 0.5, 0.07, 0.9, 0.06, 0.054, 15.0],
        [0.05, 0.20, 0.25, 0.26, 0.15, 0.23, 0.2, 0.06, 0.012, 3.33],
        [0.30, 0.20, 0.55, 0.26, 0.425, 0.23, 0.25, 0.06, 0.015, 4.16],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ]], dtype=torch.float32, device=DEVICE)
    zt_flags = torch.zeros(1, N, 16, device=DEVICE)
    zt_flags[0, 0, 0] = 1.0  # Search is editable
    with torch.no_grad():
        zt_out = model(
            goal_ids=zt_enc["input_ids"].to(DEVICE),
            goal_mask=zt_enc["attention_mask"].float().to(DEVICE),
            candidate_label_ids=ztc_enc["input_ids"].unsqueeze(0).to(DEVICE),
            candidate_label_mask=ztc_enc["attention_mask"].float().unsqueeze(0).to(DEVICE),
            candidate_bounds=zt_bounds,
            candidate_roles=zt_roles,
            candidate_flags=zt_flags,
            candidate_pad_mask=torch.ones(1, N, device=DEVICE),
        )
    zt_scores = zt_out["ranking_scores"][0]
    zt_top_idx = zt_scores.argmax().item()
    zt_pass = (zt_top_idx == 0)
    zt_margin = float(zt_scores[0] - max(zt_scores[1], zt_scores[2], zt_scores[3]))
    print(f"Zepto Search Input Box  | Top: {zepto_type_cands[zt_top_idx]:25s} | Margin vs Distractors & Complete: +{zt_margin:.2f} pts | [{'PASS' if zt_pass else 'FAIL'}]")

    all_passed = (
        (anchors_passed == len(screens))
        and z_pass
        and y_pass
        and ya_pass
        and ys_pass
        and yd_pass
        and i_pass
        and id_pass
        and zs_pass
        and zt_pass
    )
    print("\n" + "=" * 80)
    print(f"FINAL DUAL VERIFICATION: {'100% ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 80)
    return all_passed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="")
    args = parser.parse_args()
    if args.ckpt:
        target_ckpt = Path(args.ckpt)
    else:
        target_ckpt = SMOKE_CKPT if SMOKE_CKPT.exists() else SOURCE_CKPT
    verify_checkpoint(target_ckpt)
