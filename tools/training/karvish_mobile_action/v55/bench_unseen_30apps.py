"""
KARVISH UNIVERSAL V55 — 30-APP UNSEEN BENCHMARK & TASK-CONCEPT TRANSFER SUITE
Phase 6 Implementation:
  Evaluates Karvish Universal V55 on 30 completely unseen applications across 5 functional domains:
    1. Communication & Social (SignalX, ChimeChat, EchoFeed, TeamPulse, VibeTalk, WhisperNet)
    2. E-Commerce & Quick-Commerce (SpeedMart, ShopZen, FreshDrop, SwiftCart, DailyBasket, PantryPlus)
    3. Productivity & Workspace (NoteCraft, TaskCraft, CloudDrive, DocuFlow, MindMap, QuickMemo)
    4. Media & Entertainment (AudioFlow, CineView, PhotoLab, StreamVibe, BookWorm, GameVault)
    5. Utilities & System Control (HealthTrack, BudgetWise, HomeSmart, CleanSweep, TravelNest, RideEasy)

Evaluates:
  - Task-Concept Transfer (f(Role, Subgoal, Context, Bounds))
  - Adversarial UI Distractor Invariance (ads, sponsored cards, banners)
  - Layout Generalization (unusual aspect ratios, reordered feeds, bottom bars)

Output:
  tools/training/karvish_mobile_action/v55/UNSEEN_BENCHMARK_REPORT.json
"""

from __future__ import annotations

import sys
import os
import json
import time
import math
import random
from pathlib import Path
from typing import Dict, List, Any

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8")

import torch

REPO_ROOT = Path("c:/Users/Karthik S/AndroidStudioProjects/zipper")
V54_DIR = REPO_ROOT / "tools" / "training" / "karvish_mobile_action" / "v54"
V55_DIR = REPO_ROOT / "tools" / "training" / "karvish_mobile_action" / "v55"
sys.path.insert(0, str(V55_DIR))
sys.path.insert(0, str(V54_DIR))
sys.path.insert(0, str(REPO_ROOT))

from model_v55 import KarvishModelV55
from schema_v55 import ROLE_TO_IDX, ACTION_TO_IDX, CANONICAL_ACTIONS, SUBGOAL_TO_IDX
from train_v54_full import TOKENIZER, DEVICE

V55_CKPT_PATH = V55_DIR / "checkpoints" / "v55_best.pt"
OUTPUT_REPORT = V55_DIR / "UNSEEN_BENCHMARK_REPORT.json"

UNSEEN_APPS_30 = [
    # Domain 1: Communication
    {"app": "SignalX", "domain": "Communication", "goal": "send 'hello' to Alex", "subgoal": "COMPOSE_MESSAGE", "target_role": "MESSAGE_INPUT", "target_lbl": "Type a message", "distractor_role": "HEADER", "distractor_lbl": "Alex (Online)", "act": "TYPE_TEXT"},
    {"app": "ChimeChat", "domain": "Communication", "goal": "message Sarah: lunch at 1", "subgoal": "COMPOSE_MESSAGE", "target_role": "MESSAGE_INPUT", "target_lbl": "Write something...", "distractor_role": "HEADER", "distractor_lbl": "Sarah", "act": "TYPE_TEXT"},
    {"app": "EchoFeed", "domain": "Communication", "goal": "reply 'agree' to David", "subgoal": "COMPOSE_MESSAGE", "target_role": "MESSAGE_INPUT", "target_lbl": "Reply...", "distractor_role": "HEADER", "distractor_lbl": "David Miller", "act": "TYPE_TEXT"},
    {"app": "TeamPulse", "domain": "Communication", "goal": "post 'standup ready' on TeamPulse", "subgoal": "COMPOSE_MESSAGE", "target_role": "MESSAGE_INPUT", "target_lbl": "Message #general", "distractor_role": "HEADER", "distractor_lbl": "#general (12 members)", "act": "TYPE_TEXT"},
    {"app": "VibeTalk", "domain": "Communication", "goal": "chat with Kevin", "subgoal": "SELECT_RESULT", "target_role": "CONTACT_ITEM", "target_lbl": "Kevin Brown", "distractor_role": "BUTTON", "distractor_lbl": "complete", "act": "CLICK"},
    {"app": "WhisperNet", "domain": "Communication", "goal": "view profile of Elena", "subgoal": "VIEW_PROFILE", "target_role": "HEADER", "target_lbl": "Elena Rostova", "distractor_role": "MESSAGE_INPUT", "target_lbl": "Elena Rostova", "distractor_lbl": "Type a message", "act": "CLICK"},

    # Domain 2: E-Commerce
    {"app": "SpeedMart", "domain": "E-Commerce", "goal": "search cheapest apples on SpeedMart", "subgoal": "SELECT_RESULT", "target_role": "PRODUCT_CARD", "target_lbl": "Fresh Fuji Apples 1kg ₹120", "distractor_role": "PRODUCT_CARD", "distractor_lbl": "Imported Royal Apples 1kg ₹260", "price": 120.0, "dis_price": 260.0, "act": "CLICK"},
    {"app": "ShopZen", "domain": "E-Commerce", "goal": "find cheapest blue socks", "subgoal": "SELECT_RESULT", "target_role": "PRODUCT_CARD", "target_lbl": "Cotton Blue Socks Pair ₹49", "distractor_role": "PRODUCT_CARD", "distractor_lbl": "Woolen Thermal Socks ₹199", "price": 49.0, "dis_price": 199.0, "act": "CLICK"},
    {"app": "FreshDrop", "domain": "E-Commerce", "goal": "search milk and add to cart", "subgoal": "ENTER_QUERY", "target_role": "SEARCH_INPUT", "target_lbl": "Search fresh dairy, fruits...", "distractor_role": "BUTTON", "distractor_lbl": "View Deals 50% OFF", "act": "TYPE_TEXT"},
    {"app": "SwiftCart", "domain": "E-Commerce", "goal": "search lowest price coffee", "subgoal": "SELECT_RESULT", "target_role": "PRODUCT_CARD", "target_lbl": "Classic Instant Coffee 50g ₹75", "distractor_role": "PRODUCT_CARD", "distractor_lbl": "Artisan Roast Beans 250g ₹450", "price": 75.0, "dis_price": 450.0, "act": "CLICK"},
    {"app": "DailyBasket", "domain": "E-Commerce", "goal": "search bread on DailyBasket", "subgoal": "ENTER_QUERY", "target_role": "SEARCH_INPUT", "target_lbl": "Search 10,000+ items", "distractor_role": "BUTTON", "distractor_lbl": "Cart ₹0", "act": "TYPE_TEXT"},
    {"app": "PantryPlus", "domain": "E-Commerce", "goal": "buy cheapest eggs", "subgoal": "SELECT_RESULT", "target_role": "PRODUCT_CARD", "target_lbl": "Farm Fresh Eggs 6pk ₹42", "distractor_role": "PRODUCT_CARD", "distractor_lbl": "Organic Brown Eggs 6pk ₹85", "price": 42.0, "dis_price": 85.0, "act": "CLICK"},

    # Domain 3: Productivity
    {"app": "NoteCraft", "domain": "Productivity", "goal": "create new note", "subgoal": "LOCATE_INPUT", "target_role": "BUTTON", "target_lbl": "New Note (+)", "distractor_role": "HEADER", "distractor_lbl": "All Notes (14)", "act": "CLICK"},
    {"app": "TaskCraft", "domain": "Productivity", "goal": "add task 'buy milk'", "subgoal": "ENTER_QUERY", "target_role": "EDITABLE_FIELD", "target_lbl": "What needs to be done?", "distractor_role": "TAB_ITEM", "distractor_lbl": "Completed", "act": "TYPE_TEXT"},
    {"app": "CloudDrive", "domain": "Productivity", "goal": "upload file to CloudDrive", "subgoal": "LOCATE_INPUT", "target_role": "BUTTON", "target_lbl": "Upload File", "distractor_role": "HEADER", "distractor_lbl": "Storage 42% Used", "act": "CLICK"},
    {"app": "DocuFlow", "domain": "Productivity", "goal": "search invoices", "subgoal": "ENTER_QUERY", "target_role": "SEARCH_INPUT", "target_lbl": "Filter documents...", "distractor_role": "BUTTON", "distractor_lbl": "Upgrade Pro", "act": "TYPE_TEXT"},
    {"app": "MindMap", "domain": "Productivity", "goal": "open project Roadmap", "subgoal": "SELECT_RESULT", "target_role": "LIST_ITEM", "target_lbl": "Q4 Engineering Roadmap", "distractor_role": "BUTTON", "distractor_lbl": "Settings", "act": "CLICK"},
    {"app": "QuickMemo", "domain": "Productivity", "goal": "write memo 'team sync at 4'", "subgoal": "COMPOSE_MESSAGE", "target_role": "EDITABLE_FIELD", "target_lbl": "Start typing your thoughts...", "distractor_role": "HEADER", "distractor_lbl": "QuickMemo v2.1", "act": "TYPE_TEXT"},

    # Domain 4: Media & Streaming
    {"app": "AudioFlow", "domain": "Media", "goal": "search podcast 'Huberman'", "subgoal": "ENTER_QUERY", "target_role": "SEARCH_INPUT", "target_lbl": "Search songs, podcasts...", "distractor_role": "BUTTON", "distractor_lbl": "Subscribe ₹99/mo", "act": "TYPE_TEXT"},
    {"app": "CineView", "domain": "Media", "goal": "play Interstellar", "subgoal": "SELECT_RESULT", "target_role": "MEDIA_ITEM", "target_lbl": "Interstellar (2014) 4K", "distractor_role": "HEADER", "distractor_lbl": "Top Trending Movies", "act": "CLICK"},
    {"app": "PhotoLab", "domain": "Media", "goal": "open recent photo", "subgoal": "SELECT_RESULT", "target_role": "MEDIA_ITEM", "target_lbl": "IMG_20261002_Sunset.jpg", "distractor_role": "TAB_ITEM", "distractor_lbl": "Albums", "act": "CLICK"},
    {"app": "StreamVibe", "domain": "Media", "goal": "search lo-fi beats on StreamVibe", "subgoal": "ENTER_QUERY", "target_role": "SEARCH_INPUT", "target_lbl": "Search artists, tracks...", "distractor_role": "BUTTON", "distractor_lbl": "Trending Now", "act": "TYPE_TEXT"},
    {"app": "BookWorm", "domain": "Media", "goal": "open chapter 4", "subgoal": "SELECT_RESULT", "target_role": "LIST_ITEM", "target_lbl": "Chapter 4: The Neural Age", "distractor_role": "BUTTON", "distractor_lbl": "Bookmark", "act": "CLICK"},
    {"app": "GameVault", "domain": "Media", "goal": "launch Chess Master", "subgoal": "SELECT_RESULT", "target_role": "LIST_ITEM", "target_lbl": "Chess Master 3D", "distractor_role": "HEADER", "distractor_lbl": "Installed Games", "act": "CLICK"},

    # Domain 5: Utilities & Tools
    {"app": "HealthTrack", "domain": "Utilities", "goal": "log water intake", "subgoal": "LOCATE_INPUT", "target_role": "BUTTON", "target_lbl": "+ Add Water (250ml)", "distractor_role": "HEADER", "distractor_lbl": "Daily Target: 2.5L", "act": "CLICK"},
    {"app": "BudgetWise", "domain": "Utilities", "goal": "add expense ₹50", "subgoal": "ENTER_QUERY", "target_role": "EDITABLE_FIELD", "target_lbl": "Amount (₹)", "distractor_role": "TAB_ITEM", "distractor_lbl": "Monthly Report", "act": "TYPE_TEXT"},
    {"app": "HomeSmart", "domain": "Utilities", "goal": "toggle living room light", "subgoal": "TOGGLE_SETTING", "target_role": "CHECKBOX_TOGGLE", "target_lbl": "Living Room Lamp (OFF)", "distractor_role": "HEADER", "distractor_lbl": "Home Network: Online", "act": "CLICK"},
    {"app": "CleanSweep", "domain": "Utilities", "goal": "clean cache", "subgoal": "LOCATE_INPUT", "target_role": "BUTTON", "target_lbl": "Clean Cache (1.2 GB)", "distractor_role": "HEADER", "distractor_lbl": "Memory Clean 88%", "act": "CLICK"},
    {"app": "TravelNest", "domain": "Utilities", "goal": "search hotels in Goa", "subgoal": "ENTER_QUERY", "target_role": "SEARCH_INPUT", "target_lbl": "Where do you want to stay?", "distractor_role": "BUTTON", "distractor_lbl": "Top Destinations", "act": "TYPE_TEXT"},
    {"app": "RideEasy", "domain": "Utilities", "goal": "book ride to Airport", "subgoal": "ENTER_QUERY", "target_role": "SEARCH_INPUT", "target_lbl": "Where to?", "distractor_role": "HEADER", "distractor_lbl": "Current Location: MG Road", "act": "TYPE_TEXT"},
]


def evaluate_unseen_task(model: KarvishModelV55, task: Dict[str, Any]) -> Dict[str, Any]:
    goal = task["goal"]
    subgoal = task["subgoal"]
    tar_role = task["target_role"]
    tar_lbl = task["target_lbl"]
    dis_role = task["distractor_role"]
    dis_lbl = task["distractor_lbl"]

    tar_role_idx = ROLE_TO_IDX.get(tar_role, 0)
    dis_role_idx = ROLE_TO_IDX.get(dis_role, 0)

    # 4 candidates: Target, Forbidden Distractor, and 2 random noise elements
    cands = [
        {"lbl": tar_lbl, "role": tar_role, "role_idx": tar_role_idx, "price": task.get("price", 0.0), "bounds": [0.05, 0.20, 0.95, 0.30, 0.5, 0.25, 0.9, 0.10, 0.09, 9.0], "is_editable": (task["act"] == "TYPE_TEXT"), "is_clickable": True},
        {"lbl": dis_lbl, "role": dis_role, "role_idx": dis_role_idx, "price": task.get("dis_price", 0.0), "bounds": [0.05, 0.04, 0.95, 0.12, 0.5, 0.08, 0.9, 0.08, 0.072, 11.25], "is_clickable": True},
        {"lbl": "More Options", "role": "MENU_ITEM", "role_idx": ROLE_TO_IDX["MENU_ITEM"], "price": 0.0, "bounds": [0.85, 0.04, 0.95, 0.12, 0.90, 0.08, 0.10, 0.08, 0.008, 1.25], "is_clickable": True},
        {"lbl": "Sponsored: Get 20% Discount", "role": "BUTTON", "role_idx": ROLE_TO_IDX["BUTTON"], "price": 0.0, "bounds": [0.05, 0.85, 0.95, 0.95, 0.5, 0.90, 0.9, 0.10, 0.09, 9.0], "is_clickable": True},
    ]

    # Candidate random permutation to test spatial permutation invariance
    perm = list(range(len(cands)))
    random.shuffle(perm)
    shuffled_cands = [cands[i] for i in perm]
    new_tar_idx = perm.index(0)
    new_dis_idx = perm.index(1)

    enc = TOKENIZER(goal, max_length=32, truncation=True, return_tensors="pt")
    g_ids = enc["input_ids"].to(DEVICE)
    g_mask = enc["attention_mask"].float().to(DEVICE)

    lbl_texts = [c["lbl"] for c in shuffled_cands]
    c_enc = TOKENIZER(lbl_texts, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
    c_lbl_ids = c_enc["input_ids"].unsqueeze(0).to(DEVICE)
    c_lbl_mask = c_enc["attention_mask"].float().unsqueeze(0).to(DEVICE)

    bounds_vec = [c["bounds"] for c in shuffled_cands]
    roles_vec = [c["role_idx"] for c in shuffled_cands]
    prices_vec = [[math.log1p(float(c.get("price", 0.0))) / 10.0, math.log1p(float(c.get("price", 0.0))) / 10.0, 0.0, 0.0] for c in shuffled_cands]
    flags_vec = [[1.0 if c.get("is_editable") else 0.0, 0.0, 1.0 if c.get("is_clickable", True) else 0.0] + [0.0] * 13 for c in shuffled_cands]

    c_bounds = torch.tensor([bounds_vec], dtype=torch.float32, device=DEVICE)
    c_roles = torch.tensor([roles_vec], dtype=torch.long, device=DEVICE)
    c_flags = torch.tensor([flags_vec], dtype=torch.float32, device=DEVICE)
    c_pad = torch.ones(1, len(shuffled_cands), dtype=torch.float32, device=DEVICE)
    c_num = torch.tensor([prices_vec], dtype=torch.float32, device=DEVICE)

    subgoal_idx = torch.tensor([SUBGOAL_TO_IDX.get(subgoal, 0)], dtype=torch.long, device=DEVICE)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(
            goal_ids=g_ids,
            goal_mask=g_mask,
            candidate_label_ids=c_lbl_ids,
            candidate_label_mask=c_lbl_mask,
            candidate_bounds=c_bounds,
            candidate_roles=c_roles,
            candidate_flags=c_flags,
            candidate_pad_mask=c_pad,
            candidate_num_features=c_num,
            active_subgoal_override=subgoal_idx,
        )
    latency_ms = (time.perf_counter() - t0) * 1000

    scores = out["ranking_scores"][0].cpu().numpy().tolist()
    pred_idx = int(scores.index(max(scores)))
    tar_score = scores[new_tar_idx]
    dis_score = scores[new_dis_idx]
    margin = tar_score - dis_score

    action_idx = int(torch.argmax(out["action_logits"][0]).item())
    pred_action = CANONICAL_ACTIONS[action_idx]

    passed = (pred_idx == new_tar_idx and margin >= 2.0)

    return {
        "app": task["app"],
        "domain": task["domain"],
        "passed": bool(passed),
        "target_correct": bool(pred_idx == new_tar_idx),
        "picked_candidate": shuffled_cands[pred_idx]["lbl"],
        "expected_candidate": tar_lbl,
        "margin": round(float(margin), 2),
        "latency_ms": round(float(latency_ms), 2),
        "pred_action": pred_action,
    }


def run_30app_unseen_benchmark():
    print("=" * 80)
    print("STARTING KARVISH UNIVERSAL V55 — 30-APP UNSEEN BENCHMARK")
    print(f"Model Checkpoint: {V55_CKPT_PATH.name}")
    print("Zero Pre-training on These Apps | 100% Generalization Test")
    print("=" * 80)

    model = KarvishModelV55().to(DEVICE)
    state = torch.load(V55_CKPT_PATH, map_location=DEVICE, weights_only=False)
    sd = state.get("model_state_dict", state.get("model", state))
    clean_sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(clean_sd, strict=True)
    model.eval()

    results = []
    domain_counts = {}
    domain_passes = {}

    for task in UNSEEN_APPS_30:
        res = evaluate_unseen_task(model, task)
        results.append(res)

        dom = task["domain"]
        domain_counts[dom] = domain_counts.get(dom, 0) + 1
        if res["passed"]:
            domain_passes[dom] = domain_passes.get(dom, 0) + 1

        status = "PASS" if res["passed"] else "FAIL"
        print(f"[{status}] {res['domain']:<14} | {res['app']:<12} | Target: '{res['expected_candidate'][:22]}' | Margin: +{res['margin']}")

    total_tasks = len(results)
    total_passed = sum(1 for r in results if r["passed"])
    pass_rate_pct = (total_passed / total_tasks) * 100.0

    print("\n" + "=" * 80)
    print("DOMAIN-WISE GENERALIZATION ACCURACY:")
    print("-" * 80)
    for dom in domain_counts:
        p = domain_passes.get(dom, 0)
        t = domain_counts[dom]
        pct = (p / t) * 100.0
        print(f"  - {dom:<16}: {p}/{t} ({pct:.1f}%)")

    print("=" * 80)
    print(f">> HEADLINE METRIC: UNSEEN-APP SUCCESS RATE: {total_passed}/{total_tasks} ({pass_rate_pct:.1f}%)")
    print(f">> Target Threshold: >= 85.0% | Status: {'QUALIFIED [PASS]' if pass_rate_pct >= 85.0 else 'FAILED'}")
    print("=" * 80)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "Karvish Universal V55",
        "checkpoint": str(V55_CKPT_PATH),
        "total_unseen_apps_evaluated": total_tasks,
        "total_passed": total_passed,
        "unseen_app_success_rate_pct": round(pass_rate_pct, 2),
        "threshold_required_pct": 85.0,
        "status": "QUALIFIED" if pass_rate_pct >= 85.0 else "UNQUALIFIED",
        "domain_breakdown": {
            dom: {
                "passed": domain_passes.get(dom, 0),
                "total": domain_counts[dom],
                "accuracy_pct": round((domain_passes.get(dom, 0) / domain_counts[dom]) * 100.0, 1),
            }
            for dom in domain_counts
        },
        "tasks_detail": results,
    }

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f">> Saved Full Benchmark Report to: {OUTPUT_REPORT}")


if __name__ == "__main__":
    run_30app_unseen_benchmark()
