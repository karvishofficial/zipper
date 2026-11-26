"""
KARVISH V54 — BATTLE-HARDENED REAL-DEVICE AUDIT
Tests the CURRENT model against REAL Samsung Galaxy A26 Accessibility Dumps
extracted directly from live execution logs (task-6117.log).

Constitutional Rules:
- 0 hardcoded task heuristics.
- Does NOT train or modify any checkpoint.
- Rigorously tests the exact failure modes seen on device:
  1. WhatsApp 'submit focused field' parasitic loop on contact search results.
  2. Instagram 'Reshare number is 10197' video button trap vs search.
  3. Instagram Share Sheet 'submit focused field' loop.
  4. Telegram empty 'Message' input vs 'submit focused field'.
  5. WhatsApp 'Video call' accidental tap vs chat input.
  6. Zepto promo card vs search bar.
  7. Zepto product list vs 'submit focused field'.
  8. LinkedIn sponsored feed card vs search bar.
  9. YouTube voice mic button ('Voice Song') vs search bar.
 10. Android App Switcher app selection vs distractor apps.
"""

from __future__ import annotations

import sys
import os
import re
import math
import time
from pathlib import Path

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8")

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from model_v54 import KarvishModelV54
from schema_v54 import CANONICAL_ACTIONS, ACTION_TO_IDX
from train_v54_full import TOKENIZER, DEVICE

CKPT_PATH = SCRIPT_DIR / "checkpoints" / "v54_best.pt"


def evaluate_screen(
    model: KarvishModelV54,
    goal: str,
    candidates: list[dict],
    expected_target_checker,
    forbidden_distractor_checker,
    expected_action: str | None = None,
    min_margin: float = 3.0,
) -> dict:
    N = len(candidates)
    lbl_texts = [str(c.get("lbl", "")).strip() for c in candidates]

    enc = TOKENIZER(goal, max_length=32, truncation=True, return_tensors="pt")
    g_ids = enc["input_ids"].to(DEVICE)
    g_mask = enc["attention_mask"].float().to(DEVICE)

    c_enc = TOKENIZER(lbl_texts, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
    c_lbl_ids = c_enc["input_ids"].unsqueeze(0).to(DEVICE)
    c_lbl_mask = c_enc["attention_mask"].float().unsqueeze(0).to(DEVICE)

    bounds_list = []
    roles_list = []
    for i, c in enumerate(candidates):
        b = c.get("bounds", None)
        if isinstance(b, list) and len(b) == 10:
            bounds_list.append(b)
        else:
            bounds_list.append([0.05, 0.05 + 0.03 * i, 0.95, 0.08 + 0.03 * i, 0.5, 0.065 + 0.03 * i, 0.9, 0.03, 0.027, 30.0])

        r_str = str(c.get("role", "")).lower()
        act_str = str(c.get("act", "")).lower()
        if "input" in r_str or "edit" in r_str or c.get("role_idx") == 2 or c.get("is_editable"):
            roles_list.append(2)
        elif "button" in r_str or "click" in r_str or ("viewgroup" in r_str and act_str == "tap") or c.get("role_idx") == 1:
            roles_list.append(1)
        else:
            roles_list.append(0)

    c_bounds = torch.tensor([bounds_list], dtype=torch.float32, device=DEVICE)
    c_roles = torch.tensor([roles_list], dtype=torch.long, device=DEVICE)
    c_flags = torch.zeros(1, N, 16, dtype=torch.float32, device=DEVICE)
    for i, c in enumerate(candidates):
        if roles_list[i] == 2:
            c_flags[0, i, 0] = 1.0
        if c.get("is_clickable", True):
            c_flags[0, i, 2] = 1.0
    c_pad_mask = torch.ones(1, N, dtype=torch.float32, device=DEVICE)

    c_num = torch.zeros(1, N, 4, dtype=torch.float32, device=DEVICE)
    for i, c in enumerate(candidates):
        lbl = lbl_texts[i]
        cleaned = re.sub(r'([₹]|rs\.?|inr)\s*([\d]+),([\d]+)', r'\1\2\3', lbl, flags=re.I)
        matches = re.findall(r"(?:₹|rs\.?|inr)\s*([\d]+(?:\.\d+)?)", cleaned, re.IGNORECASE)
        if matches:
            valid_nums = [float(x) for x in matches if 0.0 < float(x) < 100000.0]
            if valid_nums:
                c_num[0, i, 0] = math.log1p(min(valid_nums)) / 10.0

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
            candidate_num_features=c_num,
        )

    ranking = out["ranking_scores"][0]
    action_logits = out["action_logits"][0]

    top_idx = int(torch.argmax(ranking).item())
    top_cand = candidates[top_idx]
    top_score = ranking[top_idx].item()

    pred_act_idx = int(torch.argmax(action_logits).item())
    pred_action = CANONICAL_ACTIONS[pred_act_idx]

    # Find expected target score and distractor score
    target_idx = None
    target_score = -999.0
    for i, c in enumerate(candidates):
        if expected_target_checker(c):
            if ranking[i].item() > target_score:
                target_score = ranking[i].item()
                target_idx = i

    distractor_idx = None
    distractor_score = -999.0
    for i, c in enumerate(candidates):
        if forbidden_distractor_checker(c):
            if ranking[i].item() > distractor_score:
                distractor_score = ranking[i].item()
                distractor_idx = i

    margin = target_score - distractor_score if (target_idx is not None and distractor_idx is not None) else 0.0

    is_top1_ok = expected_target_checker(top_cand)
    is_action_ok = (expected_action is None) or (pred_action == expected_action)
    is_margin_ok = margin >= min_margin

    passed = is_top1_ok and is_action_ok and is_margin_ok

    return {
        "passed": passed,
        "top_cand": top_cand,
        "top_score": top_score,
        "pred_action": pred_action,
        "target_idx": target_idx,
        "target_cand": candidates[target_idx] if target_idx is not None else None,
        "target_score": target_score,
        "distractor_idx": distractor_idx,
        "distractor_cand": candidates[distractor_idx] if distractor_idx is not None else None,
        "distractor_score": distractor_score,
        "margin": margin,
        "is_top1_ok": is_top1_ok,
        "is_action_ok": is_action_ok,
        "is_margin_ok": is_margin_ok,
    }


def parse_live_screens_from_log() -> list[dict]:
    """
    Parses live device candidate dumps from task logs using robust multiline matching.
    """
    tasks_dir = Path(r"C:\Users\Karthik S\.gemini\antigravity-ide\brain\b0921c87-82c4-4a56-9b5c-50097b3ca10d\.system_generated\tasks")
    screens = []
    log_names = ["task-6117.log", "task-6843.log"]
    for log_name in log_names:
        log_file = tasks_dir / log_name
        if not log_file.exists():
            continue

        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

        turn_blocks = re.findall(
            r">> \[V54 TURN (\d+)\] Received (\d+) candidates for goal '(.*?)':\n(.*?)(?=\d+\.\d+\.\d+\.\d+ - -|\Z|>> \[V54 TURN)",
            text,
            re.DOTALL
        )

        cand_pattern = re.compile(
            r"\[\s*(\d+)\] tool='(.*?)' \| id='(.*?)' \| lbl='(.*?)' \| role='(.*?)' \| act='(.*?)'(?=\s*\[\s*\d+\]|\Z)",
            re.DOTALL
        )

        for turn_num, count_str, goal, block in turn_blocks:
            cands = []
            for cm in cand_pattern.finditer(block):
                idx_str, tool, cid, lbl, role, act = cm.groups()
                cands.append({
                    "idx": int(idx_str),
                    "tool": tool,
                    "id": cid,
                    "lbl": lbl.strip(),
                    "role": role,
                    "act": act
                })
            screens.append({
                "turn": int(turn_num),
                "goal": goal,
                "candidates": cands
            })

    return screens


def run_battle_hardened_audit():
    print("=" * 80)
    print("KARVISH V54 — BATTLE-HARDENED REAL-DEVICE AUDIT")
    print(f"Testing Current Model: {CKPT_PATH.name}")
    print("Zero Heuristics | 100% Real-Device Samsung A26 Dumps")
    print("=" * 80)

    if not CKPT_PATH.exists():
        print(f"[FATAL] Checkpoint not found: {CKPT_PATH}")
        sys.exit(1)

    model = KarvishModelV54().to(DEVICE)
    sd = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(sd.get("model_state_dict", sd))
    model.eval()

    screens = parse_live_screens_from_log()
    print(f">> Loaded {len(screens)} real device screens directly from physical phone logs.\n")

    audit_results = []

    def print_gate(name: str, res: dict):
        status = "PASS" if res["passed"] else "FAIL (ERROR REPRODUCED)"
        print(f"[{status}] {name}")
        top_lbl = res['top_cand']['lbl'].replace('\n', ' ')[:45]
        tgt_lbl = res['target_cand']['lbl'].replace('\n', ' ')[:45] if res['target_cand'] else 'None'
        dist_lbl = res['distractor_cand']['lbl'].replace('\n', ' ')[:45] if res['distractor_cand'] else 'None'
        print(f"   Model Chose: '{top_lbl}' (Score: {res['top_score']:.2f}, Action: {res['pred_action']})")
        print(f"   Target:      '{tgt_lbl}' (Score: {res['target_score']:.2f})")
        print(f"   Distractor:  '{dist_lbl}' (Score: {res['distractor_score']:.2f})")
        print(f"   Margin:      {res['margin']:+.2f} (Required: >= +3.00)")
        print()

    # =========================================================================
    # SUITE 1: AUDITING THE EXACT PHYSICAL ERRORS YOU EXPERIENCED
    # =========================================================================
    print("-" * 80)
    print("SUITE 1: PHYSICAL DEVICE FAILURE MODES (THE ERRORS YOU REPORTED)")
    print("-" * 80)

    # GATE 1: WhatsApp Contact Search vs Submit Parasitic Loop
    # Screen from 'open WhatsApp and send hi to Karthik' with Karthik.S (You)
    for s in screens:
        if "send hi to karthik" in s["goal"].lower():
            lbls = [c["lbl"] for c in s["candidates"]]
            if any("Karthik.S (You)" in l for l in lbls) and any("submit" in l.lower() for l in lbls):
                res1 = evaluate_screen(
                    model=model,
                    goal=s["goal"],
                    candidates=s["candidates"],
                    expected_target_checker=lambda c: "Karthik.S (You)" in c["lbl"],
                    forbidden_distractor_checker=lambda c: "submit focused field" in c["lbl"],
                    expected_action="CLICK",
                    min_margin=3.0,
                )
                audit_results.append(("Gate 1: WhatsApp Contact Search vs Submit Loop", res1))
                print_gate("Gate 1: WhatsApp Contact Search vs Submit Loop", res1)
                break

    # GATE 2: Instagram Explore Search vs Video Reshare / Video Controls
    for s in screens:
        if "instagram" in s["goal"].lower() and "photography" in s["goal"].lower():
            lbls = [c["lbl"] for c in s["candidates"]]
            if any("Reshare" in l for l in lbls) and any("Search" in l for l in lbls):
                res2 = evaluate_screen(
                    model=model,
                    goal=s["goal"],
                    candidates=s["candidates"],
                    expected_target_checker=lambda c: "search" in c["lbl"].lower(),
                    forbidden_distractor_checker=lambda c: "reshare" in c["lbl"].lower(),
                    expected_action="CLICK",
                    min_margin=3.0,
                )
                audit_results.append(("Gate 2: Instagram Search vs Video Reshare Trap", res2))
                print_gate("Gate 2: Instagram Search vs Video Reshare Trap", res2)
                break

    # GATE 3: Instagram Share Sheet vs Submit Loop
    for s in screens:
        if "instagram" in s["goal"].lower():
            lbls = [c["lbl"] for c in s["candidates"]]
            if any("Bhavani" in l for l in lbls) and any("submit" in l.lower() for l in lbls):
                res3 = evaluate_screen(
                    model=model,
                    goal="open instagram and send hi to bhavani",
                    candidates=s["candidates"],
                    expected_target_checker=lambda c: "Bhavani" in c["lbl"],
                    forbidden_distractor_checker=lambda c: "submit focused field" in c["lbl"],
                    expected_action="CLICK",
                    min_margin=3.0,
                )
                audit_results.append(("Gate 3: Instagram Share Sheet vs Submit Loop", res3))
                print_gate("Gate 3: Instagram Share Sheet vs Submit Loop", res3)
                break

    # GATE 4: Telegram Open Chat Room vs Empty Submit
    for s in screens:
        if "telegram" in s["goal"].lower() and s["turn"] == 2:
            res4 = evaluate_screen(
                model=model,
                goal=s["goal"],
                candidates=s["candidates"],
                expected_target_checker=lambda c: c["lbl"] == "Message",
                forbidden_distractor_checker=lambda c: "submit focused field" in c["lbl"],
                expected_action="TYPE_TEXT",
                min_margin=3.0,
            )
            audit_results.append(("Gate 4: Telegram Open Chat vs Empty Submit", res4))
            print_gate("Gate 4: Telegram Open Chat vs Empty Submit", res4)
            break

    # GATE 5: WhatsApp Direct Contact Screen vs Video Call Trap
    for s in screens:
        if "whatsapp" in s["goal"].lower() and "bhavani" in s["goal"].lower():
            lbls = [c["lbl"] for c in s["candidates"]]
            if any("Video call" in l for l in lbls):
                res5 = evaluate_screen(
                    model=model,
                    goal=s["goal"],
                    candidates=s["candidates"],
                    expected_target_checker=lambda c: "voice call" not in c["lbl"].lower() and "video call" not in c["lbl"].lower() and ("message" in c["lbl"].lower() or "bhavani" in c["lbl"].lower()),
                    forbidden_distractor_checker=lambda c: "video call" in c["lbl"].lower(),
                    expected_action="CLICK",
                    min_margin=3.0,
                )
                audit_results.append(("Gate 5: WhatsApp Contact Screen vs Video Call Trap", res5))
                print_gate("Gate 5: WhatsApp Contact Screen vs Video Call Trap", res5)
                break

    # =========================================================================
    # SUITE 2: SURPRISING AUDITS ACROSS OTHER APPS
    # =========================================================================
    print("-" * 80)
    print("SUITE 2: SURPRISE AUDITS ACROSS OTHER APPS (ZEPTO, LINKEDIN, YOUTUBE)")
    print("-" * 80)

    # GATE 6: Zepto Promo Card vs Search Bar
    for s in screens:
        if "zepto" in s["goal"].lower() and s["turn"] == 2:
            res6 = evaluate_screen(
                model=model,
                goal=s["goal"],
                candidates=s["candidates"],
                expected_target_checker=lambda c: "search" in c["lbl"].lower(),
                forbidden_distractor_checker=lambda c: "off" in c["lbl"].lower() and "33" in c["lbl"],
                expected_action="CLICK",
                min_margin=3.0,
            )
            audit_results.append(("Gate 6: Zepto Promo Card vs Search Bar", res6))
            print_gate("Gate 6: Zepto Promo Card vs Search Bar", res6)
            break

    # GATE 7: Zepto Search Results vs Submit Loop
    for s in screens:
        if "zepto" in s["goal"].lower():
            lbls = [c["lbl"] for c in s["candidates"]]
            if any("Tomato" in l for l in lbls) and any("submit" in l.lower() for l in lbls):
                res7 = evaluate_screen(
                    model=model,
                    goal=s["goal"],
                    candidates=s["candidates"],
                    expected_target_checker=lambda c: "tomato" in c["lbl"].lower(),
                    forbidden_distractor_checker=lambda c: "submit focused field" in c["lbl"],
                    expected_action="CLICK",
                    min_margin=3.0,
                )
                audit_results.append(("Gate 7: Zepto Search Results vs Submit Loop", res7))
                print_gate("Gate 7: Zepto Search Results vs Submit Loop", res7)
                break

    # GATE 8: LinkedIn Feed Card vs Search Bar
    for s in screens:
        if "linkedin" in s["goal"].lower() and s["turn"] in [2, 5]:
            lbls = [c["lbl"] for c in s["candidates"]]
            if any("Software Engineer" in l for l in lbls) or any("Home" in l for l in lbls):
                res8 = evaluate_screen(
                    model=model,
                    goal=s["goal"],
                    candidates=s["candidates"],
                    expected_target_checker=lambda c: "search" in c["lbl"].lower(),
                    forbidden_distractor_checker=lambda c: "strac" in c["lbl"].lower() or "software engineer (in" in c["lbl"].lower(),
                    expected_action="CLICK",
                    min_margin=3.0,
                )
                audit_results.append(("Gate 8: LinkedIn Feed Card vs Search Bar", res8))
                print_gate("Gate 8: LinkedIn Feed Card vs Search Bar", res8)
                break

    # GATE 9: YouTube Voice Mic ('Voice Song') vs Text Search Bar
    for s in screens:
        if "youtube" in s["goal"].lower() and s["turn"] in [3, 4]:
            lbls = [c["lbl"] for c in s["candidates"]]
            if any("Voice" in l for l in lbls):
                res9 = evaluate_screen(
                    model=model,
                    goal=s["goal"],
                    candidates=s["candidates"],
                    expected_target_checker=lambda c: "search" in c["lbl"].lower() and "voice" not in c["lbl"].lower(),
                    forbidden_distractor_checker=lambda c: "voice" in c["lbl"].lower(),
                    expected_action="CLICK",
                    min_margin=3.0,
                )
                audit_results.append(("Gate 9: YouTube Voice Mic vs Text Search", res9))
                print_gate("Gate 9: YouTube Voice Mic vs Text Search", res9)
                break

    # GATE 10: Android Task Switcher App Selection
    for s in screens:
        lbls = [c["lbl"] for c in s["candidates"]]
        if any("active apps" in l.lower() for l in lbls) and "zepto" in s["goal"].lower():
            res10 = evaluate_screen(
                model=model,
                goal=s["goal"],
                candidates=s["candidates"],
                expected_target_checker=lambda c: "zepto" in c["lbl"].lower(),
                forbidden_distractor_checker=lambda c: "linkedin" in c["lbl"].lower() or "instagram" in c["lbl"].lower(),
                expected_action="CLICK",
                min_margin=3.0,
            )
            audit_results.append(("Gate 10: Android Task Switcher App Selection", res10))
            print_gate("Gate 10: Android Task Switcher App Selection", res10)
            break

    # Summary
    print("=" * 80)
    print("BATTLE-HARDENED AUDIT SUMMARY ON CURRENT MODEL")
    print("=" * 80)
    passed_count = sum(1 for _, r in audit_results if r["passed"])
    total_count = len(audit_results)
    print(f">> PASSED: {passed_count}/{total_count} ({passed_count/max(1, total_count)*100:.1f}%)")
    for name, r in audit_results:
        st = "PASS" if r["passed"] else "FAIL"
        print(f"  [{st}] {name:<48} | Margin: {r['margin']:+.2f}")

    print("=" * 80)


if __name__ == "__main__":
    run_battle_hardened_audit()
