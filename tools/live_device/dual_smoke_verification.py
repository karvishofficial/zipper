"""
DUAL SMOKE VERIFICATION SUITE
Evaluates sandbox checkpoint (v54_chat_smoke.pt) on:
Part A: 12 Baseline Anchor Tasks (Regression Shield: Chrome, YouTube, LinkedIn, Calculator 1-5, Instagram, Settings).
Part B: AI Chat Tasks (Launcher ChatGPT/Gemini, ChatGPT real input, Gemini real input, Send message transition, Question span).
"""

from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")

import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(".").resolve()
sys.path.insert(0, str(REPO_ROOT / "tools" / "training" / "karvish_mobile_action" / "v54"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "training" / "karvish_mobile_action" / "v52"))

from model_v54 import KarvishModelV54
from schema_v54 import ACTION_TO_IDX
from probe_v52_screen_decisions import get_test_screens, evaluate_screen

IDX_TO_ACTION = {v: k for k, v in ACTION_TO_IDX.items()}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
SMOKE_CKPT = REPO_ROOT / "tools" / "training" / "karvish_mobile_action" / "v54" / "checkpoints" / "v54_chat_smoke.pt"


def eval_single(model, goal: str, candidates: list, expected_target: str, expected_action: str, expected_span: str = ""):
    N = len(candidates)
    enc = TOKENIZER(goal, max_length=32, truncation=True, padding="max_length", return_offsets_mapping=True, return_tensors="pt")
    g_ids = enc["input_ids"].to(DEVICE)
    g_mask = enc["attention_mask"].float().to(DEVICE)
    offsets = enc["offset_mapping"][0]

    lbl_texts = [c["label"] for c in candidates]
    c_enc = TOKENIZER(lbl_texts, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
    c_lbl_ids = c_enc["input_ids"].unsqueeze(0).to(DEVICE)
    c_lbl_mask = c_enc["attention_mask"].float().unsqueeze(0).to(DEVICE)
    c_bounds = torch.tensor([c.get("bounds_vec", [0.0]*10) for c in candidates], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    c_roles = torch.tensor([c.get("role_idx", 1 if c.get("role") == "button" else (2 if c.get("role") == "input" else 0)) for c in candidates], dtype=torch.long).unsqueeze(0).to(DEVICE)
    c_flags = torch.zeros(1, N, 16, dtype=torch.float32, device=DEVICE)
    for i, c in enumerate(candidates):
        if c.get("is_editable") or c.get("role") == "input" or c.get("role_idx") == 2: c_flags[0, i, 0] = 1.0
        if c.get("is_focused"): c_flags[0, i, 1] = 1.0
        if c.get("is_clickable", True): c_flags[0, i, 2] = 1.0
    c_pad_mask = torch.ones(1, N, dtype=torch.float32, device=DEVICE)

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
        )

    scores = out["ranking_scores"][0]
    top_idx = scores.argmax().item()
    top_cand = candidates[top_idx]

    act_idx = out["action_logits"][0].argmax().item()
    pred_act = IDX_TO_ACTION.get(act_idx, "CLICK")

    # Span decoding
    s_idx = out["span_start_logits"][0].argmax().item()
    e_idx = out["span_end_logits"][0].argmax().item()
    ext_span = ""
    if 0 <= s_idx <= e_idx < len(offsets):
        s_c = offsets[s_idx][0].item()
        e_c = offsets[e_idx][1].item()
        if e_c > s_c:
            ext_span = goal[s_c:e_c].strip()

    tgt_ok = expected_target.lower() in top_cand["label"].lower() or (expected_target.lower() in top_cand.get("res", "").lower())
    act_ok = (pred_act == expected_action) or (expected_action == "OPEN_APP" and pred_act in ("OPEN_APP", "CLICK"))
    span_ok = (expected_span.lower() in ext_span.lower()) if expected_span else True

    return {
        "selected": top_cand["label"],
        "action": pred_act,
        "span": ext_span,
        "pass": tgt_ok and act_ok and span_ok,
        "tgt_ok": tgt_ok,
        "act_ok": act_ok,
        "span_ok": span_ok,
        "score": scores[top_idx].item(),
    }


def main():
    print("=" * 80)
    print("        DUAL SMOKE TEST VERIFICATION (ANCHORS + AI CHAT)")
    print("=" * 80)

    model = KarvishModelV54().to(DEVICE)
    ckpt = torch.load(SMOKE_CKPT, map_location=DEVICE, weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd, strict=False)
    model.eval()
    print(f">> Loaded candidate checkpoint: {SMOKE_CKPT.name}")

    # =========================================================================
    # PART A: 12 BASELINE ANCHORS (REGRESSION SHIELD)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART A: EVALUATING 12 BASELINE ANCHORS (EXISTING WORKING FEATURES)")
    print("=" * 80)

    anchor_screens = get_test_screens()
    anchors_passed = 0
    anchor_results = []

    for s in anchor_screens:
        res = eval_single(
            model=model,
            goal=s["goal"],
            candidates=s["candidates"][:24],
            expected_target=s["expected_target"],
            expected_action=s.get("expected_action", "CLICK"),
            expected_span=s.get("expected_text", ""),
        )
        if res["pass"]:
            anchors_passed += 1
            status = "PASS"
        else:
            status = "FAIL"
        anchor_results.append((s["name"], res["selected"], res["action"], res["span"], status))
        print(f"[{status:4s}] {s['name'][:35]:35s} | Target: '{res['selected'][:20]:20s}' | Act: {res['action']:10s} | Text: '{res['span']}'")

    print(f"\n>> ANCHORS TOTAL SCORE: {anchors_passed}/{len(anchor_screens)} ({anchors_passed/len(anchor_screens)*100:.1f}%)")

    # =========================================================================
    # PART B: AI CHAT SUITE (NEW FEATURES)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART B: EVALUATING AI CHAT CAPABILITIES (NEW EXTENSIONS)")
    print("=" * 80)

    chat_tests = [
        # Test 1: Launcher Turn (ChatGPT)
        {
            "name": "Launcher: 'Ask ChatGPT what is algae'",
            "goal": "Ask ChatGPT what is algae",
            "candidates": [
                {"id": "c_yt", "label": "YouTube", "role": "button", "role_idx": 1, "bounds_vec": [0.1, 0.1, 0.25, 0.25, 0.5, 0.5, 0.15, 0.15, 0.02, 1.0]},
                {"id": "c_gpt", "label": "ChatGPT", "role": "button", "role_idx": 1, "bounds_vec": [0.1, 0.3, 0.25, 0.45, 0.5, 0.5, 0.15, 0.15, 0.02, 1.0]},
                {"id": "c_calc", "label": "Calculator", "role": "button", "role_idx": 1, "bounds_vec": [0.1, 0.5, 0.25, 0.65, 0.5, 0.5, 0.15, 0.15, 0.02, 1.0]},
                {"id": "c_set", "label": "Settings", "role": "button", "role_idx": 1, "bounds_vec": [0.1, 0.7, 0.25, 0.85, 0.5, 0.5, 0.15, 0.15, 0.02, 1.0]},
                {"id": "c_comp", "label": "complete", "role": "button", "role_idx": 1, "bounds_vec": [0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]},
            ],
            "target": "ChatGPT",
            "action": "OPEN_APP",
            "span": "",
        },
        # Test 2: Launcher Turn (Gemini)
        {
            "name": "Launcher: 'Ask Gemini how airplanes fly'",
            "goal": "Ask Gemini how airplanes fly",
            "candidates": [
                {"id": "c_yt", "label": "YouTube", "role": "button", "role_idx": 1, "bounds_vec": [0.1, 0.1, 0.25, 0.25, 0.5, 0.5, 0.15, 0.15, 0.02, 1.0]},
                {"id": "c_gem", "label": "Gemini", "role": "button", "role_idx": 1, "bounds_vec": [0.1, 0.3, 0.25, 0.45, 0.5, 0.5, 0.15, 0.15, 0.02, 1.0]},
                {"id": "c_zepto", "label": "Zepto", "role": "button", "role_idx": 1, "bounds_vec": [0.1, 0.5, 0.25, 0.65, 0.5, 0.5, 0.15, 0.15, 0.02, 1.0]},
                {"id": "c_comp", "label": "complete", "role": "button", "role_idx": 1, "bounds_vec": [0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]},
            ],
            "target": "Gemini",
            "action": "OPEN_APP",
            "span": "",
        },
        # Test 3: ChatGPT Chat Screen (Prompt Input & Query Span)
        {
            "name": "ChatGPT Screen: Input Box & Query Span",
            "goal": "Ask ChatGPT what is algae",
            "candidates": [
                {"id": "c_in", "label": "Reply to ChatGPT", "role": "input", "role_idx": 2, "is_editable": True, "bounds_vec": [0.48, 0.17, 0.55, 0.83, 0.5, 0.51, 0.66, 0.07, 0.046, 9.4]},
                {"id": "c_send", "label": "Send Message", "role": "button", "role_idx": 1, "bounds_vec": [0.48, 0.83, 0.55, 0.96, 0.89, 0.51, 0.13, 0.07, 0.009, 1.8]},
                {"id": "c_mic", "label": "Dictation", "role": "button", "role_idx": 1, "bounds_vec": [0.48, 0.70, 0.55, 0.83, 0.76, 0.51, 0.13, 0.07, 0.009, 1.8]},
                {"id": "c_att", "label": "Attachment", "role": "button", "role_idx": 1, "bounds_vec": [0.48, 0.03, 0.55, 0.17, 0.10, 0.51, 0.14, 0.07, 0.009, 2.0]},
                {"id": "c_comp", "label": "complete", "role": "button", "role_idx": 1, "bounds_vec": [0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]},
            ],
            "target": "Reply to ChatGPT",
            "action": "TYPE_TEXT",
            "span": "algae",
        },
        # Test 4: Gemini Chat Screen (Prompt Input & Query Span)
        {
            "name": "Gemini Screen: Input Box & Query Span",
            "goal": "Ask Gemini how airplanes fly",
            "candidates": [
                {"id": "c_in", "label": "Ask Gemini", "role": "input", "role_idx": 2, "is_editable": True, "bounds_vec": [0.46, 0.25, 0.52, 0.63, 0.44, 0.49, 0.38, 0.06, 0.02, 6.3]},
                {"id": "c_side", "label": "Open sidebar", "role": "button", "role_idx": 1, "bounds_vec": [0.05, 0.04, 0.11, 0.18, 0.11, 0.08, 0.14, 0.06, 0.008, 2.3]},
                {"id": "c_mic", "label": "Microphone", "role": "button", "role_idx": 1, "bounds_vec": [0.46, 0.63, 0.52, 0.77, 0.70, 0.49, 0.14, 0.06, 0.008, 2.3]},
                {"id": "c_live", "label": "Open Gemini Live", "role": "button", "role_idx": 1, "bounds_vec": [0.46, 0.77, 0.52, 0.90, 0.83, 0.49, 0.13, 0.06, 0.008, 2.1]},
                {"id": "c_comp", "label": "complete", "role": "button", "role_idx": 1, "bounds_vec": [0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]},
            ],
            "target": "Ask Gemini",
            "action": "TYPE_TEXT",
            "span": "airplanes fly",
        },
        # Test 5: ChatGPT Send Message (Populated Field Transition)
        {
            "name": "ChatGPT Screen: Send Button Transition",
            "goal": "Ask ChatGPT what is algae",
            "candidates": [
                {"id": "c_send", "label": "Send Message", "role": "button", "role_idx": 1, "bounds_vec": [0.48, 0.83, 0.55, 0.96, 0.89, 0.51, 0.13, 0.07, 0.009, 1.8]},
                {"id": "c_in", "label": "what is algae", "role": "input", "role_idx": 2, "is_editable": True, "is_focused": True, "bounds_vec": [0.48, 0.17, 0.55, 0.83, 0.5, 0.51, 0.66, 0.07, 0.046, 9.4]},
                {"id": "c_att", "label": "Attachment", "role": "button", "role_idx": 1, "bounds_vec": [0.48, 0.03, 0.55, 0.17, 0.10, 0.51, 0.14, 0.07, 0.009, 2.0]},
                {"id": "c_comp", "label": "complete", "role": "button", "role_idx": 1, "bounds_vec": [0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]},
            ],
            "target": "Send Message",
            "action": "CLICK",
            "span": "",
        },
    ]

    chat_passed = 0
    for ct in chat_tests:
        res = eval_single(
            model=model,
            goal=ct["goal"],
            candidates=ct["candidates"],
            expected_target=ct["target"],
            expected_action=ct["action"],
            expected_span=ct["span"],
        )
        if res["pass"]:
            chat_passed += 1
            status = "PASS"
        else:
            status = "FAIL"
        print(f"[{status:4s}] {ct['name'][:38]:38s} | Selected: '{res['selected']:18s}' | Act: {res['action']:10s} | Span: '{res['span']}'")

    print(f"\n>> AI CHAT TOTAL SCORE: {chat_passed}/{len(chat_tests)} ({chat_passed/len(chat_tests)*100:.1f}%)")

    # Summary
    print("\n" + "=" * 80)
    print("FINAL SMOKE TEST VERIFICATION VERDICT")
    print("=" * 80)
    all_ok = (anchors_passed == len(anchor_screens)) and (chat_passed == len(chat_tests))
    print(f"Part A (Anchors Invariance): {anchors_passed}/{len(anchor_screens)} {'[100% PASS - ZERO REGRESSION]' if anchors_passed == len(anchor_screens) else '[REGRESSION DETECTED]'}")
    print(f"Part B (AI Chat Universal):  {chat_passed}/{len(chat_tests)} {'[100% PASS]' if chat_passed == len(chat_tests) else '[FAILURES DETECTED]'}")
    print(f"Overall Smoke Status:        {'*** ALL GATES PASSED ***' if all_ok else '!!! FAILED !!!'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
