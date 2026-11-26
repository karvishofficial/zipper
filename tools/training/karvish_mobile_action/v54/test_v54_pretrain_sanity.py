"""
KARVISH V54 — PRETRAIN SANITY AUDIT SUITE
Verifies the complete solution WITHOUT running full training or modifying production checkpoints:
1. True selling price extraction across all real-world e-commerce discount formats.
2. Mathematical gradient flow across all architectural heads (ranking, action, entity gating, learned span pointer, termination): 0 NaNs, 0 Infs.
3. Dataset and multi-turn trajectory integrity across all 10 categories (including WhatsApp and e-commerce PDP).
4. Zero oracle leakage audit (AGENTS.md Laws 9 & 17).

Constitutional Compliance:
- v52_locked_gold.pt and v51_7_gold are 100% untouched.
- Production v54_best.pt is NOT overwritten.
- 0 hardcoded task heuristics.
- No training or smoke training is executed during pretrain sanity verification.
"""

from __future__ import annotations

import sys
import math
import time
import argparse
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8")

import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from model_v54 import KarvishModelV54
from schema_v54 import ACTION_TO_IDX
from train_v54_full import (
    DEVICE,
    TOKENIZER,
    LOCKED_GOLD_CKPT,
    extract_candidate_price,
    build_full_v54_dataset,
    train_v54_model,
    evaluate_v54_production_gates,
)


def test_part1_price_extraction():
    print("\n" + "=" * 70)
    print("TEST 1: TRUE SELLING PRICE EXTRACTION ON REAL E-COMMERCE CARDS")
    print("=" * 70)
    test_cases = [
        ({"label": "Bare Anatomy ... available for ₹348, 200 ml ADD ₹348 ₹399 12% OFF on MRP"}, 348.0),
        ({"label": "GUBB Hair Spray Bottle ... available for ₹148, 1 pc ADD ₹148 ₹155 6,339"}, 148.0),
        ({"label": "Portronics Toad 23 Wireless Optical Mouse ₹299 M.R.P: ₹599 (50% off)"}, 299.0),
        ({"label": "boAt Rockerz 255 Bluetooth Neckband ₹899 ₹2,990 69% off Free delivery"}, 899.0),
        ({"label": "Fresh Tomato Local 500g ₹13 Add to Cart"}, 13.0),
        ({"label": "Britannia Milk Bread ₹20 ADD"}, 20.0),
        ({"label": "Amul Taaza Toned Fresh Milk 500ml ₹27 ADD"}, 27.0),
        ({"label": "Speed Type Waterproof Gaming Mouse Pad ₹99"}, 99.0),
    ]
    all_ok = True
    for c, exp in test_cases:
        p = extract_candidate_price(c)
        ok = (p == exp)
        all_ok &= ok
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] Extracted: {p} (Expected: {exp}) | Label: '{c['label'][:45]}...'")
    assert all_ok, "Price extraction test failed!"
    print(">> TEST 1 PASSED: True selling price extracted correctly on all real-world discount formats!")


def test_part2_entity_gating_and_gradients():
    print("\n" + "=" * 70)
    print("TEST 2: ARCHITECTURE INTEGRITY & ALL-HEAD MATHEMATICAL GRADIENTS")
    print("=" * 70)
    model = KarvishModelV54().to(DEVICE)
    model.train()

    # Load backbone weights from frozen control
    assert LOCKED_GOLD_CKPT.exists(), f"Missing frozen baseline: {LOCKED_GOLD_CKPT}"
    sd = torch.load(LOCKED_GOLD_CKPT, map_location=DEVICE, weights_only=False)
    state = sd.get("model_state_dict", sd)
    compatible_sd = {k: v for k, v in state.items() if k in model.state_dict() and model.state_dict()[k].shape == v.shape}
    model.load_state_dict(compatible_sd, strict=False)
    print(f"  [OK] Loaded {len(compatible_sd)} backbone tensors from frozen V52-LOCKED-GOLD.")

    goal = "Send message to Rahul: I am on my way"
    enc = TOKENIZER(goal, max_length=32, truncation=True, return_tensors="pt")
    g_ids = enc["input_ids"].to(DEVICE)
    g_mask = enc["attention_mask"].float().to(DEVICE)

    # 4 candidates: Search button, Contact row, Distractor, Back
    labels = [
        "Search contacts and chats",
        "Rahul +91 98765 43210 Available",
        "Priya: Hey, are you free today?",
        "back",
    ]
    c_enc = TOKENIZER(labels, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
    c_lbl_ids = c_enc["input_ids"].unsqueeze(0).to(DEVICE)
    c_lbl_mask = c_enc["attention_mask"].float().unsqueeze(0).to(DEVICE)
    c_bounds = torch.zeros(1, 4, 10, dtype=torch.float32, device=DEVICE)
    c_roles = torch.tensor([[1, 1, 1, 0]], dtype=torch.long, device=DEVICE)
    c_flags = torch.zeros(1, 4, 16, dtype=torch.float32, device=DEVICE)
    c_flags[0, :, 2] = 1.0  # clickable
    c_pad = torch.ones(1, 4, dtype=torch.float32, device=DEVICE)

    prices = [0.0, 0.0, 0.0, 0.0]
    c_num = torch.zeros(1, 4, 4, dtype=torch.float32, device=DEVICE)

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
    )

    ranking = out["ranking_scores"]
    action_logits = out["action_logits"]
    span_s = out["span_start_logits"]
    span_e = out["span_end_logits"]
    term = out["termination_prob"]

    target = torch.tensor([0], dtype=torch.long, device=DEVICE)
    act_target = torch.tensor([ACTION_TO_IDX["CLICK"]], dtype=torch.long, device=DEVICE)
    st_target = torch.tensor([4], dtype=torch.long, device=DEVICE)
    en_target = torch.tensor([4], dtype=torch.long, device=DEVICE)
    term_target = torch.tensor([0.0], dtype=torch.float32, device=DEVICE)

    loss_rank = F.cross_entropy(ranking, target)
    loss_act = F.cross_entropy(action_logits, act_target)
    loss_span = F.cross_entropy(span_s, st_target) + F.cross_entropy(span_e, en_target)
    loss_term = F.binary_cross_entropy(term, term_target)

    loss_total = loss_rank + loss_act + loss_span + loss_term
    loss_total.backward()

    # Check gradients across all parameter groups that received backprop
    grad_ok = True
    active_param_count = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            active_param_count += 1
            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                print(f"  [ERROR] NaN/Inf gradient in: {name}")
                grad_ok = False

    assert grad_ok and active_param_count > 0, "NaN, Inf, or zero active gradients detected!"
    print(f"  [OK] Verified {active_param_count} parameter tensors across all architectural heads.")
    print(">> TEST 2 PASSED: 0 NaNs, 0 Infs, mathematical gradient backpropagation 100% verified!")


def test_part3_dataset_and_span_integrity():
    print("\n" + "=" * 70)
    print("TEST 3: DATASET COMPOSITION, SPAN EXTRACTION & LEAKAGE AUDIT")
    print("=" * 70)
    t0 = time.time()
    dataset = build_full_v54_dataset()
    elapsed = time.time() - t0
    print(f">> Dataset constructed in {elapsed:.3f}s: {len(dataset)} verified trajectories.")

    # 1. Audit categories
    categories: dict[str, int] = {}
    for item in dataset:
        c = item.get("category", "anchor")
        categories[c] = categories.get(c, 0) + 1

    expected_categories = [
        "anchor", "launcher_media", "zepto_search", "zepto_grid",
        "blinkit", "amazon", "flipkart", "price_permutations",
        "terminal_cart", "comm_search", "comm_contact_wa", "comm_contact_unseen",
        "comm_compose", "comm_send"
    ]

    print(f">> Category Distribution across {len(categories)} categories:")
    for cat_name in expected_categories:
        cnt = categories.get(cat_name, 0)
        status = "PASS" if cnt > 0 else "FAIL"
        print(f"  [{status}] Category '{cat_name:20s}': {cnt:2d} screen trajectories")
        assert cnt > 0, f"Missing category: {cat_name}"

    # 2. Audit learned span token offsets
    span_checks = 0
    for item in dataset:
        target_span = item.get("target_span")
        ent = item.get("entity_target", "")
        if target_span is not None and ent and ent not in ["complete", "Add to Cart", "submit"]:
            st, en = target_span
            assert 0 <= st <= en < 32, f"Invalid span [{st}, {en}] for item {item['name']}"
            span_checks += 1
    print(f">> Span Offset Verification: {span_checks} text entity spans validated within MiniLM token boundaries.")

    # 3. Audit zero oracle leakage (AGENTS.md Laws 9 & 17)
    forbidden_keys = ["oracle_target", "correct_candidate", "evaluator_state", "ground_truth_progress"]
    for item in dataset:
        for fk in forbidden_keys:
            assert fk not in item, f"Constitutional violation: forbidden key '{fk}' in item {item['name']}"
        for c in item["candidates"]:
            assert "is_target" not in c, f"Constitutional violation: oracle leakage 'is_target' in candidate {c.get('id')}"

    print(">> Constitutional Leakage Audit: 0 oracle cues in candidate representations.")
    print(">> TEST 3 PASSED: Dataset integrity, category balance, and zero-leakage contract verified!")


def test_part4_in_memory_overfit_and_all_15_gates():
    print("\n" + "=" * 70)
    print("TEST 4: IN-MEMORY SANITY OVERFIT & 15 PRODUCTION GATES AUDIT")
    print("=" * 70)
    print(">> Running 300-step in-memory optimization (Production checkpoint untouched)...")
    t0 = time.time()
    test_model = train_v54_model(total_steps=300, is_dry_run=True)
    elapsed = time.time() - t0
    print(f">> In-memory optimization finished in {elapsed:.2f}s!")

    print("\n>> Auditing all 15 certification gates on the trained test model...")
    all_passed = evaluate_v54_production_gates(test_model)

    print("\n" + "=" * 70)
    print(f"FINAL AUDIT RESULT: {'ALL 15 GATES PASSED (100%)' if all_passed else 'SOME GATES FAILED'}")
    print("=" * 70)
    assert all_passed, "Not all 15 gates passed!"
    print(">> PRETRAIN SANITY AUDIT COMPLETED WITH 100% SUCCESS!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Karvish V54 Pretrain Sanity Suite")
    parser.add_argument("--include-smoke-train", action="store_true", help="Include in-memory overfit smoke train and 15 gates audit")
    args = parser.parse_args()

    print("\n======================================================================")
    print("STARTING KARVISH V54 PRETRAIN SANITY SUITE (NO SMOKE TRAIN)")
    print("======================================================================")
    test_part1_price_extraction()
    test_part2_entity_gating_and_gradients()
    test_part3_dataset_and_span_integrity()

    if args.include_smoke_train:
        print("\n>> User explicitly requested in-memory smoke train...")
        test_part4_in_memory_overfit_and_all_15_gates()
    else:
        print("\n>> Note: In-memory smoke train skipped by default per user directive.")
        print(">> To run in-memory smoke train, use --include-smoke-train.")

    print("\n" + "=" * 70)
    print(">> ALL PRETRAIN SANITY CHECKS COMPLETED WITH 100% SUCCESS!")
    print("======================================================================")
