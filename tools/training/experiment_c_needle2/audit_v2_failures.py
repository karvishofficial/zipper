"""
Comprehensive Failure Audit for Needle 2 v2 across the 77 Held-Out Validation Transitions.
Inspects:
1. open_app failures (0/7)
2. type_text failures (0/7)
3. scroll failures (0/7)
4. back failures (0/6)
5. tap grounding failures (42/45 failed exact grounding)
6. wait failures (4/5 failed)
7. Training set distribution and representation audit (736 examples)
"""

import os
import sys
import json
import re
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from needle.model.tokenizer import get_tokenizer

VAL_PATH = "tools/training/experiment_c_needle2/data/needle_val.jsonl"
RAW_VAL_PATH = "tools/training/data/val_unseen.jsonl"
TRAIN_PATH = "tools/training/experiment_c_needle2/data/needle_train.jsonl"
RAW_TRAIN_PATH = "tools/training/data/train.jsonl"
BENCHMARK_PATH = "tools/training/experiment_c_needle2/benchmark_results_v2.json"

tokenizer = get_tokenizer(8192)

def audit_dataset_distribution():
    print("=" * 80)
    print("1. DATASET ACTION-TYPE AUDIT (736 Train vs 77 Val)")
    print("=" * 80)

    with open(RAW_TRAIN_PATH, "r", encoding="utf-8") as f:
        train_raw = [json.loads(l) for l in f]
    with open(RAW_VAL_PATH, "r", encoding="utf-8") as f:
        val_raw = [json.loads(l) for l in f]

    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        train_needle = [json.loads(l) for l in f]
    with open(VAL_PATH, "r", encoding="utf-8") as f:
        val_needle = [json.loads(l) for l in f]

    train_c = Counter(x.get("targetActionName") for x in train_raw)
    val_c = Counter(x.get("targetActionName") for x in val_raw)

    print(f"{'Action Type':15s} | {'Train Count':12s} | {'Train %':8s} | {'Val Count':10s} | {'Val %':8s}")
    print("-" * 65)
    for act in ["tap", "scroll", "type_text", "open_app", "wait", "back"]:
        tc = train_c.get(act, 0)
        vc = val_c.get(act, 0)
        tp = tc / len(train_raw) * 100
        vp = vc / len(val_raw) * 100
        print(f"{act:15s} | {tc:5d} / {len(train_raw):3d} | {tp:6.1f}% | {vc:3d} / {len(val_raw):2d} | {vp:6.1f}%")

    # Audit tool counts per example
    train_tool_counts = [len(x["tools"]) for x in train_needle]
    val_tool_counts = [len(x["tools"]) for x in val_needle]

    print("\n--- Available Tool Count Distribution ---")
    print(f"Train Tools per sample: Min={min(train_tool_counts)}, Max={max(train_tool_counts)}, Avg={sum(train_tool_counts)/len(train_tool_counts):.1f}")
    print(f"Val Tools per sample:   Min={min(val_tool_counts)}, Max={max(val_tool_counts)}, Avg={sum(val_tool_counts)/len(val_tool_counts):.1f}")

    # Audit target token length
    train_tgt_lens = [len(tokenizer.encode(json.dumps(x["answers"]))) for x in train_needle]
    val_tgt_lens = [len(tokenizer.encode(json.dumps(x["answers"]))) for x in val_needle]

    print("\n--- Target Token Length Distribution (Answers JSON) ---")
    print(f"Train Target Tokens: Min={min(train_tgt_lens)}, Max={max(train_tgt_lens)}, Avg={sum(train_tgt_lens)/len(train_tgt_lens):.1f}")
    print(f"Val Target Tokens:   Min={min(val_tgt_lens)}, Max={max(val_tgt_lens)}, Avg={sum(val_tgt_lens)/len(val_tgt_lens):.1f}")

    # Check scroll directions
    train_scrolls = Counter(x.get("targetActionArguments", {}).get("direction") for x in train_raw if x.get("targetActionName") == "scroll")
    val_scrolls = Counter(x.get("targetActionArguments", {}).get("direction") for x in val_raw if x.get("targetActionName") == "scroll")
    print(f"\n--- Scroll Directions ---")
    print(f"Train Scrolls: {dict(train_scrolls)}")
    print(f"Val Scrolls:   {dict(val_scrolls)}")

    # Check type_text target payloads
    type_lengths = [len(x.get("targetActionArguments", {}).get("text", "")) for x in train_raw if x.get("targetActionName") == "type_text"]
    print(f"\n--- Type Text Payloads ---")
    print(f"Train Type Strings: Min chars={min(type_lengths)}, Max chars={max(type_lengths)}, Avg={sum(type_lengths)/len(type_lengths):.1f}")

    # Check multi-step history
    hist_lengths_train = [len(x.get("shortHistory", [])) for x in train_raw]
    hist_lengths_val = [len(x.get("shortHistory", [])) for x in val_raw]
    print(f"\n--- Multi-Step History ---")
    print(f"Train Steps: 0 steps (initial)={hist_lengths_train.count(0)}, multi-step={len(hist_lengths_train)-hist_lengths_train.count(0)} (Max history={max(hist_lengths_train)})")
    print(f"Val Steps:   0 steps (initial)={hist_lengths_val.count(0)}, multi-step={len(hist_lengths_val)-hist_lengths_val.count(0)} (Max history={max(hist_lengths_val)})")


def audit_validation_failures():
    print("\n" + "=" * 80)
    print("2. DETAILED AUDIT OF VALIDATION CATEGORIES AND FAILURE MODES")
    print("=" * 80)

    with open(RAW_VAL_PATH, "r", encoding="utf-8") as f:
        val_raw = [json.loads(l) for l in f]
    with open(VAL_PATH, "r", encoding="utf-8") as f:
        val_needle = [json.loads(l) for l in f]

    # Categorize by action
    by_category = {}
    for idx, (raw, nex) in enumerate(zip(val_raw, val_needle)):
        act = raw.get("targetActionName")
        if act not in by_category:
            by_category[act] = []
        by_category[act].append((idx, raw, nex))

    for act, items in by_category.items():
        print(f"\n==================== [CATEGORY: {act.upper()} ({len(items)} samples)] ====================")
        for idx, raw, nex in items:
            tools = [t["name"] for t in nex["tools"]]
            target_tool = nex["answers"][0]
            t_name = target_tool["name"]
            t_args = target_tool.get("arguments", {})
            tgt_str = f"{t_name} {t_args}"
            tok_len = len(tokenizer.encode(f'<tool_call>{json.dumps([target_tool])}</tool_call>'))
            
            # Check if target tool is in schema
            in_schema = t_name in tools
            
            print(f"Sample #{idx+1:2d} | App: {raw.get('packageName'):32s} | Target: {tgt_str[:40]:40s} | Tokens: {tok_len:2d} | In Schema: {in_schema}")
            if act in ("open_app", "scroll", "back", "type_text", "wait") or idx in [38, 40, 44, 74]:
                print(f"    Goal: {raw.get('goal')[:75]}")
                print(f"    Total tools: {len(tools)} (first 4: {tools[:4]})")


if __name__ == "__main__":
    audit_dataset_distribution()
    audit_validation_failures()
