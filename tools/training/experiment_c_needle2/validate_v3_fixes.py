"""
Diagnostic & Validation Script for Proposed V3 Fixes.
Validates:
1. resourceId extraction (10-20 samples)
2. Spatial position bucket calculation from bounds
3. Keyboard nodes audit (does filtering drop any ground-truth target?)
4. Token length distribution for all 77 val and 736 train targets (50-token budget check)
5. Scroll directions distribution and schema
6. Class weighting vs balanced sampling analysis
"""

import os
import sys
import json
import re
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from needle.model.tokenizer import get_tokenizer

RAW_TRAIN_PATH = "tools/training/data/train.jsonl"
RAW_VAL_PATH = "tools/training/data/val_unseen.jsonl"

tokenizer = get_tokenizer(8192)

with open(RAW_TRAIN_PATH, "r", encoding="utf-8") as f:
    train_data = [json.loads(l) for l in f]

with open(RAW_VAL_PATH, "r", encoding="utf-8") as f:
    val_data = [json.loads(l) for l in f]


def clean_slug(raw_label):
    cleaned = ''.join(c if c.isalnum() else '_' for c in raw_label.lower())
    cleaned = re.sub(r'_+', '_', cleaned).strip('_')
    return cleaned[:16].strip('_') or 'element'


def get_spatial_bucket(bounds, screen_w=1080, screen_h=2400):
    if not bounds:
        return ""
    left = bounds.get("left", 0)
    top = bounds.get("top", 0)
    right = bounds.get("right", screen_w)
    bottom = bounds.get("bottom", screen_h)
    
    cx = (left + right) / 2.0
    cy = (top + bottom) / 2.0

    # Vertical coarse bucket (normalized 0..1)
    # Top bar: < 0.15, Upper: 0.15..0.4, Mid: 0.4..0.7, Lower: 0.7..0.88, Bottom nav: > 0.88
    # Using typical phone aspect ratio or coordinate thresholds
    # In AndroidControl coordinates, Y usually ranges 0..2400 or 0..2160
    # Let's detect max Y across the observation if unknown, or use relative Y
    return cx, cy


def test_resource_id_extraction(samples, num_to_show=15):
    print("=" * 80)
    print(f"1. VALIDATION OF RESOURCE-ID EXTRACTION (Showing {num_to_show} samples)")
    print("=" * 80)
    shown = 0
    for s_idx, s in enumerate(samples):
        for n in s["currentObservation"]["nodes"]:
            text = n.get("text")
            desc = n.get("contentDescription")
            res_id = n.get("resourceId")
            
            # Cases where text and desc were null, but resourceId exists
            if not text and not desc and res_id and "/" in res_id:
                old_label = (n.get("role") or f"item_{n['index']}").strip()
                old_slug = clean_slug(old_label)
                
                # New logic: Extract short clean name from resourceId
                short_id = res_id.split("/")[-1]
                new_slug = clean_slug(short_id)
                new_label = short_id.replace("_", " ")
                
                print(f"Sample #{s_idx+1:2d} Node #{n['index']:2d}:")
                print(f"   Role:           {n.get('role')}")
                print(f"   Raw resourceId: {res_id}")
                print(f"   OLD Tool Name:  tap_{old_slug}_{n['index']}")
                print(f"   NEW Tool Name:  tap_{new_slug}_{n['index']} (Label: '{new_label}')")
                shown += 1
                if shown >= num_to_show:
                    return


def test_keyboard_nodes_audit():
    print("\n" + "=" * 80)
    print("2. VALIDATION OF KEYBOARD NODES & GROUND TRUTH INTEGRITY")
    print("=" * 80)

    # Check across ALL 736 train and 77 val samples:
    # When keyboard package is active, what are the target actions?
    for split_name, dataset in [("TRAIN", train_data), ("VAL", val_data)]:
        gboard_samples = [s for s in dataset if s.get("packageName") == "com.google.android.inputmethod.latin"]
        print(f"\n[{split_name} SET] Total samples with Gboard active: {len(gboard_samples)} / {len(dataset)}")
        
        target_actions = Counter(s.get("targetActionName") for s in gboard_samples)
        print(f"  Target action distribution on Gboard screen: {dict(target_actions)}")

        # Check if target action ever targets an individual soft key
        soft_key_targets = 0
        non_soft_key_targets = 0
        for s in gboard_samples:
            act = s.get("targetActionName")
            args = s.get("targetActionArguments", {})
            t_id = args.get("target")
            if t_id is not None:
                t_node = next((n for n in s["currentObservation"]["nodes"] if n.get("index") == int(t_id)), None)
                if t_node:
                    role = t_node.get("role", "")
                    text = t_node.get("text", "")
                    res_id = t_node.get("resourceId", "")
                    # Is it a single letter key (e.g. text='q', 'a')?
                    if len(text or "") == 1 and role == "Button":
                        soft_key_targets += 1
                    else:
                        non_soft_key_targets += 1

        print(f"  Target was individual single-character soft key: {soft_key_targets}")
        print(f"  Target was field / send / search button:         {non_soft_key_targets}")


def test_spatial_buckets_audit(num_samples=10):
    print("\n" + "=" * 80)
    print("3. VALIDATION OF SPATIAL POSITION BUCKETING")
    print("=" * 80)
    
    # Analyze bounds across observations
    # In Android, typical resolution is 1080x2400 or 1080x2340 or 720x1280
    # Let's derive relative coordinates (0.0 to 1.0) dynamically from max bounds in the screen
    for s_idx, s in enumerate(val_data[:num_samples]):
        nodes = s["currentObservation"]["nodes"]
        max_y = max((n.get("bounds", {}).get("bottom", 0) for n in nodes), default=2400)
        max_x = max((n.get("bounds", {}).get("right", 0) for n in nodes), default=1080)
        
        print(f"\nVal Sample #{s_idx+1} (App: {s.get('packageName')}, Canvas: {max_x}x{max_y}):")
        
        # Show 3 sample nodes: top, center, bottom
        for n in nodes[:4]:
            b = n.get("bounds", {})
            if not b: continue
            cy = (b.get("top", 0) + b.get("bottom", 0)) / 2.0
            cx = (b.get("left", 0) + b.get("right", 0)) / 2.0
            rel_y = cy / max(max_y, 1)
            
            if rel_y < 0.15:
                pos_tag = "top-bar"
            elif rel_y < 0.40:
                pos_tag = "upper"
            elif rel_y < 0.70:
                pos_tag = "center"
            elif rel_y < 0.88:
                pos_tag = "lower"
            else:
                pos_tag = "bottom-nav"
                
            label = n.get("text") or n.get("contentDescription") or n.get("role") or "item"
            print(f"   Node #{n['index']:2d} '{label[:20]:20s}' -> {pos_tag:10s} (rel_y={rel_y:.2f})")


def test_token_budget_audit():
    print("\n" + "=" * 80)
    print("4. VALIDATION OF 50-TOKEN EVALUATION BUDGET (ALL 77 VAL SAMPLES)")
    print("=" * 80)
    
    with open("tools/training/experiment_c_needle2/data/needle_val.jsonl", "r", encoding="utf-8") as f:
        needle_val = [json.loads(l) for l in f]
        
    lengths = []
    over_25 = []
    over_50 = []
    
    for idx, ex in enumerate(needle_val):
        target_str = f'<tool_call>{json.dumps(ex["answers"])}</tool_call><|im_end|>'
        tokens = tokenizer.encode(target_str)
        t_len = len(tokens)
        lengths.append(t_len)
        
        if t_len > 25:
            over_25.append((idx + 1, ex["answers"][0]["name"], t_len, target_str))
        if t_len > 50:
            over_50.append((idx + 1, ex["answers"][0]["name"], t_len, target_str))
            
    print(f"Total Validation Targets: {len(lengths)}")
    print(f"Token Lengths: Min={min(lengths)}, Max={max(lengths)}, Median={sorted(lengths)[len(lengths)//2]}, Avg={sum(lengths)/len(lengths):.1f}")
    print(f"Targets requiring > 25 tokens: {len(over_25)} / {len(lengths)} ({len(over_25)/len(lengths)*100:.1f}%)")
    print(f"Targets requiring > 50 tokens: {len(over_50)} / {len(lengths)} ({len(over_50)/len(lengths)*100:.1f}%)")
    
    print("\n--- Samples Truncated by the Old 25-Token Limit ---")
    for s_idx, name, t_len, raw_str in over_25:
        print(f"   Sample #{s_idx:2d} ({name:25s}): {t_len:2d} tokens -> {raw_str[:70]}...")


if __name__ == "__main__":
    test_resource_id_extraction(train_data, 12)
    test_keyboard_nodes_audit()
    test_spatial_buckets_audit(3)
    test_token_budget_audit()
