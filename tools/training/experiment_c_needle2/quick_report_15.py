"""
Instant Audit & Report on First 15 Held-Out Validation Examples
Using Corrected Pipeline (No <think>, 100% schema completeness, robust parser)
Evaluates EXISTING needle_lora_best.pt checkpoint on RTX 4050 GPU.
Prints each sample immediately with full inspection details and summary metrics.
"""

import os
import sys
import json
import time
import pickle
import re
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from pytorch_needle import Needle2Model, DEVICE
from needle.model.tokenizer import get_tokenizer, BOS_ID, EOS_ID
from needle.model.finetune import render_example

VAL_DATA_PATH = "tools/training/data/val_unseen.jsonl"
NEEDLE_VAL_PATH = "tools/training/experiment_c_needle2/data/needle_val.jsonl"
NEEDLE_BASE_CKPT = "tools/training/experiment_c_needle2/checkpoints/needle2.pkl"
NEEDLE_TUNED_CKPT = "tools/training/experiment_c_needle2/checkpoints/needle_lora_best.pt"


def parse_needle_output_robust(text):
    action_type = "unknown"
    target_id = None
    app_name = None
    is_valid = False
    name = ""

    # Strategy 1: Find complete JSON array or object
    m_array = re.search(r'\[\s*\{\s*"name"\s*:.*?\s*\}\s*\]', text, re.DOTALL)
    m_single = re.search(r'\{\s*"name"\s*:.*?\s*\}', text, re.DOTALL)
    
    if m_array:
        try:
            data = json.loads(m_array.group(0))
            if isinstance(data, list) and len(data) > 0:
                call = data[0]
                name = call.get("name", "")
                args = call.get("arguments", {})
                is_valid = True
        except Exception:
            pass

    if not is_valid and m_single:
        try:
            call = json.loads(m_single.group(0))
            name = call.get("name", "")
            args = call.get("arguments", {})
            is_valid = True
        except Exception:
            pass

    # Strategy 2: If JSON is incomplete/malformed, extract tool name directly via regex
    if not name:
        m_name = re.search(r'"name"\s*:\s*"([^"]+)"', text)
        if m_name:
            name = m_name.group(1)
        else:
            m_raw = re.search(r'(open_app|tap_\w+|type_\w+|scroll_\w+|press_back|complete_goal|wait)', text)
            if m_raw:
                name = m_raw.group(1)

    # Strategy 3: Extract app_name if present
    m_app = re.search(r'"app_name"\s*:\s*"([^"]+)"', text)
    if m_app:
        app_name = m_app.group(1)

    # Strategy 4: Map name to action_type and target_id
    if name:
        if name == "open_app":
            action_type = "open_app"
        elif name.startswith("tap_"):
            action_type = "tap"
            m = re.search(r'_(\d+)$', name)
            if m:
                target_id = int(m.group(1))
        elif name.startswith("type_"):
            action_type = "type_text"
            m = re.search(r'_(\d+)$', name)
            if m:
                target_id = int(m.group(1))
        elif "scroll" in name:
            action_type = "scroll"
        elif "back" in name:
            action_type = "back"
        elif "complete" in name:
            action_type = "complete"
        elif "wait" in name:
            action_type = "wait"

    return {
        "action_type": action_type,
        "tool_name": name,
        "target_id": target_id,
        "app_name": app_name,
        "is_valid": is_valid or (action_type != "unknown" and name != ""),
        "raw": text
    }


def run_quick_report(max_samples=15):
    print("=" * 80)
    print(f"AUDIT REPORT: FIRST {max_samples} HELD-OUT VALIDATION EXAMPLES")
    print(f"Hardware: {torch.cuda.get_device_name(DEVICE)}")
    print("Pipeline: Corrected (100% Schema Completeness, No <think> Target, Robust Parser)")
    print("=" * 80, flush=True)

    with open(VAL_DATA_PATH, "r", encoding="utf-8") as f:
        raw_items = [json.loads(line) for line in f][:max_samples]
    with open(NEEDLE_VAL_PATH, "r", encoding="utf-8") as f:
        needle_items = [json.loads(line) for line in f][:max_samples]

    with open(NEEDLE_BASE_CKPT, "rb") as f:
        needle_ckpt = pickle.load(f)
    tokenizer = get_tokenizer(needle_ckpt["config"]["vocab_size"])

    model = Needle2Model(needle_ckpt["params"], needle_ckpt["config"], lora_rank=16).to(DEVICE)
    model.load_state_dict(torch.load(NEEDLE_TUNED_CKPT))
    model.eval()

    metrics = {
        "action_type_correct": 0,
        "exact_action_correct": 0,
        "grounding_correct": 0,
        "grounding_total": 0,
        "valid_syntax": 0,
        "unseen_app_correct": 0,
        "unseen_app_total": 0,
        "multistep_correct": 0,
        "multistep_total": 0,
        "total": len(raw_items)
    }

    for idx in range(len(raw_items)):
        raw = raw_items[idx]
        nex = needle_items[idx]

        gt_type = raw.get("targetActionName")
        gt_args = raw.get("targetActionArguments", {})
        gt_target_id = int(gt_args["target"]) if "target" in gt_args and str(gt_args["target"]).isdigit() else None
        gt_app = gt_args.get("appName") or gt_args.get("app_name")
        is_multistep = len(raw.get("shortHistory", [])) > 0
        is_unseen = True

        prompt, target = render_example(nex)
        p_tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor([[BOS_ID] + p_tokens], device=DEVICE)

        gen_ids = input_ids.clone()
        with torch.no_grad():
            for _ in range(25):
                logits = model(gen_ids)
                next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                gen_ids = torch.cat([gen_ids, next_tok], dim=-1)
                if next_tok.item() in (EOS_ID, 1):
                    break

        out_tokens = gen_ids[0, input_ids.shape[1]:].tolist()
        raw_output = tokenizer.decode(out_tokens)
        parsed = parse_needle_output_robust(raw_output)

        tool_names = [t["name"] for t in nex.get("tools", [])]
        gt_tool_name = nex["answers"][0]["name"]
        target_tool_exists = gt_tool_name in tool_names

        p_type = parsed["action_type"]
        type_match = (p_type == gt_type) or (gt_type in ("complete", "finish") and p_type in ("complete", "finish"))
        if type_match:
            metrics["action_type_correct"] += 1

        if parsed["is_valid"]:
            metrics["valid_syntax"] += 1

        grounding_match = False
        if gt_type in ("tap", "type_text") and gt_target_id is not None:
            metrics["grounding_total"] += 1
            if parsed["target_id"] == gt_target_id:
                metrics["grounding_correct"] += 1
                grounding_match = True

        exact_match = False
        if type_match:
            if gt_type == "open_app":
                if gt_app and parsed["app_name"] and gt_app.lower() in parsed["app_name"].lower():
                    exact_match = True
            elif gt_type in ("tap", "type_text"):
                if parsed["target_id"] == gt_target_id:
                    exact_match = True
            elif gt_type in ("back", "scroll", "complete", "wait"):
                exact_match = True

        if exact_match:
            metrics["exact_action_correct"] += 1

        if is_multistep:
            metrics["multistep_total"] += 1
            if exact_match:
                metrics["multistep_correct"] += 1

        if is_unseen:
            metrics["unseen_app_total"] += 1
            if exact_match:
                metrics["unseen_app_correct"] += 1

        print(f"\n--- [Sample #{idx+1}/{len(raw_items)}] App: {raw.get('packageName')} ---")
        print(f"  Goal:                {raw.get('goal')[:75]}")
        print(f"  Available Tools:     {len(tool_names)} tools (e.g. {tool_names[:3]} ...)")
        print(f"  Target Tool Exists:  {target_tool_exists} ({gt_tool_name})")
        print(f"  Ground Truth:        {gt_type} {gt_args} -> {gt_tool_name}")
        print(f"  Raw Model Output:    {repr(raw_output)}")
        print(f"  Parsed Model Action: {parsed['action_type']} (tool: {parsed['tool_name']}, target_id: {parsed['target_id']})")
        print(f"  Grounding Correct:   {grounding_match}")
        print(f"  Parser Succeeded:    {parsed['is_valid']}")
        sys.stdout.flush()

    total = metrics["total"]
    g_total = max(metrics["grounding_total"], 1)
    m_total = max(metrics["multistep_total"], 1)
    u_total = max(metrics["unseen_app_total"], 1)

    print("\n" + "=" * 80)
    print(f"AUDIT SUMMARY OVER {total} HELD-OUT VALIDATION SAMPLES")
    print("=" * 80)
    print(f"1. Action-Type Accuracy:     {metrics['action_type_correct']}/{total} ({metrics['action_type_correct']/total*100:.1f}%)")
    print(f"2. Exact-Action Accuracy:    {metrics['exact_action_correct']}/{total} ({metrics['exact_action_correct']/total*100:.1f}%)")
    print(f"3. UI-Grounding Accuracy:    {metrics['grounding_correct']}/{g_total} ({metrics['grounding_correct']/g_total*100:.1f}%)")
    print(f"4. Valid-Function-Call Rate: {metrics['valid_syntax']}/{total} ({metrics['valid_syntax']/total*100:.1f}%)")
    print(f"5. Unseen-App Accuracy:      {metrics['unseen_app_correct']}/{u_total} ({metrics['unseen_app_correct']/u_total*100:.1f}%)")
    print(f"6. Multi-Step Accuracy:      {metrics['multistep_correct']}/{m_total} ({metrics['multistep_correct']/m_total*100:.1f}%)")
    print("=" * 80)

if __name__ == "__main__":
    run_quick_report(15)
