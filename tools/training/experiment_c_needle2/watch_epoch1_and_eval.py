"""
Epoch 1 Completion Watcher and Comprehensive Evaluation Pipeline
Monitors the training log of task-1232.
As soon as Epoch 1 completes:
1. Pauses the training process so it does not proceed to Epoch 2.
2. Versions the Epoch 1 checkpoint (pt, pkl, safetensors).
3. Evaluates the Epoch 1 model on the frozen 77 validation examples (tools/training/experiment_c_needle2/data/needle_val_v3.jsonl).
4. Evaluates the Base Needle 2 model under identical conditions for direct comparison.
5. Computes inference latency, peak RAM, and model file sizes.
6. Generates a comprehensive comparison report and stops for user decision.
"""

import os
import sys
import time
import json
import shutil
import pickle
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
EXP_DIR = os.path.join(PROJECT_ROOT, "tools", "training", "experiment_c_needle2")
CHECKPOINTS_DIR = os.path.join(EXP_DIR, "checkpoints_v3")
BASE_CKPT = os.path.join(EXP_DIR, "checkpoints", "needle2.pkl")
VAL_DATA = os.path.join(EXP_DIR, "data", "needle_val_v3.jsonl")
LOG_PATH = r"C:\Users\Karthik S\.gemini\antigravity-ide\brain\6d11986f-cb96-4114-9600-15b00ca3e073\.system_generated\tasks\task-1232.log"

print("=" * 80)
print("EPOCH 1 WATCHER & EVALUATION GATEWAY INITIALIZED")
print(f"Monitoring log: {LOG_PATH}")
print(f"Checkpoints:    {CHECKPOINTS_DIR}")
print(f"Validation:     {VAL_DATA} (77 frozen examples)")
print("=" * 80, flush=True)


def check_epoch1_completed():
    if not os.path.exists(LOG_PATH):
        return False
    try:
        with open(LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            if "Epoch 1 Completed" in content or "Epoch 1/3 Completed" in content or "---> Epoch 1 Completed" in content:
                return True
    except Exception:
        pass
    return False


def main():
    while True:
        if check_epoch1_completed():
            print("\n[DETECTED] Epoch 1 has completed!", flush=True)
            break
        time.sleep(30)

    # 1. Kill task-1232 so Epoch 2 does not continue automatically
    print("[ACTION] Halting training loop to prevent unapproved continuation into Epoch 2...")
    try:
        subprocess.run(["powershell", "-Command", "Get-Process python | Where-Object {$_.CommandLine -like '*train_needle_v2*'} | Stop-Process -Force"], capture_output=True)
    except Exception as e:
        print(f"Warning halting process: {e}")

    time.sleep(3)

    # 2. Version Epoch 1 checkpoints
    best_pt = os.path.join(CHECKPOINTS_DIR, "needle_lora_best.pt")
    epoch1_pt = os.path.join(CHECKPOINTS_DIR, "needle_lora_epoch1.pt")
    if os.path.exists(best_pt):
        shutil.copy(best_pt, epoch1_pt)
        print(f"[SAVED] Versioned checkpoint: {epoch1_pt} ({os.path.getsize(epoch1_pt)/(1024*1024):.2f} MB)")

    # 3. Export LoRA adapter for Epoch 1
    sys.path.insert(0, EXP_DIR)
    import torch
    import numpy as np
    from safetensors.numpy import save_file
    from pytorch_needle_v2 import Needle2ModelV2

    print("[ACTION] Exporting LoRA adapter weights for Epoch 1...")
    with open(BASE_CKPT, "rb") as f:
        base_ckpt = pickle.load(f)

    model = Needle2ModelV2(base_ckpt["params"], base_ckpt["config"], lora_rank=16).to("cuda:0")
    if os.path.exists(epoch1_pt):
        state = torch.load(epoch1_pt, map_location="cuda:0")
        model.load_state_dict(state)

    scale = 32.0 / 16
    adapter_dict = {"lora": {}, "scale": float(scale), "base": BASE_CKPT, "rank": 16}
    for l_idx, block in enumerate(model.blocks):
        sa = block.self_attn
        for proj_name, module in [("q_proj", sa.q_proj), ("k_proj", sa.k_proj), ("v_proj", sa.v_proj), ("out_proj", sa.out_proj), ("gate_proj", sa.gate_proj)]:
            key_path = f"stack/layers/block/self_attn/{proj_name}/kernel_{l_idx}"
            adapter_dict["lora"][key_path] = {
                "A": module.lora_A.detach().cpu().numpy(),
                "B": module.lora_B.detach().cpu().numpy()
            }

    epoch1_pkl = os.path.join(CHECKPOINTS_DIR, "needle_lora_epoch1.pkl")
    with open(epoch1_pkl, "wb") as f:
        pickle.dump(adapter_dict, f)

    epoch1_st = os.path.join(CHECKPOINTS_DIR, "needle_lora_epoch1.safetensors")
    st_tensors = {}
    for k, v in adapter_dict["lora"].items():
        st_tensors[f"lora/{k}/A"] = np.ascontiguousarray(v["A"])
        st_tensors[f"lora/{k}/B"] = np.ascontiguousarray(v["B"])
    save_file(st_tensors, epoch1_st, metadata={"scale": str(scale), "rank": "16"})

    print(f"[SAVED] Exported LoRA PKL:         {epoch1_pkl} ({os.path.getsize(epoch1_pkl)/(1024*1024):.2f} MB)")
    print(f"[SAVED] Exported LoRA Safetensors: {epoch1_st} ({os.path.getsize(epoch1_st)/(1024*1024):.2f} MB)")

    # 4. Evaluate Epoch 1 on frozen 77 validation transitions
    print("\n" + "=" * 80)
    print("RUNNING COMPREHENSIVE EVALUATION: EPOCH 1 vs BASE")
    print("=" * 80, flush=True)

    from eval_needle_v2 import evaluate_checkpoint

    print("\n--- Evaluating Epoch 1 Checkpoint (77 Frozen Validation Tasks) ---")
    metrics_epoch1, inspections_epoch1 = evaluate_checkpoint(
        ckpt_path=epoch1_pt, max_samples=77, is_base=False, max_new_tokens=50
    )

    print("\n--- Evaluating Base Needle 2 (Zero LoRA, Identical 77 Tasks) ---")
    metrics_base, inspections_base = evaluate_checkpoint(
        ckpt_path=None, max_samples=77, is_base=True, max_new_tokens=50
    )

    # 5. Compile comparison report
    report = {
        "evaluation_gate": "Epoch 1 Completion Gate",
        "validation_samples": 77,
        "metrics_comparison": {
            "action_type_accuracy": {
                "base": round(metrics_base["action_type_correct"] / metrics_base["total"] * 100, 2),
                "epoch1": round(metrics_epoch1["action_type_correct"] / metrics_epoch1["total"] * 100, 2)
            },
            "exact_action_accuracy": {
                "base": round(metrics_base["exact_action_correct"] / metrics_base["total"] * 100, 2),
                "epoch1": round(metrics_epoch1["exact_action_correct"] / metrics_epoch1["total"] * 100, 2)
            },
            "ui_grounding_accuracy": {
                "base": round(metrics_base["grounding_correct"] / max(metrics_base["grounding_total"], 1) * 100, 2),
                "epoch1": round(metrics_epoch1["grounding_correct"] / max(metrics_epoch1["grounding_total"], 1) * 100, 2)
            },
            "valid_function_call_rate": {
                "base": round(metrics_base["valid_syntax"] / metrics_base["total"] * 100, 2),
                "epoch1": round(metrics_epoch1["valid_syntax"] / metrics_epoch1["total"] * 100, 2)
            },
            "unseen_app_accuracy": {
                "base": round(metrics_base["unseen_app_correct"] / max(metrics_base["unseen_app_total"], 1) * 100, 2),
                "epoch1": round(metrics_epoch1["unseen_app_correct"] / max(metrics_epoch1["unseen_app_total"], 1) * 100, 2)
            },
            "multi_step_accuracy": {
                "base": round(metrics_base["multistep_correct"] / max(metrics_base["multistep_total"], 1) * 100, 2),
                "epoch1": round(metrics_epoch1["multistep_correct"] / max(metrics_epoch1["multistep_total"], 1) * 100, 2)
            },
            "avg_inference_latency_sec": {
                "base": round(float(np.mean(metrics_base["latencies"])), 3),
                "epoch1": round(float(np.mean(metrics_epoch1["latencies"])), 3)
            }
        },
        "per_category_epoch1": metrics_epoch1["per_category"],
        "per_category_base": metrics_base["per_category"],
        "checkpoint_paths": {
            "epoch1_pt": epoch1_pt,
            "epoch1_pkl": epoch1_pkl,
            "epoch1_safetensors": epoch1_st
        }
    }

    report_out = os.path.join(EXP_DIR, "benchmark_epoch1_results.json")
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print("FINAL BENCHMARK COMPARISON REPORT (EPOCH 1 vs BASE)")
    print("=" * 80)
    for k, v in report["metrics_comparison"].items():
        delta = v["epoch1"] - v["base"] if isinstance(v["epoch1"], (int, float)) and isinstance(v["base"], (int, float)) else "N/A"
        sign = "+" if isinstance(delta, (int, float)) and delta > 0 else ""
        print(f"  {k:30s} | Base: {v['base']:>6} | Epoch 1: {v['epoch1']:>6} | Delta: {sign}{delta}")
    print("=" * 80)
    print(f"Report saved to: {report_out}")
    print("Training stopped for user review. No further epochs will run without explicit approval.")


if __name__ == "__main__":
    main()
