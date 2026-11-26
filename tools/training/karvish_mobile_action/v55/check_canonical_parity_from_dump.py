"""
KARVISH V55 — CANONICAL INPUT & OUTPUT PARITY HARNESS
Validates the canonical input dump from Android OnDeviceRankerEngine across:
1. V54 PyTorch (v54_universal_search_best.pt)
2. V55-Warm PyTorch (v55_warm_v54.pt)
3. V54 Mobile FP16 ONNX (karvish_v54_fp16.onnx)
4. V55 Mobile FP16 ONNX (karvish_v55_fp16.onnx)

Senior Parity Gate:
- Pre-training Parity: MSE(V54, V55) < 1e-5 on identical input.
- Input Tensor Identity: Android Dump == PyTorch Input == ONNX Input.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
V54_DIR = SCRIPT_DIR.parent / "v54"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(V54_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "v52"))

from model_v54 import KarvishModelV54
from model_v55 import KarvishModelV55

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dump_from_file_or_adb(dump_path: Path | None = None) -> dict:
    if dump_path and dump_path.exists():
        with open(dump_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    # Try default local dump
    local_default = REPO_ROOT / "canonical_input_dump.json"
    if local_default.exists():
        with open(local_default, "r", encoding="utf-8") as f:
            return json.load(f)
            
    # Try pulling from connected device
    import subprocess
    adb_exe = Path.home() / "AppData" / "Local" / "Android" / "Sdk" / "platform-tools" / "adb.exe"
    if adb_exe.exists():
        try:
            print(">> Pulling canonical_input_dump.json from connected device via ADB run-as...")
            res = subprocess.run(
                [str(adb_exe), "-s", "RZCY51MQP9K", "exec-out", "run-as", "com.zipper.app", "cat", "files/canonical_input_dump.json"],
                capture_output=True,
                text=True,
                check=True
            )
            if res.stdout.strip():
                data = json.loads(res.stdout)
                with open(local_default, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                print(f">> Successfully pulled and saved to {local_default}")
                return data
        except Exception as e:
            print(f"ADB pull failed or device not ready: {e}")

    raise FileNotFoundError("canonical_input_dump.json not found locally or on device.")


def run_canonical_parity(dump_data: dict):
    print("=" * 80)
    print("KARVISH V54 <-> V55 CANONICAL INPUT & OUTPUT PARITY HARNESS")
    print(f"Goal: '{dump_data.get('goal', '')}'")
    print(f"Model Tag in Dump: {dump_data.get('modelTag', '')} | Valid Candidates: {dump_data.get('validCount', 0)}")
    print("=" * 80)

    # 1. Prepare PyTorch tensors from Dump
    goal_ids = torch.tensor([dump_data["goal_ids"]], dtype=torch.long, device=DEVICE)
    goal_mask = torch.tensor([dump_data["goal_mask"]], dtype=torch.float32, device=DEVICE)
    c_lbl_ids = torch.tensor([dump_data["candidate_label_ids"]], dtype=torch.long, device=DEVICE)
    c_lbl_mask = torch.tensor([dump_data["candidate_label_mask"]], dtype=torch.float32, device=DEVICE)
    c_bnd = torch.tensor([dump_data["candidate_bounds"]], dtype=torch.float32, device=DEVICE)
    c_roles = torch.tensor([dump_data["candidate_roles"]], dtype=torch.long, device=DEVICE)
    c_flags = torch.tensor([dump_data["candidate_flags"]], dtype=torch.float32, device=DEVICE)
    c_pad = torch.tensor([dump_data["candidate_pad_mask"]], dtype=torch.float32, device=DEVICE)
    c_num = torch.tensor([dump_data.get("candidate_num_features", [])], dtype=torch.float32, device=DEVICE)
    h_prev = torch.tensor([dump_data.get("h_prev", [0.0]*128)], dtype=torch.float32, device=DEVICE)
    c_labels = dump_data.get("candidate_labels", [])
    valid_count = dump_data["validCount"]

    N = c_lbl_ids.shape[1]
    if c_num.shape[1] != N:
        c_num = torch.zeros(1, N, 4, dtype=torch.float32, device=DEVICE)

    # 2. Load PyTorch Models
    print(">> Loading V54 PyTorch Baseline...")
    m54 = KarvishModelV54().to(DEVICE)
    sd54 = torch.load(V54_DIR / "checkpoints" / "v54_universal_search_best.pt", map_location=DEVICE, weights_only=False)
    m54.load_state_dict(sd54["model_state_dict"] if "model_state_dict" in sd54 else sd54, strict=False)
    m54.eval()

    print(">> Loading V55-Warm PyTorch Baseline...")
    m55 = KarvishModelV55().to(DEVICE)
    sd55 = torch.load(SCRIPT_DIR / "checkpoints" / "v55_warm_v54.pt", map_location=DEVICE, weights_only=False)
    m55.load_state_dict(sd55["model_state_dict"] if "model_state_dict" in sd55 else sd55, strict=True)
    m55.eval()

    # 3. Load ONNX Models
    print(">> Loading V54 FP16 ONNX Runtime Session...")
    onnx_v54_path = REPO_ROOT / "app" / "src" / "main" / "assets" / "models" / "karvish_v54_fp16.onnx"
    sess_v54 = ort.InferenceSession(str(onnx_v54_path), providers=["CPUExecutionProvider"])

    print(">> Loading V55 FP16 ONNX Runtime Session...")
    onnx_v55_path = REPO_ROOT / "app" / "src" / "main" / "assets" / "models" / "karvish_v55_fp16.onnx"
    sess_v55 = ort.InferenceSession(str(onnx_v55_path), providers=["CPUExecutionProvider"]) if onnx_v55_path.exists() else None

    # 4. PyTorch Inference
    with torch.no_grad():
        out_py54 = m54(
            goal_ids=goal_ids,
            goal_mask=goal_mask,
            candidate_label_ids=c_lbl_ids,
            candidate_label_mask=c_lbl_mask,
            candidate_bounds=c_bnd,
            candidate_roles=c_roles,
            candidate_flags=c_flags,
            candidate_pad_mask=c_pad,
            candidate_num_features=c_num,
            h_prev=h_prev,
        )
        out_py55 = m55(
            goal_ids=goal_ids,
            goal_mask=goal_mask,
            candidate_label_ids=c_lbl_ids,
            candidate_label_mask=c_lbl_mask,
            candidate_bounds=c_bnd,
            candidate_roles=c_roles,
            candidate_flags=c_flags,
            candidate_pad_mask=c_pad,
            candidate_num_features=c_num,
            h_prev=h_prev,
        )

    # 5. ONNX Inference
    onnx_inputs_54 = {
        "goal_ids": goal_ids.cpu().numpy(),
        "goal_mask": goal_mask.cpu().numpy(),
        "candidate_label_ids": c_lbl_ids.cpu().numpy(),
        "candidate_label_mask": c_lbl_mask.cpu().numpy(),
        "candidate_bounds": c_bnd.cpu().numpy(),
        "candidate_roles": c_roles.cpu().numpy(),
        "candidate_flags": c_flags.cpu().numpy(),
        "candidate_pad_mask": c_pad.cpu().numpy(),
        "candidate_num_features": c_num.cpu().numpy(),
        "h_prev": h_prev.cpu().numpy(),
    }
    out_onnx54 = sess_v54.run(None, onnx_inputs_54)

    # 6. Compare Outputs
    scores_py54 = out_py54["ranking_scores"][0, :valid_count].cpu().numpy()
    scores_py55 = out_py55["ranking_scores"][0, :valid_count].cpu().numpy()
    raw_scores_py55 = out_py55["raw_ranking_scores"][0, :valid_count].cpu().numpy()
    scores_onnx54 = out_onnx54[0][0, :valid_count]

    top_py54 = int(np.argmax(scores_py54))
    top_py55 = int(np.argmax(scores_py55))
    top_onnx54 = int(np.argmax(scores_onnx54))

    diff_py_v54_v55_raw = np.max(np.abs(scores_py54 - raw_scores_py55))
    mse_py_v54_v55_raw = np.mean((scores_py54 - raw_scores_py55) ** 2)

    diff_py_onnx_54 = np.max(np.abs(scores_py54 - scores_onnx54))

    print("\n" + "=" * 80)
    print("PARITY HARNESS VERIFICATION RESULTS:")
    print("=" * 80)
    print(f"Top Candidate (V54 PyTorch) : [{top_py54:02d}] '{c_labels[top_py54]}' (score={scores_py54[top_py54]:.4f})")
    print(f"Top Candidate (V55 PyTorch) : [{top_py55:02d}] '{c_labels[top_py55]}' (score={scores_py55[top_py55]:.4f})")
    print(f"Top Candidate (V54 ONNX)    : [{top_onnx54:02d}] '{c_labels[top_onnx54]}' (score={scores_onnx54[top_onnx54]:.4f})")
    print("-" * 80)
    print(f"Max Absolute Error | V54 PyTorch vs V55 PyTorch (Raw) : {diff_py_v54_v55_raw:.8e}")
    print(f"Mean Squared Error | V54 PyTorch vs V55 PyTorch (Raw) : {mse_py_v54_v55_raw:.8e}")
    print(f"Max Absolute Error | V54 PyTorch vs V54 ONNX (FP16)   : {diff_py_onnx_54:.8e}")
    print("-" * 80)

    if diff_py_v54_v55_raw < 1e-5:
        print(">> [RESULT: PASS] Pre-training parity gate verified: V54 and V55 have mathematical identity.")
    else:
        print(">> [RESULT: FAIL] Pre-training parity mismatch detected! Stop and fix weight transfer.")

    print("=" * 80)
    return {
        "top_py54": top_py54,
        "top_py55": top_py55,
        "top_onnx54": top_onnx54,
        "diff_py_v54_v55_raw": float(diff_py_v54_v55_raw),
        "mse_py_v54_v55_raw": float(mse_py_v54_v55_raw),
        "diff_py_onnx_54": float(diff_py_onnx_54),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check V54 vs V55 canonical parity on dumped Android input.")
    parser.add_argument("--dump-path", type=str, default=None, help="Path to canonical_input_dump.json")
    args = parser.parse_args()

    p = Path(args.dump_path) if args.dump_path else None
    data = load_dump_from_file_or_adb(p)
    run_canonical_parity(data)
