"""
KARVISH V52 — PHASE 9: ONNX MOBILE ASSET EXPORT & QUANTIZATION
Exports KarvishModelV52 to standalone ONNX runtime graph for Android deployment.

Governing Laws:
- AGENTS.md Law 34: Exported learned asset size must be strictly < 50,000,000 bytes.
- AGENTS.md Law 35: Target decision latency < 300 ms, RAM < 250 MB.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from model_v52 import KarvishModelV52

EXPORT_DIR = SCRIPT_DIR / "export"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
ONNX_EXPORT_PATH = EXPORT_DIR / "karvish_v52.onnx"
EXPORT_MANIFEST_PATH = EXPORT_DIR / "v52_export_manifest.json"


class V52ExportWrapper(nn.Module):
    """
    Slim inference wrapper for ONNX export.
    Exposes primary mobile decision outputs: ranking_scores, action_logits, span_start, span_end.
    """

    def __init__(self, model: KarvishModelV52):
        super().__init__()
        self.model = model

    def forward(
        self,
        goal_ids: torch.Tensor,
        goal_mask: torch.Tensor,
        candidate_label_ids: torch.Tensor,
        candidate_label_mask: torch.Tensor,
        candidate_bounds: torch.Tensor,
        candidate_roles: torch.Tensor,
        candidate_flags: torch.Tensor,
        candidate_pad_mask: torch.Tensor,
    ):
        out = self.model(
            goal_ids=goal_ids,
            goal_mask=goal_mask,
            candidate_label_ids=candidate_label_ids,
            candidate_label_mask=candidate_label_mask,
            candidate_bounds=candidate_bounds,
            candidate_roles=candidate_roles,
            candidate_flags=candidate_flags,
            candidate_pad_mask=candidate_pad_mask,
        )
        return (
            out["ranking_scores"],
            out["action_logits"],
            out["span_start_logits"],
            out["span_end_logits"],
            out["subgoal_logits"],
            out["progress_scalar"],
            out["termination_prob"],
        )


def export_and_verify_onnx() -> Dict[str, Any]:
    print("=" * 80)
    print("STARTING KARVISH V52 ONNX EXPORT & SIZE AUDIT...")
    print("=" * 80)

    device = torch.device("cpu")
    model = KarvishModelV52().to(device)
    model.eval()

    # Load best checkpoint if available
    ckpt_path = SCRIPT_DIR / "checkpoints" / "v52_best.pt"
    if not ckpt_path.exists():
        ckpt_path = SCRIPT_DIR / "checkpoints" / "v52_best_checkpoint.pt"
    if ckpt_path.exists():
        print(f">> Loading trained weights from {ckpt_path.name}...")
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)

    wrapper = V52ExportWrapper(model)
    wrapper.eval()

    # Dummy inputs for trace
    B = 1
    L = 32
    N = 25
    max_lbl = 16

    dummy_inputs = (
        torch.randint(1, 1000, (B, L), dtype=torch.long),
        torch.ones(B, L, dtype=torch.float32),
        torch.randint(1, 1000, (B, N, max_lbl), dtype=torch.long),
        torch.ones(B, N, max_lbl, dtype=torch.float32),
        torch.rand(B, N, 10, dtype=torch.float32),
        torch.randint(0, 18, (B, N), dtype=torch.long),
        torch.zeros(B, N, 16, dtype=torch.float32),
        torch.ones(B, N, dtype=torch.float32),
    )

    input_names = [
        "goal_ids",
        "goal_mask",
        "candidate_label_ids",
        "candidate_label_mask",
        "candidate_bounds",
        "candidate_roles",
        "candidate_flags",
        "candidate_pad_mask",
    ]

    output_names = [
        "ranking_scores",
        "action_logits",
        "span_start_logits",
        "span_end_logits",
        "subgoal_logits",
        "progress_scalar",
        "termination_prob",
    ]

    dynamic_axes = {
        "goal_ids": {0: "batch", 1: "goal_len"},
        "goal_mask": {0: "batch", 1: "goal_len"},
        "candidate_label_ids": {0: "batch", 1: "num_cands"},
        "candidate_label_mask": {0: "batch", 1: "num_cands"},
        "candidate_bounds": {0: "batch", 1: "num_cands"},
        "candidate_roles": {0: "batch", 1: "num_cands"},
        "candidate_flags": {0: "batch", 1: "num_cands"},
        "candidate_pad_mask": {0: "batch", 1: "num_cands"},
        "ranking_scores": {0: "batch", 1: "num_cands"},
        "action_logits": {0: "batch"},
    }

    t0 = time.time()
    print(f">> Exporting to ONNX: {ONNX_EXPORT_PATH}...")
    torch.onnx.export(
        wrapper,
        dummy_inputs,
        str(ONNX_EXPORT_PATH),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    export_duration = time.time() - t0

    # Constitutional Size Audit (< 50,000,000 bytes)
    file_bytes = os.path.getsize(ONNX_EXPORT_PATH)
    file_mb = file_bytes / (1024 * 1024)
    print(f">> Exported FP32 ONNX Size: {file_bytes:,} bytes ({file_mb:.2f} MB)")
    print(f">> Export Completed in {export_duration:.2f} seconds")

    # Quantize to INT8 for Mobile Deployment (< 50,000,000 bytes Law 34)
    onnx_int8_path = EXPORT_DIR / "karvish_v52_int8.onnx"
    print(f">> Applying INT8 Quantization: {onnx_int8_path.name}...")
    from onnxruntime.quantization import quantize_dynamic, QuantType
    t_q0 = time.time()
    quantize_dynamic(
        model_input=str(ONNX_EXPORT_PATH),
        model_output=str(onnx_int8_path),
        weight_type=QuantType.QInt8,
    )
    quant_duration = time.time() - t_q0
    int8_bytes = os.path.getsize(onnx_int8_path)
    int8_mb = int8_bytes / (1024 * 1024)
    print(f">> Exported INT8 ONNX Size: {int8_bytes:,} bytes ({int8_mb:.2f} MB)")
    print(f">> Quantization Completed in {quant_duration:.2f} seconds")

    assert int8_bytes < 50000000, f"INT8 asset {int8_bytes} exceeds 50,000,000 bytes limit!"
    print(f">> [LAW 34 PASSED] Final Deployable Mobile Asset: {int8_mb:.2f} MB (< 50 MB Constitutional Budget)!")

    manifest = {
        "status": "PHASE_9_ONNX_EXPORT_SUCCESS",
        "fp32_onnx_file": str(ONNX_EXPORT_PATH),
        "fp32_bytes": file_bytes,
        "fp32_mb": round(file_mb, 2),
        "int8_onnx_file": str(onnx_int8_path),
        "int8_bytes": int8_bytes,
        "int8_mb": round(int8_mb, 2),
        "constitutional_budget_bytes": 50000000,
        "law_34_compliance": True,
        "export_duration_sec": round(export_duration, 2),
        "quant_duration_sec": round(quant_duration, 2),
    }

    with open(EXPORT_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 80)
    print("PHASE 9 ONNX INT8 EXPORT QUALIFIED!")
    print("=" * 80)
    return manifest


if __name__ == "__main__":
    export_and_verify_onnx()
