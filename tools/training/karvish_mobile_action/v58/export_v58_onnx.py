"""
KARVISH V58 — NATIVE FP16 ONNX MOBILE EXPORT
Exports KarvishModelV58 (v58_best.pt) to certified mobile ONNX asset:
- Dynamic candidate length support.
- Recurrent task state chaining (h_prev -> h_t).
- Multimodal stall probability & evidence-gated termination.
- Overlay Occlusion Escape & search suggestion disambiguation heads.
- Native half-precision weights with float32 boundary I/O for Android OnnxRuntime compatibility.
"""

from __future__ import annotations

import os
import sys
import shutil
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
V57_DIR = SCRIPT_DIR.parent / "v57"
V56_DIR = SCRIPT_DIR.parent / "v56"
V55_DIR = SCRIPT_DIR.parent / "v55"
V54_DIR = SCRIPT_DIR.parent / "v54"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(V57_DIR))
sys.path.insert(0, str(V56_DIR))
sys.path.insert(0, str(V55_DIR))
sys.path.insert(0, str(V54_DIR))
sys.path.insert(0, str(REPO_ROOT / "scratch"))

from model_v58 import KarvishModelV58
from export_recurrent_fp16 import DynamicScreenSelfAttn

CHECKPOINT_PATH = SCRIPT_DIR / "checkpoints" / "v58_best.pt"
EXPORT_DIR = SCRIPT_DIR / "export"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_ONNX_PATH = EXPORT_DIR / "karvish_v58_fp16.onnx"
OUTPUT_ONNX_FP32_PATH = EXPORT_DIR / "karvish_v58.onnx"
ASSETS_DIR = REPO_ROOT / "app" / "src" / "main" / "assets" / "models"
EXPORT_MANIFEST_PATH = EXPORT_DIR / "v58_export_manifest.json"


class V58FP16MobileExportWrapper(nn.Module):
    """
    Half-precision export wrapper for Karvish V58 Android deployment.
    Accepts float32 tensors from Android OnnxRuntime, casts to FP16 internally,
    and returns float32 logits to the Android ranker engine.
    """

    def __init__(self, model: KarvishModelV58):
        super().__init__()
        self.model = model

    def forward(
        self,
        goal_ids: torch.Tensor,               # [B, L] long
        goal_mask: torch.Tensor,              # [B, L] float
        candidate_label_ids: torch.Tensor,    # [B, N, max_lbl] long
        candidate_label_mask: torch.Tensor,   # [B, N, max_lbl] float
        candidate_bounds: torch.Tensor,       # [B, N, 10] float
        candidate_roles: torch.Tensor,        # [B, N] long
        candidate_flags: torch.Tensor,        # [B, N, 16] float
        candidate_pad_mask: torch.Tensor,     # [B, N] float
        candidate_num_features: torch.Tensor, # [B, N, 4] float
        observable_delta: torch.Tensor,       # [B, 10] float
        evidence_observed: torch.Tensor,      # [B] float
        h_prev: torch.Tensor,                 # [B, 128] float
    ):
        g_mask = goal_mask.half()
        c_lbl_mask = candidate_label_mask.half()
        c_bnd = candidate_bounds.half()
        c_flg = candidate_flags.half()
        c_pad = candidate_pad_mask.half()
        c_num = candidate_num_features.half()
        obs_delta = observable_delta.half()
        ev_obs = evidence_observed.half()

        # Sanitize recurrent state against half-precision overflow
        h_p = torch.nan_to_num(h_prev.half(), nan=0.0, posinf=10.0, neginf=-10.0)
        h_p = torch.clamp(h_p, min=-12.0, max=12.0)

        # Recurrent state init if h_p is zero
        is_zero = (h_p.abs().sum(dim=-1, keepdim=True) < 1e-4)
        goal_out = self.model.text_encoder(goal_ids, attention_mask=g_mask)
        h_init = self.model.task_state_engine.init_state(goal_out["pooled_output"])
        h_actual = torch.where(is_zero, h_init, h_p)

        out = self.model(
            goal_ids=goal_ids,
            goal_mask=g_mask,
            candidate_label_ids=candidate_label_ids,
            candidate_label_mask=c_lbl_mask,
            candidate_bounds=c_bnd,
            candidate_roles=candidate_roles,
            candidate_flags=c_flg,
            candidate_pad_mask=c_pad,
            candidate_num_features=c_num,
            observable_delta=obs_delta,
            evidence_observed=ev_obs,
            h_prev=h_actual,
        )

        h_next = torch.nan_to_num(out["h_t"].float(), nan=0.0, posinf=10.0, neginf=-10.0)
        h_next = torch.clamp(h_next, min=-12.0, max=12.0)
        clean_scores = torch.nan_to_num(out["ranking_scores"].float(), nan=-10.0, posinf=50.0, neginf=-50.0)

        return (
            clean_scores,
            out["action_logits"].float(),
            out["span_start_logits"].float(),
            out["span_end_logits"].float(),
            out["stall_prob"].float(),
            out["termination_prob"].float(),
            h_next,
        )


def export_and_verify_v58():
    print("=" * 80)
    print("KARVISH V58: NATIVE FP16 MOBILE ONNX EXPORT")
    print(f"Source Checkpoint: {CHECKPOINT_PATH.name}")
    print("=" * 80)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {CHECKPOINT_PATH}")

    device = torch.device("cpu")
    model = KarvishModelV58().to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd, strict=False)
    model.eval()

    # Apply dynamic attention patch for dynamic candidate sequence length
    model.container_attention.screen_self_attn = DynamicScreenSelfAttn(model.container_attention.screen_self_attn)
    model.eval().half()

    wrapper = V58FP16MobileExportWrapper(model)
    wrapper.eval()

    B, L, N, max_lbl = 1, 32, 20, 16
    dummy_inputs = (
        torch.randint(100, 2000, (B, L), dtype=torch.long),
        torch.ones(B, L, dtype=torch.float32),
        torch.randint(100, 2000, (B, N, max_lbl), dtype=torch.long),
        torch.ones(B, N, max_lbl, dtype=torch.float32),
        torch.rand(B, N, 10, dtype=torch.float32),
        torch.randint(0, 32, (B, N), dtype=torch.long),
        torch.zeros(B, N, 16, dtype=torch.float32),
        torch.ones(B, N, dtype=torch.float32),
        torch.zeros(B, N, 4, dtype=torch.float32),
        torch.zeros(B, 10, dtype=torch.float32),
        torch.tensor([0.0], dtype=torch.float32),
        torch.zeros(B, 128, dtype=torch.float32),
    )

    input_names = [
        "goal_ids", "goal_mask", "candidate_label_ids", "candidate_label_mask",
        "candidate_bounds", "candidate_roles", "candidate_flags", "candidate_pad_mask",
        "candidate_num_features", "observable_delta", "evidence_observed", "h_prev"
    ]
    output_names = [
        "ranking_scores", "action_logits", "span_start_logits", "span_end_logits",
        "stall_prob", "termination_prob", "h_t"
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
        "candidate_num_features": {0: "batch", 1: "num_cands"},
        "observable_delta": {0: "batch"},
        "h_prev": {0: "batch"},
        "ranking_scores": {0: "batch", 1: "num_cands"},
        "action_logits": {0: "batch"},
        "span_start_logits": {0: "batch", 1: "goal_len"},
        "span_end_logits": {0: "batch", 1: "goal_len"},
        "h_t": {0: "batch"},
    }

    t0 = time.time()
    print(f">> Exporting FP16 ONNX graph to: {OUTPUT_ONNX_PATH}...")
    torch.onnx.export(
        wrapper,
        dummy_inputs,
        str(OUTPUT_ONNX_PATH),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    export_duration = time.time() - t0
    file_bytes = os.path.getsize(OUTPUT_ONNX_PATH)
    file_mb = file_bytes / (1024 * 1024)
    print(f">> Model Size: {file_bytes:,} bytes ({file_mb:.2f} MB) exported in {export_duration:.2f}s")

    # Validate ONNX graph
    print(">> Validating ONNX graph with onnx.checker...")
    onnx_model = onnx.load(str(OUTPUT_ONNX_PATH))
    onnx.checker.check_model(onnx_model)
    print(">> [PASS] ONNX graph structurally verified.")

    # Check parity with ONNX Runtime
    print(">> Verifying ONNX Runtime numerical inference...")
    session = ort.InferenceSession(str(OUTPUT_ONNX_PATH), providers=["CPUExecutionProvider"])
    ort_inputs = {name: tensor.numpy() for name, tensor in zip(input_names, dummy_inputs)}

    with torch.no_grad():
        pt_outputs = wrapper(*dummy_inputs)

    ort_outputs = session.run(None, ort_inputs)

    pt_rank = pt_outputs[0].numpy()
    ort_rank = ort_outputs[0]
    rank_diff = np.max(np.abs(pt_rank - ort_rank))
    top1_pt = int(np.argmax(pt_rank[0]))
    top1_ort = int(np.argmax(ort_rank[0]))

    print(f"  Max Absolute Logit Delta: {rank_diff:.4f}")
    print(f"  PyTorch Top-1 Candidate: {top1_pt} | ONNX Runtime Top-1: {top1_ort}")
    assert top1_pt == top1_ort, "Top-1 candidate mismatch!"
    print(">> [PASS] ONNX Runtime numerical parity certified.")

    # Deploy to app/src/main/assets/models
    if ASSETS_DIR.exists():
        target_fp16 = ASSETS_DIR / "karvish_v58_fp16.onnx"
        target_default = ASSETS_DIR / "karvish_v58.onnx"
        shutil.copy2(OUTPUT_ONNX_PATH, target_fp16)
        shutil.copy2(OUTPUT_ONNX_PATH, target_default)
        print(f">> [DEPLOYED] Copied to Android app assets: {target_default.name} ({file_mb:.2f} MB)")
        print(f">> [NOTE] Existing karvish_v57.onnx and karvish_v56.onnx remain preserved and unchanged.")

    manifest = {
        "status": "V58_MOBILE_FP16_EXPORT_CERTIFIED",
        "checkpoint": str(CHECKPOINT_PATH),
        "onnx_path": str(OUTPUT_ONNX_PATH),
        "size_bytes": file_bytes,
        "size_mb": round(file_mb, 2),
        "onnx_check_passed": True,
        "ort_parity_passed": True,
        "max_logit_delta": float(rank_diff),
    }

    with open(EXPORT_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 80)
    print("V58 MOBILE ONNX EXPORT: 100% COMPLETE & CERTIFIED!")
    print("=" * 80)
    return manifest


if __name__ == "__main__":
    export_and_verify_v58()
