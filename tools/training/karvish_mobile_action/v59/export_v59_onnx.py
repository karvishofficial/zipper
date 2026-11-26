"""
KARVISH V59 — NATIVE FP16 ONNX MOBILE EXPORT
Exports KarvishModelV59 (v59_best.pt) to certified mobile ONNX asset:
- Dynamic candidate length support.
- Recurrent task state chaining (h_prev -> h_t).
- Multimodal stall probability & evidence-gated termination.
- Token-level BIO slot classification (15 tags) for native on-device slot filling.
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
V58_DIR = SCRIPT_DIR.parent / "v58"
V57_DIR = SCRIPT_DIR.parent / "v57"
V56_DIR = SCRIPT_DIR.parent / "v56"
V55_DIR = SCRIPT_DIR.parent / "v55"
V54_DIR = SCRIPT_DIR.parent / "v54"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(V58_DIR))
sys.path.insert(0, str(V57_DIR))
sys.path.insert(0, str(V56_DIR))
sys.path.insert(0, str(V55_DIR))
sys.path.insert(0, str(V54_DIR))
sys.path.insert(0, str(REPO_ROOT / "scratch"))

from model_v59 import KarvishModelV59
from export_recurrent_fp16 import DynamicScreenSelfAttn

CHECKPOINT_PATH = SCRIPT_DIR / "checkpoints" / "v59_best.pt"
EXPORT_DIR = SCRIPT_DIR / "export"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_ONNX_PATH = EXPORT_DIR / "karvish_v59_fp16.onnx"
OUTPUT_ONNX_FP32_PATH = EXPORT_DIR / "karvish_v59.onnx"
ASSETS_DIR = REPO_ROOT / "app" / "src" / "main" / "assets" / "models"
EXPORT_MANIFEST_PATH = EXPORT_DIR / "v59_export_manifest.json"


class V59FP16MobileExportWrapper(nn.Module):
    """
    Half-precision export wrapper for Karvish V59 Android deployment.
    Accepts float32 tensors from Android OnnxRuntime, casts to FP16 internally,
    and returns float32 logits including token-level BIO slot tags to Android.
    """

    def __init__(self, model: KarvishModelV59):
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
            out["bio_slot_logits"].float(),
        )


def export_and_verify_v59():
    print("=" * 80)
    print("KARVISH V59: NATIVE FP16 MOBILE ONNX EXPORT")
    print(f"Source Checkpoint: {CHECKPOINT_PATH.name}")
    print("=" * 80)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {CHECKPOINT_PATH}")

    device = torch.device("cpu")
    model = KarvishModelV59().to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd, strict=False)
    model.eval()

    # Apply dynamic attention patch for dynamic candidate sequence length
    model.container_attention.screen_self_attn = DynamicScreenSelfAttn(model.container_attention.screen_self_attn)
    model.eval().half()

    wrapper = V59FP16MobileExportWrapper(model)
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
        "stall_prob", "termination_prob", "h_t", "bio_slot_logits"
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
        "ranking_scores": {0: "batch", 1: "num_cands"},
        "span_start_logits": {0: "batch", 1: "goal_len"},
        "span_end_logits": {0: "batch", 1: "goal_len"},
        "bio_slot_logits": {0: "batch", 1: "goal_len"},
    }

    print(">> Exporting PyTorch model to ONNX FP16...")
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

    size_mb = OUTPUT_ONNX_PATH.stat().st_size / (1024 * 1024)
    print(f">> Exported ONNX to {OUTPUT_ONNX_PATH} ({size_mb:.2f} MB)")

    # Verify ONNX model
    onnx_model = onnx.load(str(OUTPUT_ONNX_PATH))
    onnx.checker.check_model(onnx_model)
    print(">> ONNX structure verified with onnx.checker.check_model().")

    # Verify OnnxRuntime inference
    session = ort.InferenceSession(str(OUTPUT_ONNX_PATH), providers=["CPUExecutionProvider"])
    ort_inputs = {
        name: dummy_inputs[i].numpy() for i, name in enumerate(input_names)
    }

    t0 = time.perf_counter()
    ort_outputs = session.run(None, ort_inputs)
    latency_ms = (time.perf_counter() - t0) * 1000

    print(f">> OnnxRuntime inference verified. Latency: {latency_ms:.2f} ms")
    print(f"   Outputs: {len(ort_outputs)} tensors")
    print(f"   ranking_scores shape: {ort_outputs[0].shape}")
    print(f"   bio_slot_logits shape: {ort_outputs[7].shape}")

    # Copy to assets directory for Android build (both karvish_v59_fp16.onnx and karvish_v59.onnx)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    target_fp16 = ASSETS_DIR / "karvish_v59_fp16.onnx"
    target_std = ASSETS_DIR / "karvish_v59.onnx"
    shutil.copy2(OUTPUT_ONNX_PATH, target_fp16)
    shutil.copy2(OUTPUT_ONNX_PATH, target_std)
    print(f">> Copied to Android assets: {target_fp16} and {target_std} ({target_std.stat().st_size / (1024 * 1024):.2f} MB)")

    # Also save manifest
    manifest = {
        "version": "V59",
        "architecture": "KarvishModelV59",
        "source_checkpoint": str(CHECKPOINT_PATH),
        "onnx_path": str(OUTPUT_ONNX_PATH),
        "size_mb": size_mb,
        "latency_ms": latency_ms,
        "input_names": input_names,
        "output_names": output_names,
        "num_bio_tags": 15,
    }
    with open(EXPORT_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f">> Saved export manifest to {EXPORT_MANIFEST_PATH}")
    print("=" * 80)
    print("ONNX EXPORT AND VERIFICATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    export_and_verify_v59()
