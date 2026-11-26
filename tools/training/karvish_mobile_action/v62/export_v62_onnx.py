"""
KARVISH V62 — UNIFIED (K + 1) FP16 MOBILE ONNX EXPORT
Exports KarvishModelV62 (v62_best.pt) to certified mobile ONNX asset:
- Dynamic candidate length support [B, N, ...].
- Output ranking_scores has shape [B, N + 1], where index N is c_complete_goal.
- Pure neural argmax over candidates with ZERO runtime thresholds.
- Anti-wandering distractor repulsion head integrated.
- 512-D SIMD-aligned visual feature projection.
- Native half-precision weights with float32 boundary I/O for Android OnnxRuntime compatibility.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
V61_DIR = SCRIPT_DIR.parent / "v61"
V60_DIR = SCRIPT_DIR.parent / "v60"
V59_DIR = SCRIPT_DIR.parent / "v59"
V58_DIR = SCRIPT_DIR.parent / "v58"
V57_DIR = SCRIPT_DIR.parent / "v57"
V56_DIR = SCRIPT_DIR.parent / "v56"
V55_DIR = SCRIPT_DIR.parent / "v55"
V54_DIR = SCRIPT_DIR.parent / "v54"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(V61_DIR))
sys.path.insert(0, str(V60_DIR))
sys.path.insert(0, str(V59_DIR))
sys.path.insert(0, str(V58_DIR))
sys.path.insert(0, str(V57_DIR))
sys.path.insert(0, str(V56_DIR))
sys.path.insert(0, str(V55_DIR))
sys.path.insert(0, str(V54_DIR))
sys.path.insert(0, str(REPO_ROOT / "scratch"))

from model_v62 import KarvishModelV62
from export_recurrent_fp16 import DynamicScreenSelfAttn

CHECKPOINT_PATH = SCRIPT_DIR / "checkpoints" / "v62_best.pt"
EXPORT_DIR = SCRIPT_DIR / "export"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_ONNX_PATH = EXPORT_DIR / "karvish_v62_fp16.onnx"
ASSETS_DIR = REPO_ROOT / "app" / "src" / "main" / "assets" / "models"
EXPORT_MANIFEST_PATH = EXPORT_DIR / "v62_export_manifest.json"


class V62FP16MobileExportWrapper(nn.Module):
    """
    Half-precision export wrapper for Karvish V62 Android deployment.
    Accepts float32 tensors from Android OnnxRuntime, casts to FP16 internally,
    and returns float32 logits with unified (K + 1) ranking scores.
    """

    def __init__(self, model: KarvishModelV62):
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
        candidate_visual_feats: torch.Tensor, # [B, N, 512] float
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
        c_vis = candidate_visual_feats.half()
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
            candidate_visual_feats=c_vis,
            candidate_num_features=c_num,
            observable_delta=obs_delta,
            evidence_observed=ev_obs,
            h_prev=h_actual,
        )

        h_next = torch.nan_to_num(out["h_t"].float(), nan=0.0, posinf=10.0, neginf=-10.0)
        h_next = torch.clamp(h_next, min=-12.0, max=12.0)
        
        # Unified (K + 1) ranking scores: [B, N + 1]
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
            out["slip_prob"].float(),
        )


def export_and_verify_v62():
    print("=" * 80)
    print("KARVISH V62: UNIFIED (K + 1) FP16 MOBILE ONNX EXPORT")
    print(f"Source Checkpoint: {CHECKPOINT_PATH.name}")
    print("=" * 80)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {CHECKPOINT_PATH}")

    device = torch.device("cpu")
    model = KarvishModelV62().to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd, strict=False)
    model.eval()

    # Apply dynamic attention patch for dynamic candidate sequence length
    model.container_attention.screen_self_attn = DynamicScreenSelfAttn(model.container_attention.screen_self_attn)
    model.eval().half()

    wrapper = V62FP16MobileExportWrapper(model).eval()

    # Dummy Inputs for ONNX Tracing
    B, L, N, max_lbl = 1, 64, 4, 16
    dummy_inputs = (
        torch.zeros((B, L), dtype=torch.long),
        torch.ones((B, L), dtype=torch.float32),
        torch.zeros((B, N, max_lbl), dtype=torch.long),
        torch.ones((B, N, max_lbl), dtype=torch.float32),
        torch.zeros((B, N, 10), dtype=torch.float32),
        torch.zeros((B, N), dtype=torch.long),
        torch.zeros((B, N, 16), dtype=torch.float32),
        torch.ones((B, N), dtype=torch.float32),
        torch.zeros((B, N, 512), dtype=torch.float32),
        torch.zeros((B, N, 4), dtype=torch.float32),
        torch.zeros((B, 10), dtype=torch.float32),
        torch.zeros((B,), dtype=torch.float32),
        torch.zeros((B, 128), dtype=torch.float32),
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
        "candidate_visual_feats",
        "candidate_num_features",
        "observable_delta",
        "evidence_observed",
        "h_prev",
    ]

    output_names = [
        "ranking_scores",
        "action_logits",
        "span_start_logits",
        "span_end_logits",
        "stall_prob",
        "termination_prob",
        "h_t",
        "bio_slot_logits",
        "slip_prob",
    ]

    dynamic_axes = {
        "goal_ids": {0: "batch", 1: "seq_len"},
        "goal_mask": {0: "batch", 1: "seq_len"},
        "candidate_label_ids": {0: "batch", 1: "num_cands"},
        "candidate_label_mask": {0: "batch", 1: "num_cands"},
        "candidate_bounds": {0: "batch", 1: "num_cands"},
        "candidate_roles": {0: "batch", 1: "num_cands"},
        "candidate_flags": {0: "batch", 1: "num_cands"},
        "candidate_pad_mask": {0: "batch", 1: "num_cands"},
        "candidate_visual_feats": {0: "batch", 1: "num_cands"},
        "candidate_num_features": {0: "batch", 1: "num_cands"},
        "observable_delta": {0: "batch"},
        "evidence_observed": {0: "batch"},
        "h_prev": {0: "batch"},
        # Output ranking scores has dynamic shape [B, N + 1]
        "ranking_scores": {0: "batch", 1: "num_cands_plus_complete"},
        "action_logits": {0: "batch"},
        "span_start_logits": {0: "batch", 1: "seq_len"},
        "span_end_logits": {0: "batch", 1: "seq_len"},
        "stall_prob": {0: "batch"},
        "termination_prob": {0: "batch"},
        "h_t": {0: "batch"},
        "bio_slot_logits": {0: "batch", 1: "seq_len"},
        "slip_prob": {0: "batch"},
    }

    print(f">> Exporting PyTorch graph to ONNX: {OUTPUT_ONNX_PATH}...")
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

    onnx_size_mb = OUTPUT_ONNX_PATH.stat().st_size / 1e6
    print(f">> ONNX Export successful! File size: {onnx_size_mb:.2f} MB")

    # Verify ONNX model graph structure
    print(">> Validating ONNX structure with ONNX checker...")
    model_proto = onnx.load(str(OUTPUT_ONNX_PATH))
    onnx.checker.check_model(model_proto)
    print(">> ONNX model graph integrity verified: PASS.")

    # Benchmark ONNX Runtime CPU Inference Latency
    print(">> Benchmarking OnnxRuntime CPU latency (simulating mobile ARM execution)...")
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(OUTPUT_ONNX_PATH), sess_options=sess_opts, providers=["CPUExecutionProvider"])

    full_feed = {
        "goal_ids": dummy_inputs[0].numpy(),
        "goal_mask": dummy_inputs[1].numpy(),
        "candidate_label_ids": dummy_inputs[2].numpy(),
        "candidate_label_mask": dummy_inputs[3].numpy(),
        "candidate_bounds": dummy_inputs[4].numpy(),
        "candidate_roles": dummy_inputs[5].numpy(),
        "candidate_flags": dummy_inputs[6].numpy(),
        "candidate_pad_mask": dummy_inputs[7].numpy(),
        "candidate_visual_feats": dummy_inputs[8].numpy(),
        "candidate_num_features": dummy_inputs[9].numpy(),
        "observable_delta": dummy_inputs[10].numpy(),
        "evidence_observed": dummy_inputs[11].numpy(),
        "h_prev": dummy_inputs[12].numpy(),
    }
    active_in_names = {inp.name for inp in session.get_inputs()}
    feed_dict = {k: v for k, v in full_feed.items() if k in active_in_names}
    print(f">> Active Session Input Names: {sorted(list(active_in_names))}")

    # Warmup
    for _ in range(5):
        session.run(None, feed_dict)

    latencies = []
    for _ in range(30):
        t_start = time.perf_counter()
        outputs = session.run(None, feed_dict)
        latencies.append((time.perf_counter() - t_start) * 1000.0)

    median_lat = float(np.median(latencies))
    print(f">> OnnxRuntime Median Latency: {median_lat:.2f} ms (Target: < 45 ms)")
    
    # Check that output ranking scores has length N + 1
    scores_out = outputs[0]
    expected_cands = N + 1
    print(f">> Verified Output Ranking Shape: {scores_out.shape} (Expected [1, {expected_cands}]): {'PASS' if scores_out.shape[1] == expected_cands else 'FAIL'}")

    # Copy to Android assets
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    target_asset = ASSETS_DIR / "karvish_v62.onnx"
    shutil.copyfile(str(OUTPUT_ONNX_PATH), str(target_asset))
    print(f">> Copied production asset to Android: {target_asset} ({target_asset.stat().st_size / 1e6:.2f} MB)")

    manifest = {
        "version": "V62",
        "checkpoint": str(CHECKPOINT_PATH),
        "onnx_path": str(target_asset),
        "onnx_size_mb": onnx_size_mb,
        "input_names": input_names,
        "output_names": output_names,
        "latency_ms": median_lat,
        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(EXPORT_MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f">> Manifest written to {EXPORT_MANIFEST_PATH}")
    print("=" * 80)
    print("KARVISH V62 ONNX EXPORT COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    export_and_verify_v62()
