"""
KARVISH V56 — V54/V55-PRESERVING WEIGHT GRAFTING & INITIALIZATION
Transfers proven frozen weights from V54 baseline and V55 role compatibility into V56.
"""

from __future__ import annotations

import sys
import os
import torch
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
V54_DIR = SCRIPT_DIR.parent / "v54"
V55_DIR = SCRIPT_DIR.parent / "v55"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(V55_DIR))
sys.path.insert(0, str(V54_DIR))

from model_v56 import KarvishModelV56

V54_CKPT = V54_DIR / "checkpoints" / "v54_universal_search_best.pt"
V55_CKPT = V55_DIR / "checkpoints" / "v55_distilled_best.pt"
OUTPUT_CKPT = SCRIPT_DIR / "checkpoints" / "v56_warm_v54.pt"


def graft_to_v56(v54_path: Path, v55_path: Path, output_path: Path) -> KarvishModelV56:
    print("=" * 80)
    print("KARVISH V56: WEIGHT GRAFTING FROM V54 FROZEN CONTROL & V55 ADAPTERS")
    print(f"V54 Source: {v54_path}")
    print(f"V55 Source: {v55_path}")
    print("=" * 80)

    # 1. Load V54 State Dict
    ckpt_v54 = torch.load(v54_path, map_location="cpu", weights_only=False)
    v54_sd = ckpt_v54["model_state_dict"] if "model_state_dict" in ckpt_v54 else ckpt_v54

    # 2. Load V55 State Dict if available
    v55_sd = {}
    if v55_path.exists():
        ckpt_v55 = torch.load(v55_path, map_location="cpu", weights_only=False)
        v55_sd = ckpt_v55["model_state_dict"] if "model_state_dict" in ckpt_v55 else ckpt_v55

    # 3. Instantiate V56 Model
    v56_model = KarvishModelV56()
    v56_sd = v56_model.state_dict()

    transferred_params = 0
    total_v56_params = sum(p.numel() for p in v56_model.parameters())

    transferred_keys = []
    new_keys = []

    for k, v in v56_sd.items():
        # Prefer V55 if available and shape matches, else V54
        source_sd = None
        if k in v55_sd and v55_sd[k].shape == v.shape:
            source_sd = v55_sd
        elif k in v54_sd and v54_sd[k].shape == v.shape:
            source_sd = v54_sd

        if source_sd is not None:
            v56_sd[k].copy_(source_sd[k])
            transferred_keys.append(k)
            transferred_params += v.numel()
        else:
            new_keys.append(k)

    transfer_pct = (transferred_params / total_v56_params) * 100.0

    print(f"\n[TRANSFER SUMMARY]")
    print(f"Total V56 Parameters:         {total_v56_params:,}")
    print(f"Transferred from Baseline:    {transferred_params:,} ({transfer_pct:.2f}%)")
    print(f"Transferred Tensor Keys:      {len(transferred_keys)} / {len(v56_sd)}")
    print(f"New V56 Modules (Dual-Span):  {len(new_keys)} keys")
    for nk in new_keys:
        print(f"  + {nk}")

    if transfer_pct < 85.0:
        raise RuntimeError(f"FATAL: Only {transfer_pct:.2f}% transferred (< 85.0% required by AGENTS.md)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": v56_model.state_dict(),
            "transfer_stats": {
                "v54_source": str(v54_path),
                "v55_source": str(v55_path),
                "transferred_params": transferred_params,
                "total_v56_params": total_v56_params,
                "transfer_pct": transfer_pct,
            },
        },
        output_path,
    )
    print(f"\n[SUCCESS] Warm V56 Checkpoint saved to: {output_path}")
    print("=" * 80)
    return v56_model


if __name__ == "__main__":
    graft_to_v56(V54_CKPT, V55_CKPT, OUTPUT_CKPT)
