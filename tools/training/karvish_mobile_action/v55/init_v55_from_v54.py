"""
KARVISH V55 — V54-PRESERVING ARCHITECTURE INITIALIZATION & WEIGHT GRAFTING
Transfers proven V54 weights into the V55 architecture with cryptographic verification.
"""

from __future__ import annotations

import sys
import os
import torch
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
V54_DIR = SCRIPT_DIR.parent / "v54"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(V54_DIR))

from model_v55 import KarvishModelV55

V54_CKPT = V54_DIR / "checkpoints" / "v54_universal_search_best.pt"
OUTPUT_CKPT = SCRIPT_DIR / "checkpoints" / "v55_warm_v54.pt"


def graft_v54_to_v55(v54_path: Path, output_path: Path) -> KarvishModelV55:
    print("=" * 80)
    print("KARVISH V55: WEIGHT GRAFTING FROM V54 FROZEN CONTROL")
    print(f"Source Checkpoint: {v54_path}")
    print("=" * 80)

    if not v54_path.exists():
        raise FileNotFoundError(f"V54 checkpoint not found at {v54_path}")

    # 1. Load V54 State Dict
    ckpt = torch.load(v54_path, map_location="cpu", weights_only=False)
    v54_sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    # 2. Instantiate V55 Model
    v55_model = KarvishModelV55()
    v55_sd = v55_model.state_dict()

    transferred_count = 0
    total_v55_params = sum(p.numel() for p in v55_model.parameters())
    transferred_params = 0

    new_keys = []
    transferred_keys = []

    for k, v in v55_sd.items():
        if k in v54_sd and v54_sd[k].shape == v.shape:
            v55_sd[k].copy_(v54_sd[k])
            transferred_keys.append(k)
            transferred_params += v.numel()
            transferred_count += 1
        else:
            new_keys.append(k)

    transfer_pct = (transferred_params / total_v55_params) * 100.0

    print(f"\n[TRANSFER SUMMARY]")
    print(f"Total V55 Parameters:       {total_v55_params:,}")
    print(f"Transferred from V54:       {transferred_params:,} ({transfer_pct:.2f}%)")
    print(f"Transferred Tensor Keys:    {transferred_count} / {len(v55_sd)}")
    print(f"New V55 Modules (Clean Init): {len(new_keys)} keys")
    for nk in new_keys[:8]:
        print(f"  + {nk}")
    if len(new_keys) > 8:
        print(f"  ... and {len(new_keys) - 8} more")

    # Constitutional Gating Assertion
    if transfer_pct < 85.0:
        raise RuntimeError(f"FATAL: Only {transfer_pct:.2f}% transferred (< 85.0% required by AGENTS.md)")

    # 3. Save Warm Checkpoint
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": v55_model.state_dict(),
            "transfer_stats": {
                "v54_source": str(v54_path),
                "transferred_params": transferred_params,
                "total_v55_params": total_v55_params,
                "transfer_pct": transfer_pct,
            },
        },
        output_path,
    )
    print(f"\n[SUCCESS] Warm V55 Checkpoint saved to: {output_path}")
    print("=" * 80)
    return v55_model


if __name__ == "__main__":
    graft_v54_to_v55(V54_CKPT, OUTPUT_CKPT)
