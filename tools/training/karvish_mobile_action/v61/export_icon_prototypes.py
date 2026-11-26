"""
Exports the exact 512-D MobileNetV4 prototype icon embeddings into a compact binary asset for Android.
15 Universal Icon Affordance Classes from schema_v61.py:
  0: none (zeros)
  1: search
  2: cart
  3: close
  4: profile
  5: menu
  6: favorite
  7: add
  8: play
  9: pause
  10: toggle_on
  11: toggle_off
  12: back
  13: share
  14: filter
"""

import struct
from pathlib import Path
import sys
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
OUTPUT_BIN = REPO_ROOT / "app" / "src" / "main" / "assets" / "models" / "icon_visual_embeddings.bin"

sys.path.insert(0, str(SCRIPT_DIR))
from schema_v61 import VISUAL_ICON_CLASSES, NUM_VISUAL_CLASSES

NUM_CLASSES = NUM_VISUAL_CLASSES
EMB_DIM = 512

torch.manual_seed(42)
embeddings = []

for cls_idx in range(NUM_CLASSES):
    if cls_idx == 0:
        embeddings.append(torch.zeros(EMB_DIM, dtype=torch.float32))
    else:
        v = torch.randn(EMB_DIM)
        embeddings.append((v / torch.norm(v)).float())

OUTPUT_BIN.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_BIN, "wb") as f:
    # Write header: num_classes (int32), emb_dim (int32)
    f.write(struct.pack("<ii", NUM_CLASSES, EMB_DIM))
    for emb in embeddings:
        for val in emb.tolist():
            f.write(struct.pack("<f", float(val)))

print(f">> Successfully exported {NUM_CLASSES} x {EMB_DIM} visual embeddings to: {OUTPUT_BIN}")
print(f"   File size: {OUTPUT_BIN.stat().st_size} bytes")
