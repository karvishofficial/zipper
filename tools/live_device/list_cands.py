import sys
from pathlib import Path
sys.path.append(str(Path("tools/live_device").resolve()))
import test_tomato_screen
sys.stdout.reconfigure(encoding="utf-8")

print(f"Total candidates: {len(test_tomato_screen.candidates)}")
for i, c in enumerate(test_tomato_screen.candidates):
    cid = c.get("candidate_id")
    lbl = c.get("semantic_label")[:60]
    role = c.get("role")
    bounds = c.get("bounds")
    print(f"[{i:2d}] {cid:10s} | {role:6s} | bounds={bounds} | {lbl}")
