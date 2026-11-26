import torch
from model_v54 import KarvishModelV54
from train_v54_full import TOKENIZER, DEVICE, BEST_V54_CKPT
from test_v54_battle_hardened import parse_live_screens_from_log, evaluate_screen

# Test directly on BEST_V54_CKPT to see base behavior vs dry run
screens = parse_live_screens_from_log()
switcher = next(s for s in screens if any("active apps" in c["lbl"].lower() for c in s["candidates"]) and "zepto" in s["goal"].lower())

print("Switcher candidates count:", len(switcher["candidates"]))
for i, c in enumerate(switcher["candidates"]):
    print(f"  {i}: {c['lbl']}")
