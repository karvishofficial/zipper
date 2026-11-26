import sys, torch
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, "tools/training/karvish_mobile_action/v52")
from model_v52 import KarvishModelV52
from probe_v52_screen_decisions import get_test_screens, evaluate_screen

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = KarvishModelV52().to(DEVICE)
ckpt = torch.load("tools/training/karvish_mobile_action/v52/checkpoints/v52_best.pt", map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
model.eval()

screens = get_test_screens()
passed = 0
for s in screens:
    r = evaluate_screen(model, s["name"], s["goal"], s["candidates"], s["expected_target"], s["expected_action"], s["expected_text"])
    if r["overall_pass"]: passed += 1
    print(f"{s['name']:35s} | Pass: {r['overall_pass']} | Act: {r['action_predicted']} | Target: {r['target_correct']}")
print(f"Total Passed: {passed}/{len(screens)}")
