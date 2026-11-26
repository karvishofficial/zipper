import sys
from pathlib import Path
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
V58_DIR = SCRIPT_DIR.parent / "v58"
V52_DIR = SCRIPT_DIR.parent / "v52"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(V58_DIR))
sys.path.insert(0, str(V52_DIR))

from transformers import AutoTokenizer
from model_v59 import KarvishModelV59
from probe_v52_screen_decisions import get_test_screens

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

model = KarvishModelV59().to(DEVICE)
ckpt = torch.load(SCRIPT_DIR / "checkpoints" / "v59_best.pt", map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval()

screens = get_test_screens()
anchors_passed = 0
for idx, s in enumerate(screens, 1):
    cands = s["candidates"][:24]
    N = len(cands)
    enc = TOKENIZER(s["goal"], max_length=32, truncation=True, return_tensors="pt")
    lbl_texts = [c["label"] for c in cands]
    c_enc = TOKENIZER(lbl_texts, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
    bounds = [c.get("bounds_vec", [0.0] * 10) for c in cands]
    roles = [2 if c.get("is_editable") else (1 if c.get("role") == "button" else 0) for c in cands]
    flags = torch.zeros(1, N, 16, device=DEVICE)
    for i, c in enumerate(cands):
        role_str = str(c.get("role", "button")).lower()
        if c.get("is_editable") or "input" in role_str or "edit" in role_str:
            flags[0, i, 0] = 1.0
        if c.get("is_focused"):
            flags[0, i, 1] = 1.0
        if c.get("is_clickable", True):
            flags[0, i, 2] = 1.0
        if c.get("is_selected"):
            flags[0, i, 5] = 1.0

    with torch.no_grad():
        out = model(
            goal_ids=enc["input_ids"].to(DEVICE),
            goal_mask=enc["attention_mask"].float().to(DEVICE),
            candidate_label_ids=c_enc["input_ids"].unsqueeze(0).to(DEVICE),
            candidate_label_mask=c_enc["attention_mask"].float().unsqueeze(0).to(DEVICE),
            candidate_bounds=torch.tensor([bounds], dtype=torch.float32, device=DEVICE),
            candidate_roles=torch.tensor([roles], dtype=torch.long, device=DEVICE),
            candidate_flags=flags,
            candidate_pad_mask=torch.ones(1, N, device=DEVICE),
        )
    pred_idx = out["ranking_scores"][0].argmax().item()
    pred_lbl = lbl_texts[pred_idx]
    is_match = s["expected_target"].lower() in pred_lbl.lower()
    if is_match:
        anchors_passed += 1
    print(f"Anchor {idx:02d} | Exp: {s['expected_target'][:18]:18s} | Pred: {pred_lbl[:18]:18s} | [{'PASS' if is_match else 'FAIL'}]")

print(f"\nV59 Baseline Anchor Score: {anchors_passed}/{len(screens)} ({(anchors_passed/len(screens))*100:.1f}%)")
