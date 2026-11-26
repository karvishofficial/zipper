import sys
from pathlib import Path
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
V54_DIR = SCRIPT_DIR.parent / "v54"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(V54_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "v52"))

from model_v54 import KarvishModelV54
from model_v55 import KarvishModelV55
from transformers import AutoTokenizer
from probe_v52_screen_decisions import get_test_screens

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# 1. Load V54
m54 = KarvishModelV54().to(DEVICE)
m54_sd = torch.load(V54_DIR / "checkpoints" / "v54_universal_search_best.pt", map_location=DEVICE, weights_only=False)
m54.load_state_dict(m54_sd["model_state_dict"] if "model_state_dict" in m54_sd else m54_sd, strict=False)
m54.eval()

# 2. Load V55 Warm
m55 = KarvishModelV55().to(DEVICE)
m55_sd = torch.load(SCRIPT_DIR / "checkpoints" / "v55_warm_v54.pt", map_location=DEVICE, weights_only=False)
m55.load_state_dict(m55_sd["model_state_dict"], strict=True)
m55.eval()

screens = get_test_screens()
print("Comparing V54 vs V55-Warm on 12 Anchors:")
for idx, s in enumerate(screens, 1):
    cands = s["candidates"][:24]
    N = len(cands)
    enc = TOKENIZER(s["goal"], max_length=32, truncation=True, return_tensors="pt")
    lbl_texts = [c["label"] for c in cands]
    c_enc = TOKENIZER(lbl_texts, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
    bounds = [c.get("bounds_vec", [0.0]*10) for c in cands]
    roles = [2 if c.get("is_editable") else (1 if c.get("role") == "button" else 0) for c in cands]

    with torch.no_grad():
        out54 = m54(
            goal_ids=enc["input_ids"].to(DEVICE),
            goal_mask=enc["attention_mask"].float().to(DEVICE),
            candidate_label_ids=c_enc["input_ids"].unsqueeze(0).to(DEVICE),
            candidate_label_mask=c_enc["attention_mask"].float().unsqueeze(0).to(DEVICE),
            candidate_bounds=torch.tensor([bounds], dtype=torch.float32, device=DEVICE),
            candidate_roles=torch.tensor([roles], dtype=torch.long, device=DEVICE),
            candidate_flags=torch.zeros(1, N, 16, device=DEVICE),
            candidate_pad_mask=torch.ones(1, N, device=DEVICE),
        )
        out55 = m55(
            goal_ids=enc["input_ids"].to(DEVICE),
            goal_mask=enc["attention_mask"].float().to(DEVICE),
            candidate_label_ids=c_enc["input_ids"].unsqueeze(0).to(DEVICE),
            candidate_label_mask=c_enc["attention_mask"].float().unsqueeze(0).to(DEVICE),
            candidate_bounds=torch.tensor([bounds], dtype=torch.float32, device=DEVICE),
            candidate_roles=torch.tensor([roles], dtype=torch.long, device=DEVICE),
            candidate_flags=torch.zeros(1, N, 16, device=DEVICE),
            candidate_pad_mask=torch.ones(1, N, device=DEVICE),
        )
    
    p54 = out54["ranking_scores"][0].argmax().item()
    p55 = out55["ranking_scores"][0].argmax().item()
    p55_raw = out55["raw_ranking_scores"][0].argmax().item()
    diff = (out54["ranking_scores"] - out55["raw_ranking_scores"]).abs().max().item()
    print(f"Anchor {idx:02d} | Exp: {s['expected_target'][:15]:15s} | V54: {lbl_texts[p54][:15]:15s} | V55-Full: {lbl_texts[p55][:15]:15s} | V55-Raw: {lbl_texts[p55_raw][:15]:15s} | Raw Diff: {diff:.6e}")
