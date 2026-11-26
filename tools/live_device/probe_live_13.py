import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(".").resolve()
sys.path.insert(0, str(REPO_ROOT / "tools" / "training" / "karvish_mobile_action" / "v52"))
from model_v52 import KarvishModelV52
from train_v52_full_ecommerce import parse_xml_candidates, make_system_cand
from schema_v52 import ACTION_TO_IDX

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

model = KarvishModelV52().to(DEVICE)
ckpt = torch.load("tools/training/karvish_mobile_action/v52/checkpoints/v52_best.pt", map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
model.eval()

cands = parse_xml_candidates(Path("tools/live_device/live_tomato_13.xml"))
cands = [c for c in cands if c.get("role") in ["button", "input"]]
cands.insert(0, make_system_cand("c_scroll_down", "scroll down", "system"))
cands.append(make_system_cand("c_complete_goal", "complete", "button"))
cands = cands[:32]

goal = "Open Zepto and add the cheapest tomato to the cart"
enc_g = TOKENIZER(goal, max_length=32, truncation=True, return_tensors="pt")
g_ids, g_mask = enc_g["input_ids"].to(DEVICE), enc_g["attention_mask"].float().to(DEVICE)

lbl_texts = [c["label"] for c in cands]
enc_c = TOKENIZER(lbl_texts, max_length=32, padding="max_length", truncation=True, return_tensors="pt")
c_lbl_ids = enc_c["input_ids"].unsqueeze(0).to(DEVICE)
c_lbl_mask = enc_c["attention_mask"].float().unsqueeze(0).to(DEVICE)
c_bounds = torch.tensor([c["bounds_vec"] for c in cands], dtype=torch.float32).unsqueeze(0).to(DEVICE)
c_roles = torch.tensor([c["role_idx"] for c in cands], dtype=torch.long).unsqueeze(0).to(DEVICE)
c_flags = torch.zeros(1, len(cands), 16, device=DEVICE)
for i, c in enumerate(cands):
    if c.get("is_editable"): c_flags[0, i, 0] = 1.0
    if c.get("is_clickable", True): c_flags[0, i, 2] = 1.0
c_pad_mask = torch.ones(1, len(cands), device=DEVICE)

with torch.no_grad():
    out = model(goal_ids=g_ids, goal_mask=g_mask, candidate_label_ids=c_lbl_ids, candidate_label_mask=c_lbl_mask, candidate_bounds=c_bounds, candidate_roles=c_roles, candidate_flags=c_flags, candidate_pad_mask=c_pad_mask)

scores = out["ranking_scores"][0].cpu().numpy()
sorted_indices = list(range(len(cands)))
sorted_indices.sort(key=lambda idx: scores[idx], reverse=True)

print("=== TOP 10 PREDICTIONS ON LIVE TOMATO 13 SCREEN ===")
for rank, idx in enumerate(sorted_indices[:10]):
    c = cands[idx]
    print(f"#{rank+1:2d} (Score: {scores[idx]:6.2f}) | ID: {c['id']} | Label: {c['label']} | Res: {c.get('res')}")
