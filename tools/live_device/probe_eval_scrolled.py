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

tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = KarvishModelV52()
ckpt = torch.load("tools/training/karvish_mobile_action/v52/checkpoints/v52_best.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
model.eval()

scrolled_cands = parse_xml_candidates(Path("tools/live_device/scrolled_tomato.xml"))
cheap_13_btn = next((c for c in scrolled_cands if "5973feaa" in c["res"] and "add-button" in c["res"]), None)
if cheap_13_btn is None:
    cheap_13_btn = next((c for c in scrolled_cands if "5973feaa" in c["res"]), None)
other_cards = [c for c in scrolled_cands if "product-card" in c["res"] and c != cheap_13_btn]
scroll_down_cand = make_system_cand("c_scroll_down", "scroll down", "system")
complete_cand = make_system_cand("c_complete_goal", "complete", "button")

eval_scrolled = ([cheap_13_btn] + other_cards[:14] + [scroll_down_cand, complete_cand])[:24]

goal = "Open Zepto and add the cheapest tomato to the cart"
enc_g = tok(goal, max_length=32, truncation=True, return_tensors="pt")
g_ids, g_mask = enc_g["input_ids"], enc_g["attention_mask"].float()

lbl_texts = [c["label"] for c in eval_scrolled]
enc_c = tok(lbl_texts, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
c_lbl_ids = enc_c["input_ids"].unsqueeze(0)
c_lbl_mask = enc_c["attention_mask"].float().unsqueeze(0)
c_bounds = torch.tensor([c["bounds_vec"] for c in eval_scrolled], dtype=torch.float32).unsqueeze(0)
c_roles = torch.tensor([c["role_idx"] for c in eval_scrolled], dtype=torch.long).unsqueeze(0)
c_flags = torch.zeros(1, len(eval_scrolled), 16)
c_pad_mask = torch.ones(1, len(eval_scrolled))

with torch.no_grad():
    out = model(goal_ids=g_ids, goal_mask=g_mask, candidate_label_ids=c_lbl_ids, candidate_label_mask=c_lbl_mask, candidate_bounds=c_bounds, candidate_roles=c_roles, candidate_flags=c_flags, candidate_pad_mask=c_pad_mask)

scores = out["ranking_scores"][0].numpy()
sorted_indices = list(range(len(eval_scrolled)))
sorted_indices.sort(key=lambda idx: scores[idx], reverse=True)
print("=== EVAL SCROLLED TOP 5 ===")
for rank, idx in enumerate(sorted_indices[:5]):
    c = eval_scrolled[idx]
    print(f"#{rank+1:2d} (Score: {scores[idx]:.2f}) | {c['id']}: {c['label']} ({c.get('res')})")
