import torch
from pathlib import Path
from transformers import AutoTokenizer
import sys

REPO = Path("c:/Users/Karthik S/AndroidStudioProjects/zipper")
sys.path.insert(0, str(REPO / "tools/training/karvish_mobile_action/v58"))
sys.path.insert(0, str(REPO / "tools/training/karvish_mobile_action/v57"))
sys.path.insert(0, str(REPO / "tools/training/karvish_mobile_action/v56"))
sys.path.insert(0, str(REPO / "tools/training/karvish_mobile_action/v52"))

from model_v57 import KarvishModelV57
from model_v58 import KarvishModelV58
from probe_v52_screen_decisions import get_test_screens

screens = get_test_screens()
s6 = screens[5] # Anchor 6: Play Store query populated
print("Screen:", s6["name"])
print("Goal:", s6["goal"])

TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
cands = s6["candidates"]
N = len(cands)
enc = TOKENIZER(s6["goal"], max_length=32, truncation=True, return_tensors="pt")
lbl_texts = [c["label"] for c in cands]
c_enc = TOKENIZER(lbl_texts, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
bounds = [c.get("bounds_vec", [0.0] * 10) for c in cands]
roles = [2 if c.get("is_editable") else (1 if c.get("role") == "button" else 0) for c in cands]
flags = torch.zeros(1, N, 16)
for i, c in enumerate(cands):
    role_str = str(c.get("role", "button")).lower()
    if c.get("is_editable") or "input" in role_str or "edit" in role_str:
        flags[0, i, 0] = 1.0
    if c.get("is_focused"):
        flags[0, i, 1] = 1.0
    if c.get("is_clickable", True):
        flags[0, i, 2] = 1.0

# V57
m57 = KarvishModelV57()
ckpt57 = torch.load(REPO / "tools/training/karvish_mobile_action/v57/checkpoints/v57_best.pt", map_location="cpu", weights_only=False)
m57.load_state_dict(ckpt57["model_state_dict"])
m57.eval()

# V58
m58 = KarvishModelV58()
ckpt58 = torch.load(REPO / "tools/training/karvish_mobile_action/v58/checkpoints/v58_best.pt", map_location="cpu", weights_only=False)
m58.load_state_dict(ckpt58["model_state_dict"])
m58.eval()

with torch.no_grad():
    out57 = m57(
        goal_ids=enc["input_ids"],
        goal_mask=enc["attention_mask"].float(),
        candidate_label_ids=c_enc["input_ids"].unsqueeze(0),
        candidate_label_mask=c_enc["attention_mask"].float().unsqueeze(0),
        candidate_bounds=torch.tensor([bounds], dtype=torch.float32),
        candidate_roles=torch.tensor([roles], dtype=torch.long),
        candidate_flags=flags,
        candidate_pad_mask=torch.ones(1, N),
    )
    out58 = m58(
        goal_ids=enc["input_ids"],
        goal_mask=enc["attention_mask"].float(),
        candidate_label_ids=c_enc["input_ids"].unsqueeze(0),
        candidate_label_mask=c_enc["attention_mask"].float().unsqueeze(0),
        candidate_bounds=torch.tensor([bounds], dtype=torch.float32),
        candidate_roles=torch.tensor([roles], dtype=torch.long),
        candidate_flags=flags,
        candidate_pad_mask=torch.ones(1, N),
    )

print("\n--- SCORES COMPARISON ---")
for i, c in enumerate(cands):
    sc57 = out57["ranking_scores"][0, i].item()
    sc58 = out58["ranking_scores"][0, i].item()
    print(f"[{i}] {c['label'][:35]:35s} | V57: {sc57:6.2f} | V58: {sc58:6.2f}")
