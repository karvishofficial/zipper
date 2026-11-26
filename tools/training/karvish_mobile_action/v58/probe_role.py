import torch
from pathlib import Path
from transformers import AutoTokenizer
import sys

REPO = Path("c:/Users/Karthik S/AndroidStudioProjects/zipper")
sys.path.insert(0, str(REPO / "tools/training/karvish_mobile_action/v58"))
sys.path.insert(0, str(REPO / "tools/training/karvish_mobile_action/v57"))
sys.path.insert(0, str(REPO / "tools/training/karvish_mobile_action/v56"))

from model_v58 import KarvishModelV58
TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

m58 = KarvishModelV58()
ckpt58 = torch.load(REPO / "tools/training/karvish_mobile_action/v58/checkpoints/v58_best.pt", map_location="cpu", weights_only=False)
m58.load_state_dict(ckpt58["model_state_dict"])
m58.eval()

goal = "Tap home"
cands = [
    {"label": "Back", "role": 12, "is_clickable": True, "is_editable": False},
    {"label": "Dismiss", "role": 1, "is_clickable": True, "is_editable": False},
    {"label": "Searching…", "role": 0, "is_clickable": True, "is_editable": False},
    {"label": "Search Instagram", "role": 3, "is_clickable": True, "is_editable": True},
]
N = len(cands)
enc = TOKENIZER(goal, max_length=32, truncation=True, return_tensors="pt")
lbl_texts = [c["label"] for c in cands]
c_enc = TOKENIZER(lbl_texts, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
flags = torch.zeros(1, N, 16)
for i, c in enumerate(cands):
    if c.get("is_editable"):
        flags[0, i, 0] = 1.0
    if c.get("is_clickable"):
        flags[0, i, 2] = 1.0

with torch.no_grad():
    out = m58(
        goal_ids=enc["input_ids"],
        goal_mask=enc["attention_mask"].float(),
        candidate_label_ids=c_enc["input_ids"].unsqueeze(0),
        candidate_label_mask=c_enc["attention_mask"].float().unsqueeze(0),
        candidate_bounds=torch.zeros(1, N, 10),
        candidate_roles=torch.tensor([[c["role"] for c in cands]], dtype=torch.long),
        candidate_flags=flags,
        candidate_pad_mask=torch.ones(1, N),
    )

scores = out["ranking_scores"][0]
print("Scores with role 12 for Back and role 3 for Search Instagram:")
for i, c in enumerate(cands):
    print(f"  {c['label']:20s} (role {c['role']}): {scores[i].item():.2f}")
