import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(".").resolve()
sys.path.insert(0, str(REPO_ROOT / "tools" / "training" / "karvish_mobile_action" / "v54"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "training" / "karvish_mobile_action" / "v52"))

from model_v54 import KarvishModelV54
from schema_v54 import ACTION_TO_IDX

IDX_TO_ACTION = {v: k for k, v in ACTION_TO_IDX.items()}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
CKPT = REPO_ROOT / "tools" / "training" / "karvish_mobile_action" / "v54" / "checkpoints" / "v54_chat_smoke.pt"

# Extract candidates from chatgpt_curr.xml
root = ET.parse("tools/live_device/chatgpt_curr.xml").getroot()
parent_map = {c: p for p in root.iter() for c in p}

candidates = []
for n in root.iter("node"):
    txt = n.attrib.get("text", "").strip()
    cd = n.attrib.get("content-desc", "").strip()
    cls = n.attrib.get("class", "").split(".")[-1]
    clk = n.attrib.get("clickable") == "true"
    edt = "EditText" in cls or n.attrib.get("editable") == "true"
    
    # Label
    lbl = txt or cd
    if not lbl and edt:
        # Sibling text
        p = parent_map.get(n)
        if p is not None:
            for ch in p:
                if ch.attrib.get("content-desc"):
                    lbl = ch.attrib.get("content-desc")
                    break
    if not lbl:
        lbl = cls
        
    if clk or edt:
        role = "input" if edt else ("button" if "Button" in cls else "view")
        role_idx = 2 if edt else (1 if role == "button" else 0)
        candidates.append({
            "label": lbl,
            "role": role,
            "role_idx": role_idx,
            "is_editable": edt,
            "is_clickable": clk,
            "bounds": [0.1, 0.1, 0.2, 0.2, 0.5, 0.5, 0.1, 0.1, 0.01, 1.0]
        })

print(f"Total extracted candidates: {len(candidates)}")
for i, c in enumerate(candidates):
    print(f"[{i:2d}] {c['label']:<30} | role={c['role']} | editable={c['is_editable']}")

model = KarvishModelV54().to(DEVICE)
sd = torch.load(CKPT, map_location=DEVICE, weights_only=False)
model.load_state_dict(sd["model_state_dict"] if "model_state_dict" in sd else sd, strict=False)
model.eval()

goal = "Ask ChatGPT what is algae"
enc_g = TOKENIZER(goal, return_tensors="pt")
enc_c = TOKENIZER([c["label"] for c in candidates], padding="max_length", max_length=16, return_tensors="pt")
N = len(candidates)
b = torch.zeros(1, N, 10).to(DEVICE)
r = torch.tensor([[c["role_idx"] for c in candidates]]).to(DEVICE)
fl = torch.zeros(1, N, 16).to(DEVICE)
for i, c in enumerate(candidates):
    if c["is_editable"]: fl[0, i, 0] = 1.0
    if c["is_clickable"]: fl[0, i, 2] = 1.0
pad = torch.ones(1, N).to(DEVICE)

with torch.no_grad():
    out = model(enc_g["input_ids"].to(DEVICE), enc_g["attention_mask"].float().to(DEVICE),
                enc_c["input_ids"].unsqueeze(0).to(DEVICE), enc_c["attention_mask"].float().unsqueeze(0).to(DEVICE),
                b, r, fl, pad)

scores = out["ranking_scores"][0]
topk = scores.topk(5)
print("\nTop 5 Model Predictions:")
for rank, idx in enumerate(topk.indices):
    c = candidates[idx]
    sc = topk.values[rank].item()
    print(f"  Rank {rank+1}: {c['label']:<30} | Score: {sc:+.3f} | Role: {c['role']}")
