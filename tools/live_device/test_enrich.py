import sys, torch
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
from transformers import AutoTokenizer

REPO_ROOT = Path('.').resolve()
sys.path.insert(0, str(REPO_ROOT / 'tools' / 'training' / 'karvish_mobile_action' / 'v52'))
from model_v52 import KarvishModelV52
from schema_v52 import ACTION_TO_IDX

tok = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = KarvishModelV52()
ckpt = torch.load('tools/training/karvish_mobile_action/v52/checkpoints/v52_best.pt', map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
model.eval()

goal = 'Open Zepto and add the cheapest tomato to the cart'
enc_g = tok(goal, max_length=32, truncation=True, return_tensors='pt')
g_ids, g_mask = enc_g['input_ids'], enc_g['attention_mask'].float()

test_cands = [
    {'id': 'c_scroll', 'label': 'scroll down', 'role': 'system'},
    {'id': 'c_cheap_13', 'label': 'Tomato Local 250 g ₹13 Add to Cart', 'role': 'button'},
    {'id': 'c_org_26', 'label': 'Organically Grown Tomato Local 500 g ₹26 Add to Cart', 'role': 'button'},
    {'id': 'c_bloom_34', 'label': 'Bloom Tomato Hybrid 500 g ₹34 Add to Cart', 'role': 'button'},
    {'id': 'c_cart', 'label': 'Cart 13 items', 'role': 'button'},
    {'id': 'c_comp', 'label': 'complete', 'role': 'button'},
]

lbl_texts = [c['label'] for c in test_cands]
enc_c = tok(lbl_texts, max_length=32, padding='max_length', truncation=True, return_tensors='pt')
c_lbl_ids = enc_c['input_ids'].unsqueeze(0)
c_lbl_mask = enc_c['attention_mask'].float().unsqueeze(0)
c_bounds = torch.zeros(1, len(test_cands), 10)
c_roles = torch.tensor([[1 if c['role']=='button' else 0 for c in test_cands]])
c_flags = torch.zeros(1, len(test_cands), 16)
c_flags[0, :, 2] = 1.0
c_pad_mask = torch.ones(1, len(test_cands))

with torch.no_grad():
    out = model(goal_ids=g_ids, goal_mask=g_mask, candidate_label_ids=c_lbl_ids, candidate_label_mask=c_lbl_mask, candidate_bounds=c_bounds, candidate_roles=c_roles, candidate_flags=c_flags, candidate_pad_mask=c_pad_mask)

scores = out['ranking_scores'][0].numpy()
act_logits = out['action_logits'][0].numpy()
IDX_TO_ACTION = {v: k for k, v in ACTION_TO_IDX.items()}
print('=== SCORES WITH CARD CONTEXT ENRICHMENT ===')
for i, c in enumerate(test_cands):
    print(f"{c['id']:12s}: {scores[i]:6.2f} | label: {c['label']}")
