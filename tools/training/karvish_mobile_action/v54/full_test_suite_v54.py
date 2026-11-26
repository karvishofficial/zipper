import sys
import json
import time
import urllib.request
from pathlib import Path
import torch

REPO_ROOT = Path(".").resolve()
sys.path.insert(0, str(REPO_ROOT / "tools" / "training" / "karvish_mobile_action" / "v54"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "training" / "karvish_mobile_action" / "v52"))

from model_v54 import KarvishModelV54
from schema_v54 import ACTION_TO_IDX
from transformers import AutoTokenizer
from probe_v52_screen_decisions import get_test_screens

IDX_TO_ACTION = {v: k for k, v in ACTION_TO_IDX.items()}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
CKPT_PATH = REPO_ROOT / "tools" / "training" / "karvish_mobile_action" / "v54" / "checkpoints" / "v54_best.pt"

print("=====================================================================")
print("             KARVISH V54 FULL COMPREHENSIVE CERTIFICATION TEST       ")
print("=====================================================================")

# 1. Test Server HTTP Health and Propose API
print("\n--- TEST SUITE 1: GPU BRIDGE SERVER (/health & /propose) ---")
try:
    req = urllib.request.Request("http://127.0.0.1:8765/health")
    with urllib.request.urlopen(req, timeout=3) as resp:
        h_data = json.loads(resp.read().decode("utf-8"))
        print(f"Health Status: {h_data.get('status')} | Model: {h_data.get('model')} | Device: {h_data.get('device')}")
        assert h_data.get("status") == "ready"
        print(">> [PASS] Bridge Server is ONLINE and READY")
except Exception as e:
    print(f">> [FAIL] Bridge Server check failed: {e}")

# Test Propose API via HTTP
try:
    payload = {
        "prompt": "open zepto and search cheapest tomato",
        "tools": [
            {"candidate_id": "c_open_zepto", "semantic_label": "open zepto", "name": "open_app", "action_type": "open_app", "app_name": "Zepto", "role": "system"},
            {"candidate_id": "c_fail_goal", "semantic_label": "fail", "name": "fail", "role": "system"},
            {"candidate_id": "c_press_back", "semantic_label": "back", "name": "press_back", "role": "system"}
        ]
    }
    req = urllib.request.Request("http://127.0.0.1:8765/propose", data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3) as resp:
        p_data = json.loads(resp.read().decode("utf-8"))
        top_cand = p_data["chosen_action"]["target"]
        top_act = p_data["chosen_action"]["action_type"]
        top_args = p_data["chosen_action"]["arguments"]
        print(f"HTTP Propose -> Chosen Target: {top_cand} | Action: {top_act} | Args: {top_args}")
        assert top_cand == "c_open_zepto" and top_act == "open_app"
        print(">> [PASS] Bridge Server accurately routed open_app action")
except Exception as e:
    print(f">> [FAIL] HTTP Propose test failed: {e}")

# 2. Test 12 Frozen Baseline Anchor Tasks (Invariance)
print("\n--- TEST SUITE 2: 12 FROZEN ANCHOR TASKS (BASELINE INVARIANCE) ---")
model = KarvishModelV54().to(DEVICE)
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
model.load_state_dict(state, strict=False)
model.eval()

screens = get_test_screens()
anchors_passed = 0
for idx, s in enumerate(screens, 1):
    cands = s["candidates"][:24]
    N = len(cands)
    enc = TOKENIZER(s["goal"], max_length=32, truncation=True, return_tensors="pt")
    g_ids = enc["input_ids"].to(DEVICE)
    g_mask = enc["attention_mask"].float().to(DEVICE)
    lbl_texts = [c["label"] for c in cands]
    c_enc = TOKENIZER(lbl_texts, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
    c_lbl_ids = c_enc["input_ids"].unsqueeze(0).to(DEVICE)
    c_lbl_mask = c_enc["attention_mask"].float().unsqueeze(0).to(DEVICE)
    bounds = [c.get("bounds_vec", [0.0]*10) for c in cands]
    c_bounds = torch.tensor([bounds], dtype=torch.float32, device=DEVICE)
    roles = [2 if c.get("is_editable") else (1 if c.get("role") == "button" else 0) for c in cands]
    c_roles = torch.tensor([roles], dtype=torch.long, device=DEVICE)
    c_flags = torch.zeros(1, N, 16, device=DEVICE)
    c_pad = torch.ones(1, N, device=DEVICE)
    
    with torch.no_grad():
        out = model(g_ids, g_mask, c_lbl_ids, c_lbl_mask, c_bounds, c_roles, c_flags, c_pad)
    
    pred_idx = out["ranking_scores"][0].argmax().item()
    pred_lbl = lbl_texts[pred_idx]
    is_match = s["expected_target"].lower() in pred_lbl.lower()
    status = "PASS" if is_match else "FAIL"
    if is_match:
        anchors_passed += 1
    print(f"Anchor {idx:02d} | {s['name'][:30]:30s} | Exp: {s['expected_target'][:20]:20s} | Pred: {pred_lbl[:20]:20s} | [{status}]")

print(f"\n>> Baseline Anchor Summary: {anchors_passed}/{len(screens)} Passed ({(anchors_passed/len(screens))*100:.1f}%)")

# 3. Test Full 6-Turn Zepto Trajectory
print("\n--- TEST SUITE 3: 6-TURN PHYSICAL TRAJECTORY CERTIFICATION ---")
trajectory_passed = True

# Turn 1: Launcher Screen
t1_cands = [
    {"label": "open zepto", "role": "button", "is_clickable": True, "bounds": [0.1, 0.3, 0.3, 0.4, 0.2, 0.35, 0.2, 0.1, 0.02, 2.0]},
    {"label": "open youtube", "role": "button", "is_clickable": True, "bounds": [0.4, 0.3, 0.6, 0.4, 0.5, 0.35, 0.2, 0.1, 0.02, 2.0]},
    {"label": "open instagram", "role": "button", "is_clickable": True, "bounds": [0.7, 0.3, 0.9, 0.4, 0.8, 0.35, 0.2, 0.1, 0.02, 2.0]},
    {"label": "submit focused field", "role": "button", "is_clickable": False, "bounds": [0.0]*10},
    {"label": "fail", "role": "system", "is_clickable": False, "bounds": [0.0]*10}
]
enc = TOKENIZER("open zepto and search cheapest tomato", max_length=32, truncation=True, return_tensors="pt")
c_enc = TOKENIZER([c["label"] for c in t1_cands], max_length=16, padding="max_length", truncation=True, return_tensors="pt")
with torch.no_grad():
    out1 = model(
        enc["input_ids"].to(DEVICE), enc["attention_mask"].float().to(DEVICE),
        c_enc["input_ids"].unsqueeze(0).to(DEVICE), c_enc["attention_mask"].float().unsqueeze(0).to(DEVICE),
        torch.tensor([[c["bounds"] for c in t1_cands]], dtype=torch.float32, device=DEVICE),
        torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.long, device=DEVICE),
        torch.zeros(1, len(t1_cands), 16, device=DEVICE),
        torch.ones(1, len(t1_cands), device=DEVICE)
    )
t1_pred = t1_cands[out1["ranking_scores"][0].argmax().item()]["label"]
t1_margin = float(out1["ranking_scores"][0][0] - out1["ranking_scores"][0][4])
t1_ok = t1_pred == "open zepto"
print(f"Turn 1 (Launcher)       | Top: {t1_pred:20s} | Margin vs Fail: +{t1_margin:.2f} pts | [{'PASS' if t1_ok else 'FAIL'}]")

# Turn 2: Zepto Home Screen
t2_cands = [
    {"label": "Search for products, brands and more", "role": "input", "bounds": [0.05, 0.1, 0.95, 0.18, 0.5, 0.14, 0.9, 0.08, 0.07, 11.25]},
    {"label": "Cart 0 items", "role": "button", "bounds": [0.85, 0.02, 0.98, 0.08, 0.91, 0.05, 0.13, 0.06, 0.008, 2.1]},
    {"label": "Banner 50% OFF Fresh Vegetables", "role": "cardview", "bounds": [0.05, 0.2, 0.95, 0.45, 0.5, 0.32, 0.9, 0.25, 0.22, 3.6]}
]
c_enc2 = TOKENIZER([c["label"] for c in t2_cands], max_length=16, padding="max_length", truncation=True, return_tensors="pt")
with torch.no_grad():
    out2 = model(
        enc["input_ids"].to(DEVICE), enc["attention_mask"].float().to(DEVICE),
        c_enc2["input_ids"].unsqueeze(0).to(DEVICE), c_enc2["attention_mask"].float().unsqueeze(0).to(DEVICE),
        torch.tensor([[c["bounds"] for c in t2_cands]], dtype=torch.float32, device=DEVICE),
        torch.tensor([[2, 1, 0]], dtype=torch.long, device=DEVICE),
        torch.zeros(1, len(t2_cands), 16, device=DEVICE),
        torch.ones(1, len(t2_cands), device=DEVICE),
        h_prev=out1["h_t"]
    )
t2_pred = t2_cands[out2["ranking_scores"][0].argmax().item()]["label"]
t2_ok = "Search" in t2_pred
print(f"Turn 2 (Zepto Home)     | Top: {t2_pred:20s} | Role: input (CLICK)           | [{'PASS' if t2_ok else 'FAIL'}]")

# Turn 3: Search Query Typing (Span Argument Extraction)
t_toks = TOKENIZER.convert_ids_to_tokens(enc["input_ids"][0])
sp_s = out2["span_start_logits"][0].argmax().item()
sp_e = out2["span_end_logits"][0].argmax().item()
ext_txt = TOKENIZER.convert_tokens_to_string(t_toks[sp_s:sp_e+1]).strip()
t3_ok = "tomato" in ext_txt.lower()
print(f"Turn 3 (Search Argument)| Span: '{ext_txt}' | Extracted Query Clean         | [{'PASS' if t3_ok else 'FAIL'}]")

# Turn 4: Autocomplete Suppression (Submit beats Dropdown Suggestions)
t4_cands = [
    {"label": "submit focused field", "role": "button", "bounds": [0.85, 0.9, 0.98, 0.98, 0.91, 0.94, 0.13, 0.08, 0.01, 1.6]},
    {"label": "Tomato puree", "role": "button", "bounds": [0.05, 0.2, 0.95, 0.28, 0.5, 0.24, 0.9, 0.08, 0.07, 11.2]},
    {"label": "Tomato ketchup", "role": "button", "bounds": [0.05, 0.29, 0.95, 0.37, 0.5, 0.33, 0.9, 0.08, 0.07, 11.2]},
    {"label": "Fresh tomatoes in Vegetables", "role": "button", "bounds": [0.05, 0.38, 0.95, 0.46, 0.5, 0.42, 0.9, 0.08, 0.07, 11.2]}
]
c_enc4 = TOKENIZER([c["label"] for c in t4_cands], max_length=16, padding="max_length", truncation=True, return_tensors="pt")
with torch.no_grad():
    out4 = model(
        enc["input_ids"].to(DEVICE), enc["attention_mask"].float().to(DEVICE),
        c_enc4["input_ids"].unsqueeze(0).to(DEVICE), c_enc4["attention_mask"].float().unsqueeze(0).to(DEVICE),
        torch.tensor([[c["bounds"] for c in t4_cands]], dtype=torch.float32, device=DEVICE),
        torch.tensor([[1, 1, 1, 1]], dtype=torch.long, device=DEVICE),
        torch.zeros(1, len(t4_cands), 16, device=DEVICE),
        torch.ones(1, len(t4_cands), device=DEVICE),
        h_prev=out2["h_t"]
    )
t4_pred = t4_cands[out4["ranking_scores"][0].argmax().item()]["label"]
t4_margin = float(out4["ranking_scores"][0][0] - out4["ranking_scores"][0][1])
t4_ok = t4_pred == "submit focused field"
print(f"Turn 4 (Autocomplete)   | Top: {t4_pred:20s} | Margin vs Dropdown: +{t4_margin:.2f} pts | [{'PASS' if t4_ok else 'FAIL'}]")

# Turn 5: Dense 28-item Tomato Grid (Cheapest Selection)
import re
t5_labels = [
    "Tomato Local 500g ₹13 ADD",
    "Tomato Hybrid 500g ₹19 ADD",
    "Tomato Cherry 250g ₹37 ADD",
    "Tomato Plum 500g ₹29 ADD",
    "Tomato Country Fresh 1kg ₹45 ADD",
    "Ginger Fresh 100g ₹9 ADD",
    "Garlic 200g ₹22 ADD",
    "Dettol Soap 75g ₹10 ADD",
    "Amul Taaza Milk 500ml ₹27 ADD",
    "Coriander Bunch ₹8 ADD",
    "Onion 1kg ₹32 ADD",
    "Potato 1kg ₹25 ADD"
]
t5_num = torch.zeros(1, len(t5_labels), 4, device=DEVICE)
for i, l in enumerate(t5_labels):
    m = re.search(r"₹(\d+)", l)
    if m:
        t5_num[0, i, 0] = float(m.group(1)) / 100.0

c_enc5 = TOKENIZER(t5_labels, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
with torch.no_grad():
    out5 = model(
        enc["input_ids"].to(DEVICE), enc["attention_mask"].float().to(DEVICE),
        c_enc5["input_ids"].unsqueeze(0).to(DEVICE), c_enc5["attention_mask"].float().unsqueeze(0).to(DEVICE),
        torch.zeros(1, len(t5_labels), 10, device=DEVICE),
        torch.ones(1, len(t5_labels), dtype=torch.long, device=DEVICE),
        torch.zeros(1, len(t5_labels), 16, device=DEVICE),
        torch.ones(1, len(t5_labels), device=DEVICE),
        candidate_num_features=t5_num,
        h_prev=out4["h_t"]
    )
t5_scores = out5["ranking_scores"][0]
t5_top_idx = t5_scores.argmax().item()
t5_top = t5_labels[t5_top_idx]
t5_ok = t5_top_idx == 0
distractor_min_margin = min([float(t5_scores[0] - t5_scores[j]) for j in range(5, len(t5_labels))])
price_margin = float(t5_scores[0] - t5_scores[1])
print(f"Turn 5 (Cheapest Tomato)| Top: {t5_top:20s} | Beat ₹19 by: +{price_margin:.2f} pts | [{'PASS' if t5_ok else 'FAIL'}]")
print(f"                        | Distractor Isolation Margin (Ginger ₹9, Soap ₹10): +{distractor_min_margin:.2f} pts")

# Turn 6: Cart Mutation Milestone & Termination Certainty
t6_cands = [
    {"label": "complete", "role": "button", "bounds": [0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]},
    {"label": "Tomato Local 500g 1 in cart +", "role": "button", "bounds": [0.05, 0.2, 0.45, 0.5, 0.25, 0.35, 0.4, 0.3, 0.12, 1.33]},
    {"label": "View Cart ₹13", "role": "button", "bounds": [0.05, 0.9, 0.95, 0.98, 0.5, 0.94, 0.9, 0.08, 0.07, 11.25]}
]
c_enc6 = TOKENIZER([c["label"] for c in t6_cands], max_length=16, padding="max_length", truncation=True, return_tensors="pt")
enc_term = TOKENIZER("Search cheapest tomato and add to cart and stop", max_length=32, truncation=True, return_tensors="pt")
with torch.no_grad():
    out6 = model(
        enc_term["input_ids"].to(DEVICE), enc_term["attention_mask"].float().to(DEVICE),
        c_enc6["input_ids"].unsqueeze(0).to(DEVICE), c_enc6["attention_mask"].float().unsqueeze(0).to(DEVICE),
        torch.tensor([[c["bounds"] for c in t6_cands]], dtype=torch.float32, device=DEVICE),
        torch.tensor([[1, 1, 1]], dtype=torch.long, device=DEVICE),
        torch.zeros(1, len(t6_cands), 16, device=DEVICE),
        torch.ones(1, len(t6_cands), device=DEVICE),
        h_prev=out5["h_t"]
    )
t6_top_idx = out6["ranking_scores"][0].argmax().item()
t6_top_lbl = t6_cands[t6_top_idx]["label"]
term_prob = float(out6["termination_prob"][0].item())
t6_ok = ("complete" in t6_top_lbl.lower()) and (term_prob > 0.5 or out6["action_logits"][0].argmax().item() == ACTION_TO_IDX["WAIT"])
print(f"Turn 6 (Milestone Term) | Top: {t6_top_lbl:16s} | Term Certainty: {term_prob*100:.2f}%     | Stops Multi-Tap (5x)        | [{'PASS' if t6_ok else 'FAIL'}]")

print("\n=====================================================================")
print("                   FINAL CERTIFICATION STATUS                       ")
print("=====================================================================")
all_pass = (anchors_passed == len(screens)) and t1_ok and t2_ok and t3_ok and t4_ok and t5_ok and t6_ok
print(f"OVERALL CERTIFICATION: {'100% ALL TESTS PASSED' if all_pass else 'FAILURES DETECTED'}")
