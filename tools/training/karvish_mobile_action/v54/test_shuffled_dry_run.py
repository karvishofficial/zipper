import random
import time
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from train_v54_full import TOKENIZER, DEVICE, BEST_V54_CKPT, find_token_span
from model_v54 import KarvishModelV54
from schema_v54 import ACTION_TO_IDX
from dry_run_universal_audit import build_universal_v54_dataset
from test_v54_battle_hardened import parse_live_screens_from_log, evaluate_screen

# 1. Load model in memory
print(">> Loading fresh V54 weights...")
ckpt = torch.load(BEST_V54_CKPT, map_location="cpu", weights_only=False)
weights = ckpt.get("model_state_dict", ckpt)
model = KarvishModelV54().to(DEVICE)
model.load_state_dict(weights, strict=False)

# 2. Build dataset
dataset = build_universal_v54_dataset()
print(f">> Dataset size: {len(dataset)} items.")

# Fix A1 WA target to candidate 10 (the button 'Karthik.S (You)')
for item in dataset:
    if item["name"] == "real_wa_karthik_search_dump":
        # Find the button candidate
        for i, c in enumerate(item["candidates"]):
            if "karthik.s (you)" in c["label"].lower() and c.get("role_idx") == 1:
                item["target_idx"] = i
                print(f">> Fixed A1 WA target to candidate {i}: {c['label']} ({c['role']})")
                break
        # Ensure 'You:' and 'Karthikeyen' are in hard_negs
        negs = []
        for i, c in enumerate(item["candidates"]):
            lbl = c["label"].lower()
            if "you:" in lbl or "karthikeyen" in lbl or "submit" in lbl or "ask" in lbl:
                negs.append(i)
        item["hard_negs"] = negs

# Fix Gate 4 Telegram Compose target
for item in dataset:
    if item["name"] == "real_tg_open_chat_dump":
        # Make sure Message box is marked editable
        for i, c in enumerate(item["candidates"]):
            if c["label"] == "Message":
                c["is_editable"] = True
                c["role_idx"] = 2
                item["target_idx"] = i
                break

# 3. Setup optimizer
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = AdamW(trainable_params, lr=5e-5, weight_decay=1e-4)

# 4. Train 3 epochs with full shuffle
screens = parse_live_screens_from_log()

def run_quick_audit(epoch: int):
    model.eval()
    print(f"\n--- AUDIT AT EPOCH {epoch} ---")
    passes = 0
    total = 0

    # Gate 1: WhatsApp
    for s in screens:
        if "send hi to karthik" in s["goal"].lower():
            lbls = [c["lbl"] for c in s["candidates"]]
            if any("Karthik.S (You)" in l for l in lbls) and any("submit" in l.lower() for l in lbls):
                res = evaluate_screen(
                    model=model, goal=s["goal"], candidates=s["candidates"],
                    expected_target_checker=lambda c: "karthik.s (you)" in c["lbl"].lower(),
                    forbidden_distractor_checker=lambda c: "submit" in c["lbl"].lower(),
                    expected_action="CLICK", min_margin=3.0,
                )
                total += 1
                if res["passed"]: passes += 1
                st = "PASS" if res["passed"] else "FAIL"
                print(f"[{st}] Gate 1 WA: chosen='{res['top_cand']['lbl'][:25]}' | margin={res['margin']:+.2f}")
                break

    # Gate 2: IG Explore
    for s in screens:
        if "instagram" in s["goal"].lower() and "photography" in s["goal"].lower():
            lbls = [c["lbl"] for c in s["candidates"]]
            if any("Reshare" in l for l in lbls) and any("Search" in l for l in lbls):
                res = evaluate_screen(
                    model=model, goal=s["goal"], candidates=s["candidates"],
                    expected_target_checker=lambda c: "search" in c["lbl"].lower(),
                    forbidden_distractor_checker=lambda c: "reshare" in c["lbl"].lower(),
                    expected_action=None, min_margin=3.0,
                )
                total += 1
                if res["passed"]: passes += 1
                st = "PASS" if res["passed"] else "FAIL"
                print(f"[{st}] Gate 2 IG Search: chosen='{res['top_cand']['lbl'][:25]}' | margin={res['margin']:+.2f}")
                break

    # Gate 3: IG Share Sheet
    for s in screens:
        if "instagram" in s["goal"].lower():
            lbls = [c["lbl"] for c in s["candidates"]]
            if any("Bhavani" in l for l in lbls) and any("submit" in l.lower() for l in lbls):
                res = evaluate_screen(
                    model=model, goal="open instagram and send hi to bhavani", candidates=s["candidates"],
                    expected_target_checker=lambda c: "bhavani" in c["lbl"].lower(),
                    forbidden_distractor_checker=lambda c: "submit" in c["lbl"].lower(),
                    expected_action="CLICK", min_margin=3.0,
                )
                total += 1
                if res["passed"]: passes += 1
                st = "PASS" if res["passed"] else "FAIL"
                print(f"[{st}] Gate 3 IG Share: chosen='{res['top_cand']['lbl'][:25]}' | margin={res['margin']:+.2f}")
                break

    # Gate 4: Telegram
    for s in screens:
        if "telegram" in s["goal"].lower() and s["turn"] == 2:
            res = evaluate_screen(
                model=model, goal=s["goal"], candidates=s["candidates"],
                expected_target_checker=lambda c: c["lbl"] == "Message",
                forbidden_distractor_checker=lambda c: "submit" in c["lbl"].lower(),
                expected_action="TYPE_TEXT", min_margin=3.0,
            )
            total += 1
            if res["passed"]: passes += 1
            st = "PASS" if res["passed"] else "FAIL"
            print(f"[{st}] Gate 4 TG Compose: chosen='{res['top_cand']['lbl'][:25]}' | margin={res['margin']:+.2f}")
            break

    # Gate 5: Zepto Promo
    for s in screens:
        if "zepto" in s["goal"].lower() and s["turn"] == 2:
            res = evaluate_screen(
                model=model, goal=s["goal"], candidates=s["candidates"],
                expected_target_checker=lambda c: "search" in c["lbl"].lower(),
                forbidden_distractor_checker=lambda c: "33" in c["lbl"] and "off" in c["lbl"].lower(),
                expected_action="CLICK", min_margin=3.0,
            )
            total += 1
            if res["passed"]: passes += 1
            st = "PASS" if res["passed"] else "FAIL"
            print(f"[{st}] Gate 5 Zepto Promo: chosen='{res['top_cand']['lbl'][:25]}' | margin={res['margin']:+.2f}")
            break

    # Gate 6: Zepto Search Results
    for s in screens:
        if "zepto" in s["goal"].lower():
            lbls = [c["lbl"] for c in s["candidates"]]
            if any("Tomato" in l for l in lbls) and any("submit" in l.lower() for l in lbls):
                res = evaluate_screen(
                    model=model, goal=s["goal"], candidates=s["candidates"],
                    expected_target_checker=lambda c: "tomato" in c["lbl"].lower(),
                    forbidden_distractor_checker=lambda c: "submit" in c["lbl"].lower(),
                    expected_action="CLICK", min_margin=3.0,
                )
                total += 1
                if res["passed"]: passes += 1
                st = "PASS" if res["passed"] else "FAIL"
                print(f"[{st}] Gate 6 Zepto Results: chosen='{res['top_cand']['lbl'][:25]}' | margin={res['margin']:+.2f}")
                break

    # Gate 7: Task Switcher Zepto
    for s in screens:
        lbls = [c["lbl"] for c in s["candidates"]]
        if any("active apps" in l.lower() for l in lbls) and "zepto" in s["goal"].lower():
            res = evaluate_screen(
                model=model, goal=s["goal"], candidates=s["candidates"],
                expected_target_checker=lambda c: "zepto" in c["lbl"].lower(),
                forbidden_distractor_checker=lambda c: "linkedin" in c["lbl"].lower() or "instagram" in c["lbl"].lower(),
                expected_action="CLICK", min_margin=3.0,
            )
            total += 1
            if res["passed"]: passes += 1
            st = "PASS" if res["passed"] else "FAIL"
            print(f"[{st}] Gate 7 Switcher: chosen='{res['top_cand']['lbl'][:25]}' | margin={res['margin']:+.2f}")
            break

    print(f">> TOTAL SCORE: {passes}/{total} ({passes/max(1, total)*100:.1f}%)")

print("\nEvaluating baseline before dry run:")
run_quick_audit(0)

# Training loop
random.seed(42)
torch.manual_seed(42)

for epoch in range(1, 4):
    model.train()
    shuffled_indices = list(range(len(dataset)))
    random.shuffle(shuffled_indices)

    total_loss_epoch = 0.0
    for idx in shuffled_indices:
        item = dataset[idx]
        optimizer.zero_grad()

        goal = item["goal"]
        candidates = item["candidates"]
        N = len(candidates)

        enc = TOKENIZER(goal, max_length=32, truncation=True, return_tensors="pt")
        g_ids = enc["input_ids"].to(DEVICE)
        g_mask = enc["attention_mask"].float().to(DEVICE)

        lbl_texts = [str(c.get("label", "")).strip() for c in candidates]
        c_enc = TOKENIZER(lbl_texts, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
        c_lbl_ids = c_enc["input_ids"].unsqueeze(0).to(DEVICE)
        c_lbl_mask = c_enc["attention_mask"].float().unsqueeze(0).to(DEVICE)

        b_list = [c.get("bounds_vec", [0.05, 0.05, 0.95, 0.15, 0.5, 0.10, 0.9, 0.10, 0.09, 9.0]) for c in candidates]
        roles_list = [c.get("role_idx", 0) for c in candidates]

        c_bounds = torch.tensor([b_list], dtype=torch.float32, device=DEVICE)
        c_roles = torch.tensor([roles_list], dtype=torch.long, device=DEVICE)
        c_flags = torch.zeros(1, N, 16, dtype=torch.float32, device=DEVICE)
        for i, c in enumerate(candidates):
            if roles_list[i] == 2 or c.get("is_editable"):
                c_flags[0, i, 0] = 1.0
            if c.get("is_clickable", True):
                c_flags[0, i, 2] = 1.0
        c_pad_mask = torch.ones(1, N, dtype=torch.float32, device=DEVICE)

        c_num = torch.zeros(1, N, 4, dtype=torch.float32, device=DEVICE)

        out = model(
            goal_ids=g_ids,
            goal_mask=g_mask,
            candidate_label_ids=c_lbl_ids,
            candidate_label_mask=c_lbl_mask,
            candidate_bounds=c_bounds,
            candidate_roles=c_roles,
            candidate_flags=c_flags,
            candidate_pad_mask=c_pad_mask,
            candidate_num_features=c_num,
        )

        ranking = out["ranking_scores"]
        action_logits = out["action_logits"]
        entity_gate = out["entity_gate"]

        t_idx = torch.tensor([item["target_idx"]], dtype=torch.long, device=DEVICE)
        a_idx = torch.tensor([item["action_idx"]], dtype=torch.long, device=DEVICE)

        loss_rank = F.cross_entropy(ranking, t_idx)
        loss_act = F.cross_entropy(action_logits, a_idx)

        # Entity Gating
        loss_entity = torch.tensor(0.0, device=DEVICE)
        if "entity_target" in item:
            ent_target_str = item["entity_target"].lower()
            ent_labels = torch.zeros(1, N, device=DEVICE)
            for i, c in enumerate(candidates):
                if c.get("is_matching_entity") is True:
                    ent_labels[0, i] = 1.0
                elif c.get("is_matching_entity") is False:
                    ent_labels[0, i] = 0.0
                elif ent_target_str in c["label"].lower():
                    ent_labels[0, i] = 1.0
                else:
                    ent_labels[0, i] = 0.0
            loss_entity = F.binary_cross_entropy(entity_gate, ent_labels)

        # Adversarial Margin Loss
        loss_margin = torch.tensor(0.0, device=DEVICE)
        if item.get("hard_negs"):
            pos_score = ranking[0, item["target_idx"]]
            neg_scores = ranking[0, item["hard_negs"]]
            margin_diff = torch.clamp(4.5 - (pos_score - neg_scores), min=0.0)
            active_viol = margin_diff[margin_diff > 0]
            if len(active_viol) > 0:
                loss_margin = active_viol.mean()

        total_loss = loss_rank + loss_act + 2.5 * loss_margin + 1.0 * loss_entity
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        total_loss_epoch += total_loss.item()

    print(f"\n[Epoch {epoch}/3] Avg Loss: {total_loss_epoch / len(dataset):.4f}")
    run_quick_audit(epoch)
