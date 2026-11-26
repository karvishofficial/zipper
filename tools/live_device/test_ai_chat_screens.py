"""
PRE-TRAINING DIAGNOSTIC AUDIT ON REAL AI CHAT SCREENS (SAMSUNG GALAXY A26)
Tests the current frozen/trained model (v54_best.pt) on real screen dumps of:
1. ChatGPT (com.openai.chatgpt)
2. Google Gemini (com.google.android.apps.bard)
3. Claude (com.anthropic.claude)
4. Perplexity (ai.perplexity.app.android)

Evaluates:
- Top-1 candidate chosen by current weights
- Action type predicted (CLICK vs TYPE_TEXT vs SUBMIT)
- Phase D Learned Span Pointer head (arguments extracted from goal without regex)
- Hard negative distractor resilience (Voice/Mic, Camera/Attachment, Menu, Upgrade)
"""

from __future__ import annotations
import sys
import re
import json
import xml.etree.ElementTree as ET
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")

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
CKPT_PATH = REPO_ROOT / "tools" / "training" / "karvish_mobile_action" / "v54" / "checkpoints" / "v54_best.pt"


def parse_bounds(bounds_str: str, screen_w=1080.0, screen_h=2340.0):
    m = re.findall(r"\[(\d+),(\d+)\]", bounds_str)
    if len(m) == 2:
        x1, y1 = float(m[0][0]), float(m[0][1])
        x2, y2 = float(m[1][0]), float(m[1][1])
        ymin = max(0.0, min(1.0, y1 / screen_h))
        xmin = max(0.0, min(1.0, x1 / screen_w))
        ymax = max(0.0, min(1.0, y2 / screen_h))
        xmax = max(0.0, min(1.0, x2 / screen_w))
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        w = max(0.001, xmax - xmin)
        h = max(0.001, ymax - ymin)
        area = w * h
        aspect = w / h
        return [ymin, xmin, ymax, xmax, cx, cy, w, h, area, aspect]
    return [0.0] * 10


def extract_candidates_from_xml(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Pre-pass: map parent to child text/desc for bubbling up
    parent_map = {c: p for p in root.iter() for c in p}

    def get_subtree_label(node):
        txt = node.attrib.get("text", "").strip()
        cd = node.attrib.get("content-desc", "").strip()
        if txt or cd:
            return txt or cd
        for child in node.iter():
            if child is not node:
                c_txt = child.attrib.get("text", "").strip()
                c_cd = child.attrib.get("content-desc", "").strip()
                if c_txt or c_cd:
                    return c_txt or c_cd
        return ""

    candidates = []
    candidates.append({
        "id": "c_complete_goal",
        "label": "complete",
        "role": "button",
        "bounds_vec": [0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0],
        "is_clickable": True,
        "is_editable": False,
        "is_focused": False,
        "res_id": "system_complete",
    })

    node_idx = 0
    seen_boxes = set()

    for node in root.iter("node"):
        c_class = node.attrib.get("class", "")
        clickable = node.attrib.get("clickable", "false") == "true"
        editable = "EditText" in c_class or node.attrib.get("editable", "false") == "true"
        bounds_str = node.attrib.get("bounds", "")
        res_id = node.attrib.get("resource-id", "").split("/")[-1]

        label = get_subtree_label(node)

        # Skip zero area or non-interactive text-less nodes
        b_vec = parse_bounds(bounds_str)
        if b_vec[6] < 0.02 or b_vec[7] < 0.01:
            continue

        box_key = (round(b_vec[0], 3), round(b_vec[1], 3), round(b_vec[2], 3), round(b_vec[3], 3))

        if (clickable or editable or (label and len(label) > 1)) and label:
            # Deduplicate near identical boxes with same label
            if (box_key, label) in seen_boxes:
                continue
            seen_boxes.add((box_key, label))

            role = "input" if editable else ("button" if clickable else "view")
            candidates.append({
                "id": f"c_{node_idx}",
                "label": label[:64],
                "role": role,
                "bounds_vec": b_vec,
                "is_clickable": clickable,
                "is_editable": editable,
                "is_focused": node.attrib.get("focused", "false") == "true",
                "res_id": res_id,
            })
            node_idx += 1

    return candidates


def run_diagnostic(model, tokenizer, screen_name: str, xml_path: Path, goal: str, expected_target_keywords: list[str]):
    print("\n" + "=" * 80)
    print(f"AUDITING: [{screen_name}] on REAL SCREEN DUMP")
    print(f"Goal: '{goal}'")
    print(f"XML: {xml_path}")
    print("=" * 80)

    candidates = extract_candidates_from_xml(xml_path)
    print(f">> Total Extracted Candidates: {len(candidates)}")
    print(">> Candidate List (Top 15):")
    for ci, c in enumerate(candidates[:15]):
        rid = f"({c['res_id']})" if c['res_id'] else ""
        print(f"   [{ci:2d}] [{c['id']:15s}] '{c['label']:28s}' {rid:25s} role={c['role']:6s}")
    if len(candidates) > 15:
        print(f"   ... and {len(candidates) - 15} more candidates")

    # Encode Goal
    enc = tokenizer(goal, max_length=32, truncation=True, padding="max_length", return_offsets_mapping=True, return_tensors="pt")
    goal_ids = enc["input_ids"].to(DEVICE)
    goal_mask = enc["attention_mask"].float().to(DEVICE)
    offsets = enc["offset_mapping"][0]

    # Subselect or pad to max 64
    cands_eval = candidates[:64]
    N = len(cands_eval)
    max_lbl_len = 16

    cand_lbl_ids = torch.zeros(1, N, max_lbl_len, dtype=torch.long, device=DEVICE)
    cand_lbl_mask = torch.zeros(1, N, max_lbl_len, dtype=torch.float32, device=DEVICE)
    cand_bounds = torch.zeros(1, N, 10, dtype=torch.float32, device=DEVICE)
    cand_roles = torch.zeros(1, N, dtype=torch.long, device=DEVICE)
    cand_flags = torch.zeros(1, N, 16, dtype=torch.float32, device=DEVICE)
    cand_pad_mask = torch.ones(1, N, dtype=torch.float32, device=DEVICE)

    for ci, c in enumerate(cands_eval):
        c_enc = tokenizer(c["label"], max_length=max_lbl_len, truncation=True, padding=False)
        c_tids = c_enc["input_ids"][:max_lbl_len]
        for tw_i, tw_id in enumerate(c_tids):
            cand_lbl_ids[0, ci, tw_i] = tw_id
            cand_lbl_mask[0, ci, tw_i] = 1.0

        cand_bounds[0, ci] = torch.tensor(c["bounds_vec"], device=DEVICE)
        r = c["role"]
        cand_roles[0, ci] = 2 if r == "input" else (1 if r == "button" else 0)
        if c["is_editable"]: cand_flags[0, ci, 0] = 1.0
        if c["is_focused"]: cand_flags[0, ci, 1] = 1.0
        if c["is_clickable"]: cand_flags[0, ci, 2] = 1.0

    with torch.no_grad():
        out = model(
            goal_ids=goal_ids,
            goal_mask=goal_mask,
            candidate_label_ids=cand_lbl_ids,
            candidate_label_mask=cand_lbl_mask,
            candidate_bounds=cand_bounds,
            candidate_roles=cand_roles,
            candidate_flags=cand_flags,
            candidate_pad_mask=cand_pad_mask,
        )

    ranking_scores = out["ranking_scores"][0]
    action_logits = out["action_logits"][0]
    start_logits = out["span_start_logits"][0]
    end_logits = out["span_end_logits"][0]

    # Rank candidates
    sorted_indices = torch.argsort(ranking_scores, descending=True).tolist()

    print("\n>> MODEL PREDICTION RANKINGS (TOP 5):")
    for rank, idx in enumerate(sorted_indices[:5], 1):
        c = cands_eval[idx]
        score = ranking_scores[idx].item()
        marker = " <=== [SELECTED #1]" if rank == 1 else ""
        print(f"   Rank {rank}: Score={score:6.2f} | Label: '{c['label']:25s}' role={c['role']:6s} (res={c['res_id']}){marker}")

    top_idx = sorted_indices[0]
    top_c = cands_eval[top_idx]
    pred_act_idx = action_logits.argmax().item()
    pred_action = IDX_TO_ACTION.get(pred_act_idx, "CLICK")

    # Decode span from goal
    s_idx = start_logits.argmax().item()
    e_idx = end_logits.argmax().item()
    if s_idx <= e_idx and e_idx < len(offsets):
        s_char = offsets[s_idx][0].item()
        e_char = offsets[e_idx][1].item()
        extracted_span = goal[s_char:e_char].strip()
    else:
        extracted_span = ""

    # Check if target matches expected keyword
    is_target_hit = any(kw.lower() in top_c["label"].lower() or kw.lower() in top_c["res_id"].lower() for kw in expected_target_keywords)

    print("\n>> PRE-TRAINING DIAGNOSTIC SUMMARY:")
    print(f"   Selected Candidate:  '{top_c['label']}' (role: {top_c['role']})")
    print(f"   Predicted Action:    {pred_action}")
    print(f"   Span Pointer Query:  '{extracted_span}'")
    print(f"   Target Correctness:  {'*** HIT ***' if is_target_hit else '!!! MISS / DISTRACTED !!!'}")

    return {
        "screen": screen_name,
        "goal": goal,
        "selected": top_c["label"],
        "action": pred_action,
        "span": extracted_span,
        "hit": is_target_hit,
        "score": ranking_scores[top_idx].item(),
    }


def main():
    print("=" * 80)
    print("     PRE-TRAINING DIAGNOSTIC SUITE ON REAL AI APPS (CURRENT V54 MODEL)")
    print("=" * 80)

    model = KarvishModelV54().to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Loaded Checkpoint: {CKPT_PATH.name} ({sum(p.numel() for p in model.parameters()):,} parameters)")

    results = []

    # 1. ChatGPT Turn 2 (Chat input screen)
    r1 = run_diagnostic(
        model=model,
        tokenizer=TOKENIZER,
        screen_name="ChatGPT Real Chat Screen",
        xml_path=REPO_ROOT / "tools" / "live_device" / "chatgpt_dump.xml",
        goal="Ask ChatGPT what is algae",
        expected_target_keywords=["wait", "Reply to ChatGPT", "EditText"],
    )
    results.append(r1)

    # 2. Google Gemini Turn 2 (Chat input screen)
    r2 = run_diagnostic(
        model=model,
        tokenizer=TOKENIZER,
        screen_name="Google Gemini Real Half Sheet Screen",
        xml_path=REPO_ROOT / "tools" / "live_device" / "gemini_dump.xml",
        goal="Ask Gemini what is algae",
        expected_target_keywords=["Ask Gemini", "assistant_robin_input"],
    )
    results.append(r2)

    # 3. Perplexity Real Screen
    r3 = run_diagnostic(
        model=model,
        tokenizer=TOKENIZER,
        screen_name="Perplexity Real Main Screen",
        xml_path=REPO_ROOT / "tools" / "live_device" / "perplexity_dump.xml",
        goal="Ask Perplexity what is algae",
        expected_target_keywords=["Ask anything", "input-search"],
    )
    results.append(r3)

    # 4. Launcher Screen (Opening ChatGPT)
    # We can test if current model knows to launch ChatGPT or Gemini
    print("\n\n" + "=" * 80)
    print("FINAL PRE-TRAINING DIAGNOSTIC SUMMARY REPORT")
    print("=" * 80)
    print(f"{'Screen':<32s} | {'Top-1 Selected':<22s} | {'Action':<10s} | {'Extracted Span':<18s} | {'Result':<10s}")
    print("-" * 105)
    for r in results:
        status = "HIT" if r["hit"] else "MISS"
        print(f"{r['screen']:<32s} | {r['selected'][:20]:<22s} | {r['action']:<10s} | {r['span']:<18s} | {status:<10s}")
    print("=" * 80)


if __name__ == "__main__":
    main()
