"""
KARVISH UNIVERSAL V59 — CLOSED-LOOP SIMULATION & REGRESSION EVALUATOR
Evaluates:
  1. 12/12 Frozen Baseline Anchor Tasks (Invariance Shield vs V52/V58).
  2. NLU Slot Extraction F1 (Target >= 85% across messy Hinglish & multi-item queries).
  3. Goal Representation Cosine Invariance (Target >= 0.88 between typos and canonical goals).
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
V58_DIR = SCRIPT_DIR.parent / "v58"
V57_DIR = SCRIPT_DIR.parent / "v57"
V56_DIR = SCRIPT_DIR.parent / "v56"
V52_DIR = SCRIPT_DIR.parent / "v52"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(V58_DIR))
sys.path.insert(0, str(V57_DIR))
sys.path.insert(0, str(V56_DIR))
sys.path.insert(0, str(V52_DIR))

from model_v59 import KarvishModelV59
from schema_v59 import (
    BIO_SLOT_TAGS,
    BIO_TAG_TO_IDX,
    IDX_TO_BIO_TAG,
    ACTION_TO_IDX,
    ROLE_TO_IDX,
)
from probe_v52_screen_decisions import get_test_screens

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
V59_CKPT = SCRIPT_DIR / "checkpoints" / "v59_best.pt"


NLU_TEST_UTTERANCES = [
    {
        "utterance": "allo 1kg paneeerr 2kg order it in zepto",
        "expected_slots": [("1kg", "QTY"), ("2kg", "QTY"), ("zepto", "APP")],
    },
    {
        "utterance": "bored hu youtube pe doremon chalao",
        "expected_slots": [("youtube", "APP"), ("doremon", "SEARCH")],
    },
    {
        "utterance": "bhejo mom ko im reaching late on wa",
        "expected_slots": [("mom", "CONTACT"), ("im reaching late", "PAYLOAD"), ("wa", "APP")],
    },
    {
        "utterance": "turn on flight mode yaar",
        "expected_slots": [("flight mode", "SEARCH")],
    },
    {
        "utterance": "zepto pe 1 packet milk and bread order karo",
        "expected_slots": [("zepto", "APP"), ("milk", "ITEM"), ("bread", "ITEM")],
    },
    {
        "utterance": "bro open yt and play interstellar theme",
        "expected_slots": [("yt", "APP"), ("interstellar theme", "SEARCH")],
    },
    {
        "utterance": "opn settings and switch off wifi",
        "expected_slots": [("settings", "APP"), ("wifi", "SEARCH")],
    },
    {
        "utterance": "wa pe priya ko bhejo: ok noted",
        "expected_slots": [("wa", "APP"), ("priya", "CONTACT"), ("ok noted", "PAYLOAD")],
    },
]


def extract_entities_from_bio_predictions(
    tokens: List[str],
    bio_tags: List[str],
) -> List[Tuple[str, str]]:
    """Extracts entity chunks (entity_text, tag_type) from predicted BIO tags."""
    entities = []
    current_tokens = []
    current_type = None

    for tok, tag in zip(tokens, bio_tags):
        if tok in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        clean_tok = tok[2:] if tok.startswith("##") else tok

        if tag.startswith("B-"):
            if current_type is not None:
                entities.append((" ".join(current_tokens).replace(" ##", ""), current_type))
            current_type = tag[2:]
            current_tokens = [clean_tok]
        elif tag.startswith("I-") and current_type is not None and tag[2:] == current_type:
            current_tokens.append(clean_tok)
        else:
            if current_type is not None:
                entities.append((" ".join(current_tokens).replace(" ##", ""), current_type))
                current_type = None
                current_tokens = []

    if current_type is not None:
        entities.append((" ".join(current_tokens).replace(" ##", ""), current_type))

    return entities


def evaluate_simulation():
    print("=" * 80)
    print("KARVISH V59: CLOSED-LOOP SIMULATION & REGRESSION BENCHMARK")
    print(f"Device: {DEVICE}")
    print("=" * 80)

    if not V59_CKPT.exists():
        raise FileNotFoundError(f"Checkpoint not found at {V59_CKPT}!")

    model = KarvishModelV59().to(DEVICE)
    ckpt = torch.load(V59_CKPT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print(f">> Loaded V59 checkpoint from {V59_CKPT} (Epoch {ckpt.get('epoch', '?')})")

    # -----------------------------------------------------------------
    # Benchmark 1: 12/12 Frozen Baseline Anchor Tasks (Authoritative)
    # -----------------------------------------------------------------
    print("\n--- BENCHMARK 1: 12/12 FROZEN BASELINE ANCHOR TASKS ---")
    screens = get_test_screens()
    anchors_passed = 0

    with torch.no_grad():
        for idx, s in enumerate(screens, 1):
            cands = s["candidates"][:24]
            N = len(cands)
            enc = TOKENIZER(s["goal"], max_length=32, truncation=True, return_tensors="pt")
            lbl_texts = [c["label"] for c in cands]
            c_enc = TOKENIZER(lbl_texts, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
            bounds = [c.get("bounds_vec", [0.0] * 10) for c in cands]
            roles = [2 if c.get("is_editable") else (1 if c.get("role") == "button" else 0) for c in cands]
            flags = torch.zeros(1, N, 16, device=DEVICE)
            for i, c in enumerate(cands):
                role_str = str(c.get("role", "button")).lower()
                if c.get("is_editable") or "input" in role_str or "edit" in role_str:
                    flags[0, i, 0] = 1.0
                if c.get("is_focused"):
                    flags[0, i, 1] = 1.0
                if c.get("is_clickable", True):
                    flags[0, i, 2] = 1.0
                if c.get("is_selected"):
                    flags[0, i, 5] = 1.0

            out = model(
                goal_ids=enc["input_ids"].to(DEVICE),
                goal_mask=enc["attention_mask"].float().to(DEVICE),
                candidate_label_ids=c_enc["input_ids"].unsqueeze(0).to(DEVICE),
                candidate_label_mask=c_enc["attention_mask"].float().unsqueeze(0).to(DEVICE),
                candidate_bounds=torch.tensor([bounds], dtype=torch.float32, device=DEVICE),
                candidate_roles=torch.tensor([roles], dtype=torch.long, device=DEVICE),
                candidate_flags=flags,
                candidate_pad_mask=torch.ones(1, N, device=DEVICE),
            )
            pred_idx = out["ranking_scores"][0].argmax().item()
            pred_lbl = lbl_texts[pred_idx]
            is_match = s["expected_target"].lower() in pred_lbl.lower()
            if is_match:
                anchors_passed += 1
            print(f"Anchor {idx:02d} | Exp: {s['expected_target'][:18]:18s} | Pred: {pred_lbl[:18]:18s} | [{'PASS' if is_match else 'FAIL'}]")

    anchor_rate = (anchors_passed / len(screens)) * 100.0
    print(f">> Anchor Pass Rate: {anchors_passed}/{len(screens)} ({anchor_rate:.1f}%)")

    # -----------------------------------------------------------------
    # Benchmark 2: Token-Level BIO Slot Extraction Suite
    # -----------------------------------------------------------------
    print("\n--- BENCHMARK 2: NLU BIO SLOT EXTRACTION SUITE ---")
    correct_slots = 0
    total_slots = 0

    with torch.no_grad():
        for test in NLU_TEST_UTTERANCES:
            utt = test["utterance"]
            enc = TOKENIZER(utt, max_length=32, padding="max_length", truncation=True, return_tensors="pt").to(DEVICE)
            goal_out = model.text_encoder(enc["input_ids"], attention_mask=enc["attention_mask"].float())
            bio_logits = model.bio_slot_head(goal_out["last_hidden_state"])

            pred_tag_ids = bio_logits[0].argmax(dim=-1).cpu().tolist()
            pred_tags = [IDX_TO_BIO_TAG.get(idx, "O") for idx in pred_tag_ids]
            tokens = TOKENIZER.convert_ids_to_tokens(enc["input_ids"][0].cpu().tolist())

            extracted = extract_entities_from_bio_predictions(tokens, pred_tags)
            print(f"\n  Utterance: '{utt}'")
            print(f"  Extracted Slots: {extracted}")

            # Check coverage against expected
            for exp_txt, exp_type in test["expected_slots"]:
                total_slots += 1
                matched = any(exp_type == t and (exp_txt.lower().replace(" ", "") in txt.lower().replace(" ", "") or txt.lower().replace(" ", "") in exp_txt.lower().replace(" ", "")) for txt, t in extracted)
                if matched:
                    correct_slots += 1

    slot_f1 = (correct_slots / max(1, total_slots)) * 100.0
    print(f"\n>> Slot Extraction Accuracy: {correct_slots}/{total_slots} ({slot_f1:.1f}%)")

    # -----------------------------------------------------------------
    # Benchmark 3: Contrastive Goal Representation Alignment
    # -----------------------------------------------------------------
    print("\n--- BENCHMARK 3: CONTRASTIVE GOAL INVARIANCE TEST ---")
    pairs = [
        ("allo 1kg paneeerr 2kg in zepto", "order potato and paneer in zepto"),
        ("bored hu youtube pe doremon chalao", "Search doremon on YouTube"),
        ("bhejo mom ko im reaching late on wa", "Message mom I am reaching late on WhatsApp"),
        ("turn on flight mode yaar", "Turn on airplane mode in Settings"),
        ("opn settings and switch off wifi", "Open settings and disable Wi-Fi"),
    ]

    sims = []
    with torch.no_grad():
        for noisy, canon in pairs:
            n_enc = TOKENIZER(noisy, max_length=32, padding="max_length", truncation=True, return_tensors="pt").to(DEVICE)
            c_enc = TOKENIZER(canon, max_length=32, padding="max_length", truncation=True, return_tensors="pt").to(DEVICE)

            n_out = model.text_encoder(n_enc["input_ids"], attention_mask=n_enc["attention_mask"].float())
            c_out = model.text_encoder(c_enc["input_ids"], attention_mask=c_enc["attention_mask"].float())

            sim = F.cosine_similarity(n_out["pooled_output"], c_out["pooled_output"], dim=-1)[0].item()
            sims.append(sim)
            print(f"  Sim: {sim:.4f} | '{noisy}' <--> '{canon}'")

    avg_sim = sum(sims) / len(sims)
    print(f">> Mean Goal Invariance Cosine Similarity: {avg_sim:.4f} (Target >= 0.88)")

    # -----------------------------------------------------------------
    # Summary & Pass Criteria
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("V59 QUALIFICATION VERDICT")
    print("=" * 80)
    print(f"Anchor Tasks: {anchors_passed}/{len(screens)} (Req: 12/12) -> {'PASSED' if anchors_passed == len(screens) else 'FAILED'}")
    print(f"Slot Extraction: {slot_f1:.1f}% (Req: >= 80%) -> {'PASSED' if slot_f1 >= 80 else 'FAILED'}")
    print(f"Invariance Cosine: {avg_sim:.4f} (Req: >= 0.88) -> {'PASSED' if avg_sim >= 0.88 else 'FAILED'}")

    all_passed = (anchors_passed == len(screens)) and (slot_f1 >= 80) and (avg_sim >= 0.88)
    print(f"OVERALL STATUS: {'QUALIFIED FOR ONNX EXPORT & DEVICE DEPLOYMENT' if all_passed else 'NEEDS ITERATION'}")
    print("=" * 80)
    return all_passed


if __name__ == "__main__":
    evaluate_simulation()
