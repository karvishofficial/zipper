"""
KARVISH UNIVERSAL V59 — AUTHORITATIVE NLU & MOBILE ACTION DATASET GENERATOR
Conforming strictly to Karvish Constitution (AGENTS.md):
  1. Ingests all V58 authoritative baseline items (V57 anchors, Social Multihop, Search History,
     Overlay Occlusion Escape, Zero-Evidence Completion Calibration).
  2. Ingests seed utterances from seed_utterances_v59.json (500 real-world human phrasing patterns).
  3. Applies multi-class adversarial augmentations:
     - QWERTY keyboard adjacent typos & vowel elongations ("paneeerr", "bbored", "momm")
     - Slang & shorthand aliases ("yt", "wa", "insta", "calc")
     - Conversational preambles & Hinglish syntax ("bored hu yaar play doremon on yt")
     - Multi-slot grocery cart extraction ("allo 1kg paneeerr 2kg in zepto")
  4. Generates precise token-level BIO slot labels (15 tags) using MiniLM character offset mapping.
  5. Provides paired (noisy_goal, canonical_goal) trajectories for Contrastive Goal Invariance loss.
"""

from __future__ import annotations

import copy
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
V58_DIR = SCRIPT_DIR.parent / "v58"
V57_DIR = SCRIPT_DIR.parent / "v57"
V56_DIR = SCRIPT_DIR.parent / "v56"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(V58_DIR))
sys.path.insert(0, str(V57_DIR))
sys.path.insert(0, str(V56_DIR))

from schema_v59 import (
    BIO_SLOT_TAGS,
    BIO_TAG_TO_IDX,
    CANONICAL_ACTIONS,
    ACTION_TO_IDX,
    ROLE_TO_IDX,
)
from build_v58_dataset import generate_v58_authoritative_dataset
from build_v56_generalized_dataset import _normalize_item, find_token_span

TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

QWERTY_NEIGHBORS = {
    'a': ['q', 'w', 's', 'z'],
    'b': ['v', 'g', 'h', 'n'],
    'c': ['x', 'd', 'f', 'v'],
    'd': ['s', 'e', 'r', 'f', 'c', 'x'],
    'e': ['w', 'r', 'd', 's'],
    'f': ['d', 'r', 't', 'g', 'v', 'c'],
    'g': ['f', 't', 'y', 'h', 'b', 'v'],
    'h': ['g', 'y', 'u', 'j', 'n', 'b'],
    'i': ['u', 'o', 'k', 'j'],
    'j': ['h', 'u', 'i', 'k', 'm', 'n'],
    'k': ['j', 'i', 'o', 'l', 'm'],
    'l': ['k', 'o', 'p'],
    'm': ['n', 'j', 'k'],
    'n': ['b', 'h', 'j', 'm'],
    'o': ['i', 'p', 'l', 'k'],
    'p': ['o', 'l'],
    'q': ['w', 'a'],
    'r': ['e', 't', 'f', 'd'],
    's': ['a', 'w', 'e', 'd', 'x', 'z'],
    't': ['r', 'y', 'g', 'f'],
    'u': ['y', 'i', 'j', 'h'],
    'v': ['c', 'f', 'g', 'b'],
    'w': ['q', 'e', 's', 'a'],
    'x': ['z', 's', 'd', 'c'],
    'y': ['t', 'u', 'h', 'g'],
    'z': ['a', 's', 'x'],
}

CONVERSATIONAL_PREAMBLES = [
    "hey please ",
    "can you ",
    "bro ",
    "bro please ",
    "mujhe ",
    "yaar ",
    "suno ",
    "i am bored ",
    "bored hu ",
    "bore ho raha hu ",
    "kuch accha ",
    "please go ahead and ",
    "quickly ",
    "just ",
]


def inject_qwerty_typo(text: str, p: float = 0.15) -> str:
    """Injects human-like adjacent key QWERTY typos or vowel elongations."""
    chars = list(text)
    new_chars = []
    for c in chars:
        lower_c = c.lower()
        if lower_c in 'aeiou' and random.random() < 0.12:
            # Vowel elongation (e.g. paneeerr, doreemonn)
            new_chars.append(c * random.choice([2, 3]))
        elif lower_c in QWERTY_NEIGHBORS and random.random() < p:
            replacement = random.choice(QWERTY_NEIGHBORS[lower_c])
            new_chars.append(replacement.upper() if c.isupper() else replacement)
        else:
            new_chars.append(c)
    return "".join(new_chars)


def compute_token_bio_tags(
    text: str,
    spans: List[Tuple[int, int, str]],
    tokenizer: AutoTokenizer,
    max_len: int = 32,
) -> List[int]:
    """
    Given text and character spans (start_char, end_char, slot_type),
    returns a list of BIO tag indices aligned with tokenized input_ids.
    """
    encoding = tokenizer(
        text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
    )
    offsets = encoding["offset_mapping"]
    bio_tags = [BIO_TAG_TO_IDX["O"]] * len(offsets)

    for start_char, end_char, slot_type in spans:
        b_tag = f"B-{slot_type.upper()}"
        i_tag = f"I-{slot_type.upper()}"
        b_idx = BIO_TAG_TO_IDX.get(b_tag, BIO_TAG_TO_IDX["O"])
        i_idx = BIO_TAG_TO_IDX.get(i_tag, BIO_TAG_TO_IDX["O"])

        first_token_marked = False
        for tok_i, (tok_start, tok_end) in enumerate(offsets):
            if tok_start == tok_end:
                continue  # CLS, SEP, PAD
            # Check overlap between token char offset and span
            if max(tok_start, start_char) < min(tok_end, end_char):
                if not first_token_marked:
                    bio_tags[tok_i] = b_idx
                    first_token_marked = True
                else:
                    bio_tags[tok_i] = i_idx

    return bio_tags


def levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def find_entity_spans(text: str, entities: List[Tuple[str, str]]) -> List[Tuple[int, int, str]]:
    """
    Finds (start, end, tag) character spans for a list of (entity_string, tag) pairs,
    with exact substring matching and fuzzy word-level alignment for typos and Hinglish variants.
    """
    spans: List[Tuple[int, int, str]] = []
    text_lower = text.lower()
    words_with_offsets = []
    for match in re.finditer(r'\b\w+\b', text_lower):
        words_with_offsets.append((match.start(), match.end(), match.group()))

    for entity_str, tag in entities:
        if not entity_str:
            continue
        ent_lower = entity_str.lower().strip().strip(':').strip('"').strip('\'').strip(',')
        if not ent_lower:
            continue

        # 1. Direct exact substring match (or stripped match)
        pos = text_lower.find(ent_lower)
        if pos != -1:
            spans.append((pos, pos + len(ent_lower), tag))
            continue

        # Also try matching with punctuation replaced by space in text
        text_no_punct = re.sub(r'[:",]', ' ', text_lower)
        pos_np = text_no_punct.find(ent_lower)
        if pos_np != -1:
            spans.append((pos_np, pos_np + len(ent_lower), tag))
            continue

        # 2. Token-by-token fuzzy match for typos and Hinglish variations
        ent_tokens = ent_lower.split()
        for ent_tok in ent_tokens:
            if len(ent_tok) <= 2:
                for w_start, w_end, w_txt in words_with_offsets:
                    if w_txt == ent_tok:
                        spans.append((w_start, w_end, tag))
            else:
                max_allowed_dist = 1 if len(ent_tok) <= 4 else 2
                for w_start, w_end, w_txt in words_with_offsets:
                    if len(w_txt) >= 3 and (ent_tok in w_txt or w_txt in ent_tok or levenshtein(ent_tok, w_txt) <= max_allowed_dist):
                        spans.append((w_start, w_end, tag))

    # Deduplicate overlapping spans (keep longest / first)
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    clean_spans: List[Tuple[int, int, str]] = []
    for s in spans:
        if not any(max(s[0], c[0]) < min(s[1], c[1]) for c in clean_spans):
            clean_spans.append(s)
    return clean_spans


def build_ui_candidates_for_seed(seed: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int, int, str]:
    """
    Constructs realistic UI candidate nodes for a seed utterance.
    Returns: (candidates, target_idx, action_idx, argument_value)
    """
    cat = seed.get("category", "groceries")
    app = seed.get("app", "App")
    slots = seed.get("slots", {})

    cands: List[Dict[str, Any]] = []
    target_idx = 0
    action_idx = 0
    arg_value = ""

    if cat == "groceries":
        # Zepto / Blinkit / Instamart style screen
        items = slots.get("items", [["item", "1"]])
        first_item = items[0][0] if items else "groceries"
        arg_value = first_item

        cands = [
            # 0: Search input bar (TARGET for typing search)
            {"label": f"Search '{first_item}' or groceries", "role": "SEARCH_INPUT", "bounds": [48, 120, 1032, 220], "is_clickable": True, "is_editable": True, "is_focused": False},
            # 1: Cart button
            {"label": "Cart (0 items)", "role": "BUTTON", "bounds": [920, 120, 1032, 220], "is_clickable": True, "is_editable": False},
            # 2: Category Chip
            {"label": "Fresh Vegetables & Fruits", "role": "TAB", "bounds": [48, 250, 400, 330], "is_clickable": True, "is_editable": False},
            # 3: Category Chip
            {"label": "Dairy, Bread & Eggs", "role": "TAB", "bounds": [420, 250, 800, 330], "is_clickable": True, "is_editable": False},
            # 4: Popular product
            {"label": f"Fresh {first_item.capitalize()} (500g) - Add", "role": "BUTTON", "bounds": [48, 360, 520, 680], "is_clickable": True, "is_editable": False},
            # 5: Dismiss / Back
            {"label": "Back", "role": "BACK_BUTTON", "bounds": [24, 40, 120, 100], "is_clickable": True, "is_editable": False},
        ]
        target_idx = 0
        action_idx = ACTION_TO_IDX["TYPE_TEXT"]

    elif cat == "entertainment":
        # YouTube / Spotify / Instagram style screen
        query = slots.get("query", "video")
        arg_value = query
        cands = [
            # 0: App Search button (TARGET)
            {"label": "Search", "role": "BUTTON", "bounds": [800, 80, 920, 180], "is_clickable": True, "is_editable": False},
            # 1: Home tab
            {"label": "Home", "role": "TAB", "bounds": [48, 2150, 260, 2280], "is_clickable": True, "is_editable": False, "is_selected": True},
            # 2: Subscriptions
            {"label": "Subscriptions", "role": "TAB", "bounds": [280, 2150, 560, 2280], "is_clickable": True, "is_editable": False},
            # 3: Video feed item
            {"label": f"Recommended: {query} official teaser", "role": "ITEM", "bounds": [48, 300, 1032, 900], "is_clickable": True, "is_editable": False},
            # 4: Account profile
            {"label": "Account and settings", "role": "BUTTON", "bounds": [940, 80, 1032, 180], "is_clickable": True, "is_editable": False},
        ]
        target_idx = 0
        action_idx = ACTION_TO_IDX["CLICK"]

    elif cat == "messaging":
        # WhatsApp / Contacts style screen
        contact = slots.get("contact", "contact")
        payload = slots.get("payload", "")
        arg_value = payload if payload else contact
        cands = [
            # 0: Contact row / Chat row (TARGET)
            {"label": f"{contact.capitalize()}: Active now", "role": "ITEM", "bounds": [48, 260, 1032, 400], "is_clickable": True, "is_editable": False},
            # 1: Search chats
            {"label": "Search contacts or messages", "role": "SEARCH_INPUT", "bounds": [48, 120, 950, 220], "is_clickable": True, "is_editable": True},
            # 2: Floating New Chat
            {"label": "New chat", "role": "BUTTON", "bounds": [880, 1950, 1020, 2090], "is_clickable": True, "is_editable": False},
            # 3: Calls tab
            {"label": "Calls", "role": "TAB", "bounds": [720, 120, 950, 200], "is_clickable": True, "is_editable": False},
        ]
        target_idx = 0
        action_idx = ACTION_TO_IDX["CLICK"]

    elif cat == "system":
        # Settings / System quick toggle
        setting = slots.get("setting", "Settings")
        arg_value = setting
        cands = [
            # 0: Toggle Switch (TARGET)
            {"label": f"{setting.capitalize()} Switch Toggle", "role": "BUTTON", "bounds": [850, 280, 1000, 360], "is_clickable": True, "is_editable": False, "is_selected": False},
            # 1: Setting row title
            {"label": f"{setting.capitalize()}", "role": "ITEM", "bounds": [48, 280, 800, 360], "is_clickable": True, "is_editable": False},
            # 2: Search settings
            {"label": "Search settings", "role": "SEARCH_INPUT", "bounds": [48, 100, 1032, 200], "is_clickable": True, "is_editable": True},
            # 3: More options
            {"label": "More options", "role": "BUTTON", "bounds": [950, 40, 1032, 100], "is_clickable": True, "is_editable": False},
        ]
        target_idx = 0
        action_idx = ACTION_TO_IDX["CLICK"]

    else:
        # Generic app opening
        cands = [
            {"label": f"Open {app}", "role": "BUTTON", "bounds": [48, 300, 500, 450], "is_clickable": True, "is_editable": False},
            {"label": "App Drawer", "role": "CONTAINER", "bounds": [0, 0, 1080, 2400], "is_clickable": False, "is_editable": False},
        ]
        target_idx = 0
        action_idx = ACTION_TO_IDX["CLICK"]

    return cands, target_idx, action_idx, arg_value


def extract_entities_from_seed(seed: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Extracts (string_to_find, tag) from seed utterance dictionary."""
    entities: List[Tuple[str, str]] = []
    app = seed.get("app")
    if app:
        entities.append((app, "APP"))
        # Add common aliases if present in text
        if app.lower() == "youtube":
            entities.append(("yt", "APP"))
        elif app.lower() == "whatsapp":
            entities.append(("wa", "APP"))
        elif app.lower() == "instagram":
            entities.append(("insta", "APP"))
        elif app.lower() == "calculator":
            entities.append(("calc", "APP"))

    slots = seed.get("slots", {})
    if "items" in slots:
        for item_entry in slots["items"]:
            if isinstance(item_entry, (list, tuple)) and len(item_entry) >= 2:
                it_name, it_qty = item_entry[0], item_entry[1]
                if it_name:
                    entities.append((str(it_name), "ITEM"))
                if it_qty:
                    entities.append((str(it_qty), "QTY"))

    if "contact" in slots:
        entities.append((str(slots["contact"]), "CONTACT"))

    if "payload" in slots:
        entities.append((str(slots["payload"]), "PAYLOAD"))

    if "query" in slots:
        entities.append((str(slots["query"]), "SEARCH"))

    if "setting" in slots:
        entities.append((str(slots["setting"]), "SEARCH"))

    return entities


def build_v59_seed_items(seed_path: Path) -> List[Dict[str, Any]]:
    """Builds training items from seed utterances with adversarial augmentations."""
    with open(seed_path, "r", encoding="utf-8") as f:
        seeds = json.load(f)

    print(f">> Loaded {len(seeds)} raw seed utterances from {seed_path}")

    v59_items: List[Dict[str, Any]] = []

    for seed in seeds:
        raw_utt = seed["utterance"]
        app_name = seed.get("app", "App")
        entities = extract_entities_from_seed(seed)
        cands, target_idx, action_idx, arg_value = build_ui_candidates_for_seed(seed)

        # -------------------------------------------------------------
        # 1. Base Seed Item (Real human phrasing)
        # -------------------------------------------------------------
        spans = find_entity_spans(raw_utt, entities)
        bio_tags = compute_token_bio_tags(raw_utt, spans, TOKENIZER)
        token_span = find_token_span(raw_utt, arg_value, TOKENIZER)

        v59_items.append(_normalize_item({
            "goal": raw_utt,
            "canonical_goal": f"Open {app_name} and perform {seed.get('intent', 'ACTION')}",
            "dataset_class": "v59_seed_base",
            "current_phase": "NLU_EXECUTE",
            "expected_state": "GOAL_DISAMBIGUATED",
            "candidates": cands,
            "target_idx": target_idx,
            "action_idx": action_idx,
            "slot_type": 0,
            "target_span": token_span,
            "argument_value": arg_value,
            "bio_slot_tags": bio_tags,
            "is_final": False,
            "verification_evidence": 0.0,
        }))

        # -------------------------------------------------------------
        # 2. Typos & Elongations Augmentation ("paneeerr", "doreemonn")
        # -------------------------------------------------------------
        noisy_utt = inject_qwerty_typo(raw_utt, p=0.15)
        # Recalculate spans on noisy text
        noisy_spans = find_entity_spans(noisy_utt, entities)
        noisy_bio = compute_token_bio_tags(noisy_utt, noisy_spans, TOKENIZER)
        noisy_token_span = find_token_span(noisy_utt, arg_value, TOKENIZER)

        v59_items.append(_normalize_item({
            "goal": noisy_utt,
            "canonical_goal": raw_utt,
            "dataset_class": "v59_noisy_typo",
            "current_phase": "NLU_EXECUTE",
            "expected_state": "GOAL_DISAMBIGUATED",
            "candidates": cands,
            "target_idx": target_idx,
            "action_idx": action_idx,
            "slot_type": 0,
            "target_span": noisy_token_span,
            "argument_value": arg_value,
            "bio_slot_tags": noisy_bio,
            "is_final": False,
            "verification_evidence": 0.0,
        }))

        # -------------------------------------------------------------
        # 3. Conversational Preamble Augmentation ("bro please ...")
        # -------------------------------------------------------------
        preamble = random.choice(CONVERSATIONAL_PREAMBLES)
        preamble_utt = preamble + raw_utt
        preamble_spans = find_entity_spans(preamble_utt, entities)
        preamble_bio = compute_token_bio_tags(preamble_utt, preamble_spans, TOKENIZER)
        preamble_token_span = find_token_span(preamble_utt, arg_value, TOKENIZER)

        v59_items.append(_normalize_item({
            "goal": preamble_utt,
            "canonical_goal": raw_utt,
            "dataset_class": "v59_conversational_filler",
            "current_phase": "NLU_EXECUTE",
            "expected_state": "GOAL_DISAMBIGUATED",
            "candidates": cands,
            "target_idx": target_idx,
            "action_idx": action_idx,
            "slot_type": 0,
            "target_span": preamble_token_span,
            "argument_value": arg_value,
            "bio_slot_tags": preamble_bio,
            "is_final": False,
            "verification_evidence": 0.0,
        }))

    print(f">> Synthesized {len(v59_items)} NLU-augmented items from seed utterances.")
    return v59_items


def attach_bio_tags_to_v58_items(v58_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensures every historical V58 baseline item has valid bio_slot_tags tensor
    to avoid shape mismatch during batch collation.
    """
    for it in v58_items:
        if "bio_slot_tags" not in it:
            goal = it.get("goal", "")
            arg_val = it.get("argument_value", "")
            spans = []
            if arg_val:
                spans.append((0, 0, "SEARCH")) # default tag if span found
                spans = find_entity_spans(goal, [(arg_val, "SEARCH")])
            it["bio_slot_tags"] = compute_token_bio_tags(goal, spans, TOKENIZER)
        if "canonical_goal" not in it:
            it["canonical_goal"] = it.get("goal", "")
    return v58_items


def generate_v59_authoritative_dataset() -> List[Dict[str, Any]]:
    """
    Compiles the authoritative V59 dataset:
    - 3,423 V58 baseline items with BIO tag supervision.
    - 1,500+ NLU seed & augmented items (typos, slang, Hinglish, preambles, multi-slot cart).
    """
    print("=" * 80)
    print("COMPILING KARVISH V59 AUTHORITATIVE DATASET")
    print("=" * 80)

    # 1. Ingest V58 Baseline
    v58_ckpt_file = V58_DIR / "checkpoints" / "v58_authoritative_dataset.pt"
    if v58_ckpt_file.exists():
        v58_items = torch.load(v58_ckpt_file, weights_only=False)
        print(f">> Ingested {len(v58_items)} verified V58 foundation items from checkpoint.")
    else:
        v58_items = generate_v58_authoritative_dataset()
        print(f">> Generated {len(v58_items)} verified V58 foundation items.")

    v58_items = attach_bio_tags_to_v58_items(v58_items)

    # 2. Ingest & Augment V59 Seeds
    seed_file = SCRIPT_DIR / "seed_utterances_v59.json"
    if not seed_file.exists():
        raise FileNotFoundError(f"Missing {seed_file}! Run generate_seed_corpus.py first.")

    v59_nlu_items = build_v59_seed_items(seed_file)

    all_items = v58_items + v59_nlu_items
    random.seed(42)
    random.shuffle(all_items)

    print("=" * 80)
    print(f"V59 AUTHORITATIVE DATASET READY: {len(all_items)} TOTAL SAMPLES")
    print(f"  - Baseline V58 items: {len(v58_items)}")
    print(f"  - NLU Augmented items: {len(v59_nlu_items)}")
    print("=" * 80)

    return all_items


if __name__ == "__main__":
    out_dir = SCRIPT_DIR / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = generate_v59_authoritative_dataset()
    out_file = out_dir / "v59_authoritative_dataset.pt"
    torch.save(dataset, out_file)
    print(f">> Successfully saved V59 dataset to {out_file} ({len(dataset)} items).")
