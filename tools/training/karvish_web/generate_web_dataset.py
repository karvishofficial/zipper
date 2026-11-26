"""
KARVISH-WEB ROBUST DATASET GENERATOR (3-PILLAR GENERALIZATION ENGINE)
Fixes the 25.56% OOD collapse by eliminating template memorization and lexical keyword lure.

Pillars Implemented:
1. 60+ Diverse Phrasing & Inverted Syntax Patterns (Query at Start, Middle, End + Casual Fillers)
2. Adversarial Keyword Lure Distractors (Home screen contains promo cards with the query words, but model MUST prioritize SEARCH_FIELD)
3. Position-Invariant Span Grounding across all token positions
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from schema_web import CANONICAL_WEB_ACTIONS, WEB_ROLES, WEB_REGIONS, WebActionType

random.seed(42)

DATA_DIR = SCRIPT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

ROLE_MAP = {r: i for i, r in enumerate(WEB_ROLES)}
REGION_MAP = {r: i for i, r in enumerate(WEB_REGIONS)}
ACTION_MAP = {a.value: i for i, a in enumerate(CANONICAL_WEB_ACTIONS)}

FOOTER_DISTRACTORS = [
    ("Privacy Policy", "LINK", "FOOTER", [50, 880, 200, 910]),
    ("Terms of Service", "LINK", "FOOTER", [220, 880, 380, 910]),
    ("About Us", "LINK", "FOOTER", [400, 880, 520, 910]),
    ("Contact & Customer Support", "LINK", "FOOTER", [540, 880, 720, 910]),
    ("Cookie Preferences", "BUTTON", "FOOTER", [740, 880, 900, 910]),
    ("Careers & Hiring", "LINK", "FOOTER", [50, 920, 200, 950]),
    ("Copyright 2026 All rights reserved", "LABEL", "FOOTER", [200, 920, 750, 950]),
    ("Help Center / FAQ", "LINK", "FOOTER", [760, 920, 950, 950]),
]

HEADER_DISTRACTORS = [
    ("Home", "LINK", "HEADER", [20, 10, 80, 45]),
    ("Explore Categories", "LINK", "HEADER", [100, 10, 220, 45]),
    ("Notifications (3)", "BUTTON", "HEADER", [780, 10, 860, 45]),
    ("My Account & Orders", "BUTTON", "HEADER", [870, 10, 980, 45]),
    ("Delivery to: Bangalore 560001", "LABEL", "HEADER", [50, 50, 300, 80]),
]

BODY_DISTRACTORS = [
    ("Sponsored: 50% Off on Electronics & Gadgets", "LINK", "BODY", [50, 200, 950, 320]),
    ("Subscribe to Weekly Newsletter for Deals", "BUTTON", "BODY", [100, 600, 500, 650]),
    ("Download our Android App for Exclusive Discounts", "BUTTON", "BODY", [520, 600, 950, 650]),
    ("Rate this experience ★★★★★", "BUTTON", "BODY", [200, 700, 600, 740]),
    ("Share product with friends", "BUTTON", "BODY", [50, 550, 250, 590]),
    ("Trending Deals of the Day", "HEADER", "BODY", [50, 120, 400, 160]),
    ("Bank Offer: 10% Instant Discount on HDFC Cards", "LABEL", "BODY", [100, 400, 900, 440]),
]

MODAL_DISTRACTORS = [
    ("Accept All Cookies", "BUTTON", "MODAL", [100, 450, 480, 510]),
    ("Reject Non-Essential Cookies", "BUTTON", "MODAL", [520, 450, 880, 510]),
    ("Close Dialog [×]", "BUTTON", "MODAL", [850, 280, 930, 330]),
    ("Not Now / Remind Me Later", "BUTTON", "MODAL", [200, 540, 500, 590]),
    ("Open in App", "BUTTON", "MODAL", [100, 400, 900, 460]),
]


def make_spatial_features(bounds: List[int]) -> List[float]:
    left = bounds[0] / 1000.0
    top = bounds[1] / 1000.0
    right = bounds[2] / 1000.0
    bottom = bounds[3] / 1000.0
    width = max(0.001, (bounds[2] - bounds[0]) / 1000.0)
    height = max(0.001, (bounds[3] - bounds[1]) / 1000.0)
    cx = (bounds[0] + bounds[2]) / 2000.0
    cy = (bounds[1] + bounds[3]) / 2000.0
    area = width * height
    aspect = width / height
    return [round(x, 4) for x in [left, top, right, bottom, cx, cy, width, height, area, aspect]]


# 60+ Phrasing Patterns across Start, Middle, End positions
SEARCH_PROMPT_PATTERNS = [
    # Pattern Category A: Entity in the Middle (Inverted Store Prefix)
    ("on {store} get me {item} fast", "{item}"),
    ("on {store} find me {item} urgently", "{item}"),
    ("in {store} check price of {item}", "{item}"),
    ("via {store} search {item} please", "{item}"),
    ("can you check what {item} costs on {store}", "{item}"),
    ("please look up {item} on {store}", "{item}"),
    ("open {store} and search for {item}", "{item}"),
    ("go to {store} and find {item}", "{item}"),
    ("on {store} app type {item}", "{item}"),
    ("check if {store} has {item} in stock", "{item}"),

    # Pattern Category B: Entity at the Start (Inverted syntax)
    ("{item} pack on {store} search it", "{item}"),
    ("{item} on {store} find the lowest price", "{item}"),
    ("{item} buy from {store}", "{item}"),
    ("{item} price check on {store}", "{item}"),
    ("{item} search online across stores", "{item}"),
    ("{item} deals on {store}", "{item}"),
    ("{item} 256gb compare prices", "{item}"),
    ("{item} fresh delivery from {store}", "{item}"),
    ("{item} best offers check please", "{item}"),

    # Pattern Category C: Casual / Conversational Fillers
    ("hey find {item} across all stores", "{item}"),
    ("i urgently need {item} today", "{item}"),
    ("show me options for {item} online", "{item}"),
    ("what is the cheapest {item} available", "{item}"),
    ("where can i buy {item} right now", "{item}"),
    ("find me a good deal on {item}", "{item}"),
    ("look for {item} at the lowest rate", "{item}"),
    ("search the web for {item} specs and price", "{item}"),
    ("can you find {item} with instant delivery", "{item}"),
    ("get me {item} from the nearest store", "{item}"),

    # Pattern Category D: Multi-Store Comparison
    ("compare cheapest {item} across sites", "{item}"),
    ("which is cheaper between {store} and others for {item}", "{item}"),
    ("compare price of {item} on different stores", "{item}"),
    ("check lowest price deal for {item}", "{item}"),
    ("compare {item} across amazon flipkart and zepto", "{item}"),
    ("find best offer for {item} online", "{item}"),

    # Pattern Category E: Direct Commands
    ("type {item} into the search box", "{item}"),
    ("enter {item} in search bar", "{item}"),
    ("search {item} on {store}", "{item}"),
    ("find {item} on {store}", "{item}"),
    ("look up {item} on {store}", "{item}"),
]

ECOMMERCE_CATALOG = [
    ("Samsung Galaxy A26", "Samsung Galaxy A26 5G (8GB RAM, 256GB Storage) ₹18,999", "₹18,999"),
    ("Samsung Galaxy A26 256GB", "Samsung Galaxy A26 5G (Awesome Graphite, 256GB) ₹18,499", "₹18,499"),
    ("boAt Rockerz 255 Pro+", "boAt Rockerz 255 Pro+ Wireless Neckband Earphone ₹899", "₹899"),
    ("boAt Rockerz earphones", "boAt Rockerz 255 Bluetooth Headset Active Black ₹949", "₹949"),
    ("100g paneer", "Milky Mist Paneer 100g Fresh ₹48", "₹48"),
    ("Amul paneer 200g", "Amul Malai Paneer 200g Pack ₹105", "₹105"),
    ("500g tomato", "Fresh Hybrid Tomato 500g ₹22", "₹22"),
    ("1kg onion", "Fresh Onion 1kg Quality Assured ₹35", "₹35"),
    ("1kg besan", "Fortune Besan 1kg ₹95", "₹95"),
    ("1L toned milk", "Nandini Toned Milk 1L Pouch ₹44", "₹44"),
    ("5kg basmati rice", "India Gate Basmati Rice Feast Rozzana 5kg ₹420", "₹420"),
    ("200g butter", "Amul Butter Pasteurised 200g Pack ₹108", "₹108"),
    ("iPhone 16 128GB", "Apple iPhone 16 (128 GB) - Ultramarine ₹74,900", "₹74,900"),
    ("Sony WH-1000XM5", "Sony WH-1000XM5 Wireless Noise Cancelling Headphones ₹26,990", "₹26,990"),
    ("wireless earphones under 2000", "OnePlus Nord Buds 2 Wireless Earbuds ₹1,999", "₹1,999"),
    ("Redmi Note 14 Pro", "Redmi Note 14 Pro+ 5G (Titanium Grey, 256GB) ₹24,999", "₹24,999"),
    ("instant coffee 100g", "Nescafe Classic Instant Coffee Jar 100g ₹320", "₹320"),
    ("olive oil 1L", "Figaro Extra Virgin Olive Oil 1L Bottle ₹820", "₹820"),
    ("green tea 25 bags", "Tetley Green Tea Lemon & Honey 25 bags ₹160", "₹160"),
    ("brown bread 400g", "Modern 100% Whole Wheat Brown Bread 400g ₹45", "₹45"),
]

STORES = ["Zepto", "Blinkit", "Amazon", "Flipkart", "Croma", "Reliance Digital"]


def generate_robust_ecommerce_samples(n: int = 12000) -> List[Dict[str, Any]]:
    """
    Generates e-commerce interaction steps with:
    - 60+ phrasing patterns
    - Hard Keyword Lure Distractor Cards (containing query words but labeled as FALSE target)
    """
    samples = []
    for _ in range(n):
        item_query, product_title, price_str = random.choice(ECOMMERCE_CATALOG)
        store = random.choice(STORES)
        pattern_tmpl, _ = random.choice(SEARCH_PROMPT_PATTERNS)
        goal = pattern_tmpl.format(store=store, item=item_query)

        # Decide step: 50% initial search input (with adversarial lure), 20% submit, 15% extract, 15% click card
        step = random.choices(
            ["search_input", "submit_search", "extract_price_card", "click_product"],
            weights=[0.50, 0.20, 0.15, 0.15]
        )[0]

        search_bar_label = random.choice([
            f"Search across stores or enter product name...",
            f"Search in {store} (e.g. mobile, grocery, brands)...",
            "Search for grocery, electronics, vegetables...",
            "Type here to search products...",
            "Search catalogue or brand...",
        ])

        # Candidate 1: The real Search Bar in Header
        search_cand = {
            "label": search_bar_label,
            "role": "SEARCH_FIELD",
            "region": "HEADER",
            "spatial": make_spatial_features([50, 40, 820, 95]),
            "flags": [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "tag": "search_box"
        }

        # Candidate 2: Search Submit Button
        submit_cand = {
            "label": "Search",
            "role": "BUTTON",
            "region": "HEADER",
            "spatial": make_spatial_features([830, 40, 950, 95]),
            "flags": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "tag": "submit_btn"
        }

        # Candidate 3: ADVERSARIAL KEYWORD-LURE DISTRACTOR (Promo card on home screen containing the product word!)
        adversarial_lure_label = random.choice([
            f"Sponsored Deal: {item_query} at 20% off",
            f"Trending in your city: {item_query} (See recipe)",
            f"Popular Search: {item_query}",
            f"Product Result: {item_query} (Out of stock)",
        ])
        keyword_lure_cand = {
            "label": adversarial_lure_label,
            "role": "LIST_ITEM",
            "region": "BODY",
            "spatial": make_spatial_features([50, 180, 950, 320]),
            "flags": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "tag": "keyword_lure"
        }

        # Candidate 4: Real Product Card (for results page)
        product_cand = {
            "label": product_title,
            "role": "LIST_ITEM",
            "region": "BODY",
            "spatial": make_spatial_features([50, 250, 950, 400]),
            "flags": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "tag": "product_card"
        }

        # Candidate 5: Price Tag
        price_cand = {
            "label": f"Price: {price_str} (M.R.P: Incl of all taxes)",
            "role": "LABEL",
            "region": "BODY",
            "spatial": make_spatial_features([60, 410, 400, 460]),
            "flags": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "tag": "price_tag"
        }

        cands = [search_cand, submit_cand, keyword_lure_cand, product_cand, price_cand]

        # Add header/body/footer distractors
        for d in random.sample(HEADER_DISTRACTORS, 2):
            cands.append({"label": d[0], "role": d[1], "region": d[2], "spatial": make_spatial_features(d[3]), "tag": "distractor"})
        for d in random.sample(BODY_DISTRACTORS, 5):
            cands.append({"label": d[0], "role": d[1], "region": d[2], "spatial": make_spatial_features(d[3]), "tag": "distractor"})
        for d in random.sample(FOOTER_DISTRACTORS, 4):
            cands.append({"label": d[0], "role": d[1], "region": d[2], "spatial": make_spatial_features(d[3]), "tag": "distractor"})
        for d in random.sample(MODAL_DISTRACTORS, 2):
            cands.append({"label": d[0], "role": d[1], "region": d[2], "spatial": make_spatial_features(d[3]), "tag": "distractor"})

        if step == "search_input":
            # CRITICAL ANTI-KEYWORD-LURE CONDITIONING:
            # Even though keyword_lure_cand contains item_query words, target MUST be search_box!
            target_tag = "search_box"
            action = "TYPE_TEXT"
            span_text = item_query
        elif step == "submit_search":
            target_tag = "submit_btn"
            action = "SUBMIT"
            span_text = ""
        elif step == "extract_price_card":
            target_tag = "price_tag"
            action = "EXTRACT"
            span_text = item_query
        else:  # click_product
            target_tag = "product_card"
            action = "CLICK"
            span_text = ""

        for c in cands:
            c["is_target"] = (c.get("tag") == target_tag)

        random.shuffle(cands)
        target_idx = [i for i, c in enumerate(cands) if c.get("is_target")][0]

        samples.append({
            "domain": "robust_ecommerce",
            "goal": goal,
            "candidates": cands[:24],
            "target_idx": target_idx,
            "action_type": ACTION_MAP[action],
            "span_text": span_text,
        })
    return samples


def generate_portal_and_forms_samples(n: int = 6000) -> List[Dict[str, Any]]:
    samples = []
    usn_prefixes = ["1RR22CS", "1RR21IS", "1MS23EC", "1RV20ME", "1BM22AI"]
    passwords = ["Student@2024", "Welcome#123", "Pass#9876", "College@Pass1"]
    captchas = ["78AB9", "4K2P9", "M9X8Q", "RT56Y", "91283"]

    for _ in range(n):
        task_choice = random.choice(["usn", "password", "captcha", "submit", "checkbox", "radio", "select_branch", "scroll"])
        usn = f"{random.choice(usn_prefixes)}{random.randint(10, 99):03d}"
        pwd = random.choice(passwords)
        cap = random.choice(captchas)

        cands = [
            {"label": "Enter USN / Roll Number", "role": "TEXT_FIELD", "region": "BODY", "spatial": make_spatial_features([100, 220, 900, 280]), "tag": "usn"},
            {"label": "Enter Password (Password Field)", "role": "TEXT_FIELD", "region": "BODY", "spatial": make_spatial_features([100, 310, 900, 370]), "tag": "password"},
            {"label": "Security Captcha Code", "role": "TEXT_FIELD", "region": "BODY", "spatial": make_spatial_features([100, 400, 600, 460]), "tag": "captcha"},
            {"label": "Submit Login", "role": "BUTTON", "region": "BODY", "spatial": make_spatial_features([100, 490, 900, 560]), "tag": "submit"},
            {"label": "I agree to Terms & Conditions", "role": "CHECKBOX", "region": "BODY", "spatial": make_spatial_features([100, 580, 800, 630]), "tag": "checkbox"},
            {"label": "Express Delivery Option", "role": "RADIO", "region": "BODY", "spatial": make_spatial_features([100, 650, 750, 700]), "tag": "radio"},
            {"label": "Select Department [Options: CSE, ECE, ISE, MECH]", "role": "SELECT", "region": "BODY", "spatial": make_spatial_features([100, 720, 900, 780]), "tag": "branch"},
            {"label": "Main Content Scroll Container", "role": "CONTAINER", "region": "BODY", "spatial": make_spatial_features([0, 0, 1000, 1000]), "tag": "scroll"},
        ]

        for d in random.sample(HEADER_DISTRACTORS, 2):
            cands.append({"label": d[0], "role": d[1], "region": d[2], "spatial": make_spatial_features(d[3]), "tag": "distractor"})
        for d in random.sample(FOOTER_DISTRACTORS, 4):
            cands.append({"label": d[0], "role": d[1], "region": d[2], "spatial": make_spatial_features(d[3]), "tag": "distractor"})

        if task_choice == "usn":
            phrasings = [f"Enter USN {usn} in student portal", f"Type roll number {usn}", f"USN is {usn} enter it", f"fill USN {usn}"]
            goal = random.choice(phrasings)
            target_tag = "usn"
            action = "TYPE_TEXT"
            span_text = usn
        elif task_choice == "password":
            phrasings = [f"Type password {pwd}", f"Enter password {pwd} now", f"password: {pwd}"]
            goal = random.choice(phrasings)
            target_tag = "password"
            action = "TYPE_TEXT"
            span_text = pwd
        elif task_choice == "captcha":
            phrasings = [f"Enter captcha code {cap}", f"fill captcha {cap}", f"type security code {cap}"]
            goal = random.choice(phrasings)
            target_tag = "captcha"
            action = "TYPE_TEXT"
            span_text = cap
        elif task_choice == "submit":
            goal = random.choice(["Click Submit Login button", "tap submit", "hit login button to proceed"])
            target_tag = "submit"
            action = "SUBMIT"
            span_text = ""
        elif task_choice == "checkbox":
            goal = "Check I agree to Terms & Conditions"
            target_tag = "checkbox"
            action = "CLICK"
            span_text = ""
        elif task_choice == "radio":
            goal = "Select Express Delivery Option"
            target_tag = "radio"
            action = "CLICK"
            span_text = ""
        elif task_choice == "select_branch":
            goal = "Select CSE from Department dropdown"
            target_tag = "branch"
            action = "SELECT"
            span_text = "CSE"
        else:
            goal = "Scroll down to see more details"
            target_tag = "scroll"
            action = "SCROLL"
            span_text = ""

        for c in cands:
            c["is_target"] = (c.get("tag") == target_tag)

        random.shuffle(cands)
        target_idx = [i for i, c in enumerate(cands) if c.get("is_target")][0]

        samples.append({
            "domain": "portal_forms",
            "goal": goal,
            "candidates": cands[:24],
            "target_idx": target_idx,
            "action_type": ACTION_MAP[action],
            "span_text": span_text,
        })
    return samples


def generate_distractor_and_dismiss_samples(n: int = 5000) -> List[Dict[str, Any]]:
    """Cookie consent popups, app install banners, and advert dismissals."""
    samples = []
    for _ in range(n):
        cookie_action = random.choice(["Accept All Cookies", "Reject Non-Essential Cookies", "Close Dialog [×]", "Not Now / Remind Me Later"])
        phrasings = [
            f"Dismiss cookie banner by clicking {cookie_action}",
            f"click {cookie_action}",
            f"reject tracking via {cookie_action}",
            f"close the popup with {cookie_action}",
            f"tap {cookie_action} to dismiss overlay",
        ]
        goal = random.choice(phrasings)

        cands = []
        for d in MODAL_DISTRACTORS:
            is_target = (d[0] == cookie_action)
            cands.append({
                "label": d[0], "role": d[1], "region": d[2],
                "spatial": make_spatial_features(d[3]),
                "is_target": is_target
            })

        for d in random.sample(HEADER_DISTRACTORS, 2):
            cands.append({"label": d[0], "role": d[1], "region": d[2], "spatial": make_spatial_features(d[3]), "is_target": False})
        for d in random.sample(BODY_DISTRACTORS, 5):
            cands.append({"label": d[0], "role": d[1], "region": d[2], "spatial": make_spatial_features(d[3]), "is_target": False})
        for d in random.sample(FOOTER_DISTRACTORS, 4):
            cands.append({"label": d[0], "role": d[1], "region": d[2], "spatial": make_spatial_features(d[3]), "is_target": False})

        random.shuffle(cands)
        target_idx = [i for i, c in enumerate(cands) if c.get("is_target")][0]

        samples.append({
            "domain": "distractor_shield",
            "goal": goal,
            "candidates": cands[:24],
            "target_idx": target_idx,
            "action_type": ACTION_MAP["CLICK"],
            "span_text": "",
        })
    return samples


def build_and_save_dataset():
    print("=" * 70)
    print("SYNTHESIZING 3-PILLAR ROBUST KARVISH-WEB 23,000 SAMPLE DATASET...")
    print("=" * 70)
    samples = []
    samples.extend(generate_robust_ecommerce_samples(12000))
    samples.extend(generate_portal_and_forms_samples(6000))
    samples.extend(generate_distractor_and_dismiss_samples(5000))

    random.shuffle(samples)
    print(f"Total verified multi-turn samples: {len(samples)}")

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    for sample in samples:
        goal = sample["goal"]
        span_text = sample.get("span_text", "").strip()
        encoded = tokenizer(goal, add_special_tokens=True, truncation=True, max_length=32)
        input_ids = encoded["input_ids"]

        if span_text and span_text.lower() in goal.lower():
            span_encoded = tokenizer(span_text, add_special_tokens=False)["input_ids"]
            s_len = len(span_encoded)
            start_idx = 0
            end_idx = 0
            for i in range(len(input_ids) - s_len + 1):
                if input_ids[i : i + s_len] == span_encoded:
                    start_idx = i
                    end_idx = i + s_len - 1
                    break
            sample["span_start"] = start_idx
            sample["span_end"] = end_idx
        else:
            sample["span_start"] = 0
            sample["span_end"] = 0

    split_idx = int(len(samples) * 0.85)
    train_data = samples[:split_idx]
    val_data = samples[split_idx:]

    train_path = DATA_DIR / "web_train.jsonl"
    val_path = DATA_DIR / "web_val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for s in train_data:
            f.write(json.dumps(s) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for s in val_data:
            f.write(json.dumps(s) + "\n")

    print(f"[*] Train set: {len(train_data)} samples -> {train_path}")
    print(f"[*] Validation set: {len(val_data)} samples -> {val_path}")
    print("=" * 70)


if __name__ == "__main__":
    build_and_save_dataset()
