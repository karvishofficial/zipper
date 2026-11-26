"""
KARVISH UNIVERSAL V60 — CANONICAL SCHEMA & DATA CONTRACT
Authoritative Data Specification for Karvish Universal Agent V60.

Key Advancements over V59:
  1. Preserves 15-tag BIO Slot Tagging Contract (100% backwards-compatible with Android tokenizer & ONNX).
  2. Introduces Explicit Multi-Turn Milestone Supervision:
     - `is_terminal_complete`: Ground-truth binary flag (1.0 only on true task completion).
     - `is_near_miss_negative`: Flag for tricky intermediate states that resemble completion.
     - `is_slip_stagnation`: Flag for recovery turns (delta=0, dialog dismissal, retry).
  3. Universal 50-Domain App Categorization:
     - E-commerce, Food Delivery, Ride Hailing, Messaging, Productivity, System Settings,
       Media/Streaming, Utilities, Travel, Social, Finance, Health.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
V59_DIR = SCRIPT_DIR.parent / "v59"
V58_DIR = SCRIPT_DIR.parent / "v58"
sys.path.insert(0, str(V59_DIR))
sys.path.insert(0, str(V58_DIR))

from schema_v59 import (
    CanonicalActionType,
    CANONICAL_ACTIONS,
    ACTION_TO_IDX,
    UniversalSemanticRole,
    UNIVERSAL_ROLES,
    ROLE_TO_IDX,
    NUM_UNIVERSAL_ROLES,
    UNIVERSAL_SUBGOALS,
    SUBGOAL_TO_IDX,
    NUM_UNIVERSAL_SUBGOALS,
    ScrollDirection,
    ArgumentSlotType,
    BIO_SLOT_TAGS,
    BIO_TAG_TO_IDX,
    IDX_TO_BIO_TAG,
    NUM_BIO_TAGS,
)

DATASET_VERSION = "V60"
MODEL_VERSION = "KarvishModelV60"

# 50 Universal App Domains for Zero-Shot Cross-App Generalization
APP_DOMAINS = [
    # Core Anchors
    "youtube", "whatsapp", "instagram", "settings", "calculator", "zepto", "clock", "contacts", "playstore",
    # E-Commerce & Shopping
    "amazon", "flipkart", "blinkit", "instamart", "meesho", "myntra", "ajio", "nykaa", "tatacliq",
    # Food & Grocery
    "swiggy", "zomato", "eatclub", "dominos", "mcdonalds", "bigbasket", "bbnow", "dunzo",
    # Ride Hailing & Travel
    "uber", "ola", "rapido", "makemytrip", "goibibo", "irctc", "redbus", "airbnb",
    # Productivity & Notes
    "keep", "gmail", "outlook", "slack", "notion", "google_docs", "drive", "files",
    # Social & Media
    "reddit", "twitter", "linkedin", "spotify", "wynk", "jiosaavn", "prime_video", "netflix", "hotstar"
]
