"""
KARVISH UNIVERSAL V59 — CANONICAL SCHEMA & DATA CONTRACT
Authoritative Data Specification for Karvish Universal Agent V59.

Preserves V52-LOCKED-GOLD, V56, V57, and V58 contracts 100% intact.
Adds:
  1. Token-level BIO Slot Tags for multi-slot extraction:
     ["O", "B-APP", "I-APP", "B-ITEM", "I-ITEM", "B-QTY", "I-QTY",
      "B-CONTACT", "I-CONTACT", "B-PAYLOAD", "I-PAYLOAD", "B-SEARCH",
      "I-SEARCH", "B-TIME", "I-TIME"]
  2. Contrastive Goal Invariance specification (pairings of messy/slang goals with canonical goals).
  3. Universal NLU evaluation benchmarks for intent accuracy, slot F1, and typo robustness.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

SCRIPT_DIR = Path(__file__).resolve().parent
V58_DIR = SCRIPT_DIR.parent / "v58"
V57_DIR = SCRIPT_DIR.parent / "v57"
V56_DIR = SCRIPT_DIR.parent / "v56"
sys.path.insert(0, str(V58_DIR))
sys.path.insert(0, str(V57_DIR))
sys.path.insert(0, str(V56_DIR))

from schema_v58 import (
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
)

DATASET_VERSION = "V59"
MODEL_VERSION = "KarvishModelV59"

BIO_SLOT_TAGS = [
    "O",          # 0: Outside / Filler / Preambles
    "B-APP",      # 1: Beginning of Target Application
    "I-APP",      # 2: Inside of Target Application
    "B-ITEM",     # 3: Beginning of Grocery / Product Item
    "I-ITEM",     # 4: Inside of Grocery / Product Item
    "B-QTY",      # 5: Beginning of Quantity / Count
    "I-QTY",      # 6: Inside of Quantity / Count
    "B-CONTACT",  # 7: Beginning of Contact Name
    "I-CONTACT",  # 8: Inside of Contact Name
    "B-PAYLOAD",  # 9: Beginning of Message Body / Query Argument
    "I-PAYLOAD",  # 10: Inside of Message Body / Query Argument
    "B-SEARCH",   # 11: Beginning of Search Query / Video Name / Title
    "I-SEARCH",   # 12: Inside of Search Query / Video Name / Title
    "B-TIME",     # 13: Beginning of Time / Duration / Date
    "I-TIME",     # 14: Inside of Time / Duration / Date
]

BIO_TAG_TO_IDX = {tag: idx for idx, tag in enumerate(BIO_SLOT_TAGS)}
IDX_TO_BIO_TAG = {idx: tag for idx, tag in enumerate(BIO_SLOT_TAGS)}
NUM_BIO_TAGS = len(BIO_SLOT_TAGS)
