"""
KARVISH MULTIMODAL UNIVERSAL V62 — CANONICAL SCHEMA & DATA CONTRACT
Authoritative Data Specification for Karvish Universal Agent V62.

Key Advancements over V61:
  1. Unified (K + 1) Differentiable Action Space:
     - `c_complete_goal` is candidate index K (first-class candidate in the softmax).
     - No -1000 penalty gates, no 0.15 hard clamps, no 0.70 runtime thresholds.
     - Autonomous stopping and anti-wandering emerge naturally from Softmax competition.
  2. 512-D SIMD-Aligned Latent Representations:
     - Candidate encodings, visual features, and goal embeddings aligned to 512-D (8x64 ARM NEON lanes).
  3. Anti-Wandering Contrastive Distractor Supervision:
     - `is_distractor`: Ground truth flag for feed cards, sponsored banners, and off-path clickbait.
  4. Multi-Slot Conversational NLU:
     - 15-tag BIO sequence tagging contract for multi-lingual and slang argument extraction.
  5. Observable Delta & Evidence Feature Ingestion:
     - Screen delta and external verifier signals ingested as continuous input embeddings,
       never as decision overrides or output clamps.
"""

from __future__ import annotations

import sys
from pathlib import Path
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
V61_DIR = SCRIPT_DIR.parent / "v61"
V60_DIR = SCRIPT_DIR.parent / "v60"
V59_DIR = SCRIPT_DIR.parent / "v59"

sys.path.insert(0, str(V61_DIR))
sys.path.insert(0, str(V60_DIR))
sys.path.insert(0, str(V59_DIR))

from schema_v61 import (
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
    APP_DOMAINS,
    VISUAL_ICON_CLASSES,
    VISUAL_CLASS_TO_IDX,
    IDX_TO_VISUAL_CLASS,
    NUM_VISUAL_CLASSES,
)

DATASET_VERSION = "V62"
MODEL_VERSION = "KarvishModelV62"

# Canonical Completion Identifier in Unified Action Space
COMPLETE_CANDIDATE_ID = "c_complete_goal"
COMPLETE_SEMANTIC_LABEL = "complete"
COMPLETE_ROLE_NAME = "system"

# Feature Dimension Standards (ARM NEON SIMD Vector Aligned)
LATENT_DIM = 512
TASK_STATE_DIM = 128
VISUAL_FEATURE_DIM = 512
MAX_CANDIDATES = 32
MAX_GOAL_TOKENS = 64
NUM_ACTIONS = len(CANONICAL_ACTIONS)  # 8

__all__ = [
    "DATASET_VERSION",
    "MODEL_VERSION",
    "COMPLETE_CANDIDATE_ID",
    "COMPLETE_SEMANTIC_LABEL",
    "COMPLETE_ROLE_NAME",
    "LATENT_DIM",
    "TASK_STATE_DIM",
    "VISUAL_FEATURE_DIM",
    "MAX_CANDIDATES",
    "MAX_GOAL_TOKENS",
    "NUM_ACTIONS",
    "CanonicalActionType",
    "CANONICAL_ACTIONS",
    "ACTION_TO_IDX",
    "UniversalSemanticRole",
    "UNIVERSAL_ROLES",
    "ROLE_TO_IDX",
    "NUM_UNIVERSAL_ROLES",
    "UNIVERSAL_SUBGOALS",
    "SUBGOAL_TO_IDX",
    "NUM_UNIVERSAL_SUBGOALS",
    "ScrollDirection",
    "ArgumentSlotType",
    "BIO_SLOT_TAGS",
    "BIO_TAG_TO_IDX",
    "IDX_TO_BIO_TAG",
    "NUM_BIO_TAGS",
    "APP_DOMAINS",
    "VISUAL_ICON_CLASSES",
    "VISUAL_CLASS_TO_IDX",
    "IDX_TO_VISUAL_CLASS",
    "NUM_VISUAL_CLASSES",
]
