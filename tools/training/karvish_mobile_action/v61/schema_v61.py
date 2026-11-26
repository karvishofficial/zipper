"""
KARVISH MULTIMODAL UNIVERSAL V61 — CANONICAL SCHEMA & DATA CONTRACT
Authoritative Data Specification for Karvish Universal Agent V61.

Key Advancements over V60:
  1. Integrates 512-D Visual Feature Embeddings for candidate UI nodes.
  2. Expands Vocabulary to 32,522 WordPiece Tokens with Subword Mean Decomposition.
  3. Supports Naked / Unlabeled Icon Affinity Classes:
     - 🔍 Search, 🛒 Cart, ✕ Close/Dismiss, 👤 Profile, ☰ Menu, ❤️ Favorite, 
     - ➕ Add, ▶️ Play, ⏸️ Pause, 🟢 Toggle_ON, ⚪ Toggle_OFF.
  4. Multi-Turn Gated Hindsight Distillation (GHD) Signals:
     - `hindsight_next_screen_delta`: Observed UI state progression flag.
     - `popup_dismissal_required`: Flag for interstitial promo overlays.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
V60_DIR = SCRIPT_DIR.parent / "v60"
V59_DIR = SCRIPT_DIR.parent / "v59"
sys.path.insert(0, str(V60_DIR))
sys.path.insert(0, str(V59_DIR))

from schema_v60 import (
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
)

DATASET_VERSION = "V61"
MODEL_VERSION = "KarvishModelV61"

# Visual Icon Affordance Classes for MobileNetV4
VISUAL_ICON_CLASSES = [
    "none",          # 0: Standard text button (no icon)
    "search",        # 1: Magnifying glass 🔍
    "cart",          # 2: Shopping cart / bag 🛒
    "close",         # 3: Circular ✕ or cross dismiss button
    "profile",       # 4: Avatar / user icon 👤
    "menu",          # 5: Hamburger / three dots ☰
    "favorite",      # 6: Heart / star ❤️
    "add",           # 7: Plus / new button ➕
    "play",          # 8: Play triangle ▶️
    "pause",         # 9: Pause bars ⏸️
    "toggle_on",     # 10: Active green/blue toggle switch 🟢
    "toggle_off",    # 11: Inactive gray toggle switch ⚪
    "back",          # 12: Back arrow / chevron ⬅️
    "share",         # 13: Share arrow / node ↗️
    "filter",        # 14: Filter funnel / sliders ⚡
]

VISUAL_CLASS_TO_IDX = {name: idx for idx, name in enumerate(VISUAL_ICON_CLASSES)}
IDX_TO_VISUAL_CLASS = {idx: name for idx, name in enumerate(VISUAL_ICON_CLASSES)}
NUM_VISUAL_CLASSES = len(VISUAL_ICON_CLASSES)
