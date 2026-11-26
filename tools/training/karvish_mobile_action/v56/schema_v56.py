"""
KARVISH UNIVERSAL V56 — CANONICAL SCHEMA & DATA CONTRACT
Authoritative Data Specification for Karvish Universal Agent V56.

Governing Standard: Karvish Agent Constitution (AGENTS.md)
Target Architecture: Karvish V56 Universal Action Policy
Design Pillars:
  1. Universal Semantic UI Roles (32 canonical roles, zero app-specific heuristics).
  2. Contextual Role Compatibility: f(Role, Subgoal, Context, Bounds).
  3. Dual-Slot Action-Conditioned Span Extraction: separates recipient from message payload.
  4. Multimodal Delta State: DeltaTree + DeltaSemantics + DeltaViewport + Keyboard.
  5. Evidence-Gated Completion: P(complete | evidence=0) < 0.15.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


# =====================================================================
# 1. CANONICAL ACTION VOCABULARY (8 Actions)
# =====================================================================
class CanonicalActionType(str, Enum):
    CLICK = "CLICK"
    LONG_PRESS = "LONG_PRESS"
    SCROLL = "SCROLL"
    OPEN_APP = "OPEN_APP"
    TYPE_TEXT = "TYPE_TEXT"
    HOME = "HOME"
    BACK = "BACK"
    WAIT = "WAIT"

    @classmethod
    def from_raw(cls, raw: Optional[str]) -> "CanonicalActionType":
        if not raw:
            return cls.WAIT
        clean = str(raw).upper().strip().replace(" ", "_").replace("-", "_")
        if clean in ("TAP", "CLICK", "PRESS", "SUBMIT", "PRESS_ENTER", "IME_ENTER", "ENTER", "SEARCH", "CONFIRM", "SELECT"):
            return cls.CLICK
        if clean in ("LONG_PRESS", "LONGPRESS", "LONG_CLICK"):
            return cls.LONG_PRESS
        if clean in ("SCROLL", "SWIPE", "FLING", "DRAG"):
            return cls.SCROLL
        if clean in ("OPEN_APP", "LAUNCH_APP", "OPENAPP", "LAUNCH", "OPEN"):
            return cls.OPEN_APP
        if clean in ("TYPE_TEXT", "TYPE", "INPUT_TEXT", "INPUT", "SET_TEXT"):
            return cls.TYPE_TEXT
        if clean in ("HOME", "GLOBAL_HOME"):
            return cls.HOME
        if clean in ("BACK", "GLOBAL_BACK"):
            return cls.BACK
        if clean in ("WAIT", "SLEEP", "COMPLETE", "SUCCESS"):
            return cls.WAIT
        return cls.WAIT


CANONICAL_ACTIONS: List[str] = [a.value for a in CanonicalActionType]
ACTION_TO_IDX: Dict[str, int] = {a: i for i, a in enumerate(CANONICAL_ACTIONS)}


# =====================================================================
# 2. UNIVERSAL UI SEMANTIC ROLES (32 Canonical Roles)
# =====================================================================
class UniversalSemanticRole(str, Enum):
    UNKNOWN = "UNKNOWN"                          # 0
    BUTTON = "BUTTON"                            # 1
    EDITABLE_FIELD = "EDITABLE_FIELD"            # 2
    SEARCH_INPUT = "SEARCH_INPUT"                # 3
    MESSAGE_INPUT = "MESSAGE_INPUT"              # 4
    CONTACT_ITEM = "CONTACT_ITEM"                # 5
    CHAT_ITEM = "CHAT_ITEM"                      # 6
    PRODUCT_CARD = "PRODUCT_CARD"                # 7
    ADD_TO_CART_BUTTON = "ADD_TO_CART_BUTTON"    # 8
    PRICE_TAG = "PRICE_TAG"                      # 9
    APP_ICON = "APP_ICON"                        # 10
    HEADER = "HEADER"                            # 11
    BACK_BUTTON = "BACK_BUTTON"                  # 12
    MENU_ITEM = "MENU_ITEM"                      # 13
    TAB_ITEM = "TAB_ITEM"                        # 14
    LIST_ITEM = "LIST_ITEM"                      # 15
    SCROLL_CONTAINER = "SCROLL_CONTAINER"        # 16
    DIALOG_PROMPT = "DIALOG_PROMPT"              # 17
    CONFIRM_BUTTON = "CONFIRM_BUTTON"            # 18
    CANCEL_BUTTON = "CANCEL_BUTTON"              # 19
    CHECKBOX_TOGGLE = "CHECKBOX_TOGGLE"          # 20
    FILTER_CHIP = "FILTER_CHIP"                  # 21
    PROFILE_HEADER = "PROFILE_HEADER"            # 22
    STATUS_BAR = "STATUS_BAR"                    # 23
    MEDIA_ITEM = "MEDIA_ITEM"                    # 24
    RATING_REVIEW = "RATING_REVIEW"              # 25
    LINK_URL = "LINK_URL"                        # 26
    KEYBOARD_KEY = "KEYBOARD_KEY"                # 27
    SEND_BUTTON = "SEND_BUTTON"                  # 28
    SEARCH_SUBMIT_BUTTON = "SEARCH_SUBMIT_BUTTON"# 29
    CLEAR_BUTTON = "CLEAR_BUTTON"                # 30
    REFRESH_BUTTON = "REFRESH_BUTTON"            # 31


UNIVERSAL_ROLES: List[str] = [r.value for r in UniversalSemanticRole]
ROLE_TO_IDX: Dict[str, int] = {r: i for i, r in enumerate(UNIVERSAL_ROLES)}
NUM_UNIVERSAL_ROLES: int = len(UNIVERSAL_ROLES)  # 32


# =====================================================================
# 3. UNIVERSAL SUBGOALS (12 Canonical Subgoals)
# =====================================================================
UNIVERSAL_SUBGOALS: List[str] = [
    "LOCATE_INPUT",       # 0: Find search or text entry field
    "ENTER_QUERY",        # 1: Type query text or message
    "SUBMIT_SEARCH",      # 2: Commit search via IME or submit button
    "SELECT_RESULT",      # 3: Click matching result / contact / card
    "COMPOSE_MESSAGE",    # 4: Type into messaging text box
    "SEND_MESSAGE",       # 5: Tap send button
    "ADD_TO_CART",        # 6: Add product to shopping cart
    "VIEW_PROFILE",       # 7: Inspect profile or details header
    "NAVIGATE_TAB",       # 8: Switch bottom nav / tab
    "TOGGLE_SETTING",     # 9: Change switch / checkbox state
    "RECOVER_ERROR",      # 10: Dismiss dialog / dismiss keyboard / scroll away from trap
    "COMPLETE_TASK",      # 11: Task completed with observable evidence
]
SUBGOAL_TO_IDX: Dict[str, int] = {s: i for i, s in enumerate(UNIVERSAL_SUBGOALS)}
NUM_UNIVERSAL_SUBGOALS: int = len(UNIVERSAL_SUBGOALS)  # 12


# =====================================================================
# 4. SCROLL DIRECTIONS
# =====================================================================
class ScrollDirection(str, Enum):
    DOWN = "DOWN"
    UP = "UP"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    NONE = "NONE"

    @classmethod
    def from_raw(cls, raw: Optional[str]) -> "ScrollDirection":
        if not raw:
            return cls.DOWN
        clean = str(raw).upper().strip()
        if clean in ("DOWN", "FORWARD", "SCROLL_DOWN"):
            return cls.DOWN
        if clean in ("UP", "BACKWARD", "SCROLL_UP"):
            return cls.UP
        if clean in ("LEFT", "SCROLL_LEFT"):
            return cls.LEFT
        if clean in ("RIGHT", "SCROLL_RIGHT"):
            return cls.RIGHT
        return cls.DOWN


# =====================================================================
# 5. DUAL-SLOT ARGUMENT TYPES
# =====================================================================
class ArgumentSlotType(int, Enum):
    RECIPIENT_OR_QUERY = 0
    MESSAGE_PAYLOAD = 1
    NUMERIC_CONSTRAINT = 2
    NONE = 3
