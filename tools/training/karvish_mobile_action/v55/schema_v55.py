"""
KARVISH UNIVERSAL V55 — CANONICAL SCHEMA & DATA CONTRACT
Authoritative Data Specification for Karvish Universal Agent V55.

Governing Standard: Karvish Agent Constitution (AGENTS.md)
Target Architecture: Karvish V55 Universal Action Policy
Design Pillars:
  1. Universal Semantic UI Roles (32 canonical roles, zero app-specific heuristics).
  2. Contextual Role Compatibility: f(Role, Subgoal, Context, Bounds).
  3. Action-Conditioned Span Extraction: separates entity noun from numerical constraints.
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
# 5. CONTINUOUS 2D SPATIAL BOUNDS (10 Continuous Dimensions)
# =====================================================================
@dataclass
class SpatialBounds:
    left: float
    top: float
    right: float
    bottom: float

    @property
    def cx(self) -> float:
        return (self.left + self.right) / 2.0

    @property
    def cy(self) -> float:
        return (self.top + self.bottom) / 2.0

    @property
    def width(self) -> float:
        return max(0.0, self.right - self.left)

    @property
    def height(self) -> float:
        return max(0.0, self.bottom - self.top)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        h = max(self.height, 0.001)
        return min((self.width / h) / 5.0, 2.0)

    def to_vector(self) -> List[float]:
        return [
            min(max(self.left, 0.0), 1.0),
            min(max(self.top, 0.0), 1.0),
            min(max(self.right, 0.0), 1.0),
            min(max(self.bottom, 0.0), 1.0),
            min(max(self.cx, 0.0), 1.0),
            min(max(self.cy, 0.0), 1.0),
            min(max(self.width, 0.0), 1.0),
            min(max(self.height, 0.0), 1.0),
            min(max(self.area, 0.0), 1.0),
            self.aspect_ratio,
        ]

    @classmethod
    def from_dict(cls, data: Dict[str, Any], screen_width: float = 1080.0, screen_height: float = 2400.0) -> "SpatialBounds":
        l = float(data.get("left", data.get("x1", 0.0)))
        t = float(data.get("top", data.get("y1", 0.0)))
        r = float(data.get("right", data.get("x2", 1.0)))
        b = float(data.get("bottom", data.get("y2", 1.0)))
        if r > 1.0 or b > 1.0:
            l = l / screen_width
            r = r / screen_width
            t = t / screen_height
            b = b / screen_height
        return cls(left=l, top=t, right=r, bottom=b)


# =====================================================================
# 6. MULTIMODAL OBSERVABLE DELTA STATE (10 Dimensions)
# =====================================================================
@dataclass
class MultimodalDeltaState:
    delta_tree_nodes: float = 0.0       # Normalized difference in candidate node count
    delta_text_entropy: float = 0.0     # Change in unique token diversity
    delta_viewport_scroll: float = 0.0  # Measured scroll pixel displacement
    is_keyboard_visible: float = 0.0    # 1.0 if soft keyboard open, else 0.0
    is_dialog_open: float = 0.0         # 1.0 if modal / permission dialog active
    is_loading_spinner: float = 0.0     # 1.0 if loading spinner / progressbar active
    is_package_changed: float = 0.0     # 1.0 if foreground app package changed
    is_toast_or_snackbar: float = 0.0   # 1.0 if notification toast / snackbar present
    consecutive_noop_count: float = 0.0 # Normalized count of 0-delta actions (0 to 1)
    action_success_signal: float = 0.0  # -1.0 (failed), 0.0 (neutral), +1.0 (progress)

    def to_vector(self) -> List[float]:
        return [
            min(max(self.delta_tree_nodes, -1.0), 1.0),
            min(max(self.delta_text_entropy, -1.0), 1.0),
            min(max(self.delta_viewport_scroll, -1.0), 1.0),
            1.0 if self.is_keyboard_visible > 0.5 else 0.0,
            1.0 if self.is_dialog_open > 0.5 else 0.0,
            1.0 if self.is_loading_spinner > 0.5 else 0.0,
            1.0 if self.is_package_changed > 0.5 else 0.0,
            1.0 if self.is_toast_or_snackbar > 0.5 else 0.0,
            min(max(self.consecutive_noop_count, 0.0), 1.0),
            min(max(self.action_success_signal, -1.0), 1.0),
        ]


# =====================================================================
# 7. STRUCTURED GOAL DECOMPOSITION
# =====================================================================
@dataclass
class UserGoalStructured:
    raw_goal: str
    primary_intent: str = "SEARCH"      # SEARCH, MESSAGE, ECOMMERCE, NAVIGATION, SETTING
    entity_noun: str = ""               # "tomato", "Bhavani", "SpaceX"
    constraint_type: str = "NONE"       # "CHEAPEST", "HIGHEST_RATED", "NONE"
    argument_payload: str = ""          # Text to be typed
    subgoals: List[str] = field(default_factory=list)

    @classmethod
    def decompose(cls, goal: str) -> "UserGoalStructured":
        """
        Structured decomposition helper for training and evaluation.
        Decouples entity noun from numerical constraints ('cheapest tomato' -> noun='tomato', constraint='CHEAPEST').
        """
        clean = goal.strip().lower()
        constraint = "NONE"
        if "cheapest" in clean or "lowest price" in clean or "low price" in clean:
            constraint = "CHEAPEST"
        elif "highest rated" in clean or "best rated" in clean or "top rated" in clean:
            constraint = "HIGHEST_RATED"

        intent = "SEARCH"
        if any(w in clean for w in ("send", "message", "text", "whatsapp", "chat")):
            intent = "MESSAGE"
        elif any(w in clean for w in ("add to cart", "buy", "cart", "order")):
            intent = "ECOMMERCE"
        elif any(w in clean for w in ("settings", "turn on", "turn off", "toggle", "bluetooth", "wifi")):
            intent = "SETTING"
        elif any(w in clean for w in ("profile", "view profile", "header")):
            intent = "NAVIGATION"

        # Separate entity noun
        entity = ""
        # Common entity candidates
        for token in clean.replace("open", "").replace("search", "").replace("and add to cart", "").replace("on zepto", "").replace("cheapest", "").split():
            if len(token) > 2 and token not in ("the", "for", "and", "in", "to", "app"):
                entity = token
                break

        return cls(
            raw_goal=goal,
            primary_intent=intent,
            entity_noun=entity,
            constraint_type=constraint,
            argument_payload=entity,
            subgoals=[],
        )


# =====================================================================
# 8. CANDIDATE NODE SCHEMA
# =====================================================================
@dataclass
class CandidateNodeV55:
    candidate_id: str
    label: str
    bounds: SpatialBounds
    role: str = "UNKNOWN"
    role_idx: int = 0
    is_clickable: bool = True
    is_editable: bool = False
    is_scrollable: bool = False
    is_focusable: bool = False
    is_selected: bool = False
    is_checked: bool = False
    is_visible: bool = True
    num_features: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["bounds"] = self.bounds.to_vector()
        return d


# =====================================================================
# 9. UNIVERSAL TASK TRAJECTORY STEP
# =====================================================================
@dataclass
class TrajectoryStepV55:
    step_id: str
    goal: str
    structured_goal: UserGoalStructured
    candidates: List[CandidateNodeV55]
    target_candidate_id: str
    action_type: CanonicalActionType
    argument_text: Optional[str] = None
    scroll_direction: ScrollDirection = ScrollDirection.NONE
    observable_delta: MultimodalDeltaState = field(default_factory=MultimodalDeltaState)
    current_subgoal: str = "LOCATE_INPUT"
    task_progress: float = 0.0
    completion_evidence_observed: bool = False
    is_terminal: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "goal": self.goal,
            "candidates": [c.to_dict() for c in self.candidates],
            "target_candidate_id": self.target_candidate_id,
            "action_type": self.action_type.value,
            "argument_text": self.argument_text,
            "scroll_direction": self.scroll_direction.value,
            "observable_delta": self.observable_delta.to_vector(),
            "current_subgoal": self.current_subgoal,
            "task_progress": self.task_progress,
            "completion_evidence_observed": self.completion_evidence_observed,
            "is_terminal": self.is_terminal,
        }
