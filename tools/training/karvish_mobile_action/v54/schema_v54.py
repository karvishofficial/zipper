"""
KARVISH V54 — CANONICAL SCHEMA & DATA CONTRACT
Authoritative Data Specification for V54 Architecture.

Governing Standard: Karvish Agent Constitution (AGENTS.md)
Target Model: Karvish V54 (Compact Mobile Neural Policy with Multiplicative Entity Gating & Numerical Grounding)

Key Innovations:
1. 8 Canonical Actions mapped directly from AndroidControl:
   CLICK, LONG_PRESS, SCROLL, OPEN_APP, TYPE_TEXT, HOME, BACK, WAIT.
2. Continuous 2D Spatial Geometry: [left, top, right, bottom, cx, cy, w, h, area, aspect].
3. Hierarchical Container Modeling: parent_id, children_ids, is_container, card_binding.
4. Numerical Attribute Features: [selling_price, mrp, discount_pct, unit_rate].
5. Entity Grounding Targets: entity_text, entity_span, entity_role.
6. Dynamic Goal Coverage metadata & Terminal Milestone State flags.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


# =====================================================================
# 1. CANONICAL ACTION VOCABULARY (8 Actions from AndroidControl)
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
# 2. SCROLL DIRECTIONS
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
# 3. CONTINUOUS 2D SPATIAL BOUNDS
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
        """Returns 10-dim continuous geometry vector normalized to [0, 1]."""
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
        # Normalize if in absolute pixel coordinates
        if r > 1.0 or b > 1.0:
            l = l / screen_width
            r = r / screen_width
            t = t / screen_height
            b = b / screen_height
        return cls(left=l, top=t, right=r, bottom=b)


# =====================================================================
# 4. HIERARCHICAL CARD CONTAINER METADATA
# =====================================================================
@dataclass
class ContainerMetadata:
    is_container: bool = False
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    card_title: Optional[str] = None
    card_selling_price: Optional[float] = None
    card_mrp: Optional[float] = None
    card_discount_pct: Optional[float] = None
    card_weight_grams: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =====================================================================
# 5. CANDIDATE NODE SCHEMA
# =====================================================================
@dataclass
class CandidateNode:
    candidate_id: str
    label: str
    bounds: SpatialBounds
    role: str = "unknown"
    role_idx: int = 0
    is_clickable: bool = True
    is_editable: bool = False
    is_scrollable: bool = False
    is_focusable: bool = False
    is_selected: bool = False
    is_checked: bool = False
    is_visible: bool = True
    container: ContainerMetadata = field(default_factory=ContainerMetadata)
    num_features: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["bounds"] = self.bounds.to_vector()
        return d


# =====================================================================
# 6. UNIVERSAL TASK TRAJECTORY STEP
# =====================================================================
@dataclass
class TrajectoryStep:
    step_id: str
    goal: str
    candidates: List[CandidateNode]
    target_candidate_id: str
    action_type: CanonicalActionType
    argument_text: Optional[str] = None
    scroll_direction: ScrollDirection = ScrollDirection.NONE
    observable_delta: List[float] = field(default_factory=lambda: [0.0] * 10)
    task_progress: float = 0.0
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
            "observable_delta": self.observable_delta,
            "task_progress": self.task_progress,
            "is_terminal": self.is_terminal,
        }
