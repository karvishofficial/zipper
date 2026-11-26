"""
KARVISH V52 — CANONICAL SCHEMA & DATA CONTRACT
Authoritative Data Specification for Phase 1 to Phase 10.

Governing Standard: Karvish Agent Constitution (AGENTS.md)
Target Model: Karvish V52 (28.5M INT8 Compact Mobile Policy)

Key Innovations:
1. 8 Canonical Actions mapped directly from AndroidControl:
   CLICK, LONG_PRESS, SCROLL, OPEN_APP, TYPE_TEXT, HOME, BACK, WAIT.
2. Continuous 2D Spatial Geometry: [left, top, right, bottom, cx, cy, w, h, area, aspect].
3. Hierarchical Container Modeling: parent_id, children_ids, is_container.
4. Dynamic Goal Coverage metadata (token-level sequence & consumption flags).
5. Phase D Learned Span Pointer targets (start_token, end_token, argument_text).
6. Provenance & Teacher Consensus auditing fields.
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
        # Supports [l, t, r, b] or absolute pixel coordinates
        l = float(data.get("left", data.get("x1", 0.0)))
        t = float(data.get("top", data.get("y1", 0.0)))
        r = float(data.get("right", data.get("x2", 1.0)))
        b = float(data.get("bottom", data.get("y2", 1.0)))

        # Normalize pixel coordinates if > 1.0
        if r > 1.0 or b > 1.0 or l > 1.0 or t > 1.0:
            l = l / screen_width
            r = r / screen_width
            t = t / screen_height
            b = b / screen_height

        return cls(
            left=max(0.0, min(1.0, l)),
            top=max(0.0, min(1.0, t)),
            right=max(0.0, min(1.0, r)),
            bottom=max(0.0, min(1.0, b)),
        )


# =====================================================================
# 4. HIERARCHICAL CANDIDATE NODE
# =====================================================================
@dataclass
class CandidateNode:
    id: str
    role: str
    label: str
    bounds: SpatialBounds
    is_clickable: bool = True
    is_editable: bool = False
    is_scrollable: bool = False
    is_focused: bool = False
    is_enabled: bool = True
    is_checked: bool = False
    is_container: bool = False
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    child_text: str = ""
    semantic_tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "label": self.label,
            "bounds": asdict(self.bounds),
            "bounds_vector": self.bounds.to_vector(),
            "is_clickable": self.is_clickable,
            "is_editable": self.is_editable,
            "is_scrollable": self.is_scrollable,
            "is_focused": self.is_focused,
            "is_enabled": self.is_enabled,
            "is_checked": self.is_checked,
            "is_container": self.is_container,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "child_text": self.child_text,
            "semantic_tags": self.semantic_tags,
        }


# =====================================================================
# 5. PHASE D LEARNED SPAN TARGET
# =====================================================================
@dataclass
class ArgumentSpanTarget:
    start_token: int = -1
    end_token: int = -1
    argument_text: str = ""
    direction: ScrollDirection = ScrollDirection.DOWN
    target_package: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_token": self.start_token,
            "end_token": self.end_token,
            "argument_text": self.argument_text,
            "direction": self.direction.value,
            "target_package": self.target_package,
        }


# =====================================================================
# 6. CONSENSUS & PROVENANCE METADATA
# =====================================================================
@dataclass
class ConsensusMetadata:
    source_dataset: str                         # "android_control" or "rico"
    trajectory_id: str
    step_index: int
    infigui_proposal: Optional[Dict[str, Any]] = None
    uitars_proposal: Optional[Dict[str, Any]] = None
    consensus_agreed: bool = True
    is_hard_negative: bool = False
    hard_negative_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_dataset": self.source_dataset,
            "trajectory_id": self.trajectory_id,
            "step_index": self.step_index,
            "infigui_proposal": self.infigui_proposal,
            "uitars_proposal": self.uitars_proposal,
            "consensus_agreed": self.consensus_agreed,
            "is_hard_negative": self.is_hard_negative,
            "hard_negative_ids": self.hard_negative_ids,
        }


# =====================================================================
# 7. THE MASTER TRANSITION SAMPLE (V52 Canonical Transition)
# =====================================================================
@dataclass
class V52TransitionSample:
    sample_id: str
    goal: str
    current_subgoal: str
    history: List[str]
    candidates: List[CandidateNode]
    target_candidate_id: Optional[str]
    action_type: CanonicalActionType
    argument_span: ArgumentSpanTarget
    consensus: ConsensusMetadata
    step_index: int = 0
    total_steps: int = 1
    screen_id: str = ""
    app_package: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "goal": self.goal,
            "current_subgoal": self.current_subgoal,
            "history": self.history,
            "candidates": [c.to_dict() for c in self.candidates],
            "target_candidate_id": self.target_candidate_id,
            "action_type": self.action_type.value,
            "argument_span": self.argument_span.to_dict(),
            "consensus": self.consensus.to_dict(),
            "step_index": self.step_index,
            "total_steps": self.total_steps,
            "screen_id": self.screen_id,
            "app_package": self.app_package,
        }

    def compute_sha256(self) -> str:
        s = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()
