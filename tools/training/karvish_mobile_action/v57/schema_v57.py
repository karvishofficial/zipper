"""
KARVISH UNIVERSAL V57 — CANONICAL SCHEMA & DATA CONTRACT
Authoritative Data Specification for Karvish Universal Agent V57.
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
V56_DIR = SCRIPT_DIR.parent / "v56"
sys.path.insert(0, str(V56_DIR))

from schema_v56 import (
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
