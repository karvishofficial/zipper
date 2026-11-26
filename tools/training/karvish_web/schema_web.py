"""
KARVISH-WEB — CANONICAL WEB ACTION SCHEMA & SPECIFICATION
Dedicated Web Agent Specification.
Completely isolated from the Android V52 physical baseline.
"""

from __future__ import annotations
from enum import Enum
from typing import List, Optional

class WebActionType(str, Enum):
    CLICK = "CLICK"
    TYPE_TEXT = "TYPE_TEXT"
    SUBMIT = "SUBMIT"
    SCROLL = "SCROLL"
    SELECT = "SELECT"
    EXTRACT = "EXTRACT"
    WAIT = "WAIT"
    NAVIGATE = "NAVIGATE"

CANONICAL_WEB_ACTIONS = [
    WebActionType.CLICK,
    WebActionType.TYPE_TEXT,
    WebActionType.SUBMIT,
    WebActionType.SCROLL,
    WebActionType.SELECT,
    WebActionType.EXTRACT,
    WebActionType.WAIT,
    WebActionType.NAVIGATE,
]

WEB_ROLES = [
    "BUTTON",
    "SEARCH_FIELD",
    "TEXT_FIELD",
    "LINK",
    "SELECT",
    "CHECKBOX",
    "RADIO",
    "HEADER",
    "LIST_ITEM",
    "TAB",
    "MENU_ITEM",
    "LABEL",
    "CONTAINER",
]

WEB_REGIONS = [
    "HEADER",
    "BODY",
    "MODAL",
    "FOOTER",
]
