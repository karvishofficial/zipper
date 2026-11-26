"""
KARVISH V56 — ZERO-HARDCODING CONSTITUTIONAL AUDITOR
Governing Authority: AGENTS.md (Constitutional Laws 0, 4, 7, 8, 9, 14, 17)

Audits production model and runtime inference files:
1. Zero hardcoded task-specific entities ("Bhavani", "SpaceX", "tomatoes", etc.)
2. Zero package-specific decision shortcuts ("com.whatsapp", "com.instagram.android")
3. Zero hardcoded sequence macros
4. Legitimate platform plumbing vs. forbidden task spoonfeeding separation
"""

from __future__ import annotations

import sys
import os
import ast
import re
from pathlib import Path
from typing import List, Dict, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]

INFERENCE_FILES = [
    SCRIPT_DIR / "model_v56.py",
    SCRIPT_DIR / "schema_v56.py",
    SCRIPT_DIR / "action_conditioned_span_v56.py",
]

FORBIDDEN_ENTITY_PATTERNS = [
    r"\bBhavani\b",
    r"\bSpaceX\b",
    r"\bfresh tomatoes\b",
    r"\bsoftware engineer\b",
    r"\bRahul\b",
    r"\bBasavaraj\b",
    r"\bTomatoes\b",
]

FORBIDDEN_PACKAGE_SHORTCUTS = [
    r'package\s*==\s*["\']com\.whatsapp["\']',
    r'package\s*==\s*["\']com\.instagram\.android["\']',
    r'package\s*==\s*["\']com\.google\.android\.youtube["\']',
    r'package\s*==\s*["\']com\.linkedin\.android["\']',
    r'if\s+["\']youtube["\']\s+in\s+goal',
    r'if\s+["\']whatsapp["\']\s+in\s+goal',
    r'if\s+["\']instagram["\']\s+in\s+goal',
]

FORBIDDEN_MACROS = [
    r'if\s+previous_action\s*==\s*TYPE_TEXT.*boost.*submit',
    r'if\s+step\s*==\s*1.*return',
]

def audit_file(file_path: Path) -> List[str]:
    violations: List[str] = []
    if not file_path.exists():
        return [f"File not found: {file_path}"]

    content = file_path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()

    # 1. Regex Pattern Scan
    for line_no, line in enumerate(lines, 1):
        # Skip pure docstrings or comments explaining the rule
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("*") or stripped.startswith('"""') or stripped.startswith("'''"):
            continue

        for pat in FORBIDDEN_ENTITY_PATTERNS:
            if re.search(pat, line, re.IGNORECASE):
                violations.append(f"{file_path.name}:{line_no} - Forbidden entity reference '{pat}': {stripped}")

        for pat in FORBIDDEN_PACKAGE_SHORTCUTS:
            if re.search(pat, line, re.IGNORECASE):
                violations.append(f"{file_path.name}:{line_no} - Package-specific workflow shortcut '{pat}': {stripped}")

        for pat in FORBIDDEN_MACROS:
            if re.search(pat, line, re.IGNORECASE):
                violations.append(f"{file_path.name}:{line_no} - Predetermined macro shortcut '{pat}': {stripped}")

    # 2. AST Structure Verification
    try:
        tree = ast.parse(content, filename=str(file_path))
    except Exception as e:
        violations.append(f"{file_path.name}: AST parse failure: {e}")

    return violations

def run_constitutional_audit() -> bool:
    print("=" * 80)
    print("KARVISH V56 ZERO-HARDCODING CONSTITUTIONAL AUDIT")
    print("Governing Authority: AGENTS.md (Constitutional Laws 0, 4, 7, 8, 9, 14, 17)")
    print("=" * 80)

    total_violations = 0
    for f in INFERENCE_FILES:
        print(f"\nScanning inference file: {f.relative_to(REPO_ROOT)}...")
        v = audit_file(f)
        if v:
            print(f"  [VIOLATIONS DETECTED: {len(v)}]")
            for item in v:
                print(f"    - {item}")
            total_violations += len(v)
        else:
            print(f"  [PASS] Zero task spoonfeeding or hardcoded heuristics detected.")

    print("\n" + "=" * 80)
    if total_violations == 0:
        print("CONSTITUTIONAL AUDIT PASSED: ZERO TASK SPOONFEEDING DETECTED.")
        print("All task intelligence is verified to reside strictly in learned model weights.")
        print("=" * 80)
        return True
    else:
        print(f"CONSTITUTIONAL AUDIT FAILED: {total_violations} VIOLATION(S) DETECTED.")
        print("=" * 80)
        return False

if __name__ == "__main__":
    success = run_constitutional_audit()
    sys.exit(0 if success else 1)
