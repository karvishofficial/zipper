import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

def read_file(p):
    for enc in ['utf-16', 'utf-16le', 'utf-8', 'latin1']:
        try:
            with open(p, 'r', encoding=enc) as f:
                c = f.read()
                if "10-03" in c or "zipper" in c:
                    return c
        except Exception:
            continue
    return ""

folder = Path("tools/live_device")
for file in folder.glob("*.txt"):
    c = read_file(file)
    print(f"\n=== FILE: {file.name} (length {len(c)}) ===")
    goals_found = set()
    for line in c.splitlines():
        if "GOAL:" in line or "run_agent_goal" in line or "Starting autonomous" in line:
            m = re.search(r"GOAL:\s*'([^']+)'", line)
            if not m:
                m = re.search(r"run_agent_goal \"([^\"]+)\"", line)
            if m:
                goals_found.add(m.group(1).strip())
    print(f"Goals found in {file.name}: {len(goals_found)}")
    for g in goals_found:
        print(f"  * '{g}'")
