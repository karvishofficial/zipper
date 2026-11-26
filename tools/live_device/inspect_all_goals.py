import sys
import re
import subprocess
import os

sys.stdout.reconfigure(encoding='utf-8')

adb = os.path.expandvars(r'%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe')
raw = subprocess.check_output([adb, 'logcat', '-d'], text=True, encoding='utf-8', errors='ignore')

goals = []
decisions = []
achieved = []

for line in raw.splitlines():
    if "GOAL:" in line:
        m = re.search(r"GOAL:\s*'([^']+)'", line)
        if m:
            g = m.group(1).strip()
            if not goals or goals[-1] != g:
                goals.append(g)
    if "AUTONOMOUS GOAL ACHIEVED" in line:
        m = re.search(r"AUTONOMOUS GOAL ACHIEVED:\s*'([^']+)'", line)
        if m:
            achieved.append(m.group(1).strip())
    if "[V56 ON-DEVICE DECISION]" in line:
        decisions.append(line.strip())

print(f"Total Unique Goals Run: {len(goals)}")
for idx, g in enumerate(goals):
    is_done = g in achieved
    print(f"  {idx+1:2d}. [{'SUCCESS' if is_done else 'IN-PROGRESS/PARTIAL'}] {g}")

print(f"\nTotal On-Device Decisions: {len(decisions)}")
print("\n--- RECENT 10 DECISIONS ---")
for d in decisions[-10:]:
    m_picked = re.search(r"Picked:\s*'([^']+)'", d)
    m_act = re.search(r"Action:\s*([^|]+)", d)
    m_conf = re.search(r"conf:\s*([0-9\.]+)\%", d)
    m_score = re.search(r"score:\s*([0-9\.\-]+)", d)
    picked = m_picked.group(1) if m_picked else '?'
    act = m_act.group(1).strip() if m_act else '?'
    conf = m_conf.group(1) if m_conf else '?'
    score = m_score.group(1) if m_score else '?'
    print(f"  Picked: '{picked}' | Score: {score} | Conf: {conf}% | Action: {act}")
