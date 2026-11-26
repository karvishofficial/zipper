import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

log_path = 'tools/live_device/recent_logcat_1616.txt'
try:
    with open(log_path, 'r', encoding='utf-16', errors='ignore') as f:
        lines = f.readlines()
    if len(lines) < 10:
        raise ValueError()
except Exception:
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

print(f'Total recent lines: {len(lines)}')

decisions = []
audit_traces = []
goals = []
errors = []

for l in lines:
    l_str = l.strip()
    if '[V56 ON-DEVICE DECISION]' in l_str or 'ON-DEVICE DECISION' in l_str:
        decisions.append(l_str)
    if 'GOAL:' in l_str or 'run_agent_goal' in l_str:
        goals.append(l_str)
    if 'AUDIT TRACE' in l_str or 'PARSED ACTION' in l_str:
        audit_traces.append(l_str)
    if any(k in l_str for k in ['FATAL', 'CRASH', 'Exception', 'AndroidRuntime: E', 'NullPointerException']) and 'com.zipper.app' in l_str:
        errors.append(l_str)

print(f'Decisions count: {len(decisions)}')
print(f'Goals count: {len(goals)}')
print(f'Audit traces: {len(audit_traces)}')
print(f'Errors found: {len(errors)}')

print('\n--- ALL RECENT GOALS ---')
for g in set(goals):
    print('  ', g)

print('\n--- ALL RECENT DECISIONS (LAST 25) ---')
for d in decisions[-25:]:
    m_picked = re.search(r"Picked:\s*'([^']+)'", d)
    m_act = re.search(r"Action:\s*([^|]+)", d)
    m_conf = re.search(r"conf:\s*([0-9\.]+)\%", d)
    m_score = re.search(r"score:\s*([0-9\.\-]+)", d)
    picked = m_picked.group(1) if m_picked else '?'
    act = m_act.group(1).strip() if m_act else '?'
    conf = m_conf.group(1) if m_conf else '?'
    score = m_score.group(1) if m_score else '?'
    print(f"Picked: '{picked}' | Score: {score} | Conf: {conf}% | Action: {act}")

if errors:
    print('\n--- ERRORS ---')
    for e in errors[:10]:
        print('  ', e)
