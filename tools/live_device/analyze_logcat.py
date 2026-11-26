import re
import os
from collections import defaultdict

log_path = "tools/live_device/device_full_logcat.txt"
if not os.path.exists(log_path):
    print("Log file not found")
    exit(1)

try:
    with open(log_path, "r", encoding="utf-16", errors="ignore") as f:
        lines = f.readlines()
    if len(lines) < 10:
        raise ValueError("Too few lines")
except Exception:
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

print(f"Total logcat lines: {len(lines)}")

# Look for agent sessions / goals
goal_events = []
decisions_by_turn = []
stalls = []
all_goals = []

current_goal = None

for line in lines:
    # Check for goal started
    # e.g., AgentRuntimeService: ... Starting goal ... or run_agent_goal
    if "Starting goal" in line or "startGoal" in line or "run_agent_goal" in line or "Execute goal:" in line or "goal='" in line:
        m = re.search(r"goal=['\"]([^'\"]+)['\"]", line)
        if not m:
            m = re.search(r"run_agent_goal\s+([^,;\n]+)", line)
        if m:
            g = m.group(1).strip()
            if not all_goals or all_goals[-1] != g:
                all_goals.append(g)
            current_goal = g

    if "[V56 ON-DEVICE DECISION]" in line or "ON-DEVICE DECISION" in line:
        decisions_by_turn.append((current_goal, line.strip()))

    if "Multimodal Delta Verifier" in line or "stall_prob" in line:
        stalls.append((current_goal, line.strip()))

print(f"Total Unique Goals detected: {len(set(all_goals))}")
print(f"Total Goal Runs: {len(all_goals)}")
print(f"Total Decisions recorded: {len(decisions_by_turn)}")
print(f"Total Stall / Recovery Detections: {len(stalls)}")

print("\n--- ALL DETECTED GOALS ---")
for i, g in enumerate(dict.fromkeys(all_goals)):
    print(f"{i+1:2d}. {g}")

print("\n--- RECENT DECISION EXAMPLES ---")
for g, d in decisions_by_turn[-15:]:
    # extract picked and action
    m_picked = re.search(r"Picked:\s*'([^']+)'", d)
    m_act = re.search(r"Action:\s*([^|]+)", d)
    m_lat = re.search(r"Latency:\s*([^|]+)", d)
    picked = m_picked.group(1) if m_picked else "Unknown"
    act = m_act.group(1).strip() if m_act else "Unknown"
    lat = m_lat.group(1).strip() if m_lat else ""
    print(f"Goal: '{g}' -> Picked: '{picked}' | Action: {act} | {lat}")

print("\n--- STALL DETECTIONS ---")
for g, s in stalls[-10:]:
    print(f"Goal: '{g}' -> {s}")
