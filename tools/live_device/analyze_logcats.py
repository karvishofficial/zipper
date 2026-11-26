import re
import sys
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

def read_file_safe(path):
    for enc in ['utf-8', 'utf-16', 'utf-16-le', 'latin1', 'cp1252']:
        try:
            with open(path, "r", encoding=enc) as f:
                content = f.read()
                if "com.zipper" in content or "Karvish" in content or "Agent" in content or len(content) > 100:
                    return content
        except Exception:
            continue
    with open(path, "rb") as f:
        return f.read().decode('latin1', errors='ignore')

def analyze_log(file_path):
    print("=" * 80)
    print(f"ANALYZING LOG FILE: {file_path}")
    print("=" * 80)
    
    content = read_file_safe(file_path)
    lines = content.splitlines()
    print(f"Read {len(lines)} lines from {file_path}")
    
    goals = []
    current_goal = None
    goal_data = defaultdict(lambda: {"turns": [], "success": False, "reason": None, "decisions": [], "all_turns_text": []})
    
    for line in lines:
        if "run_agent_goal" in line or "GOAL:" in line or "Starting autonomous agent loop for goal:" in line or "AgentRuntimeService: runGoal:" in line:
            m = re.search(r"GOAL:\s*'([^']+)'", line)
            if not m:
                m = re.search(r"Starting autonomous agent loop for goal:\s*'([^']+)'", line)
            if not m:
                m = re.search(r"run_agent_goal \"([^\"]+)\"", line)
            if not m:
                m = re.search(r"runGoal:\s*'([^']+)'", line)
            if not m:
                m = re.search(r"goal=['\"]([^'\"]+)['\"]", line)
            if m:
                g = m.group(1).strip()
                if not current_goal or current_goal != g:
                    current_goal = g
                    if g not in goals:
                        goals.append(g)
                        
        if current_goal:
            if "Turn " in line and ("->" in line or ":" in line):
                goal_data[current_goal]["turns"].append(line.strip())
            if "PARSED ACTION:" in line:
                goal_data[current_goal]["all_turns_text"].append(line.strip())
            if "[V56 ON-DEVICE DECISION]" in line or "[V54 ON-DEVICE DECISION]" in line or "RAW NEEDLE OUTPUT" in line or "Picked:" in line:
                goal_data[current_goal]["decisions"].append(line.strip())
            if "AUTONOMOUS GOAL ACHIEVED" in line:
                goal_data[current_goal]["success"] = True
                goal_data[current_goal]["reason"] = "AUTONOMOUS_GOAL_ACHIEVED"
            elif "Goal finished: success=true" in line:
                goal_data[current_goal]["success"] = True
                m_reason = re.search(r"summary=([^,\n]+)", line)
                goal_data[current_goal]["reason"] = m_reason.group(1) if m_reason else "SUCCESS"
            elif "Goal finished: success=false" in line or "TERMINATING (FAILED)" in line or "FAILED" in line:
                if not goal_data[current_goal]["success"]:
                    m_fail = re.search(r"reason=([^,\n]+)", line)
                    goal_data[current_goal]["reason"] = m_fail.group(1) if m_fail else "FAILED"
                    
    print(f"Total Unique Goals Found: {len(goals)}\n")
    
    success_count = 0
    fail_count = 0
    
    for idx, g in enumerate(goals):
        info = goal_data[g]
        status = "PASS" if info["success"] else "FAIL/INCOMPLETE"
        if info["success"]:
            success_count += 1
        else:
            fail_count += 1
        print(f"[{idx+1:2d}] [{status:15s}] Goal: '{g}'")
        print(f"     Status: {'SUCCESS' if info['success'] else 'FAIL'}")
        if info['reason']:
            print(f"     Reason: {info['reason']}")
        print(f"     Turns Count: {len(info['turns'])}")
        
        # Print all turns
        for t in info["turns"]:
            print(f"       -> {t[:140]}")
        for p in info["all_turns_text"]:
            print(f"       [ACT] {p[:140]}")
        for d in info["decisions"]:
            print(f"       [DEC] {d[:140]}")
        print("-" * 60)
            
    print("\n" + "=" * 80)
    print(f"SUMMARY: Total={len(goals)} | PASS={success_count} | FAIL={fail_count}")
    print("=" * 80)

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "tools/live_device/device_full_logcat.txt"
    analyze_log(target)
