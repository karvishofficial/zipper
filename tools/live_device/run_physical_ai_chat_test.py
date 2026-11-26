"""
PHYSICAL DEVICE AI CHAT TEST RUNNER (SAMSUNG GALAXY A26)
Executes physical device tests for AI chat flow:
- Goal: "Ask ChatGPT what is algae"
- Uses Live Host GPU Server (localhost:8765) with v54_chat_smoke.pt
- Monitors logcat, captures screenshots per turn, verifies complete physical execution.
"""

import subprocess
import time
import os
import sys
import json
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")

REPO_ROOT = Path(".").resolve()
ADB_EXE = os.path.expandvars(r"%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe")
DEVICE_ID = "RZCY51MQP9K"
OUTPUT_DIR = REPO_ROOT / "tools" / "live_device" / "physical_test_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_adb(args):
    cmd = [ADB_EXE, "-s", DEVICE_ID] + args
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def capture_screenshot(step_name: str) -> Path:
    out_file = OUTPUT_DIR / f"{step_name}.png"
    img_data = subprocess.check_output([ADB_EXE, "-s", DEVICE_ID, "exec-out", "screencap", "-p"])
    with open(out_file, "wb") as f:
        f.write(img_data)
    print(f">> [SCREENSHOT] Saved: {out_file.name} ({len(img_data):,} bytes)")
    return out_file


def run_physical_test(goal: str = "Ask ChatGPT what is algae"):
    print("=" * 80)
    print(f"STARTING PHYSICAL TEST ON SAMSUNG GALAXY A26: '{goal}'")
    print("=" * 80)

    # 1. Health check
    try:
        with urllib.request.urlopen("http://127.0.0.1:8765/health", timeout=3) as resp:
            data = json.loads(resp.read().decode())
            print(f">> [GPU SERVER] OK: {data}")
    except Exception as e:
        print(f"[FATAL ERROR] Cannot reach GPU server on port 8765: {e}")
        return

    # 2. Reverse port forward
    run_adb(["reverse", "tcp:8765", "tcp:8765"])
    print(">> [ADB REVERSE] Configured tcp:8765 -> tcp:8765")

    # 3. Clean device state
    print(">> [PRE-FLIGHT] Resetting device state to Home Screen...")
    run_adb(["shell", "input keyevent 224"])  # Wake
    run_adb(["shell", "input keyevent 82"])   # Unlock
    run_adb(["shell", "am force-stop com.openai.chatgpt"])
    run_adb(["shell", "am force-stop com.google.android.apps.bard"])
    run_adb(["shell", "am force-stop com.zipper.app"])
    run_adb(["shell", "input keyevent 3"])   # Home
    run_adb(["shell", "input keyevent 3"])
    time.sleep(1)

    capture_screenshot("00_initial_home")

    # 4. Check Accessibility
    a11y_dump = run_adb(["shell", "dumpsys accessibility"]).stdout
    if "KarvishAccessibilityService" not in a11y_dump:
        print(">> Enabling KarvishAccessibilityService...")
        run_adb(["shell", "settings put secure enabled_accessibility_services com.zipper.app/com.zipper.app.accessibility.KarvishAccessibilityService"])
        run_adb(["shell", "settings put secure accessibility_enabled 1"])
        time.sleep(1)

    # 5. Timestamp for logcat monitoring
    t_res = run_adb(["shell", "date '+%m-%d %H:%M:%S.000'"])
    start_time = t_res.stdout.strip()
    print(f">> Logcat start time: {start_time}")

    # 6. Launch agent with goal
    print(f">> Launching Zipper Agent with goal: '{goal}'...")
    launch_str = f'am start -n com.zipper.app/.MainActivity --es run_agent_goal "{goal}" --ez use_remote_ranker true --ez enable_active_recovery false'
    res = run_adb(["shell", launch_str])
    print(f">> Launch output: {res.stdout.strip()}")

    # 7. Monitor execution loop
    print("\n>> Monitoring live physical execution on device...")
    finished = False
    turn_count = 0
    for tick in range(1, 30):
        time.sleep(2)
        elapsed = tick * 2
        
        # Grab focused window
        focus_res = run_adb(["shell", "dumpsys window | grep mCurrentFocus"]).stdout.strip()
        
        # Grab agent logcat
        log_res = run_adb([
            "logcat", "-d", "-v", "time", "-t", start_time,
            "AgentRuntimeService:V", "KarvishAgentRuntime:V", "LiveRankerModelEngine:V", "ActionDispatcher:V", "*:S"
        ])
        logs = log_res.stdout or ""

        # Print new significant logs
        for line in logs.splitlines():
            if any(k in line for k in ["Dispatched action:", "Action failed", "Model proposed:", "Goal finished", "AUTONOMOUS", "openAppByName"]):
                if line not in seen_lines:
                    seen_lines.add(line)
                    print(f"   [{elapsed}s] {line.strip()}")

        if "Goal finished:" in logs or "AUTONOMOUS GOAL ACHIEVED" in logs or "Early termination" in logs:
            print(f"\n>> [TERMINATION DETECTED] Goal completed in ~{elapsed}s!")
            finished = True
            break

        if tick % 3 == 0:
            capture_screenshot(f"turn_{tick//3:02d}_{elapsed}s")

    capture_screenshot("final_device_state")

    # Save full trace
    full_logs = run_adb([
        "logcat", "-d", "-v", "time", "-t", start_time,
        "AgentRuntimeService:V", "KarvishAgentRuntime:V", "KarvishA11yService:V", "LiveRankerModelEngine:V", "ActionDispatcher:V", "*:S"
    ]).stdout
    trace_file = OUTPUT_DIR / "device_execution_trace.log"
    with open(trace_file, "w", encoding="utf-8") as f:
        f.write(full_logs)
    print(f"\n>> Full execution trace saved to: {trace_file}")


seen_lines = set()

if __name__ == "__main__":
    goal_arg = sys.argv[1] if len(sys.argv) > 1 else "Ask ChatGPT what is algae"
    run_physical_test(goal_arg)
