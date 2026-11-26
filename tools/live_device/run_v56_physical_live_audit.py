"""
KARVISH V56 — PHYSICAL CLOSED-LOOP LIVE DEVICE AUDIT
Step 7 of Universal Agent Roadmap
Runs real-time autonomous test suites on physical Samsung Galaxy A26 (RZCY51MQP9K).
Executes natively on-device with the deployed Karvish V56 ONNX model (models/karvish_v56.onnx).
"""

from __future__ import annotations

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
ADB_PATH = os.path.expandvars(r"%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe")
DEVICE_SERIAL = "RZCY51MQP9K"

OUTPUT_DIR = REPO_ROOT / "tools" / "live_device" / "v56_live_audit_reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FILE = OUTPUT_DIR / f"v56_physical_audit_{int(time.time())}.json"


def run_adb(args: List[str], timeout: int = 15) -> str:
    cmd = [ADB_PATH, "-s", DEVICE_SERIAL] + args
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
        return res.stdout.strip()
    except Exception as e:
        return f"ADB_ERROR: {e}"


def run_v56_live_goal(
    goal: str,
    target_app_pkg: Optional[str] = None,
    timeout: int = 35,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"RUNNING V56 PHYSICAL ON-DEVICE TASK: '{goal}'")
    print(f"Device: {DEVICE_SERIAL} (Samsung Galaxy A26)")
    print(f"Execution: Pure On-Device ONNX (models/karvish_v56.onnx, 51.01 MB)")
    print("=" * 80, flush=True)

    # 1. Wake & unlock
    run_adb(["shell", "input", "keyevent", "224"])
    run_adb(["shell", "input", "keyevent", "82"])
    run_adb(["shell", "input", "keyevent", "3"])

    # 2. Reset app states
    if target_app_pkg:
        run_adb(["shell", "am", "force-stop", target_app_pkg])
        if "calc" in target_app_pkg:
            run_adb(["shell", "pm", "clear", target_app_pkg])

    # 3. Clear logcat
    run_adb(["shell", "logcat", "-c"])

    # 4. Dispatch Intent to Karvish
    t0 = time.time()
    intent_cmd = (
        f"am start -n com.zipper.app/.MainActivity "
        f"--es run_agent_goal \"{goal}\" "
        f"--ez enable_active_recovery true "
        f"--ez use_remote_ranker false"
    )
    launch_res = run_adb(["shell", intent_cmd])
    print(f">> Dispatched goal to device: {launch_res}", flush=True)

    # 5. Monitor Live Turns
    turns: List[Dict[str, Any]] = []
    completed = False
    start_time = time.time()

    while time.time() - start_time < timeout:
        time.sleep(2.0)
        elapsed = round(time.time() - start_time, 1)

        raw_logs = run_adb(["shell", "logcat", "-d", "-s", "OnDeviceRankerEngine:*", "AgentRuntimeService:*", "KarvishAgentRuntime:*", "KarvishAgentService:*"])
        lines = raw_logs.splitlines()

        for line in lines:
            if "[V56 ON-DEVICE DECISION]" in line:
                if line not in [t.get("raw") for t in turns]:
                    turns.append({"raw": line, "time": elapsed})
                    print(f"  [T+{elapsed}s] {line.split('[V56 ON-DEVICE DECISION]')[-1].strip()}")

            if "AUTONOMOUS GOAL ACHIEVED" in line or "GOAL_COMPLETED" in line:
                completed = True

        if completed:
            print(f">> [SUCCESS] Physical Goal Achieved in {elapsed}s!")
            break

    total_time = round(time.time() - start_time, 1)

    # 6. Screen Capture
    screenshot_name = f"v56_{int(time.time())}_{goal[:12].replace(' ', '_')}.png"
    screen_path = OUTPUT_DIR / screenshot_name
    try:
        img_bytes = subprocess.check_output([ADB_PATH, "-s", DEVICE_SERIAL, "exec-out", "screencap", "-p"], timeout=10)
        with open(screen_path, "wb") as f:
            f.write(img_bytes)
        print(f">> Verified screenshot saved: {screenshot_name}")
    except Exception as e:
        print(f">> Screenshot capture warning: {e}")

    result = {
        "goal": goal,
        "device": DEVICE_SERIAL,
        "completed": completed,
        "turns_count": len(turns),
        "turns": turns,
        "duration_sec": total_time,
        "screenshot": str(screen_path),
    }
    return result


def run_full_v56_physical_audit():
    test_cases = [
        {
            "name": "Calculator Arithmetic (Anchor)",
            "goal": "open Calculator and calculate 50 - 20",
            "pkg": "com.sec.android.app.popupcalculator",
            "timeout": 25,
        },
        {
            "name": "Instagram Search Photography (Failure Class B - Tab vs Input)",
            "goal": "open Instagram and search photography",
            "pkg": "com.instagram.android",
            "timeout": 25,
        },
        {
            "name": "WhatsApp Contact & Composer (Failure Class A - Header vs Composer)",
            "goal": "open WhatsApp and send hi to Ajay",
            "pkg": "com.whatsapp",
            "timeout": 25,
        },
        {
            "name": "Zepto Quick Commerce (Search & Price Grounding)",
            "goal": "Open Zepto and search fresh tomatoes",
            "pkg": "com.zeptoconsumerapp",
            "timeout": 30,
        },
    ]

    print("=" * 80)
    print("STARTING KARVISH V56 PHYSICAL DEVICE QUALIFICATION AUDIT")
    print(f"Target Device: {DEVICE_SERIAL}")
    print("=" * 80)

    results = []
    for tc in test_cases:
        res = run_v56_live_goal(tc["goal"], target_app_pkg=tc.get("pkg"), timeout=tc["timeout"])
        results.append(res)
        time.sleep(2.0)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"PHYSICAL DEVICE AUDIT REPORT GENERATED: {REPORT_FILE.name}")
    print(f"Total Live Tasks Executed: {len(results)}")
    print("=" * 80)


if __name__ == "__main__":
    run_full_v56_physical_audit()
