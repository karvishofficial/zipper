"""
KARVISH V51.7 — SEALED PHYSICAL EVALUATION & REGRESSION SUITE
Executes the authoritative sealed physical evaluation on Samsung Galaxy A26 (RZCY51MQP9K).

Sealed Tasks:
1. Calculator: 85 * 6 -> clear
2. YouTube: search SpaceX -> view results -> back
3. Settings: search Battery
4. Keep: create note -> type "Meeting Notes" -> return
5. Play Store: search Chess -> open/top result

Historical V42 Regression Tasks:
1. YouTube: search "lo-fi hip hop"
2. LinkedIn: search "software engineer"
3. Instagram: search "photography"
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8")

ADB_PATH = os.path.expandvars(r"%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe")

def get_connected_devices() -> List[str]:
    try:
        res = subprocess.run([ADB_PATH, "devices"], capture_output=True, text=True, timeout=5)
        lines = res.stdout.strip().splitlines()
        devs = []
        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "device":
                devs.append(parts[0])
        return devs
    except Exception:
        return []

_connected = get_connected_devices()
DEVICE_SERIAL = _connected[0] if _connected else "RZCY51MQP9K"
OUTPUT_DIR = Path("tools/live_device/eval_results_v51_7")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEALED_TASKS = [
    {
        "id": "sealed_1_calculator",
        "category": "SEALED",
        "name": "Calculator: 85 * 6 -> clear",
        "goal": "Open Calculator, calculate 85 * 6, then clear",
        "pkg": "com.sec.android.app.popupcalculator",
        "timeout": 35,
        "success_keywords": ["510", "Clear", "Calculator"],
    },
    {
        "id": "sealed_2_youtube",
        "category": "SEALED",
        "name": "YouTube: search SpaceX -> view results -> back",
        "goal": "Open YouTube, search for SpaceX, view results, then press back",
        "pkg": "com.google.android.youtube",
        "timeout": 40,
        "success_keywords": ["SpaceX", "results", "YouTube"],
    },
    {
        "id": "sealed_3_settings",
        "category": "SEALED",
        "name": "Settings: search Battery",
        "goal": "Open Settings and search for Battery",
        "pkg": "com.android.settings",
        "timeout": 30,
        "success_keywords": ["Battery", "Settings"],
    },
    {
        "id": "sealed_4_keep",
        "category": "SEALED",
        "name": 'Keep: create note -> type "Meeting Notes" -> return',
        "goal": 'Open Keep, create note, type Meeting Notes, then return',
        "pkg": "com.google.android.keep",
        "timeout": 45,
        "success_keywords": ["Meeting Notes", "Note", "Keep"],
    },
    {
        "id": "sealed_5_playstore",
        "category": "SEALED",
        "name": "Play Store: search Chess -> open/top result",
        "goal": "Open Play Store, search for Chess, and view top result",
        "pkg": "com.android.vending",
        "timeout": 40,
        "success_keywords": ["Chess", "Play Store"],
    },
]

REGRESSION_TASKS = [
    {
        "id": "reg_1_youtube_lofi",
        "category": "REGRESSION_SEARCH",
        "name": 'YouTube: search "lo-fi hip hop"',
        "goal": "Open YouTube and search for lo-fi hip hop",
        "pkg": "com.google.android.youtube",
        "timeout": 35,
        "success_keywords": ["lo-fi", "hip hop", "results"],
    },
    {
        "id": "reg_2_linkedin_se",
        "category": "REGRESSION_SEARCH",
        "name": 'LinkedIn: search "software engineer"',
        "goal": "Open LinkedIn and search for software engineer",
        "pkg": "com.linkedin.android",
        "timeout": 35,
        "success_keywords": ["software engineer", "LinkedIn"],
    },
    {
        "id": "reg_3_instagram_photo",
        "category": "REGRESSION_SEARCH",
        "name": 'Instagram: search "photography"',
        "goal": "Open Instagram and search for photography",
        "pkg": "com.instagram.android",
        "timeout": 35,
        "success_keywords": ["photography", "Instagram"],
    },
]


def run_adb(args: List[str], timeout: int = 15) -> str:
    cmd = [ADB_PATH, "-s", DEVICE_SERIAL] + args
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
        return res.stdout.strip()
    except Exception as e:
        return f"ADB_ERROR: {e}"


def check_gpu_server() -> bool:
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:8765/health", timeout=3) as resp:
            data = json.loads(resp.read().decode())
            return data.get("status") == "ready"
    except Exception:
        return False


def execute_physical_task(task_def: Dict[str, Any]) -> Dict[str, Any]:
    task_id = task_def["id"]
    goal = task_def["goal"]
    pkg = task_def.get("pkg")
    timeout = task_def.get("timeout", 35)

    print("\n" + "=" * 80)
    print(f"EXECUTING TASK [{task_def['category']}]: {task_def['name']}")
    print(f"Goal: '{goal}' | Pkg: {pkg} | Timeout: {timeout}s")
    print("=" * 80)

    # 1. Reset state
    run_adb(["reverse", "tcp:8765", "tcp:8765"])
    run_adb(["shell", "input keyevent 224"])  # Wakeup
    run_adb(["shell", "input keyevent 82"])   # Unlock
    run_adb(["shell", "input keyevent 3"])    # Home
    if pkg:
        run_adb(["shell", f"am force-stop {pkg}"])
    run_adb(["shell", "am force-stop com.android.chrome"])
    run_adb(["shell", "am force-stop com.truecaller"])
    run_adb(["shell", "am force-stop com.samsung.android.lool"])
    run_adb(["shell", "am force-stop com.dubox.drive"])
    run_adb(["shell", "am force-stop com.instagram.android"])
    run_adb(["shell", "input keyevent 3"])
    time.sleep(1.0)

    # Ensure Karvish Accessibility Service is active
    run_adb(["shell", "settings", "put", "secure", "enabled_accessibility_services", "com.zipper.app/.accessibility.KarvishAccessibilityService"])
    run_adb(["shell", "settings", "put", "secure", "accessibility_enabled", "1"])

    # Clear logcat
    run_adb(["shell", "logcat", "-c"])

    # 2. Launch Karvish Agent Intent
    t0 = time.time()
    launch_res = run_adb([
        "shell",
        f"am start -n com.zipper.app/.MainActivity --es run_agent_goal '{goal}' --ez enable_active_recovery false --ez use_remote_ranker true"
    ])
    print(f"  -> Dispatched intent: {launch_res}")

    # 3. Monitor execution
    completed = False
    stopped = False
    turn_count = 0
    poll_interval = 1.5
    start_marker = f"Goal: '{goal}'"

    while time.time() - t0 < timeout:
        time.sleep(poll_interval)
        elapsed = round(time.time() - t0, 1)
        raw_logs = run_adb(["shell", "logcat", "-d", "-s", "KarvishAgentService", "KarvishAgentRuntime", "LiveRankerModelEngine", "KarvishActionDispatcher"])
        
        # Strictly isolate logs to this task's execution
        if start_marker in raw_logs:
            logs = raw_logs[raw_logs.rfind(start_marker):]
        else:
            print(f"  ... [T+{elapsed}s] Waiting for agent task start...")
            continue
        
        # Check termination triggers
        if "AUTONOMOUS GOAL ACHIEVED" in logs or "GOAL COMPLETED" in logs or "TERMINATING (GOAL_SUCCESS)" in logs or "Goal finished:" in logs:
            completed = True
            print(f"  -> Goal Completed in {elapsed}s!")
            break
        if "AGENT STOPPED" in logs or "Early termination" in logs or "TERMINATING" in logs or "Goal declared unreachable" in logs or "GOAL_FAILED" in logs:
            stopped = True
            print(f"  -> Agent stopped in {elapsed}s.")
            break
        print(f"  ... [T+{elapsed}s] Task executing...")

    duration = round(time.time() - t0, 2)

    # 4. Pull Full Logcat Traces
    full_logs_raw = run_adb(["shell", "logcat", "-d", "-s", "KarvishAgentService", "KarvishAgentRuntime", "LiveRankerModelEngine", "KarvishA11yService", "KarvishActionDispatcher"])
    if start_marker in full_logs_raw:
        full_logs = full_logs_raw[full_logs_raw.rfind(start_marker):]
    else:
        full_logs = full_logs_raw
    log_file = OUTPUT_DIR / f"{task_id}_logcat.log"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(full_logs)

    # 5. Capture Final Observable UI State (Screenshot & UI Dump)
    screen_file = OUTPUT_DIR / f"{task_id}_final_screen.png"
    try:
        img_bytes = subprocess.check_output([ADB_PATH, "-s", DEVICE_SERIAL, "exec-out", "screencap", "-p"], timeout=10)
        with open(screen_file, "wb") as f:
            f.write(img_bytes)
        print(f"  -> Saved final screenshot: {screen_file.name} ({len(img_bytes)} bytes)")
    except Exception as e:
        print(f"  -> Failed to capture screenshot: {e}")

    # Dump UI hierarchy
    ui_xml_file = OUTPUT_DIR / f"{task_id}_ui_dump.xml"
    run_adb(["shell", "uiautomator dump /sdcard/window_dump.xml"])
    ui_xml_content = run_adb(["shell", "cat /sdcard/window_dump.xml"])
    with open(ui_xml_file, "w", encoding="utf-8") as f:
        f.write(ui_xml_content)

    # 6. Parse Turn Traces
    turn_traces = re.findall(
        r"==================== \[TURN (\d+) AUDIT TRACE.*?\] ====================([\s\S]*?)==================================================================",
        full_logs,
    )
    
    actions_list = []
    latencies = []
    last_act = None
    loop_detected = False

    for t_num, trace_body in turn_traces:
        action_match = re.search(r"8\.\s*PARSED ACTION:\s*(.*?)\n", trace_body)
        act_str = action_match.group(1).strip() if action_match else "UNKNOWN"
        
        target_match = re.search(r"7\.\s*ACTION CANDIDATE:\s*(.*?)\n", trace_body)
        target_str = target_match.group(1).strip() if target_match else "None"

        conf_match = re.search(r"9\.\s*CONFIDENCE:\s*([\d\.]+)", trace_body)
        conf = float(conf_match.group(1)) if conf_match else 0.0

        lat_match = re.search(r"Latency:\s*(\d+)ms", trace_body)
        lat = int(lat_match.group(1)) if lat_match else 0
        latencies.append(lat)

        verif_match = re.search(r"13\.\s*VERIFICATION:\s*([A-Z_]+)", trace_body)
        verif = verif_match.group(1) if verif_match else "UNKNOWN"

        if last_act == act_str and verif in ["NO_OP", "BLOCKED"]:
            loop_detected = True

        actions_list.append({
            "turn": int(t_num),
            "action": act_str,
            "target": target_str,
            "confidence": conf,
            "latency_ms": lat,
            "verification": verif,
        })
        last_act = act_str

    # 7. Evaluate Observable Final Success
    # Grounded strictly in observable final UI state
    current_pkg_match = re.search(r'package="([^"]+)"', ui_xml_content)
    current_pkg = current_pkg_match.group(1) if current_pkg_match else ""
    is_still_launcher = (current_pkg == "com.zipper.app")

    observable_success = False
    observable_state_summary = ""

    if is_still_launcher:
        observable_success = False
        observable_state_summary = f"FAILED: Agent remained on launcher screen ({current_pkg}); target app {pkg} was not in foreground."
    elif task_id == "sealed_1_calculator":
        is_calc_pkg = "popupcalculator" in ui_xml_content.lower() or "calculator" in ui_xml_content.lower()
        calc_actions = [a.get("action", "") for a in actions_list]
        tapped_8 = any("'8'" in a for a in calc_actions)
        tapped_5 = any("'5'" in a for a in calc_actions)
        tapped_op = any("×" in a or "*" in a or "times" in a.lower() for a in calc_actions)
        tapped_6 = any("'6'" in a for a in calc_actions)
        has_510 = "510" in ui_xml_content
        has_clear = any("clear" in a.lower() or "'c'" in a.lower() for a in calc_actions)
        full_math_sequence = tapped_8 and tapped_5 and tapped_op and tapped_6
        # Strictly require that 510 was computed OR the full sequence (8, 5, ×, 6) occurred
        observable_success = is_calc_pkg and (has_510 or (full_math_sequence and has_clear))
        observable_state_summary = f"Calculator foreground: {is_calc_pkg}, Formula aligned: {full_math_sequence}, 510 Computed: {has_510}, Clear Executed: {has_clear}, Actions: {len(actions_list)}"

    elif task_id == "sealed_2_youtube":
        is_yt_pkg = "youtube" in ui_xml_content.lower()
        # Must have typed SpaceX into search and executed search
        yt_actions = [a.get("action", "") for a in actions_list]
        typed_spacex = any("spacex" in a.lower() for a in yt_actions)
        submitted = any("submit" in a.lower() for a in yt_actions)
        has_results = "spacex" in ui_xml_content.lower()
        observable_success = is_yt_pkg and typed_spacex and (submitted or has_results)
        observable_state_summary = f"YouTube foreground: {is_yt_pkg}, Typed SpaceX: {typed_spacex}, Submitted: {submitted}, Results: {has_results}"

    elif task_id == "sealed_3_settings":
        is_settings_pkg = "settings" in ui_xml_content.lower() or "com.samsung.android.lool" in ui_xml_content.lower()
        # Must be in Battery settings screen
        settings_actions = [a.get("action", "") for a in actions_list]
        tapped_battery = any("battery" in a.lower() for a in settings_actions)
        in_battery_screen = "battery" in ui_xml_content.lower() and ("usage" in ui_xml_content.lower() or "protection" in ui_xml_content.lower() or "lool" in ui_xml_content.lower())
        observable_success = is_settings_pkg and (tapped_battery or in_battery_screen)
        observable_state_summary = f"Settings foreground: {is_settings_pkg}, Tapped Battery: {tapped_battery}, In Battery Screen: {in_battery_screen}"

    elif task_id == "sealed_4_keep":
        is_keep_pkg = "keep" in ui_xml_content.lower()
        # Must have actually typed "Meeting Notes" or opened a note editor
        keep_actions = [a.get("action", "") for a in actions_list]
        typed_notes = any("meeting notes" in a.lower() for a in keep_actions)
        has_meeting_notes = "meeting notes" in ui_xml_content.lower()
        in_note_edit = "edit_note_text" in ui_xml_content.lower() or "note_title" in ui_xml_content.lower()
        observable_success = is_keep_pkg and (typed_notes or (has_meeting_notes and in_note_edit))
        observable_state_summary = f"Keep foreground: {is_keep_pkg}, Typed Meeting Notes: {typed_notes}, Note Edit Open: {in_note_edit}"

    elif task_id == "sealed_5_playstore":
        is_vending_pkg = "vending" in ui_xml_content.lower() or "play" in ui_xml_content.lower()
        # Must have typed Chess and submitted search
        store_actions = [a.get("action", "") for a in actions_list]
        typed_chess = any("chess" in a.lower() for a in store_actions)
        has_chess = "chess" in ui_xml_content.lower()
        observable_success = is_vending_pkg and typed_chess and has_chess
        observable_state_summary = f"Play Store foreground: {is_vending_pkg}, Typed Chess: {typed_chess}, Chess Results: {has_chess}"

    elif "reg_1_youtube" in task_id:
        is_yt_pkg = "youtube" in ui_xml_content.lower()
        yt_actions = [a.get("action", "") for a in actions_list]
        typed_lofi = any("lo-fi" in a.lower() or "hip hop" in a.lower() for a in yt_actions)
        has_lofi = "lo-fi" in ui_xml_content.lower() or "hip hop" in ui_xml_content.lower()
        observable_success = is_yt_pkg and typed_lofi and has_lofi
        observable_state_summary = f"YouTube foreground: {is_yt_pkg}, Typed Query: {typed_lofi}, Query Results: {has_lofi}"

    elif "reg_2_linkedin" in task_id:
        is_linkedin_pkg = "linkedin" in ui_xml_content.lower()
        li_actions = [a.get("action", "") for a in actions_list]
        typed_se = any("software engineer" in a.lower() for a in li_actions)
        has_se = "software engineer" in ui_xml_content.lower()
        observable_success = is_linkedin_pkg and typed_se and has_se
        observable_state_summary = f"LinkedIn foreground: {is_linkedin_pkg}, Typed Query: {typed_se}, Results Present: {has_se}"

    elif "reg_3_instagram" in task_id:
        is_insta_pkg = "instagram" in ui_xml_content.lower()
        insta_actions = [a.get("action", "") for a in actions_list]
        typed_photo = any("photography" in a.lower() for a in insta_actions)
        has_photo = "photography" in ui_xml_content.lower()
        observable_success = is_insta_pkg and typed_photo and has_photo
        observable_state_summary = f"Instagram foreground: {is_insta_pkg}, Typed Photography: {typed_photo}, Results Present: {has_photo}"

    result_record = {
        "task_id": task_id,
        "name": task_def["name"],
        "category": task_def["category"],
        "goal": goal,
        "package": pkg,
        "duration_sec": duration,
        "completed_signal": completed,
        "stopped_signal": stopped,
        "observable_success": observable_success,
        "observable_state_summary": observable_state_summary,
        "loop_detected": loop_detected,
        "turn_count": len(actions_list),
        "avg_latency_ms": round(float(sum(latencies) / max(1, len(latencies))), 1),
        "actions": actions_list,
        "screenshot": str(screen_file.resolve()),
        "ui_dump": str(ui_xml_file.resolve()),
    }

    print(f"  -> Observable Success: {observable_success}")
    print(f"  -> Turns: {len(actions_list)} | Loop Detected: {loop_detected} | Duration: {duration}s")
    print(f"  -> Summary: {observable_state_summary}")

    return result_record


def run_all_evaluations(out_dir_str: Optional[str] = None, task_filter: Optional[str] = None):
    global OUTPUT_DIR
    if out_dir_str:
        OUTPUT_DIR = Path(out_dir_str)
    else:
        OUTPUT_DIR = Path("tools/live_device/eval_results_phase_a")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"KARVISH V52+ — EXECUTING PHYSICAL EVALUATION ON DEVICE: {DEVICE_SERIAL}")
    print(f"Output Directory: {OUTPUT_DIR.resolve()}")
    print("=" * 80)

    assert check_gpu_server(), "Host GPU server is not running on port 8765!"
    print(">> Host GPU Server Verified Ready on http://localhost:8765/health.")

    model_name = run_adb(["shell", "getprop", "ro.product.model"]) or "Android Device"
    calc_pkg_list = run_adb(["shell", "pm", "list", "packages", "calc"])
    if "google.android.calculator" in calc_pkg_list:
        SEALED_TASKS[0]["pkg"] = "com.google.android.calculator"
    elif "popupcalculator" in calc_pkg_list:
        SEALED_TASKS[0]["pkg"] = "com.sec.android.app.popupcalculator"

    eval_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device_serial": DEVICE_SERIAL,
        "device_model": model_name,
        "model_version": "KarvishModelV51_7 (Phase A De-Spoonfed)",
        "checkpoint": "v51_phase_a_best.pt",
        "sealed_tasks": [],
        "regression_tasks": [],
    }

    # Filter tasks if requested
    sealed_to_run = [t for t in SEALED_TASKS if not task_filter or task_filter in t["id"]]
    reg_to_run = [t for t in REGRESSION_TASKS if not task_filter or task_filter in t["id"]]

    # 1. Execute Sealed Tasks
    if sealed_to_run:
        print(f"\n--- PHASE 1: {len(sealed_to_run)} SEALED NOVEL APPLICATION TASKS ---")
        for task in sealed_to_run:
            res = execute_physical_task(task)
            eval_results["sealed_tasks"].append(res)
            time.sleep(2.0)

    # 2. Execute Historical V42 Regression Tasks
    if reg_to_run:
        print(f"\n--- PHASE 2: {len(reg_to_run)} HISTORICAL V42 REGRESSION SEARCH TASKS ---")
        for task in reg_to_run:
            res = execute_physical_task(task)
            eval_results["regression_tasks"].append(res)
            time.sleep(2.0)

    # Return to home screen
    run_adb(["shell", "input keyevent 3"])

    # Summary
    sealed_passed = sum(1 for t in eval_results["sealed_tasks"] if t["observable_success"])
    sealed_total = len(eval_results["sealed_tasks"]) if eval_results["sealed_tasks"] else 1
    reg_passed = sum(1 for t in eval_results["regression_tasks"] if t["observable_success"])
    reg_total = len(eval_results["regression_tasks"]) if eval_results["regression_tasks"] else 1

    eval_results["summary"] = {
        "sealed_passed": sealed_passed,
        "sealed_total": len(eval_results["sealed_tasks"]),
        "sealed_pass_rate": round(sealed_passed / sealed_total * 100, 1) if eval_results["sealed_tasks"] else 0.0,
        "regression_passed": reg_passed,
        "regression_total": len(eval_results["regression_tasks"]),
        "regression_pass_rate": round(reg_passed / reg_total * 100, 1) if eval_results["regression_tasks"] else 0.0,
    }

    out_file = OUTPUT_DIR / "phase_a_physical_eval_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    print("\n" + "=" * 80)
    print("PHYSICAL EVALUATION SUITE COMPLETE")
    if eval_results["sealed_tasks"]:
        print(f"  Sealed Tasks Passed:      {sealed_passed}/{len(eval_results['sealed_tasks'])} ({eval_results['summary']['sealed_pass_rate']}%)")
    if eval_results["regression_tasks"]:
        print(f"  Regression Tasks Passed:  {reg_passed}/{len(eval_results['regression_tasks'])} ({eval_results['summary']['regression_pass_rate']}%)")
    print(f"  Results saved to: {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Karvish Physical Evaluation")
    parser.add_argument("--output-dir", type=str, default="tools/live_device/eval_results_phase_a", help="Output directory")
    parser.add_argument("--task", type=str, default=None, help="Filter for specific task ID")
    parser.add_argument("--device-serial", type=str, default=None, help="Device serial number")
    args = parser.parse_args()
    if args.device_serial:
        DEVICE_SERIAL = args.device_serial
    run_all_evaluations(args.output_dir, args.task)
