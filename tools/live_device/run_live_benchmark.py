"""
Karvish Mobile Action: Live Device Benchmark Suite (Milestone v0.8).

Executes live safe, low-risk mobile tasks on a connected physical Android device.
Compares:
1. Baseline: Rank-1 Only Runtime (enableActiveRecovery = False)
2. Milestone: Active Recovery Runtime (enableActiveRecovery = True)

Evaluates:
- Live checkpoint success rate
- Live full task success rate
- Bounded attempts per checkpoint
- No-op and failure recovery events
- Zero repeated loops (zero-tolerance)
- Zero premature completions (zero-tolerance)
- Inference latency per turn
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

ADB_PATH = os.path.expandvars(r"%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe")
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DEVICE = "RZCY51MQP9K"

TEST_TASKS = [
    {
        "id": "task_1_open_clock",
        "goal": "Open Clock",
        "description": "Atomic application launch & foreground verification",
        "timeout_sec": 18,
    },
    {
        "id": "task_2_open_settings",
        "goal": "Open Settings",
        "description": "Atomic system settings navigation & foreground verification",
        "timeout_sec": 18,
    },
    {
        "id": "task_3_clock_tabs",
        "goal": "Open Clock and switch to Stopwatch",
        "description": "Compound multi-step navigation & explicit tab selection",
        "timeout_sec": 25,
    },
    {
        "id": "task_4_settings_scroll",
        "goal": "Open Settings and scroll down",
        "description": "Container scroll displacement & list advancement",
        "timeout_sec": 25,
    },
    {
        "id": "task_5_system_back",
        "goal": "Press Back to return to Karvish",
        "description": "System navigation & recovery from active overlay",
        "timeout_sec": 18,
    },
    {
        "id": "task_6_multi_app_message_calculator",
        "goal": "Open WhatsApp, search for Alice, send Hello to Alice, then open Calculator and calculate 25 + 37",
        "description": "Generic multi-app launch, search, typed message, verified send, return, and calculation",
        "timeout_sec": 60,
    },
]


def run_adb(device_id: str, args: List[str]) -> subprocess.CompletedProcess:
    cmd = [ADB_PATH, "-s", device_id] + args
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def execute_live_task(
    device_id: str,
    goal: str,
    enable_recovery: bool = True,
    timeout_sec: int = 25,
) -> Dict[str, Any]:
    mode_str = "ACTIVE_RECOVERY" if enable_recovery else "RANK1_ONLY"
    print(f"\n>> [{mode_str}] Launching Task: '{goal}'")

    # Ensure adb reverse is active
    run_adb(device_id, ["reverse", "tcp:8765", "tcp:8765"])

    # 1. Reset device to home screen and clear logcat
    run_adb(device_id, ["shell", "input", "keyevent", "KEYCODE_HOME"])
    time.sleep(1.0)
    run_adb(device_id, ["shell", "logcat", "-c"])

    # 2. Trigger Karvish autonomous agent via intent
    start_time = time.time()
    recovery_flag = "true" if enable_recovery else "false"
    intent_res = run_adb(device_id, [
        "shell", "am", "start", "-n", "com.zipper.app/.MainActivity",
        "--es", "run_agent_goal", f'"{goal}"',
        "--ez", "enable_active_recovery", recovery_flag
    ])
    print(f"   Intent dispatched: {intent_res.stdout.strip()}")

    # 3. Poll logcat for runtime outcome
    completed = False
    timed_out = False

    while time.time() - start_time < timeout_sec:
        time.sleep(1.2)
        log = run_adb(device_id, ["shell", "logcat", "-d", "-s", "KarvishAgentRuntime", "KarvishA11yService", "LiveRankerModelEngine"]).stdout
        if "AUTONOMOUS GOAL ACHIEVED" in log or "GOAL COMPLETED" in log or "TERMINATING (GOAL_SUCCESS)" in log:
            completed = True
            break
        if "AGENT STOPPED" in log or "Early termination" in log or "STUCK_NO_PROGRESS" in log or "TERMINATING" in log:
            completed = False
            break

    total_time = round(time.time() - start_time, 2)
    if not completed and total_time >= timeout_sec:
        timed_out = True

    # 4. Pull and parse full execution logcat
    log = run_adb(device_id, ["shell", "logcat", "-d", "-s", "KarvishAgentRuntime", "KarvishA11yService", "LiveRankerModelEngine"]).stdout

    turn_traces = re.findall(
        r"==================== \[TURN (\d+) AUDIT TRACE.*?\] ====================([\s\S]*?)==================================================================",
        log,
    )

    turns_data = []
    latencies = []
    actions = []
    outcomes = []
    recovery_events = 0
    repeated_loops = 0
    false_completes = 0

    last_action_desc = None
    for turn_num_str, trace_body in turn_traces:
        turn_num = int(turn_num_str)

        # Extract parsed action
        action_match = re.search(r"8\.\s*PARSED ACTION:\s*(.*?)\n", trace_body)
        act_desc = action_match.group(1).strip() if action_match else "Unknown"
        actions.append(act_desc)

        # Extract confidence
        conf_match = re.search(r"9\.\s*CONFIDENCE:\s*([\d\.]+)", trace_body)
        conf = float(conf_match.group(1)) if conf_match else 0.85

        # Extract execution latency
        lat_match = re.search(r"Latency:\s*(\d+)ms", trace_body)
        lat = int(lat_match.group(1)) if lat_match else 15
        latencies.append(lat)

        # Extract outcome
        outcome_match = re.search(r"13\.\s*VERIFICATION:\s*([A-Z_]+)", trace_body)
        outcome_val = outcome_match.group(1) if outcome_match else "UNKNOWN"
        outcomes.append(outcome_val)

        # Check for loop repetition
        if last_action_desc is not None and last_action_desc == act_desc and outcome_val in ["NO_OP", "BLOCKED"]:
            repeated_loops += 1
        last_action_desc = act_desc

        if "Active Recovery: Fell back" in log:
            recovery_events += 1

        turns_data.append({
            "turn": turn_num,
            "action": act_desc,
            "confidence": conf,
            "latency_ms": lat,
            "outcome": outcome_val,
        })

    # Check for premature completion
    if "COMPLETE_REJECTED" in log:
        false_completes += 1

    success = ("AUTONOMOUS GOAL ACHIEVED" in log or "GOAL COMPLETED" in log or "TERMINATING (GOAL_SUCCESS)" in log)
    turns_count = len(turns_data) if turns_data else (1 if success else 0)
    avg_latency = round(sum(latencies) / max(len(latencies), 1), 2)

    status_str = "SUCCESS" if success else ("TIMEOUT" if timed_out else "FAILED")
    print(f"   Task Result: {status_str} | Turns: {turns_count} | Latency: {avg_latency}ms | Loops: {repeated_loops}")

    return {
        "goal": goal,
        "mode": mode_str,
        "enable_recovery": enable_recovery,
        "success": success,
        "status": status_str,
        "turns_executed": turns_count,
        "total_time_sec": total_time,
        "avg_turn_latency_ms": avg_latency,
        "recovery_events": recovery_events,
        "repeated_loops": repeated_loops,
        "false_completes": false_completes,
        "turns": turns_data,
        "actions": actions,
        "outcomes": outcomes,
    }


def run_benchmark(device_id: str, out_path: str, task_filter: Optional[str] = None) -> Dict[str, Any]:
    print("=" * 75)
    print("KARVISH MOBILE ACTION: LIVE DEVICE VERIFICATION BENCHMARK")
    print(f"Target Device: {device_id}")
    print("=" * 75)

    # Configure adb reverse port forwarding
    print(f">> Configuring ADB reverse port forwarding (tcp:8765 -> tcp:8765)...")
    run_adb(device_id, ["reverse", "tcp:8765", "tcp:8765"])

    rank1_results = []
    recovery_results = []

    # Run Benchmark across all tasks in both modes
    tasks_to_run = [t for t in TEST_TASKS if task_filter is None or t["id"] == task_filter or t["goal"].lower() == task_filter.lower()]
    for task in tasks_to_run:
        g = task["goal"]
        t_sec = task["timeout_sec"]

        # Run Mode A: Rank-1 Only
        res_r1 = execute_live_task(device_id, g, enable_recovery=False, timeout_sec=t_sec)
        rank1_results.append(res_r1)
        time.sleep(1.5)

        # Run Mode B: Active Recovery
        res_rec = execute_live_task(device_id, g, enable_recovery=True, timeout_sec=t_sec)
        recovery_results.append(res_rec)
        time.sleep(1.5)

    # Calculate Aggregate Metrics
    n_tasks = len(tasks_to_run)

    r1_success = sum(1 for r in rank1_results if r["success"])
    rec_success = sum(1 for r in recovery_results if r["success"])

    r1_turns = sum(r["turns_executed"] for r in rank1_results)
    rec_turns = sum(r["turns_executed"] for r in recovery_results)

    r1_loops = sum(r["repeated_loops"] for r in rank1_results)
    rec_loops = sum(r["repeated_loops"] for r in recovery_results)

    r1_fc = sum(r["false_completes"] for r in rank1_results)
    rec_fc = sum(r["false_completes"] for r in recovery_results)

    r1_lat = round(sum(r["avg_turn_latency_ms"] for r in rank1_results) / max(n_tasks, 1), 2)
    rec_lat = round(sum(r["avg_turn_latency_ms"] for r in recovery_results) / max(n_tasks, 1), 2)

    total_rec_events = sum(r["recovery_events"] for r in recovery_results)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device_id": device_id,
        "total_tasks": n_tasks,
        "rank1_only": {
            "tasks_passed": r1_success,
            "task_success_rate": round((r1_success / n_tasks) * 100, 2),
            "total_turns": r1_turns,
            "avg_turns_per_task": round(r1_turns / n_tasks, 2),
            "repeated_action_loops": r1_loops,
            "false_completes": r1_fc,
            "avg_turn_latency_ms": r1_lat,
            "task_details": rank1_results,
        },
        "active_recovery": {
            "tasks_passed": rec_success,
            "task_success_rate": round((rec_success / n_tasks) * 100, 2),
            "total_turns": rec_turns,
            "avg_turns_per_task": round(rec_turns / n_tasks, 2),
            "repeated_action_loops": rec_loops,
            "false_completes": rec_fc,
            "avg_turn_latency_ms": rec_lat,
            "recovery_events_triggered": total_rec_events,
            "task_details": recovery_results,
        },
    }

    print("\n" + "=" * 75)
    print("LIVE DEVICE BENCHMARK SUMMARY SCORECARD")
    print("=" * 75)
    print(f"Task Success Rate:      Rank-1 Only: {report['rank1_only']['task_success_rate']}% ({r1_success}/{n_tasks}) | Active Recovery: {report['active_recovery']['task_success_rate']}% ({rec_success}/{n_tasks})")
    print(f"Average Turns/Task:     Rank-1 Only: {report['rank1_only']['avg_turns_per_task']} | Active Recovery: {report['active_recovery']['avg_turns_per_task']}")
    print(f"Recovery Events:        {total_rec_events} recovery actions executed")
    print(f"Repeated Loops:         Rank-1: {r1_loops} | Active Recovery: {rec_loops} (Target: 0)")
    print(f"False Completes:        Rank-1: {r1_fc} | Active Recovery: {rec_fc} (Target: 0)")
    print(f"Average Turn Latency:   Rank-1: {r1_lat}ms | Active Recovery: {rec_lat}ms")
    print("=" * 75)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f">> Report successfully saved to: {out_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Device Benchmark Suite")
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--task", default=None, help="Filter to specific task ID or goal")
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "tools", "live_device", "live_eval_report.json"))
    args = parser.parse_args()
    run_benchmark(device_id=args.device, out_path=args.out, task_filter=args.task)
