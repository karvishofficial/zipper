"""
Karvish Mobile Action: Device Rollout Collector & Telemetry (Step 3).

Executes autonomous episodes on physical Android device (Samsung Galaxy SM-A266B),
captures granular turn transitions (s_t, a_t, r_t, s_{t+1}), evaluates transitions
with TaskStateOracleVerifier, and records verified trajectories for RL / RSI.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRAIN_DIR = os.path.join(REPO_ROOT, "tools", "training", "karvish_mobile_action")
LIVE_DIR = os.path.join(REPO_ROOT, "tools", "live_device")
sys.path.insert(0, TRAIN_DIR)
sys.path.insert(0, LIVE_DIR)

from task_state_verifier import TaskStateOracleVerifier, VerificationState  # noqa: E402
from trajectory_logger import EpisodeTrajectory, TurnTransition  # noqa: E402

ADB_PATH = os.path.expandvars(r"%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe")
DEFAULT_DEVICE = "RZCY51MQP9K"


def run_adb(device_id: str, args: List[str]) -> subprocess.CompletedProcess:
    cmd = [ADB_PATH, "-s", device_id] + args
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def ensure_adb_reverse(device_id: str, port: int = 8765) -> bool:
    res = run_adb(device_id, ["reverse", f"tcp:{port}", f"tcp:{port}"])
    return res.returncode == 0


def collect_device_rollout(
    goal: str,
    device_id: str = DEFAULT_DEVICE,
    enable_recovery: bool = True,
    timeout_sec: int = 30,
    out_dir: Optional[str] = None,
) -> EpisodeTrajectory:
    verifier = TaskStateOracleVerifier()
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_goal_name = re.sub(r"[^a-zA-Z0-9]+", "_", goal)[:30].strip("_")
    episode_id = f"ep_{timestamp_str}_{clean_goal_name}"

    out_dir = out_dir or os.path.join(REPO_ROOT, "tools", "live_device", "trajectories")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=======================================================")
    print(f"[ROLLOUT] Episode: {episode_id}")
    print(f"[ROLLOUT] Goal: '{goal}'")
    print(f"[ROLLOUT] Target Device: {device_id}")
    print(f"=======================================================")

    # Ensure port forwarding
    ensure_adb_reverse(device_id)

    # 1. TEST SETUP (Per Rule 16): Reset device to home screen, clean target app state, and clear logcat
    run_adb(device_id, ["shell", "input", "keyevent", "KEYCODE_HOME"])
    lower_goal = goal.lower()
    if "youtube" in lower_goal:
        run_adb(device_id, ["shell", "am", "force-stop", "com.google.android.youtube"])
    elif "chrome" in lower_goal:
        run_adb(device_id, ["shell", "am", "force-stop", "com.android.chrome"])
    elif "google" in lower_goal:
        run_adb(device_id, ["shell", "am", "force-stop", "com.google.android.googlequicksearchbox"])
    elif "clock" in lower_goal or "stopwatch" in lower_goal or "timer" in lower_goal or "alarm" in lower_goal:
        run_adb(device_id, ["shell", "am", "force-stop", "com.sec.android.app.clockpackage"])
    elif "calculator" in lower_goal:
        run_adb(device_id, ["shell", "am", "force-stop", "com.sec.android.app.popupcalculator"])
    elif "linkedin" in lower_goal:
        run_adb(device_id, ["shell", "am", "force-stop", "com.linkedin.android"])
    elif "instagram" in lower_goal:
        run_adb(device_id, ["shell", "am", "force-stop", "com.instagram.android"])
    time.sleep(1.0)
    run_adb(device_id, ["shell", "logcat", "-c"])

    # 2. Launch Karvish Agent Runtime
    recovery_flag = "true" if enable_recovery else "false"
    t0 = time.time()
    run_adb(device_id, [
        "shell", "am", "start", "-n", "com.zipper.app/.MainActivity",
        "--es", "run_agent_goal", f'"{goal}"',
        "--ez", "enable_active_recovery", recovery_flag
    ])

    # 3. Monitor logcat until completion, termination, or timeout
    completed = False
    timed_out = False

    while time.time() - t0 < timeout_sec:
        time.sleep(1.2)
        log = run_adb(device_id, ["shell", "logcat", "-d", "-s", "KarvishAgentRuntime", "KarvishA11yService", "LiveRankerModelEngine"]).stdout
        if "AUTONOMOUS GOAL ACHIEVED" in log or "GOAL COMPLETED" in log or "TERMINATING (GOAL_SUCCESS)" in log:
            completed = True
            break
        if "AGENT STOPPED" in log or "Early termination" in log or "STUCK_NO_PROGRESS" in log or "TERMINATING" in log:
            completed = False
            break

    total_time = round(time.time() - t0, 2)
    if not completed and total_time >= timeout_sec:
        timed_out = True

    # 4. Extract turns and audit trace from logcat
    log = run_adb(device_id, ["shell", "logcat", "-d", "-s", "KarvishAgentRuntime", "KarvishA11yService", "LiveRankerModelEngine"]).stdout
    turn_traces = re.findall(
        r"==================== \[TURN (\d+) AUDIT TRACE.*?\] ====================([\s\S]*?)==================================================================",
        log,
    )

    history_acc: List[Dict[str, Any]] = []
    turns: List[TurnTransition] = []
    cumulative_reward = 0.0
    recovery_events = 0
    cycle_oscillations = 0
    false_completes = 0
    last_action_desc = ""

    for turn_num_str, trace_body in turn_traces:
        turn_num = int(turn_num_str)

        # Parse current app
        m_app = re.search(r"2\.\s*CURRENT APP:\s*(.*?)\n", trace_body)
        current_app = m_app.group(1).strip() if m_app else "unknown"

        # Parse activity
        m_act = re.search(r"3\.\s*CURRENT ACTIVITY:\s*(.*?)\n", trace_body)
        activity_name = m_act.group(1).strip() if m_act else None
        if activity_name == "null":
            activity_name = None

        # Parse node count
        m_nodes = re.search(r"4\.\s*ACCESSIBILITY NODE COUNT:\s*(\d+)", trace_body)
        node_count = int(m_nodes.group(1)) if m_nodes else 0

        # Parse chosen action
        m_action = re.search(r"8\.\s*PARSED ACTION:\s*(.*?)\n", trace_body)
        action_desc = m_action.group(1).strip() if m_action else "Unknown"

        # Parse confidence
        m_conf = re.search(r"9\.\s*CONFIDENCE:\s*([\d\.]+)", trace_body)
        conf = float(m_conf.group(1)) if m_conf else 0.85

        # Parse latency
        m_lat = re.search(r"Latency:\s*(\d+)ms", trace_body)
        latency_ms = int(m_lat.group(1)) if m_lat else 15

        # Parse post-action app and nodes
        m_post = re.search(r"12\.\s*POST-ACTION STATE:\s*([^\(]+)\((\d+)\s+nodes\)", trace_body)
        post_app = m_post.group(1).strip() if m_post else current_app
        post_node_count = int(m_post.group(2)) if m_post else node_count

        # Parse element changes
        m_app_nodes = re.search(r"(\d+)\s+new element\(s\)\s+appeared", trace_body)
        m_dis_nodes = re.search(r"(\d+)\s+element\(s\)\s+closed/disappeared", trace_body)
        elements_appeared = int(m_app_nodes.group(1)) if m_app_nodes else 0
        elements_disappeared = int(m_dis_nodes.group(1)) if m_dis_nodes else 0

        # Parse verification string
        m_ver = re.search(r"13\.\s*VERIFICATION:\s*(.*?)\n", trace_body)
        ver_str = m_ver.group(1).strip() if m_ver else "UNKNOWN"

        state_changed = (post_app != current_app) or (elements_appeared > 0) or (elements_disappeared > 0)
        is_cycle = (action_desc == last_action_desc and not state_changed)
        if is_cycle:
            cycle_oscillations += 1
        last_action_desc = action_desc

        if "Active Recovery" in trace_body:
            recovery_events += 1

        # Map action_desc to action_dict
        act_dict = {"action_type": "tap", "target": action_desc}
        if "OpenApp" in action_desc:
            act_dict["action_type"] = "open_app"
        elif "PressBack" in action_desc:
            act_dict["action_type"] = "back"
        elif "Complete" in action_desc:
            act_dict["action_type"] = "complete"
        elif "Type" in action_desc:
            act_dict["action_type"] = "type_text"
        elif "Submit" in action_desc:
            act_dict["action_type"] = "submit"

        obs_dict = {
            "package_name": post_app,
            "activity_name": activity_name,
            "nodes": [{"text": action_desc}],
        }

        # Evaluate structured turn reward using TaskStateOracleVerifier
        v_state = verifier.evaluate_turn(
            goal=goal,
            observation=obs_dict,
            action=act_dict,
            history=history_acc,
            state_changed=state_changed,
            is_cycle=is_cycle,
        )

        if v_state.is_violation:
            false_completes += 1

        turn_reward = v_state.reward
        cumulative_reward += turn_reward

        turns.append(
            TurnTransition(
                turn=turn_num,
                goal=goal,
                current_app=current_app,
                activity_name=activity_name,
                node_count=node_count,
                candidates=[],
                chosen_candidate_id=action_desc,
                chosen_action_type=act_dict["action_type"],
                chosen_label=action_desc,
                confidence=conf,
                latency_ms=latency_ms,
                actual_execution=True,
                post_app=post_app,
                post_node_count=post_node_count,
                elements_appeared=elements_appeared,
                elements_disappeared=elements_disappeared,
                verification=ver_str,
                step_reward=round(turn_reward, 3),
                cumulative_return=round(cumulative_reward, 3),
                raw_trace=trace_body.strip(),
            )
        )

        history_acc.append({
            "action_type": act_dict["action_type"],
            "target": action_desc,
            "package_name": post_app,
        })

    final_status = "SUCCESS" if completed else ("TIMEOUT" if timed_out else "FAILED")
    episode = EpisodeTrajectory(
        episode_id=episode_id,
        timestamp=timestamp_str,
        device_id=device_id,
        goal=goal,
        mode="LIVE_ROLLOUT",
        success=completed,
        final_status=final_status,
        total_turns=len(turns),
        total_time_sec=total_time,
        total_return=round(cumulative_reward, 3),
        recovery_events=recovery_events,
        cycle_oscillations=cycle_oscillations,
        false_completes=false_completes,
        turns=turns,
    )

    out_file = os.path.join(out_dir, f"{episode_id}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(asdict(episode), f, indent=2)

    print(f"[ROLLOUT COMPLETE] Status={final_status} Turns={len(turns)} "
          f"Return={cumulative_reward:.2f} FalseCompletes={false_completes} -> {out_file}")

    return episode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Device Rollout Collector")
    parser.add_argument("--goal", type=str, default="Open Clock and switch to Stopwatch")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    collect_device_rollout(
        goal=args.goal,
        device_id=args.device,
        timeout_sec=args.timeout,
        out_dir=args.out_dir,
    )
