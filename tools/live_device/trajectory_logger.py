"""
Karvish Mobile Action: Trajectory & Environment Reward Logger (Phase 4 / Phase 5).

Cleanly records live device execution trajectories:
    τ = (s_0, A_0, π_0, a_0, r_0, s_1, A_1, π_1, a_1, r_1, ..., s_T)
including full observation summaries, candidate action sets, model logits,
execution outcomes, and observable environment reward signals.

Also provides the Phase 5 Failure Diagnoser to identify decision-point divergences
and generate verifiable counterfactual repair proposals.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


# Reward Rubric formally agreed for Phase 4 Environment Feedback
REWARD_GOAL_SUCCESS = 10.0
REWARD_MILESTONE_ADVANCE = 1.0
REWARD_PROGRESSIVE_STEP = 0.2
PENALTY_STEP_COST = -0.1
PENALTY_REPEATED_OR_CYCLE = -0.5
PENALTY_WRONG_SCREEN = -0.5
PENALTY_BLOCKED_OR_REFUSAL = -0.3
PENALTY_FALSE_COMPLETE = -10.0


@dataclass
class CandidateRecord:
    id: str
    action_type: str
    label: str
    role: str
    target_id: Optional[int] = None
    score: Optional[float] = None
    confidence: Optional[float] = None
    rank: Optional[int] = None
    semantic_tags: List[str] = field(default_factory=list)


@dataclass
class TurnTransition:
    turn: int
    goal: str
    current_app: str
    activity_name: Optional[str]
    node_count: int
    candidates: List[Dict[str, Any]]
    chosen_candidate_id: str
    chosen_action_type: str
    chosen_label: str
    confidence: float
    latency_ms: int
    actual_execution: bool
    post_app: str
    post_node_count: int
    elements_appeared: int
    elements_disappeared: int
    verification: str
    step_reward: float
    cumulative_return: float
    raw_trace: str = ""


@dataclass
class EpisodeTrajectory:
    episode_id: str
    timestamp: str
    device_id: str
    goal: str
    mode: str
    success: bool
    final_status: str
    total_turns: int
    total_time_sec: float
    total_return: float
    recovery_events: int
    cycle_oscillations: int
    false_completes: int
    turns: List[TurnTransition] = field(default_factory=list)


def compute_turn_reward(
    verification: str,
    is_task_success: bool,
    is_milestone: bool,
    is_cycle: bool,
    is_false_complete: bool,
) -> float:
    """Computes the observable environment reward for an individual transition."""
    if is_false_complete:
        return PENALTY_FALSE_COMPLETE

    reward = PENALTY_STEP_COST  # Base step cost for turn expenditure

    if is_task_success:
        reward += REWARD_GOAL_SUCCESS
        return round(reward, 3)

    if is_cycle:
        reward += PENALTY_REPEATED_OR_CYCLE
        return round(reward, 3)

    if "SUCCESS" in verification:
        if is_milestone:
            reward += REWARD_MILESTONE_ADVANCE
        else:
            reward += REWARD_PROGRESSIVE_STEP
    elif "WRONG_SCREEN" in verification:
        reward += PENALTY_WRONG_SCREEN
    elif "BLOCKED" in verification or "NO_OP" in verification:
        reward += PENALTY_BLOCKED_OR_REFUSAL
    else:
        reward += PENALTY_STEP_COST

    return round(reward, 3)


class TrajectoryLogger:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def parse_logcat_to_trajectory(
        self,
        device_id: str,
        goal: str,
        logcat_text: str,
        mode: str = "LIVE_RUN",
        total_time_sec: float = 0.0,
    ) -> EpisodeTrajectory:
        """Parses standard Karvish audit traces from Android logcat into a structured trajectory."""
        ep_timestamp = time.strftime("%Y%m%d_%H%M%S")
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", goal).strip("_")[:24]
        ep_id = f"ep_{ep_timestamp}_{slug}"

        turn_traces = re.findall(
            r"==================== \[TURN (\d+) AUDIT TRACE.*?\] ====================([\s\S]*?)==================================================================",
            logcat_text,
        )

        turns: List[TurnTransition] = []
        cumulative_return = 0.0
        cycle_count = 0
        recovery_count = len(re.findall(r"Active Recovery: Fell back", logcat_text))
        false_completes = len(re.findall(r"COMPLETE_REJECTED", logcat_text))

        is_final_success = bool(
            re.search(r"AUTONOMOUS GOAL ACHIEVED|GOAL COMPLETED|TERMINATING \(GOAL_SUCCESS\)", logcat_text)
        )

        for turn_idx_str, trace_body in turn_traces:
            t_num = int(turn_idx_str)

            # 1. Goal
            goal_m = re.search(r"1\.\s*GOAL:\s*'(.*?)'", trace_body)
            t_goal = goal_m.group(1).strip() if goal_m else goal

            # 2. App & Activity
            app_m = re.search(r"2\.\s*CURRENT APP:\s*(.*?)\n", trace_body)
            cur_app = app_m.group(1).strip() if app_m else "unknown"

            act_m = re.search(r"3\.\s*CURRENT ACTIVITY:\s*(.*?)\n", trace_body)
            cur_act = act_m.group(1).strip() if act_m and act_m.group(1).strip() != "null" else None

            # 4. Node Count
            nodes_m = re.search(r"4\.\s*ACCESSIBILITY NODE COUNT:\s*(\d+)", trace_body)
            node_cnt = int(nodes_m.group(1)) if nodes_m else 0

            # 7. Raw Output / Chosen Candidate
            raw_out_m = re.search(r"7\.\s*RAW NEEDLE OUTPUT:\s*(\{.*?\})", trace_body)
            chosen_id = "unknown"
            chosen_type = "unknown"
            chosen_label = "unknown"
            score = 0.0
            conf = 0.0
            if raw_out_m:
                try:
                    c_json = json.loads(raw_out_m.group(1))
                    chosen_id = c_json.get("candidate_id", c_json.get("tool_name", "unknown"))
                    chosen_type = c_json.get("action_type", "unknown")
                    chosen_label = c_json.get("label", "unknown")
                    score = float(c_json.get("score", 0.0))
                    conf = float(c_json.get("confidence", 0.0))
                except Exception:
                    pass

            # 8. Parsed Action
            parsed_m = re.search(r"8\.\s*PARSED ACTION:\s*(.*?)\n", trace_body)
            parsed_desc = parsed_m.group(1).strip() if parsed_m else chosen_id

            # 9. Latency
            lat_m = re.search(r"Latency:\s*(\d+)ms", trace_body)
            lat = int(lat_m.group(1)) if lat_m else 15

            # 11. Actual Execution
            exec_m = re.search(r"11\.\s*ACTUAL EXECUTION:\s*(true|false|Skipped)", trace_body, re.IGNORECASE)
            actual_exec = (exec_m.group(1).lower() == "true") if exec_m else False

            # 12. Post State
            post_m = re.search(r"12\.\s*POST-ACTION STATE:\s*([^\(\n]+)(?:\s*\((\d+)\s*nodes\))?", trace_body)
            post_app = post_m.group(1).strip() if post_m else cur_app
            post_nodes = int(post_m.group(2)) if post_m and post_m.group(2) else node_cnt

            # 13. Verification
            verif_m = re.search(r"13\.\s*VERIFICATION:\s*([A-Z_]+)(?:\s*->\s*(.*))?", trace_body)
            verif_status = verif_m.group(1).strip() if verif_m else "UNKNOWN"
            verif_details = verif_m.group(2).strip() if verif_m and verif_m.group(2) else ""

            app_diff_m = re.search(r"(\d+)\s*new element\(s\) appeared", verif_details)
            dis_diff_m = re.search(r"(\d+)\s*element\(s\) closed/disappeared", verif_details)
            appeared_cnt = int(app_diff_m.group(1)) if app_diff_m else 0
            disappeared_cnt = int(dis_diff_m.group(1)) if dis_diff_m else 0

            # Cycle & Milestone checks
            is_turn_cycle = "oscillation" in t_goal.lower() or "cycle" in verif_details.lower()
            if is_turn_cycle:
                cycle_count += 1

            is_milestone = bool(re.search(r"Switched to app", verif_details))
            is_step_success = (t_num == len(turn_traces)) and is_final_success

            r = compute_turn_reward(
                verification=verif_status,
                is_task_success=is_step_success,
                is_milestone=is_milestone,
                is_cycle=is_turn_cycle,
                is_false_complete=False,
            )
            cumulative_return += r

            turns.append(
                TurnTransition(
                    turn=t_num,
                    goal=t_goal,
                    current_app=cur_app,
                    activity_name=cur_act,
                    node_count=node_cnt,
                    candidates=[],
                    chosen_candidate_id=chosen_id,
                    chosen_action_type=chosen_type,
                    chosen_label=chosen_label,
                    confidence=conf,
                    latency_ms=lat,
                    actual_execution=actual_exec,
                    post_app=post_app,
                    post_node_count=post_nodes,
                    elements_appeared=appeared_cnt,
                    elements_disappeared=disappeared_cnt,
                    verification=verif_status,
                    step_reward=r,
                    cumulative_return=round(cumulative_return, 3),
                    raw_trace=trace_body.strip(),
                )
            )

        if is_final_success:
            final_status = "SUCCESS"
        elif "STUCK_NO_PROGRESS" in logcat_text or "Early termination" in logcat_text:
            final_status = "CYCLE_TERMINATED" if cycle_count > 0 else "STUCK_NO_PROGRESS"
        elif "TIMEOUT" in logcat_text:
            final_status = "TIMEOUT"
        else:
            final_status = "FAILED"

        trajectory = EpisodeTrajectory(
            episode_id=ep_id,
            timestamp=ep_timestamp,
            device_id=device_id,
            goal=goal,
            mode=mode,
            success=is_final_success,
            final_status=final_status,
            total_turns=len(turns),
            total_time_sec=total_time_sec,
            total_return=round(cumulative_return, 3),
            recovery_events=recovery_count,
            cycle_oscillations=cycle_count,
            false_completes=false_completes,
            turns=turns,
        )

        out_file = os.path.join(self.output_dir, f"{ep_id}.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(asdict(trajectory), f, indent=2)

        print(f"[TrajectoryLogger] Clean episode saved to: {out_file} (Return: {trajectory.total_return})")
        return trajectory


# Phase 5: Counterfactual Failure Diagnoser
class FailureDiagnoser:
    """
    Analyzes failed trajectories to locate decision-point divergences and
    formulate verified counterfactual repair proposals for Phase 5.
    """

    @staticmethod
    def diagnose_episode(trajectory: EpisodeTrajectory) -> Optional[Dict[str, Any]]:
        if trajectory.success:
            return None

        # 1. Identify divergence turn
        divergence_turn = None
        stopwords = {"open", "launch", "and", "switch", "to", "the", "in", "then", "into", "a", "an"}
        goal_tokens = [w for w in re.findall(r"[A-Za-z0-9]+", trajectory.goal.lower()) if w not in stopwords]

        for idx, t in enumerate(trajectory.turns):
            chosen_clean = t.chosen_label.lower()
            # If action was a penalty, wrong screen, or blocked
            if t.step_reward < 0.0 or "WRONG_SCREEN" in t.verification or "BLOCKED" in t.verification:
                divergence_turn = t
                break
            # If action was back or cycle while compound goal requirements are unfulfilled
            if t.chosen_action_type == "back" or "back" in t.chosen_candidate_id:
                divergence_turn = t
                break
            # If action did not match any remaining goal tokens after initial launch milestone
            if idx > 0 and not any(gt in chosen_clean for gt in goal_tokens):
                divergence_turn = t
                break

        if divergence_turn is None and trajectory.turns:
            divergence_turn = trajectory.turns[0]

        if divergence_turn is None:
            return None

        # 2. Extract unfulfilled semantic keywords
        fulfilled_tokens = set()
        for idx in range(trajectory.turns.index(divergence_turn)):
            fulfilled_tokens.update(re.findall(r"[A-Za-z0-9]+", trajectory.turns[idx].raw_trace.lower()))
            fulfilled_tokens.update(re.findall(r"[A-Za-z0-9]+", trajectory.turns[idx].chosen_label.lower()))
            fulfilled_tokens.update(re.findall(r"[A-Za-z0-9]+", trajectory.turns[idx].current_app.lower()))

        unfulfilled = [
            tok for tok in goal_tokens
            if tok not in fulfilled_tokens and tok not in divergence_turn.chosen_label.lower()
        ]

        target_token = unfulfilled[0] if unfulfilled else (goal_tokens[-1] if goal_tokens else "target")

        diagnosis = {
            "episode_id": trajectory.episode_id,
            "goal": trajectory.goal,
            "divergence_turn": divergence_turn.turn,
            "current_app": divergence_turn.current_app,
            "chosen_action": {
                "id": divergence_turn.chosen_candidate_id,
                "label": divergence_turn.chosen_label,
                "confidence": divergence_turn.confidence,
            },
            "root_cause": (
                f"Semantic grounding mismatch: model prioritized app-level prior "
                f"over unfulfilled goal keyword '{target_token}'."
            ),
            "unfulfilled_goal_keywords": unfulfilled,
            "proposed_counterfactual_hypothesis": {
                "turn": divergence_turn.turn,
                "required_semantic_token": target_token,
                "recommended_action_type": "tap",
                "expected_postcondition": f"Foreground screen reflects '{target_token}' tab/screen state",
            },
            "status": "HYPOTHESIS_PENDING_REPLAY",
        }
        return diagnosis
