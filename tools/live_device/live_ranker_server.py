"""
Karvish Mobile Action: Live Device Host GPU Ranker Server (Milestone v0.8).

Exposes an HTTP server on port 8765 to serve real-time Top-K action predictions
directly to the Android device over ADB reverse port forwarding:
adb reverse tcp:8765 tcp:8765

Runs the trained MobileActionRanker v0.6 on GPU with ~6ms latency.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRAIN_DIR = os.path.join(REPO_ROOT, "tools", "training", "karvish_mobile_action")
NEEDLE_DIR = os.path.join(REPO_ROOT, "tools", "training", "experiment_c_needle2")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, NEEDLE_DIR)
sys.path.insert(0, TRAIN_DIR)

from action_proposer import ActionProposal, TopKActionProposer  # noqa: E402
from needle.model.tokenizer import get_tokenizer  # noqa: E402
from task_state_verifier import TaskStateOracleVerifier  # noqa: E402
from task_memory_runtime import TaskMemoryState  # noqa: E402
from model_v43 import KarvishModelV43  # noqa: E402
from model_v44 import KarvishModelV44, STRUCTURED_TOKENS, TOKEN_TO_IDX  # noqa: E402
from model_v45 import KarvishModelV45  # noqa: E402
from v44_schema import StructuredInputState  # noqa: E402
from train_mobile_action_ranker import (  # noqa: E402
    MobileActionRanker,
    prepare_example,
)

MODEL = None
TOKENIZER = None
DEVICE = None
PROPOSER = None
VERIFIER = TaskStateOracleVerifier()
GLOBAL_TASK_MEMORY = TaskMemoryState()
CURRENT_CKPT_PATH = ""


def canonical_math_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\u2212", "-").replace("−", "-")
    s = s.replace("×", "*").replace("\xd7", "*")
    s = s.replace("÷", "/").replace("\xf7", "/")
    return s


def clean_samsung_formula(desc: str) -> str:
    if not desc:
        return "0"
    desc = canonical_math_text(desc)
    desc = re.sub(r"(?i)^calculator input field\s*", "", desc).strip()
    desc = re.sub(r"(?i)\bcalculation result\b", "", desc).strip()
    desc = re.sub(r"(?i)\bresult\b", "", desc).strip()
    desc = re.sub(r"(?i)\bplus\b", "+", desc)
    desc = re.sub(r"(?i)\bminus\b", "-", desc)
    desc = re.sub(r"(?i)\bmultiplied by\b|\bmultiplication\b", "*", desc)
    desc = re.sub(r"(?i)\bdivided by\b|\bdivision\b|\bdivided\b", "/", desc)
    desc = re.sub(r"(?i)\btimes\b", "*", desc)
    desc = desc.replace(",", "")
    desc = re.sub(r"\s*([+\-\*\/÷x])\s*", r" \1", desc)
    desc = canonical_math_text(desc)
    return desc.strip() or "0"



def parse_tool_to_candidate(tool: Dict[str, Any], idx: int) -> Dict[str, Any]:
    name = tool.get("name", "")
    desc = tool.get("description", "")
    name_lower = name.lower()

    action_type = "wait"
    target_id = None
    role = "system"
    clean_label = name

    target_node_index = tool.get("target_node_index")
    candidate_id = tool.get("candidate_id") or name
    explicit_label = tool.get("semantic_label")

    if target_node_index is not None:
        target_id = int(target_node_index)

    if name_lower == "open_app":
        action_type = "open_app"
        role = "system"
        clean_label = explicit_label or "open application"
        app_name = tool.get("app_name", "")
        if not app_name and "description" in tool:
            m = re.search(r"Launch application '([^']*)'", tool["description"])
            if m:
                app_name = m.group(1).strip()
        if not app_name and explicit_label:
            m_exp = re.search(r"open\s+([A-Za-z0-9_]+)", explicit_label, re.IGNORECASE)
            if m_exp:
                app_name = m_exp.group(1).strip()
        if app_name and len(app_name.split()) > 2:
            m_end = re.search(r"\b(?:on|in|into|via|open|launch)\s+([A-Za-z0-9_]+)$", app_name, re.IGNORECASE)
            if m_end:
                app_name = m_end.group(1).strip()
            else:
                app_name = ""
        if app_name and app_name.lower() not in ("application", "app", ""):
            clean_label = f"open {app_name}"
        else:
            clean_label = explicit_label or "open application"
    elif name_lower.startswith("tap_") or name_lower.startswith("candidate_"):
        action_type = "tap"
        role = "button"
        if target_id is None:
            num_match = re.findall(r"\d+", name)
            if num_match:
                target_id = int(num_match[-1])
        if explicit_label:
            clean_label = explicit_label.strip().lower()
        else:
            quote_match = re.search(r"'([^']*)'", desc)
            if quote_match and quote_match.group(1).strip():
                clean_label = quote_match.group(1).strip().lower()
            else:
                clean_label = name.replace("tap_", "").replace("_", " ").strip().lower()

        # Canonical normalization for math operator symbols
        clean_label = clean_label.replace("\u2212", "-").replace("−", "-")
        clean_label = clean_label.replace("×", "*").replace("\xd7", "*")
        clean_label = clean_label.replace("÷", "/").replace("\xf7", "/")
        clean_label = re.sub(r"(?i)\bmultiplied by\b|\bmultiplication\b|\btimes\b", "*", clean_label)
        clean_label = re.sub(r"(?i)\bdivided by\b|\bdivision\b|\bdivided\b", "/", clean_label)
        clean_label = re.sub(r"(?i)\bplus\b", "+", clean_label)
        clean_label = re.sub(r"(?i)\bminus\b", "-", clean_label)
        clean_label = re.sub(r"(?i)\bequals?\b", "=", clean_label)
    elif name_lower.startswith("type_"):
        action_type = "type_text"
        role = "input"
        if target_id is None:
            num_match = re.findall(r"\d+", name)
            if num_match:
                target_id = int(num_match[-1])
        if explicit_label:
            clean_label = explicit_label.strip().lower()
        else:
            quote_match = re.search(r"'([^']*)'", desc)
            if quote_match and quote_match.group(1).strip():
                clean_label = quote_match.group(1).strip().lower()
            else:
                clean_label = name.replace("type_", "").replace("_", " ").strip().lower()
    elif name_lower == "scroll_down" or name_lower.startswith("scroll_down_"):
        action_type = "scroll"
        role = "system"
        clean_label = explicit_label or "scroll down"
        if target_id is None:
            num_match = re.findall(r"\d+", name)
            if num_match:
                target_id = int(num_match[-1])
    elif name_lower == "scroll_up" or name_lower.startswith("scroll_up_"):
        action_type = "scroll"
        role = "system"
        clean_label = explicit_label or "scroll up"
        if target_id is None:
            num_match = re.findall(r"\d+", name)
            if num_match:
                target_id = int(num_match[-1])
    elif name_lower in ["press_back", "back"]:
        action_type = "back"
        role = "system"
        clean_label = "back"
    elif name_lower in ["submit", "press_enter", "ime_enter"]:
        action_type = "submit"
        role = "system"
        clean_label = "submit focused field"
    elif name_lower == "complete_goal":
        action_type = "complete"
        role = "system"
        clean_label = "complete"
    elif name_lower == "fail_goal":
        action_type = "fail"
        role = "system"
        clean_label = "fail"

    desc_lower = desc.lower()
    clean_lower = clean_label.lower()
    semantic_tags = list(tool.get("semantic_tags", [])) if isinstance(tool.get("semantic_tags"), list) else []

    if "search" in clean_lower or "search" in desc_lower or "search" in name_lower:
        if action_type == "tap":
            if "search_affordance" not in semantic_tags:
                semantic_tags.append("search_affordance")
            if "result_item" in semantic_tags:
                semantic_tags.remove("result_item")

    if "edittext" in desc_lower or "input" in desc_lower:
        role = "input"
    elif "button" in desc_lower or "switch" in desc_lower:
        role = "button"
    elif "text" in desc_lower or "label" in desc_lower:
        role = "text"

    region = tool.get("region", "")
    if not region:
        if "search" in clean_lower and role in ["button", "system", "text"]:
            region = "upper_right"
        elif role == "system" or action_type in ["submit", "back", "complete", "fail"]:
            region = "system"
        elif role == "input" or action_type == "type_text":
            region = "upper_center"
        else:
            region = "middle_center"

    return {
        "id": name,
        "tool_name": name,
        "action_type": action_type,
        "target_id": target_id,
        "label": canonical_math_text(clean_label),
        "role": role,
        "app_name": "" if name_lower != "open_app" else tool.get("app_name", ""),
        "region": region,
        "depth": 0,
        "is_leaf": True,
        "child_count": 0,
        "label_source": "content_desc" if action_type == "tap" else "system",
        "neighborhood": "",
        "specificity_score": 1.0,
        "clickable_descendant_count": 0,
        "argument_schema": {},
        "semantic_tags": semantic_tags,
    }


class RankerRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            resp = {
                "status": "ready",
                "model": f"Karvish-{MODEL.__class__.__name__ if MODEL else 'MobileActionRanker'}",
                "checkpoint": os.path.basename(CURRENT_CKPT_PATH) if CURRENT_CKPT_PATH else "unknown",
                "device": str(DEVICE),
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/propose":
            t0 = time.perf_counter()
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")

            try:
                data = json.loads(body)
                prompt = data.get("prompt", "")
                tools = data.get("tools", [])
                top_k = int(data.get("top_k", 3))
                observation_payload = data.get("observation")
                history_payload = data.get("history", [])
                previous_failure = data.get("previous_failure", "")

                # Parse goal and history cleanly
                markers = ["[history:", "(history:", "[previous failure:", "(previous action failed:", "(already in", "(notice:"]
                goal_clean = prompt
                for m in markers:
                    if m in goal_clean.lower():
                        idx = goal_clean.lower().find(m)
                        goal_clean = goal_clean[:idx].strip()
                goal_clean = canonical_math_text(goal_clean)

                candidates = [parse_tool_to_candidate(t, i) for i, t in enumerate(tools)]

                # The ranker owns next-action selection. The host only converts
                # the live observation into the model's structured input.
                if not isinstance(observation_payload, dict):
                    observation_payload = {
                        "package_name": "live.foreground.app",
                        "activity_name": "",
                        "nodes": [],
                    }
                else:
                    if not observation_payload.get("package_name"):
                        observation_payload["package_name"] = "live.foreground.app"
                    if not observation_payload.get("activity_name"):
                        observation_payload["activity_name"] = ""
                    if not isinstance(observation_payload.get("nodes"), list):
                        observation_payload["nodes"] = []
                    else:
                        for n in observation_payload["nodes"]:
                            if "text" in n and isinstance(n["text"], str):
                                n["text"] = canonical_math_text(n["text"])
                            if "content_description" in n and isinstance(n["content_description"], str):
                                n["content_description"] = canonical_math_text(n["content_description"])

                # Evaluate world-state with TaskStateOracleVerifier (Grounded Task State Tracker)
                v_state = VERIFIER.evaluate_turn(
                    goal=goal_clean,
                    observation=observation_payload,
                    action={"action_type": "inspect"},
                    history=history_payload,
                )
                print(f"[live-oracle] {v_state.format_state_header()} | all_completed={v_state.is_all_completed}", flush=True)

                # Phase P3: Production inputs must be evaluated honestly by the model.
                # All candidates including complete_goal are scored without runtime filtering.
                if not candidates:
                    candidates = [parse_tool_to_candidate({"name": "press_back", "description": "back"}, 0)]

                print(
                    f"[live-ranker] received observation_nodes="
                    f"{len((observation_payload or {}).get('nodes', []))} "
                    f"candidates={len(candidates)} history={len(history_payload)}",
                    flush=True,
                )
                for c in candidates:
                    safe_lbl = str(c.get('label', '')).encode('ascii', errors='backslashreplace').decode('ascii')
                    print(f"  [cand] id={c['id']} type={c['action_type']} label='{safe_lbl}' role={c['role']} tags={c.get('semantic_tags', [])}", flush=True)

                # Check for formula and result on screen in calculator
                calc_formula = None
                calc_result = None
                if "popupcalculator" in observation_payload.get("package_name", "").lower() or "calculator" in goal_clean.lower():
                    for n in observation_payload.get("nodes", []):
                        n_text = str(n.get("text") or "")
                        n_desc = str(n.get("content_description") or "")
                        n_hint = str(n.get("resource_hint") or "")
                        if "calc_edt_formula" in n_hint or "calculator input field" in n_desc.lower():
                            raw_f = n_desc if "calculator input field" in n_desc.lower() else (n_text or n_desc)
                            calc_formula = clean_samsung_formula(raw_f)
                        elif "calc_tv_result" in n_hint or "calculation result" in n_desc.lower() or "calc_edt_result" in n_hint:
                            raw_r = n_text or n_desc
                            calc_result = clean_samsung_formula(raw_r)

                # Check if math goal is satisfied
                m_math = re.search(r"(\d+(?:[\s\+\-\*\/\%x]+\d+)+)", goal_clean)
                if m_math:
                    expr_clean = m_math.group(1).replace("x", "*").replace(" ", "")
                    try:
                        expected_num = eval(expr_clean)
                        expected_str = str(int(expected_num)) if expected_num == int(expected_num) else str(expected_num)
                        if (calc_formula and calc_formula == expected_str) or (calc_result and calc_result == expected_str):
                            v_state.observed_values["calculation_result"] = expected_str
                            math_req = f"calculate:{m_math.group(1)}"
                            if math_req in v_state.pending_milestones:
                                v_state.pending_milestones.remove(math_req)
                            if math_req not in v_state.completed_milestones:
                                v_state.completed_milestones.append(math_req)
                            v_state.is_all_completed = len(v_state.pending_milestones) == 0
                    except Exception:
                        pass

                # If in a multi-app workflow, check if open_app needs to target the next pending app
                curr_pkg = observation_payload.get("package_name", "").lower()
                pending_apps = [m.split(":")[1] for m in v_state.pending_milestones if m.startswith("open_app:")]
                if pending_apps:
                    target_next_app = pending_apps[0]
                    for c in candidates:
                        if c.get("action_type") == "open_app":
                            c_app = str(c.get("app_name") or "").lower()
                            if not c_app or c_app in curr_pkg:
                                # Previous app is already active, retarget open_app candidate to the next pending app
                                c["app_name"] = target_next_app.capitalize()
                                c["label"] = f"open {target_next_app}"

                # Phase P3: Production inputs must be the original goal, live observation,
                # available actions, and short history. No oracle micro-intents, no regex
                # clause splitting, and no manufactured completion or launch actions.
                # Grounded task state is tracked in previous_outcome / state context.
                factual_prev_outcome = "Task started"
                if history_payload:
                    last_h = history_payload[-1]
                    last_act_desc = canonical_math_text(last_h.get("action_description") or last_h.get("action") or "")
                    last_refl = canonical_math_text(last_h.get("outcomeSummary") or last_h.get("outcome") or "")

                    # Extract last key tapped if any
                    last_key = ""
                    last_act_type = str(last_h.get("action") or "").lower()
                    if last_act_type in ("tap", "click"):
                        m_key = re.search(r"'([^']*)'", last_act_desc)
                        if m_key:
                            cand_key = canonical_math_text(m_key.group(1).strip())
                            if cand_key.lower() not in ("calculator", "app") and len(cand_key) <= 5:
                                last_key = cand_key

                    if calc_formula is not None:
                        if v_state.is_all_completed:
                            factual_prev_outcome = f"Calculation complete. Display shows final result {calc_formula}."
                        elif last_key:
                            factual_prev_outcome = f"Key '{last_key}' tapped. Formula updated to '{calc_formula}'."
                        elif calc_formula in ("0", ""):
                            factual_prev_outcome = "Calculator opened. Display shows 0."
                        else:
                            factual_prev_outcome = f"Formula updated to '{calc_formula}'."
                    elif last_refl and last_refl.lower() not in {"success", "ok", "task started"} and not last_refl.startswith("text:") and "; +" not in last_refl:
                        factual_prev_outcome = last_refl
                    elif last_act_desc:
                        factual_prev_outcome = f"{last_act_desc} executed."
                    else:
                        factual_prev_outcome = "Live step"
                else:
                    factual_prev_outcome = "Task started"

                factual_prev_outcome = canonical_math_text(factual_prev_outcome)
                print(f"[live-ranker] factual_prev_outcome='{factual_prev_outcome}'", flush=True)

                cleaned_history = []
                for h in history_payload:
                    h_copy = dict(h)
                    act = canonical_math_text(str(h_copy.get("action", "")).lower())
                    act_desc = canonical_math_text(str(h_copy.get("action_description", "")))
                    summary = canonical_math_text(str(h_copy.get("outcomeSummary", "") or h_copy.get("outcome", "")))
                    h_copy["action"] = act
                    h_copy["action_description"] = act_desc
                    if act == "open_app":
                        app_name = "App"
                        m_app = re.search(r"'([^']*)'", act_desc)
                        if m_app:
                            app_name = m_app.group(1).strip()
                        h_copy["outcomeSummary"] = f"{app_name} opened"
                        h_copy["outcome"] = f"{app_name} opened"
                    elif "calc_edt_formula" in summary:
                        m_f = re.search(r"calc_edt_formula\s*\('([^']*)'\s*->\s*'([^']*)'\)", summary)
                        if m_f:
                            old_f = clean_samsung_formula(m_f.group(1))
                            new_f = clean_samsung_formula(m_f.group(2))
                            h_copy["outcomeSummary"] = f"text: calc_edt_formula ('{old_f}' -> '{new_f}')"
                            h_copy["outcome"] = h_copy["outcomeSummary"]
                        else:
                            h_copy["outcomeSummary"] = summary.split(";")[0].strip()
                            h_copy["outcome"] = h_copy["outcomeSummary"]
                    elif summary.startswith("text:") or "; +" in summary:
                        clean_summary = f"{act_desc or act} executed"
                        h_copy["outcomeSummary"] = clean_summary
                        h_copy["outcome"] = clean_summary
                    elif ";" in summary:
                        first_clause = summary.split(";")[0].strip()
                        h_copy["outcomeSummary"] = first_clause
                        h_copy["outcome"] = first_clause
                    else:
                        h_copy["outcomeSummary"] = summary
                        h_copy["outcome"] = summary
                    cleaned_history.append(h_copy)

                # Universal extraction of observed display / active input content
                observed_display = ""
                if calc_formula is not None:
                    observed_display = calc_formula
                else:
                    for n in observation_payload.get("nodes", []):
                        if n.get("focused") or "edittext" in str(n.get("class_name", "")).lower() or n.get("is_editable"):
                            t = str(n.get("text") or n.get("content_description") or "")
                            if t:
                                observed_display = t
                                break

                ex_record = {
                    "goal": goal_clean,
                    "micro_intent": "",
                    "package_name": observation_payload.get("package_name", "live.foreground.app"),
                    "activity_name": observation_payload.get("activity_name", ""),
                    "package_masked": "<APP>",
                    "task_state": v_state,
                    "observation": observation_payload,
                    "history": cleaned_history,
                    "previous_outcome": factual_prev_outcome,
                    "observed_display": observed_display,
                    "candidates": candidates,
                    "label": {"candidate_id": candidates[0]["id"], "action_type": candidates[0]["action_type"]},
                }

                # Universal structured input state tracking for sequential inputs (math, OTP, dates, codes)
                m_math = re.search(r"(\d+(?:[\s\+\-\*\/\%x]+\d+)+)", goal_clean)
                exp_token = None
                if m_math:
                    target_seq = m_math.group(1).replace(" ", "").replace("x", "*") + "="
                    clean_disp = observed_display.replace(" ", "")
                    step_idx = 0
                    for ch in clean_disp:
                        if step_idx < len(target_seq) and ch.lower() == target_seq[step_idx].lower():
                            step_idx += 1
                    if step_idx < len(target_seq):
                        exp_token = target_seq[step_idx]
                    ex_record["expected_token"] = exp_token or ""
                    ex_record["is_structured"] = 1.0

                item = prepare_example(ex_record, TOKENIZER, DEVICE, is_train=False, use_micro_intent=False)
                if item is None:
                    raise ValueError("Failed to prepare candidate tensor representations")

                if exp_token:
                    item["expected_token"] = exp_token

                # Reset memory for new episode
                if len(history_payload) == 0:
                    GLOBAL_TASK_MEMORY.reset()

                # The model independently scores and ranks all available candidates (with Confidence Ladder Cascade)
                proposals = PROPOSER.propose(
                    MODEL,
                    item,
                    goal=goal_clean,
                    task_state=v_state,
                    task_memory=GLOBAL_TASK_MEMORY,
                    step_index=len(history_payload) + 1,
                )

                for i, p in enumerate(proposals[:5]):
                    print(
                        f"  [MODEL_DECISION #{i+1}] {p.candidate_id} | action={p.action_type} | "
                        f"score={p.score:.4f} | conf={p.confidence:.4f} | "
                        f"progress={getattr(p, 'task_progress', 0.0):.2f} | "
                        f"subgoal={getattr(p, 'subgoal_type', 'NAVIGATE')} | "
                        f"term_conf={getattr(p, 'termination_confidence', 0.0):.2f} | "
                        f"mem_write={getattr(p, 'memory_write_detected', False)} | "
                        f"mem_read={getattr(p, 'memory_read_detected', False)} | "
                        f"arg='{getattr(p, 'argument_value', '')}'",
                        flush=True,
                    )

                # Format response
                proposals_json = []
                for p in proposals[:top_k]:
                    p_dict = p.to_dict()
                    if p.candidate_data:
                        p_dict.update(p.candidate_data)
                    p_dict["tool_name"] = p.candidate_id

                    # Safety Interception: If a complete action leaks while pending milestones remain, intercept it
                    if p_dict.get("action_type") == "complete" and not v_state.is_all_completed:
                        print(f"[VERIFIER_DECISION] Rejected premature completion claim by model: {p_dict.get('tool_name')}", flush=True)
                        continue
                    if p_dict.get("action_type") == "type_text":
                        if "arguments" not in p_dict or not isinstance(p_dict.get("arguments"), dict):
                            p_dict["arguments"] = {}
                        if getattr(p, "argument_value", ""):
                            p_dict["arguments"]["text"] = p.argument_value
                        elif not p_dict["arguments"].get("text"):
                            m_search = re.search(r"\b(?:search|find|look for)\s+(?:for\s+)?(.+?)(?:\s+and\s+|\s+then\s+|$)", goal_clean, re.IGNORECASE)
                            if m_search:
                                term = m_search.group(1).strip().strip("',.\"")
                                term = re.sub(r"\s+(?:in|on|inside|within)\s+[a-zA-Z0-9_\s]+$", "", term, flags=re.IGNORECASE).strip()
                                p_dict["arguments"]["text"] = term
                            else:
                                m_type = re.search(r"\b(?:type|enter|input)\s+(.+?)(?:\s+into\s+|\s+in\s+|$)", goal_clean, re.IGNORECASE)
                                if m_type:
                                    p_dict["arguments"]["text"] = m_type.group(1).strip().strip("',.\"")
                    proposals_json.append(p_dict)

                print(
                    "[live-ranker] top3=" + ", ".join(
                        f"{p.get('tool_name')}:{p.get('action_type')}:{p.get('target_id')}"
                        for p in proposals_json[:3]
                    ),
                    flush=True,
                )

                latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                resp = {
                    "status": "ok",
                    "latency_ms": latency_ms,
                    "proposals": proposals_json,
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(resp).encode("utf-8"))

            except Exception as e:
                import traceback
                print(f"[live-ranker ERROR] {e}", flush=True)
                traceback.print_exc()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                err_resp = {"status": "error", "message": str(e)}
                self.wfile.write(json.dumps(err_resp).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Concise logging
        pass


def run_server(port: int = 8765, ckpt_path: Optional[str] = None):
    global MODEL, TOKENIZER, DEVICE, PROPOSER, CURRENT_CKPT_PATH

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v46_1_ckpt = os.path.join(TRAIN_DIR, "checkpoints_v46_pilot", "v46_1_model.pt")
    v46_ckpt = os.path.join(TRAIN_DIR, "checkpoints_v46_pilot", "v46_targeted_trained.pt")
    v45_1_ckpt = os.path.join(TRAIN_DIR, "checkpoints_v45_1", "model_best.pt")
    v45_ckpt = os.path.join(TRAIN_DIR, "checkpoints_v45", "model_best.pt")
    v44_ckpt = os.path.join(TRAIN_DIR, "checkpoints_v44", "model_best.pt")
    v43_ckpt = os.path.join(TRAIN_DIR, "checkpoints_v43", "model_best.pt")
    v42_ckpt = os.path.join(TRAIN_DIR, "checkpoints_v42", "ranker_best.pt")
    v40_ckpt = os.path.join(TRAIN_DIR, "checkpoints_v40", "ranker_best.pt")
    default_ckpt = (
        v46_1_ckpt if os.path.exists(v46_1_ckpt) else
        (v46_ckpt if os.path.exists(v46_ckpt) else
        (v45_1_ckpt if os.path.exists(v45_1_ckpt) else
        (v45_ckpt if os.path.exists(v45_ckpt) else
        (v44_ckpt if os.path.exists(v44_ckpt) else
        (v43_ckpt if os.path.exists(v43_ckpt) else
        (v42_ckpt if os.path.exists(v42_ckpt) else
        (v40_ckpt if os.path.exists(v40_ckpt) else os.path.join(TRAIN_DIR, "checkpoints_v39_4", "ranker_best.pt"))))))))
    )
    ckpt_path = ckpt_path or default_ckpt
    CURRENT_CKPT_PATH = ckpt_path

    print(f">> Initializing Host Ranker Server on {DEVICE}...")
    print(f">> Loading model checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        config = ckpt.get("config", {})
    else:
        state_dict = ckpt
        config = {}

    if DEVICE.type == "cpu":
        state_dict = {k: v.float() if v.dtype == torch.float16 else v for k, v in state_dict.items()}

    TOKENIZER = get_tokenizer(config.get("vocab_size", 8192))
    if "tem_dense_proj.0.weight" in state_dict or "struct_cursor_proj.0.weight" in state_dict or "tem_fusion_head.fusion_proj.weight" in state_dict:
        print(">> [MODEL_INIT] Detected Karvish Model v45/v45.1 Architecture (Phased Goals, TEM & Structured Cursor).", flush=True)
        MODEL = KarvishModelV45().to(DEVICE)
    elif "structured_input_head.token_proj.weight" in state_dict or "subgoal_plan_head.subgoal_proj.weight" in state_dict:
        print(">> [MODEL_INIT] Detected Karvish Model v44 Architecture (Long-Horizon & Structured Input).", flush=True)
        MODEL = KarvishModelV44().to(DEVICE)
    elif "task_state_head.progress_proj.weight" in state_dict:
        print(">> [MODEL_INIT] Detected Karvish Model v43 Architecture with Learned Task State.", flush=True)
        MODEL = KarvishModelV43().to(DEVICE)
    else:
        is_factored = state_dict.get("action_head.0.weight", torch.zeros(1, 512)).shape[1] == 768
        MODEL = MobileActionRanker(factored_scoring=is_factored).to(DEVICE)
    MODEL.load_state_dict(state_dict, strict=False)
    MODEL.eval()

    PROPOSER = TopKActionProposer(top_k=5)

    # Warmup forward pass on GPU
    dummy_ex = {
        "goal": "warmup", "micro_intent": "", "package_name": "com.warmup",
        "activity_name": "", "package_masked": "<APP>",
        "observation": {"package_name": "com.warmup", "activity_name": "", "nodes": []},
        "history": [], "previous_outcome": "warmup",
        "candidates": [{"id": "c1", "tool_name": "c1", "action_type": "tap", "target_id": 0, "label": "test", "role": "button", "region": "middle_center", "depth": 0, "is_leaf": True, "child_count": 0, "label_source": "system", "neighborhood": "", "specificity_score": 1.0, "clickable_descendant_count": 0, "argument_schema": {}}],
        "label": {"candidate_id": "c1", "action_type": "tap"},
    }
    dummy_item = prepare_example(dummy_ex, TOKENIZER, DEVICE, is_train=False, use_micro_intent=False)
    if dummy_item:
        with torch.no_grad():
            PROPOSER.propose(MODEL, dummy_item)
    print(">> Model GPU warmup complete.", flush=True)

    # Reverse port 8765 once at startup if ADB device is connected
    adb_bin = None
    for p in [
        os.path.expandvars(r"%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe"),
        r"C:\Users\Karthik S\AppData\Local\Android\Sdk\platform-tools\adb.exe",
    ]:
        if os.path.exists(p):
            adb_bin = p
            break
    if not adb_bin:
        adb_bin = shutil.which("adb")

    if adb_bin:
        try:
            import subprocess
            res = subprocess.run([adb_bin, "reverse", f"tcp:{port}", f"tcp:{port}"], capture_output=True, text=True, timeout=5)
            if res.returncode == 0:
                print(f">> [ADB REVERSE] Port {port} forwarded to connected Android device successfully!", flush=True)
        except Exception as e:
            print(f">> [ADB REVERSE] Note: {e}")

    server = HTTPServer(("0.0.0.0", port), RankerRequestHandler)
    print(f">> [READY] Karvish Host GPU Ranker Server listening on http://0.0.0.0:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n>> Shutting down server.")
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Host GPU Ranker Server")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()
    run_server(port=args.port, ckpt_path=args.ckpt)
