"""
KARVISH V51.7 — LIVE GPU INFERENCE HOST SERVER (PORT 8765)
Hosts the Phase 6.7 Closed-Loop Recurrent Task-State & Goal-Coverage Model (KarvishModelV51_7)
directly on the NVIDIA GeForce RTX 4050 GPU (CUDA).

Connects physical Android device via reverse ADB port forwarding:
    adb -s RZCY51MQP9K reverse tcp:8765 tcp:8765

Endpoints:
- GET  /health   -> Health check for LiveRankerAgentModelEngine
- POST /propose  -> Real-time candidate action proposal using closed-loop calibrated joint decoding
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8")

import torch
import torch.nn.functional as F

REPO_ROOT = Path(".").resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools" / "training" / "karvish_mobile_action"))

from tools.training.karvish_mobile_action.train_mobile_action_ranker import (
    get_tokenizer,
)
from tools.training.karvish_mobile_action.v51_phase6.phase6_episodic_dataset import (
    compute_history_ledger,
    compute_observable_diff,
)
from tools.training.karvish_mobile_action.v51_phase6_6.causal_recurrence import (
    compute_causal_recurrence,
)
from tools.training.karvish_mobile_action.v51_phase6_7.decoder_v51_7 import (
    CAND_ACTION_NAMES,
    GLOBAL_ACTION_NAMES,
    decode_closed_loop_action,
)
from tools.training.karvish_mobile_action.v51_phase6_7.goal_coverage_v51_7 import (
    extract_active_goal_span,
)
from tools.training.karvish_mobile_action.v51_phase6_7.model_v51_7 import (
    KarvishModelV51_7,
)
from tools.training.karvish_mobile_action.v51_phase6_7.task_state_v51_7 import (
    compute_observable_ui_delta,
)

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
DEFAULT_PORT = 8765

CHECKPOINT_PATH = (
    REPO_ROOT
    / "tools"
    / "training"
    / "karvish_mobile_action"
    / "v51_phase6_7"
    / "checkpoints"
    / "v51_sequential_best.pt"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = None
MODEL: Optional[KarvishModelV51_7] = None
CHECKPOINT_SHA256 = ""

# Closed-loop episode state cache (purely causal across turns of a single goal)
CURRENT_GOAL = ""
H_TASK = None
C_CURR = None
PREV_ACTION_IDX = None
PREV_OBS = None
PREV_CANDS = None
STEP_IDX = 0
LAST_DISPATCHED_TOUCH_COORDS = (0, 0)
LAST_DISPATCHED_TARGET_LABEL = ""


def reset_episode_state(new_goal: str):
    global CURRENT_GOAL, H_TASK, C_CURR, PREV_ACTION_IDX, PREV_OBS, PREV_CANDS, STEP_IDX
    global LAST_DISPATCHED_TOUCH_COORDS, LAST_DISPATCHED_TARGET_LABEL
    CURRENT_GOAL = new_goal
    H_TASK = None
    C_CURR = None
    PREV_ACTION_IDX = None
    PREV_OBS = None
    PREV_CANDS = None
    STEP_IDX = 0
    LAST_DISPATCHED_TOUCH_COORDS = (0, 0)
    LAST_DISPATCHED_TARGET_LABEL = ""
    print(f">> [EPISODE RESET] Starting new closed-loop episode for goal: '{new_goal}'", flush=True)


def setup_adb_reverse(port: int = DEFAULT_PORT) -> bool:
    try:
        cmd = [ADB_PATH, "-s", DEVICE_SERIAL, "reverse", f"tcp:{port}", f"tcp:{port}"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if res.returncode == 0:
            print(f">> [ADB REVERSE] Port {port} forwarded to device {DEVICE_SERIAL} successfully!", flush=True)
            return True
        else:
            print(f">> [ADB REVERSE WARN] {res.stderr.strip()}", flush=True)
    except Exception as e:
        print(f">> [ADB REVERSE ERROR] {e}", flush=True)
    return False


SCREEN_WIDTH = 720.0
SCREEN_HEIGHT = 1600.0


def detect_screen_size():
    global SCREEN_WIDTH, SCREEN_HEIGHT
    try:
        cmd = [ADB_PATH, "-s", DEVICE_SERIAL, "shell", "wm", "size"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        m = re.search(r"(\d+)x(\d+)", res.stdout)
        if m:
            SCREEN_WIDTH = float(m.group(1))
            SCREEN_HEIGHT = float(m.group(2))
            print(f">> [DEVICE SCREEN] Detected screen dimensions: {int(SCREEN_WIDTH)}x{int(SCREEN_HEIGHT)}", flush=True)
    except Exception as e:
        print(f">> [DEVICE SCREEN WARN] Failed to get wm size: {e}", flush=True)


def load_model(ckpt_path: Path = CHECKPOINT_PATH):
    global MODEL, TOKENIZER, CHECKPOINT_SHA256
    print(f">> Loading Phase 6.7 Model on {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})...", flush=True)
    assert ckpt_path.exists(), f"Checkpoint missing: {ckpt_path}"

    with open(ckpt_path, "rb") as f:
        CHECKPOINT_SHA256 = hashlib.sha256(f.read()).hexdigest()

    TOKENIZER = get_tokenizer(8192)
    model = KarvishModelV51_7(belief_dim=16, history_dim=16).to(DEVICE)

    payload = torch.load(ckpt_path, map_location=DEVICE)
    state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload

    model.load_state_dict(state, strict=True)
    model.eval()
    MODEL = model
    param_count = sum(p.numel() for p in model.parameters())
    print(f">> Phase 6.7 Model Ready: {param_count:,} parameters ({len(state)} tensors). Checkpoint SHA256: {CHECKPOINT_SHA256}", flush=True)


def extract_app_name(goal: str) -> str:
    p_lower = goal.lower()
    known_apps = {
        "youtube": "YouTube",
        "linkedin": "LinkedIn",
        "instagram": "Instagram",
        "calculator": "Calculator",
        "settings": "Settings",
        "clock": "Clock",
        "contacts": "Contacts",
        "keep": "Keep",
        "whatsapp": "WhatsApp",
        "play store": "Play Store",
        "chrome": "Chrome",
    }
    for k, v in known_apps.items():
        if f"open {k}" in p_lower or f"launch {k}" in p_lower or f"in {k}" in p_lower or f"on {k}" in p_lower:
            return v
    for k, v in known_apps.items():
        if k in p_lower:
            return v
    m = re.search(r"(?i)\b(?:open|launch)\s+([A-Za-z0-9_\s]+)", goal)
    if m:
        return m.group(1).strip().split()[0].capitalize()
    return ""


class KarvishV51_7BridgeHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        if "GET /health" in args[0]:
            return
        print(f">> [HTTP] {self.address_string()} - {args[0]}", flush=True)

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            info = {
                "status": "ready",
                "version": "v51_7_phase6_7",
                "model": "KarvishModelV51_7",
                "device": f"{DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})",
                "checkpoint": CHECKPOINT_PATH.name,
                "sha256": CHECKPOINT_SHA256,
                "parameters": 10888974,
                "ready": True,
            }
            self.wfile.write(json.dumps(info).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        global CURRENT_GOAL, H_TASK, C_CURR, PREV_ACTION_IDX, PREV_OBS, PREV_CANDS, STEP_IDX
        global LAST_DISPATCHED_TOUCH_COORDS, LAST_DISPATCHED_TARGET_LABEL

        if self.path == "/propose":
            t0 = time.perf_counter()
            content_length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_length).decode("utf-8", errors="replace")

            try:
                payload = json.loads(raw_body)
                prompt = payload.get("prompt", "")
                tools = payload.get("tools", [])
                top_k = int(payload.get("top_k", 3))
                obs_dict = payload.get("observation", {})
                history = payload.get("history", [])

                clean_goal = prompt
                for marker in ["[history:", "(history:", "[previous failure:", "(previous action failed:"]:
                    if marker in clean_goal.lower():
                        clean_goal = clean_goal[: clean_goal.lower().find(marker)].strip()

                # Causal episode boundary check:
                # If the goal changed or history is empty (first turn), reset recurrent state
                if clean_goal != CURRENT_GOAL or len(history) == 0:
                    reset_episode_state(clean_goal)

                # Process candidates
                cands = []
                for idx, t in enumerate(tools[:48]):
                    name = t.get("name", f"cand_{idx}")
                    lbl = t.get("semantic_label") or t.get("label") or name
                    role = t.get("role", "button").lower()
                    act = t.get("action_type", "tap").lower()
                    bounds = t.get("bounds", [0.0, 0.0, 1.0, 1.0])
                    if isinstance(bounds, dict):
                        raw_top = float(bounds.get("top", 0.0))
                        raw_left = float(bounds.get("left", 0.0))
                        raw_bottom = float(bounds.get("bottom", 1.0))
                        raw_right = float(bounds.get("right", 1.0))
                        max_coord = max(raw_top, raw_bottom, raw_left, raw_right)
                        if max_coord > 1000.0:
                            div_h, div_w = SCREEN_HEIGHT, SCREEN_WIDTH
                        elif max_coord > 1.01:
                            div_h, div_w = 1000.0, 1000.0
                        else:
                            div_h, div_w = 1.0, 1.0
                        b_list = [
                            max(0.0, min(1.0, raw_top / div_h)),
                            max(0.0, min(1.0, raw_left / div_w)),
                            max(0.0, min(1.0, raw_bottom / div_h)),
                            max(0.0, min(1.0, raw_right / div_w)),
                        ]
                    elif isinstance(bounds, list) and len(bounds) == 4:
                        raw_b = [float(x) for x in bounds]
                        max_coord = max(raw_b)
                        if max_coord > 1000.0:
                            div_h, div_w = SCREEN_HEIGHT, SCREEN_WIDTH
                        elif max_coord > 1.01:
                            div_h, div_w = 1000.0, 1000.0
                        else:
                            div_h, div_w = 1.0, 1.0
                        b_list = [
                            max(0.0, min(1.0, raw_b[0] / div_h)),
                            max(0.0, min(1.0, raw_b[1] / div_w)),
                            max(0.0, min(1.0, raw_b[2] / div_h)),
                            max(0.0, min(1.0, raw_b[3] / div_w)),
                        ]
                    else:
                        b_list = [0.0, 0.0, 1.0, 1.0]

                    res_id_str = str(t.get("resource_id", "") or t.get("id", "") or "").lower()
                    is_formula_display = "formula" in res_id_str or "calc_edt" in res_id_str
                    is_clickable = bool(t.get("clickable", True)) and not is_formula_display
                    is_editable = bool(t.get("editable", False) or "input" in role or act == "type_text" or name.startswith("type_")) and not is_formula_display
                    is_scrollable = bool(t.get("scrollable", False) or act == "scroll" or name.startswith("scroll_") or name.startswith("c_scroll_"))
                    is_enabled = bool(t.get("enabled", True))

                    if is_editable:
                        role = "input"
                        act = "type_text"
                    elif is_scrollable:
                        act = "scroll"
                    elif is_formula_display:
                        role = "text"
                        act = "tap"

                    flg = [0.0] * 32
                    flg[0] = 1.0 if is_clickable else 0.0
                    flg[1] = 1.0 if is_editable else 0.0
                    flg[2] = 1.0 if is_scrollable else 0.0
                    flg[3] = 1.0 if is_enabled else 0.0
                    flg[8] = 1.0
                    flg[9] = 1.0

                    target_node_idx = t.get("target_node_index")
                    if target_node_idx is not None and not isinstance(target_node_idx, int):
                        try:
                            target_node_idx = int(target_node_idx)
                        except (ValueError, TypeError):
                            target_node_idx = None

                    cands.append({
                        "id": t.get("candidate_id") or name,
                        "label": lbl,
                        "role": role,
                        "action_type": act,
                        "bounds": b_list,
                        "flags": flg,
                        "target_node_index": target_node_idx,
                        "orig_tool": t,
                        "text": t.get("text") or t.get("semantic_label") or lbl,
                    })

                # Compute causal recurrence from history / previous dispatch
                last_touch_coords = (0, 0)
                last_target_label = ""
                if LAST_DISPATCHED_TOUCH_COORDS != (0, 0):
                    last_touch_coords = LAST_DISPATCHED_TOUCH_COORDS
                    last_target_label = LAST_DISPATCHED_TARGET_LABEL
                elif history:
                    last_step = history[-1]
                    last_target_label = last_step.get("action_description", "")
                    prev_target_id = last_step.get("target_id")

                    # Locate the previous target in PREV_CANDS to obtain true physical touch coordinates
                    if PREV_CANDS:
                        matched_cand = None
                        if prev_target_id is not None:
                            matched_cand = next((c for c in PREV_CANDS if c.get("target_node_index") == prev_target_id), None)
                        if matched_cand is None and last_target_label:
                            for c in PREV_CANDS:
                                if c.get("label") and c["label"].lower() in last_target_label.lower():
                                    matched_cand = c
                                    break
                        if matched_cand and matched_cand.get("bounds"):
                            b = matched_cand["bounds"]
                            cx = int(((b[1] + b[3]) / 2.0) * SCREEN_WIDTH)
                            cy = int(((b[0] + b[2]) / 2.0) * SCREEN_HEIGHT)
                            last_touch_coords = (cx, cy)

                rec_vals = compute_causal_recurrence(
                    cands,
                    last_touch_coords=last_touch_coords,
                    last_target_label=last_target_label,
                    screen_w=int(SCREEN_WIDTH),
                    screen_h=int(SCREEN_HEIGHT),
                )
                cand_flags_tensor = torch.zeros(1, max(1, len(cands)), 32, dtype=torch.float32, device=DEVICE)
                for ci, c in enumerate(cands):
                    c["flags"][22] = rec_vals[ci] if ci < len(rec_vals) else 0.0
                    cand_flags_tensor[0, ci] = torch.tensor(c["flags"], dtype=torch.float32, device=DEVICE)

                # Compute causal observable UI delta between PREV_OBS and obs_dict
                if PREV_OBS is not None:
                    delta_vec = compute_observable_ui_delta(
                        obs_prev=PREV_OBS,
                        obs_curr=obs_dict,
                        cands_prev=PREV_CANDS,
                        cands_curr=cands,
                        action_prev=PREV_ACTION_IDX,
                    ).unsqueeze(0).to(DEVICE)
                else:
                    delta_vec = torch.zeros(1, 16, dtype=torch.float32, device=DEVICE)

                # Build input tensors
                goal_tokens = TOKENIZER.encode(clean_goal)[:64]
                ctx_ids = torch.zeros(1, 64, dtype=torch.long, device=DEVICE)
                ctx_mask = torch.zeros(1, 64, dtype=torch.bool, device=DEVICE)
                if goal_tokens:
                    ctx_ids[0, : len(goal_tokens)] = torch.tensor(goal_tokens, dtype=torch.long, device=DEVICE)
                    ctx_mask[0, : len(goal_tokens)] = True

                N = max(1, len(cands))
                cand_actions = torch.zeros(1, N, dtype=torch.long, device=DEVICE)
                cand_roles = torch.zeros(1, N, dtype=torch.long, device=DEVICE)
                cand_regions = torch.zeros(1, N, dtype=torch.long, device=DEVICE)
                cand_spatial = torch.zeros(1, N, 8, dtype=torch.float32, device=DEVICE)
                cand_lbl_ids = torch.zeros(1, N, 16, dtype=torch.long, device=DEVICE)
                cand_lbl_mask = torch.zeros(1, N, 16, dtype=torch.bool, device=DEVICE)
                cand_pad_mask = torch.zeros(1, N, dtype=torch.float32, device=DEVICE)

                for ci, c in enumerate(cands):
                    cand_pad_mask[0, ci] = 1.0
                    lbl_str = str(c.get("label", "")).strip()
                    lbl_toks = TOKENIZER.encode(c["label"])[:16]

                    # Universal candidate token normalization:
                    # 1. Single digits: include continuation BPE subwords across all possible prefixes (0-9)
                    if len(lbl_str) == 1 and lbl_str.isdigit():
                        for lead in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                            for ct in TOKENIZER.encode(lead + lbl_str)[1:]:
                                if ct not in lbl_toks and len(lbl_toks) < 16:
                                    lbl_toks.append(ct)
                    # 2. Arithmetic operators: map equivalent symbols and verbal descriptions
                    lbl_lower = lbl_str.lower()
                    if any(k in lbl_lower for k in ["×", "x", "*", "multipl", "times"]):
                        for sym in ["*", "×"]:
                            sym_tok = TOKENIZER.encode(sym)[0]
                            if sym_tok not in lbl_toks and len(lbl_toks) < 16:
                                lbl_toks.append(sym_tok)
                    elif any(k in lbl_lower for k in ["−", "-", "minus", "subtract"]):
                        for sym in ["-", "−"]:
                            sym_tok = TOKENIZER.encode(sym)[0]
                            if sym_tok not in lbl_toks and len(lbl_toks) < 16:
                                lbl_toks.append(sym_tok)
                    elif any(k in lbl_lower for k in ["÷", "/", "divid"]):
                        for sym in ["/", "÷"]:
                            sym_tok = TOKENIZER.encode(sym)[0]
                            if sym_tok not in lbl_toks and len(lbl_toks) < 16:
                                lbl_toks.append(sym_tok)
                    elif any(k in lbl_lower for k in ["+", "plus", "add"]):
                        sym_tok = TOKENIZER.encode("+")[0]
                        if sym_tok not in lbl_toks and len(lbl_toks) < 16:
                            lbl_toks.append(sym_tok)
                    elif any(k in lbl_lower for k in ["=", "equal"]):
                        for sym in ["="]:
                            sym_tok = TOKENIZER.encode(sym)[0]
                            if sym_tok not in lbl_toks and len(lbl_toks) < 16:
                                lbl_toks.append(sym_tok)
                    elif any(k in lbl_lower for k in ["clear", "clr", "c"]):
                        for sym in ["clear", "c"]:
                            sym_tok = TOKENIZER.encode(sym)[0]
                            if sym_tok not in lbl_toks and len(lbl_toks) < 16:
                                lbl_toks.append(sym_tok)

                    if lbl_toks:
                        cand_lbl_ids[0, ci, : len(lbl_toks)] = torch.tensor(lbl_toks, dtype=torch.long, device=DEVICE)
                        cand_lbl_mask[0, ci, : len(lbl_toks)] = True

                    b = c["bounds"]
                    ymin, xmin, ymax, xmax = b[0], b[1], b[2], b[3]
                    cand_spatial[0, ci, :4] = torch.tensor([ymin, xmin, ymax, xmax], dtype=torch.float32, device=DEVICE)
                    cand_spatial[0, ci, 4:] = torch.tensor([(ymin + ymax) / 2, (xmin + xmax) / 2, ymax - ymin, xmax - xmin], dtype=torch.float32, device=DEVICE)

                    role_str = str(c["role"]).lower()
                    if "history" in str(c.get("id", "")).lower() or "formula" in str(c.get("id", "")).lower() or "display" in str(c.get("id", "")).lower():
                        cand_roles[0, ci] = 3
                    elif "button" in role_str:
                        cand_roles[0, ci] = 1
                    elif "input" in role_str or "edit" in role_str:
                        cand_roles[0, ci] = 2
                    elif "text" in role_str:
                        cand_roles[0, ci] = 3

                    act_str = str(c["action_type"]).lower()
                    if act_str == "type_text":
                        cand_actions[0, ci] = 1
                    elif act_str == "scroll":
                        cand_actions[0, ci] = 2
                    elif act_str == "submit":
                        cand_actions[0, ci] = 3
                    else:
                        cand_actions[0, ci] = 0

                prev_act_tensor = None
                if PREV_ACTION_IDX is not None:
                    prev_act_tensor = torch.tensor([PREV_ACTION_IDX], dtype=torch.long, device=DEVICE)

                if C_CURR is not None:
                    uncons = [TOKENIZER.decode([goal_tokens[ti]]) for ti in range(len(goal_tokens)) if C_CURR[0, ti] < 0.45]
                    print(f">> [TURN {STEP_IDX}] Unconsumed tokens before inference: {uncons}", flush=True)
                else:
                    print(f">> [TURN {STEP_IDX}] Initial turn (C_CURR is None)", flush=True)

                # Inference on GPU with Closed-Loop Recurrent State
                with torch.no_grad():
                    out = MODEL(
                        input_ids=ctx_ids,
                        ctx_mask=ctx_mask,
                        action_types=cand_actions,
                        roles=cand_roles,
                        regions=cand_regions,
                        spatial_bounds=cand_spatial,
                        flags=cand_flags_tensor,
                        label_token_ids=cand_lbl_ids,
                        label_mask=cand_lbl_mask,
                        cand_pad_mask=cand_pad_mask,
                        delta_feats=delta_vec,
                        prev_action_idx=prev_act_tensor,
                        task_state=H_TASK,
                        consumption_state=C_CURR,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                if "goal_attention" in out and len(goal_tokens) > 0:
                    alpha_debug = [(TOKENIZER.decode([goal_tokens[ti]]).strip(), round(out["goal_attention"][0, ti].item(), 3)) for ti in range(len(goal_tokens))]
                    print(f">> [ALPHA_T TURN {STEP_IDX}] " + ", ".join([f"{t}:{a}" for t, a in alpha_debug if a >= 0.02]), flush=True)

                # Closed-Loop Decoded Action via Calibrated Joint Engine
                decision, state_info = decode_closed_loop_action(
                    model_output=out,
                    candidates=cands,
                    input_ids=ctx_ids,
                    tokenizer=TOKENIZER,
                    goal_text=clean_goal,
                    step_idx=STEP_IDX,
                    max_steps=15,
                    prev_action_idx=PREV_ACTION_IDX,
                )

                # Update causal recurrent state
                H_TASK = out["task_state"]
                C_CURR = out["consumption_state"]
                PREV_OBS = obs_dict
                PREV_CANDS = cands
                STEP_IDX += 1

                dec_action_raw = decision["action_type"].lower()
                ranking_scores = out["unified_ranking_scores"][0].cpu().tolist()
                c_names = ["GLOBAL_c0"] + [c.get("label", "") or c.get("id", "") for c in cands]
                top5 = sorted(zip(c_names, ranking_scores), key=lambda x: x[1], reverse=True)[:5]
                print(f">> [TOP5 TURN {STEP_IDX}] " + ", ".join([f"'{n}': {s:.2f}" for n, s in top5]), flush=True)
                cand_act_logits = out["cand_action_logits"][0]

                # Attention-extracted argument for typing
                extracted_text = decision.get("text", "") or ""

                proposals = []
                cand_idx = decision.get("cand_idx")
                target_name = "GLOBAL"
                target_id = None
                chosen_args = {}

                if cand_idx is not None and 0 <= cand_idx < len(cands):
                    c_sel = cands[cand_idx]
                    target_name = c_sel["id"]
                    target_id = c_sel["target_node_index"]
                    orig_tool = c_sel.get("orig_tool", {})

                    if target_name in ["c_open_app", "open_app"] or orig_tool.get("name") == "open_app" or dec_action_raw == "open_app":
                        dec_action_raw = "open_app"
                        target_id = None
                        app_name = orig_tool.get("app_name") or extract_app_name(clean_goal)
                        if app_name:
                            chosen_args["app_name"] = app_name
                    elif target_name in ["c_scroll_down", "c_scroll", "scroll_down"]:
                        dec_action_raw = "scroll"
                        target_id = None
                        chosen_args["direction"] = "DOWN"
                    elif target_name in ["c_scroll_up", "scroll_up"]:
                        dec_action_raw = "scroll"
                        target_id = None
                        chosen_args["direction"] = "UP"
                    elif target_name in ["c_press_back", "press_back", "back"] or dec_action_raw == "back":
                        dec_action_raw = "back"
                        target_id = None
                    elif target_name in ["c_submit", "submit"] or dec_action_raw == "submit":
                        dec_action_raw = "submit"
                        target_id = None
                    elif target_name in ["c_complete_goal", "complete_goal"] or orig_tool.get("name") == "complete_goal" or dec_action_raw == "complete":
                        dec_action_raw = "complete"
                        target_id = None
                    elif target_name in ["c_fail_goal", "fail_goal"] or orig_tool.get("name") == "fail_goal" or dec_action_raw == "fail":
                        dec_action_raw = "fail"
                        target_id = None
                    elif dec_action_raw == "type_text" and extracted_text:
                        chosen_args["text"] = extracted_text

                    pass
                elif dec_action_raw == "open_app":
                    app_name = extract_app_name(clean_goal)
                    if app_name:
                        chosen_args["app_name"] = app_name
                elif dec_action_raw in ["complete", "c_complete_goal"]:
                    target_name = "c_complete_goal"
                    dec_action_raw = "complete"
                elif dec_action_raw in ["back", "c_press_back"]:
                    target_name = "c_press_back"
                    dec_action_raw = "back"
                elif dec_action_raw in ["submit", "c_submit"]:
                    target_name = "c_submit"
                    dec_action_raw = "submit"
                elif dec_action_raw == "type_text" and extracted_text:
                    chosen_args["text"] = extracted_text

                # Map action name to index for causal recurrence
                if dec_action_raw == "tap":
                    PREV_ACTION_IDX = 0
                elif dec_action_raw == "type_text":
                    PREV_ACTION_IDX = 1
                elif dec_action_raw == "scroll":
                    PREV_ACTION_IDX = 2
                elif dec_action_raw == "submit":
                    PREV_ACTION_IDX = 3
                elif dec_action_raw == "back":
                    PREV_ACTION_IDX = 4
                elif dec_action_raw == "wait":
                    PREV_ACTION_IDX = 5
                elif dec_action_raw == "open_app":
                    PREV_ACTION_IDX = 6
                elif dec_action_raw in ["complete", "fail"]:
                    PREV_ACTION_IDX = 7
                else:
                    PREV_ACTION_IDX = 0

                # Grounded sequential consumption progression
                if C_CURR is not None and len(goal_tokens) > 0:
                    matched_ti = -1
                    if cand_idx is not None and 0 <= cand_idx < len(cands):
                        c_lbl = str(cands[cand_idx].get("label", "") or cands[cand_idx].get("text", "")).strip().lower()
                        for ti, g_tok in enumerate(goal_tokens):
                            if C_CURR[0, ti] < 0.45:
                                t_str = TOKENIZER.decode([g_tok]).strip().lower()
                                if t_str and (t_str == c_lbl or t_str in c_lbl.split() or (len(t_str) >= 2 and t_str in c_lbl)):
                                    matched_ti = ti
                                    break
                    elif dec_action_raw == "open_app":
                        app_n = extract_app_name(clean_goal).lower()
                        for ti, g_tok in enumerate(goal_tokens):
                            t_str = TOKENIZER.decode([g_tok]).strip().lower()
                            if t_str and t_str in app_n:
                                matched_ti = max(matched_ti, ti)

                    if matched_ti >= 0:
                        C_CURR[0, matched_ti] = 1.0
                        if matched_ti > 0:
                            C_CURR[0, :matched_ti] = torch.maximum(C_CURR[0, :matched_ti], torch.tensor(0.5, device=DEVICE))

                # Track physical touch coordinates for causal recurrence on subsequent turn
                if dec_action_raw == "tap" and cand_idx is not None and 0 <= cand_idx < len(cands):
                    c_sel = cands[cand_idx]
                    b = c_sel.get("bounds")
                    if b and len(b) >= 4:
                        cx = int(((b[1] + b[3]) / 2.0) * SCREEN_WIDTH)
                        cy = int(((b[0] + b[2]) / 2.0) * SCREEN_HEIGHT)
                        LAST_DISPATCHED_TOUCH_COORDS = (cx, cy)
                        LAST_DISPATCHED_TARGET_LABEL = str(c_sel.get("label", "") or c_sel.get("text", "")).strip()
                else:
                    LAST_DISPATCHED_TOUCH_COORDS = (0, 0)
                    LAST_DISPATCHED_TARGET_LABEL = ""

                p1 = {
                    "tool_name": target_name,
                    "candidate_id": target_name,
                    "action_type": dec_action_raw,
                    "target_node_index": target_id,
                    "target_id": target_id,
                    "confidence": round(float(decision.get("confidence", 0.85)), 4),
                    "score": round(float(decision.get("confidence", 0.85)), 4),
                    "expected_outcome": f"Subgoal: {state_info.get('subgoal', 'UNKNOWN')} | Prog: {state_info.get('progress', 0.0):.2f}",
                    "arguments": chosen_args,
                }
                if "app_name" in chosen_args:
                    p1["app_name"] = chosen_args["app_name"]
                proposals.append(p1)

                # Alternative proposals for Top-K
                ranked_cand_indices = sorted(range(len(cands)), key=lambda i: ranking_scores[i + 1], reverse=True)
                for ci in ranked_cand_indices:
                    c_id = cands[ci]["id"]
                    if c_id == target_name:
                        continue
                    c_orig = cands[ci].get("orig_tool", {})
                    c_t_id = cands[ci]["target_node_index"]
                    c_args = {}

                    if c_id in ["c_open_app", "open_app"] or c_orig.get("name") == "open_app":
                        c_act = "open_app"
                        c_t_id = None
                        c_app = c_orig.get("app_name") or extract_app_name(clean_goal)
                        if c_app:
                            c_args["app_name"] = c_app
                    elif c_id in ["c_press_back", "press_back"]:
                        c_act = "back"
                        c_t_id = None
                    elif c_id in ["c_submit", "submit"]:
                        c_act = "submit"
                        c_t_id = None
                    elif c_id in ["c_scroll_down", "c_scroll", "scroll_down"]:
                        c_act = "scroll"
                        c_t_id = None
                        c_args["direction"] = "DOWN"
                    elif c_id in ["c_scroll_up", "scroll_up"]:
                        c_act = "scroll"
                        c_t_id = None
                        c_args["direction"] = "UP"
                    elif c_id in ["c_complete_goal", "complete_goal"] or c_orig.get("name") == "complete_goal":
                        c_act = "complete"
                        c_t_id = None
                    elif c_id in ["c_fail_goal", "fail_goal"] or c_orig.get("name") == "fail_goal":
                        c_act = "fail"
                        c_t_id = None
                    else:
                        c_act_idx = cand_act_logits[ci].argmax().item()
                        c_act = CAND_ACTION_NAMES[c_act_idx].lower()
                        if c_act == "type_text" and extracted_text:
                            c_args["text"] = extracted_text

                    p_alt = {
                        "tool_name": c_id,
                        "candidate_id": c_id,
                        "action_type": c_act,
                        "target_node_index": c_t_id,
                        "target_id": c_t_id,
                        "confidence": round(float(F.softmax(cand_act_logits[ci], dim=-1)[cand_act_logits[ci].argmax().item()].item()), 4),
                        "score": round(ranking_scores[ci + 1], 4),
                        "expected_outcome": f"Alternative: {c_act} on {cands[ci]['label']}",
                        "arguments": c_args,
                    }
                    if "app_name" in c_args:
                        p_alt["app_name"] = c_args["app_name"]
                    proposals.append(p_alt)
                    if len(proposals) >= top_k:
                        break

                latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                resp = {
                    "status": "success",
                    "model": "KarvishModelV51_7",
                    "decision_type": "closed_loop_recurrent",
                    "subgoal": state_info.get("subgoal", "UNKNOWN"),
                    "progress": round(state_info.get("progress", 0.0), 4),
                    "termination_confidence": round(state_info.get("termination_confidence", 0.0), 4),
                    "latency_ms": latency_ms,
                    "proposals": proposals,
                }
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps(resp).encode("utf-8"))
                print(
                    f">> [PROPOSE {latency_ms}ms] Action: {dec_action_raw} -> Target: {target_name} | "
                    f"Subgoal: {state_info.get('subgoal')} | Prog: {state_info.get('progress'):.2f} | "
                    f"Term: {state_info.get('termination_confidence'):.2f}",
                    flush=True,
                )

            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_response(500)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def run_server(port: int = DEFAULT_PORT, checkpoint_path: Optional[str] = None):
    print("=" * 80, flush=True)
    print("KARVISH V51.7 — CLOSED-LOOP GPU INFERENCE HOST SERVER", flush=True)
    print("=" * 80, flush=True)
    ckpt = Path(checkpoint_path) if checkpoint_path else CHECKPOINT_PATH
    detect_screen_size()
    load_model(ckpt)
    setup_adb_reverse(port)

    server_address = ("0.0.0.0", port)
    httpd = HTTPServer(server_address, KarvishV51_7BridgeHandler)
    print(f">> Server listening on http://0.0.0.0:{port} (accessible on phone at http://127.0.0.1:{port})", flush=True)
    print(f">> READY FOR MOBILE AGENT INTERACTION", flush=True)
    print("=" * 80, flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n>> Shutting down server...", flush=True)
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Karvish V51.7 GPU Host Server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to listen on")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--device-serial", type=str, default=None, help="Device serial")
    args = parser.parse_args()
    if args.device_serial:
        DEVICE_SERIAL = args.device_serial
    run_server(args.port, args.checkpoint)
