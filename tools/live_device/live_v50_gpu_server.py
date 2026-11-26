"""
KARVISH V50 — LIVE GPU INFERENCE HOST SERVER

Hosts KarvishModelV50 directly in PyTorch on NVIDIA GPU (CUDA).
Serves real-time action proposals to connected physical Android devices over ADB reverse port forwarding (port 8765).

Features:
- Pure PyTorch on CUDA (No ONNX conversion required)
- Sub-5ms GPU tensor inference
- Dynamic UI tree serialization & V50 tensor encoding
- 7 Decision Modes, 14 Universal Actions, 13 Postcondition Transitions, Progress & Recovery
- Automatic ADB reverse port forwarding (tcp:8765 -> tcp:8765)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
TRAIN_DIR = REPO_ROOT / "tools" / "training" / "karvish_mobile_action"
NEEDLE_DIR = REPO_ROOT / "tools" / "training" / "experiment_c_needle2"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(TRAIN_DIR))
sys.path.insert(0, str(NEEDLE_DIR))

from needle.model.tokenizer import get_tokenizer
from universal_schema_v50 import (
    UniversalSemanticRole,
    DecisionMode,
    UniversalActionType,
    ExpectedTransition,
    ProgressStatus,
    RecoveryClass,
    CandidateSignature,
)
from ui_relation_graph import LocalUiRelationGraph
from task_belief_state import TaskBeliefStateEngine
from semantic_memory import SemanticMemoryEngine
from model_v50 import KarvishModelV50

RELATION_GRAPH = LocalUiRelationGraph()
BELIEF_ENGINE = TaskBeliefStateEngine()
MEMORY_ENGINE = SemanticMemoryEngine()

ADB_PATH = os.path.expandvars(r"%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe")
DEVICE_SERIAL = "RZCY51MQP9K"

MODEL: Optional[KarvishModelV50] = None
TOKENIZER = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = TRAIN_DIR / "checkpoints_v50" / "v50_model.pt"


def setup_adb_reverse(port: int = 8765) -> bool:
    adb_bins = [
        ADB_PATH,
        "adb",
        r"C:\Users\Karthik S\AppData\Local\Android\Sdk\platform-tools\adb.exe",
    ]
    for bin_path in adb_bins:
        try:
            res = subprocess.run(
                [bin_path, "-s", DEVICE_SERIAL, "reverse", f"tcp:{port}", f"tcp:{port}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode == 0:
                print(f">> [ADB REVERSE] Port {port} forwarded to device {DEVICE_SERIAL} successfully!")
                return True
        except Exception:
            continue
    print(f">> [WARN] Could not automatically forward port {port} via ADB. Device may need manual adb reverse.")
    return False


def load_v50_model(ckpt_path: Path = CKPT_PATH) -> KarvishModelV50:
    global MODEL, TOKENIZER, DEVICE

    print(f">> Initializing Karvish V50 on {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})...")
    print(f">> Checkpoint: {ckpt_path}")

    TOKENIZER = get_tokenizer(8192)
    model = KarvishModelV50()

    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    # GPU Warmup
    B, C = 1, 24
    with torch.no_grad():
        dummy_g = torch.randint(1, 1000, (B, 64), device=DEVICE, dtype=torch.long)
        dummy_c = torch.randint(1, 1000, (B, C, 16), device=DEVICE, dtype=torch.long)
        dummy_r = torch.zeros(B, C, device=DEVICE, dtype=torch.long)
        dummy_b = torch.zeros(B, C, 4, device=DEVICE, dtype=torch.float32)
        dummy_fl = torch.zeros(B, C, 32, device=DEVICE, dtype=torch.float32)
        dummy_rel = torch.zeros(B, C, 16, device=DEVICE, dtype=torch.float32)
        dummy_bf = torch.zeros(B, 8, device=DEVICE, dtype=torch.float32)
        dummy_mf = torch.zeros(B, 8, device=DEVICE, dtype=torch.float32)
        dummy_hf = torch.zeros(B, 8, device=DEVICE, dtype=torch.float32)
        dummy_m = torch.ones(B, C, device=DEVICE, dtype=torch.float32)

        t0 = time.perf_counter()
        model(dummy_g, dummy_c, dummy_r, dummy_b, dummy_fl, dummy_rel, dummy_bf, dummy_mf, dummy_hf, dummy_m)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        warmup_ms = (time.perf_counter() - t0) * 1000.0
        print(f">> Warmup GPU pass complete in {warmup_ms:.2f} ms")

    MODEL = model
    return model


def parse_tool_to_candidate_sig(t: Dict[str, Any], idx: int) -> CandidateSignature:
    name = t.get("name", f"tool_{idx}")
    desc = t.get("description", "")
    explicit_label = t.get("semantic_label") or t.get("label")

    clean_label = explicit_label or ""
    if not clean_label:
        q_match = re.search(r"'([^']*)'", desc)
        if q_match:
            clean_label = q_match.group(1).strip()
        else:
            clean_label = name.replace("tap_", "").replace("type_", "").replace("_", " ")

    desc_lower = desc.lower()
    name_lower = name.lower()

    if name_lower == "open_app" or name_lower.startswith("open_"):
        app_name = t.get("app_name", "")
        if not app_name:
            m = re.search(r"Launch application '([^']*)'", desc)
            if m:
                app_name = m.group(1).strip()
        if not app_name and explicit_label:
            m_exp = re.search(r"open\s+([A-Za-z0-9_]+)", explicit_label, re.IGNORECASE)
            if m_exp:
                app_name = m_exp.group(1).strip()
        clean_label = f"open {app_name}" if app_name else (explicit_label or "open application")
        role = UniversalSemanticRole.BUTTON
        is_editable = False
        is_clickable = True
    elif "edittext" in desc_lower or "input" in desc_lower or name_lower.startswith("type_"):
        role = UniversalSemanticRole.TEXT_FIELD
        is_editable = True
        is_clickable = True
    elif "search" in clean_label.lower() and ("input" in desc_lower or "field" in desc_lower):
        role = UniversalSemanticRole.SEARCH_FIELD
        is_editable = True
        is_clickable = True
    elif "button" in desc_lower or name_lower.startswith("tap_") or name_lower.startswith("btn_"):
        role = UniversalSemanticRole.BUTTON
        is_editable = False
        is_clickable = True
    elif "switch" in desc_lower or "toggle" in desc_lower:
        role = UniversalSemanticRole.SWITCH
        is_editable = False
        is_clickable = True
    elif "tab" in desc_lower:
        role = UniversalSemanticRole.TAB
        is_editable = False
        is_clickable = True
    elif "scroll" in name_lower:
        role = UniversalSemanticRole.SCROLL_CONTAINER
        is_editable = False
        is_clickable = True
    elif name_lower in ("submit", "press_enter", "ime_enter"):
        role = UniversalSemanticRole.BUTTON
        is_editable = False
        is_clickable = True
    elif name_lower in ("press_back", "back"):
        role = UniversalSemanticRole.NAVIGATION
        is_editable = False
        is_clickable = True
    else:
        role = UniversalSemanticRole.from_str(t.get("role", "button"))
        is_editable = False
        is_clickable = True

    bounds = t.get("bounds")
    if not bounds:
        desc_lower = desc.lower()
        if "top_right" in desc_lower or "upper_right" in desc_lower:
            norm_b = (700, 50, 950, 150)
        elif "top_left" in desc_lower or "upper_left" in desc_lower:
            norm_b = (50, 50, 300, 150)
        elif "top_center" in desc_lower or "upper_center" in desc_lower:
            norm_b = (150, 50, 850, 150)
        elif "bottom_right" in desc_lower or "lower_right" in desc_lower:
            norm_b = (700, 2000, 950, 2200)
        elif "bottom_center" in desc_lower or "lower_center" in desc_lower:
            norm_b = (350, 2100, 650, 2250)
        elif "bottom_left" in desc_lower or "lower_left" in desc_lower:
            norm_b = (50, 2100, 350, 2250)
        elif "middle_left" in desc_lower:
            norm_b = (50, 800, 450, 1000)
        elif "middle_center" in desc_lower:
            norm_b = (200, 900, 800, 1100)
        elif "scroll" in name_lower:
            norm_b = (100, 300, 900, 1800)
        else:
            norm_b = (100, 500, 900, 700)
    elif isinstance(bounds, list) and len(bounds) == 4:
        norm_b = (int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3]))
    else:
        norm_b = (100, 500, 900, 700)

    return CandidateSignature(
        role=role,
        label=clean_label,
        content_description=t.get("content_description", ""),
        normalized_bounds=norm_b,
        is_clickable=is_clickable,
        is_editable=is_editable,
        is_scrollable="scroll" in name_lower,
        is_enabled=t.get("enabled", True),
        is_focused=t.get("focused", False),
        is_checked=t.get("checked", False),
    )


def extract_argument_text(goal: str, cand_label: str) -> Optional[str]:
    patterns = [
        r"(?i)search for\s+['\"]?([^'\"\n\r]+?)['\"]?(?:\s+in|\s+on|\s+into|$)",
        r"(?i)type\s+['\"]?([^'\"\n\r]+?)['\"]?(?:\s+in|\s+into|$)",
        r"(?i)enter\s+['\"]?([^'\"\n\r]+?)['\"]?(?:\s+in|\s+into|$)",
        r"(?i)calculate\s+['\"]?([0-9\+\-\*\/\.\s]+)['\"]?",
    ]
    for p in patterns:
        m = re.search(p, goal)
        if m:
            val = m.group(1).strip()
            val = re.sub(r"(?i)\s+(?:in|into|on)\s+.*$", "", val).strip()
            if val:
                return val
    return None


class KarvishV50GpuHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            info = {
                "status": "ready",
                "version": "v50_universal",
                "model": "KarvishModelV50",
                "device": f"{DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})",
                "cuda": torch.cuda.is_available(),
                "parameters": 6440123,
                "checkpoint": str(CKPT_PATH.name),
            }
            self.wfile.write(json.dumps(info).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/propose":
            t0 = time.perf_counter()
            content_length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_length).decode("utf-8")

            try:
                payload = json.loads(raw_body)
                prompt = payload.get("prompt", "")
                tools = payload.get("tools", [])
                top_k = int(payload.get("top_k", 3))
                history = payload.get("history", [])

                clean_goal = prompt
                for marker in ["[history:", "(history:", "[previous failure:", "(previous action failed:"]:
                    if marker in clean_goal.lower():
                        clean_goal = clean_goal[:clean_goal.lower().find(marker)].strip()

                raw_cands = [parse_tool_to_candidate_sig(t, i) for i, t in enumerate(tools)]
                cands = raw_cands[:24]
                actual_cands = max(1, len(cands))

                goal_toks = TOKENIZER.encode(clean_goal)[:64]
                goal_t = torch.zeros(1, 64, dtype=torch.long, device=DEVICE)
                if goal_toks:
                    goal_t[0, :len(goal_toks)] = torch.tensor(goal_toks, dtype=torch.long, device=DEVICE)

                cand_t = torch.zeros(1, actual_cands, 16, dtype=torch.long, device=DEVICE)
                cand_roles = torch.zeros(1, actual_cands, dtype=torch.long, device=DEVICE)
                cand_bounds = torch.zeros(1, actual_cands, 4, dtype=torch.float32, device=DEVICE)
                cand_flags = torch.zeros(1, actual_cands, 32, dtype=torch.float32, device=DEVICE)
                cand_relations = torch.zeros(1, actual_cands, 16, dtype=torch.float32, device=DEVICE)
                cand_mask = torch.zeros(1, actual_cands, dtype=torch.float32, device=DEVICE)

                relations = RELATION_GRAPH.build_relations(cands)
                for ci, c in enumerate(cands):
                    lbl_toks = TOKENIZER.encode(c.label)[:16]
                    if lbl_toks:
                        cand_t[0, ci, :len(lbl_toks)] = torch.tensor(lbl_toks, dtype=torch.long, device=DEVICE)
                    cand_roles[0, ci] = UniversalSemanticRole.role_to_id(c.role) + 1
                    cand_bounds[0, ci] = torch.tensor([b / 1000.0 for b in c.normalized_bounds], dtype=torch.float32, device=DEVICE)

                    flg = [0.0] * 32
                    flg[0] = 1.0 if c.is_clickable else 0.0
                    flg[1] = 1.0 if c.is_editable else 0.0
                    flg[2] = 1.0 if c.is_scrollable else 0.0
                    flg[3] = 1.0 if c.is_enabled else 0.0
                    flg[4] = 1.0 if c.is_focused else 0.0
                    flg[5] = 1.0 if c.is_checked else 0.0
                    cand_flags[0, ci] = torch.tensor(flg, dtype=torch.float32, device=DEVICE)

                    rel_f = relations[ci].feature_vector if ci < len(relations) else [0.0] * 16
                    cand_relations[0, ci] = torch.tensor(rel_f[:16], dtype=torch.float32, device=DEVICE)
                    cand_mask[0, ci] = 1.0

                belief_t = torch.zeros(1, 8, dtype=torch.float32, device=DEVICE)
                memory_t = torch.zeros(1, 8, dtype=torch.float32, device=DEVICE)
                history_t = torch.zeros(1, 8, dtype=torch.float32, device=DEVICE)

                with torch.no_grad():
                    out = MODEL(
                        goal_t,
                        cand_t,
                        cand_roles,
                        cand_bounds,
                        cand_flags,
                        cand_relations,
                        belief_t,
                        memory_t,
                        history_t,
                        cand_mask,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                cand_logits = out["cand_logits"]
                mode_logits = out["mode_logits"]
                action_logits = out["action_logits"]
                trans_logits = out["transition_logits"]
                prog_status = out["progress_status_logits"]
                prog_scalar = out["progress_scalar"]
                term_conf = out["termination_confidence"]
                rec_logits = out["recovery_logits"]

                scores = cand_logits[0].cpu().numpy().tolist()
                probs = F.softmax(cand_logits[0], dim=0).cpu().numpy().tolist()

                pred_mode_id = int(torch.argmax(mode_logits[0]).cpu().item())
                pred_mode = DecisionMode.id_to_mode(pred_mode_id).value

                pred_act_id = int(torch.argmax(action_logits[0]).cpu().item())
                pred_action = UniversalActionType.id_to_action(pred_act_id).value

                pred_trans_id = int(torch.argmax(trans_logits[0]).cpu().item())
                pred_trans = ExpectedTransition.id_to_transition(pred_trans_id).value

                pred_prog_scalar = float(prog_scalar[0].cpu().item())
                pred_term_conf = float(term_conf[0].cpu().item())

                ranked_indices = sorted(range(len(cands)), key=lambda i: scores[i], reverse=True)

                proposals = []
                for rank_pos, c_idx in enumerate(ranked_indices[:top_k]):
                    c_sig = cands[c_idx]
                    orig_tool = tools[c_idx] if c_idx < len(tools) else {}
                    t_name = orig_tool.get("name", f"tool_{c_idx}")
                    score = float(scores[c_idx])
                    conf = float(probs[c_idx])

                    t_name_lower = t_name.lower()
                    c_act = orig_tool.get("action_type")
                    if not c_act:
                        if t_name_lower == "open_app" or t_name_lower.startswith("open_"):
                            c_act = "open_app"
                        elif c_sig.is_editable:
                            c_act = "type_text"
                        elif "scroll" in t_name_lower:
                            c_act = "scroll"
                        elif t_name_lower in ("submit", "press_enter"):
                            c_act = "submit"
                        elif t_name_lower in ("back", "press_back"):
                            c_act = "back"
                        elif t_name_lower in ("complete_goal", "complete"):
                            c_act = "complete"
                        elif t_name_lower in ("fail_goal", "fail"):
                            c_act = "fail"
                        else:
                            c_act = "tap"

                    # Determine target_node_index: system actions MUST NOT have a screen node target!
                    if c_act in ("open_app", "back", "complete", "fail"):
                        target_id = None
                    else:
                        target_id = orig_tool.get("target_node_index")
                        if target_id is None:
                            target_id = orig_tool.get("target_id")
                        if target_id is None and c_act in ("tap", "type_text"):
                            target_id = c_idx

                    args = {}
                    app_name = ""
                    if c_act == "open_app":
                        app_name = orig_tool.get("app_name", "")
                        if not app_name:
                            m = re.search(r"Launch application '([^']*)'", orig_tool.get("description", ""))
                            if m:
                                app_name = m.group(1).strip()
                        if not app_name:
                            m = re.search(r"(?i)\b(?:open|launch)\s+([A-Za-z0-9_\s]+)", clean_goal)
                            if m:
                                app_name = m.group(1).strip().split()[0]
                        if app_name:
                            args["app_name"] = app_name
                    elif c_act == "type_text":
                        extracted = extract_argument_text(clean_goal, c_sig.label)
                        if extracted:
                            args["text"] = extracted

                    prop_entry = {
                        "tool_name": t_name,
                        "action_type": c_act,
                        "label": c_sig.label,
                        "role": c_sig.role.value,
                        "score": round(score, 4),
                        "confidence": round(conf, 4),
                        "arguments": args,
                    }
                    if target_id is not None:
                        prop_entry["target_node_index"] = target_id
                        prop_entry["target_id"] = target_id
                    if app_name:
                        prop_entry["app_name"] = app_name

                    proposals.append(prop_entry)

                total_lat = (time.perf_counter() - t0) * 1000.0

                resp = {
                    "status": "success",
                    "model": "KarvishModelV50",
                    "device": str(DEVICE),
                    "proposals": proposals,
                    "mode": pred_mode,
                    "expected_transition": pred_trans,
                    "progress_scalar": round(pred_prog_scalar, 3),
                    "termination_confidence": round(pred_term_conf, 3),
                    "latency_ms": round(total_lat, 2),
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(resp).encode("utf-8"))

            except Exception as e:
                err_resp = {"status": "error", "error": str(e)}
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(err_resp).encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def run_v50_server(port: int = 8765):
    load_v50_model()
    setup_adb_reverse(port)

    server = HTTPServer(("0.0.0.0", port), KarvishV50GpuHandler)
    print(f"\n=========================================================================")
    print(f">> [READY] Karvish V50 PyTorch GPU Server listening on http://0.0.0.0:{port}")
    print(f">> Model running directly on: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f">> Connected Device ADB Reverse Port: tcp:{port} <-> tcp:{port}")
    print(f"=========================================================================\n", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n>> Shutting down GPU server.")
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Karvish V50 GPU Host Server")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    run_v50_server(port=args.port)
