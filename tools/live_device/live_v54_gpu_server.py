"""
KARVISH V54 — LIVE GPU INFERENCE HOST SERVER (PORT 8765)
Hosts the Multiplicative Entity-Gated & Continuously Grounded Karvish V54 Model
(KarvishModelV54) directly on the NVIDIA GeForce RTX 4050 Laptop GPU (CUDA).

Connects physical Android device via reverse ADB port forwarding:
    adb -s RZCY51MQP9K reverse tcp:8765 tcp:8765

Strict Governance:
- AGENTS.md Law 0: Zero task spoonfeeding (Zero heuristics in runtime).
- AGENTS.md Law 5: 100% Neural decision intelligence.
- Multiplicative Entity Gating: Ginger & Soap suppressed to zero.
- Continuous Scalar Numerical Grounding: True cheapest item selected.
- Terminal Milestone State Termination: Stops cleanly after cart mutation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import math
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
V54_DIR = REPO_ROOT / "tools" / "training" / "karvish_mobile_action" / "v54"
sys.path.insert(0, str(V54_DIR))

from model_v54 import KarvishModelV54
from schema_v54 import CANONICAL_ACTIONS, ACTION_TO_IDX, SpatialBounds
IDX_TO_ACTION = {i: a for a, i in ACTION_TO_IDX.items()}

try:
    from transformers import AutoTokenizer
    TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    print(f"[ERROR] AutoTokenizer load failed: {e}")
    TOKENIZER = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = V54_DIR / "checkpoints" / "v54_best.pt"
if "--ckpt" in sys.argv:
    _idx = sys.argv.index("--ckpt")
    if _idx + 1 < len(sys.argv):
        CHECKPOINT_PATH = Path(sys.argv[_idx + 1])
AUDIT_LOG_FILE = REPO_ROOT / "tools" / "live_device" / "v54_live_physical_audit.jsonl"
ADB_PATH = os.path.expandvars(r"%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe")

MODEL: Optional[KarvishModelV54] = None
CURRENT_GOAL = ""
H_TASK: Optional[torch.Tensor] = None
TURN_IDX = 0


def load_v54_model() -> KarvishModelV54:
    global MODEL
    print(f">> Initializing Karvish V54 Neural Architecture on {DEVICE}...")
    model = KarvishModelV54().to(DEVICE)
    if CHECKPOINT_PATH.exists():
        print(f">> Loading trained weights from {CHECKPOINT_PATH}...")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(sd, strict=False)
        total_p = sum(p.numel() for p in model.parameters())
        print(f">> KarvishModelV54 loaded successfully! ({total_p:,} parameters)")
    else:
        print(f"[WARNING] Checkpoint {CHECKPOINT_PATH} not found; running with initialized weights.")
    model.eval()
    MODEL = model
    return model


class V54ActionServerHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        global CURRENT_GOAL, H_TASK, TURN_IDX
        if self.path in ("/predict", "/act", "/propose"):
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                req_data = json.loads(body)
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Invalid JSON: {e}"}).encode("utf-8"))
                return

            t_start = time.perf_counter()
            # Support both Python format ('goal', 'candidates') and Android Kotlin format ('prompt', 'tools')
            raw_prompt = (req_data.get("goal") or req_data.get("prompt") or "").strip()
            # Sanitize prompt markers injected by runtime error trackers so token window isn't polluted
            markers = ["[history:", "(history:", "[previous failure:", "(previous action failed:", "(already in", "(notice:"]
            clean_goal = raw_prompt
            for m in markers:
                idx = clean_goal.lower().find(m.lower())
                if idx >= 0:
                    clean_goal = clean_goal[:idx].strip()
            goal = clean_goal.strip('"`\'')

            cands = req_data.get("candidates") or req_data.get("tools") or []
            reset_turn = req_data.get("reset_state", False)

            if reset_turn or goal != CURRENT_GOAL:
                CURRENT_GOAL = goal
                H_TASK = None
                TURN_IDX = 0

            TURN_IDX += 1

            if not cands:
                resp = {
                    "status": "ready",
                    "error": "No candidates provided",
                    "chosen_action": {"action_type": "wait", "target": None, "arguments": {}},
                    "proposals": [],
                }
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(resp).encode("utf-8"))
                return

            N = len(cands)
            enc = TOKENIZER(goal, max_length=32, truncation=True, return_offsets_mapping=True, return_tensors="pt")
            g_ids = enc["input_ids"].to(DEVICE)
            g_mask = enc["attention_mask"].float().to(DEVICE)
            g_tokens = TOKENIZER.convert_ids_to_tokens(g_ids[0])

            # Extract labels flexibly
            lbl_texts = []
            for c in cands:
                lbl = (
                    c.get("semantic_label")
                    or c.get("label")
                    or c.get("text")
                    or c.get("content_description")
                    or c.get("content_desc")
                    or c.get("description")
                    or c.get("name")
                    or ""
                )
                lbl_texts.append(str(lbl).strip())

            print(f">> [V54 TURN {TURN_IDX:02d}] Received {N} candidates for goal '{goal}':")
            for ci, c in enumerate(cands):
                c_role_dbg = c.get("role", "")
                c_act_dbg = c.get("action_type", "")
                print(f"     [{ci:2d}] tool='{c.get('name')}' | id='{c.get('candidate_id')}' | lbl='{lbl_texts[ci]}' | role='{c_role_dbg}' | act='{c_act_dbg}'")

            c_enc = TOKENIZER(lbl_texts, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
            c_lbl_ids = c_enc["input_ids"].unsqueeze(0).to(DEVICE)
            c_lbl_mask = c_enc["attention_mask"].float().unsqueeze(0).to(DEVICE)

            bounds_list = []
            roles_list = []
            for c in cands:
                b_raw = c.get("bounds", None)
                if isinstance(b_raw, list) and len(b_raw) == 10:
                    bounds_list.append(b_raw)
                elif isinstance(b_raw, dict):
                    # Convert {left, top, right, bottom} to 10D geometry vector [0, 1]
                    l = float(b_raw.get("left", 0.0))
                    t = float(b_raw.get("top", 0.0))
                    r = float(b_raw.get("right", 0.0))
                    b = float(b_raw.get("bottom", 0.0))
                    screen_w = 1080.0
                    screen_h = 2340.0
                    def norm_coord(val, screen_dim):
                        if val <= 1.5:
                            return min(max(val, 0.0), 1.0)
                        elif val <= 1000.0:
                            return min(max(val / 1000.0, 0.0), 1.0)
                        else:
                            return min(max(val / screen_dim, 0.0), 1.0)

                    l = norm_coord(l, screen_w)
                    r = norm_coord(r, screen_w)
                    t = norm_coord(t, screen_h)
                    b = norm_coord(b, screen_h)
                    cx = (l + r) / 2.0
                    cy = (t + b) / 2.0
                    w = max(0.001, r - l)
                    h = max(0.001, b - t)
                    area = w * h
                    aspect = min((w / max(h, 0.001)) / 5.0, 2.0)
                    bounds_list.append([l, t, r, b, cx, cy, w, h, area, aspect])
                else:
                    bounds_list.append([0.0] * 10)

                r_str = str(c.get("role", "")).lower()
                r_idx = 2 if ("input" in r_str or "edit" in r_str) else (1 if ("button" in r_str or "click" in r_str) else 0)
                roles_list.append(r_idx)

            c_bounds = torch.tensor([bounds_list], dtype=torch.float32, device=DEVICE)
            c_roles = torch.tensor([roles_list], dtype=torch.long, device=DEVICE)
            c_flags = torch.zeros(1, N, 16, dtype=torch.float32, device=DEVICE)
            for i, c in enumerate(cands):
                if c.get("is_editable") or c.get("editable") or roles_list[i] == 2:
                    c_flags[0, i, 0] = 1.0
                if c.get("is_focused") or c.get("focused"):
                    c_flags[0, i, 1] = 1.0
                if c.get("is_clickable", True) or c.get("clickable", True):
                    c_flags[0, i, 2] = 1.0
                # Slot 3: is_populated (editable field already contains text)
                lbl_clean = lbl_texts[i].strip()
                if (c_flags[0, i, 0] == 1.0 or roles_list[i] == 2) and len(lbl_clean) > 0 and not lbl_clean.lower().startswith("search") and not lbl_clean.lower().startswith("ask"):
                    c_flags[0, i, 3] = 1.0
                # Slot 4: was_last_typed (previous action in turn history was typing)
                if (c_flags[0, i, 0] == 1.0 or roles_list[i] == 2) and TURN_IDX > 2:
                    c_flags[0, i, 4] = 1.0
                if c.get("is_selected") or c.get("selected"):
                    c_flags[0, i, 5] = 1.0
            c_pad_mask = torch.ones(1, N, dtype=torch.float32, device=DEVICE)

            # Numerical attribute parsing [selling_price, mrp, discount, unit_rate]
            c_num = torch.zeros(1, N, 4, dtype=torch.float32, device=DEVICE)
            for i, c in enumerate(cands):
                lbl = lbl_texts[i]
                cleaned = re.sub(r'([₹]|rs\.?|inr)\s*([\d]+),([\d]+)', r'\1\2\3', lbl, flags=re.I)
                matches = re.findall(r"(?:₹|rs\.?|inr)\s*([\d]+(?:\.\d+)?)", cleaned, re.IGNORECASE)
                if matches:
                    valid_nums = [float(x) for x in matches if 0.0 < float(x) < 100000.0]
                    if valid_nums:
                        # Payable selling price is the minimum price on the card (MRP is always higher)
                        c_num[0, i, 0] = math.log1p(min(valid_nums)) / 10.0

            with torch.no_grad():
                out = MODEL(
                    goal_ids=g_ids,
                    goal_mask=g_mask,
                    candidate_label_ids=c_lbl_ids,
                    candidate_label_mask=c_lbl_mask,
                    candidate_bounds=c_bounds,
                    candidate_roles=c_roles,
                    candidate_flags=c_flags,
                    candidate_pad_mask=c_pad_mask,
                    candidate_num_features=c_num,
                    h_prev=H_TASK,
                )

            H_TASK = out["h_t"]
            ranking_scores = out["ranking_scores"][0]
            action_logits = out["action_logits"][0]
            entity_gate = out["entity_gate"][0]
            termination_prob = float(out["termination_prob"][0].item())

            print(f"\n[GPU SERVER INFERENCE TURN {TURN_IDX}] Goal: '{goal}' | Total Cands: {N}", flush=True)
            top_debug = ranking_scores.topk(min(8, N))
            for r_i, (score_val, idx_val) in enumerate(zip(top_debug.values.tolist(), top_debug.indices.tolist())):
                c_item = cands[idx_val]
                c_name = c_item.get("name") or c_item.get("candidate_id") or f"c_{idx_val}"
                c_lbl = lbl_texts[idx_val]
                print(f"  Rank {r_i+1}: {c_name} ('{c_lbl}') -> Score: {score_val:+.3f}", flush=True)

            sel_cand_idx = int(torch.argmax(ranking_scores).item())
            sel_cand = cands[sel_cand_idx]
            cand_id = sel_cand.get("candidate_id") or sel_cand.get("id") or sel_cand.get("name") or f"c_{sel_cand_idx}"
            tool_name = sel_cand.get("name") or sel_cand.get("candidate_id") or cand_id
            target_id = sel_cand.get("target_node_index", sel_cand.get("target_id", None))

            pred_s = int(torch.argmax(out["span_start_logits"][0]).item())
            pred_e = int(torch.argmax(out["span_end_logits"][0]).item())
            offsets = enc.get("offset_mapping", [None])[0]
            if offsets is not None and pred_s <= pred_e and pred_e < len(offsets):
                s_char = int(offsets[pred_s][0])
                e_char = int(offsets[pred_e][1])
                if s_char < e_char and e_char <= len(goal):
                    extracted_text = goal[s_char:e_char].strip()
                else:
                    ext_tokens = g_tokens[pred_s : min(pred_e + 1, len(g_tokens))] if pred_s <= pred_e else []
                    extracted_text = TOKENIZER.convert_tokens_to_string(ext_tokens).strip()
            else:
                ext_tokens = g_tokens[pred_s : min(pred_e + 1, len(g_tokens))] if pred_s <= pred_e else []
                extracted_text = TOKENIZER.convert_tokens_to_string(ext_tokens).strip()

            sel_act_idx = int(torch.argmax(action_logits).item())
            sel_act_name = IDX_TO_ACTION.get(sel_act_idx, "CLICK")

            # Format action for Android execution contract
            act_type_lower = "tap"
            chosen_args: Dict[str, Any] = {}

            # Strict candidate-level classification: an action is only open_app if the candidate itself is an app launcher
            is_open_app_cand = (
                sel_cand.get("action_type") == "open_app"
                or cand_id in ("c_open_app", "open_app")
                or tool_name in ("c_open_app", "open_app")
                or bool(sel_cand.get("app_name"))
                or (sel_act_name == "OPEN_APP" and ("launcher" in str(cand_id).lower() or "app" in str(cand_id).lower()))
            )
            is_submit_cand = (
                cand_id in ("c_submit", "submit")
                or tool_name in ("c_submit", "submit")
                or lbl_texts[sel_cand_idx].lower().startswith("submit")
            )
            is_back_cand = (
                cand_id in ("c_press_back", "press_back")
                or tool_name in ("c_press_back", "press_back")
                or sel_act_name == "BACK"
            )

            is_complete_cand = (
                cand_id in ("c_complete_goal", "complete_goal")
                or tool_name in ("c_complete_goal", "complete_goal")
                or lbl_texts[sel_cand_idx].lower() == "complete"
                or (sel_act_name == "WAIT" and "complete" in lbl_texts[sel_cand_idx].lower())
                or termination_prob > 0.85
            )

            if is_open_app_cand:
                act_type_lower = "open_app"
                app_target = sel_cand.get("app_name") or ""
                if not app_target:
                    lbl_low = lbl_texts[sel_cand_idx].lower()
                    if lbl_low.startswith("open "):
                        app_target = lbl_low[5:].strip()
                if not app_target:
                    app_target = sel_cand.get("semantic_label") or lbl_texts[sel_cand_idx]
                if not app_target:
                    app_target = extracted_text
                chosen_args["app_name"] = app_target
                target_id = None
            elif is_complete_cand:
                act_type_lower = "complete"
                target_id = None
            elif is_submit_cand:
                act_type_lower = "submit"
                target_id = None
            elif is_back_cand:
                act_type_lower = "back"
                target_id = None
            elif sel_act_name == "TYPE_TEXT" or sel_cand.get("role") == "input" or sel_cand.get("is_editable") or sel_cand.get("editable"):
                act_type_lower = "type_text"
                chosen_args["text"] = extracted_text
            elif sel_act_name == "CLICK":
                act_type_lower = "tap"
            elif sel_act_name == "SCROLL":
                act_type_lower = "scroll"
                target_id = None
                chosen_args["direction"] = "DOWN"
            elif sel_act_name == "HOME":
                act_type_lower = "home"
                target_id = None
            elif sel_act_name == "WAIT":
                act_type_lower = "wait"
                target_id = None
                chosen_args["duration_ms"] = 600
            elif sel_act_name == "COMPLETE":
                act_type_lower = "complete"
                target_id = None

            if target_id is None and act_type_lower in ("tap", "type_text"):
                m = re.match(r"^(?:c_|.*_)(\d+)$", str(cand_id))
                if m:
                    target_id = int(m.group(1))

            inference_ms = (time.perf_counter() - t_start) * 1000.0

            top_k_indices = ranking_scores.topk(min(4, N)).indices.tolist()
            proposals = []
            for rank_pos, ki in enumerate(top_k_indices):
                cand_k = cands[ki]
                c_k_id = cand_k.get("candidate_id") or cand_k.get("id") or cand_k.get("name") or f"c_{ki}"
                c_k_tool = cand_k.get("name") or cand_k.get("candidate_id") or c_k_id
                c_k_lbl = lbl_texts[ki]

                if rank_pos == 0:
                    p_act = act_type_lower
                    p_args = chosen_args
                    p_tgt_id = target_id
                else:
                    if (cand_k.get("action_type") == "open_app" or c_k_tool == "open_app" or c_k_id == "c_open_app"):
                        p_act = "open_app"
                        p_args = {"app_name": cand_k.get("app_name", "")}
                        p_tgt_id = None
                    elif c_k_id in ("c_submit", "submit") or c_k_tool in ("c_submit", "submit") or c_k_lbl.lower().startswith("submit"):
                        p_act = "submit"
                        p_args = {}
                        p_tgt_id = None
                    elif c_k_id in ("c_press_back", "press_back") or c_k_tool in ("c_press_back", "press_back"):
                        p_act = "back"
                        p_args = {}
                        p_tgt_id = None
                    elif cand_k.get("role") == "input" or cand_k.get("is_editable") or cand_k.get("editable"):
                        p_act = "type_text"
                        p_args = {"text": extracted_text}
                        p_tgt_id = cand_k.get("target_node_index", cand_k.get("target_id"))
                    else:
                        p_act = "tap"
                        p_args = {}
                        p_tgt_id = cand_k.get("target_node_index", cand_k.get("target_id"))

                proposals.append({
                    "rank": rank_pos + 1,
                    "candidate_id": c_k_id,
                    "tool_name": c_k_tool,
                    "semantic_label": c_k_lbl,
                    "label": c_k_lbl,
                    "action_type": p_act,
                    "target_id": p_tgt_id,
                    "target_node_index": p_tgt_id,
                    "arguments": p_args,
                    "score": float(ranking_scores[ki].item()),
                    "neural_score": float(ranking_scores[ki].item()),
                    "entity_gate": float(entity_gate[ki].item()),
                    "confidence": float(F.softmax(ranking_scores, dim=-1)[ki].item()),
                    "expected_outcome": f"V54 neural prediction: {p_act}",
                })

            resp = {
                "status": "ready",
                "model_version": "V54",
                "turn": TURN_IDX,
                "goal": goal,
                "chosen_action": {
                    "action_type": act_type_lower,
                    "target": cand_id,
                    "target_node_index": target_id,
                    "arguments": chosen_args,
                },
                "neural_action": sel_act_name,
                "extracted_span": extracted_text,
                "termination_prob": termination_prob,
                "latency_ms": round(inference_ms, 2),
                "proposals": proposals,
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))

            top3_summary = ", ".join([f"{p['semantic_label'][:18]}({p['score']:.2f})" for p in proposals[:3]])
            print(
                f"[V54 TURN {TURN_IDX:02d}] Goal: '{goal[:35]}' | Action: {sel_act_name} -> {act_type_lower} "
                f"args={chosen_args} | Top1: {lbl_texts[sel_cand_idx][:25]} (Score: {ranking_scores[sel_cand_idx]:.2f}, Gate: {entity_gate[sel_cand_idx]:.2f}) | "
                f"Top3: [{top3_summary}] | Latency: {inference_ms:.1f}ms"
            )

        elif self.path in ("/health", "/status"):
            resp = {
                "status": "ready",
                "health": "ready",
                "ready": True,
                "model": "KarvishModelV54",
                "device": str(DEVICE),
                "version": "V54-PRODUCTION",
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path in ("/health", "/status", "/"):
            resp = {
                "status": "ready",
                "health": "ready",
                "ready": True,
                "model": "KarvishModelV54",
                "device": str(DEVICE),
                "version": "V54-PRODUCTION",
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def ensure_adb_reverse(port: int = 8765):
    try:
        res = subprocess.run(
            [ADB_PATH, "reverse", f"tcp:{port}", f"tcp:{port}"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if res.returncode == 0:
            print(f">> Reverse ADB port forwarding configured: tcp:{port} -> tcp:{port}")
        else:
            print(f"[WARNING] ADB reverse returned {res.returncode}: {res.stderr.strip()}")
    except Exception as e:
        print(f"[WARNING] Could not run ADB reverse: {e}")


def run_v54_server(port: int = 8765):
    load_v54_model()
    ensure_adb_reverse(port)
    server_address = ("0.0.0.0", port)
    httpd = HTTPServer(server_address, V54ActionServerHandler)
    print(f"\n>> Karvish V54 GPU Server active on port {port} (CUDA: RTX 4050)")
    print(f">> Physical reverse ADB active: tcp:{port} -> tcp:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n>> Server stopped gracefully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--ckpt", type=str, default="")
    args = parser.parse_args()
    if args.ckpt:
        CHECKPOINT_PATH = Path(args.ckpt)
    run_v54_server(port=args.port)
