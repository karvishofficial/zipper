"""
KARVISH V51.6 — LIVE GPU INFERENCE HOST SERVER (PORT 8765)
Hosts the Phase 6.6 Candidate-Conditioned Action Intelligence model (KarvishModelV51_6)
directly on the NVIDIA GeForce RTX 4050 GPU (CUDA).

Connects physical Android device via reverse ADB port forwarding:
    adb -s RZCY51MQP9K reverse tcp:8765 tcp:8765

Endpoints:
- GET  /health   -> Health check for LiveRankerAgentModelEngine
- POST /propose  -> Real-time candidate action proposal using calibrated joint decoding
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
    inject_causal_recurrence_into_flags,
)
from tools.training.karvish_mobile_action.v51_phase6_6.decoder_v51_6 import (
    CANDIDATE_ACTION_TYPES,
    GLOBAL_ACTION_TYPES,
    decode_calibrated_joint_action,
)
from tools.training.karvish_mobile_action.v51_phase6_6.model_v51_6 import (
    KarvishModelV51_6,
)

ADB_PATH = os.path.expandvars(r"%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe")
DEVICE_SERIAL = "RZCY51MQP9K"
DEFAULT_PORT = 8765

CHECKPOINT_PATH = (
    REPO_ROOT
    / "tools"
    / "training"
    / "karvish_mobile_action"
    / "v51_phase6_6"
    / "checkpoints"
    / "v51_phase6_6_best_checkpoint.pt"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = None
MODEL: Optional[KarvishModelV51_6] = None
CHECKPOINT_SHA256 = ""


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


def load_model(ckpt_path: Path = CHECKPOINT_PATH):
    global MODEL, TOKENIZER, CHECKPOINT_SHA256
    print(f">> Loading Phase 6.6 Model on {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})...", flush=True)
    assert ckpt_path.exists(), f"Checkpoint missing: {ckpt_path}"

    with open(ckpt_path, "rb") as f:
        CHECKPOINT_SHA256 = hashlib.sha256(f.read()).hexdigest()

    TOKENIZER = get_tokenizer(8192)
    model = KarvishModelV51_6(belief_dim=16, history_dim=16).to(DEVICE)

    payload = torch.load(ckpt_path, map_location=DEVICE)
    state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload

    if "cand_action_head.0.weight" in state:
        model.load_state_dict(state, strict=True)
        print(f">> Loaded full Phase 6.6 state dict ({len(state)} tensors).", flush=True)
    else:
        transferred = model.warm_start_from_phase6(state)
        print(f">> Warm-started from Phase 6 base ({transferred} tensors).", flush=True)

    model.eval()
    MODEL = model
    param_count = sum(p.numel() for p in model.parameters())
    print(f">> Model Ready: {param_count:,} parameters. Checkpoint SHA256: {CHECKPOINT_SHA256}", flush=True)


def extract_goal_argument_text(goal: str) -> Optional[str]:
    patterns = [
        r"(?i)search\s+for\s+['\"]?([^'\"\n\r]+?)['\"]?(?:\s+in|\s+on|\s+into|$)",
        r"(?i)search\s+['\"]?([^'\"\n\r]+?)['\"]?(?:\s+in|\s+on|\s+into|$)",
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


class KarvishV51_6BridgeHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Concise logging
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
                "version": "v51_6_phase6_6",
                "model": "KarvishModelV51_6",
                "device": f"{DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})",
                "checkpoint": CHECKPOINT_PATH.name,
                "sha256": CHECKPOINT_SHA256,
                "parameters": 10585092,
                "ready": True,
            }
            self.wfile.write(json.dumps(info).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
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

                # Process candidates
                cands = []
                for idx, t in enumerate(tools[:40]):
                    name = t.get("name", f"cand_{idx}")
                    lbl = t.get("semantic_label") or t.get("label") or name
                    role = t.get("role", "button").lower()
                    act = t.get("action_type", "tap").lower()
                    bounds = t.get("bounds", [0.0, 0.0, 1.0, 1.0])
                    if isinstance(bounds, dict):
                        b_list = [bounds.get("top", 0.0), bounds.get("left", 0.0), bounds.get("bottom", 1.0), bounds.get("right", 1.0)]
                    elif isinstance(bounds, list) and len(bounds) == 4:
                        b_list = bounds
                    else:
                        b_list = [0.0, 0.0, 1.0, 1.0]

                    is_clickable = bool(t.get("clickable", True))
                    is_editable = bool(t.get("editable", False) or "input" in role or act == "type_text")
                    is_scrollable = bool(t.get("scrollable", False) or act == "scroll")
                    is_enabled = bool(t.get("enabled", True))

                    flg = [0.0] * 32
                    flg[0] = 1.0 if is_clickable else 0.0
                    flg[1] = 1.0 if is_editable else 0.0
                    flg[2] = 1.0 if is_scrollable else 0.0
                    flg[3] = 1.0 if is_enabled else 0.0
                    flg[8] = 1.0
                    flg[9] = 1.0

                    cands.append({
                        "id": t.get("candidate_id") or name,
                        "label": lbl,
                        "role": role,
                        "action_type": "type_text" if is_editable else ("scroll" if is_scrollable else ("tap" if is_clickable else act)),
                        "bounds": b_list,
                        "flags": flg,
                        "target_node_index": t.get("target_node_index", idx),
                        "orig_tool": t,
                    })

                # Compute causal recurrence from history
                last_touch_coords = (0, 0)
                last_target_label = ""
                if history:
                    last_step = history[-1]
                    last_target_label = last_step.get("action_description", "")

                rec_vals = compute_causal_recurrence(
                    cands,
                    last_touch_coords=last_touch_coords,
                    last_target_label=last_target_label,
                )
                cand_flags_tensor = torch.zeros(1, max(1, len(cands)), 32, dtype=torch.float32, device=DEVICE)
                for ci, c in enumerate(cands):
                    c["flags"][22] = rec_vals[ci] if ci < len(rec_vals) else 0.0
                    cand_flags_tensor[0, ci] = torch.tensor(c["flags"], dtype=torch.float32, device=DEVICE)

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
                    lbl_toks = TOKENIZER.encode(c["label"])[:16]
                    if lbl_toks:
                        cand_lbl_ids[0, ci, : len(lbl_toks)] = torch.tensor(lbl_toks, dtype=torch.long, device=DEVICE)
                        cand_lbl_mask[0, ci, : len(lbl_toks)] = True

                    b = c["bounds"]
                    ymin, xmin, ymax, xmax = b[0], b[1], b[2], b[3]
                    cand_spatial[0, ci, :4] = torch.tensor([ymin, xmin, ymax, xmax], dtype=torch.float32, device=DEVICE)
                    cand_spatial[0, ci, 4:] = torch.tensor([(ymin + ymax) / 2, (xmin + xmax) / 2, ymax - ymin, xmax - xmin], dtype=torch.float32, device=DEVICE)

                # Belief and History features
                past_step_records = []
                for h in history:
                    past_step_records.append({
                        "action": h.get("action", "wait"),
                        "outcome": h.get("outcome", "SUCCESS"),
                        "state_changed": h.get("state_changed", True),
                    })
                b_feat = compute_observable_diff(None, obs_dict, None, cands).unsqueeze(0).to(DEVICE)
                h_feat = compute_history_ledger(past_step_records).unsqueeze(0).to(DEVICE)

                # Inference on GPU
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
                        belief_feats=b_feat,
                        history_feats=h_feat,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                # Calibrated Joint Decoding
                decision = decode_calibrated_joint_action(
                    unified_ranking_scores=out["unified_ranking_scores"],
                    cand_action_logits=out["cand_action_logits"],
                    global_action_logits=out["global_action_logits"],
                    candidates=cands,
                    goal_text=clean_goal,
                    cand_pad_mask=cand_pad_mask,
                    temperature=1.0,
                )

                ranking_scores = out["unified_ranking_scores"][0].cpu().tolist()
                cand_act_logits = out["cand_action_logits"][0]  # (N, 4)

                # Top-K candidate ranking
                ranked_cand_indices = sorted(range(len(cands)), key=lambda i: ranking_scores[i + 1], reverse=True)

                arg_text = extract_goal_argument_text(clean_goal) or ""

                proposals = []
                dec_type = decision.get("type", "candidate")
                dec_action = decision.get("action", "tap")
                dec_cand = decision.get("candidate")
                dec_arg = decision.get("argument", "") or arg_text
                dec_conf = decision.get("confidence", 0.5)
                dec_logp = decision.get("log_probability", -1.0)

                chosen_args = {}
                target_id = None
                target_name = "GLOBAL"
                if dec_type == "candidate" and dec_cand is not None:
                    target_name = dec_cand.get("id") or "cand_0"
                    target_id = dec_cand.get("target_node_index")
                    if dec_action == "type_text" and dec_arg:
                        chosen_args["text"] = dec_arg
                elif dec_action == "type_text" and dec_arg:
                    chosen_args["text"] = dec_arg
                elif dec_action == "open_app":
                    app_name = extract_app_name(clean_goal)
                    if app_name:
                        chosen_args["app_name"] = app_name

                p1 = {
                    "tool_name": target_name,
                    "candidate_id": target_name,
                    "action_type": dec_action,
                    "target_node_index": target_id,
                    "target_id": target_id,
                    "confidence": round(dec_conf, 4),
                    "score": round(dec_logp, 4),
                    "expected_outcome": f"Model predicted {dec_action}",
                    "arguments": chosen_args,
                }
                if "app_name" in chosen_args:
                    p1["app_name"] = chosen_args["app_name"]
                proposals.append(p1)

                # Alternative proposals for Top-K
                for ci in ranked_cand_indices:
                    c_id = cands[ci]["id"]
                    if c_id == target_name:
                        continue
                    c_act_idx = cand_act_logits[ci].argmax().item()
                    c_act = CANDIDATE_ACTION_TYPES[c_act_idx]
                    c_args = {}
                    if c_act == "type_text" and arg_text:
                        c_args["text"] = arg_text

                    p_alt = {
                        "tool_name": c_id,
                        "candidate_id": c_id,
                        "action_type": c_act,
                        "target_node_index": cands[ci]["target_node_index"],
                        "target_id": cands[ci]["target_node_index"],
                        "confidence": round(float(F.softmax(cand_act_logits[ci], dim=-1)[c_act_idx].item()), 4),
                        "score": round(ranking_scores[ci + 1], 4),
                        "expected_outcome": f"Alternative proposal: {c_act} on {cands[ci]['label']}",
                        "arguments": c_args,
                    }
                    proposals.append(p_alt)
                    if len(proposals) >= top_k:
                        break

                latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                resp = {
                    "status": "success",
                    "model": "KarvishModelV51_6",
                    "decision_type": dec_type,
                    "latency_ms": latency_ms,
                    "proposals": proposals,
                }
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps(resp).encode("utf-8"))
                print(f">> [PROPOSE {latency_ms}ms] Action: {dec_action} -> Target: {target_name} (conf={dec_conf:.3f})", flush=True)

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


def run_server(port: int = DEFAULT_PORT):
    print("=" * 80, flush=True)
    print("KARVISH V51.6 — HOST GPU INFERENCE SERVER", flush=True)
    print("=" * 80, flush=True)
    load_model()
    setup_adb_reverse(port)

    server_address = ("0.0.0.0", port)
    httpd = HTTPServer(server_address, KarvishV51_6BridgeHandler)
    print(f">> Server listening on http://0.0.0.0:{port} (accessible on phone at http://127.0.0.1:{port})", flush=True)
    print(f">> READY FOR MOBILE AGENT INTERACTION", flush=True)
    print("=" * 80, flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n>> Shutting down server...", flush=True)
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Karvish V51.6 GPU Host Server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to listen on")
    args = parser.parse_args()
    run_server(args.port)
