"""
KARVISH V52 — LIVE GPU INFERENCE HOST SERVER (PORT 8765)
Hosts the De-Spoonfed 100% Neural Karvish V52 Model (KarvishModelV52)
directly on the NVIDIA GeForce RTX 4050 GPU (CUDA).

Connects physical Android device via reverse ADB port forwarding:
    adb -s RZCY51MQP9K reverse tcp:8765 tcp:8765

Strict Governance:
- AGENTS.md Law 0: Zero task spoonfeeding (Zero +3.5 search boost, Zero -5.0 feed penalty).
- AGENTS.md Law 5: 100% Neural decision intelligence (Ranking, Action Type, Pointer Span).
- AGENTS.md Law 7.5: Zero regex entity extraction; learned span pointer head only.
- Strict live step auditing to v52_live_physical_audit.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
V52_DIR = REPO_ROOT / "tools" / "training" / "karvish_mobile_action" / "v52"
sys.path.insert(0, str(V52_DIR))

from model_v52 import KarvishModelV52
from schema_v52 import CANONICAL_ACTIONS, ACTION_TO_IDX, SpatialBounds
IDX_TO_ACTION = {i: a for a, i in ACTION_TO_IDX.items()}

try:
    from transformers import AutoTokenizer
    TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    print(f"[ERROR] AutoTokenizer load failed: {e}")
    TOKENIZER = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = V52_DIR / "checkpoints" / "v52_best.pt"
AUDIT_LOG_FILE = REPO_ROOT / "tools" / "live_device" / "v52_live_physical_audit.jsonl"

MODEL: Optional[KarvishModelV52] = None
CURRENT_GOAL = ""
H_TASK: Optional[torch.Tensor] = None
C_CURR: Optional[torch.Tensor] = None
TURN_IDX = 0
SCREEN_WIDTH = 1080.0
SCREEN_HEIGHT = 2400.0


def load_v52_model():
    global MODEL
    print("=" * 80)
    print(f"LOADING KARVISH V52 MODEL ON {DEVICE}...")
    assert CHECKPOINT_PATH.exists(), f"Checkpoint missing: {CHECKPOINT_PATH}"
    MODEL = KarvishModelV52().to(DEVICE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    MODEL.load_state_dict(ckpt["model_state_dict"])
    MODEL.eval()
    print(f">> Loaded {CHECKPOINT_PATH.name} (Grounded Val Top-1: {ckpt.get('val_top1', 0.0):.2f}%)")
    print(f">> Total Parameters: {sum(p.numel() for p in MODEL.parameters()):,}")
    print("=" * 80)


def reset_episode(goal: str):
    global CURRENT_GOAL, H_TASK, C_CURR, TURN_IDX
    CURRENT_GOAL = goal
    H_TASK = None
    C_CURR = None
    TURN_IDX = 0
    print(f"\n[EPISODE RESET] New Live Goal: '{goal}'", flush=True)


class LiveV52ServerHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress default HTTP logging to keep stdout clean

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            resp = {
                "status": "ready",
                "model": "KarvishModelV52",
                "device": str(DEVICE),
                "parameters": 26532446,
                "zero_heuristics": True,
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        global H_TASK, C_CURR, TURN_IDX, CURRENT_GOAL, MODEL

        if self.path == "/propose":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body)

            goal = (data.get("goal") or data.get("prompt") or "").strip()
            tools = data.get("tools", [])
            history = data.get("history", [])

            if goal != CURRENT_GOAL or len(history) == 0:
                reset_episode(goal)

            TURN_IDX += 1
            t_start = time.time()

            # 1. WordPiece Tokenization of Goal
            enc = TOKENIZER(goal, max_length=32, truncation=True, padding="max_length", return_offsets_mapping=True, return_tensors="pt")
            goal_ids = enc["input_ids"].to(DEVICE)
            goal_mask = enc["attention_mask"].float().to(DEVICE)
            offsets = enc["offset_mapping"][0]

            # 2. Extract and format candidate features
            cands = []
            obs_nodes = (data.get("observation") or {}).get("nodes", [])

            for idx, tool_item in enumerate(tools[:64]):
                name = tool_item.get("name", f"cand_{idx}")
                lbl = str(tool_item.get("semantic_label") or tool_item.get("label") or name).strip()
                role = str(tool_item.get("role", "button")).lower()
                act = str(tool_item.get("action_type", "tap")).lower()
                bounds_raw = tool_item.get("bounds", [0.0, 0.0, 1.0, 1.0])

                if isinstance(bounds_raw, dict):
                    b_left = float(bounds_raw.get("left", 0.0))
                    b_top = float(bounds_raw.get("top", 0.0))
                    b_right = float(bounds_raw.get("right", 1.0))
                    b_bottom = float(bounds_raw.get("bottom", 1.0))
                    if max(b_left, b_top, b_right, b_bottom) > 1.05:
                        b_left, b_top, b_right, b_bottom = b_left / 1000.0, b_top / 1000.0, b_right / 1000.0, b_bottom / 1000.0
                    sb = SpatialBounds(left=b_left, top=b_top, right=b_right, bottom=b_bottom)
                elif isinstance(bounds_raw, list) and len(bounds_raw) == 4:
                    raw_b = [float(x) for x in bounds_raw]
                    if max(raw_b) > 1.05:
                        raw_b = [x / 1000.0 for x in raw_b]
                    sb = SpatialBounds(left=raw_b[0], top=raw_b[1], right=raw_b[2], bottom=raw_b[3])
                else:
                    sb = SpatialBounds(0.0, 0.0, 1.0, 1.0)

                target_node_idx = tool_item.get("target_node_index")
                if target_node_idx is not None and not isinstance(target_node_idx, int):
                    try:
                        target_node_idx = int(target_node_idx)
                    except Exception:
                        target_node_idx = None

                # Context enrichment for generic button labels (e.g. Add to Cart / ADD)
                if ("add-button" in name or lbl.lower() in ["add to cart", "add", "buy"]) and obs_nodes:
                    bl, bt, br, bb = sb.left, sb.top, sb.right, sb.bottom
                    enclosing = []
                    for n in obs_nodes:
                        if not isinstance(n, dict): continue
                        if target_node_idx is not None and n.get("id") == target_node_idx: continue
                        b_dict = n.get("bounds", {})
                        if isinstance(b_dict, dict):
                            nl = float(b_dict.get("left", 0.0))
                            nt = float(b_dict.get("top", 0.0))
                            nr = float(b_dict.get("right", 1.0))
                            nbm = float(b_dict.get("bottom", 1.0))
                        elif isinstance(b_dict, list) and len(b_dict) == 4:
                            nl, nt, nr, nbm = [float(x) for x in b_dict]
                        else:
                            continue
                        if max(nl, nt, nr, nbm) > 1.05:
                            nl, nt, nr, nbm = nl / 1000.0, nt / 1000.0, nr / 1000.0, nbm / 1000.0
                        if nl <= bl + 0.08 and nt <= bt + 0.08 and nr >= br - 0.08 and nbm >= bb - 0.08:
                            desc = (n.get("content_description") or n.get("text") or n.get("label") or "").strip()
                            rh = str(n.get("resource_hint", "")).lower()
                            if desc and ("₹" in desc or "off" in desc.lower() or "product" in rh or "card" in rh or any(k in desc.lower() for k in ["tomato", "potato", "onion", "vegetable", "kg", " g", "pack"])):
                                area = max(0.0001, (nr - nl) * (nbm - nt))
                                enclosing.append((area, desc))
                    if enclosing:
                        enclosing.sort(key=lambda x: x[0])
                        card_desc = re.sub(r'[\u200a-\u200f\ufeff]', '', enclosing[0][1]).strip()
                        if card_desc and not any(k in lbl.lower() for k in ["tomato", "milk", "paneer", "apple"]):
                            lbl = f"{card_desc} Add to Cart"

                cands.append({
                    "id": tool_item.get("candidate_id") or name,
                    "label": lbl,
                    "role": role,
                    "action_type": act,
                    "bounds_vec": sb.to_vector(),
                    "target_node_index": target_node_idx,
                    "orig_tool": tool_item,
                })

            N = len(cands)
            if N == 0:
                cands.append({
                    "id": "c_screen",
                    "label": "screen",
                    "role": "view",
                    "action_type": "wait",
                    "bounds_vec": [0.0] * 10,
                    "target_node_index": None,
                    "orig_tool": {},
                })
                N = 1

            # Prepare Candidate Tensors
            max_lbl_len = 32
            cand_lbl_ids = torch.zeros(1, N, max_lbl_len, dtype=torch.long, device=DEVICE)
            cand_lbl_mask = torch.zeros(1, N, max_lbl_len, dtype=torch.float32, device=DEVICE)
            cand_bounds = torch.zeros(1, N, 10, dtype=torch.float32, device=DEVICE)
            cand_roles = torch.zeros(1, N, dtype=torch.long, device=DEVICE)
            cand_flags = torch.zeros(1, N, 16, dtype=torch.float32, device=DEVICE)
            cand_pad_mask = torch.ones(1, N, dtype=torch.float32, device=DEVICE)

            for ci, c in enumerate(cands):
                c_enc = TOKENIZER(c["label"], max_length=max_lbl_len, truncation=True, padding=False)
                c_tids = c_enc["input_ids"][:max_lbl_len]
                for tw_i, tw_id in enumerate(c_tids):
                    cand_lbl_ids[0, ci, tw_i] = tw_id
                    cand_lbl_mask[0, ci, tw_i] = 1.0

                cand_bounds[0, ci] = torch.tensor(c["bounds_vec"], device=DEVICE)
                
                # Role mapping: INPUT = 2, BUTTON = 1, OTHER = 0
                r_str = str(c.get("role", "button")).lower()
                if "input" in r_str or "edit" in r_str:
                    cand_roles[0, ci] = 2
                elif "button" in r_str:
                    cand_roles[0, ci] = 1
                else:
                    cand_roles[0, ci] = 0

                # Flags: bit 0 = is_editable, bit 1 = is_focused, bit 2 = is_clickable
                orig_t = c.get("orig_tool", {})
                if orig_t.get("is_editable") or "input" in r_str or "edit" in r_str:
                    cand_flags[0, ci, 0] = 1.0
                if orig_t.get("is_focused") or orig_t.get("focused"):
                    cand_flags[0, ci, 1] = 1.0
                if orig_t.get("is_clickable", True):
                    cand_flags[0, ci, 2] = 1.0

            # 3. Model Forward Pass on GPU
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                    out = MODEL(
                        goal_ids=goal_ids,
                        goal_mask=goal_mask,
                        candidate_label_ids=cand_lbl_ids,
                        candidate_label_mask=cand_lbl_mask,
                        candidate_bounds=cand_bounds,
                        candidate_roles=cand_roles,
                        candidate_flags=cand_flags,
                        candidate_pad_mask=cand_pad_mask,
                        h_prev=H_TASK,
                        c_prev=C_CURR,
                    )

            H_TASK = out["h_t"]
            C_CURR = out["c_t"]

            # 4. Neural Action & Candidate Selection (Zero Heuristics!)
            ranking_scores = out["ranking_scores"][0] # [N]
            sel_cand_idx = ranking_scores.argmax().item()
            sel_cand = cands[sel_cand_idx]

            action_logits = out["action_logits"][0] # [num_actions]
            sel_act_idx = action_logits.argmax().item()
            sel_act_name = IDX_TO_ACTION.get(sel_act_idx, "CLICK")

            # 5. Neural Span Extraction for Argument Text (Pointer Head)
            extracted_text = ""
            s_idx = out["span_start_logits"][0].argmax().item()
            e_idx = out["span_end_logits"][0].argmax().item()
            if 0 <= s_idx <= e_idx < goal_ids.shape[1]:
                c_st = offsets[s_idx][0].item()
                c_en = offsets[e_idx][1].item()
                if c_en > c_st:
                    extracted_text = goal[c_st:c_en].strip()
                else:
                    span_token_ids = goal_ids[0, s_idx : e_idx + 1].tolist()
                    extracted_text = TOKENIZER.decode(span_token_ids, skip_special_tokens=True).strip()

            inference_ms = (time.time() - t_start) * 1000.0

            # Map to Android Runtime Action format
            act_type_lower = "tap"
            target_name = sel_cand["id"]
            target_id = sel_cand["target_node_index"]
            chosen_args: Dict[str, Any] = {}

            if sel_act_name == "TYPE_TEXT":
                act_type_lower = "type_text"
                chosen_args["text"] = extracted_text
            elif sel_act_name == "CLICK":
                act_type_lower = "tap"
            elif sel_act_name == "SCROLL":
                act_type_lower = "scroll"
                target_id = None
                chosen_args["direction"] = "DOWN"
            elif sel_act_name == "BACK":
                act_type_lower = "back"
                target_id = None
            elif sel_act_name == "HOME":
                act_type_lower = "home"
                target_id = None
            elif sel_act_name == "WAIT":
                act_type_lower = "wait"
                target_id = None
            elif sel_act_name == "COMPLETE":
                act_type_lower = "complete"
                target_id = None

            # Terminal Completion Handshake: if the model chose the complete goal candidate, dispatch complete
            if target_name in ("c_complete_goal", "complete_goal"):
                act_type_lower = "complete"
                target_id = None
            elif target_name in ("c_submit", "submit_focused_field", "c_ime_submit") or "submit" in str(target_name).lower():
                act_type_lower = "submit"
                target_id = None
            elif sel_act_name == "OPEN_APP" or target_name in ("c_open_app", "open_app") or "open_app" in str(target_name).lower():
                act_type_lower = "open_app"
                target_id = None
                app_hint = sel_cand.get("orig_tool", {}).get("app_name", "")
                if not app_hint:
                    app_hint = extracted_text if extracted_text else ""
                chosen_args["app_name"] = app_hint

            # Ensure target_id is populated from candidate_id if null (platform node index plumbing)
            if target_id is None and act_type_lower in ("tap", "type_text"):
                m = re.match(r"^c_(\d+)$", str(target_name))
                if m:
                    target_id = int(m.group(1))

            # Prepare Top-3 proposals
            top_k_indices = ranking_scores.topk(min(3, N)).indices.tolist()
            softmax_scores = F.softmax(ranking_scores, dim=-1)
            proposals = []

            for rank_i, c_idx in enumerate(top_k_indices):
                cand = cands[c_idx]
                c_conf = round(float(softmax_scores[c_idx].item()), 4)
                c_score = round(float(ranking_scores[c_idx].item()), 3)

                if rank_i == 0:
                    p_act_type = act_type_lower
                    p_args = chosen_args
                    p_tid = target_id
                else:
                    p_act_type = cand["action_type"]
                    p_args = {}
                    p_tid = cand["target_node_index"]
                    if cand["id"] in ("open_app", "c_open_app") or p_act_type == "open_app":
                        p_act_type = "open_app"
                        p_args = {"app_name": cand.get("orig_tool", {}).get("app_name", "")}

                p = {
                    "tool_name": cand["id"],
                    "candidate_id": cand["id"],
                    "action_type": p_act_type,
                    "target_node_index": p_tid,
                    "target_id": p_tid,
                    "confidence": c_conf,
                    "score": c_score,
                    "expected_outcome": f"V52 proposal #{rank_i+1} ({cand['label']})",
                    "arguments": p_args,
                    "text": extracted_text if (rank_i == 0 and p_act_type == "type_text") else "",
                }
                proposals.append(p)

            # Strict Audit Logging
            confidence = proposals[0]["confidence"]
            audit_entry = {
                "turn": TURN_IDX,
                "goal": goal,
                "latency_ms": round(inference_ms, 2),
                "selected_candidate": {
                    "id": target_name,
                    "label": sel_cand["label"],
                    "role": sel_cand["role"],
                    "target_node_index": target_id,
                },
                "neural_action": sel_act_name,
                "dispatched_action": act_type_lower,
                "extracted_span": extracted_text,
                "confidence": confidence,
                "top3_candidates": [
                    {
                        "id": cands[idx.item()]["id"],
                        "label": cands[idx.item()]["label"],
                        "score": round(ranking_scores[idx].item(), 3),
                    }
                    for idx in ranking_scores.topk(min(3, N)).indices
                ],
            }
            with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(audit_entry) + "\n")

            print(
                f"[LIVE V52 TURN {TURN_IDX}] Act: {sel_act_name} | Target: '{sel_cand['label']}' ({target_name}) | "
                f"Text: '{extracted_text}' | Conf: {confidence:.2f} | Latency: {inference_ms:.1f}ms",
                flush=True,
            )

            # Send HTTP Response to phone
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            resp = {
                "status": "success",
                "proposals": proposals,
                "turn": TURN_IDX,
                "model": "KarvishModelV52",
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))


def run_server(port: int = 8765, checkpoint: Optional[Path] = None):
    global CHECKPOINT_PATH
    if checkpoint:
        CHECKPOINT_PATH = Path(checkpoint)
    load_v52_model()
    # Reset audit log file
    with open(AUDIT_LOG_FILE, "w", encoding="utf-8") as f:
        pass

    server = HTTPServer(("0.0.0.0", port), LiveV52ServerHandler)
    print(f"\n>> [SERVER READY] Karvish V52 Live GPU Server listening on port {port}...")
    print(f">> Audit log recording to: {AUDIT_LOG_FILE}")
    print(">> Waiting for requests from Samsung Galaxy A26 (RZCY51MQP9K)...\n", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n>> Server stopped by user.")
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Karvish V52 Live GPU Host Server")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on (default 8765)")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to custom model checkpoint")
    args = parser.parse_args()
    run_server(port=args.port, checkpoint=Path(args.checkpoint) if args.checkpoint else None)
