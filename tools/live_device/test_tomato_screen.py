import json
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

# Parse current_tomato.xml into tools JSON format as live_v52_gpu_server expects
tree = ET.parse("tools/live_device/current_tomato.xml")

candidates = []
node_idx = 0
for node in tree.iter("node"):
    c = node.attrib.get("class", "")
    cd = node.attrib.get("content-desc", "")
    txt = node.attrib.get("text", "")
    rid = node.attrib.get("resource-id", "")
    clickable = node.attrib.get("clickable", "false") == "true"
    bounds_str = node.attrib.get("bounds", "[0,0][0,0]")

    label = (cd or txt).strip()
    if not label and not clickable:
        continue
    
    # parse bounds [l,t][r,b]
    try:
        parts = bounds_str.replace("][", ",").replace("[", "").replace("]", "").split(",")
        l, t, r, b = [int(p) for p in parts]
    except Exception:
        l, t, r, b = 0, 0, 0, 0

    if (r - l) <= 10 or (b - t) <= 10:
        continue

    role = "button" if "Button" in c or clickable else "text"
    cid = f"c_{node_idx}"
    node_idx += 1

    cand = {
        "candidate_id": cid,
        "semantic_label": label[:100],
        "role": role,
        "target_node_index": node_idx,
        "bounds": {"left": l, "top": t, "right": r, "bottom": b},
        "description": f"[{role}] '{label}'"
    }
    candidates.append(cand)

# Add system fallback tools
candidates.append({"candidate_id": "c_complete_goal", "semantic_label": "complete", "role": "button", "target_node_index": None, "bounds": {"left": 0, "top": 0, "right": 1080, "bottom": 100}, "description": "Declare complete"})
candidates.append({"candidate_id": "c_scroll_down", "semantic_label": "scroll down", "role": "system", "target_node_index": None, "bounds": {"left": 0, "top": 0, "right": 1080, "bottom": 100}, "description": "Scroll down"})

req_body = {
    "goal": "Open Zepto and add the cheapest tomato to the cart",
    "turn": 5,
    "tools": candidates
}

req_json = json.dumps(req_body).encode("utf-8")
req = urllib.request.Request("http://localhost:8765/propose", data=req_json, headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req) as resp:
    result = json.loads(resp.read().decode())

print("=== V52 MODEL PREDICTION ON TOMATO SEARCH SCREEN ===")
proposals = result.get("proposals", [])
for p in proposals[:5]:
    print(f"Tool: {p.get('tool_name')} | Action: {p.get('action_type')} | Score: {p.get('score')} | Label: {p.get('expected_outcome')}")

