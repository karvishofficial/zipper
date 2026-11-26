import urllib.request
import json

data = {
    "prompt": "Open Calculator and calculate 45 + 55",
    "tools": [
        {"name": "icon_calculator", "description": "Calculator", "action_type": "open_app", "semantic_label": "Calculator"},
        {"name": "icon_clock", "description": "Clock", "action_type": "open_app", "semantic_label": "Clock"},
        {"name": "icon_calendar", "description": "Calendar", "action_type": "open_app", "semantic_label": "Calendar"}
    ],
    "observation": {"package_name": "com.android.launcher", "nodes": []},
    "top_k": 3
}

req = urllib.request.Request("http://localhost:8765/propose", data=json.dumps(data).encode("utf-8"), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req) as resp:
    res = json.loads(resp.read().decode("utf-8"))

print(f"Status: {res.get('status')} | Latency: {res.get('latency_ms')} ms")
for i, p in enumerate(res.get("proposals", [])):
    print(f"  #{i+1}: {p.get('candidate_id')} ({p.get('action_type')}) | score={p.get('score')} | conf={p.get('confidence')} | progress={p.get('task_progress')}")
