"""
Converts Karvish TrainingExample JSONL (from AndroidControl)
into Cactus Needle 2 training format (V3):
{
  "query": "Goal: ... | Screen Context ...",
  "tools": [... dynamic JSON schema tools ...],
  "answers": [{"name": "...", "arguments": {...}}],
  "reasoning": null
}

V3 improvements over V2:
  - resourceId extraction: elements without text/contentDescription get labeled from resourceId
  - Spatial position tags: each element description includes position (top-bar/upper/center/lower/bottom-nav)
  - scroll_left/scroll_right: complete scroll direction schema
  - Gboard keyboard filtering: filter individual soft-key tools when keyboard package is active
  - wait tool always present
  - No truncation of any field
"""

import json
import os
import re


def clean_slug(raw_label, max_len=16):
    """Create a clean tool-name slug from a label."""
    cleaned = ''.join(c if c.isalnum() else '_' for c in raw_label.lower())
    cleaned = re.sub(r'_+', '_', cleaned).strip('_')
    return cleaned[:max_len].strip('_') or 'element'


def get_spatial_tag(bounds, max_x, max_y):
    """Compute spatial position tag from node bounds."""
    if not bounds or max_y <= 0:
        return ""
    top = bounds.get("top", 0)
    bottom = bounds.get("bottom", max_y)
    cy = (top + bottom) / 2.0
    rel_y = cy / max(max_y, 1)
    
    if rel_y < 0.15:
        return "top-bar"
    elif rel_y < 0.40:
        return "upper"
    elif rel_y < 0.70:
        return "center"
    elif rel_y < 0.88:
        return "lower"
    else:
        return "bottom-nav"


def get_label_from_node(node):
    """Extract the best available label for a node, falling back to resourceId."""
    text = node.get('text')
    desc = node.get('contentDescription')
    res_id = node.get('resourceId')
    role = node.get('role')
    idx = node.get('index', 0)
    
    # Priority: text > contentDescription > resourceId short name > role > generic
    if text and text.strip():
        return text.strip()
    if desc and desc.strip():
        return desc.strip()
    if res_id and '/' in res_id:
        # Extract short name: "com.app:id/btn_search" -> "btn search"
        short_id = res_id.split('/')[-1]
        return short_id.replace('_', ' ')
    if role:
        return role
    return f'item_{idx}'


def is_keyboard_soft_key(node, package_name):
    """Check if a node is an individual keyboard soft key (single char button)."""
    if package_name != 'com.google.android.inputmethod.latin':
        return False
    text = node.get('text', '')
    role = node.get('role', '')
    # Single-character buttons on Gboard are soft keys
    if role == 'Button' and len(text or '') == 1:
        return True
    return False


def format_record_for_needle(rec):
    goal = rec['goal']
    history = rec.get('shortHistory', [])
    hist_str = ''
    if history:
        steps_summary = [f"Step {h['stepIndex']}: {h['actionSummary']}" for h in history]
        hist_str = ' | History: ' + '; '.join(steps_summary)
    
    pkg = rec.get('packageName', '')
    query = f"Goal: {goal} | Current App: {pkg}{hist_str}"

    nodes = rec['currentObservation']['nodes']
    
    # Compute screen dimensions for spatial tagging
    max_y = max((n.get('bounds', {}).get('bottom', 0) for n in nodes), default=2400) or 2400
    max_x = max((n.get('bounds', {}).get('right', 0) for n in nodes), default=1080) or 1080

    tools = []
    # 1. open_app primitive
    tools.append({
        'name': 'open_app',
        'description': 'Launch any installed Android application by its common name, e.g. Calculator, Clock, Settings, Camera, Messages, Chrome, Calendar.',
        'parameters': {
            'type': 'object',
            'properties': {'app_name': {'type': 'string', 'description': 'The name of the application to open'}},
            'required': ['app_name']
        }
    })

    # Track ground-truth target to ensure it's never filtered out
    act_name = rec['targetActionName']
    act_args = rec.get('targetActionArguments', {})
    gt_target_id = None
    if 'target' in act_args:
        try:
            gt_target_id = int(act_args['target'])
        except (ValueError, TypeError):
            pass

    for n in nodes:
        idx = n['index']
        
        # Filter keyboard soft keys (unless this node is the ground-truth target)
        if is_keyboard_soft_key(n, pkg) and idx != gt_target_id:
            continue
        
        raw_label = get_label_from_node(n)
        slug = clean_slug(raw_label)
        
        # Add spatial position tag
        bounds = n.get('bounds', {})
        spatial = get_spatial_tag(bounds, max_x, max_y)
        pos_str = f" [{spatial}]" if spatial else ""

        if n.get('isEditable'):
            tools.append({
                'name': f'type_{slug}_{idx}',
                'description': f"Type text into [{n['role']}] '{raw_label}'{pos_str} on screen (element #{idx})",
                'parameters': {
                    'type': 'object',
                    'properties': {'text': {'type': 'string', 'description': 'Text or numbers to type'}},
                    'required': ['text']
                }
            })
        elif n.get('isClickable'):
            tools.append({
                'name': f'tap_{slug}_{idx}',
                'description': f"Tap [{n['role']}] '{raw_label}'{pos_str} on screen (element #{idx})",
                'parameters': {'type': 'object', 'properties': {}}
            })

    # Navigation primitives (all 4 scroll directions)
    tools.append({'name': 'scroll_down', 'description': 'Scroll down to reveal elements currently below the screen', 'parameters': {'type': 'object', 'properties': {}}})
    tools.append({'name': 'scroll_up', 'description': 'Scroll up to reveal elements currently above the screen', 'parameters': {'type': 'object', 'properties': {}}})
    tools.append({'name': 'scroll_left', 'description': 'Scroll left to reveal elements to the left', 'parameters': {'type': 'object', 'properties': {}}})
    tools.append({'name': 'scroll_right', 'description': 'Scroll right to reveal elements to the right', 'parameters': {'type': 'object', 'properties': {}}})
    tools.append({'name': 'press_back', 'description': 'Navigate back to the previous screen or dismiss dialog', 'parameters': {'type': 'object', 'properties': {}}})
    tools.append({'name': 'navigate_home', 'description': 'Navigate to the device home screen', 'parameters': {'type': 'object', 'properties': {}}})
    tools.append({'name': 'wait', 'description': 'Wait for ongoing UI transition or content to load', 'parameters': {'type': 'object', 'properties': {}}})
    tools.append({
        'name': 'complete_goal',
        'description': f"Call this tool ONLY when the user's goal ('{goal}') is visibly achieved on screen",
        'parameters': {
            'type': 'object',
            'properties': {'summary': {'type': 'string', 'description': 'Explanation of how the goal was achieved'}},
            'required': ['summary']
        }
    })

    # Ground-truth target action mapping with 100% schema completeness
    target_tool = None

    if act_name == 'open_app':
        target_tool = {'name': 'open_app', 'arguments': {'app_name': act_args.get('appName', '')}}
    elif act_name == 'tap':
        t_id = int(act_args.get('target', 0))
        matched = [t['name'] for t in tools if t['name'].endswith(f'_{t_id}')]
        if matched:
            target_tool = {'name': matched[0], 'arguments': {}}
        else:
            # Ground-truth target element was not marked clickable; add explicit tool to guarantee schema completeness
            target_node = next((n for n in nodes if n.get('index') == t_id), None)
            if target_node:
                raw_label = get_label_from_node(target_node)
                slug = clean_slug(raw_label)
                tool_name = f'tap_{slug}_{t_id}'
                bounds = target_node.get('bounds', {})
                spatial = get_spatial_tag(bounds, max_x, max_y)
                pos_str = f" [{spatial}]" if spatial else ""
                desc = f"Tap [{target_node.get('role', 'View')}] '{raw_label}'{pos_str} on screen (element #{t_id})"
            else:
                tool_name = f'tap_element_{t_id}'
                desc = f"Tap on screen element #{t_id}"
            new_tool = {'name': tool_name, 'description': desc, 'parameters': {'type': 'object', 'properties': {}}}
            tools.append(new_tool)
            target_tool = {'name': tool_name, 'arguments': {}}
    elif act_name == 'type_text':
        t_id = int(act_args.get('target', 0)) if 'target' in act_args else None
        matched = [t['name'] for t in tools if t['name'].startswith('type_') and (t_id is None or t['name'].endswith(f'_{t_id}'))]
        if matched:
            target_tool = {'name': matched[0], 'arguments': {'text': act_args.get('text', '')}}
        else:
            target_node = next((n for n in nodes if n.get('index') == t_id), None) if t_id else None
            if target_node:
                raw_label = get_label_from_node(target_node)
                slug = clean_slug(raw_label)
                tool_name = f'type_{slug}_{t_id}'
                bounds = target_node.get('bounds', {})
                spatial = get_spatial_tag(bounds, max_x, max_y)
                pos_str = f" [{spatial}]" if spatial else ""
                desc = f"Type text into [{target_node.get('role', 'Input')}] '{raw_label}'{pos_str} on screen (element #{t_id})"
            else:
                tool_name = f'type_element_{t_id}' if t_id else 'type_text'
                desc = f"Type text into screen element #{t_id if t_id else ''}"
            new_tool = {
                'name': tool_name,
                'description': desc,
                'parameters': {
                    'type': 'object',
                    'properties': {'text': {'type': 'string', 'description': 'Text or numbers to type'}},
                    'required': ['text']
                }
            }
            tools.append(new_tool)
            target_tool = {'name': tool_name, 'arguments': {'text': act_args.get('text', '')}}
    elif act_name == 'scroll':
        dir_val = act_args.get('direction', 'down')
        tool_name = f"scroll_{dir_val}"
        if not any(t['name'] == tool_name for t in tools):
            tools.append({'name': tool_name, 'description': f'Scroll {dir_val}', 'parameters': {'type': 'object', 'properties': {}}})
        target_tool = {'name': tool_name, 'arguments': {}}
    elif act_name in ('back', 'press_back'):
        target_tool = {'name': 'press_back', 'arguments': {}}
    elif act_name in ('home', 'navigate_home'):
        target_tool = {'name': 'navigate_home', 'arguments': {}}
    elif act_name == 'wait':
        target_tool = {'name': 'wait', 'arguments': {}}
    elif act_name in ('complete', 'finish'):
        target_tool = {'name': 'complete_goal', 'arguments': {'summary': 'Goal achieved'}}
    else:
        if not any(t['name'] == act_name for t in tools):
            tools.append({'name': act_name, 'description': f'Action {act_name}', 'parameters': {'type': 'object', 'properties': {}}})
        target_tool = {'name': act_name, 'arguments': act_args}

    return {
        'query': query,
        'tools': tools,
        'answers': [target_tool],
        'reasoning': None  # No artificial <think> reasoning
    }


def convert_file(src_path, dst_path, max_count=None):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    count = 0
    skipped = 0
    with open(src_path, 'r', encoding='utf-8') as fin, open(dst_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            try:
                converted = format_record_for_needle(rec)
                fout.write(json.dumps(converted, ensure_ascii=False) + '\n')
                count += 1
            except Exception as e:
                skipped += 1
                if skipped <= 5:
                    print(f"  WARNING: Skipped record (error: {e})")
            if max_count and count >= max_count:
                break
    print(f"Converted {count} records from {src_path} -> {dst_path} (skipped: {skipped})")
    return count


if __name__ == '__main__':
    # 1. Sanity subset (first 50 examples)
    convert_file('tools/training/data/train.jsonl', 'tools/training/experiment_c_needle2/data/sanity_50_v3.jsonl', max_count=50)
    # 2. Full training set
    convert_file('tools/training/data/train.jsonl', 'tools/training/experiment_c_needle2/data/needle_train_v3.jsonl')
    # 3. Held-out validation set (77 examples) — FROZEN, same data, V3 format
    convert_file('tools/training/data/val_unseen.jsonl', 'tools/training/experiment_c_needle2/data/needle_val_v3.jsonl')
    print("\nV3 conversion complete.")
