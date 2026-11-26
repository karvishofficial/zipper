import xml.etree.ElementTree as ET
import re

tree = ET.parse('tools/live_device/scrolled_tomato.xml')
root = tree.getroot()
cands = []
for idx, el in enumerate(root.iter('node')):
    txt = el.get('text', '').strip()
    desc = el.get('content-desc', '').strip()
    res = el.get('resource-id', '').strip()
    cls = el.get('class', '').strip()
    clickable = el.get('clickable') == 'true'
    editable = el.get('editable') == 'true' or 'EditText' in cls
    bounds_str = el.get('bounds', '')
    lbl = txt if txt else desc
    if not lbl and not clickable and not editable:
        continue
    res_short = res.split('/')[-1] if res else ''
    full_lbl = lbl if lbl else res_short
    cands.append({'id': f'node_{idx}', 'label': full_lbl, 'res': res_short, 'clickable': clickable, 'bounds': bounds_str})

cheap = [c for c in cands if '973feaa' in c['res']]
for c in cheap:
    clean = f"{c['id']} | {c['res']} | {c['label']} | {c['bounds']}".encode('ascii', errors='replace').decode()
    print(clean)
