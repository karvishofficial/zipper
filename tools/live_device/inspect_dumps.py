import sys
import xml.etree.ElementTree as ET

sys.stdout.reconfigure(encoding='utf-8')

apps = [
    ('ChatGPT', 'tools/live_device/chatgpt_dump.xml'),
    ('Gemini', 'tools/live_device/gemini_dump.xml'),
]

for name, path in apps:
    print(f"\n{'='*25} {name} {'='*25}")
    try:
        tree = ET.parse(path)
        count = 0
        for node in tree.iter('node'):
            txt = node.attrib.get('text', '')
            cd = node.attrib.get('content-desc', '')
            rid = node.attrib.get('resource-id', '')
            c = node.attrib.get('class', '')
            clickable = node.attrib.get('clickable', '')
            bounds = node.attrib.get('bounds', '')
            label = (cd or txt).strip()
            if label or clickable == 'true':
                short_c = c.split('.')[-1]
                short_rid = rid.split('/')[-1] if rid else ''
                print(f"[{short_c:12s}] label='{label[:40]:40s}' | rid='{short_rid:25s}' | click={clickable:5s} | bounds={bounds}")
                count += 1
        print(f"Total interactive/labeled nodes: {count}")
    except Exception as e:
        print(f"Error parsing {path}: {e}")
