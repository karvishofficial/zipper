import xml.etree.ElementTree as ET

tree = ET.parse("tools/live_device/ui_clock.xml")
nodes = tree.findall(".//node")
for i, n in enumerate(nodes):
    clickable = n.attrib.get("clickable") == "true"
    text = n.attrib.get("text", "")
    desc = n.attrib.get("content-desc", "")
    res_id = n.attrib.get("resource-id", "")
    if clickable or text or desc:
        print(f"Node #{i}: text='{text}', desc='{desc}', id='{res_id}', clickable={clickable}")
