import xml.etree.ElementTree as ET

root = ET.parse("tools/live_device/chatgpt_curr.xml").getroot()
for i, n in enumerate(root.iter("node")):
    txt = n.attrib.get("text", "").strip()
    cd = n.attrib.get("content-desc", "").strip()
    cls = n.attrib.get("class", "").split(".")[-1]
    clk = n.attrib.get("clickable") == "true"
    edt = "EditText" in cls or n.attrib.get("editable") == "true"
    b = n.attrib.get("bounds")
    if clk or edt or (txt and len(txt) < 40) or cd:
        print(f"[{i:2d}] {cls:<16} | clk={clk} edt={edt} | txt='{txt}' | cd='{cd}' | {b}")
