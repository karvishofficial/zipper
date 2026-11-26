import sys, json, xml.etree.ElementTree as ET, re
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

tree = ET.parse("tools/live_device/scrolled_tomato.xml")
root = tree.getroot()

nodes = []
for idx, el in enumerate(root.iter("node")):
    b_str = el.get("bounds", "")
    m = re.findall(r"\[(\d+),(\d+)\]", b_str)
    if len(m) == 2:
        bounds = {
            "left": float(m[0][0])/1080.0,
            "top": float(m[0][1])/2340.0,
            "right": float(m[1][0])/1080.0,
            "bottom": float(m[1][1])/2340.0,
        }
    else:
        bounds = {"left": 0, "top": 0, "right": 1, "bottom": 1}
    nodes.append({
        "id": idx,
        "resource_hint": el.get("resource-id", ""),
        "content_description": el.get("content-desc", ""),
        "text": el.get("text", ""),
        "bounds": bounds,
    })

add_buttons = [n for n in nodes if "add-button" in n["resource_hint"] and not n["resource_hint"].endswith("-minus") and not n["resource_hint"].endswith("-plus")]

print(f"Found {len(add_buttons)} add buttons")
for btn in add_buttons:
    bl, bt, br, bb = btn["bounds"]["left"], btn["bounds"]["top"], btn["bounds"]["right"], btn["bounds"]["bottom"]
    enclosing = [
        n for n in nodes 
        if n["id"] != btn["id"] 
        and n["bounds"]["left"] <= bl + 0.05 
        and n["bounds"]["top"] <= bt + 0.05 
        and n["bounds"]["right"] >= br - 0.05 
        and n["bounds"]["bottom"] >= bb - 0.05 
        and (n["content_description"] or n["text"])
        and ("₹" in n["content_description"] or "Tomato" in n["content_description"] or "product-card" in n["resource_hint"])
    ]
    if enclosing:
        enclosing.sort(key=lambda n: (n["bounds"]["right"]-n["bounds"]["left"]) * (n["bounds"]["bottom"]-n["bounds"]["top"]))
        card = enclosing[0]
        desc = re.sub(r"[\u200a-\u200f\ufeff]", "", card["content_description"] or card["text"])
        print(f"Btn {btn['id']} -> {desc} Add to Cart")
