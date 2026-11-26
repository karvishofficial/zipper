import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

with open("tools/live_device/device_full_logcat.txt", "r", encoding="latin1", errors="ignore") as f:
    lines = f.readlines()

in_g1 = False
for line in lines:
    if "open WhatsApp and send hi to Ajay" in line:
        in_g1 = True
    if "Open Zepto and search fresh tomatoes" in line:
        in_g1 = False
    if in_g1:
        if any(k in line for k in ["KarvishAgentService:", "KarvishAgentRuntime:", "OnDeviceRankerEngine:", "ExternalOracle"]):
            print(line.strip())
