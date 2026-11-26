import os
import subprocess
import sys
from pathlib import Path

adb = os.path.expandvars(r'%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe')
device = "RZCY51MQP9K"

def run_cmd(cmd):
    res = subprocess.run([adb, "-s", device] + cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    return res.stdout.strip()

print("Current Focus Activity:")
focus = run_cmd(["shell", "dumpsys", "window", "displays"])
for line in focus.splitlines():
    if "mCurrentFocus" in line or "mFocusedApp" in line:
        print(" ", line.strip())

print("\nCapturing screenshot to tools/live_device/v56_live_audit_reports/screen_now.png...")
out_img = Path("tools/live_device/v56_live_audit_reports/screen_now.png")
res_img = subprocess.run([adb, "-s", device, "exec-out", "screencap", "-p"], capture_output=True)
if res_img.stdout:
    out_img.write_bytes(res_img.stdout)
    print(f"Screenshot saved ({len(res_img.stdout)} bytes).")
else:
    print("Failed to capture screenshot.")

print("\nRecent zipper logcat messages (last 200 lines):")
logs = subprocess.run([adb, "-s", device, "logcat", "-d", "-t", "200"], capture_output=True, text=True, encoding='utf-8', errors='replace')
for line in logs.stdout.splitlines():
    if any(k in line for k in ["Zipper", "Karvish", "AgentRuntime", "Ranker", "ExternalOracle", "ACTION", "GOAL", "PICKED", "TURN"]):
        print(" ", line)
