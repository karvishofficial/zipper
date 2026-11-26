import subprocess
import time
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

adb_exe = os.path.expandvars(r"%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe")
device_id = "RZCY51MQP9K"

def run_adb(args):
    cmd = [adb_exe, "-s", device_id] + args
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")

def run_test(goal, app_pkg_to_kill=None, use_remote=True):
    print(f"\n========================================================")
    print(f"STARTING LIVE TEST: Goal = '{goal}' (Remote GPU Ranker = {use_remote})")
    print(f"========================================================")
    
    # 0. Preflight GPU Server Check (only if using remote ranker)
    if use_remote:
        try:
            import urllib.request
            with urllib.request.urlopen("http://localhost:8765/health", timeout=3) as resp:
                health_data = resp.read().decode()
                print(f">> [PRE-FLIGHT] Host GPU Ranker Server is READY: {health_data.strip()}")
        except Exception as e:
            print(f">> [PRE-FLIGHT ERROR] Cannot reach GPU Ranker Server on http://localhost:8765/health: {e}")
            print(">> Make sure live_ranker_server.py is running on the host GPU!")
            return

        # 1. Establish Port Forwarding
        print(">> Setting up reverse ADB port 8765...")
        run_adb(["reverse", "tcp:8765", "tcp:8765"])
    else:
        print(">> [OFFLINE MODE] Using Pure On-Device Karvish Model V46.1 ONNX Engine (<50 MB) on device CPU!")

    # 2. Clear state and ensure clean Home screen
    run_adb(["shell", "input keyevent 224"]) # Keycode WAKEUP
    run_adb(["shell", "input keyevent 82"])  # Keycode UNLOCK
    run_adb(["shell", "input keyevent 3"])   # Home
    if app_pkg_to_kill:
        print(f">> Resetting test state for {app_pkg_to_kill}...")
        run_adb(["shell", f"am force-stop {app_pkg_to_kill}"])
    run_adb(["shell", "am force-stop com.android.settings"])
    run_adb(["shell", "am force-stop com.instagram.android"])
    run_adb(["shell", "settings put secure enabled_accessibility_services com.zipper.app/com.zipper.app.accessibility.KarvishAccessibilityService"])
    run_adb(["shell", "settings put secure accessibility_enabled 1"])
    run_adb(["shell", "input keyevent 3"]) # Home
    run_adb(["shell", "input keyevent 3"]) # Home twice to clear overlays
    time.sleep(1)

    # 3. Verify Accessibility Service is active
    a11y_dump = run_adb(["shell", "dumpsys accessibility"]).stdout
    if "KarvishAccessibilityService" not in a11y_dump:
        print(">> [WARNING] KarvishAccessibilityService is not showing as enabled in dumpsys!")
    else:
        print(">> [PRE-FLIGHT] KarvishAccessibilityService is ACTIVE on device.")

    # 4. Get current timestamp for logcat
    t_res = run_adb(["shell", "date '+%m-%d %H:%M:%S.000'"])
    start_time = t_res.stdout.strip()
    print(f">> Device test start timestamp: {start_time}")

    # 5. Launch via am start with properly quoted goal and remote ranker flag
    remote_flag = "--ez use_remote_ranker true" if use_remote else "--ez use_remote_ranker false"
    launch_cmd = f"am start -n com.zipper.app/.MainActivity --es run_agent_goal \"{goal}\" --ez enable_active_recovery true {remote_flag}"
    print(f">> Running: {launch_cmd}")
    l_res = run_adb(["shell", launch_cmd])
    print(f">> Launch output: {l_res.stdout.strip()}")

    # 4. Monitor logcat for completion or up to 100 seconds
    print("Monitoring execution...")
    finished = False
    for i in range(50):
        time.sleep(2)
        log_res = run_adb(["logcat", "-d", "-v", "time", "-t", start_time, "AgentRuntimeService:V", "KarvishAgentRuntime:V", "OnDeviceRankerEngine:V", "LiveRankerModelEngine:V", "*:S"])
        out = log_res.stdout or ""
        if any(marker in out for marker in ["Goal finished:", "Early termination", "AUTONOMOUS GOAL ACHIEVED", "TERMINATING", "Completed:", "Stopped:"]):
            print(f"Task finished in ~{(i+1)*2}s!")
            finished = True
            break
        print(f"  ... turn progress (elapsed {(i+1)*2}s)")
    
    # 5. Capture final screen
    time.sleep(1)
    safe_name = goal.replace(" ", "_").replace("'", "").replace('"', "")[:20].lower()
    out_dir = os.path.join(os.path.dirname(__file__), "physical_test_output")
    os.makedirs(out_dir, exist_ok=True)
    screen_path = os.path.join(out_dir, f"{safe_name}_screen.png")
    img_data = subprocess.check_output([adb_exe, "-s", device_id, "exec-out", "screencap", "-p"])
    with open(screen_path, "wb") as f:
        f.write(img_data)
    print(f"Saved screenshot to {screen_path} ({len(img_data)} bytes)")

    # 6. Dump all logs for this test
    dump_res = run_adb(["logcat", "-d", "-v", "time", "-t", start_time, 
                        "AgentRuntimeService:V", "KarvishAgentRuntime:V", "KarvishA11yService:V", "OnDeviceRankerEngine:V", "LiveRankerModelEngine:V", "*:S"])
    log_file = os.path.join(out_dir, f"{safe_name}_trace.log")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(dump_res.stdout)
    print(f"Saved full audit trace to {log_file}")
    return log_file, screen_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Karvish live test on device")
    parser.add_argument("goal", help="Agent natural language goal")
    parser.add_argument("pkg_to_kill", nargs="?", default=None, help="Optional package to kill prior to test")
    parser.add_argument("--offline", action="store_true", help="Run purely on-device using ONNX model without GPU bridge")
    args = parser.parse_args()

    use_remote = not args.offline
    run_test(args.goal, args.pkg_to_kill, use_remote=use_remote)
