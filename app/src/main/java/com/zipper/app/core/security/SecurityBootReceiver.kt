package com.zipper.app.core.security

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log

/**
 * High-priority Boot BroadcastReceiver that ensures Zipper Anti-Theft Protection
 * survives physical phone reboots.
 *
 * If the device restarts while Stealth / Mimic mode is active:
 * 1. Forcibly re-engages GPS, Wi-Fi, Mobile Data, and Bluetooth.
 * 2. Re-pins the Fake Shutdown Blackout screen over the lockscreen.
 * 3. Captures an immediate mugshot of whoever booted the phone and streams GPS to Firebase & Email.
 */
class SecurityBootReceiver : BroadcastReceiver() {

    companion object {
        private const val TAG = "SecurityBootReceiver"
    }

    override fun onReceive(context: Context, intent: Intent) {
        val action = intent.action ?: return
        if (action == Intent.ACTION_BOOT_COMPLETED || 
            action == Intent.ACTION_LOCKED_BOOT_COMPLETED ||
            action == "android.intent.action.QUICKBOOT_POWERON"
        ) {
            Log.i(TAG, "Device reboot completed ($action). Checking Zipper Security State...")

            val isStealthActive = SecurityConfig.isStealthModeActive(context)
            val isGuardEnabled = SecurityConfig.isGuardEnabled(context)

            if (isStealthActive || isGuardEnabled) {
                Log.w(TAG, "🚨 REBOOT OCCURRED WHILE SECURITY ACTIVE! Re-enforcing hardware radios and tracking telemetry!")

                // 1. Force all communication radios ON immediately
                HardwareEnforcer.enforceStealthConnectivity(context)

                if (isStealthActive) {
                    // 2. Re-pin blackout fake shutdown screen
                    FakeShutdownOverlayActivity.start(context)

                    // 3. Trigger immediate camera snap of whoever booted the phone and resume Firebase tracking
                    IntruderCaptureService.start(context, "DEVICE_REBOOT_DETECTED")
                }
            }
        }
    }
}
