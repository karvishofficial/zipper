package com.zipper.app.core.security

import android.app.admin.DevicePolicyManager
import android.content.BroadcastReceiver
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.util.Log

/**
 * BroadcastReceiver that triggers an immediate panic protocol when a thief ejects
 * the physical SIM card tray.
 */
class SimStateReceiver : BroadcastReceiver() {

    companion object {
        private const val TAG = "SimStateReceiver"
    }

    override fun onReceive(context: Context, intent: Intent) {
        val action = intent.action ?: return
        if (action == "android.intent.action.SIM_STATE_CHANGED" ||
            action == "android.intent.action.SIM_CARD_STATE_CHANGED") {

            val state = intent.getStringExtra("ss") ?: ""
            Log.d(TAG, "SIM state event received: $state")

            if (state.equals("ABSENT", ignoreCase = true) || state.equals("UNKNOWN", ignoreCase = true)) {
                if (!SecurityConfig.isGuardEnabled(context) || !SecurityConfig.isTriggerOnSimEjectEnabled(context)) {
                    Log.d(TAG, "SIM eject trigger disabled in config; skipping")
                    return
                }

                Log.w(TAG, "🚨 PHYSICAL SIM CARD REMOVAL DETECTED! Starting immediate intruder telemetry!")

                // 1. Instantly lock the screen so intruder cannot navigate anywhere
                runCatching {
                    val dpm = context.getSystemService(Context.DEVICE_POLICY_SERVICE) as? DevicePolicyManager
                    val adminComponent = ComponentName(context, ZipperDeviceAdminReceiver::class.java)
                    if (dpm != null && dpm.isAdminActive(adminComponent)) {
                        dpm.lockNow()
                        Log.i(TAG, "Screen locked immediately upon SIM ejection")
                    }
                }

                // 2. Launch headless camera & GPS capture burst
                IntruderCaptureService.start(context, "SIM_CARD_REMOVED")
            }
        }
    }
}
