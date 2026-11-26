package com.zipper.app.core.intelligence.commute

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log
import android.widget.Toast
import com.zipper.app.core.remote.EmergencyRinger

/**
 * Handles tactile 1-tap responses from the Commute Safety Guard notification HUD.
 */
class CommuteSafetyReceiver : BroadcastReceiver() {

    companion object {
        private const val TAG = "CommuteSafetyReceiver"
    }

    override fun onReceive(context: Context?, intent: Intent?) {
        if (context == null || intent == null) return

        when (intent.action) {
            CommuteSafetyGuard.ACTION_SAFE_ARRIVED -> {
                Log.i(TAG, "Safe arrival button tapped.")
                EmergencyRinger.stopRinging(context)
                val msg = CommuteSafetyGuard.markSafeArrival(context)
                Toast.makeText(context, msg, Toast.LENGTH_SHORT).show()
            }
            CommuteSafetyGuard.ACTION_TRIGGER_SOS -> {
                Log.w(TAG, "Manual SOS button tapped.")
                val msg = CommuteSafetyGuard.triggerSos(context, "1-Tap SOS Notification Tap")
                Toast.makeText(context, "🚨 SOS Alert Dispatched!", Toast.LENGTH_LONG).show()
            }
        }
    }
}
