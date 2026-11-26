package com.zipper.app.core.security

import android.app.admin.DeviceAdminReceiver
import android.app.admin.DevicePolicyManager
import android.content.Context
import android.content.Intent
import android.util.Log

/**
 * Device Administrator Receiver that intercepts lock screen authentication events.
 * Specifically monitors failed password/PIN/pattern attempts to catch intruders in real time.
 */
class ZipperDeviceAdminReceiver : DeviceAdminReceiver() {

    companion object {
        private const val TAG = "ZipperDeviceAdmin"
    }

    @Suppress("DEPRECATION")
    @Deprecated("Deprecated in Java")
    override fun onPasswordFailed(context: Context, intent: Intent) {
        super.onPasswordFailed(context, intent)

        if (!SecurityConfig.isGuardEnabled(context)) {
            Log.d(TAG, "Security guard disabled; ignoring failed password attempt")
            return
        }

        val dpm = context.getSystemService(Context.DEVICE_POLICY_SERVICE) as? DevicePolicyManager
        val attempts = dpm?.currentFailedPasswordAttempts ?: 1
        val threshold = SecurityConfig.getFailedAttemptThreshold(context)

        Log.w(TAG, "Lock screen password attempt failed! Current count: $attempts (Threshold: $threshold)")

        if (attempts >= threshold) {
            Log.i(TAG, "Threshold reached! Triggering intruder mugshot capture and location telemetry.")
            IntruderCaptureService.start(context, "WRONG_LOCKSCREEN_PIN (Attempt #$attempts)")
        }
    }

    @Suppress("DEPRECATION")
    @Deprecated("Deprecated in Java")
    override fun onPasswordSucceeded(context: Context, intent: Intent) {
        super.onPasswordSucceeded(context, intent)
        Log.d(TAG, "Lock screen password succeeded - owner unlocked device.")
    }

    override fun onEnabled(context: Context, intent: Intent) {
        super.onEnabled(context, intent)
        Log.i(TAG, "Zipper Device Administrator activated successfully!")
    }

    override fun onDisabled(context: Context, intent: Intent) {
        super.onDisabled(context, intent)
        Log.w(TAG, "Zipper Device Administrator deactivated.")
    }
}
