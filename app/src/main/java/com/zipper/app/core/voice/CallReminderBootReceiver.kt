package com.zipper.app.core.voice

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log

/**
 * CallReminderBootReceiver
 *
 * Restores all pending Call-Kind Voice Reminders when device reboots or
 * app is updated.
 */
class CallReminderBootReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        val action = intent.action
        if (action == Intent.ACTION_BOOT_COMPLETED || action == Intent.ACTION_MY_PACKAGE_REPLACED) {
            Log.i("CallReminderBoot", "Restoring scheduled reminders on $action")
            CallReminderScheduler.restoreAllAlarms(context)
        }
    }
}
