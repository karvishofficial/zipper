package com.zipper.app.core.voice

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.media.AudioAttributes
import android.media.RingtoneManager
import android.os.Build
import android.os.PowerManager
import android.util.Log
import androidx.core.app.NotificationCompat
import com.zipper.app.R

/**
 * CallReminderReceiver
 *
 * BroadcastReceiver triggered by AlarmManager when a CallReminder fires.
 * Launches the full-screen IncomingCallActivity and issues a high-priority
 * CATEGORY_CALL notification with ringtone audio.
 */
class CallReminderReceiver : BroadcastReceiver() {

    companion object {
        private const val TAG = "CallReminderReceiver"
        const val CHANNEL_ID = "zipper_voice_call_reminders"
        const val NOTIFICATION_ID = 91101
    }

    override fun onReceive(context: Context, intent: Intent) {
        val reminderId = intent.getStringExtra(CallReminderScheduler.EXTRA_REMINDER_ID) ?: return
        val reminder = CallReminderScheduler.getReminder(context, reminderId) ?: return

        if (reminder.isCompleted) {
            Log.i(TAG, "Reminder '$reminderId' already marked complete. Ignoring.")
            return
        }

        val prefs = com.zipper.app.core.intelligence.store.LifeRhythmPrefs(context)
        if (prefs.isCallMutedNow()) {
            Log.w(TAG, "Call reminder '$reminderId' (${reminder.title}) suppressed because Quiet Hours or 'Don't Call Today' is active.")
            return
        }

        Log.i(TAG, "Triggering incoming call for reminder: '${reminder.title}'")

        // 1. Wake the CPU briefly to start the incoming call UI
        val powerManager = context.getSystemService(Context.POWER_SERVICE) as? PowerManager
        val wakeLock = powerManager?.newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK or PowerManager.ACQUIRE_CAUSES_WAKEUP,
            "zipper:CallReminderWakeLock"
        )
        wakeLock?.acquire(15000L) // 15 seconds max

        // 2. Prepare Notification Channel
        val notificationManager = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        createCallNotificationChannel(context, notificationManager)

        // 3. Prepare IncomingCallActivity Intent
        val callIntent = Intent(context, IncomingCallActivity::class.java).apply {
            putExtra(CallReminderScheduler.EXTRA_REMINDER_ID, reminder.id)
            putExtra(IncomingCallActivity.EXTRA_REMINDER_TITLE, reminder.title)
            putExtra(IncomingCallActivity.EXTRA_REMINDER_PROMPT, reminder.spokenPrompt)
            putExtra(IncomingCallActivity.EXTRA_ACTION_TYPE, reminder.actionType.name)
            putExtra(IncomingCallActivity.EXTRA_TARGET_GOAL, reminder.targetGoal)
            reminder.commitmentId?.let { putExtra(IncomingCallActivity.EXTRA_COMMITMENT_ID, it) }
            reminder.category?.let { putExtra(IncomingCallActivity.EXTRA_CATEGORY, it) }
            if (reminder.candidatePackages.isNotEmpty()) {
                putExtra(IncomingCallActivity.EXTRA_CANDIDATE_PACKAGES, reminder.candidatePackages.toTypedArray())
            }
            reminder.sourcePerson?.let { putExtra(IncomingCallActivity.EXTRA_SOURCE_PERSON, it) }
            addFlags(
                Intent.FLAG_ACTIVITY_NEW_TASK or
                Intent.FLAG_ACTIVITY_CLEAR_TOP or
                Intent.FLAG_ACTIVITY_EXCLUDE_FROM_RECENTS
            )
        }

        val fullScreenPendingIntent = PendingIntent.getActivity(
            context,
            reminder.id.hashCode(),
            callIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        // 4. Build Full-Screen Call Notification
        val notification = NotificationCompat.Builder(context, CHANNEL_ID)
            .setSmallIcon(R.mipmap.ic_launcher)
            .setContentTitle("📞 ${reminder.title}")
            .setContentText(reminder.spokenPrompt)
            .setPriority(NotificationCompat.PRIORITY_MAX)
            .setCategory(NotificationCompat.CATEGORY_CALL)
            .setVisibility(NotificationCompat.VISIBILITY_PUBLIC)
            .setOngoing(true)
            .setAutoCancel(true)
            .setFullScreenIntent(fullScreenPendingIntent, true)
            .setContentIntent(fullScreenPendingIntent)
            .build()

        notificationManager.notify(NOTIFICATION_ID, notification)
    }

    private fun createCallNotificationChannel(context: Context, manager: NotificationManager) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val ringtoneUri = RingtoneManager.getDefaultUri(RingtoneManager.TYPE_RINGTONE)
            val audioAttributes = AudioAttributes.Builder()
                .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                .setUsage(AudioAttributes.USAGE_NOTIFICATION_RINGTONE)
                .build()

            val channel = NotificationChannel(
                CHANNEL_ID,
                "Zipper Call-Kind Voice Alerts",
                NotificationManager.IMPORTANCE_HIGH
            ).apply {
                description = "Full-screen incoming phone call alerts for critical reminders"
                setSound(ringtoneUri, audioAttributes)
                enableVibration(true)
                vibrationPattern = longArrayOf(0, 1000, 800, 1000, 800)
                lockscreenVisibility = android.app.Notification.VISIBILITY_PUBLIC
            }
            manager.createNotificationChannel(channel)
        }
    }
}
