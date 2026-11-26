package com.zipper.app.core.voice

import android.app.AlarmManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import android.os.Build
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.util.Calendar
import java.util.UUID

/**
 * CallReminderScheduler
 *
 * Manages reliable offline scheduling, snooze cycles, and persistence of
 * call-kind voice reminders across reboots using Android's exact alarm subsystem.
 */
object CallReminderScheduler {
    private const val TAG = "CallReminderScheduler"
    private const val PREFS_NAME = "zipper_call_reminders"
    private const val KEY_REMINDERS_JSON = "reminders_list"

    const val EXTRA_REMINDER_ID = "com.zipper.app.EXTRA_REMINDER_ID"

    private fun getPrefs(context: Context): SharedPreferences {
        return context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    }

    /**
     * Schedules a call reminder with the system AlarmManager and saves it locally.
     */
    fun schedule(context: Context, reminder: CallReminder) {
        val all = getAllReminders(context).toMutableList()
        all.removeAll { it.id == reminder.id }
        all.add(reminder)
        saveReminders(context, all)

        val alarmManager = context.getSystemService(Context.ALARM_SERVICE) as? AlarmManager ?: return
        val pendingIntent = createAlarmPendingIntent(context, reminder)

        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                alarmManager.setExactAndAllowWhileIdle(
                    AlarmManager.RTC_WAKEUP,
                    reminder.triggerTimeEpochMs,
                    pendingIntent
                )
            } else {
                alarmManager.setExact(
                    AlarmManager.RTC_WAKEUP,
                    reminder.triggerTimeEpochMs,
                    pendingIntent
                )
            }
            Log.i(TAG, "Scheduled CallReminder '${reminder.title}' at ${reminder.triggerTimeEpochMs}")
        } catch (e: SecurityException) {
            Log.w(TAG, "Exact alarm permission missing, falling back to inexact: ${e.message}")
            alarmManager.set(
                AlarmManager.RTC_WAKEUP,
                reminder.triggerTimeEpochMs,
                pendingIntent
            )
        }
    }

    /**
     * Snoozes an existing reminder by N minutes and reschedules.
     */
    fun snooze(context: Context, reminderId: String, minutes: Int = 10): CallReminder? {
        val existing = getReminder(context, reminderId) ?: return null
        val newTriggerTime = System.currentTimeMillis() + (minutes * 60 * 1000L)
        val snoozed = existing.copy(
            triggerTimeEpochMs = newTriggerTime,
            isSnoozed = true,
            snoozeCount = existing.snoozeCount + 1
        )
        schedule(context, snoozed)
        Log.i(TAG, "Snoozed reminder '$reminderId' for $minutes minutes.")
        return snoozed
    }

    /**
     * Marks a reminder as completed and cancels pending alarms.
     */
    fun markComplete(context: Context, reminderId: String) {
        cancelAlarmIntent(context, reminderId)
        val all = getAllReminders(context).toMutableList()
        val index = all.indexOfFirst { it.id == reminderId }
        if (index != -1) {
            all[index] = all[index].copy(isCompleted = true)
            saveReminders(context, all)
            Log.i(TAG, "Marked reminder '$reminderId' as complete.")
        }
    }

    /**
     * Cancels and deletes a reminder.
     */
    fun cancel(context: Context, reminderId: String) {
        cancelAlarmIntent(context, reminderId)
        val all = getAllReminders(context).toMutableList()
        all.removeAll { it.id == reminderId }
        saveReminders(context, all)
        Log.i(TAG, "Cancelled reminder '$reminderId'.")
    }

    fun getReminder(context: Context, reminderId: String): CallReminder? {
        return getAllReminders(context).find { it.id == reminderId }
    }

    fun getActiveReminders(context: Context): List<CallReminder> {
        val now = System.currentTimeMillis()
        return getAllReminders(context)
            .filter { !it.isCompleted }
            .sortedBy { it.triggerTimeEpochMs }
    }

    fun getAllReminders(context: Context): List<CallReminder> {
        val jsonStr = getPrefs(context).getString(KEY_REMINDERS_JSON, "[]") ?: "[]"
        val list = mutableListOf<CallReminder>()
        try {
            val array = JSONArray(jsonStr)
            for (i in 0 until array.length()) {
                val obj = array.getJSONObject(i)
                list.add(CallReminder.fromJson(obj))
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse reminders JSON: ${e.message}")
        }
        return list
    }

    private fun saveReminders(context: Context, reminders: List<CallReminder>) {
        val array = JSONArray()
        reminders.forEach { array.put(it.toJson()) }
        getPrefs(context).edit().putString(KEY_REMINDERS_JSON, array.toString()).apply()
    }

    private fun createAlarmPendingIntent(context: Context, reminder: CallReminder): PendingIntent {
        val intent = Intent(context, CallReminderReceiver::class.java).apply {
            action = "com.zipper.app.ACTION_TRIGGER_CALL_REMINDER"
            putExtra(EXTRA_REMINDER_ID, reminder.id)
        }
        val flags = PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        return PendingIntent.getBroadcast(
            context,
            reminder.id.hashCode(),
            intent,
            flags
        )
    }

    private fun cancelAlarmIntent(context: Context, reminderId: String) {
        val intent = Intent(context, CallReminderReceiver::class.java).apply {
            action = "com.zipper.app.ACTION_TRIGGER_CALL_REMINDER"
            putExtra(EXTRA_REMINDER_ID, reminderId)
        }
        val flags = PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        val pendingIntent = PendingIntent.getBroadcast(
            context,
            reminderId.hashCode(),
            intent,
            flags
        )
        val alarmManager = context.getSystemService(Context.ALARM_SERVICE) as? AlarmManager
        alarmManager?.cancel(pendingIntent)
    }

    /**
     * Re-registers all uncompleted alarms upon system reboot.
     */
    fun restoreAllAlarms(context: Context) {
        val active = getActiveReminders(context)
        val now = System.currentTimeMillis()
        Log.i(TAG, "Restoring ${active.size} active call reminders after reboot.")
        for (reminder in active) {
            if (reminder.triggerTimeEpochMs > now) {
                schedule(context, reminder)
            }
        }
    }

    /**
     * Flexible natural language parser for reminders.
     * Examples:
     * - "call me in 20 minutes to turn off water motor"
     * - "call me in 10 mins: geyser"
     * - "call me at 7:30 pm to hit the gym"
     * - "bill reminder electricity 1450 tomorrow at 10 am"
     */
    fun parseNaturalReminder(input: String): CallReminder? {
        val text = input.trim()
        if (text.isBlank()) return null

        val now = Calendar.getInstance()

        // 1. Relative minutes: "in 20 minutes / in 15 mins / in 1 hour"
        val relativeRegex = Regex("""(?i)(?:call\s+me|remind\s+me|set\s+(?:me\s+)?(?:an?\s+)?(?:alarm|reminder))?\s*in\s+(\d+)\s*(minutes|minute|mins|min|hours|hour|hrs|hr|m|h)\s*(?:to|for|:)?\s*(.*)$""")
        val relMatch = relativeRegex.find(text)
        if (relMatch != null) {
            val amount = relMatch.groupValues[1].toIntOrNull() ?: 10
            val unit = relMatch.groupValues[2].lowercase()
            val rawChore = relMatch.groupValues[3].trim()
            val chore = if (rawChore.isNotBlank()) rawChore else "Alarm Reminder"

            val delayMs = if (unit.startsWith("h")) {
                amount * 60 * 60 * 1000L
            } else {
                amount * 60 * 1000L
            }

            val triggerTime = System.currentTimeMillis() + delayMs
            val isBill = chore.contains(Regex("""(?i)\b(bill|pay|electricity|rent|recharge|dth|wifi)\b"""))
            val actionType = if (isBill) CallReminderActionType.AGENT_HANDOFF else CallReminderActionType.VOICE_ONLY

            return CallReminder(
                title = chore.replaceFirstChar { it.uppercase() },
                spokenPrompt = "Hey Karthik, it's time for $chore.",
                triggerTimeEpochMs = triggerTime,
                actionType = actionType,
                targetGoal = if (isBill) "Pay bill: $chore" else null
            )
        }

        // 2. Exact time: "at 7:30 pm to gym" or "set me an alarm for 2:41" or "alarm for 6 am"
        val timeRegex = Regex("""(?i)(?:call\s+me|remind\s+me|wake\s+me\s+up|set\s+(?:me\s+)?(?:an?\s+)?(?:alarm|reminder))?\s*(?:today|tomorrow)?\s*(?:at|for)?\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*(?:to|for|:)?\s*(.*)$""")
        val timeMatch = timeRegex.find(text)
        if (timeMatch != null && (timeMatch.groupValues[2].isNotBlank() || timeMatch.groupValues[3].isNotBlank())) {
            var hour = timeMatch.groupValues[1].toIntOrNull() ?: 12
            val minute = timeMatch.groupValues[2].toIntOrNull() ?: 0
            val amPm = timeMatch.groupValues[3].lowercase()
            val rawChore = timeMatch.groupValues[4].trim()
            val chore = if (rawChore.isNotBlank()) rawChore else "Alarm"

            if (amPm == "pm" && hour < 12) hour += 12
            if (amPm == "am" && hour == 12) hour = 0
            // If AM/PM omitted, pick next occurrence
            if (amPm.isBlank() && hour < 12) {
                val currentHour = now.get(Calendar.HOUR_OF_DAY)
                if (currentHour >= 12 && hour + 12 > currentHour) {
                    hour += 12
                }
            }

            val targetCal = Calendar.getInstance().apply {
                set(Calendar.HOUR_OF_DAY, hour)
                set(Calendar.MINUTE, minute)
                set(Calendar.SECOND, 0)
                set(Calendar.MILLISECOND, 0)
            }

            if (targetCal.before(now) || text.contains(Regex("(?i)tomorrow"))) {
                targetCal.add(Calendar.DAY_OF_YEAR, 1)
            }

            val isBill = chore.contains(Regex("""(?i)\b(bill|pay|electricity|rent|recharge|dth|wifi)\b"""))
            val actionType = if (isBill) CallReminderActionType.AGENT_HANDOFF else CallReminderActionType.VOICE_ONLY

            return CallReminder(
                title = "$chore ($hour:${String.format(java.util.Locale.US, "%02d", minute)})",
                spokenPrompt = "Hey Karthik, alarm: $chore.",
                triggerTimeEpochMs = targetCal.timeInMillis,
                actionType = actionType,
                targetGoal = if (isBill) "Pay bill: $chore" else null
            )
        }

        // Fallback default: +15 minutes
        val cleanChore = text
            .replace(Regex("""(?i)^(?:call\s+me|remind\s+me)\s*(?:to|for)?\s*"""), "")
            .trim()
            .ifBlank { "General task" }

        return CallReminder(
            title = cleanChore.replaceFirstChar { it.uppercase() },
            spokenPrompt = "Hey Karthik, reminder for $cleanChore.",
            triggerTimeEpochMs = System.currentTimeMillis() + (15 * 60 * 1000L),
            actionType = CallReminderActionType.VOICE_ONLY
        )
    }
}
