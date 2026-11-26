package com.zipper.app.core.intelligence.scheduler

import android.content.Context
import android.util.Log
import com.zipper.app.core.intelligence.store.CommitmentCategory
import com.zipper.app.core.intelligence.store.CommitmentDatabase
import com.zipper.app.core.intelligence.store.CommitmentEntity
import com.zipper.app.core.intelligence.store.CommitmentStatus
import com.zipper.app.core.voice.CallReminder
import com.zipper.app.core.voice.CallReminderActionType
import com.zipper.app.core.voice.CallReminderScheduler

/**
 * Schedules 5-minute pre-event awakening phone calls for validated daily commitments.
 * Arms Android's AlarmManager subsystem so the device rings reliably.
 */
object LifePlanCallScheduler {

    private const val TAG = "LifePlanCallScheduler"
    private const val PRE_CALL_LEAD_TIME_MS = 5 * 60 * 1000L // 5 minutes before scheduled start

    /**
     * Scans pending commitments in CommitmentDatabase and arms 5-minute pre-call alarms.
     * Strictly enforces Quiet Hours (10:00 PM - 07:00 AM), user 'Don't Call Today' mutes,
     * and exempts quick checklist to-dos from ever arming phone calls.
     */
    fun scheduleUpcomingAlarms(context: Context) {
        val prefs = com.zipper.app.core.intelligence.store.LifeRhythmPrefs(context)
        if (prefs.isCallMutedNow()) {
            Log.i(TAG, "AI pre-call awakenings are currently MUTED (user mute or overnight quiet hours). Skipping alarm arming.")
            return
        }

        val db = CommitmentDatabase.getInstance(context)
        val pending = db.getPendingCommitments()
        val now = System.currentTimeMillis()

        Log.i(TAG, "Scanning ${pending.size} pending commitments for 5-minute pre-call alarms")

        for (commitment in pending) {
            // Checklist to-dos must NEVER ring or trigger phone calls
            if (commitment.isChecklist) {
                continue
            }

            val triggerTime = commitment.scheduledTimeMs - PRE_CALL_LEAD_TIME_MS
            // Skip past triggers, triggers inside overnight quiet hours, or user mute
            if (triggerTime > now) {
                if (prefs.isInsideQuietHours(triggerTime)) {
                    Log.i(TAG, "Suppressed 5-min pre-call alarm for '${commitment.title}' because $triggerTime falls in quiet hours.")
                    continue
                }

                val prompt = generateSpokenPrompt(commitment)
                val reminder = CallReminder(
                    id = "life_plan_${commitment.id}",
                    title = commitment.title,
                    spokenPrompt = prompt,
                    triggerTimeEpochMs = triggerTime,
                    actionType = CallReminderActionType.VOICE_ONLY,
                    targetPackage = commitment.candidatePackages.firstOrNull(),
                    commitmentId = commitment.id,
                    category = commitment.category.name,
                    candidatePackages = commitment.candidatePackages,
                    sourcePerson = commitment.sourcePerson
                )
                CallReminderScheduler.schedule(context, reminder)
                Log.d(TAG, "Armed 5-min pre-call alarm for '${commitment.title}' at $triggerTime")
            }
        }
    }

    /**
     * Finds the next upcoming daytime pre-call alarm.
     * Returns Pair(CommitmentEntity, triggerTimeEpochMs) or null if no calls scheduled.
     */
    fun getNextUpcomingCall(context: Context): Pair<CommitmentEntity, Long>? {
        val prefs = com.zipper.app.core.intelligence.store.LifeRhythmPrefs(context)
        if (prefs.isCallMutedNow()) return null

        val db = CommitmentDatabase.getInstance(context)
        val pending = db.getPendingCommitments()
        val now = System.currentTimeMillis()

        return pending
            .filter { !it.isChecklist }
            .map { it to (it.scheduledTimeMs - PRE_CALL_LEAD_TIME_MS) }
            .filter { (_, triggerTime) -> triggerTime > now && !prefs.isInsideQuietHours(triggerTime) }
            .minByOrNull { it.second }
    }

    /**
     * Mutes all pre-call awakening alarms for the rest of today until tomorrow 07:00 AM.
     * Cancels any active alarms currently scheduled in AlarmManager.
     */
    fun muteAllCallsToday(context: Context) {
        val prefs = com.zipper.app.core.intelligence.store.LifeRhythmPrefs(context)
        prefs.muteCallsForToday()

        val active = CallReminderScheduler.getAllReminders(context)
        for (reminder in active) {
            if (reminder.id.startsWith("life_plan_") || reminder.id.startsWith("test_call_")) {
                CallReminderScheduler.cancel(context, reminder.id)
            }
        }
        Log.i(TAG, "Muted all AI calls today and cancelled pending alarms.")
    }

    /**
     * Restores call reminders and re-arms daytime alarms.
     */
    fun unmuteCalls(context: Context) {
        val prefs = com.zipper.app.core.intelligence.store.LifeRhythmPrefs(context)
        prefs.unmuteCalls()
        scheduleUpcomingAlarms(context)
        Log.i(TAG, "Unmuted AI calls and re-armed upcoming daytime alarms.")
    }

    /**
     * Immediately triggers an instant test incoming call for a commitment
     * (fires in 3 seconds) for live demonstration and physical verification.
     */
    fun triggerTestCallNow(context: Context, commitment: CommitmentEntity) {
        val triggerTime = System.currentTimeMillis() + 3000L
        val prompt = generateSpokenPrompt(commitment)
        val reminder = CallReminder(
            id = "test_call_${commitment.id}_${System.currentTimeMillis()}",
            title = commitment.title,
            spokenPrompt = prompt,
            triggerTimeEpochMs = triggerTime,
            actionType = CallReminderActionType.VOICE_ONLY,
            targetPackage = commitment.candidatePackages.firstOrNull(),
            commitmentId = commitment.id,
            category = commitment.category.name,
            candidatePackages = commitment.candidatePackages,
            sourcePerson = commitment.sourcePerson
        )
        CallReminderScheduler.schedule(context, reminder)
        Log.i(TAG, "Armed INSTANT test call for '${commitment.title}' (fires in 3s)")
    }

    /**
     * Cancels an existing commitment's scheduled call alarm.
     */
    fun cancelAlarmsForCommitment(context: Context, commitmentId: Long) {
        CallReminderScheduler.cancel(context, "life_plan_$commitmentId")
    }

    private fun generateSpokenPrompt(c: CommitmentEntity): String {
        return when (c.category) {
            CommitmentCategory.EDUCATION, CommitmentCategory.WORK -> {
                "Hey Karthik, your ${c.title} is due soon. Do you want to work on it now or should I reschedule it?"
            }
            CommitmentCategory.FITNESS, CommitmentCategory.HEALTH -> {
                "Hey Karthik, your workout session '${c.title}' starts in 5 minutes! Do you want to start now or should I reschedule it?"
            }
            CommitmentCategory.COMMUNICATION -> {
                if (c.sourcePerson.isNotBlank()) {
                    "Hey Karthik, reminder to connect with ${c.sourcePerson}. Do you want to call now or should I reschedule it?"
                } else {
                    "Hey Karthik, reminder for '${c.title}'. Do you want to do it now or should I reschedule it?"
                }
            }
            CommitmentCategory.GENERAL -> {
                "Hey Karthik, your ${c.title} is scheduled now. Do you want to work on it now or should I reschedule it?"
            }
        }
    }
}
