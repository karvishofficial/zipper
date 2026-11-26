package com.zipper.app.core.intelligence.scheduler

import android.util.Log
import com.zipper.app.core.intelligence.store.CommitmentCategory
import com.zipper.app.core.intelligence.store.CommitmentEntity
import com.zipper.app.core.intelligence.store.CommitmentStatus
import com.zipper.app.core.intelligence.store.LifeRhythmPrefs
import java.util.Calendar

/**
 * Deterministic schedule validation engine.
 *
 * Enforces:
 * 1. Transit Buffers (protects user commute time after college/work).
 * 2. Energy Curves (heavy tasks positioned when user has focus).
 * 3. Overlap / Clash Resolution (shifts overlapping tasks).
 * 4. Sleep & Anchor Window Protection (no intrusive tasks during college or night sleep).
 */
class SmartScheduleValidator(private val prefs: LifeRhythmPrefs) {

    companion object {
        private const val TAG = "SmartScheduleValidator"
        const val WAKE_HOUR = 6 // 6:00 AM
    }

    /**
     * Validates and adjusts a list of proposed commitments for a specific day.
     * Ensures all commitments respect college/work anchor hours, transit buffers,
     * sleep boundaries, and have zero chronological clashes.
     */
    fun validateAndRebalance(
        proposals: List<CommitmentEntity>,
        targetDayTimestampMs: Long = System.currentTimeMillis()
    ): List<CommitmentEntity> {
        if (proposals.isEmpty()) return emptyList()

        val cal = Calendar.getInstance().apply {
            timeInMillis = targetDayTimestampMs
            set(Calendar.HOUR_OF_DAY, 0)
            set(Calendar.MINUTE, 0)
            set(Calendar.SECOND, 0)
            set(Calendar.MILLISECOND, 0)
        }
        val dayStartMs = cal.timeInMillis

        val collegeStartMs = dayStartMs + (prefs.collegeStartHour * 3600000L)
        val collegeEndMs = dayStartMs + (prefs.collegeEndHour * 3600000L)
        val transitBufferMs = prefs.transitBufferMinutes * 60000L
        val freeEveningStartMs = collegeEndMs + transitBufferMs
        val quietHoursStartHour = prefs.quietHoursStart.coerceIn(18, 23)
        val quietHoursEndHour = prefs.quietHoursEnd.coerceIn(5, 10)
        val sleepStartMs = dayStartMs + (quietHoursStartHour * 3600000L)
        val wakeStartMs = dayStartMs + (quietHoursEndHour * 3600000L)

        val scheduledItems = mutableListOf<CommitmentEntity>()

        // Sort incoming proposals by priority: with deadline first, then category
        val sortedProposals = proposals.sortedWith(
            compareBy<CommitmentEntity> { it.deadlineMs ?: Long.MAX_VALUE }
                .thenBy { it.scheduledTimeMs }
        )

        for (proposal in sortedProposals) {
            var proposedTime = proposal.scheduledTimeMs
            val durationMs = proposal.durationMinutes * 60000L

            // 1. If proposed time is not set for target day, choose intelligent default slot based on category
            if (proposedTime < dayStartMs || proposedTime > dayStartMs + 86400000L) {
                proposedTime = getDefaultCategoryTime(proposal.category, dayStartMs, freeEveningStartMs, wakeStartMs)
            }

            // 2. Adjust if inside College/Work Anchor hours
            if (proposedTime in collegeStartMs..collegeEndMs) {
                Log.d(TAG, "Task '${proposal.title}' overlaps college hours. Shifting to post-transit evening.")
                proposedTime = freeEveningStartMs
            } else if (proposedTime < freeEveningStartMs && (proposedTime + durationMs) > collegeStartMs && proposedTime >= collegeStartMs) {
                proposedTime = freeEveningStartMs
            }

            // 3. Adjust if inside Transit Buffer
            if (proposedTime in collegeEndMs until freeEveningStartMs) {
                Log.d(TAG, "Task '${proposal.title}' inside transit buffer. Shifting to free evening start.")
                proposedTime = freeEveningStartMs
            }

            // 4. Adjust if inside Sleep / Night Quiet Hours Window (strictly blocks 22:00 to 07:00 AM)
            if (proposedTime >= sleepStartMs || (proposedTime + durationMs) > sleepStartMs || proposedTime < wakeStartMs) {
                Log.d(TAG, "Task '${proposal.title}' encroaches sleep/quiet hours ($quietHoursStartHour:00 - $quietHoursEndHour:00). Shifting to morning.")
                proposedTime = wakeStartMs + 1800000L // 7:30 AM
            }

            // 5. Clash Resolution: check against already scheduled items
            proposedTime = findNextAvailableSlot(
                candidateStartMs = proposedTime,
                durationMs = durationMs,
                existing = scheduledItems,
                dayStartMs = dayStartMs,
                collegeStartMs = collegeStartMs,
                collegeEndMs = collegeEndMs,
                transitBufferMs = transitBufferMs,
                sleepStartMs = sleepStartMs,
                wakeStartMs = wakeStartMs
            )

            val adjusted = proposal.copy(
                scheduledTimeMs = proposedTime,
                status = CommitmentStatus.PENDING
            )
            scheduledItems.add(adjusted)
            Log.i(TAG, "Validated slot for '${adjusted.title}': ${formatTime(adjusted.scheduledTimeMs)} (${adjusted.durationMinutes}m)")
        }

        return scheduledItems.sortedBy { it.scheduledTimeMs }
    }

    private fun getDefaultCategoryTime(
        category: CommitmentCategory,
        dayStartMs: Long,
        freeEveningStartMs: Long,
        wakeStartMs: Long
    ): Long {
        return when (category) {
            CommitmentCategory.FITNESS, CommitmentCategory.HEALTH -> {
                // Default to post-commute evening (e.g. 6:30 PM) or early morning (6:30 AM)
                freeEveningStartMs
            }
            CommitmentCategory.EDUCATION, CommitmentCategory.WORK -> {
                // Deep work after dinner / evening (e.g. 7:30 PM)
                freeEveningStartMs + 3600000L // 1 hour after arriving home
            }
            CommitmentCategory.COMMUNICATION -> {
                // Quick follow up around 7:00 PM
                freeEveningStartMs + (30 * 60000L)
            }
            CommitmentCategory.GENERAL -> {
                freeEveningStartMs + 7200000L // 8:30 PM
            }
        }
    }

    private fun findNextAvailableSlot(
        candidateStartMs: Long,
        durationMs: Long,
        existing: List<CommitmentEntity>,
        dayStartMs: Long,
        collegeStartMs: Long,
        collegeEndMs: Long,
        transitBufferMs: Long,
        sleepStartMs: Long,
        wakeStartMs: Long
    ): Long {
        var start = candidateStartMs

        var collisionFound = true
        var iterations = 0

        while (collisionFound && iterations < 20) {
            collisionFound = false
            iterations++

            val end = start + durationMs

            // College clash
            val freeEvening = collegeEndMs + transitBufferMs
            if (start in collegeStartMs until freeEvening || (start < collegeStartMs && end > collegeStartMs)) {
                start = freeEvening
                collisionFound = true
                continue
            }

            // Sleep clash: strictly prevent overnight slots (quiet hours start to wake)
            if (start >= sleepStartMs || end > sleepStartMs || start < wakeStartMs) {
                // Wrap to morning
                start = wakeStartMs + 1800000L // 7:30 AM
                collisionFound = true
                continue
            }

            // Existing tasks clash
            for (item in existing) {
                val itemStart = item.scheduledTimeMs
                val itemEnd = itemStart + (item.durationMinutes * 60000L)

                val overlaps = (start < itemEnd) && (end > itemStart)
                if (overlaps) {
                    // Shift start to 10 minutes after existing item ends
                    start = itemEnd + (10 * 60000L)
                    collisionFound = true
                    break
                }
            }
        }

        return start
    }

    private fun formatTime(timeMs: Long): String {
        val cal = Calendar.getInstance().apply { timeInMillis = timeMs }
        val hour = cal.get(Calendar.HOUR_OF_DAY)
        val minute = cal.get(Calendar.MINUTE)
        return String.format(java.util.Locale.US, "%02d:%02d", hour, minute)
    }
}
