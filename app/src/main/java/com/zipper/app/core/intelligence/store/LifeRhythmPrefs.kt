package com.zipper.app.core.intelligence.store

import android.content.Context
import android.content.SharedPreferences

/**
 * Manages user life rhythm anchor windows (e.g. college/work hours,
 * nightly planning time, sleep hours).
 */
class LifeRhythmPrefs(context: Context? = null, customPrefs: SharedPreferences? = null) {

    private val prefs: SharedPreferences? =
        customPrefs ?: context?.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    private val memoryStore = mutableMapOf<String, Any>()

    companion object {
        private const val PREFS_NAME = "karvish_life_rhythm_prefs"

        private const val KEY_COLLEGE_START_HOUR = "college_start_hour"
        private const val KEY_COLLEGE_END_HOUR = "college_end_hour"
        private const val KEY_NIGHT_PLAN_HOUR = "night_plan_hour"
        private const val KEY_NIGHT_PLAN_MINUTE = "night_plan_minute"
        private const val KEY_SLEEP_HOUR = "sleep_hour"
        private const val KEY_GUARD_ACTIVE = "guard_active"
        private const val KEY_TRANSIT_BUFFER_MINS = "transit_buffer_mins"
        private const val KEY_CALLS_MUTED_UNTIL_EPOCH_MS = "calls_muted_until_epoch_ms"
        private const val KEY_QUIET_HOURS_START = "quiet_hours_start"
        private const val KEY_QUIET_HOURS_END = "quiet_hours_end"
        private const val KEY_LLM_API_KEY = "llm_api_key"
        private const val KEY_LLM_ENDPOINT = "llm_endpoint"
        private const val KEY_LLM_MODEL = "llm_model"
        private const val KEY_LLM_PLATFORM = "llm_platform"

        const val DEFAULT_COLLEGE_START_HOUR = 9   // 09:00 AM
        const val DEFAULT_COLLEGE_END_HOUR = 18    // 06:00 PM
        const val DEFAULT_NIGHT_PLAN_HOUR = 22     // 10:00 PM
        const val DEFAULT_NIGHT_PLAN_MINUTE = 0
        const val DEFAULT_SLEEP_HOUR = 23          // 11:00 PM
        const val DEFAULT_QUIET_HOURS_START = 22   // 10:00 PM
        const val DEFAULT_QUIET_HOURS_END = 7      // 07:00 AM
        const val DEFAULT_TRANSIT_BUFFER_MINS = 30 // 30 minutes after college
        const val DEFAULT_LLM_MODEL = "gemini-1.5-flash"
        const val DEFAULT_LLM_PLATFORM = "gemini"
    }

    var collegeStartHour: Int
        get() = prefs?.getInt(KEY_COLLEGE_START_HOUR, DEFAULT_COLLEGE_START_HOUR)
            ?: (memoryStore[KEY_COLLEGE_START_HOUR] as? Int ?: DEFAULT_COLLEGE_START_HOUR)
        set(value) {
            prefs?.edit()?.putInt(KEY_COLLEGE_START_HOUR, value)?.apply()
            memoryStore[KEY_COLLEGE_START_HOUR] = value
        }

    var collegeEndHour: Int
        get() = prefs?.getInt(KEY_COLLEGE_END_HOUR, DEFAULT_COLLEGE_END_HOUR)
            ?: (memoryStore[KEY_COLLEGE_END_HOUR] as? Int ?: DEFAULT_COLLEGE_END_HOUR)
        set(value) {
            prefs?.edit()?.putInt(KEY_COLLEGE_END_HOUR, value)?.apply()
            memoryStore[KEY_COLLEGE_END_HOUR] = value
        }

    var nightPlanHour: Int
        get() = prefs?.getInt(KEY_NIGHT_PLAN_HOUR, DEFAULT_NIGHT_PLAN_HOUR)
            ?: (memoryStore[KEY_NIGHT_PLAN_HOUR] as? Int ?: DEFAULT_NIGHT_PLAN_HOUR)
        set(value) {
            prefs?.edit()?.putInt(KEY_NIGHT_PLAN_HOUR, value)?.apply()
            memoryStore[KEY_NIGHT_PLAN_HOUR] = value
        }

    var nightPlanMinute: Int
        get() = prefs?.getInt(KEY_NIGHT_PLAN_MINUTE, DEFAULT_NIGHT_PLAN_MINUTE)
            ?: (memoryStore[KEY_NIGHT_PLAN_MINUTE] as? Int ?: DEFAULT_NIGHT_PLAN_MINUTE)
        set(value) {
            prefs?.edit()?.putInt(KEY_NIGHT_PLAN_MINUTE, value)?.apply()
            memoryStore[KEY_NIGHT_PLAN_MINUTE] = value
        }

    var sleepHour: Int
        get() = prefs?.getInt(KEY_SLEEP_HOUR, DEFAULT_SLEEP_HOUR)
            ?: (memoryStore[KEY_SLEEP_HOUR] as? Int ?: DEFAULT_SLEEP_HOUR)
        set(value) {
            prefs?.edit()?.putInt(KEY_SLEEP_HOUR, value)?.apply()
            memoryStore[KEY_SLEEP_HOUR] = value
        }

    var isGuardActive: Boolean
        get() = prefs?.getBoolean(KEY_GUARD_ACTIVE, true)
            ?: (memoryStore[KEY_GUARD_ACTIVE] as? Boolean ?: true)
        set(value) {
            prefs?.edit()?.putBoolean(KEY_GUARD_ACTIVE, value)?.apply()
            memoryStore[KEY_GUARD_ACTIVE] = value
        }

    var transitBufferMinutes: Int
        get() = prefs?.getInt(KEY_TRANSIT_BUFFER_MINS, DEFAULT_TRANSIT_BUFFER_MINS)
            ?: (memoryStore[KEY_TRANSIT_BUFFER_MINS] as? Int ?: DEFAULT_TRANSIT_BUFFER_MINS)
        set(value) {
            prefs?.edit()?.putInt(KEY_TRANSIT_BUFFER_MINS, value)?.apply()
            memoryStore[KEY_TRANSIT_BUFFER_MINS] = value
        }

    var llmApiKey: String
        get() = prefs?.getString(KEY_LLM_API_KEY, "").orEmpty()
            .ifEmpty { (memoryStore[KEY_LLM_API_KEY] as? String).orEmpty() }
        set(value) {
            prefs?.edit()?.putString(KEY_LLM_API_KEY, value)?.apply()
            memoryStore[KEY_LLM_API_KEY] = value
        }

    var llmEndpoint: String
        get() = prefs?.getString(KEY_LLM_ENDPOINT, "").orEmpty()
            .ifEmpty { (memoryStore[KEY_LLM_ENDPOINT] as? String).orEmpty() }
        set(value) {
            prefs?.edit()?.putString(KEY_LLM_ENDPOINT, value)?.apply()
            memoryStore[KEY_LLM_ENDPOINT] = value
        }

    var llmModel: String
        get() = prefs?.getString(KEY_LLM_MODEL, DEFAULT_LLM_MODEL).orEmpty()
            .ifEmpty { (memoryStore[KEY_LLM_MODEL] as? String) ?: DEFAULT_LLM_MODEL }
        set(value) {
            prefs?.edit()?.putString(KEY_LLM_MODEL, value)?.apply()
            memoryStore[KEY_LLM_MODEL] = value
        }

    var llmPlatform: String
        get() = prefs?.getString(KEY_LLM_PLATFORM, DEFAULT_LLM_PLATFORM).orEmpty()
            .ifEmpty { (memoryStore[KEY_LLM_PLATFORM] as? String) ?: DEFAULT_LLM_PLATFORM }
        set(value) {
            prefs?.edit()?.putString(KEY_LLM_PLATFORM, value)?.apply()
            memoryStore[KEY_LLM_PLATFORM] = value
        }

    var callsMutedUntilEpochMs: Long
        get() = prefs?.getLong(KEY_CALLS_MUTED_UNTIL_EPOCH_MS, 0L)
            ?: (memoryStore[KEY_CALLS_MUTED_UNTIL_EPOCH_MS] as? Long ?: 0L)
        set(value) {
            prefs?.edit()?.putLong(KEY_CALLS_MUTED_UNTIL_EPOCH_MS, value)?.apply()
            memoryStore[KEY_CALLS_MUTED_UNTIL_EPOCH_MS] = value
        }

    var quietHoursStart: Int
        get() = prefs?.getInt(KEY_QUIET_HOURS_START, DEFAULT_QUIET_HOURS_START)
            ?: (memoryStore[KEY_QUIET_HOURS_START] as? Int ?: DEFAULT_QUIET_HOURS_START)
        set(value) {
            prefs?.edit()?.putInt(KEY_QUIET_HOURS_START, value)?.apply()
            memoryStore[KEY_QUIET_HOURS_START] = value
        }

    var quietHoursEnd: Int
        get() = prefs?.getInt(KEY_QUIET_HOURS_END, DEFAULT_QUIET_HOURS_END)
            ?: (memoryStore[KEY_QUIET_HOURS_END] as? Int ?: DEFAULT_QUIET_HOURS_END)
        set(value) {
            prefs?.edit()?.putInt(KEY_QUIET_HOURS_END, value)?.apply()
            memoryStore[KEY_QUIET_HOURS_END] = value
        }

    /**
     * Checks if AI pre-event awakening phone calls are currently muted.
     * Evaluates both manual 'Don't Call Today' mute timestamp and automatic overnight quiet hours.
     */
    fun isCallMutedNow(): Boolean {
        val now = System.currentTimeMillis()
        if (now < callsMutedUntilEpochMs) return true
        return isInsideQuietHours(now)
    }

    /**
     * Checks if the given epoch timestamp falls within night quiet hours (e.g. 10:00 PM to 07:00 AM).
     */
    fun isInsideQuietHours(epochMs: Long): Boolean {
        val cal = java.util.Calendar.getInstance().apply { timeInMillis = epochMs }
        val hour = cal.get(java.util.Calendar.HOUR_OF_DAY)
        // If quietHoursStart is 22 and quietHoursEnd is 7, any hour >= 22 OR < 7 is quiet hours
        return if (quietHoursStart > quietHoursEnd) {
            hour >= quietHoursStart || hour < quietHoursEnd
        } else {
            hour in quietHoursStart until quietHoursEnd
        }
    }

    /**
     * Mutes all phone calls for the rest of today until tomorrow morning at quietHoursEnd (07:00 AM).
     */
    fun muteCallsForToday() {
        val cal = java.util.Calendar.getInstance().apply {
            add(java.util.Calendar.DAY_OF_YEAR, 1)
            set(java.util.Calendar.HOUR_OF_DAY, quietHoursEnd)
            set(java.util.Calendar.MINUTE, 0)
            set(java.util.Calendar.SECOND, 0)
            set(java.util.Calendar.MILLISECOND, 0)
        }
        callsMutedUntilEpochMs = cal.timeInMillis
    }

    /**
     * Resets mute status so pre-call awakenings are re-enabled.
     */
    fun unmuteCalls() {
        callsMutedUntilEpochMs = 0L
    }
}
