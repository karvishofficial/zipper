package com.zipper.app.core.security

import android.content.Context
import android.content.SharedPreferences
import com.zipper.app.BuildConfig

/**
 * Security Configuration for Zipper Anti-Theft, Intruder Detection, and Stealth Mode.
 * Stores credentials, alert thresholds, and flags safely in SharedPreferences.
 */
object SecurityConfig {

    private const val PREFS_NAME = "zipper_security_guard_prefs"

    private const val KEY_ENABLED = "security_guard_enabled"
    private const val KEY_FAILED_ATTEMPT_THRESHOLD = "failed_attempt_threshold"
    private const val KEY_TELEGRAM_BOT_TOKEN = "telegram_bot_token"
    private const val KEY_TELEGRAM_CHAT_ID = "telegram_chat_id"
    private const val KEY_TRIGGER_ON_SIM_EJECT = "trigger_on_sim_eject"
    private const val KEY_ALERT_EMAIL = "alert_email"
    const val DEFAULT_ALERT_EMAIL = BuildConfig.ALERT_EMAIL
    private const val KEY_SMTP_APP_PASSWORD = "smtp_app_password"
    const val DEFAULT_SMTP_APP_PASSWORD = BuildConfig.SMTP_APP_PASSWORD
    private const val KEY_CUSTOM_ALERT_WEBHOOK = "custom_alert_webhook"
    private const val KEY_STEALTH_MODE_ACTIVE = "stealth_mode_active"
    private const val KEY_LIVE_TRACKING_ACTIVE = "live_tracking_active"
    private const val KEY_LAST_INTRUDER_TIMESTAMP = "last_intruder_timestamp"

    private fun getPrefs(context: Context): SharedPreferences {
        return context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    }

    fun getAlertEmail(context: Context): String {
        return getPrefs(context).getString(KEY_ALERT_EMAIL, DEFAULT_ALERT_EMAIL) ?: DEFAULT_ALERT_EMAIL
    }

    fun setAlertEmail(context: Context, email: String) {
        getPrefs(context).edit().putString(KEY_ALERT_EMAIL, email.trim()).apply()
    }

    fun getSmtpAppPassword(context: Context): String {
        return getPrefs(context).getString(KEY_SMTP_APP_PASSWORD, DEFAULT_SMTP_APP_PASSWORD) ?: DEFAULT_SMTP_APP_PASSWORD
    }

    fun setSmtpAppPassword(context: Context, password: String) {
        getPrefs(context).edit().putString(KEY_SMTP_APP_PASSWORD, password.trim()).apply()
    }

    fun getCustomAlertWebhook(context: Context): String {
        return getPrefs(context).getString(KEY_CUSTOM_ALERT_WEBHOOK, "").orEmpty()
    }

    fun setCustomAlertWebhook(context: Context, url: String) {
        getPrefs(context).edit().putString(KEY_CUSTOM_ALERT_WEBHOOK, url.trim()).apply()
    }

    fun isLiveTrackingActive(context: Context): Boolean {
        return getPrefs(context).getBoolean(KEY_LIVE_TRACKING_ACTIVE, false)
    }

    fun setLiveTrackingActive(context: Context, active: Boolean) {
        getPrefs(context).edit().putBoolean(KEY_LIVE_TRACKING_ACTIVE, active).apply()
    }

    fun isGuardEnabled(context: Context): Boolean {
        return getPrefs(context).getBoolean(KEY_ENABLED, true)
    }

    fun setGuardEnabled(context: Context, enabled: Boolean) {
        getPrefs(context).edit().putBoolean(KEY_ENABLED, enabled).apply()
    }

    fun getFailedAttemptThreshold(context: Context): Int {
        return getPrefs(context).getInt(KEY_FAILED_ATTEMPT_THRESHOLD, 2)
    }

    fun setFailedAttemptThreshold(context: Context, threshold: Int) {
        getPrefs(context).edit().putInt(KEY_FAILED_ATTEMPT_THRESHOLD, threshold).apply()
    }

    fun getTelegramBotToken(context: Context): String {
        return getPrefs(context).getString(KEY_TELEGRAM_BOT_TOKEN, "").orEmpty()
    }

    fun setTelegramBotToken(context: Context, token: String) {
        getPrefs(context).edit().putString(KEY_TELEGRAM_BOT_TOKEN, token.trim()).apply()
    }

    fun getTelegramChatId(context: Context): String {
        return getPrefs(context).getString(KEY_TELEGRAM_CHAT_ID, "").orEmpty()
    }

    fun setTelegramChatId(context: Context, chatId: String) {
        getPrefs(context).edit().putString(KEY_TELEGRAM_CHAT_ID, chatId.trim()).apply()
    }

    fun isTriggerOnSimEjectEnabled(context: Context): Boolean {
        return getPrefs(context).getBoolean(KEY_TRIGGER_ON_SIM_EJECT, true)
    }

    fun setTriggerOnSimEjectEnabled(context: Context, enabled: Boolean) {
        getPrefs(context).edit().putBoolean(KEY_TRIGGER_ON_SIM_EJECT, enabled).apply()
    }

    fun isStealthModeActive(context: Context): Boolean {
        return getPrefs(context).getBoolean(KEY_STEALTH_MODE_ACTIVE, false)
    }

    fun setStealthModeActive(context: Context, active: Boolean) {
        getPrefs(context).edit().putBoolean(KEY_STEALTH_MODE_ACTIVE, active).apply()
    }

    fun recordIntruderIncident(context: Context) {
        getPrefs(context).edit().putLong(KEY_LAST_INTRUDER_TIMESTAMP, System.currentTimeMillis()).apply()
    }

    fun getLastIntruderTimestamp(context: Context): Long {
        return getPrefs(context).getLong(KEY_LAST_INTRUDER_TIMESTAMP, 0L)
    }

    private const val KEY_NETLIFY_BASE_URL = "netlify_tracking_base_url"
    const val DEFAULT_NETLIFY_BASE_URL = "https://zipper-sentry-radar.netlify.app"
    private const val KEY_ACTIVE_SESSION_ID = "active_tracking_session_id"
    const val DEFAULT_SESSION_ID = "SESSION_A26_LATEST"

    fun getNetlifyBaseUrl(context: Context): String {
        return getPrefs(context).getString(KEY_NETLIFY_BASE_URL, DEFAULT_NETLIFY_BASE_URL) ?: DEFAULT_NETLIFY_BASE_URL
    }

    fun setNetlifyBaseUrl(context: Context, url: String) {
        getPrefs(context).edit().putString(KEY_NETLIFY_BASE_URL, url.trim()).apply()
    }

    fun getActiveTrackingSessionId(context: Context): String {
        return getPrefs(context).getString(KEY_ACTIVE_SESSION_ID, DEFAULT_SESSION_ID) ?: DEFAULT_SESSION_ID
    }

    fun setActiveTrackingSessionId(context: Context, sessionId: String) {
        getPrefs(context).edit().putString(KEY_ACTIVE_SESSION_ID, sessionId.trim()).apply()
    }
}
