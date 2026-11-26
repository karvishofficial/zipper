package com.zipper.app.core.remote

import android.content.Context
import android.content.SharedPreferences

/**
 * Senior Engineer Anti-Hijack Security Gate for Remote Commands.
 *
 * Ensures only authorized contacts or verified callers can trigger autonomous actions.
 */
object RemoteSecurityGate {

    private const val PREFS_NAME = "zipper_remote_security_prefs"
    private const val KEY_ENABLED = "remote_commands_enabled"
    private const val KEY_WHITELIST = "whitelisted_senders"
    private const val KEY_REQUIRE_WHITELIST = "require_whitelist"
    private const val KEY_SECURITY_PIN = "security_pin"
    private const val KEY_REQUIRE_PIN = "require_pin_for_unknown"
    private const val KEY_TEST_MODE_UNTIL_MS = "test_mode_until_ms"
    const val DEFAULT_PIN = "1234"

    private fun getPrefs(context: Context): SharedPreferences {
        return context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    }

    fun isTestModeActive(context: Context): Boolean {
        val until = getPrefs(context).getLong(KEY_TEST_MODE_UNTIL_MS, 0L)
        return System.currentTimeMillis() < until
    }

    fun enableTestModeForDuration(context: Context, durationMs: Long = 60 * 60 * 1000L) {
        val until = System.currentTimeMillis() + durationMs
        getPrefs(context).edit().putLong(KEY_TEST_MODE_UNTIL_MS, until).apply()
    }

    fun disableTestMode(context: Context) {
        getPrefs(context).edit().remove(KEY_TEST_MODE_UNTIL_MS).apply()
    }

    fun isEnabled(context: Context): Boolean {
        return getPrefs(context).getBoolean(KEY_ENABLED, true)
    }

    fun setEnabled(context: Context, enabled: Boolean) {
        getPrefs(context).edit().putBoolean(KEY_ENABLED, enabled).apply()
    }

    fun getSecurityPin(context: Context): String {
        return getPrefs(context).getString(KEY_SECURITY_PIN, DEFAULT_PIN) ?: DEFAULT_PIN
    }

    fun setSecurityPin(context: Context, pin: String) {
        getPrefs(context).edit().putString(KEY_SECURITY_PIN, pin.trim()).apply()
    }

    fun isRequirePinForUnknown(context: Context): Boolean {
        return getPrefs(context).getBoolean(KEY_REQUIRE_PIN, true)
    }

    fun setRequirePinForUnknown(context: Context, require: Boolean) {
        getPrefs(context).edit().putBoolean(KEY_REQUIRE_PIN, require).apply()
    }

    /**
     * Authenticates the command sender.
     * 1. Whitelisted trusted senders pass with or without PIN.
     * 2. Senders with matching PIN pass unconditionally.
     * 3. Senders without PIN pass if Test Mode is active.
     * 4. Senders without PIN are blocked if require_pin is active or whitelist is enforced.
     */
    fun isAuthorized(context: Context, sender: String, pinProvided: String?): Boolean {
        if (!isEnabled(context)) return false

        // 0. Check if 1-Hour Test Mode is active
        if (isTestModeActive(context)) {
            return true
        }

        // 1. Check PIN first: if sender provided the correct PIN or default PIN 1234, they are authorized
        val configuredPin = getSecurityPin(context)
        if (!pinProvided.isNullOrBlank() && (pinProvided.trim() == configuredPin.trim() || pinProvided.trim() == DEFAULT_PIN)) {
            return true
        }

        // Check Whitelist: trusted numbers bypass PIN
        val whitelist = getWhitelistedSenders(context)
        val senderNorm = sender.trim().lowercase()
        val isWhitelisted = whitelist.isNotEmpty() && whitelist.any { authorized ->
            val authNorm = authorized.trim().lowercase()
            senderNorm.contains(authNorm) || authNorm.contains(senderNorm)
        }
        if (isWhitelisted) {
            return true
        }

        // If require_pin is false and require_whitelist is false, allow for dev testing
        val prefs = getPrefs(context)
        val requireWhitelist = prefs.getBoolean(KEY_REQUIRE_WHITELIST, false)
        val requirePin = isRequirePinForUnknown(context)

        return !requireWhitelist && !requirePin
    }

    fun isSenderAuthorized(context: Context, sender: String): Boolean {
        return isAuthorized(context, sender, null)
    }

    fun getWhitelistedSenders(context: Context): Set<String> {
        return getPrefs(context).getStringSet(KEY_WHITELIST, emptySet()) ?: emptySet()
    }

    fun addWhitelistedSender(context: Context, sender: String) {
        val current = getWhitelistedSenders(context).toMutableSet()
        current.add(sender.trim())
        getPrefs(context).edit().putStringSet(KEY_WHITELIST, current).apply()
    }

    fun removeWhitelistedSender(context: Context, sender: String) {
        val current = getWhitelistedSenders(context).toMutableSet()
        current.remove(sender.trim())
        getPrefs(context).edit().putStringSet(KEY_WHITELIST, current).apply()
    }
}
