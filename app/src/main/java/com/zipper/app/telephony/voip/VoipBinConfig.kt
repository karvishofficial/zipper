package com.zipper.app.telephony.voip

import android.content.Context
import android.content.SharedPreferences
import com.zipper.app.BuildConfig

/**
 * Configuration model for VoIPBin CPaaS Cloud Telephony & AI Assistant integration.
 */
data class VoipConfig(
    val apiUrl: String = "https://api.voipbin.net/v1.0",
    val apiKey: String = "",
    val assistantId: String = "8caad583-40ce-4374-a656-09c56b06571b",
    val sipUri: String = "sip:direct.c95b295237fe@sip.voipbin.net",
    val isEnabled: Boolean = false,
    val autoForwardNumber: String = "",
    val isVerified: Boolean = false,
    val lastVerified: Long = 0L
) {
    val isConfigured: Boolean
        get() = apiKey.isNotBlank() && assistantId.isNotBlank()
}

/**
 * Manages persistence and retrieval of VoIPBin telephony configuration.
 */
class VoipConfigManager(private val context: Context) {

    companion object {
        private const val PREFS_NAME = "zipper_voipbin_config"
        private const val KEY_API_URL = "api_url"
        private const val KEY_API_KEY = "api_key"
        private const val KEY_ASSISTANT_ID = "assistant_id"
        private const val KEY_SIP_URI = "sip_uri"
        private const val KEY_IS_ENABLED = "is_enabled"
        private const val KEY_AUTO_FORWARD_NUMBER = "auto_forward_number"
        private const val KEY_IS_VERIFIED = "is_verified"
        private const val KEY_LAST_VERIFIED = "last_verified"

        const val DEFAULT_API_URL = "https://api.voipbin.net/v1.0"
        const val DEFAULT_ASSISTANT_ID = "8caad583-40ce-4374-a656-09c56b06571b"
        const val DEFAULT_SIP_URI = "sip:direct.c95b295237fe@sip.voipbin.net"
        const val DEFAULT_API_KEY = BuildConfig.VOIPBIN_API_KEY
    }

    private val prefs: SharedPreferences by lazy {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    }

    fun getConfig(): VoipConfig {
        val storedKey = prefs.getString(KEY_API_KEY, "") ?: ""
        val effectiveKey = storedKey.ifBlank { DEFAULT_API_KEY }
        return VoipConfig(
            apiUrl = prefs.getString(KEY_API_URL, DEFAULT_API_URL) ?: DEFAULT_API_URL,
            apiKey = effectiveKey,
            assistantId = prefs.getString(KEY_ASSISTANT_ID, DEFAULT_ASSISTANT_ID) ?: DEFAULT_ASSISTANT_ID,
            sipUri = prefs.getString(KEY_SIP_URI, DEFAULT_SIP_URI) ?: DEFAULT_SIP_URI,
            isEnabled = prefs.getBoolean(KEY_IS_ENABLED, true),
            autoForwardNumber = prefs.getString(KEY_AUTO_FORWARD_NUMBER, "") ?: "",
            isVerified = true,
            lastVerified = System.currentTimeMillis()
        )
    }

    fun saveConfig(config: VoipConfig) {
        prefs.edit()
            .putString(KEY_API_URL, config.apiUrl.ifBlank { DEFAULT_API_URL })
            .putString(KEY_API_KEY, config.apiKey.trim())
            .putString(KEY_ASSISTANT_ID, config.assistantId.ifBlank { DEFAULT_ASSISTANT_ID })
            .putString(KEY_SIP_URI, config.sipUri.ifBlank { DEFAULT_SIP_URI })
            .putBoolean(KEY_IS_ENABLED, config.isEnabled)
            .putString(KEY_AUTO_FORWARD_NUMBER, config.autoForwardNumber.trim())
            .putBoolean(KEY_IS_VERIFIED, config.isVerified)
            .putLong(KEY_LAST_VERIFIED, config.lastVerified)
            .apply()
    }

    fun setVerificationStatus(isVerified: Boolean) {
        prefs.edit()
            .putBoolean(KEY_IS_VERIFIED, isVerified)
            .putLong(KEY_LAST_VERIFIED, System.currentTimeMillis())
            .apply()
    }

    fun setEnabled(enabled: Boolean) {
        prefs.edit()
            .putBoolean(KEY_IS_ENABLED, enabled)
            .apply()
    }
}
