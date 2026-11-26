package com.zipper.app.core.web

import android.content.Context
import android.content.SharedPreferences
import com.zipper.app.BuildConfig

data class WebAiConfig(
    val provider: String = "Groq",
    val apiKey: String = "",
    val model: String = "qwen/qwen3.8-27b",
    val customUrl: String = "",
    val isVerified: Boolean = false,
    val lastVerified: Long = 0L,
    val fallbackProvider: String = "Google Gemini",
    val fallbackApiKey: String = "",
    val fallbackModel: String = "gemini-2.0-flash",
    val secondaryGroqApiKey: String = WebConfigManager.DEFAULT_GROQ_API_KEY
) {
    val isConfigured: Boolean
        get() = apiKey.isNotBlank() ||
                fallbackApiKey.isNotBlank() ||
                secondaryGroqApiKey.isNotBlank() ||
                provider.contains("ChatGPT Web", true) ||
                provider.equals("Groq", true)

    /**
     * Resolves the tiered execution chain:
     * Tier 1: Primary Groq (configured via secrets.properties or user prefs)
     * Tier 2: Secondary Groq (backup key with fast failover model)
     * Tier 3: Cloud Fallback / Additional Providers
     */
    fun getFallbackChain(): List<WebAiConfig> {
        val chain = mutableListOf<WebAiConfig>()

        // 1. Resolve Groq Keys (Primary and Secondary)
        val groq1Key = when {
            provider.equals("Groq", true) && apiKey.isNotBlank() -> apiKey.trim()
            fallbackApiKey.startsWith("gsk_") -> fallbackApiKey.trim()
            else -> WebConfigManager.PRIMARY_GROQ_API_KEY
        }
        val groq2Key = when {
            secondaryGroqApiKey.isNotBlank() && secondaryGroqApiKey != groq1Key -> secondaryGroqApiKey.trim()
            groq1Key != WebConfigManager.SECONDARY_GROQ_API_KEY -> WebConfigManager.SECONDARY_GROQ_API_KEY
            else -> WebConfigManager.SECONDARY_GROQ_API_KEY
        }

        // Prioritize Groq 1 and Groq 2 for instant <300ms sub-second response
        if (groq1Key.isNotBlank()) {
            chain.add(
                WebAiConfig(
                    provider = "Groq",
                    apiKey = groq1Key,
                    model = "qwen/qwen3.8-27b",
                    isVerified = true
                )
            )
        }
        if (groq2Key.isNotBlank() && groq2Key != groq1Key) {
            chain.add(
                WebAiConfig(
                    provider = "Groq",
                    apiKey = groq2Key,
                    model = "llama-3.3-70b-versatile",
                    isVerified = true
                )
            )
        }

        // Tier 3: Additional custom or ChatGPT Web if configured
        if (provider.contains("ChatGPT Web", true) || (apiKey.isNotBlank() && !provider.equals("Groq", true) && !apiKey.startsWith("gsk_"))) {
            chain.add(
                copy(
                    apiKey = apiKey.trim(),
                    isVerified = true
                )
            )
        }

        return chain.distinctBy { "${it.provider}_${it.apiKey}_${it.model}" }
    }

    fun getFallbackConfig(): WebAiConfig? {
        val chain = getFallbackChain()
        return chain.getOrNull(1) ?: chain.getOrNull(2)
    }
}

/**
 * Manages persistence of LLM API configurations for Karvish Web Assistant.
 */
class WebConfigManager(private val context: Context) {

    companion object {
        const val PRIMARY_GROQ_API_KEY = BuildConfig.PRIMARY_GROQ_API_KEY
        const val SECONDARY_GROQ_API_KEY = BuildConfig.SECONDARY_GROQ_API_KEY
        const val DEFAULT_GROQ_API_KEY = PRIMARY_GROQ_API_KEY

        private const val PREFS_NAME = "karvish_web_ai_config"
        private const val KEY_PROVIDER = "provider"
        private const val KEY_API_KEY = "api_key"
        private const val KEY_MODEL = "model"
        private const val KEY_CUSTOM_URL = "custom_url"
        private const val KEY_IS_VERIFIED = "is_verified"
        private const val KEY_LAST_VERIFIED = "last_verified"
        private const val KEY_FALLBACK_PROVIDER = "fallback_provider"
        private const val KEY_FALLBACK_API_KEY = "fallback_api_key"
        private const val KEY_FALLBACK_MODEL = "fallback_model"
        private const val KEY_SECONDARY_GROQ_API_KEY = "secondary_groq_api_key"

        val SUPPORTED_PROVIDERS = listOf(
            "Groq",
            "Google Gemini",
            "OpenAI",
            "OpenRouter",
            "ChatGPT Web (Free)",
            "Custom (OpenAI-compatible)"
        )

        fun getDefaultModelForProvider(provider: String): String {
            return when (provider) {
                "Groq" -> "qwen/qwen3.8-27b"
                "Google Gemini" -> "gemini-3-flash-preview"
                "OpenAI" -> "gpt-4o-mini"
                "OpenRouter" -> "google/gemini-2.0-flash-exp:free"
                "ChatGPT Web (Free)" -> "gpt-4o-guest"
                else -> "custom-model"
            }
        }
    }

    private val prefs: SharedPreferences =
        context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun getConfig(): WebAiConfig {
        val provider = prefs.getString(KEY_PROVIDER, "Groq") ?: "Groq"
        var model = prefs.getString(KEY_MODEL, "qwen/qwen3.8-27b") ?: "qwen/qwen3.8-27b"
        if (provider.equals("Groq", true) && (model.isBlank() || model.contains("llama", true))) {
            model = "qwen/qwen3.8-27b"
            prefs.edit().putString(KEY_MODEL, model).apply()
        }
        if (provider.contains("Gemini", true) && (model.contains("gemini-1.5", true) || model.isBlank())) {
            model = "gemini-2.0-flash"
            prefs.edit().putString(KEY_MODEL, model).apply()
        }

        val fallbackProvider = prefs.getString(KEY_FALLBACK_PROVIDER, "Google Gemini") ?: "Google Gemini"
        var fallbackModel = prefs.getString(KEY_FALLBACK_MODEL, "gemini-2.0-flash") ?: "gemini-2.0-flash"

        var apiKey = prefs.getString(KEY_API_KEY, "") ?: ""
        var fallbackApiKey = prefs.getString(KEY_FALLBACK_API_KEY, "") ?: ""
        val secondaryGroqKey = prefs.getString(KEY_SECONDARY_GROQ_API_KEY, DEFAULT_GROQ_API_KEY) ?: DEFAULT_GROQ_API_KEY

        // If primary is Groq and apiKey is blank, auto-seed with DEFAULT_GROQ_API_KEY
        if (provider.equals("Groq", true) && apiKey.isBlank()) {
            apiKey = DEFAULT_GROQ_API_KEY
            prefs.edit().putString(KEY_API_KEY, apiKey).putBoolean(KEY_IS_VERIFIED, true).apply()
        }

        // Auto-inherit Gemini credentials from LifeRhythmPrefs if fallbackApiKey is blank
        if (fallbackApiKey.isBlank()) {
            try {
                val rhythmPrefs = com.zipper.app.core.intelligence.store.LifeRhythmPrefs(context)
                if (rhythmPrefs.llmApiKey.isNotBlank()) {
                    fallbackApiKey = rhythmPrefs.llmApiKey.trim()
                    prefs.edit().putString(KEY_FALLBACK_API_KEY, fallbackApiKey).apply()
                }
            } catch (_: Exception) {}
        }

        val isVerified = prefs.getBoolean(KEY_IS_VERIFIED, false) || apiKey.isNotBlank()

        return WebAiConfig(
            provider = provider,
            apiKey = apiKey,
            model = model,
            customUrl = prefs.getString(KEY_CUSTOM_URL, "") ?: "",
            isVerified = isVerified,
            lastVerified = prefs.getLong(KEY_LAST_VERIFIED, 0L),
            fallbackProvider = fallbackProvider,
            fallbackApiKey = fallbackApiKey,
            fallbackModel = fallbackModel,
            secondaryGroqApiKey = secondaryGroqKey
        )
    }

    fun saveConfig(config: WebAiConfig) {
        val sanitizedModel = if (config.provider.equals("Groq", true) && (config.model.isBlank() || config.model.contains("llama", true))) {
            "qwen/qwen3.8-27b"
        } else {
            config.model
        }
        val sanitizedFallbackModel = if (config.fallbackProvider.equals("Groq", true) && (config.fallbackModel.isBlank() || config.fallbackModel.contains("llama", true))) {
            "qwen/qwen3.8-27b"
        } else if (config.fallbackProvider.contains("Gemini", true) && (config.fallbackModel.isBlank() || config.fallbackModel.contains("1.5", true))) {
            "gemini-2.0-flash"
        } else {
            config.fallbackModel
        }
        prefs.edit()
            .putString(KEY_PROVIDER, config.provider)
            .putString(KEY_API_KEY, config.apiKey)
            .putString(KEY_MODEL, sanitizedModel)
            .putString(KEY_CUSTOM_URL, config.customUrl)
            .putBoolean(KEY_IS_VERIFIED, config.isVerified)
            .putLong(KEY_LAST_VERIFIED, config.lastVerified)
            .putString(KEY_FALLBACK_PROVIDER, config.fallbackProvider)
            .putString(KEY_FALLBACK_API_KEY, config.fallbackApiKey)
            .putString(KEY_FALLBACK_MODEL, sanitizedFallbackModel)
            .putString(KEY_SECONDARY_GROQ_API_KEY, config.secondaryGroqApiKey)
            .apply()
    }

    fun clearConfig() {
        prefs.edit().clear().apply()
    }
}
