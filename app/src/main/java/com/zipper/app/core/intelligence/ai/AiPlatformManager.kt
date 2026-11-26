package com.zipper.app.core.intelligence.ai

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL

/**
 * Universal AI Platform & Model Discovery Engine for Karvish / Zipper.
 *
 * Supports live dynamic discovery of available models with ZERO hardcoding.
 * Connects directly to Google Gemini, xAI Grok, NVIDIA NIM, OpenAI, Groq, and Custom endpoints.
 */
object AiPlatformManager {

    private const val TAG = "AiPlatformManager"

    data class PlatformConfig(
        val id: String,
        val displayName: String,
        val defaultBaseUrl: String,
        val isOpenAiCompatible: Boolean
    )

    val SUPPORTED_PLATFORMS = listOf(
        PlatformConfig("gemini", "Google Gemini", "https://generativelanguage.googleapis.com/v1beta", isOpenAiCompatible = false),
        PlatformConfig("grok", "xAI Grok", "https://api.x.ai/v1", isOpenAiCompatible = true),
        PlatformConfig("nvidia", "NVIDIA NIM", "https://integrate.api.nvidia.com/v1", isOpenAiCompatible = true),
        PlatformConfig("groq", "Groq Cloud", "https://api.groq.com/openai/v1", isOpenAiCompatible = true),
        PlatformConfig("openai", "OpenAI", "https://api.openai.com/v1", isOpenAiCompatible = true),
        PlatformConfig("custom", "Custom / Self-Hosted", "http://localhost:11434/v1", isOpenAiCompatible = true)
    )

    fun getPlatform(id: String): PlatformConfig {
        return SUPPORTED_PLATFORMS.firstOrNull { it.id.equals(id, ignoreCase = true) }
            ?: SUPPORTED_PLATFORMS[0]
    }

    /**
     * Dynamically queries the provider's /models endpoint to fetch actual available models.
     * ZERO hardcoded models.
     */
    suspend fun fetchLiveModels(
        platformId: String,
        apiKey: String,
        baseUrl: String = ""
    ): Result<List<String>> = withContext(Dispatchers.IO) {
        val platform = getPlatform(platformId)
        val effectiveBase = baseUrl.ifBlank { platform.defaultBaseUrl }.trimEnd('/')

        try {
            val endpointUrl = if (!platform.isOpenAiCompatible) {
                // Gemini Discovery URL
                "$effectiveBase/models?key=$apiKey"
            } else {
                // OpenAI-compatible Discovery URL
                "$effectiveBase/models"
            }

            val url = URL(endpointUrl)
            val conn = (url.openConnection() as HttpURLConnection).apply {
                requestMethod = "GET"
                connectTimeout = 8000
                readTimeout = 8000
                if (platform.isOpenAiCompatible && apiKey.isNotBlank()) {
                    setRequestProperty("Authorization", "Bearer $apiKey")
                }
            }

            val responseCode = conn.responseCode
            if (responseCode != HttpURLConnection.HTTP_OK) {
                val errText = BufferedReader(InputStreamReader(conn.errorStream ?: conn.inputStream)).use { it.readText() }
                return@withContext Result.failure(Exception("HTTP $responseCode: ${extractErrorMessage(errText)}"))
            }

            val responseText = BufferedReader(InputStreamReader(conn.inputStream)).use { it.readText() }
            val models = parseModelsResponse(responseText, platform.isOpenAiCompatible)
            if (models.isEmpty()) {
                return@withContext Result.failure(Exception("No compatible chat models found in account."))
            }

            Log.i(TAG, "Successfully fetched ${models.size} live models for platform ${platform.id}")
            Result.success(models)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to fetch live models for ${platform.id}: ${e.message}")
            Result.failure(e)
        }
    }

    /**
     * Tests connection with a tiny test prompt and returns round-trip latency in milliseconds.
     */
    suspend fun testConnection(
        platformId: String,
        apiKey: String,
        model: String,
        baseUrl: String = ""
    ): Result<Long> = withContext(Dispatchers.IO) {
        val platform = getPlatform(platformId)
        val effectiveBase = baseUrl.ifBlank { platform.defaultBaseUrl }.trimEnd('/')
        val startMs = System.currentTimeMillis()

        try {
            if (!platform.isOpenAiCompatible) {
                // Gemini Test Call
                val endpoint = "$effectiveBase/models/$model:generateContent?key=$apiKey"
                val url = URL(endpoint)
                val conn = (url.openConnection() as HttpURLConnection).apply {
                    requestMethod = "POST"
                    connectTimeout = 10000
                    readTimeout = 10000
                    doOutput = true
                    setRequestProperty("Content-Type", "application/json")
                }

                val body = JSONObject().apply {
                    val contents = JSONArray().apply {
                        val part = JSONObject().apply { put("text", "Hi") }
                        put(JSONObject().apply { put("parts", JSONArray().apply { put(part) }) })
                    }
                    put("contents", contents)
                }

                OutputStreamWriter(conn.outputStream).use { it.write(body.toString()) }
                if (conn.responseCode != HttpURLConnection.HTTP_OK) {
                    val errText = BufferedReader(InputStreamReader(conn.errorStream ?: conn.inputStream)).use { it.readText() }
                    return@withContext Result.failure(Exception("HTTP ${conn.responseCode}: ${extractErrorMessage(errText)}"))
                }
            } else {
                // OpenAI-compatible Test Call
                val endpoint = "$effectiveBase/chat/completions"
                val url = URL(endpoint)
                val conn = (url.openConnection() as HttpURLConnection).apply {
                    requestMethod = "POST"
                    connectTimeout = 10000
                    readTimeout = 10000
                    doOutput = true
                    setRequestProperty("Content-Type", "application/json")
                    setRequestProperty("Authorization", "Bearer $apiKey")
                }

                val body = JSONObject().apply {
                    put("model", model)
                    put("max_tokens", 5)
                    val messages = JSONArray().apply {
                        put(JSONObject().apply {
                            put("role", "user")
                            put("content", "Hi")
                        })
                    }
                    put("messages", messages)
                }

                OutputStreamWriter(conn.outputStream).use { it.write(body.toString()) }
                if (conn.responseCode != HttpURLConnection.HTTP_OK) {
                    val errText = BufferedReader(InputStreamReader(conn.errorStream ?: conn.inputStream)).use { it.readText() }
                    return@withContext Result.failure(Exception("HTTP ${conn.responseCode}: ${extractErrorMessage(errText)}"))
                }
            }

            val latency = System.currentTimeMillis() - startMs
            Result.success(latency)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    private fun parseModelsResponse(jsonStr: String, isOpenAi: Boolean): List<String> {
        val result = mutableListOf<String>()
        val root = JSONObject(jsonStr)

        if (!isOpenAi) {
            // Gemini format: {"models": [{"name": "models/gemini-1.5-flash", "supportedGenerationMethods": ["generateContent"]}]}
            val modelsArr = root.optJSONArray("models") ?: return emptyList()
            for (i in 0 until modelsArr.length()) {
                val m = modelsArr.getJSONObject(i)
                val name = m.optString("name").removePrefix("models/")
                val methods = m.optJSONArray("supportedGenerationMethods")
                val isChat = if (methods != null) {
                    (0 until methods.length()).any { methods.getString(it) == "generateContent" }
                } else true

                if (isChat && name.isNotBlank() && !name.contains("embedding") && !name.contains("aqa")) {
                    result.add(name)
                }
            }
        } else {
            // OpenAI / Grok / Nvidia / Groq format: {"data": [{"id": "meta/llama-3.3-70b-instruct"}]}
            val dataArr = root.optJSONArray("data") ?: return emptyList()
            for (i in 0 until dataArr.length()) {
                val m = dataArr.getJSONObject(i)
                val id = m.optString("id")
                if (id.isNotBlank() && !id.contains("embedding") && !id.contains("whisper") && !id.contains("dall-e") && !id.contains("tts")) {
                    result.add(id)
                }
            }
        }

        return result.distinct().sorted()
    }

    private fun extractErrorMessage(errJson: String): String {
        return try {
            val root = JSONObject(errJson)
            root.optJSONObject("error")?.optString("message")
                ?: root.optString("message", errJson.take(120))
        } catch (_: Exception) {
            errJson.take(120)
        }
    }
}
