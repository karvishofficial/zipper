package com.zipper.app.core.intelligence.voice

import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import com.zipper.app.BuildConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL

data class TencentRtcConfig(
    val sdkAppId: Long = BuildConfig.TENCENT_SDK_APP_ID,
    val sdkSecretKey: String = BuildConfig.TENCENT_SDK_SECRET_KEY,
    val secretId: String = BuildConfig.TENCENT_SECRET_ID,
    val secretKey: String = BuildConfig.TENCENT_SECRET_KEY,
    val region: String = "ap-singapore",
    val webClientBaseUrl: String = "https://zipper-call.netlify.app",
    val isEnabled: Boolean = true
) {
    val isConfigured: Boolean
        get() = sdkAppId > 0 && sdkSecretKey.isNotBlank() && secretId.isNotBlank() && secretKey.isNotBlank()
}

/**
 * Manager for Tencent RTC and Conversational AI services.
 * Handles server-side StartAIConversation / StopAIConversation lifecycle and UserSig generation.
 */
class TencentRtcManager private constructor(context: Context) {

    private val prefs: SharedPreferences =
        context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    companion object {
        private const val TAG = "TencentRtcManager"
        private const val PREFS_NAME = "zipper_tencent_rtc_prefs"

        private const val KEY_SDK_APP_ID = "sdk_app_id"
        private const val KEY_SDK_SECRET_KEY = "sdk_secret_key"
        private const val KEY_SECRET_ID = "secret_id"
        private const val KEY_SECRET_KEY = "secret_key"
        private const val KEY_REGION = "region"
        private const val KEY_WEB_CLIENT_BASE_URL = "web_client_base_url"
        private const val KEY_IS_ENABLED = "is_enabled"

        private const val TRTC_API_HOST = "trtc.intl.tencentcloudapi.com"
        private const val TRTC_API_VERSION = "2019-07-22"

        @Volatile
        private var instance: TencentRtcManager? = null

        fun getInstance(context: Context): TencentRtcManager {
            return instance ?: synchronized(this) {
                instance ?: TencentRtcManager(context).also { instance = it }
            }
        }
    }

    fun getConfig(): TencentRtcConfig {
        return TencentRtcConfig(
            sdkAppId = prefs.getLong(KEY_SDK_APP_ID, BuildConfig.TENCENT_SDK_APP_ID),
            sdkSecretKey = prefs.getString(KEY_SDK_SECRET_KEY, BuildConfig.TENCENT_SDK_SECRET_KEY) ?: "",
            secretId = prefs.getString(KEY_SECRET_ID, BuildConfig.TENCENT_SECRET_ID) ?: "",
            secretKey = prefs.getString(KEY_SECRET_KEY, BuildConfig.TENCENT_SECRET_KEY) ?: "",
            region = prefs.getString(KEY_REGION, "ap-singapore") ?: "ap-singapore",
            webClientBaseUrl = run {
                val saved = prefs.getString(KEY_WEB_CLIENT_BASE_URL, "https://zipper-call.netlify.app") ?: "https://zipper-call.netlify.app"
                if (saved.contains("zipper-voice-twin.netlify.app") || saved.isBlank()) "https://zipper-call.netlify.app" else saved
            },
            isEnabled = prefs.getBoolean(KEY_IS_ENABLED, true)
        )
    }

    fun saveConfig(config: TencentRtcConfig) {
        prefs.edit()
            .putLong(KEY_SDK_APP_ID, config.sdkAppId)
            .putString(KEY_SDK_SECRET_KEY, config.sdkSecretKey)
            .putString(KEY_SECRET_ID, config.secretId)
            .putString(KEY_SECRET_KEY, config.secretKey)
            .putString(KEY_REGION, config.region)
            .putString(KEY_WEB_CLIENT_BASE_URL, config.webClientBaseUrl)
            .putBoolean(KEY_IS_ENABLED, config.isEnabled)
            .apply()
    }

    /**
     * Spawns a conversational AI agent into the TRTC room to conduct a real-time voice call.
     */
    suspend fun startAiConversation(
        roomId: String,
        aiUserId: String = "zipper_voice_twin",
        welcomeMessage: String,
        systemPrompt: String,
        llmApiKey: String,
        llmApiUrl: String = "https://api.groq.com/openai/v1/chat/completions",
        llmModel: String = "llama-3.3-70b-versatile"
    ): Result<String> = withContext(Dispatchers.IO) {
        val config = getConfig()
        if (!config.isConfigured || !config.isEnabled) {
            return@withContext Result.failure(IllegalStateException("Tencent RTC is not enabled or credentials missing"))
        }

        try {
            val userSig = TrtcUserSigGenerator.genUserSig(
                sdkAppId = config.sdkAppId,
                secretKey = config.sdkSecretKey,
                userId = aiUserId
            )

            val sessionId = "session_${roomId}_${System.currentTimeMillis()}"

            // Build LLMConfig JSON string
            val llmConfigJson = JSONObject().apply {
                put("LLMType", "openai")
                put("Model", llmModel)
                put("APIKey", llmApiKey)
                put("APIUrl", llmApiUrl)
                put("Streaming", true)
                put("SystemPrompt", systemPrompt)
            }.toString()

            val agentConfig = JSONObject().apply {
                put("UserId", aiUserId)
                put("UserSig", userSig)
                put("WelcomeMessage", welcomeMessage)
                put("InterruptMode", 0) // Intelligent full-duplex barge-in
                put("InterruptSpeechDuration", 500)
                put("MaxIdleTime", 120) // Terminate if silence exceeds 2 minutes
            }

            val requestBody = JSONObject().apply {
                put("SdkAppId", config.sdkAppId)
                put("RoomId", roomId)
                put("RoomIdType", 1) // 1 = string roomId
                put("SessionId", sessionId)
                put("AgentConfig", agentConfig)
                put("STTConfig", JSONObject().put("Language", "en"))
                put("LLMConfig", llmConfigJson)
                put("TTSConfig", JSONObject().put("TTSType", "tencent"))
            }.toString()

            val headers = TencentCloudSigner.signRequest(
                secretId = config.secretId,
                secretKey = config.secretKey,
                service = "trtc",
                host = TRTC_API_HOST,
                action = "StartAIConversation",
                version = TRTC_API_VERSION,
                region = config.region,
                payloadJson = requestBody
            )

            val connection = (URL("https://$TRTC_API_HOST").openConnection() as HttpURLConnection).apply {
                requestMethod = "POST"
                connectTimeout = 8000
                readTimeout = 8000
                doOutput = true
                headers.forEach { (k, v) -> setRequestProperty(k, v) }
            }

            OutputStreamWriter(connection.outputStream, "UTF-8").use { os ->
                os.write(requestBody)
                os.flush()
            }

            val statusCode = connection.responseCode
            val responseStream = if (statusCode in 200..299) connection.inputStream else connection.errorStream
            val responseText = BufferedReader(InputStreamReader(responseStream, "UTF-8")).use { it.readText() }

            Log.i(TAG, "StartAIConversation HTTP $statusCode: $responseText")

            if (statusCode in 200..299) {
                val json = JSONObject(responseText)
                val respObj = json.optJSONObject("Response")
                val taskId = respObj?.optString("TaskId").orEmpty()
                val finalSessionId = respObj?.optString("SessionId").orEmpty().ifBlank { sessionId }
                Result.success(taskId.ifBlank { finalSessionId })
            } else {
                Result.failure(Exception("HTTP $statusCode: $responseText"))
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start AI conversation: ${e.message}", e)
            Result.failure(e)
        }
    }

    /**
     * Stops an active AI conversation task.
     */
    suspend fun stopAiConversation(taskId: String): Result<Boolean> = withContext(Dispatchers.IO) {
        val config = getConfig()
        try {
            val requestBody = JSONObject().apply {
                put("SdkAppId", config.sdkAppId)
                put("TaskId", taskId)
            }.toString()

            val headers = TencentCloudSigner.signRequest(
                secretId = config.secretId,
                secretKey = config.secretKey,
                service = "trtc",
                host = TRTC_API_HOST,
                action = "StopAIConversation",
                version = TRTC_API_VERSION,
                region = config.region,
                payloadJson = requestBody
            )

            val connection = (URL("https://$TRTC_API_HOST").openConnection() as HttpURLConnection).apply {
                requestMethod = "POST"
                connectTimeout = 6000
                readTimeout = 6000
                doOutput = true
                headers.forEach { (k, v) -> setRequestProperty(k, v) }
            }

            OutputStreamWriter(connection.outputStream, "UTF-8").use { os ->
                os.write(requestBody)
                os.flush()
            }

            val statusCode = connection.responseCode
            if (statusCode in 200..299) {
                Result.success(true)
            } else {
                Result.failure(Exception("HTTP $statusCode"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Generates a 1-tap browser voice call link for a human caller.
     */
    fun generateCallerWebLink(roomId: String, callerName: String): String {
        val config = getConfig()
        val callerUserId = "caller_${callerName.lowercase().replace(Regex("[^a-z0-9]"), "_").take(16)}"
        val userSig = TrtcUserSigGenerator.genUserSig(
            sdkAppId = config.sdkAppId,
            secretKey = config.sdkSecretKey,
            userId = callerUserId,
            expireSeconds = 86400
        )

        val baseUrl = config.webClientBaseUrl.trimEnd('/')
        val encodedCaller = try { java.net.URLEncoder.encode(callerName, "UTF-8") } catch (_: Exception) { callerName }
        return "$baseUrl/?sdkAppId=${config.sdkAppId}&roomId=$roomId&userId=$callerUserId&userSig=$userSig&callerName=$encodedCaller"
    }
}
