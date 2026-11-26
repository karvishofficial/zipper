package com.zipper.app.telephony.voip

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL

data class VoipCallResponse(
    val callId: String,
    val status: String,
    val phoneNumber: String,
    val createdAt: String = "",
    val rawResponse: String = ""
)

data class VoipCallDetails(
    val callId: String,
    val status: String, // "initiated", "ringing", "in-progress", "completed", "failed", "busy"
    val durationSeconds: Int = 0,
    val transcript: String = "",
    val summary: String = "",
    val isEnded: Boolean = false
)

/**
 * High-performance HTTP client for VoIPBin CPaaS voice calls and AI assistant bridging.
 * Zero-dependency, lightweight, non-blocking via Kotlin Coroutines.
 */
class VoipBinClient(private val config: VoipConfig) {

    companion object {
        private const val TAG = "VoipBinClient"
        private const val TIMEOUT_MS = 15000
    }

    /**
     * Authenticates with VoIPBin using username and password to retrieve the JWT access token.
     */
    suspend fun login(username: String, password: String): Result<String> = withContext(Dispatchers.IO) {
        try {
            val endpoint = "${config.apiUrl.trimEnd('/')}/auth/login"
            val requestBody = JSONObject().apply {
                put("username", username.trim())
                put("password", password.trim())
            }.toString()

            val (code, response) = executeHttpRequest(endpoint, "POST", requestBody)
            if (code in 200..299) {
                val json = JSONObject(response)
                val token = json.optString("token", "")
                if (token.isNotBlank()) {
                    Result.success(token)
                } else {
                    Result.failure(Exception("Login succeeded but token was empty."))
                }
            } else {
                Result.failure(Exception("Login failed (HTTP $code): $response"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Initiates a live conversational AI outbound phone call via VoIPBin.
     * The cloud assistant will dial the target number, converse using Groq, and follow the specified goal.
     */
    suspend fun createCall(
        phoneNumber: String,
        goal: String,
        targetName: String = "Contact"
    ): Result<VoipCallResponse> = withContext(Dispatchers.IO) {
        if (!config.isConfigured) {
            return@withContext Result.failure(IllegalStateException("VoIPBin API key or Assistant ID is not configured."))
        }

        try {
            val endpoint = "${config.apiUrl.trimEnd('/')}/calls"
            val requestBody = JSONObject().apply {
                put("source", JSONObject().apply {
                    put("type", "ai")
                    put("target", config.assistantId)
                })
                put("destinations", org.json.JSONArray().apply {
                    put(JSONObject().apply {
                        put("type", "tel")
                        put("target", phoneNumber)
                    })
                })
                put("variables", JSONObject().apply {
                    put("goal", goal)
                    put("contact_name", targetName)
                    put("user_name", "Karthik")
                })
            }.toString()

            Log.i(TAG, "Creating outbound VoIP call to $targetName ($phoneNumber) via assistant ${config.assistantId}")

            val (code, response) = executeHttpRequest(endpoint, "POST", requestBody)
            if (code in 200..299) {
                val json = JSONObject(response)
                val callId = json.optString("id", json.optString("call_id", "voip_${System.currentTimeMillis()}"))
                val status = json.optString("status", "initiated")
                val number = json.optString("phone_number", phoneNumber)

                Log.i(TAG, "VoIP call created successfully! Call ID: $callId, Status: $status")
                Result.success(VoipCallResponse(callId, status, number, rawResponse = response))
            } else if (code == 403 && response.contains("IDENTITY_VERIFICATION_REQUIRED", true)) {
                Log.w(TAG, "VoIPBin Outbound PSTN requires Customer Identity Verification in admin.voipbin.net.")
                Result.failure(Exception("Identity Verification Required on VoIPBin for external PSTN dialing. (Falling back to SIM call)"))
            } else {
                Log.e(TAG, "VoIP call creation failed with code $code: $response")
                Result.failure(Exception("VoIPBin HTTP $code: $response"))
            }
        } catch (e: Exception) {
            Log.e(TAG, "Exception creating VoIP call: ${e.message}", e)
            Result.failure(e)
        }
    }

    /**
     * Queries the live status and transcript of an active or completed call.
     */
    suspend fun getCall(callId: String): Result<VoipCallDetails> = withContext(Dispatchers.IO) {
        if (!config.isConfigured) {
            return@withContext Result.failure(IllegalStateException("VoIPBin not configured."))
        }

        try {
            val endpoint = "${config.apiUrl.trimEnd('/')}/calls/$callId"
            val (code, response) = executeHttpRequest(endpoint, "GET", null)

            if (code in 200..299) {
                val json = JSONObject(response)
                val status = json.optString("status", "unknown").lowercase()
                val duration = json.optInt("duration", 0)
                val transcript = json.optString("transcript", "")
                val summary = json.optString("summary", "")

                val isEnded = status in listOf("completed", "ended", "failed", "busy", "canceled", "no-answer")
                Result.success(VoipCallDetails(callId, status, duration, transcript, summary, isEnded))
            } else {
                Result.failure(Exception("Failed to get call status ($code): $response"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Terminates an active call immediately.
     */
    suspend fun endCall(callId: String): Result<Boolean> = withContext(Dispatchers.IO) {
        try {
            val endpoint = "${config.apiUrl.trimEnd('/')}/calls/$callId"
            val (code, response) = executeHttpRequest(endpoint, "DELETE", null)
            if (code in 200..299) {
                Result.success(true)
            } else {
                Result.failure(Exception("Failed to end call ($code): $response"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Verifies API Key validity and Assistant accessibility by probing the VoIPBin API.
     */
    suspend fun verifyConnection(): Result<String> = withContext(Dispatchers.IO) {
        if (!config.isConfigured) {
            return@withContext Result.failure(IllegalStateException("API key or Assistant ID is blank."))
        }

        try {
            // Test query on the calls or agent endpoint
            val endpoint = "${config.apiUrl.trimEnd('/')}/calls?limit=1"
            val (code, response) = executeHttpRequest(endpoint, "GET", null)

            if (code in 200..299) {
                Result.success("Connected to VoIPBin Cloud (Assistant ${config.assistantId})")
            } else if (code == 401 || code == 403) {
                Result.failure(Exception("Authentication failed (HTTP $code). Please check your VoIPBin API key."))
            } else {
                Result.failure(Exception("VoIPBin responded with HTTP $code: $response"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    private fun executeHttpRequest(
        urlStr: String,
        method: String,
        body: String?
    ): Pair<Int, String> {
        val finalUrl = if (config.apiKey.isNotBlank() && !urlStr.contains("accesskey=")) {
            val delimiter = if (urlStr.contains("?")) "&" else "?"
            "$urlStr${delimiter}accesskey=${config.apiKey}"
        } else {
            urlStr
        }

        val url = URL(finalUrl)
        val conn = (url.openConnection() as HttpURLConnection).apply {
            requestMethod = method
            connectTimeout = TIMEOUT_MS
            readTimeout = TIMEOUT_MS
            setRequestProperty("Content-Type", "application/json")
            setRequestProperty("Accept", "application/json")

            if (body != null && (method == "POST" || method == "PUT")) {
                doOutput = true
            }
        }

        if (body != null && conn.doOutput) {
            OutputStreamWriter(conn.outputStream, "UTF-8").use { os ->
                os.write(body)
                os.flush()
            }
        }

        val code = conn.responseCode
        val stream = if (code in 200..299) conn.inputStream else conn.errorStream
        val responseText = stream?.use { s ->
            BufferedReader(InputStreamReader(s, "UTF-8")).readText()
        } ?: ""

        return Pair(code, responseText)
    }
}
