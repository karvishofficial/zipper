package com.zipper.app.core.web

import android.util.Log
import com.zipper.app.core.model.DistilledNode
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL

sealed class VerificationResult {
    data class Success(val provider: String, val model: String) : VerificationResult()
    data class Failure(val errorMessage: String) : VerificationResult()
}

data class PlannedWebAction(
    val action: String,
    val targetIndex: Int = -1,
    val text: String? = null,
    val value: String? = null,
    val description: String? = null
)

/**
 * Universal Multi-Platform LLM Client for Karvish Web Assistant.
 * Supports: Google Gemini, Groq, OpenAI, OpenRouter, and Custom (OpenAI-compatible) APIs.
 */
class WebLlmClient {

    companion object {
        private const val TAG = "WebLlmClient"
        private const val TIMEOUT_MS = 6000
    }

    /**
     * Verifies the provided API configuration using a low-overhead, 1-token test ping.
     */
    suspend fun verifyKey(config: WebAiConfig): VerificationResult = withContext(Dispatchers.IO) {
        if (config.provider.contains("ChatGPT Web", true)) {
            return@withContext VerificationResult.Success(config.provider, "gpt-4o-guest (No Login Required)")
        }

        if (config.apiKey.isBlank() && !config.provider.contains("Custom", true)) {
            return@withContext VerificationResult.Failure("API key cannot be empty.")
        }

        try {
            val (endpoint, requestBody, headers) = buildVerificationRequest(config)
            val connection = (URL(endpoint).openConnection() as HttpURLConnection).apply {
                requestMethod = "POST"
                connectTimeout = TIMEOUT_MS
                readTimeout = TIMEOUT_MS
                doOutput = true
                headers.forEach { (k, v) -> setRequestProperty(k, v) }
            }

            OutputStreamWriter(connection.outputStream, "UTF-8").use { os ->
                os.write(requestBody)
                os.flush()
            }

            val statusCode = connection.responseCode
            val responseBody = readResponseString(connection, statusCode)

            if (statusCode in 200..299) {
                VerificationResult.Success(config.provider, config.model)
            } else {
                val errorDetail = parseErrorMessage(responseBody, statusCode)
                VerificationResult.Failure(errorDetail)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Verification exception for ${config.provider}", e)
            VerificationResult.Failure("Network error: ${e.message ?: "Failed to connect"}")
        }
    }

    /**
     * Uses LLM to extract the clean, exact target URL from natural language query.
     * Prevents commas, trailing punctuation, or typos from breaking navigation.
     */
    suspend fun resolveTargetUrl(goal: String, config: WebAiConfig): String? = withContext(Dispatchers.IO) {
        val prompt = """
            Extract the target website URL from this user instruction.
            Remove any trailing commas, periods, or punctuation.
            If a port number is present (e.g. :4043), preserve it.
            If no protocol is provided, prefix with https://.
            If the query is a general search without a specific URL/domain, respond with NONE.
            
            Instruction: $goal
            
            Respond ONLY with the clean URL (e.g. https://rrce.contineo.in:4043/students/index.php) or NONE.
        """.trimIndent()

        val chain = config.getFallbackChain()
        for (tier in chain) {
            if (!tier.isConfigured && tier.apiKey.isBlank() && !tier.provider.contains("ChatGPT Web", true)) continue

            if (tier.provider.contains("ChatGPT Web", true)) {
                val bridgeResult = com.zipper.app.core.web.chatgpt.ChatGptHeadlessBridge.ask(null, prompt)
                val raw = bridgeResult.getOrNull()?.trim()?.replace("`", "")?.trim()
                if (!raw.isNullOrBlank() && !raw.equals("NONE", true) && (raw.contains("http://") || raw.contains("https://") || raw.contains("."))) {
                    val clean = raw.lines().firstOrNull()?.trim()
                        ?.trimEnd(',', '.', ';', ':', ')', ']', '}', '"', '\'') ?: ""
                    if (clean.isNotBlank()) {
                        return@withContext if (clean.startsWith("http://") || clean.startsWith("https://")) clean else "https://$clean"
                    }
                }
                continue
            }

            try {
                val (endpoint, requestBody, headers) = buildPlanningRequest(tier, prompt)
                val connection = (URL(endpoint).openConnection() as HttpURLConnection).apply {
                    requestMethod = "POST"
                    connectTimeout = 5000
                    readTimeout = 5000
                    doOutput = true
                    headers.forEach { (k, v) -> setRequestProperty(k, v) }
                }

                OutputStreamWriter(connection.outputStream, "UTF-8").use { os ->
                    os.write(requestBody)
                    os.flush()
                }

                val statusCode = connection.responseCode
                if (statusCode in 200..299) {
                    val responseBody = readResponseString(connection, statusCode)
                    val raw = extractLlmResponseText(tier.provider, responseBody).trim()
                        .replace("`", "")
                        .trim()
                    if (raw.isNotBlank() && !raw.equals("NONE", true) && (raw.contains("http://") || raw.contains("https://") || raw.contains("."))) {
                        val clean = raw.lines().firstOrNull()?.trim()
                            ?.trimEnd(',', '.', ';', ':', ')', ']', '}', '"', '\'') ?: ""
                        if (clean.isNotBlank()) {
                            return@withContext if (clean.startsWith("http://") || clean.startsWith("https://")) clean else "https://$clean"
                        }
                    }
                } else {
                    Log.w(TAG, "resolveTargetUrl ${tier.provider} returned HTTP $statusCode, trying next tier in chain...")
                }
            } catch (e: Exception) {
                Log.w(TAG, "Failed to resolve target URL via ${tier.provider}: ${e.message}, trying next tier in chain...")
            }
        }
        null
    }

    /**
     * Calls the configured LLM with distilled interactive elements to plan sequential form-filling actions.
     */
    suspend fun planWebActions(
        goal: String,
        currentUrl: String,
        nodes: List<DistilledNode>,
        config: WebAiConfig
    ): List<PlannedWebAction> = withContext(Dispatchers.IO) {
        val interactiveNodes = nodes.filter {
            it.isClickable || it.isEditable || it.role in setOf("SELECT", "TEXT_FIELD", "SEARCH_FIELD", "BUTTON", "MENU_ITEM")
        }.take(35)

        Log.i(TAG, "Planning via ${config.provider} with ${interactiveNodes.size} interactive nodes for goal: '$goal'")
        val prompt = buildDistilledPrompt(goal, currentUrl, interactiveNodes)
        val chain = config.getFallbackChain()

        for (tier in chain) {
            if (!tier.isConfigured && tier.apiKey.isBlank() && !tier.provider.contains("ChatGPT Web", true)) continue

            if (tier.provider.contains("ChatGPT Web", true)) {
                val bridgeResult = com.zipper.app.core.web.chatgpt.ChatGptHeadlessBridge.ask(null, prompt)
                val rawContent = bridgeResult.getOrNull()
                if (!rawContent.isNullOrBlank()) {
                    val actions = parsePlannedActions(rawContent)
                    if (actions.isNotEmpty()) {
                        Log.i(TAG, "Parsed ${actions.size} planned actions from ChatGPT Web")
                        return@withContext actions
                    }
                }
                continue
            }

            try {
                val (endpoint, requestBody, headers) = buildPlanningRequest(tier, prompt)
                val connection = (URL(endpoint).openConnection() as HttpURLConnection).apply {
                    requestMethod = "POST"
                    connectTimeout = TIMEOUT_MS
                    readTimeout = TIMEOUT_MS
                    doOutput = true
                    headers.forEach { (k, v) -> setRequestProperty(k, v) }
                }

                OutputStreamWriter(connection.outputStream, "UTF-8").use { os ->
                    os.write(requestBody)
                    os.flush()
                }

                val statusCode = connection.responseCode
                if (statusCode in 200..299) {
                    val responseBody = readResponseString(connection, statusCode)
                    val rawContent = extractLlmResponseText(tier.provider, responseBody)
                    Log.i(TAG, "Raw LLM plan response: $rawContent")
                    val actions = parsePlannedActions(rawContent)
                    if (actions.isNotEmpty()) {
                        Log.i(TAG, "Parsed ${actions.size} planned actions from ${tier.provider} (${tier.model})")
                        return@withContext actions
                    }
                } else {
                    Log.w(TAG, "Planning request failed on ${tier.provider} with code $statusCode: trying next fallback...")
                }
            } catch (e: Exception) {
                Log.w(TAG, "Failed to plan web actions via ${tier.provider}: ${e.message}, trying next fallback...")
            }
        }
        emptyList()
    }

    private fun buildDistilledPrompt(
        goal: String,
        currentUrl: String,
        nodes: List<DistilledNode>
    ): String {
        val candidatesList = nodes.joinToString("\n") { node ->
            val label = node.semanticLabel.take(90)
            val id = node.resourceId?.takeIf { it.isNotBlank() }?.let { " id=\"$it\"" } ?: ""
            "[${node.index}] ${node.role}$id: $label"
        }

        return """
            You are a web automation planner. Given a user goal and interactive elements on the page, return a JSON array of sequential actions to achieve the goal (e.g. fill inputs, select dropdowns, click submit).
            
            Supported action formats:
            - {"action": "TYPE", "targetIndex": 0, "text": "value"}
            - {"action": "SELECT", "targetIndex": 1, "value": "option"}
            - {"action": "CLICK", "targetIndex": 2}
            - {"action": "SUBMIT"}
            
            Rules:
            1. Match elements by role, ID, placeholder, and label.
            2. For Date of Birth (DOB) or dates: If Day, Month, and Year are separate dropdowns (SELECT), generate 3 separate SELECT actions (e.g. Day "29", Month "Sep" or "September", Year "2005").
            3. For Username / USN: Enter the requested USN/ID into the username or USN text field.
            4. For post-login or secondary verification screens (OTP/PIN/parent mobile/security questions):
               - If there is a SELECT dropdown for verification type (e.g. "Father Mobile Last 4 Digits"), generate a SELECT action for that option.
               - Enter the requested PIN/digits/code into the digit/PIN text field (typing into the first digit box will automatically distribute the digits across split boxes).
               - Click the submit/verify button.
            5. If the current screen is the initial login form and has an unknown visual CAPTCHA or image code, fill USN and DOB and do NOT submit (CAPTCHA is solved autonomously).
            6. ONLY output the valid JSON array of actions. No markdown code blocks, no explanations.
            
            USER GOAL: $goal
            CURRENT URL: $currentUrl
            
            INTERACTIVE ELEMENTS:
            $candidatesList
        """.trimIndent()
    }

    private fun buildVerificationRequest(config: WebAiConfig): Triple<String, String, Map<String, String>> {
        val prompt = "ping"
        return when {
            config.provider.contains("Gemini", true) -> {
                val model = config.model.ifBlank { "gemini-1.5-flash" }
                val url = "https://generativelanguage.googleapis.com/v1beta/models/$model:generateContent?key=${config.apiKey.trim()}"
                val body = JSONObject().apply {
                    put("contents", JSONArray().put(JSONObject().put("parts", JSONArray().put(JSONObject().put("text", prompt)))))
                    put("generationConfig", JSONObject().put("maxOutputTokens", 2))
                }.toString()
                Triple(url, body, mapOf("Content-Type" to "application/json"))
            }
            config.provider.contains("Groq", true) -> {
                val url = "https://api.groq.com/openai/v1/chat/completions"
                val model = if (config.model.isBlank() || config.model.contains("llama", true)) "qwen/qwen3.8-27b" else config.model
                val body = buildOpenAiStyleBody(model, prompt, 2)
                Triple(url, body, mapOf(
                    "Content-Type" to "application/json",
                    "Authorization" to "Bearer ${config.apiKey.trim()}",
                    "User-Agent" to "Mozilla/5.0 (Linux; Android 14; Mobile)"
                ))
            }
            config.provider.contains("OpenRouter", true) -> {
                val url = "https://openrouter.ai/api/v1/chat/completions"
                val model = config.model.ifBlank { "google/gemini-2.0-flash-exp:free" }
                val body = buildOpenAiStyleBody(model, prompt, 2)
                Triple(url, body, mapOf(
                    "Content-Type" to "application/json",
                    "Authorization" to "Bearer ${config.apiKey.trim()}",
                    "HTTP-Referer" to "https://zipper.app"
                ))
            }
            config.provider.contains("OpenAI", true) -> {
                val url = "https://api.openai.com/v1/chat/completions"
                val model = config.model.ifBlank { "gpt-4o-mini" }
                val body = buildOpenAiStyleBody(model, prompt, 2)
                Triple(url, body, mapOf(
                    "Content-Type" to "application/json",
                    "Authorization" to "Bearer ${config.apiKey.trim()}"
                ))
            }
            else -> {
                // Custom Provider
                val base = config.customUrl.trim().removeSuffix("/")
                val url = if (base.endsWith("/chat/completions")) base else "$base/chat/completions"
                val model = config.model.ifBlank { "custom-model" }
                val body = buildOpenAiStyleBody(model, prompt, 2)
                val headers = mutableMapOf("Content-Type" to "application/json")
                if (config.apiKey.isNotBlank()) {
                    headers["Authorization"] = "Bearer ${config.apiKey.trim()}"
                }
                Triple(url, body, headers)
            }
        }
    }

    private fun buildPlanningRequest(config: WebAiConfig, prompt: String): Triple<String, String, Map<String, String>> {
        return when {
            config.provider.contains("Gemini", true) -> {
                val model = config.model.ifBlank { "gemini-3-flash-preview" }
                val url = "https://generativelanguage.googleapis.com/v1beta/models/$model:generateContent?key=${config.apiKey.trim()}"
                val body = JSONObject().apply {
                    put("contents", JSONArray().put(JSONObject().put("parts", JSONArray().put(JSONObject().put("text", prompt)))))
                    put("generationConfig", JSONObject().apply {
                        put("temperature", 0.1)
                        put("maxOutputTokens", 1500)
                    })
                }.toString()
                Triple(url, body, mapOf("Content-Type" to "application/json"))
            }
            config.provider.contains("Groq", true) -> {
                val url = "https://api.groq.com/openai/v1/chat/completions"
                val model = if (config.model.isBlank() || config.model.contains("llama", true)) "qwen/qwen3.8-27b" else config.model
                val body = buildOpenAiStyleBody(model, prompt, 1500)
                Triple(url, body, mapOf(
                    "Content-Type" to "application/json",
                    "Authorization" to "Bearer ${config.apiKey.trim()}",
                    "User-Agent" to "Mozilla/5.0 (Linux; Android 14; Mobile)"
                ))
            }
            config.provider.contains("OpenRouter", true) -> {
                val url = "https://openrouter.ai/api/v1/chat/completions"
                val model = config.model.ifBlank { "google/gemini-2.0-flash-exp:free" }
                val body = buildOpenAiStyleBody(model, prompt, 1200)
                Triple(url, body, mapOf(
                    "Content-Type" to "application/json",
                    "Authorization" to "Bearer ${config.apiKey.trim()}",
                    "HTTP-Referer" to "https://zipper.app"
                ))
            }
            config.provider.contains("OpenAI", true) -> {
                val url = "https://api.openai.com/v1/chat/completions"
                val model = config.model.ifBlank { "gpt-4o-mini" }
                val body = buildOpenAiStyleBody(model, prompt, 1200)
                Triple(url, body, mapOf(
                    "Content-Type" to "application/json",
                    "Authorization" to "Bearer ${config.apiKey.trim()}"
                ))
            }
            else -> {
                val base = config.customUrl.trim().removeSuffix("/")
                val url = if (base.endsWith("/chat/completions")) base else "$base/chat/completions"
                val model = config.model.ifBlank { "custom-model" }
                val body = buildOpenAiStyleBody(model, prompt, 1200)
                val headers = mutableMapOf("Content-Type" to "application/json")
                if (config.apiKey.isNotBlank()) {
                    headers["Authorization"] = "Bearer ${config.apiKey.trim()}"
                }
                Triple(url, body, headers)
            }
        }
    }

    private fun buildOpenAiStyleBody(model: String, prompt: String, maxTokens: Int): String {
        return JSONObject().apply {
            put("model", model)
            put("messages", JSONArray().put(JSONObject().apply {
                put("role", "user")
                put("content", prompt)
            }))
            put("temperature", 0.1)
            put("max_tokens", maxTokens)
        }.toString()
    }

    private fun readResponseString(conn: HttpURLConnection, code: Int): String {
        val stream = if (code in 200..299) conn.inputStream else conn.errorStream ?: return ""
        return BufferedReader(InputStreamReader(stream, "UTF-8")).use { it.readText() }
    }

    private fun parseErrorMessage(rawError: String, code: Int): String {
        return try {
            val json = JSONObject(rawError)
            val msg = when {
                json.has("error") -> {
                    val errObj = json.get("error")
                    if (errObj is JSONObject) errObj.optString("message", rawError)
                    else errObj.toString()
                }
                json.has("message") -> json.getString("message")
                else -> rawError
            }
            "HTTP $code: $msg"
        } catch (_: Exception) {
            when (code) {
                401, 403 -> "HTTP $code: Invalid API Key or Unauthorized"
                404 -> "HTTP 404: Model or Endpoint not found"
                429 -> "HTTP 429: Rate Limit Exceeded or Quota Reached"
                else -> "HTTP $code: $rawError".take(120)
            }
        }
    }

    private fun extractLlmResponseText(provider: String, responseBody: String): String {
        return try {
            val json = JSONObject(responseBody)
            if (provider.contains("Gemini", true)) {
                val candidates = json.optJSONArray("candidates")
                val content = candidates?.optJSONObject(0)?.optJSONObject("content")
                val parts = content?.optJSONArray("parts")
                parts?.optJSONObject(0)?.optString("text") ?: ""
            } else {
                val choices = json.optJSONArray("choices")
                val message = choices?.optJSONObject(0)?.optJSONObject("message")
                val content = message?.optString("content").orEmpty()
                if (content.isNotBlank()) content
                else message?.optString("reasoning").orEmpty()
            }
        } catch (e: Exception) {
            Log.w(TAG, "Error extracting content from LLM response", e)
            ""
        }
    }

    private fun parsePlannedActions(rawContent: String): List<PlannedWebAction> {
        val cleanJson = rawContent.trim()
            .replace(Regex("^```json\\s*", RegexOption.IGNORE_CASE), "")
            .replace(Regex("^```\\s*"), "")
            .replace(Regex("\\s*```$"), "")
            .trim()

        val startIdx = cleanJson.indexOf('[')
        val endIdx = cleanJson.lastIndexOf(']')

        // Resilient JSON Slice with auto-healing if truncated
        val jsonSlice = when {
            startIdx != -1 && endIdx != -1 && startIdx < endIdx -> {
                cleanJson.substring(startIdx, endIdx + 1)
            }
            startIdx != -1 -> {
                var partial = cleanJson.substring(startIdx).trim()
                when {
                    partial.endsWith("\"") -> partial + "}]"
                    partial.endsWith("}") -> partial + "]"
                    Regex(""".*:\s*"?\w+$""").matches(partial) -> partial + "\"}]"
                    else -> partial + "}]"
                }
            }
            else -> cleanJson
        }

        val actions = mutableListOf<PlannedWebAction>()

        try {
            val array = JSONArray(jsonSlice)
            for (i in 0 until array.length()) {
                val obj = array.optJSONObject(i) ?: continue
                parseSingleActionObject(obj)?.let { actions.add(it) }
            }
        } catch (_: Exception) {
            // Fallback: Parse individual completed { ... } objects via regex if array parsing fails
            val objectRegex = Regex("""\{[^{}]*\}""")
            for (match in objectRegex.findAll(cleanJson)) {
                try {
                    val obj = JSONObject(match.value)
                    parseSingleActionObject(obj)?.let { actions.add(it) }
                } catch (_: Exception) {}
            }
        }

        return actions
    }

    /**
     * Executes a general text synthesis completion prompt (e.g. for notification briefings or summarization).
     */
    suspend fun generateCompletion(prompt: String, config: WebAiConfig): String? = withContext(Dispatchers.IO) {
        val chain = config.getFallbackChain()
        for (tier in chain) {
            if (!tier.isConfigured && tier.apiKey.isBlank() && !tier.provider.contains("ChatGPT Web", true)) continue

            if (tier.provider.contains("ChatGPT Web", true)) {
                val bridgeResult = com.zipper.app.core.web.chatgpt.ChatGptHeadlessBridge.ask(null, prompt)
                val ans = bridgeResult.getOrNull()
                if (!ans.isNullOrBlank()) return@withContext ans
                continue
            }

            try {
                val (endpoint, requestBody, headers) = buildPlanningRequest(tier, prompt)
                val connection = (URL(endpoint).openConnection() as java.net.HttpURLConnection).apply {
                    requestMethod = "POST"
                    connectTimeout = TIMEOUT_MS
                    readTimeout = TIMEOUT_MS
                    doOutput = true
                    headers.forEach { (k, v) -> setRequestProperty(k, v) }
                }

                java.io.OutputStreamWriter(connection.outputStream, "UTF-8").use { os ->
                    os.write(requestBody)
                    os.flush()
                }

                val statusCode = connection.responseCode
                if (statusCode in 200..299) {
                    val responseBody = readResponseString(connection, statusCode)
                    val text = extractLlmResponseText(tier.provider, responseBody).trim()
                    if (text.isNotBlank()) {
                        Log.i(TAG, "generateCompletion succeeded via ${tier.provider} (${tier.model})")
                        return@withContext text
                    }
                } else {
                    Log.w(TAG, "generateCompletion HTTP $statusCode on ${tier.provider}, falling over to next tier...")
                }
            } catch (e: Exception) {
                Log.w(TAG, "Failed generateCompletion via ${tier.provider}: ${e.message}, falling over to next tier...")
            }
        }
        null
    }

    private fun parseSingleActionObject(obj: JSONObject): PlannedWebAction? {
        val rawAction = obj.optString("action", "").uppercase()
        if (rawAction.isBlank()) return null

        val actionType = when (rawAction) {
            "INPUT", "TYPE", "ENTER", "WRITE", "FILL" -> "TYPE"
            "SELECT", "CHOOSE", "PICK" -> "SELECT"
            "CLICK", "TAP", "PRESS" -> "CLICK"
            "SUBMIT" -> "SUBMIT"
            else -> rawAction
        }

        val targetIndex = when {
            obj.has("targetIndex") -> obj.optInt("targetIndex", -1)
            obj.has("index") -> obj.optInt("index", -1)
            obj.has("target_index") -> obj.optInt("target_index", -1)
            obj.has("element") -> obj.optInt("element", -1)
            obj.has("node") -> obj.optInt("node", -1)
            obj.has("target") -> obj.optInt("target", -1)
            obj.has("id") -> {
                val rawId = obj.optString("id")
                rawId.removePrefix("k_node_").toIntOrNull() ?: obj.optInt("id", -1)
            }
            else -> -1
        }
        val text = obj.optString("text").takeIf { it.isNotBlank() }
            ?: obj.optString("value").takeIf { it.isNotBlank() }
            ?: obj.optString("argument").takeIf { it.isNotBlank() }
        val value = obj.optString("value").takeIf { it.isNotBlank() }
            ?: obj.optString("text").takeIf { it.isNotBlank() }
            ?: obj.optString("option").takeIf { it.isNotBlank() }
        val desc = obj.optString("description").takeIf { it.isNotBlank() }

        return PlannedWebAction(
            action = actionType,
            targetIndex = targetIndex,
            text = text,
            value = value,
            description = desc
        )
    }

    /**
     * Generates a context-aware chat reply on behalf of the user to an incoming message.
     * Uses the configured LLM API (Gemini, Groq, OpenAI, OpenRouter, Custom).
     * Returns the drafted reply text, or null if no reply needed (e.g. closing message).
     */
    suspend fun generateChatReply(
        contactName: String,
        incomingMessage: String,
        recentHistory: List<String>,
        userContext: String,
        personaTone: String,
        config: WebAiConfig
    ): String? = withContext(Dispatchers.IO) {
        if (!config.isConfigured) return@withContext null

        val historyBlock = if (recentHistory.isNotEmpty()) {
            "RECENT CONVERSATION THREAD SO FAR (Anchor + Recent Turns):\n" + recentHistory.joinToString("\n") { "  $it" }
        } else {
            ""
        }

        val isPro = personaTone.contains("Professional", true) || personaTone.contains("Formal", true)
        val isKanglishContext = incomingMessage.contains("bro", true) ||
            incomingMessage.contains("macha", true) ||
            incomingMessage.contains("guru", true) ||
            incomingMessage.contains("maga", true) ||
            incomingMessage.contains("oota", true) ||
            incomingMessage.contains("sigana", true) ||
            incomingMessage.contains("elli", true) ||
            incomingMessage.contains("yavaga", true) ||
            incomingMessage.contains("samachara", true) ||
            incomingMessage.contains("aitha", true) ||
            incomingMessage.contains("aaythu", true) ||
            incomingMessage.contains("baa", true) ||
            incomingMessage.contains("bega", true) ||
            incomingMessage.contains("clg", true) ||
            incomingMessage.contains("tmmorw", true) ||
            incomingMessage.contains("tmrw", true) ||
            personaTone.contains("Kanglish", true) ||
            personaTone.contains("Kannada", true)

        val prompt = """
            You are Karthik texting on WhatsApp. You are actively conversing with $contactName.

            YOUR CURRENT CONTEXT, STATUS & INSTRUCTIONS:
            ${userContext.ifBlank { "You are currently busy with software development work, but you can converse and answer questions naturally as Karthik." }}

            YOUR PERSONA & TONE FOR THIS SPECIFIC CONTACT ($contactName):
            ${personaTone.ifBlank { "Polite, friendly, and authentic texting style. Speak in the first person ('I', 'me', 'my')." }}

            $historyBlock

            NEW INCOMING MESSAGE FROM $contactName:
            "$incomingMessage"

            CRITICAL CHATTING RULES:
            1. Speak as Karthik in the FIRST PERSON ("I", "my", "we"). NEVER say "Karthik is..." or "On behalf of Karthik".
            ${if (isPro) {
                """2. STRICT PROFESSIONAL & POLITE TONE:
               - Speak in clean, polite, formal, and respectful English.
               - NEVER use "bro", "dude", colloquialisms, or casual slang!
               - Maintain a courteous, composed tone suitable for a professional colleague, client, professor, or elder."""
            } else {
                """2. AUTHENTIC CASUAL SLANG & KANGLISH ADAPTATION (MANDATORY):
               - You are Karthik, a college student / software developer in Bangalore texting a friend on WhatsApp.
               - NATURALLY ADOPT CASUAL TEXTING SLANG: Freely, warmly, and authentically use casual slang such as "bro", "done bro", "ha bro", "yeah bro", "scenes", "chill", "catch you in a bit", etc.
               - KANGLISH & BANGALORE SLANG ADAPTATION:
                 Deeply understand and actively adopt Romanized Kanglish / Bangalore lingo:
                 (*bro, macha, guru, maga, en samachara, sigana, yavaga, yav time, aitha, aaythu, oota, baa, bega, clg, tension illa, scenes*).
                 ${if (isKanglishContext) {
                     "Since $contactName texted using informal/Kanglish/Bangalore lingo or slang, ACTIVELY ADOPT AND MIRROR their Romanized Kanglish vibe! Reply naturally in Romanized Kanglish (e.g., 'Ha done bro, nale sigana 👍', 'Oota aaythu bro, needhu aaytha?', 'Ha bro clg ge barthini, nee barthiya?', 'Outside idini bro, en samachara?')."
                 } else {
                     "Even when $contactName texts in standard English (e.g. 'are you free tomorrow?', 'tmw clg?', 'where are you?', 'can we meet?'), ALWAYS text back like a real close friend using natural casual slang ('Yeah bro, tomorrow works for me!', 'Ha bro coming to clg! You coming?', 'Outside for some work bro, what's up?'). If they ask in Kanglish, reply in Kanglish. NEVER sound stiff, formal, or like an AI bot!"
                 }}"""
            }}
            3. CORE FACTS ABOUT KARTHIK:
               - Karthik is an engineering student living and studying in Bangalore. His family and hometown is Mysore.
               - If casually asked where he is ("Where are you?", "elli iddiya?", "Bangalore or Mysore?"):
                 * For Elders / Family: Answer naturally and respectfully in Kannada (e.g. "Bangalore nalli idini Aunty, clg kelasadalli idini. En samachara?" or "Bangalore nalli idini Amma.").
                 * For Friends: Answer in natural casual slang (e.g. "Bangalore nalli idini bro, clg scenes. What's up?").
                 * NEVER say "outside for some work" repeatedly like a bot! Answer truthfully that he is in Bangalore.

            4. INTENT DISCRIMINATION & LIVE CONFIRMATION CALLS:
               - CATEGORY A: CASUAL STATUS & ACTIVITY INQUIRIES (DO NOT CALL! CHAT DIRECTLY AS KARTHIK):
                 In ANY language or Romanized regional slang (Kanglish, Hinglish, Bangalore slang, casual English):
                 Questions about current activity, status, or casual greetings ("what are you doing?", "what's up?", "wyd?", "en madtidya?", "enmadtiydya?", "enu madtidiya?", "kya kar raha hai?", "en scenes?", "how are you?", "oota aaytha?"):
                 * THIS IS A CASUAL ACTIVITY INQUIRY, NOT A MEETING INVITATION OR TIME REQUEST!
                 * NEVER TRIGGER A LIVE CALL (CALL_USER) FOR ACTIVITY INQUIRIES!
                 * Answer directly in chat as Karthik describing your current activity / status (e.g. "Kelasa madtha idini bro, en samachara?" / "Working on some code right now, what's up?").
                 * If a user context is configured, use it naturally (e.g. "College nalli idini bro", "In gym right now bro").

               - CATEGORY B: ACTUAL INVITATIONS & PROPOSED MEETUPS (TRIGGER A LIVE CALL):
                 When the contact EXPLICITLY proposes a physical meetup, outing, hanging out, or fixing a calendar schedule ("Will you come tomorrow?", "When can we meet?", "Can we go out?", "Nale sigana?"):
                 Karthik has not agreed to this yet!
                 TRIGGER A LIVE CALL TO KARTHIK'S PHONE:
                 CALL_USER: $contactName is asking: "$incomingMessage" | [Polite holding phrase in their language, e.g. "Swalpa check maadi helthini" or "1 sec bro checking schedule"]

               - CATEGORY C: NEGOTIATION LOOP & PREVIOUSLY PROPOSED TIME:
                 Check the recent conversation history above! If Karthik already proposed a time (e.g. 4 PM, Saturday, or tomorrow):
                 * IF $contactName AGREES (e.g. "sari 4 PM ge", "4 PM ok", "cool", "done", "works", "sure"):
                   DO NOT CALL! The contact accepted Karthik's proposed time!
                   Confirm warmly and lock it in:
                   - For Family / Elder: "Sari, aavaga sigana 👍" (address respectfully as Amma/Appa/Aunty/Uncle based on relationship).
                   - For Friend: "Done bro, see you then!"
                 * IF $contactName REJECTS OR PROPOSES A CLASH (e.g. "No 4 PM won't work, come at 2 PM", "No change it", "very urgent", "emergency"):
                   DO NOT stubbornly repeat the old time!
                   TRIGGER A LIVE CALL TO KARTHIK WITH THEIR EXACT OBJECTION:
                   CALL_USER: $contactName says: "$incomingMessage" | [Polite holding phrase in their language]

               - CATEGORY D: ROUTINE CHAT & CLOSURES (DO NOT CALL):
                 Do NOT call for normal messages ("how are you?", "send notes", "oota aaytha?", "ok", "cool"). Only call for actual unconfirmed meeting proposals or time clashes!
            6. FINANCIAL & SENSITIVE LOCKOUT:
               - NEVER agree to send money, GPay, PhonePe, UPI, OTP, passwords, or confidential files.
               - Reply: "Can't access GPay/banking right now bro, caught up with work. Ping you later!"
            7. SLANG & EXACT LANGUAGE LEVEL MIRRORING:
               ${if (isPro) {
                   "- Keep the reply composed, courteous, and strictly free of slang."
               } else {
                   """- Match the friend's exact language level and energy!
               - If the contact texts in broken, simple, or non-native English (e.g. 'wr u', 'u comming', 'call me fast', 'wat r u doin'), YOU MUST MATCH their simple vocabulary and short structure. DO NOT use complex grammar, fancy words, or write essays!
               - If the contact texts in ultra-short 2-4 words, reply in 2-4 words.
               - If the contact texts in Bangalore slang or Romanized Kanglish, warmly mirror their Kanglish vibe."""
               }}
            8. YOU HAVE COMPLETE MEMORY of the conversation thread above (including what Karthik typed or confirmed). Answer naturally with full context!
            9. DO NOT MINDLESSLY REPEAT your status phrase in every message. Treat the conversation like a real ongoing chat.
            10. Keep the reply authentic and concise: 1 to 2 short sentences (natural WhatsApp texting format).
            11. NEVER sound like an AI assistant or bot. Never say "As an AI...", "How can I assist you?", or "Best regards".
            12. If the incoming message is just a pure conversation closing or acknowledgment (e.g. "ok", "k", "cya", "bye", "👍", "thanks"), reply with EXACTLY: NO_REPLY
            13. Output ONLY the message text, no quotation marks, no preamble.
        """.trimIndent()

        if (config.provider.contains("ChatGPT Web", true)) {
            val bridgeResult = com.zipper.app.core.web.chatgpt.ChatGptHeadlessBridge.ask(null, prompt)
            val raw = bridgeResult.getOrNull()?.trim()
            if (!raw.isNullOrBlank()) {
                val clean = if (raw.startsWith("\"") && raw.endsWith("\"") && raw.length > 1) {
                    raw.substring(1, raw.length - 1).trim()
                } else raw
                if (clean.equals("NO_REPLY", ignoreCase = true) || clean.equals("NONE", ignoreCase = true)) return@withContext null
                return@withContext clean
            }
            return@withContext null
        }

        val chain = config.getFallbackChain()
        for (tier in chain) {
            if (!tier.isConfigured && tier.apiKey.isBlank() && !tier.provider.contains("ChatGPT Web", true)) continue

            try {
                val (endpoint, requestBody, headers) = buildPlanningRequest(tier, prompt)
                val connection = (URL(endpoint).openConnection() as HttpURLConnection).apply {
                    requestMethod = "POST"
                    connectTimeout = 5000
                    readTimeout = 5000
                    doOutput = true
                    headers.forEach { (k, v) -> setRequestProperty(k, v) }
                }

                OutputStreamWriter(connection.outputStream, "UTF-8").use { os ->
                    os.write(requestBody)
                    os.flush()
                }

                val statusCode = connection.responseCode
                val responseBody = readResponseString(connection, statusCode)

                if (statusCode in 200..299) {
                    val raw = extractLlmResponseText(tier.provider, responseBody).trim()
                        .replace("`", "")
                        .removeSurrounding("\"")
                        .trim()
                    if (raw.isNotBlank() && !raw.equals("NO_REPLY", ignoreCase = true) && !raw.equals("NONE", ignoreCase = true)) {
                        Log.i(TAG, "Chat reply successfully generated via ${tier.provider} (${tier.model}): '$raw'")
                        return@withContext raw
                    }
                } else {
                    Log.w(TAG, "Chat reply generation failed on ${tier.provider} (${tier.model}) with status $statusCode, falling over...")
                }
            } catch (e: Exception) {
                Log.w(TAG, "Error generating chat reply via ${tier.provider} (${tier.model}): ${e.message}, falling over...")
            }
        }
        null
    }

    /**
     * Takes Karthik's live in-call instruction/response (e.g. "tmw is okk ask her what time" or "free at 5pm")
     * and synthesizes a natural, polite, first-person WhatsApp message to the contact matching their persona/slang.
     */
    suspend fun synthesizeUserDecision(
        contactName: String,
        incomingMessage: String,
        userInstruction: String,
        personaTone: String,
        recentHistory: List<String>,
        config: WebAiConfig
    ): String = withContext(Dispatchers.IO) {
        if (userInstruction.isBlank()) return@withContext ""

        val historyBlock = if (recentHistory.isNotEmpty()) {
            "RECENT CONVERSATION THREAD SO FAR:\n" + recentHistory.joinToString("\n") { "  $it" }
        } else {
            ""
        }

        val isPro = personaTone.contains("Professional", true) || personaTone.contains("Formal", true)
        val dossier = com.zipper.app.core.intelligence.reply.ContactKnowledgeBook.getInstanceOrNull()?.getDossier(contactName)
        val isElder = dossier?.isElderOrFamily() == true
        val isKan = dossier?.isKannadaPreferred() == true || dossier?.languageStyle == com.zipper.app.core.intelligence.reply.LanguageStyle.KANNADA_REGIONAL
        val honorific = dossier?.resolveHonorific() ?: ""

        val relationship = dossier?.relationship?.ifBlank { "Friend / Contact" } ?: "Friend / Contact"
        val languagePreference = when {
            isKan -> "Kannada / Regional"
            dossier?.languageStyle == com.zipper.app.core.intelligence.reply.LanguageStyle.FORMAL_ENGLISH || isPro -> "Formal English"
            dossier?.languageStyle == com.zipper.app.core.intelligence.reply.LanguageStyle.BROKEN_BASIC_ENGLISH -> "Casual Broken/Basic English"
            else -> "Casual English / Bengaluru Slang"
        }

        val styleRules = """
            - RELATIONSHIP TO KARTHIK: $relationship.
            - PREFERRED LANGUAGE & STYLE: $languagePreference.
            ${if (honorific.isNotBlank()) "- Address $contactName respectfully as '$honorific'. DO NOT call them any other title (e.g. do NOT use 'Amma' if they are an Aunt, and do NOT use 'bro')." else "- $contactName is a friend/peer. Address them by their first name or friendly slang ('bro'). DO NOT use 'Amma', 'Appa', 'Uncle', or 'Sir'."}
            - If $contactName texted in Kannada / Kanglish, reply in natural, authentic Kannada / Kanglish.
            - If $contactName texted in Hindi, reply in natural Hindi.
            - If $contactName texted in English, reply in clean, natural English.
        """.trimIndent()

        val prompt = """
            You are Karthik texting on WhatsApp. You are replying to $contactName.
            $contactName's latest message to you was:
            "$incomingMessage"

            Karthik was just consulted via a live phone call and provided this raw executive direction:
            "$userInstruction"

            YOUR PERSONA & TONE FOR $contactName:
            ${personaTone.ifBlank { "Polite, authentic texting style. Speak in the first person ('I', 'me', 'my')." }}

            $historyBlock

            CRITICAL RULES:
            1. Speak as Karthik in the FIRST PERSON ("I", "my", "we"). NEVER say "Karthik says" or speak about him in the 3rd person.
            2. DO NOT output Karthik's raw words verbatim (e.g. NEVER just send "$userInstruction" directly).
            3. Properly TUNE, POLISH, and PERSUASIVELY ARTICULATE his intent into an authentic, respectful, conversational message:
               - PROPOSING A TIME / AVAILABILITY (e.g. "$userInstruction"):
                 Actively persuade and propose that availability warmly and politely:
                 * For Family / Elder in Kannada: Propose the time respectfully:
                   e.g., "${if (honorific.isNotBlank()) "$honorific, " else ""}[time] ge aagutha? Swalpa kelasa mugisi [time] ge barthini."
                 * For Friend / Peer: Persuade casually:
                   e.g., "Bro [time] works best for me! Little stuck till then, can we do [time]? Let me know!"
               - CONFIRMATION (e.g. "Sari", "Yes", "Confirm", "works"):
                 Confirm warmly in their language:
                 * Elder / Kannada: "Sari${if (honorific.isNotBlank()) " $honorific" else ""}, aaythu. Aavaga sigana."
                 * Friend: "Done bro, see you then!"
               - COLLEGE / WORK / BUSY:
                 If Karthik mentioned college, work, or being occupied:
                 * Elder / Kannada: "Kelasa ide${if (honorific.isNotBlank()) " $honorific" else ""}, so baroke aagalla. Weekend barthini."
                 * Friend: "Work/clg ide bro, cannot make it right now!"
            4. Convert Karthik's intent into natural phrasing matching their relationship:
            $styleRules
            5. Keep it concise: 1 natural sentence. Output ONLY the message text, no quotation marks, no preamble.
        """.trimIndent()

        val chain = config.getFallbackChain()
        for (tier in chain) {
            if (!tier.isConfigured && tier.apiKey.isBlank() && !tier.provider.contains("ChatGPT Web", true)) continue

            // If tier is ChatGPT Web, try headless bridge only if running
            if (tier.provider.contains("ChatGPT Web", true)) {
                val bridgeResult = com.zipper.app.core.web.chatgpt.ChatGptHeadlessBridge.ask(null, prompt)
                val raw = bridgeResult.getOrNull()?.trim()
                if (!raw.isNullOrBlank()) {
                    val clean = if (raw.startsWith("\"") && raw.endsWith("\"") && raw.length > 1) {
                        raw.substring(1, raw.length - 1).trim()
                    } else raw
                    return@withContext clean
                }
                continue
            }

            try {
                val (endpoint, requestBody, headers) = buildPlanningRequest(tier, prompt)
                val connection = (URL(endpoint).openConnection() as HttpURLConnection).apply {
                    requestMethod = "POST"
                    connectTimeout = 6000
                    readTimeout = 7000
                    doOutput = true
                    headers.forEach { (k, v) -> setRequestProperty(k, v) }
                }

                OutputStreamWriter(connection.outputStream, "UTF-8").use { os ->
                    os.write(requestBody)
                    os.flush()
                }

                val statusCode = connection.responseCode
                val responseBody = readResponseString(connection, statusCode)

                if (statusCode in 200..299) {
                    val raw = extractLlmResponseText(tier.provider, responseBody).trim()
                        .replace("`", "")
                        .removeSurrounding("\"")
                        .trim()
                    if (raw.isNotBlank() && !raw.equals("NO_REPLY", ignoreCase = true)) {
                        Log.i(TAG, "Synthesized user decision via ${tier.provider} (${tier.model}): '$userInstruction' -> '$raw'")
                        return@withContext raw
                    }
                } else {
                    Log.w(TAG, "Synthesize user decision HTTP $statusCode on ${tier.provider}, trying fallback...")
                }
            } catch (e: Exception) {
                Log.w(TAG, "Error synthesizing user decision via ${tier.provider} (${tier.model}): ${e.message}, trying fallback...")
            }
        }

        // Fallback to offline universal rule engine
        fallbackSynthesizeInstruction(contactName, incomingMessage, userInstruction, personaTone)
    }

    internal fun fallbackSynthesizeInstruction(
        contactName: String,
        incomingMessage: String,
        userInstruction: String,
        personaTone: String
    ): String {
        val lower = userInstruction.lowercase(java.util.Locale.ROOT).trim()
        val dossier = com.zipper.app.core.intelligence.reply.ContactKnowledgeBook.getInstanceOrNull()?.getDossier(contactName)
        val isElder = dossier?.isElderOrFamily() == true
        val isKan = dossier?.isKannadaPreferred() == true ||
            dossier?.languageStyle == com.zipper.app.core.intelligence.reply.LanguageStyle.KANNADA_REGIONAL ||
            incomingMessage.contains("yavaga", true) ||
            incomingMessage.contains("barthya", true) ||
            incomingMessage.contains("baralva", true) ||
            incomingMessage.contains("oorige", true) ||
            incomingMessage.contains("mysore", true) ||
            incomingMessage.contains("sigana", true)

        val isPro = personaTone.contains("Professional", true) || personaTone.contains("Formal", true)
        val isKanglish = isKan || incomingMessage.contains("bro", true) || incomingMessage.contains("macha", true) || incomingMessage.contains("guru", true)
        val honorific = dossier?.resolveHonorific() ?: ""
        val tag = if (honorific.isNotBlank()) " $honorific" else ""

        // Universal time extractor: matches "3pm", "3 pm", "4", "4pm", "5:30", "11am", etc.
        val timeRegex = Regex("""\b(?:at\s+)?(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b""", RegexOption.IGNORE_CASE)
        val timeMatch = timeRegex.find(lower)?.groupValues?.get(1)?.trim()
        val formattedTime = timeMatch?.let { t ->
            val numOnly = t.toIntOrNull()
            if (numOnly != null && numOnly in 1..12) {
                val isMorning = lower.contains("morning") || lower.contains("belagge")
                if (isMorning) "$numOnly AM" else "$numOnly PM"
            } else {
                t.replace(Regex("""(?i)(\d+)\s*(am|pm)"""), "$1 $2").uppercase(java.util.Locale.ROOT)
            }
        }

        val hasCollege = lower.contains("clg") || lower.contains("college")
        val hasConfirm = lower.contains("works") || lower.contains("see you") || lower.contains("confirm") ||
            lower == "yes" || lower == "ok" || lower == "sari" || lower.contains("fine") || lower == "done" || lower == "sure"

        return when {
            isElder && isKan -> {
                when {
                    formattedTime != null -> "Nale $formattedTime ge aagutha$tag? Swalpa kelasa mugisi $formattedTime ge barthini."
                    hasCollege -> "Nale clg kelasa ide$tag, so baroke agalla. Ee weekend barthini."
                    hasConfirm -> "Sari$tag, aaythu. Aavaga sigana."
                    lower.contains("saturday") || lower.contains("shanivara") -> "Saturday barthini$tag."
                    lower.contains("sunday") || lower.contains("bhanuvara") -> "Sunday barthini$tag."
                    lower.contains("tomorrow") || lower.contains("nale") -> "Nale barthini$tag."
                    lower.contains("at home") || lower.contains("maneli") -> "Maneli idini$tag."
                    lower.contains("busy") || lower.contains("later") || lower.contains("tied up") -> "Swalpa kelasa ide$tag, aamele call madthini."
                    userInstruction.startsWith("Sari", true) || userInstruction.startsWith("Swalpa", true) -> userInstruction
                    userInstruction.length <= 5 -> "Sari$tag, $userInstruction ge arrange madona."
                    else -> "Sari$tag, $userInstruction"
                }
            }
            isElder -> {
                when {
                    formattedTime != null -> "Would $formattedTime work instead$tag? I can wrap up my work and meet at $formattedTime."
                    hasCollege -> "I have college commitments tomorrow$tag, will try to visit this weekend."
                    hasConfirm -> "Yes$tag, that works. Confirmed."
                    lower.contains("busy") || lower.contains("later") -> "Currently occupied$tag, will follow up with you as soon as I am free."
                    userInstruction.length <= 5 -> "Understood$tag, I will plan for $userInstruction."
                    else -> "Understood$tag, $userInstruction."
                }
            }
            formattedTime != null -> {
                if (isPro) "Would $formattedTime work for you instead?"
                else if (isKan) "Ha bro, $formattedTime works best for me! $formattedTime ge sigana?"
                else "$formattedTime works best for me bro! Can we do $formattedTime instead?"
            }
            hasCollege -> {
                if (isPro) "I have college commitments tomorrow, let us plan for later."
                else if (isKan) "Ha bro clg ide nale, cannot make it. Later sigana!"
                else "Have college tomorrow bro! Can't make it then."
            }
            hasConfirm -> {
                if (isPro) "Confirmed, looking forward to meeting."
                else if (isKan) "Done bro, aavaga sigana 👍"
                else "Done bro, see you then!"
            }
            lower.contains("at home") || lower.contains("maneli") -> {
                if (isPro) "I am currently at home. Please let me know what happened."
                else if (isKan) "Ha bro, I'm at home only! What happened, all good? Baa meet maadona."
                else "I'm at home right now bro! What happened, all good?"
            }
            lower.contains("traffic") || lower.contains("silk board") -> {
                if (isPro) "I am currently delayed in traffic. I will reach out as soon as I arrive."
                else "Stuck in traffic right now bro! What happened, everything ok?"
            }
            lower.contains("can't meet") || lower.contains("cant meet") || lower.contains("call in") -> {
                if (isPro) "I am unable to meet at the moment. Could you please call me in 10 minutes?"
                else "Can't meet right now bro, caught up with something. Can you call me in 10 mins?"
            }
            lower.contains("ask") && (lower.contains("time") || lower.contains("yav time")) -> {
                if (isPro) {
                    "Tomorrow works for me. What time would be convenient for you?"
                } else if (isKanglish) {
                    "Ha done bro, tomorrow is fine! What time works for you?"
                } else {
                    "Tomorrow works for me bro! What time were you thinking?"
                }
            }
            lower.contains("5pm") || lower.contains("5 pm") -> {
                if (isPro) {
                    "I am available at 5 PM. Would that time suit you?"
                } else if (isKanglish) {
                    "A bit tied up earlier bro, but I'm completely free around 5 PM! Will that work?"
                } else {
                    "I am completely free around 5 PM bro! Does that work for you?"
                }
            }
            lower.contains("4pm") || lower.contains("4 pm") -> {
                if (isPro) {
                    "I am available at 4 PM. Would that time suit you?"
                } else {
                    "Can't make that time bro, can we do 4 PM instead?"
                }
            }
            lower == "yes" || lower == "confirm" || lower == "ok" -> {
                if (isPro) "Confirmed, that time works. See you then." else "Yeah, that time works bro! See you then."
            }
            lower == "busy" || lower == "later" -> {
                if (isPro) "I am currently occupied, will get back to you in 30 minutes." else "A bit tied up right now bro, will check and ping you in 30 mins!"
            }
            else -> {
                val dossier = com.zipper.app.core.intelligence.reply.ContactKnowledgeBook.getInstanceOrNull()?.getDossier(contactName)
                val directiveTranslated = com.zipper.app.core.intelligence.reply.SpeechActTranslator.translateDirective(
                    userInstruction,
                    dossier,
                    incomingMessage
                )
                if (directiveTranslated != userInstruction && directiveTranslated.isNotBlank()) {
                    directiveTranslated
                } else {
                    val clean = userInstruction
                        .replaceFirst(Regex("""(?i)^tell (her|him|them)\s*"""), "")
                        .replaceFirst(Regex("""(?i)^ask (her|him|them)\s*"""), "")
                        .trim()
                    clean
                }
            }
        }
    }

    /**
     * Synthesizes a natural, concise, first-person executive briefing for the incoming phone call screen
     * instead of dumping raw, all-caps message text verbatim into quotes.
     */
    suspend fun synthesizeSpokenBriefing(
        contactName: String,
        incomingMessage: String,
        config: WebAiConfig
    ): String = withContext(Dispatchers.IO) {
        val cleanName = contactName.trim().replaceFirstChar { if (it.isLowerCase()) it.titlecase(java.util.Locale.ROOT) else it.toString() }
        if (incomingMessage.isBlank()) return@withContext "$cleanName is calling. What should I reply?"

        val prompt = """
            You are Karthik's AI Copilot on his Android phone.
            A contact named $cleanName sent this incoming message to Karthik:
            "$incomingMessage"

            Karthik's phone is now ringing to ask him what to reply.
            Your task is to summarize what $cleanName is saying, asking, or objecting to in ONE concise, natural spoken briefing sentence.
            Format: "$cleanName says/asks [brief summary in plain English]. What should I reply?"

            CRITICAL ACCURACY & SLANG RULES:
            1. TRUTHFULNESS MANDATE: Summarize ONLY what was explicitly written in the message. NEVER invent or assume $cleanName is asking when Karthik is free or asking to meet unless the message literally asks to meet or asks about free time!
            2. UNDERSTAND REGIONAL SLANG: Contacts frequently text in Romanized regional slang (Kanglish, Kannada, Hindi, etc.). Accurately understand the literal meaning:
               - "en madtidya" / "enmadtiydya" / "enu madtidiya" / "kya kar raha hai" means "asking what you are doing".
               - "oota aaytha" means "asking if you had food".
               - "yavaga barthya" / "yavaga barodu" means "asking when you will come".
               - "sigana" / "meet agana" means "asking to meet up".
               - "clg barthya" means "asking if you are coming to college".
            3. Output EXACTLY ONE concise sentence (max 18 words).
            4. Never output quotation marks, all-caps yelling, or raw text verbatim.

            Examples:
            - If message is "Can we meet tomorrow at 5 PM?":
              "$cleanName wants to meet tomorrow at 5 PM. What should I reply?"
            - If message is "IVATHE BA":
              "Urgent from $cleanName: they need you to come today. What should I reply?"
            - If message is "No that time won't work":
              "$cleanName says that time won't work. What should I reply?"
            - If message is "WILL YOU COME TOMORROW":
              "$cleanName is asking if you will come tomorrow. What should I reply?"
            - If message is "en madtidya":
              "$cleanName is asking what you are doing right now. What should I reply?"
        """.trimIndent()

        val chain = config.getFallbackChain()
        for (tier in chain) {
            if (!tier.isConfigured && tier.apiKey.isBlank() && !tier.provider.contains("ChatGPT Web", true)) continue
            try {
                val (endpoint, requestBody, headers) = buildPlanningRequest(tier, prompt)
                val connection = (URL(endpoint).openConnection() as HttpURLConnection).apply {
                    requestMethod = "POST"
                    connectTimeout = 3000
                    readTimeout = 3000
                    doOutput = true
                    headers.forEach { (k, v) -> setRequestProperty(k, v) }
                }

                OutputStreamWriter(connection.outputStream, "UTF-8").use { os ->
                    os.write(requestBody)
                    os.flush()
                }

                val statusCode = connection.responseCode
                val responseBody = readResponseString(connection, statusCode)

                if (statusCode in 200..299) {
                    val raw = extractLlmResponseText(tier.provider, responseBody).trim()
                        .replace("`", "")
                        .removeSurrounding("\"")
                        .trim()
                    if (raw.isNotBlank() && !raw.equals("NO_REPLY", ignoreCase = true)) {
                        Log.i(TAG, "Synthesized in-call spoken briefing via ${tier.provider}: '$raw'")
                        return@withContext raw
                    }
                }
            } catch (e: Exception) {
                Log.w(TAG, "Error synthesizing spoken briefing via ${tier.provider}: ${e.message}")
            }
        }

        fallbackSpokenBriefing(cleanName, incomingMessage)
    }

    internal fun fallbackSpokenBriefing(contactName: String, incomingMessage: String): String {
        val cleanName = contactName.trim().replaceFirstChar { if (it.isLowerCase()) it.titlecase(java.util.Locale.ROOT) else it.toString() }
        val lower = incomingMessage.lowercase(java.util.Locale.ROOT).trim()

        val isWord = { word: String -> lower.split(Regex("""[\s,?.!]+""")).contains(word) }

        val hasRejection = lower.contains("agala") || lower.contains("aagalla") || lower.contains("beda") ||
            lower.contains("cannot") || lower.contains("can't") || lower.contains("cant") || lower.contains("won't") ||
            isWord("no") || isWord("nope") || isWord("not")

        val hasTimeInquiry = lower.contains("yesht gante") || lower.contains("yavaga") || lower.contains("what time") ||
            lower.contains("which time") || (isWord("when") && !lower.contains("what")) || isWord("free")

        val hasActivityInquiry = lower.contains("madtidya") || lower.contains("madtiya") || lower.contains("madtidiya") ||
            lower.contains("madtha idya") || lower.contains("doing") || lower.contains("wyd") ||
            lower.contains("what up") || lower.contains("whats up") || lower.contains("what's up") ||
            lower.contains("wassup") || lower.contains("kya kar") || lower.contains("scenes")

        val hasUrgent = lower.contains("urgent") || lower.contains("bega") || lower.contains("ivathe") || lower.contains("ivatte")

        return when {
            hasRejection && hasTimeInquiry -> "$cleanName says that won't work, and asks what time you're free. What should I reply?"
            hasRejection -> "$cleanName says that won't work. What should I reply?"
            hasUrgent -> "Urgent from $cleanName: they need you right away. What should I reply?"
            hasActivityInquiry -> "$cleanName is asking what you're doing right now. What should I reply?"
            hasTimeInquiry -> "$cleanName is asking what time you will be free. What should I reply?"
            lower.contains("clg") || lower.contains("college") -> "$cleanName is asking about college tomorrow. What should I reply?"
            lower.contains("oota") -> "$cleanName is asking if you had food. What should I reply?"
            lower.contains("meet") || lower.contains("sigana") || lower.contains("barodu") || lower.contains("barthya") -> "$cleanName is asking to meet up. What should I reply?"
            else -> "$cleanName asks: \"${incomingMessage.take(45)}\". What should I reply?"
        }
    }
}

