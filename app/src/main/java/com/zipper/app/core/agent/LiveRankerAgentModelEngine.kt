package com.zipper.app.core.agent

import android.util.Log
import com.zipper.app.core.needle.AgentPrediction
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

/**
 * Karvish Live Ranker Model Engine Adapter (Milestone v0.8).
 *
 * Implements dual-mode Top-K proposal generation:
 * 1. Primary: Real-time HTTP bridge to host-GPU MobileActionRanker model over reverse ADB port 8765.
 * 2. Standalone Fallback: On-device semantic tool scoring when bridge server is disconnected.
 *
 * In accordance with AGENTS.md:
 * - Independently evaluates live UI tool schemas.
 * - Returns structured Top-1, Top-2, and Top-3 candidate proposals.
 * - Zero hardcoded app workflows or coordinates.
 */
class LiveRankerAgentModelEngine(
    private val bridgeUrl: String = "http://127.0.0.1:8765/propose",
    private val connectTimeoutMs: Int = 1200,
    private val readTimeoutMs: Int = 10000
) : AgentModelEngine {

    companion object {
        private const val TAG = "LiveRankerModelEngine"
        @Volatile
        private var activeWorkingBridgeUrl: String? = null
    }

    override suspend fun runAgentInference(
        prompt: String,
        toolsArray: JSONArray
    ): AgentPrediction = runInference(prompt, toolsArray, null)

    override suspend fun runAgentInference(
        prompt: String,
        toolsArray: JSONArray,
        context: AgentInferenceContext
    ): AgentPrediction = runInference(prompt, toolsArray, context)

    private suspend fun runInference(
        prompt: String,
        toolsArray: JSONArray,
        context: AgentInferenceContext?
    ): AgentPrediction = withContext(Dispatchers.IO) {
        val startTime = System.currentTimeMillis()

        // 1. Try host GPU bridge over reverse ADB or active Wi-Fi endpoint
        val bridgeResult = queryBridgeServer(prompt, toolsArray, context)
        if (bridgeResult != null) {
            val latency = System.currentTimeMillis() - startTime
            Log.i(TAG, "Bridge inference success (${latency}ms): ${bridgeResult.rawToolName} (conf=${bridgeResult.confidence})")
            return@withContext bridgeResult.copy(latencyMs = latency)
        }

        // Do not synthesize an action from stale local semantics after a live bridge failure.
        // An explicit refusal lets the runtime retry or stop without dispatching an unrelated tap.
        Log.w(TAG, "Bridge server unavailable; refusing to dispatch an ungrounded fallback action")
        val latency = System.currentTimeMillis() - startTime
        return@withContext AgentPrediction(
            action = null,
            confidence = 0.0f,
            latencyMs = latency,
            explanation = "Live ranker bridge unavailable; no grounded action produced",
            rawToolName = null,
            isRefusal = true
        )
    }

    private fun queryBridgeServer(
        prompt: String,
        toolsArray: JSONArray,
        context: AgentInferenceContext?
    ): AgentPrediction? {
        val defaultUrls = listOf(bridgeUrl, "http://192.168.31.218:8765/propose")
        val candidateUrls = if (activeWorkingBridgeUrl != null) {
            listOf(activeWorkingBridgeUrl!!) + defaultUrls.filter { it != activeWorkingBridgeUrl }
        } else {
            defaultUrls
        }
        for (urlStr in candidateUrls) {
            var conn: HttpURLConnection? = null
            try {
                val url = URL(urlStr)
                conn = url.openConnection() as HttpURLConnection
                conn.requestMethod = "POST"
                conn.connectTimeout = connectTimeoutMs
                conn.readTimeout = readTimeoutMs
                conn.doOutput = true
                conn.doInput = true
                conn.setRequestProperty("Content-Type", "application/json; charset=UTF-8")

                val payload = JSONObject().apply {
                    put("prompt", prompt)
                    put("tools", toolsArray)
                    put("top_k", 3)
                    if (context != null) {
                        put("observation", serializeObservation(context.uiState))
                        put("history", serializeHistory(context.history))
                        put("previous_failure", context.previousFailure ?: "")
                    }
                }

                Log.i(
                    TAG,
                    "Sending live observation to $urlStr: nodes=${context?.uiState?.nodes?.size ?: 0}, " +
                        "candidates=${toolsArray.length()}, history=${context?.history?.size ?: 0}"
                )

                OutputStreamWriter(conn.outputStream, "UTF-8").use { writer ->
                    writer.write(payload.toString())
                    writer.flush()
                }

                val code = conn.responseCode
                if (code == 200) {
                    val responseText = BufferedReader(InputStreamReader(conn.inputStream, "UTF-8")).use { it.readText() }
                    val respJson = JSONObject(responseText)
                    val parsed = parseBridgeResponse(respJson, toolsArray, prompt, context)
                    if (parsed != null) {
                        activeWorkingBridgeUrl = urlStr
                        val top3 = buildList {
                            add(parsed.rawToolName ?: "unknown")
                            addAll(parsed.alternativeProposals.take(2).map { it.rawToolName ?: "unknown" })
                        }
                        Log.i(TAG, "Ranker returned top-k: ${top3.joinToString(", ")}; chosen=${parsed.rawToolName}")
                    }
                    return parsed
                }
            } catch (e: Exception) {
                Log.d(TAG, "Bridge query to $urlStr skipped (${e.message})")
                if (activeWorkingBridgeUrl == urlStr) {
                    activeWorkingBridgeUrl = null
                }
            } finally {
                conn?.disconnect()
            }
        }
        return null
    }

    fun checkBridgeHealth(): Boolean {
        val defaultHealth = listOf("http://127.0.0.1:8765/health", "http://192.168.31.218:8765/health")
        val candidateUrls = if (activeWorkingBridgeUrl != null) {
            val cachedHealth = activeWorkingBridgeUrl!!.replace("/propose", "/health")
            listOf(cachedHealth) + defaultHealth.filter { it != cachedHealth }
        } else {
            defaultHealth
        }
        for (urlStr in candidateUrls) {
            var conn: HttpURLConnection? = null
            try {
                val url = URL(urlStr)
                conn = url.openConnection() as HttpURLConnection
                conn.requestMethod = "GET"
                conn.connectTimeout = 1500
                conn.readTimeout = 1500
                if (conn.responseCode == 200) {
                    val resp = BufferedReader(InputStreamReader(conn.inputStream, "UTF-8")).use { it.readText() }
                    if (resp.contains("ready")) {
                        Log.i(TAG, "Bridge health check PASSED at $urlStr: $resp")
                        activeWorkingBridgeUrl = urlStr.replace("/health", "/propose")
                        return true
                    }
                }
            } catch (e: Exception) {
                Log.d(TAG, "Bridge health check failed for $urlStr: ${e.message}")
            } finally {
                conn?.disconnect()
            }
        }
        return false
    }

    private fun serializeObservation(uiState: com.zipper.app.core.model.DistilledUiState): JSONObject {
        val nodes = JSONArray()
        uiState.nodes.forEach { node ->
            nodes.put(serializeNode(node))
        }
        return JSONObject().apply {
            put("package_name", uiState.packageName ?: JSONObject.NULL)
            put("activity_name", uiState.activityName ?: JSONObject.NULL)
            put("raw_node_count", uiState.rawNodeCount)
            put("timestamp", uiState.timestamp)
            put("nodes", nodes)
        }
    }

    private fun serializeNode(node: DistilledNode): JSONObject {
        val bounds = JSONObject().apply {
            put("left", node.bounds.left / 1000.0)
            put("top", node.bounds.top / 1000.0)
            put("right", node.bounds.right / 1000.0)
            put("bottom", node.bounds.bottom / 1000.0)
            put("cx", node.bounds.centerX / 1000.0)
            put("cy", node.bounds.centerY / 1000.0)
        }
        return JSONObject().apply {
            put("id", node.index)
            put("role", node.role.lowercase())
            put("label", node.semanticLabel)
            put("text", node.text ?: JSONObject.NULL)
            put("content_description", node.contentDescription ?: JSONObject.NULL)
            put("resource_hint", node.resourceId ?: JSONObject.NULL)
            put("region", normalizedRegion(node))
            put("bounds", bounds)
            put("clickable", node.isClickable)
            put("editable", node.isEditable)
            put("scrollable", node.isScrollable)
            put("enabled", node.isEnabled)
            put("focused", node.isFocused)
            put("checked", node.isChecked)
            put("checkable", node.isCheckable)
            put("parent_id", node.parentIndex ?: JSONObject.NULL)
            put("class_name", node.className ?: JSONObject.NULL)
            put("current_value", node.currentValue ?: JSONObject.NULL)
            put("capabilities", JSONArray(node.capabilities.toList()))
            put("depth", node.depth)
            put("child_count", 0)
            put("is_leaf", true)
            put("is_container", node.role.lowercase() in setOf("container", "list", "scroll", "webview"))
            put("label_source", when {
                !node.text.isNullOrBlank() -> "self_text"
                !node.contentDescription.isNullOrBlank() -> "content_desc"
                !node.resourceId.isNullOrBlank() -> "resource_id"
                else -> "role"
            })
        }
    }

    private fun normalizedRegion(node: DistilledNode): String {
        val horizontal = when {
            node.bounds.centerX < 333 -> "left"
            node.bounds.centerX < 667 -> "center"
            else -> "right"
        }
        val vertical = when {
            node.bounds.centerY < 333 -> "upper"
            node.bounds.centerY < 667 -> "middle"
            else -> "lower"
        }
        return "${vertical}_$horizontal"
    }

    private fun serializeHistory(history: List<AgentStepRecord>): JSONArray {
        val result = JSONArray()
        history.takeLast(6).forEach { step ->
            result.put(JSONObject().apply {
                put("step", step.turn)
                put("action", actionType(step.action))
                put("target_id", targetId(step.action) ?: JSONObject.NULL)
                put("outcome", step.outcome.name)
                put("outcomeSummary", step.reflection)
                put("state_changed", step.outcome != StepOutcome.NO_OP && step.outcome != StepOutcome.SAFETY_BLOCKED)
                put("confidence", step.confidence)
                put("action_description", step.actionDescription)
            })
        }
        return result
    }

    private fun actionType(action: AgentAction): String = when (action) {
        is AgentAction.OpenApp -> "open_app"
        is AgentAction.Tap -> "tap"
        is AgentAction.TypeText -> "type_text"
        is AgentAction.Scroll -> "scroll"
        AgentAction.PressBack -> "back"
        AgentAction.Submit -> "submit"
        AgentAction.PressHome -> "home"
        is AgentAction.Wait -> "wait"
        is AgentAction.CompleteGoal -> "complete"
        is AgentAction.FailGoal -> "fail"
    }

    private fun targetId(action: AgentAction): Int? = when (action) {
        is AgentAction.Tap -> action.targetNodeIndex
        is AgentAction.TypeText -> action.targetNodeIndex
        is AgentAction.Scroll -> action.targetNodeIndex
        else -> null
    }

    private fun parseBridgeResponse(
        respJson: JSONObject,
        toolsArray: JSONArray,
        prompt: String,
        context: AgentInferenceContext?
    ): AgentPrediction? {
        val proposalsArr = respJson.optJSONArray("proposals") ?: return null
        if (proposalsArr.length() == 0) return null

        val predictions = mutableListOf<AgentPrediction>()
        for (i in 0 until proposalsArr.length()) {
            val p = proposalsArr.getJSONObject(i)
            val toolName = p.optString("tool_name", p.optString("candidate_id"))
            val actType = p.optString("action_type")
            val targetId = if (p.has("target_id") && !p.isNull("target_id")) p.optInt("target_id") else null
            val conf = p.optDouble("confidence", 0.85).toFloat()
            val score = p.optDouble("score", 0.0)
            val expOutcome = p.optString("expected_outcome", "Model selected action")
            val rawArgs = p.optJSONObject("arguments") ?: JSONObject()

            val matchingTool = (0 until toolsArray.length())
                .map { toolsArray.getJSONObject(it) }
                .firstOrNull { it.optString("candidate_id") == toolName || it.optString("name") == toolName }
            val cId = matchingTool?.optString("candidate_id")?.takeIf { it.isNotBlank() } ?: toolName
            val sLabel = matchingTool?.optString("semantic_label")?.takeIf { it.isNotBlank() } ?: p.optString("label", "")
            val nId = matchingTool?.optString("node_identity")?.takeIf { it.isNotBlank() }
            val tNodeIdx = if (matchingTool != null && matchingTool.has("target_node_index") && !matchingTool.isNull("target_node_index")) {
                matchingTool.optInt("target_node_index").takeIf { it >= 0 }
            } else {
                targetId
            }

            val action: AgentAction? = when {
                actType == "open_app" || toolName == "c_open_app" || toolName == "open_app" -> {
                    var appName = rawArgs.optString("app_name", "").ifBlank { rawArgs.optString("appName", "") }
                    if (appName.isBlank()) {
                        appName = p.optString("app_name", "")
                    }
                    if (appName.isBlank()) {
                        val pLower = prompt.lowercase()
                        appName = when {
                            pLower.contains("open ") -> pLower.substringAfter("open ").split(Regex("\\s+(?:and|then)\\s+|,")).first().trim().split(" ")[0].trim()
                            pLower.contains("launch ") -> pLower.substringAfter("launch ").split(Regex("\\s+(?:and|then)\\s+|,")).first().trim().split(" ")[0].trim()
                            pLower.contains("ask ") -> pLower.substringAfter("ask ").split(Regex("\\s+(?:and|then|what|how|why|who|to|about)\\s+|,")).first().trim().split(" ")[0].trim()
                            pLower.contains("in settings") || pLower.contains("settings") -> "Settings"
                            pLower.contains("in contacts") || pLower.contains("contacts") -> "Contacts"
                            pLower.contains("in clock") || pLower.contains("clock") -> "Clock"
                            pLower.contains("in whatsapp") || pLower.contains("whatsapp") -> "WhatsApp"
                            pLower.contains("in calculator") || pLower.contains("calculator") -> "Calculator"
                            else -> ""
                        }
                    }
                    AgentAction.OpenApp(appName)
                }
                actType == "back" || toolName == "c_press_back" || toolName == "press_back" -> AgentAction.PressBack
                actType == "submit" || toolName == "c_submit" || toolName == "submit" -> AgentAction.Submit
                actType == "home" || toolName == "press_home" -> AgentAction.PressHome
                actType == "wait" || toolName == "wait" -> AgentAction.Wait(600L)
                actType == "complete" || toolName == "c_complete_goal" || toolName == "complete_goal" -> AgentAction.CompleteGoal(expOutcome)
                actType == "fail" || toolName == "c_fail_goal" || toolName == "fail_goal" -> AgentAction.FailGoal(expOutcome)
                actType == "scroll" || toolName.startsWith("c_scroll_") || toolName.startsWith("scroll_") -> {
                    val dirStr = rawArgs.optString("direction", if (toolName.contains("up")) "UP" else "DOWN").uppercase()
                    val dir = if (dirStr == "UP") ScrollDirection.UP else ScrollDirection.DOWN
                    AgentAction.Scroll(
                        targetNodeIndex = tNodeIdx,
                        direction = dir,
                        candidateId = cId,
                        semanticLabel = sLabel,
                        nodeIdentity = nId
                    )
                }
                actType == "type_text" || toolName.startsWith("type_") -> tNodeIdx?.let {
                    val targetNode = context?.uiState?.nodes?.firstOrNull { node -> node.index == it }
                    val comparative = ProductExplorationEngine.detectComparativeIntent(prompt)
                    val modelText = rawArgs.optString("text", "").ifBlank {
                        rawArgs.optString("extracted_text", "")
                    }.trim()
                    val resolvedText = if (comparative != null && comparative.targetEntity.isNotBlank()) {
                        comparative.targetEntity
                    } else {
                        modelText.ifBlank {
                            GoalArgumentResolver.resolveText(prompt, sLabel, targetNode)
                        }
                    }
                    AgentAction.TypeText(
                        targetNodeIndex = it,
                        text = resolvedText,
                        candidateId = cId,
                        semanticLabel = sLabel,
                        nodeIdentity = nId
                    )
                }
                actType == "tap" -> tNodeIdx?.let {
                    AgentAction.Tap(
                        targetNodeIndex = it,
                        candidateId = cId,
                        semanticLabel = sLabel,
                        nodeIdentity = nId
                    )
                }
                else -> parseToolNameToFallbackAction(toolName, prompt)
            }

            if (action != null) {
                predictions.add(
                    AgentPrediction(
                        action = action,
                        confidence = conf,
                        latencyMs = 0L,
                        explanation = expOutcome,
                        rawToolName = toolName,
                        rawJson = p.toString(),
                        isRefusal = false
                    )
                )
            }
        }

        if (predictions.isEmpty()) return null

        val primary = predictions[0]
        val alternatives = predictions.drop(1)
        val currentPkg = (context?.uiState?.packageName ?: "").lowercase()
        val targetApp = (primary.action as? AgentAction.OpenApp)?.appName?.trim()?.lowercase() ?: ""
        val isAlreadyInTargetApp = targetApp.isNotBlank() && currentPkg.contains(targetApp)
        val selected = if (primary.action is AgentAction.OpenApp && isAlreadyInTargetApp) {
            alternatives.firstOrNull { it.action != null && it.action !is AgentAction.OpenApp } ?: primary
        } else {
            primary
        }
        return selected.copy(alternativeProposals = listOf(primary) + alternatives.filter { it !== selected })
    }

    /**
     * Generic semantic tool scoring when host GPU bridge is offline.
     * Evaluates action intent and target token similarity dynamically
     * without any hardcoded app names or coordinate shortcuts.
     */
    internal fun rankToolsLocally(prompt: String, toolsArray: JSONArray): AgentPrediction {
        val pLower = prompt.lowercase()
        val cleanGoal = pLower
            .substringBefore(" [history:")
            .substringBefore(" (history:")
            .substringBefore(" [previous failure:")
            .substringBefore(" (previous action failed:")
            .trim()

        val completeToolDesc = (0 until toolsArray.length())
            .mapNotNull { toolsArray.optJSONObject(it) }
            .firstOrNull { it.optString("name") == "complete_goal" }
            ?.optString("description") ?: ""

        val historyPart = when {
            completeToolDesc.contains(" (history:") -> completeToolDesc.substringAfter(" (history:").substringBefore(")")
            pLower.contains(" [history:") -> pLower.substringAfter(" [history:").substringBefore("]")
            pLower.contains(" (history:") -> {
                val rem = pLower.substringAfter(" (history:")
                if (rem.contains(" (previous action failed:")) {
                    rem.substringBefore(" (previous action failed:").trimEnd(')', ' ')
                } else if (rem.contains(" [previous failure:")) {
                    rem.substringBefore(" [previous failure:").trimEnd(')', ' ')
                } else {
                    rem.trimEnd(')', ' ')
                }
            }
            else -> ""
        }

        // Check if all action clauses of compound goal are already completed in history
        val goalClauses = cleanGoal.split(Regex("\\s*,\\s*|\\s+(?:and|then)\\s+"))
            .map { it.trim() }
            .filter { it.isNotBlank() }
        val isCompound = goalClauses.size > 1
        val allClausesCompleted = if (isCompound && historyPart.isNotBlank()) {
            val clauses = goalClauses
            val actionClauses = clauses.filterNot { it.startsWith("open ") || it.startsWith("launch ") }
            actionClauses.isNotEmpty() && actionClauses.all { clause ->
                if (clause.startsWith("scroll") || clause.startsWith("swipe")) {
                    historyPart.contains("scroll") && (historyPart.contains("success") || historyPart.contains("displace"))
                } else if (clause.startsWith("tap ") || clause.startsWith("click ") || clause.startsWith("press ") || clause.startsWith("select ")) {
                    val targetConcept = clause.substringAfter(" ").trim()
                    val stopwords = setOf("the", "a", "an", "button", "icon", "on", "to", "item")
                    val targetTokens = targetConcept.split(Regex("[^a-z0-9]+")).filter { it.isNotBlank() && it !in stopwords }
                    historyPart.contains("tap") && historyPart.contains("success") &&
                            (targetTokens.isEmpty() || targetTokens.any { token -> historyPart.contains(token) })
                } else {
                    false
                }
            }
        } else {
            false
        }

        val scoredTools = mutableListOf<Pair<JSONObject, Float>>()

        // Determine active clause and intent
        val hasOpenAppTool = (0 until toolsArray.length()).any {
            toolsArray.optJSONObject(it)?.optString("name")?.lowercase() == "open_app"
        }

        val isOpenPhase = hasOpenAppTool && goalClauses.any {
            it.startsWith("open ") || it.startsWith("launch ")
        }

        val hasEditableTool = (0 until toolsArray.length()).any {
            toolsArray.optJSONObject(it)?.optString("name")?.lowercase()?.startsWith("type_") == true
        }
        val hasSubmitTool = (0 until toolsArray.length()).any {
            toolsArray.optJSONObject(it)?.optString("name")?.lowercase() in setOf("submit", "press_enter", "ime_enter")
        }
        val activeClause = when {
            isOpenPhase -> goalClauses.firstOrNull { it.startsWith("open ") || it.startsWith("launch ") }
            hasEditableTool -> goalClauses.firstOrNull {
                it.contains(Regex("\\b(search|find|look for|type|enter|input|send)\\b"))
            }
            hasSubmitTool -> goalClauses.firstOrNull {
                it.contains(Regex("\\b(send|submit|search|find)\\b"))
            }
            else -> goalClauses.lastOrNull()
        } ?: cleanGoal

        // Extract action verb from active clause
        val isScrollIntent = activeClause.contains("scroll") || activeClause.contains("swipe")
        val isScrollDown = activeClause.contains("down")
        val isScrollUp = activeClause.contains("up")
        val isBackIntent = activeClause.contains("back")
        val isTypeIntent = activeClause.contains(Regex("\\b(type|enter|input|search|find|look for|send)\\b"))
        val isSubmitIntent = activeClause.contains(Regex("\\b(send|submit|search)\\b")) &&
            !hasEditableTool && hasSubmitTool

        // Extract target tokens excluding action stopwords
        val stopwords = setOf("open", "launch", "tap", "click", "press", "select", "the", "a", "an", "on", "to", "button", "icon", "and", "then", "into", "in", "down", "up", "scroll", "swipe")
        val targetTokens = activeClause.split(Regex("[^a-z0-9]+")).filter { it.isNotBlank() && it !in stopwords && it.length > 1 }

        for (i in 0 until toolsArray.length()) {
            val t = toolsArray.optJSONObject(i) ?: continue
            val name = t.optString("name").lowercase()
            val desc = t.optString("description").lowercase()

            var score = 0.05f

            if (name == "open_app") {
                if (isOpenPhase) {
                    score += 0.90f
                } else {
                    score = 0.01f
                }
            } else if (name == "scroll_down") {
                if (allClausesCompleted) {
                    score = 0.05f
                } else if (isScrollIntent && (isScrollDown || !isScrollUp)) {
                    score += 0.85f
                }
            } else if (name == "scroll_up") {
                if (allClausesCompleted) {
                    score = 0.05f
                } else if (isScrollIntent && isScrollUp) {
                    score += 0.85f
                }
            } else if (name == "press_back") {
                if (isBackIntent) {
                    score += 0.85f
                }
            } else if (name == "submit" || name == "press_enter" || name == "ime_enter") {
                if (isSubmitIntent) {
                    score += 0.90f
                }
            } else if (name == "complete_goal") {
                if (allClausesCompleted) {
                    score = 1.50f
                } else {
                    score = 0.02f
                }
            } else if (name == "fail_goal") {
                score = 0.01f
            } else if (name.startsWith("type_")) {
                if (isTypeIntent) {
                    score += 0.50f
                    val toolWords = desc.split(Regex("[^a-z0-9]+")).filter { it.length > 1 }
                    val overlap = targetTokens.count { it in toolWords }
                    score += (overlap * 0.35f)
                }
            } else if (name.startsWith("tap_")) {
                if (allClausesCompleted) {
                    score = 0.05f
                } else if (!isScrollIntent && !isBackIntent) {
                    val toolWords = desc.split(Regex("[^a-z0-9]+")).filter { it.length > 1 }.toSet()
                    if (targetTokens.isNotEmpty()) {
                        val matchingTokens = targetTokens.filter { it in toolWords }
                        val overlapCount = matchingTokens.size

                        if (overlapCount == targetTokens.size) {
                            // Perfect target phrase match
                            score += 1.10f
                        } else if (overlapCount > 0) {
                            score += (overlapCount.toFloat() / targetTokens.size.toFloat()) * 0.85f
                        }
                    } else {
                        score += 0.15f
                    }
                }
            }

            scoredTools.add(t to score)
        }

        scoredTools.sortByDescending { it.second }

        val predictions = mutableListOf<AgentPrediction>()
        for ((t, score) in scoredTools.take(3)) {
            val name = t.optString("name")
            val desc = t.optString("description")
            val act = parseToolNameToFallbackAction(name, prompt, desc)
            if (act != null) {
                predictions.add(
                    AgentPrediction(
                        action = act,
                        confidence = score.coerceIn(0.10f, 0.95f),
                        latencyMs = 1L,
                        explanation = desc.ifBlank { name },
                        rawToolName = name,
                        isRefusal = false
                    )
                )
            }
        }

        if (predictions.isEmpty()) {
            return AgentPrediction(
                action = AgentAction.PressBack,
                confidence = 0.30f,
                latencyMs = 1L,
                explanation = "Default fallback action",
                rawToolName = "press_back",
                isRefusal = false
            )
        }

        val primary = predictions[0]
        return primary.copy(alternativeProposals = predictions.drop(1))
    }

    private fun parseToolNameToFallbackAction(toolName: String, prompt: String = "", description: String = ""): AgentAction? {
        val lower = toolName.lowercase()
        return when {
            lower == "open_app" -> {
                val pLower = prompt.lowercase()
                val targetApp = when {
                    pLower.contains("open ") -> pLower.substringAfter("open ").split(Regex("\\s+(?:and|then)\\s+|,")).first().trim().split(" ")[0].trim()
                    pLower.contains("launch ") -> pLower.substringAfter("launch ").split(Regex("\\s+(?:and|then)\\s+|,")).first().trim().split(" ")[0].trim()
                    pLower.contains(" on ") -> pLower.substringAfter(" on ").split(Regex("\\s+(?:and|then)\\s+|,")).first().trim().split(" ")[0].trim()
                    pLower.contains(" in ") -> pLower.substringAfter(" in ").split(Regex("\\s+(?:and|then)\\s+|,")).first().trim().split(" ")[0].trim()
                    pLower.contains(" from ") -> pLower.substringAfter(" from ").split(Regex("\\s+(?:and|then)\\s+|,")).first().trim().split(" ")[0].trim()
                    else -> ""
                }.filter { it.isLetterOrDigit() }
                AgentAction.OpenApp(targetApp)
            }
            lower.startsWith("tap_") -> {
                val idx = Regex("""\d+""").findAll(toolName).lastOrNull()?.value?.toIntOrNull()
                idx?.let { AgentAction.Tap(it) }
            }
            lower.startsWith("type_") -> {
                val idx = Regex("""\d+""").findAll(toolName).lastOrNull()?.value?.toIntOrNull()
                idx?.let { AgentAction.TypeText(it, GoalArgumentResolver.resolveText(prompt, description)) }
            }
            lower == "scroll_down" || lower.startsWith("scroll_down_") -> {
                val idx = Regex("""\d+""").findAll(toolName).lastOrNull()?.value?.toIntOrNull()
                AgentAction.Scroll(targetNodeIndex = idx, direction = ScrollDirection.DOWN)
            }
            lower == "scroll_up" || lower.startsWith("scroll_up_") -> {
                val idx = Regex("""\d+""").findAll(toolName).lastOrNull()?.value?.toIntOrNull()
                AgentAction.Scroll(targetNodeIndex = idx, direction = ScrollDirection.UP)
            }
            lower == "press_back" -> AgentAction.PressBack
            lower == "submit" || lower == "press_enter" || lower == "ime_enter" -> AgentAction.Submit
            lower == "press_home" -> AgentAction.PressHome
            lower == "complete_goal" -> AgentAction.CompleteGoal("Goal verified")
            lower == "fail_goal" -> AgentAction.FailGoal("Cannot achieve goal")
            else -> {
                val idx = Regex("""\d+""").findAll(toolName).lastOrNull()?.value?.toIntOrNull()
                idx?.let { AgentAction.Tap(it) }
            }
        }
    }
}
