package com.zipper.app.core.needle

import android.content.Context
import android.util.Log
import com.zipper.app.core.model.ActionPrediction
import com.zipper.app.core.model.ActionType
import com.zipper.app.core.model.DistilledNode
import com.zipper.app.core.model.DistilledUiState
import com.zipper.app.core.model.FlowStep
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import kotlin.system.measureTimeMillis

/**
 * Genuine Native Cactus Needle 2 Engine (45M CQ2-bit, Native ARM64).
 *
 * Executes the official compiled binary from Cactus Compute (Apache License 2.0).
 * Invokes the model with dynamic JSON tool schemas derived from the live distilled UI state.
 */
class Needle2NativeEngine(private val context: Context) : com.zipper.app.core.agent.AgentModelEngine {

    companion object {
        private const val TAG = "Needle2NativeEngine"
        private const val ASSET_NAME = "needle/needle-arm64"
        private const val BINARY_NAME = "needle_arm64_bin"
    }

    private var executableFile: File? = null

    init {
        ensureBinaryExtracted()
    }

    private fun ensureBinaryExtracted() {
        try {
            // 1. Check official native library path (installed via jniLibs/arm64-v8a/libneedle.so)
            val nativeLibDir = context.applicationInfo.nativeLibraryDir
            val nativeBin = File(nativeLibDir, "libneedle.so")
            if (nativeBin.exists() && nativeBin.length() > 10_000_000L) {
                executableFile = nativeBin
                Log.i(TAG, "Official Needle 2 binary loaded from nativeLibraryDir: ${nativeBin.absolutePath} (size: ${nativeBin.length()} bytes)")
            } else {
                // 2. Fallback: extract from assets to filesDir if present
                val targetFile = File(context.filesDir, BINARY_NAME)
                if (!targetFile.exists() || targetFile.length() < 10_000_000L) {
                    try {
                        Log.i(TAG, "Extracting official Needle 2 ARM64 binary from assets...")
                        context.assets.open(ASSET_NAME).use { input ->
                            FileOutputStream(targetFile).use { output ->
                                input.copyTo(output)
                            }
                        }
                        targetFile.setExecutable(true, false)
                        Runtime.getRuntime().exec("chmod 755 ${targetFile.absolutePath}").waitFor()
                        Log.i(TAG, "Needle 2 binary extracted successfully: ${targetFile.absolutePath} (size: ${targetFile.length()} bytes)")
                    } catch (e: Exception) {
                        Log.d(TAG, "Fallback asset binary not bundled: ${e.message}")
                    }
                }
                if (targetFile.exists()) {
                    executableFile = targetFile
                }
            }

            // 3. Extract V3 LoRA adapter if bundled in assets
            try {
                val loraFile = File(context.filesDir, "needle_lora_v3.safetensors")
                if (!loraFile.exists() || loraFile.length() < 100_000L) {
                    context.assets.open("needle/needle_lora_v3.safetensors").use { input ->
                        FileOutputStream(loraFile).use { output ->
                            input.copyTo(output)
                        }
                    }
                    Log.i(TAG, "Extracted Needle 2 V3 LoRA adapter: ${loraFile.absolutePath} (${loraFile.length()} bytes)")
                }
            } catch (e: Exception) {
                Log.d(TAG, "No needle_lora_v3.safetensors in assets: ${e.message}")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to prepare native Needle 2 binary: ${e.message}", e)
        }
    }

    suspend fun runInference(
        step: FlowStep,
        uiState: DistilledUiState
    ): ActionPrediction = withContext(Dispatchers.IO) {
        val bin = executableFile
        if (bin == null || !bin.exists()) {
            return@withContext ActionPrediction(
                selectedNodeIndex = null,
                action = step.action,
                confidence = 0.0f,
                latencyMs = 0L,
                explanation = "ERROR: Native Needle 2 binary not available",
                modelName = "REAL_MODEL_UNAVAILABLE"
            )
        }

        // 1. Build Tools JSON from Distilled UI State
        val toolsArray = JSONArray()
        for (node in uiState.nodes) {
            val toolObj = JSONObject()
            toolObj.put("name", "candidate_${node.index}")
            val richLabel = listOfNotNull(node.text, node.contentDescription, node.resourceId?.substringAfterLast("/"))
                .filter { it.isNotBlank() }
                .distinct()
                .joinToString(" ")
            val labelDesc = if (richLabel.isNotBlank()) richLabel else node.semanticLabel
            toolObj.put("description", "${node.role} labeled '$labelDesc' at bounds (${node.bounds.centerX}, ${node.bounds.centerY})")

            val paramsObj = JSONObject()
            paramsObj.put("type", "object")
            val propsObj = JSONObject()
            val intentProp = JSONObject().put("type", "string").put("description", "intent action")
            propsObj.put("intent", intentProp)
            paramsObj.put("properties", propsObj)
            toolObj.put("parameters", paramsObj)

            toolsArray.put(toolObj)
        }

        val toolsFile = File(context.cacheDir, "needle_tools.json")
        toolsFile.writeText(toolsArray.toString())

        val prompt = step.intent.trim()

        var selectedIndex: Int? = null
        var confidence = 0.0f
        var explanation = ""
        var isRealModel = false
        var decodeTps: Double? = null
        var peakRamMb: Double? = null
        var isRefusal = false

        val latencyMs = measureTimeMillis {
            try {
                val processBuilder = ProcessBuilder(
                    bin.absolutePath,
                    "--tools", toolsFile.absolutePath,
                    "--prompt", prompt
                )
                processBuilder.redirectErrorStream(true)
                val process = processBuilder.start()

                val outputText = process.inputStream.bufferedReader().use { it.readText() }.trim()
                process.waitFor()

                Log.d(TAG, "REAL_MODEL raw stdout: $outputText")

                if (outputText.startsWith("{") && outputText.contains("\"function_calls\"")) {
                    isRealModel = true
                    val json = JSONObject(outputText)
                    confidence = json.optDouble("confidence", 0.0).toFloat()
                    val reasoning = json.optString("reasoning", "")
                    decodeTps = json.optDouble("decode_tps", 0.0)
                    peakRamMb = json.optDouble("peak_ram_mb", 0.0)

                    val functionCalls = json.optJSONArray("function_calls")
                    if (functionCalls != null && functionCalls.length() > 0) {
                        val firstCall = functionCalls.getJSONObject(0)
                        val toolName = firstCall.getString("name") // e.g. "candidate_8"
                        val idxMatch = Regex("""\d+""").findAll(toolName).lastOrNull()?.value
                        selectedIndex = idxMatch?.toIntOrNull()
                        isRefusal = false
                        explanation = "REAL_MODEL: matched '$toolName' | confidence: ${"%.4f".format(confidence)} | reasoning: '$reasoning' | decode_speed: ${"%.1f".format(decodeTps)} tps | ram: ${"%.1f".format(peakRamMb)} MB"
                    } else {
                        isRefusal = true
                        selectedIndex = null
                        explanation = "REAL_MODEL: refusal / no tool declared | reasoning: '$reasoning' | confidence: ${"%.4f".format(confidence)}"
                    }
                } else {
                    explanation = "REAL_MODEL execution error or non-JSON output: $outputText"
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error executing native Needle binary", e)
                explanation = "Execution error: ${e.message}"
            }
        }

        val modelTag = if (isRealModel) "REAL_MODEL: Cactus-Needle-2-SAN-45M (Native ARM64)" else "REAL_MODEL_ERROR"
        val argument = step.literalArgument ?: step.argumentSlot

        ActionPrediction(
            selectedNodeIndex = selectedIndex,
            action = step.action,
            argument = argument,
            confidence = confidence,
            latencyMs = latencyMs,
            explanation = explanation,
            modelName = modelTag,
            decodeTps = decodeTps,
            peakRamMb = peakRamMb,
            isRefusal = isRefusal
        )
    }

    /**
     * Executes autonomous agent inference where Needle 2 chooses ANY semantic action tool
     * from dynamic schemas (Tap, Type, Scroll, Back, CompleteGoal, FailGoal).
     */
    override suspend fun runAgentInference(
        prompt: String,
        toolsArray: JSONArray
    ): AgentPrediction = withContext(Dispatchers.IO) {
        val bin = executableFile
        if (bin == null || !bin.exists()) {
            return@withContext AgentPrediction(
                action = null,
                confidence = 0.0f,
                latencyMs = 0L,
                explanation = "Native Needle 2 binary unavailable",
                rawToolName = null,
                decodeTps = null,
                peakRamMb = null,
                isRefusal = true
            )
        }

        val toolsFile = File.createTempFile("needle_agent_tools_", ".json", context.cacheDir)
        var action: com.zipper.app.core.agent.AgentAction? = null
        var confidence = 0.0f
        var explanation = ""
        var rawToolName: String? = null
        var decodeTps: Double? = null
        var peakRamMb: Double? = null
        var isRefusal = false

        var outputText = ""
        val latencyMs = measureTimeMillis {
            try {
                toolsFile.writeText(toolsArray.toString())

                val processBuilder = ProcessBuilder(
                    bin.absolutePath,
                    "--tools", toolsFile.absolutePath,
                    "--prompt", prompt.trim()
                )
                processBuilder.redirectErrorStream(true)
                val process = processBuilder.start()

                outputText = process.inputStream.bufferedReader().use { it.readText() }.trim()
                process.waitFor()

                Log.d(TAG, "Agent inference raw stdout: $outputText")

                if (outputText.startsWith("{") && outputText.contains("\"function_calls\"")) {
                    val json = JSONObject(outputText)
                    confidence = json.optDouble("confidence", 0.0).toFloat()
                    val reasoning = json.optString("reasoning", "")
                    decodeTps = json.optDouble("decode_tps", 0.0)
                    peakRamMb = json.optDouble("peak_ram_mb", 0.0)

                    val functionCalls = json.optJSONArray("function_calls")
                    if (functionCalls != null && functionCalls.length() > 0) {
                        var selectedToolName: String? = null
                        var selectedAction: com.zipper.app.core.agent.AgentAction? = null

                        // If the model predicted both an active interaction (tap/type/scroll) and completion,
                        // execute the active interaction first so the screen state actually updates.
                        val callList = (0 until functionCalls.length()).map { functionCalls.getJSONObject(it) }
                        val activeCall = callList.firstOrNull {
                            val n = it.getString("name").lowercase()
                            n != "complete_goal" && n != "fail_goal" && parseToolCallToAction(n, it.optJSONObject("arguments") ?: it.optJSONObject("parameters") ?: JSONObject(), reasoning) != null
                        } ?: callList.firstOrNull {
                            val n = it.getString("name")
                            parseToolCallToAction(n, it.optJSONObject("arguments") ?: it.optJSONObject("parameters") ?: JSONObject(), reasoning) != null
                        }

                        if (activeCall != null) {
                            val name = activeCall.getString("name")
                            val args = activeCall.optJSONObject("arguments") ?: activeCall.optJSONObject("parameters") ?: JSONObject()
                            selectedToolName = name
                            selectedAction = parseToolCallToAction(name, args, reasoning)
                        }

                        action = selectedAction
                        rawToolName = selectedToolName
                        isRefusal = action == null
                        explanation = "Needle 2 chose '$selectedToolName' | confidence: ${"%.4f".format(confidence)} | reasoning: '$reasoning'"
                    } else {
                        isRefusal = true
                        explanation = "Needle 2 refusal / no tool called: '$reasoning'"
                    }
                } else {
                    isRefusal = true
                    explanation = "Needle 2 output non-JSON or error: $outputText"
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error executing agent Needle inference", e)
                isRefusal = true
                explanation = "Execution error: ${e.message}"
            } finally {
                try { toolsFile.delete() } catch (_: Exception) {}
            }
        }

        AgentPrediction(
            action = action,
            confidence = confidence,
            latencyMs = latencyMs,
            explanation = explanation,
            rawToolName = rawToolName,
            rawJson = outputText,
            decodeTps = decodeTps,
            peakRamMb = peakRamMb,
            isRefusal = isRefusal
        )
    }

    private fun parseToolCallToAction(
        toolName: String,
        params: JSONObject,
        reasoning: String
    ): com.zipper.app.core.agent.AgentAction? {
        val lower = toolName.lowercase()
        return when {
            lower == "open_app" -> {
                val appName = params.optString("app_name", "").ifBlank { params.optString("name", "") }
                if (appName.isNotBlank()) com.zipper.app.core.agent.AgentAction.OpenApp(appName) else null
            }
            lower.startsWith("tap_") || lower.startsWith("candidate_") -> {
                val idx = Regex("""\d+""").findAll(toolName).lastOrNull()?.value?.toIntOrNull()
                    ?: params.optInt("element_id", -1).takeIf { it >= 0 }
                idx?.let { com.zipper.app.core.agent.AgentAction.Tap(it) }
            }
            lower.startsWith("type_") -> {
                val idx = Regex("""\d+""").findAll(toolName).lastOrNull()?.value?.toIntOrNull()
                    ?: params.optInt("element_id", -1).takeIf { it >= 0 }
                val text = params.optString("text", "")
                idx?.let { com.zipper.app.core.agent.AgentAction.TypeText(it, text) }
            }
            lower == "scroll_down" -> {
                com.zipper.app.core.agent.AgentAction.Scroll(direction = com.zipper.app.core.agent.ScrollDirection.DOWN)
            }
            lower == "scroll_up" -> {
                com.zipper.app.core.agent.AgentAction.Scroll(direction = com.zipper.app.core.agent.ScrollDirection.UP)
            }
            lower == "scroll" || lower == "scroll_screen" -> {
                val dirStr = params.optString("direction", "DOWN")
                val dir = if (dirStr.equals("UP", true)) com.zipper.app.core.agent.ScrollDirection.UP else com.zipper.app.core.agent.ScrollDirection.DOWN
                com.zipper.app.core.agent.AgentAction.Scroll(direction = dir)
            }
            lower == "press_back" -> {
                com.zipper.app.core.agent.AgentAction.PressBack
            }
            lower == "press_home" -> {
                com.zipper.app.core.agent.AgentAction.PressHome
            }
            lower == "complete_goal" -> {
                val summary = params.optString("summary", reasoning.ifBlank { "Goal completed" })
                com.zipper.app.core.agent.AgentAction.CompleteGoal(summary)
            }
            lower == "fail_goal" -> {
                val reason = params.optString("reason", reasoning.ifBlank { "Goal failed" })
                com.zipper.app.core.agent.AgentAction.FailGoal(reason)
            }
            else -> {
                val idx = Regex("""\d+""").findAll(toolName).lastOrNull()?.value?.toIntOrNull()
                idx?.let { com.zipper.app.core.agent.AgentAction.Tap(it) }
            }
        }
    }
}

data class AgentPrediction(
    val action: com.zipper.app.core.agent.AgentAction?,
    val confidence: Float,
    val latencyMs: Long,
    val explanation: String,
    val rawToolName: String?,
    val rawJson: String? = null,
    val decodeTps: Double? = null,
    val peakRamMb: Double? = null,
    val isRefusal: Boolean = false,
    val alternativeProposals: List<AgentPrediction> = emptyList(),
    val extractedText: String? = null,
    val isVisionActive: Boolean = false,
    val visionIconClass: String? = null
)
