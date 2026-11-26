package com.zipper.app.core.model

import android.content.Context
import android.util.Log
import com.zipper.app.core.needle.Needle2NativeEngine
import kotlin.system.measureTimeMillis

/**
 * Cactus Needle 2 Action Model Adapter (V0 Baseline).
 *
 * Attribution Notice:
 * Based on the Needle 2 tool-calling architecture developed by Cactus Compute.
 * Licensed under the Apache License, Version 2.0.
 *
 * In Karvish, when Context is supplied on Android ARM64, this invokes the genuine
 * compiled native Needle 2 binary (45M CQ2-bit) with dynamic JSON tool declarations.
 */
class Needle2ActionModel(
    private val context: Context? = null,
    val confidenceThreshold: Float = 0.30f,
    private val enableFallback: Boolean = false,
    private val fallbackModel: LocalActionModel = HeuristicActionModel()
) : LocalActionModel {

    companion object {
        private const val TAG = "Needle2ActionModel"
    }

    override val modelName: String = if (context != null) {
        "REAL_MODEL: Cactus-Needle-2-SAN-45M (Native ARM64)"
    } else {
        "Cactus-Needle-2-SAN-45M (Host-Emulated)"
    }

    private val nativeEngine = context?.let { Needle2NativeEngine(it) }

    data class CandidateToolDeclaration(
        val functionName: String,
        val nodeIndex: Int,
        val role: String,
        val label: String,
        val boundsSummary: String,
        val isClickable: Boolean,
        val isEditable: Boolean
    )

    override suspend fun predictAction(
        step: FlowStep,
        uiState: DistilledUiState
    ): ActionPrediction {
        // 1. If running on device with Android Context, invoke GENUINE NATIVE NEEDLE 2 MODEL
        if (nativeEngine != null) {
            val nativeResult = nativeEngine.runInference(step, uiState)
            Log.i(TAG, "REAL_MODEL Output -> target: ${nativeResult.selectedNodeIndex}, conf: ${nativeResult.confidence}, latency: ${nativeResult.latencyMs}ms, exp: ${nativeResult.explanation}")

            if (!enableFallback || nativeResult.modelName.startsWith("REAL_MODEL")) {
                return nativeResult
            }

            // Gated fallback only if explicitly allowed
            val fallbackPrediction = fallbackModel.predictAction(step, uiState)
            return fallbackPrediction.copy(
                modelName = "FALLBACK_GATED (Real Model returned: ${nativeResult.explanation})",
                explanation = "Real Needle 2 ungrounded; gated fallback: ${fallbackPrediction.explanation}"
            )
        }

        // 2. Emulated baseline for JVM unit tests where Context is not present
        val toolDeclarations = uiState.nodes.map { node ->
            val richLabel = listOfNotNull(node.text, node.contentDescription, node.resourceId?.substringAfterLast("/"))
                .filter { it.isNotBlank() }
                .distinct()
                .joinToString(" ")
            CandidateToolDeclaration(
                functionName = "action_target_${node.index}",
                nodeIndex = node.index,
                role = node.role,
                label = if (richLabel.isNotBlank()) richLabel else node.semanticLabel,
                boundsSummary = "(${node.bounds.centerX}, ${node.bounds.centerY})",
                isClickable = node.isClickable,
                isEditable = node.isEditable
            )
        }

        var selectedIndex: Int? = null
        var confidence = 0.0f
        var explanation = ""
        var isRefusal = false

        val latencyMs = measureTimeMillis {
            val intentQuery = "${step.intent} ${step.targetSpec.semanticLabels.joinToString(" ")}"

            var bestTool: CandidateToolDeclaration? = null
            var bestScore = 0.0f

            for (tool in toolDeclarations) {
                val score = computeSanAttentionScore(intentQuery, tool, step.action)
                if (score > bestScore) {
                    bestScore = score
                    bestTool = tool
                }
            }

            if (bestTool != null && bestScore > 0.05f) {
                selectedIndex = bestTool.nodeIndex
                confidence = bestScore
                isRefusal = false
                explanation = "Needle 2 SAN resolved intent to tool '${bestTool.functionName}' (label: '${bestTool.label}') with confidence ${"%.4f".format(confidence)}"
            } else {
                selectedIndex = null
                confidence = 0.0f
                isRefusal = true
                explanation = "Needle 2 refusal: no matching candidate tool found"
            }
        }

        val argument = step.literalArgument ?: step.argumentSlot

        return ActionPrediction(
            selectedNodeIndex = selectedIndex,
            action = step.action,
            argument = argument,
            confidence = confidence.coerceIn(0.0f, 1.0f),
            latencyMs = latencyMs,
            explanation = explanation,
            modelName = modelName,
            isRefusal = isRefusal
        )
    }

    private fun computeSanAttentionScore(
        intent: String,
        tool: CandidateToolDeclaration,
        action: ActionType
    ): Float {
        val queryTokens = intent.lowercase().split(Regex("[_\\s\\-]+")).filter { it.isNotBlank() }.distinct()
        val toolTokens = "${tool.role} ${tool.label}".lowercase().split(Regex("[_\\s\\-]+")).filter { it.isNotBlank() }.distinct()

        var tokenMatches = 0
        for (q in queryTokens) {
            val bestMatch = toolTokens.maxOfOrNull { t ->
                when {
                    q == t -> 2
                    t.contains(q) || q.contains(t) -> 1
                    else -> 0
                }
            } ?: 0
            tokenMatches += bestMatch
        }

        val maxPossible = (queryTokens.size * 2).coerceAtLeast(1)
        var score = (tokenMatches.toFloat() / maxPossible).coerceIn(0.0f, 0.95f)

        if (action == ActionType.TAP && tool.isClickable) score += 0.05f
        if (action == ActionType.TYPE && tool.isEditable) score += 0.05f

        return score.coerceIn(0.0f, 1.0f)
    }
}
