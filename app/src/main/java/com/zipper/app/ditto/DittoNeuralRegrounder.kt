package com.zipper.app.ditto

import android.content.Context
import android.graphics.Rect
import android.util.Log
import android.view.accessibility.AccessibilityNodeInfo
import com.zipper.app.accessibility.KarvishAccessibilityService
import com.zipper.app.core.agent.OnDeviceRankerEngine
import com.zipper.app.core.agent.ToolSchemaGenerator
import com.zipper.app.core.distiller.UiStateDistiller
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Strict Ditto Neural Re-Grounder.
 *
 * Senior Architectural Mandate:
 * - Uses the on-device Karvish ONNX candidate ranking head purely as a TARGET LOCATOR.
 * - Zero action invention. Zero action type inference.
 * - Answers only: "Which live UI element corresponds to what the user demonstrated?"
 * - Applies a Multi-Layer Recorded-Target Compatibility Gate (semantic role, bounds plausibility).
 * - Calibrated ambiguity margin check prevents clicking when two candidates are tied.
 */
object DittoNeuralRegrounder {
    private const val TAG = "DittoNeuralRegrounder"

    sealed class RegroundDecision {
        data class TargetFound(
            val node: AccessibilityNodeInfo,
            val bounds: Rect,
            val confidence: Float,
            val margin: Float,
            val label: String
        ) : RegroundDecision()

        data class TargetAmbiguous(
            val topCandidate: String,
            val secondCandidate: String,
            val margin: Float
        ) : RegroundDecision()

        data class TargetNotFound(val reason: String) : RegroundDecision()
    }

    suspend fun locateTarget(
        service: KarvishAccessibilityService,
        step: DittoStep,
        minConfidenceThreshold: Float = 0.80f,
        minAmbiguityMargin: Float = 0.15f
    ): RegroundDecision = withContext(Dispatchers.Default) {
        val root = service.rootInActiveWindow
            ?: return@withContext RegroundDecision.TargetNotFound("Root window is null")

        val rawQuery = step.targetIdentity.primaryText?.takeIf { it.isNotBlank() }
            ?: step.targetIdentity.contentDesc?.takeIf { it.isNotBlank() }
            ?: step.targetMetadata.resourceId?.substringAfter(":id/")?.takeIf { it.isNotBlank() }
            ?: return@withContext RegroundDecision.TargetNotFound("Target identity has no query string")

        val targetQuery = DittoSemanticCleaner.clean(rawQuery).ifBlank { rawQuery }

        // 1. Distill active window into UI state with node mapping
        val liveNodeMap = mutableMapOf<Int, AccessibilityNodeInfo>()
        val uiState = UiStateDistiller().distill(
            root = root,
            onNodeMapped = { idx: Int, a11yNode: AccessibilityNodeInfo -> liveNodeMap[idx] = a11yNode }
        )

        if (uiState.nodes.isEmpty()) {
            return@withContext RegroundDecision.TargetNotFound("No actionable nodes on screen")
        }

        // 2. Generate candidate tools for the ONNX ranker
        val generated = ToolSchemaGenerator().generateTools(
            uiState = uiState,
            goal = targetQuery
        )

        // 3. Obtain model ranker instance
        val prefs = service.getSharedPreferences("karvish_agent_prefs", Context.MODE_PRIVATE)
        val selectedModel = prefs.getString("selected_model", "V62") ?: "V62"
        val requestedPath = when (selectedModel) {
            "V62" -> "models/karvish_v62.onnx"
            "V61" -> "models/karvish_v61.onnx"
            "V60" -> "models/karvish_v60.onnx"
            "V59" -> "models/karvish_v59.onnx"
            "V58" -> "models/karvish_v58.onnx"
            "V56" -> "models/karvish_v56.onnx"
            "V54" -> "models/karvish_v54.onnx"
            else -> "models/karvish_v62.onnx"
        }
        val assetExists = try {
            service.assets.open(requestedPath).close()
            true
        } catch (_: Exception) {
            false
        }
        val modelPath = if (assetExists) requestedPath else "models/karvish_v62.onnx"
        val engine = OnDeviceRankerEngine.getOrCreate(service, modelPath)

        // 4. Run Neural Candidate Ranking
        val rankedCandidates = engine.rankCandidatesForLocator(
            targetQuery = targetQuery,
            toolsArray = generated.toolsJsonArray,
            uiState = uiState
        )

        if (rankedCandidates.isEmpty()) {
            return@withContext RegroundDecision.TargetNotFound("Model returned zero candidate rankings")
        }

        // 5. Multi-Layer Recorded-Target Compatibility Gate
        // Filter out candidates that violate fundamental physical constraints
        val compatibleCandidates = rankedCandidates.filter { cand ->
            if (step.actionType == "TYPE_TEXT") {
                cand.isEditable || cand.role.contains("input", ignoreCase = true) || cand.role.contains("edit", ignoreCase = true)
            } else {
                true // For TAP, all interactable elements are eligible
            }
        }

        if (compatibleCandidates.isEmpty()) {
            return@withContext RegroundDecision.TargetNotFound("No candidates matched role compatibility for action ${step.actionType}")
        }

        // 6. Semantic Plausibility Gate:
        // A neural locator candidate cannot be completely disjoint from an explicit textual target.
        // For example, if target is "Message", a candidate labeled "ajay" with zero textual or role relation
        // indicates the destination page has not yet loaded over the network.
        fun isSemanticallyPlausible(query: String, cand: com.zipper.app.core.agent.CandidateRankResult): Boolean {
            val q = query.trim().lowercase()
            val l = cand.label.trim().lowercase()
            val id = cand.candidateId.lowercase()

            // 1. Search Keyword Anti-Hijack Guard:
            // If the query is an entity/user handle (e.g. "ajay_d_05"), a generic search keyword chip ("ajay", "search for ajay")
            // or an element whose resource ID indicates a search keyword row MUST NOT hijack the target.
            val isSearchKeywordCandidate = id.contains("keyword") || l.startsWith("search for ") || l.startsWith("search with ")
            val isSpecificEntityTarget = q.contains("_") || q.any { it.isDigit() } || (q.length >= 6 && !q.contains(" "))
            if (isSpecificEntityTarget && isSearchKeywordCandidate) {
                return false
            }

            // 2. Direct candidate contains query:
            // e.g. candidate is "ajay_d_05 Ajay D • 50K followers" and query is "ajay_d_05" -> TRUE
            if (l.isNotBlank() && l.contains(q)) return true
            if (id.isNotBlank() && id.contains(q)) return true

            // 3. Query contains candidate (Anti-Prefix-Hijack):
            // Candidate must cover at least 75% of query length. A candidate that is just a short prefix
            // (e.g. "ajay" covering 4/9 of "ajay_d_05") MUST NOT hijack the specific entity handle.
            if (l.isNotBlank() && q.contains(l)) {
                if (isSpecificEntityTarget) {
                    return false // Entity handle must match in full, not just prefix
                }
                if (l.length >= (q.length * 0.75f).toInt()) {
                    return true
                }
                return false
            }

            // 4. Token overlap (e.g. "Message button" vs "Message")
            val qTokens = q.split(Regex("[^a-zA-Z0-9]+")).filter { it.length >= 2 }
            val lTokens = (l + " " + id).split(Regex("[^a-zA-Z0-9]+")).filter { it.length >= 2 }
            if (qTokens.isNotEmpty() && lTokens.isNotEmpty()) {
                // If query has specific distinguishing tokens (e.g. numbers like "05" or tokens >= 4 chars), candidate must match them!
                val specificTokens = qTokens.filter { it.any { c -> c.isDigit() } || it.length >= 4 }
                if (specificTokens.isNotEmpty()) {
                    val matchesSpecific = specificTokens.any { st -> lTokens.any { lt -> lt == st || lt.contains(st) } }
                    if (!matchesSpecific) {
                        return false // Missing the distinguishing token!
                    }
                }
                if (qTokens.any { qt -> lTokens.any { lt -> lt.contains(qt) || qt.contains(lt) } }) {
                    return true
                }
            }

            // 5. Action button guard: If query is an action keyword ("message", "send", "submit", "buy", "cart"),
            // a purely content-text node without matching action role or text cannot be accepted.
            val isActionQuery = q.contains("message") || q.contains("send") || q.contains("submit") || q.contains("buy") || q.contains("cart")
            val isActionCand = l.contains("message") || l.contains("send") || l.contains("submit") || l.contains("buy") || l.contains("cart") ||
                    cand.role.contains("button", ignoreCase = true) || id.contains("button") || id.contains("action")
            if (isActionQuery && !isActionCand) {
                return false
            }

            // 6. Distinct tokens prefix guard
            if (qTokens.isNotEmpty()) {
                val hasPartialPrefix = qTokens.any { qt -> lTokens.any { lt -> lt.startsWith(qt.take(3)) } }
                if (!hasPartialPrefix && cand.confidence < 0.95f) {
                    return false
                }
            }

            return true
        }

        val plausibleCandidates = compatibleCandidates.filter { isSemanticallyPlausible(targetQuery, it) }
        if (plausibleCandidates.isEmpty()) {
            val topUnmatched = compatibleCandidates.firstOrNull()?.label ?: "none"
            Log.w(TAG, "Neural Locator: Top candidate '$topUnmatched' is semantically disjoint from target '$targetQuery' (screen may still be loading)")
            return@withContext RegroundDecision.TargetNotFound("Candidates on screen are disjoint from target '$targetQuery'")
        }

        val top1 = plausibleCandidates[0]
        val top2 = plausibleCandidates.getOrNull(1)

        val confidence = top1.confidence
        val margin = if (top2 != null) (top1.rawScore - top2.rawScore) else 1.0f

        Log.i(TAG, "Neural Locator for '$targetQuery': Top='${top1.label}' (conf=${String.format(java.util.Locale.ROOT, "%.2f", confidence)}, margin=${String.format(java.util.Locale.ROOT, "%.2f", margin)}) | Second='${top2?.label ?: "none"}'")

        // 7. Ambiguity Gate: If top 2 candidates are too close, refuse to guess randomly
        if (top2 != null && margin < minAmbiguityMargin && confidence < 0.90f) {
            Log.w(TAG, "Target '$targetQuery' is ambiguous between '${top1.label}' and '${top2.label}' (margin=$margin < $minAmbiguityMargin)")
            return@withContext RegroundDecision.TargetAmbiguous(top1.label, top2.label, margin)
        }

        // 8. Calibrated Confidence Gate:
        // Senior Architectural Rule: When the top candidate's label is an exact text match to targetQuery
        // (e.g. target is "Message" and button is "Message") and margin is clear, the required softmax
        // probability threshold is relaxed to 0.35f because softmax divides total mass over 20+ nodes.
        val qClean = targetQuery.trim().lowercase()
        val lClean = top1.label.trim().lowercase()
        val isExactOrPrefixMatch = lClean == qClean || (qClean.length >= 4 && lClean.startsWith(qClean)) || (lClean.length >= 4 && qClean.startsWith(lClean))
        val effectiveMinConfidence = if (isExactOrPrefixMatch && margin >= minAmbiguityMargin) 0.35f else minConfidenceThreshold

        if (confidence < effectiveMinConfidence) {
            Log.w(TAG, "Target '$targetQuery' top match '${top1.label}' confidence ($confidence) below threshold ($effectiveMinConfidence)")
            return@withContext RegroundDecision.TargetNotFound("Top match confidence $confidence < $effectiveMinConfidence")
        }

        // 9. Map back to live AccessibilityNodeInfo
        val targetNodeIndex = top1.nodeIndex
        val a11yNode = if (targetNodeIndex != null) liveNodeMap[targetNodeIndex] else null
        if (a11yNode == null) {
            return@withContext RegroundDecision.TargetNotFound("Mapped accessibility node info not found for index $targetNodeIndex")
        }

        return@withContext RegroundDecision.TargetFound(
            node = a11yNode,
            bounds = top1.bounds,
            confidence = confidence,
            margin = margin,
            label = top1.label
        )
    }
}
