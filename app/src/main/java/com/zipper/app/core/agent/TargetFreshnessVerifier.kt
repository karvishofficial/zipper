package com.zipper.app.core.agent

import com.zipper.app.core.model.CandidateSignature
import com.zipper.app.core.model.DistilledNode
import com.zipper.app.core.model.DistilledUiState
import com.zipper.app.core.model.UniversalSemanticRole
import kotlin.math.sqrt

/**
 * KARVISH V50 — Target Freshness Recheck Gate (Android Runtime)
 *
 * Enforces the authoritative principle:
 * OBSERVE -> DECIDE -> RECHECK -> EXECUTE -> OBSERVE
 *
 * Verifies that the model-selected target element is still present, visible, enabled,
 * and within spatial tolerance before the platform dispatches any input event.
 */
data class TargetFreshnessResult(
    val isFresh: Boolean,
    val failureReason: String? = null,
    val remappedNodeIndex: Int? = null,
    val confidence: Float = 1.0f,
    val boundsDrift: Float = 0.0f
)

class TargetFreshnessVerifier(
    private val maxBoundsDriftUnits: Int = 80 // [0, 1000] coordinate grid tolerance
) {

    fun verifyTargetFreshness(
        targetSig: CandidateSignature,
        liveState: DistilledUiState
    ): TargetFreshnessResult {
        val targetHash = targetSig.signatureHash
        val nodes = liveState.nodes

        // 1. Exact hash match
        for (node in nodes) {
            val liveSig = buildCandidateSignature(node)
            if (liveSig.signatureHash == targetHash) {
                if (!node.isEnabled) {
                    return TargetFreshnessResult(
                        isFresh = false,
                        failureReason = "DISABLED_CONTROL",
                        remappedNodeIndex = node.index,
                        confidence = 0.0f
                    )
                }
                return TargetFreshnessResult(
                    isFresh = true,
                    remappedNodeIndex = node.index,
                    confidence = 1.0f,
                    boundsDrift = 0.0f
                )
            }
        }

        // 2. Spatial & Semantic tolerance match (for slight scrolling or animations)
        var bestMatchNode: DistilledNode? = null
        var minDrift = Float.MAX_VALUE

        val tgtCx = targetSig.normalizedBounds.centerX.toFloat()
        val tgtCy = targetSig.normalizedBounds.centerY.toFloat()

        for (node in nodes) {
            val liveRole = UniversalSemanticRole.fromRoleString(node.role)
            if (liveRole != targetSig.role) continue

            val liveLabel = node.semanticLabel.trim().lowercase()
            val tgtLabel = targetSig.label.trim().lowercase()
            val liveDesc = (node.contentDescription ?: "").trim().lowercase()
            val tgtDesc = targetSig.contentDescription.trim().lowercase()

            val labelMatched = (tgtLabel.isNotEmpty() && liveLabel == tgtLabel) ||
                    (tgtDesc.isNotEmpty() && liveDesc == tgtDesc) ||
                    (tgtLabel.isEmpty() && liveLabel.isEmpty())

            if (!labelMatched) continue

            val liveCx = node.bounds.centerX.toFloat()
            val liveCy = node.bounds.centerY.toFloat()
            val dx = tgtCx - liveCx
            val dy = tgtCy - liveCy
            val drift = sqrt(dx * dx + dy * dy)

            if (drift < minDrift) {
                minDrift = drift
                bestMatchNode = node
            }
        }

        if (bestMatchNode != null && minDrift <= maxBoundsDriftUnits) {
            if (!bestMatchNode.isEnabled) {
                return TargetFreshnessResult(
                    isFresh = false,
                    failureReason = "DISABLED_CONTROL",
                    remappedNodeIndex = bestMatchNode.index,
                    confidence = 0.0f
                )
            }
            val conf = (1.0f - (minDrift / maxBoundsDriftUnits) * 0.5f).coerceIn(0.5f, 1.0f)
            return TargetFreshnessResult(
                isFresh = true,
                remappedNodeIndex = bestMatchNode.index,
                confidence = conf,
                boundsDrift = minDrift
            )
        }

        // Target disappeared or shifted beyond tolerance
        return TargetFreshnessResult(
            isFresh = false,
            failureReason = "STALE_TARGET",
            remappedNodeIndex = null,
            confidence = 0.0f,
            boundsDrift = if (minDrift != Float.MAX_VALUE) minDrift else 1000.0f
        )
    }

    private fun buildCandidateSignature(node: DistilledNode): CandidateSignature {
        val role = UniversalSemanticRole.fromRoleString(node.role)
        return CandidateSignature(
            role = role,
            label = node.text ?: "",
            contentDescription = node.contentDescription ?: "",
            normalizedBounds = node.bounds,
            isClickable = node.isClickable,
            isEditable = node.isEditable,
            isScrollable = node.isScrollable,
            isEnabled = node.isEnabled,
            isFocused = node.isFocused,
            isChecked = node.isChecked
        )
    }
}
