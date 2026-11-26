package com.zipper.app.core.model

import kotlin.system.measureTimeMillis

/**
 * Deterministic semantic action model used as a reference baseline and offline test engine.
 * Computes semantic similarity over roles, labels, and normalized spatial regions.
 */
class HeuristicActionModel : LocalActionModel {
    override val modelName: String = "Heuristic-Semantic-Matcher-v1"

    override suspend fun predictAction(
        step: FlowStep,
        uiState: DistilledUiState
    ): ActionPrediction {
        var bestIndex: Int? = null
        var bestScore = 0.0f
        var bestExplanation = "No candidates matched"

        val latencyMs = measureTimeMillis {
            val spec = step.targetSpec
            val labels = spec.semanticLabels.map { it.lowercase() }
            val roles = spec.semanticRoles.map { it.uppercase() }

            for (node in uiState.nodes) {
                var score = 0.0f
                val reasons = mutableListOf<String>()

                // 1. Role Match (weight: 0.25)
                if (roles.isNotEmpty()) {
                    if (roles.contains(node.role)) {
                        score += 0.25f
                        reasons.add("role match (+0.25)")
                    }
                } else if (node.isClickable || node.isEditable) {
                    score += 0.15f
                }

                // 2. Exact or Partial Label Match (weight: 0.50)
                val nodeText = node.text?.lowercase() ?: ""
                val nodeDesc = node.contentDescription?.lowercase() ?: ""
                val nodeId = node.resourceId?.lowercase() ?: ""

                var labelScore = 0.0f
                for (targetLabel in labels) {
                    when {
                        nodeText == targetLabel || nodeDesc == targetLabel -> {
                            labelScore = maxOf(labelScore, 0.50f)
                            reasons.add("exact label match '$targetLabel' (+0.50)")
                        }
                        nodeText.contains(targetLabel) || nodeDesc.contains(targetLabel) -> {
                            labelScore = maxOf(labelScore, 0.35f)
                            reasons.add("partial label match '$targetLabel' (+0.35)")
                        }
                        nodeId.contains(targetLabel) -> {
                            labelScore = maxOf(labelScore, 0.25f)
                            reasons.add("resource id match '$targetLabel' (+0.25)")
                        }
                    }
                }
                score += labelScore

                // 3. Fallback Resource ID match (weight: 0.15)
                if (!spec.fallbackResourceId.isNullOrBlank() && nodeId.contains(spec.fallbackResourceId.lowercase())) {
                    score += 0.15f
                    reasons.add("fallback id match (+0.15)")
                }

                // 4. Spatial Region Preference (e.g. FAB at bottom-right) (weight: 0.10)
                if (spec.preferredBoundsRegion != null) {
                    val inRegion = when (spec.preferredBoundsRegion.lowercase()) {
                        "bottom_right" -> node.bounds.centerX >= 600 && node.bounds.centerY >= 600
                        "top_right" -> node.bounds.centerX >= 600 && node.bounds.centerY <= 400
                        "top_left" -> node.bounds.centerX <= 400 && node.bounds.centerY <= 400
                        "bottom_left" -> node.bounds.centerX <= 400 && node.bounds.centerY >= 600
                        "center" -> node.bounds.centerX in 300..700 && node.bounds.centerY in 300..700
                        else -> false
                    }
                    if (inRegion) {
                        score += 0.10f
                        reasons.add("region '${spec.preferredBoundsRegion}' match (+0.10)")
                    }
                }

                if (score > bestScore) {
                    bestScore = score
                    bestIndex = node.index
                    bestExplanation = reasons.joinToString("; ")
                }
            }
        }

        // Determine effective argument (literal or slot variable)
        val argument = step.literalArgument ?: step.argumentSlot

        return ActionPrediction(
            selectedNodeIndex = if (bestScore >= 0.20f) bestIndex else null,
            action = step.action,
            argument = argument,
            confidence = bestScore.coerceIn(0.0f, 1.0f),
            latencyMs = latencyMs,
            explanation = "Score: $bestScore | $bestExplanation",
            modelName = modelName
        )
    }
}
