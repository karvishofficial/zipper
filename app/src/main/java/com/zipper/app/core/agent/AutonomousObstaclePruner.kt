package com.zipper.app.core.agent

import android.util.Log
import com.zipper.app.core.model.DistilledNode
import com.zipper.app.core.model.DistilledUiState
import java.util.Locale

/**
 * Autonomous Obstacle & Interstitial Pruning Engine.
 *
 * In real-world Android mobile navigation, unexpected modals, promotional banners,
 * notification sheets, and permission prompts constantly derail autonomous agents.
 *
 * This engine detects uninvited UI obstacles in the foreground and autonomously
 * prunes them (dismiss, close, 'not now', 'later', 'skip') without burning model turns
 * or halting the primary goal trajectory.
 *
 * Zero external cloud APIs, 100% on-device deterministic inspection.
 */
object AutonomousObstaclePruner {

    private const val TAG = "ObstaclePruner"

    data class PruneDecision(
        val isObstacleDetected: Boolean,
        val pruneAction: AgentAction?,
        val reason: String,
        val obstacleTitle: String? = null
    )

    // Standard dismissive tokens across Android dialogs and promotional bottom-sheets
    private val DISMISS_LABELS = setOf(
        "✕", "x", "close", "dismiss", "not now", "later", "maybe later",
        "remind me later", "no thanks", "no, thanks", "skip", "skip for now",
        "cancel", "don't allow", "dont allow", "deny", "while using the app",
        "only this time", "continue without", "got it", "i'll do it later",
        "never", "ask me later", "ignore", "not today"
    )

    private val DISMISS_RESOURCE_TOKENS = listOf(
        "close", "dismiss", "cancel", "skip", "btn_close", "close_button",
        "iv_close", "img_close", "negative_button", "button2", "btn_cancel",
        "permission_allow_foreground_only_button", "permission_deny_button"
    )

    private val OBSTACLE_INDICATOR_TOKENS = listOf(
        "special offer", "flat discount", "update available", "rate us",
        "give us 5 stars", "enable notification", "allow location", "permission",
        "enjoying the app", "terms and conditions", "what's new", "festive sale",
        "claim offer", "subscribe to premium", "turn on gps", "device location"
    )

    /**
     * Inspects current UI state and determines if a blocking obstacle should be pruned.
     *
     * @param uiState Current visible distilled screen state
     * @param primaryGoal User's active goal
     * @return PruneDecision containing the dismissal action if an obstacle is confirmed
     */
    fun evaluate(uiState: DistilledUiState, primaryGoal: String): PruneDecision {
        val lowerGoal = primaryGoal.lowercase(Locale.ROOT).trim()

        // If user's explicit goal specifically asks for interacting with the dialog/permission, do not prune
        val isGoalTargetingObstacle = lowerGoal.contains("allow") || lowerGoal.contains("permission") ||
            lowerGoal.contains("discount") || lowerGoal.contains("update") || lowerGoal.contains("rate") ||
            lowerGoal.contains("claim") || lowerGoal.contains("agree")

        if (isGoalTargetingObstacle) {
            return PruneDecision(isObstacleDetected = false, pruneAction = null, reason = "Goal directly targets current screen content")
        }

        val nodes = uiState.nodes
        if (nodes.isEmpty()) {
            return PruneDecision(isObstacleDetected = false, pruneAction = null, reason = "No nodes on screen")
        }

        // 1. Check for promotional, permission, or rating obstacle signatures
        val screenTexts = nodes.mapNotNull { it.text?.takeIf { t -> t.isNotBlank() } ?: it.contentDescription?.takeIf { c -> c.isNotBlank() } }
        val screenCombinedText = screenTexts.joinToString(" ").lowercase(Locale.ROOT)

        val hasObstacleSignature = OBSTACLE_INDICATOR_TOKENS.any { token -> screenCombinedText.contains(token) }

        // 2. Identify candidate dismiss affordance
        var dismissCandidate: DistilledNode? = null
        var dismissReason = ""

        for (node in nodes) {
            if (!node.isClickable && node.role != "BUTTON" && node.role != "IMAGE") continue

            val label = (node.text ?: node.contentDescription ?: "").trim().lowercase(Locale.ROOT)
            val resId = node.resourceId?.lowercase(Locale.ROOT).orEmpty()

            // Exact or prefix match on standard dismiss tokens
            if (label in DISMISS_LABELS || DISMISS_LABELS.any { label == it || label.startsWith("$it ") || label.endsWith(" $it") }) {
                dismissCandidate = node
                dismissReason = "Found explicit dismiss button: '$label'"
                break
            }

            // Resource ID token match
            if (DISMISS_RESOURCE_TOKENS.any { resId.contains(it) }) {
                // Secondary check: ensure it's not a positive submission or main action button
                if (!label.contains("submit") && !label.contains("buy") && !label.contains("order") && !label.contains("pay") && !label.contains("send")) {
                    dismissCandidate = node
                    dismissReason = "Found dismiss button via resource ID: '$resId'"
                    break
                }
            }

            // Standalone single-character 'X' or multiplication symbol in top corners
            if ((label == "x" || label == "✕" || label == "×") && node.bounds.top < 350) {
                dismissCandidate = node
                dismissReason = "Found top-screen close icon: '$label'"
                break
            }
        }

        // 3. Confirm decision
        if (dismissCandidate != null && (hasObstacleSignature || isModalContainerPresent(uiState))) {
            Log.i(TAG, "Autonomous Obstacle Pruner triggered: $dismissReason at node ${dismissCandidate.index}")
            val pruneAction = AgentAction.Tap(
                targetNodeIndex = dismissCandidate.index,
                candidateId = "prune_obstacle_${dismissCandidate.index}",
                semanticLabel = "Dismiss Obstacle"
            )
            return PruneDecision(
                isObstacleDetected = true,
                pruneAction = pruneAction,
                reason = dismissReason,
                obstacleTitle = screenTexts.firstOrNull()
            )
        }

        // If screen is a known Android system permission dialog without a direct 'close' button, allow foreground use
        val permissionAllowNode = nodes.find {
            val l = (it.text ?: it.contentDescription ?: "").lowercase(Locale.ROOT)
            l.contains("while using the app") || l.contains("only this time")
        }
        if (permissionAllowNode != null && screenCombinedText.contains("permission")) {
            val pruneAction = AgentAction.Tap(
                targetNodeIndex = permissionAllowNode.index,
                candidateId = "prune_permission_${permissionAllowNode.index}",
                semanticLabel = "Grant Foreground Permission"
            )
            return PruneDecision(
                isObstacleDetected = true,
                pruneAction = pruneAction,
                reason = "Autonomous permission bypass: 'While using the app'",
                obstacleTitle = "System Permission Dialog"
            )
        }

        return PruneDecision(isObstacleDetected = false, pruneAction = null, reason = "No uninvited obstacle detected")
    }

    private fun isModalContainerPresent(uiState: DistilledUiState): Boolean {
        return uiState.nodes.any {
            val cls = it.className?.lowercase(Locale.ROOT).orEmpty()
            val res = it.resourceId?.lowercase(Locale.ROOT).orEmpty()
            cls.contains("dialog") || cls.contains("bottomsheet") || res.contains("dialog") || res.contains("modal") || res.contains("sheet")
        }
    }
}
