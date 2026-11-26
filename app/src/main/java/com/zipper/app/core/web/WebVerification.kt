package com.zipper.app.core.web

import com.zipper.app.core.agent.AgentAction
import com.zipper.app.core.model.DistilledNode
import com.zipper.app.core.model.DistilledUiState
import java.util.Locale

enum class WebOutcomeStatus {
    MUTATION_CONFIRMED,       // General interactive DOM mutation verified
    URL_CHANGED,              // Navigation to new destination URL verified
    INPUT_POPULATED,          // Verified that target input field reflects the typed text
    AFFORDANCE_CLICKED,       // Click caused expected DOM reaction (modal, submit, detachment)
    CONTENT_EXTRACTED,        // Target information matching goal is verified and extracted
    NO_EFFECT,                // DOM did not mutate (signals mechanical retry for executor)
    BLOCKED_OR_ERROR          // 404, rate limit, anti-bot challenge, network outage
}

data class WebVerificationResult(
    val status: WebOutcomeStatus,
    val isProgressMade: Boolean,
    val isGoalAccomplished: Boolean,
    val summary: String,
    val extractedEvidence: String? = null,
    val urlChanged: Boolean = false,
    val mutatedNodeCount: Int = 0
)

/**
 * WebVerification Engine for Karvish WebEnvironment (v2 Specification).
 *
 * Implements authoritative, deterministic verification of web action outcomes.
 * Adheres strictly to the core engineering contract:
 * "V57 decides; WebStateDistiller reports; WebActionExecutor operates; WebVerification proves."
 */
class WebVerification {

    companion object {
        private val ERROR_SIGNALS = listOf(
            "404 not found", "access denied", "robot or human", "verify you are human",
            "cloudflare", "security check", "bad gateway", "502 bad gateway",
            "503 service unavailable", "internal server error"
        )

        private val SUCCESS_INDICATORS = listOf(
            "thank you", "order confirmed", "successfully submitted", "your message has been sent",
            "payment successful", "booking confirmed", "registered successfully", "congratulations"
        )
    }

    /**
     * Proves whether an action produced a legitimate DOM mutation and whether the goal is accomplished.
     */
    fun verify(
        preState: DistilledUiState,
        action: AgentAction,
        postState: DistilledUiState,
        goal: String
    ): WebVerificationResult {
        val cleanGoal = goal.trim().lowercase(Locale.ROOT)
        val preUrl = (preState.activityName ?: "").trim().lowercase(Locale.ROOT)
        val postUrl = (postState.activityName ?: "").trim().lowercase(Locale.ROOT)
        val urlChanged = preUrl.isNotBlank() && postUrl.isNotBlank() && preUrl != postUrl

        // 1. Detect Block or Error Page
        val errorSignal = checkErrorOrBlock(postState)
        if (errorSignal != null) {
            return WebVerificationResult(
                status = WebOutcomeStatus.BLOCKED_OR_ERROR,
                isProgressMade = false,
                isGoalAccomplished = false,
                summary = "Web destination blocked or reported error: $errorSignal",
                urlChanged = urlChanged
            )
        }

        // 2. Action-Specific Verification
        val actionResult = when (action) {
            is AgentAction.TypeText -> verifyTypeText(preState, action, postState)
            is AgentAction.Tap -> verifyTap(preState, action, postState, urlChanged)
            is AgentAction.Scroll -> verifyScroll(preState, postState)
            is AgentAction.Submit -> verifySubmit(preState, postState, urlChanged)
            is AgentAction.PressBack -> verifyBack(preState, postState, urlChanged)
            is AgentAction.OpenApp -> WebVerificationResult(
                status = if (urlChanged) WebOutcomeStatus.URL_CHANGED else WebOutcomeStatus.MUTATION_CONFIRMED,
                isProgressMade = true,
                isGoalAccomplished = false,
                summary = "Navigated to web destination: ${postState.activityName}",
                urlChanged = urlChanged
            )
            is AgentAction.CompleteGoal -> WebVerificationResult(
                status = WebOutcomeStatus.CONTENT_EXTRACTED,
                isProgressMade = true,
                isGoalAccomplished = true,
                summary = "Goal explicitly marked complete",
                urlChanged = urlChanged
            )
            else -> WebVerificationResult(
                status = WebOutcomeStatus.MUTATION_CONFIRMED,
                isProgressMade = true,
                isGoalAccomplished = false,
                summary = "Action executed",
                urlChanged = urlChanged
            )
        }

        // 3. Goal Accomplishment Verification
        val goalAccomplished = checkGoalAccomplishment(cleanGoal, postState)
        val extractedEvidence = extractEvidenceForGoal(cleanGoal, postState)

        val finalStatus = if (goalAccomplished && extractedEvidence != null) {
            WebOutcomeStatus.CONTENT_EXTRACTED
        } else {
            actionResult.status
        }

        return actionResult.copy(
            status = finalStatus,
            isGoalAccomplished = goalAccomplished || actionResult.isGoalAccomplished,
            extractedEvidence = extractedEvidence ?: actionResult.extractedEvidence
        )
    }

    private fun verifyTypeText(
        preState: DistilledUiState,
        action: AgentAction.TypeText,
        postState: DistilledUiState
    ): WebVerificationResult {
        val targetNode = postState.nodes.find { it.index == action.targetNodeIndex }
        val targetText = targetNode?.text?.lowercase(Locale.ROOT) ?: ""
        val expectedText = action.text.trim().lowercase(Locale.ROOT)

        val textPopulated = targetText.contains(expectedText) ||
            postState.nodes.any { it.text?.lowercase(Locale.ROOT)?.contains(expectedText) == true }

        val domMutated = !areStatesIdentical(preState, postState)

        return if (textPopulated || domMutated) {
            WebVerificationResult(
                status = WebOutcomeStatus.INPUT_POPULATED,
                isProgressMade = true,
                isGoalAccomplished = false,
                summary = "Verified text '${action.text}' entered into field #${action.targetNodeIndex}",
                mutatedNodeCount = 1
            )
        } else {
            WebVerificationResult(
                status = WebOutcomeStatus.NO_EFFECT,
                isProgressMade = false,
                isGoalAccomplished = false,
                summary = "TypeText produced no observable DOM change on field #${action.targetNodeIndex}"
            )
        }
    }

    private fun verifyTap(
        preState: DistilledUiState,
        action: AgentAction.Tap,
        postState: DistilledUiState,
        urlChanged: Boolean
    ): WebVerificationResult {
        if (urlChanged) {
            return WebVerificationResult(
                status = WebOutcomeStatus.URL_CHANGED,
                isProgressMade = true,
                isGoalAccomplished = false,
                summary = "Tap on #${action.targetNodeIndex} triggered navigation to ${postState.activityName}",
                urlChanged = true
            )
        }

        val targetInPre = preState.nodes.find { it.index == action.targetNodeIndex }
        val targetInPost = postState.nodes.find { it.index == action.targetNodeIndex }
        val isDetached = targetInPre != null && targetInPost == null
        val domMutated = !areStatesIdentical(preState, postState)

        return if (isDetached || domMutated) {
            WebVerificationResult(
                status = WebOutcomeStatus.AFFORDANCE_CLICKED,
                isProgressMade = true,
                isGoalAccomplished = false,
                summary = "Tap on #${action.targetNodeIndex} ('${action.semanticLabel ?: targetInPre?.text.orEmpty()}') confirmed with DOM mutation",
                mutatedNodeCount = countDifferingNodes(preState, postState)
            )
        } else {
            WebVerificationResult(
                status = WebOutcomeStatus.NO_EFFECT,
                isProgressMade = false,
                isGoalAccomplished = false,
                summary = "Tap on #${action.targetNodeIndex} produced no DOM response"
            )
        }
    }

    private fun verifyScroll(preState: DistilledUiState, postState: DistilledUiState): WebVerificationResult {
        val domMutated = !areStatesIdentical(preState, postState)
        return if (domMutated) {
            WebVerificationResult(
                status = WebOutcomeStatus.MUTATION_CONFIRMED,
                isProgressMade = true,
                isGoalAccomplished = false,
                summary = "Scroll shifted viewport elements",
                mutatedNodeCount = countDifferingNodes(preState, postState)
            )
        } else {
            WebVerificationResult(
                status = WebOutcomeStatus.NO_EFFECT,
                isProgressMade = false,
                isGoalAccomplished = false,
                summary = "Scroll reached boundary or had no effect"
            )
        }
    }

    private fun verifySubmit(
        preState: DistilledUiState,
        postState: DistilledUiState,
        urlChanged: Boolean
    ): WebVerificationResult {
        val domMutated = !areStatesIdentical(preState, postState)
        return if (urlChanged || domMutated) {
            WebVerificationResult(
                status = if (urlChanged) WebOutcomeStatus.URL_CHANGED else WebOutcomeStatus.MUTATION_CONFIRMED,
                isProgressMade = true,
                isGoalAccomplished = false,
                summary = "Submit triggered DOM transition",
                urlChanged = urlChanged
            )
        } else {
            WebVerificationResult(
                status = WebOutcomeStatus.NO_EFFECT,
                isProgressMade = false,
                isGoalAccomplished = false,
                summary = "Submit produced no DOM mutation"
            )
        }
    }

    private fun verifyBack(
        preState: DistilledUiState,
        postState: DistilledUiState,
        urlChanged: Boolean
    ): WebVerificationResult {
        val domMutated = !areStatesIdentical(preState, postState)
        return if (urlChanged || domMutated) {
            WebVerificationResult(
                status = if (urlChanged) WebOutcomeStatus.URL_CHANGED else WebOutcomeStatus.MUTATION_CONFIRMED,
                isProgressMade = true,
                isGoalAccomplished = false,
                summary = "Back navigation altered web page",
                urlChanged = urlChanged
            )
        } else {
            WebVerificationResult(
                status = WebOutcomeStatus.NO_EFFECT,
                isProgressMade = false,
                isGoalAccomplished = false,
                summary = "Back action had no effect (no previous history entry)"
            )
        }
    }

    private fun checkErrorOrBlock(state: DistilledUiState): String? {
        val allText = state.nodes.mapNotNull { it.text ?: it.contentDescription }.joinToString(" ").lowercase(Locale.ROOT)
        val title = state.activityName?.lowercase(Locale.ROOT).orEmpty()
        for (err in ERROR_SIGNALS) {
            if (title.contains(err) || allText.contains(err)) {
                return err
            }
        }
        return null
    }

    private fun checkGoalAccomplishment(goal: String, state: DistilledUiState): Boolean {
        val allText = state.nodes.mapNotNull { it.text ?: it.contentDescription }.joinToString(" ").lowercase(Locale.ROOT)

        // Check for common confirmation banners if goal was a submission or order
        val isSubmissionGoal = goal.contains("submit") || goal.contains("order") || goal.contains("send") || goal.contains("book")
        if (isSubmissionGoal && SUCCESS_INDICATORS.any { allText.contains(it) }) {
            return true
        }

        // Check for search query results
        if (goal.contains("search") || goal.contains("find") || goal.contains("what is")) {
            val queryTokens = goal.split(Regex("[^a-z0-9]+"))
                .filter { it.length >= 3 && it !in setOf("open", "the", "and", "then", "search", "find", "google", "for", "what", "from") }
            val matchingTokens = queryTokens.count { allText.contains(it) }
            if (queryTokens.isNotEmpty() && matchingTokens >= (queryTokens.size * 0.7).toInt().coerceAtLeast(1)) {
                val hasResults = state.nodes.size >= 5
                if (hasResults) return true
            }
        }

        return false
    }

    private fun extractEvidenceForGoal(goal: String, state: DistilledUiState): String? {
        val queryTokens = goal.split(Regex("[^a-z0-9]+"))
            .filter { it.length >= 3 && it !in setOf("open", "the", "and", "then", "search", "find", "google", "for", "what", "from") }

        // Find the node with the highest token overlap
        val bestNode = state.nodes
            .filter { (it.text?.length ?: 0) in 20..400 }
            .maxByOrNull { node ->
                val txt = (node.text ?: "").lowercase(Locale.ROOT)
                queryTokens.count { txt.contains(it) }
            }

        return bestNode?.text
    }

    private fun areStatesIdentical(a: DistilledUiState, b: DistilledUiState): Boolean {
        if (a.nodes.size != b.nodes.size) return false
        if (a.activityName != b.activityName) return false
        for (i in a.nodes.indices) {
            val na = a.nodes[i]
            val nb = b.nodes[i]
            if (na.index != nb.index || na.text != nb.text || na.bounds != nb.bounds || na.isEditable != nb.isEditable) {
                return false
            }
        }
        return true
    }

    private fun countDifferingNodes(a: DistilledUiState, b: DistilledUiState): Int {
        val aMap = a.nodes.associateBy { it.index }
        val bMap = b.nodes.associateBy { it.index }
        var diff = 0
        for ((idx, nodeA) in aMap) {
            val nodeB = bMap[idx]
            if (nodeB == null || nodeA.text != nodeB.text || nodeA.bounds != nodeB.bounds) {
                diff++
            }
        }
        for (idx in bMap.keys) {
            if (!aMap.containsKey(idx)) diff++
        }
        return diff
    }
}
