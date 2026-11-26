package com.zipper.app.core.agent

import com.zipper.app.core.model.DistilledNode

/**
 * Universal Action Execution Boundary (AGENTS.md Section 4 & 5).
 * Decouples model intent (AgentAction) from physical environment execution:
 * - Android OS via Accessibility APIs (ActionDispatcher)
 * - Web Pages via Hidden WebView DOM events (WebActionExecutor)
 */
interface ActionExecutor {
    suspend fun execute(
        action: AgentAction,
        candidateMap: Map<Int, DistilledNode> = emptyMap(),
        goal: String = "",
        actionCandidateMap: Map<String, ActionCandidate> = emptyMap()
    ): Boolean
}
