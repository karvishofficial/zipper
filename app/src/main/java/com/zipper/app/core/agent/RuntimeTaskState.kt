package com.zipper.app.core.agent

/**
 * Universal 9 Generic Task Phases (Senior Step 4 & Requirement 4)
 * Generic capabilities applicable to any mobile task without app-specific hardcoding.
 */
enum class TaskPhase {
    REACH_CONTEXT,    // Navigating to target app / screen
    FIND_TARGET,      // Locating recipient, search affordance, or item
    SELECT_TARGET,    // Tapping verified item row or search result
    FOCUS_INPUT,      // Activating text field / search composer
    ENTER_ARGUMENT,   // Typing query or message payload
    COMMIT,           // Tapping send, search, or enter key
    VERIFY_OUTCOME,   // Verifying observable screen evidence
    RECOVER,          // Backtracking, dismissing modal, or alternate candidate
    DONE              // Task completed with verified evidence
}

/**
 * Observable Action Transition Outcome (Senior Step 2 & Requirement 2)
 * Decided by the transition verifier based on post-state observable changes.
 */
enum class ActionOutcome {
    INITIAL,
    SUCCESS_TRANSITION,
    UNCHANGED_NOOP,
    WRONG_MODAL_OPENED,
    COMPLETE_REJECTED_MISSING_EVIDENCE,
    ENVIRONMENT_ERROR_REJECTED
}

/**
 * Runtime-Owned Task State Container (Senior Step 3 & Requirement 3)
 * The deterministic runtime owns task progress, phase transitions, and evidence tracking.
 */
data class RuntimeTaskState(
    val goal: String,
    var targetEntity: String? = null,
    var argumentText: String? = null,
    var currentPhase: TaskPhase = TaskPhase.REACH_CONTEXT,
    val requiredEvidence: MutableSet<String> = mutableSetOf(),
    val completedEvidence: MutableSet<String> = mutableSetOf(),
    var lastAction: AgentAction? = null,
    var lastOutcome: ActionOutcome = ActionOutcome.INITIAL,
    var consecutiveNoOpCount: Int = 0,
    var attemptCountForCurrentPhase: Int = 0,
    var lastCandidateId: String? = null,
    var hasExecutedSearch: Boolean = false
)
