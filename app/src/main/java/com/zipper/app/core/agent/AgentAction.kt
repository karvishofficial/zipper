package com.zipper.app.core.agent

/**
 * Generic semantic action primitives executed by the Karvish V0.2 Agent Runtime.
 *
 * These primitives decouple agent intent from OEM-specific or screen-specific
 * coordinate hacks.
 */
sealed class AgentAction {
    /**
     * Launches an installed Android application by its common name
     * (e.g. "Calculator", "Clock", "Settings", "Camera", "Messages").
     */
    data class OpenApp(val appName: String) : AgentAction()

    /**
     * Taps a grounded UI element.
     * The candidate the model ranks must be the exact node the executor executes.
     */
    data class Tap(
        val targetNodeIndex: Int,
        val candidateId: String? = null,
        val semanticLabel: String? = null,
        val nodeIdentity: String? = null
    ) : AgentAction()

    /**
     * Types text into an editable element.
     * The runtime will attempt AccessibilityNodeInfo.ACTION_SET_TEXT.
     */
    data class TypeText(
        val targetNodeIndex: Int,
        val text: String,
        val candidateId: String? = null,
        val semanticLabel: String? = null,
        val nodeIdentity: String? = null
    ) : AgentAction()

    /**
     * Scrolls a container (or whole screen) in a given direction.
     * The runtime attempts ACTION_SCROLL_FORWARD/BACKWARD first,
     * falling back to a directional swipe gesture.
     */
    data class Scroll(
        val targetNodeIndex: Int? = null,
        val direction: ScrollDirection = ScrollDirection.DOWN,
        val candidateId: String? = null,
        val semanticLabel: String? = null,
        val nodeIdentity: String? = null
    ) : AgentAction()

    /**
     * Triggers the Android system back action.
     */
    object PressBack : AgentAction()

    /** Submits the currently focused editable field through the Android IME action. */
    object Submit : AgentAction()

    /**
     * Triggers the Android system home action.
     */
    object PressHome : AgentAction()

    /**
     * Pauses for UI animations, asynchronous loading, or dialog transitions.
     */
    data class Wait(val durationMs: Long = 600L) : AgentAction()

    /**
     * The agent declares the user's high-level goal is successfully completed.
     */
    data class CompleteGoal(val summary: String) : AgentAction()

    /**
     * The agent determines the goal cannot be satisfied (e.g. element missing, permission error).
     */
    data class FailGoal(val reason: String) : AgentAction()
}

enum class ScrollDirection {
    UP,
    DOWN,
    LEFT,
    RIGHT
}

enum class StepOutcome {
    SUCCESS,
    SUCCESS_PROGRESS,
    NO_OP,
    WRONG_SCREEN,
    BLOCKED_BY_POPUP,
    ERROR_DETECTED,
    SAFETY_BLOCKED,
    GOAL_COMPLETED,
    GOAL_FAILED,
    COMPLETE_VERIFIED,
    COMPLETE_REJECTED,
    STUCK_NO_PROGRESS,
    REPEATED_REFUSAL
}

/**
 * Audit record of a single turn in the SDAVR agent execution loop.
 */
data class AgentStepRecord(
    val turn: Int,
    val action: AgentAction,
    val actionDescription: String,
    val preScreenSummary: String,
    val postScreenSummary: String,
    val outcome: StepOutcome,
    val reflection: String,
    val confidence: Float,
    val latencyMs: Long,
    val stateDelta: StateDelta? = null,
    val packageName: String? = null
)
