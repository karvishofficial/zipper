package com.zipper.app.core.model

/**
 * Standard primitive action types supported by the Karvish deterministic runtime.
 */
enum class ActionType {
    TAP,
    TYPE,
    SCROLL_UP,
    SCROLL_DOWN,
    BACK,
    WAIT,
    TERMINATE_SUCCESS,
    SIGNAL_FAILURE
}

/**
 * Semantic criteria for locating an actionable UI target without relying on hardcoded coordinates.
 */
data class TargetSpec(
    val semanticRoles: List<String> = emptyList(),
    val semanticLabels: List<String> = emptyList(),
    val preferredBoundsRegion: String? = null,
    val fallbackResourceId: String? = null
)

/**
 * Supported postcondition assertion types for deterministic verification.
 */
enum class PostconditionType {
    ELEMENT_EXISTS,
    ELEMENT_DISAPPEARED,
    TEXT_CONTAINS,
    FIELD_EQUALS,
    SCREEN_CHANGED
}

/**
 * A declarative structural postcondition that must be satisfied for a step to be deemed successful.
 */
data class Postcondition(
    val type: PostconditionType,
    val expectedRole: String? = null,
    val expectedText: String? = null,
    val textSlot: String? = null,
    val expectedPackage: String? = null,
    val timeoutMs: Long = 2500L
)

/**
 * An individual procedural step within a Karvish Flow.
 */
data class FlowStep(
    val stepId: String,
    val intent: String,
    val targetSpec: TargetSpec,
    val action: ActionType,
    val argumentSlot: String? = null,
    val literalArgument: String? = null,
    val postcondition: Postcondition? = null
)

/**
 * A complete, compiled, versioned procedural flow emitted by the Cloud LLM or Teach Compiler.
 */
data class KarvishFlow(
    val flowId: String,
    val name: String,
    val targetAppPackage: String,
    val version: String = "1.0.0",
    val parameters: Map<String, String> = emptyMap(),
    val steps: List<FlowStep> = emptyList()
)
