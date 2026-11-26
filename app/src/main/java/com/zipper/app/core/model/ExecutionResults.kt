package com.zipper.app.core.model

/**
 * Result of an action prediction by the Local Action Model.
 */
data class ActionPrediction(
    val selectedNodeIndex: Int?,
    val action: ActionType,
    val argument: String? = null,
    val confidence: Float,
    val latencyMs: Long,
    val explanation: String? = null,
    val modelName: String,
    val decodeTps: Double? = null,
    val peakRamMb: Double? = null,
    val isRefusal: Boolean = false
)

/**
 * Explicit Gating Decision evaluating model confidence and grounding prior to action dispatch.
 */
enum class GatingDecision {
    DISPATCH_ACTION,
    BLOCKED_LOW_CONFIDENCE,
    BLOCKED_REFUSAL,
    BLOCKED_UNGROUNDED
}

/**
 * Tri-state verification status ensuring zero silent failures.
 */
enum class VerificationStatus {
    VERIFIED_SUCCESS,
    CONFIRMED_FAILURE,
    UNCERTAIN
}

/**
 * Verification evaluation outcome against a declarative postcondition.
 */
data class VerificationResult(
    val status: VerificationStatus,
    val message: String,
    val matchedNodeIndex: Int? = null,
    val latencyMs: Long = 0L
)

/**
 * Audit record of an individual step execution.
 */
data class StepExecutionRecord(
    val stepId: String,
    val intent: String,
    val prediction: ActionPrediction,
    val verificationResult: VerificationResult,
    val gatingDecision: GatingDecision = GatingDecision.DISPATCH_ACTION,
    val wasRepaired: Boolean = false,
    val repairAttempts: Int = 0
)

/**
 * Aggregated execution report for a complete procedural flow.
 */
data class FlowExecutionResult(
    val flowId: String,
    val isSuccessful: Boolean,
    val records: List<StepExecutionRecord> = emptyList(),
    val totalLatencyMs: Long = 0L,
    val failureReason: String? = null
)
