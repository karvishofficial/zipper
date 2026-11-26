package com.zipper.app.core.orchestrator

import com.zipper.app.core.model.ActionPrediction
import com.zipper.app.core.model.ActionType
import com.zipper.app.core.model.DistilledUiState
import com.zipper.app.core.model.FlowExecutionResult
import com.zipper.app.core.model.FlowStep
import com.zipper.app.core.model.KarvishFlow
import com.zipper.app.core.model.LocalActionModel
import com.zipper.app.core.model.StepExecutionRecord
import com.zipper.app.core.model.VerificationResult
import com.zipper.app.core.model.VerificationStatus
import com.zipper.app.core.verifier.VerificationEngine
import kotlinx.coroutines.delay

/**
 * Interface representing the capability to dispatch native actions and capture live UI states.
 */
interface ActionExecutor {
    suspend fun captureCurrentState(): DistilledUiState
    suspend fun executeAction(nodeIndex: Int?, action: ActionType, argument: String?): Boolean
}

/**
 * Comprehensive 7-point audit diagnostic trace for real-world execution.
 */
data class StepDiagnosticTrace(
    val stepId: String,
    val intent: String,
    // 1. What UI state Karvish receives
    val initialPackage: String?,
    val rawPreNodeCount: Int,
    // 2. What candidates the distiller produces
    val distilledCandidatesCount: Int,
    val topCandidates: List<String>,
    // 3. What action/target Needle predicts
    val predictedTargetIndex: Int?,
    val predictedAction: ActionType,
    val predictionConfidence: Float,
    val predictionExplanation: String,
    val decodeTps: Double? = null,
    val peakRamMb: Double? = null,
    // Gating Decision
    val gatingDecision: com.zipper.app.core.model.GatingDecision,
    val gatingReason: String,
    // 4. What action is executed
    val actionDispatched: Boolean,
    val targetDescription: String,
    // 5. What UI state appears afterward
    val postPackage: String?,
    val rawPostNodeCount: Int,
    // 6. Whether verification passes or fails
    val verificationStatus: VerificationStatus,
    val verificationMessage: String,
    // 7. Why it failed if it fails
    val failureReason: String? = null
)

/**
 * Karvish Orchestrator: Central coordinator executing the closed neuro-symbolic loop:
 * FlowStep -> Distill -> Action Model -> Execute -> Verify -> Memory/Repair
 */
class KarvishOrchestrator(
    private val actionModel: LocalActionModel,
    private val verifier: VerificationEngine = VerificationEngine(),
    private val executor: ActionExecutor,
    val confidenceThreshold: Float = 0.30f
) {
    /**
     * Executes a full procedural flow step by step with closed-loop verification and self-repair.
     */
    suspend fun executeFlow(flow: KarvishFlow): FlowExecutionResult {
        val startTime = System.currentTimeMillis()
        val records = mutableListOf<StepExecutionRecord>()

        for (step in flow.steps) {
            val record = executeStepWithRepair(step, flow.parameters)
            records.add(record)

            if (record.verificationResult.status != VerificationStatus.VERIFIED_SUCCESS) {
                return FlowExecutionResult(
                    flowId = flow.flowId,
                    isSuccessful = false,
                    records = records,
                    totalLatencyMs = System.currentTimeMillis() - startTime,
                    failureReason = "Step '${step.stepId}' (${step.intent}) failed verification: ${record.verificationResult.message}"
                )
            }
        }

        return FlowExecutionResult(
            flowId = flow.flowId,
            isSuccessful = true,
            records = records,
            totalLatencyMs = System.currentTimeMillis() - startTime
        )
    }

    /**
     * Executes a step while recording an explicit 7-point audit diagnostic trace.
     */
    suspend fun executeStepWithDiagnostics(
        step: FlowStep,
        flowParameters: Map<String, String> = emptyMap()
    ): Pair<StepExecutionRecord, StepDiagnosticTrace> {
        val resolvedArg = step.literalArgument ?: step.argumentSlot?.let { flowParameters[it] }

        // 1. Capture Pre-Action UI State
        val preState = executor.captureCurrentState()

        // 2. Distilled Candidates
        val candidateSummaries = preState.nodes.map { node ->
            "[#${node.index}] ${node.role}: '${node.semanticLabel}' @ bounds(${node.bounds.left},${node.bounds.top},${node.bounds.right},${node.bounds.bottom})"
        }

        // 3. Action Model Prediction (Needle 2)
        val prediction = actionModel.predictAction(step, preState)
        val selectedNode = prediction.selectedNodeIndex?.let { idx ->
            preState.nodes.getOrNull(idx)
        }
        val targetDesc = selectedNode?.let { "[#${it.index}] ${it.role}: '${it.semanticLabel}'" } ?: "None"

        // Evaluate Safety Gating Decision: Protect device from low-confidence, ungrounded, or random taps
        val (gatingDecision, gatingReason) = when {
            prediction.isRefusal || prediction.selectedNodeIndex == null -> {
                Pair(com.zipper.app.core.model.GatingDecision.BLOCKED_REFUSAL, "Model refused: no matching candidate tool found for intent '${step.intent}'")
            }
            prediction.confidence < confidenceThreshold -> {
                Pair(com.zipper.app.core.model.GatingDecision.BLOCKED_LOW_CONFIDENCE, "Model confidence ${"%.4f".format(prediction.confidence)} is below safety threshold ${"%.2f".format(confidenceThreshold)}")
            }
            selectedNode == null -> {
                Pair(com.zipper.app.core.model.GatingDecision.BLOCKED_UNGROUNDED, "Selected candidate #${prediction.selectedNodeIndex} is not grounded in visible screen hierarchy")
            }
            else -> {
                Pair(com.zipper.app.core.model.GatingDecision.DISPATCH_ACTION, "Confidence ${"%.4f".format(prediction.confidence)} >= threshold ${"%.2f".format(confidenceThreshold)}. Approved for native dispatch.")
            }
        }

        val executed: Boolean
        val postState: DistilledUiState
        val verification: VerificationResult
        val failureReason: String?

        // 4. Action Dispatch - ONLY dispatched if gating passes
        if (gatingDecision == com.zipper.app.core.model.GatingDecision.DISPATCH_ACTION) {
            executed = executor.executeAction(prediction.selectedNodeIndex, prediction.action, resolvedArg)
            delay(600) // Allow UI transition and animations to complete
            postState = executor.captureCurrentState()
            verification = verifier.verify(step.postcondition, preState, postState, resolvedArg)
            failureReason = if (verification.status != VerificationStatus.VERIFIED_SUCCESS) {
                "Verification failed: ${verification.message}. Post-action package: '${postState.packageName}', visible elements count: ${postState.nodes.size}"
            } else null
        } else {
            // GATED: Zero screen taps! Protect device state and escalate
            executed = false
            postState = preState
            verification = VerificationResult(
                status = VerificationStatus.CONFIRMED_FAILURE,
                message = "Action dispatch blocked by safety gate: $gatingReason"
            )
            failureReason = "SAFETY GATE REFUSAL: $gatingReason. Escalated without dispatching action."
        }

        val trace = StepDiagnosticTrace(
            stepId = step.stepId,
            intent = step.intent,
            initialPackage = preState.packageName,
            rawPreNodeCount = preState.rawNodeCount,
            distilledCandidatesCount = preState.nodes.size,
            topCandidates = candidateSummaries.take(15),
            predictedTargetIndex = prediction.selectedNodeIndex,
            predictedAction = prediction.action,
            predictionConfidence = prediction.confidence,
            predictionExplanation = prediction.explanation ?: "",
            decodeTps = prediction.decodeTps,
            peakRamMb = prediction.peakRamMb,
            gatingDecision = gatingDecision,
            gatingReason = gatingReason,
            actionDispatched = executed,
            targetDescription = targetDesc,
            postPackage = postState.packageName,
            rawPostNodeCount = postState.rawNodeCount,
            verificationStatus = verification.status,
            verificationMessage = verification.message,
            failureReason = failureReason
        )

        val record = StepExecutionRecord(
            stepId = step.stepId,
            intent = step.intent,
            prediction = prediction,
            verificationResult = verification,
            gatingDecision = gatingDecision,
            wasRepaired = false,
            repairAttempts = 0
        )

        return Pair(record, trace)
    }

    /**
     * Executes a single step. If drift is detected or verification fails, attempts local repair.
     */
    suspend fun executeStepWithRepair(
        step: FlowStep,
        flowParameters: Map<String, String>,
        maxRepairAttempts: Int = 1
    ): StepExecutionRecord {
        val resolvedArg = step.literalArgument
            ?: step.argumentSlot?.let { flowParameters[it] }

        val preState = executor.captureCurrentState()
        var prediction = actionModel.predictAction(step, preState)

        // Initial execution attempt
        val executed = executor.executeAction(prediction.selectedNodeIndex, prediction.action, resolvedArg)
        delay(400)

        var postState = executor.captureCurrentState()
        var verification = verifier.verify(step.postcondition, preState, postState, resolvedArg)

        // If verified, return immediately
        if (verification.status == VerificationStatus.VERIFIED_SUCCESS && executed) {
            return StepExecutionRecord(
                stepId = step.stepId,
                intent = step.intent,
                prediction = prediction,
                verificationResult = verification,
                wasRepaired = false,
                repairAttempts = 0
            )
        }

        // UI Drift / Failure detected -> Model-Driven Self-Repair
        var repairAttempts = 0
        var repairedSuccess = false

        while (repairAttempts < maxRepairAttempts && !repairedSuccess) {
            repairAttempts++

            val repairPreState = executor.captureCurrentState()
            val candidatePrediction = actionModel.predictAction(step, repairPreState)

            // Strictly model-driven: NO programmatic search, NO alternative node substitution
            if (candidatePrediction.selectedNodeIndex == null || candidatePrediction.isRefusal) {
                prediction = candidatePrediction
                break
            }

            executor.executeAction(candidatePrediction.selectedNodeIndex, candidatePrediction.action, resolvedArg)
            delay(500)

            val repairPostState = executor.captureCurrentState()
            val repairVerification = verifier.verify(step.postcondition, repairPreState, repairPostState, resolvedArg)

            prediction = candidatePrediction
            verification = repairVerification

            if (repairVerification.status == VerificationStatus.VERIFIED_SUCCESS) {
                repairedSuccess = true
            }
        }

        return StepExecutionRecord(
            stepId = step.stepId,
            intent = step.intent,
            prediction = prediction,
            verificationResult = verification,
            wasRepaired = repairedSuccess,
            repairAttempts = repairAttempts
        )
    }
}
