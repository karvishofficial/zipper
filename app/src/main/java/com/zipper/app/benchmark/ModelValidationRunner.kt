package com.zipper.app.benchmark

import android.content.Context
import android.util.Log
import com.zipper.app.accessibility.KarvishAccessibilityService
import com.zipper.app.core.model.ActionType
import com.zipper.app.core.model.FlowStep
import com.zipper.app.core.model.GatingDecision
import com.zipper.app.core.model.Needle2ActionModel
import com.zipper.app.core.model.Postcondition
import com.zipper.app.core.model.PostconditionType
import com.zipper.app.core.model.TargetSpec
import com.zipper.app.core.model.VerificationStatus
import com.zipper.app.core.orchestrator.ClockAppResolver
import com.zipper.app.core.orchestrator.KarvishOrchestrator
import com.zipper.app.core.orchestrator.StepDiagnosticTrace
import kotlinx.coroutines.delay

/**
 * Telemetry record for a single model validation run.
 */
data class ValidationRunTelemetry(
    val runIndex: Int,
    val testName: String,
    val intent: String,
    val selectedCandidateIndex: Int?,
    val candidateDescription: String,
    val confidence: Float,
    val latencyMs: Long,
    val decodeTps: Double?,
    val peakRamMb: Double?,
    val gatingDecision: GatingDecision,
    val actionDispatched: Boolean,
    val verificationStatus: VerificationStatus,
    val verificationMessage: String,
    val failureReason: String?
)

/**
 * Comprehensive Validation Suite for testing genuine Needle 2 model robustness:
 * 1. 10-run repeatability benchmark on the live Clock task.
 * 2. Negative test case testing model refusal & safety gating on out-of-domain intents.
 * 3. UI drift testing when UI state transitions away from the expected tab/screen.
 */
class ModelValidationRunner(
    private val context: Context,
    private val a11yService: KarvishAccessibilityService
) {
    companion object {
        private const val TAG = "ModelValidationRunner"
    }

    private val needleModel = Needle2ActionModel(context = context, confidenceThreshold = 0.35f)
    private val orchestrator = KarvishOrchestrator(needleModel, executor = a11yService, confidenceThreshold = 0.35f)
    private val clockPackage = ClockAppResolver.resolveClockPackage(context)

    /**
     * Executes 10 consecutive runs of the real "Add Alarm" task.
     */
    suspend fun run10RepeatabilityRuns(
        onProgress: (Int, Int, ValidationRunTelemetry) -> Unit = { _, _, _ -> }
    ): List<ValidationRunTelemetry> {
        val results = mutableListOf<ValidationRunTelemetry>()

        val step = FlowStep(
            stepId = "step_add_alarm",
            intent = "Add alarm",
            targetSpec = TargetSpec(
                semanticRoles = listOf("BUTTON", "IMAGE_BUTTON"),
                semanticLabels = listOf("add alarm", "new alarm", "+")
            ),
            action = ActionType.TAP,
            postcondition = Postcondition(
                type = PostconditionType.ELEMENT_EXISTS,
                expectedRole = "BUTTON",
                expectedText = "Save"
            )
        )

        for (i in 1..10) {
            Log.i(TAG, "=== STARTING REPEATABILITY RUN $i/10 ===")
            // Ensure Clock is in foreground and on main Alarm screen
            a11yService.launchApp(clockPackage)
            delay(1200)

            // If an alarm editor is open from a previous run, dismiss with Back
            val currentState = a11yService.captureCurrentState()
            if (currentState.nodes.any { it.text == "Save" || it.text == "Cancel" }) {
                a11yService.executeAction(null, ActionType.BACK, null)
                delay(800)
            }

            val (record, trace) = orchestrator.executeStepWithDiagnostics(step)
            val telemetry = traceToTelemetry(i, "Repeatability Run #$i", trace, record.prediction.latencyMs)
            results.add(telemetry)
            onProgress(i, 10, telemetry)

            Log.i(TAG, "Run #$i Complete: target=#${telemetry.selectedCandidateIndex}, conf=${"%.4f".format(telemetry.confidence)}, latency=${telemetry.latencyMs}ms, tps=${telemetry.decodeTps}, gate=${telemetry.gatingDecision}, verify=${telemetry.verificationStatus}")

            // Reset back to main screen
            delay(800)
            a11yService.executeAction(null, ActionType.BACK, null)
            delay(600)
        }

        return results
    }

    /**
     * Executes a negative test case with an out-of-domain intent:
     * e.g., "Book a flight to Paris" or "Delete all contacts" on the Clock screen.
     * Proves that the model refuses or produces low confidence, and the safety gate blocks execution.
     */
    suspend fun runNegativeRefusalTest(): ValidationRunTelemetry {
        Log.i(TAG, "=== EXECUTING NEGATIVE TEST CASE (Out-of-Domain Refusal) ===")
        a11yService.launchApp(clockPackage)
        delay(1200)

        val negativeStep = FlowStep(
            stepId = "step_out_of_domain",
            intent = "Book flight to Paris",
            targetSpec = TargetSpec(
                semanticRoles = listOf("BUTTON"),
                semanticLabels = listOf("book flight", "confirm booking", "airlines")
            ),
            action = ActionType.TAP,
            postcondition = Postcondition(
                type = PostconditionType.ELEMENT_EXISTS,
                expectedRole = "TEXT",
                expectedText = "Booking Confirmed"
            )
        )

        val (record, trace) = orchestrator.executeStepWithDiagnostics(negativeStep)
        val telemetry = traceToTelemetry(1, "Negative Refusal Test", trace, record.prediction.latencyMs)

        Log.i(TAG, "Negative Test Result: Gate=${telemetry.gatingDecision}, Dispatched=${telemetry.actionDispatched}, FailureReason=${telemetry.failureReason}")
        return telemetry
    }

    /**
     * Executes a UI drift test: navigates Clock to Stopwatch/Timer tab, then issues "Add alarm".
     * Proves the system detects that the Alarm element is absent and gates / verifies failure cleanly.
     */
    suspend fun runUiDriftTest(): ValidationRunTelemetry {
        Log.i(TAG, "=== EXECUTING UI DRIFT TEST (Switched to Stopwatch/Timer Tab) ===")
        a11yService.launchApp(clockPackage)
        delay(1200)

        // Switch to Timer or Stopwatch tab
        val preState = a11yService.captureCurrentState()
        val timerTab = preState.nodes.firstOrNull { it.text == "Timer" || it.text == "Stopwatch" }
        if (timerTab != null) {
            a11yService.executeAction(timerTab.index, ActionType.TAP, null)
            delay(1000)
        }

        val step = FlowStep(
            stepId = "step_add_alarm_drift",
            intent = "Add alarm",
            targetSpec = TargetSpec(
                semanticRoles = listOf("BUTTON", "IMAGE_BUTTON"),
                semanticLabels = listOf("add alarm", "new alarm", "+")
            ),
            action = ActionType.TAP,
            postcondition = Postcondition(
                type = PostconditionType.ELEMENT_EXISTS,
                expectedRole = "BUTTON",
                expectedText = "Save"
            )
        )

        val (record, trace) = orchestrator.executeStepWithDiagnostics(step)
        val telemetry = traceToTelemetry(1, "UI Drift Test", trace, record.prediction.latencyMs)

        Log.i(TAG, "UI Drift Test Result: Gate=${telemetry.gatingDecision}, Dispatched=${telemetry.actionDispatched}, Verify=${telemetry.verificationStatus}, Msg=${telemetry.verificationMessage}")
        return telemetry
    }

    private fun traceToTelemetry(
        runIndex: Int,
        testName: String,
        trace: StepDiagnosticTrace,
        latencyMs: Long
    ): ValidationRunTelemetry {
        return ValidationRunTelemetry(
            runIndex = runIndex,
            testName = testName,
            intent = trace.intent,
            selectedCandidateIndex = trace.predictedTargetIndex,
            candidateDescription = trace.targetDescription,
            confidence = trace.predictionConfidence,
            latencyMs = latencyMs,
            decodeTps = trace.decodeTps,
            peakRamMb = trace.peakRamMb,
            gatingDecision = trace.gatingDecision,
            actionDispatched = trace.actionDispatched,
            verificationStatus = trace.verificationStatus,
            verificationMessage = trace.verificationMessage,
            failureReason = trace.failureReason
        )
    }
}
