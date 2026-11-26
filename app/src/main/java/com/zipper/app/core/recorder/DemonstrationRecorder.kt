package com.zipper.app.core.recorder

import android.util.Log
import com.zipper.app.core.agent.AgentAction
import com.zipper.app.core.agent.AgentRuntime
import com.zipper.app.core.agent.StateDifferentialEngine
import com.zipper.app.core.agent.StepOutcome
import com.zipper.app.core.agent.ToolSchemaGenerator
import com.zipper.app.core.model.DistilledUiState
import java.util.UUID

/**
 * Orchestrates real-time demonstration recording of human interactions on Android.
 *
 * Implements Phase 5A requirements:
 * - Records what the human actually did (zero synthetic predetermination)
 * - Captures pre-action live observation, dynamic available tools, exact human action,
 *   post-action observation, state transition differential, and success/recovery tags.
 * - Validates trajectories before export to JSONL.
 */
class DemonstrationRecorder(
    private val stateCaptureProvider: (suspend () -> DistilledUiState)? = null,
    private val toolSchemaGenerator: ToolSchemaGenerator = ToolSchemaGenerator(),
    private val differentialEngine: StateDifferentialEngine = StateDifferentialEngine(),
    private val validator: DemonstrationValidator = DemonstrationValidator()
) {

    companion object {
        private const val TAG = "KarvishDemoRecorder"
    }

    private var activeGoal: String? = null
    private var activeMetadata: DemonstrationMetadata? = null
    private var currentObservation: DistilledUiState? = null
    private val recordedSteps = mutableListOf<RecordedStep>()
    private var isRecording: Boolean = false

    val isSessionActive: Boolean get() = isRecording
    val recordedStepsCount: Int get() = recordedSteps.size

    /**
     * Initiates a new human demonstration recording session for the specified user goal.
     */
    suspend fun startRecording(
        goal: String,
        initialUiState: DistilledUiState? = null,
        targetAppPackage: String? = null,
        deviceModel: String = "Android Device"
    ) {
        val cleanGoal = goal.trim()
        require(cleanGoal.isNotBlank()) { "Goal must not be blank" }

        recordedSteps.clear()
        activeGoal = cleanGoal
        activeMetadata = DemonstrationMetadata(
            demonstrationId = UUID.randomUUID().toString(),
            schemaVersion = DemonstrationValidator.SUPPORTED_SCHEMA_VERSION,
            createdAt = System.currentTimeMillis(),
            deviceModel = deviceModel,
            targetAppPackage = targetAppPackage
        )

        val preState = initialUiState ?: stateCaptureProvider?.invoke() ?: DistilledUiState()
        currentObservation = preState
        isRecording = true

        Log.i(TAG, "Demonstration session started. Goal: '$cleanGoal', ID: ${activeMetadata?.demonstrationId}")
    }

    /**
     * Records an individual interaction step performed by the human.
     */
    suspend fun recordStep(
        humanAction: AgentAction,
        actionArguments: Map<String, String> = emptyMap(),
        postUiState: DistilledUiState? = null,
        failureRecoveryInfo: String? = null
    ): RecordedStep {
        check(isRecording) { "Cannot record step: No active demonstration session" }
        val goal = activeGoal ?: throw IllegalStateException("Active goal is null")
        val preState = currentObservation ?: throw IllegalStateException("Current observation is null")

        // 1. Generate available tools at this exact observation
        val generated = toolSchemaGenerator.generateTools(uiState = preState, goal = goal)
        val toolDescriptors = mutableListOf<ActionToolDescriptor>()
        for (i in 0 until generated.toolsJsonArray.length()) {
            val t = generated.toolsJsonArray.getJSONObject(i)
            toolDescriptors.add(
                ActionToolDescriptor(
                    name = t.getString("name"),
                    description = t.getString("description"),
                    parametersJson = t.optJSONObject("parameters")?.toString() ?: "{}"
                )
            )
        }

        // 2. Capture post-action UI observation
        val postState = postUiState ?: stateCaptureProvider?.invoke() ?: preState

        // 3. Evaluate state differential
        val delta = differentialEngine.computeDelta(preState, postState, humanAction)
        val isStateChanged = !AgentRuntime.isStateEffectivelyIdentical(preState, postState)
        val isSuccessful = failureRecoveryInfo == null && delta.outcome != StepOutcome.ERROR_DETECTED

        val stepIndex = recordedSteps.size + 1
        val recorded = RecordedStep(
            stepIndex = stepIndex,
            timestamp = System.currentTimeMillis(),
            goal = goal,
            packageName = preState.packageName,
            activityName = preState.activityName,
            preObservation = preState,
            availableTools = toolDescriptors,
            humanAction = humanAction,
            actionArguments = actionArguments,
            postObservation = postState,
            isStateChanged = isStateChanged,
            isSuccessful = isSuccessful,
            deltaSummary = delta.summary,
            failureRecoveryInfo = failureRecoveryInfo
        )

        recordedSteps.add(recorded)
        currentObservation = postState

        Log.i(TAG, "Step $stepIndex recorded: ${humanAction.javaClass.simpleName}, changed=$isStateChanged, success=$isSuccessful")
        return recorded
    }

    /**
     * Finalizes the active demonstration session and returns the verified trajectory.
     */
    fun finishDemonstration(
        isSuccess: Boolean = true,
        finalOutcomeSummary: String = "Goal successfully demonstrated by human"
    ): DemonstrationTrajectory {
        check(isRecording) { "Cannot finish demonstration: No active session" }
        val goal = activeGoal ?: throw IllegalStateException("Active goal is null")
        val metadata = activeMetadata ?: throw IllegalStateException("Metadata is null")

        val trajectory = DemonstrationTrajectory(
            metadata = metadata,
            goal = goal,
            totalSteps = recordedSteps.size,
            isSuccess = isSuccess,
            finalOutcome = finalOutcomeSummary,
            steps = recordedSteps.toList()
        )

        // Validate trajectory before returning
        val validation = validator.validate(trajectory)
        if (!validation.isValid) {
            val errorMsg = "Demonstration validation failed: ${validation.errors.joinToString("; ")}"
            Log.e(TAG, errorMsg)
            throw IllegalStateException(errorMsg)
        }

        isRecording = false
        activeGoal = null
        activeMetadata = null
        currentObservation = null

        Log.i(TAG, "Demonstration session finished successfully with ${trajectory.totalSteps} steps.")
        return trajectory
    }

    /**
     * Cancels the active demonstration session without saving.
     */
    fun cancelRecording() {
        isRecording = false
        activeGoal = null
        activeMetadata = null
        currentObservation = null
        recordedSteps.clear()
        Log.w(TAG, "Demonstration session cancelled.")
    }
}
