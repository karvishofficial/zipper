package com.zipper.app.core.recorder

import com.zipper.app.core.agent.AgentAction
import com.zipper.app.core.model.DistilledUiState
import java.util.UUID

/**
 * Metadata identifying a recorded demonstration session.
 * Includes schema versioning to allow dataset evolution over time.
 */
data class DemonstrationMetadata(
    val demonstrationId: String = UUID.randomUUID().toString(),
    val schemaVersion: String = "1.0.0",
    val createdAt: Long = System.currentTimeMillis(),
    val deviceModel: String = "Android Device",
    val androidApiLevel: Int = 34,
    val targetAppPackage: String? = null
)

/**
 * Representation of an individual action tool available to the agent at a specific turn.
 */
data class ActionToolDescriptor(
    val name: String,
    val description: String,
    val parametersJson: String = "{}"
)

/**
 * Audit record of a single step in a human demonstration.
 * Captures live state before and after the human action, the exact action performed,
 * and whether the step was successful or required recovery.
 */
data class RecordedStep(
    val stepIndex: Int,
    val timestamp: Long = System.currentTimeMillis(),
    val goal: String,
    val packageName: String? = null,
    val activityName: String? = null,
    val preObservation: DistilledUiState,
    val availableTools: List<ActionToolDescriptor> = emptyList(),
    val humanAction: AgentAction,
    val actionArguments: Map<String, String> = emptyMap(),
    val postObservation: DistilledUiState,
    val isStateChanged: Boolean,
    val isSuccessful: Boolean,
    val deltaSummary: String,
    val failureRecoveryInfo: String? = null
)

/**
 * A complete human demonstration trajectory for a specific task goal.
 * Composed of ordered, validated interaction steps from initial state to goal completion.
 */
data class DemonstrationTrajectory(
    val metadata: DemonstrationMetadata,
    val goal: String,
    val totalSteps: Int,
    val isSuccess: Boolean,
    val finalOutcome: String,
    val steps: List<RecordedStep>
)
