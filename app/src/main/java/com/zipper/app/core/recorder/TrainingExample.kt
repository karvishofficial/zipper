package com.zipper.app.core.recorder

import com.zipper.app.core.agent.AgentAction
import com.zipper.app.core.model.DistilledUiState

/**
 * Compact record of prior actions and their observable outcomes in the demonstration.
 */
data class HistoryStep(
    val stepIndex: Int,
    val actionSummary: String,
    val outcomeSummary: String,
    val isStateChanged: Boolean
)

/**
 * A normalized, machine-readable training example for the Karvish Action Model.
 *
 * Directly maps the conceptual training formula:
 * Goal + Current Observation + Available Actions + Short History + Previous Outcome -> Next Action
 */
data class TrainingExample(
    val exampleId: String,
    val goal: String,
    val packageName: String?,
    val currentObservation: DistilledUiState,
    val availableActions: List<ActionToolDescriptor>,
    val shortHistory: List<HistoryStep>,
    val previousOutcome: String?,
    val targetAction: AgentAction,
    val targetActionName: String,
    val targetActionArguments: Map<String, String>
)
