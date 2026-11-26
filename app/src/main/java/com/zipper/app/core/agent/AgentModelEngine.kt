package com.zipper.app.core.agent

import com.zipper.app.core.model.DistilledUiState
import com.zipper.app.core.needle.AgentPrediction
import org.json.JSONArray

data class AgentInferenceContext(
    val uiState: DistilledUiState,
    val history: List<AgentStepRecord> = emptyList(),
    val previousFailure: String? = null,
    val currentPhase: TaskPhase = TaskPhase.REACH_CONTEXT,
    val lastOutcome: ActionOutcome = ActionOutcome.INITIAL
)

/**
 * Model-agnostic interface for agent action inference.
 *
 * Adheres to AGENTS.md Constitution Rule 7:
 * "The architecture must keep the model interface replaceable so that
 * a future Karvish Action Model can replace Needle 2."
 */
fun interface AgentModelEngine {
    suspend fun runAgentInference(
        prompt: String,
        toolsArray: JSONArray
    ): AgentPrediction

    suspend fun runAgentInference(
        prompt: String,
        toolsArray: JSONArray,
        context: AgentInferenceContext
    ): AgentPrediction = runAgentInference(prompt, toolsArray)
}
