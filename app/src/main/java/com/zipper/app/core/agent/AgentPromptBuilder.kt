package com.zipper.app.core.agent

import com.zipper.app.core.model.DistilledUiState

/**
 * Builds grounded prompts for Cactus Needle 2 in compliance with AGENTS.md.
 *
 * Supplies the exact, original user goal without pre-computation, goal slicing,
 * or token decomposition. Supplies failure reflections to guide recovery when actions fail.
 */
class AgentPromptBuilder {

    fun buildPrompt(
        goal: String,
        currentUiState: DistilledUiState,
        stepHistory: List<AgentStepRecord>,
        activeReflection: String? = null,
        blacklistedIndices: Set<Int> = emptySet()
    ): String {
        val trimmedGoal = goal.trim()
        val formulaNode = currentUiState.nodes.find {
            val res = (it.resourceId ?: "").lowercase()
            res.contains("calc_edt_formula") || res.contains("formula")
        }
        val rawFormula = formulaNode?.text?.replace("Calculator input field", "")?.trim()?.takeIf { it.isNotBlank() && it != "0" }
        val resultNode = currentUiState.nodes.find {
            val res = (it.resourceId ?: "").lowercase()
            res.contains("calc_tv_result") || res.contains("result")
        }
        val resultText = resultNode?.text?.trim()?.takeIf { it.isNotBlank() }

        val stateSuffix = when {
            resultText != null && rawFormula != null -> " [state: $rawFormula=$resultText]"
            rawFormula != null -> " [state: $rawFormula]"
            else -> ""
        }

        if (!activeReflection.isNullOrBlank()) {
            return "$trimmedGoal$stateSuffix (previous action failed: $activeReflection)"
        }
        return "$trimmedGoal$stateSuffix"
    }
}
