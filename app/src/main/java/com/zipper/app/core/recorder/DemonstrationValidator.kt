package com.zipper.app.core.recorder

import com.zipper.app.core.agent.AgentAction

/**
 * Result of validating a demonstration trajectory against dataset integrity constraints.
 */
data class ValidationResult(
    val isValid: Boolean,
    val errors: List<String>
)

/**
 * Enforces dataset integrity to prevent malformed or ungrounded demonstrations
 * from entering the training dataset for the Karvish Action Model.
 */
class DemonstrationValidator {

    companion object {
        const val SUPPORTED_SCHEMA_VERSION = "1.0.0"
    }

    fun validate(trajectory: DemonstrationTrajectory): ValidationResult {
        val errors = mutableListOf<String>()

        if (trajectory.goal.isBlank()) {
            errors.add("Demonstration goal must not be blank")
        }

        if (trajectory.metadata.demonstrationId.isBlank()) {
            errors.add("Demonstration ID must not be blank")
        }

        if (trajectory.metadata.schemaVersion != SUPPORTED_SCHEMA_VERSION) {
            errors.add("Unsupported schema version '${trajectory.metadata.schemaVersion}'. Expected '$SUPPORTED_SCHEMA_VERSION'")
        }

        if (trajectory.steps.isEmpty()) {
            errors.add("Demonstration must contain at least one step")
        }

        if (trajectory.totalSteps != trajectory.steps.size) {
            errors.add("totalSteps (${trajectory.totalSteps}) does not match steps list size (${trajectory.steps.size})")
        }

        trajectory.steps.forEachIndexed { idx, step ->
            val expectedIndex = idx + 1
            if (step.stepIndex != expectedIndex) {
                errors.add("Step at position $idx has invalid stepIndex ${step.stepIndex}, expected $expectedIndex")
            }

            if (step.goal.isBlank()) {
                errors.add("Step ${step.stepIndex} has a blank goal")
            }

            if (step.deltaSummary.isBlank()) {
                errors.add("Step ${step.stepIndex} has a blank deltaSummary")
            }

            // Action Grounding Validation: Action must target a node present in preObservation
            val availableNodeIndices = step.preObservation.nodes.map { it.index }.toSet()
            when (val action = step.humanAction) {
                is AgentAction.Tap -> {
                    if (action.targetNodeIndex !in availableNodeIndices) {
                        errors.add("Step ${step.stepIndex}: Ungrounded Tap action. Node #${action.targetNodeIndex} does not exist in preObservation (${availableNodeIndices.size} nodes available)")
                    }
                }
                is AgentAction.TypeText -> {
                    if (action.targetNodeIndex !in availableNodeIndices) {
                        errors.add("Step ${step.stepIndex}: Ungrounded TypeText action. Node #${action.targetNodeIndex} does not exist in preObservation (${availableNodeIndices.size} nodes available)")
                    }
                }
                is AgentAction.Scroll -> {
                    if (action.targetNodeIndex !in availableNodeIndices) {
                        errors.add("Step ${step.stepIndex}: Ungrounded Scroll action. Node #${action.targetNodeIndex} does not exist in preObservation (${availableNodeIndices.size} nodes available)")
                    }
                }
                is AgentAction.OpenApp -> {
                    if (action.appName.isBlank()) {
                        errors.add("Step ${step.stepIndex}: OpenApp action has blank appName")
                    }
                }
                else -> {
                    // Back, Home, Wait, CompleteGoal, FailGoal do not require target node grounding
                }
            }
        }

        return ValidationResult(isValid = errors.isEmpty(), errors = errors)
    }
}
