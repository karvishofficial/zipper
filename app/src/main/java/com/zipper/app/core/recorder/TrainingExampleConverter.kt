package com.zipper.app.core.recorder

import com.zipper.app.core.agent.AgentAction

/**
 * Converts raw multi-step demonstration trajectories into standardized, independent training examples
 * for the future Karvish Action Model.
 *
 * Each training example directly maps:
 * Goal + Current Observation + Available Actions + Short History + Previous Outcome -> Next Action
 */
class TrainingExampleConverter {

    fun convertTrajectoryToTrainingExamples(
        trajectory: DemonstrationTrajectory,
        maxHistorySteps: Int = 3
    ): List<TrainingExample> {
        val examples = mutableListOf<TrainingExample>()

        trajectory.steps.forEachIndexed { idx, step ->
            // Collect short history of preceding steps (up to maxHistorySteps)
            val historyStartIndex = (idx - maxHistorySteps).coerceAtLeast(0)
            val shortHistory = mutableListOf<HistoryStep>()
            for (hIdx in historyStartIndex until idx) {
                val pastStep = trajectory.steps[hIdx]
                shortHistory.add(
                    HistoryStep(
                        stepIndex = pastStep.stepIndex,
                        actionSummary = describeAction(pastStep.humanAction),
                        outcomeSummary = pastStep.deltaSummary,
                        isStateChanged = pastStep.isStateChanged
                    )
                )
            }

            val prevOutcome = if (idx > 0) trajectory.steps[idx - 1].deltaSummary else null

            val targetActionName = getActionTypeName(step.humanAction)
            val targetArgs = step.actionArguments.toMutableMap()
            if (targetArgs.isEmpty()) {
                targetArgs.putAll(extractActionArguments(step.humanAction))
            }

            examples.add(
                TrainingExample(
                    exampleId = "${trajectory.metadata.demonstrationId}_step_${step.stepIndex}",
                    goal = step.goal,
                    packageName = step.packageName,
                    currentObservation = step.preObservation,
                    availableActions = step.availableTools,
                    shortHistory = shortHistory,
                    previousOutcome = prevOutcome,
                    targetAction = step.humanAction,
                    targetActionName = targetActionName,
                    targetActionArguments = targetArgs
                )
            )
        }

        return examples
    }

    private fun describeAction(action: AgentAction): String {
        return when (action) {
            is AgentAction.OpenApp -> "OpenApp('${action.appName}')"
            is AgentAction.Tap -> "Tap [#${action.targetNodeIndex}]"
            is AgentAction.TypeText -> "Type '${action.text}' into [#${action.targetNodeIndex}]"
            is AgentAction.Scroll -> "Scroll ${action.direction}"
            is AgentAction.PressBack -> "Press Back"
            AgentAction.Submit -> "Submit focused field"
            is AgentAction.PressHome -> "Press Home"
            is AgentAction.Wait -> "Wait (${action.durationMs}ms)"
            is AgentAction.CompleteGoal -> "CompleteGoal('${action.summary}')"
            is AgentAction.FailGoal -> "FailGoal('${action.reason}')"
        }
    }

    private fun getActionTypeName(action: AgentAction): String {
        return when (action) {
            is AgentAction.OpenApp -> "open_app"
            is AgentAction.Tap -> "tap"
            is AgentAction.TypeText -> "type_text"
            is AgentAction.Scroll -> "scroll"
            is AgentAction.PressBack -> "press_back"
            AgentAction.Submit -> "submit"
            is AgentAction.PressHome -> "press_home"
            is AgentAction.Wait -> "wait"
            is AgentAction.CompleteGoal -> "complete_goal"
            is AgentAction.FailGoal -> "fail_goal"
        }
    }

    private fun extractActionArguments(action: AgentAction): Map<String, String> {
        return when (action) {
            is AgentAction.OpenApp -> mapOf("app_name" to action.appName)
            is AgentAction.Tap -> mapOf("node_index" to action.targetNodeIndex.toString())
            is AgentAction.TypeText -> mapOf("node_index" to action.targetNodeIndex.toString(), "text" to action.text)
            is AgentAction.Scroll -> mapOf("node_index" to action.targetNodeIndex.toString(), "direction" to action.direction.name)
            is AgentAction.Wait -> mapOf("duration_ms" to action.durationMs.toString())
            is AgentAction.CompleteGoal -> mapOf("summary" to action.summary)
            is AgentAction.FailGoal -> mapOf("reason" to action.reason)
            else -> emptyMap()
        }
    }
}
