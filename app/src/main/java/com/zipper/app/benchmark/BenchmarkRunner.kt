package com.zipper.app.benchmark

import com.zipper.app.core.model.ActionPrediction
import com.zipper.app.core.model.ActionType
import com.zipper.app.core.model.DistilledNode
import com.zipper.app.core.model.DistilledUiState
import com.zipper.app.core.model.FlowExecutionResult
import com.zipper.app.core.model.FlowStep
import com.zipper.app.core.model.HeuristicActionModel
import com.zipper.app.core.model.KarvishFlow
import com.zipper.app.core.model.LocalActionModel
import com.zipper.app.core.model.Needle2ActionModel
import com.zipper.app.core.model.NormalizedBounds
import com.zipper.app.core.model.Postcondition
import com.zipper.app.core.model.PostconditionType
import com.zipper.app.core.model.TargetSpec
import com.zipper.app.core.model.VerificationStatus
import com.zipper.app.core.orchestrator.ActionExecutor
import com.zipper.app.core.orchestrator.KarvishOrchestrator
import com.zipper.app.core.verifier.VerificationEngine

/**
 * Report containing aggregate evaluation metrics across a benchmark suite.
 */
data class BenchmarkMetrics(
    val modelName: String,
    val totalTasks: Int,
    val tasksCompletedSuccessfully: Int,
    val totalSteps: Int,
    val stepsVerifiedSuccessfully: Int,
    val averageInferenceLatencyMs: Double,
    val averageTotalTurnaroundMs: Double,
    val silentFailureCount: Int,
    val repairSuccessCount: Int,
    val taskSuccessRate: Double,
    val verificationPassRate: Double,
    val silentFailureRate: Double
)

/**
 * Simulated in-memory executor for running benchmark flows against predefined multi-step UI transitions.
 */
class SimulatedBenchmarkExecutor(
    private val uiStates: MutableList<DistilledUiState>
) : ActionExecutor {
    private var currentStateIndex = 0

    override suspend fun captureCurrentState(): DistilledUiState {
        return uiStates[currentStateIndex.coerceIn(0, uiStates.lastIndex)]
    }

    override suspend fun executeAction(nodeIndex: Int?, action: ActionType, argument: String?): Boolean {
        // Transition to next UI state if available
        if (currentStateIndex < uiStates.lastIndex) {
            currentStateIndex++
        }
        return nodeIndex != null
    }
}

/**
 * Benchmark Runner: Benchmarks the Local Action Model (Cactus Needle 2 baseline vs Heuristic/DeBERTa baselines)
 * across realistic Android mobile task flows.
 */
class BenchmarkRunner {

    /**
     * Creates standard benchmark flows for evaluation.
     */
    fun createStandardBenchmarkSuite(): List<Pair<KarvishFlow, List<DistilledUiState>>> {
        // Task 1: Calendar - Create Event
        val calendarFlow = KarvishFlow(
            flowId = "bench_calendar_event_v1",
            name = "Create Calendar Event",
            targetAppPackage = "com.google.android.calendar",
            parameters = mapOf("title" to "Team Sync 10AM"),
            steps = listOf(
                FlowStep(
                    stepId = "step_open_create",
                    intent = "tap create event button",
                    targetSpec = TargetSpec(
                        semanticRoles = listOf("BUTTON"),
                        semanticLabels = listOf("create", "add", "new event", "+"),
                        preferredBoundsRegion = "bottom_right"
                    ),
                    action = ActionType.TAP,
                    postcondition = Postcondition(
                        type = PostconditionType.ELEMENT_EXISTS,
                        expectedRole = "INPUT",
                        expectedText = "Add title"
                    )
                ),
                FlowStep(
                    stepId = "step_type_title",
                    intent = "type event title",
                    targetSpec = TargetSpec(
                        semanticRoles = listOf("INPUT"),
                        semanticLabels = listOf("add title", "title")
                    ),
                    action = ActionType.TYPE,
                    argumentSlot = "title",
                    postcondition = Postcondition(
                        type = PostconditionType.TEXT_CONTAINS,
                        expectedText = "Team Sync 10AM"
                    )
                ),
                FlowStep(
                    stepId = "step_save",
                    intent = "save event",
                    targetSpec = TargetSpec(
                        semanticRoles = listOf("BUTTON"),
                        semanticLabels = listOf("save", "done", "check")
                    ),
                    action = ActionType.TAP,
                    postcondition = Postcondition(
                        type = PostconditionType.ELEMENT_EXISTS,
                        expectedRole = "TEXT",
                        expectedText = "Team Sync 10AM"
                    )
                )
            )
        )

        val calendarStates = listOf(
            // Initial Screen: Calendar Month View with FAB
            DistilledUiState(
                packageName = "com.google.android.calendar",
                activityName = "AllInOneActivity",
                nodes = listOf(
                    DistilledNode(0, "TEXT", "September 2026", null, "month_title", NormalizedBounds(50, 60, 400, 100)),
                    DistilledNode(1, "BUTTON", "Search", "Search events", "search_btn", NormalizedBounds(850, 50, 950, 110), isClickable = true),
                    DistilledNode(2, "BUTTON", "Create", "Create new event", "floating_action_button", NormalizedBounds(820, 850, 960, 950), isClickable = true)
                )
            ),
            // Screen 2: Event Edit Screen
            DistilledUiState(
                packageName = "com.google.android.calendar",
                activityName = "EditEventActivity",
                nodes = listOf(
                    DistilledNode(0, "BUTTON", "Close", "Discard event", "close_btn", NormalizedBounds(40, 50, 120, 110), isClickable = true),
                    DistilledNode(1, "INPUT", "Add title", "Event title input", "title_input", NormalizedBounds(60, 140, 940, 220), isClickable = true, isEditable = true),
                    DistilledNode(2, "BUTTON", "Save", "Save event", "save_btn", NormalizedBounds(840, 50, 960, 110), isClickable = true)
                )
            ),
            // Screen 3: Event Edit with typed title
            DistilledUiState(
                packageName = "com.google.android.calendar",
                activityName = "EditEventActivity",
                nodes = listOf(
                    DistilledNode(0, "BUTTON", "Close", "Discard event", "close_btn", NormalizedBounds(40, 50, 120, 110), isClickable = true),
                    DistilledNode(1, "INPUT", "Team Sync 10AM", "Event title", "title_input", NormalizedBounds(60, 140, 940, 220), isClickable = true, isEditable = true),
                    DistilledNode(2, "BUTTON", "Save", "Save event", "save_btn", NormalizedBounds(840, 50, 960, 110), isClickable = true)
                )
            ),
            // Screen 4: Returned to Month View with new Event
            DistilledUiState(
                packageName = "com.google.android.calendar",
                activityName = "AllInOneActivity",
                nodes = listOf(
                    DistilledNode(0, "TEXT", "September 2026", null, "month_title", NormalizedBounds(50, 60, 400, 100)),
                    DistilledNode(1, "TEXT", "Team Sync 10AM", "Event entry", "event_item", NormalizedBounds(60, 300, 940, 400), isClickable = true),
                    DistilledNode(2, "BUTTON", "Create", "Create new event", "floating_action_button", NormalizedBounds(820, 850, 960, 950), isClickable = true)
                )
            )
        )

        return listOf(calendarFlow to calendarStates)
    }

    /**
     * Executes the benchmark suite on a given action model.
     */
    suspend fun runBenchmark(actionModel: LocalActionModel): BenchmarkMetrics {
        val suite = createStandardBenchmarkSuite()
        var completedTasks = 0
        var totalSteps = 0
        var verifiedSteps = 0
        var totalInferenceLatency = 0L
        var totalTurnaroundLatency = 0L
        var silentFailures = 0
        var repairCount = 0

        val verifier = VerificationEngine()

        for ((flow, states) in suite) {
            val executor = SimulatedBenchmarkExecutor(states.toMutableList())
            val orchestrator = KarvishOrchestrator(actionModel, verifier, executor)

            val result = orchestrator.executeFlow(flow)
            if (result.isSuccessful) {
                completedTasks++
            }

            for (record in result.records) {
                totalSteps++
                totalInferenceLatency += record.prediction.latencyMs
                totalTurnaroundLatency += record.prediction.latencyMs + record.verificationResult.latencyMs

                if (record.verificationResult.status == VerificationStatus.VERIFIED_SUCCESS) {
                    verifiedSteps++
                }

                if (record.wasRepaired) {
                    repairCount++
                }

                // Check for silent failure (model claimed success on bad action)
                if (record.prediction.selectedNodeIndex == null && record.verificationResult.status == VerificationStatus.VERIFIED_SUCCESS) {
                    silentFailures++
                }
            }
        }

        val stepCount = totalSteps.coerceAtLeast(1)
        val taskCount = suite.size.coerceAtLeast(1)

        return BenchmarkMetrics(
            modelName = actionModel.modelName,
            totalTasks = suite.size,
            tasksCompletedSuccessfully = completedTasks,
            totalSteps = totalSteps,
            stepsVerifiedSuccessfully = verifiedSteps,
            averageInferenceLatencyMs = totalInferenceLatency.toDouble() / stepCount,
            averageTotalTurnaroundMs = totalTurnaroundLatency.toDouble() / stepCount,
            silentFailureCount = silentFailures,
            repairSuccessCount = repairCount,
            taskSuccessRate = (completedTasks.toDouble() / taskCount) * 100.0,
            verificationPassRate = (verifiedSteps.toDouble() / stepCount) * 100.0,
            silentFailureRate = (silentFailures.toDouble() / stepCount) * 100.0
        )
    }
}
