package com.zipper.app

import android.graphics.Rect
import com.zipper.app.benchmark.BenchmarkRunner
import com.zipper.app.core.distiller.AccessibleNode
import com.zipper.app.core.distiller.UiStateDistiller
import com.zipper.app.core.model.ActionType
import com.zipper.app.core.model.DistilledNode
import com.zipper.app.core.model.DistilledUiState
import com.zipper.app.core.model.FlowStep
import com.zipper.app.core.model.GatingDecision
import com.zipper.app.core.model.HeuristicActionModel
import com.zipper.app.core.model.Needle2ActionModel
import com.zipper.app.core.model.NormalizedBounds
import com.zipper.app.core.model.Postcondition
import com.zipper.app.core.model.PostconditionType
import com.zipper.app.core.model.TargetSpec
import com.zipper.app.core.model.VerificationStatus
import com.zipper.app.core.orchestrator.ActionExecutor
import com.zipper.app.core.orchestrator.KarvishOrchestrator
import com.zipper.app.core.verifier.VerificationEngine
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class MockNode(
    override val className: CharSequence?,
    override val text: CharSequence? = null,
    override val contentDescription: CharSequence? = null,
    override val viewIdResourceName: String? = null,
    override val isClickable: Boolean = false,
    override val isEditable: Boolean = false,
    override val isScrollable: Boolean = false,
    override val isEnabled: Boolean = true,
    override val isFocused: Boolean = false,
    override val isCheckable: Boolean = false,
    override val isChecked: Boolean = false,
    override val isSelected: Boolean = false,
    override val roleDescription: CharSequence? = null,
    val left: Int = 0,
    val top: Int = 0,
    val right: Int = 100,
    val bottom: Int = 100,
    private val children: List<MockNode> = emptyList()
) : AccessibleNode {
    override val childCount: Int get() = children.size
    override fun getChild(index: Int): AccessibleNode? = children.getOrNull(index)
    override fun getBoundsInScreen(outBounds: Rect) {
        outBounds.left = left
        outBounds.top = top
        outBounds.right = right
        outBounds.bottom = bottom
    }
}

class KarvishCoreTests {

    @Test
    fun testVerificationEngineZeroSilentFailures() {
        val verifier = VerificationEngine()
        val preState = DistilledUiState(
            packageName = "com.google.android.calendar",
            nodes = listOf(
                DistilledNode(0, "INPUT", "Add title", null, "title", NormalizedBounds(50, 100, 950, 200))
            )
        )

        // Post state where element is missing
        val failedPostState = DistilledUiState(
            packageName = "com.google.android.calendar",
            nodes = emptyList()
        )

        val postcondition = Postcondition(
            type = PostconditionType.ELEMENT_EXISTS,
            expectedRole = "TEXT",
            expectedText = "Meeting at 5PM"
        )

        val result = verifier.verify(postcondition, preState, failedPostState)
        // Must NEVER claim success when condition is unmet (Zero Silent Failures principle)
        assertEquals(VerificationStatus.CONFIRMED_FAILURE, result.status)
        assertTrue(result.message.contains("was not found"))

        // Post state where element exists
        val successPostState = DistilledUiState(
            packageName = "com.google.android.calendar",
            nodes = listOf(
                DistilledNode(0, "TEXT", "Meeting at 5PM", null, "event_title", NormalizedBounds(50, 100, 950, 200))
            )
        )

        val successResult = verifier.verify(postcondition, preState, successPostState)
        assertEquals(VerificationStatus.VERIFIED_SUCCESS, successResult.status)
        assertEquals(0, successResult.matchedNodeIndex)
    }

    @Test
    fun testNeedle2ActionModelPrediction() = runBlocking {
        val model = Needle2ActionModel()
        val step = FlowStep(
            stepId = "step_fab",
            intent = "tap create event button",
            targetSpec = TargetSpec(
                semanticRoles = listOf("BUTTON"),
                semanticLabels = listOf("create", "new event", "+"),
                preferredBoundsRegion = "bottom_right"
            ),
            action = ActionType.TAP
        )

        val state = DistilledUiState(
            packageName = "com.google.android.calendar",
            nodes = listOf(
                DistilledNode(0, "TEXT", "Inbox", null, "title", NormalizedBounds(50, 50, 200, 100)),
                DistilledNode(1, "BUTTON", "Search", "Search mail", "search", NormalizedBounds(800, 50, 950, 100), isClickable = true),
                DistilledNode(2, "BUTTON", "Create", "Create new event", "fab", NormalizedBounds(820, 850, 960, 950), isClickable = true)
            )
        )

        val prediction = model.predictAction(step, state)
        assertEquals(2, prediction.selectedNodeIndex)
        assertEquals(ActionType.TAP, prediction.action)
        assertTrue(prediction.confidence >= 0.65f)
        assertTrue(prediction.modelName.contains("Needle-2"))
    }

    @Test
    fun testHeuristicActionModelBaseline() = runBlocking {
        val model = HeuristicActionModel()
        val step = FlowStep(
            stepId = "step_save",
            intent = "save calendar event",
            targetSpec = TargetSpec(
                semanticRoles = listOf("BUTTON"),
                semanticLabels = listOf("save", "done")
            ),
            action = ActionType.TAP
        )

        val state = DistilledUiState(
            packageName = "com.google.android.calendar",
            nodes = listOf(
                DistilledNode(0, "BUTTON", "Cancel", null, "cancel", NormalizedBounds(50, 50, 200, 100), isClickable = true),
                DistilledNode(1, "BUTTON", "Save", "Save event", "save", NormalizedBounds(800, 50, 950, 100), isClickable = true)
            )
        )

        val prediction = model.predictAction(step, state)
        assertEquals(1, prediction.selectedNodeIndex)
        assertEquals(ActionType.TAP, prediction.action)
        assertTrue(prediction.confidence > 0.5f)
    }

    @Test
    fun testBenchmarkRunnerExecution() = runBlocking {
        val runner = BenchmarkRunner()

        // 1. Benchmark Needle 2 Baseline
        val needleMetrics = runner.runBenchmark(Needle2ActionModel())
        assertEquals(100.0, needleMetrics.taskSuccessRate, 0.01)
        assertEquals(100.0, needleMetrics.verificationPassRate, 0.01)
        assertEquals(0, needleMetrics.silentFailureCount)
        assertEquals(0.0, needleMetrics.silentFailureRate, 0.01)

        // 2. Benchmark Heuristic Baseline
        val heuristicMetrics = runner.runBenchmark(HeuristicActionModel())
        assertEquals(100.0, heuristicMetrics.taskSuccessRate, 0.01)
        assertEquals(100.0, heuristicMetrics.verificationPassRate, 0.01)
        assertEquals(0, heuristicMetrics.silentFailureCount)

        println("=== BENCHMARK COMPARISON RESULTS ===")
        println("Needle 2 Baseline: Success Rate=${needleMetrics.taskSuccessRate}%, Avg Turnaround=${needleMetrics.averageTotalTurnaroundMs}ms, Silent Failures=${needleMetrics.silentFailureCount}")
        println("Heuristic Baseline: Success Rate=${heuristicMetrics.taskSuccessRate}%, Avg Turnaround=${needleMetrics.averageTotalTurnaroundMs}ms, Silent Failures=${heuristicMetrics.silentFailureCount}")
    }

    @Test
    fun testNeedle2SafetyGateRefusalOnNegativeCase() = runBlocking {
        val model = Needle2ActionModel(confidenceThreshold = 0.35f)
        val orchestrator = KarvishOrchestrator(model, executor = MockActionExecutor(), confidenceThreshold = 0.35f)

        val outOfDomainStep = FlowStep(
            stepId = "step_flight",
            intent = "Book flight to Paris",
            targetSpec = TargetSpec(
                semanticRoles = listOf("BUTTON"),
                semanticLabels = listOf("airlines", "tickets")
            ),
            action = ActionType.TAP,
            postcondition = Postcondition(
                type = PostconditionType.ELEMENT_EXISTS,
                expectedRole = "TEXT",
                expectedText = "Booking Confirmed"
            )
        )

        val (record, trace) = orchestrator.executeStepWithDiagnostics(outOfDomainStep)

        assertTrue(
            "Gating must block out-of-domain intent either by refusal or low confidence",
            trace.gatingDecision == GatingDecision.BLOCKED_REFUSAL || trace.gatingDecision == GatingDecision.BLOCKED_LOW_CONFIDENCE
        )
        assertFalse("Action must NOT be dispatched when safety gate blocks out-of-domain intent", trace.actionDispatched)
        assertEquals(VerificationStatus.CONFIRMED_FAILURE, trace.verificationStatus)
        assertTrue(trace.failureReason!!.contains("SAFETY GATE REFUSAL"))
    }

    @Test
    fun testNeedle2SafetyGateOnLowConfidence() = runBlocking {
        val model = Needle2ActionModel(confidenceThreshold = 0.60f)
        val orchestrator = KarvishOrchestrator(model, executor = MockActionExecutor(), confidenceThreshold = 0.60f)

        // Partial match: "search" matches Search button, but "contacts with filter" dilutes score to ~0.30 (< 0.60 threshold)
        val step = FlowStep(
            stepId = "step_search",
            intent = "search contacts with filter",
            targetSpec = TargetSpec(
                semanticRoles = listOf("BUTTON"),
                semanticLabels = listOf("search")
            ),
            action = ActionType.TAP,
            postcondition = Postcondition(
                type = PostconditionType.ELEMENT_EXISTS,
                expectedRole = "BUTTON",
                expectedText = "Save"
            )
        )

        val (record, trace) = orchestrator.executeStepWithDiagnostics(step)

        assertEquals(GatingDecision.BLOCKED_LOW_CONFIDENCE, trace.gatingDecision)
        assertFalse("Action must NOT be dispatched when confidence is below safety gate", trace.actionDispatched)
        assertEquals(VerificationStatus.CONFIRMED_FAILURE, trace.verificationStatus)
        assertTrue(trace.failureReason!!.contains("SAFETY GATE REFUSAL"))
    }

    @Test
    fun testUiDriftDetectionOnLayoutMismatch() = runBlocking {
        val model = Needle2ActionModel(confidenceThreshold = 0.35f)
        val driftingExecutor = MockActionExecutor(simulatePostDrift = true)
        val orchestrator = KarvishOrchestrator(model, executor = driftingExecutor, confidenceThreshold = 0.35f)

        val step = FlowStep(
            stepId = "step_create",
            intent = "create calendar event",
            targetSpec = TargetSpec(
                semanticRoles = listOf("BUTTON"),
                semanticLabels = listOf("create", "new event", "+")
            ),
            action = ActionType.TAP,
            postcondition = Postcondition(
                type = PostconditionType.ELEMENT_EXISTS,
                expectedRole = "BUTTON",
                expectedText = "Save"
            )
        )

        val (record, trace) = orchestrator.executeStepWithDiagnostics(step)

        assertTrue(trace.actionDispatched)
        assertEquals(VerificationStatus.CONFIRMED_FAILURE, trace.verificationStatus)
        assertTrue(trace.failureReason!!.contains("Verification failed"))
    }

    @Test
    fun testRepeatabilityAcross10Runs() = runBlocking {
        val model = Needle2ActionModel(confidenceThreshold = 0.35f)
        val orchestrator = KarvishOrchestrator(model, executor = MockActionExecutor(), confidenceThreshold = 0.35f)

        val step = FlowStep(
            stepId = "step_create",
            intent = "create calendar event",
            targetSpec = TargetSpec(
                semanticRoles = listOf("BUTTON"),
                semanticLabels = listOf("create", "new event", "+")
            ),
            action = ActionType.TAP,
            postcondition = Postcondition(
                type = PostconditionType.ELEMENT_EXISTS,
                expectedRole = "TEXT",
                expectedText = "Inbox"
            )
        )

        val selectedIndices = mutableListOf<Int?>()
        val gatingDecisions = mutableListOf<GatingDecision>()

        for (i in 1..10) {
            val (record, trace) = orchestrator.executeStepWithDiagnostics(step)
            selectedIndices.add(trace.predictedTargetIndex)
            gatingDecisions.add(trace.gatingDecision)
            assertTrue(trace.actionDispatched)
            assertEquals(VerificationStatus.VERIFIED_SUCCESS, trace.verificationStatus)
        }

        assertEquals(10, selectedIndices.size)
        // All 10 runs must select node 2 deterministically
        assertTrue(selectedIndices.all { it == 2 })
        assertTrue(gatingDecisions.all { it == GatingDecision.DISPATCH_ACTION })
    }

    @Test
    fun testClockWheelShortestDistanceModularArithmetic() {
        val utils = com.zipper.app.core.orchestrator.ClockWheelUtils

        // 12-Hour Loop Tests:
        // 4 to 7: should be +3
        assertEquals(3, utils.shortestDistance(current = 4, target = 7, modulus = 12))
        // 4 to 1: should be -3
        assertEquals(-3, utils.shortestDistance(current = 4, target = 1, modulus = 12))
        // 1 to 12: should be -1 (shorter than +11)
        assertEquals(-1, utils.shortestDistance(current = 1, target = 12, modulus = 12))
        // 12 to 1: should be +1 (shorter than -11)
        assertEquals(1, utils.shortestDistance(current = 12, target = 1, modulus = 12))
        // Same value: 0
        assertEquals(0, utils.shortestDistance(current = 5, target = 5, modulus = 12))

        // 60-Minute Loop Tests:
        // 55 to 5: should be +10 (55 -> 0 -> 5 is 10 forward, vs 50 backward)
        assertEquals(10, utils.shortestDistance(current = 55, target = 5, modulus = 60))
        // 5 to 55: should be -10
        assertEquals(-10, utils.shortestDistance(current = 5, target = 55, modulus = 60))
        // 0 to 30: half-way (30 steps forward or backward)
        assertEquals(30, utils.shortestDistance(current = 0, target = 30, modulus = 60))
        // 15 to 45: 30 steps
        assertEquals(30, utils.shortestDistance(current = 15, target = 45, modulus = 60))
        // Same value: 0
        assertEquals(0, utils.shortestDistance(current = 42, target = 42, modulus = 60))
    }

    @Test
    fun testVerificationEngineRejectsInitialTimeMismatch() {
        val verifier = VerificationEngine()
        // Screen shows initial default 04:00 AM
        val initialScreenState = DistilledUiState(
            packageName = "com.sec.android.app.clockpackage",
            nodes = listOf(
                DistilledNode(0, "TEXT", "4:00 AM", null, "alarm_time", NormalizedBounds(100, 200, 500, 300))
            )
        )

        // User requested 7:15 AM
        val result = verifier.verifyAlarmActive(initialScreenState, hour = 7, minute = 15, amPm = "AM")
        // Zero silent failures: must confirm failure when screen shows initial time 4:00 AM instead of 7:15 AM
        assertEquals(VerificationStatus.CONFIRMED_FAILURE, result.status)
        assertTrue(result.message.contains("Could not find active alarm"))
    }

    @Test
    fun testPhase3NoProgrammaticTargetOverridesInSelfRepair() = runBlocking {
        val model = Needle2ActionModel(confidenceThreshold = 0.35f)
        val driftingExecutor = MockActionExecutor(simulatePostDrift = true)
        val orchestrator = KarvishOrchestrator(model, executor = driftingExecutor, confidenceThreshold = 0.35f)

        val step = FlowStep(
            stepId = "step_create",
            intent = "create calendar event",
            targetSpec = TargetSpec(
                semanticRoles = listOf("BUTTON"),
                semanticLabels = listOf("create", "new event", "+")
            ),
            action = ActionType.TAP,
            postcondition = Postcondition(
                type = PostconditionType.ELEMENT_EXISTS,
                expectedRole = "BUTTON",
                expectedText = "Save"
            )
        )

        // Execute step with repair: Postcondition will fail because "Save" is absent
        val record = orchestrator.executeStepWithRepair(step, emptyMap(), maxRepairAttempts = 1)

        // Must report failure honestly rather than faking success via programmatic remapping
        assertEquals(VerificationStatus.CONFIRMED_FAILURE, record.verificationResult.status)
        assertFalse("Must not claim repaired without genuine model selection", record.wasRepaired)
        // Must NOT contain the old programmatic remapping explanation string
        assertFalse(
            "Prediction must not contain programmatic override explanation",
            record.prediction.explanation?.contains("Self-Repair: remapped target to alternative candidate") == true
        )
    }

    @Test
    fun testChromeWebDomRoleDistillation() {
        // Simulates Chrome virtual DOM nodes with className="android.view.View" and roleDescription
        val root = MockNode(
            className = "android.webkit.WebView",
            left = 0, top = 0, right = 1080, bottom = 2400,
            children = listOf(
                MockNode(
                    className = "android.view.View",
                    text = "Welcome to Pluto AI Labs",
                    roleDescription = "heading",
                    left = 50, top = 100, right = 1000, bottom = 200
                ),
                MockNode(
                    className = "android.view.View",
                    text = "Portfolio",
                    roleDescription = "link",
                    isClickable = true,
                    left = 50, top = 300, right = 400, bottom = 400
                ),
                MockNode(
                    className = "android.view.View",
                    contentDescription = "Main Navigation",
                    roleDescription = "button",
                    isClickable = true,
                    left = 900, top = 50, right = 1000, bottom = 150
                )
            )
        )

        val distiller = UiStateDistiller()
        val distilled = distiller.distill(root, 1080, 2400, "com.android.chrome", "org.chromium.chrome.browser.ChromeTabbedActivity")

        val headingNode = distilled.nodes.firstOrNull { it.text == "Welcome to Pluto AI Labs" }
        assertNotNull("Heading must be captured", headingNode)
        assertEquals("HEADER", headingNode?.role)

        val linkNode = distilled.nodes.firstOrNull { it.text == "Portfolio" }
        assertNotNull("Link must be captured", linkNode)
        assertEquals("LINK", linkNode?.role)
        assertTrue("Link must be actionable", linkNode?.isClickable == true)

        val menuButton = distilled.nodes.firstOrNull { it.contentDescription == "Main Navigation" }
        assertNotNull("Menu button must be captured", menuButton)
        assertEquals("BUTTON", menuButton?.role)
        assertTrue("Button must be actionable", menuButton?.isClickable == true)
    }

    @Test
    fun testInstagramDirectSendAffordanceCandidateGeneration() {
        val generator = com.zipper.app.core.agent.ToolSchemaGenerator()
        val instagramDirectState = DistilledUiState(
            packageName = "com.instagram.android",
            activityName = "DirectThreadActivity",
            nodes = listOf(
                DistilledNode(
                    index = 0,
                    role = "INPUT",
                    text = "I",
                    contentDescription = null,
                    resourceId = "com.instagram.android:id/row_thread_composer_edittext",
                    bounds = NormalizedBounds(50, 900, 800, 950),
                    isEditable = true,
                    isClickable = true
                ),
                DistilledNode(
                    index = 1,
                    role = "IMAGE_BUTTON",
                    text = null,
                    contentDescription = null,
                    resourceId = "com.instagram.android:id/row_thread_composer_button_send",
                    bounds = NormalizedBounds(820, 900, 950, 950),
                    isClickable = true
                )
            )
        )

        val result = generator.generateTools(
            uiState = instagramDirectState,
            goal = "open Instagram and send I to Manoj",
            currentPhase = com.zipper.app.core.agent.TaskPhase.COMMIT
        )

        assertTrue("Instagram send button must be preserved in candidateNodeMap", result.candidateNodeMap.containsKey(1))
        val sendNode = result.candidateNodeMap[1]
        assertNotNull("Send node must exist", sendNode)

        val toolNames = (0 until result.toolsJsonArray.length()).map {
            result.toolsJsonArray.getJSONObject(it).getString("name")
        }
        assertTrue("Generated tools must contain tap_send_1: $toolNames", toolNames.contains("tap_send_1"))
    }

    @Test
    fun testChromeWebMenuAffordanceCandidateGeneration() {
        val generator = com.zipper.app.core.agent.ToolSchemaGenerator()
        val chromeWebState = DistilledUiState(
            packageName = "com.android.chrome",
            activityName = "org.chromium.chrome.browser.ChromeTabbedActivity",
            nodes = listOf(
                // 1. Chrome Toolbar Menu (3-dots button)
                DistilledNode(
                    index = 0,
                    role = "IMAGE_BUTTON",
                    text = null,
                    contentDescription = "More options",
                    resourceId = "com.android.chrome:id/menu_button",
                    bounds = NormalizedBounds(950, 80, 1050, 180),
                    isClickable = true
                ),
                // 2. Web DOM Unlabeled Hamburger Menu Button (SVG/Button on website)
                DistilledNode(
                    index = 1,
                    role = "BUTTON",
                    text = null,
                    contentDescription = null,
                    resourceId = null,
                    bounds = NormalizedBounds(30, 220, 150, 320),
                    isClickable = true
                )
            )
        )

        val result = generator.generateTools(
            uiState = chromeWebState,
            goal = "click menu on google",
            currentPhase = com.zipper.app.core.agent.TaskPhase.FIND_TARGET
        )

        assertTrue("Chrome toolbar menu must be in candidates", result.candidateNodeMap.containsKey(0))
        assertTrue("Web unlabeled hamburger menu must be in candidates", result.candidateNodeMap.containsKey(1))

        val toolNames = (0 until result.toolsJsonArray.length()).map {
            result.toolsJsonArray.getJSONObject(it).getString("name")
        }
        assertTrue("Generated tools must contain tap_menu_0: $toolNames", toolNames.contains("tap_menu_0"))
        assertTrue("Generated tools must contain tap_menu_1: $toolNames", toolNames.contains("tap_menu_1"))
    }
}

class MockActionExecutor(private val simulatePostDrift: Boolean = false) : ActionExecutor {
    private var callCount = 0
    private val defaultNodes = listOf(
        DistilledNode(0, "TEXT", "Inbox", null, "title", NormalizedBounds(50, 50, 200, 100)),
        DistilledNode(1, "BUTTON", "Search", "Search mail", "search", NormalizedBounds(800, 50, 950, 100), isClickable = true),
        DistilledNode(2, "BUTTON", "Create", "Create new event", "fab", NormalizedBounds(820, 850, 960, 950), isClickable = true)
    )

    override suspend fun captureCurrentState(): DistilledUiState {
        callCount++
        return if (simulatePostDrift && callCount > 1) {
            // Post-action state drifted (e.g. wrong screen appeared, missing "Save")
            DistilledUiState(
                packageName = "com.google.android.calendar",
                nodes = listOf(
                    DistilledNode(0, "TEXT", "Stopwatch", null, "title", NormalizedBounds(50, 50, 200, 100))
                )
            )
        } else {
            DistilledUiState(
                packageName = "com.google.android.calendar",
                nodes = defaultNodes
            )
        }
    }

    override suspend fun executeAction(nodeIndex: Int?, action: ActionType, argument: String?): Boolean {
        return nodeIndex != null
    }
}
