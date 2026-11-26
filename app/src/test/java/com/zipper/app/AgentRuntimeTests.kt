package com.zipper.app

import com.zipper.app.core.agent.AgentAction
import com.zipper.app.core.agent.AgentModelEngine
import com.zipper.app.core.agent.AgentPromptBuilder
import com.zipper.app.core.agent.GoalArgumentResolver
import com.zipper.app.core.agent.AgentRuntime
import com.zipper.app.core.agent.AgentStepRecord
import com.zipper.app.core.agent.ScrollDirection
import com.zipper.app.core.agent.StateDelta
import com.zipper.app.core.agent.StateDifferentialEngine
import com.zipper.app.core.agent.StepOutcome
import com.zipper.app.core.agent.TerminationReason
import com.zipper.app.core.agent.ToolSchemaGenerator
import com.zipper.app.core.model.DistilledNode
import com.zipper.app.core.model.DistilledUiState
import com.zipper.app.core.model.NormalizedBounds
import com.zipper.app.core.needle.AgentPrediction
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class AgentRuntimeTests {

    @Test
    fun testGoalArgumentResolverHandlesAlarmFieldsAndMessageText() {
        val hourNode = DistilledNode(
            index = 1,
            role = "EDITTEXT",
            text = "",
            contentDescription = "Hour",
            resourceId = "clock_hour",
            bounds = NormalizedBounds(0, 0, 100, 100)
        )
        val minuteNode = hourNode.copy(index = 2, contentDescription = "Minute", resourceId = "clock_minute")
        val messageNode = hourNode.copy(index = 3, contentDescription = "Message")
        val contactNode = hourNode.copy(index = 4, contentDescription = "Search contacts")

        assertEquals("5", GoalArgumentResolver.resolveText("Set alarm for 5:00 AM", "", hourNode))
        assertEquals("00", GoalArgumentResolver.resolveText("Set alarm for 5:00 AM", "", minuteNode))
        assertEquals("", GoalArgumentResolver.resolveText("Set alarm for 5:00 AM", "Alarm name"))
        assertEquals("hello", GoalArgumentResolver.resolveText("Open WhatsApp and send hello to Alice", "", messageNode))
        assertEquals("Alice", GoalArgumentResolver.resolveText("Open WhatsApp and send hello to Alice", "", contactNode))
        assertEquals("Karthik", GoalArgumentResolver.resolveText("Search for Karthik", "", contactNode))
        assertEquals("Karthik", GoalArgumentResolver.resolveText("Search for Karthik in Contacts", "", contactNode))
        assertEquals("Alice Smith", GoalArgumentResolver.resolveText("Search for Alice Smith in Contacts", "", contactNode))
        assertEquals("Display", GoalArgumentResolver.resolveText("Search for Display in Settings", "", contactNode))
        assertEquals("Wi-Fi", GoalArgumentResolver.resolveText("Search for Wi-Fi in Settings", "", contactNode))
        assertEquals("name", GoalArgumentResolver.resolveText("search name", "", contactNode))
    }

    @Test
    fun testModelScrollPreservesTargetAndDirection() {
        val engine = com.zipper.app.core.agent.KarvishActionModelEngine()
        val prediction = engine.parseModelOutput(
            "<action>[INTENT: Adjust the Hour picker] scroll target=#17 direction=UP</action>",
            setOf("scroll_up_hour_17"),
            1L
        )

        assertEquals(AgentAction.Scroll(targetNodeIndex = 17, direction = ScrollDirection.UP), prediction.action)
    }

    @Test
    fun testAlarmCompletionRequiresVisibleRequestedTime() {
        val state = createSampleUiState(
            nodes = listOf(
                DistilledNode(
                    index = 0,
                    role = "TEXT",
                    text = "5:00",
                    contentDescription = "5:00 AM",
                    bounds = NormalizedBounds(0, 0, 100, 100)
                )
            )
        )
        val complete = AgentAction.CompleteGoal("Alarm is set")
        assertTrue(AgentRuntime.isGoalAccomplished("Set alarm for 5:00 AM", complete, state))
        assertFalse(AgentRuntime.isGoalAccomplished("Set alarm for 6:00 AM", complete, state))
    }

    private fun createSampleUiState(
        nodes: List<DistilledNode> = listOf(
            DistilledNode(
                index = 0,
                role = "BUTTON",
                text = "Add alarm",
                contentDescription = "Create new alarm",
                resourceId = "com.sec.android.app.clockpackage:id/menu_alarm_add",
                bounds = NormalizedBounds(800, 50, 900, 100),
                isClickable = true,
                isEditable = false,
                isScrollable = false
            ),
            DistilledNode(
                index = 1,
                role = "EDITTEXT",
                text = "Alarm name",
                contentDescription = null,
                resourceId = "com.sec.android.app.clockpackage:id/alarm_name",
                bounds = NormalizedBounds(100, 400, 900, 480),
                isClickable = true,
                isEditable = true,
                isScrollable = false
            ),
            DistilledNode(
                index = 2,
                role = "BUTTON",
                text = "Save",
                contentDescription = "Save alarm",
                resourceId = "com.sec.android.app.clockpackage:id/save",
                bounds = NormalizedBounds(400, 900, 600, 960),
                isClickable = true,
                isEditable = false,
                isScrollable = false
            )
        ),
        packageName: String = "com.sec.android.app.clockpackage",
        activityName: String = "com.sec.android.app.clockpackage.ClockPackage"
    ): DistilledUiState {
        return DistilledUiState(
            packageName = packageName,
            activityName = activityName,
            nodes = nodes,
            timestamp = System.currentTimeMillis()
        )
    }

    @Test
    fun testToolSchemaGeneratorCreatesValidTools() {
        val generator = ToolSchemaGenerator()
        val uiState = createSampleUiState()

        val result = generator.generateTools(uiState, goal = "Set alarm for 7:00 AM")
        val toolsJson = result.toolsJsonArray

        // The alarm name field is incompatible with a time-only goal and is excluded.
        // Assert tool count: 1 open_app + 2 compatible screen elements + 5 global tools + submit = 9
        assertEquals(9, toolsJson.length())

        val toolNames = mutableListOf<String>()
        for (i in 0 until toolsJson.length()) {
            toolNames.add(toolsJson.getJSONObject(i).getString("name"))
        }

        assertTrue(toolNames.contains("open_app"))
        assertTrue(toolNames.any { it.startsWith("tap_") && it.endsWith("_0") })
        assertFalse(toolNames.any { it.startsWith("type_") && it.endsWith("_1") })
        assertTrue(toolNames.any { it.startsWith("tap_") && it.endsWith("_2") })
        assertTrue(toolNames.contains("scroll_down"))
        assertTrue(toolNames.contains("scroll_up"))
        assertTrue(toolNames.contains("press_back"))
        assertTrue(toolNames.contains("complete_goal"))
        assertTrue(toolNames.contains("fail_goal"))
    }

    @Test
    fun testToolSchemaGeneratorExcludesBlacklistedNodes() {
        val generator = ToolSchemaGenerator()
        val uiState = createSampleUiState()

        // Node 0 previously failed -> blacklisted
        val result = generator.generateTools(
            uiState = uiState,
            blacklistedNodeIndices = setOf(0),
            goal = "Set alarm for 7:00 AM"
        )
        val toolsJson = result.toolsJsonArray

        val toolNames = mutableListOf<String>()
        for (i in 0 until toolsJson.length()) {
            toolNames.add(toolsJson.getJSONObject(i).getString("name"))
        }

        assertFalse("Blacklisted node 0 must NOT appear as a tool", toolNames.any { it.endsWith("_0") })
        assertFalse("Name field must remain excluded for a time-only goal", toolNames.any { it.endsWith("_1") })
        assertTrue("Non-blacklisted node 2 must appear", toolNames.any { it.endsWith("_2") })
    }

    @Test
    fun testToolSchemaAddsGenericSearchOrdinalAndCommitSemantics() {
        val generator = ToolSchemaGenerator()
        val state = createSampleUiState(
            nodes = listOf(
                DistilledNode(
                    index = 10,
                    role = "EDITTEXT",
                    text = "",
                    contentDescription = "Search videos",
                    bounds = NormalizedBounds(0, 0, 900, 100),
                    isClickable = true,
                    isEditable = true
                ),
                DistilledNode(
                    index = 11,
                    role = "TEXTVIEW",
                    text = "Second result",
                    contentDescription = "Video result",
                    bounds = NormalizedBounds(0, 200, 900, 350),
                    isClickable = true
                ),
                DistilledNode(
                    index = 12,
                    role = "BUTTON",
                    text = "Send",
                    contentDescription = "Send message",
                    bounds = NormalizedBounds(0, 800, 900, 900),
                    isClickable = true
                )
            )
        )

        val result = generator.generateTools(
            uiState = state,
            goal = "Open YouTube and search for cats and select 2nd video"
        )
        val tools = (0 until result.toolsJsonArray.length()).map { result.toolsJsonArray.getJSONObject(it) }
        val searchTool = tools.first { it.getString("name").endsWith("_10") }
        val resultTool = tools.first { it.getString("name").endsWith("_11") }
        val sendTool = tools.first { it.getString("name").endsWith("_12") }

        assertTrue(searchTool.getJSONArray("semantic_tags").toString().contains("search_input"))
        assertTrue(resultTool.getJSONArray("semantic_tags").toString().contains("ordinal_candidate"))
        assertTrue(sendTool.getJSONArray("semantic_tags").toString().contains("commit_affordance"))
    }

    @Test
    fun testToolSchemaCarriesDescendantLabelToClickableResultRow() {
        val generator = ToolSchemaGenerator()
        val state = createSampleUiState(
            nodes = listOf(
                DistilledNode(
                    index = 30,
                    role = "LINEAR_LAYOUT",
                    bounds = NormalizedBounds(0, 200, 1000, 320),
                    isClickable = true
                ),
                DistilledNode(
                    index = 31,
                    role = "TEXTVIEW",
                    text = "Battery",
                    bounds = NormalizedBounds(100, 220, 500, 300),
                    parentIndex = 30
                )
            )
        )

        val generated = generator.generateTools(state, goal = "Open the Battery result")
        val tools = (0 until generated.toolsJsonArray.length())
            .map { generated.toolsJsonArray.getJSONObject(it) }
        val resultTool = tools.first { it.getString("name").startsWith("tap_") && it.getString("name").endsWith("_30") }

        assertTrue(resultTool.getString("description").contains("Battery"))
        assertTrue(resultTool.getJSONArray("semantic_tags").toString().contains("result_item"))
    }

    @Test
    fun testToolSchemaExposesGenericControlCapabilities() {
        val generator = ToolSchemaGenerator()
        val state = createSampleUiState(
            nodes = listOf(
                DistilledNode(
                    index = 20,
                    role = "SCROLL_VIEW",
                    contentDescription = "Results",
                    bounds = NormalizedBounds(0, 0, 1000, 900),
                    isScrollable = true
                ),
                DistilledNode(
                    index = 21,
                    role = "EDITTEXT",
                    contentDescription = "Search",
                    bounds = NormalizedBounds(0, 0, 500, 100),
                    isClickable = true,
                    isEditable = true
                )
            )
        )

        val tools = generator.generateTools(state, goal = "Search results")
        val scrollDescription = (0 until tools.toolsJsonArray.length())
            .map { tools.toolsJsonArray.getJSONObject(it) }
            .first { it.getString("name").contains("scroll_down") && it.getString("name").endsWith("_20") }
            .getString("description")
        val typeDescription = (0 until tools.toolsJsonArray.length())
            .map { tools.toolsJsonArray.getJSONObject(it) }
            .first { it.getString("name").contains("type_search_21") }
            .getString("description")

        assertTrue(scrollDescription.contains("scroll_capability"))
        assertTrue(typeDescription.contains("type_capability"))
    }

    @Test
    fun testToolSchemaOffersOnlyUnfulfilledLaunchTarget() {
        val generator = ToolSchemaGenerator()
        val goal = "Open WhatsApp and send hello, then open Calculator and calculate 2 + 2"
        val whatsappState = createSampleUiState(
            packageName = "com.whatsapp",
            nodes = emptyList()
        )
        val calculatorState = createSampleUiState(
            packageName = "com.sec.android.app.popupcalculator",
            nodes = emptyList()
        )

        val nextLaunchTools = generator.generateTools(whatsappState, goal = goal).toolsJsonArray
        val completedTools = generator.generateTools(
            calculatorState,
            goal = "Open Calculator and calculate 2 + 2"
        ).toolsJsonArray

        assertTrue("A later unfulfilled launch must remain available", (0 until nextLaunchTools.length())
            .any { nextLaunchTools.getJSONObject(it).getString("name") == "open_app" })
        assertFalse("No launch tool is needed after all launch targets are visible", (0 until completedTools.length())
            .any { completedTools.getJSONObject(it).getString("name") == "open_app" })
    }

    @Test
    fun testToolSchemaTagsStructuralTabControlWithoutAppVocabulary() {
        val generator = ToolSchemaGenerator()
        val state = createSampleUiState(
            nodes = listOf(
                DistilledNode(
                    index = 40,
                    role = "TAB",
                    text = "Section B",
                    className = "NavigationTab",
                    bounds = NormalizedBounds(0, 0, 500, 100),
                    isClickable = true
                )
            )
        )

        val tool = generator.generateTools(state, goal = "switch to Section B").toolsJsonArray
            .let { tools -> (0 until tools.length()).map { tools.getJSONObject(it) }.first { it.getString("name").endsWith("_40") } }

        assertTrue(tool.getJSONArray("semantic_tags").toString().contains("tab_control"))
        assertTrue(tool.getJSONArray("semantic_tags").toString().contains("navigation_choice"))
    }

    @Test
    fun testStateDifferentialDetectsNoOp() {
        val engine = StateDifferentialEngine()
        val preState = createSampleUiState()
        val postState = createSampleUiState() // Exact duplicate

        val delta = engine.computeDelta(preState, postState, AgentAction.Tap(0))

        assertEquals(StepOutcome.NO_OP, delta.outcome)
        assertTrue(delta.summary.contains("NO-OP"))
        assertFalse(delta.packageChanged)
        assertFalse(delta.activityChanged)
    }

    @Test
    fun testStateDifferentialDetectsScreenProgress() {
        val engine = StateDifferentialEngine()
        val preState = createSampleUiState()

        // Post-state has an additional dialog node
        val postNodes = preState.nodes + DistilledNode(
            index = 3,
            role = "TEXTVIEW",
            text = "Alarm set for 12 hours from now",
            contentDescription = null,
            resourceId = null,
            bounds = NormalizedBounds(200, 800, 800, 850),
            isClickable = false,
            isEditable = false,
            isScrollable = false
        )
        val postState = preState.copy(nodes = postNodes)

        val delta = engine.computeDelta(preState, postState, AgentAction.Tap(2))

        assertEquals(StepOutcome.SUCCESS, delta.outcome)
        assertEquals(1, delta.nodesAddedCount)
        assertTrue(delta.summary.contains("new element"))
    }

    @Test
    fun testStateDifferentialDetectsErrorModal() {
        val engine = StateDifferentialEngine()
        val preState = createSampleUiState()

        val postNodes = listOf(
            DistilledNode(
                index = 10,
                role = "TEXTVIEW",
                text = "Clock app stopped working. Something went wrong.",
                contentDescription = null,
                resourceId = null,
                bounds = NormalizedBounds(100, 300, 900, 500),
                isClickable = false,
                isEditable = false,
                isScrollable = false
            )
        )
        val postState = preState.copy(nodes = postNodes)

        val delta = engine.computeDelta(preState, postState, AgentAction.Tap(0))

        assertEquals(StepOutcome.ERROR_DETECTED, delta.outcome)
        assertNotNull(delta.detectedErrorMessage)
        assertTrue(delta.summary.contains("Error indicator detected"))
    }

    @Test
    fun testAgentPromptBuilderIncludesGoalAndReflection() {
        val builder = AgentPromptBuilder()
        val uiState = createSampleUiState()

        val history = listOf(
            AgentStepRecord(
                turn = 1,
                action = AgentAction.Tap(0),
                actionDescription = "Tap [#0] 'Add alarm'",
                preScreenSummary = "Clock (3 nodes)",
                postScreenSummary = "Clock (3 nodes)",
                outcome = StepOutcome.NO_OP,
                reflection = "Screen did not change",
                confidence = 0.85f,
                latencyMs = 120L
            )
        )

        val prompt = builder.buildPrompt(
            goal = "Set alarm for 07:30 AM",
            currentUiState = uiState,
            stepHistory = history,
            activeReflection = "Tapping element #0 resulted in NO-OP.",
            blacklistedIndices = setOf(0)
        )

        assertTrue(prompt.contains("Set alarm for 07:30 AM"))
        assertTrue(prompt.contains("previous action failed"))
    }

    @Test
    fun testPhase1PromptBuilderPreservesExactGoalAndNeverInjectsDirectives() {
        val builder = AgentPromptBuilder()
        val uiState = createSampleUiState()

        // Arithmetic goal
        val mathGoal = "Open Calculator and calculate 25 + 37"
        val historyWithTaps = listOf(
            AgentStepRecord(
                turn = 1,
                action = AgentAction.OpenApp("Calculator"),
                actionDescription = "OpenApp('Calculator')",
                preScreenSummary = "Launcher",
                postScreenSummary = "Calculator",
                outcome = StepOutcome.SUCCESS,
                reflection = "App opened",
                confidence = 0.90f,
                latencyMs = 100L
            ),
            AgentStepRecord(
                turn = 2,
                action = AgentAction.Tap(10),
                actionDescription = "Tap [#10] '2'",
                preScreenSummary = "Calculator",
                postScreenSummary = "Calculator",
                outcome = StepOutcome.SUCCESS,
                reflection = "Digit 2 typed",
                confidence = 0.88f,
                latencyMs = 90L
            )
        )

        // Turn 3: In older code, this would produce "Tap 5". In Phase 1, it must be the exact user goal!
        val prompt = builder.buildPrompt(
            goal = mathGoal,
            currentUiState = uiState,
            stepHistory = historyWithTaps
        )

        assertEquals("Prompt must be exact user goal", mathGoal, prompt)
        assertFalse("Prompt must NEVER contain precomputed directives like 'Tap 5'", prompt.contains("Tap 5"))
        assertFalse("Prompt must NEVER contain 'Tap plus'", prompt.contains("Tap plus"))
        assertFalse("Prompt must NEVER contain token slicing", prompt == "calculate 25 + 37")
    }

    @Test
    fun testPhase1ToolSchemaGeneratorNeverSuppressesToolsForArithmetic() {
        val generator = ToolSchemaGenerator()
        val uiState = createSampleUiState()

        val historyWithTaps = listOf(
            AgentStepRecord(
                turn = 1,
                action = AgentAction.Tap(1),
                actionDescription = "Tap [#1] '2'",
                preScreenSummary = "Calculator",
                postScreenSummary = "Calculator",
                outcome = StepOutcome.SUCCESS,
                reflection = "OK",
                confidence = 0.9f,
                latencyMs = 100L
            )
        )

        val result = generator.generateTools(
            uiState = uiState,
            goal = "Open Calculator and calculate 25 + 37",
            stepHistory = historyWithTaps
        )

        val toolNames = mutableListOf<String>()
        for (i in 0 until result.toolsJsonArray.length()) {
            toolNames.add(result.toolsJsonArray.getJSONObject(i).getString("name"))
        }

        assertTrue("scroll_down must be available", toolNames.contains("scroll_down"))
        assertTrue("scroll_up must be available", toolNames.contains("scroll_up"))
        assertTrue("press_back must be available", toolNames.contains("press_back"))
        assertTrue("complete_goal must be available", toolNames.contains("complete_goal"))
        assertTrue("fail_goal must be available", toolNames.contains("fail_goal"))
    }

    @Test
    fun testPhase2GoalCompletionRequiresModelDeclarationOrAtomicLaunch() {
        val uiState = createSampleUiState()

        // 1. Compound Goal: Intermediate tap must NOT complete the goal
        val compoundGoal1 = "Open Clock and switch to Stopwatch"
        val tapStopwatch = AgentAction.Tap(1)
        assertFalse(
            "Tapping Stopwatch must NOT prematurely complete compound goal in runtime",
            com.zipper.app.core.agent.AgentRuntime.isGoalAccomplished(
                goal = compoundGoal1,
                action = tapStopwatch,
                postState = uiState,
                actionDescription = "Tap [#1] 'Stopwatch'"
            )
        )

        // 2. Compound Goal: Intermediate tap on Add alarm must NOT complete goal
        val compoundGoal2 = "Open Clock and tap Add alarm"
        val tapAdd = AgentAction.Tap(0)
        assertFalse(
            "Tapping Add alarm must NOT complete goal in runtime",
            com.zipper.app.core.agent.AgentRuntime.isGoalAccomplished(
                goal = compoundGoal2,
                action = tapAdd,
                postState = uiState,
                actionDescription = "Tap [#0] 'Add alarm'"
            )
        )

        val commaSeparatedGoal = "Open Settings, search for Battery, and open the Battery result"
        assertFalse(
            "Opening the app must not complete a comma-separated compound goal",
            com.zipper.app.core.agent.AgentRuntime.isGoalAccomplished(
                goal = commaSeparatedGoal,
                action = AgentAction.OpenApp("settings"),
                postState = createSampleUiState(packageName = "com.android.settings"),
                actionDescription = "OpenApp('settings')"
            )
        )

        // 3. Calculation Goal: Intermediate tap on '=' must NOT complete goal
        val calcGoal = "Calculate 25 + 37"
        val tapEqual = AgentAction.Tap(10)
        assertFalse(
            "Tapping equal must NOT complete math goal without CompleteGoal call",
            com.zipper.app.core.agent.AgentRuntime.isGoalAccomplished(
                goal = calcGoal,
                action = tapEqual,
                postState = uiState,
                actionDescription = "Tap [#10] '='"
            )
        )

        // 4. Atomic Launch Goal: Verified launch DOES complete atomic goal
        val atomicGoal = "Open Calculator"
        val openApp = AgentAction.OpenApp("Calculator")
        val calcState = createSampleUiState(packageName = "com.sec.android.app.popupcalculator")
        assertTrue(
            "Atomic OpenApp with matching target package must complete atomic launch goal",
            com.zipper.app.core.agent.AgentRuntime.isGoalAccomplished(
                goal = atomicGoal,
                action = openApp,
                postState = calcState,
                actionDescription = "OpenApp('Calculator')"
            )
        )

        // 5. CompleteGoal action declared by model:
        // Must reject if required tab (Stopwatch) was not verified in post-state
        val completeAction = AgentAction.CompleteGoal("Switched to Stopwatch screen")
        assertFalse(
            "CompleteGoal declared by model must be REJECTED if target tab was not verified",
            com.zipper.app.core.agent.AgentRuntime.isGoalAccomplished(
                goal = compoundGoal1,
                action = completeAction,
                postState = uiState,
                actionDescription = "CompleteGoal"
            )
        )

        // Must accept if target tab (Stopwatch) is verified in post-state
        val stopwatchState = uiState.copy(
            nodes = uiState.nodes + DistilledNode(
                index = 5,
                role = "BUTTON",
                text = "Lap",
                contentDescription = "Lap",
                bounds = NormalizedBounds(0, 0, 100, 100),
                isClickable = true
            )
        )
        assertTrue(
            "CompleteGoal declared by model must be accepted when target tab controls are verified",
            com.zipper.app.core.agent.AgentRuntime.isGoalAccomplished(
                goal = compoundGoal1,
                action = completeAction,
                postState = stopwatchState,
                actionDescription = "CompleteGoal"
            )
        )
    }

    @Test
    fun testRepeatedIdenticalStateTerminatesEarly() = runBlocking {
        val staticState = createSampleUiState()
        // Model picks a different valid action each turn, but state stays unchanged (NO-OP)
        val actions = listOf(
            AgentPrediction(AgentAction.Tap(0), 0.9f, 50L, "Tap 0", "tap_add_alarm_0", isRefusal = false),
            AgentPrediction(AgentAction.Tap(2), 0.9f, 50L, "Tap 2", "tap_save_2", isRefusal = false),
            AgentPrediction(AgentAction.Scroll(0, ScrollDirection.DOWN), 0.9f, 50L, "Scroll", "scroll_down", isRefusal = false),
            AgentPrediction(AgentAction.Tap(1), 0.9f, 50L, "Tap 1", "tap_edit_1", isRefusal = false)
        )
        var inferenceCount = 0
        val mockEngine = AgentModelEngine { _, _ ->
            actions[inferenceCount++ % actions.size]
        }

        val runtime = AgentRuntime(
            needleEngine = mockEngine,
            maxTurns = 10,
            maxConsecutiveNoProgressTurns = 3,
            maxConsecutiveIdenticalActions = 2,
            stepDelayMs = 0L,
            stateProvider = { staticState },
            actionExecutor = { _, _ -> true }
        )

        val result = runtime.executeGoal("Perform test task")
        assertFalse("Run must terminate with failure when state is stuck", result.success)
        assertEquals("Must terminate after exactly 3 no-progress turns", 3, result.turnsExecuted)
        assertEquals(TerminationReason.STUCK_NO_PROGRESS, result.terminationReason)
        assertTrue(result.finalSummary.contains("STUCK_NO_PROGRESS"))
        assertTrue(result.finalSummary.contains("state remained unchanged"))
    }

    @Test
    fun testRepeatedIdenticalActionTerminatesEarly() = runBlocking {
        val staticState = createSampleUiState()
        // Model repeats the exact same action twice on an unchanged state
        val mockEngine = AgentModelEngine { _, _ ->
            AgentPrediction(
                action = AgentAction.Tap(0),
                confidence = 0.85f,
                latencyMs = 50L,
                explanation = "Tap Add Alarm again",
                rawToolName = "tap_add_alarm_0",
                isRefusal = false
            )
        }

        val runtime = AgentRuntime(
            needleEngine = mockEngine,
            maxTurns = 10,
            maxConsecutiveIdenticalActions = 2,
            maxConsecutiveNoProgressTurns = 5,
            stepDelayMs = 0L,
            stateProvider = { staticState },
            actionExecutor = { _, _ -> true }
        )

        val result = runtime.executeGoal("Set alarm for 7 AM")
        assertFalse("Run must fail when repeating same action with no progress", result.success)
        assertEquals("Must terminate within the configured no-progress bound", 5, result.turnsExecuted)
        assertEquals(TerminationReason.STUCK_NO_PROGRESS, result.terminationReason)
        assertTrue(result.finalSummary.contains("STUCK_NO_PROGRESS"))
        assertTrue(result.finalSummary.contains("state remained unchanged"))
    }

    @Test
    fun testRepeatedRefusalTerminatesEarly() = runBlocking {
        val staticState = createSampleUiState()
        // Model refuses or outputs empty function calls for 3 consecutive turns
        var callCount = 0
        val mockEngine = AgentModelEngine { _, _ ->
            callCount++
            AgentPrediction(
                action = null,
                confidence = 0.0f,
                latencyMs = 40L,
                explanation = "I cannot perform this task",
                rawToolName = null,
                rawJson = "{}",
                decodeTps = 25.0,
                peakRamMb = 120.0,
                isRefusal = true
            )
        }

        val runtime = AgentRuntime(
            needleEngine = mockEngine,
            maxTurns = 15,
            maxConsecutiveRefusals = 3,
            stepDelayMs = 0L,
            stateProvider = { staticState },
            actionExecutor = { _, _ -> true }
        )

        val result = runtime.executeGoal("Unsupported request")
        assertFalse("Run must fail on repeated refusal", result.success)
        assertEquals("Must terminate after 3 consecutive refusals", 3, result.turnsExecuted)
        assertEquals(TerminationReason.REPEATED_REFUSAL, result.terminationReason)
        assertTrue(result.finalSummary.contains("REPEATED_REFUSAL"))
        assertTrue(result.finalSummary.contains("Model refused or produced no action"))
    }

    @Test
    fun testStateChangingActionDoesNotPrematurelyTerminate() = runBlocking {
        // State changes on every turn
        var turnCounter = 0
        val states = listOf(
            createSampleUiState(),
            createSampleUiState().copy(nodes = createSampleUiState().nodes + DistilledNode(
                index = 3,
                role = "TEXTVIEW",
                text = "Step 1 Done",
                bounds = NormalizedBounds(100, 100, 200, 200)
            )),
            createSampleUiState().copy(nodes = createSampleUiState().nodes + DistilledNode(
                index = 4,
                role = "TEXTVIEW",
                text = "Step 2 Done",
                bounds = NormalizedBounds(100, 200, 200, 300)
            )),
            createSampleUiState().copy(nodes = createSampleUiState().nodes + DistilledNode(
                index = 5,
                role = "TEXTVIEW",
                text = "All Steps Complete",
                bounds = NormalizedBounds(100, 300, 200, 400)
            ))
        )

        val predictions = listOf(
            AgentPrediction(AgentAction.Tap(0), 0.9f, 50L, "Tap 0", "tap_0", isRefusal = false),
            AgentPrediction(AgentAction.Tap(1), 0.9f, 50L, "Tap 1", "tap_1", isRefusal = false),
            AgentPrediction(AgentAction.Tap(2), 0.9f, 50L, "Tap 2", "tap_2", isRefusal = false),
            AgentPrediction(AgentAction.CompleteGoal("All Steps Complete"), 0.95f, 50L, "Done", "complete_goal", isRefusal = false)
        )

        val mockEngine = AgentModelEngine { _, _ ->
            predictions[turnCounter]
        }

        val runtime = AgentRuntime(
            needleEngine = mockEngine,
            maxTurns = 10,
            maxConsecutiveNoProgressTurns = 3,
            maxConsecutiveIdenticalActions = 2,
            stepDelayMs = 0L,
            stateProvider = {
                val s = states[turnCounter.coerceAtMost(states.size - 1)]
                s
            },
            actionExecutor = { _, _ ->
                turnCounter++
                true
            }
        )

        val result = runtime.executeGoal("Complete multi-step form")
        assertTrue("Must succeed and not terminate prematurely", result.success)
        assertEquals(4, result.turnsExecuted)
        assertEquals(TerminationReason.GOAL_COMPLETED, result.terminationReason)
    }

    @Test
    fun testNormalMultiTurnProgressSucceeds() = runBlocking {
        // Multi-turn progress: OpenApp -> Tap -> CompleteGoal
        val state1 = createSampleUiState(packageName = "com.android.launcher")
        val state2 = createSampleUiState(packageName = "com.sec.android.app.popupcalculator")
        val state3 = state2.copy(nodes = state2.nodes + DistilledNode(
            index = 10,
            role = "TEXTVIEW",
            text = "62",
            bounds = NormalizedBounds(100, 100, 900, 200)
        ))

        var currentPhase = 0
        val mockEngine = AgentModelEngine { _, _ ->
            when (currentPhase) {
                0 -> AgentPrediction(AgentAction.OpenApp("Calculator"), 0.9f, 50L, "Open Calc", "open_app", isRefusal = false)
                1 -> AgentPrediction(AgentAction.Tap(0), 0.88f, 50L, "Tap digit", "tap_0", isRefusal = false)
                else -> AgentPrediction(AgentAction.CompleteGoal("Calculation complete: 62"), 0.95f, 50L, "Done", "complete_goal", isRefusal = false)
            }
        }

        val runtime = AgentRuntime(
            needleEngine = mockEngine,
            maxTurns = 10,
            maxConsecutiveNoProgressTurns = 3,
            maxConsecutiveIdenticalActions = 2,
            stepDelayMs = 0L,
            stateProvider = {
                when (currentPhase) {
                    0 -> state1
                    1 -> state2
                    else -> state3
                }
            },
            actionExecutor = { _, _ ->
                currentPhase++
                true
            }
        )

        val result = runtime.executeGoal("Open Calculator and calculate 25 + 37")
        assertTrue("Multi-turn sequence must succeed", result.success)
        assertEquals(TerminationReason.GOAL_COMPLETED, result.terminationReason)
        assertTrue(result.finalSummary.contains("Goal completed"))
    }

    @Test
    fun testIsStateEffectivelyIdentical() {
        val stateA = createSampleUiState()
        val stateB = createSampleUiState()
        assertTrue(AgentRuntime.isStateEffectivelyIdentical(stateA, stateB))

        val diffPackage = stateA.copy(packageName = "other.package")
        assertFalse(AgentRuntime.isStateEffectivelyIdentical(stateA, diffPackage))

        val diffNodes = stateA.copy(nodes = stateA.nodes.drop(1))
        assertFalse(AgentRuntime.isStateEffectivelyIdentical(stateA, diffNodes))
    }

    @Test
    fun testIsActionIdentical() {
        assertTrue(AgentRuntime.isActionIdentical(AgentAction.Tap(0), "tap_0", AgentAction.Tap(0), "tap_0"))
        assertFalse(AgentRuntime.isActionIdentical(AgentAction.Tap(0), "tap_0", AgentAction.Tap(1), "tap_1"))
        assertTrue(AgentRuntime.isActionIdentical(null, "open_app", null, "open_app"))
        assertFalse(AgentRuntime.isActionIdentical(null, null, null, null))
        assertFalse(AgentRuntime.isActionIdentical(AgentAction.Tap(0), "tap_0", null, null))
    }

    @Test
    fun testTwoStepPingPongOscillationDetectedAndTerminated() = runBlocking {
        val stateA = createSampleUiState(packageName = "com.android.launcher")
        val stateB = createSampleUiState(packageName = "com.sec.android.app.popupcalculator")

        var currentState = stateA
        val mockEngine = AgentModelEngine { _, _ ->
            if (currentState.packageName == "com.android.launcher") {
                AgentPrediction(AgentAction.OpenApp("Calculator"), 0.9f, 20L, "Open Calculator", "open_app", isRefusal = false)
            } else {
                AgentPrediction(AgentAction.PressBack, 0.85f, 20L, "Press Back", "press_back", isRefusal = false)
            }
        }

        val runtime = AgentRuntime(
            needleEngine = mockEngine,
            maxTurns = 15,
            stepDelayMs = 0L,
            stateProvider = { currentState },
            actionExecutor = { action, _ ->
                currentState = if (action is AgentAction.OpenApp) stateB else stateA
                true
            }
        )

        val result = runtime.executeGoal("Open Calculator and calculate 25 + 37")
        assertFalse("Oscillating ping-pong loop must fail", result.success)
        assertEquals(TerminationReason.CYCLE_DETECTED, result.terminationReason)
        assertTrue("Must terminate early due to cycle detection", result.turnsExecuted <= 6)
        assertTrue(result.finalSummary.contains("CYCLE_DETECTED"))
    }
}
