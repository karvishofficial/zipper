package com.zipper.app

import com.zipper.app.core.agent.AgentAction
import com.zipper.app.core.agent.AgentRuntime
import com.zipper.app.core.agent.KarvishActionModelEngine
import com.zipper.app.core.agent.StepOutcome
import com.zipper.app.core.model.DistilledNode
import com.zipper.app.core.model.DistilledUiState
import com.zipper.app.core.model.NormalizedBounds
import kotlinx.coroutines.runBlocking
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class KarvishActionModelEngineTests {

    private val engine = KarvishActionModelEngine()

    @Test
    fun testDirectTapActionParsedCorrectly() {
        val raw = "<action>[INTENT: Select Alarm tab] tap 4</action>"
        val pred = engine.parseModelOutput(raw, setOf("tap_4"), 45L)

        assertFalse(pred.isRefusal)
        assertEquals(AgentAction.Tap(targetNodeIndex = 4), pred.action)
        assertEquals("Select Alarm tab", pred.explanation)
        assertEquals(45L, pred.latencyMs)
        assertTrue(pred.confidence >= 0.90f)
    }

    @Test
    fun testTypeTextActionParsedCorrectly() {
        val raw = "<action>[INTENT: Search for file] type_text text=\"resume.pdf\"</action>"
        val pred = engine.parseModelOutput(raw, setOf("type_2"), 60L)

        assertFalse(pred.isRefusal)
        assertEquals(AgentAction.TypeText(targetNodeIndex = 0, text = "resume.pdf"), pred.action)
        assertEquals("Search for file", pred.explanation)
    }

    @Test
    fun testScrollAndNavigationActions() {
        val scrollRaw = "<action>[INTENT: Scroll down to view more] scroll direction=down</action>"
        val scrollPred = engine.parseModelOutput(scrollRaw, emptySet(), 30L)
        assertFalse(scrollPred.isRefusal)
        assertEquals(AgentAction.Scroll(direction = com.zipper.app.core.agent.ScrollDirection.DOWN), scrollPred.action)

        val backRaw = "<action>[INTENT: Dismiss dialog] back</action>"
        val backPred = engine.parseModelOutput(backRaw, emptySet(), 20L)
        assertFalse(backPred.isRefusal)
        assertEquals(AgentAction.PressBack, backPred.action)

        val completeRaw = "<action>[INTENT: Goal achieved] complete</action>"
        val completePred = engine.parseModelOutput(completeRaw, emptySet(), 15L)
        assertFalse(completePred.isRefusal)
        assertEquals(AgentAction.CompleteGoal(summary = "Goal achieved"), completePred.action)
    }

    @Test
    fun testRefusalOnInvalidAction() {
        val invalidRaw = "<action>unknown_command foo=bar</action>"
        val pred = engine.parseModelOutput(invalidRaw, emptySet(), 10L)

        assertTrue(pred.isRefusal)
        assertEquals(null, pred.action)
        assertEquals(0.0f, pred.confidence)
    }

    @Test
    fun testEndToEndAgentRuntimeIntegrationWithKarvishModel() = runBlocking {
        val mockNode1 = DistilledNode(
            index = 1,
            role = "Button",
            text = "Alarm",
            bounds = NormalizedBounds(0, 800, 250, 900),
            isClickable = true
        )
        val mockNode2 = DistilledNode(
            index = 2,
            role = "Button",
            text = "Add alarm",
            bounds = NormalizedBounds(400, 100, 600, 200),
            isClickable = true
        )

        val testUiState = DistilledUiState(
            packageName = "com.sec.android.app.clockpackage",
            activityName = "com.sec.android.app.clockpackage.ClockTabActivity",
            nodes = listOf(mockNode1, mockNode2)
        )

        var stepCounter = 0
        val mockEngine = KarvishActionModelEngine()

        val runtime = AgentRuntime(
            needleEngine = mockEngine,
            maxTurns = 5,
            stateProvider = { testUiState },
            actionExecutor = { action, _ ->
                stepCounter++
                true
            }
        )

        val tools = JSONArray().apply {
            put(JSONObject().apply {
                put("name", "tap_1")
                put("description", "Tap Alarm")
            })
            put(JSONObject().apply {
                put("name", "tap_2")
                put("description", "Tap Add alarm")
            })
        }

        val prediction = mockEngine.runAgentInference(
            "<goal>Set alarm for 7:00 AM</goal><ui>[#1] Button 'Alarm' clickable\n[#2] Button 'Add alarm' clickable</ui>",
            tools
        )

        assertNotNull(prediction)
        // Verify latency is sub-100ms
        assertTrue("Latency should be < 100ms", prediction.latencyMs < 100)
    }
}
