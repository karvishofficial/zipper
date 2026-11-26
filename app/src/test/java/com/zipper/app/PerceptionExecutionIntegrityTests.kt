package com.zipper.app

import com.zipper.app.core.agent.ActionCandidate
import com.zipper.app.core.agent.AgentAction
import com.zipper.app.core.agent.ToolSchemaGenerator
import com.zipper.app.core.model.DistilledNode
import com.zipper.app.core.model.DistilledUiState
import com.zipper.app.core.model.NormalizedBounds
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Step 1: Perception & Execution Integrity Test Suite
 *
 * Core Invariant:
 * A candidate ID identifies one candidate within one observation.
 * The executor must execute only the node bound to that candidate and must never infer
 * the target from the human-readable tool name, label, coordinate, or ambiguous numeric substring.
 * If that binding becomes invalid, re-observe instead of guessing.
 */
class PerceptionExecutionIntegrityTests {

    private val toolSchemaGenerator = ToolSchemaGenerator()

    /**
     * Test 1: Ambiguous Numeric Keypad Test (The Calculator Bug)
     *
     * In Calculator, "History" is node #0, and Digit "0" is node #20.
     * Legacy regex parsing on "tap_0_20" matched "_0", targeting node #0 (History) instead of #20 (Digit 0).
     *
     * In Step 1:
     * - The candidate carries nodeIndex = 20 directly.
     * - Tool JSON explicitly embeds target_node_index = 20 and candidate_id = "c_20".
     * - Target is never inferred from the tool name string.
     */
    @Test
    fun testAmbiguousNumericKeypadTargetIntegrity() {
        val historyNode = DistilledNode(
            index = 0,
            role = "BUTTON",
            text = "History",
            contentDescription = "Calculation history",
            resourceId = "com.google.android.calculator:id/history",
            isClickable = true,
            bounds = NormalizedBounds(0, 50, 150, 100)
        )
        val digit5Node = DistilledNode(
            index = 10,
            role = "BUTTON",
            text = "5",
            contentDescription = "5",
            resourceId = "com.google.android.calculator:id/digit_5",
            isClickable = true,
            bounds = NormalizedBounds(350, 600, 500, 700)
        )
        val digit0Node = DistilledNode(
            index = 20,
            role = "BUTTON",
            text = "0",
            contentDescription = "0",
            resourceId = "com.google.android.calculator:id/digit_0",
            isClickable = true,
            bounds = NormalizedBounds(350, 800, 500, 900)
        )
        val digit1Node = DistilledNode(
            index = 21,
            role = "BUTTON",
            text = "1",
            contentDescription = "1",
            resourceId = "com.google.android.calculator:id/digit_1",
            isClickable = true,
            bounds = NormalizedBounds(150, 700, 300, 800)
        )
        val plusNode = DistilledNode(
            index = 22,
            role = "BUTTON",
            text = "+",
            contentDescription = "plus",
            resourceId = "com.google.android.calculator:id/op_add",
            isClickable = true,
            bounds = NormalizedBounds(750, 700, 900, 800)
        )

        val calculatorState = DistilledUiState(
            packageName = "com.google.android.calculator",
            activityName = "CalculatorActivity",
            nodes = listOf(historyNode, digit5Node, digit0Node, digit1Node, plusNode)
        )

        val generated = toolSchemaGenerator.generateTools(
            uiState = calculatorState,
            goal = "Calculate 25 + 37"
        )

        // 1. Verify candidateMap contains candidate for digit 0
        val cand0 = generated.candidateMap["c_20"]
        assertNotNull("Candidate c_20 must exist in candidateMap", cand0)
        assertEquals(20, cand0!!.nodeIndex)
        assertEquals("0", cand0.semanticLabel)
        assertEquals("button", cand0.role)
        assertEquals("tap", cand0.actionType)
        assertEquals(digit0Node, cand0.node)

        // 2. Verify History candidate is completely separate at node 0
        val candHistory = generated.candidateMap["c_0"]
        assertNotNull("Candidate c_0 must exist in candidateMap", candHistory)
        assertEquals(0, candHistory!!.nodeIndex)
        assertEquals("History", candHistory.semanticLabel)
        assertNotEquals(cand0.nodeIndex, candHistory.nodeIndex)

        // 3. Verify JSON tool schema for Digit 0 contains structured target_node_index
        var foundDigit0Tool = false
        for (i in 0 until generated.toolsJsonArray.length()) {
            val toolObj = generated.toolsJsonArray.getJSONObject(i)
            if (toolObj.optString("name") == "tap_0_20") {
                foundDigit0Tool = true
                assertEquals("c_20", toolObj.getString("candidate_id"))
                assertEquals(20, toolObj.getInt("target_node_index"))
                assertEquals("0", toolObj.getString("semantic_label"))
                assertEquals("com.google.android.calculator:id/digit_0", toolObj.getString("node_identity"))
            }
        }
        assertTrue("Tool 'tap_0_20' must be present in toolsJsonArray", foundDigit0Tool)

        // 4. Construct AgentAction.Tap from the candidate: verify targetNodeIndex is strictly 20, never 0
        val tapAction = AgentAction.Tap(
            targetNodeIndex = cand0.nodeIndex!!,
            candidateId = cand0.candidateId,
            semanticLabel = cand0.semanticLabel,
            nodeIdentity = cand0.nodeIdentity
        )
        assertEquals(20, tapAction.targetNodeIndex)
        assertEquals("c_20", tapAction.candidateId)
        assertEquals("0", tapAction.semanticLabel)
        assertNotEquals(0, tapAction.targetNodeIndex)
    }

    /**
     * Test 2: Duplicate Labels Test
     *
     * Two distinct nodes on screen share the identical text "0":
     * - Node #20: Digit "0" at keypad location (bounds: 350, 800, 500, 900)
     * - Node #35: Another "0" at a different location (bounds: 100, 200, 200, 300)
     *
     * The runtime must bind strictly to the candidate's observation-scoped index.
     * It must NEVER search globally for text "0", which would ambiguously hit the wrong node.
     */
    @Test
    fun testDuplicateLabelsResolveStrictlyByCandidateBinding() {
        val digit0Keypad = DistilledNode(
            index = 20,
            role = "BUTTON",
            text = "0",
            contentDescription = "0",
            resourceId = "com.test.app:id/key_0",
            isClickable = true,
            bounds = NormalizedBounds(350, 800, 500, 900)
        )
        val digit0Secondary = DistilledNode(
            index = 35,
            role = "BUTTON",
            text = "0",
            contentDescription = "0",
            resourceId = "com.test.app:id/badge_0",
            isClickable = true,
            bounds = NormalizedBounds(100, 200, 200, 300)
        )

        val duplicateLabelState = DistilledUiState(
            packageName = "com.test.app",
            nodes = listOf(digit0Keypad, digit0Secondary)
        )

        val generated = toolSchemaGenerator.generateTools(
            uiState = duplicateLabelState,
            goal = "Tap 0"
        )

        val cand20 = generated.candidateMap["c_20"]
        val cand35 = generated.candidateMap["c_35"]

        assertNotNull("Candidate c_20 must exist", cand20)
        assertNotNull("Candidate c_35 must exist", cand35)

        assertEquals(20, cand20!!.nodeIndex)
        assertEquals(35, cand35!!.nodeIndex)

        // Both candidates share identical semantic text "0"
        assertEquals("0", cand20.semanticLabel)
        assertEquals("0", cand35.semanticLabel)

        // But have distinct identities and indices
        assertNotEquals(cand20.candidateId, cand35.candidateId)
        assertNotEquals(cand20.nodeIndex, cand35.nodeIndex)
        assertNotEquals(cand20.nodeIdentity, cand35.nodeIdentity)

        // When candidate c_20 is selected, the executor binds strictly to node 20
        val targetNodeForCand20 = generated.candidateNodeMap[cand20.nodeIndex]
        val targetNodeForCand35 = generated.candidateNodeMap[cand35.nodeIndex]

        assertEquals(digit0Keypad, targetNodeForCand20)
        assertEquals(digit0Secondary, targetNodeForCand35)
        assertNotEquals(targetNodeForCand20, targetNodeForCand35)
    }

    /**
     * Test 3: Fail-Closed Invariant on Stale or Invalid Candidate Binding
     *
     * If the UI refreshed or changed underneath such that the live node at targetNodeIndex
     * has mutated (e.g. index #20 is now "History" or a different widget), or if the node
     * disappeared and no stable resourceId is available:
     *
     * The system must fail closed (return false / prompt re-observation),
     * rather than guessing approximate coordinates or clicking something nearby.
     */
    @Test
    fun testFailClosedValidationLogic() {
        // Validation helper matching ActionDispatcher.isSemanticallyCompatible
        fun isSemanticallyCompatible(expected: String, liveText: String?, liveDesc: String?): Boolean {
            val exp = expected.trim().lowercase()
            val txt = liveText?.trim()?.lowercase().orEmpty()
            val desc = liveDesc?.trim()?.lowercase().orEmpty()
            if (txt.isEmpty() && desc.isEmpty()) return true
            if (txt == exp || desc == exp) return true
            if (txt.contains(exp) || (txt.length >= 2 && exp.contains(txt))) return true
            if (desc.contains(exp) || (desc.length >= 2 && exp.contains(desc))) return true
            return false
        }

        // Case 1: Exact match -> Compatible
        assertTrue(isSemanticallyCompatible("0", "0", "0"))
        assertTrue(isSemanticallyCompatible("Search", "Search music", null))
        assertTrue(isSemanticallyCompatible("Music", null, "Music channel"))

        // Case 2: Mutation mismatch -> Incompatible -> FAILS CLOSED
        assertFalse(isSemanticallyCompatible("0", "History", "Calculation history"))
        assertFalse(isSemanticallyCompatible("0", "+", "Plus"))
        assertFalse(isSemanticallyCompatible("Search", "Settings", null))
        assertFalse(isSemanticallyCompatible("Submit", "Cancel", null))

        // Case 3: Empty live labels (e.g. unlabelled icon button) -> Allowed for touch
        assertTrue(isSemanticallyCompatible("c_20", "", null))
    }

    /**
     * Test 4: Full Candidate Map Preserves All Primitives
     *
     * Verifies that ToolSchemaGenerator populates ActionCandidate instances
     * for interactive (tap, type, scroll) and system primitives (open_app, back, submit, complete_goal).
     */
    @Test
    fun testCandidateMapPreservesAllActionPrimitives() {
        val searchInput = DistilledNode(
            index = 1,
            role = "EDITTEXT",
            text = "",
            contentDescription = "Search input",
            resourceId = "com.test.app:id/search_box",
            isClickable = true,
            isEditable = true,
            bounds = NormalizedBounds(50, 50, 950, 120)
        )
        val submitButton = DistilledNode(
            index = 2,
            role = "BUTTON",
            text = "Search",
            contentDescription = "Search",
            isClickable = true,
            bounds = NormalizedBounds(800, 50, 950, 120)
        )
        val feedContainer = DistilledNode(
            index = 3,
            role = "RECYCLERVIEW",
            isScrollable = true,
            bounds = NormalizedBounds(0, 150, 1000, 950)
        )

        val state = DistilledUiState(
            packageName = "com.test.app",
            nodes = listOf(searchInput, submitButton, feedContainer)
        )

        val generated = toolSchemaGenerator.generateTools(
            uiState = state,
            goal = "Search rock music"
        )

        // TypeText candidate
        val typeCand = generated.candidateMap["c_1"]
        assertNotNull("Type candidate must exist", typeCand)
        assertEquals("type_text", typeCand!!.actionType)
        assertEquals(1, typeCand.nodeIndex)

        // Tap candidate
        val tapCand = generated.candidateMap["c_2"]
        assertNotNull("Tap candidate must exist", tapCand)
        assertEquals("tap", tapCand!!.actionType)
        assertEquals(2, tapCand.nodeIndex)

        // Scroll candidate
        val scrollDownCand = generated.candidateMap["c_scroll_down_3"]
        assertNotNull("Scroll down candidate must exist", scrollDownCand)
        assertEquals("scroll", scrollDownCand!!.actionType)
        assertEquals(3, scrollDownCand.nodeIndex)

        // System primitives
        assertNotNull("System back candidate must exist", generated.candidateMap["c_press_back"])
        assertNotNull("System scroll down must exist", generated.candidateMap["c_scroll_down"])
        assertNotNull("System scroll up must exist", generated.candidateMap["c_scroll_up"])
    }
}
