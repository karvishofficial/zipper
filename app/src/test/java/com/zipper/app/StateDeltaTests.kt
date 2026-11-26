package com.zipper.app

import com.zipper.app.core.agent.AgentAction
import com.zipper.app.core.agent.StateDifferentialEngine
import com.zipper.app.core.agent.StepOutcome
import com.zipper.app.core.model.DistilledNode
import com.zipper.app.core.model.DistilledUiState
import com.zipper.app.core.model.NormalizedBounds
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Step 2: Factual State Delta Unit Tests
 *
 * Validates that the runtime captures objective, lossless state diffs
 * across button taps, text entries, focus transitions, and element additions
 * with zero hardcoded semantic interpretations.
 */
class StateDeltaTests {

    private val differentialEngine = StateDifferentialEngine()

    /**
     * Test 1: Calculator Formula Text Diff on Button Tap
     *
     * In Calculator, tapping a digit button updates the formula input field.
     * The delta engine must detect this text change even though the action was a Tap,
     * not a TypeText.
     */
    @Test
    fun testCalculatorFormulaTextDiffOnTap() {
        val formulaPre = DistilledNode(
            index = 3,
            role = "EDITTEXT",
            text = "0",
            resourceId = "com.sec.android.app.popupcalculator:id/calc_edt_formula",
            bounds = NormalizedBounds(50, 100, 950, 250)
        )
        val digit2 = DistilledNode(
            index = 10,
            role = "BUTTON",
            text = "2",
            resourceId = "com.sec.android.app.popupcalculator:id/calc_keypad_btn_02",
            bounds = NormalizedBounds(350, 600, 500, 700)
        )

        val preState = DistilledUiState(
            packageName = "com.sec.android.app.popupcalculator",
            nodes = listOf(formulaPre, digit2)
        )

        // After tapping "2", the formula displays "2"
        val formulaPost = formulaPre.copy(text = "2")
        val postState = preState.copy(nodes = listOf(formulaPost, digit2))

        val delta = differentialEngine.computeDelta(
            preState = preState,
            postState = postState,
            action = AgentAction.Tap(targetNodeIndex = 10, candidateId = "c_10", semanticLabel = "2")
        )

        assertFalse("Action should not be NO-OP", delta.isNoOp)
        assertTrue("Observation should be changed", delta.observationChanged)
        assertEquals(StepOutcome.SUCCESS, delta.outcome)
        assertEquals(1, delta.textChanges.size)

        val textDiff = delta.textChanges[0]
        assertEquals(3, textDiff.nodeIndex)
        assertEquals("calc_edt_formula", textDiff.label)
        assertEquals("0", textDiff.oldText)
        assertEquals("2", textDiff.newText)

        val summary = delta.formatFactualSummary()
        assertTrue("Summary should contain formula text change", summary.contains("calc_edt_formula ('0' -> '2')"))
    }

    /**
     * Test 2: Focus Transition Detection
     *
     * When tapping a search button or input, the input element gains focus.
     */
    @Test
    fun testFocusTransitionDetection() {
        val searchBoxPre = DistilledNode(
            index = 5,
            role = "EDITTEXT",
            text = "",
            isFocused = false,
            resourceId = "com.google.android.youtube:id/search_edit_text",
            bounds = NormalizedBounds(100, 50, 900, 120)
        )
        val preState = DistilledUiState(
            packageName = "com.google.android.youtube",
            nodes = listOf(searchBoxPre)
        )

        val searchBoxPost = searchBoxPre.copy(isFocused = true)
        val postState = preState.copy(nodes = listOf(searchBoxPost))

        val delta = differentialEngine.computeDelta(
            preState = preState,
            postState = postState,
            action = AgentAction.Tap(targetNodeIndex = 5, candidateId = "c_5", semanticLabel = "Search")
        )

        assertFalse("Action should not be NO-OP", delta.isNoOp)
        assertTrue("Observation should be changed", delta.observationChanged)
        assertEquals(1, delta.focusChanges.size)

        val focusDiff = delta.focusChanges[0]
        assertEquals(5, focusDiff.nodeIndex)
        assertFalse(focusDiff.hadFocus)
        assertTrue(focusDiff.hasFocus)

        val summary = delta.formatFactualSummary()
        assertTrue("Summary should report focus", summary.contains("focused"))
    }

    /**
     * Test 3: NO-OP Detection when Tapping Ineffective Button
     *
     * If the screen state before and after are completely unchanged (e.g. tapping "0" when "0" is already displayed),
     * the delta engine must explicitly mark isNoOp = true and observationChanged = false.
     */
    @Test
    fun testNoOpDetectionOnUnchangedScreen() {
        val formula = DistilledNode(
            index = 3,
            role = "EDITTEXT",
            text = "0",
            resourceId = "com.sec.android.app.popupcalculator:id/calc_edt_formula",
            bounds = NormalizedBounds(50, 100, 950, 250)
        )
        val digit0 = DistilledNode(
            index = 20,
            role = "BUTTON",
            text = "0",
            resourceId = "com.sec.android.app.popupcalculator:id/calc_keypad_btn_00",
            bounds = NormalizedBounds(350, 800, 500, 900)
        )

        val preState = DistilledUiState(
            packageName = "com.sec.android.app.popupcalculator",
            nodes = listOf(formula, digit0)
        )
        // Exact identical post-state
        val postState = preState.copy()

        val delta = differentialEngine.computeDelta(
            preState = preState,
            postState = postState,
            action = AgentAction.Tap(targetNodeIndex = 20, candidateId = "c_20", semanticLabel = "0")
        )

        assertTrue("Action must be marked as NO-OP", delta.isNoOp)
        assertFalse("Observation must NOT be marked as changed", delta.observationChanged)
        assertEquals(StepOutcome.NO_OP, delta.outcome)
        assertEquals("NO-OP: Screen state unchanged", delta.formatFactualSummary())
    }

    /**
     * Test 4: Element Set Diff (Search Results Appearance)
     *
     * Submitting a search query reveals a list of new result elements.
     */
    @Test
    fun testSearchResultsAppearanceDiff() {
        val searchBar = DistilledNode(
            index = 1,
            role = "EDITTEXT",
            text = "music",
            resourceId = "com.google.android.youtube:id/search_edit_text",
            bounds = NormalizedBounds(100, 50, 900, 120)
        )

        val preState = DistilledUiState(
            packageName = "com.google.android.youtube",
            nodes = listOf(searchBar)
        )

        // 15 video cards appear in postState
        val videoCards = (10..24).map { i ->
            DistilledNode(
                index = i,
                role = "VIEW",
                text = "Video item #$i - Artist Track",
                resourceId = "com.google.android.youtube:id/video_card_$i",
                bounds = NormalizedBounds(0, 150 + (i - 10) * 80, 1000, 220 + (i - 10) * 80),
                isClickable = true
            )
        }

        val postState = preState.copy(nodes = listOf(searchBar) + videoCards)

        val delta = differentialEngine.computeDelta(
            preState = preState,
            postState = postState,
            action = AgentAction.Submit
        )

        assertFalse(delta.isNoOp)
        assertTrue(delta.observationChanged)
        assertTrue(delta.screenChanged)
        assertEquals(15, delta.nodesAddedCount)
        assertEquals(0, delta.nodesRemovedCount)

        val summary = delta.formatFactualSummary()
        assertTrue("Summary should report added elements count", summary.contains("+15 element(s)"))
    }
}
