package com.zipper.app

import com.zipper.app.core.agent.UniversalTaskStateEngine
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class UniversalTaskStateParityTests {

    private val engine = UniversalTaskStateEngine()

    @Test
    fun testLoginScenarioParity() {
        val goal = "Log in with username student and password secret"

        // 1. Username entered, password empty
        val nodesPartial = listOf(
            mapOf("label" to "student", "text" to "student", "editable" to true, "focused" to false),
            mapOf("label" to "Password", "text" to "", "editable" to true, "focused" to false),
            mapOf("label" to "Log In", "text" to "Log In", "editable" to false, "focused" to false)
        )
        val historyPartial = listOf(
            mapOf("action" to "type_text", "action_description" to "Type student into Username")
        )
        val state1 = engine.extractAndGround(goal, nodesPartial, historyPartial)
        assertEquals("INPUT:password", state1.currentSubgoal)
        assertEquals("NOT_READY", state1.submissionReadiness)
        assertEquals(1, state1.satisfiedRequirements.size)
        assertEquals(1, state1.pendingRequirements.size)
        assertEquals(
            "TaskState:[Done: username=student | Pending: password=secret | Subgoal: INPUT:password | Submit: NOT_READY]",
            state1.formatCanonicalString()
        )

        // 2. Both fields entered
        val nodesComplete = listOf(
            mapOf("label" to "student", "text" to "student", "editable" to true, "focused" to false),
            mapOf("label" to "••••••", "text" to "••••••", "editable" to true, "focused" to true),
            mapOf("label" to "Log In", "text" to "Log In", "editable" to false, "focused" to false)
        )
        val historyComplete = listOf(
            mapOf("action" to "type_text", "action_description" to "Type student into Username"),
            mapOf("action" to "type_text", "action_description" to "Type secret into Password")
        )
        val state2 = engine.extractAndGround(goal, nodesComplete, historyComplete)
        assertEquals("SUBMIT", state2.currentSubgoal)
        assertEquals("READY", state2.submissionReadiness)
        assertEquals(2, state2.satisfiedRequirements.size)
        assertEquals(0, state2.pendingRequirements.size)
        assertEquals(
            "TaskState:[Done: username=student, password=secret | Pending: none | Subgoal: SUBMIT | Submit: READY]",
            state2.formatCanonicalString()
        )
    }

    @Test
    fun testRegistrationScenarioParity() {
        val goal = "Register with first name John, last name Smith, email john@example.com"

        // Step 1: First name only
        val nodes1 = listOf(
            mapOf("label" to "John", "text" to "John", "editable" to true),
            mapOf("label" to "Last Name", "text" to "", "editable" to true),
            mapOf("label" to "Email Address", "text" to "", "editable" to true)
        )
        val hist1 = listOf(mapOf("action" to "type_text", "action_description" to "Type John into First Name"))
        val state1 = engine.extractAndGround(goal, nodes1, hist1)
        assertEquals("INPUT:last_name", state1.currentSubgoal)
        assertEquals("NOT_READY", state1.submissionReadiness)
        assertEquals(1, state1.satisfiedRequirements.size)
        assertEquals(2, state1.pendingRequirements.size)

        // Step 2: First + Last
        val nodes2 = listOf(
            mapOf("label" to "John", "text" to "John", "editable" to true),
            mapOf("label" to "Smith", "text" to "Smith", "editable" to true),
            mapOf("label" to "Email Address", "text" to "", "editable" to true)
        )
        val hist2 = hist1 + listOf(mapOf("action" to "type_text", "action_description" to "Type Smith into Last Name"))
        val state2 = engine.extractAndGround(goal, nodes2, hist2)
        assertEquals("INPUT:email", state2.currentSubgoal)
        assertEquals("NOT_READY", state2.submissionReadiness)
        assertEquals(2, state2.satisfiedRequirements.size)
        assertEquals(1, state2.pendingRequirements.size)

        // Step 3: All 3 fields
        val nodes3 = listOf(
            mapOf("label" to "John", "text" to "John", "editable" to true),
            mapOf("label" to "Smith", "text" to "Smith", "editable" to true),
            mapOf("label" to "john@example.com", "text" to "john@example.com", "editable" to true)
        )
        val hist3 = hist2 + listOf(mapOf("action" to "type_text", "action_description" to "Type john@example.com into Email"))
        val state3 = engine.extractAndGround(goal, nodes3, hist3)
        assertEquals("SUBMIT", state3.currentSubgoal)
        assertEquals("READY", state3.submissionReadiness)
        assertEquals(3, state3.satisfiedRequirements.size)
        assertEquals(0, state3.pendingRequirements.size)
        assertEquals(
            "TaskState:[Done: first_name=John, last_name=Smith, email=john@example.com | Pending: none | Subgoal: SUBMIT | Submit: READY]",
            state3.formatCanonicalString()
        )
    }

    @Test
    fun testYouTubeAndChromeScenarioParity() {
        // YouTube
        val ytGoal = "Open YouTube and search lo-fi hip hop"
        val ytNodes = listOf(mapOf("label" to "lo-fi hip hop", "text" to "lo-fi hip hop", "editable" to true))
        val ytState = engine.extractAndGround(ytGoal, ytNodes, emptyList())
        assertEquals("SUBMIT", ytState.currentSubgoal)
        assertEquals("READY", ytState.submissionReadiness)
        assertEquals("TaskState:[Done: query=lo-fi hip hop | Pending: none | Subgoal: SUBMIT | Submit: READY]", ytState.formatCanonicalString())

        // Chrome
        val chromeGoal = "Open Chrome and go to wikipedia.org"
        val chromeNodes = listOf(mapOf("label" to "wikipedia.org", "text" to "wikipedia.org", "editable" to true))
        val chromeState = engine.extractAndGround(chromeGoal, chromeNodes, emptyList())
        assertEquals("SUBMIT", chromeState.currentSubgoal)
        assertEquals("READY", chromeState.submissionReadiness)
        assertEquals("TaskState:[Done: url=wikipedia.org | Pending: none | Subgoal: SUBMIT | Submit: READY]", chromeState.formatCanonicalString())
    }

    @Test
    fun testNegativeConstraintMessageParity() {
        val goalNeg = "Type I will be late into message box but do not send"
        val nodes = listOf(mapOf("label" to "I will be late", "text" to "I will be late", "editable" to true))
        val stateNeg = engine.extractAndGround(goalNeg, nodes, emptyList())
        assertEquals("COMPLETE_WITHOUT_SUBMIT", stateNeg.currentSubgoal)
        assertEquals("NOT_READY", stateNeg.submissionReadiness)
        assertTrue(stateNeg.negativeConstraint)

        val goalPos = "Send a message saying I will be late"
        val statePos = engine.extractAndGround(goalPos, nodes, emptyList())
        assertEquals("SUBMIT", statePos.currentSubgoal)
        assertEquals("READY", statePos.submissionReadiness)
        assertFalse(statePos.negativeConstraint)
    }
}
