package com.zipper.app

import com.zipper.app.core.agent.AgentAction
import com.zipper.app.core.agent.ScrollDirection
import com.zipper.app.core.model.DistilledNode
import com.zipper.app.core.model.DistilledUiState
import com.zipper.app.core.model.NormalizedBounds
import com.zipper.app.core.recorder.ActionToolDescriptor
import com.zipper.app.core.recorder.DemonstrationMetadata
import com.zipper.app.core.recorder.DemonstrationRecorder
import com.zipper.app.core.recorder.DemonstrationSerializer
import com.zipper.app.core.recorder.DemonstrationTrajectory
import com.zipper.app.core.recorder.DemonstrationValidator
import com.zipper.app.core.recorder.RecordedStep
import com.zipper.app.core.recorder.TrainingExampleConverter
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.UUID

class DemonstrationRecorderTests {

    private fun createSampleUiState(
        packageName: String = "com.example.notes",
        activityName: String = "com.example.notes.MainActivity",
        nodes: List<DistilledNode> = listOf(
            DistilledNode(
                index = 0,
                role = "BUTTON",
                text = "New note",
                contentDescription = "Create a note",
                resourceId = "com.example.notes:id/create_button",
                bounds = NormalizedBounds(800, 50, 950, 150),
                isClickable = true
            ),
            DistilledNode(
                index = 1,
                role = "EDITTEXT",
                text = "Title",
                contentDescription = null,
                resourceId = "com.example.notes:id/note_title",
                bounds = NormalizedBounds(50, 200, 950, 300),
                isClickable = true,
                isEditable = true
            ),
            DistilledNode(
                index = 2,
                role = "BUTTON",
                text = "Save",
                contentDescription = "Save note",
                resourceId = "com.example.notes:id/save_button",
                bounds = NormalizedBounds(400, 850, 600, 950),
                isClickable = true
            )
        )
    ): DistilledUiState {
        return DistilledUiState(
            packageName = packageName,
            activityName = activityName,
            nodes = nodes,
            timestamp = System.currentTimeMillis()
        )
    }

    // =========================================================================
    // 1. Recording a single action
    // =========================================================================
    @Test
    fun testRecordingSingleAction() = runBlocking {
        val recorder = DemonstrationRecorder()
        val preState = createSampleUiState(packageName = "com.android.launcher")
        val postState = createSampleUiState(packageName = "com.google.android.GoogleCamera")

        recorder.startRecording(
            goal = "Open Camera",
            initialUiState = preState,
            targetAppPackage = "com.google.android.GoogleCamera"
        )

        val step = recorder.recordStep(
            humanAction = AgentAction.OpenApp("Camera"),
            actionArguments = mapOf("app_name" to "Camera"),
            postUiState = postState
        )

        val trajectory = recorder.finishDemonstration(
            isSuccess = true,
            finalOutcomeSummary = "Camera application opened successfully"
        )

        assertEquals("Must contain exactly 1 step", 1, trajectory.totalSteps)
        assertEquals(1, trajectory.steps.size)
        assertEquals("Open Camera", trajectory.goal)
        assertTrue(trajectory.isSuccess)
        assertEquals(AgentAction.OpenApp("Camera"), step.humanAction)
        assertTrue(step.isStateChanged)
        assertTrue(step.isSuccessful)
        assertEquals("com.android.launcher", step.packageName)
        assertEquals("com.google.android.GoogleCamera", step.postObservation.packageName)
        assertTrue("Available tools must be captured", step.availableTools.isNotEmpty())
    }

    // =========================================================================
    // 2. Recording a multi-step trajectory
    // =========================================================================
    @Test
    fun testRecordingMultiStepTrajectory() = runBlocking {
        val recorder = DemonstrationRecorder()
        val state0 = createSampleUiState()
        val state1 = state0.copy(
            nodes = state0.nodes + DistilledNode(
                index = 3,
                role = "EDITTEXT",
                text = "Type note body here...",
                bounds = NormalizedBounds(50, 320, 950, 600),
                isEditable = true,
                isClickable = true
            )
        )
        val state2 = state1.copy(
            nodes = state1.nodes.map {
                if (it.index == 1) it.copy(text = "Grocery List") else it
            }
        )
        val state3 = state2.copy(
            nodes = state2.nodes + DistilledNode(
                index = 4,
                role = "TEXTVIEW",
                text = "Note saved to storage",
                bounds = NormalizedBounds(200, 800, 800, 860)
            )
        )

        recorder.startRecording(
            goal = "Create a note titled Grocery List and save it",
            initialUiState = state0
        )

        // Step 1: Tap "New note"
        recorder.recordStep(
            humanAction = AgentAction.Tap(0),
            postUiState = state1
        )

        // Step 2: Type note title
        recorder.recordStep(
            humanAction = AgentAction.TypeText(1, "Grocery List"),
            actionArguments = mapOf("text" to "Grocery List"),
            postUiState = state2
        )

        // Step 3: Tap "Save"
        recorder.recordStep(
            humanAction = AgentAction.Tap(2),
            postUiState = state3
        )

        // Step 4: Complete Goal
        recorder.recordStep(
            humanAction = AgentAction.CompleteGoal("Note saved"),
            postUiState = state3
        )

        val trajectory = recorder.finishDemonstration(
            isSuccess = true,
            finalOutcomeSummary = "Grocery List note created and saved"
        )

        assertEquals(4, trajectory.totalSteps)
        assertEquals(4, trajectory.steps.size)
        assertTrue(trajectory.isSuccess)

        // Validate step monotonicity and action correctness
        for (i in 0 until 4) {
            assertEquals("Step index must be 1-based sequential", i + 1, trajectory.steps[i].stepIndex)
        }

        assertEquals(AgentAction.Tap(0), trajectory.steps[0].humanAction)
        assertEquals(AgentAction.TypeText(1, "Grocery List"), trajectory.steps[1].humanAction)
        assertEquals(AgentAction.Tap(2), trajectory.steps[2].humanAction)
        assertEquals(AgentAction.CompleteGoal("Note saved"), trajectory.steps[3].humanAction)
    }

    // =========================================================================
    // 3. Recording a failed action and recovery
    // =========================================================================
    @Test
    fun testRecordingFailedActionAndRecovery() = runBlocking {
        val recorder = DemonstrationRecorder()
        val stateInitial = createSampleUiState()

        // State where save button is disabled or tapping it causes validation error modal
        val stateError = stateInitial.copy(
            nodes = stateInitial.nodes + DistilledNode(
                index = 5,
                role = "TEXTVIEW",
                text = "Error: Note title cannot be empty",
                bounds = NormalizedBounds(100, 400, 900, 500)
            )
        )

        // State after typing title and fixing error
        val stateFixed = stateInitial.copy(
            nodes = stateInitial.nodes.map {
                if (it.index == 1) it.copy(text = "Valid Title") else it
            }
        )

        val stateFinal = stateFixed.copy(
            nodes = stateFixed.nodes + DistilledNode(
                index = 6,
                role = "TEXTVIEW",
                text = "Saved successfully",
                bounds = NormalizedBounds(100, 900, 900, 950)
            )
        )

        recorder.startRecording(
            goal = "Save note with validation check",
            initialUiState = stateInitial
        )

        // Step 1: Premature tap on Save resulting in validation error
        val failedStep = recorder.recordStep(
            humanAction = AgentAction.Tap(2),
            postUiState = stateError,
            failureRecoveryInfo = "Validation error: Title is required before saving"
        )

        assertFalse("Step with failure info must not be marked successful", failedStep.isSuccessful)
        assertNotNull(failedStep.failureRecoveryInfo)

        // Step 2: Recovery action: Type title
        val recoveryStep = recorder.recordStep(
            humanAction = AgentAction.TypeText(1, "Valid Title"),
            postUiState = stateFixed
        )
        assertTrue(recoveryStep.isSuccessful)

        // Step 3: Tap Save again
        val finalStep = recorder.recordStep(
            humanAction = AgentAction.Tap(2),
            postUiState = stateFinal
        )
        assertTrue(finalStep.isSuccessful)

        val trajectory = recorder.finishDemonstration(
            isSuccess = true,
            finalOutcomeSummary = "Recovered from validation error and saved note"
        )

        assertEquals(3, trajectory.totalSteps)
        assertFalse("Step 1 must be marked as failed", trajectory.steps[0].isSuccessful)
        assertEquals("Validation error: Title is required before saving", trajectory.steps[0].failureRecoveryInfo)
        assertTrue("Step 2 must be successful recovery", trajectory.steps[1].isSuccessful)
        assertTrue("Step 3 must be successful final action", trajectory.steps[2].isSuccessful)
        assertTrue("Trajectory overall must be marked successful", trajectory.isSuccess)
    }

    // =========================================================================
    // 4. Serialization / Deserialization (JSON and JSONL)
    // =========================================================================
    @Test
    fun testSerializationAndDeserialization() = runBlocking {
        val recorder = DemonstrationRecorder()
        val state0 = createSampleUiState()
        val state1 = state0.copy(activityName = "com.example.notes.EditActivity")

        recorder.startRecording(goal = "Test Serialization", initialUiState = state0)
        recorder.recordStep(
            humanAction = AgentAction.Tap(0),
            actionArguments = mapOf("target" to "New note"),
            postUiState = state1
        )
        val trajectory = recorder.finishDemonstration(isSuccess = true, finalOutcomeSummary = "Done")

        val serializer = DemonstrationSerializer()

        // 1. Trajectory JSON serialization round-trip
        val jsonObject = serializer.serializeTrajectoryToJson(trajectory)
        val deserializedTrajectory = serializer.deserializeTrajectoryFromJson(jsonObject)

        assertEquals(trajectory.metadata.demonstrationId, deserializedTrajectory.metadata.demonstrationId)
        assertEquals(trajectory.metadata.schemaVersion, deserializedTrajectory.metadata.schemaVersion)
        assertEquals(trajectory.goal, deserializedTrajectory.goal)
        assertEquals(trajectory.totalSteps, deserializedTrajectory.totalSteps)
        assertEquals(trajectory.isSuccess, deserializedTrajectory.isSuccess)
        assertEquals(trajectory.finalOutcome, deserializedTrajectory.finalOutcome)
        assertEquals(trajectory.steps.size, deserializedTrajectory.steps.size)

        val origStep = trajectory.steps[0]
        val desStep = deserializedTrajectory.steps[0]
        assertEquals(origStep.stepIndex, desStep.stepIndex)
        assertEquals(origStep.humanAction, desStep.humanAction)
        assertEquals(origStep.actionArguments, desStep.actionArguments)
        assertEquals(origStep.isStateChanged, desStep.isStateChanged)
        assertEquals(origStep.isSuccessful, desStep.isSuccessful)
        assertEquals(origStep.preObservation.nodes.size, desStep.preObservation.nodes.size)
        assertEquals(origStep.postObservation.activityName, desStep.postObservation.activityName)

        // 2. Trajectory JSONL line round-trip
        val jsonlLine = serializer.trajectoryToJsonlLine(trajectory)
        assertFalse("JSONL line must not contain newlines", jsonlLine.contains("\n"))
        val fromJsonl = serializer.parseTrajectoryFromJsonlLine(jsonlLine)
        assertEquals(trajectory.goal, fromJsonl.goal)
        assertEquals(trajectory.totalSteps, fromJsonl.totalSteps)

        // 3. Training Example JSONL line round-trip
        val converter = TrainingExampleConverter()
        val examples = converter.convertTrajectoryToTrainingExamples(trajectory)
        assertEquals(1, examples.size)

        val exampleJsonl = serializer.trainingExampleToJsonlLine(examples[0])
        val parsedExample = serializer.parseTrainingExampleFromJsonlLine(exampleJsonl)
        assertEquals(examples[0].goal, parsedExample.goal)
        assertEquals(examples[0].targetAction, parsedExample.targetAction)
        assertEquals(examples[0].targetActionName, parsedExample.targetActionName)
    }

    // =========================================================================
    // 5. Schema Validation
    // =========================================================================
    @Test
    fun testSchemaValidationPreventsMalformedDemonstrations() {
        val validator = DemonstrationValidator()
        val validState = createSampleUiState()

        val validStep = RecordedStep(
            stepIndex = 1,
            goal = "Valid Goal",
            preObservation = validState,
            humanAction = AgentAction.Tap(0),
            postObservation = validState,
            isStateChanged = true,
            isSuccessful = true,
            deltaSummary = "State changed"
        )

        val validTrajectory = DemonstrationTrajectory(
            metadata = DemonstrationMetadata(demonstrationId = "test-id-1", schemaVersion = "1.0.0"),
            goal = "Valid Goal",
            totalSteps = 1,
            isSuccess = true,
            finalOutcome = "Goal Achieved",
            steps = listOf(validStep)
        )

        // 1. Valid trajectory must pass
        val validRes = validator.validate(validTrajectory)
        assertTrue("Valid trajectory must pass", validRes.isValid)
        assertTrue(validRes.errors.isEmpty())

        // 2. Blank goal must fail
        val blankGoal = validTrajectory.copy(goal = "   ")
        assertFalse("Blank goal must fail", validator.validate(blankGoal).isValid)

        // 3. Empty steps must fail
        val emptySteps = validTrajectory.copy(steps = emptyList(), totalSteps = 0)
        assertFalse("Empty steps must fail", validator.validate(emptySteps).isValid)

        // 4. Mismatched totalSteps must fail
        val mismatchedCount = validTrajectory.copy(totalSteps = 5)
        assertFalse("Mismatched totalSteps must fail", validator.validate(mismatchedCount).isValid)

        // 5. Invalid non-monotonic stepIndex must fail
        val badStepIndex = validTrajectory.copy(steps = listOf(validStep.copy(stepIndex = 3)))
        assertFalse("Non-sequential step index must fail", validator.validate(badStepIndex).isValid)

        // 6. Unsupported schema version must fail
        val badVersion = validTrajectory.copy(
            metadata = validTrajectory.metadata.copy(schemaVersion = "99.0.0")
        )
        assertFalse("Unsupported schema version must fail", validator.validate(badVersion).isValid)

        // 7. Ungrounded action targeting non-existent node must fail
        val ungroundedAction = validTrajectory.copy(
            steps = listOf(validStep.copy(humanAction = AgentAction.Tap(999))) // Node 999 does not exist
        )
        val ungroundedRes = validator.validate(ungroundedAction)
        assertFalse("Ungrounded action must fail validation", ungroundedRes.isValid)
        assertTrue(ungroundedRes.errors.any { it.contains("Ungrounded Tap action") })
    }

    // =========================================================================
    // 6. Unseen App and Package Names
    // =========================================================================
    @Test
    fun testUnseenAppAndPackageNamesAreHandledGenerically() = runBlocking {
        val recorder = DemonstrationRecorder()

        // Completely novel, unseen third-party crypto wallet app
        val novelPackage = "org.quantum.secure.vault"
        val novelActivity = "org.quantum.secure.vault.ui.SendTokensActivity"
        val novelNodes = listOf(
            DistilledNode(
                index = 0,
                role = "EDITTEXT",
                text = "Recipient Address",
                bounds = NormalizedBounds(50, 100, 950, 200),
                isEditable = true,
                isClickable = true
            ),
            DistilledNode(
                index = 1,
                role = "BUTTON",
                text = "Confirm Send",
                bounds = NormalizedBounds(200, 800, 800, 900),
                isClickable = true
            )
        )

        val preState = createSampleUiState(
            packageName = novelPackage,
            activityName = novelActivity,
            nodes = novelNodes
        )
        val postState = preState.copy(
            nodes = novelNodes.map { if (it.index == 0) it.copy(text = "0xABCDEF123456") else it }
        )

        recorder.startRecording(
            goal = "Send 10 tokens to recipient",
            initialUiState = preState,
            targetAppPackage = novelPackage
        )

        val step = recorder.recordStep(
            humanAction = AgentAction.TypeText(0, "0xABCDEF123456"),
            actionArguments = mapOf("text" to "0xABCDEF123456"),
            postUiState = postState
        )

        val trajectory = recorder.finishDemonstration(
            isSuccess = true,
            finalOutcomeSummary = "Tokens sent"
        )

        assertEquals(novelPackage, trajectory.metadata.targetAppPackage)
        assertEquals(novelPackage, step.packageName)
        assertEquals(novelActivity, step.activityName)
        assertEquals(AgentAction.TypeText(0, "0xABCDEF123456"), step.humanAction)
        assertTrue(step.isStateChanged)

        // Verify JSONL line can be parsed without issue
        val serializer = DemonstrationSerializer()
        val jsonl = serializer.trajectoryToJsonlLine(trajectory)
        val parsed = serializer.parseTrajectoryFromJsonlLine(jsonl)
        assertEquals(novelPackage, parsed.metadata.targetAppPackage)
        assertEquals(novelPackage, parsed.steps[0].packageName)
    }

    // =========================================================================
    // 7. Arbitrary Generic Actions & Training Example Conversion
    // =========================================================================
    @Test
    fun testArbitraryGenericActionsAndTrainingExampleConversion() = runBlocking {
        val recorder = DemonstrationRecorder()
        val state = createSampleUiState()

        recorder.startRecording(goal = "Execute diverse generic primitives", initialUiState = state)

        // OpenApp
        recorder.recordStep(humanAction = AgentAction.OpenApp("Calculator"), postUiState = state)
        // Tap
        recorder.recordStep(humanAction = AgentAction.Tap(0), postUiState = state)
        // TypeText
        recorder.recordStep(humanAction = AgentAction.TypeText(1, "Test text"), postUiState = state)
        // Scroll
        recorder.recordStep(humanAction = AgentAction.Scroll(0, ScrollDirection.DOWN), postUiState = state)
        // PressBack
        recorder.recordStep(humanAction = AgentAction.PressBack, postUiState = state)
        // PressHome
        recorder.recordStep(humanAction = AgentAction.PressHome, postUiState = state)
        // Wait
        recorder.recordStep(humanAction = AgentAction.Wait(1000L), postUiState = state)
        // CompleteGoal
        recorder.recordStep(humanAction = AgentAction.CompleteGoal("All primitives exercised"), postUiState = state)

        val trajectory = recorder.finishDemonstration(isSuccess = true, finalOutcomeSummary = "All primitives demonstrated")
        assertEquals(8, trajectory.totalSteps)

        val converter = TrainingExampleConverter()
        val examples = converter.convertTrajectoryToTrainingExamples(trajectory, maxHistorySteps = 3)
        assertEquals("Must produce exactly 8 training examples", 8, examples.size)

        // Check training example structure for Step 4 (Scroll)
        val ex4 = examples[3]
        assertEquals("Step 4 must have 3 history steps", 3, ex4.shortHistory.size)
        assertEquals("scroll", ex4.targetActionName)
        assertEquals("DOWN", ex4.targetActionArguments["direction"])

        // Check Step 8 (CompleteGoal)
        val ex8 = examples[7]
        assertEquals("complete_goal", ex8.targetActionName)
        assertEquals("All primitives exercised", ex8.targetActionArguments["summary"])

        // Verify training examples can be serialized to JSONL without exceptions
        val serializer = DemonstrationSerializer()
        examples.forEach { ex ->
            val line = serializer.trainingExampleToJsonlLine(ex)
            assertFalse(line.contains("\n"))
            val reparsed = serializer.parseTrainingExampleFromJsonlLine(line)
            assertEquals(ex.exampleId, reparsed.exampleId)
            assertEquals(ex.targetActionName, reparsed.targetActionName)
        }
    }
}
