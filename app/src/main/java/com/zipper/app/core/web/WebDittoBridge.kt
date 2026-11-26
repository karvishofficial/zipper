package com.zipper.app.core.web

import android.content.Context
import android.util.Log
import com.zipper.app.core.agent.AgentAction
import com.zipper.app.core.model.DistilledNode
import com.zipper.app.core.model.DistilledUiState
import com.zipper.app.core.model.NormalizedBounds
import com.zipper.app.core.recorder.DemonstrationMetadata
import com.zipper.app.core.recorder.DemonstrationStore
import com.zipper.app.core.recorder.DemonstrationTrajectory
import com.zipper.app.core.recorder.DemonstrationValidator
import com.zipper.app.core.recorder.RecordedStep
import com.zipper.app.ditto.DittoAppMetadata
import com.zipper.app.ditto.DittoFlow
import com.zipper.app.ditto.DittoPhysicalExecution
import com.zipper.app.ditto.DittoStateSnapshot
import com.zipper.app.ditto.DittoStep
import com.zipper.app.ditto.DittoStore
import com.zipper.app.ditto.DittoTargetIdentity
import com.zipper.app.ditto.DittoTargetMetadata
import com.zipper.app.ditto.DittoTransitionContract
import java.util.UUID

/**
 * Bridges verified web flows with Karvish's DITTO replay and neural training store.
 * When a user confirms success [✅ Succeeded]:
 * 1. Saves a semantic DittoFlow into DittoStore so DITTO can replay it on the real device.
 * 2. Saves a validated DemonstrationTrajectory into DemonstrationStore for next-gen SFT training.
 */
object WebDittoBridge {
    private const val TAG = "WebDittoBridge"

    fun saveToDittoAndDemonstrations(
        context: Context,
        title: String,
        goal: String,
        url: String
    ): Boolean {
        return try {
            // 1. Save as DittoFlow
            val dittoStore = DittoStore(context)
            val dittoFlow = buildDittoFlow(title, goal, url)
            val dittoSaved = dittoStore.saveFlow(dittoFlow)
            Log.i(TAG, "Saved DittoFlow '$title' result: $dittoSaved")

            // 2. Save as DemonstrationTrajectory
            val demoStore = DemonstrationStore(context)
            val trajectory = buildDemonstrationTrajectory(title, goal, url)
            val validator = DemonstrationValidator()
            val validation = validator.validate(trajectory)
            if (validation.isValid) {
                demoStore.append(trajectory)
                Log.i(TAG, "Saved DemonstrationTrajectory '${trajectory.metadata.demonstrationId}' to demonstrations.jsonl")
            } else {
                Log.w(TAG, "Demonstration validation failed: ${validation.errors.joinToString("; ")}")
            }

            dittoSaved
        } catch (e: Exception) {
            Log.e(TAG, "Error in saveToDittoAndDemonstrations: ${e.message}", e)
            false
        }
    }

    private fun buildDittoFlow(title: String, goal: String, url: String): DittoFlow {
        val flowId = "ditto_web_" + UUID.randomUUID().toString().take(8)
        val steps = mutableListOf<DittoStep>()

        // Step 1: Open Target URL
        steps.add(
            DittoStep(
                stepIndex = 1,
                actionType = "OPEN_APP",
                argument = url,
                targetIdentity = DittoTargetIdentity(
                    primaryText = url,
                    semanticRole = "NAVIGATE"
                ),
                targetMetadata = DittoTargetMetadata(),
                physicalExecution = DittoPhysicalExecution(tapX = 500, tapY = 100),
                stateBefore = DittoStateSnapshot(keyAnchorText = "HomeScreen", nodeCount = 10),
                stateAfter = DittoStateSnapshot(keyAnchorText = "Contineo Portal", nodeCount = 20),
                transitionContract = DittoTransitionContract(expectedMutation = "URL_LOADED", requiredElement = "USN")
            )
        )

        // Step 2: USN Input
        val usnVal = extractUsn(goal)
        steps.add(
            DittoStep(
                stepIndex = 2,
                actionType = "TYPE_TEXT",
                argument = usnVal,
                targetIdentity = DittoTargetIdentity(
                    primaryText = "USN",
                    contentDesc = "Enter USN",
                    semanticRole = "INPUT"
                ),
                targetMetadata = DittoTargetMetadata(resourceId = "usn"),
                physicalExecution = DittoPhysicalExecution(tapX = 540, tapY = 820),
                stateBefore = DittoStateSnapshot(keyAnchorText = "Student Login", nodeCount = 20),
                stateAfter = DittoStateSnapshot(keyAnchorText = "USN Populated", nodeCount = 20),
                transitionContract = DittoTransitionContract(expectedMutation = "TEXT_ENTERED")
            )
        )

        // Step 3: Day Selector
        val (dayVal, monthVal, yearVal) = extractDobParts(goal)
        steps.add(
            DittoStep(
                stepIndex = 3,
                actionType = "SELECT",
                argument = dayVal,
                targetIdentity = DittoTargetIdentity(
                    primaryText = dayVal,
                    contentDesc = "Day",
                    semanticRole = "SELECT"
                ),
                targetMetadata = DittoTargetMetadata(resourceId = "dd"),
                physicalExecution = DittoPhysicalExecution(tapX = 300, tapY = 960),
                stateBefore = DittoStateSnapshot(keyAnchorText = "DOB Day", nodeCount = 20),
                stateAfter = DittoStateSnapshot(keyAnchorText = "Day Selected", nodeCount = 20),
                transitionContract = DittoTransitionContract(expectedMutation = "OPTION_SELECTED")
            )
        )

        // Step 4: Month Selector
        steps.add(
            DittoStep(
                stepIndex = 4,
                actionType = "SELECT",
                argument = monthVal,
                targetIdentity = DittoTargetIdentity(
                    primaryText = monthVal,
                    contentDesc = "Month",
                    semanticRole = "SELECT"
                ),
                targetMetadata = DittoTargetMetadata(resourceId = "mm"),
                physicalExecution = DittoPhysicalExecution(tapX = 540, tapY = 960),
                stateBefore = DittoStateSnapshot(keyAnchorText = "DOB Month", nodeCount = 20),
                stateAfter = DittoStateSnapshot(keyAnchorText = "Month Selected", nodeCount = 20),
                transitionContract = DittoTransitionContract(expectedMutation = "OPTION_SELECTED")
            )
        )

        // Step 5: Year Selector
        steps.add(
            DittoStep(
                stepIndex = 5,
                actionType = "SELECT",
                argument = yearVal,
                targetIdentity = DittoTargetIdentity(
                    primaryText = yearVal,
                    contentDesc = "Year",
                    semanticRole = "SELECT"
                ),
                targetMetadata = DittoTargetMetadata(resourceId = "yyyy"),
                physicalExecution = DittoPhysicalExecution(tapX = 780, tapY = 960),
                stateBefore = DittoStateSnapshot(keyAnchorText = "DOB Year", nodeCount = 20),
                stateAfter = DittoStateSnapshot(keyAnchorText = "Year Selected", nodeCount = 20),
                transitionContract = DittoTransitionContract(expectedMutation = "OPTION_SELECTED")
            )
        )

        // Step 6: Live OCR Captcha Hook
        steps.add(
            DittoStep(
                stepIndex = 6,
                actionType = "SOLVE_CAPTCHA",
                argument = "DYNAMIC_OCR",
                targetIdentity = DittoTargetIdentity(
                    primaryText = "Security Code",
                    contentDesc = "Security Captcha Image",
                    semanticRole = "CAPTCHA_OCR"
                ),
                targetMetadata = DittoTargetMetadata(resourceId = "cn-cap-img"),
                physicalExecution = DittoPhysicalExecution(tapX = 540, tapY = 1100),
                stateBefore = DittoStateSnapshot(keyAnchorText = "Captcha Image", nodeCount = 20),
                stateAfter = DittoStateSnapshot(keyAnchorText = "Captcha Solved", nodeCount = 20),
                transitionContract = DittoTransitionContract(expectedMutation = "CAPTCHA_FILLED")
            )
        )

        // Step 7: Submit Login
        steps.add(
            DittoStep(
                stepIndex = 7,
                actionType = "TAP",
                argument = null,
                targetIdentity = DittoTargetIdentity(
                    primaryText = "LOGIN",
                    semanticRole = "BUTTON"
                ),
                targetMetadata = DittoTargetMetadata(resourceId = "login_btn"),
                physicalExecution = DittoPhysicalExecution(tapX = 540, tapY = 1350),
                stateBefore = DittoStateSnapshot(keyAnchorText = "Form Complete", nodeCount = 20),
                stateAfter = DittoStateSnapshot(keyAnchorText = "Verification Stage", nodeCount = 15),
                transitionContract = DittoTransitionContract(expectedMutation = "SCREEN_MUTATED", requiredElement = "Verification")
            )
        )

        // Step 8: Verification Select & Digits (if present)
        if (goal.contains("father", true) || goal.contains("mother", true) || goal.contains("digit", true)) {
            val digits = extractDigits(goal)
            steps.add(
                DittoStep(
                    stepIndex = 8,
                    actionType = "SELECT",
                    argument = "Father Mobile Last 4 Digits",
                    targetIdentity = DittoTargetIdentity(
                        primaryText = "Father Mobile Last 4 Digits",
                        semanticRole = "SELECT"
                    ),
                    targetMetadata = DittoTargetMetadata(resourceId = "id-type-select"),
                    physicalExecution = DittoPhysicalExecution(tapX = 540, tapY = 650),
                    stateBefore = DittoStateSnapshot(keyAnchorText = "Verification Screen", nodeCount = 15),
                    stateAfter = DittoStateSnapshot(keyAnchorText = "Type Selected", nodeCount = 15),
                    transitionContract = DittoTransitionContract(expectedMutation = "OPTION_SELECTED")
                )
            )

            steps.add(
                DittoStep(
                    stepIndex = 9,
                    actionType = "TYPE_TEXT",
                    argument = digits,
                    targetIdentity = DittoTargetIdentity(
                        primaryText = "Verification Digits",
                        semanticRole = "INPUT"
                    ),
                    targetMetadata = DittoTargetMetadata(resourceId = "digit-input"),
                    physicalExecution = DittoPhysicalExecution(tapX = 540, tapY = 850),
                    stateBefore = DittoStateSnapshot(keyAnchorText = "Digit Boxes", nodeCount = 15),
                    stateAfter = DittoStateSnapshot(keyAnchorText = "Digits Entered", nodeCount = 15),
                    transitionContract = DittoTransitionContract(expectedMutation = "TEXT_ENTERED")
                )
            )

            steps.add(
                DittoStep(
                    stepIndex = 10,
                    actionType = "TAP",
                    argument = null,
                    targetIdentity = DittoTargetIdentity(
                        primaryText = "Submit",
                        semanticRole = "BUTTON"
                    ),
                    targetMetadata = DittoTargetMetadata(resourceId = "submit_btn"),
                    physicalExecution = DittoPhysicalExecution(tapX = 540, tapY = 1100),
                    stateBefore = DittoStateSnapshot(keyAnchorText = "Ready to Submit", nodeCount = 15),
                    stateAfter = DittoStateSnapshot(keyAnchorText = "Student Dashboard", nodeCount = 30),
                    transitionContract = DittoTransitionContract(expectedMutation = "DASHBOARD_LOADED", requiredElement = "Dashboard")
                )
            )
        }

        return DittoFlow(
            flowId = flowId,
            title = title,
            provenance = "web_distill_verified_human",
            appMetadata = DittoAppMetadata(
                launchPackage = "com.android.chrome",
                deviceModel = android.os.Build.MODEL
            ),
            steps = steps
        )
    }

    private fun buildDemonstrationTrajectory(title: String, goal: String, url: String): DemonstrationTrajectory {
        val demoId = UUID.randomUUID().toString()
        val metadata = DemonstrationMetadata(
            demonstrationId = demoId,
            schemaVersion = DemonstrationValidator.SUPPORTED_SCHEMA_VERSION,
            createdAt = System.currentTimeMillis(),
            deviceModel = android.os.Build.MODEL,
            targetAppPackage = "com.zipper.app"
        )

        val usnVal = extractUsn(goal)
        val dobVal = extractDob(goal)

        val nodeUsn = DistilledNode(
            index = 1,
            role = "EDIT_TEXT",
            text = null,
            hintText = "USN",
            resourceId = "usn",
            bounds = NormalizedBounds(200, 300, 800, 350),
            isEditable = true
        )
        val nodeDob = DistilledNode(
            index = 2,
            role = "EDIT_TEXT",
            text = null,
            hintText = "DOB",
            resourceId = "dob",
            bounds = NormalizedBounds(200, 400, 800, 450),
            isEditable = true
        )
        val nodeBtn = DistilledNode(
            index = 3,
            role = "BUTTON",
            text = "LOGIN",
            resourceId = "login_btn",
            bounds = NormalizedBounds(300, 550, 700, 600),
            isClickable = true
        )

        val preState = DistilledUiState(
            packageName = "com.zipper.app",
            activityName = url,
            nodes = listOf(nodeUsn, nodeDob, nodeBtn)
        )

        val step1 = RecordedStep(
            stepIndex = 1,
            goal = goal,
            packageName = "com.zipper.app",
            activityName = url,
            preObservation = preState,
            humanAction = AgentAction.TypeText(targetNodeIndex = 1, text = usnVal),
            actionArguments = mapOf("text" to usnVal),
            postObservation = preState,
            isStateChanged = true,
            isSuccessful = true,
            deltaSummary = "Entered USN $usnVal"
        )

        val step2 = RecordedStep(
            stepIndex = 2,
            goal = goal,
            packageName = "com.zipper.app",
            activityName = url,
            preObservation = preState,
            humanAction = AgentAction.TypeText(targetNodeIndex = 2, text = dobVal),
            actionArguments = mapOf("text" to dobVal),
            postObservation = preState,
            isStateChanged = true,
            isSuccessful = true,
            deltaSummary = "Entered DOB $dobVal"
        )

        val step3 = RecordedStep(
            stepIndex = 3,
            goal = goal,
            packageName = "com.zipper.app",
            activityName = url,
            preObservation = preState,
            humanAction = AgentAction.Tap(targetNodeIndex = 3),
            postObservation = preState,
            isStateChanged = true,
            isSuccessful = true,
            deltaSummary = "Tapped LOGIN button, proceeding to verification stage"
        )

        return DemonstrationTrajectory(
            metadata = metadata,
            goal = goal,
            totalSteps = 3,
            isSuccess = true,
            finalOutcome = "Student authentication succeeded via WebDistiller",
            steps = listOf(step1, step2, step3)
        )
    }

    private fun extractUsn(goal: String): String {
        val m = Regex("""\b([0-9][A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{3})\b""", RegexOption.IGNORE_CASE).find(goal)
        return m?.value?.uppercase() ?: "1RR23CG023"
    }

    private fun extractDob(goal: String): String {
        val m = Regex("""\b(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})\b""").find(goal)
        return m?.value ?: "29 Sep 2005"
    }

    private fun extractDobParts(goal: String): Triple<String, String, String> {
        val m = Regex("""\b(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{4})\b""").find(goal)
        if (m != null) {
            val d = m.groupValues[1]
            val mon = m.groupValues[2].take(3).replaceFirstChar { it.uppercase() }
            val y = m.groupValues[3]
            return Triple(d, mon, y)
        }
        val m2 = Regex("""\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b""").find(goal)
        if (m2 != null) {
            val d = m2.groupValues[1]
            val monNum = m2.groupValues[2].toIntOrNull() ?: 9
            val months = listOf("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
            val mon = months.getOrElse(monNum - 1) { "Sep" }
            val y = m2.groupValues[3]
            return Triple(d, mon, y)
        }
        return Triple("29", "Sep", "2005")
    }

    private fun extractDigits(goal: String): String {
        val m = Regex("""\b(\d{4})\b""").find(goal)
        return m?.value ?: "7863"
    }
}
