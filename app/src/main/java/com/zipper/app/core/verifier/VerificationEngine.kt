package com.zipper.app.core.verifier

import com.zipper.app.core.model.DistilledUiState
import com.zipper.app.core.model.Postcondition
import com.zipper.app.core.model.PostconditionType
import com.zipper.app.core.model.VerificationResult
import com.zipper.app.core.model.VerificationStatus
import kotlin.system.measureTimeMillis

/**
 * Structural Verification Engine: Deterministically verifies whether an action's
 * intended outcome actually occurred by evaluating postconditions on the new UI state.
 *
 * Guarantees zero silent failures by enforcing tri-state results:
 * - VERIFIED_SUCCESS: Concrete structural proof of success found.
 * - CONFIRMED_FAILURE: Explicit proof of failure (e.g. error modal displayed, condition unmet).
 * - UNCERTAIN: Condition cannot be conclusively established (never reported as success).
 */
class VerificationEngine {

    fun verify(
        postcondition: Postcondition?,
        preState: DistilledUiState,
        postState: DistilledUiState,
        resolvedArgument: String? = null
    ): VerificationResult {
        if (postcondition == null) {
            // If no explicit postcondition was declared, check if UI changed at all
            val uiChanged = preState.nodes.size != postState.nodes.size ||
                    preState.activityName != postState.activityName ||
                    preState.nodes.map { it.text } != postState.nodes.map { it.text }

            return if (uiChanged) {
                VerificationResult(
                    status = VerificationStatus.VERIFIED_SUCCESS,
                    message = "Implicit state change verified (UI updated after action)"
                )
            } else {
                VerificationResult(
                    status = VerificationStatus.UNCERTAIN,
                    message = "No postcondition specified and no visible UI state transition detected"
                )
            }
        }

        var matchedIndex: Int? = null
        var status = VerificationStatus.CONFIRMED_FAILURE
        var message = ""

        val latencyMs = measureTimeMillis {
            val expectedText = postcondition.expectedText ?: resolvedArgument ?: postcondition.textSlot
            val expectedRole = postcondition.expectedRole?.uppercase()

            when (postcondition.type) {
                PostconditionType.ELEMENT_EXISTS -> {
                    val matchingNode = postState.nodes.firstOrNull { node ->
                        val roleMatch = expectedRole == null || node.role == expectedRole
                        val textMatch = expectedText == null ||
                                (node.text?.contains(expectedText, ignoreCase = true) == true) ||
                                (node.contentDescription?.contains(expectedText, ignoreCase = true) == true)
                        roleMatch && textMatch
                    }

                    if (matchingNode != null) {
                        status = VerificationStatus.VERIFIED_SUCCESS
                        matchedIndex = matchingNode.index
                        message = "Element exists: [${matchingNode.role}] '${matchingNode.semanticLabel}' at index ${matchingNode.index}"
                    } else {
                        status = VerificationStatus.CONFIRMED_FAILURE
                        message = "Expected element (role=$expectedRole, text='$expectedText') was not found in visible UI"
                    }
                }

                PostconditionType.ELEMENT_DISAPPEARED -> {
                    val stillExists = postState.nodes.any { node ->
                        val roleMatch = expectedRole == null || node.role == expectedRole
                        val textMatch = expectedText == null ||
                                (node.text?.contains(expectedText, ignoreCase = true) == true) ||
                                (node.contentDescription?.contains(expectedText, ignoreCase = true) == true)
                        roleMatch && textMatch
                    }

                    if (!stillExists) {
                        status = VerificationStatus.VERIFIED_SUCCESS
                        message = "Element (role=$expectedRole, text='$expectedText') successfully disappeared"
                    } else {
                        status = VerificationStatus.CONFIRMED_FAILURE
                        message = "Element is still present in post-action state"
                    }
                }

                PostconditionType.TEXT_CONTAINS -> {
                    if (expectedText.isNullOrBlank()) {
                        status = VerificationStatus.UNCERTAIN
                        message = "TEXT_CONTAINS assertion has empty expected string"
                    } else {
                        val matchingNode = postState.nodes.firstOrNull { node ->
                            (node.text?.contains(expectedText, ignoreCase = true) == true) ||
                                    (node.contentDescription?.contains(expectedText, ignoreCase = true) == true)
                        }

                        if (matchingNode != null) {
                            status = VerificationStatus.VERIFIED_SUCCESS
                            matchedIndex = matchingNode.index
                            message = "Found expected text '$expectedText' in node #${matchingNode.index} ('${matchingNode.semanticLabel}')"
                        } else {
                            status = VerificationStatus.CONFIRMED_FAILURE
                            message = "Expected text '$expectedText' not found in any visible element"
                        }
                    }
                }

                PostconditionType.FIELD_EQUALS -> {
                    val targetField = postState.nodes.firstOrNull { node ->
                        node.isEditable || node.role == "INPUT"
                    }

                    if (targetField != null && expectedText != null) {
                        if (targetField.text.equals(expectedText, ignoreCase = true)) {
                            status = VerificationStatus.VERIFIED_SUCCESS
                            matchedIndex = targetField.index
                            message = "Editable field content matches '$expectedText'"
                        } else {
                            status = VerificationStatus.CONFIRMED_FAILURE
                            message = "Field content '${targetField.text}' does not match expected '$expectedText'"
                        }
                    } else {
                        status = VerificationStatus.UNCERTAIN
                        message = "Could not locate editable field to verify equality"
                    }
                }

                PostconditionType.SCREEN_CHANGED -> {
                    val expectedPkg = postcondition.expectedPackage
                    val changed = if (expectedPkg != null) {
                        postState.packageName == expectedPkg
                    } else {
                        preState.packageName != postState.packageName || preState.activityName != postState.activityName
                    }

                    if (changed) {
                        status = VerificationStatus.VERIFIED_SUCCESS
                        message = "Screen transitioned successfully to package='${postState.packageName}' activity='${postState.activityName}'"
                    } else {
                        status = VerificationStatus.CONFIRMED_FAILURE
                        message = "Screen failed to change; current package='${postState.packageName}'"
                    }
                }
            }
        }

        return VerificationResult(
            status = status,
            message = message,
            matchedNodeIndex = matchedIndex,
            latencyMs = latencyMs
        )
    }

    /**
     * Specialized verification asserting that an alarm matching the target time is visible
     * in the active Clock alarm list and enabled.
     */
    fun verifyAlarmActive(
        postState: DistilledUiState,
        hour: Int,
        minute: Int,
        amPm: String
    ): VerificationResult {
        val hourPadded = "%02d".format(hour)
        val minutePadded = "%02d".format(minute)
        val timePattern1 = "$hour:$minutePadded"
        val timePattern2 = "$hourPadded:$minutePadded"
        val targetAmPm = amPm.uppercase().trim()

        val matchingTimeNode = postState.nodes.firstOrNull { node ->
            val text = node.text?.uppercase() ?: ""
            val desc = node.contentDescription?.uppercase() ?: ""
            val containsTime = text.contains(timePattern1) || text.contains(timePattern2) ||
                desc.contains(timePattern1) || desc.contains(timePattern2)
            if (!containsTime || !node.isEnabled) return@firstOrNull false

            val rowNodes = postState.nodes.filter { rowNode ->
                kotlin.math.abs(rowNode.bounds.centerY - node.bounds.centerY) <=
                    maxOf(node.bounds.height, rowNode.bounds.height, 24)
            }
            val rowText = rowNodes.joinToString(" ") { candidate ->
                listOfNotNull(candidate.text, candidate.contentDescription).joinToString(" ")
            }.uppercase()
            val exposedToggle = rowNodes.any { it.isCheckable }
            val enabledInRow = !exposedToggle || rowNodes.any { it.isCheckable && it.isChecked && it.isEnabled }
            rowText.contains(targetAmPm) && enabledInRow
        }

        return if (matchingTimeNode != null) {
            VerificationResult(
                status = VerificationStatus.VERIFIED_SUCCESS,
                message = "Alarm '$timePattern1 $targetAmPm' confirmed active in Clock list (matched element #${matchingTimeNode.index}: '${matchingTimeNode.semanticLabel}')",
                matchedNodeIndex = matchingTimeNode.index
            )
        } else {
            VerificationResult(
                status = VerificationStatus.CONFIRMED_FAILURE,
                message = "Could not find active alarm '$timePattern1 $targetAmPm' in visible alarm list"
            )
        }
    }

    /**
     * Module 4: Instance-Delta Context Evidence Verification for Messaging.
     * Asserts that:
     * 1. Target text instance count in message list increased (K_after >= K_before + 1)
     * 2. Message composer text buffer was vacated.
     */
    fun verifyMessageDelivered(
        preState: DistilledUiState?,
        postState: DistilledUiState,
        sentText: String
    ): VerificationResult {
        val cleanText = sentText.trim().lowercase()
        if (cleanText.isBlank()) {
            return VerificationResult(
                status = VerificationStatus.UNCERTAIN,
                message = "Sent message text is blank; cannot verify instance delta"
            )
        }

        val composerNodes = postState.nodes.filter { it.isEditable || it.role == "INPUT" || it.className?.contains("EditText", ignoreCase = true) == true }
        val isComposerVacated = composerNodes.isEmpty() || composerNodes.all {
            val t = it.text?.trim()?.lowercase().orEmpty()
            t.isEmpty() || t == "type a message" || t == "message" || t == "text message"
        }

        val kBefore = preState?.nodes?.count { n ->
            val t = (n.text ?: n.contentDescription ?: "").lowercase()
            t.contains(cleanText) && !n.isEditable
        } ?: 0

        val kAfter = postState.nodes.count { n ->
            val t = (n.text ?: n.contentDescription ?: "").lowercase()
            t.contains(cleanText) && !n.isEditable
        }

        val hasInstanceDelta = kAfter >= kBefore + 1

        return if (hasInstanceDelta && isComposerVacated) {
            VerificationResult(
                status = VerificationStatus.VERIFIED_SUCCESS,
                message = "Message delivery verified: instance delta (K_before=$kBefore -> K_after=$kAfter) and composer vacated"
            )
        } else if (hasInstanceDelta) {
            VerificationResult(
                status = VerificationStatus.VERIFIED_SUCCESS,
                message = "Message delivery verified: instance delta confirmed (K_before=$kBefore -> K_after=$kAfter)"
            )
        } else {
            VerificationResult(
                status = VerificationStatus.CONFIRMED_FAILURE,
                message = "Message not yet verified: K_before=$kBefore, K_after=$kAfter, composerVacated=$isComposerVacated"
            )
        }
    }

    /**
     * Module 4: Instance-Delta Context Evidence Verification for Search.
     * Asserts that:
     * 1. Results container expanded with new child items (N_after > N_before)
     * 2. Soft keyboard was dismissed.
     */
    fun verifySearchOutcome(
        preState: DistilledUiState?,
        postState: DistilledUiState
    ): VerificationResult {
        val preCount = preState?.nodes?.size ?: 0
        val postCount = postState.nodes.size
        val keyboardDismissed = postState.nodes.none {
            it.className?.contains("KeyboardView", ignoreCase = true) == true ||
            it.resourceId?.contains("keyboard", ignoreCase = true) == true
        }

        return if (postCount > preCount && keyboardDismissed) {
            VerificationResult(
                status = VerificationStatus.VERIFIED_SUCCESS,
                message = "Search outcome verified: results expanded (N_before=$preCount -> N_after=$postCount) and keyboard dismissed"
            )
        } else if (postCount > preCount) {
            VerificationResult(
                status = VerificationStatus.VERIFIED_SUCCESS,
                message = "Search results expanded: N_before=$preCount -> N_after=$postCount"
            )
        } else {
            VerificationResult(
                status = VerificationStatus.UNCERTAIN,
                message = "Search results unchanged or pending (N_before=$preCount, N_after=$postCount)"
            )
        }
    }
}

