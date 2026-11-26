package com.zipper.app.core.agent

import com.zipper.app.core.model.DistilledNode
import com.zipper.app.core.model.DistilledUiState

data class TextDiff(
    val nodeIndex: Int,
    val label: String,
    val oldText: String,
    val newText: String
)

data class FocusDiff(
    val nodeIndex: Int,
    val label: String,
    val hadFocus: Boolean,
    val hasFocus: Boolean
)

data class StateDelta(
    val outcome: StepOutcome,
    val summary: String,
    val packageChanged: Boolean,
    val activityChanged: Boolean,
    val nodesAddedCount: Int,
    val nodesRemovedCount: Int,
    val textChangedCount: Int,
    val detectedErrorMessage: String? = null,
    val screenChanged: Boolean = false,
    val observationChanged: Boolean = false,
    val isNoOp: Boolean = false,
    val textChanges: List<TextDiff> = emptyList(),
    val focusChanges: List<FocusDiff> = emptyList(),
    val addedNodes: List<String> = emptyList(),
    val removedNodes: List<String> = emptyList()
) {
    /**
     * Formats a clean, factual, compact delta line for agent history/training.
     * Examples:
     * - text: formula ('0' -> '2')
     * - text: search_box ('music'); +15 element(s)
     * - NO-OP: Screen state unchanged
     */
    fun formatFactualSummary(): String {
        if (isNoOp) return "NO-OP: Screen state unchanged"
        val parts = mutableListOf<String>()
        if (packageChanged) parts.add("package_changed")
        if (activityChanged) parts.add("screen_transitioned")
        if (textChanges.isNotEmpty()) {
            val tcStr = textChanges.joinToString(", ") {
                "${it.label} ('${it.oldText}' -> '${it.newText}')"
            }
            parts.add("text: $tcStr")
        }
        if (focusChanges.isNotEmpty()) {
            val fcStr = focusChanges.joinToString(", ") {
                "${it.label} (${if (it.hasFocus) "focused" else "unfocused"})"
            }
            parts.add("focus: $fcStr")
        }
        if (nodesAddedCount > 0) {
            parts.add("+$nodesAddedCount element(s)")
        }
        if (nodesRemovedCount > 0) {
            parts.add("-$nodesRemovedCount element(s)")
        }
        return parts.joinToString("; ").ifBlank { "UI state updated" }
    }
}

/**
 * Evaluates the factual differential between the UI state before an action and after.
 *
 * Provides closed-loop observation feedback:
 * - Factual text changes (e.g. calculator formula updates on button taps, search box inputs)
 * - Factual focus changes (e.g. search field gained/lost focus)
 * - Added/removed node sets
 * - Explicit NO-OP identification when actions produce zero observable change.
 */
class StateDifferentialEngine {

    companion object {
        private val ERROR_KEYWORDS = listOf(
            "error",
            "failed",
            "something went wrong",
            "stopped working",
            "isn't responding",
            "permission denied",
            "invalid",
            "try again",
            "can't connect",
            "couldn't"
        )

        private val POPUP_KEYWORDS = listOf(
            "permission",
            "allow",
            "deny",
            "while using the app",
            "only this time",
            "don't allow",
            "access your",
            "grant permission"
        )
    }

    fun computeDelta(
        preState: DistilledUiState,
        postState: DistilledUiState,
        action: AgentAction
    ): StateDelta {
        val packageChanged = preState.packageName != postState.packageName
        val activityChanged = preState.activityName != postState.activityName

        // 1. Check for unexpected app/launcher exit (WRONG_SCREEN)
        if (packageChanged && action !is AgentAction.OpenApp && action !is AgentAction.PressHome) {
            val postPkg = postState.packageName?.lowercase() ?: ""
            if (postPkg.contains("launcher") || postPkg.contains("home") || postPkg == "com.zipper.app") {
                return StateDelta(
                    outcome = StepOutcome.WRONG_SCREEN,
                    summary = "WRONG_SCREEN: Unexpected exit to launcher/host app ($postPkg)",
                    packageChanged = true,
                    activityChanged = activityChanged,
                    nodesAddedCount = 0,
                    nodesRemovedCount = 0,
                    textChangedCount = 0,
                    screenChanged = true,
                    observationChanged = true,
                    isNoOp = false
                )
            }
        }

        // 2. Check for blocking popup or permission dialog
        val preNodeKeys = preState.nodes.map { nodeKey(it) }.toSet()
        val hasPopup = postState.nodes.any { node ->
            val nodeText = listOfNotNull(node.text, node.contentDescription)
                .joinToString(" ")
                .lowercase()
            val isNewNode = nodeKey(node) !in preNodeKeys
            val isDialogLike = node.role.contains("DIALOG", ignoreCase = true) ||
                node.className?.contains("dialog", ignoreCase = true) == true
            isNewNode && isDialogLike && POPUP_KEYWORDS.any { kw -> nodeText.contains(kw) }
        }
        if (hasPopup) {
            return StateDelta(
                outcome = StepOutcome.BLOCKED_BY_POPUP,
                summary = "BLOCKED_BY_POPUP: Screen was intercepted by system permission or alert dialog",
                packageChanged = packageChanged,
                activityChanged = activityChanged,
                nodesAddedCount = 0,
                nodesRemovedCount = 0,
                textChangedCount = 0,
                screenChanged = true,
                observationChanged = true,
                isNoOp = false
            )
        }

        // 3. Check for error modals or toast messages
        val errorMessage = postState.nodes.firstOrNull { node ->
            val text = node.text?.lowercase() ?: ""
            val desc = node.contentDescription?.lowercase() ?: ""
            ERROR_KEYWORDS.any { kw -> text.contains(kw) || desc.contains(kw) }
        }?.let { it.text ?: it.contentDescription }

        if (errorMessage != null) {
            return StateDelta(
                outcome = StepOutcome.ERROR_DETECTED,
                summary = "Error indicator detected on screen: '$errorMessage'",
                packageChanged = packageChanged,
                activityChanged = activityChanged,
                nodesAddedCount = 0,
                nodesRemovedCount = 0,
                textChangedCount = 0,
                detectedErrorMessage = errorMessage,
                screenChanged = true,
                observationChanged = true,
                isNoOp = false
            )
        }

        // 4. Calculate factual text and focus changes across all matching nodes
        val preNodesById = preState.nodes.filter { !it.resourceId.isNullOrBlank() }.associateBy { it.resourceId!! }
        val preNodesByIndex = preState.nodes.associateBy { it.index }

        val textDiffs = mutableListOf<TextDiff>()
        val focusDiffs = mutableListOf<FocusDiff>()

        for (postNode in postState.nodes) {
            val matchingPreNode = (postNode.resourceId?.let { preNodesById[it] })
                ?: preNodesByIndex[postNode.index]?.takeIf { it.role == postNode.role }

            if (matchingPreNode != null) {
                val preText = matchingPreNode.text?.trim().orEmpty()
                val postText = postNode.text?.trim().orEmpty()
                if (preText != postText) {
                    val label = postNode.resourceId?.substringAfterLast("/")
                        ?: postNode.contentDescription
                        ?: postNode.role.lowercase()
                    textDiffs.add(TextDiff(postNode.index, label, preText, postText))
                }

                if (matchingPreNode.isFocused != postNode.isFocused) {
                    val label = postNode.resourceId?.substringAfterLast("/")
                        ?: postNode.contentDescription
                        ?: postNode.text
                        ?: postNode.role.lowercase()
                    focusDiffs.add(FocusDiff(postNode.index, label, matchingPreNode.isFocused, postNode.isFocused))
                }
            }
        }

        // Explicit TypeText fallback check if node index shifted
        if (action is AgentAction.TypeText && textDiffs.isEmpty()) {
            val matchedTypedNode = postState.nodes.firstOrNull {
                it.text?.contains(action.text, ignoreCase = true) == true
            }
            if (matchedTypedNode != null) {
                val label = matchedTypedNode.resourceId?.substringAfterLast("/") ?: "input_field"
                textDiffs.add(TextDiff(matchedTypedNode.index, label, "", action.text))
            }
        }

        // 5. Calculate node set differences (added / removed)
        val preKeySet = preState.nodes.map { nodeKey(it) }.toSet()
        val postKeySet = postState.nodes.map { nodeKey(it) }.toSet()

        val addedNodes = postState.nodes.filter { nodeKey(it) !in preKeySet }
            .map { it.text?.takeIf { t -> t.isNotBlank() } ?: it.contentDescription ?: it.role }
            .take(10)
        val removedNodes = preState.nodes.filter { nodeKey(it) !in postKeySet }
            .map { it.text?.takeIf { t -> t.isNotBlank() } ?: it.contentDescription ?: it.role }
            .take(10)

        val addedCount = postState.nodes.count { nodeKey(it) !in preKeySet }
        val removedCount = preState.nodes.count { nodeKey(it) !in postKeySet }

        val preLabels = preState.nodes.map { "${it.role}:${it.text ?: it.contentDescription ?: ""}" }
        val postLabels = postState.nodes.map { "${it.role}:${it.text ?: it.contentDescription ?: ""}" }
        val preBounds = preState.nodes.map { it.bounds }
        val postBounds = postState.nodes.map { it.bounds }

        val screenChanged = packageChanged || activityChanged || addedCount > 0 || removedCount > 0
        val observationChanged = screenChanged || textDiffs.isNotEmpty() || focusDiffs.isNotEmpty()
        val isNoOp = !packageChanged && !activityChanged && !observationChanged &&
            preLabels == postLabels && preBounds == postBounds

        if (isNoOp) {
            return StateDelta(
                outcome = StepOutcome.NO_OP,
                summary = "NO-OP: Screen state did not change after action. The element may be non-interactive or blocked.",
                packageChanged = false,
                activityChanged = false,
                nodesAddedCount = 0,
                nodesRemovedCount = 0,
                textChangedCount = 0,
                screenChanged = false,
                observationChanged = false,
                isNoOp = true,
                textChanges = emptyList(),
                focusChanges = emptyList(),
                addedNodes = emptyList(),
                removedNodes = emptyList()
            )
        }

        val progressDesc = buildString {
            if (packageChanged) append("Switched to app '${postState.packageName}'. ")
            if (activityChanged) append("Navigated to screen '${postState.activityName}'. ")
            if (addedCount > 0) append("$addedCount new element(s) appeared. ")
            if (removedCount > 0) append("$removedCount element(s) closed/disappeared. ")
            if (textDiffs.isNotEmpty()) {
                val tdDesc = textDiffs.take(2).joinToString(", ") { "${it.label}: '${it.oldText}' -> '${it.newText}'" }
                append("Text changed ($tdDesc). ")
            }
            if (focusDiffs.isNotEmpty()) {
                val fdDesc = focusDiffs.take(2).joinToString(", ") { "${it.label} (${if (it.hasFocus) "focused" else "unfocused"})" }
                append("Focus changed ($fdDesc). ")
            }
        }.trim().ifBlank { "UI state updated successfully." }

        return StateDelta(
            outcome = StepOutcome.SUCCESS,
            summary = progressDesc,
            packageChanged = packageChanged,
            activityChanged = activityChanged,
            nodesAddedCount = addedCount,
            nodesRemovedCount = removedCount,
            textChangedCount = textDiffs.size,
            screenChanged = screenChanged,
            observationChanged = observationChanged,
            isNoOp = false,
            textChanges = textDiffs,
            focusChanges = focusDiffs,
            addedNodes = addedNodes,
            removedNodes = removedNodes
        )
    }

    private fun nodeKey(node: DistilledNode): String =
        "${node.role}:${node.resourceId ?: ""}:${node.text ?: node.contentDescription ?: ""}:${node.bounds}"
}

