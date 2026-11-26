package com.zipper.app.core.agent

import android.graphics.Rect
import android.os.Bundle
import android.util.Log
import android.view.accessibility.AccessibilityNodeInfo
import com.zipper.app.accessibility.KarvishAccessibilityService
import com.zipper.app.core.model.DistilledNode
import kotlinx.coroutines.delay

/**
 * Dispatches generic AgentAction primitives to the Android OS via Accessibility APIs.
 *
 * Core Invariant:
 * A candidate ID identifies one candidate within one observation.
 * The executor must execute only the node bound to that candidate and must never infer
 * the target from the human-readable tool name, label, coordinate, or ambiguous numeric substring.
 * If that binding becomes invalid, re-observe instead of guessing.
 */
class ActionDispatcher(private val a11yService: KarvishAccessibilityService) : ActionExecutor {

    companion object {
        private const val TAG = "KarvishActionDispatcher"
    }

    override suspend fun execute(
        action: AgentAction,
        candidateMap: Map<Int, DistilledNode>,
        goal: String,
        actionCandidateMap: Map<String, ActionCandidate>
    ): Boolean = dispatch(action, candidateMap, goal, actionCandidateMap)


    suspend fun dispatch(
        action: AgentAction,
        candidateMap: Map<Int, DistilledNode> = emptyMap(),
        goal: String = "",
        actionCandidateMap: Map<String, ActionCandidate> = emptyMap()
    ): Boolean {
        val candidate = when (action) {
            is AgentAction.Tap -> action.candidateId?.let { actionCandidateMap[it] }
            is AgentAction.TypeText -> action.candidateId?.let { actionCandidateMap[it] }
            is AgentAction.Scroll -> action.candidateId?.let { actionCandidateMap[it] }
            else -> null
        }
        return when (action) {
            is AgentAction.OpenApp -> if (action.appName.isBlank()) {
                a11yService.openAppForGoal(goal)
            } else {
                a11yService.openAppByName(action.appName)
            }
            is AgentAction.Tap -> executeTap(action.targetNodeIndex, candidateMap[action.targetNodeIndex], action, candidate)
            is AgentAction.TypeText -> executeTypeText(action.targetNodeIndex, action.text, candidateMap[action.targetNodeIndex], action, candidate)
            is AgentAction.Scroll -> executeScroll(action, candidateMap[action.targetNodeIndex], candidate)
            is AgentAction.PressBack -> a11yService.performGlobalBack()
            AgentAction.Submit -> executeSubmit()
            is AgentAction.PressHome -> a11yService.performGlobalHome()
            is AgentAction.Wait -> {
                delay(action.durationMs)
                true
            }
            is AgentAction.CompleteGoal, is AgentAction.FailGoal -> true
        }
    }

    private fun isSemanticallyCompatible(expected: String, liveText: String?, liveDesc: String?): Boolean {
        val exp = expected.trim().lowercase()
        val txt = liveText?.trim()?.lowercase().orEmpty()
        val desc = liveDesc?.trim()?.lowercase().orEmpty()
        if (txt.isEmpty() && desc.isEmpty()) {
            return true
        }
        if (txt == exp || desc == exp) return true
        if (txt.contains(exp) || (txt.length >= 2 && exp.contains(txt))) return true
        if (desc.contains(exp) || (desc.length >= 2 && exp.contains(desc))) return true
        val isBothCartActions = (exp.contains("add") || exp.contains("cart")) &&
            (desc.contains("add") || desc.contains("cart") || txt.contains("add") || txt.contains("cart"))
        if (isBothCartActions) return true
        return false
    }

    private suspend fun executeTap(
        nodeIndex: Int,
        candidateNode: DistilledNode?,
        action: AgentAction.Tap? = null,
        candidate: ActionCandidate? = null
    ): Boolean {
        // 1. Primary path: Live AccessibilityNodeInfo from activeNodeMap (1-to-1 grounded to current observation)
        val liveNode = a11yService.activeNodeMap[nodeIndex]
        val expectedLabel = action?.semanticLabel
            ?: candidate?.semanticLabel
            ?: candidateNode?.text
            ?: candidateNode?.contentDescription

        if (liveNode != null) {
            // Semantic validation: Ensure liveNode hasn't mutated into an incompatible element
            if (!expectedLabel.isNullOrBlank()) {
                val liveText = liveNode.text?.toString()?.trim()
                val liveDesc = liveNode.contentDescription?.toString()?.trim()
                if (!isSemanticallyCompatible(expectedLabel, liveText, liveDesc)) {
                    Log.w(TAG, "executeTap: Semantic mismatch on node #$nodeIndex! Expected '$expectedLabel', found text='$liveText', desc='$liveDesc'. Failing closed to re-observe.")
                    return false
                }
            }

            val rect = Rect()
            liveNode.getBoundsInScreen(rect)

            // Web Context Handling:
            // Mobile browser engines (Chromium/WebView) often consume AccessibilityNodeInfo.ACTION_CLICK
            // synthetically without triggering JavaScript touch/pointer handlers (e.g. delegated touchstart/pointerdown).
            // For WebViews or web page elements, dispatch a real touch gesture to guarantee DOM events fire!
            val isWebNode = liveNode.packageName?.toString()?.contains("chrome") == true ||
                liveNode.className?.contains("WebView", ignoreCase = true) == true ||
                candidateNode?.role in setOf("LINK", "WEBVIEW")

            if (isWebNode && rect.width() > 0 && rect.height() > 0) {
                Log.i(TAG, "executeTap: Dispatching physical coordinate gesture on web node #$nodeIndex at $rect")
                val tapResult = a11yService.clickAtCoordinates(rect.centerX().toFloat(), rect.centerY().toFloat())
                if (tapResult) return true
            }

            // Semantic ACTION_CLICK on the live node
            if (liveNode.performAction(AccessibilityNodeInfo.ACTION_CLICK)) {
                Log.i(TAG, "executeTap: Semantic ACTION_CLICK succeeded on live node #$nodeIndex (${liveNode.className})")
                return true
            }

            // Check immediate parent hierarchy for clickable wrapper
            var parent = liveNode.parent
            var depth = 0
            while (parent != null && depth < 4) {
                if (parent.isClickable) {
                    if (parent.performAction(AccessibilityNodeInfo.ACTION_CLICK)) {
                        Log.i(TAG, "executeTap: Clicked clickable parent of node #$nodeIndex (${parent.className})")
                        return true
                    }
                }
                parent = parent.parent
                depth++
            }

            // Dynamic live bounding box center tap strictly on this node
            if (rect.width() > 0 && rect.height() > 0) {
                Log.i(TAG, "executeTap: Dynamic center tap on live node #$nodeIndex at $rect")
                val tapResult = a11yService.clickAtCoordinates(rect.centerX().toFloat(), rect.centerY().toFloat())
                if (tapResult) return true
            }
        }

        // 2. Secondary path: If liveNode reference was detached from activeNodeMap,
        // attempt resolution by stable resourceId from nodeIdentity if present
        val stableId = action?.nodeIdentity?.takeIf { !it.startsWith("node_") && !it.startsWith("system:") }
            ?: candidateNode?.resourceId?.takeIf { it.isNotBlank() }

        if (stableId != null) {
            val root = a11yService.rootInActiveWindow
            val nodesWithId = root?.findAccessibilityNodeInfosByViewId(stableId)
            val matchingNode = nodesWithId?.firstOrNull { node ->
                if (expectedLabel.isNullOrBlank()) true
                else isSemanticallyCompatible(expectedLabel, node.text?.toString(), node.contentDescription?.toString())
            }
            if (matchingNode != null) {
                if (matchingNode.performAction(AccessibilityNodeInfo.ACTION_CLICK)) {
                    Log.i(TAG, "executeTap: Clicked node resolved via stable resourceId '$stableId'")
                    return true
                }
                val rect = Rect()
                matchingNode.getBoundsInScreen(rect)
                if (rect.width() > 0 && rect.height() > 0) {
                    return a11yService.clickAtCoordinates(rect.centerX().toFloat(), rect.centerY().toFloat())
                }
            }
        }

        // Web / Coordinate Fallback: If node reference was detached in a browser or dynamic web context
        // but we have valid screen bounds from candidate capture, dispatch physical tap at the captured center!
        val capturedBounds = candidateNode?.rawBounds
        if (capturedBounds != null && capturedBounds.width() > 0 && capturedBounds.height() > 0) {
            Log.i(TAG, "executeTap: Fallback to physical gesture at captured bounds center (${capturedBounds.centerX()}, ${capturedBounds.centerY()}) for node #$nodeIndex ('$expectedLabel')")
            val tapResult = a11yService.clickAtCoordinates(capturedBounds.centerX().toFloat(), capturedBounds.centerY().toFloat())
            if (tapResult) return true
        }

        // Fail-Closed: Never guess approximate coordinates or perform blind global text searches
        // that cause duplicate-label collisions (e.g. clicking the wrong "0" button).
        Log.w(TAG, "executeTap: Candidate binding invalid for node #$nodeIndex ('$expectedLabel'). Failing closed to re-observe.")
        return false
    }

    private suspend fun executeTypeText(
        nodeIndex: Int,
        text: String,
        candidateNode: DistilledNode?,
        action: AgentAction.TypeText? = null,
        candidate: ActionCandidate? = null
    ): Boolean {
        val liveNode = a11yService.activeNodeMap[nodeIndex]
        val expectedLabel = action?.semanticLabel
            ?: candidate?.semanticLabel
            ?: candidateNode?.text
            ?: candidateNode?.contentDescription

        if (liveNode != null) {
            if (!expectedLabel.isNullOrBlank()) {
                val liveText = liveNode.text?.toString()?.trim()
                val liveDesc = liveNode.contentDescription?.toString()?.trim()
                if (!isSemanticallyCompatible(expectedLabel, liveText, liveDesc)) {
                    Log.w(TAG, "executeTypeText: Semantic mismatch on node #$nodeIndex! Expected '$expectedLabel'. Failing closed.")
                    return false
                }
            }

            // Tap the editable node via coordinates to guarantee touch focus and soft keyboard activation
            val rect = Rect()
            liveNode.getBoundsInScreen(rect)
            if (rect.width() > 0 && rect.height() > 0) {
                a11yService.clickAtCoordinates(rect.centerX().toFloat(), rect.centerY().toFloat())
                delay(200)
            } else {
                liveNode.performAction(AccessibilityNodeInfo.ACTION_CLICK)
                liveNode.performAction(AccessibilityNodeInfo.ACTION_FOCUS)
                delay(150)
            }

            val emptyBundle = Bundle().apply {
                putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, "")
            }
            liveNode.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, emptyBundle)

            val bundle = Bundle().apply {
                putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, text)
            }
            if (liveNode.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, bundle)) {
                Log.i(TAG, "executeTypeText: Successfully set text '$text' on live node #$nodeIndex")
                return true
            }

            // Fallback 1: Poll if tap activated a new focused input field (e.g. Chrome search_box_text -> url_bar)
            for (w in 0 until 5) {
                delay(100)
                val currentRoot = a11yService.rootInActiveWindow
                val focusedInput = currentRoot?.findFocus(AccessibilityNodeInfo.FOCUS_INPUT)
                    ?: currentRoot?.findAccessibilityNodeInfosByViewId("com.android.chrome:id/url_bar")?.firstOrNull()
                if (focusedInput != null && focusedInput.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, bundle)) {
                    Log.i(TAG, "executeTypeText: Successfully set text '$text' on newly focused input node")
                    return true
                }
            }

            // Fallback 2: retry coordinate tap and set text
            if (rect.width() > 0 && rect.height() > 0) {
                a11yService.clickAtCoordinates(rect.centerX().toFloat(), rect.centerY().toFloat())
                delay(200)
                val retryFocus = a11yService.rootInActiveWindow?.findFocus(AccessibilityNodeInfo.FOCUS_INPUT) ?: liveNode
                if (retryFocus.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, bundle)) {
                    return true
                }
            }
        }

        // Secondary path: stable resourceId fallback if detached
        val stableId = action?.nodeIdentity?.takeIf { !it.startsWith("node_") && !it.startsWith("system:") }
            ?: candidateNode?.resourceId?.takeIf { it.isNotBlank() }

        if (stableId != null) {
            val root = a11yService.rootInActiveWindow
            val nodesWithId = root?.findAccessibilityNodeInfosByViewId(stableId)
            val matchingNode = nodesWithId?.firstOrNull()
            if (matchingNode != null) {
                val bundle = Bundle().apply {
                    putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, text)
                }
                if (matchingNode.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, bundle)) {
                    return true
                }
            }
        }

        Log.w(TAG, "executeTypeText: Failed to safely resolve editable node #$nodeIndex. Failing closed to re-observe.")
        return false
    }

    private fun commitInput(node: AccessibilityNodeInfo?) {
        try {
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R && node != null) {
                node.performAction(AccessibilityNodeInfo.AccessibilityAction.ACTION_IME_ENTER.id)
            }
        } catch (_: Throwable) {}
    }

    private suspend fun executeScroll(
        action: AgentAction.Scroll,
        candidateNode: DistilledNode?,
        candidate: ActionCandidate?
    ): Boolean {
        val targetIdx = action.targetNodeIndex ?: candidate?.nodeIndex
        if (targetIdx != null) {
            val liveNode = a11yService.activeNodeMap[targetIdx]
            if (liveNode != null) {
                val scrollAction = if (action.direction == ScrollDirection.UP) {
                    AccessibilityNodeInfo.ACTION_SCROLL_BACKWARD
                } else {
                    AccessibilityNodeInfo.ACTION_SCROLL_FORWARD
                }
                if (liveNode.performAction(scrollAction)) return true
            }
        }
        return a11yService.genericScroll(targetIdx, action.direction)
    }

    private suspend fun executeSubmit(): Boolean {
        // 0. Messaging / Search Submit Affordance: If a dedicated Send or Search button is present, dispatch click directly
        val liveSubmitNode = a11yService.activeNodeMap.values.firstOrNull { node ->
            val isInput = node.isEditable || node.className?.toString()?.contains("EditText", ignoreCase = true) == true
            if (isInput) return@firstOrNull false

            val res = node.viewIdResourceName?.lowercase().orEmpty()
            val text = node.text?.toString()?.lowercase().orEmpty()
            val desc = node.contentDescription?.toString()?.lowercase().orEmpty()

            // Filter out AI / Bot triggers (e.g. Meta AI send button in WhatsApp search)
            val isAiDistractor = res.contains("meta_ai") || res.contains("ai_send") || res.contains("assistant") ||
                res.contains("ai_input_send") || desc.contains("meta ai") || text.contains("meta ai") ||
                desc.contains("ask ai") || text.contains("ask ai")
            if (isAiDistractor) return@firstOrNull false

            val isSend = res.contains("send") || desc == "send" || desc.startsWith("send ") || text == "send"
            val isSearch = res.contains("search_button") || res.contains("btn_search") || res.contains("search_mag_icon") ||
                res.contains("nav_search") || ((desc == "search" || text == "search") && (node.className?.contains("Button", true) == true || node.className?.contains("ImageView", true) == true))
            (isSend || isSearch) && (node.isClickable || node.className?.contains("Button", ignoreCase = true) == true || node.className?.contains("ImageView", ignoreCase = true) == true)
        }
        if (liveSubmitNode != null) {
            Log.i(TAG, "executeSubmit: Found live Submit/Search button (${liveSubmitNode.viewIdResourceName} / ${liveSubmitNode.text}). Performing ACTION_CLICK.")
            if (liveSubmitNode.performAction(AccessibilityNodeInfo.ACTION_CLICK)) {
                return true
            }
            var p = liveSubmitNode.parent
            while (p != null) {
                if (p.isClickable && p.performAction(AccessibilityNodeInfo.ACTION_CLICK)) {
                    return true
                }
                p = p.parent
            }
            val rect = Rect()
            liveSubmitNode.getBoundsInScreen(rect)
            if (rect.width() > 0 && rect.height() > 0) {
                return a11yService.clickAtCoordinates(rect.centerX().toFloat(), rect.centerY().toFloat())
            }
        }

        // 1. Try finding input-focused node or any editable node in activeNodeMap
        val focusedNode = a11yService.rootInActiveWindow?.findFocus(AccessibilityNodeInfo.FOCUS_INPUT)
            ?: a11yService.activeNodeMap.values.firstOrNull { it.isEditable }
        if (focusedNode != null) {
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R) {
                if (focusedNode.performAction(AccessibilityNodeInfo.AccessibilityAction.ACTION_IME_ENTER.id)) {
                    Log.i(TAG, "executeSubmit: Dispatched ACTION_IME_ENTER successfully on focused node")
                    return true
                }
            }
        }

        // 2. Check if a soft keyboard (IME window) is actively present on screen
        val imeWindow = try {
            a11yService.windows.firstOrNull { it.type == android.view.accessibility.AccessibilityWindowInfo.TYPE_INPUT_METHOD }
        } catch (e: Exception) {
            null
        }

        if (imeWindow != null) {
            val imeBounds = Rect()
            imeWindow.getBoundsInScreen(imeBounds)
            if (imeBounds.width() > 0 && imeBounds.height() > 0) {
                val imeEnterX = imeBounds.left + imeBounds.width() * 0.90f
                val imeEnterY = imeBounds.top + imeBounds.height() * 0.85f
                Log.i(TAG, "executeSubmit: Dispatching IME Enter gesture to active keyboard window at ($imeEnterX, $imeEnterY)")
                return a11yService.clickAtCoordinates(imeEnterX, imeEnterY)
            }
        }

        // 3. Fallback: Commit input on focused node if available
        if (focusedNode != null) {
            commitInput(focusedNode)
            return true
        }

        return false
    }
}
