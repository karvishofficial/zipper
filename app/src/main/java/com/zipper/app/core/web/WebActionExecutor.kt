package com.zipper.app.core.web

import android.util.Log
import com.zipper.app.core.agent.ActionCandidate
import com.zipper.app.core.agent.ActionExecutor
import com.zipper.app.core.agent.AgentAction
import com.zipper.app.core.model.DistilledNode
import kotlinx.coroutines.delay

/**
 * Executes V57 AgentAction primitives within the Hidden WebView environment.
 * Maps high-level intent (Tap, TypeText, Scroll, Submit, Back) to DOM events.
 */
class WebActionExecutor(private val webController: WebController) : ActionExecutor {

    companion object {
        private const val TAG = "WebActionExecutor"
    }

    override suspend fun execute(
        action: AgentAction,
        candidateMap: Map<Int, DistilledNode>,
        goal: String,
        actionCandidateMap: Map<String, ActionCandidate>
    ): Boolean {
        return when (action) {
            is AgentAction.Tap -> {
                val node = candidateMap[action.targetNodeIndex]
                val karvishId = node?.resourceId?.takeIf { it.startsWith("k_node_") }
                    ?: "k_node_${action.targetNodeIndex}"
                Log.d(TAG, "Executing Web Tap on index ${action.targetNodeIndex} (karvishId: $karvishId)")
                webController.clickElement(karvishId)
            }
            is AgentAction.TypeText -> {
                val node = candidateMap[action.targetNodeIndex]
                val karvishId = node?.resourceId?.takeIf { it.startsWith("k_node_") }
                    ?: "k_node_${action.targetNodeIndex}"
                Log.d(TAG, "Executing Web TypeText on index ${action.targetNodeIndex} (karvishId: $karvishId): '${action.text}'")
                webController.typeText(karvishId, action.text)
            }
            is AgentAction.Scroll -> {
                Log.d(TAG, "Executing Web Scroll: ${action.direction}")
                webController.scroll(action.direction)
            }
            is AgentAction.Submit -> {
                Log.d(TAG, "Executing Web Submit")
                webController.submit()
            }
            is AgentAction.PressBack -> {
                Log.d(TAG, "Executing Web GoBack")
                webController.goBack()
            }
            is AgentAction.Wait -> {
                delay(action.durationMs)
                true
            }
            is AgentAction.CompleteGoal, is AgentAction.FailGoal -> true
            else -> {
                Log.w(TAG, "Action $action is not applicable in Web environment")
                false
            }
        }
    }
}
