package com.zipper.app.core.web

import android.content.Context
import com.zipper.app.core.agent.ActionCandidate
import com.zipper.app.core.agent.AgentAction
import com.zipper.app.core.model.DistilledNode
import com.zipper.app.core.model.DistilledUiState
import kotlinx.coroutines.delay

data class WebStepResult(
    val action: AgentAction,
    val executed: Boolean,
    val preState: DistilledUiState,
    val postState: DistilledUiState,
    val verification: WebVerificationResult
)

/**
 * Authoritative WebExecutionEnvironment for Karvish V57+ Universal Agent.
 *
 * Implements the frozen v2 WebView Architecture:
 * V57
 *  ↓
 * WebEnvironment
 *  ├── HiddenWebViewWorker (WebController)
 *  ├── WebStateDistiller
 *  ├── WebActionExecutor
 *  └── WebVerification
 *
 * Principle: "V57 decides; WebStateDistiller reports; WebActionExecutor operates; WebVerification proves."
 */
class WebEnvironment(
    val controller: WebController,
    val distiller: WebStateDistiller = WebStateDistiller(),
    val executor: WebActionExecutor = WebActionExecutor(controller),
    val verifier: WebVerification = WebVerification()
) {

    companion object {
        fun createDefault(context: Context): WebEnvironment {
            val ctrl = HiddenWebController(context)
            return WebEnvironment(controller = ctrl)
        }
    }

    private var latestState: DistilledUiState = DistilledUiState(
        packageName = "com.zipper.web",
        activityName = "about:blank"
    )

    /**
     * Observes the current DOM state and distills it into a standardized DistilledUiState.
     */
    suspend fun observe(): DistilledUiState {
        val state = controller.distillWebState()
        latestState = state
        return state
    }

    /**
     * Navigates to the specified URL and returns the initial distilled state.
     */
    suspend fun navigate(url: String, timeoutMs: Long = 12000L): Boolean {
        val success = controller.navigate(url, timeoutMs)
        if (success) {
            observe()
        }
        return success
    }

    /**
     * Executes an agent action primitive, observes the DOM outcome, and formally verifies it.
     */
    suspend fun step(
        action: AgentAction,
        goal: String,
        candidateMap: Map<Int, DistilledNode>,
        actionCandidateMap: Map<String, ActionCandidate> = emptyMap()
    ): WebStepResult {
        val preState = latestState

        // 1. WebActionExecutor operates
        val executed = executor.execute(action, candidateMap, goal, actionCandidateMap)

        // 2. Allow DOM mutation to settle (debounce)
        val debounceMs = when (action) {
            is AgentAction.Tap -> 500L
            is AgentAction.Submit -> 800L
            is AgentAction.TypeText -> 200L
            is AgentAction.Scroll -> 350L
            is AgentAction.PressBack -> 500L
            else -> 150L
        }
        delay(debounceMs)

        // 3. WebStateDistiller reports
        val postState = observe()

        // 4. WebVerification proves
        val verificationResult = verifier.verify(
            preState = preState,
            action = action,
            postState = postState,
            goal = goal
        )

        return WebStepResult(
            action = action,
            executed = executed,
            preState = preState,
            postState = postState,
            verification = verificationResult
        )
    }

    suspend fun clear() {
        controller.clear()
        latestState = DistilledUiState(packageName = "com.zipper.web", activityName = "about:blank")
    }

    suspend fun destroy() {
        controller.destroy()
    }
}
