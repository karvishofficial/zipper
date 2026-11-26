package com.zipper.app.core.web

import android.util.Log
import com.zipper.app.core.agent.ActionExecutor
import com.zipper.app.core.agent.AgentAction
import com.zipper.app.core.agent.ScrollDirection
import com.zipper.app.core.model.DistilledNode
import com.zipper.app.core.model.DistilledUiState
import kotlinx.coroutines.delay
import java.net.URLEncoder

data class WebEpisodeResult(
    val success: Boolean,
    val finalUrl: String,
    val summary: String,
    val turnsExecuted: Int,
    val fieldsFilled: List<Pair<String, String>> = emptyList(),
    val actionsTaken: List<String> = emptyList(),
    val extractedContent: String? = null,
    val imageUrls: List<String> = emptyList()
)

/**
 * Multi-Turn Autonomous Web Runner for Karvish.
 * Executes closed-loop web navigation, interactive candidate grounding,
 * automated form filling, and live section extraction.
 */
class WebAutonomousRunner(
    private val webController: WebController,
    private val actionExecutor: ActionExecutor,
    private val webLlmClient: WebLlmClient? = null,
    private val aiConfig: WebAiConfig? = null,
    private val webOnDeviceEngine: WebOnDeviceEngine? = null,
    private val onProgressUpdate: (String) -> Unit = {}
) {

    companion object {
        private const val TAG = "WebAutonomousRunner"
        private const val MAX_TURNS = 10
    }

    private val captchaSolver = WebCaptchaSolver()
    private val hiddenController = webController as? HiddenWebController
    private val context = hiddenController?.getContext()
    private val vaultManager by lazy { context?.let { WebVaultManager(it) } }
    private val macroEngine by lazy { context?.let { WebMacroEngine(it) } }

    suspend fun runEpisode(goal: String): WebEpisodeResult {
        val cleanGoal = goal.trim()
        val actionsTaken = mutableListOf<String>()
        val fieldsFilled = mutableListOf<Pair<String, String>>()

        // 1. Resolve Initial URL (AI-first with sanitized fallback)
        var targetUrl: String? = null
        if (webLlmClient != null && aiConfig != null && aiConfig.isConfigured) {
            onProgressUpdate("🧠 ${aiConfig.provider} is resolving target destination...")
            targetUrl = webLlmClient.resolveTargetUrl(cleanGoal, aiConfig)
            if (!targetUrl.isNullOrBlank()) {
                Log.i(TAG, "LLM resolved target URL: $targetUrl")
            }
        }
        if (targetUrl.isNullOrBlank()) {
            targetUrl = resolveInitialUrl(cleanGoal)
        }

        val initialUrl = targetUrl
        onProgressUpdate("🌐 Navigating to $initialUrl...")
        Log.i(TAG, "Navigating to initial URL: $initialUrl")

        var loaded = webController.navigate(initialUrl, timeoutMs = 12000L)
        var effectiveUrl = initialUrl
        if (!loaded && initialUrl.startsWith("https://") && initialUrl.contains(":")) {
            val fallbackHttp = "http://" + initialUrl.removePrefix("https://")
            onProgressUpdate("🌐 Retrying over HTTP: $fallbackHttp...")
            loaded = webController.navigate(fallbackHttp, timeoutMs = 12000L)
            if (loaded) {
                effectiveUrl = fallbackHttp
            }
        }

        if (!loaded) {
            return WebEpisodeResult(
                success = false,
                finalUrl = effectiveUrl,
                summary = "Failed to load initial webpage: $effectiveUrl within timeout.",
                turnsExecuted = 0
            )
        }

        // 1b. Smart DOM settle: Fast MutationObserver event bridge
        onProgressUpdate("🌐 Waiting for webpage elements to render...")
        hiddenController?.waitForDomStability(2500L)
        var initialDistilled = webController.distillWebState()
        if (initialDistilled.nodes.count { it.isClickable || it.isEditable } < 2) {
            delay(350L)
            initialDistilled = webController.distillWebState()
        }
        Log.i(TAG, "Initial distillation ready: ${initialDistilled.nodes.size} candidates found")

        // 1c. Check for saved 1-Tap Web Macro (Ditto Web 2.0)
        val macro = macroEngine?.findMacroForUrl(effectiveUrl)
        if (macro != null && hiddenController != null) {
            onProgressUpdate("▶ Executing saved Web Macro: '${macro.title}'...")
            val macroRes = macroEngine!!.executeMacro(hiddenController, macro)
            if (macroRes.success) {
                hiddenController.waitForDomStability(1800L)
                val finalState = webController.distillWebState()
                return WebEpisodeResult(
                    success = true,
                    finalUrl = finalState.activityName.orEmpty(),
                    summary = "⚡ Instant Macro Execution Completed (${macroRes.stepsCompleted} steps): ${macro.title}",
                    turnsExecuted = 1,
                    actionsTaken = listOf("Executed macro: ${macro.title}"),
                    extractedContent = extractRelevantText(cleanGoal, finalState)
                )
            } else {
                Log.w(TAG, "Macro execution failed, falling back to autonomous agent: ${macroRes.summary}")
            }
        }

        // 2. Check if LLM API is configured for pure autonomous planning
        if (webLlmClient != null && aiConfig != null && aiConfig.isConfigured) {
            onProgressUpdate("🧠 Observing web elements & planning via ${aiConfig.provider}...")
            val candidateMap = initialDistilled.nodes.associateBy { it.index }

            val stage1Goal = if (cleanGoal.contains("after login", ignoreCase = true)) {
                cleanGoal.substringBefore("after login", cleanGoal).trim().trimEnd(',', '.', ';', ':')
            } else {
                cleanGoal
            }

            val plannedActions = webLlmClient.planWebActions(
                goal = stage1Goal,
                currentUrl = initialDistilled.activityName.orEmpty(),
                nodes = initialDistilled.nodes,
                config = aiConfig
            )

            if (plannedActions.isNotEmpty()) {
                onProgressUpdate("⚡ Executing ${plannedActions.size} autonomous actions via ${aiConfig.provider}...")
                var submitTriggered = false

                for ((idx, plan) in plannedActions.withIndex()) {
                    val targetNode = candidateMap[plan.targetIndex]
                    val targetLabel = targetNode?.semanticLabel ?: "Element #${plan.targetIndex}"
                    val karvishId = "k_node_${plan.targetIndex}"

                    when (plan.action) {
                        "TYPE" -> {
                            val textToType = plan.text ?: ""
                            onProgressUpdate("⌨️ (${idx + 1}/${plannedActions.size}) Entering '$textToType' in [$targetLabel]")
                            val action = AgentAction.TypeText(plan.targetIndex, textToType)
                            actionExecutor.execute(action, candidateMap, cleanGoal)
                            fieldsFilled.add(targetLabel to textToType)
                            actionsTaken.add("Entered '$textToType' in $targetLabel")
                            delay(400L)
                        }
                        "SELECT" -> {
                            val valToSelect = plan.value ?: ""
                            onProgressUpdate("🔽 (${idx + 1}/${plannedActions.size}) Selecting '$valToSelect' in [$targetLabel]")
                            webController.selectOption(karvishId, valToSelect)
                            fieldsFilled.add(targetLabel to valToSelect)
                            actionsTaken.add("Selected '$valToSelect' in $targetLabel")
                            delay(400L)
                        }
                        "CLICK" -> {
                            onProgressUpdate("👆 (${idx + 1}/${plannedActions.size}) Clicking [$targetLabel]")
                            val action = AgentAction.Tap(plan.targetIndex)
                            actionExecutor.execute(action, candidateMap, cleanGoal)
                            actionsTaken.add("Clicked $targetLabel")
                            if (targetLabel.contains("submit", true) || targetLabel.contains("login", true)) {
                                submitTriggered = true
                            }
                            delay(800L)
                        }
                        "SUBMIT" -> {
                            onProgressUpdate("🚀 (${idx + 1}/${plannedActions.size}) Submitting form...")
                            actionExecutor.execute(AgentAction.Submit, candidateMap, cleanGoal)
                            actionsTaken.add("Submitted form")
                            submitTriggered = true
                            delay(1200L)
                        }
                    }
                }

                // Pre-Submit Completeness Guard: Ensure all required fields from goal are filled
                ensureRequiredFormFieldsFilled(webController, candidateMap, cleanGoal, fieldsFilled, actionsTaken)

                // Check if a CAPTCHA security code is on this page and solve it with auto-retry
                var captchaInfo = captchaSolver.detectCaptcha(webController)
                if (captchaInfo != null) {
                    onProgressUpdate("🔐 Detecting and reading security code...")
                    var solvedCode: String? = null
                    var attempts = 0
                    while (attempts < 3) {
                        attempts++
                        val code = captchaSolver.solve(captchaInfo!!.base64Data)
                        if (!code.isNullOrBlank() && code.length in 4..6) {
                            solvedCode = code
                            break
                        }
                        Log.w(TAG, "Captcha attempt $attempts read invalid code '$code'. Refreshing...")
                        onProgressUpdate("🔄 Captcha read '$code'. Refreshing for clear image (attempt $attempts)...")
                        captchaSolver.refreshCaptcha(webController)
                        delay(600L)
                        captchaInfo = captchaSolver.detectCaptcha(webController)
                        if (captchaInfo == null) break
                    }

                    if (!solvedCode.isNullOrBlank()) {
                        onProgressUpdate("✍️ Entering security code '$solvedCode'...")
                        webController.typeText(captchaInfo!!.inputKarvishId, solvedCode)
                        fieldsFilled.add("Security Code" to solvedCode)
                        actionsTaken.add("Entered Security Code '$solvedCode'")
                        delay(400L)

                        // Click the submit/login button (excluding refresh or forgot buttons)
                        val currentState = webController.distillWebState()
                        val loginBtn = currentState.nodes.firstOrNull { node ->
                            node.isClickable && (
                                node.semanticLabel.contains("login", true) ||
                                node.semanticLabel.contains("log in", true) ||
                                node.semanticLabel.contains("sign in", true)
                            ) && !node.semanticLabel.contains("refresh", true) && !node.semanticLabel.contains("forgot", true)
                        } ?: currentState.nodes.firstOrNull { node ->
                            node.isClickable && (
                                node.semanticLabel.contains("submit", true) ||
                                node.semanticLabel.contains("continue", true)
                            ) && !node.semanticLabel.contains("refresh", true) && !node.semanticLabel.contains("forgot", true)
                        }

                        if (loginBtn != null) {
                            onProgressUpdate("🚀 Clicking [${loginBtn.semanticLabel}]...")
                            actionExecutor.execute(AgentAction.Tap(loginBtn.index), currentState.nodes.associateBy { it.index }, cleanGoal)
                            actionsTaken.add("Clicked ${loginBtn.semanticLabel}")
                            submitTriggered = true
                        } else {
                            onProgressUpdate("🚀 Submitting login form...")
                            webController.submit()
                            actionsTaken.add("Submitted form")
                            submitTriggered = true
                        }
                        delay(2000L)
                    }
                }

                // If submitted, check if a second screen (e.g. 2FA, parent phone, PIN) needs action
                var postSubmissionState: DistilledUiState? = null
                if (submitTriggered) {
                    onProgressUpdate("🔄 Waiting for post-login page transition...")
                    val transitionStart = System.currentTimeMillis()
                    while (System.currentTimeMillis() - transitionStart < 7000L) {
                        delay(600L)
                        val curr = webController.distillWebState()
                        val hasNextScreenInputs = curr.nodes.any {
                            it.isEditable || it.role == "SELECT" || it.semanticLabel.contains("father", true) ||
                            it.semanticLabel.contains("mother", true) || it.semanticLabel.contains("otp", true) ||
                            it.semanticLabel.contains("digit", true) || it.semanticLabel.contains("pin", true)
                        }
                        val loginGone = curr.nodes.none { it.semanticLabel.contains("login", true) && it.isClickable }
                        if (hasNextScreenInputs || loginGone || curr.activityName != initialDistilled.activityName) {
                            postSubmissionState = curr
                            break
                        }
                    }
                    if (postSubmissionState == null) {
                        postSubmissionState = webController.distillWebState()
                    }

                    // Check for secondary screen inputs (e.g. father/mother last 4 digits)
                    val hasNextScreenInputs = postSubmissionState.nodes.any {
                        it.isEditable || it.role == "SELECT" || it.semanticLabel.contains("father", true) || it.semanticLabel.contains("mother", true)
                    }

                    val goalHasNextStep = cleanGoal.contains("father", true) || cleanGoal.contains("mother", true) ||
                            cleanGoal.contains("digit", true) || cleanGoal.contains("otp", true) || cleanGoal.contains("pin", true) || cleanGoal.contains("after login", true)

                    if (hasNextScreenInputs && goalHasNextStep) {
                        onProgressUpdate("🧠 Planning Stage 2 verification via ${aiConfig.provider}...")

                        val stage2GoalClean = if (cleanGoal.contains("after login", ignoreCase = true)) {
                            cleanGoal.substringAfter("after login", cleanGoal).trim().trimStart(',', ':', '-', ' ')
                        } else {
                            cleanGoal
                        }
                        val scopedGoal = "Complete post-login verification: $stage2GoalClean (Note: Login credentials USN, DOB, and CAPTCHA were already entered and submitted on the previous screen. Only plan actions for the current verification screen)."

                        var stage2Actions = webLlmClient.planWebActions(
                            goal = scopedGoal,
                            currentUrl = postSubmissionState.activityName.orEmpty(),
                            nodes = postSubmissionState.nodes,
                            config = aiConfig
                        )

                        // If LLM returned empty actions (e.g. rate limit or token exhaustion), use autonomous fallback
                        if (stage2Actions.isEmpty()) {
                            Log.w(TAG, "Stage 2 LLM returned 0 actions. Running autonomous verification fallback...")
                            val fallbackActions = mutableListOf<PlannedWebAction>()
                            val nodes = postSubmissionState.nodes

                            // 1. Dropdown for verification type (Father, Mother, Guardian, etc.)
                            val selectNode = nodes.find { it.role == "SELECT" }
                            if (selectNode != null) {
                                val goalLower = cleanGoal.lowercase()
                                val optionsMatch = Regex("""\[Options:\s*([^\]]+)\]""").find(selectNode.semanticLabel)
                                val options = optionsMatch?.groupValues?.get(1)?.split(",")?.map { it.trim() } ?: emptyList()

                                val matchedOpt = options.find { opt ->
                                    val optLower = opt.lowercase()
                                    (goalLower.contains("father") && optLower.contains("father")) ||
                                    (goalLower.contains("mother") && optLower.contains("mother")) ||
                                    (goalLower.contains("guardian") && optLower.contains("guardian")) ||
                                    (goalLower.contains("mobile") && optLower.contains("mobile"))
                                } ?: options.find { !it.contains("select", true) && it.isNotBlank() }

                                if (matchedOpt != null) {
                                    fallbackActions.add(PlannedWebAction(action = "SELECT", targetIndex = selectNode.index, value = matchedOpt))
                                }
                            }

                            // 2. Target digits/OTP
                            val digitInputs = nodes.filter { it.isEditable || it.role == "TEXT_FIELD" }
                            val numberMatch = Regex("""\b(\d{4,8})\b""").find(cleanGoal)?.value
                            if (digitInputs.isNotEmpty() && numberMatch != null) {
                                fallbackActions.add(PlannedWebAction(action = "TYPE", targetIndex = digitInputs.first().index, text = numberMatch))
                            }

                            // 3. Submit button
                            val submitNode = nodes.find {
                                it.role == "BUTTON" && (
                                    it.semanticLabel.contains("submit", true) ||
                                    it.semanticLabel.contains("verify", true) ||
                                    it.semanticLabel.contains("continue", true) ||
                                    it.semanticLabel.contains("proceed", true)
                                )
                            }
                            if (submitNode != null) {
                                fallbackActions.add(PlannedWebAction(action = "CLICK", targetIndex = submitNode.index))
                            }

                            if (fallbackActions.isNotEmpty()) {
                                stage2Actions = fallbackActions
                            }
                        }

                        if (stage2Actions.isNotEmpty()) {
                            onProgressUpdate("⚡ Executing ${stage2Actions.size} verification actions...")
                            val stage2Map = postSubmissionState.nodes.associateBy { it.index }
                            for ((sIdx, sPlan) in stage2Actions.withIndex()) {
                                val sNode = stage2Map[sPlan.targetIndex]
                                val sLabel = sNode?.semanticLabel ?: "Element #${sPlan.targetIndex}"
                                val sId = "k_node_${sPlan.targetIndex}"
                                when (sPlan.action) {
                                    "TYPE" -> {
                                        val txt = sPlan.text ?: ""
                                        onProgressUpdate("⌨️ (${sIdx + 1}/${stage2Actions.size}) Entering '$txt' in [$sLabel]")
                                        actionExecutor.execute(AgentAction.TypeText(sPlan.targetIndex, txt), stage2Map, cleanGoal)
                                        fieldsFilled.add(sLabel to txt)
                                        actionsTaken.add("Entered '$txt' in $sLabel")
                                        delay(400L)
                                    }
                                    "SELECT" -> {
                                        val v = sPlan.value ?: ""
                                        onProgressUpdate("🔽 (${sIdx + 1}/${stage2Actions.size}) Selecting '$v' in [$sLabel]")
                                        webController.selectOption(sId, v)
                                        fieldsFilled.add(sLabel to v)
                                        actionsTaken.add("Selected '$v' in $sLabel")
                                        delay(400L)
                                    }
                                    "CLICK", "SUBMIT" -> {
                                        onProgressUpdate("👆 (${sIdx + 1}/${stage2Actions.size}) Submitting verification [$sLabel]")
                                        actionExecutor.execute(AgentAction.Tap(sPlan.targetIndex), stage2Map, cleanGoal)
                                        actionsTaken.add("Submitted verification")
                                        delay(1500L)
                                    }
                                }
                            }

                            // Wait for post-verification dashboard transition
                            val verifyStart = System.currentTimeMillis()
                            while (System.currentTimeMillis() - verifyStart < 8000L) {
                                delay(600L)
                                val curr = webController.distillWebState()
                                val isDashboard = curr.nodes.any {
                                    it.semanticLabel.contains("attendance", true) ||
                                    it.semanticLabel.contains("dashboard", true) ||
                                    it.semanticLabel.contains("logout", true) ||
                                    it.semanticLabel.contains("course", true) ||
                                    it.semanticLabel.contains("semester", true)
                                }
                                if (isDashboard) {
                                    postSubmissionState = curr
                                    break
                                }
                            }
                            if (postSubmissionState == null) {
                                postSubmissionState = webController.distillWebState()
                            }
                        }
                    }
                }

                val postText = postSubmissionState?.nodes?.mapNotNull { it.text }
                    ?.find { it.contains("thank", true) || it.contains("success", true) || it.contains("welcome", true) || it.contains("attendance", true) }
                    ?: if (submitTriggered) "Current page: ${postSubmissionState?.activityName}" else null

                val extracted = postText ?: extractRelevantText(cleanGoal, initialDistilled)

                val summary = buildString {
                    if (fieldsFilled.isNotEmpty()) {
                        append("Successfully executed autonomous form filling (${fieldsFilled.size} fields via ${aiConfig.provider}): ")
                        append(fieldsFilled.joinToString(", ") { "${it.first}=${it.second}" })
                        append(". ")
                    }
                    if (actionsTaken.isNotEmpty()) {
                        append("Actions executed: ${actionsTaken.joinToString(" ➔ ")}. ")
                    }
                    if (extracted.isNotBlank()) {
                        append("\n\nExtracted / Status:\n$extracted")
                    }
                }.trim()

                val liveTargetUrl = postSubmissionState?.activityName?.takeIf { it.isNotBlank() && !it.endsWith("blank") }
                    ?: (webController as? HiddenWebController)?.getWebView()?.url
                    ?: initialUrl

                return WebEpisodeResult(
                    success = true,
                    finalUrl = liveTargetUrl,
                    summary = summary,
                    turnsExecuted = plannedActions.size,
                    fieldsFilled = fieldsFilled,
                    actionsTaken = actionsTaken,
                    extractedContent = extracted
                )
            } else {
                Log.w(TAG, "${aiConfig.provider} returned 0 planned actions. Running heuristic fallback...")
                onProgressUpdate("ℹ️ ${aiConfig.provider} planned 0 actions. Running fallback...")
            }
        }

        // 2b. On-Device Neural Planning (Karvish-Web INT8 ONNX Engine, 0 Cloud Tokens)
        if (webOnDeviceEngine != null && webOnDeviceEngine.isAvailable() && (aiConfig == null || !aiConfig.isConfigured || actionsTaken.isEmpty())) {
            onProgressUpdate("⚡ Running Karvish-Web On-Device Model (26M INT8, 0 tokens)...")
            var currentTurn = 0
            var isGoalCompleted = false
            var lastExtractedContent: String? = null

            while (currentTurn < MAX_TURNS && !isGoalCompleted) {
                currentTurn++
                (webController as? HiddenWebController)?.waitForDomStability(timeoutMs = 2500L)
                val state = webController.distillWebState()

                // Check for CAPTCHA security code first
                val captchaInfo = captchaSolver.detectCaptcha(webController)
                if (captchaInfo != null) {
                    onProgressUpdate("🔐 Detecting and solving security code locally...")
                    val code = captchaSolver.solve(captchaInfo.base64Data)
                    if (!code.isNullOrBlank()) {
                        webController.typeText(captchaInfo.inputKarvishId, code)
                        fieldsFilled.add("Security Code" to code)
                        actionsTaken.add("Entered Security Code '$code'")
                        delay(400L)
                        webController.submit()
                        actionsTaken.add("Submitted form with security code")
                        delay(1500L)
                        continue
                    }
                }

                // Query on-device neural model
                val decision = webOnDeviceEngine.predictNextAction(cleanGoal, state)
                val targetNode = decision.targetNode
                val confPercent = (decision.confidence * 100).toInt()

                Log.i(TAG, "OnDevice Turn $currentTurn: action=${decision.action}, target=${targetNode?.semanticLabel}, conf=$confPercent%, time=${decision.inferenceTimeMs}ms")

                when (decision.action) {
                    "CLICK" -> {
                        if (targetNode != null) {
                            val karvishId = "k_node_${targetNode.index}"
                            onProgressUpdate("👆 [${decision.inferenceTimeMs}ms] Tapping [${targetNode.semanticLabel}] ($confPercent% conf)...")
                            val touched = (webController as? HiddenWebController)?.dispatchPhysicalTouch(
                                targetNode.bounds.centerX.toFloat(),
                                targetNode.bounds.centerY.toFloat()
                            ) ?: false
                            if (!touched) {
                                webController.clickElement(karvishId)
                            }
                            actionsTaken.add("Clicked ${targetNode.semanticLabel}")
                            val lbl = targetNode.semanticLabel.lowercase()
                            if (lbl.contains("submit") || lbl.contains("login") || lbl.contains("search") || lbl.contains("sign in")) {
                                delay(1200L)
                            } else {
                                delay(600L)
                            }
                        } else {
                            delay(400L)
                        }
                    }
                    "TYPE_TEXT" -> {
                        if (targetNode != null) {
                            val karvishId = "k_node_${targetNode.index}"
                            val textToType = decision.argument ?: extractArgumentFallback(cleanGoal, targetNode)
                            onProgressUpdate("⌨️ [${decision.inferenceTimeMs}ms] Entering '$textToType' in [${targetNode.semanticLabel}] ($confPercent% conf)...")
                            webController.typeText(karvishId, textToType)
                            fieldsFilled.add(targetNode.semanticLabel to textToType)
                            actionsTaken.add("Entered '$textToType' in ${targetNode.semanticLabel}")
                            delay(400L)
                        } else {
                            delay(400L)
                        }
                    }
                    "SELECT" -> {
                        if (targetNode != null) {
                            val karvishId = "k_node_${targetNode.index}"
                            val valToSelect = decision.argument ?: ""
                            onProgressUpdate("🔽 [${decision.inferenceTimeMs}ms] Selecting '$valToSelect' in [${targetNode.semanticLabel}]...")
                            webController.selectOption(karvishId, valToSelect)
                            fieldsFilled.add(targetNode.semanticLabel to valToSelect)
                            actionsTaken.add("Selected '$valToSelect' in ${targetNode.semanticLabel}")
                            delay(400L)
                        } else {
                            delay(400L)
                        }
                    }
                    "SUBMIT" -> {
                        onProgressUpdate("🚀 [${decision.inferenceTimeMs}ms] Submitting web form ($confPercent% conf)...")
                        webController.submit()
                        actionsTaken.add("Submitted form")
                        delay(1500L)
                    }
                    "SCROLL" -> {
                        onProgressUpdate("📜 [${decision.inferenceTimeMs}ms] Scrolling page down...")
                        webController.scroll(ScrollDirection.DOWN)
                        actionsTaken.add("Scrolled down")
                        delay(600L)
                    }
                    "EXTRACT" -> {
                        onProgressUpdate("✅ [${decision.inferenceTimeMs}ms] Target content identified!")
                        isGoalCompleted = true
                        lastExtractedContent = targetNode?.semanticLabel ?: state.activityName
                        break
                    }
                    "WAIT" -> {
                        delay(800L)
                    }
                    "NAVIGATE" -> {
                        delay(600L)
                    }
                }

                // Check termination conditions
                val postState = webController.distillWebState()
                val hasSuccessMessage = postState.nodes.any { n ->
                    val txt = (n.text ?: "").lowercase()
                    txt.contains("success") || txt.contains("thank you") || txt.contains("dashboard") ||
                            txt.contains("welcome") || txt.contains("confirmed") || txt.contains("results")
                }
                if (hasSuccessMessage) {
                    isGoalCompleted = true
                    lastExtractedContent = postState.nodes.find {
                        val t = (it.text ?: "").lowercase()
                        t.contains("success") || t.contains("thank") || t.contains("results")
                    }?.text
                    break
                }
            }

            if (actionsTaken.isNotEmpty()) {
                val finalState = webController.distillWebState()
                val summaryText = if (isGoalCompleted) {
                    "⚡ **Karvish-Web On-Device Autonomous Execution Completed:**\n\n" +
                            actionsTaken.joinToString("\n") { "• $it" } + "\n\n" +
                            (lastExtractedContent?.let { "📄 **Extracted Output:**\n$it\n\n" } ?: "") +
                            "🌐 Final URL: `${finalState.activityName}`"
                } else {
                    "⚡ **Karvish-Web Executed ${actionsTaken.size} On-Device Actions:**\n\n" +
                            actionsTaken.joinToString("\n") { "• $it" } + "\n\n" +
                            "🌐 Final URL: `${finalState.activityName}`"
                }

                val extractedImages = (finalState.images + finalState.nodes.mapNotNull { it.imageUrl })
                    .filter { it.startsWith("http://") || it.startsWith("https://") }
                    .distinct()
                    .take(3)

                return WebEpisodeResult(
                    success = true,
                    finalUrl = finalState.activityName.orEmpty(),
                    summary = summaryText,
                    turnsExecuted = currentTurn,
                    fieldsFilled = fieldsFilled,
                    actionsTaken = actionsTaken,
                    extractedContent = lastExtractedContent,
                    imageUrls = extractedImages
                )
            }
        }

        // 3. Fallback: Parse Form Entities if user requested form filling
        val pendingFormFields = extractFormEntities(cleanGoal).toMutableMap()
        val filledKeys = mutableSetOf<String>()

        var currentTurn = 0
        var isGoalCompleted = false
        var lastExtractedContent: String? = null

        while (currentTurn < MAX_TURNS && !isGoalCompleted) {
            currentTurn++
            onProgressUpdate("Turn $currentTurn/$MAX_TURNS: Observing web state...")

            val state = webController.distillWebState()
            val candidateMap = state.nodes.associateBy { it.index }

            Log.i(TAG, "Turn $currentTurn: Observed ${state.nodes.size} candidates on ${state.activityName}")

            // A. Check if Form Fields need to be filled
            if (pendingFormFields.isNotEmpty() && filledKeys.size < pendingFormFields.size) {
                var fieldFilledInThisTurn = false

                for ((key, value) in pendingFormFields) {
                    if (filledKeys.contains(key)) continue

                    // Find matching editable or select candidate
                    val targetField = findMatchingInput(key, state.nodes)
                    if (targetField != null) {
                        if (targetField.role == "SELECT") {
                            onProgressUpdate("🔽 Selecting '$value' in [${targetField.semanticLabel}]...")
                            val karvishId = "k_node_${targetField.index}"
                            val success = webController.selectOption(karvishId, value)
                            if (success) {
                                filledKeys.add(key)
                                fieldsFilled.add(key to value)
                                actionsTaken.add("Selected '$value' in ${targetField.semanticLabel}")
                                fieldFilledInThisTurn = true
                                delay(400L)
                                break
                            }
                        } else {
                            onProgressUpdate("⌨️ Entering '$value' into [${targetField.semanticLabel}]...")
                            val action = AgentAction.TypeText(targetField.index, value)
                            val success = actionExecutor.execute(action, candidateMap, cleanGoal)

                            if (success) {
                                filledKeys.add(key)
                                fieldsFilled.add(key to value)
                                actionsTaken.add("Entered '$value' in ${targetField.semanticLabel}")
                                fieldFilledInThisTurn = true
                                delay(500L)
                                break // Re-observe after input
                            }
                        }
                    }
                }

                if (fieldFilledInThisTurn) {
                    continue
                }
            }

            // B. If all form fields are filled, solve captcha if present, then tap Submit / Login
            if (pendingFormFields.isNotEmpty() && filledKeys.size >= pendingFormFields.size) {
                // Check if a CAPTCHA security code is on this page and solve it
                val captchaInfo = captchaSolver.detectCaptcha(webController)
                if (captchaInfo != null && !filledKeys.contains("captcha")) {
                    onProgressUpdate("🔐 Detecting and reading security code...")
                    val solvedCode = captchaSolver.solve(captchaInfo.base64Data)
                    if (!solvedCode.isNullOrBlank()) {
                        onProgressUpdate("✍️ Entering security code '$solvedCode'...")
                        webController.typeText(captchaInfo.inputKarvishId, solvedCode)
                        filledKeys.add("captcha")
                        fieldsFilled.add("Security Code" to solvedCode)
                        actionsTaken.add("Entered Security Code '$solvedCode'")
                        delay(400L)
                    }
                }

                val submitBtn = state.nodes.firstOrNull { node ->
                    node.isClickable && (
                        node.semanticLabel.contains("login", true) ||
                        node.semanticLabel.contains("log in", true) ||
                        node.semanticLabel.contains("sign in", true)
                    ) && !node.semanticLabel.contains("refresh", true) && !node.semanticLabel.contains("forgot", true)
                } ?: state.nodes.firstOrNull { node ->
                    node.isClickable && (
                        node.semanticLabel.contains("submit", true) ||
                        node.semanticLabel.contains("continue", true) ||
                        node.semanticLabel.contains("send", true) ||
                        node.semanticLabel.contains("register", true) ||
                        node.semanticLabel.contains("sign up", true)
                    ) && !node.semanticLabel.contains("refresh", true) && !node.semanticLabel.contains("forgot", true)
                }

                if (submitBtn != null) {
                    onProgressUpdate("👆 Tapping [${submitBtn.semanticLabel}] to submit...")
                    val action = AgentAction.Tap(submitBtn.index)
                    actionExecutor.execute(action, candidateMap, cleanGoal)
                    actionsTaken.add("Tapped ${submitBtn.semanticLabel}")
                    delay(1200L) // Wait for submission transition

                    // Verify post-submission
                    val postState = webController.distillWebState()
                    val confirmationText = postState.nodes
                        .mapNotNull { it.text }
                        .find { it.contains("thank", true) || it.contains("success", true) || it.contains("received", true) }

                    isGoalCompleted = true
                    lastExtractedContent = confirmationText ?: "Form submitted. Current page: ${postState.activityName}"
                    break
                }
            }

            // C. Search & Click Workflow (e.g. "search Quantum Computing and click History")
            val searchDirective = extractSearchDirective(cleanGoal)
            if (searchDirective != null && !actionsTaken.any { it.startsWith("Typed search query") }) {
                val searchInput = state.nodes.firstOrNull { it.isEditable || it.role == "SEARCH_FIELD" }
                if (searchInput != null) {
                    onProgressUpdate("⌨️ Typing '$searchDirective' into search box...")
                    actionExecutor.execute(AgentAction.TypeText(searchInput.index, searchDirective), candidateMap, cleanGoal)
                    actionsTaken.add("Typed search query '$searchDirective'")
                    delay(400L)

                    // Find and click search submit
                    val searchBtn = state.nodes.firstOrNull {
                        it.isClickable && (it.role == "BUTTON" || it.semanticLabel.contains("search", true) || it.semanticLabel.contains("go", true))
                    }
                    if (searchBtn != null) {
                        onProgressUpdate("👆 Tapping search button...")
                        actionExecutor.execute(AgentAction.Tap(searchBtn.index), candidateMap, cleanGoal)
                        actionsTaken.add("Tapped search button")
                    } else {
                        actionExecutor.execute(AgentAction.Submit, candidateMap, cleanGoal)
                        actionsTaken.add("Dispatched form submit")
                    }
                    delay(1200L)
                    continue
                }
            }

            // D. Target Click / Section Navigation (e.g. "click History", "click Electronics")
            val targetClickKeyword = extractTargetClickKeyword(cleanGoal, actionsTaken)
            if (targetClickKeyword != null) {
                val matchingClickable = state.nodes.firstOrNull { node ->
                    node.isClickable && node.semanticLabel.contains(targetClickKeyword, true)
                }
                if (matchingClickable != null) {
                    onProgressUpdate("👆 Clicking section [${matchingClickable.semanticLabel}]...")
                    actionExecutor.execute(AgentAction.Tap(matchingClickable.index), candidateMap, cleanGoal)
                    actionsTaken.add("Clicked '${matchingClickable.semanticLabel}'")
                    delay(1000L)
                    continue
                }
            }

            // E. Extraction & Completion
            onProgressUpdate("📊 Extracting content from page...")
            val extracted = extractRelevantText(cleanGoal, state)
            lastExtractedContent = extracted
            isGoalCompleted = true
            break
        }

        val summary = buildString {
            if (fieldsFilled.isNotEmpty()) {
                append("Successfully filled ${fieldsFilled.size} form fields: ")
                append(fieldsFilled.joinToString(", ") { "${it.first}=${it.second}" })
                append(". ")
            }
            if (actionsTaken.isNotEmpty()) {
                append("Actions executed: ${actionsTaken.joinToString(" ➔ ")}. ")
            }
            if (lastExtractedContent != null) {
                append("\n\nExtracted Information:\n$lastExtractedContent")
            }
        }.trim()

        val finalState = webController.distillWebState()
        val extractedImages = (finalState.images + finalState.nodes.mapNotNull { it.imageUrl })
            .filter { it.startsWith("http://") || it.startsWith("https://") }
            .distinct()
            .take(3)

        return WebEpisodeResult(
            success = true,
            finalUrl = initialUrl,
            summary = summary,
            turnsExecuted = currentTurn,
            fieldsFilled = fieldsFilled,
            actionsTaken = actionsTaken,
            extractedContent = lastExtractedContent,
            imageUrls = extractedImages
        )
    }

    private fun resolveInitialUrl(goal: String): String {
        val urlRegex = Regex("""https?://[^\s]+|\b([a-zA-Z0-9-]+\.(?:com|org|in|net|edu|io|co|gov)[^\s]*)""", RegexOption.IGNORE_CASE)
        val match = urlRegex.find(goal)
        if (match != null) {
            val raw = match.value.trim()
                .trimEnd(',', '.', ';', ':', ')', ']', '}', '"', '\'', '>', '?')
                .trimStart('(', '[', '{', '<', '"', '\'')
            return if (raw.startsWith("http://") || raw.startsWith("https://")) raw else "https://$raw"
        }

        val lower = goal.lowercase()
        return when {
            lower.contains("wikipedia") -> "https://en.wikipedia.org"
            lower.contains("amazon") -> "https://www.amazon.in"
            lower.contains("flipkart") -> "https://www.flipkart.com"
            lower.contains("zepto") -> "https://www.zeptonow.com"
            lower.contains("blinkit") -> "https://blinkit.com"
            lower.contains("reddit") -> "https://www.reddit.com"
            lower.contains("github") -> "https://github.com"
            WebProductFinder.isGroceryQuery(goal) -> "https://www.zeptonow.com"
            WebProductFinder.isShoppingQuery(goal) -> "https://www.amazon.in"
            else -> "https://www.google.com/search?q=" + URLEncoder.encode(goal, "UTF-8")
        }
    }

    private fun extractFormEntities(goal: String): Map<String, String> {
        val entities = mutableMapOf<String, String>()
        val kvRegex = Regex("""\b([a-zA-Z_]+)\s*[:=]\s*([^,:\n]+?)(?=\s*,\s*|\s+[a-zA-Z_]+\s*[:=]|\s*$)""", RegexOption.IGNORE_CASE)
        for (match in kvRegex.findAll(goal)) {
            val key = match.groupValues[1].trim().lowercase()
            val value = match.groupValues[2].trim()
            if (key !in setOf("http", "https", "form", "submit", "open", "url") && value.isNotBlank()) {
                entities[key] = value
            }
        }

        // USN extraction: "USN 1RR22CS001" or "USN: 1RR22CS001"
        if (!entities.containsKey("usn") && !entities.containsKey("username")) {
            val usnMatch = Regex("""\bUSN\s*[:=]?\s*([a-zA-Z0-9]+)\b""", RegexOption.IGNORE_CASE).find(goal)
            if (usnMatch != null) {
                entities["usn"] = usnMatch.groupValues[1].trim()
            }
        }

        // DOB extraction: "DOB 29 September 2005", "date of birth 29 Sep 2005", or "DOB 29-09-2005"
        val dobTextMatch = Regex("""\b(?:DOB|date\s+of\s+birth|birth\s*date)\s*[:=]?\s*([0-9]{1,2})\s+([a-zA-Z]+)\s+([0-9]{4})\b""", RegexOption.IGNORE_CASE).find(goal)
        if (dobTextMatch != null) {
            entities["dd"] = dobTextMatch.groupValues[1].trim()
            entities["mm"] = dobTextMatch.groupValues[2].trim()
            entities["yyyy"] = dobTextMatch.groupValues[3].trim()
        } else {
            val dobNumMatch = Regex("""\b(?:DOB|date\s+of\s+birth|birth\s*date)\s*[:=]?\s*([0-9]{1,2})[-/]([0-9]{1,2})[-/]([0-9]{4})\b""", RegexOption.IGNORE_CASE).find(goal)
            if (dobNumMatch != null) {
                entities["dd"] = dobNumMatch.groupValues[1].trim()
                entities["mm"] = dobNumMatch.groupValues[2].trim()
                entities["yyyy"] = dobNumMatch.groupValues[3].trim()
            }
        }

        return entities
    }

    private fun findMatchingInput(key: String, nodes: List<DistilledNode>): DistilledNode? {
        val lowerKey = key.lowercase()
        return nodes.filter { it.isEditable || it.role == "TEXT_FIELD" || it.role == "SEARCH_FIELD" || it.role == "SELECT" }
            .minByOrNull { node ->
                val label = node.semanticLabel.lowercase()
                val res = node.resourceId?.lowercase().orEmpty()
                val hint = node.hintText?.lowercase().orEmpty()
                when {
                    label == lowerKey || res == lowerKey || hint == lowerKey -> 0
                    label.contains(lowerKey) || res.contains(lowerKey) || hint.contains(lowerKey) -> 1
                    lowerKey == "usn" && (label.contains("username") || res.contains("username") || label.contains("usn")) -> 2
                    lowerKey == "dd" && (label.contains("day") || res.contains("dd") || label.contains("dd")) -> 2
                    lowerKey == "mm" && (label.contains("month") || res.contains("mm") || label.contains("mm")) -> 2
                    lowerKey == "yyyy" && (label.contains("year") || res.contains("yyyy") || label.contains("yyyy")) -> 2
                    lowerKey == "name" && (label.contains("first") || label.contains("full")) -> 3
                    lowerKey == "email" && label.contains("mail") -> 3
                    lowerKey == "message" && (label.contains("comment") || label.contains("inquiry")) -> 3
                    else -> 99
                }
            }?.takeIf { node ->
                val label = node.semanticLabel.lowercase()
                val res = node.resourceId?.lowercase().orEmpty()
                val hint = node.hintText?.lowercase().orEmpty()
                label.contains(lowerKey) || res.contains(lowerKey) || hint.contains(lowerKey) ||
                    (lowerKey == "usn" && (label.contains("username") || res.contains("username") || label.contains("usn"))) ||
                    (lowerKey == "dd" && (label.contains("day") || res.contains("dd"))) ||
                    (lowerKey == "mm" && (label.contains("month") || res.contains("mm"))) ||
                    (lowerKey == "yyyy" && (label.contains("year") || res.contains("yyyy"))) ||
                    (lowerKey == "name" && label.contains("name")) ||
                    (lowerKey == "email" && label.contains("mail")) ||
                    (lowerKey == "message" && (label.contains("comment") || label.contains("message")))
            }
    }

    private fun extractSearchDirective(goal: String): String? {
        val lower = goal.lowercase()
        val searchIdx = lower.indexOf("search ")
        if (searchIdx != -1) {
            val after = goal.substring(searchIdx + 7).trim()
            val andIdx = after.indexOf(" and ", ignoreCase = true)
            val thenIdx = after.indexOf(" then ", ignoreCase = true)
            val endIdx = listOfNotNull(
                if (andIdx != -1) andIdx else null,
                if (thenIdx != -1) thenIdx else null
            ).minOrNull() ?: after.length
            return after.substring(0, endIdx).replace("\"", "").trim()
        }
        return null
    }

    private fun extractTargetClickKeyword(goal: String, actionsTaken: List<String>): String? {
        val lower = goal.lowercase()
        val clickIdx = lower.indexOf("click ")
        if (clickIdx != -1) {
            val after = goal.substring(clickIdx + 6).trim()
            val word = after.split(Regex("[^a-zA-Z0-9]+")).firstOrNull()?.trim()
            if (word != null && word.length >= 3 && !actionsTaken.any { it.contains(word, true) }) {
                return word
            }
        }
        return null
    }

    private fun extractArgumentFallback(goal: String, targetNode: DistilledNode): String {
        val formEntities = extractFormEntities(goal)
        val targetLabel = targetNode.semanticLabel.lowercase()
        for ((key, value) in formEntities) {
            if (targetLabel.contains(key, ignoreCase = true)) {
                return value
            }
        }
        val searchDir = extractSearchDirective(goal)
        if (!searchDir.isNullOrBlank() && (targetNode.isEditable || targetNode.role == "SEARCH_FIELD")) {
            return searchDir
        }
        return goal.trim()
    }

    private fun extractRelevantText(goal: String, state: DistilledUiState): String {
        val navChromeBlacklist = setOf(
            "main menu", "images", "google apps", "sign in", "google search", "search with ai",
            "quick settings", "privacy", "terms", "tools", "feedback", "advertising", "business",
            "about", "settings", "all", "videos", "news", "maps", "shopping", "books", "flights",
            "finance", "next >", "clear search", "switch account", "dark theme", "light theme",
            "share", "close", "menu", "search", "more results", "past hour", "past 24 hours",
            "past week", "past month", "past year", "verbatim", "any time", "all results"
        )

        val texts = state.nodes
            .filter { !it.isEditable && it.text != null && it.text.isNotBlank() }
            .map { it.text!!.trim() }
            .distinct()
            .filter { txt ->
                val lower = txt.lowercase()
                !navChromeBlacklist.contains(lower) && !txt.contains("©")
            }

        val priceTexts = texts.filter { it.contains("₹") || it.contains("Rs", true) || it.contains("$") }
        val substantiveTexts = texts.filter { it.length in 12..160 && !it.startsWith("http") }

        return buildString {
            if (priceTexts.isNotEmpty()) {
                append("Prices & Key Badges:\n")
                priceTexts.take(5).forEach { append(" • $it\n") }
                append("\n")
            }
            if (substantiveTexts.isNotEmpty()) {
                append("Key Findings:\n")
                substantiveTexts.take(6).forEach { append(" • $it\n") }
            }
        }.trim().ifBlank {
            texts.filter { it.length >= 8 }.take(5).joinToString("\n• ", prefix = "• ")
        }
    }

    private suspend fun ensureRequiredFormFieldsFilled(
        webController: WebController,
        candidateMap: Map<Int, DistilledNode>,
        cleanGoal: String,
        fieldsFilled: MutableList<Pair<String, String>>,
        actionsTaken: MutableList<String>
    ) {
        // 1. Check USN field (if present in goal and not yet filled)
        val usnMatch = Regex("""\b(1[A-Za-z]{2}\d{2}[A-Za-z]{2}\d{3})\b""").find(cleanGoal)?.value
            ?: Regex("""usn\s*[:=]?\s*([A-Za-z0-9]+)""", RegexOption.IGNORE_CASE).find(cleanGoal)?.groupValues?.get(1)
        if (usnMatch != null && fieldsFilled.none { it.second.equals(usnMatch, true) }) {
            val usnNode = candidateMap.values.find {
                it.isEditable && (it.resourceId?.contains("usn", true) == true || it.semanticLabel.contains("usn", true) || it.semanticLabel.contains("username", true))
            }
            if (usnNode != null) {
                val karvishId = usnNode.resourceId?.takeIf { it.startsWith("k_node_") } ?: "k_node_${usnNode.index}"
                webController.typeText(karvishId, usnMatch)
                fieldsFilled.add("USN" to usnMatch)
                actionsTaken.add("Auto-filled USN '$usnMatch'")
                delay(300L)
            }
        }

        // 2. Check DOB Day dropdown (#dd)
        val dayMatch = Regex("""\b([0-2]?[0-9]|3[01])\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)""", RegexOption.IGNORE_CASE).find(cleanGoal)?.groupValues?.get(1)
            ?: Regex("""dob\s*[:=]?\s*([0-2]?[0-9]|3[01])""", RegexOption.IGNORE_CASE).find(cleanGoal)?.groupValues?.get(1)
        if (dayMatch != null && fieldsFilled.none { it.first.contains("day", true) || it.second == dayMatch }) {
            val ddNode = candidateMap.values.find {
                it.role == "SELECT" && (it.resourceId?.equals("dd", true) == true || it.semanticLabel.contains("day", true))
            }
            if (ddNode != null) {
                val cleanDay = dayMatch.toIntOrNull()?.toString() ?: dayMatch
                val karvishId = ddNode.resourceId?.takeIf { it.startsWith("k_node_") } ?: "k_node_${ddNode.index}"
                webController.selectOption(karvishId, cleanDay)
                fieldsFilled.add("Day" to cleanDay)
                actionsTaken.add("Auto-selected Day '$cleanDay'")
                delay(300L)
            }
        }

        // 3. Check DOB Month dropdown (#mm)
        val monthMatch = Regex("""\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b""", RegexOption.IGNORE_CASE).find(cleanGoal)?.value
        if (monthMatch != null && fieldsFilled.none { it.first.contains("month", true) || it.second.contains(monthMatch, true) }) {
            val mmNode = candidateMap.values.find {
                it.role == "SELECT" && (it.resourceId?.equals("mm", true) == true || it.semanticLabel.contains("month", true))
            }
            if (mmNode != null) {
                val cleanMonth = monthMatch.take(3).replaceFirstChar { it.uppercase() }
                val karvishId = mmNode.resourceId?.takeIf { it.startsWith("k_node_") } ?: "k_node_${mmNode.index}"
                webController.selectOption(karvishId, cleanMonth)
                fieldsFilled.add("Month" to cleanMonth)
                actionsTaken.add("Auto-selected Month '$cleanMonth'")
                delay(300L)
            }
        }

        // 4. Check DOB Year dropdown (#yyyy)
        val yearMatch = Regex("""\b(19\d{2}|20\d{2})\b""").find(cleanGoal)?.value
        if (yearMatch != null && fieldsFilled.none { it.first.contains("year", true) || it.second == yearMatch }) {
            val yyyyNode = candidateMap.values.find {
                it.role == "SELECT" && (it.resourceId?.equals("yyyy", true) == true || it.semanticLabel.contains("year", true))
            }
            if (yyyyNode != null) {
                val karvishId = yyyyNode.resourceId?.takeIf { it.startsWith("k_node_") } ?: "k_node_${yyyyNode.index}"
                webController.selectOption(karvishId, yearMatch)
                fieldsFilled.add("Year" to yearMatch)
                actionsTaken.add("Auto-selected Year '$yearMatch'")
                delay(300L)
            }
        }

        // 5. Encrypted Hardware Vault Fallback (Zero-Prompt Autofill)
        val vault = vaultManager
        if (vault != null) {
            for (node in candidateMap.values) {
                if (node.isEditable || node.role == "SELECT") {
                    val alreadyFilled = fieldsFilled.any {
                        it.first.equals(node.semanticLabel, true) || it.first.equals(node.resourceId, true)
                    }
                    if (!alreadyFilled) {
                        val match = vault.matchVaultFieldForNode(node)
                        if (match != null) {
                            val (vaultKey, vaultVal) = match
                            val karvishId = node.resourceId?.takeIf { it.startsWith("k_node_") } ?: "k_node_${node.index}"
                            if (node.role == "SELECT") {
                                webController.selectOption(karvishId, vaultVal)
                            } else {
                                webController.typeText(karvishId, vaultVal)
                            }
                            fieldsFilled.add(node.semanticLabel to vaultVal)
                            actionsTaken.add("Auto-filled ${node.semanticLabel} from Vault ($vaultKey)")
                            delay(300L)
                        }
                    }
                }
            }
        }
    }
}
