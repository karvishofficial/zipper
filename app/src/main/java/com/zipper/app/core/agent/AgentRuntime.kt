package com.zipper.app.core.agent

import android.util.Log
import com.zipper.app.accessibility.KarvishAccessibilityService
import com.zipper.app.core.model.DistilledNode
import com.zipper.app.core.model.DistilledUiState
import com.zipper.app.core.needle.AgentPrediction
import com.zipper.app.core.verifier.VerificationEngine
import kotlinx.coroutines.delay
import org.json.JSONArray

enum class TerminationReason {
    GOAL_COMPLETED,
    GOAL_FAILED,
    TURN_BUDGET_EXHAUSTED,
    STUCK_NO_PROGRESS,
    REPEATED_REFUSAL,
    CYCLE_DETECTED
}

class AgentRuntime(
    private val a11yService: KarvishAccessibilityService? = null,
    private val needleEngine: AgentModelEngine? = null,
    val maxTurns: Int = 15,
    val confidenceThreshold: Float = 0.30f,
    val maxConsecutiveRefusals: Int = 3,
    val maxConsecutiveIdenticalActions: Int = 2,
    val maxConsecutiveNoProgressTurns: Int = 3,
    val stepDelayMs: Long = 400L,
    val enableActiveRecovery: Boolean = true,
    var autoApproveGate: Boolean = false,
    private val stateProvider: (suspend () -> DistilledUiState)? = null,
    private val actionExecutor: (suspend (AgentAction, Map<Int, DistilledNode>) -> Boolean)? = null,
    private val customActionExecutor: ActionExecutor? = null
) {
    @Volatile var isPaused: Boolean = false
    @Volatile var isCancelled: Boolean = false

    companion object {
        private const val TAG = "KarvishAgentRuntime"

        val MESSAGING_PACKAGES = setOf(
            "com.whatsapp",
            "com.whatsapp.w4b",
            "com.instagram.android",
            "org.telegram.messenger",
            "org.telegram.plus",
            "com.google.android.apps.messaging",
            "com.samsung.android.messaging",
            "com.facebook.orca",
            "com.facebook.mlite",
            "com.discord",
            "com.slack",
            "org.thoughtcrime.securesms"
        )

        fun isLauncherOrSystemPackage(pkg: String?): Boolean {
            if (pkg.isNullOrBlank()) return true
            val lower = pkg.lowercase(java.util.Locale.ROOT)
            return lower == "com.zipper.app" ||
                   lower == "com.sec.android.app.launcher" ||
                   lower.contains("launcher") ||
                   lower == "android" ||
                   lower == "com.google.android.apps.nexuslauncher" ||
                   lower == "com.miui.home"
        }

        val EXEMPT_SYSTEM_PACKAGES = setOf(
            "android",
            "com.android.systemui",
            "com.google.android.permissioncontroller",
            "com.android.permissioncontroller",
            "com.samsung.android.honeyboard",
            "com.google.android.inputmethod.latin",
            "com.android.documentsui",
            "com.google.android.documentsui"
        )

        fun isExemptSystemPackage(pkg: String?): Boolean {
            if (pkg.isNullOrBlank()) return true
            val lower = pkg.lowercase(java.util.Locale.ROOT)
            return lower in EXEMPT_SYSTEM_PACKAGES ||
                   lower.contains("keyboard") ||
                   lower.contains("honeyboard") ||
                   lower.contains("permission") ||
                   lower.contains("inputmethod")
        }

        fun isAppMatchingTarget(pkg: String?, targetApp: String): Boolean {
            if (pkg.isNullOrBlank()) return false
            val p = pkg.lowercase(java.util.Locale.ROOT)
            val t = targetApp.lowercase(java.util.Locale.ROOT)
            return p.contains(t) ||
                   (t in setOf("play", "playstore", "play store", "vending") && (p.contains("vending") || p.contains("play"))) ||
                   (t == "zepto" && p.contains("zeptoconsumerapp")) ||
                   (t == "amazon" && p.contains("amazon")) ||
                   (t == "blinkit" && p.contains("grofers")) ||
                   (t == "settings" && (p.contains("settings") || p == "com.android.settings")) ||
                   (t == "calculator" && (p.contains("calculator") || p.contains("calc"))) ||
                   (t == "clock" && (p.contains("clock") || p.contains("alarm"))) ||
                   (t == "youtube" && p.contains("youtube")) ||
                   (t == "whatsapp" && p.contains("whatsapp")) ||
                   (t == "instagram" && p.contains("instagram"))
        }

        val KNOWN_TARGET_APPS = setOf(
            "youtube", "whatsapp", "instagram", "chrome", "firefox", "browser", "spotify",
            "calculator", "settings", "zepto", "blinkit", "swiggy", "zomato", "gmail",
            "keep", "notes", "maps", "playstore", "play store", "telegram", "twitter", "x",
            "linkedin", "reddit", "phone", "contacts", "messages", "camera", "gallery",
            "clock", "calendar", "files", "amazon", "flipkart"
        )

        val UI_ELEMENT_EXCLUDE = setOf(
            "the", "a", "an", "this", "that", "these", "those", "my", "your", "his", "her", "their",
            "search", "bar", "field", "top", "bottom", "for", "about", "cart",
            "usn", "box", "input", "form", "text", "textbox", "edittext", "button", "query",
            "card", "item", "list", "row", "col", "column", "container", "area", "screen",
            "view", "page", "dialog", "tab", "header", "footer", "password", "username",
            "email", "id", "name", "number", "code", "otp",
            "section", "category", "chip", "chipgroup", "book", "books", "app", "apps",
            "icon", "profile", "feed", "home", "menu", "option", "options", "setting",
            "element", "tile"
        )

        fun resolveTargetAppsFromGoal(goal: String, comparativeIntent: ComparativeIntent? = null): List<String> {
            if (comparativeIntent?.targetApp != null) {
                return listOf(comparativeIntent.targetApp)
            }
            val results = mutableListOf<String>()
            val hasInScreenNavQualifier = Regex(
                "\\b(?:section|tab|page|screen|button|icon|category|chip|menu|view|header|footer|tile)\\b",
                RegexOption.IGNORE_CASE
            ).containsMatchIn(goal)

            // Explicit launch verbs: "open <app>", "launch <app>", "switch to <app>", "go to <app>"
            val explicitMatches = Regex("\\b(?:open|launch|switch\\s+to|go\\s+to)\\s+([A-Za-z0-9_]+)", RegexOption.IGNORE_CASE).findAll(goal)
            for (explicitMatch in explicitMatches) {
                val candidate = explicitMatch.groupValues[1].trim().lowercase(java.util.Locale.ROOT)
                val fullMatchedText = explicitMatch.value.lowercase(java.util.Locale.ROOT)
                val isGoToOrSwitch = fullMatchedText.startsWith("go") || fullMatchedText.startsWith("switch")

                if (isGoToOrSwitch && hasInScreenNavQualifier) {
                    continue
                }

                if (isGoToOrSwitch && candidate !in KNOWN_TARGET_APPS && !goal.contains(Regex("\\b$candidate\\s+app\\b", RegexOption.IGNORE_CASE))) {
                    continue
                }

                if (candidate !in UI_ELEMENT_EXCLUDE) {
                    results.add(candidate)
                }
            }

            // Trailing preposition: "search dog on youtube", "play music in spotify"
            val prepMatches = Regex("\\b(?:on|in)\\s+([A-Za-z0-9_]+)", RegexOption.IGNORE_CASE).findAll(goal.trim())
            for (prepMatch in prepMatches) {
                val candidate = prepMatch.groupValues[1].trim().lowercase(java.util.Locale.ROOT)
                if (candidate in KNOWN_TARGET_APPS && candidate !in results) {
                    results.add(candidate)
                }
            }
            return results.distinct()
        }

        fun resolveTargetAppFromGoal(goal: String, comparativeIntent: ComparativeIntent?): String? {
            return resolveTargetAppsFromGoal(goal, comparativeIntent).firstOrNull()
        }

        internal fun isStateEffectivelyIdentical(a: DistilledUiState?, b: DistilledUiState?): Boolean {
            if (a === b) return true
            if (a == null || b == null) return false
            return a.packageName == b.packageName && a.activityName == b.activityName && a.nodes == b.nodes
        }

        internal fun isActionIdentical(aAction: AgentAction?, aRawTool: String?, bAction: AgentAction?, bRawTool: String?): Boolean {
            if (aAction == null || bAction == null) {
                return aAction == null && bAction == null && !aRawTool.isNullOrBlank() && aRawTool == bRawTool
            }
            if (aAction == bAction) return true

            return when {
                aAction is AgentAction.Tap && bAction is AgentAction.Tap -> {
                    val sameIndex = aAction.targetNodeIndex == bAction.targetNodeIndex
                    val sameIdentity = !aAction.nodeIdentity.isNullOrBlank() && aAction.nodeIdentity == bAction.nodeIdentity
                    val aLabel = aAction.semanticLabel?.trim()?.lowercase().orEmpty()
                    val bLabel = bAction.semanticLabel?.trim()?.lowercase().orEmpty()
                    val sameLabel = aLabel.isNotBlank() && aLabel == bLabel
                    sameIndex || sameIdentity || sameLabel
                }
                aAction is AgentAction.TypeText && bAction is AgentAction.TypeText -> {
                    val sameText = aAction.text.trim().equals(bAction.text.trim(), ignoreCase = true)
                    val sameIndex = aAction.targetNodeIndex == bAction.targetNodeIndex
                    val aLabel = aAction.semanticLabel?.trim()?.lowercase().orEmpty()
                    val bLabel = bAction.semanticLabel?.trim()?.lowercase().orEmpty()
                    val sameLabel = aLabel.isNotBlank() && aLabel == bLabel
                    sameText && (sameIndex || sameLabel)
                }
                aAction is AgentAction.Scroll && bAction is AgentAction.Scroll -> {
                    aAction.direction == bAction.direction && (aAction.targetNodeIndex == bAction.targetNodeIndex)
                }
                aAction is AgentAction.OpenApp && bAction is AgentAction.OpenApp -> {
                    aAction.appName.trim().equals(bAction.appName.trim(), ignoreCase = true)
                }
                aAction is AgentAction.PressBack && bAction is AgentAction.PressBack -> true
                aAction is AgentAction.Submit && bAction is AgentAction.Submit -> true
                else -> false
            }
        }

        private fun isRepeatableInputTarget(node: DistilledNode?): Boolean =
            node != null && (node.isEditable || node.role.contains("EDIT", true) || node.role.contains("INPUT", true))

        private fun isEmptyTypeText(action: AgentAction?): Boolean =
            action is AgentAction.TypeText && action.text.isBlank()

        internal fun computeEditDistance(s1: String, s2: String): Int {
            val dp = IntArray(s2.length + 1) { it }
            for (i in 1..s1.length) {
                var prev = dp[0]
                dp[0] = i
                for (j in 1..s2.length) {
                    val temp = dp[j]
                    dp[j] = if (s1[i - 1] == s2[j - 1]) prev else 1 + minOf(prev, dp[j], dp[j - 1])
                    prev = temp
                }
            }
            return dp[s2.length]
        }

        internal fun isGoalAccomplished(
            goal: String,
            action: AgentAction,
            postState: DistilledUiState,
            actionDescription: String = "",
            stepHistory: List<AgentStepRecord> = emptyList(),
            preState: DistilledUiState? = null
        ): Boolean {
            val cleanGoal = goal.replace(Regex("\\[state:[^\\]]*\\]", RegexOption.IGNORE_CASE), "")
                .replace(Regex("\\[history:[^\\]]*\\]", RegexOption.IGNORE_CASE), "")
                .replace(Regex("\\(previous action failed:[^)]*\\)", RegexOption.IGNORE_CASE), "")
                .trim()
            val lowerGoal = cleanGoal.lowercase(java.util.Locale.ROOT)
            val currentPkg = (postState.packageName ?: "").lowercase()

            // 1. Strict Grounding: A goal can NEVER be accomplished while sitting inside Karvish's own launcher
            if (currentPkg == "com.zipper.app" || currentPkg.isBlank()) {
                return false
            }

            // 2. Oracle Verifier: Check observable world state progress
            val oracleVerifier = ExternalOracleVerifier()
            val progress = oracleVerifier.evaluateProgress(cleanGoal, postState, stepHistory)

            // 3. If model explicitly declares CompleteGoal
            if (action is AgentAction.CompleteGoal) {
                if (action.summary.isBlank()) return false
                // Strict Physical Grounding: Click/tap goals must have physical evidence of target interaction
                val clickTarget = GoalArgumentResolver.resolveClickTarget(cleanGoal)?.lowercase(java.util.Locale.ROOT)
                if (clickTarget != null) {
                    val targetInteracted = stepHistory.any {
                        it.action is AgentAction.Tap && it.actionDescription.lowercase(java.util.Locale.ROOT).contains(clickTarget)
                    } || postState.nodes.any {
                        val t = (it.text ?: it.contentDescription ?: "").lowercase(java.util.Locale.ROOT)
                        t.contains(clickTarget) && (it.isSelected || it.isChecked)
                    } || progress.completedRequirements.any {
                        it.lowercase(java.util.Locale.ROOT).contains(clickTarget)
                    }
                    if (!targetInteracted) return false
                }
                val hasProgress = progress.completedRequirements.isNotEmpty() || stepHistory.isNotEmpty()
                if (!hasProgress) return false
                // AGENTS.md Law 9 & Section 18: Completion requires physical verification of all requirements
                if (!progress.isAllCompleted) return false

                // Observable state verification: if goal specifies an alarm time, verify it appears on screen
                val timeMatch = Regex("""\b(\d{1,2}:\d{2})\b""").find(cleanGoal)
                if (lowerGoal.contains("alarm") && timeMatch != null) {
                    val requestedTime = timeMatch.groupValues[1]
                    val visibleTexts = postState.nodes.mapNotNull { it.text ?: it.contentDescription }
                    return visibleTexts.any { it.contains(requestedTime) }
                }
                return true
            }

            // 4. Launch Goal: If all progress milestones are completed and goal is pure launch
            val hasActionVerbs = lowerGoal.startsWith("ask ") || lowerGoal.contains(" ask ") || 
                lowerGoal.contains("search ") || lowerGoal.contains("type ") || lowerGoal.contains("what is") || 
                lowerGoal.contains("how ") || lowerGoal.contains("tell ") || lowerGoal.contains("prompt ")
            val isPureLaunchGoal = !hasActionVerbs && !lowerGoal.contains(" and ") && !lowerGoal.contains(" then ")
            if (action is AgentAction.OpenApp && progress.isAllCompleted && isPureLaunchGoal) {
                val appName = action.appName.trim().lowercase()
                if (currentPkg.isNotBlank() && (currentPkg.contains(appName) || (appName.contains("calc") && currentPkg.contains("calc")))) {
                    return true
                }
            }

            // 5. Add to Cart Milestone Completion:
            val hasCheckoutRequest = lowerGoal.contains("checkout") || lowerGoal.contains("pay") || 
                lowerGoal.contains("place order") || lowerGoal.contains("order now") || lowerGoal.contains("buy now")
            if (!hasCheckoutRequest && progress.completedRequirements.contains("add_to_cart") && progress.isAllCompleted) {
                return true
            }

            // 6. Module 4: Instance-Delta Context Evidence Verification for Messaging
            // Requirements:
            // - Conversation container experienced a target text instance delta (K_after >= K_before + 1)
            // - Composer text buffer was vacated
            val hasMessageRequest = lowerGoal.contains("send ") || lowerGoal.contains("message ") || lowerGoal.contains("text ") ||
                lowerGoal.startsWith("say ") || lowerGoal.contains(" say ") || lowerGoal.startsWith("tell ") || lowerGoal.contains(" tell ")
            if (hasMessageRequest) {
                val typedRecord = stepHistory.lastOrNull { it.action is AgentAction.TypeText && !it.actionDescription.lowercase().contains("search") }
                val typedText = (typedRecord?.action as? AgentAction.TypeText)?.text?.trim()?.lowercase()

                if (!typedText.isNullOrBlank()) {
                    // Check if composer text is vacated
                    val composerNodes = postState.nodes.filter { it.isEditable || it.role == "INPUT" || it.className?.contains("EditText", ignoreCase = true) == true }
                    val isComposerVacated = composerNodes.isEmpty() || composerNodes.all {
                        val t = it.text?.trim()?.lowercase().orEmpty()
                        t.isEmpty() || t == "type a message" || t == "message" || t == "text message"
                    }

                    // Check instance delta in message list (non-editable nodes)
                    val kBefore = preState?.nodes?.count { n ->
                        val t = (n.text ?: n.contentDescription ?: "").lowercase()
                        t.contains(typedText) && !n.isEditable
                    } ?: 0
                    val kAfter = postState.nodes.count { n ->
                        val t = (n.text ?: n.contentDescription ?: "").lowercase()
                        t.contains(typedText) && !n.isEditable
                    }

                    val hasInstanceDelta = kAfter >= kBefore + 1
                    val isSendTap = (action is AgentAction.Tap && actionDescription.lowercase().contains("send")) ||
                        action is AgentAction.Submit

                    if ((hasInstanceDelta && isComposerVacated) || (isSendTap && hasInstanceDelta)) {
                        Log.i(TAG, "[Module 4 Evidence] Messaging verified: instance delta K_after($kAfter) >= K_before($kBefore) + 1 and composer vacated")
                        return true
                    }
                }
            }

            // 7. Module 4: Instance-Delta Context Evidence Verification for Search
            val hasSearchRequest = lowerGoal.startsWith("search ") || lowerGoal.contains(" search ") || lowerGoal.startsWith("find ")
            val hasTrailingClick = lowerGoal.contains("click") || lowerGoal.contains("play") || lowerGoal.contains("watch") || lowerGoal.contains("back") || lowerGoal.contains("open the ")
            val isComparativeOrCart = lowerGoal.contains("cheapest") || lowerGoal.contains("best offer") || lowerGoal.contains("lowest price") ||
                lowerGoal.contains("best deal") || lowerGoal.contains("cart") || lowerGoal.contains("add ") || lowerGoal.contains("buy ") || lowerGoal.contains("order ")
            if (hasSearchRequest && !hasTrailingClick && !isComparativeOrCart) {
                val queryRecord = stepHistory.lastOrNull { it.action is AgentAction.TypeText }
                val query = (queryRecord?.action as? AgentAction.TypeText)?.text?.trim()
                if (!query.isNullOrBlank()) {
                    val preCount = preState?.nodes?.size ?: 0
                    val postCount = postState.nodes.size
                    val isSearchSubmitted = action is AgentAction.Submit || (action is AgentAction.Tap && actionDescription.lowercase(java.util.Locale.ROOT).contains("search"))
                    val isSearchProgressComplete = progress.completedRequirements.any { it.startsWith("search:") } && progress.pendingRequirements.isEmpty()
                    if ((postCount > preCount || isSearchSubmitted || isSearchProgressComplete) && isSearchProgressComplete) {
                        Log.i(TAG, "[Module 4 Evidence] Search verified: Query '$query' executed and search results visible.")
                        return true
                    }
                }
            }

            // 8. Module 4: Observable Physical Evidence Verification for Calculator / Arithmetic
            // Requirements:
            // - Active window is a calculator package
            // - The goal contains an arithmetic expression (e.g. "50 - 20" -> 30)
            // - The postState contains the mathematical result in a calculation result or formula node
            val isCalcApp = currentPkg.contains("calculator") || currentPkg.contains("calc")
            if (isCalcApp) {
                val calcMatch = Regex("""\b(\d+)\s*([\+\-\*xX/÷])\s*(\d+)\b""").find(cleanGoal)
                if (calcMatch != null) {
                    val n1 = calcMatch.groupValues[1].toDoubleOrNull()
                    val op = calcMatch.groupValues[2]
                    val n2 = calcMatch.groupValues[3].toDoubleOrNull()
                    if (n1 != null && n2 != null) {
                        val expected = when (op) {
                            "+" -> n1 + n2
                            "-", "–" -> n1 - n2
                            "*", "x", "X" -> n1 * n2
                            "/", "÷" -> if (n2 != 0.0) n1 / n2 else null
                            else -> null
                        }
                        if (expected != null) {
                            val expectedIntStr = if (expected % 1.0 == 0.0) expected.toLong().toString() else expected.toString()
                            val visibleTexts = postState.nodes.mapNotNull { it.text ?: it.contentDescription }
                            val hasResult = visibleTexts.any {
                                it.equals(expectedIntStr, ignoreCase = true) ||
                                it.startsWith("$expectedIntStr ") ||
                                it.contains("$expectedIntStr Calculation result", ignoreCase = true)
                            }
                            val hasExecutedCalcAction = action is AgentAction.Tap || action is AgentAction.TypeText
                            if (hasResult && hasExecutedCalcAction) {
                                Log.i(TAG, "[Module 4 Evidence] Calculation verified: Expected result '$expectedIntStr' found on screen.")
                                return true
                            }
                        }
                    }
                }
            }

            return false
        }
    }

    private val toolSchemaGenerator = ToolSchemaGenerator()
    private val promptBuilder = AgentPromptBuilder()
    private val actionDispatcher: ActionDispatcher? = a11yService?.let { ActionDispatcher(it) }
    private val differentialEngine = StateDifferentialEngine()

    private suspend fun captureState(): DistilledUiState {
        var state = stateProvider?.invoke()
            ?: a11yService?.captureCurrentState()
            ?: DistilledUiState()
        if (state.packageName == "com.zipper.app") {
            return state
        }
        var attempts = 0
        while (state.nodes.size <= 3 && attempts < 4) {
            delay(500L)
            val retried = stateProvider?.invoke() ?: a11yService?.captureCurrentState()
            if (retried != null && retried.nodes.size > 3) {
                state = retried
                break
            } else if (retried != null && retried.nodes.isNotEmpty()) {
                state = retried
            }
            attempts++
        }
        return state
    }

    private suspend fun dispatchAction(
        action: AgentAction,
        candidateMap: Map<Int, DistilledNode>,
        goal: String,
        actionCandidates: Map<String, ActionCandidate> = emptyMap()
    ): Boolean {
        if (customActionExecutor != null) {
            return customActionExecutor.execute(action, candidateMap, goal, actionCandidates)
        }
        return actionExecutor?.invoke(action, candidateMap)
            ?: actionDispatcher?.dispatch(action, candidateMap, goal, actionCandidates)
            ?: true
    }

    private suspend fun inferAction(
        prompt: String,
        toolsJson: JSONArray,
        inferenceContext: AgentInferenceContext
    ): AgentPrediction {
        return needleEngine?.runAgentInference(prompt, toolsJson, inferenceContext)
            ?: AgentPrediction(
                action = null,
                confidence = 0.0f,
                latencyMs = 0L,
                explanation = "No model engine configured",
                rawToolName = null,
                decodeTps = null,
                peakRamMb = null,
                isRefusal = true
            )
    }

    data class AgentRunResult(
        val success: Boolean,
        val turnsExecuted: Int,
        val finalSummary: String,
        val history: List<AgentStepRecord>,
        val finalUiState: DistilledUiState?,
        val terminationReason: TerminationReason = if (success) TerminationReason.GOAL_COMPLETED else TerminationReason.TURN_BUDGET_EXHAUSTED
    )

    @Volatile
    var activeGoal: String = ""
        private set

    fun updateCurrentGoal(newGoal: String) {
        val cleaned = newGoal.trim().trim('"', '\'', '`', '\\')
        if (cleaned.isNotBlank()) {
            activeGoal = cleaned
            Log.i(TAG, "[GOAL UPDATE] Runtime activeGoal dynamically updated to: '$activeGoal'")
        }
    }

    suspend fun executeGoal(
        goal: String,
        onStepUpdate: ((AgentStepRecord) -> Unit)? = null
    ): AgentRunResult {
        val cleanGoal = goal.trim().trim('"', '\'', '`', '\\')
        activeGoal = cleanGoal
        Log.i(TAG, "=== STARTING AUTONOMOUS AGENT RUNTIME V0.2 ===")
        Log.i(TAG, "Goal: '$cleanGoal', Max Turns: $maxTurns, Conf Threshold: $confidenceThreshold")

        val stepHistory = mutableListOf<AgentStepRecord>()
        val blacklistedIndices = mutableSetOf<Int>()
        val blacklistedSemanticKeys = mutableSetOf<String>()
        var activeReflection: String? = null
        var turn = 1
        var latestUiState: DistilledUiState? = null

        var consecutiveRefusals = 0
        var consecutiveBlockedTurns = 0
        var consecutiveIdenticalActions = 0
        var consecutiveNoProgressTurns = 0
        var consecutiveCycleTurns = 0
        val stateHistory = mutableListOf<DistilledUiState>()
        // Senior Step 3: Runtime-Owned Task State & Generic Phase Tracker
        val runtimeTaskState = RuntimeTaskState(goal = cleanGoal)
        var hasUserDisambiguated = false
        var disambiguatedNodeIndex: Int? = null
        var hasUserConfirmedSend = false
        var hasUserApprovedSensitiveAction = false
        var recoveryBackCount = 0
        val maxRecoveryBacks = 2
        var sessionTargetPackage: String? = null
        var consecutiveDriftCount = 0
        val maxDriftRecoveries = 3

        val comparativeIntent = ProductExplorationEngine.detectComparativeIntent(cleanGoal)
        val explorationEngine = comparativeIntent?.let { ProductExplorationEngine(it) }

        var lastAttemptedAction: AgentAction? = null
        var lastAttemptedRawTool: String? = null
        var previousTurnState: DistilledUiState? = null

        while (turn <= maxTurns) {
            while (isPaused && !isCancelled) {
                delay(200)
            }
            if (isCancelled) {
                Log.i(TAG, "Goal execution cancelled by user via Karvish Control Overlay")
                com.zipper.app.accessibility.KarvishControlOverlay.hide()
                return AgentRunResult(
                    success = false,
                    turnsExecuted = turn,
                    finalSummary = "Cancelled by user via Karvish Control HUD",
                    history = stepHistory,
                    finalUiState = latestUiState,
                    terminationReason = TerminationReason.GOAL_FAILED
                )
            }

            // 1. SENSE: Capture current live UI state
            val currentState = captureState()
            latestUiState = currentState
            if (stateHistory.isEmpty()) {
                stateHistory.add(currentState)
            }

            // Phase synchronization & Target Package Locking
            val resolvedTargetApp = resolveTargetAppFromGoal(cleanGoal, comparativeIntent)
            if (runtimeTaskState.currentPhase == TaskPhase.REACH_CONTEXT) {
                val hasReachedTargetApp = if (resolvedTargetApp != null) {
                    isAppMatchingTarget(currentState.packageName, resolvedTargetApp)
                } else {
                    !isLauncherOrSystemPackage(currentState.packageName)
                }
                if (hasReachedTargetApp) {
                    runtimeTaskState.currentPhase = TaskPhase.FIND_TARGET
                    sessionTargetPackage = currentState.packageName
                    Log.i(TAG, "[TASK PHASE] Advanced to FIND_TARGET in ${currentState.packageName} (Locked target: $sessionTargetPackage)")
                }
            } else if (sessionTargetPackage == null && !currentState.packageName.isNullOrBlank()) {
                if (resolvedTargetApp != null && isAppMatchingTarget(currentState.packageName, resolvedTargetApp)) {
                    sessionTargetPackage = currentState.packageName
                    Log.i(TAG, "[ANTI-DRIFT] Locked target package: $sessionTargetPackage")
                } else if (!isLauncherOrSystemPackage(currentState.packageName) && !isExemptSystemPackage(currentState.packageName)) {
                    sessionTargetPackage = currentState.packageName
                    Log.i(TAG, "[ANTI-DRIFT] Inferred and locked target package: $sessionTargetPackage")
                }
            }

            // Anti-Drift Guardrail: Sub-microsecond check against accidental external app redirection (e.g. ad popups, notification head clicks)
            if (sessionTargetPackage != null &&
                !currentState.packageName.isNullOrBlank() &&
                currentState.packageName != sessionTargetPackage &&
                runtimeTaskState.currentPhase != TaskPhase.REACH_CONTEXT
            ) {
                val isExempt = isExemptSystemPackage(currentState.packageName)
                val isExplicitGoalTarget = resolvedTargetApp != null && isAppMatchingTarget(currentState.packageName, resolvedTargetApp)

                if (!isExempt && !isExplicitGoalTarget) {
                    consecutiveDriftCount++
                    Log.w(TAG, "[ANTI-DRIFT] External app drift detected: currently in '${currentState.packageName}', expected '$sessionTargetPackage'. Recovery attempt #$consecutiveDriftCount")

                    val a11y = a11yService ?: KarvishAccessibilityService.instance
                    a11y?.let { ctx ->
                        com.zipper.app.accessibility.KarvishControlOverlay.updateTurnProgress(
                            context = ctx,
                            turn = turn,
                            maxTurns = maxTurns,
                            actionName = "Anti-Drift Guardrail",
                            statusText = "External app detected: restoring target..."
                        )
                    }

                    if (consecutiveDriftCount <= 1) {
                        // Tier 1: Hardware BACK gesture dismisses 95% of popups/notification banners/in-app webviews
                        Log.i(TAG, "[ANTI-DRIFT Tier 1] Dismissing external overlay via Global BACK")
                        val backSuccess = a11y?.performGlobalBack() ?: false
                        if (!backSuccess && resolvedTargetApp != null) {
                            a11y?.openAppByName(resolvedTargetApp)
                        }
                        delay(600L)
                    } else if (consecutiveDriftCount < maxDriftRecoveries) {
                        // Tier 2: Directly re-launch target app into foreground
                        Log.i(TAG, "[ANTI-DRIFT Tier 2] Bringing target app '$sessionTargetPackage' back to foreground")
                        val launched = if (resolvedTargetApp != null) {
                            a11y?.openAppByName(resolvedTargetApp) ?: false
                        } else {
                            a11y?.launchPackage(sessionTargetPackage!!) ?: false
                        }
                        if (!launched) {
                            a11y?.launchPackage(sessionTargetPackage)
                        }
                        delay(800L)
                    } else {
                        Log.w(TAG, "[ANTI-DRIFT] Exceeded max drift recoveries ($maxDriftRecoveries), accepting new context: ${currentState.packageName}")
                        sessionTargetPackage = currentState.packageName
                        consecutiveDriftCount = 0
                    }

                    continue
                } else {
                    consecutiveDriftCount = 0
                }
            } else {
                consecutiveDriftCount = 0
            }

            // 1B. Autonomous Obstacle & Interstitial Pruning Gate
            val obstacleDecision = AutonomousObstaclePruner.evaluate(currentState, cleanGoal)
            if (obstacleDecision.isObstacleDetected && obstacleDecision.pruneAction != null) {
                Log.i(TAG, "[OBSTACLE PRUNER] Detected uninvited modal: ${obstacleDecision.reason}")
                val pruneAction = obstacleDecision.pruneAction
                val pruneDesc = "PruneObstacle('${obstacleDecision.reason}')"
                val pruneExecuted = dispatchAction(pruneAction, currentState.nodes.associateBy { it.index }, cleanGoal)
                delay(600L)
                val postPruneState = captureState()
                val pruneRecord = AgentStepRecord(
                    turn = turn,
                    action = pruneAction,
                    actionDescription = pruneDesc,
                    preScreenSummary = "${currentState.packageName} (Modal Obstacle)",
                    postScreenSummary = "${postPruneState.packageName} (${postPruneState.nodes.size} nodes)",
                    outcome = if (pruneExecuted) StepOutcome.SUCCESS_PROGRESS else StepOutcome.BLOCKED_BY_POPUP,
                    reflection = "Autonomous Obstacle Pruner cleared modal: ${obstacleDecision.reason}",
                    confidence = 1.0f,
                    latencyMs = 25L
                )
                stepHistory.add(pruneRecord)
                onStepUpdate?.invoke(pruneRecord)
                continue
            }

            // 2. Dynamic Tool Generation (Conditioned on TaskPhase with Safety Fallback)
            val generatedTools = toolSchemaGenerator.generateTools(
                uiState = currentState,
                blacklistedNodeIndices = blacklistedIndices,
                goal = cleanGoal,
                stepHistory = stepHistory,
                blacklistedSemanticKeys = blacklistedSemanticKeys,
                currentPhase = runtimeTaskState.currentPhase
            )
            val toolsJson = generatedTools.toolsJsonArray
            val candidateMap = generatedTools.candidateNodeMap
            val toolSummary = generatedTools.summary
            val actionCandidates = generatedTools.candidateMap

            // 3. Build Context Prompt
            val prompt = promptBuilder.buildPrompt(
                goal = cleanGoal,
                currentUiState = currentState,
                stepHistory = stepHistory,
                activeReflection = activeReflection,
                blacklistedIndices = blacklistedIndices
            )

            // 4. DECIDE: Action Model Inference (Feedback Conditioned) or Human-in-the-Loop Disambiguation
            val targetEntity = GoalArgumentResolver.resolveTargetEntity(cleanGoal)?.lowercase().orEmpty()
            val isCommunicationGoal = cleanGoal.contains(Regex("\\b(send|message|text|call|chat|tell|say)\\b", RegexOption.IGNORE_CASE))
            if (isCommunicationGoal && (runtimeTaskState.currentPhase == TaskPhase.SELECT_TARGET || runtimeTaskState.currentPhase == TaskPhase.FIND_TARGET) &&
                targetEntity.isNotBlank() && !hasUserDisambiguated && disambiguatedNodeIndex == null) {
                val targetTokens = targetEntity.split(Regex("[^a-z0-9]+")).filter { it.length >= 2 }
                val matchingCards = candidateMap.values.filter { node ->
                    val isCandidateNonInput = !node.isEditable && !node.role.equals("INPUT", true)
                    val res = node.resourceId?.lowercase().orEmpty()
                    val rawText = listOfNotNull(
                        node.text?.takeIf { it.isNotBlank() },
                        node.contentDescription?.takeIf { it.isNotBlank() },
                        node.semanticLabel?.takeIf { it.isNotBlank() }
                    ).firstOrNull().orEmpty()
                    val rawLower = rawText.lowercase().trim()

                    // Exclude System Navigation, Action Bars, and Search Buttons
                    val isSystemNavOrHeader = rawLower in setOf("back", "navigate up", "clear", "clear all", "search", "see all", "voice search", "camera search", "options", "menu", "close", "cancel", "more options", "unread", "recent", "messages", "channels") ||
                        res.contains("action_bar") || res.contains("back") || res.contains("clear") || res.contains("search_mag_icon") || res.contains("up_button")

                    val isRowOrCard = isCandidateNonInput && !isSystemNavOrHeader && (
                        res.contains("contact") || res.contains("conversation") || res.contains("recipient") ||
                        res.contains("row_inbox") || res.contains("inbox_row") || res.contains("user_row") ||
                        node.role in setOf("ITEM", "ROW", "CARD", "VIEW", "BUTTON")
                    )

                    // Universal Layer 1: Filter out Universal Media/Avatars/Photos
                    val isMediaOrAvatar = node.className?.contains("ImageView", true) == true ||
                        node.role.equals("IMAGE", true) ||
                        rawLower.contains(Regex("\\b(picture|photo|avatar|thumbnail|image|icon|logo)\\b"))

                    // Universal Layer 2: Filter out Universal Timestamps and Call/Status metadata
                    val isTimestampOrStatus = rawLower.contains(Regex("\\b(\\d{1,2}:\\d{2}|\\d{1,2}[smhdw]|yesterday|today|\\d{1,2}/\\d{1,2}/\\d{2,4})\\b")) ||
                        rawLower.contains(Regex("\\b(voice call|missed call|video call|online|typing|delivered|read|seen)\\b"))

                    // Universal Layer 3: Filter out Groups and Channel Snippets when target is a person
                    val isGroupOrChannel = !targetEntity.contains("group") && !targetEntity.contains("channel") &&
                        (rawLower.contains("group") || rawLower.contains("channel") || rawLower.contains("is also in this group") || rawLower.contains("is in this group"))

                    val isAux = rawLower.contains("matched messages") || rawLower.contains("matched message") ||
                        rawLower.contains("movie") || rawLower.contains("suggestion") || rawLower.contains("search for")

                    val candTokens = rawLower.split(Regex("[^a-z0-9]+")).filter { it.length >= 2 }
                    val matchesTarget = targetTokens.any { t ->
                        candTokens.any { c ->
                            c == t || c.startsWith(t) || t.startsWith(c) ||
                            (t.length >= 3 && c.length >= 3 && computeEditDistance(t, c) <= 1)
                        }
                    }

                    isRowOrCard && matchesTarget && !isMediaOrAvatar && !isTimestampOrStatus && !isGroupOrChannel && !isAux
                }

                // Clean and normalize contact name (strip auxiliary phone/status labels)
                fun cleanContactTitle(raw: String): String {
                    return raw
                        .replace(Regex("(?i)\\b\\S+\\s+is\\s+(also\\s+)?in\\s+this\\s+group.*"), "")
                        .replace(Regex("(?i)\\b(mobile|phone|work|home|main|other|picture|photo|avatar)\\b"), "")
                        .replace(Regex("[^\\p{L}\\p{N}\\s.'_\\-—]"), " ")
                        .replace(Regex("\\s+"), " ")
                        .trim()
                }

                val forbiddenContactLabels = setOf("back", "cancel", "search", "navigate up", "options", "menu", "unread", "draft", "missed voice call", "missed call", "voice call")

                val distinctContacts = matchingCards
                    .mapNotNull { node ->
                        val raw = listOfNotNull(
                            node.text?.takeIf { it.isNotBlank() },
                            node.contentDescription?.takeIf { it.isNotBlank() },
                            node.semanticLabel?.takeIf { it.isNotBlank() }
                        ).firstOrNull().orEmpty()
                        val title = cleanContactTitle(raw)
                        if (title.length >= 2 && title.lowercase() !in forbiddenContactLabels) node to title else null
                    }
                    .distinctBy { it.second.lowercase() }
                    .map { it.first }

                val hasAvailableSearchInput = candidateMap.values.any { node ->
                    val res = node.resourceId?.lowercase().orEmpty()
                    val text = node.text?.lowercase().orEmpty()
                    val desc = node.contentDescription?.lowercase().orEmpty()
                    val isSearchField = res.contains("search") || text.contains("search") || desc.contains("search") ||
                        res.contains("action_bar_search") || res.contains("menu_search") || res.contains("find")
                    val isUsable = node.isEditable || node.isClickable || node.role in setOf("INPUT", "BUTTON", "VIEW")
                    isSearchField && isUsable
                }

                val hasExactMatch = distinctContacts.any { node ->
                    val raw = listOfNotNull(
                        node.text?.takeIf { it.isNotBlank() },
                        node.contentDescription?.takeIf { it.isNotBlank() },
                        node.semanticLabel?.takeIf { it.isNotBlank() }
                    ).firstOrNull().orEmpty()
                    cleanContactTitle(raw).equals(targetEntity, ignoreCase = true)
                }

                val shouldSearchFirst = !runtimeTaskState.hasExecutedSearch && hasAvailableSearchInput && !hasExactMatch

                if (shouldSearchFirst) {
                    runtimeTaskState.currentPhase = TaskPhase.FIND_TARGET
                    Log.i(TAG, "[SELECT_TARGET] Only partial matches (or no exact match) on un-searched screen with search bar. Transitioning to FIND_TARGET to search for '$targetEntity'.")
                } else if (distinctContacts.size > 1) {
                    val contactOptions = distinctContacts.map { node ->
                        val raw = listOfNotNull(
                            node.text?.takeIf { it.isNotBlank() },
                            node.contentDescription?.takeIf { it.isNotBlank() },
                            node.semanticLabel?.takeIf { it.isNotBlank() }
                        ).firstOrNull().orEmpty()
                        cleanContactTitle(raw)
                    }
                    Log.i(TAG, "[DISAMBIGUATION] Found ${distinctContacts.size} contacts matching '$targetEntity': $contactOptions. Prompting user to select.")
                    isPaused = true
                    com.zipper.app.accessibility.KarvishControlOverlay.showDisambiguationCard(
                        title = "Select Contact",
                        prompt = "Multiple contacts match \"$targetEntity\". Say 1, 2, 3 or tap:",
                        options = contactOptions,
                        boundsList = distinctContacts.map { it.rawBounds },
                        onSelect = { selectedIdx ->
                            val chosen = distinctContacts.getOrNull(selectedIdx)
                            if (chosen != null) {
                                disambiguatedNodeIndex = chosen.index
                                hasUserDisambiguated = true
                                isPaused = false
                                Log.i(TAG, "[DISAMBIGUATION] User selected: ${chosen.text ?: chosen.contentDescription}")
                            }
                        },
                        onCancel = {
                            hasUserDisambiguated = true
                            isCancelled = true
                            isPaused = false
                            Log.i(TAG, "[DISAMBIGUATION] Disambiguation cancelled by user. Aborting goal.")
                        }
                    )
                    while (isPaused && !isCancelled) {
                        delay(200)
                    }
                    if (isCancelled) {
                        Log.i(TAG, "[DISAMBIGUATION] Aborting goal execution due to user cancellation")
                        return AgentRunResult(
                            success = false,
                            turnsExecuted = turn,
                            finalSummary = "Goal cancelled by user during contact selection",
                            history = stepHistory,
                            finalUiState = latestUiState,
                            terminationReason = TerminationReason.GOAL_FAILED
                        )
                    }
                } else if (distinctContacts.size == 1) {
                    val singleTarget = distinctContacts[0]
                    val singleRaw = listOfNotNull(
                        singleTarget.text?.takeIf { it.isNotBlank() },
                        singleTarget.contentDescription?.takeIf { it.isNotBlank() },
                        singleTarget.semanticLabel?.takeIf { it.isNotBlank() }
                    ).firstOrNull().orEmpty()
                    val singleName = cleanContactTitle(singleRaw)

                    val isExactMatch = singleName.equals(targetEntity, ignoreCase = true)

                    if (isExactMatch) {
                        disambiguatedNodeIndex = singleTarget.index
                        Log.i(TAG, "[SELECT_TARGET] Exact 1:1 match '$targetEntity' == '$singleName' (node #${singleTarget.index}). Auto-targeting with zero friction.")
                    } else {
                        // Partial match (e.g. goal is "Shobha", but contact is "Shobha Aunty", or "Ajay sir" for "Ajay"):
                        // Prompt user on HUD so they have 100% control to confirm!
                        Log.i(TAG, "[DISAMBIGUATION] Partial match found: '$singleName' for '$targetEntity'. Prompting user to confirm.")
                        isPaused = true
                        com.zipper.app.accessibility.KarvishControlOverlay.showDisambiguationCard(
                            title = "Select Contact",
                            prompt = "Found contact for \"$targetEntity\". Say 1 to confirm or tap:",
                            options = listOf(singleName),
                            boundsList = listOf(singleTarget.rawBounds),
                            onSelect = {
                                disambiguatedNodeIndex = singleTarget.index
                                hasUserDisambiguated = true
                                isPaused = false
                                Log.i(TAG, "[DISAMBIGUATION] User confirmed single partial match: $singleName")
                            },
                            onCancel = {
                                hasUserDisambiguated = true
                                isCancelled = true
                                isPaused = false
                                Log.i(TAG, "[DISAMBIGUATION] Partial match confirmation cancelled by user. Aborting goal.")
                            }
                        )
                        while (isPaused && !isCancelled) {
                            delay(200)
                        }
                        if (isCancelled) {
                            Log.i(TAG, "[DISAMBIGUATION] Aborting goal execution due to user cancellation")
                            return AgentRunResult(
                                success = false,
                                turnsExecuted = turn,
                                finalSummary = "Goal cancelled by user during contact selection",
                                history = stepHistory,
                                finalUiState = latestUiState,
                                terminationReason = TerminationReason.GOAL_FAILED
                            )
                        }
                    }
                } else if (distinctContacts.isEmpty() && !runtimeTaskState.hasExecutedSearch && hasAvailableSearchInput) {
                    runtimeTaskState.currentPhase = TaskPhase.FIND_TARGET
                    Log.i(TAG, "[SELECT_TARGET] No matching contacts on un-searched screen with search bar. Transitioning to FIND_TARGET to search.")
                }
            }

            val rawPrediction = if (disambiguatedNodeIndex != null) {
                val nodeIdx = disambiguatedNodeIndex!!
                val targetNode = candidateMap[nodeIdx] ?: currentState.nodes.find { it.index == nodeIdx }
                val targetLabel = targetNode?.text?.takeIf { it.isNotBlank() } ?: targetNode?.contentDescription ?: "contact"
                disambiguatedNodeIndex = null
                AgentPrediction(
                    action = AgentAction.Tap(
                        targetNodeIndex = nodeIdx,
                        candidateId = "tap_disambiguated_$nodeIdx",
                        semanticLabel = targetLabel,
                        nodeIdentity = targetNode?.resourceId
                    ),
                    confidence = 1.0f,
                    latencyMs = 0L,
                    explanation = "User selected '$targetLabel' via Karvish Disambiguation",
                    rawToolName = "tap_disambiguated_$nodeIdx"
                )
            } else if (runtimeTaskState.currentPhase == TaskPhase.REACH_CONTEXT &&
                       resolvedTargetApp != null &&
                       !isAppMatchingTarget(currentState.packageName, resolvedTargetApp)) {
                AgentPrediction(
                    action = AgentAction.OpenApp(resolvedTargetApp),
                    confidence = 1.0f,
                    latencyMs = 0L,
                    explanation = "Launching target app '$resolvedTargetApp' from current screen",
                    rawToolName = "open_app"
                )
            } else if (explorationEngine != null && runtimeTaskState.hasExecutedSearch) {
                val exploreAction = explorationEngine.evaluateNextAction(currentState)
                if (exploreAction != null) {
                    val toolName = when (exploreAction) {
                        is AgentAction.Scroll -> "explore_scroll_${exploreAction.direction.name.lowercase(java.util.Locale.ROOT)}"
                        is AgentAction.Tap -> "tap_add_${exploreAction.targetNodeIndex}"
                        else -> "explore_action"
                    }
                    val expl = if (explorationEngine.isWinnerActionDispatched) {
                        "Selected winner: '${explorationEngine.winner?.title}' (Price: ₹${explorationEngine.winner?.price}, Unit: ₹${"%.2f".format(explorationEngine.winner?.unitPrice ?: 0.0)}) from ${explorationEngine.totalAccumulatedCount} compared items."
                    } else {
                        "Exploring catalog depth (buffer: ${explorationEngine.totalAccumulatedCount} items)..."
                    }
                    AgentPrediction(
                        action = exploreAction,
                        confidence = 0.95f,
                        latencyMs = 0L,
                        explanation = expl,
                        rawToolName = toolName
                    )
                } else {
                    inferAction(
                        prompt = prompt,
                        toolsJson = toolsJson,
                        inferenceContext = AgentInferenceContext(
                            uiState = currentState,
                            history = stepHistory,
                            previousFailure = activeReflection,
                            currentPhase = runtimeTaskState.currentPhase,
                            lastOutcome = runtimeTaskState.lastOutcome
                        )
                    )
                }
            } else {
                inferAction(
                    prompt = prompt,
                    toolsJson = toolsJson,
                    inferenceContext = AgentInferenceContext(
                        uiState = currentState,
                        history = stepHistory,
                        previousFailure = activeReflection,
                        currentPhase = runtimeTaskState.currentPhase,
                        lastOutcome = runtimeTaskState.lastOutcome
                    )
                )
            }

            // Select active prediction with Active Recovery Top-K fallback
            var activePrediction = rawPrediction
            var targetIdx = when (val a = activePrediction.action) {
                is AgentAction.Tap -> a.targetNodeIndex
                is AgentAction.TypeText -> a.targetNodeIndex
                is AgentAction.Scroll -> a.targetNodeIndex
                else -> null
            }
            val isKeypadAffordance = when (val a = activePrediction.action) {
                is AgentAction.Tap -> {
                    val lbl = a.semanticLabel?.trim().orEmpty()
                    lbl.length <= 2 && lbl.all { it.isDigit() || it in "+-*/=.%" }
                }
                else -> false
            }
            val lastStep = stepHistory.lastOrNull()
            val lastStepHadUiTransition = lastStep != null && lastStep.outcome == StepOutcome.SUCCESS &&
                lastStep.postScreenSummary != "Unchanged"
            val repeatedNoOpAction = consecutiveIdenticalActions > 0 &&
                !lastStepHadUiTransition &&
                !isKeypadAffordance &&
                isActionIdentical(
                    activePrediction.action,
                    activePrediction.rawToolName,
                    lastAttemptedAction,
                    lastAttemptedRawTool
                )

            val isPrematureComplete = activePrediction.action is AgentAction.CompleteGoal &&
                !isGoalAccomplished(cleanGoal, activePrediction.action, currentState, "CompleteGoal", stepHistory)

            // Module 3: Typed Action Escalation with Hard Ceiling (Ceiling 3)
            if (repeatedNoOpAction) {
                runtimeTaskState.attemptCountForCurrentPhase++
                when (runtimeTaskState.attemptCountForCurrentPhase) {
                    1 -> {
                        // Attempt 1: Invalidate offending candidate and re-sample from alternatives
                        val alt = rawPrediction.alternativeProposals.firstOrNull { altProposal ->
                            val altTarget = when (val a = altProposal.action) {
                                is AgentAction.Tap -> a.targetNodeIndex
                                is AgentAction.TypeText -> a.targetNodeIndex
                                is AgentAction.Scroll -> a.targetNodeIndex
                                else -> null
                            }
                            !altProposal.isRefusal && altProposal.action != null && altProposal.action !is AgentAction.CompleteGoal &&
                                !isEmptyTypeText(altProposal.action) &&
                                !isActionIdentical(altProposal.action, altProposal.rawToolName, lastAttemptedAction, lastAttemptedRawTool) &&
                                (altTarget == null || (altTarget in candidateMap.keys && altTarget !in blacklistedIndices))
                        }
                        if (alt != null) {
                            Log.w(TAG, "[Module 3 Escalation Attempt 1]: Resampling to alternate candidate ${alt.rawToolName}")
                            activePrediction = alt
                            targetIdx = when (val a = activePrediction.action) {
                                is AgentAction.Tap -> a.targetNodeIndex
                                is AgentAction.TypeText -> a.targetNodeIndex
                                is AgentAction.Scroll -> a.targetNodeIndex
                                else -> null
                            }
                        } else {
                            Log.w(TAG, "[Module 3 Escalation Attempt 1]: No alternative proposal available; proceeding to Attempt 2")
                        }
                    }
                    2 -> {
                        // Attempt 2: If a scrollable container exists, emit recovery micro-scroll to force relayout; else Bounded WAIT
                        val scrollableCandidate = candidateMap.values.firstOrNull { it.isScrollable || it.role == "SCROLL_VIEW" || it.role == "LIST" }
                        if (scrollableCandidate != null) {
                            Log.w(TAG, "[Module 3 Escalation Attempt 2]: Emitting Layout-Refresh Micro-Scroll on node ${scrollableCandidate.index} to unfreeze layout")
                            activePrediction = AgentPrediction(
                                action = AgentAction.Scroll(
                                    targetNodeIndex = scrollableCandidate.index,
                                    direction = ScrollDirection.DOWN
                                ),
                                confidence = 0.85f,
                                latencyMs = 0L,
                                explanation = "Recovery Micro-Scroll to refresh stalled UI layout",
                                rawToolName = "recovery_scroll"
                            )
                            targetIdx = scrollableCandidate.index
                        } else {
                            Log.w(TAG, "[Module 3 Escalation Attempt 2]: Emitting Bounded WAIT (3.0s) for UI settling/mutation")
                            activePrediction = AgentPrediction(
                                action = AgentAction.Wait(3000L),
                                confidence = 0.90f,
                                latencyMs = 0L,
                                explanation = "Bounded WAIT: Waiting up to 3.0s for UI mutation/animation settling",
                                rawToolName = "bounded_wait"
                            )
                            targetIdx = null
                        }
                    }
                    3 -> {
                        // Attempt 3 (Terminal Ceiling): Emit BLOCKED with human-in-the-loop handshake
                        Log.w(TAG, "[Module 3 Escalation Attempt 3]: Hard ceiling 3 reached! Emitting BLOCKED with user handshake")
                        val blockedMsg = "Action Blocked: No valid input field or action candidate found."
                        com.zipper.app.accessibility.KarvishControlOverlay.showBlockedBanner(
                            message = blockedMsg,
                            onResume = {
                                isPaused = false
                                runtimeTaskState.attemptCountForCurrentPhase = 0
                                Log.i(TAG, "User resumed blocked execution")
                            },
                            onAbort = {
                                isCancelled = true
                                isPaused = false
                                Log.i(TAG, "User aborted blocked execution")
                            }
                        )
                        isPaused = a11yService != null
                        activePrediction = AgentPrediction(
                            action = AgentAction.Wait(500L),
                            confidence = 0.0f,
                            latencyMs = 0L,
                            explanation = "Blocked: Waiting for human guidance",
                            rawToolName = "blocked_handshake"
                        )
                        targetIdx = null
                    }
                    else -> {
                        Log.e(TAG, "[Module 3 Escalation] Hard ceiling exceeded after attempt 3; safe abort")
                        com.zipper.app.accessibility.KarvishControlOverlay.hide()
                        return AgentRunResult(
                            success = false,
                            turnsExecuted = turn,
                            finalSummary = "Action Blocked: No valid input field or target in phase ${runtimeTaskState.currentPhase}",
                            history = stepHistory,
                            finalUiState = latestUiState,
                            terminationReason = TerminationReason.STUCK_NO_PROGRESS
                        )
                    }
                }
            } else {
                runtimeTaskState.attemptCountForCurrentPhase = 0
            }

            if (enableActiveRecovery && !repeatedNoOpAction && (activePrediction.isRefusal || activePrediction.action == null ||
                isEmptyTypeText(activePrediction.action) ||
                isPrematureComplete ||
                (targetIdx != null && (targetIdx in blacklistedIndices || targetIdx !in candidateMap.keys)))) {
                val viableFallback = rawPrediction.alternativeProposals.firstOrNull { alt ->
                    val altTarget = when (val a = alt.action) {
                        is AgentAction.Tap -> a.targetNodeIndex
                        is AgentAction.TypeText -> a.targetNodeIndex
                        is AgentAction.Scroll -> a.targetNodeIndex
                        else -> null
                    }
                    !alt.isRefusal && alt.action != null && alt.action !is AgentAction.CompleteGoal &&
                        !isEmptyTypeText(alt.action) &&
                        !isActionIdentical(alt.action, alt.rawToolName, lastAttemptedAction, lastAttemptedRawTool) &&
                        (altTarget == null || (altTarget in candidateMap.keys && altTarget !in blacklistedIndices))
                }
                if (viableFallback != null) {
                    Log.i(TAG, "Active Recovery: Fell back from ${rawPrediction.rawToolName} to ${viableFallback.rawToolName}")
                    activePrediction = viableFallback
                    targetIdx = when (val a = activePrediction.action) {
                        is AgentAction.Tap -> a.targetNodeIndex
                        is AgentAction.TypeText -> a.targetNodeIndex
                        is AgentAction.Scroll -> a.targetNodeIndex
                        else -> null
                    }
                }
            }

            if (isEmptyTypeText(activePrediction.action)) {
                val viableAlternative = rawPrediction.alternativeProposals.firstOrNull { alternative ->
                    !alternative.isRefusal && alternative.action != null &&
                        !isEmptyTypeText(alternative.action) &&
                        !isActionIdentical(
                            alternative.action,
                            alternative.rawToolName,
                            lastAttemptedAction,
                            lastAttemptedRawTool
                        )
                }
                if (viableAlternative != null) {
                    activePrediction = viableAlternative
                    targetIdx = when (val action = activePrediction.action) {
                        is AgentAction.Tap -> action.targetNodeIndex
                        is AgentAction.TypeText -> action.targetNodeIndex
                        is AgentAction.Scroll -> action.targetNodeIndex
                        else -> null
                    }
                } else {
                    activePrediction = activePrediction.copy(
                        action = null,
                        confidence = 0f,
                        rawToolName = "empty_type_rejected",
                        isRefusal = true,
                        explanation = "Empty text is not a valid action."
                    )
                    targetIdx = null
                }
            }

            val prediction = activePrediction

            // MuSpark "Thinking Mode" Telemetry Stream
            val thinkReason = prediction.explanation.ifBlank { "Selected candidate '${prediction.rawToolName}'" }
            val liveThinkingMsg = "Phase: ${runtimeTaskState.currentPhase} | Conf: ${(prediction.confidence * 100).toInt()}% | $thinkReason"
            com.zipper.app.accessibility.KarvishControlOverlay.updateThinkingTrace(liveThinkingMsg)

            // 5. SAFETY GATE Evaluation
            var gatingDecision = "APPROVED"
            var isSafetyBlocked = false
            if (prediction.isRefusal || prediction.action == null) {
                gatingDecision = "BLOCKED_REFUSAL"
                isSafetyBlocked = true
            } else if (prediction.confidence < confidenceThreshold) {
                gatingDecision = "BLOCKED_LOW_CONFIDENCE (${"%.4f".format(prediction.confidence)} < $confidenceThreshold)"
                isSafetyBlocked = true
            }

            if (!isSafetyBlocked && targetIdx != null && targetIdx !in candidateMap.keys) {
                gatingDecision = "BLOCKED_UNGROUNDED (node #$targetIdx not on screen)"
                isSafetyBlocked = true
            }

            // Financial & Order Safety Gate (AGENTS.md Section 4.3):
            // Prevent unauthorized checkout, pay, or place order actions unless explicitly in user goal
            val targetLabel = targetIdx?.let { candidateMap[it]?.let { n -> listOfNotNull(n.text, n.contentDescription, n.resourceId).joinToString(" ").lowercase() } }.orEmpty()
            val isPaymentAffordance = targetLabel.contains(Regex("\\b(pay|checkout|place order|order now|proceed to pay|click to pay|buy now)\\b"))
            val userRequestedPayment = cleanGoal.contains("pay", ignoreCase = true) || 
                cleanGoal.contains("checkout", ignoreCase = true) || 
                cleanGoal.contains("place order", ignoreCase = true) || 
                cleanGoal.contains("order now", ignoreCase = true) || 
                cleanGoal.contains("buy now", ignoreCase = true)
            if (!isSafetyBlocked && isPaymentAffordance && !userRequestedPayment) {
                gatingDecision = "BLOCKED_UNAUTHORIZED_PAYMENT (User goal does not request payment/checkout)"
                isSafetyBlocked = true
            }

            // Telephony & Calling Safety Boundary (AGENTS.md Section 4.3 & 5.1):
            // Prevent unauthorized voice call, video call, or dialer actions unless explicitly requested by the user.
            // Do NOT confuse a chat/contact list row whose previous message snippet was a call with an actual call trigger button!
            val targetNode = targetIdx?.let { candidateMap[it] }
            val targetResId = targetNode?.resourceId?.lowercase().orEmpty()
            val isChatOrContactRow = targetResId.contains("contact") || targetResId.contains("conversation") ||
                targetResId.contains("recipient") || targetResId.contains("chat_row") || targetResId.contains("inbox_row") || targetResId.contains("row_inbox") ||
                (targetNode?.role in setOf("ITEM", "ROW", "CARD") && !targetResId.contains("button"))

            val targetTextDesc = targetIdx?.let { candidateMap[it]?.let { n -> listOfNotNull(n.text, n.contentDescription).joinToString(" ").lowercase() } }.orEmpty()
            val isCallAffordance = !isChatOrContactRow && (
                targetLabel.contains(Regex("\\b(start call|video call|voice call|dial|call button)\\b")) ||
                (targetLabel.contains(Regex("\\b(call)\\b")) && (targetNode?.role?.contains("BUTTON", ignoreCase = true) == true || targetResId.contains("call"))) ||
                targetTextDesc.contains(Regex("\\b(start call|video call|voice call|dial button)\\b")) ||
                targetResId.contains("action_call") || targetResId.contains("menu_call") || targetResId.contains("video_call") || targetResId.contains("dial_button")
            )
            val userRequestedCall = cleanGoal.contains("call", ignoreCase = true) || cleanGoal.contains("dial", ignoreCase = true)
            if (!isSafetyBlocked && isCallAffordance && !userRequestedCall) {
                gatingDecision = "BLOCKED_UNAUTHORIZED_CALL (User goal does not request voice/video call)"
                isSafetyBlocked = true
            }

            // Platform Validity Check: TypeText requires an editable input field
            if (!isSafetyBlocked && prediction.action is AgentAction.TypeText && targetIdx != null) {
                val targetNode = candidateMap[targetIdx]
                val isEditableField = targetNode?.isEditable == true ||
                    targetNode?.role?.contains("EDIT", ignoreCase = true) == true ||
                    targetNode?.role?.contains("INPUT", ignoreCase = true) == true ||
                    targetNode?.className?.contains("EditText", ignoreCase = true) == true
                if (!isEditableField) {
                    gatingDecision = "BLOCKED_INVALID_ACTION (node #$targetIdx is not an editable input field)"
                    isSafetyBlocked = true
                }
            }

            // Platform Safety Boundary: Submit Action Validity (AGENTS.md Section 4.2 & 5.1)
            // An IME Submit action is only structurally valid if text is present in the active input
            // or was recently entered. Submitting an empty, unpopulated input field is an invalid action.
            if (!isSafetyBlocked && prediction.action is AgentAction.Submit) {
                val focusedInput = currentState.nodes.firstOrNull { it.isFocused && (it.isEditable || it.role.contains("INPUT", true)) }
                    ?: currentState.nodes.firstOrNull { it.isEditable || it.role.contains("INPUT", true) }
                val hasText = focusedInput?.text?.isNotBlank() == true
                val hasRecentTypeText = stepHistory.takeLast(2).any { it.action is AgentAction.TypeText }
                if (!hasText && !hasRecentTypeText) {
                    gatingDecision = "BLOCKED_INVALID_ACTION (Cannot submit an empty, unpopulated input field)"
                    isSafetyBlocked = true
                }
            }

            // Platform Safety Boundary: Target Conflict Protection (AGENTS.md Section 4.3 & 5.1)
            // If the user's goal targets an explicit named entity/recipient (e.g. "Bhavani"),
            // do not allow tapping a conflicting contact/person/profile item that has 0% lexical or semantic overlap.
            if (!isSafetyBlocked && prediction.action is AgentAction.Tap && targetIdx != null) {
                val targetNode = candidateMap[targetIdx]
                val candTool = actionCandidates.values.find { it.nodeIndex == targetIdx }
                val candChildText = currentState.nodes
                    .filter { it.parentIndex == targetIdx }
                    .mapNotNull { it.text?.takeIf { t -> t.isNotBlank() } ?: it.contentDescription?.takeIf { d -> d.isNotBlank() } ?: it.semanticLabel?.takeIf { s -> s.isNotBlank() } }
                    .joinToString(" ")
                val candLabel = listOfNotNull(
                    candTool?.semanticLabel,
                    targetNode?.semanticLabel,
                    targetNode?.text,
                    targetNode?.contentDescription,
                    candChildText.takeIf { it.isNotBlank() }
                ).firstOrNull { it.isNotBlank() }?.trim().orEmpty()
                val extractedArg = (prediction.extractedText?.trim()?.takeIf { it.isNotBlank() })
                    ?: GoalArgumentResolver.resolveTargetEntity(cleanGoal)?.takeIf { it.isNotBlank() }

                if (extractedArg != null && candLabel.isNotBlank()) {
                    val resId = targetNode?.resourceId?.lowercase().orEmpty()
                    val cName = targetNode?.className?.lowercase().orEmpty()
                    val isContactOrUserCard = resId.contains("contact") ||
                        resId.contains("row_inbox") ||
                        resId.contains("conversation") ||
                        resId.contains("profile") ||
                        resId.contains("recent") ||
                        resId.contains("chat") ||
                        resId.contains("user") ||
                        resId.contains("item") ||
                        cName.contains("contact")

                    val candLower = candLabel.lowercase()
                    val resTokens = resId.split(Regex("[^a-z0-9]+")).filter { it.isNotBlank() }.toSet()
                    val hasAiToken = resTokens.any { it in setOf("ai", "bot", "assistant", "metaai", "copilot", "gemini", "chatbot") }
                    val isAiPromptOrSendTrigger = candLower in setOf("ask", "ask ai", "generate") ||
                        (candLower.startsWith("ask ") && !candLower.contains("search") && !candLower.contains("find")) ||
                        resId.contains("meta_ai") ||
                        resId.contains("metaai") ||
                        resId.contains("ai_send") ||
                        resId.contains("ai_input_send") ||
                        resId.contains("assistant_send") ||
                        (resTokens.contains("send") && hasAiToken)

                    val isRealSearchField = targetNode?.isEditable == true ||
                        targetNode?.role.equals("INPUT", ignoreCase = true) ||
                        targetNode?.className?.contains("EditText", ignoreCase = true) == true ||
                        resId.contains("search_bar") || resId.contains("search_edit_text") || resId.contains("search_src_text") ||
                        candLower.contains("search or ask")

                    val isNavOrExploreTab = targetNode?.isSelected != null ||
                        targetNode?.role.equals("TAB", ignoreCase = true) ||
                        candLower.contains("explore") || candLower.contains("tab") ||
                        candLower == "search and explore"

                    val isTrendingOrRecentCard = !isRealSearchField && !isNavOrExploreTab && (
                        candLower.startsWith("trending search") || candLower.startsWith("trending searches") ||
                        candLower.startsWith("recent search") || candLower.startsWith("popular searches") ||
                        (candLower.startsWith("search ") && candLower.length > 15 && !candLower.contains(Regex("""\b(button|bar|box|field|input|icon|tab|explore)\b""")))
                    )
                    val isSearchAffordance = isRealSearchField || isNavOrExploreTab || (!isAiPromptOrSendTrigger && !isTrendingOrRecentCard &&
                        (candLower.contains("search") || resId.contains("search")))
                    val isSystemOrNav = isSearchAffordance || isNavOrExploreTab || candLower in setOf("search", "explore", "back", "home", "submit", "clear", "cancel", "done", "close", "menu", "notifications") ||
                        (candLower.startsWith("search") && !isTrendingOrRecentCard) || candLower.startsWith("clear")

                    // Brand-Agnostic Assistant / Bot Recognition:
                    // Candidate is an AI bot distractor if it is a prompt/send trigger, or matches generic assistant signatures
                    val isAiBotCandidate = !isRealSearchField && (isAiPromptOrSendTrigger || (!isSearchAffordance && run {
                        val matchesGenericAiId = hasAiToken ||
                            resId.contains("meta_ai") || resId.contains("ai_fab") || resId.contains("ai_chat") ||
                            resId.contains("ai_input_send") || resId.contains("ai_send")
                        val matchesAiText = candLower.contains("message your assistant") ||
                            candLower.contains("chat with ai") ||
                            candLower.contains("meta ai") ||
                            candLower.contains("galaxy ai") ||
                            candLower.contains("copilot") ||
                            candLower.contains("gemini") ||
                            candLower.contains("apple intelligence") ||
                            candLower == "assistant" ||
                            candLower == "ask ai" ||
                            candLower == "ask" ||
                            // Generative query prompt chips in search results (e.g. "Ask Meta AI 'ajay'", "Ask Gemini 'ajay'")
                            (candLower.startsWith("ask ") && (candLower.contains(" ai") || candLower.contains("about") || candLower.contains("'") || candLower.contains("\"")))
                        matchesGenericAiId || matchesAiText
                    }))

                    val isTargetPerson = !extractedArg.contains(Regex("""\b(ai|meta|assistant|copilot|gemini|bot)\b"""))

                    if (isAiBotCandidate && isTargetPerson) {
                        gatingDecision = "BLOCKED_MISMATCHED_TARGET (Candidate '$candLabel' is an AI assistant, not target '$extractedArg')"
                        isSafetyBlocked = true
                    } else if (isTrendingOrRecentCard && !candLower.contains(extractedArg.lowercase())) {
                        gatingDecision = "BLOCKED_MISMATCHED_TARGET (Candidate '$candLabel' is a search suggestion topic, not target '$extractedArg')"
                        isSafetyBlocked = true
                    } else if (isContactOrUserCard && !isSystemOrNav) {
                        val isGroupMemberSnippet = candLower.contains("is also in this group") || candLower.contains("is in this group")
                        val cleanCand = candLower.replace(Regex("(?i)\\b\\S+\\s+is\\s+(also\\s+)?in\\s+this\\s+group.*"), "").trim()
                        val argTokens = extractedArg.lowercase().split(Regex("[^a-z0-9]+")).filter { it.length >= 2 }
                        val candTokens = cleanCand.split(Regex("[^a-z0-9]+")).filter { it.length >= 2 }
                        val hasOverlap = argTokens.any { t ->
                            candTokens.any { c ->
                                c == t || c.contains(t) || t.contains(c) ||
                                (t.length >= 3 && c.length >= 3 && computeEditDistance(t, c) <= 1)
                            }
                        } || (runtimeTaskState.hasExecutedSearch && isContactOrUserCard)

                        if (isGroupMemberSnippet && !extractedArg.contains("group", ignoreCase = true)) {
                            gatingDecision = "BLOCKED_MISMATCHED_TARGET (Candidate '$candLabel' is a group association snippet, not target '$extractedArg')"
                            isSafetyBlocked = true
                        } else if (!hasOverlap && argTokens.isNotEmpty()) {
                            gatingDecision = "BLOCKED_MISMATCHED_TARGET (Candidate '$candLabel' does not match target '$extractedArg')"
                            isSafetyBlocked = true
                        }
                    }
                }
            }

            // Handle Safety Blocked / Refusals
            if (isSafetyBlocked) {
                val isRefusal = prediction.isRefusal || prediction.action == null
                if (isRefusal) {
                    consecutiveRefusals++
                    consecutiveIdenticalActions = 0
                } else {
                    consecutiveBlockedTurns++
                    val isRepeated = isActionIdentical(
                        prediction.action, prediction.rawToolName,
                        lastAttemptedAction, lastAttemptedRawTool
                    )
                    if (isRepeated) {
                        consecutiveIdenticalActions++
                    } else {
                        consecutiveIdenticalActions = 1
                    }
                }
                consecutiveNoProgressTurns++

                lastAttemptedAction = prediction.action
                lastAttemptedRawTool = prediction.rawToolName
                previousTurnState = currentState

                // Early Termination Checks for Refusals / Blocked Actions
                if (consecutiveRefusals >= maxConsecutiveRefusals) {
                    val summary = "REPEATED_REFUSAL: Model refused or produced no action for $consecutiveRefusals consecutive turns"
                    Log.w(TAG, "Early termination at Turn $turn: $summary")
                    val record = AgentStepRecord(
                        turn = turn,
                        action = prediction.action ?: AgentAction.FailGoal("Repeated Refusal"),
                        actionDescription = prediction.rawToolName ?: "Refusal",
                        preScreenSummary = "${currentState.packageName} (${currentState.nodes.size} nodes)",
                        postScreenSummary = "Unchanged",
                        outcome = StepOutcome.REPEATED_REFUSAL,
                        reflection = summary,
                        confidence = prediction.confidence,
                        latencyMs = prediction.latencyMs
                    )
                    stepHistory.add(record)
                    onStepUpdate?.invoke(record)
                    return AgentRunResult(
                        success = false,
                        turnsExecuted = turn,
                        finalSummary = summary,
                        history = stepHistory,
                        finalUiState = currentState,
                        terminationReason = TerminationReason.REPEATED_REFUSAL
                    )
                }

                if (consecutiveIdenticalActions >= maxConsecutiveIdenticalActions) {
                    if (recoveryBackCount < maxRecoveryBacks) {
                        recoveryBackCount++
                        val actionDesc = prediction.rawToolName ?: (prediction.action?.let { describeAction(it, candidateMap) } ?: "Blocked Action")
                        Log.w(TAG, "[AUTONOMOUS RECOVERY] Repeated blocked action '$actionDesc' ($consecutiveIdenticalActions attempts). Attempting universal PressBack escape ($recoveryBackCount/$maxRecoveryBacks)...")
                        consecutiveIdenticalActions = 0
                        runtimeTaskState.currentPhase = TaskPhase.RECOVER

                        a11yService?.let { service ->
                            com.zipper.app.accessibility.KarvishControlOverlay.updateTurn(
                                context = service,
                                turn = turn,
                                maxTurns = maxTurns,
                                action = AgentAction.PressBack,
                                targetLabel = "Back (Escape sub-screen)",
                                confidence = 0.95f,
                                bounds = null
                            )
                        }
                        dispatchAction(AgentAction.PressBack, candidateMap, cleanGoal, actionCandidates)
                        delay(1000L)
                        val recoveryPostState = captureState()
                        latestUiState = recoveryPostState
                        stateHistory.add(recoveryPostState)
                        activeReflection = "Recovered from dead-end via PressBack. Choose an alternative valid tool."
                        turn++
                        continue
                    }
                    val actionDesc = prediction.rawToolName ?: (prediction.action?.let { describeAction(it, candidateMap) } ?: "Blocked Action")
                    val summary = "STUCK_NO_PROGRESS: Repeated identical action '$actionDesc' produced no state transition ($consecutiveIdenticalActions attempts)"
                    Log.w(TAG, "Early termination at Turn $turn: $summary")
                    val record = AgentStepRecord(
                        turn = turn,
                        action = prediction.action ?: AgentAction.FailGoal("Repeated Blocked Action"),
                        actionDescription = actionDesc,
                        preScreenSummary = "${currentState.packageName} (${currentState.nodes.size} nodes)",
                        postScreenSummary = "Unchanged",
                        outcome = StepOutcome.STUCK_NO_PROGRESS,
                        reflection = summary,
                        confidence = prediction.confidence,
                        latencyMs = prediction.latencyMs
                    )
                    stepHistory.add(record)
                    onStepUpdate?.invoke(record)
                    return AgentRunResult(
                        success = false,
                        turnsExecuted = turn,
                        finalSummary = summary,
                        history = stepHistory,
                        finalUiState = currentState,
                        terminationReason = TerminationReason.STUCK_NO_PROGRESS
                    )
                }

                if (consecutiveNoProgressTurns >= maxConsecutiveNoProgressTurns) {
                    val summary = "STUCK_NO_PROGRESS: Environment state remained unchanged for $consecutiveNoProgressTurns consecutive turns"
                    Log.w(TAG, "Early termination at Turn $turn: $summary")
                    val record = AgentStepRecord(
                        turn = turn,
                        action = prediction.action ?: AgentAction.FailGoal("Blocked Action"),
                        actionDescription = prediction.rawToolName ?: "Blocked Action",
                        preScreenSummary = "${currentState.packageName} (${currentState.nodes.size} nodes)",
                        postScreenSummary = "Unchanged",
                        outcome = StepOutcome.STUCK_NO_PROGRESS,
                        reflection = summary,
                        confidence = prediction.confidence,
                        latencyMs = prediction.latencyMs
                    )
                    stepHistory.add(record)
                    onStepUpdate?.invoke(record)
                    return AgentRunResult(
                        success = false,
                        turnsExecuted = turn,
                        finalSummary = summary,
                        history = stepHistory,
                        finalUiState = currentState,
                        terminationReason = TerminationReason.STUCK_NO_PROGRESS
                    )
                }

                val record = AgentStepRecord(
                    turn = turn,
                    action = prediction.action ?: AgentAction.FailGoal("Refusal"),
                    actionDescription = prediction.rawToolName ?: "Blocked Action",
                    preScreenSummary = "${currentState.packageName} (${currentState.nodes.size} nodes)",
                    postScreenSummary = "Unchanged",
                    outcome = StepOutcome.SAFETY_BLOCKED,
                    reflection = "Safety Gate: $gatingDecision | ${prediction.explanation}",
                    confidence = prediction.confidence,
                    latencyMs = prediction.latencyMs
                )
                stepHistory.add(record)
                onStepUpdate?.invoke(record)

                Log.w(TAG, """
                ==================== [TURN $turn AUDIT TRACE (BLOCKED)] ====================
                1. GOAL: '$cleanGoal'
                2. CURRENT APP: ${currentState.packageName}
                3. CURRENT ACTIVITY: ${currentState.activityName}
                4. ACCESSIBILITY NODE COUNT: ${currentState.nodes.size}
                5. TOOLS SENT TO NEEDLE: ${toolsJson.length()} tools ($toolSummary)
                6. PROMPT SENT TO NEEDLE: '$prompt'
                7. RAW NEEDLE OUTPUT: ${prediction.rawJson}
                8. PARSED ACTION: ${prediction.action?.javaClass?.simpleName ?: "None"} (${prediction.rawToolName})
                9. CONFIDENCE: ${"%.4f".format(prediction.confidence)}
                10. SAFETY GATE: $gatingDecision
                11. ACTUAL EXECUTION: Skipped
                12. POST-ACTION STATE: Unchanged
                13. VERIFICATION: BLOCKED
                14. NEXT TURN: PROCEEDING TO TURN ${turn + 1}
                ===========================================================================
                """.trimIndent())

                if (targetIdx != null) {
                    blacklistedIndices.add(targetIdx)
                    candidateMap[targetIdx]?.let { node ->
                        val label = listOfNotNull(node.text, node.contentDescription, node.resourceId)
                            .joinToString(" ")
                            .lowercase()
                            .replace(Regex("\\s+"), " ")
                            .trim()
                        val stableId = node.resourceId?.lowercase()?.takeIf { it.isNotBlank() }
                        blacklistedSemanticKeys.add("${node.role.lowercase()}|${stableId ?: label}")
                    }
                }

                activeReflection = "Safety Gate: $gatingDecision. Choose an alternative valid tool."
                turn++
                continue
            }

            // Terminal Action: CompleteGoal
            if (prediction.action is AgentAction.CompleteGoal) {
                val isSatisfied = isGoalAccomplished(
                    goal = cleanGoal,
                    action = prediction.action,
                    postState = currentState,
                    actionDescription = "CompleteGoal('${prediction.action.summary}')",
                    stepHistory = stepHistory
                )

                if (!isSatisfied) {
                    runtimeTaskState.lastOutcome = ActionOutcome.COMPLETE_REJECTED_MISSING_EVIDENCE
                    val rejectSummary = "COMPLETE_REJECTED: Goal '$cleanGoal' requires pending action(s) that have not been performed yet."
                    Log.w(TAG, "Turn $turn: $rejectSummary")
                    val record = AgentStepRecord(
                        turn = turn,
                        action = prediction.action,
                        actionDescription = "CompleteGoal('${prediction.action.summary}')",
                        preScreenSummary = "${currentState.packageName} (${currentState.nodes.size} nodes)",
                        postScreenSummary = "Unchanged",
                        outcome = StepOutcome.COMPLETE_REJECTED,
                        reflection = rejectSummary,
                        confidence = prediction.confidence,
                        latencyMs = prediction.latencyMs
                    )
                    stepHistory.add(record)
                    onStepUpdate?.invoke(record)
                    activeReflection = "Premature completion rejected. Goal '$cleanGoal' has remaining actions. Choose the next action."
                    turn++
                    continue
                }

                val record = AgentStepRecord(
                    turn = turn,
                    action = prediction.action,
                    actionDescription = "CompleteGoal('${prediction.action.summary}')",
                    preScreenSummary = "${currentState.packageName} (${currentState.nodes.size} nodes)",
                    postScreenSummary = "Goal Completed",
                    outcome = StepOutcome.GOAL_COMPLETED,
                    reflection = "Goal declared complete and verified: ${prediction.action.summary}",
                    confidence = prediction.confidence,
                    latencyMs = prediction.latencyMs
                )
                stepHistory.add(record)
                onStepUpdate?.invoke(record)
                Log.i(TAG, "🎉 GOAL COMPLETED at Turn $turn: ${prediction.action.summary}")
                return AgentRunResult(
                    success = true,
                    turnsExecuted = turn,
                    finalSummary = "Goal completed: ${prediction.action.summary}",
                    history = stepHistory,
                    finalUiState = currentState,
                    terminationReason = TerminationReason.GOAL_COMPLETED
                )
            }

            // Terminal Action: FailGoal
            if (prediction.action is AgentAction.FailGoal) {
                val record = AgentStepRecord(
                    turn = turn,
                    action = prediction.action,
                    actionDescription = "FailGoal('${prediction.action.reason}')",
                    preScreenSummary = "${currentState.packageName} (${currentState.nodes.size} nodes)",
                    postScreenSummary = "Goal Failed",
                    outcome = StepOutcome.GOAL_FAILED,
                    reflection = "Goal declared failed by action model: ${prediction.action.reason}",
                    confidence = prediction.confidence,
                    latencyMs = prediction.latencyMs
                )
                stepHistory.add(record)
                onStepUpdate?.invoke(record)
                Log.w(TAG, "Goal failed at Turn $turn: ${prediction.action.reason}")
                return AgentRunResult(
                    success = false,
                    turnsExecuted = turn,
                    finalSummary = "Goal declared unreachable: ${prediction.action.reason}",
                    history = stepHistory,
                    finalUiState = currentState,
                    terminationReason = TerminationReason.GOAL_FAILED
                )
            }

            // 6. ACT: Execute Primitive
            val action = prediction.action!!
            val isRepeatedAction = isActionIdentical(
                action, prediction.rawToolName,
                lastAttemptedAction, lastAttemptedRawTool
            )
            if (isRepeatedAction) {
                consecutiveIdenticalActions++
            } else {
                consecutiveIdenticalActions = 1
            }
            lastAttemptedAction = action
            lastAttemptedRawTool = prediction.rawToolName

            val actionDesc = describeAction(action, candidateMap)
            a11yService?.let { service ->
                val targetScreenBounds = targetIdx?.let { idx ->
                    val candidate = candidateMap[idx]
                    val parentBounds = candidate?.parentIndex?.let { pIdx ->
                        val parent = candidateMap[pIdx]
                        val pBounds = parent?.rawBounds
                        val cBounds = candidate.rawBounds
                        if (parent != null && parent.isClickable && pBounds != null && cBounds != null &&
                            pBounds.left <= cBounds.left && pBounds.top <= cBounds.top &&
                            pBounds.right >= cBounds.right && pBounds.bottom >= cBounds.bottom) {
                            pBounds
                        } else null
                    }
                    parentBounds ?: candidate?.rawBounds ?: run {
                        val liveNode = service.activeNodeMap[idx]
                        if (liveNode != null) {
                            val r = android.graphics.Rect()
                            liveNode.getBoundsInScreen(r)
                            r
                        } else null
                    }
                }
                val chosenTargetName = targetIdx?.let { idx ->
                    candidateMap[idx]?.text ?: candidateMap[idx]?.contentDescription
                } ?: actionDesc

                com.zipper.app.accessibility.KarvishControlOverlay.updateTurn(
                    context = service,
                    turn = turn,
                    maxTurns = maxTurns,
                    action = action,
                    targetLabel = chosenTargetName,
                    confidence = prediction.confidence,
                    bounds = targetScreenBounds,
                    isVisionActive = prediction.isVisionActive,
                    visionIconClass = prediction.visionIconClass
                )
                com.zipper.app.core.graphics.LiveOverlayManager.onStepUpdated(
                    turn = turn,
                    actionText = actionDesc,
                    coreState = com.zipper.app.core.graphics.ZipperCoreState.WORKING
                )
            }

            // Human-in-the-Loop Pre-Send Safety Gate:
            // Intercept any message send action (tapping Send button or submitting composer in messaging apps)
            // and prompt the user to confirm sending or abort with draft intact.
            val isMessagingApp = currentState.packageName in MESSAGING_PACKAGES
            val isSendTap = if (action is AgentAction.Tap) {
                val tNode = candidateMap[action.targetNodeIndex]
                val resId = tNode?.resourceId?.lowercase().orEmpty()
                val desc = listOfNotNull(tNode?.text, tNode?.contentDescription, tNode?.semanticLabel, action.semanticLabel).joinToString(" ").lowercase()
                val candId = action.candidateId?.lowercase().orEmpty()

                // Never treat AI bot buttons as send buttons
                val isAiDistractor = resId.contains("meta_ai") || resId.contains("ai_send") || resId.contains("assistant") ||
                    resId.contains("ai_input_send") || desc.contains("meta ai") || desc.contains("ask ai")
                if (isAiDistractor) {
                    false
                } else {
                    val isSendAffordance = candId.contains("send") || resId.contains("send") || desc == "send" || desc.startsWith("send ") || desc.contains("send message") || action.semanticLabel?.lowercase() == "send"
                    val isBottomRightComposerSend = isMessagingApp &&
                        (tNode?.bounds?.bottom ?: 0) > 1300 && (tNode?.bounds?.right ?: 0) > 700 &&
                        (resId.contains("button") || resId.contains("send") || tNode?.role in setOf("BUTTON", "IMAGE", "VIEW"))
                    isSendAffordance || isBottomRightComposerSend || (runtimeTaskState.currentPhase == TaskPhase.COMMIT && (resId.contains("send") || desc.contains("send")))
                }
            } else false

            val hasRecentTextEntry = stepHistory.any { it.action is AgentAction.TypeText }
            val isSendSubmit = (action is AgentAction.Submit && runtimeTaskState.currentPhase == TaskPhase.COMMIT && hasRecentTextEntry && isMessagingApp) ||
                (isMessagingApp && runtimeTaskState.currentPhase == TaskPhase.COMMIT && hasRecentTextEntry && isSendTap)

            if (isMessagingApp && (isSendTap || isSendSubmit) && !hasUserConfirmedSend) {
                val recipient = GoalArgumentResolver.resolveTargetEntity(cleanGoal)?.ifBlank { null }
                    ?: currentState.nodes.firstOrNull { it.resourceId?.contains("conversation_contact_name") == true || it.resourceId?.contains("action_bar_title") == true }?.text
                    ?: "recipient"
                val messageText = stepHistory.lastOrNull { it.action is AgentAction.TypeText }?.let { (it.action as AgentAction.TypeText).text }
                    ?: GoalArgumentResolver.resolveMessageBody(cleanGoal)
                    ?: "message"

                Log.i(TAG, "[PRE_SEND_GATE] Intercepted send action. Prompting user for confirmation before sending to '$recipient'.")
                isPaused = true
                var sendApproved = false
                com.zipper.app.accessibility.KarvishControlOverlay.showConfirmationCard(
                    title = "Confirm Send",
                    prompt = "Send \"$messageText\" to $recipient?",
                    confirmText = "Send",
                    cancelText = "Don't Send",
                    onConfirm = {
                        sendApproved = true
                        hasUserConfirmedSend = true
                        isPaused = false
                        Log.i(TAG, "[PRE_SEND_GATE] User approved message send.")
                    },
                    onCancel = {
                        isCancelled = true
                        isPaused = false
                        Log.i(TAG, "[PRE_SEND_GATE] User cancelled message send.")
                    }
                )

                while (isPaused && !isCancelled) {
                    delay(200)
                }

                if (isCancelled || !sendApproved) {
                    Log.i(TAG, "[PRE_SEND_GATE] Send aborted by user. Preserving composer draft.")
                    return AgentRunResult(
                        success = false,
                        turnsExecuted = turn,
                        finalSummary = "Message sending cancelled by user. Draft preserved.",
                        history = stepHistory,
                        finalUiState = currentState,
                        terminationReason = TerminationReason.GOAL_FAILED
                    )
                }
            }

            // Autonomous Approval Gate (Grok Bot Safety Architecture)
            // Intercept irreversible financial payments and destructive commitments
            val targetNodeForApproval = targetIdx?.let { candidateMap[it] }
            val approvalDecision = AutonomousApprovalGate.evaluate(
                targetNode = targetNodeForApproval,
                action = action,
                packageName = currentState.packageName
            )
            if (approvalDecision.isApprovalRequired && !hasUserApprovedSensitiveAction && !autoApproveGate) {
                Log.w(TAG, "[APPROVAL GATE] Triggered: ${approvalDecision.title} - ${approvalDecision.targetDetail}")
                var userApprovedAction = false
                isPaused = true
                com.zipper.app.accessibility.KarvishControlOverlay.showApprovalGateCard(
                    title = approvalDecision.title,
                    prompt = approvalDecision.prompt,
                    targetDetail = approvalDecision.targetDetail,
                    onApprove = {
                        userApprovedAction = true
                        hasUserApprovedSensitiveAction = true
                        isPaused = false
                        Log.i(TAG, "[APPROVAL GATE] User APPROVED sensitive action: ${approvalDecision.targetDetail}")
                    },
                    onAbort = {
                        isCancelled = true
                        isPaused = false
                        Log.i(TAG, "[APPROVAL GATE] User ABORTED sensitive action: ${approvalDecision.targetDetail}")
                    }
                )

                while (isPaused && !isCancelled) {
                    delay(200)
                }

                if (isCancelled || !userApprovedAction) {
                    Log.i(TAG, "[APPROVAL GATE] Action halted safely by user authorization rejection.")
                    com.zipper.app.accessibility.KarvishControlOverlay.hide()
                    return AgentRunResult(
                        success = false,
                        turnsExecuted = turn,
                        finalSummary = "Halted by user at Approval Gate: ${approvalDecision.targetDetail}",
                        history = stepHistory,
                        finalUiState = currentState,
                        terminationReason = TerminationReason.GOAL_FAILED
                    )
                }
            }

            val executed = dispatchAction(action, candidateMap, cleanGoal, actionCandidates)

            val waitDelay = when (action) {
                is AgentAction.OpenApp -> 1200L
                is AgentAction.Submit -> 1000L
                is AgentAction.Tap -> {
                    if (action.candidateId?.contains("suggestion") == true || action.semanticLabel?.startsWith("Search") == true) {
                        900L
                    } else if (explorationEngine != null) {
                        300L
                    } else {
                        350L
                    }
                }
                is AgentAction.Scroll -> if (explorationEngine != null) 350L else 450L
                is AgentAction.TypeText -> 300L
                else -> stepDelayMs
            }
            if (waitDelay > 0L) {
                delay(waitDelay)
            }

            // 7. VERIFY: Capture Post-State and Compute Differential
            val postState = captureState()
            latestUiState = postState
            val delta = differentialEngine.computeDelta(currentState, postState, action)

            // MuSpark Post-Action State Delta Trace
            val postTrace = "Delta: ${delta.outcome} | Nodes: ${currentState.nodes.size} -> ${postState.nodes.size} | Phase: ${runtimeTaskState.currentPhase}"
            com.zipper.app.accessibility.KarvishControlOverlay.updateThinkingTrace(postTrace)

            val currentRecord = AgentStepRecord(
                turn = turn,
                action = action,
                actionDescription = actionDesc,
                preScreenSummary = "${currentState.packageName} (${currentState.nodes.size} nodes)",
                postScreenSummary = "${postState.packageName} (${postState.nodes.size} nodes)",
                outcome = delta.outcome,
                reflection = delta.summary,
                confidence = prediction.confidence,
                latencyMs = prediction.latencyMs,
                packageName = postState.packageName ?: currentState.packageName
            )
            val historyWithCurrent = stepHistory + currentRecord
            val stateVerification = StateDeltaVerifier.verifyTransition(cleanGoal, currentState, postState, action)
            val isNegativeError = stateVerification.status == StateDeltaVerifier.VerificationStatus.REJECTED_ERROR

            val isGoalSatisfied = !isNegativeError && ((explorationEngine?.isWinnerActionDispatched == true) ||
                isGoalAccomplished(cleanGoal, action, postState, actionDesc, historyWithCurrent, currentState))
            val isStateChanged = !isStateEffectivelyIdentical(currentState, postState)

            if (isNegativeError) {
                Log.w(TAG, "[STATE VERIFIER] Action rejected by environment: ${stateVerification.summary}")
                activeReflection = "Environment error: ${stateVerification.rawErrorMessage}"
                runtimeTaskState.lastOutcome = ActionOutcome.ENVIRONMENT_ERROR_REJECTED
                runtimeTaskState.currentPhase = TaskPhase.RECOVER
            }

            // Senior Step 2 & 3: Verified Control Loop Outcome Tracking & Phase Advancement
            val isActionMeaningful = executed && (isStateChanged || action is AgentAction.TypeText || action is AgentAction.Submit)
            if (!isActionMeaningful && (delta.outcome == StepOutcome.NO_OP || !isStateChanged)) {
                runtimeTaskState.lastOutcome = ActionOutcome.UNCHANGED_NOOP
                runtimeTaskState.consecutiveNoOpCount++
            } else if (!isNegativeError) {
                runtimeTaskState.lastOutcome = ActionOutcome.SUCCESS_TRANSITION
                runtimeTaskState.consecutiveNoOpCount = 0

                if (action is AgentAction.Submit && (comparativeIntent != null || cleanGoal.contains("search", ignoreCase = true) || cleanGoal.contains("find", ignoreCase = true))) {
                    runtimeTaskState.hasExecutedSearch = true
                    runtimeTaskState.currentPhase = TaskPhase.SELECT_TARGET
                    Log.i(TAG, "[TASK PHASE] Search submitted via Submit action -> Advanced to SELECT_TARGET")
                }

                when (runtimeTaskState.currentPhase) {
                    TaskPhase.REACH_CONTEXT -> {
                        val hasReachedTargetApp = if (resolvedTargetApp != null) {
                            isAppMatchingTarget(postState.packageName, resolvedTargetApp)
                        } else {
                            !isLauncherOrSystemPackage(postState.packageName)
                        }
                        if (hasReachedTargetApp) {
                            runtimeTaskState.currentPhase = TaskPhase.FIND_TARGET
                            Log.i(TAG, "[TASK PHASE] Advanced to FIND_TARGET in ${postState.packageName}")
                        }
                    }
                    TaskPhase.FIND_TARGET -> {
                        if (action is AgentAction.TypeText && action.text.isNotBlank()) {
                            runtimeTaskState.currentPhase = TaskPhase.COMMIT
                            Log.i(TAG, "[TASK PHASE] Text entered during FIND_TARGET -> Advanced to COMMIT to submit")
                        } else {
                            val hasActiveInput = postState.nodes.any { it.isFocused && (it.isEditable || it.role.contains("EDIT", true)) }
                            if (hasActiveInput) {
                                runtimeTaskState.currentPhase = TaskPhase.ENTER_ARGUMENT
                                Log.i(TAG, "[TASK PHASE] Active input focused -> Advanced to ENTER_ARGUMENT")
                            } else if (action is AgentAction.Tap) {
                            val tappedNode = currentState.nodes.find { it.index == (action as? AgentAction.Tap)?.targetNodeIndex }
                            val tappedLabel = "${tappedNode?.text.orEmpty()} ${tappedNode?.contentDescription.orEmpty()} ${tappedNode?.resourceId.orEmpty()}".lowercase()
                            val isSearchAffordance = tappedLabel.contains("search") || tappedLabel.contains("find")

                            val postHasSearchInput = postState.nodes.any { n ->
                                val res = n.resourceId?.lowercase().orEmpty()
                                val text = n.text?.lowercase().orEmpty()
                                val desc = n.contentDescription?.lowercase().orEmpty()
                                (res.contains("search") || text.contains("search") || desc.contains("search")) &&
                                (n.isEditable || n.isFocused)
                            }

                            if (isSearchAffordance || postHasSearchInput) {
                                runtimeTaskState.currentPhase = TaskPhase.ENTER_ARGUMENT
                                Log.i(TAG, "[TASK PHASE] Search affordance tapped -> Advanced to ENTER_ARGUMENT")
                            } else {
                                val targetEntity = GoalArgumentResolver.resolveTargetEntity(cleanGoal)?.lowercase().orEmpty()
                                val hasExactMatch = if (targetEntity.isNotBlank()) {
                                    postState.nodes.any { n ->
                                        val candText = (n.text?.takeIf { it.isNotBlank() } ?: n.contentDescription).orEmpty()
                                        candText.trim().equals(targetEntity, ignoreCase = true)
                                    }
                                } else false

                                val hasUnusedSearch = postState.nodes.any { n ->
                                    val res = n.resourceId?.lowercase().orEmpty()
                                    val text = n.text?.lowercase().orEmpty()
                                    val desc = n.contentDescription?.lowercase().orEmpty()
                                    (res.contains("search") || text.contains("search") || desc.contains("search")) &&
                                    (n.isEditable || n.isClickable)
                                }

                                if (!runtimeTaskState.hasExecutedSearch && hasUnusedSearch && !hasExactMatch) {
                                    runtimeTaskState.currentPhase = TaskPhase.FIND_TARGET
                                    Log.i(TAG, "[TASK PHASE] Un-executed search bar available and no exact target match -> Remaining in FIND_TARGET to search")
                                } else {
                                    runtimeTaskState.currentPhase = TaskPhase.SELECT_TARGET
                                    Log.i(TAG, "[TASK PHASE] Advanced to SELECT_TARGET")
                                }
                            }
                        }
                    }
                    }
                    TaskPhase.SELECT_TARGET -> {
                        val isMessagingApp = currentState.packageName in MESSAGING_PACKAGES
                        if (action is AgentAction.TypeText && action.text.isNotBlank()) {
                            if (isMessagingApp) {
                                runtimeTaskState.hasExecutedSearch = true
                                runtimeTaskState.currentPhase = TaskPhase.SELECT_TARGET
                                Log.i(TAG, "[TASK PHASE] Contact search entered in messaging app -> Remaining in SELECT_TARGET to pick contact")
                            } else {
                                runtimeTaskState.currentPhase = TaskPhase.COMMIT
                                Log.i(TAG, "[TASK PHASE] Text entered during SELECT_TARGET -> Advanced directly to COMMIT")
                            }
                        } else {
                            val hasComposerInput = postState.nodes.any { n ->
                                (n.isEditable || n.role.contains("EDIT", true)) &&
                                !n.resourceId.orEmpty().contains("search", true) &&
                                !n.text.orEmpty().contains("search", true) &&
                                !n.contentDescription.orEmpty().contains("search", true)
                            }
                            if (hasComposerInput) {
                                runtimeTaskState.currentPhase = TaskPhase.ENTER_ARGUMENT
                                Log.i(TAG, "[TASK PHASE] Composer detected -> Advanced to ENTER_ARGUMENT")
                            }
                        }
                    }
                    TaskPhase.ENTER_ARGUMENT -> {
                        if (action is AgentAction.TypeText && action.text.isNotBlank()) {
                            val isMessagingApp = currentState.packageName in MESSAGING_PACKAGES
                            val targetEntity = GoalArgumentResolver.resolveTargetEntity(cleanGoal)?.lowercase().orEmpty()
                            val typedTargetNode = currentState.nodes.find { it.index == action.targetNodeIndex }
                            val typedLabel = "${action.semanticLabel.orEmpty()} ${typedTargetNode?.resourceId.orEmpty()} ${typedTargetNode?.text.orEmpty()}".lowercase()
                            val isSearchInput = typedLabel.contains("search") || typedLabel.contains("find") || typedLabel.contains("query") ||
                                (targetEntity.isNotBlank() && action.text.trim().equals(targetEntity, ignoreCase = true))
                            if (isMessagingApp && isSearchInput) {
                                runtimeTaskState.hasExecutedSearch = true
                                runtimeTaskState.currentPhase = TaskPhase.SELECT_TARGET
                                Log.i(TAG, "[TASK PHASE] Contact search query entered in messaging app -> Advanced to SELECT_TARGET to pick contact")
                            } else if (isSearchInput) {
                                runtimeTaskState.currentPhase = TaskPhase.COMMIT
                                Log.i(TAG, "[TASK PHASE] Search query entered -> Advanced to COMMIT to submit search")
                            } else {
                                runtimeTaskState.currentPhase = TaskPhase.COMMIT
                                Log.i(TAG, "[TASK PHASE] Argument entered -> Advanced to COMMIT")
                            }
                        }
                    }
                    TaskPhase.COMMIT -> {
                        val isTypeTextAction = action is AgentAction.TypeText
                        val targetIdx = when (action) {
                            is AgentAction.Tap -> action.targetNodeIndex
                            is AgentAction.TypeText -> action.targetNodeIndex
                            is AgentAction.Scroll -> action.targetNodeIndex
                            else -> null
                        }
                        val targetNode = targetIdx?.let { idx -> currentState.nodes.find { it.index == idx } }
                        val nodeRes = targetNode?.resourceId?.lowercase().orEmpty()
                        val semanticLbl = when (action) {
                            is AgentAction.Tap -> action.semanticLabel
                            is AgentAction.TypeText -> action.semanticLabel
                            else -> null
                        }
                        val nodeText = (targetNode?.text ?: semanticLbl).orEmpty().lowercase()
                        val isExactSuggestionTap = action is AgentAction.Tap && (action.candidateId?.contains("exact_suggestion") == true || action.semanticLabel?.contains("Search '", ignoreCase = true) == true)
                        val isSendOrSubmit = nodeRes.contains("send") || nodeRes.contains("submit") ||
                            nodeRes.contains("search") || nodeText.contains("search") ||
                            nodeText.contains("send") || action is AgentAction.Submit || isExactSuggestionTap

                        if (isTypeTextAction) {
                            Log.i(TAG, "[TASK PHASE] Text entered during COMMIT -> Remaining in COMMIT to click Send")
                        } else if (isSendOrSubmit) {
                            if (comparativeIntent != null || cleanGoal.contains("search", ignoreCase = true) || cleanGoal.contains("find", ignoreCase = true)) {
                                runtimeTaskState.hasExecutedSearch = true
                                runtimeTaskState.currentPhase = TaskPhase.SELECT_TARGET
                                Log.i(TAG, "[TASK PHASE] Search submitted -> Advanced to SELECT_TARGET")
                            } else if (isStateChanged) {
                                runtimeTaskState.currentPhase = TaskPhase.VERIFY_OUTCOME
                                Log.i(TAG, "[TASK PHASE] Send/Commit executed successfully -> Advanced to VERIFY_OUTCOME")
                            }
                        }
                    }
                    TaskPhase.RECOVER -> {
                        runtimeTaskState.currentPhase = TaskPhase.FIND_TARGET
                        Log.i(TAG, "[TASK PHASE] Recovered back to FIND_TARGET")
                    }
                    else -> {}
                }
            }

            // Multi-step oscillation / ping-pong cycle detection:
            // Check if postState is effectively identical to any prior state in recent history
            // (excluding currentState which is stateHistory.last())
            val recentPriorStates = stateHistory.takeLast(6).dropLast(1)
            val isCycle = recentPriorStates.any { isStateEffectivelyIdentical(postState, it) }

            if (isCycle) {
                consecutiveCycleTurns++
                Log.w(TAG, "Multi-step cycle detected at Turn $turn (consecutive cycle turns: $consecutiveCycleTurns)")
            } else if (isStateChanged) {
                consecutiveCycleTurns = 0
            }

            stateHistory.add(postState)
            if (stateHistory.size > 15) {
                stateHistory.removeAt(0)
            }

            val isExploringCatalog = explorationEngine != null && !explorationEngine.isWinnerActionDispatched
            val madeMeaningfulProgress = isExploringCatalog || (!isCycle && (isStateChanged ||
                (action is AgentAction.TypeText && executed && delta.outcome == StepOutcome.SUCCESS) ||
                (action is AgentAction.Scroll && executed && delta.outcome == StepOutcome.SUCCESS)))

            if (madeMeaningfulProgress || isGoalSatisfied) {
                consecutiveNoProgressTurns = 0
                consecutiveRefusals = 0
                consecutiveBlockedTurns = 0
            } else {
                consecutiveNoProgressTurns++
            }

            previousTurnState = postState
            val outcome = if (isGoalSatisfied) StepOutcome.GOAL_COMPLETED else delta.outcome

            // 8. Log Full 14-Point Diagnostic Trace
            Log.i(TAG, """
            ==================== [TURN $turn AUDIT TRACE] ====================
            1. GOAL: '$cleanGoal'
            2. CURRENT APP: ${currentState.packageName}
            3. CURRENT ACTIVITY: ${currentState.activityName}
            4. ACCESSIBILITY NODE COUNT: ${currentState.nodes.size}
            5. TOOLS SENT TO NEEDLE: ${toolsJson.length()} tools ($toolSummary)
            6. PROMPT SENT TO NEEDLE: '$prompt'
            7. RAW NEEDLE OUTPUT: ${prediction.rawJson}
            8. PARSED ACTION: ${prediction.action.javaClass.simpleName} ($actionDesc)
            9. CONFIDENCE: ${"%.4f".format(prediction.confidence)}
            10. SAFETY GATE: $gatingDecision
            11. ACTUAL EXECUTION: $executed
            12. POST-ACTION STATE: ${postState.packageName} (${postState.nodes.size} nodes)
            13. VERIFICATION: $outcome -> ${delta.summary}
            14. NEXT TURN: ${if (isGoalSatisfied) "TERMINATING (GOAL_SUCCESS)" else "PROCEEDING TO TURN ${turn + 1}"}
            ==================================================================
            """.trimIndent())

            // 9. REFLECT: If NO-OP, cycle, or failure outcome, blacklist target element
            val isFailureOutcome = outcome == StepOutcome.NO_OP ||
                outcome == StepOutcome.ERROR_DETECTED ||
                outcome == StepOutcome.WRONG_SCREEN ||
                outcome == StepOutcome.BLOCKED_BY_POPUP ||
                outcome == StepOutcome.COMPLETE_REJECTED

            if (isCycle) {
                if (targetIdx != null) {
                    blacklistedIndices.add(targetIdx)
                    candidateMap[targetIdx]?.let { node ->
                        val label = listOfNotNull(node.text, node.contentDescription, node.resourceId)
                            .joinToString(" ")
                            .lowercase()
                            .replace(Regex("\\s+"), " ")
                            .trim()
                        val stableId = node.resourceId?.lowercase()?.takeIf { it.isNotBlank() }
                        blacklistedSemanticKeys.add("${node.role.lowercase()}|${stableId ?: label}")
                    }
                }
                activeReflection = "Detected oscillation loop back to previous screen (${postState.packageName}). Choose a different, progressive action."
            } else if (isFailureOutcome) {
                if (targetIdx != null) {
                    blacklistedIndices.add(targetIdx)
                    candidateMap[targetIdx]?.let { node ->
                        val label = listOfNotNull(node.text, node.contentDescription, node.resourceId)
                            .joinToString(" ")
                            .lowercase()
                            .replace(Regex("\\s+"), " ")
                            .trim()
                        val stableId = node.resourceId?.lowercase()?.takeIf { it.isNotBlank() }
                        blacklistedSemanticKeys.add("${node.role.lowercase()}|${stableId ?: label}")
                    }
                }
                activeReflection = "Action on element #${targetIdx ?: 0} resulted in $outcome: ${delta.summary}"
            } else {
                activeReflection = null
            }

            if (action is AgentAction.Tap && (!isStateChanged || consecutiveIdenticalActions >= maxConsecutiveIdenticalActions)) {
                val repeatedNode = candidateMap[action.targetNodeIndex]
                if (!isRepeatableInputTarget(repeatedNode)) {
                    blacklistedIndices.add(action.targetNodeIndex)
                    val label = listOfNotNull(repeatedNode?.text, repeatedNode?.contentDescription, repeatedNode?.resourceId)
                        .joinToString(" ")
                        .lowercase()
                        .replace(Regex("\\s+"), " ")
                        .trim()
                    val stableId = repeatedNode?.resourceId?.lowercase()?.takeIf { it.isNotBlank() }
                    blacklistedSemanticKeys.add("${repeatedNode?.role?.lowercase() ?: "element"}|${stableId ?: label}")
                    activeReflection = "Target '${repeatedNode?.semanticLabel ?: action.targetNodeIndex}' produced no UI transition; choose an alternative progressive action."
                    Log.w(TAG, "Blacklisted non-transition tap target #${action.targetNodeIndex} ('$label')")
                }
            }

            // Check goal satisfaction first
            if (isGoalSatisfied) {
                val record = AgentStepRecord(
                    turn = turn,
                    action = action,
                    actionDescription = actionDesc,
                    preScreenSummary = "${currentState.packageName} (${currentState.nodes.size} nodes)",
                    postScreenSummary = "${postState.packageName} (${postState.nodes.size} nodes)",
                    outcome = outcome,
                    reflection = "Goal successfully achieved on device.",
                    confidence = prediction.confidence,
                    latencyMs = prediction.latencyMs
                )
                stepHistory.add(record)
                onStepUpdate?.invoke(record)
                Log.i(TAG, "🎉 AUTONOMOUS GOAL ACHIEVED: '$goal'")
                val completedSummary = if (explorationEngine?.winner != null) {
                    val w = explorationEngine.winner!!
                    "Compared ${explorationEngine.totalAccumulatedCount} items on ${postState.packageName ?: "app"}: ${w.title} (${w.quantityText}) at ₹${w.price} is ${explorationEngine.intent.mode.name.lowercase(java.util.Locale.ROOT)} (₹${"%.2f".format(w.unitPrice)}/unit). Added to cart."
                } else {
                    "Goal completed: '$goal' achieved on device in $turn turn(s)"
                }
                return AgentRunResult(
                    success = true,
                    turnsExecuted = turn,
                    finalSummary = completedSummary,
                    history = stepHistory,
                    finalUiState = postState,
                    terminationReason = TerminationReason.GOAL_COMPLETED
                )
            }

            // Early termination check: Multi-step ping-pong cycle detected
            if (consecutiveCycleTurns >= 3) {
                val summary = "CYCLE_DETECTED: Agent trapped in multi-step oscillation loop across visited screens ($consecutiveCycleTurns consecutive cyclic transitions)"
                Log.w(TAG, "Early termination at Turn $turn: $summary")
                val record = AgentStepRecord(
                    turn = turn,
                    action = action,
                    actionDescription = actionDesc,
                    preScreenSummary = "${currentState.packageName} (${currentState.nodes.size} nodes)",
                    postScreenSummary = "${postState.packageName} (${postState.nodes.size} nodes)",
                    outcome = StepOutcome.STUCK_NO_PROGRESS,
                    reflection = summary,
                    confidence = prediction.confidence,
                    latencyMs = prediction.latencyMs
                )
                stepHistory.add(record)
                onStepUpdate?.invoke(record)
                return AgentRunResult(
                    success = false,
                    turnsExecuted = turn,
                    finalSummary = summary,
                    history = stepHistory,
                    finalUiState = postState,
                    terminationReason = TerminationReason.CYCLE_DETECTED
                )
            }

            // Early termination check: Repeated identical action producing no state transition
            val repeatedBackRecovery = action is AgentAction.PressBack &&
                consecutiveIdenticalActions >= maxConsecutiveIdenticalActions
            if (!isExploringCatalog && ((!isStateChanged && consecutiveIdenticalActions >= maxConsecutiveIdenticalActions) || repeatedBackRecovery)) {
                if (action !is AgentAction.PressBack && recoveryBackCount < maxRecoveryBacks) {
                    recoveryBackCount++
                    Log.w(TAG, "[AUTONOMOUS RECOVERY] Sub-screen or dead-end detected ($consecutiveIdenticalActions unprogressed attempts on '$actionDesc'). Attempting universal PressBack escape ($recoveryBackCount/$maxRecoveryBacks)...")
                    consecutiveIdenticalActions = 0
                    runtimeTaskState.currentPhase = TaskPhase.RECOVER

                    a11yService?.let { service ->
                        com.zipper.app.accessibility.KarvishControlOverlay.updateTurn(
                            context = service,
                            turn = turn,
                            maxTurns = maxTurns,
                            action = AgentAction.PressBack,
                            targetLabel = "Back (Escape sub-screen)",
                            confidence = 0.95f,
                            bounds = null
                        )
                    }
                    dispatchAction(AgentAction.PressBack, candidateMap, cleanGoal, actionCandidates)
                    delay(1000L)
                    val recoveryPostState = captureState()
                    latestUiState = recoveryPostState
                    stateHistory.add(recoveryPostState)
                    activeReflection = "Recovered from dead-end via PressBack. Observe new screen and continue towards goal."
                    turn++
                    continue
                }

                val summary = "STUCK_NO_PROGRESS: Repeated identical action '$actionDesc' produced no state transition ($consecutiveIdenticalActions attempts)"
                Log.w(TAG, "Early termination at Turn $turn: $summary")
                val record = AgentStepRecord(
                    turn = turn,
                    action = action,
                    actionDescription = actionDesc,
                    preScreenSummary = "${currentState.packageName} (${currentState.nodes.size} nodes)",
                    postScreenSummary = "${postState.packageName} (${postState.nodes.size} nodes)",
                    outcome = StepOutcome.STUCK_NO_PROGRESS,
                    reflection = summary,
                    confidence = prediction.confidence,
                    latencyMs = prediction.latencyMs
                )
                stepHistory.add(record)
                onStepUpdate?.invoke(record)
                return AgentRunResult(
                    success = false,
                    turnsExecuted = turn,
                    finalSummary = summary,
                    history = stepHistory,
                    finalUiState = postState,
                    terminationReason = TerminationReason.STUCK_NO_PROGRESS
                )
            }

            // Early termination check: Environment state remained unchanged across consecutive turns
            if (!isExploringCatalog && !isStateChanged && consecutiveNoProgressTurns >= maxConsecutiveNoProgressTurns) {
                if (action !is AgentAction.PressBack && recoveryBackCount < maxRecoveryBacks) {
                    recoveryBackCount++
                    Log.w(TAG, "[AUTONOMOUS RECOVERY] Environment unchanged for $consecutiveNoProgressTurns turns. Attempting universal PressBack escape ($recoveryBackCount/$maxRecoveryBacks)...")
                    consecutiveIdenticalActions = 0
                    runtimeTaskState.currentPhase = TaskPhase.RECOVER

                    a11yService?.let { service ->
                        com.zipper.app.accessibility.KarvishControlOverlay.updateTurn(
                            context = service,
                            turn = turn,
                            maxTurns = maxTurns,
                            action = AgentAction.PressBack,
                            targetLabel = "Back (Escape sub-screen)",
                            confidence = 0.95f,
                            bounds = null
                        )
                    }
                    dispatchAction(AgentAction.PressBack, candidateMap, cleanGoal, actionCandidates)
                    delay(1000L)
                    val recoveryPostState = captureState()
                    latestUiState = recoveryPostState
                    stateHistory.add(recoveryPostState)
                    activeReflection = "Recovered from dead-end via PressBack. Observe new screen and continue towards goal."
                    turn++
                    continue
                }

                val summary = "STUCK_NO_PROGRESS: Environment state remained unchanged for $consecutiveNoProgressTurns consecutive turns"
                Log.w(TAG, "Early termination at Turn $turn: $summary")
                val record = AgentStepRecord(
                    turn = turn,
                    action = action,
                    actionDescription = actionDesc,
                    preScreenSummary = "${currentState.packageName} (${currentState.nodes.size} nodes)",
                    postScreenSummary = "${postState.packageName} (${postState.nodes.size} nodes)",
                    outcome = StepOutcome.STUCK_NO_PROGRESS,
                    reflection = summary,
                    confidence = prediction.confidence,
                    latencyMs = prediction.latencyMs
                )
                stepHistory.add(record)
                onStepUpdate?.invoke(record)
                return AgentRunResult(
                    success = false,
                    turnsExecuted = turn,
                    finalSummary = summary,
                    history = stepHistory,
                    finalUiState = postState,
                    terminationReason = TerminationReason.STUCK_NO_PROGRESS
                )
            }

            val record = AgentStepRecord(
                turn = turn,
                action = action,
                actionDescription = actionDesc,
                preScreenSummary = "${currentState.packageName} (${currentState.nodes.size} nodes)",
                postScreenSummary = "${postState.packageName} (${postState.nodes.size} nodes)",
                outcome = outcome,
                reflection = delta.formatFactualSummary(),
                confidence = prediction.confidence,
                latencyMs = prediction.latencyMs,
                stateDelta = delta
            )
            stepHistory.add(record)
            onStepUpdate?.invoke(record)

            turn++
        }

        Log.w(TAG, "Agent turn budget exhausted ($maxTurns turns)")
        return AgentRunResult(
            success = false,
            turnsExecuted = maxTurns,
            finalSummary = "Turn budget exhausted ($maxTurns turns) before goal completion",
            history = stepHistory,
            finalUiState = latestUiState,
            terminationReason = TerminationReason.TURN_BUDGET_EXHAUSTED
        )
    }

    private fun describeAction(
        action: AgentAction,
        candidateMap: Map<Int, DistilledNode>
    ): String {
        return when (action) {
            is AgentAction.OpenApp -> "OpenApp('${action.appName}')"
            is AgentAction.Tap -> {
                val node = candidateMap[action.targetNodeIndex]
                val label = node?.let { it.text ?: it.contentDescription ?: it.semanticLabel } ?: "unknown"
                "Tap [#${action.targetNodeIndex}] '$label'"
            }
            is AgentAction.TypeText -> {
                val node = candidateMap[action.targetNodeIndex]
                val label = node?.let { it.text ?: it.contentDescription ?: it.semanticLabel } ?: "input"
                "Type '${action.text}' into [#${action.targetNodeIndex}] '$label'"
            }
            is AgentAction.Scroll -> "Scroll ${action.direction}"
            is AgentAction.PressBack -> "Press Back"
            AgentAction.Submit -> "Submit focused field"
            is AgentAction.PressHome -> "Press Home"
            is AgentAction.Wait -> "Wait (${action.durationMs}ms)"
            is AgentAction.CompleteGoal -> "CompleteGoal('${action.summary}')"
            is AgentAction.FailGoal -> "FailGoal('${action.reason}')"
        }
    }
}
