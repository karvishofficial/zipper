package com.zipper.app.core.agent

import com.zipper.app.core.model.DistilledNode
import com.zipper.app.core.model.DistilledUiState
import java.util.Locale

/**
 * External Oracle Verifier grounded strictly in observable world state (Accessibility tree,
 * package name, node texts, focus, and step history).
 *
 * Implements AGENTS.md Section 18 (Verification) and Section 26 Phase 6 (Task State):
 * The policy model CANNOT declare its own success. Completion is gated by physical verification.
 */
class ExternalOracleVerifier {

    data class TaskProgress(
        val completedRequirements: List<String>,
        val pendingRequirements: List<String>,
        val isAllCompleted: Boolean
    ) {
        fun formatStateHeader(): String {
            val doneStr = if (completedRequirements.isNotEmpty()) "Done:[${completedRequirements.joinToString(";")}]" else "Done:[Init]"
            val pendStr = if (pendingRequirements.isNotEmpty()) "Pending:[${pendingRequirements.joinToString(";")}]" else "Pending:[None]"
            return "State: $doneStr | $pendStr"
        }
    }

    /**
     * Evaluates live task progress from physical world state.
     */
    fun evaluateProgress(
        goal: String,
        uiState: DistilledUiState,
        stepHistory: List<AgentStepRecord> = emptyList()
    ): TaskProgress {
        val cleanGoal = goal.replace(Regex("\\[state:[^\\]]*\\]", RegexOption.IGNORE_CASE), "")
            .replace(Regex("\\[history:[^\\]]*\\]", RegexOption.IGNORE_CASE), "")
            .replace(Regex("\\(previous action failed:[^)]*\\)", RegexOption.IGNORE_CASE), "")
            .trim()
        val lowerGoal = cleanGoal.lowercase(Locale.ROOT)
        val currentPkg = (uiState.packageName ?: "").lowercase(Locale.ROOT)

        val completed = mutableListOf<String>()
        val pending = mutableListOf<String>()

        // 1. App Launch Milestones
        val targetApps = detectTargetApps(lowerGoal)
        for (targetApp in targetApps) {
            val appReached = currentPkg.contains(targetApp.pkgSubstring) ||
                stepHistory.any {
                    it.preScreenSummary.lowercase(Locale.ROOT).contains(targetApp.pkgSubstring) ||
                    it.postScreenSummary.lowercase(Locale.ROOT).contains(targetApp.pkgSubstring) ||
                    (it.action is AgentAction.OpenApp && it.action.appName.contains(targetApp.pkgSubstring, ignoreCase = true))
                }
            if (appReached && currentPkg != "com.zipper.app" && currentPkg.isNotBlank()) {
                completed.add("open_app:${targetApp.name.lowercase(Locale.ROOT)}")
            } else {
                pending.add("open_app:${targetApp.name.lowercase(Locale.ROOT)}")
            }
        }

        // 2. Secondary Milestone: Tab Switch (e.g. "switch to stopwatch", "go to timer", "display")
        val tabTarget = detectTabTarget(lowerGoal)
        if (tabTarget != null) {
            val isTabVerified = isTabActive(tabTarget, uiState, stepHistory)
            if (isTabVerified) {
                completed.add("select_tab:$tabTarget")
            } else {
                pending.add("select_tab:$tabTarget")
            }
        }

        // 3. Secondary Milestone: Search / Query (e.g. "search for Karthik", "search Display")
        val searchQuery = detectSearchQuery(lowerGoal)
        if (searchQuery != null) {
            val isSearchVerified = isSearchExecuted(searchQuery, uiState, stepHistory)
            if (isSearchVerified) {
                completed.add("search:$searchQuery")
            } else {
                pending.add("search:$searchQuery")
            }
        }

        // 4. Secondary Milestone: Arithmetic Calculation (e.g. "calculate 25 + 37")
        val mathExpr = detectMathExpression(lowerGoal)
        if (mathExpr != null) {
            val isMathVerified = isCalculationResultVisible(mathExpr, uiState)
            if (isMathVerified) {
                completed.add("calculate:$mathExpr")
            } else {
                pending.add("calculate:$mathExpr")
            }
        }

        // 5. Secondary Milestone: Click / Play Result (e.g. "and click it", "and click the video", "and play it", "and view top result")
        val hasClickInstruction = lowerGoal.contains("click") || lowerGoal.contains("play") || lowerGoal.contains("watch") || lowerGoal.contains("first video") || lowerGoal.contains("top result") || lowerGoal.contains("open result")
        if (searchQuery != null && hasClickInstruction && !lowerGoal.startsWith("click")) {
            val isResultClicked = isSearchResultClicked(uiState, stepHistory)
            if (isResultClicked) {
                completed.add("click_result")
            } else {
                pending.add("click_result")
            }
        }

        // 6. Secondary Milestone: Trailing Back (e.g. "then press back", "and back", "return")
        val hasBackInstruction = lowerGoal.contains("press back") || lowerGoal.contains("then back") || lowerGoal.contains("then return")
        if (hasBackInstruction) {
            val hasBackExecuted = stepHistory.any { it.action is AgentAction.PressBack }
            if (hasBackExecuted) {
                completed.add("press_back")
            } else {
                pending.add("press_back")
            }
        }

        // 7. Alarm Specific Verification (time on screen)
        val timeMatch = Regex("""\b(\d{1,2}:\d{2})\b""").find(cleanGoal)
        if (lowerGoal.contains("alarm") && timeMatch != null) {
            val requestedTime = timeMatch.groupValues[1]
            val visibleTexts = uiState.nodes.mapNotNull { it.text ?: it.contentDescription }
            if (visibleTexts.any { it.contains(requestedTime) }) {
                completed.add("alarm_time:$requestedTime")
            } else {
                pending.add("alarm_time:$requestedTime")
            }
        }

        // 8. E-Commerce / Cart Milestone (e.g. "add to cart", "add the cheapest tomato to the cart")
        val hasCartInstruction = lowerGoal.contains("cart") || lowerGoal.contains("add ") || lowerGoal.contains("buy ") || lowerGoal.contains("order ") ||
            lowerGoal.contains("cheapest") || lowerGoal.contains("lowest price") || lowerGoal.contains("best offer") || lowerGoal.contains("best deal")
        if (hasCartInstruction) {
            val isCartAdded = stepHistory.any {
                val desc = it.actionDescription.lowercase(Locale.ROOT)
                desc.contains("cart") || desc.contains("add")
            } || uiState.nodes.any {
                val t = (it.text ?: it.contentDescription ?: "").lowercase(Locale.ROOT)
                t.contains("item added") || t.contains("added to cart") || t.contains("1 item in cart") || t.contains("view cart")
            }
            if (isCartAdded && stepHistory.size >= 2) {
                completed.add("add_to_cart")
            } else {
                pending.add("add_to_cart")
            }
        }

        // 9. Messaging Milestone (e.g. "send hi to Bhavani", "message Mom", "text John")
        val hasMessageInstruction = lowerGoal.contains("send ") || lowerGoal.contains("message ") || lowerGoal.contains("text ")
        if (hasMessageInstruction) {
            val isMessageSent = isMessageSentVerified(uiState, stepHistory, lowerGoal)
            if (isMessageSent) {
                completed.add("send_message")
            } else {
                pending.add("send_message")
            }
        }

        // 10. Generic / Form Milestone (when no specific app/tab/search/math/alarm/click/cart/message is requested)
        if (targetApps.isEmpty() && tabTarget == null && searchQuery == null && mathExpr == null && timeMatch == null && !hasClickInstruction && !hasBackInstruction && !hasCartInstruction && !hasMessageInstruction) {
            if (currentPkg != "com.zipper.app" && currentPkg.isNotBlank()) {
                completed.add("interaction_progress")
            }
        }

        return TaskProgress(
            completedRequirements = completed,
            pendingRequirements = pending,
            isAllCompleted = pending.isEmpty() && completed.isNotEmpty()
        )
    }

    private data class TargetAppInfo(val name: String, val pkgSubstring: String)

    private fun detectTargetApps(lowerGoal: String): List<TargetAppInfo> {
        val appMap = listOf(
            TargetAppInfo("Clock", "clock"),
            TargetAppInfo("Alarm", "clock"),
            TargetAppInfo("Settings", "setting"),
            TargetAppInfo("Contacts", "contact"),
            TargetAppInfo("Calculator", "calculat"),
            TargetAppInfo("YouTube", "youtube"),
            TargetAppInfo("Files", "files"),
            TargetAppInfo("Messages", "messag"),
            TargetAppInfo("Gallery", "gallery"),
            TargetAppInfo("Maps", "maps"),
            TargetAppInfo("Camera", "camera"),
            TargetAppInfo("Calendar", "calendar"),
            TargetAppInfo("Chrome", "chrome"),
            TargetAppInfo("LinkedIn", "linkedin"),
            TargetAppInfo("Instagram", "instagram"),
            TargetAppInfo("WhatsApp", "whatsapp"),
            TargetAppInfo("Zepto", "zepto"),
            TargetAppInfo("Play Store", "vending"),
            TargetAppInfo("Play", "vending"),
            TargetAppInfo("Keep", "keep"),
            TargetAppInfo("ChatGPT", "chatgpt"),
            TargetAppInfo("Gemini", "bard"),
            TargetAppInfo("Claude", "claude"),
            TargetAppInfo("Perplexity", "perplexity")
        )
        val matched = appMap.filter { lowerGoal.contains(it.name.lowercase(Locale.ROOT)) }.toMutableList()
        // Universal dynamic app detection for unseen apps (e.g. "open Swiggy", "launch Uber")
        val m = Regex("""\b(?:open|launch)\s+([a-zA-Z0-9]+)""").find(lowerGoal)
        if (m != null) {
            val candidateName = m.groupValues[1].lowercase(Locale.ROOT)
            val stopWords = setOf("the", "a", "an", "app", "tab", "setting", "search", "and", "then")
            if (candidateName !in stopWords && matched.none { it.name.lowercase(Locale.ROOT) == candidateName }) {
                matched.add(TargetAppInfo(candidateName.replaceFirstChar { it.uppercase() }, candidateName))
            }
        }
        return matched
    }

    private fun detectTabTarget(lowerGoal: String): String? {
        val clickTarget = GoalArgumentResolver.resolveClickTarget(lowerGoal)?.lowercase(Locale.ROOT)
        if (clickTarget != null && clickTarget !in AgentRuntime.KNOWN_TARGET_APPS) {
            return clickTarget
        }
        val tabKeywords = listOf("stopwatch", "timer", "world clock", "display", "sound", "battery", "apps", "notifications", "favorites", "groups", "downloads", "images", "videos", "home", "books", "games", "reels", "shorts", "chats", "messages")
        val patterns = listOf(
            Regex("""\b(?:switch to|go to|select|open tab|click|tap)\s+([a-z\s]+)"""),
            Regex("""\b([a-z]+)\s+(?:tab|section|page)\b""")
        )
        for (pattern in patterns) {
            val match = pattern.find(lowerGoal)
            if (match != null) {
                val matched = match.groupValues[1].trim()
                val keyword = tabKeywords.firstOrNull { matched.contains(it) }
                if (keyword != null) return keyword
            }
        }
        return tabKeywords.firstOrNull { lowerGoal.contains(it) && !lowerGoal.startsWith("open $it") }
    }

    private fun isTabActive(tabTarget: String, uiState: DistilledUiState, stepHistory: List<AgentStepRecord>): Boolean {
        // Physical verification: Is the tab selected, or are its unique controls visible on screen?
        val tabTargetLower = tabTarget.lowercase(Locale.ROOT)

        // 1. Direct tab element with isSelected or checked
        val matchingNodes = uiState.nodes.filter {
            val label = listOfNotNull(it.text, it.contentDescription).joinToString(" ").lowercase(Locale.ROOT)
            label.contains(tabTargetLower)
        }
        if (matchingNodes.any { it.role.contains("TAB", true) || it.role.contains("BUTTON", true) }) {
            // Check if selected or title
            if (matchingNodes.any { it.text?.equals(tabTarget, ignoreCase = true) == true && it.bounds.height < 200 }) {
                val tappedTab = stepHistory.any {
                    it.action is AgentAction.Tap && it.actionDescription.lowercase(Locale.ROOT).contains(tabTargetLower)
                }
                if (tappedTab) return true
            }
        }

        // 2. Tab-specific controls (e.g. Stopwatch has Lap, Reset, or Start buttons)
        if (tabTargetLower == "stopwatch") {
            val hasStopwatchControls = uiState.nodes.any {
                val lbl = listOfNotNull(it.text, it.contentDescription).joinToString(" ").lowercase(Locale.ROOT)
                lbl.contains("lap") || lbl.contains("reset") || lbl.contains("start") || lbl == "00:00.00"
            }
            if (hasStopwatchControls) return true
        }

        // 3. Tab-specific controls for Timer
        if (tabTargetLower == "timer") {
            val hasTimerControls = uiState.nodes.any {
                val lbl = listOfNotNull(it.text, it.contentDescription).joinToString(" ").lowercase(Locale.ROOT)
                lbl.contains("cancel") || lbl.contains("hour") || lbl.contains("start")
            }
            if (hasTimerControls) return true
        }

        return stepHistory.any {
            it.action is AgentAction.Tap && it.actionDescription.lowercase(Locale.ROOT).contains(tabTargetLower) &&
            it.outcome == StepOutcome.SUCCESS
        }
    }

    private fun detectSearchQuery(lowerGoal: String): String? {
        val searchRegexes = listOf(
            Regex("""\b(?:search|find|look for)\s+(?:for\s+)?(.+?)(?:\s+(?:in|and|then|view|to|comma)\s*|,|$)"""),
            Regex("""\b(?:type|enter|input)\s+['\"]?(.+?)['\"]?(?:\s+(?:into|in|then|and)\s+|,|$)"""),
            Regex("""\b(?:ask|tell|prompt)\s+[A-Za-z0-9_]+\s+(?:to\s+)?(?:about\s+)?(.+?)(?:\s+(?:in|and|then)\s*|,|$)""")
        )
        for (r in searchRegexes) {
            val m = r.find(lowerGoal)
            if (m != null) {
                var query = m.groupValues[1].trim().trim('\'', '"', '.', ',')
                query = query.split(Regex("""\s*,\s*|\s+(?:view|and|then|to|press)\s+"""))[0].trim().trim('\'', '"', '.', ',')
                if (query.isNotBlank() && query != "contacts" && query != "settings" && query != "for") {
                    return query
                }
            }
        }
        return null
    }

    private fun isSearchExecuted(query: String, uiState: DistilledUiState, stepHistory: List<AgentStepRecord>): Boolean {
        val cleanQuery = query.lowercase(Locale.ROOT)
        val typeIndex = stepHistory.indexOfLast {
            it.action is AgentAction.TypeText && (
                it.action.text.lowercase(Locale.ROOT).contains(cleanQuery) ||
                cleanQuery.contains(it.action.text.lowercase(Locale.ROOT))
            )
        }
        val hasTypedQuery = typeIndex != -1
        val hasSubmitted = typeIndex != -1 && stepHistory.drop(typeIndex).any { it.action is AgentAction.Submit }
        val hasTappedSendAfterType = typeIndex != -1 && stepHistory.drop(typeIndex + 1).any {
            it.action is AgentAction.Tap && (
                it.actionDescription.lowercase(Locale.ROOT).contains("send") ||
                it.actionDescription.lowercase(Locale.ROOT).contains("submit") ||
                it.actionDescription.lowercase(Locale.ROOT).contains("search")
            )
        }
        val hasTappedSuggestion = stepHistory.any {
            it.action is AgentAction.Tap && (
                it.actionDescription.lowercase(Locale.ROOT).contains(cleanQuery) ||
                it.actionDescription.lowercase(Locale.ROOT).contains("show all results")
            ) && !it.actionDescription.lowercase(Locale.ROOT).contains("show all recent")
        }
        val isEditorOrCreateScreen = uiState.nodes.any {
            val t = listOfNotNull(it.text, it.contentDescription).joinToString(" ").lowercase(Locale.ROOT)
            val res = it.resourceId?.lowercase(Locale.ROOT).orEmpty()
            t.contains("quiz") || t.contains("add a quiz") || res.contains("attach_quiz") || res.contains("post_editor") || t.contains("quiz your community")
        }
        val isRecentPage = uiState.nodes.any { 
            val t = listOfNotNull(it.text, it.contentDescription).joinToString(" ").lowercase(Locale.ROOT)
            t.contains("recent") || t.contains("clear all")
        }
        val isSuggestionList = uiState.nodes.any {
            val res = it.resourceId?.lowercase(Locale.ROOT).orEmpty()
            val desc = it.contentDescription?.lowercase(Locale.ROOT).orEmpty()
            res.contains("edit_suggestion") || desc.contains("edit suggestion")
        }
        val hasFilterChips = uiState.nodes.any {
            val t = listOfNotNull(it.text, it.contentDescription).joinToString(" ").lowercase(Locale.ROOT)
            t.contains("filter by") || t.contains("for you") || t.contains("accounts") || t.contains("channels") || t.contains("search filters") || t.contains("videos") || t.contains("shorts")
        }
        val hasSearchResultsOnScreen = !isSuggestionList && uiState.nodes.any {
            if (it.isEditable || it.role.contains("EDIT", true) || it.role.contains("INPUT", true) || it.className?.contains("EditText", true) == true) {
                false
            } else {
                val t = listOfNotNull(it.text, it.contentDescription).joinToString(" ").lowercase(Locale.ROOT)
                t.contains(cleanQuery) && !t.contains("search") && !t.contains("clear")
            }
        }
        return !isEditorOrCreateScreen && ((hasTypedQuery && (hasSubmitted || hasTappedSendAfterType)) || hasTappedSuggestion || ((hasTypedQuery || hasTappedSuggestion) && (hasFilterChips || hasSearchResultsOnScreen) && !isRecentPage && !isSuggestionList))
    }

    private fun detectMathExpression(lowerGoal: String): String? {
        val match = Regex("""(\d+(?:[\s\+\-\*\/\%x\=]+\d+)+)""").find(lowerGoal)
        return match?.groupValues?.get(1)?.trim()
    }

    private fun isCalculationResultVisible(mathExpr: String, uiState: DistilledUiState): Boolean {
        val parsed = GenericExpressionEvaluator.parseAndEvaluate(mathExpr) ?: return false
        val resultStr = parsed.expectedResult
        return uiState.nodes.any {
            val lbl = listOfNotNull(it.text, it.contentDescription).joinToString(" ")
            lbl.contains(resultStr)
        }
    }

    private fun isSearchResultClicked(uiState: DistilledUiState, stepHistory: List<AgentStepRecord>): Boolean {
        // Grounded observable evidence: Active player/viewer controls
        val hasPlayerControls = uiState.nodes.any {
            val lbl = listOfNotNull(it.text, it.contentDescription).joinToString(" ").lowercase(Locale.ROOT)
            lbl.contains("pause") || lbl.contains("play video") || lbl.contains("seek") || lbl.contains("player")
        }
        if (hasPlayerControls) return true

        // If the screen still has search suggestions / search input affordances, result is NOT clicked
        val isSuggestionScreen = uiState.nodes.any {
            val lbl = listOfNotNull(it.text, it.contentDescription).joinToString(" ").lowercase(Locale.ROOT)
            lbl.contains("edit suggestion") || lbl.contains("voice search") || lbl.contains("clear search") || lbl.trim() == "clear"
        }
        if (isSuggestionScreen) return false

        val submitIndex = stepHistory.indexOfFirst {
            it.action is AgentAction.Submit && it.outcome == StepOutcome.SUCCESS
        }
        if (submitIndex >= 0 && submitIndex < stepHistory.size - 1) {
            val postSubmitTaps = stepHistory.drop(submitIndex + 1).any {
                it.action is AgentAction.Tap && it.outcome == StepOutcome.SUCCESS
            }
            if (postSubmitTaps) return true
        }
        return false
    }

    private fun isMessageSentVerified(uiState: DistilledUiState, stepHistory: List<AgentStepRecord>, lowerGoal: String): Boolean {
        val hasTypedMessage = stepHistory.any { 
            it.action is AgentAction.TypeText && 
            it.outcome == StepOutcome.SUCCESS &&
            !it.actionDescription.lowercase(Locale.ROOT).contains("search")
        }
        val hasTappedSend = stepHistory.any {
            it.action is AgentAction.Tap &&
            it.actionDescription.lowercase(Locale.ROOT).contains("send") &&
            it.outcome == StepOutcome.SUCCESS
        }
        if (!hasTypedMessage || !hasTappedSend) return false

        // Grounded recipient physical verification (AGENTS.md Section 18)
        val recipientMatch = Regex("""\b(?:to|message|text)\s+([a-zA-Z0-9_\.]+)""").find(lowerGoal)
        if (recipientMatch != null) {
            val targetRecipient = recipientMatch.groupValues[1].lowercase(Locale.ROOT)
            val stopRecipients = setOf("the", "a", "an", "all", "cart", "chat", "this", "my")
            if (targetRecipient !in stopRecipients && targetRecipient.length >= 2) {
                val visibleTexts = uiState.nodes.mapNotNull { it.text ?: it.contentDescription }.joinToString(" ").lowercase(Locale.ROOT)
                val historyTexts = stepHistory.joinToString(" ") { it.actionDescription }.lowercase(Locale.ROOT)
                val recipientFound = visibleTexts.contains(targetRecipient) || historyTexts.contains(targetRecipient)
                if (!recipientFound) {
                    return false // Sent to wrong recipient! Fail closed.
                }
            }
        }
        return true
    }
}
