package com.zipper.app.core.agent

import com.zipper.app.core.model.DistilledNode
import com.zipper.app.core.model.DistilledUiState
import org.json.JSONArray
import org.json.JSONObject

/**
 * Dynamically converts the live DistilledUiState into structured JSON tool schemas
 * for the Cactus Needle 2 45M model.
 *
 * Excludes or penalizes blacklisted nodes that failed in previous turns.
 */
class ToolSchemaGenerator {

    companion object {
        private const val MAX_CANDIDATE_TOOLS = 58
    }

    data class GeneratedTools(
        val toolsJsonArray: JSONArray,
        val candidateNodeMap: Map<Int, DistilledNode>,
        val summary: String,
        val candidateMap: Map<String, ActionCandidate> = emptyMap()
    )

    private fun deriveSemanticTags(node: DistilledNode, rawLabel: String, goal: String = ""): List<String> {
        val label = rawLabel.lowercase()
        val resource = node.resourceId?.lowercase().orEmpty()
        val goalLower = goal.lowercase()
        val tags = linkedSetOf<String>()
        if (node.isClickable) tags.add("click_capability")
        if (node.isEditable) tags.add("type_capability")
        val controlMarker = listOf(node.role, node.className.orEmpty(), node.resourceId.orEmpty(), label)
            .joinToString(" ")
            .lowercase()
        if (controlMarker.contains("tab") || controlMarker.contains("tablayout") ||
            controlMarker.contains("segmented") || controlMarker.contains("navigation")) {
            tags.add("tab_control")
            tags.add("navigation_choice")
        }
        val targetEntity = GoalArgumentResolver.resolveTargetEntity(goal)?.lowercase().orEmpty()
        if (targetEntity.isNotBlank()) {
            val candTokens = (label + " " + resource).split(Regex("[^a-z0-9]+")).filter { it.length >= 2 }
            val targetTokens = targetEntity.split(Regex("[^a-z0-9]+")).filter { it.length >= 2 }
            if (targetTokens.any { t ->
                candTokens.any { c ->
                    c == t || c.startsWith(t) || t.startsWith(c) ||
                    (t.endsWith("s") && c == t.dropLast(1)) ||
                    (c.endsWith("s") && t == c.dropLast(1))
                }
            }) {
                tags.add("target_match")
                if (controlMarker.contains("tab") || controlMarker.contains("navigation")) {
                    tags.add("requested_section")
                }
            }
        }
        if (node.isScrollable) {
            tags.add("scroll_capability")
            tags.add("scroll_forward_capability")
            tags.add("scroll_backward_capability")
        }
        if (node.isCheckable) {
            tags.add("toggle_capability")
            tags.add("select_capability")
        }
        if (node.isSelected) {
            tags.add("already_selected")
            tags.add("active_tab")
        }
        if (node.role.contains("RADIO", ignoreCase = true) ||
            node.role.contains("CHECK", ignoreCase = true) ||
            node.role.contains("SELECT", ignoreCase = true)
        ) {
            tags.add("select_capability")
        }
        if (node.isScrollable && (node.role.contains("NUMBER", ignoreCase = true) || node.role.contains("PICKER", ignoreCase = true))) {
            tags.add("value_picker_control")
            tags.add("time_affordance")
        }
        if (node.isEditable) {
            tags.add("input_field")
            if (label.contains("search") || label.contains("query") || label.contains("find")) tags.add("search_input")
            if (label.contains("message") || label.contains("comment") || label.contains("text")) tags.add("message_input")
        }
        if (node.isCheckable) tags.add("toggle_control")
        if (node.isChecked) tags.add("toggle_on")
        if (label.contains("add") || label.contains("create") || label.contains("new")) tags.add("create_affordance")
        if (label.contains("save") || label.contains("done") || label.contains("ok") || label.contains("set") ||
            label.contains("send") || label.contains("submit") || label.contains("confirm") || resource.contains("send")) {
            tags.add("commit_affordance")
            tags.add("send_affordance")
        }
        if (label.contains("menu") || resource.contains("menu") || label.contains("more options") || resource.contains("more_options")) {
            tags.add("menu_control")
            tags.add("navigation_affordance")
        }
        val isEditSuggestionArrow = label.contains("edit suggestion")
        if (isEditSuggestionArrow) {
            tags.add("refine_affordance")
        } else if (label.contains("suggestion") || resource.contains("suggestion") || node.className?.contains("suggestion", ignoreCase = true) == true) {
            tags.add("autocomplete_suggestion")
        }
        val isCardOrContentItem = rawLabel.length > 20 ||
            label.contains("video") || label.contains("views") || label.contains("channel") ||
            label.contains("result") || label.contains("item") || label.contains("download")
        if (node.isClickable && !node.isEditable && (node.role !in setOf("BUTTON", "IMAGE_BUTTON") || isCardOrContentItem) &&
            rawLabel.isNotBlank() && !isEditSuggestionArrow && !tags.contains("autocomplete_suggestion")) {
            tags.add("result_item")
        }
        if (goalLower.contains("second") || goalLower.contains("third") || goalLower.contains("fourth") ||
            goalLower.contains("1st") || goalLower.contains("2nd") || goalLower.contains("3rd") || goalLower.contains("4th")) {
            if (node.isClickable && !node.isEditable) tags.add("ordinal_candidate")
        }
        val isRealPicker = isPickerLike(node) || (node.isScrollable && (node.role.contains("NUMBER", ignoreCase = true) || node.role.contains("PICKER", ignoreCase = true)))
        if (isRealPicker) {
            tags.add("value_picker_control")
            tags.add("time_affordance")
        }
        if (label.matches(Regex(".*\\b\\d{1,2}[:.]\\d{2}\\b.*"))) tags.add("time_value")
        if (label.matches(Regex(".*\\b[0-9+\\-*/=.:%]+\\b.*"))) tags.add("numeric_affordance")
        if (label.contains("date") || label.contains("monday") || label.contains("tuesday") || label.contains("wednesday") ||
            label.contains("thursday") || label.contains("friday") || label.contains("saturday") || label.contains("sunday") ||
            label.contains("january") || label.contains("february") || label.contains("march") || label.contains("april") ||
            label.contains("may") || label.contains("june") || label.contains("july") || label.contains("august") ||
            label.contains("september") || label.contains("october") || label.contains("november") || label.contains("december")) {
            tags.add("date_affordance")
        }
        return tags.toList()
    }

    fun generateTools(
        uiState: DistilledUiState,
        blacklistedNodeIndices: Set<Int> = emptySet(),
        goal: String = "",
        stepHistory: List<AgentStepRecord> = emptyList(),
        blacklistedSemanticKeys: Set<String> = emptySet(),
        currentPhase: TaskPhase = TaskPhase.REACH_CONTEXT
    ): GeneratedTools {
        val toolsArray = JSONArray()
        val candidateMap = mutableMapOf<Int, DistilledNode>()
        val actionCandidateMap = mutableMapOf<String, ActionCandidate>()

        // 1. App Launching Primitive
        val isKarvishLauncher = uiState.packageName == "com.zipper.app" || AgentRuntime.isLauncherOrSystemPackage(uiState.packageName)
        val launchTargets = AgentRuntime.resolveTargetAppsFromGoal(goal, null)
        val currentScreenIdentity = listOfNotNull(
            uiState.packageName,
            uiState.activityName,
            uiState.nodes.firstOrNull()?.text,
            uiState.nodes.firstOrNull()?.contentDescription
        ).joinToString(" ").lowercase()
        val appsAlreadyOpened = stepHistory.mapNotNull { (it.action as? AgentAction.OpenApp)?.appName?.lowercase() }
        val unfulfilledTargets = launchTargets.filter { target ->
            val targetLower = target.lowercase()
            val targetTokens = targetLower.split(Regex("[^a-z0-9]+"))
                .filter { it.length >= 3 }
            val isCurrentScreen = AgentRuntime.isAppMatchingTarget(uiState.packageName, target) ||
                (targetTokens.isNotEmpty() && targetTokens.all { it in currentScreenIdentity })
            val isAlreadyOpened = appsAlreadyOpened.any { it.contains(targetLower) || targetLower.contains(it) }
            !isCurrentScreen && !isAlreadyOpened
        }
        val targetAppHint = unfulfilledTargets.firstOrNull() ?: launchTargets.firstOrNull() ?: ""
        val hasUnfulfilledLaunch = unfulfilledTargets.isNotEmpty()
        if (launchTargets.isEmpty() || hasUnfulfilledLaunch || isKarvishLauncher) {
            val appDesc = if (targetAppHint.isNotBlank()) {
                "Launch application '$targetAppHint'"
            } else {
                "Launch the installed Android application that provides the capability required by this goal: '$goal'"
            }
            val openAppCand = ActionCandidate(
                candidateId = "c_open_app",
                actionType = "open_app",
                nodeIndex = null,
                nodeIdentity = "system:open_app",
                semanticLabel = if (targetAppHint.isNotBlank()) "open ${targetAppHint.lowercase()}" else "open application",
                role = "system",
                node = null
            )
            actionCandidateMap["c_open_app"] = openAppCand
            actionCandidateMap["open_app"] = openAppCand

            toolsArray.put(JSONObject().apply {
                put("name", "open_app")
                put("candidate_id", "c_open_app")
                put("action_type", "open_app")
                put("role", "system")
                put("semantic_label", openAppCand.semanticLabel)
                put("node_identity", "system:open_app")
                put("description", appDesc)
                put("app_name", targetAppHint)
                put("semantic_tags", JSONArray().put("external_capability_launch"))
                put("parameters", JSONObject().apply {
                    put("type", "object")
                    put("properties", JSONObject().apply {
                        put("app_name", JSONObject().put("type", "string").put("description", "The name of the application to open"))
                    })
                    put("required", JSONArray().put("app_name"))
                })
            })
        }

        // 2. Interactive Elements on Screen (Tap & Type)
        // When on Karvish's own launcher activity, do not generate element tools from Karvish's own controls
        val nodesByIndex = uiState.nodes.associateBy { it.index }
        fun effectiveLabel(node: DistilledNode): String {
            val cleanText = node.text?.takeIf { it.isNotBlank() }
            val cleanDesc = node.contentDescription?.takeIf { it.isNotBlank() }
            val cleanHint = node.hintText?.takeIf { it.isNotBlank() }

            // If this node is a contact row or conversation row container, prioritize the primary contact name child!
            val resId = node.resourceId?.lowercase().orEmpty()
            val isConversationOrContactRow = resId.contains("contact") || resId.contains("conversation") || resId.contains("recipient") || resId.contains("chat_row") || resId.contains("user_row") || resId.contains("row_inbox") || resId.contains("inbox_row")
            if (isConversationOrContactRow) {
                val primaryNameChild = uiState.nodes.firstOrNull { descendant ->
                    var parent = descendant.parentIndex
                    var isDescendant = false
                    while (parent != null) {
                        if (parent == node.index) { isDescendant = true; break }
                        parent = nodesByIndex[parent]?.parentIndex
                    }
                    isDescendant && run {
                        val dRes = descendant.resourceId?.lowercase().orEmpty()
                        dRes.contains("contact_name") || dRes.contains("user_name") || dRes.contains("username") || dRes.contains("row_inbox_username") || dRes.contains("name_tv") || dRes.contains("title_tv") || dRes.contains("chat_title")
                    }
                }
                val primaryName = primaryNameChild?.text?.trim()
                if (!primaryName.isNullOrBlank()) {
                    return primaryName
                }
            }

            val sanitizedDesc = cleanDesc?.let { desc ->
                var d = desc
                if (d.contains("edit suggestion", ignoreCase = true)) {
                    d = d.split(Regex("(?i),?\\s*edit suggestion.*"))[0].trim()
                }
                // Strip group member presence snippets (e.g. "Ajay is also in this group", "X is in this group")
                // so a group chat does not falsely pose as the named contact.
                d = d.replace(Regex("(?i)\\b\\S+\\s+is\\s+(also\\s+)?in\\s+this\\s+group.*"), "").trim()
                d.ifBlank { desc }
            }

            val labels = linkedSetOf<String>()
            if (cleanText != null) {
                labels.add(cleanText)
                if (sanitizedDesc != null && !sanitizedDesc.contains(cleanText, ignoreCase = true) && !cleanText.contains(sanitizedDesc, ignoreCase = true)) {
                    labels.add(sanitizedDesc)
                }
            } else if (sanitizedDesc != null) {
                labels.add(sanitizedDesc)
            } else if (cleanHint != null) {
                labels.add(cleanHint)
            } else {
                // If text/desc is blank, check if resourceId indicates a standard universal action (Send, Submit, Back, Close, Menu)
                val res = resId.lowercase()
                if (res.contains(Regex("\\b(send|button_send|composer_send)\\b")) || res.endsWith("send") || res.contains("send_button") || res.contains("composer_button_send") || res.contains("row_thread_composer")) {
                    labels.add("Send")
                } else if (node.role.contains("send", ignoreCase = true) || node.semanticLabel.contains("send", ignoreCase = true)) {
                    labels.add("Send")
                } else if (res.contains("menu_button") || res.contains("action_menu") || res.contains("toolbar_menu") || res.endsWith("menu")) {
                    labels.add("Menu")
                }
            }

            // Chrome & Web Navigation Menu Recognition:
            // 1. Android toolbar "More options" or menu_button represents the primary menu
            val hasMenuGlyph = (cleanText?.any { it in '\uE000'..'\uF8FF' || it in setOf('\u2630', '\u22EE', '\u22EF') } == true) ||
                (sanitizedDesc?.any { it in '\uE000'..'\uF8FF' || it in setOf('\u2630', '\u22EE', '\u22EF') } == true)
            if (cleanDesc?.contains("more options", ignoreCase = true) == true || resId.contains("menu_button") || resId.contains("action_menu") || hasMenuGlyph) {
                labels.add("Menu")
            }
            // 2. Web DOM header icon buttons (hamburger menus, nav icons) often lack accessibility text in mobile Chromium
            val isTopHeaderNav = node.bounds.top < 700 && node.bounds.height in 20..350
            val isActionableWebAffordance = (node.isClickable || node.role in setOf("BUTTON", "IMAGE_BUTTON", "LINK", "IMAGE", "VIEW") || node.className?.contains("Button", ignoreCase = true) == true || node.className?.contains("View", ignoreCase = true) == true)
            val isWebOrChromeContext = uiState.packageName?.contains("chrome") == true || uiState.nodes.any { it.role == "WEBVIEW" || it.className?.contains("WebView", ignoreCase = true) == true }
            if (labels.isEmpty() && isWebOrChromeContext && isTopHeaderNav && isActionableWebAffordance) {
                if (node.bounds.left < 300) {
                    labels.add("Menu")
                } else if (node.bounds.left > 750) {
                    labels.add("Menu")
                } else {
                    labels.add("Navigation")
                }
            }

            if (node.bounds.height <= 800) {
                uiState.nodes
                    .filter { descendant ->
                        var parent = descendant.parentIndex
                        while (parent != null) {
                            if (parent == node.index) return@filter true
                            parent = nodesByIndex[parent]?.parentIndex
                        }
                        false
                    }
                    .filter { descendant ->
                        // Don't include chat preview snippets in the row's identity,
                        // otherwise past message text mentioning a name will cause
                        // another contact to falsely match the goal's target contact.
                        val res = descendant.resourceId?.lowercase().orEmpty()
                        val text = descendant.text?.lowercase().orEmpty()
                        !res.contains("single_msg") &&
                        !res.contains("msg_tv") &&
                        !res.contains("snippet") &&
                        !res.contains("preview") &&
                        !res.contains("last_msg") &&
                        !res.contains("status_msg") &&
                        !res.contains("digest") &&
                        !res.contains("subtitle") &&
                        !res.contains("follow") &&
                        !text.contains(Regex("\\bis\\s+(also\\s+)?in\\s+this\\s+group\\b"))
                    }
                    .flatMap { descendant -> listOfNotNull(descendant.text, descendant.contentDescription, descendant.hintText) }
                    .filter { it.isNotBlank() && !it.contains("edit suggestion", ignoreCase = true) }
                    .forEach { descStr ->
                        val cleanDescStr = descStr.trim()
                        if (labels.none { it.contains(cleanDescStr, ignoreCase = true) || cleanDescStr.contains(it, ignoreCase = true) }) {
                            labels.add(cleanDescStr)
                        }
                    }
            }

            // Universal Platform Observation: When an action button on an e-commerce item card has only an ambiguous verb
            // (e.g. "add", "buy", "+", "order"), propagate the surrounding card context (title, price)
            // so the neural model can ground what item is being acted upon.
            // NEVER corrupt self-sufficient buttons (Search, Back, Home, Notifications, etc.) or climb into full-screen coordinators.
            val currentText = labels.joinToString(" ").trim()
            val currentLower = currentText.lowercase()
            val isAmbiguousItemAction = currentLower in setOf("add", "buy", "cart", "plus", "+", "order", "add to cart") ||
                currentLower.startsWith("add to cart") || currentLower.startsWith("add ") || currentLower == "add"

            if (isAmbiguousItemAction && node.parentIndex != null) {
                var ancestor = nodesByIndex[node.parentIndex]
                var levels = 0
                val contextTexts = linkedSetOf<String>()
                while (ancestor != null && levels < 3) {
                    val aBounds = ancestor.bounds
                    // Stop if we hit a screen-wide container or scroll list
                    if (aBounds.height > 800 || ancestor.role in setOf("SCROLL_VIEW", "RECYCLER_VIEW")) {
                        break
                    }
                    if (ancestor.contentDescription?.isNotBlank() == true) {
                        val clean = ancestor.contentDescription!!.trim()
                        if (labels.none { it.contains(clean, ignoreCase = true) || clean.contains(it, ignoreCase = true) }) {
                            contextTexts.add(clean)
                        }
                    }
                    if (ancestor.text?.isNotBlank() == true) {
                        val clean = ancestor.text!!.trim()
                        if (labels.none { it.contains(clean, ignoreCase = true) || clean.contains(it, ignoreCase = true) }) {
                            contextTexts.add(clean)
                        }
                    }
                    val aIndex = ancestor.index
                    uiState.nodes
                        .filter { other ->
                            other.index != node.index && (other.parentIndex == aIndex || nodesByIndex[other.parentIndex]?.parentIndex == aIndex)
                        }
                        .flatMap { other -> listOfNotNull(other.text, other.contentDescription, other.hintText) }
                        .filter { it.isNotBlank() && !it.contains("edit suggestion", ignoreCase = true) }
                        .forEach { descStr ->
                            val clean = descStr.trim()
                            if (labels.none { it.contains(clean, ignoreCase = true) || clean.contains(it, ignoreCase = true) }) {
                                contextTexts.add(clean)
                            }
                        }
                    if (contextTexts.isNotEmpty() && contextTexts.any { it.contains(Regex("[₹$€£]")) }) {
                        break
                    }
                    ancestor = nodesByIndex[ancestor.parentIndex]
                    levels++
                }
                if (contextTexts.isNotEmpty()) {
                    val fullList = contextTexts.toList() + labels.toList()
                    return fullList.distinct().joinToString(" ")
                }
            }

            return labels.joinToString(" ")
        }

        fun semanticKey(node: DistilledNode, label: String): String {
            val normalized = label.lowercase().replace(Regex("\\s+"), " ").trim()
            val stableId = node.resourceId?.lowercase()?.takeIf { it.isNotBlank() }
            return "${node.role.lowercase()}|${stableId ?: normalized}"
        }

        val visibleNodes = if (uiState.packageName == "com.zipper.app") {
            emptyList()
        } else {
            uiState.nodes.filter { node ->
                val isBlacklisted = node.index in blacklistedNodeIndices
                val isContainer = node.role in setOf("SCROLL_VIEW", "LIST", "CONTAINER", "WEBVIEW") ||
                    (!node.isClickable && !node.isEditable && node.isScrollable)
                val rawLabel = effectiveLabel(node)
                val isSemanticallyBlacklisted = semanticKey(node, rawLabel) in blacklistedSemanticKeys
                val resourceLower = node.resourceId?.lowercase().orEmpty()
                val rawLower = rawLabel.lowercase()
                val isSendOrSubmitAction = resourceLower.contains("send") ||
                    rawLower == "send" || rawLower.startsWith("send ") ||
                    rawLower.contains(Regex("\\b(send|submit)\\b")) ||
                    resourceLower.contains("submit")
                val isMenuAction = rawLower.contains("menu") || resourceLower.contains("menu") || rawLower.contains("more options")
                val isTransientProgress = rawLower in setOf("searching…", "searching...", "loading…", "loading...", "please wait", "refreshing...") ||
                    node.role in setOf("PROGRESS_BAR", "SPINNER") ||
                    node.className?.contains("ProgressBar", ignoreCase = true) == true
                if (isTransientProgress) return@filter false
                val hasLabel = rawLabel.isNotBlank() || node.isEditable || node.role.equals("INPUT", ignoreCase = true) || isSendOrSubmitAction || isMenuAction ||
                    resourceLower.contains(Regex("search|query|find|submit|confirm|save|next|done|send|menu|input|edit|text"))
                val isAiPromptOrSendTrigger = !isSendOrSubmitAction && (rawLower in setOf("ask", "ask ai", "generate") ||
                    (rawLower.startsWith("ask ") && !rawLower.contains("search") && !rawLower.contains("find")) ||
                    resourceLower.contains("meta_ai") ||
                    resourceLower.contains("metaai") ||
                    resourceLower.contains("ai_send") ||
                    resourceLower.contains("ai_input_send") ||
                    resourceLower.contains("assistant_send") ||
                    (resourceLower.contains("send_button") && (resourceLower.contains("ai") || resourceLower.contains("bot") || resourceLower.contains("assistant"))))

                val isTopSearchAffordance = !isAiPromptOrSendTrigger &&
                    (rawLower.contains("search") || resourceLower.contains("search")) &&
                    node.bounds.top < 600 && node.bounds.height in 25..350

                // Senior Step 2: Generic Accessibility-Tree Normalization (Multi-Signal)
                // 1. Parent-Child Action Binding: If this node is a child visual element (avatar/image/icon)
                // or status badge (unread count/snippet) whose parent/ancestor container is already an actionable clickable row or card,
                // the parent container represents the action. The child element does not compete as a separate tool.
                val isVisualElement = !isSendOrSubmitAction && !isMenuAction && (node.role == "IMAGE" ||
                    node.className?.contains("image", ignoreCase = true) == true ||
                    node.resourceId?.contains("avatar", ignoreCase = true) == true ||
                    node.resourceId?.contains("photo", ignoreCase = true) == true ||
                    node.resourceId?.contains("icon", ignoreCase = true) == true ||
                    node.resourceId?.contains("contact_selector", ignoreCase = true) == true ||
                    node.resourceId?.contains("thumbnail", ignoreCase = true) == true)

                val isUnreadBadgeOrSnippet = (resourceLower.contains("badge") ||
                    resourceLower.contains("unread") ||
                    resourceLower.contains("count") ||
                    rawLower.matches(Regex("""^\d+\s+unread(\s+messages?)?$""")) ||
                    rawLower in setOf("photo", "video", "audio", "voice message", "missed call")) &&
                    !node.isEditable

                val isChildVisualOrBadgeDecoration = (isVisualElement || isUnreadBadgeOrSnippet) &&
                    run {
                        var p = node.parentIndex
                        var depth = 0
                        var foundActionableAncestor = false
                        while (p != null && depth < 4) {
                            val ancestor = nodesByIndex[p]
                            if (ancestor != null && (ancestor.isClickable || ancestor.role in setOf("ITEM", "ROW", "CARD")) && ancestor.role !in setOf("SCROLL_VIEW", "LIST", "WEBVIEW")) {
                                val ancestorArea = ancestor.bounds.width * ancestor.bounds.height
                                val nodeArea = node.bounds.width * node.bounds.height
                                if (ancestorArea.toFloat() > nodeArea * 1.1f || ancestor.resourceId?.contains("row", ignoreCase = true) == true) {
                                    foundActionableAncestor = true
                                    break
                                }
                            }
                            p = ancestor?.parentIndex
                            depth++
                        }
                        foundActionableAncestor
                    }

                // 2. Multi-Signal Informative Metric / Stat Recognition:
                // If a non-editable, non-actionable TextView represents pure numeric metrics/counters (e.g. "45.7K", "266 posts"),
                // demote it so it does not distract from real action targets like "Message" or "Search".
                val isInformationalMetric = run {
                    val rawText = (node.text ?: rawLabel).trim()
                    if (node.isEditable || node.isCheckable || node.role.contains("BUTTON", ignoreCase = true) || (node.isClickable && rawText.length <= 4)) return@run false
                    val isMetricPattern = rawText.matches(Regex("""^[\d.,]+[KMBkmb]?\s*(posts?|views?|followers?|following|likes?|reels?|subscribers?)?$""", RegexOption.IGNORE_CASE)) ||
                        rawText.contains(Regex("""\b\d+([.,]\d+)?[KMBkmb]?\s*(views|likes|plays|reels)\b""", RegexOption.IGNORE_CASE))
                    val hasActionVerb = rawLower.contains(Regex("""\b(message|chat|send|search|call|follow|share|edit|open|view|play|buy|add|cart)\b"""))
                    val isSocialMediaGridTile = (rawLower.contains("reel by") || rawLower.contains("photo by") || rawLower.contains("video by")) && !hasActionVerb
                    (isMetricPattern && !hasActionVerb) || isSocialMediaGridTile
                }

                val isSearchAffordanceNode = !isAiPromptOrSendTrigger &&
                    (isTopSearchAffordance || rawLower.contains(Regex("\\bsearch\\b")) || resourceLower.contains(Regex("\\b(search|query|find)\\b")))

                // Brand-Agnostic Assistant / Bot Recognition:
                // Dedicated AI bots or chatbots (e.g. FABs, standalone bot shortcuts, generative query chips, or AI send buttons like 'ASK')
                // that are NOT search affordances and NOT editable input fields.
                val isAssistantChatbot = (!isSearchAffordanceNode && !node.isEditable && run {
                    val matchesGenericAiId = resourceLower.contains("assistant") || resourceLower.contains("chatbot") ||
                        resourceLower.contains("copilot") || resourceLower.contains("gemini") || resourceLower.contains("meta_ai") ||
                        resourceLower.contains("metaai") || resourceLower.contains("ai_fab") || resourceLower.contains("ai_chat") ||
                        resourceLower.contains("ai_input_send") || resourceLower.contains("ai_send")
                    val matchesAiText = rawLower.contains("message your assistant") ||
                        rawLower.contains("chat with ai") ||
                        rawLower.contains("meta ai") ||
                        rawLower.contains("galaxy ai") ||
                        rawLower.contains("copilot") ||
                        rawLower.contains("gemini") ||
                        rawLower.contains("apple intelligence") ||
                        rawLower == "assistant" ||
                        rawLower == "ask ai" ||
                        rawLower == "ask" ||
                        // Generative query prompt chips in search results (e.g. "Ask Meta AI 'ajay'", "Ask Gemini 'ajay'")
                        (rawLower.startsWith("ask ") && (rawLower.contains(" ai") || rawLower.contains("about") || rawLower.contains("'") || rawLower.contains("\"")))
                    matchesGenericAiId || matchesAiText || isAiPromptOrSendTrigger
                }) || isAiPromptOrSendTrigger
                val isGoalTargetingPerson = run {
                    val target = GoalArgumentResolver.resolveTargetEntity(goal)?.lowercase().orEmpty()
                    target.isNotBlank() && !target.contains(Regex("""\b(ai|meta|assistant|copilot|gemini|bot|group)\b"""))
                }
                val isAssistantDistractor = isAssistantChatbot && isGoalTargetingPerson

                // If user targets a specific person, a card whose label or snippet indicates it is merely a group association
                // (e.g. "Ajay is also in this group") is a distractor when targeting a person.
                val isGroupAssociationCard = (rawLower.contains("is also in this group") || rawLower.contains("is in this group")) &&
                    isGoalTargetingPerson

                // In Instagram/social SERP, do not let the top search edit container header hijack candidate clicks
                // when actual search results or user accounts are visible below it.
                val isSerpHeaderDistractor = resourceLower.contains("serp_journey_header_edit_tap_target") &&
                    uiState.nodes.any { it.resourceId?.contains("row_search_user") == true || it.resourceId?.contains("row_search_keyword") == true }

                val isParentClickable = node.parentIndex?.let { pIdx -> nodesByIndex[pIdx]?.isClickable == true } ?: false
                val isActionable = (node.isClickable || node.isEditable || (isParentClickable && !isChildVisualOrBadgeDecoration) || isTopSearchAffordance || isSendOrSubmitAction || isMenuAction) &&
                    !isContainer && !isChildVisualOrBadgeDecoration && !isInformationalMetric && !isAssistantDistractor && !isGroupAssociationCard && !isSerpHeaderDistractor
                val isIncompatibleNamedField = node.isEditable &&
                    (rawLower + " " + resourceLower).contains(Regex("\\b(name|title|label|subject)\\b")) &&
                    goal.contains(Regex("\\b\\d{1,2}[:.]\\d{2}\\s*(am|pm)?\\b", RegexOption.IGNORE_CASE)) &&
                    !goal.contains(Regex("\\b(named|name|title|label)\\s+['\"]?[^'\"]+", RegexOption.IGNORE_CASE))
                !isBlacklisted && !isSemanticallyBlacklisted && isActionable && hasLabel && !isIncompatibleNamedField
            }.sortedByDescending { node ->
                val rawLabel = effectiveLabel(node).lowercase()
                val resourceLabel = node.resourceId?.lowercase().orEmpty()
                val isTabControl = rawLabel.contains(Regex("\\b(tab|section|page|mode)\\b")) ||
                    resourceLabel.contains(Regex("\\b(tab|navigation_bar|bottom_navigation|nav_item|search_tab)\\b")) ||
                    node.role.contains("TAB", ignoreCase = true) ||
                    node.className?.contains("tab", ignoreCase = true) == true
                val isAiPromptOrSendTrigger = rawLabel in setOf("ask", "ask ai", "generate") ||
                    (rawLabel.startsWith("ask ") && !rawLabel.contains("search") && !rawLabel.contains("find")) ||
                    resourceLabel.contains("meta_ai") ||
                    resourceLabel.contains("metaai") ||
                    resourceLabel.contains("ai_send") ||
                    resourceLabel.contains("ai_input_send") ||
                    resourceLabel.contains("assistant_send") ||
                    (resourceLabel.contains("send_button") && (resourceLabel.contains("ai") || resourceLabel.contains("bot") || resourceLabel.contains("assistant")))

                val isSearchAffordance = !isAiPromptOrSendTrigger &&
                    (rawLabel.contains(Regex("\\bsearch\\b")) ||
                    resourceLabel.contains(Regex("\\b(search|query|find)\\b")) ||
                    node.role.equals("INPUT", ignoreCase = true) || node.isEditable) && (!node.isSelected || !isTabControl)
                val isModalDismissal = rawLabel.contains(Regex("\\b(close|dismiss|cancel|not now|skip)\\b")) ||
                    resourceLabel.contains(Regex("\\b(close|dismiss|cancel|skip)\\b"))
                val isSearchGoal = goal.contains(Regex("\\b(search|find|look for)\\b", RegexOption.IGNORE_CASE))

                val searchBoost = if (isSearchGoal && isSearchAffordance) 25 else 0
                val dismissBoost = if (isModalDismissal) 30 else 0

                val isMessagingGoal = goal.contains(Regex("\\b(message|chat|dm|send|text)\\b", RegexOption.IGNORE_CASE))
                val isChatInput = (node.isEditable || node.role.equals("INPUT", ignoreCase = true)) &&
                    (rawLabel.contains(Regex("\\b(message|type a message|chat|text|enter)\\b")) ||
                     resourceLabel.contains(Regex("\\b(entry|message_text|chat_input|input_text)\\b")))
                val chatInputBoost = if (isMessagingGoal && isChatInput) 35 else 0

                val isGenericAiBot = isAiPromptOrSendTrigger || (!isSearchAffordance && !node.isEditable && run {
                    val matchesGenericAiId = resourceLabel.contains("assistant") || resourceLabel.contains("chatbot") ||
                        resourceLabel.contains("copilot") || resourceLabel.contains("gemini") || resourceLabel.contains("meta_ai") ||
                        resourceLabel.contains("metaai") || resourceLabel.contains("ai_fab") || resourceLabel.contains("ai_chat") ||
                        resourceLabel.contains("ai_input_send") || resourceLabel.contains("ai_send")
                    val matchesAiText = rawLabel.contains("message your assistant") ||
                        rawLabel.contains("chat with ai") ||
                        rawLabel.contains("meta ai") ||
                        rawLabel.contains("galaxy ai") ||
                        rawLabel.contains("copilot") ||
                        rawLabel.contains("gemini") ||
                        rawLabel.contains("apple intelligence") ||
                        rawLabel == "assistant" ||
                        rawLabel == "ask ai" ||
                        rawLabel == "ask" ||
                        (rawLabel.startsWith("ask ") && (rawLabel.contains(" ai") || rawLabel.contains("about") || rawLabel.contains("'") || rawLabel.contains("\"")))
                    matchesGenericAiId || matchesAiText
                })
                val isMessageAffordance = (rawLabel.contains(Regex("\\b(message|chat|send message|direct message)\\b")) ||
                    resourceLabel.contains(Regex("\\b(message|chat|direct|action_bar_send)\\b"))) && !isTabControl && !isGenericAiBot
                val messageBoost = if (isMessagingGoal && isMessageAffordance) 30 else 0

                val isAiDistractor = isGenericAiBot &&
                    !goal.contains(Regex("""\b(ai|assistant|chatbot|copilot|gemini|meta)\b""", RegexOption.IGNORE_CASE))
                val aiDistractorPenalty = if (isAiDistractor) -50 else 0

                val goalTokens = goal.lowercase().split(Regex("[^a-z0-9]+"))
                    .filter { it.length >= 3 && it !in setOf("open", "the", "and", "then", "for") }
                val goalOverlap = if (isTabControl && node.isSelected) 0 else goalTokens.count { token ->
                    rawLabel.contains(token) || resourceLabel.contains(token)
                }

                // If chat input field is active on screen, don't let top action bar title steal focus from message input
                val isHeaderTitle = node.bounds.top < 350 && !rawLabel.contains("back") && !rawLabel.contains("navigate") && !node.isEditable
                val effectiveGoalOverlap = if (chatInputBoost > 0 && isHeaderTitle) 0 else goalOverlap

                val isRecentTextEntry = stepHistory.takeLast(1).any { it.action is AgentAction.TypeText }
                val isSuggestion = rawLabel.contains("suggestion") || resourceLabel.contains("suggestion") ||
                    node.className?.contains("suggestion", ignoreCase = true) == true ||
                    (isRecentTextEntry && goalOverlap > 0 && node.isClickable && !node.isEditable)
                val suggestionBoost = if (isRecentTextEntry && isSuggestion) 20 else 0

                // Direct Target Grounding:
                // When goal specifies a target entity (contact, list item, or navigation element),
                // matching affordances receive high target alignment boost.
                val targetEntity = GoalArgumentResolver.resolveTargetEntity(goal)?.lowercase().orEmpty()
                val isDirectTargetMatch = !node.isEditable && targetEntity.isNotBlank() && run {
                    val candTokens = (rawLabel + " " + resourceLabel).split(Regex("[^a-z0-9]+")).filter { it.length >= 2 }
                    val targetTokens = targetEntity.split(Regex("[^a-z0-9]+")).filter { it.length >= 2 }
                    targetTokens.any { t ->
                        candTokens.any { c ->
                            c == t || c.startsWith(t) || t.startsWith(c) ||
                            (t.endsWith("s") && c == t.dropLast(1)) ||
                            (c.endsWith("s") && t == c.dropLast(1))
                        }
                    }
                }
                // Form Field Grounding:
                // When goal specifies a form field or target (e.g. "USN", "password", "email"),
                // editable inputs whose label, hint or resourceId matches that token receive strong boost.
                val isDirectFormTargetMatch = (node.isEditable || node.role.equals("INPUT", ignoreCase = true)) && targetEntity.isNotBlank() && run {
                    val candTokens = (rawLabel + " " + resourceLabel).split(Regex("[^a-z0-9]+")).filter { it.length >= 2 }
                    val targetTokens = targetEntity.split(Regex("[^a-z0-9]+")).filter { it.length >= 2 }
                    targetTokens.any { t -> candTokens.any { c -> c == t || c.startsWith(t) || t.startsWith(c) } }
                }
                val formTargetMatchBoost = if (isDirectFormTargetMatch) 35 else 0

                val isNavigationTargetMatch = isTabControl && isDirectTargetMatch && !node.isSelected
                val navigationMatchBoost = if (isNavigationTargetMatch) 45 else 0

                val targetMatchBoost = if (isDirectTargetMatch && !isTabControl && !isHeaderTitle && currentPhase != TaskPhase.COMMIT) 30 else 0

                val semanticPriority = when {
                    node.isEditable || node.role.equals("INPUT", ignoreCase = true) -> if (currentPhase == TaskPhase.SELECT_TARGET) 0 else 6
                    isTabControl -> if (node.isSelected) 0 else 3
                    node.isCheckable -> 3
                    node.role.contains("BUTTON", ignoreCase = true) -> 3
                    node.isScrollable -> 2
                    else -> 0
                }
                val labelPriority = when {
                    rawLabel.contains(Regex("\\b(search|submit|confirm|save|next|done|send|menu|options|timer|stopwatch|alarm|world clock)\\b")) && (!node.isSelected || !isTabControl) -> 3
                    rawLabel.isNotBlank() -> 1
                    else -> 0
                }
                val isPopulatedSearchInSelectTarget = currentPhase == TaskPhase.SELECT_TARGET && 
                    (node.isEditable || node.role.equals("INPUT", ignoreCase = true)) && 
                    (rawLabel.contains("search", ignoreCase = true) || resourceLabel.contains("search", ignoreCase = true))
                val populatedSearchPenalty = if (isPopulatedSearchInSelectTarget) -25 else 0

                val isCommitSendAffordance = currentPhase == TaskPhase.COMMIT && (
                    rawLabel.contains(Regex("\\b(send|submit|done|enter|go|search)\\b", RegexOption.IGNORE_CASE)) ||
                    resourceLabel.contains(Regex("send|submit|done|search"))
                )
                val commitBoost = if (isCommitSendAffordance) 40 else 0

                searchBoost + dismissBoost + suggestionBoost + chatInputBoost + messageBoost + commitBoost + aiDistractorPenalty + targetMatchBoost + navigationMatchBoost + formTargetMatchBoost + populatedSearchPenalty + effectiveGoalOverlap * 5 + labelPriority * 3 + semanticPriority
            }.take(MAX_CANDIDATE_TOOLS)
        }

        // Senior Step 7: Phase-Conditioned Candidate Pre-Filter with Safety Fallback Expansion
        val phaseFilteredNodes = when (currentPhase) {
            TaskPhase.FIND_TARGET -> {
                val searchNodes = visibleNodes.filter { n ->
                    val res = n.resourceId?.lowercase().orEmpty()
                    val textLower = n.text?.lowercase().orEmpty()
                    val descLower = n.contentDescription?.lowercase().orEmpty()
                    val lbl = effectiveLabel(n).lowercase()
                    (res.contains("search") || textLower.contains("search") || descLower.contains("search") ||
                        lbl.contains("search") || res.contains("action_bar_search") || res.contains("menu_search") || res.contains("find")) &&
                        (n.isEditable || n.isClickable || n.role in setOf("INPUT", "BUTTON", "VIEW"))
                }
                if (searchNodes.isNotEmpty()) {
                    val targetEntity = GoalArgumentResolver.resolveTargetEntity(goal)?.lowercase().orEmpty()
                    val filtered = visibleNodes.filter { n ->
                        val res = n.resourceId?.lowercase().orEmpty()
                        val isContactCard = res.contains("contact") || res.contains("conversation") || res.contains("row_inbox")
                        if (isContactCard) {
                            if (targetEntity.isNotBlank()) {
                                val lbl = effectiveLabel(n).lowercase()
                                lbl.contains(targetEntity)
                            } else false
                        } else {
                            true
                        }
                    }
                    if (filtered.isNotEmpty()) filtered else visibleNodes
                } else {
                    visibleNodes
                }
            }
            TaskPhase.FOCUS_INPUT -> {
                val inputNodes = visibleNodes.filter { it.isEditable || it.role.equals("INPUT", true) || it.className?.contains("EditText", true) == true }
                if (inputNodes.size >= 1) inputNodes else visibleNodes
            }
            TaskPhase.ENTER_ARGUMENT -> {
                val inputNodes = visibleNodes.filter { it.isEditable || it.role.equals("INPUT", true) || it.className?.contains("EditText", true) == true }
                val otherNodes = visibleNodes.filter { n ->
                    !n.isEditable && !n.role.equals("INPUT", true) && n.className?.contains("EditText", true) != true && run {
                        val res = n.resourceId?.lowercase().orEmpty()
                        val textLower = n.text?.lowercase().orEmpty()
                        val descLower = n.contentDescription?.lowercase().orEmpty()
                        val isMessageBubbleOrLink = res.contains("message_text") || res.contains("link_preview") ||
                            res.contains("snippet") || res.contains("url") || res.contains("web_page") ||
                            res.contains("thumb_view") || res.contains("caption") || res.contains("media_preview") ||
                            res.contains("copy_link")
                        val isSearchHistoryOrSuggestion = (res.contains("suggestion") || res.contains("recent") || res.contains("history") ||
                            textLower.contains("recent search") || descLower.contains("recent search") ||
                            textLower.contains("suggested search") || descLower.contains("suggested search") ||
                            textLower.contains("trending search") || descLower.contains("trending search") ||
                            textLower.contains("trending searches") || descLower.contains("trending searches") ||
                            textLower.contains("popular searches") || descLower.contains("popular searches") ||
                            textLower.contains("search history") ||
                            (textLower.startsWith("search ") && textLower.length > 15) ||
                            (descLower.startsWith("search ") && descLower.length > 15)) && run {
                                val label = "${n.text.orEmpty()} ${n.contentDescription.orEmpty()}".lowercase()
                                val goalTokens = goal.lowercase().split(Regex("[^a-z0-9]+"))
                                    .filter { it.length >= 3 && it !in setOf("open", "the", "and", "then", "for", "search", "find", "google") }
                                goalTokens.none { label.contains(it) }
                            }
                        !isMessageBubbleOrLink && !isSearchHistoryOrSuggestion
                    }
                }
                val hasRecentText = stepHistory.any { it.action is AgentAction.TypeText }
                val commitNodes = visibleNodes.filter {
                    val lbl = effectiveLabel(it).lowercase()
                    val res = it.resourceId?.lowercase().orEmpty()
                    val isSendOrSubmit = lbl.contains(Regex("\\b(send|submit|done|save|enter|go|next)\\b")) || res.contains(Regex("send|submit|done|save"))
                    val isInteractive = it.isClickable || it.role.contains("BUTTON", true) || it.role.equals("IMAGE", true) || it.role.equals("VIEW", true)
                    isSendOrSubmit && isInteractive
                }
                if (hasRecentText && commitNodes.isNotEmpty()) {
                    (commitNodes + inputNodes + otherNodes).distinctBy { it.index }
                } else if (inputNodes.isNotEmpty()) {
                    (inputNodes + otherNodes).distinctBy { it.index }
                } else {
                    visibleNodes
                }
            }
            TaskPhase.COMMIT -> {
                val commitNodes = visibleNodes.filter {
                    val lbl = effectiveLabel(it).lowercase()
                    val res = it.resourceId?.lowercase().orEmpty()
                    val isSendOrSubmit = lbl.contains(Regex("\\b(send|submit|done|enter|go|search)\\b")) || res.contains(Regex("send|submit|done|search")) ||
                        (it.bounds.bottom > 1300 && it.bounds.right > 700 && (res.contains("button") || it.role in setOf("BUTTON", "IMAGE", "VIEW")))
                    val isInteractive = it.isClickable || it.role.contains("BUTTON", true) || it.role.equals("IMAGE", true) || it.role.equals("VIEW", true)
                    isSendOrSubmit && isInteractive
                }
                if (commitNodes.isNotEmpty()) commitNodes else visibleNodes
            }
            else -> visibleNodes
        }

        for (node in phaseFilteredNodes) {
            candidateMap[node.index] = node
            val rawLabel = effectiveLabel(node)
                .ifBlank { node.resourceId?.substringAfterLast("/") ?: node.semanticLabel }

            val cleanSlug = when {
                node.text == "=" || node.contentDescription?.contains("Calculation", ignoreCase = true) == true || node.contentDescription?.contains("Equal", ignoreCase = true) == true -> "equal"
                node.text == "+" || node.contentDescription?.contains("Plus", ignoreCase = true) == true -> "plus"
                node.text == "−" || node.text == "-" || node.contentDescription?.contains("Minus", ignoreCase = true) == true -> "minus"
                node.text == "×" || node.text == "*" || node.contentDescription?.contains("Multiplication", ignoreCase = true) == true -> "mul"
                node.text == "÷" || node.text == "/" || node.contentDescription?.contains("Division", ignoreCase = true) == true -> "div"
                node.text == "%" || node.contentDescription?.contains("Percentage", ignoreCase = true) == true -> "percent"
                node.text == "." || node.contentDescription?.contains("Decimal", ignoreCase = true) == true -> "dot"
                rawLabel.equals("send", ignoreCase = true) || node.resourceId?.lowercase()?.contains("send") == true -> "send"
                rawLabel.contains("menu", ignoreCase = true) || node.resourceId?.lowercase()?.contains("menu") == true -> "menu"
                else -> (node.text?.takeIf { it.isNotBlank() } ?: rawLabel).lowercase()
                    .replace(Regex("[^a-z0-9]+"), "_")
                    .trim('_')
                    .take(24)
                    .ifBlank { "item" }
            }

            val cleanDesc = "Tap [${node.role}] '$rawLabel' on screen (element #${node.index})"
            val semanticTags = deriveSemanticTags(node, rawLabel, goal)
            val candId = "c_${node.index}"
            val nodeIdentity = node.resourceId?.takeIf { it.isNotBlank() }
                ?: "node_${node.index}_${node.bounds.left}_${node.bounds.top}"

            val isFormulaOrDisplay = node.resourceId?.contains("formula", ignoreCase = true) == true ||
                node.resourceId?.contains("display", ignoreCase = true) == true ||
                node.role.equals("DISPLAY", true)
            val isInputNode = !isFormulaOrDisplay && (node.isEditable || node.role.equals("INPUT", true) || node.className?.contains("EditText", true) == true)
            if (isInputNode) {
                val actionCand = ActionCandidate(
                    candidateId = candId,
                    actionType = "type_text",
                    nodeIndex = node.index,
                    nodeIdentity = nodeIdentity,
                    semanticLabel = rawLabel,
                    role = node.role.lowercase(),
                    node = node
                )
                actionCandidateMap[candId] = actionCand
                actionCandidateMap["type_${cleanSlug}_${node.index}"] = actionCand

                // TypeText Tool
                val typeTool = JSONObject().apply {
                    put("name", "type_${cleanSlug}_${node.index}")
                    put("candidate_id", candId)
                    put("target_node_index", node.index)
                    put("node_identity", nodeIdentity)
                    put("semantic_label", rawLabel)
                    put("bounds", JSONObject().apply {
                        put("left", node.bounds.left)
                        put("top", node.bounds.top)
                        put("right", node.bounds.right)
                        put("bottom", node.bounds.bottom)
                    })
                    put("text", node.text ?: "")
                    put("content_description", node.contentDescription ?: "")
                    put("action_type", "type_text")
                    put("role", "input")
                    put("editable", true)
                    put("clickable", node.isClickable)
                    put("selected", node.isSelected)
                    put("description", "Type text into [${node.role}] '$rawLabel' on screen (element #${node.index}); semantic role: ${semanticTags.joinToString(", ")}")
                    put("semantic_tags", JSONArray(semanticTags))
                    put("parameters", JSONObject().apply {
                        put("type", "object")
                        put("properties", JSONObject().apply {
                            put("text", JSONObject().put("type", "string").put("description", "Text or numbers to type"))
                        })
                        put("required", JSONArray().put("text"))
                    })
                }
                toolsArray.put(typeTool)
            } else {
                val actionCand = ActionCandidate(
                    candidateId = candId,
                    actionType = "tap",
                    nodeIndex = node.index,
                    nodeIdentity = nodeIdentity,
                    semanticLabel = rawLabel,
                    role = node.role.lowercase(),
                    node = node
                )
                actionCandidateMap[candId] = actionCand
                actionCandidateMap["tap_${cleanSlug}_${node.index}"] = actionCand

                // Tap Tool (Clicking requires zero parameters)
                val tapTool = JSONObject().apply {
                    put("name", "tap_${cleanSlug}_${node.index}")
                    put("candidate_id", candId)
                    put("target_node_index", node.index)
                    put("node_identity", nodeIdentity)
                    put("semantic_label", rawLabel)
                    put("bounds", JSONObject().apply {
                        put("left", node.bounds.left)
                        put("top", node.bounds.top)
                        put("right", node.bounds.right)
                        put("bottom", node.bounds.bottom)
                    })
                    put("text", node.text ?: "")
                    put("content_description", node.contentDescription ?: "")
                    put("action_type", "tap")
                    put("role", node.role.lowercase())
                    put("clickable", node.isClickable)
                    put("editable", node.isEditable)
                    put("scrollable", node.isScrollable)
                    put("selected", node.isSelected)
                    put("description", "$cleanDesc; semantic role: ${semanticTags.joinToString(", ")}")
                    put("semantic_tags", JSONArray(semanticTags))
                    put("parameters", JSONObject().apply {
                        put("type", "object")
                        put("properties", JSONObject())
                    })
                }
                toolsArray.put(tapTool)
            }
        }

        // 2b. Targeted scroll tools for value pickers and other scrollable controls.
        // This enables generic wheel/selector adjustment without app-specific coordinates.
        val scrollableNodes = if (uiState.packageName == "com.zipper.app") {
            emptyList()
        } else {
            uiState.nodes.filter { node ->
                node.index !in blacklistedNodeIndices && (node.isScrollable || isPickerLike(node))
            }.take(8)
        }

        for (node in scrollableNodes) {
            candidateMap[node.index] = node
            val rawLabel = listOfNotNull(node.text, node.contentDescription)
                .filter { it.isNotBlank() }
                .distinct()
                .joinToString(" ")
                .ifBlank { node.semanticLabel }
            val scrollSlug = rawLabel.lowercase()
                .replace(Regex("[^a-z0-9]+"), "_")
                .trim('_')
                .take(20)
                .ifBlank { "scrollable" }

            val tags = deriveSemanticTags(node, rawLabel, goal).toMutableList()
            if (!tags.contains("scroll_control")) tags.add("scroll_control")
            if (!tags.contains("value_picker_control") &&
                rawLabel.contains(Regex("hour|minute|am|pm|time", RegexOption.IGNORE_CASE))
            ) {
                tags.add("value_picker_control")
            }

            val candIdDown = "c_scroll_down_${node.index}"
            val candIdUp = "c_scroll_up_${node.index}"
            val nodeIdentity = node.resourceId?.takeIf { it.isNotBlank() }
                ?: "node_${node.index}_${node.bounds.left}_${node.bounds.top}"

            val scrollDownCand = ActionCandidate(
                candidateId = candIdDown,
                actionType = "scroll",
                nodeIndex = node.index,
                nodeIdentity = nodeIdentity,
                semanticLabel = "scroll down $rawLabel",
                role = node.role.lowercase(),
                node = node
            )
            val scrollUpCand = ActionCandidate(
                candidateId = candIdUp,
                actionType = "scroll",
                nodeIndex = node.index,
                nodeIdentity = nodeIdentity,
                semanticLabel = "scroll up $rawLabel",
                role = node.role.lowercase(),
                node = node
            )
            actionCandidateMap[candIdDown] = scrollDownCand
            actionCandidateMap["scroll_down_${scrollSlug}_${node.index}"] = scrollDownCand
            actionCandidateMap[candIdUp] = scrollUpCand
            actionCandidateMap["scroll_up_${scrollSlug}_${node.index}"] = scrollUpCand

            toolsArray.put(JSONObject().apply {
                put("name", "scroll_down_${scrollSlug}_${node.index}")
                put("candidate_id", candIdDown)
                put("target_node_index", node.index)
                put("node_identity", nodeIdentity)
                put("semantic_label", "scroll down $rawLabel")
                put("bounds", JSONObject().apply {
                    put("left", node.bounds.left)
                    put("top", node.bounds.top)
                    put("right", node.bounds.right)
                    put("bottom", node.bounds.bottom)
                })
                put("text", node.text ?: "")
                put("content_description", node.contentDescription ?: "")
                put("description", "Scroll down [${node.role}] '$rawLabel' (element #${node.index}); capabilities: ${tags.joinToString(", ")}")
                put("semantic_tags", JSONArray(tags))
                put("parameters", JSONObject().put("type", "object"))
            })

            toolsArray.put(JSONObject().apply {
                put("name", "scroll_up_${scrollSlug}_${node.index}")
                put("candidate_id", candIdUp)
                put("target_node_index", node.index)
                put("node_identity", nodeIdentity)
                put("semantic_label", "scroll up $rawLabel")
                put("bounds", JSONObject().apply {
                    put("left", node.bounds.left)
                    put("top", node.bounds.top)
                    put("right", node.bounds.right)
                    put("bottom", node.bounds.bottom)
                })
                put("text", node.text ?: "")
                put("content_description", node.contentDescription ?: "")
                put("description", "Scroll up [${node.role}] '$rawLabel' (element #${node.index}); capabilities: ${tags.joinToString(", ")}")
                put("semantic_tags", JSONArray(tags))
                put("parameters", JSONObject().put("type", "object"))
            })
        }

        // 3. Navigation / System Primitives
        if (!isKarvishLauncher) {
            val scrollDownSystem = ActionCandidate("c_scroll_down", "scroll", null, "system:scroll_down", "scroll down", "system")
            actionCandidateMap["c_scroll_down"] = scrollDownSystem
            actionCandidateMap["scroll_down"] = scrollDownSystem
            toolsArray.put(JSONObject().apply {
                put("name", "scroll_down")
                put("candidate_id", "c_scroll_down")
                put("action_type", "scroll")
                put("role", "system")
                put("node_identity", "system:scroll_down")
                put("semantic_label", "scroll down")
                put("description", "Scroll down to reveal elements currently below the screen")
                put("semantic_tags", JSONArray().put("scroll_control"))
                put("parameters", JSONObject().put("type", "object"))
            })

            val scrollUpSystem = ActionCandidate("c_scroll_up", "scroll", null, "system:scroll_up", "scroll up", "system")
            actionCandidateMap["c_scroll_up"] = scrollUpSystem
            actionCandidateMap["scroll_up"] = scrollUpSystem
            toolsArray.put(JSONObject().apply {
                put("name", "scroll_up")
                put("candidate_id", "c_scroll_up")
                put("action_type", "scroll")
                put("role", "system")
                put("node_identity", "system:scroll_up")
                put("semantic_label", "scroll up")
                put("description", "Scroll up to reveal elements currently above the screen")
                put("semantic_tags", JSONArray().put("scroll_control"))
                put("parameters", JSONObject().put("type", "object"))
            })

            val backSystem = ActionCandidate("c_press_back", "back", null, "system:press_back", "back", "system")
            actionCandidateMap["c_press_back"] = backSystem
            actionCandidateMap["press_back"] = backSystem
            toolsArray.put(JSONObject().apply {
                put("name", "press_back")
                put("candidate_id", "c_press_back")
                put("action_type", "back")
                put("role", "system")
                put("node_identity", "system:press_back")
                put("semantic_label", "back")
                put("description", "Navigate back to the previous screen or dismiss dialog")
                put("semantic_tags", JSONArray().put("navigation_control"))
                put("parameters", JSONObject().put("type", "object"))
            })

            // 3b. Submit / IME Enter action: Expose as an available platform affordance whenever
            // the Android accessibility hierarchy reports an editable element, a focused input field,
            // or when text was recently entered into the active window.
            val hasActiveEditable = visibleNodes.any { it.isEditable } || uiState.nodes.any { it.isEditable }
            val hasFocusedInput = visibleNodes.any { it.isFocused && it.role.equals("INPUT", ignoreCase = true) }
            val hasRecentTextEntry = stepHistory.takeLast(1).any { it.action is AgentAction.TypeText }
            val hasImeAffordance = hasActiveEditable || hasFocusedInput || hasRecentTextEntry
            if (hasImeAffordance) {
                val submitCand = ActionCandidate("c_submit", "submit", null, "system:submit", "submit focused field", "system")
                actionCandidateMap["c_submit"] = submitCand
                actionCandidateMap["submit"] = submitCand
                toolsArray.put(JSONObject().apply {
                    put("name", "submit")
                    put("candidate_id", "c_submit")
                    put("action_type", "submit")
                    put("role", "system")
                    put("node_identity", "system:submit")
                    put("semantic_label", "submit focused field")
                    put("description", "Submit focused field or search query via IME Enter action")
                    put("semantic_tags", JSONArray().put("input_control").put("submit_action"))
                    put("bounds", JSONObject().apply {
                        put("left", 900)
                        put("top", 2200)
                        put("right", 1080)
                        put("bottom", 2340)
                    })
                    put("parameters", JSONObject().put("type", "object"))
                })
            }
        }

        // 4. Terminal Goal Verification Tools
        val inTargetApp = uiState.packageName != null && uiState.packageName != "com.zipper.app"
        if (inTargetApp) {
            val successfulActions = stepHistory
                .filter { it.outcome == StepOutcome.SUCCESS }
                .takeLast(3)
                .map {
                    val desc = it.actionDescription
                        .replace('(', ' ')
                        .replace(')', ' ')
                        .replace('[', ' ')
                        .replace(']', ' ')
                        .trim()
                    "$desc:SUCCESS"
                }
            val historyNotice = if (successfulActions.isNotEmpty()) {
                " (history: ${successfulActions.joinToString("; ")})"
            } else ""

            val completeCand = ActionCandidate("c_complete_goal", "complete", null, "system:complete_goal", "complete", "system")
            actionCandidateMap["c_complete_goal"] = completeCand
            actionCandidateMap["complete_goal"] = completeCand
            toolsArray.put(JSONObject().apply {
                put("name", "complete_goal")
                put("candidate_id", "c_complete_goal")
                put("node_identity", "system:complete_goal")
                put("semantic_label", "complete")
                put("description", "Declare the task complete when the goal has been visibly verified and achieved$historyNotice")
                put("semantic_tags", JSONArray().put("terminal_success"))
                put("parameters", JSONObject().apply {
                    put("type", "object")
                    put("properties", JSONObject().apply {
                        put("summary", JSONObject().put("type", "string").put("description", "Explanation of how the goal was achieved"))
                    })
                    put("required", JSONArray().put("summary"))
                })
            })
        }

        val failCand = ActionCandidate("c_fail_goal", "fail", null, "system:fail_goal", "fail", "system")
        actionCandidateMap["c_fail_goal"] = failCand
        actionCandidateMap["fail_goal"] = failCand
        toolsArray.put(JSONObject().apply {
            put("name", "fail_goal")
            put("candidate_id", "c_fail_goal")
            put("node_identity", "system:fail_goal")
            put("semantic_label", "fail")
            put("description", "Call this tool when the goal is impossible, blocked, or not supported")
            put("semantic_tags", JSONArray().put("terminal_failure"))
            put("parameters", JSONObject().apply {
                put("type", "object")
                put("properties", JSONObject().apply {
                    put("reason", JSONObject().put("type", "string").put("description", "Why the goal cannot be achieved"))
                })
                put("required", JSONArray().put("reason"))
            })
        })

        val summary = "Generated ${toolsArray.length()} tools (${visibleNodes.size} screen elements, ${blacklistedNodeIndices.size} blacklisted)"
        return GeneratedTools(toolsArray, candidateMap, summary, actionCandidateMap)
    }

    private fun isPickerLike(node: DistilledNode): Boolean {
        val label = listOfNotNull(node.text, node.contentDescription, node.resourceId)
            .joinToString(" ").lowercase()
        return label.contains(Regex("\\b(hour|minute|ampm|am|pm|period|time|picker|wheel)\\b"))
    }
}
