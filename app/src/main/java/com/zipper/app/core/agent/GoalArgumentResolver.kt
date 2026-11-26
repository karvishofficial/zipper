package com.zipper.app.core.agent

import com.zipper.app.core.model.DistilledNode

/** Resolves user-provided values for generic editable UI targets. */
object GoalArgumentResolver {
    private fun cleanGoal(goal: String): String {
        return goal.replace(Regex("\\[state:[^\\]]*\\]", RegexOption.IGNORE_CASE), "")
            .replace(Regex("\\[history:[^\\]]*\\]", RegexOption.IGNORE_CASE), "")
            .replace(Regex("\\(previous action failed:[^)]*\\)", RegexOption.IGNORE_CASE), "")
            .trim()
    }

    fun resolveText(goal: String, targetDescription: String, targetNode: DistilledNode? = null): String {
        val normalizedGoal = cleanGoal(goal)
        val target = listOf(
            targetDescription,
            targetNode?.text,
            targetNode?.contentDescription,
            targetNode?.resourceId
        ).filterNotNull().joinToString(" ").lowercase()
        val targetTokens = target.split(Regex("[^a-z0-9]+"))
            .filter { it.isNotBlank() }
            .toSet()

        val isSearchField = target.contains("search") ||
            target.contains("find") ||
            target.contains("query") ||
            targetNode?.resourceId?.contains("search") == true

        val isChatComposer = !isSearchField && (
            target.contains("message") ||
            target.contains("composer") ||
            target.contains("type a message") ||
            target.contains("send a message") ||
            (target.contains("entry") && !target.contains("search")) ||
            (target.contains("chat") && !target.contains("search") && !target.contains("find")) ||
            targetNode?.resourceId?.contains("composer") == true ||
            (targetNode?.resourceId?.contains("entry") == true && targetNode?.resourceId?.contains("search") != true)
        )

        if (isChatComposer) {
            val msg = resolveMessageBody(normalizedGoal)
            if (!msg.isNullOrBlank()) {
                return msg
            }
        }

        val isNamedField = targetTokens.any { it in setOf("name", "title", "label", "subject") } &&
            targetTokens.none { it in setOf("hour", "minute", "min", "meridiem", "period", "time") }

        val alarmTime = Regex("\\b(\\d{1,2})[:.](\\d{2})\\s*(am|pm)?\\b", RegexOption.IGNORE_CASE)
            .find(normalizedGoal)
        if (alarmTime != null) {
            if (isNamedField) {
                val explicitName = Regex("\\b(?:named|name(?:d)?|label(?:ed)?)\\s+['\"]?([^'\"]+?)['\"]?(?:\\s+for\\s+|$)", RegexOption.IGNORE_CASE)
                    .find(normalizedGoal)?.groupValues?.getOrNull(1)?.trim()
                return explicitName ?: ""
            }
            val hour = alarmTime.groupValues[1]
            val minute = alarmTime.groupValues[2]
            val meridiem = alarmTime.groupValues[3]
            return when {
                "minute" in targetTokens || "min" in targetTokens -> minute
                "hour" in targetTokens -> hour
                "am" in targetTokens || "pm" in targetTokens || "meridiem" in targetTokens -> meridiem.uppercase()
                meridiem.isNotBlank() -> "$hour:$minute ${meridiem.uppercase()}"
                else -> "$hour:$minute"
            }
        }

        // 1. Messaging Pattern: 'send/say <msg> to <recipient> [on/in app]'
        val sendMatch = Regex("\\b(?:send|say)\\s+(.+?)\\s+to\\s+(.+?)(?:\\s+(?:on|in|inside)\\s+[A-Za-z0-9_\\s]+)?$", RegexOption.IGNORE_CASE)
            .find(normalizedGoal)
        val isContactOrSearchTarget = target.contains("contact") ||
            target.contains("recipient") ||
            target.contains("search") ||
            target.contains("query") ||
            target.contains("find")

        if (sendMatch != null) {
            val msg = sendMatch.groupValues[1].trim().trim('"', '\'')
            var recipient = sendMatch.groupValues[2].trim().trim('"', '\'')
            recipient = recipient.replace(Regex("\\s+(?:on|in|inside)\\s+[A-Za-z0-9_\\s]+$", RegexOption.IGNORE_CASE), "").trim()
            return if (isContactOrSearchTarget) recipient else msg
        }

        // 2. Messaging Pattern: 'message/text/tell/chat with <recipient> [saying/that] <msg> [on/in app]'
        val messageMatch = Regex("\\b(?:message|text|tell|chat with)\\s+([A-Za-z0-9_]+)\\s+(?:saying\\s+|that\\s+)?(.+?)(?:\\s+(?:on|in|inside)\\s+[A-Za-z0-9_\\s]+)?$", RegexOption.IGNORE_CASE)
            .find(normalizedGoal)
        if (messageMatch != null) {
            val recipient = messageMatch.groupValues[1].trim().trim('"', '\'')
            var msg = messageMatch.groupValues[2].trim().trim('"', '\'')
            msg = msg.replace(Regex("\\s+(?:on|in|inside)\\s+[A-Za-z0-9_\\s]+$", RegexOption.IGNORE_CASE), "").trim()
            return if (isContactOrSearchTarget) recipient else msg
        }

        // 3. Compound Pattern: 'search <recipient> [on/in app] and (then)? message/send/text [him/her/them] <msg>'
        val compoundMatch = Regex("\\b(?:search|find)\\s+([A-Za-z0-9_]+)(?:\\s+(?:on|in|inside)\\s+[A-Za-z0-9_]+)?\\s+(?:and|then)?\\s*(?:message|send|text)\\s+(?:(?:him|her|them)\\s+)?['\"]?([^'\"\\n]+?)['\"]?$", RegexOption.IGNORE_CASE)
            .find(normalizedGoal)
        if (compoundMatch != null) {
            val recipient = compoundMatch.groupValues[1].trim()
            val msg = compoundMatch.groupValues[2].trim()
            return if (isContactOrSearchTarget) recipient else msg
        }

        val comparative = ProductExplorationEngine.detectComparativeIntent(normalizedGoal)
        if (comparative != null && isSearchField) {
            return comparative.targetEntity
        }

        val searchMatch = Regex("\\b(?:search|find|look for)\\s+(?:for\\s+)?(.+?)(?:\\s+(?:in|on|inside|within)\\s+[A-Za-z0-9_\\s]+|\\s+and\\s+|\\s+then\\s+|$)", RegexOption.IGNORE_CASE)
            .find(normalizedGoal)
        if (searchMatch != null) {
            var term = searchMatch.groupValues[1].trim().trim(',', '.', '"', '\'')
            term = term.replace(Regex("\\s+(?:in|on|inside|within)\\s+[A-Za-z0-9_\\s]+$", RegexOption.IGNORE_CASE), "").trim()
            if (term.isNotBlank()) return term
        }

        // 4. Form Filling Pattern: 'fill/type/enter/input [the] <field> with <value>' (e.g. 'fill the USN box with 1MS21CS001')
        val fillWithMatch = Regex("\\b(?:fill|enter|type|input|write|put)\\s+(?:the\\s+)?(.+?)\\s+with\\s+['\"]?([^'\"\\n]+?)['\"]?$", RegexOption.IGNORE_CASE)
            .find(normalizedGoal)
        if (fillWithMatch != null) {
            val value = fillWithMatch.groupValues[2].trim().trim('"', '\'')
            if (value.isNotBlank()) return value
        }

        // 5. Form Filling Pattern: 'fill/type/enter/input/put <value> into/in [the] <field>' (e.g. 'put 1MS21CS001 into USN box')
        val fillIntoMatch = Regex("\\b(?:fill|enter|type|input|write|put)\\s+['\"]?([^'\"\\n]+?)['\"]?\\s+(?:into|in)\\s+(?:the\\s+)?(.+?)$", RegexOption.IGNORE_CASE)
            .find(normalizedGoal)
        if (fillIntoMatch != null) {
            val value = fillIntoMatch.groupValues[1].trim().trim('"', '\'')
            if (value.isNotBlank() && value.lowercase() !in setOf("the", "a", "an", "this", "my", "your")) return value
        }

        // 6. Direct fill with: 'fill with <value>'
        val simpleFillWith = Regex("\\b(?:fill|type|enter|input|write)\\s+with\\s+['\"]?([^'\"\\n]+?)['\"]?$", RegexOption.IGNORE_CASE)
            .find(normalizedGoal)
        if (simpleFillWith != null) {
            val value = simpleFillWith.groupValues[1].trim().trim('"', '\'')
            if (value.isNotBlank()) return value
        }

        val inputMatch = Regex("\\b(?:type|enter|input|write)\\s+(.+?)(?:\\s+into\\s+|\\s+in\\s+|$)", RegexOption.IGNORE_CASE).find(normalizedGoal)
        return inputMatch?.groupValues?.get(1)?.trim()?.trim('"', '\'') ?: ""
    }

    /** Extracts explicit message body from compound or direct messaging goals. */
    fun resolveMessageBody(goal: String): String? {
        val normalized = cleanGoal(goal)
        val compoundMatch = Regex("\\b(?:search|find)\\s+[A-Za-z0-9_]+(?:\\s+(?:on|in|inside)\\s+[A-Za-z0-9_]+)?\\s+(?:and|then)?\\s*(?:message|send|text)\\s+(?:(?:him|her|them)\\s+)?['\"]?([^'\"\\n]+?)['\"]?$", RegexOption.IGNORE_CASE)
            .find(normalized)
        if (compoundMatch != null) {
            val msg = compoundMatch.groupValues[2].trim().trim('"', '\'')
            if (msg.isNotBlank()) return msg
        }

        // 1. Support 'say/send <msg> to <recipient>' (e.g. 'open Instagram send hi to ajay' -> msg='hi')
        val sayMatch = Regex("\\b(?:say|send)\\s+(.+?)\\s+to\\s+(.+?)(?:\\s+(?:on|in|inside)\\s+[A-Za-z0-9_\\s]+)?$", RegexOption.IGNORE_CASE)
            .find(normalized)
        if (sayMatch != null) {
            val msg = sayMatch.groupValues[1].trim().trim('"', '\'')
            if (msg.isNotBlank() && !msg.lowercase().startsWith("to ")) return msg
        }

        // 2. Direct message with saying/that or trailing body
        val andMsg = Regex("\\b(?:and|then)?\\s*(?:message|send|text|tell|say)\\s+(?:(?:him|her|them|[A-Za-z0-9_]+)\\s+)?(?:saying\\s+|that\\s+)?['\"]?([^'\"\\n]+?)['\"]?(?:\\s+(?:on|in|inside)\\s+[A-Za-z0-9_\\s]+)?$", RegexOption.IGNORE_CASE)
            .find(normalized)
        if (andMsg != null) {
            val body = andMsg.groupValues[1].trim().trim('"', '\'')
            if (body.isNotBlank() && !body.lowercase().startsWith("to ") && body.lowercase() !in setOf("him", "her", "them", "message", "text")) {
                return body
            }
        }

        val messageMatch = Regex("\\b(?:message|text|tell|chat with)\\s+([A-Za-z0-9_]+)\\s+(?:saying\\s+|that\\s+)?(.+?)(?:\\s+(?:on|in|inside)\\s+[A-Za-z0-9_\\s]+)?$", RegexOption.IGNORE_CASE)
            .find(normalized)
        if (messageMatch != null) {
            var msg = messageMatch.groupValues[2].trim().trim('"', '\'')
            msg = msg.replace(Regex("\\s+(?:on|in|inside)\\s+[A-Za-z0-9_\\s]+$", RegexOption.IGNORE_CASE), "").trim()
            if (msg.isNotBlank()) return msg
        }
        return null
    }

    /** Resolves click/tap target or target entity from user goal. */
    fun resolveClickTarget(goal: String): String? = resolveTargetEntity(goal)

    /** Resolves the target entity/person/query the user explicitly named in their goal. */
    fun resolveTargetEntity(goal: String): String? {
        val normalized = cleanGoal(goal)
        val comparative = ProductExplorationEngine.detectComparativeIntent(normalized)
        if (comparative != null) {
            return comparative.targetEntity
        }

        // Form field target from: 'fill [the] <field> with <value>' (e.g. 'fill the USN box with 1MS21CS001' -> 'USN')
        val fillFieldMatch = Regex("\\b(?:fill|enter|type|input|write|put)\\s+(?:the\\s+)?(.+?)\\s+with\\s+['\"]?([^'\"\\n]+?)['\"]?$", RegexOption.IGNORE_CASE)
            .find(normalized)
        if (fillFieldMatch != null) {
            val rawField = fillFieldMatch.groupValues[1].trim()
                .replace(Regex("(?i)\\b(?:box|field|input|container|textbox|area|line)\\b"), "")
                .trim()
            if (rawField.isNotBlank() && rawField.lowercase() !in setOf("the", "a", "an", "this", "my", "your")) {
                return rawField
            }
        }

        // Form field target from: 'put/type <value> into/in [the] <field>' (e.g. 'put 1MS21CS001 into USN box' -> 'USN')
        val fillIntoFieldMatch = Regex("\\b(?:fill|enter|type|input|write|put)\\s+['\"]?[^'\"\\n]+?['\"]?\\s+(?:into|in)\\s+(?:the\\s+)?(.+?)$", RegexOption.IGNORE_CASE)
            .find(normalized)
        if (fillIntoFieldMatch != null) {
            val rawField = fillIntoFieldMatch.groupValues[1].trim()
                .replace(Regex("(?i)\\b(?:box|field|input|container|textbox|area|line)\\b"), "")
                .trim()
            if (rawField.isNotBlank() && rawField.lowercase() !in setOf("the", "a", "an", "this", "my", "your")) {
                return rawField
            }
        }

        val compoundMatch = Regex("\\b(?:search|find)\\s+([A-Za-z0-9_]+)(?:\\s+(?:on|in|inside)\\s+[A-Za-z0-9_]+)?\\s+(?:and|then)?\\s*(?:message|send|text)", RegexOption.IGNORE_CASE)
            .find(normalized)
        if (compoundMatch != null) {
            val cand = compoundMatch.groupValues[1].trim()
            if (cand.lowercase() !in setOf("the", "a", "an", "this", "my", "our", "him", "her", "them")) {
                return cand
            }
        }

        // Support 'say/send <msg> to <recipient>' (e.g. 'Say Hi to Manoj' -> 'Manoj', 'send I to Ajay s' -> 'Ajay s')
        val sayToMatch = Regex("\\b(?:say|send|tell|message|text)\\s+(?:.+?)\\s+to\\s+([^,.;:!?\n\r]+?)(?:\\s+(?:on|in|inside)\\s+[A-Za-z0-9_]+|\\s+(?:and|then|with)\\s+|$)", RegexOption.IGNORE_CASE).find(normalized)
        if (sayToMatch != null) {
            val cand = sayToMatch.groupValues[1].trim()
            if (cand.lowercase() !in setOf("the", "a", "an", "this", "my", "our", "him", "her", "them")) {
                return cand
            }
        }

        // Navigation & Click Pattern:
        // 1. "click/tap/press/select/choose/hit/open [on] [the] <target> [tab/section/button/page/category/chip/menu/view]"
        val clickMatches = Regex(
            """\b(?:click|tap|press|select|choose|hit|open)\s+(?:on\s+)?(?:the\s+)?([^,.;:!?\n\r]+?)(?:\s+(?:section|tab|page|screen|button|icon|category|chip|menu|view))?(?:\s+(?:and|then)\b|$)""",
            RegexOption.IGNORE_CASE
        ).findAll(normalized)
        for (clickMatch in clickMatches) {
            var raw = clickMatch.groupValues[1].trim()
            raw = raw.replace(Regex("(?i)\\b(?:section|tab|page|screen|button|icon|category|chip|menu|view)\\b"), "").trim()
            raw = raw.replace(Regex("^(?i)(?:the|a|an|on|to)\\s+"), "").trim()
            val lower = raw.lowercase(java.util.Locale.ROOT)
            val isKnownAppLaunchOnly = lower in AgentRuntime.KNOWN_TARGET_APPS &&
                !normalized.contains(Regex("(?i)\\b(?:section|tab|page|screen|button|icon|category|chip)\\b"))
            if (raw.isNotBlank() && !isKnownAppLaunchOnly && lower !in setOf("the", "a", "an", "this", "my", "your", "on", "it", "here", "there")) {
                return raw
            }
        }

        // 2. "go to / navigate to / switch to / take me to / show me [the] <target> [tab/section/page/category...]"
        val navMatches = Regex(
            """\b(?:go\s+to|navigate\s+to|switch\s+to|take\s+me\s+to|show\s+me)\s+(?:the\s+)?([^,.;:!?\n\r]+?)(?:\s+(?:section|tab|page|screen|button|icon|category|chip|menu|view))?(?:\s+(?:and|then)\b|$)""",
            RegexOption.IGNORE_CASE
        ).findAll(normalized)
        for (navMatch in navMatches) {
            var raw = navMatch.groupValues[1].trim()
            raw = raw.replace(Regex("(?i)\\b(?:section|tab|page|screen|button|icon|category|chip|menu|view)\\b"), "").trim()
            raw = raw.replace(Regex("^(?i)(?:the|a|an|on|to)\\s+"), "").trim()
            val lower = raw.lowercase(java.util.Locale.ROOT)
            val isKnownAppLaunchOnly = lower in AgentRuntime.KNOWN_TARGET_APPS &&
                !normalized.contains(Regex("(?i)\\b(?:section|tab|page|screen|button|icon|category|chip)\\b"))
            if (raw.isNotBlank() && !isKnownAppLaunchOnly && lower !in setOf("the", "a", "an", "this", "my", "your", "on", "it", "here", "there")) {
                return raw
            }
        }

        // 3. Standalone Section / Tab reference: '<target> section' or '<target> tab' (e.g. 'book section', 'books tab')
        val standaloneSectionMatch = Regex(
            """^([^,.;:!?\n\r]+?)\s+(?:section|tab|page|screen|category|chip)$""",
            RegexOption.IGNORE_CASE
        ).find(normalized)
        if (standaloneSectionMatch != null) {
            var raw = standaloneSectionMatch.groupValues[1].trim()
            raw = raw.replace(Regex("^(?i)(?:the|a|an|on|to)\\s+"), "").trim()
            val lower = raw.lowercase(java.util.Locale.ROOT)
            if (raw.isNotBlank() && lower !in setOf("the", "a", "an", "this", "my", "your", "on", "it")) {
                return raw
            }
        }

        // 4. Communication recipient pattern (message/text/call/chat with <recipient>)
        val toMatch = Regex("\\b(?:message|text|call|chat with)\\s+([^,.;:!?\n\r]+?)(?:\\s+(?:on|in|inside)\\s+[A-Za-z0-9_]+|\\s+(?:and|then|with)\\s+|$)", RegexOption.IGNORE_CASE).find(normalized)
        if (toMatch != null) {
            val candidate = toMatch.groupValues[1].trim()
            if (candidate.lowercase() !in setOf("the", "a", "an", "this", "my", "our", "him", "her", "them")) {
                return candidate
            }
        }

        val searchMatch = Regex("\\b(?:search|find|look for)\\s+(?:for\\s+)?(.+?)(?:\\s+(?:in|on|inside|within)\\s+[A-Za-z0-9_\\s]+|\\s+and\\s+|\\s+then\\s+|$)", RegexOption.IGNORE_CASE).find(normalized)
        if (searchMatch != null) {
            var term = searchMatch.groupValues[1].trim().trim(',', '.', '"', '\'')
            term = term.replace(Regex("\\s+(?:in|on|inside|within)\\s+[A-Za-z0-9_\\s]+$", RegexOption.IGNORE_CASE), "").trim()
            if (term.isNotBlank() && term.lowercase() !in setOf("the", "a", "an", "for")) return term
        }
        return null
    }
}