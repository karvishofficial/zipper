package com.zipper.app.core.agent

import java.util.regex.Pattern

/**
 * Universal Task-State Engine (Android Runtime Implementation).
 *
 * Implements the Universal Task-State Representation Contract:
 * - Generic requirement extraction from goal text (key-value slots, expressions, action targets).
 * - Evidence grounding from observable UI elements, input field values, masks, and verified history.
 * - Subgoal derivation and Submission Readiness (READY | NOT_READY | UNKNOWN).
 * - 1:1 Parity with Python training implementation (universal_task_state.py).
 * - ZERO application-specific or hardcoded rules.
 */
class UniversalTaskStateEngine {

    data class TaskRequirement(
        val reqId: String,
        val key: String,
        val expectedValue: String,
        var status: String, // "SATISFIED" | "PENDING" | "UNKNOWN"
        var evidence: String = ""
    )

    data class UniversalTaskState(
        val requirements: List<TaskRequirement>,
        val currentSubgoal: String,
        val submissionReadiness: String, // "READY" | "NOT_READY" | "UNKNOWN"
        val negativeConstraint: Boolean
    ) {
        val satisfiedRequirements: List<TaskRequirement>
            get() = requirements.filter { it.status == "SATISFIED" }

        val pendingRequirements: List<TaskRequirement>
            get() = requirements.filter { it.status == "PENDING" }

        /**
         * Formats the canonical, token-efficient string representation
         * identical to Python universal_task_state.py.
         */
        fun formatCanonicalString(): String {
            val doneParts = satisfiedRequirements.map { "${it.key}=${it.expectedValue}" }
            val doneStr = if (doneParts.isNotEmpty()) doneParts.joinToString(", ") else "none"

            val pendingParts = pendingRequirements.map { "${it.key}=${it.expectedValue}" }
            val pendingStr = if (pendingParts.isNotEmpty()) pendingParts.joinToString(", ") else "none"

            return "TaskState:[Done: $doneStr | Pending: $pendingStr | Subgoal: $currentSubgoal | Submit: $submissionReadiness]"
        }
    }

    companion object {
        private val NEGATIVE_SUBMIT_PATTERNS = listOf(
            Pattern.compile("\\b(?:do\\s+not|don't|without)\\s+(?:search|send|submit|click|execute|evaluate)\\b", Pattern.CASE_INSENSITIVE),
            Pattern.compile("\\btype\\s+only\\b", Pattern.CASE_INSENSITIVE),
            Pattern.compile("\\bdo\\s+not\\s+(?:enter|press\\s+enter)\\b", Pattern.CASE_INSENSITIVE)
        )
    }

    fun extractAndGround(
        goal: String,
        visibleNodes: List<Map<String, Any?>>,
        history: List<Map<String, Any?>>,
        currentPackage: String = ""
    ): UniversalTaskState {
        // 1. Detect negative submission constraints
        val hasNegativeConstraint = NEGATIVE_SUBMIT_PATTERNS.any { it.matcher(goal).find() }

        // 2. Extract Generic Requirements from the Goal
        val requirements = extractRequirements(goal)

        // 3. Ground each requirement against observable UI state & history
        groundRequirements(requirements, visibleNodes, history)

        // 4. Check if submit or evaluate has already occurred in history
        val hasSubmitted = history.any { h ->
            val act = h["action"]?.toString()?.lowercase() ?: ""
            val desc = h["action_description"]?.toString()?.lowercase() ?: ""
            act == "submit" || desc.contains("submit")
        }

        // 5. Derive Current Subgoal & Submission Readiness
        val pending = requirements.filter { it.status == "PENDING" }
        val submissionReadiness: String
        val currentSubgoal: String

        if (hasNegativeConstraint) {
            submissionReadiness = "NOT_READY"
            currentSubgoal = if (pending.isEmpty()) "COMPLETE_WITHOUT_SUBMIT" else "INPUT:${pending.first().key}"
        } else if (pending.isNotEmpty()) {
            submissionReadiness = "NOT_READY"
            currentSubgoal = "INPUT:${pending.first().key}"
        } else if (hasSubmitted) {
            submissionReadiness = "DONE"
            currentSubgoal = "COMPLETE"
        } else {
            submissionReadiness = "READY"
            currentSubgoal = if (requirements.any { it.key == "expression" }) "EVALUATE" else "SUBMIT"
        }

        return UniversalTaskState(
            requirements = requirements,
            currentSubgoal = currentSubgoal,
            submissionReadiness = submissionReadiness,
            negativeConstraint = hasNegativeConstraint
        )
    }

    private fun extractRequirements(goal: String): List<TaskRequirement> {
        val clean = goal.trim()

        // Pattern 1: Multi-field extraction after 'with' (generic key-value pairs)
        val mWith = Pattern.compile("\\b(?:with)\\s+(.+)$", Pattern.CASE_INSENSITIVE).matcher(clean)
        if (mWith.find()) {
            val body = mWith.group(1) ?: ""
            val clauses = body.split(Regex("[,;]|\\band\\b"))
            val kvs = mutableListOf<Pair<String, String>>()
            for (c in clauses) {
                val w = c.trim().split(Regex("\\s+")).filter { it.isNotBlank() }
                if (w.size >= 2) {
                    val valStr = w.last().trim('\'', '"')
                    val keyStr = w.subList(0, w.size - 1).joinToString("_").lowercase()
                    if (keyStr !in setOf("the", "a", "an", "and", "or", "in", "on", "for")) {
                        kvs.add(Pair(keyStr, valStr))
                    }
                }
            }
            if (kvs.size >= 2) {
                return kvs.mapIndexed { idx, (k, v) -> TaskRequirement("req_$idx", k, v, "PENDING") }
            }
        }

        // Pattern 2: Calculator Expression
        val mCalc = Pattern.compile("\\b(?:calculate|compute|eval)\\s+([0-9\\s.+\\-*\\/÷×−]+)", Pattern.CASE_INSENSITIVE).matcher(clean)
        if (mCalc.find()) {
            val rawExpr = (mCalc.group(1) ?: "").replace(" ", "")
            return listOf(TaskRequirement("req_expr", "expression", rawExpr, "PENDING"))
        }

        // Pattern 3: Type into field
        val mType = Pattern.compile("\\btype\\s+(.+?)\\s+into\\s+([a-zA-Z_\\s]+?)(?:\\s+but|$)", Pattern.CASE_INSENSITIVE).matcher(clean)
        if (mType.find()) {
            val valStr = (mType.group(1) ?: "").trim('\'', '"', ' ')
            val fldStr = (mType.group(2) ?: "").trim().lowercase().replace(" ", "_")
            return listOf(TaskRequirement("req_type", fldStr, valStr, "PENDING"))
        }

        // Pattern 4: Message saying
        val mMsg = Pattern.compile("\\b(?:saying)\\s+(.+?)(?:\\s+(?:to|into|but)|$)", Pattern.CASE_INSENSITIVE).matcher(clean)
        if (mMsg.find()) {
            val valStr = (mMsg.group(1) ?: "").trim('\'', '"', ' ')
            return listOf(TaskRequirement("req_msg", "message", valStr, "PENDING"))
        }

        // Pattern 5: Search
        val mSearch = Pattern.compile("\\b(?:search|find|look for)\\s+(?:for\\s+)?(.+?)(?:\\s+(?:in|and|then|but)|$)", Pattern.CASE_INSENSITIVE).matcher(clean)
        if (mSearch.find()) {
            val valStr = (mSearch.group(1) ?: "").trim('\'', '"', ' ')
            return listOf(TaskRequirement("req_search", "query", valStr, "PENDING"))
        }

        // Pattern 6: Go to / URL
        val mGoto = Pattern.compile("\\b(?:go\\s+to|navigate\\s+to|visit)\\s+(.+?)(?:\\s+|$)", Pattern.CASE_INSENSITIVE).matcher(clean)
        if (mGoto.find()) {
            val valStr = (mGoto.group(1) ?: "").trim('\'', '"', ' ')
            return listOf(TaskRequirement("req_url", "url", valStr, "PENDING"))
        }

        // Fallback: single generic target
        return listOf(TaskRequirement("req_goal", "target", clean, "PENDING"))
    }

    private fun groundRequirements(
        requirements: List<TaskRequirement>,
        visibleNodes: List<Map<String, Any?>>,
        history: List<Map<String, Any?>>
    ) {
        val visibleTexts = mutableListOf<String>()
        var hasMaskedPasswordNode = false
        for (n in visibleNodes) {
            val txt = (n["text"] as? String ?: n["label"] as? String ?: "").trim().lowercase()
            if (txt.isNotBlank()) visibleTexts.add(txt)
            val cd = (n["contentDescription"] as? String ?: "").trim().lowercase()
            if (cd.isNotBlank()) visibleTexts.add(cd)

            val isEditable = (n["editable"] as? Boolean) ?: (n["isEditable"] as? Boolean) ?: false
            if (isEditable && ("•" in txt || "*" in txt || txt == "••••••")) {
                hasMaskedPasswordNode = true
            }
        }

        val typedEntries = mutableListOf<String>()
        val historyDescriptions = mutableListOf<String>()
        for (h in history) {
            val desc = (h["action_description"] as? String ?: h["outcome"] as? String ?: "").lowercase()
            historyDescriptions.add(desc)
            val action = h["action"] as? String ?: ""
            if ("type" in desc || action == "type_text") {
                typedEntries.add(desc)
            }
        }

        for (req in requirements) {
            val valClean = req.expectedValue.trim().lowercase()

            if (req.key == "expression") {
                val compactVal = valClean.replace(" ", "")
                val isComplete = historyDescriptions.any { it.replace(" ", "").contains(compactVal) } ||
                        visibleTexts.any { it.replace(" ", "").contains(compactVal) }
                if (isComplete) {
                    req.status = "SATISFIED"
                    req.evidence = "display_complete:${req.expectedValue}"
                } else {
                    req.status = "PENDING"
                    req.evidence = "expression_incomplete"
                }
                continue
            }

            val matchedOnScreen = visibleTexts.any { it.contains(valClean) }
            val matchedPassword = "password" in req.key && (hasMaskedPasswordNode || typedEntries.any { "password" in it })
            val matchedHistory = typedEntries.any { it.contains(valClean) }

            if (matchedOnScreen || matchedPassword || matchedHistory) {
                req.status = "SATISFIED"
                req.evidence = "observed_value:${req.expectedValue}"
            } else {
                req.status = "PENDING"
                req.evidence = "unmatched"
            }
        }
    }
}
