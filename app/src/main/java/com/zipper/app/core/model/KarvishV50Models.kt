package com.zipper.app.core.model

import java.security.MessageDigest

/**
 * KARVISH V50 — Universal Android Data Contracts & Closed-Loop Schemas
 * Strictly matches universal_schema_v50.py with zero app-specific hardcoding.
 */

enum class UniversalSemanticRole {
    BUTTON,
    TEXT_FIELD,
    SEARCH_FIELD,
    CHECKBOX,
    RADIO,
    TAB,
    LIST_ITEM,
    MENU_ITEM,
    LINK,
    IMAGE,
    ICON,
    DIALOG,
    HEADER,
    LABEL,
    SWITCH,
    SLIDER,
    SCROLL_CONTAINER,
    NAVIGATION,
    UNKNOWN;

    companion object {
        fun fromRoleString(role: String?): UniversalSemanticRole {
            if (role.isNullOrBlank()) return UNKNOWN
            val upper = role.uppercase()
            return when {
                upper == "SEARCH_FIELD" || (upper.contains("SEARCH") && upper.contains("EDIT")) -> SEARCH_FIELD
                upper == "TEXT_FIELD" || upper == "INPUT" || upper.contains("EDITTEXT") -> TEXT_FIELD
                upper == "BUTTON" -> BUTTON
                upper == "CHECKBOX" -> CHECKBOX
                upper == "RADIO" || upper.contains("RADIO") -> RADIO
                upper == "TAB" -> TAB
                upper == "SWITCH" || upper == "TOGGLE" -> SWITCH
                upper == "SLIDER" || upper.contains("SEEK") -> SLIDER
                upper == "IMAGE" || upper.contains("IMAGEVIEW") -> IMAGE
                upper == "ICON" -> ICON
                upper == "DIALOG" -> DIALOG
                upper == "HEADER" -> HEADER
                upper == "LINK" || upper.contains("LINK") -> LINK
                upper == "MENU_ITEM" || upper.contains("MENU") -> MENU_ITEM
                upper == "LABEL" || upper.contains("TEXTVIEW") -> LABEL
                upper.contains("SCROLL") || upper.contains("LIST") || upper.contains("RECYCLER") -> SCROLL_CONTAINER
                upper == "NAVIGATION" || upper.contains("NAV") -> NAVIGATION
                else -> UNKNOWN
            }
        }
    }
}

enum class DecisionMode {
    ACT,
    WAIT,
    OBSERVE,
    SEARCH,
    RECOVER,
    ABSTAIN,
    COMPLETE;

    companion object {
        fun fromIndex(idx: Int): DecisionMode = values().getOrElse(idx) { ACT }
    }
}

enum class UniversalActionType(val actionName: String) {
    TAP("tap"),
    TYPE_TEXT("type_text"),
    CLEAR_TEXT("clear_text"),
    SUBMIT("submit"),
    SCROLL("scroll"),
    SWIPE("swipe"),
    SELECT("select"),
    BACK("back"),
    HOME("home"),
    WAIT("wait"),
    COMPLETE("complete"),
    FAIL("fail"),
    OPEN_APP("open_app"),
    RECOVER("recover");

    companion object {
        fun fromString(name: String): UniversalActionType {
            val clean = name.lowercase().trim()
            return values().firstOrNull { it.actionName == clean } ?: when (clean) {
                "click", "press" -> TAP
                "type", "input" -> TYPE_TEXT
                "complete_goal", "done", "finish" -> COMPLETE
                "scroll_forward", "scroll_backward", "scroll_down", "scroll_up" -> SCROLL
                "press_back" -> BACK
                else -> WAIT
            }
        }
    }
}

enum class ExpectedTransition {
    TEXT_CHANGED,
    FOCUS_CHANGED,
    NAVIGATION_CHANGED,
    APP_CHANGED,
    SCROLL_CHANGED,
    DIALOG_APPEARED,
    DIALOG_DISMISSED,
    KEYBOARD_APPEARED,
    KEYBOARD_DISMISSED,
    CANDIDATE_SET_CHANGED,
    NO_VISIBLE_CHANGE,
    TASK_STATE_ADVANCED,
    TASK_COMPLETED;

    companion object {
        fun fromIndex(idx: Int): ExpectedTransition = values().getOrElse(idx) { NO_VISIBLE_CHANGE }
    }
}

enum class ProgressStatus {
    ADVANCED,
    PARTIAL,
    UNCHANGED,
    REGRESSED,
    COMPLETED,
    UNKNOWN;

    companion object {
        fun fromIndex(idx: Int): ProgressStatus = values().getOrElse(idx) { UNKNOWN }
    }
}

enum class RecoveryClass {
    BACK_OFF,
    RESET_STATE,
    DISMISS_INTERRUPTION,
    SCROLL_SEARCH,
    RETRY_AFFORDANCE,
    REOBSERVE,
    ABSTAIN;

    companion object {
        fun fromIndex(idx: Int): RecoveryClass = values().getOrElse(idx) { REOBSERVE }
    }
}

/**
 * Candidate Signature: Invariant descriptor used for target rechecking and ledger tracking.
 */
data class CandidateSignature(
    val role: UniversalSemanticRole,
    val label: String,
    val contentDescription: String,
    val normalizedBounds: NormalizedBounds,
    val isClickable: Boolean = false,
    val isEditable: Boolean = false,
    val isScrollable: Boolean = false,
    val isEnabled: Boolean = true,
    val isFocused: Boolean = false,
    val isChecked: Boolean = false,
    val parentRole: UniversalSemanticRole? = null
) {
    val signatureHash: String by lazy {
        val raw = "${role.name}|${label.trim().lowercase()}|${contentDescription.trim().lowercase()}|" +
                "${normalizedBounds.left},${normalizedBounds.top},${normalizedBounds.right},${normalizedBounds.bottom}|" +
                "${if (isClickable) 1 else 0}${if (isEditable) 1 else 0}${if (isEnabled) 1 else 0}"
        val bytes = MessageDigest.getInstance("SHA-256").digest(raw.toByteArray(Charsets.UTF_8))
        bytes.take(8).joinToString("") { "%02x".format(it) }
    }
}

/**
 * Task / Belief State: Tracks active requirements and submission readiness based strictly on evidence.
 */
data class TaskBeliefState(
    val activeSubgoal: String = "NAVIGATE",
    val completedRequirements: List<String> = emptyList(),
    val pendingRequirements: List<String> = emptyList(),
    val submissionReadiness: String = "NOT_READY", // READY, NOT_READY, FORBIDDEN
    val progressStatus: ProgressStatus = ProgressStatus.UNKNOWN,
    val confidence: Float = 1.0f
)

/**
 * Typed Semantic Task Memory: Evidence-backed fact persistence across apps.
 */
data class MemorySlot(
    val key: String,
    val value: String,
    val valueType: String = "string",
    val sourceContext: String = "",
    val timestampMs: Long = System.currentTimeMillis(),
    val confidence: Float = 1.0f,
    val evidence: String = ""
)

class SemanticTaskMemory(private val maxSlots: Int = 4) {
    private val slots = mutableListOf<MemorySlot>()

    fun write(slot: MemorySlot): Boolean {
        val existingIndex = slots.indexOfFirst { it.key == slot.key }
        if (existingIndex >= 0) {
            slots[existingIndex] = slot
            return true
        }
        if (slots.size < maxSlots) {
            slots.add(slot)
            return true
        }
        // Replace lowest confidence slot if new slot has higher confidence
        val lowestConfidenceIdx = slots.indices.minByOrNull { slots[it].confidence } ?: 0
        if (slot.confidence > slots[lowestConfidenceIdx].confidence) {
            slots[lowestConfidenceIdx] = slot
            return true
        }
        return false
    }

    fun read(key: String): MemorySlot? = slots.firstOrNull { it.key == key }

    fun getAllSlots(): List<MemorySlot> = slots.toList()

    fun clear() = slots.clear()
}

/**
 * Persistent Immutable Action Ledger Entry.
 */
data class ActionLedgerEntry(
    val stepId: Int,
    val timestampMs: Long,
    val mode: DecisionMode,
    val action: UniversalActionType,
    val targetCandidateHash: String?,
    val targetLabel: String?,
    val arguments: Map<String, Any> = emptyMap(),
    val confidence: Float = 1.0f,
    val expectedTransition: ExpectedTransition = ExpectedTransition.NO_VISIBLE_CHANGE,
    val observedTransition: ExpectedTransition? = null,
    val progress: ProgressStatus = ProgressStatus.UNKNOWN,
    val verificationPassed: Boolean = false,
    val recoveryInvoked: RecoveryClass? = null
)

class ActionLedger {
    private val entries = mutableListOf<ActionLedgerEntry>()

    fun append(entry: ActionLedgerEntry) {
        entries.add(entry)
    }

    fun getAll(): List<ActionLedgerEntry> = entries.toList()

    fun toSummaryString(maxRecent: Int = 5): String {
        if (entries.isEmpty()) return "No previous actions"
        return entries.takeLast(maxRecent).joinToString(" | ") { e ->
            val tgt = e.targetLabel ?: "None"
            "${e.action.actionName}('$tgt')->${e.progress.name}"
        }
    }

    fun hasRepeatingLoop(windowSize: Int = 3): Boolean {
        if (entries.size < windowSize) return false
        val tail = entries.takeLast(windowSize)
        val firstAct = tail[0].action
        val firstTgt = tail[0].targetCandidateHash
        return tail.all { it.action == firstAct && it.targetCandidateHash == firstTgt &&
                (it.progress == ProgressStatus.UNCHANGED || it.progress == ProgressStatus.UNKNOWN) }
    }
}

/**
 * Comprehensive V50 Decision Model Output.
 */
data class KarvishV50Decision(
    val mode: DecisionMode,
    val action: UniversalActionType,
    val targetCandidateIndex: Int?,
    val targetCandidateHash: String?,
    val argumentText: String?,
    val confidence: Float,
    val expectedTransition: ExpectedTransition,
    val progressEstimate: ProgressStatus,
    val terminationConfidence: Float,
    val recoveryClass: RecoveryClass? = null
)
