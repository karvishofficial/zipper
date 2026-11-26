package com.zipper.app.core.intelligence.reply

import org.json.JSONObject
import java.util.UUID

/**
 * Data model for a contact delegated to the AI auto-reply agent.
 */
data class DelegatedContact(
    val id: String = UUID.randomUUID().toString(),
    val contactName: String,
    val phoneNumber: String = "",
    val customContext: String = "",
    val personaTone: String = "Casual & Friendly",
    val isEnabled: Boolean = true,
    val maxRepliesPerSession: Int = 0,
    val requireApproval: Boolean = false,
    val appendAiBadge: Boolean = false,
    val lastReplyTimestampMs: Long = 0L,
    val sessionReplyCount: Int = 0
) {

    /**
     * Checks if the session budget has been exhausted.
     * If maxRepliesPerSession is <= 0 or >= 100, continuous chat is enabled without limits.
     * If more than [staleThresholdMs] (default 15 minutes) has elapsed since the last reply,
     * the conversation session is considered reset.
     */
    fun isSessionBudgetExceeded(staleThresholdMs: Long = 15 * 60 * 1000L): Boolean {
        if (!isEnabled) return true
        if (maxRepliesPerSession <= 0 || maxRepliesPerSession >= 100) return false // Continuous mode (no limits)
        val now = System.currentTimeMillis()
        if (now - lastReplyTimestampMs > staleThresholdMs) {
            // Fresh conversation session
            return false
        }
        return sessionReplyCount >= maxRepliesPerSession
    }

    /**
     * Returns an updated copy with an incremented reply count and current timestamp.
     */
    fun recordReplySent(staleThresholdMs: Long = 15 * 60 * 1000L): DelegatedContact {
        val now = System.currentTimeMillis()
        val isNewSession = (now - lastReplyTimestampMs) > staleThresholdMs
        val newCount = if (isNewSession) 1 else sessionReplyCount + 1
        return copy(
            lastReplyTimestampMs = now,
            sessionReplyCount = newCount
        )
    }

    fun toJson(): JSONObject {
        return JSONObject().apply {
            put("id", id)
            put("contactName", contactName)
            put("phoneNumber", phoneNumber)
            put("customContext", customContext)
            put("personaTone", personaTone)
            put("isEnabled", isEnabled)
            put("maxRepliesPerSession", maxRepliesPerSession)
            put("requireApproval", requireApproval)
            put("appendAiBadge", appendAiBadge)
            put("lastReplyTimestampMs", lastReplyTimestampMs)
            put("sessionReplyCount", sessionReplyCount)
        }
    }

    companion object {
        fun fromJson(json: JSONObject): DelegatedContact {
            return DelegatedContact(
                id = json.optString("id", UUID.randomUUID().toString()),
                contactName = json.optString("contactName", ""),
                phoneNumber = json.optString("phoneNumber", ""),
                customContext = json.optString("customContext", ""),
                personaTone = json.optString("personaTone", "Casual & Friendly"),
                isEnabled = json.optBoolean("isEnabled", true),
                maxRepliesPerSession = json.optInt("maxRepliesPerSession", 0),
                requireApproval = json.optBoolean("requireApproval", false),
                appendAiBadge = json.optBoolean("appendAiBadge", false),
                lastReplyTimestampMs = json.optLong("lastReplyTimestampMs", 0L),
                sessionReplyCount = json.optInt("sessionReplyCount", 0)
            )
        }
    }
}
