package com.zipper.app.core.voice

import org.json.JSONObject
import java.util.UUID

enum class CallReminderActionType {
    VOICE_ONLY,      // Rings call, speaks reminder, awaits spoken "done" or snooze
    AGENT_HANDOFF,   // "Pay Now / Do it" hands off to Karvish AgentRuntime
    PERSISTENT_ALARM // Ring persistently until verified
}

data class CallReminder(
    val id: String = UUID.randomUUID().toString(),
    val title: String,
    val spokenPrompt: String,
    val triggerTimeEpochMs: Long,
    val actionType: CallReminderActionType = CallReminderActionType.VOICE_ONLY,
    val targetGoal: String? = null,
    val targetPackage: String? = null,
    val isCompleted: Boolean = false,
    val isSnoozed: Boolean = false,
    val snoozeCount: Int = 0,
    val createdAtEpochMs: Long = System.currentTimeMillis(),
    val commitmentId: Long? = null,
    val category: String? = null,
    val candidatePackages: List<String> = emptyList(),
    val sourcePerson: String? = null
) {
    fun toJson(): JSONObject {
        return JSONObject().apply {
            put("id", id)
            put("title", title)
            put("spokenPrompt", spokenPrompt)
            put("triggerTimeEpochMs", triggerTimeEpochMs)
            put("actionType", actionType.name)
            put("targetGoal", targetGoal ?: JSONObject.NULL)
            put("targetPackage", targetPackage ?: JSONObject.NULL)
            put("isCompleted", isCompleted)
            put("isSnoozed", isSnoozed)
            put("snoozeCount", snoozeCount)
            put("createdAtEpochMs", createdAtEpochMs)
            put("commitmentId", commitmentId ?: JSONObject.NULL)
            put("category", category ?: JSONObject.NULL)
            put("candidatePackages", candidatePackages.joinToString(","))
            put("sourcePerson", sourcePerson ?: JSONObject.NULL)
        }
    }

    companion object {
        fun fromJson(json: JSONObject): CallReminder {
            val typeStr = json.optString("actionType", CallReminderActionType.VOICE_ONLY.name)
            val actionType = try {
                CallReminderActionType.valueOf(typeStr)
            } catch (e: Exception) {
                CallReminderActionType.VOICE_ONLY
            }

            val pkgStr = json.optString("candidatePackages", "")
            val pkgs = if (pkgStr.isNotBlank()) pkgStr.split(",").map { it.trim() }.filter { it.isNotEmpty() } else emptyList()

            return CallReminder(
                id = json.optString("id", UUID.randomUUID().toString()),
                title = json.optString("title", "Reminder"),
                spokenPrompt = json.optString("spokenPrompt", "You have a reminder."),
                triggerTimeEpochMs = json.optLong("triggerTimeEpochMs", System.currentTimeMillis()),
                actionType = actionType,
                targetGoal = if (json.isNull("targetGoal")) null else json.optString("targetGoal"),
                targetPackage = if (json.isNull("targetPackage")) null else json.optString("targetPackage"),
                isCompleted = json.optBoolean("isCompleted", false),
                isSnoozed = json.optBoolean("isSnoozed", false),
                snoozeCount = json.optInt("snoozeCount", 0),
                createdAtEpochMs = json.optLong("createdAtEpochMs", System.currentTimeMillis()),
                commitmentId = if (json.isNull("commitmentId")) null else json.optLong("commitmentId"),
                category = if (json.isNull("category")) null else json.optString("category"),
                candidatePackages = pkgs,
                sourcePerson = if (json.isNull("sourcePerson")) null else json.optString("sourcePerson")
            )
        }
    }
}
