package com.zipper.app.core.remote

enum class RemoteActionType {
    FIND_MY_PHONE_RING,
    FIND_MY_PHONE_STOP,
    GET_BATTERY_STATUS,
    GET_LOCATION,
    GET_LAST_OTP,
    RECALL_SCREEN_MEMORY,
    SEND_WHATSAPP_MESSAGE,
    SCHEDULE_MESSAGE,
    CONFIRM_CONTACT_SELECTION,
    SEND_FILE,
    CONFIRM_FILE_SELECTION,
    EXECUTE_SHOPPING,
    QUERY_STATUS_OTP,
    GET_EXPENSE_SUMMARY,
    ARM_SENTRY,
    DISARM_SENTRY,
    EVALUATE_CART_ARBITRAGE,
    START_COMMUTE_GUARD,
    STOP_COMMUTE_GUARD,
    TRIGGER_SOS,
    REMEMBER_CONTACT_KNOWLEDGE,
    SNAP_INTRUDER,
    ACTIVATE_MIMIC,
    DISARM_MIMIC,
    GENERAL_AGENT_GOAL,
    UNKNOWN
}

data class ResolvedFile(
    val name: String,
    val path: String,
    val sizeBytes: Long,
    val mimeType: String,
    val lastModified: Long = 0L
)

data class RemoteCommand(
    val rawText: String,
    val sender: String,
    val sourceApp: String,
    val actionType: RemoteActionType,
    val pinProvided: String? = null,
    val isAuthorized: Boolean = false,
    val targetApp: String? = null,
    val recipient: String? = null,
    val messageBody: String? = null,
    val scheduledTimeMs: Long? = null,
    val shoppingQuery: String? = null,
    val fileQuery: String? = null,
    val selectedContactIndex: Int? = null,
    val resolvedContact: ResolvedContact? = null,
    val selectedFileIndex: Int? = null,
    val resolvedFile: ResolvedFile? = null,
    val rawGoal: String = "",
    val timestamp: Long = System.currentTimeMillis()
)

