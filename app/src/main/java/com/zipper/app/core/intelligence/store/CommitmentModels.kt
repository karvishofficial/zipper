package com.zipper.app.core.intelligence.store

/**
 * Clean message captured by the 24/7 background guard.
 * Stored in a 24-hour rolling window and auto-purged to guarantee < 2 MB DB size.
 */
data class ContextMessage(
    val id: Long = 0L,
    val sender: String,
    val appName: String,
    val packageName: String,
    val summaryText: String,
    val timestampMs: Long = System.currentTimeMillis()
)

/**
 * Category of personal commitment.
 */
enum class CommitmentCategory {
    FITNESS,
    EDUCATION,
    WORK,
    HEALTH,
    COMMUNICATION,
    GENERAL
}

/**
 * Lifecycle status of a commitment.
 */
enum class CommitmentStatus {
    PENDING,
    IN_PROGRESS,
    COMPLETED,
    SNOOZED,
    BLOCKED
}

/**
 * A persistent personal commitment, task, or planned routine.
 */
data class CommitmentEntity(
    val id: Long = 0L,
    val title: String,
    val category: CommitmentCategory,
    val deadlineMs: Long? = null,
    val scheduledTimeMs: Long = 0L,
    val durationMinutes: Int = 30,
    val status: CommitmentStatus = CommitmentStatus.PENDING,
    val sourceApp: String = "",
    val sourcePerson: String = "",
    val candidatePackages: List<String> = emptyList(),
    val notes: String = "",
    val createdTimeMs: Long = System.currentTimeMillis()
) {
    val isChecklist: Boolean
        get() = notes.contains("[TYPE:CHECKLIST]") ||
                category == CommitmentCategory.COMMUNICATION ||
                title.startsWith("Return call", ignoreCase = true) ||
                title.startsWith("Call ", ignoreCase = true) ||
                title.startsWith("Send ", ignoreCase = true) ||
                title.startsWith("Message ", ignoreCase = true) ||
                title.startsWith("Check ", ignoreCase = true) ||
                title.startsWith("Reply ", ignoreCase = true) ||
                title.startsWith("Ask ", ignoreCase = true) ||
                (durationMinutes <= 10 && !notes.contains("[TYPE:TIMELINE]"))
}
