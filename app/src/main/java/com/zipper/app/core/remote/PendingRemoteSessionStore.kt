package com.zipper.app.core.remote

import java.util.concurrent.ConcurrentHashMap

/**
 * Tracks multi-turn conversational disambiguation for remote commands
 * (e.g., choosing between multiple contacts matching "Manoj").
 */
data class PendingDisambiguationSession(
    val senderId: String,
    val contactNameQuery: String = "",
    val messageText: String = "",
    val candidates: List<ResolvedContact> = emptyList(),
    val fileCandidates: List<ResolvedFile> = emptyList(),
    val targetRecipient: String? = null,
    val timestamp: Long = System.currentTimeMillis()
)

object PendingRemoteSessionStore {

    private const val SESSION_TTL_MS = 5 * 60 * 1000L // 5 minutes timeout
    private val activeSessions = ConcurrentHashMap<String, PendingDisambiguationSession>()

    fun setPending(senderId: String, session: PendingDisambiguationSession) {
        activeSessions[senderId] = session
    }

    fun getPending(senderId: String): PendingDisambiguationSession? {
        val session = activeSessions[senderId] ?: return null
        if (System.currentTimeMillis() - session.timestamp > SESSION_TTL_MS) {
            activeSessions.remove(senderId)
            return null
        }
        return session
    }

    fun removePending(senderId: String): PendingDisambiguationSession? {
        return activeSessions.remove(senderId)
    }

    fun clearAll() {
        activeSessions.clear()
    }
}
