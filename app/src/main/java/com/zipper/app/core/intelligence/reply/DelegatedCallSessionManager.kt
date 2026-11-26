package com.zipper.app.core.intelligence.reply

import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.os.Build
import android.service.notification.StatusBarNotification
import android.util.Log
import androidx.core.app.NotificationCompat
import com.zipper.app.R
import com.zipper.app.core.remote.RemoteCommandDispatcher
import com.zipper.app.core.web.WebConfigManager
import com.zipper.app.core.web.WebLlmClient
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.util.concurrent.ConcurrentHashMap

/**
 * Manages active pending call sessions when an AI LLM triggers an incoming call
 * to discuss a proposed meeting or schedule with Karthik.
 */
object DelegatedCallSessionManager {

    private const val TAG = "DelegatedCallSessionMgr"
    private const val STATUS_CHANNEL_ID = "karvish_ai_replies"
    private val scope = CoroutineScope(Dispatchers.IO)
    private val webLlmClient = WebLlmClient()

    data class PendingCallSession(
        val sessionId: String,
        val contact: DelegatedContact,
        val sbn: StatusBarNotification,
        val deflectionReply: String,
        val incomingMessage: String,
        val createdAtMs: Long = System.currentTimeMillis()
    )

    private val sessions = ConcurrentHashMap<String, PendingCallSession>()

    fun registerSession(sessionId: String, session: PendingCallSession) {
        sessions[sessionId] = session
        Log.i(TAG, "Registered pending call session $sessionId for contact '${session.contact.contactName}'")
    }

    fun getSession(sessionId: String): PendingCallSession? = sessions[sessionId]

    fun removeSession(sessionId: String): PendingCallSession? = sessions.remove(sessionId)

    /**
     * Called when Karthik answers the call and either speaks a decision or taps a quick action pill.
     */
    fun handleUserDecision(context: Context, sessionId: String, decisionText: String) {
        val session = removeSession(sessionId) ?: run {
            Log.w(TAG, "Session $sessionId not found or already handled.")
            return
        }

        // 1. Set a protective 45-second call cooldown so rapid follow-up texts (like 'baralva?') don't spam-ring
        DelegatedAutoReplyDispatcher.setCallCooldown(session.contact.id, 45000L)

        scope.launch {
            try {
                val webAiConfig = WebConfigManager(context).getConfig()
                com.zipper.app.core.intelligence.reply.ContactKnowledgeBook.getInstance(context)
                val recentHistory = DelegatedConversationMemory.getInstance(context)
                    .getHistory(session.contact.contactName, maxTurns = 6)
                    .map { "${it.sender}: ${it.message}" }

                // Synthesize raw user instruction ("tmw is okk ask her what time", "free at 5pm", etc.)
                // into a natural, first-person WhatsApp reply matching the contact's tone/persona.
                val synthesized = webLlmClient.synthesizeUserDecision(
                    contactName = session.contact.contactName,
                    incomingMessage = session.incomingMessage,
                    userInstruction = decisionText,
                    personaTone = session.contact.personaTone,
                    recentHistory = recentHistory,
                    config = webAiConfig
                ).ifBlank { decisionText }

                val finalReply = if (session.contact.appendAiBadge) {
                    "$synthesized\n\n— Sent via Zipper AI on behalf of Karthik"
                } else {
                    synthesized
                }

                val sent = RemoteCommandDispatcher.trySendInlineReply(context, session.sbn, finalReply)
                if (sent) {
                    DelegatedConversationMemory.getInstance(context).recordMessage(session.contact.contactName, "Karthik", synthesized)
                    DelegatedContactManager.getInstance(context).recordReply(session.contact.id)
                    Log.i(TAG, "✅ In-call user decision synthesized & sent to '${session.contact.contactName}': '$finalReply' (raw instruction: '$decisionText')")
                    postStatusNotification(
                        context = context,
                        title = "📞 Sent to ${session.contact.contactName}",
                        text = synthesized
                    )
                } else {
                    Log.w(TAG, "Failed to send inline reply for session $sessionId")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error synthesizing user decision: ${e.message}", e)
                val fallbackReply = if (session.contact.appendAiBadge) {
                    "$decisionText\n\n— Sent via Zipper AI on behalf of Karthik"
                } else {
                    decisionText
                }
                val sent = RemoteCommandDispatcher.trySendInlineReply(context, session.sbn, fallbackReply)
                if (sent) {
                    DelegatedConversationMemory.getInstance(context).recordMessage(session.contact.contactName, "Karthik", decisionText)
                    DelegatedContactManager.getInstance(context).recordReply(session.contact.id)
                }
            }
        }
    }

    /**
     * Called when Karthik declines the call or the call times out.
     * Automatically dispatches the polite human deflection message without committing.
     */
    fun handleDecline(context: Context, sessionId: String) {
        val session = removeSession(sessionId) ?: run {
            Log.d(TAG, "Session $sessionId already concluded.")
            return
        }

        DelegatedAutoReplyDispatcher.setCallCooldown(session.contact.id, 45000L)

        val deflection = session.deflectionReply.ifBlank {
            "Not 100% sure about that time yet bro, let me check my schedule once I'm free and ping you in a bit!"
        }

        val finalReply = if (session.contact.appendAiBadge) {
            "$deflection\n\n— Sent via Zipper AI on behalf of Karthik"
        } else {
            deflection
        }

        val sent = RemoteCommandDispatcher.trySendInlineReply(context, session.sbn, finalReply)
        if (sent) {
            DelegatedConversationMemory.getInstance(context).recordMessage(session.contact.contactName, "Karthik", deflection)
            DelegatedContactManager.getInstance(context).recordReply(session.contact.id)
            Log.i(TAG, "📞 Call declined. Sent deflection to '${session.contact.contactName}': '$finalReply'")
            postStatusNotification(
                context = context,
                title = "🤖 Deflection sent to ${session.contact.contactName}",
                text = deflection
            )
        } else {
            Log.w(TAG, "Failed to send deflection inline reply for session $sessionId")
        }
    }

    private fun postStatusNotification(context: Context, title: String, text: String) {
        val notificationManager = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                STATUS_CHANNEL_ID,
                "AI Auto-Replies",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Confirmations when the AI agent auto-replies on WhatsApp/SMS"
            }
            notificationManager.createNotificationChannel(channel)
        }

        val notification = NotificationCompat.Builder(context, STATUS_CHANNEL_ID)
            .setSmallIcon(R.mipmap.ic_launcher)
            .setContentTitle(title)
            .setContentText(text)
            .setStyle(NotificationCompat.BigTextStyle().bigText(text))
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setAutoCancel(true)
            .build()

        notificationManager.notify((System.currentTimeMillis() % 100000).toInt(), notification)
    }
}
