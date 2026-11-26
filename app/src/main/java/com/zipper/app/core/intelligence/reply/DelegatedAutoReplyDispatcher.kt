package com.zipper.app.core.intelligence.reply

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.content.Intent
import android.os.Build
import android.service.notification.StatusBarNotification
import android.util.Log
import androidx.core.app.NotificationCompat
import com.zipper.app.R
import com.zipper.app.core.remote.RemoteCommandDispatcher
import com.zipper.app.core.voice.IncomingCallActivity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.util.Locale

/**
 * Handles incoming notifications, checks delegated contact rules,
 * generates AI replies via LLM, and sends silent inline replies via Android RemoteInput.
 */
object DelegatedAutoReplyDispatcher {

    private const val TAG = "DelegatedAutoReply"
    private const val STATUS_CHANNEL_ID = "karvish_ai_replies"
    private val scope = CoroutineScope(Dispatchers.IO)

    private val MESSAGING_PACKAGES = setOf(
        "com.whatsapp",
        "com.whatsapp.w4b",
        "org.telegram.messenger",
        "com.google.android.apps.messaging",
        "com.facebook.orca",
        "com.instagram.android",
        "com.slack"
    )

    // Deduplication cache: Key = "${pkg}_${sender}_${cleanText}", Value = timestampMs
    private val processedMessageCache = java.util.concurrent.ConcurrentHashMap<String, Long>()
    // Cooldown cache per contact to prevent double-firing: Key = contactId, Value = timestampMs
    private val contactCooldownCache = java.util.concurrent.ConcurrentHashMap<String, Long>()
    // Cooldown cache for live phone call triggers: Key = contactId, Value = expiryTimestampMs
    private val contactCallCooldownUntilMs = java.util.concurrent.ConcurrentHashMap<String, Long>()
    private val contactCallCooldownCache = contactCallCooldownUntilMs
    // Last sent reply cache per contact to prevent self-echo loops
    private val lastSentReplyCache = java.util.concurrent.ConcurrentHashMap<String, String>()
    // Active notification cache holding valid RemoteInput actions per contact (used for missed call replies)
    private val activeRemoteInputSbn = java.util.concurrent.ConcurrentHashMap<String, StatusBarNotification>()

    fun getActiveRemoteInputSbn(sender: String): StatusBarNotification? {
        return activeRemoteInputSbn[sender.lowercase(Locale.ROOT)]
    }

    fun isObjectionOrCounterProposal(incomingMessage: String): Boolean {
        val lower = incomingMessage.lowercase(Locale.ROOT).trim()

        // 1. Explicit refusal, rejection, objection or change of schedule
        val hasRejection = lower.contains(Regex("""\b(no|nope|not)\b""")) ||
            lower.contains("cannot") ||
            lower.contains("can't") ||
            lower.contains("cant") ||
            lower.contains("won't") ||
            lower.contains("wont") ||
            lower.contains("beda") ||
            lower.contains("agalla") ||
            lower.contains("aagalla") ||
            lower.contains("baralva") ||
            lower.contains("change") ||
            lower.contains("different time") ||
            lower.contains("bere time")

        // 2. Urgent counter-demands ("IVATHE BA", "bega baa", "come today", "urgent", etc.)
        val hasUrgentDemand = lower.contains("ivathe") ||
            lower.contains("ivatte") ||
            lower.contains("bega ba") ||
            lower.contains("bega baa") ||
            lower.contains("today itself") ||
            lower.contains("right now") ||
            lower.contains("urgent") ||
            lower.contains("urgently") ||
            lower.contains("bega") ||
            lower.contains("emergency")

        // 3. Agreement / confirmation check ("Sari 4 PM ge", "Ok done 4 PM", etc.)
        val hasAgreement = lower.startsWith("sari") ||
            lower.startsWith("ok") ||
            lower.startsWith("okay") ||
            lower.startsWith("done") ||
            lower.startsWith("fine") ||
            lower.startsWith("sure") ||
            lower.contains("aaythu") ||
            lower.contains("aguthe") ||
            lower.contains("works for me")

        if (hasAgreement && !hasRejection && !hasUrgentDemand) {
            return false
        }

        // 4. Counter-proposed time (e.g. contact suggests 2 PM, 6 PM, etc.)
        val proposedTime = SmartReplyGenerator.extractProposedTime(incomingMessage)
        val hasCounterTime = proposedTime != null

        return hasRejection || hasUrgentDemand || hasCounterTime
    }

    fun clearCallCooldown(contactId: String) {
        contactCallCooldownUntilMs.remove(contactId)
        Log.d(TAG, "Cleared call cooldown for contact $contactId to allow follow-up proposal calls")
    }

    fun setCallCooldown(contactId: String, durationMs: Long = 45000L) {
        val until = System.currentTimeMillis() + durationMs
        contactCallCooldownUntilMs[contactId] = until
        Log.d(TAG, "Set call cooldown for contact $contactId for ${durationMs / 1000}s (until $until)")
    }

    /**
     * Checks if the incoming notification is from a delegated contact and triggers an automated reply.
     * Returns true if handled.
     */
    fun handleIncomingNotification(
        context: Context,
        sbn: StatusBarNotification,
        pkg: String,
        sender: String,
        messageText: String
    ): Boolean {
        // 1. Package filter
        if (pkg !in MESSAGING_PACKAGES) {
            return false
        }

        val notification = sbn.notification ?: return false
        val extras = notification.extras ?: return false

        // 2. Ignore Group Summary Notifications (WhatsApp always posts a summary notification + message notification)
        val isGroupSummary = (notification.flags and Notification.FLAG_GROUP_SUMMARY) != 0
        if (isGroupSummary) {
            return false
        }

        // 3. Ignore Group Conversations (Group chats)
        val isGroup = extras.getBoolean(Notification.EXTRA_IS_GROUP_CONVERSATION, false) ||
            sender.contains(":") ||
            extras.containsKey("android.conversationTitle")
        if (isGroup) {
            return false
        }

        val cleanText = messageText.trim()
        if (cleanText.isBlank()) return false

        // Cache latest valid messaging notification for this contact (used for missed call follow-ups)
        activeRemoteInputSbn[sender.lowercase(Locale.ROOT)] = sbn

        // 3.5. Ignore Missed Call Notifications (handled specifically by MissedCallFollowupEngine)
        if (MissedCallFollowupEngine.isMissedCallNotification(cleanText, sender)) {
            Log.d(TAG, "Notification from '$sender' is a missed call event. Bypassing text auto-reply.")
            return false
        }

        // 4. Match Delegated Contact
        val manager = DelegatedContactManager.getInstance(context)
        val contact = manager.findMatching(sender) ?: return false

        if (!contact.isEnabled) {
            return false
        }

        val now = System.currentTimeMillis()

        // 5. Post-Reply Cooldown Guard: Prevent double-firing within 3 seconds of a dispatched reply
        val lastReplyTime = contactCooldownCache[contact.id] ?: 0L
        if (now - lastReplyTime < 3000L) {
            Log.d(TAG, "Contact '${contact.contactName}' is in post-reply cooldown (${now - lastReplyTime}ms < 3000ms). Skipping.")
            return false
        }

        // 6. Message Deduplication Guard (Ignore identical message posted within 20 seconds)
        val dedupKey = "${pkg}_${sender.lowercase()}_${cleanText.lowercase()}"
        val lastProcessedTime = processedMessageCache[dedupKey] ?: 0L
        if (now - lastProcessedTime < 20000L) {
            Log.d(TAG, "Duplicate notification event detected for '$dedupKey'. Skipping.")
            return false
        }

        // 7. Self-Echo Guard (WhatsApp updates notification with our own reply text)
        val lastSent = lastSentReplyCache[contact.id]
        if (!lastSent.isNullOrBlank() && cleanText.contains(lastSent, ignoreCase = true)) {
            Log.d(TAG, "Notification text is an echo of our own sent reply ('$lastSent'). Skipping.")
            return false
        }

        // 8. Rate-limiting & Loop Breaker Check
        if (contact.isSessionBudgetExceeded()) {
            Log.w(TAG, "Session reply budget exceeded (${contact.sessionReplyCount}/${contact.maxRepliesPerSession}) for contact '${contact.contactName}'. Skipping auto-reply.")
            return false
        }

        processedMessageCache[dedupKey] = now

        // 9. Record incoming message into ContactKnowledgeBook for dynamic style and dossier history
        ContactKnowledgeBook.getInstance(context).recordChatMessage(contact.contactName, cleanText, isFromUser = false)

        Log.i(TAG, "⚡ Enqueuing ping from '${contact.contactName}' on $pkg to MultiPingDebouncer: '$cleanText'")

        val senderKey = "${pkg}_${contact.id}"
        MultiPingDebouncer.enqueueMessage(
            senderKey = senderKey,
            senderName = contact.contactName,
            text = cleanText,
            sbn = sbn
        ) { _, _, unifiedText, latestSbn ->
            scope.launch {
                try {
                    val targetSbn = latestSbn ?: sbn
                    processAndReply(context, targetSbn, contact, unifiedText, manager)
                } catch (e: Exception) {
                    Log.e(TAG, "Error in auto-reply pipeline for ${contact.contactName}: ${e.message}", e)
                }
            }
        }

        return true
    }

    data class CallUserRequest(
        val reason: String,
        val deflectionReply: String
    )

    fun parseCallUserPayload(payload: String, contactName: String): CallUserRequest {
        val payloadWithoutPrefix = payload
            .replaceFirst(Regex("""(?i)^CALL_USER:\s*"""), "")
            .trim()
        val parts = payloadWithoutPrefix.split("|", limit = 2)
        val reason = parts.getOrNull(0)?.trim()?.ifBlank { null }
            ?: "Discuss schedule with $contactName"
        val deflection = parts.getOrNull(1)?.trim()?.ifBlank { null }
            ?: "Not 100% sure about that time yet bro, let me check my schedule once I'm free and ping you in a bit!"
        return CallUserRequest(reason, deflection)
    }

    private suspend fun processAndReply(
        context: Context,
        sbn: StatusBarNotification,
        contact: DelegatedContact,
        incomingMessage: String,
        manager: DelegatedContactManager
    ) {
        val rawReplyText = SmartReplyGenerator.generateReply(context, contact, incomingMessage) ?: return

        // Check if LLM requested a live phone call to Karthik regarding schedule/meeting confirmation
        if (rawReplyText.startsWith("CALL_USER:", ignoreCase = true)) {
            val now = System.currentTimeMillis()
            val cooldownUntil = contactCallCooldownUntilMs[contact.id] ?: 0L
            val isObjectionOrCounter = isObjectionOrCounterProposal(incomingMessage)

            // If cooldown is active AND this is NOT an objection/counter-proposal, suppress repeat call
            if (now < cooldownUntil && !isObjectionOrCounter) {
                Log.i(TAG, "Call cooldown active for '${contact.contactName}' (${(cooldownUntil - now) / 1000}s remaining). Suppressing repeat ringing; sending deflection.")
                val parsed = parseCallUserPayload(rawReplyText, contact.contactName)
                val deflection = parsed.deflectionReply
                val safeDeflection = CommitmentGuard.sanitizeReply(incomingMessage, deflection)
                val finalReply = if (contact.appendAiBadge) {
                    "$safeDeflection\n\n— Sent via Zipper AI on behalf of Karthik"
                } else {
                    safeDeflection
                }
                val sent = RemoteCommandDispatcher.trySendInlineReply(context, sbn, finalReply)
                if (sent) {
                    DelegatedConversationMemory.getInstance(context).recordMessage(contact.contactName, "Karthik", safeDeflection)
                    ContactKnowledgeBook.getInstance(context).recordChatMessage(contact.contactName, safeDeflection, isFromUser = true)
                    manager.recordReply(contact.id)
                }
                return
            }

            // Set a short 45-second debounce window for identical repeated pings
            setCallCooldown(contact.id, 45000L)
            handleCallUserRequest(context, sbn, contact, incomingMessage, rawReplyText)
            return
        }

        // Run through CommitmentGuard to guarantee zero fake excuses, no financial liabilities, and no unauthorized calendar promises
        val replyText = CommitmentGuard.sanitizeReply(incomingMessage, rawReplyText)

        if (contact.requireApproval) {
            Log.i(TAG, "Contact '${contact.contactName}' requires approval. Posting co-pilot suggestion.")
            postApprovalNotification(context, contact, replyText, sbn)
            return
        }

        // Silent background dispatch via RemoteInput
        val sent = RemoteCommandDispatcher.trySendInlineReply(context, sbn, replyText)
        if (sent) {
            lastSentReplyCache[contact.id] = replyText
            contactCooldownCache[contact.id] = System.currentTimeMillis()
            manager.recordReply(contact.id)
            ContactKnowledgeBook.getInstance(context).recordChatMessage(contact.contactName, replyText, isFromUser = true)
            Log.i(TAG, "✅ Single AI reply sent silently to '${contact.contactName}': '$replyText'")
            postStatusNotification(
                context = context,
                title = "🤖 Replied to ${contact.contactName}",
                text = replyText
            )
        } else {
            Log.w(TAG, "Could not send inline reply via RemoteInput to ${contact.contactName}. Action might not have RemoteInput.")
        }
    }

    private suspend fun handleCallUserRequest(
        context: Context,
        sbn: StatusBarNotification,
        contact: DelegatedContact,
        incomingMessage: String,
        callUserPayload: String
    ) {
        val cleanName = contact.contactName.trim().replaceFirstChar { if (it.isLowerCase()) it.titlecase(Locale.ROOT) else it.toString() }
        val request = parseCallUserPayload(callUserPayload, cleanName)
        val sessionId = "call_${contact.id}_${System.currentTimeMillis()}"
        val session = DelegatedCallSessionManager.PendingCallSession(
            sessionId = sessionId,
            contact = contact.copy(contactName = cleanName),
            sbn = sbn,
            deflectionReply = request.deflectionReply,
            incomingMessage = incomingMessage
        )
        DelegatedCallSessionManager.registerSession(sessionId, session)

        Log.i(TAG, "📞 Triggering IncomingCallActivity for contact '$cleanName' (reason: '${request.reason}')")

        try {
            val webAiConfig = com.zipper.app.core.web.WebConfigManager(context).getConfig()
            val webLlm = com.zipper.app.core.web.WebLlmClient()
            val spokenPrompt = webLlm.synthesizeSpokenBriefing(
                contactName = cleanName,
                incomingMessage = incomingMessage,
                config = webAiConfig
            )

            val callIntent = Intent(context, IncomingCallActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP
                putExtra(IncomingCallActivity.EXTRA_REMINDER_TITLE, "Meeting / Schedule Proposal")
                putExtra(IncomingCallActivity.EXTRA_REMINDER_PROMPT, spokenPrompt)
                putExtra(IncomingCallActivity.EXTRA_SOURCE_PERSON, cleanName)
                putExtra(IncomingCallActivity.EXTRA_CALL_SESSION_ID, sessionId)
                putExtra(IncomingCallActivity.EXTRA_DEFLECTION_REPLY, request.deflectionReply)
                putExtra(IncomingCallActivity.EXTRA_ORIGINAL_MESSAGE, incomingMessage)
            }
            context.startActivity(callIntent)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to launch IncomingCallActivity: ${e.message}", e)
            // Fallback immediately to deflection if call activity cannot be launched
            DelegatedCallSessionManager.handleDecline(context, sessionId)
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

    private fun postApprovalNotification(
        context: Context,
        contact: DelegatedContact,
        draftReply: String,
        sbn: StatusBarNotification
    ) {
        val notificationManager = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                STATUS_CHANNEL_ID,
                "AI Auto-Replies",
                NotificationManager.IMPORTANCE_HIGH
            )
            notificationManager.createNotificationChannel(channel)
        }

        val notification = NotificationCompat.Builder(context, STATUS_CHANNEL_ID)
            .setSmallIcon(R.mipmap.ic_launcher)
            .setContentTitle("🤖 Suggested Reply for ${contact.contactName}")
            .setContentText(draftReply)
            .setStyle(NotificationCompat.BigTextStyle().bigText("Draft: \"$draftReply\""))
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setAutoCancel(true)
            .build()

        notificationManager.notify((System.currentTimeMillis() % 100000).toInt(), notification)
    }
}
