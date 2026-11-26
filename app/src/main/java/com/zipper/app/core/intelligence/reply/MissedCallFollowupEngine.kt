package com.zipper.app.core.intelligence.reply

import android.content.Context
import android.util.Log
import com.zipper.app.core.notification.NotificationCategory
import com.zipper.app.core.notification.NotificationStorageManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import android.service.notification.StatusBarNotification
import java.util.Locale

/**
 * Missed Call Follow-Up Engine.
 *
 * Automatically inspects recent conversation history when an incoming call is missed or declined,
 * synthesizes a grounded, human follow-up message in Karthik's voice, and sends it via WhatsApp/SMS.
 */
object MissedCallFollowupEngine {

    private const val TAG = "MissedCallFollowup"
    private val scope = CoroutineScope(Dispatchers.IO)

    // Cooldown per contact to avoid spamming multiple follow-ups (10 minutes)
    private val contactFollowupCooldown = java.util.concurrent.ConcurrentHashMap<String, Long>()

    /**
     * Checks if notification text / title represents a missed call event.
     * Matches WhatsApp missed voice/video/group calls and Phone dialer missed calls.
     */
    fun isMissedCallNotification(text: String, title: String = ""): Boolean {
        val combined = "$title $text".lowercase(Locale.ROOT)
        return combined.contains("missed voice call") ||
            combined.contains("missed video call") ||
            combined.contains("missed call") ||
            combined.contains("missed group call") ||
            combined.contains("missed audio call") ||
            combined.contains(Regex("""\b\d+\s+missed\s+calls?\b""")) ||
            combined.contains(Regex("""\bmissed\s+\d+\s+calls?\b"""))
    }

    /**
     * Extracts clean caller name from title and text.
     * Handles:
     * - Title: "Rahul", Text: "Missed voice call" -> "Rahul"
     * - Title: "Missed voice call", Text: "from Rahul" -> "Rahul"
     * - Title: "Missed voice call", Text: "Rahul at 12:30 PM" -> "Rahul"
     * - Title: "WhatsApp", Text: "Missed voice call from Rahul" -> "Rahul"
     * - Title: "Rahul (2)", Text: "Missed call" -> "Rahul"
     */
    fun extractCallerName(title: String, text: String): String {
        val tClean = title.trim()
        val textClean = text.trim()

        val isTitleIndicator = isMissedCallNotification(tClean) || tClean.equals("WhatsApp", ignoreCase = true)
        val rawName = if (isTitleIndicator) textClean else tClean

        // Strip prefixes: "Missed voice call from Rahul" -> "Rahul", "from Rahul" -> "Rahul"
        var candidate = rawName
            .replaceFirst(Regex("""(?i)^(missed\s+(voice\s+|video\s+|audio\s+|group\s+)?call(s)?\s*(from)?\s*)"""), "")
            .replaceFirst(Regex("""(?i)^\d+\s+missed\s+calls?\s*(from)?\s*"""), "")
            .replaceFirst(Regex("""(?i)^from\s+"""), "")
            .trim()

        // Strip timestamp suffixes: "Rahul at 12:45 PM", "Rahul - 12:45"
        candidate = candidate
            .replace(Regex("""(?i)\s+at\s+\d{1,2}:\d{2}(\s*[ap]m)?"""), "")
            .replace(Regex("""\s*[-–]\s*\d{1,2}:\d{2}(\s*[ap]m)?"""), "")
            .trim()

        // Strip count suffix: "Rahul (2)" -> "Rahul"
        candidate = candidate
            .replace(Regex("""\s*\(\d+\)$"""), "")
            .trim()

        return if (candidate.isNotBlank() && !isMissedCallNotification(candidate)) {
            candidate
        } else if (tClean.isNotBlank() && !isMissedCallNotification(tClean) && !tClean.equals("WhatsApp", ignoreCase = true)) {
            tClean
        } else {
            "Caller"
        }
    }

    /**
     * Handles a missed call notification event.
     */
    fun handleMissedCall(
        context: Context,
        callerName: String,
        callerNumber: String = "",
        sbn: StatusBarNotification? = null
    ): String? {
        val cleanName = callerName.trim()
        if (cleanName.isBlank()) return null

        // 1. Delegated Contact Authorization Check: Only send follow-ups to active, authorized contacts
        val delegatedManager = DelegatedContactManager.getInstance(context)
        var matchedContact = delegatedManager.findMatching(cleanName)
        if (matchedContact == null && callerNumber.isNotBlank()) {
            matchedContact = delegatedManager.findMatching(callerNumber)
        }
        if (matchedContact == null) {
            val resolvedNumber = resolveMissedCallNumberFromLog(context, cleanName)
            if (!resolvedNumber.isNullOrBlank()) {
                matchedContact = delegatedManager.findMatching(resolvedNumber)
            }
        }

        if (matchedContact == null || !matchedContact.isEnabled) {
            Log.d(TAG, "Skipping missed call follow-up: '$cleanName' is not in active delegated contacts")
            return null
        }

        if (matchedContact.isSessionBudgetExceeded()) {
            Log.w(TAG, "Session reply budget exceeded for delegated contact '${matchedContact.contactName}'. Skipping missed call follow-up.")
            return null
        }

        val now = System.currentTimeMillis()
        val lastSent = contactFollowupCooldown[cleanName.lowercase(Locale.ROOT)] ?: 0L
        if (now - lastSent < 10 * 60 * 1000L) {
            Log.d(TAG, "Skipping duplicate missed call follow-up for '$cleanName' (Cooldown active)")
            return null
        }

        val book = ContactKnowledgeBook.getInstance(context)
        val dossier = book.getDossier(cleanName)

        val rtcManager = com.zipper.app.core.intelligence.voice.TencentRtcManager.getInstance(context)
        val rtcConfig = rtcManager.getConfig()
        val webLink = if (rtcConfig.isEnabled) {
            val roomId = "zipper_${cleanName.lowercase(Locale.ROOT).replace(Regex("[^a-z0-9]"), "_").take(8)}_${now % 100000}"
            rtcManager.generateCallerWebLink(roomId, cleanName)
        } else null

        val baseReply = synthesizeFollowupMessage(cleanName, dossier, context, matchedContact.customContext)
        var generatedReply = if (!webLink.isNullOrBlank()) {
            "$baseReply\n\n🎙️ Or talk directly to my AI Voice Twin: $webLink"
        } else {
            baseReply
        }

        if (matchedContact.appendAiBadge) {
            generatedReply = "$generatedReply\n\n— Sent via Zipper AI on behalf of Karthik"
        }

        val sanitized = CommitmentGuard.sanitizeReply("Missed call", generatedReply)

        contactFollowupCooldown[cleanName.lowercase(Locale.ROOT)] = now

        // Dispatch silently via Android RemoteInput if notification actions support it (e.g. WhatsApp inline reply)
        var dispatchedInline = false
        val candidateSbn = sbn ?: DelegatedAutoReplyDispatcher.getActiveRemoteInputSbn(cleanName)
        if (candidateSbn != null) {
            dispatchedInline = com.zipper.app.core.remote.RemoteCommandDispatcher.trySendInlineReply(context, candidateSbn, sanitized)
            if (dispatchedInline) {
                Log.i(TAG, "✅ Dispatched missed call follow-up directly via RemoteInput to ${candidateSbn.packageName} for $cleanName: '$sanitized'")
                book.recordChatMessage(cleanName, sanitized, isFromUser = true)
                delegatedManager.recordReply(matchedContact.id)
            }
        }

        // If not sent via inline RemoteInput, try finding phone number and sending via WhatsApp automation
        if (!dispatchedInline) {
            val fallbackPhone = callerNumber.ifBlank {
                matchedContact.phoneNumber.ifBlank {
                    dossier?.phoneNumber.orEmpty().ifBlank {
                        resolveMissedCallNumberFromLog(context, cleanName).orEmpty().ifBlank {
                            com.zipper.app.core.remote.RemoteContactResolver.findMatchingContacts(context, cleanName).firstOrNull()?.normalizedPhone.orEmpty()
                        }
                    }
                }
            }
            if (fallbackPhone.isNotBlank()) {
                scope.launch {
                    val sent = com.zipper.app.core.remote.WhatsAppAutomationExecutor.executeSendToPhone(context, fallbackPhone, sanitized)
                    if (sent) {
                        Log.i(TAG, "✅ Dispatched missed call follow-up via WhatsApp direct automation to $cleanName ($fallbackPhone): '$sanitized'")
                        book.recordChatMessage(cleanName, sanitized, isFromUser = true)
                        delegatedManager.recordReply(matchedContact.id)
                    }
                }
            }
        }

        Log.i(TAG, "Generated missed call follow-up for '$cleanName' (inline: $dispatchedInline): '$sanitized'")

        // Post confirmation notification so user knows Zipper took care of it
        postFollowupNotification(context, cleanName, sanitized, dispatchedInline)

        return sanitized
    }

    fun synthesizeFollowupMessage(
        callerName: String,
        dossier: ContactDossier?,
        context: Context? = null,
        customContext: String = ""
    ): String {
        val style = dossier?.languageStyle ?: LanguageStyle.CASUAL_SLANG
        val rel = dossier?.relationship?.lowercase(Locale.ROOT) ?: "friend"
        val recentMsgs = dossier?.recentHistory?.map { it.text } ?: emptyList()
        val combinedRecent = recentMsgs.joinToString(" ").lowercase(Locale.ROOT)
        val cleanCustomContext = customContext.trim()
        val contextSnippet = if (cleanCustomContext.isNotBlank()) " ($cleanCustomContext)" else ""

        val isLandlordOrElder = rel.contains("landlord") || rel.contains("elder") || rel.contains("uncle") || rel.contains("professor") || rel.contains("aunty")
        val isMom = rel.contains("mom") || rel.contains("mother") || rel.contains("ma") || rel.contains("amma")
        val isDad = rel.contains("dad") || rel.contains("father") || rel.contains("appa")

        // 1. Contextual Project / Assignment / Notes discussion
        if (combinedRecent.contains("notes") || combinedRecent.contains("project") || combinedRecent.contains("assignment") || combinedRecent.contains("code")) {
            return when {
                style == LanguageStyle.KANNADA_REGIONAL -> "Hey $callerName, call nod-de. Project/work nalli iddini. Enadru urgent idya?"
                style == LanguageStyle.BROKEN_BASIC_ENGLISH -> "Hey bro saw your call. Tied up in work. Asking about notes or what?"
                style == LanguageStyle.CASUAL_SHORT -> "Saw call bro. About project or?"
                style == LanguageStyle.FORMAL_ENGLISH -> "Hello $callerName, saw your call. I am occupied with work right now. Is this regarding the project/notes?"
                else -> "Hey $callerName, saw your call, was occupied. Are you asking about the project / notes or something else?"
            }
        }

        // 2. Contextual Meeting / Plans / Gym discussion
        if (combinedRecent.contains("meet") || combinedRecent.contains("plans") || combinedRecent.contains("gym") || combinedRecent.contains("dinner") || combinedRecent.contains("evening")) {
            return when {
                style == LanguageStyle.KANNADA_REGIONAL -> "Hey $callerName, call nod-de. Swalpa busy iddini. Later plan on na?"
                style == LanguageStyle.BROKEN_BASIC_ENGLISH -> "Hey bro saw call. Busy now, still meeting later or what?"
                style == LanguageStyle.CASUAL_SHORT -> "Busy bro. On for later?"
                style == LanguageStyle.FORMAL_ENGLISH -> "Hello $callerName, saw your call. Tied up right now. Are we still on for later?"
                else -> "Hey $callerName, saw your call, tied up right now. Are we still on for later or what's up?"
            }
        }

        // 3. Fallback based on relationship and language style
        return when {
            isMom -> if (style == LanguageStyle.KANNADA_REGIONAL) {
                "Amma, call nod-de. Swalpa kelasa ide, innondu 15-20 mins nalli call madthini."
            } else {
                "Hey Mom, saw your missed call. A bit busy with work right now, will call you back in 20 mins ❤️"
            }
            isDad -> if (style == LanguageStyle.KANNADA_REGIONAL) {
                "Appa, call nod-de. Swalpa kelasa ide, thakshana call madthini."
            } else {
                "Hey Dad, saw your missed call. Occupied with some work right now, will call you back shortly."
            }
            isLandlordOrElder -> {
                val title = if (rel.contains("landlord")) "Sir" else if (rel.contains("aunty")) "Aunty" else "Uncle"
                "Hello $title, saw your call. Occupied right now$contextSnippet, will call you back shortly. Thank you."
            }
            style == LanguageStyle.KANNADA_REGIONAL -> "Hey $callerName, call nod-de. Swalpa busy iddini, en samachara? Text madi."
            style == LanguageStyle.CASUAL_SHORT -> if (cleanCustomContext.isNotBlank()) "Busy bro ($cleanCustomContext), ping here what's up" else "Busy bro, ping here what's up"
            style == LanguageStyle.BROKEN_BASIC_ENGLISH -> "Hey bro saw call. In work now$contextSnippet, text me what happened."
            style == LanguageStyle.FORMAL_ENGLISH -> "Hello $callerName, saw your call. I am currently occupied$contextSnippet and will get back to you shortly."
            else -> "Hey $callerName, saw your call, occupied right now$contextSnippet. What's up? Text me here 👍"
        }
    }

    private fun postFollowupNotification(
        context: Context,
        callerName: String,
        replyText: String,
        sentDirectly: Boolean = false
    ) {
        try {
            val nm = context.getSystemService(Context.NOTIFICATION_SERVICE) as? android.app.NotificationManager ?: return
            val channelId = "zipper_missed_call_followup"
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
                val chan = android.app.NotificationChannel(
                    channelId,
                    "Missed Call Follow-Up",
                    android.app.NotificationManager.IMPORTANCE_DEFAULT
                )
                nm.createNotificationChannel(chan)
            }

            val title = if (sentDirectly) "💬 Auto-replied to $callerName" else "💬 Follow-up ready for $callerName"
            val body = if (sentDirectly) {
                "Sent inline reply via WhatsApp to $callerName:\n\"$replyText\""
            } else {
                "Follow-up for $callerName:\n\"$replyText\""
            }

            val notif = androidx.core.app.NotificationCompat.Builder(context, channelId)
                .setSmallIcon(com.zipper.app.R.drawable.ic_launcher_foreground)
                .setContentTitle(title)
                .setContentText(replyText)
                .setStyle(androidx.core.app.NotificationCompat.BigTextStyle().bigText(body))
                .setAutoCancel(true)
                .build()

            nm.notify(callerName.hashCode(), notif)
        } catch (e: Exception) {
            Log.w(TAG, "Notification warning: ${e.message}")
        }
    }

    /**
     * Resolves the latest missed call phone number directly from Android CallLog
     * so contacts without manual entries can still receive the Voice Twin web link.
     */
    fun resolveMissedCallNumberFromLog(context: Context, callerName: String): String? {
        return try {
            val projection = arrayOf(
                android.provider.CallLog.Calls.NUMBER,
                android.provider.CallLog.Calls.CACHED_NAME,
                android.provider.CallLog.Calls.DATE,
                android.provider.CallLog.Calls.TYPE
            )
            val cursor = context.contentResolver.query(
                android.provider.CallLog.Calls.CONTENT_URI,
                projection,
                "${android.provider.CallLog.Calls.TYPE} = ?",
                arrayOf(android.provider.CallLog.Calls.MISSED_TYPE.toString()),
                "${android.provider.CallLog.Calls.DATE} DESC"
            )
            cursor?.use {
                if (it.moveToFirst()) {
                    val number = it.getString(it.getColumnIndexOrThrow(android.provider.CallLog.Calls.NUMBER))
                    val cachedName = it.getString(it.getColumnIndexOrThrow(android.provider.CallLog.Calls.CACHED_NAME))
                    val date = it.getLong(it.getColumnIndexOrThrow(android.provider.CallLog.Calls.DATE))
                    val isRecent = System.currentTimeMillis() - date < 300000L // 5 minutes
                    if (isRecent && (cachedName.isNullOrBlank() || cachedName.contains(callerName, ignoreCase = true) || callerName.contains(cachedName, ignoreCase = true))) {
                        Log.i(TAG, "Resolved missed call number from CallLog: $number (name: $cachedName)")
                        return number
                    }
                }
            }
            null
        } catch (e: Exception) {
            Log.w(TAG, "Could not resolve missed call from CallLog: ${e.message}")
            null
        }
    }
}
