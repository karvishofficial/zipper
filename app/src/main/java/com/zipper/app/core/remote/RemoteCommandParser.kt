package com.zipper.app.core.remote

import java.util.Locale

/**
 * Parses remote commands arriving via WhatsApp, SMS, or other notification channels.
 */
object RemoteCommandParser {

    // Matches "zipper: ...", "zipper #1234: ...", "hey zipper 1234: ...", "zipper, ..."
    // Also matches direct "#1234: ...", "#1234 ...", "#1234 - ...", "#12: ..."
    private val TRIGGER_PREFIX_REGEX = Regex(
        """(?i)^(?:(?:zipper|hey\s+zipper)(?:(?:\s*[:,-]\s*|\s+)(?:#([a-zA-Z0-9]{2,16})|([0-9]{2,8})))?|#([a-zA-Z0-9]{2,16}))(?:\s*[:,-]|\s+)\s*(.+)"""
    )

    // Pattern 1: "send <message> to <recipient> [on whatsapp]"
    // e.g. "send hi to shobha on whatsapp", "open whatsapp send hi to shobha"
    private val SEND_MSG_REGEX_1 = Regex(
        """(?i)^(?:open\s+whatsapp\s+(?:and\s+)?)?(?:send|message|text)\s+["']?(.+?)["']?\s+(?:to\s+)([a-zA-Z0-9\s]+?)(?:\s+on\s+whatsapp)?$"""
    )

    // Pattern 2: "send to <recipient> <message>" or "message <recipient> saying <message>"
    private val SEND_MSG_REGEX_2 = Regex(
        """(?i)^(?:open\s+whatsapp\s+(?:and\s+)?)?(?:send\s+to|message|text)\s+([a-zA-Z0-9\s]+?)(?:\s+saying|\s*:\s*|\s+that\s+|\s+)["']?(.+?)["']?$"""
    )

    // Shopping Pattern: "order <items> on <store>" or "buy <items> on <store>"
    private val SHOPPING_REGEX = Regex(
        """(?i)^(?:order|buy|get)\s+(.+?)(?:\s+on\s+(zepto|blinkit|amazon|instamart|swiggy))?$"""
    )

    // File Sharing Pattern: "send (my )?<file> [to <recipient>]"
    private val SEND_FILE_REGEX = Regex(
        """(?i)^(?:open\s+whatsapp\s+(?:and\s+)?)?(?:send|share|give)\s+(?:me\s+)?(?:my\s+)?(?:the\s+)?(.+?)(?:\s+(?:to|with)\s+([a-zA-Z0-9\s]+?))?(?:\s+on\s+whatsapp)?$"""
    )

    private val FILE_INDICATORS = setOf(
        "pdf", "doc", "docx", "file", "document", "sheet", "excel", "xlsx",
        "marks card", "markscard", "hall ticket", "hallticket", "admit card",
        "resume", "cv", "photo", "image", "pic", "certificate", "id card",
        "aadhaar", "pan card", "passport", "invoice", "receipt", "notes",
        "png", "jpg", "jpeg", "webp", "gif", "screen", "screenshot", "mp4", "mp3", "apk", "zip"
    )

    fun isTriggerCommand(text: String, sender: String? = null): Boolean {
        if (TRIGGER_PREFIX_REGEX.containsMatchIn(text.trim())) return true
        if (sender != null) {
            val pending = PendingRemoteSessionStore.getPending(sender)
            if (pending != null && parseSelectionIndex(text.trim(), pending) != null) {
                return true
            }
        }
        return false
    }

    fun parse(rawText: String, sender: String, sourceApp: String): RemoteCommand? {
        // Check if there is an active pending contact or file disambiguation session
        val pendingSession = PendingRemoteSessionStore.getPending(sender)
        if (pendingSession != null) {
            val selectedIdx = parseSelectionIndex(rawText.trim(), pendingSession)
            if (selectedIdx != null) {
                if (pendingSession.fileCandidates.isNotEmpty() && selectedIdx in pendingSession.fileCandidates.indices) {
                    val file = pendingSession.fileCandidates[selectedIdx]
                    return RemoteCommand(
                        rawText = rawText,
                        sender = sender,
                        sourceApp = sourceApp,
                        actionType = RemoteActionType.CONFIRM_FILE_SELECTION,
                        targetApp = "com.whatsapp",
                        recipient = pendingSession.targetRecipient,
                        selectedFileIndex = selectedIdx,
                        resolvedFile = file,
                        rawGoal = "Send file '${file.name}' to ${pendingSession.targetRecipient ?: sender}"
                    )
                } else if (pendingSession.candidates.isNotEmpty() && selectedIdx in pendingSession.candidates.indices) {
                    val candidate = pendingSession.candidates[selectedIdx]
                    return RemoteCommand(
                        rawText = rawText,
                        sender = sender,
                        sourceApp = sourceApp,
                        actionType = RemoteActionType.CONFIRM_CONTACT_SELECTION,
                        targetApp = "com.whatsapp",
                        recipient = candidate.name,
                        messageBody = pendingSession.messageText,
                        selectedContactIndex = selectedIdx,
                        resolvedContact = candidate,
                        rawGoal = "Send '${pendingSession.messageText}' to ${candidate.name} (${candidate.phoneNumber})"
                    )
                }
            }
        }

        val match = TRIGGER_PREFIX_REGEX.find(rawText.trim()) ?: return null
        val pin1 = match.groupValues[1].takeIf { it.isNotBlank() }
        val pin2 = match.groupValues[2].takeIf { it.isNotBlank() }
        val pin3 = match.groupValues[3].takeIf { it.isNotBlank() }
        val pinProvided = pin1 ?: pin2 ?: pin3
        val payload = match.groupValues[4].trim()
        val lower = payload.lowercase(Locale.ROOT)

        // 0. Check for Time-Based Scheduled Messaging (e.g. "at 2:30 send hello to bhavani", "send hi to bhavani at 2:30pm")
        val scheduledCmd = parseScheduledCommand(payload, rawText, sender, sourceApp, pinProvided)
        if (scheduledCmd != null) return scheduledCmd

        // 1. Emergency Remote Background Utilities
        // A. Ring / Find My Phone
        if (lower in listOf("ring", "ring phone", "find phone", "find my phone", "alarm", "siren", "loud", "ring my phone", "ring the phone", "sound alarm") ||
            lower.startsWith("ring ") || lower.startsWith("alarm ") || lower == "alarm"
        ) {
            return RemoteCommand(
                rawText = rawText,
                sender = sender,
                sourceApp = sourceApp,
                actionType = RemoteActionType.FIND_MY_PHONE_RING,
                pinProvided = pinProvided,
                rawGoal = payload
            )
        }

        // B. Stop Ringing
        if (lower in listOf("stop", "stop ring", "stop ringing", "stop alarm", "silent", "silence", "quiet")) {
            return RemoteCommand(
                rawText = rawText,
                sender = sender,
                sourceApp = sourceApp,
                actionType = RemoteActionType.FIND_MY_PHONE_STOP,
                pinProvided = pinProvided,
                rawGoal = payload
            )
        }

        // C. Battery & Device Status
        if (lower in listOf("battery", "battery status", "status", "charge", "battery level", "power", "battery info") ||
            lower.startsWith("battery ")
        ) {
            return RemoteCommand(
                rawText = rawText,
                sender = sender,
                sourceApp = sourceApp,
                actionType = RemoteActionType.GET_BATTERY_STATUS,
                pinProvided = pinProvided,
                rawGoal = payload
            )
        }

        // D. GPS Location
        if (lower in listOf("location", "loc", "gps", "where is my phone", "where is the phone", "where's my phone", "where is phone", "where are you", "locate", "locate phone", "find device", "track phone", "coords") ||
            lower.startsWith("where is ") || lower.startsWith("where's ") || lower.startsWith("locate ")
        ) {
            return RemoteCommand(
                rawText = rawText,
                sender = sender,
                sourceApp = sourceApp,
                actionType = RemoteActionType.GET_LOCATION,
                pinProvided = pinProvided,
                rawGoal = payload
            )
        }

        // D2. Silent Intruder Photo Capture
        if (lower in listOf("snap", "photo", "pic", "take photo", "take pic", "camera", "intruder", "front photo", "capture photo", "snap photo") ||
            lower.startsWith("snap ") || lower.startsWith("take photo")
        ) {
            return RemoteCommand(
                rawText = rawText,
                sender = sender,
                sourceApp = sourceApp,
                actionType = RemoteActionType.SNAP_INTRUDER,
                pinProvided = pinProvided,
                rawGoal = payload
            )
        }

        // D3. Full Protection / The Mimic (Fake Shutdown & Full Lockdown)
        if (lower in listOf("protect", "full protect", "full protection", "lockdown", "lost", "phone lost", "stolen", "protect phone", "mimic", "fake shutdown", "fake off", "stealth", "stealth mode", "black screen") ||
            lower.startsWith("protect") || lower.startsWith("lockdown") || lower.startsWith("mimic ") || lower.startsWith("stealth ")
        ) {
            val emailMatch = Regex("""[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}""").find(payload)
            val customEmail = emailMatch?.value?.trim()

            return RemoteCommand(
                rawText = rawText,
                sender = sender,
                sourceApp = sourceApp,
                actionType = RemoteActionType.ACTIVATE_MIMIC,
                pinProvided = pinProvided,
                recipient = customEmail,
                rawGoal = payload
            )
        }

        // D4. Disarm Mimic
        if (lower in listOf("disarm", "disarm mimic", "disarm stealth", "wake", "wake up", "exit mimic", "exit stealth")) {
            return RemoteCommand(
                rawText = rawText,
                sender = sender,
                sourceApp = sourceApp,
                actionType = RemoteActionType.DISARM_MIMIC,
                pinProvided = pinProvided,
                rawGoal = payload
            )
        }

        // E. Last OTP Retrieval
        if (lower in listOf("last otp", "get otp", "otp", "show otp", "forward otp", "bank otp", "latest otp") ||
            lower.startsWith("last otp") || lower.startsWith("get otp") || lower.startsWith("what was my last otp")
        ) {
            return RemoteCommand(
                rawText = rawText,
                sender = sender,
                sourceApp = sourceApp,
                actionType = RemoteActionType.GET_LAST_OTP,
                pinProvided = pinProvided,
                rawGoal = payload
            )
        }

        // F. Screen Memory Recall (Rewind)
        if (lower.startsWith("recall ") || lower.startsWith("memory ") || lower.startsWith("find screen ") ||
            lower.startsWith("what did i see") || lower.startsWith("rewind ")
        ) {
            val query = payload
                .removePrefix("recall ")
                .removePrefix("memory ")
                .removePrefix("find screen ")
                .removePrefix("rewind ")
                .removePrefix("what did i see on ")
                .removePrefix("what did i see in ")
                .removePrefix("what did i see ")
                .trim()
            return RemoteCommand(
                rawText = rawText,
                sender = sender,
                sourceApp = sourceApp,
                actionType = RemoteActionType.RECALL_SCREEN_MEMORY,
                pinProvided = pinProvided,
                rawGoal = query
            )
        }

        // F2. Cross-App Cart Arbitrage
        if (lower.startsWith("compare cart") || lower.startsWith("cart compare") || lower.startsWith("arbitrage") ||
            lower.startsWith("which store is cheaper") || lower.startsWith("check cart") || lower.startsWith("cheapest cart")
        ) {
            val q = payload.replace(Regex("""(?i)^(?:compare\s+cart|cart\s+compare|arbitrage|which\s+store\s+is\s+cheaper|check\s+cart|cheapest\s+cart)[:\s]*"""), "").trim()
            return RemoteCommand(
                rawText = rawText,
                sender = sender,
                sourceApp = sourceApp,
                actionType = RemoteActionType.EVALUATE_CART_ARBITRAGE,
                pinProvided = pinProvided,
                shoppingQuery = q.ifBlank { null },
                rawGoal = payload
            )
        }

        // G. Auto-Expense Ledger
        if (lower in listOf("expenses", "expense", "today expense", "today expenses", "expenses today", "today's expenses", "todays expenses", "spent today", "spending", "budget", "ledger") ||
            lower.startsWith("how much did i spend") || lower.startsWith("expense today") || lower.startsWith("expenses today") || lower.startsWith("daily expense")
        ) {
            return RemoteCommand(
                rawText = rawText,
                sender = sender,
                sourceApp = sourceApp,
                actionType = RemoteActionType.GET_EXPENSE_SUMMARY,
                pinProvided = pinProvided,
                rawGoal = payload
            )
        }

        // G. Sentry Anti-Theft Guard
        if (lower in listOf("arm sentry", "sentry on", "guard phone", "enable sentry", "lock sentry")) {
            return RemoteCommand(
                rawText = rawText,
                sender = sender,
                sourceApp = sourceApp,
                actionType = RemoteActionType.ARM_SENTRY,
                pinProvided = pinProvided,
                rawGoal = payload
            )
        }
        if (lower in listOf("disarm sentry", "sentry off", "disable sentry", "stop sentry")) {
            return RemoteCommand(
                rawText = rawText,
                sender = sender,
                sourceApp = sourceApp,
                actionType = RemoteActionType.DISARM_SENTRY,
                pinProvided = pinProvided,
                rawGoal = payload
            )
        }

        // H. Commute / SOS Safety Guard
        if (lower in listOf("sos", "emergency", "help", "send help", "alert sos", "trigger sos")) {
            return RemoteCommand(
                rawText = rawText,
                sender = sender,
                sourceApp = sourceApp,
                actionType = RemoteActionType.TRIGGER_SOS,
                pinProvided = pinProvided,
                rawGoal = payload
            )
        }

        if (lower in listOf("safe", "im safe", "i am safe", "arrived", "reached", "safe arrived", "arrived safely", "safely arrived", "disarm commute", "stop commute")) {
            return RemoteCommand(
                rawText = rawText,
                sender = sender,
                sourceApp = sourceApp,
                actionType = RemoteActionType.STOP_COMMUTE_GUARD,
                pinProvided = pinProvided,
                rawGoal = payload
            )
        }

        if (lower.startsWith("commute ") || lower.startsWith("guard commute") || lower.startsWith("ride guard") || lower.startsWith("travel to")) {
            val dest = payload.replace(Regex("""(?i)^(?:commute|guard\s+commute|ride\s+guard|travel)\s*(?:to)?\s*"""), "").trim()
            return RemoteCommand(
                rawText = rawText,
                sender = sender,
                sourceApp = sourceApp,
                actionType = RemoteActionType.START_COMMUTE_GUARD,
                pinProvided = pinProvided,
                rawGoal = dest.ifBlank { "Destination" }
            )
        }

        // I. Contact Knowledge & Dossier Teaching
        if (lower.startsWith("remember ") || lower.startsWith("note about ") || lower.startsWith("note down about ")) {
            return RemoteCommand(
                rawText = rawText,
                sender = sender,
                sourceApp = sourceApp,
                actionType = RemoteActionType.REMEMBER_CONTACT_KNOWLEDGE,
                pinProvided = pinProvided,
                rawGoal = payload
            )
        }


        // 2. Check for File Sharing (e.g. "send 12th marks card pdf", "share resume to rahul")
        if (isFileCommand(payload)) {
            val fileCmd = parseFileCommand(payload, rawText, sender, sourceApp, pinProvided)
            if (fileCmd != null) return fileCmd
        }

        // 3. Check for WhatsApp Messaging
        if (payload.contains("whatsapp", ignoreCase = true) ||
            payload.startsWith("send ", ignoreCase = true) ||
            payload.startsWith("message ", ignoreCase = true) ||
            payload.startsWith("text ", ignoreCase = true)
        ) {
            val m1 = SEND_MSG_REGEX_1.find(payload)
            if (m1 != null) {
                val message = m1.groupValues[1].trim()
                val recipient = m1.groupValues[2].trim()
                return RemoteCommand(
                    rawText = rawText,
                    sender = sender,
                    sourceApp = sourceApp,
                    actionType = RemoteActionType.SEND_WHATSAPP_MESSAGE,
                    pinProvided = pinProvided,
                    targetApp = "com.whatsapp",
                    recipient = recipient,
                    messageBody = message,
                    rawGoal = payload
                )
            }

            val m2 = SEND_MSG_REGEX_2.find(payload)
            if (m2 != null) {
                val recipient = m2.groupValues[1].trim()
                val message = m2.groupValues[2].trim()
                return RemoteCommand(
                    rawText = rawText,
                    sender = sender,
                    sourceApp = sourceApp,
                    actionType = RemoteActionType.SEND_WHATSAPP_MESSAGE,
                    pinProvided = pinProvided,
                    targetApp = "com.whatsapp",
                    recipient = recipient,
                    messageBody = message,
                    rawGoal = payload
                )
            }
        }

        // 4. Check for Shopping Request
        val shopMatch = SHOPPING_REGEX.find(payload)
        if (shopMatch != null) {
            val query = shopMatch.groupValues[1].trim()
            val store = shopMatch.groupValues[2].takeIf { it.isNotBlank() } ?: "Zepto"
            return RemoteCommand(
                rawText = rawText,
                sender = sender,
                sourceApp = sourceApp,
                actionType = RemoteActionType.EXECUTE_SHOPPING,
                pinProvided = pinProvided,
                targetApp = store,
                shoppingQuery = query,
                rawGoal = payload
            )
        }

        // 5. Check for Order Status query
        if (lower.contains("where is my order") || lower.contains("track swiggy") || lower.contains("track zomato")) {
            return RemoteCommand(
                rawText = rawText,
                sender = sender,
                sourceApp = sourceApp,
                actionType = RemoteActionType.QUERY_STATUS_OTP,
                pinProvided = pinProvided,
                rawGoal = payload
            )
        }

        // 6. Fallback: General Agent Goal
        return RemoteCommand(
            rawText = rawText,
            sender = sender,
            sourceApp = sourceApp,
            actionType = RemoteActionType.GENERAL_AGENT_GOAL,
            pinProvided = pinProvided,
            rawGoal = payload
        )
    }

    private fun isFileCommand(payload: String): Boolean {
        val lower = payload.lowercase(Locale.ROOT)
        return FILE_INDICATORS.any { indicator ->
            lower.contains(indicator)
        } || Regex("""\.\s*[a-zA-Z0-9]{2,5}\b""").containsMatchIn(lower)
    }

    private fun parseFileCommand(
        payload: String,
        rawText: String,
        sender: String,
        sourceApp: String,
        pinProvided: String?
    ): RemoteCommand? {
        val match = SEND_FILE_REGEX.find(payload) ?: return null
        val rawFileQuery = match.groupValues[1].trim()
        val recipient = match.groupValues[2].trim().takeIf { it.isNotBlank() }

        val cleanQuery = rawFileQuery
            .replace(Regex("""^["']|["']$"""), "")
            .replace(Regex("""\.\s+([a-zA-Z0-9]{2,5})\b"""), ".$1")
            .trim()
        if (cleanQuery.isBlank()) return null

        return RemoteCommand(
            rawText = rawText,
            sender = sender,
            sourceApp = sourceApp,
            actionType = RemoteActionType.SEND_FILE,
            pinProvided = pinProvided,
            targetApp = "com.whatsapp",
            recipient = recipient,
            fileQuery = cleanQuery,
            rawGoal = payload
        )
    }

    fun parseTimeToEpochMs(timeStr: String): Long? {
        val clean = timeStr.trim().lowercase(Locale.ROOT)
        val regex = Regex("""^(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$""")
        val match = regex.find(clean) ?: return null

        var hour = match.groupValues[1].toIntOrNull() ?: return null
        val minute = match.groupValues[2].takeIf { it.isNotBlank() }?.toIntOrNull() ?: 0
        val amPm = match.groupValues[3]

        if (amPm == "pm" && hour < 12) hour += 12
        if (amPm == "am" && hour == 12) hour = 0

        val now = java.util.Calendar.getInstance()
        if (amPm.isBlank() && hour < 12) {
            val currentHour = now.get(java.util.Calendar.HOUR_OF_DAY)
            if (currentHour in 1..11 && hour in 1..currentHour) {
                // morning has passed this hour, try afternoon
                hour += 12
            } else if (currentHour >= 12 && hour + 12 > currentHour) {
                // afternoon time today
                hour += 12
            }
        }

        val targetCal = java.util.Calendar.getInstance().apply {
            set(java.util.Calendar.HOUR_OF_DAY, hour)
            set(java.util.Calendar.MINUTE, minute)
            set(java.util.Calendar.SECOND, 0)
            set(java.util.Calendar.MILLISECOND, 0)
        }

        if (targetCal.before(now)) {
            targetCal.add(java.util.Calendar.DAY_OF_YEAR, 1)
        }

        return targetCal.timeInMillis
    }

    private fun parseScheduledCommand(
        payload: String,
        rawText: String,
        sender: String,
        sourceApp: String,
        pinProvided: String?
    ): RemoteCommand? {
        // Pattern A: "at 2:30 [pm] send <msg> to <recipient>" or "schedule at 2:30 [pm] send <msg> to <recipient>"
        val atRegex = Regex("""(?i)^(?:schedule\s+)?(?:at\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?))\s*[:,-]?\s*(.+)$""")
        val atMatch = atRegex.find(payload)
        if (atMatch != null) {
            val timeStr = atMatch.groupValues[1].trim()
            val subPayload = atMatch.groupValues[2].trim()
            val scheduledEpochMs = parseTimeToEpochMs(timeStr)
            if (scheduledEpochMs != null) {
                val subCmd = parseMessagingSubPayload(subPayload, rawText, sender, sourceApp, pinProvided, scheduledEpochMs)
                if (subCmd != null) return subCmd
            }
        }

        // Pattern B: "in <N> mins/hours send <msg> to <recipient>"
        val inRegex = Regex("""(?i)^(?:schedule\s+)?(?:in\s+(\d+)\s*(mins|min|minutes|minute|hours|hour|hrs|hr))\s*[:,-]?\s*(.+)$""")
        val inMatch = inRegex.find(payload)
        if (inMatch != null) {
            val count = inMatch.groupValues[1].toIntOrNull() ?: 10
            val unit = inMatch.groupValues[2].lowercase(Locale.ROOT)
            val delayMs = if (unit.startsWith("h")) count * 3600000L else count * 60000L
            val scheduledEpochMs = System.currentTimeMillis() + delayMs
            val subPayload = inMatch.groupValues[3].trim()
            val subCmd = parseMessagingSubPayload(subPayload, rawText, sender, sourceApp, pinProvided, scheduledEpochMs)
            if (subCmd != null) return subCmd
        }

        // Pattern C: "send <msg> to <recipient> at 2:30 [pm]"
        val trailingAtRegex = Regex("""(?i)^(.+?)\s+(?:at\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?))$""")
        val trailingMatch = trailingAtRegex.find(payload)
        if (trailingMatch != null) {
            val subPayload = trailingMatch.groupValues[1].trim()
            val timeStr = trailingMatch.groupValues[2].trim()
            val scheduledEpochMs = parseTimeToEpochMs(timeStr)
            if (scheduledEpochMs != null) {
                val subCmd = parseMessagingSubPayload(subPayload, rawText, sender, sourceApp, pinProvided, scheduledEpochMs)
                if (subCmd != null) return subCmd
            }
        }

        return null
    }

    private fun parseMessagingSubPayload(
        subPayload: String,
        rawText: String,
        sender: String,
        sourceApp: String,
        pinProvided: String?,
        scheduledEpochMs: Long
    ): RemoteCommand? {
        val m1 = SEND_MSG_REGEX_1.find(subPayload)
        if (m1 != null) {
            return RemoteCommand(
                rawText = rawText,
                sender = sender,
                sourceApp = sourceApp,
                actionType = RemoteActionType.SCHEDULE_MESSAGE,
                pinProvided = pinProvided,
                targetApp = "com.whatsapp",
                recipient = m1.groupValues[2].trim(),
                messageBody = m1.groupValues[1].trim(),
                scheduledTimeMs = scheduledEpochMs,
                rawGoal = subPayload
            )
        }

        val m2 = SEND_MSG_REGEX_2.find(subPayload)
        if (m2 != null) {
            return RemoteCommand(
                rawText = rawText,
                sender = sender,
                sourceApp = sourceApp,
                actionType = RemoteActionType.SCHEDULE_MESSAGE,
                pinProvided = pinProvided,
                targetApp = "com.whatsapp",
                recipient = m2.groupValues[1].trim(),
                messageBody = m2.groupValues[2].trim(),
                scheduledTimeMs = scheduledEpochMs,
                rawGoal = subPayload
            )
        }

        // Fallback for simple "send hello bhavani to bhavani"
        val simpleSendRegex = Regex("""(?i)^(?:send|message|text)\s+["']?(.+?)["']?\s+(?:to\s+)([a-zA-Z0-9\s]+?)$""")
        val mSimple = simpleSendRegex.find(subPayload)
        if (mSimple != null) {
            return RemoteCommand(
                rawText = rawText,
                sender = sender,
                sourceApp = sourceApp,
                actionType = RemoteActionType.SCHEDULE_MESSAGE,
                pinProvided = pinProvided,
                targetApp = "com.whatsapp",
                recipient = mSimple.groupValues[2].trim(),
                messageBody = mSimple.groupValues[1].trim(),
                scheduledTimeMs = scheduledEpochMs,
                rawGoal = subPayload
            )
        }

        return null
    }

    fun parseSelectionIndex(text: String, session: PendingDisambiguationSession): Int? {
        val clean = text.replace(Regex("""(?i)^(?:(?:zipper|hey\s+zipper)(?:\s+#?[a-zA-Z0-9]{4,8})?|#[a-zA-Z0-9]{4,8})(?:\s*[:,-]|\s+)\s*"""), "").trim()

        val totalCandidates = if (session.fileCandidates.isNotEmpty()) session.fileCandidates.size else session.candidates.size

        // 1. Check for standalone or trailing number: "1", "2", "option 2", "#2"
        val numMatch = Regex("""(?i)(?:option\s+|choice\s+|#)?(\d+)\b""").findAll(clean).lastOrNull()
        if (numMatch != null) {
            val num = numMatch.groupValues[1].toIntOrNull()
            if (num != null && num in 1..totalCandidates) {
                return num - 1 // 0-based
            }
        }

        // 2. Check if clean text matches any candidate name
        if (session.fileCandidates.isNotEmpty()) {
            session.fileCandidates.forEachIndexed { idx, f ->
                if (clean.equals(f.name, ignoreCase = true) ||
                    (clean.length >= 3 && f.name.contains(clean, ignoreCase = true))) {
                    return idx
                }
            }
        } else {
            session.candidates.forEachIndexed { idx, candidate ->
                if (clean.equals(candidate.name, ignoreCase = true) ||
                    (clean.length >= 3 && candidate.name.contains(clean, ignoreCase = true)) ||
                    (clean.length >= 4 && candidate.phoneNumber.contains(clean))) {
                    return idx
                }
            }
        }

        return null
    }
}
