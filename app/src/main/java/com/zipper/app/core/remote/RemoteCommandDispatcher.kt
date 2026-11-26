package com.zipper.app.core.remote

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.RemoteInput
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.service.notification.StatusBarNotification
import android.util.Log
import androidx.core.app.NotificationCompat
import com.zipper.app.R
import com.zipper.app.core.notification.KarvishNotificationItem
import com.zipper.app.core.notification.NotificationCategory
import com.zipper.app.core.notification.NotificationQueryEngine
import com.zipper.app.core.notification.NotificationStorageManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

/**
 * Dispatches and executes remote autonomous commands triggered by notifications.
 * Supports 100% background silent operations (Ring Phone, Battery, Location, OTP)
 * and silent inline reply via Android Notification RemoteInput.
 */
object RemoteCommandDispatcher {

    private const val TAG = "RemoteCommandDispatcher"
    private const val CHANNEL_ID = "karvish_remote_actions"
    private val scope = CoroutineScope(Dispatchers.Default)

    fun handleIncomingNotification(
        context: Context,
        sender: String,
        sourceApp: String,
        rawText: String,
        sbn: StatusBarNotification? = null
    ): Boolean {
        if (!RemoteCommandParser.isTriggerCommand(rawText, sender)) {
            return false
        }

        val command = RemoteCommandParser.parse(rawText, sender, sourceApp) ?: return false

        // Security authorization check (PIN or Trusted Whitelist)
        val isDisambiguation = command.actionType == RemoteActionType.CONFIRM_CONTACT_SELECTION ||
            command.actionType == RemoteActionType.CONFIRM_FILE_SELECTION
        if (!isDisambiguation && !RemoteSecurityGate.isAuthorized(context, sender, command.pinProvided)) {
            Log.w(TAG, "Blocked unauthorized remote trigger from sender: '$sender' (Invalid PIN or untrusted contact)")
            postConfirmationNotification(
                context,
                "Karvish Security Guard",
                "Blocked remote command from '$sender'. PIN verification failed or untrusted contact."
            )
            return false
        }

        Log.i(TAG, "Authorized remote command received: ${command.actionType} from '$sender' (PIN verified: ${command.pinProvided != null})")

        scope.launch {
            executeCommand(context, command, sbn)
        }

        return true
    }

    private suspend fun executeCommand(
        context: Context,
        command: RemoteCommand,
        sbn: StatusBarNotification?
    ) {
        when (command.actionType) {
            RemoteActionType.FIND_MY_PHONE_RING -> {
                EmergencyRinger.startRinging(context)
                val reply = "🔊 Phone is ringing at maximum volume! Send 'Zipper: stop' or touch phone to silence."
                replySilently(context, sbn, reply)
                recordToAuditStorage(context, command, "TRIGGERED_EMERGENCY_RING")
            }

            RemoteActionType.FIND_MY_PHONE_STOP -> {
                EmergencyRinger.stopRinging(context)
                val reply = "🔇 Emergency alarm silenced."
                replySilently(context, sbn, reply)
                recordToAuditStorage(context, command, "STOPPED_EMERGENCY_RING")
            }

            RemoteActionType.GET_BATTERY_STATUS -> {
                val reply = RemoteSystemDiagnostics.getBatteryAndNetworkStatus(context)
                replySilently(context, sbn, reply)
                recordToAuditStorage(context, command, "SENT_BATTERY_STATUS: $reply")
            }

            RemoteActionType.GET_LOCATION -> {
                com.zipper.app.core.security.HardwareEnforcer.enforceStealthConnectivity(context)
                val reply = RemoteSystemDiagnostics.getLocationUrl(context)
                replySilently(context, sbn, reply)
                recordToAuditStorage(context, command, "SENT_LOCATION: $reply")
            }

            RemoteActionType.SNAP_INTRUDER -> {
                com.zipper.app.core.security.HardwareEnforcer.enforceStealthConnectivity(context)
                com.zipper.app.core.security.IntruderCaptureService.start(context, "REMOTE_COMMAND_${command.sender}")
                val reply = "📸 Headless intruder mugshot & GPS capture triggered! Telemetry dispatching to Telegram."
                replySilently(context, sbn, reply)
                recordToAuditStorage(context, command, "TRIGGERED_INTRUDER_CAPTURE")
            }

            RemoteActionType.ACTIVATE_MIMIC -> {
                if (!command.recipient.isNullOrBlank() && command.recipient.contains("@")) {
                    com.zipper.app.core.security.SecurityConfig.setAlertEmail(context, command.recipient.trim())
                    Log.i(TAG, "Custom alert destination email updated to: ${command.recipient.trim()}")
                }
                val sessionId = "SESSION_A26_" + System.currentTimeMillis().toString().takeLast(6)
                com.zipper.app.core.security.SecurityConfig.setActiveTrackingSessionId(context, sessionId)
                com.zipper.app.core.security.HardwareEnforcer.enforceStealthConnectivity(context)
                com.zipper.app.core.security.FakeShutdownOverlayActivity.start(context)
                com.zipper.app.core.security.IntruderCaptureService.start(context, "REMOTE_COMMAND_${command.sender}")
                val target = com.zipper.app.core.security.SecurityConfig.getAlertEmail(context)
                val netlifyBase = com.zipper.app.core.security.SecurityConfig.getNetlifyBaseUrl(context)
                val radarLink = "$netlifyBase/?session=$sessionId#key=1234"
                val reply = "🕶️ Mimic Mode ACTIVATED.\n🔴 Live Radar: $radarLink\nPhotos & GPS sent to $target. Disarm with 3x Volume-Up or '#1234: disarm'."
                replySilently(context, sbn, reply)
                recordToAuditStorage(context, command, "ACTIVATED_MIMIC_SHUTDOWN")
            }

            RemoteActionType.DISARM_MIMIC -> {
                com.zipper.app.core.security.SecurityConfig.setLiveTrackingActive(context, false)
                com.zipper.app.core.security.IntruderCaptureService.stopTracking(context)
                com.zipper.app.core.security.FakeShutdownOverlayActivity.disarm(context)
                val reply = "☀️ Mimic & Protection Mode DISARMED. Tracking and cameras stopped. Phone restored to normal."
                replySilently(context, sbn, reply)
                recordToAuditStorage(context, command, "DISARMED_MIMIC_SHUTDOWN")
            }

            RemoteActionType.GET_LAST_OTP -> {
                val reply = RemoteSystemDiagnostics.getLastOtp(context)
                replySilently(context, sbn, reply)
                recordToAuditStorage(context, command, "SENT_LAST_OTP")
            }

            RemoteActionType.QUERY_STATUS_OTP -> {
                val localRes = NotificationQueryEngine.handleLocalNotificationQuery(context, command.rawGoal)
                val reply = localRes?.markdownResponse ?: "No active status found."
                replySilently(context, sbn, reply.replace(Regex("""[*`#]"""), ""))
                recordToAuditStorage(context, command, "SENT_STATUS_QUERY")
            }

            RemoteActionType.GET_EXPENSE_SUMMARY -> {
                val summary = com.zipper.app.core.plugins.BackgroundPluginManager.getInstance(context).expenseLedger.getTodaySummary()
                val reply = if (summary.transactionCount == 0) {
                    "📊 No expenses recorded today yet."
                } else {
                    buildString {
                        append("📊 Today's Expense Summary:\n")
                        append("💰 Total Spent: ₹${"%,.2f".format(summary.totalAmount)} (${summary.transactionCount} transactions)\n\n")
                        append("📂 Breakdown:\n")
                        summary.categoryTotals.forEach { (cat, amt) ->
                            append("• ${cat.name}: ₹${"%,.0f".format(amt)}\n")
                        }
                        summary.topMerchant?.let { append("\n🏆 Top Merchant: $it") }
                    }
                }
                replySilently(context, sbn, reply)
                recordToAuditStorage(context, command, "SENT_EXPENSE_SUMMARY")
            }

            RemoteActionType.ARM_SENTRY -> {
                com.zipper.app.core.plugins.BackgroundPluginManager.getInstance(context).armSentryGuard(
                    onAlarm = { reason ->
                        postConfirmationNotification(context, "🚨 Sentry Intruder Alert", reason)
                    }
                )
                val reply = "🛡️ Sentry Guard ARMED. Phone will alarm if moved or unplugged."
                replySilently(context, sbn, reply)
                recordToAuditStorage(context, command, "ARMED_SENTRY")
            }

            RemoteActionType.EVALUATE_CART_ARBITRAGE -> {
                val report = com.zipper.app.core.web.features.CartArbitrageEngine.evaluateBasketArbitrage(
                    fallbackText = command.shoppingQuery ?: command.rawGoal
                )
                val cleanReply = report.summaryMarkdown.replace(Regex("""[*`#]"""), "").trim()
                replySilently(context, sbn, cleanReply)
                recordToAuditStorage(context, command, "SENT_CART_ARBITRAGE")
            }

            RemoteActionType.DISARM_SENTRY -> {
                com.zipper.app.core.plugins.BackgroundPluginManager.getInstance(context).disarmSentryGuard()
                val reply = "🛡️ Sentry Guard DISARMED."
                replySilently(context, sbn, reply)
                recordToAuditStorage(context, command, "DISARMED_SENTRY")
            }

            RemoteActionType.RECALL_SCREEN_MEMORY -> {
                val query = command.rawGoal.ifBlank { "recent" }
                val summary = com.zipper.app.core.intelligence.memory.ScreenMemoryStore.formatMemorySummary(query)
                replySilently(context, sbn, summary)
                recordToAuditStorage(context, command, "RECALLED_SCREEN_MEMORY")
            }

            RemoteActionType.START_COMMUTE_GUARD -> {
                val dest = command.rawGoal.ifBlank { "Destination" }
                val session = com.zipper.app.core.intelligence.commute.CommuteSafetyGuard.startGuard(
                    context = context,
                    destination = dest,
                    durationMinutes = 35
                )
                val reply = "🛡️ Commute Safety Guard ARMED: En route to $dest (~35 mins). Periodic delay monitoring and 1-tap SOS active."
                replySilently(context, sbn, reply)
                recordToAuditStorage(context, command, "ARMED_COMMUTE_GUARD: $dest")
            }

            RemoteActionType.STOP_COMMUTE_GUARD -> {
                val reply = com.zipper.app.core.intelligence.commute.CommuteSafetyGuard.markSafeArrival(context)
                replySilently(context, sbn, reply)
                recordToAuditStorage(context, command, "STOPPED_COMMUTE_GUARD")
            }

            RemoteActionType.TRIGGER_SOS -> {
                val reply = com.zipper.app.core.intelligence.commute.CommuteSafetyGuard.triggerSos(context, "Remote SOS Command from ${command.sender}")
                replySilently(context, sbn, reply)
                recordToAuditStorage(context, command, "TRIGGERED_EMERGENCY_SOS")
            }

            RemoteActionType.REMEMBER_CONTACT_KNOWLEDGE -> {
                val parsed = com.zipper.app.core.intelligence.reply.ContactKnowledgeBook.getInstance(context)
                    .parseNaturalRememberCommand(command.rawGoal)
                val reply = if (parsed != null) {
                    val (contact, fact) = parsed
                    val updated = com.zipper.app.core.intelligence.reply.ContactKnowledgeBook.getInstance(context)
                        .addMemoryPoint(contact, fact)
                    "✅ Remembered for ${updated.contactName}: \"$fact\" (${updated.memoryPoints.size} memory points in dossier)"
                } else {
                    "ℹ️ Please format as: 'remember <Contact>: <facts>' (e.g. 'remember Manoj: close friend, indiranagar')"
                }
                replySilently(context, sbn, reply)
                recordToAuditStorage(context, command, "SAVED_CONTACT_KNOWLEDGE: $reply")
            }


            RemoteActionType.SEND_WHATSAPP_MESSAGE -> {
                val recipient = command.recipient ?: "Contact"
                val body = command.messageBody ?: ""

                // 1. Resolve matching contacts
                val matches = RemoteContactResolver.findMatchingContacts(context, recipient)

                if (matches.size > 1) {
                    // MULTIPLE CANDIDATES: Send interactive numbered disambiguation menu!
                    PendingRemoteSessionStore.setPending(
                        command.sender,
                        PendingDisambiguationSession(
                            senderId = command.sender,
                            contactNameQuery = recipient,
                            messageText = body,
                            candidates = matches
                        )
                    )

                    val sb = StringBuilder()
                    sb.append("Found ${matches.size} contacts for '$recipient':\n")
                    matches.forEachIndexed { idx, contact ->
                        val emoji = when (idx + 1) {
                            1 -> "1️⃣"
                            2 -> "2️⃣"
                            3 -> "3️⃣"
                            4 -> "4️⃣"
                            5 -> "5️⃣"
                            else -> "${idx + 1}."
                        }
                        sb.append("$emoji ${contact.name} (${contact.phoneNumber})\n")
                    }
                    sb.append("Reply with number (e.g. '$recipient 2' or '2') to send.")
                    replySilently(context, sbn, sb.toString().trim())
                    recordToAuditStorage(context, command, "PROMPTED_CONTACT_DISAMBIGUATION: ${matches.size} matches")
                } else if (matches.size == 1) {
                    // EXACT SINGLE MATCH: Send directly via WhatsApp direct intent & motor
                    val match = matches.first()
                    val sent = WhatsAppAutomationExecutor.executeSendToPhone(context, match.normalizedPhone, body)
                    val ack = if (sent) {
                        "✅ Sent \"$body\" to ${match.name} (${match.phoneNumber}) on WhatsApp."
                    } else {
                        "🚀 Dispatched message to ${match.name} on WhatsApp."
                    }
                    replySilently(context, sbn, ack)
                    recordToAuditStorage(context, command, "SENT_WHATSAPP_DIRECT: ${match.name}")
                } else {
                    // No direct phone match in local contacts -> Fall back to search in WhatsApp UI
                    val ack = "🚀 Karvish is sending message to $recipient: \"$body\""
                    replySilently(context, sbn, ack)

                    val goal = "Open WhatsApp and send message \"$body\" to $recipient"
                    dispatchAgentGoal(context, command, goal)
                }
            }

            RemoteActionType.SCHEDULE_MESSAGE -> {
                val recipient = command.recipient ?: "Contact"
                val body = command.messageBody ?: ""
                val targetMs = command.scheduledTimeMs ?: (System.currentTimeMillis() + 60000L)
                val timeFmt = java.text.SimpleDateFormat("hh:mm a", java.util.Locale.getDefault()).format(java.util.Date(targetMs))

                scheduleMessage(
                    context = context,
                    recipient = recipient,
                    message = body,
                    triggerTimeEpochMs = targetMs,
                    enableChat = true
                )

                val ack = "⏰ Scheduled message to $recipient for $timeFmt:\n\"$body\"\n(AI auto-reply will continue chat once sent)"
                replySilently(context, sbn, ack)
                recordToAuditStorage(context, command, "SCHEDULED_MESSAGE: $recipient at $timeFmt")
            }

            RemoteActionType.CONFIRM_CONTACT_SELECTION -> {
                val contact = command.resolvedContact
                val body = command.messageBody ?: ""
                PendingRemoteSessionStore.removePending(command.sender)

                if (contact != null) {
                    val sent = WhatsAppAutomationExecutor.executeSendToPhone(context, contact.normalizedPhone, body)
                    val ack = if (sent) {
                        "✅ Sent \"$body\" to ${contact.name} (${contact.phoneNumber}) on WhatsApp."
                    } else {
                        "🚀 Dispatched message to ${contact.name} on WhatsApp."
                    }
                    replySilently(context, sbn, ack)
                    recordToAuditStorage(context, command, "DISAMBIGUATED_SENT_WHATSAPP: ${contact.name}")
                } else {
                    replySilently(context, sbn, "⚠️ Could not resolve selected contact option.")
                }
            }

            RemoteActionType.SEND_FILE -> {
                val fileQuery = command.fileQuery ?: command.rawGoal
                val recipientName = command.recipient

                // 1. Resolve matching files from device storage
                val fileMatches = RemoteFileResolver.findMatchingFiles(context, fileQuery)

                if (fileMatches.isEmpty()) {
                    val reply = "⚠️ Could not find any files matching '$fileQuery' in storage (Downloads/Documents)."
                    replySilently(context, sbn, reply)
                    recordToAuditStorage(context, command, "FILE_NOT_FOUND: $fileQuery")
                    return
                }

                // Determine target recipient phone
                val targetPhone = resolveTargetPhoneNumber(context, recipientName, command.sender)

                if (fileMatches.size > 1) {
                    // MULTIPLE MATCHES: Show numbered disambiguation list!
                    PendingRemoteSessionStore.setPending(
                        command.sender,
                        PendingDisambiguationSession(
                            senderId = command.sender,
                            contactNameQuery = fileQuery,
                            targetRecipient = recipientName,
                            fileCandidates = fileMatches
                        )
                    )

                    val sb = StringBuilder()
                    sb.append("Found ${fileMatches.size} files for '$fileQuery':\n")
                    fileMatches.forEachIndexed { idx, f ->
                        val emoji = when (idx + 1) {
                            1 -> "1️⃣"
                            2 -> "2️⃣"
                            3 -> "3️⃣"
                            4 -> "4️⃣"
                            5 -> "5️⃣"
                            else -> "${idx + 1}."
                        }
                        val sizeStr = RemoteFileResolver.formatFileSize(f.sizeBytes)
                        sb.append("$emoji ${f.name} ($sizeStr)\n")
                    }
                    sb.append("Reply with number (e.g. '1' or '2') to send.")
                    replySilently(context, sbn, sb.toString().trim())
                    recordToAuditStorage(context, command, "PROMPTED_FILE_DISAMBIGUATION: ${fileMatches.size} files")
                } else {
                    // EXACT SINGLE MATCH: Send directly via WhatsApp file motor!
                    val match = fileMatches.first()
                    val targetFile = java.io.File(match.path)
                    val sent = WhatsAppAutomationExecutor.executeSendFile(
                        context = context,
                        targetPhone = targetPhone,
                        file = targetFile,
                        mimeType = match.mimeType
                    )
                    val targetDesc = recipientName ?: "you"
                    val ack = if (sent) {
                        "✅ Sent \"${match.name}\" to $targetDesc on WhatsApp."
                    } else {
                        "🚀 Dispatched \"${match.name}\" to $targetDesc on WhatsApp."
                    }
                    replySilently(context, sbn, ack)
                    recordToAuditStorage(context, command, "SENT_FILE_DIRECT: ${match.name}")
                }
            }

            RemoteActionType.CONFIRM_FILE_SELECTION -> {
                val selectedFile = command.resolvedFile
                val recipientName = command.recipient
                PendingRemoteSessionStore.removePending(command.sender)

                if (selectedFile != null) {
                    val targetPhone = resolveTargetPhoneNumber(context, recipientName, command.sender)
                    val targetFile = java.io.File(selectedFile.path)
                    val sent = WhatsAppAutomationExecutor.executeSendFile(
                        context = context,
                        targetPhone = targetPhone,
                        file = targetFile,
                        mimeType = selectedFile.mimeType
                    )
                    val targetDesc = recipientName ?: "you"
                    val ack = if (sent) {
                        "✅ Sent \"${selectedFile.name}\" to $targetDesc on WhatsApp."
                    } else {
                        "🚀 Dispatched \"${selectedFile.name}\" to $targetDesc on WhatsApp."
                    }
                    replySilently(context, sbn, ack)
                    recordToAuditStorage(context, command, "DISAMBIGUATED_SENT_FILE: ${selectedFile.name}")
                } else {
                    replySilently(context, sbn, "⚠️ Could not resolve selected file option.")
                }
            }

            RemoteActionType.EXECUTE_SHOPPING -> {
                val store = command.targetApp ?: "Zepto"
                val query = command.shoppingQuery ?: command.rawGoal
                val ack = "🛒 Karvish started shopping for \"$query\" on $store"
                replySilently(context, sbn, ack)

                val goal = "Order $query on $store"
                dispatchAgentGoal(context, command, goal)
            }

            RemoteActionType.GENERAL_AGENT_GOAL, RemoteActionType.UNKNOWN -> {
                val goal = command.rawGoal
                Log.i(TAG, "Dispatching goal to Karvish Autonomous Agent: '$goal'")
                replySilently(context, sbn, "⚡ Karvish Autonomous Agent executing: \"$goal\"")
                dispatchAgentGoal(context, command, goal)
            }
        }
    }

    private fun dispatchAgentGoal(context: Context, command: RemoteCommand, goal: String) {
        val serviceIntent = Intent(context, com.zipper.app.core.agent.AgentRuntimeService::class.java).apply {
            putExtra(com.zipper.app.core.agent.AgentRuntimeService.EXTRA_GOAL, goal)
            putExtra(com.zipper.app.core.agent.AgentRuntimeService.EXTRA_RECOVERY, true)
            putExtra(com.zipper.app.core.agent.AgentRuntimeService.EXTRA_USE_REMOTE_RANKER, false)
        }
        androidx.core.content.ContextCompat.startForegroundService(context, serviceIntent)
        recordToAuditStorage(context, command, "DISPATCHED_AGENT_GOAL: $goal")
    }

    /**
     * Replies silently via Android Notification RemoteInput inline reply if available.
     * Falls back to a local confirmation notification.
     */
    fun replySilently(context: Context, sbn: StatusBarNotification?, replyText: String) {
        var repliedViaRemoteInput = false
        if (sbn != null) {
            repliedViaRemoteInput = trySendInlineReply(context, sbn, replyText)
        }

        // Post confirmation so device owner always sees audit trail
        postConfirmationNotification(
            context,
            "Karvish Remote Co-Pilot",
            replyText
        )
    }

    /**
     * Sends an inline reply back into the source chat (e.g. WhatsApp) via notification RemoteInput.
     * Works 100% in the background without turning on the screen.
     */
    fun trySendInlineReply(
        context: Context,
        sbn: StatusBarNotification,
        replyText: String
    ): Boolean {
        return try {
            val notification = sbn.notification ?: return false
            val actions = notification.actions ?: return false

            for (action in actions) {
                val remoteInputs = action.remoteInputs ?: continue
                if (remoteInputs.isEmpty()) continue

                val intent = Intent()
                val bundle = Bundle()
                for (remoteInput in remoteInputs) {
                    bundle.putCharSequence(remoteInput.resultKey, replyText)
                }
                RemoteInput.addResultsToIntent(remoteInputs, intent, bundle)

                action.actionIntent.send(context, 0, intent)
                Log.i(TAG, "Sent silent inline reply via RemoteInput to ${sbn.packageName}: '$replyText'")
                return true
            }
            false
        } catch (e: Exception) {
            Log.e(TAG, "Failed to send inline reply: ${e.message}")
            false
        }
    }

    /**
     * Schedules a WhatsApp message via AlarmManager to be dispatched silently at triggerTimeEpochMs.
     */
    fun scheduleMessage(
        context: Context,
        recipient: String,
        message: String,
        triggerTimeEpochMs: Long,
        enableChat: Boolean = true
    ) {
        val alarmManager = context.getSystemService(Context.ALARM_SERVICE) as? android.app.AlarmManager ?: return
        val intent = Intent(context, ScheduledMessageReceiver::class.java).apply {
            putExtra(ScheduledMessageReceiver.EXTRA_RECIPIENT, recipient)
            putExtra(ScheduledMessageReceiver.EXTRA_MESSAGE, message)
            putExtra(ScheduledMessageReceiver.EXTRA_ENABLE_CHAT, enableChat)
        }
        val requestCode = (recipient.hashCode() xor triggerTimeEpochMs.hashCode()) and 0x7FFFFFFF
        val pendingIntent = android.app.PendingIntent.getBroadcast(
            context,
            requestCode,
            intent,
            android.app.PendingIntent.FLAG_UPDATE_CURRENT or android.app.PendingIntent.FLAG_IMMUTABLE
        )

        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                alarmManager.setExactAndAllowWhileIdle(
                    android.app.AlarmManager.RTC_WAKEUP,
                    triggerTimeEpochMs,
                    pendingIntent
                )
            } else {
                alarmManager.setExact(
                    android.app.AlarmManager.RTC_WAKEUP,
                    triggerTimeEpochMs,
                    pendingIntent
                )
            }
            Log.i(TAG, "Scheduled WhatsApp message to '$recipient' at epoch $triggerTimeEpochMs")
        } catch (e: SecurityException) {
            Log.w(TAG, "Missing exact alarm permission, using inexact: ${e.message}")
            alarmManager.set(
                android.app.AlarmManager.RTC_WAKEUP,
                triggerTimeEpochMs,
                pendingIntent
            )
        }
    }

    internal fun postConfirmationNotification(context: Context, title: String, text: String) {
        val notificationManager = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Karvish Remote Execution",
                NotificationManager.IMPORTANCE_HIGH
            ).apply {
                description = "Status confirmations for autonomous remote actions"
            }
            notificationManager.createNotificationChannel(channel)
        }

        val notification = NotificationCompat.Builder(context, CHANNEL_ID)
            .setSmallIcon(R.mipmap.ic_launcher)
            .setContentTitle(title)
            .setContentText(text)
            .setStyle(NotificationCompat.BigTextStyle().bigText(text))
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setAutoCancel(true)
            .build()

        notificationManager.notify((System.currentTimeMillis() % 100000).toInt(), notification)
    }

    private fun resolveTargetPhoneNumber(context: Context, recipientName: String?, senderName: String): String? {
        val target = recipientName ?: senderName
        val digitsOnly = target.replace(Regex("""[^0-9+]"""), "")
        if (digitsOnly.length >= 10) {
            return RemoteContactResolver.normalizePhoneNumber(digitsOnly)
        }
        val contactMatches = RemoteContactResolver.findMatchingContacts(context, target)
        if (contactMatches.isNotEmpty()) {
            return contactMatches.first().normalizedPhone
        }
        return null
    }

    private fun recordToAuditStorage(context: Context, cmd: RemoteCommand, outcome: String) {
        val auditItem = KarvishNotificationItem(
            id = "remote_cmd_${System.currentTimeMillis()}",
            packageName = "com.zipper.app",
            appName = "Remote Commander",
            title = "Command: ${cmd.rawGoal}",
            text = "From: ${cmd.sender} (${cmd.sourceApp}) -> $outcome",
            category = NotificationCategory.GENERAL,
            timestamp = System.currentTimeMillis()
        )
        NotificationStorageManager.getInstance(context).insert(auditItem)
    }
}
