package com.zipper.app.core.remote

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log
import com.zipper.app.core.intelligence.reply.DelegatedContact
import com.zipper.app.core.intelligence.reply.DelegatedContactManager
import com.zipper.app.core.intelligence.reply.DelegatedConversationMemory
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

/**
 * BroadcastReceiver triggered by AlarmManager to send a scheduled WhatsApp message.
 * Once sent, it registers or activates the recipient in DelegatedContactManager
 * so that when the recipient replies, Zipper continues the chat naturally via LLM.
 */
class ScheduledMessageReceiver : BroadcastReceiver() {

    companion object {
        private const val TAG = "ScheduledMessageReceiver"
        const val EXTRA_RECIPIENT = "extra_recipient"
        const val EXTRA_MESSAGE = "extra_message"
        const val EXTRA_ENABLE_CHAT = "extra_enable_chat"
    }

    override fun onReceive(context: Context, intent: Intent) {
        val recipient = intent.getStringExtra(EXTRA_RECIPIENT)?.trim().orEmpty()
        val message = intent.getStringExtra(EXTRA_MESSAGE)?.trim().orEmpty()
        val enableChat = intent.getBooleanExtra(EXTRA_ENABLE_CHAT, true)

        if (recipient.isBlank() || message.isBlank()) return

        Log.i(TAG, "⏰ Scheduled message alarm fired! Sending \"$message\" to $recipient (Chat continuation: $enableChat)")

        CoroutineScope(Dispatchers.IO).launch {
            try {
                // 1. Resolve contact phone number
                val matches = RemoteContactResolver.findMatchingContacts(context, recipient)
                val targetPhone = if (matches.isNotEmpty()) {
                    matches.first().normalizedPhone
                } else {
                    recipient.filter { it.isDigit() || it == '+' }
                }

                // 2. Dispatch via WhatsApp direct motor
                val sent = WhatsAppAutomationExecutor.executeSendToPhone(context, targetPhone, message)
                Log.i(TAG, "Sent scheduled message to $recipient ($targetPhone): success=$sent")

                // 3. Record into 2-way chat memory
                val memory = DelegatedConversationMemory.getInstance(context)
                memory.recordMessage(recipient, "Karthik", message)

                // 4. If chat continuation is allowed, ensure contact is active in DelegatedContactManager
                if (enableChat) {
                    val manager = DelegatedContactManager.getInstance(context)
                    val existing = manager.findMatching(recipient)
                    if (existing != null) {
                        if (!existing.isEnabled) {
                            manager.saveContact(existing.copy(isEnabled = true))
                        }
                    } else {
                        manager.saveContact(
                            DelegatedContact(
                                contactName = recipient,
                                phoneNumber = targetPhone,
                                isEnabled = true,
                                personaTone = "Casual & Friendly, Bangalore slang & Kanglish",
                                customContext = "You initiated this conversation by sending: \"$message\". Continue chatting warmly and naturally as Karthik."
                            )
                        )
                    }
                    Log.i(TAG, "Enabled delegated AI chat session for $recipient to continue conversation.")
                }

                // 5. Post notification confirmation
                RemoteCommandDispatcher.postConfirmationNotification(
                    context,
                    "Zipper Scheduled Message Sent",
                    "Sent \"$message\" to $recipient at scheduled time. (AI chat continuation active)"
                )
            } catch (e: Exception) {
                Log.e(TAG, "Failed executing scheduled message to $recipient: ${e.message}", e)
            }
        }
    }
}
