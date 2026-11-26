package com.zipper.app.core.notification

import android.content.Context
import android.content.Intent
import android.provider.Settings
import androidx.core.app.NotificationManagerCompat
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

data class NotificationQueryResult(
    val title: String,
    val markdownResponse: String,
    val copyableOtp: String? = null,
    val items: List<KarvishNotificationItem> = emptyList(),
    val requiresAiSynthesis: Boolean = false
)

/**
 * Universal Query & Briefing Engine for Karvish Notification Life-Stream.
 *
 * Bridges incoming device notifications with both Local Mode (instant < 5ms lookups)
 * and AI Mode (executive summaries and semantic reasoning).
 */
object NotificationQueryEngine {

    fun isPermissionGranted(context: Context): Boolean {
        val enabledPackages = NotificationManagerCompat.getEnabledListenerPackages(context)
        return enabledPackages.contains(context.packageName)
    }

    fun getPermissionSettingsIntent(): Intent {
        return Intent(Settings.ACTION_NOTIFICATION_LISTENER_DETAIL_SETTINGS).apply {
            putExtra(Settings.EXTRA_NOTIFICATION_LISTENER_COMPONENT_NAME, "com.zipper.app/.core.notification.KarvishNotificationListenerService")
        }.let { intent ->
            // Fallback to general listener settings if detail settings not available
            Intent(Settings.ACTION_NOTIFICATION_LISTENER_SETTINGS)
        }
    }

    fun isNotificationRelatedQuery(rawQuery: String): Boolean {
        val q = rawQuery.lowercase(Locale.ROOT)
        val triggers = setOf(
            "otp", "code", "verification code", "notification", "notifications",
            "what did i miss", "catch me up", "summarize messages", "summary of messages",
            "whatsapp", "swiggy", "zomato", "uber", "delivery status", "order status",
            "track order", "track delivery", "bank alert", "debited", "credited"
        )
        return triggers.any { q.contains(it) }
    }

    fun handleLocalNotificationQuery(context: Context, rawQuery: String): NotificationQueryResult? {
        if (!isNotificationRelatedQuery(rawQuery)) return null

        val storage = NotificationStorageManager.getInstance(context)
        val q = rawQuery.lowercase(Locale.ROOT)
        val timeFormat = SimpleDateFormat("hh:mm a", Locale.getDefault())

        // 1. Check for OTP query
        if (q.contains("otp") || q.contains("code") || q.contains("verification")) {
            val otpItem = storage.getLatestOtp()
            return if (otpItem != null && !otpItem.otpCode.isNullOrBlank()) {
                val timeStr = timeFormat.format(Date(otpItem.timestamp))
                val minutesAgo = ((System.currentTimeMillis() - otpItem.timestamp) / 60000).coerceAtLeast(0)
                NotificationQueryResult(
                    title = "Latest OTP Verification Code",
                    markdownResponse = buildString {
                        append("🔐 **Latest OTP Code Found:**\n\n")
                        append("🔢 **`${otpItem.otpCode}`**\n\n")
                        append("• **App/Sender:** ${otpItem.appName} (${otpItem.title})\n")
                        append("• **Received:** $timeStr ($minutesAgo min ago)\n")
                        append("• **Full Text:** *\"${otpItem.text}\"*")
                    },
                    copyableOtp = otpItem.otpCode,
                    items = listOf(otpItem)
                )
            } else {
                NotificationQueryResult(
                    title = "No Recent OTP",
                    markdownResponse = "ℹ️ No recent OTP or verification code was detected in your notifications in the last 15 minutes."
                )
            }
        }

        // 2. Check for Food / Cab / Delivery Logistics query
        if (q.contains("swiggy") || q.contains("zomato") || q.contains("uber") || q.contains("delivery") || q.contains("order status") || q.contains("track")) {
            val deliveries = storage.getByCategory(NotificationCategory.DELIVERY_LOGISTICS, limit = 5)
            return if (deliveries.isNotEmpty()) {
                NotificationQueryResult(
                    title = "Live Delivery & Logistics Status",
                    markdownResponse = buildString {
                        append("🛵 **Live Delivery & Logistics Updates:**\n\n")
                        deliveries.forEachIndexed { i, item ->
                            val timeStr = timeFormat.format(Date(item.timestamp))
                            append("${i + 1}. **${item.appName}** — *${item.title}*\n")
                            append("   \"${item.text}\" `($timeStr)`\n\n")
                        }
                    },
                    items = deliveries
                )
            } else {
                NotificationQueryResult(
                    title = "No Active Deliveries",
                    markdownResponse = "ℹ️ No recent delivery or ride updates found from Swiggy, Zomato, Zepto, Blinkit, or Uber."
                )
            }
        }

        // 3. Check for specific Sender (e.g. "What did Mom say?", "Messages from Rahul")
        val senderRegex = Regex("""(?i)\b(?:from|said|sent|message\s+from)\s+([a-z0-9\s]+?)(?:$|[?.])""")
        val match = senderRegex.find(q)
        if (match != null) {
            val targetSender = match.groupValues[1].trim()
            val matches = storage.search(targetSender, limit = 5)
            if (matches.isNotEmpty()) {
                return NotificationQueryResult(
                    title = "Messages from '$targetSender'",
                    markdownResponse = buildString {
                        append("💬 **Recent Notifications matching \"$targetSender\":**\n\n")
                        matches.forEachIndexed { i, item ->
                            val timeStr = timeFormat.format(Date(item.timestamp))
                            append("${i + 1}. **${item.appName}** (${item.title}):\n")
                            append("   *\"${item.text}\"* — `$timeStr`\n\n")
                        }
                    },
                    items = matches
                )
            }
        }

        // 4. Executive Briefing / Catch Me Up
        if (q.contains("summarize") || q.contains("catch me up") || q.contains("what did i miss") || q.contains("briefing")) {
            val recent = storage.getRecent(hours = 3, limit = 30)
            if (recent.isEmpty()) {
                return NotificationQueryResult(
                    title = "Notification Briefing",
                    markdownResponse = "ℹ️ No notifications captured in the last 3 hours. Your notification tray is all clear!"
                )
            }

            // Signal that AI synthesis is desired if AI mode is enabled
            return NotificationQueryResult(
                title = "Notification Executive Briefing",
                markdownResponse = buildString {
                    append("📋 **Notification Briefing (Last 3 Hours — ${recent.size} alerts):**\n\n")
                    val grouped = recent.groupBy { it.category }
                    grouped[NotificationCategory.MESSAGING]?.let { msgs ->
                        append("💬 **Messages (${msgs.size}):**\n")
                        msgs.take(3).forEach { append("• **${it.appName}** (${it.title}): ${it.text}\n") }
                        append("\n")
                    }
                    grouped[NotificationCategory.DELIVERY_LOGISTICS]?.let { dels ->
                        append("🛵 **Deliveries & Rides (${dels.size}):**\n")
                        dels.take(2).forEach { append("• **${it.appName}**: ${it.text}\n") }
                        append("\n")
                    }
                    grouped[NotificationCategory.TRANSACTION_OTP]?.let { txs ->
                        append("💳 **Finance & Alerts (${txs.size}):**\n")
                        txs.take(2).forEach { append("• **${it.appName}**: ${it.text}\n") }
                        append("\n")
                    }
                },
                items = recent,
                requiresAiSynthesis = true
            )
        }

        return null
    }

    fun buildAiSynthesisPrompt(items: List<KarvishNotificationItem>): String {
        val timeFormat = SimpleDateFormat("hh:mm a", Locale.getDefault())
        return buildString {
            append("You are Karvish, an intelligent executive personal assistant on Android. ")
            append("The user asked for an executive summary of their recent device notifications. ")
            append("Analyze the following ${items.size} notifications. Group them logically, eliminate promotional spam, ")
            append("highlight urgent action items or requests from people, and format as a crisp markdown briefing:\n\n")
            items.take(25).forEachIndexed { i, n ->
                val timeStr = timeFormat.format(Date(n.timestamp))
                append("${i + 1}. [${n.appName}] (${n.category}) From: \"${n.title}\" | Text: \"${n.text}\" ($timeStr)\n")
            }
        }
    }
}
