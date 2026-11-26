package com.zipper.app.core.intelligence.ambient

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.service.notification.StatusBarNotification
import android.util.Log
import android.widget.Toast
import androidx.core.app.NotificationCompat
import com.zipper.app.R
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap

/**
 * Senior Engineer Ambient Intelligence Engine:
 * 1. Zero-Click OTP Autopilot: Automatically extracts and copies OTPs to clipboard.
 * 2. Delivery & Logistics HUD: Surfaces real-time delivery PINs and rider actions for Swiggy, Zomato, Zepto, Blinkit, Uber.
 */
object OtpAndDeliveryAutopilot {

    private const val TAG = "OtpDeliveryAutopilot"
    private const val CHANNEL_OTP = "karvish_ambient_otp"
    private const val CHANNEL_DELIVERY = "karvish_ambient_delivery"

    private val OTP_KEYWORDS = Regex("""(?i)\b(?:otp|code|verification|passcode|one-time\s+password|security\s+code|login\s+code|auth\s+code)\b""")
    private val OTP_DIGITS_REGEX = Regex("""\b(\d{4,8})\b""")

    private val DELIVERY_PIN_REGEX = Regex("""(?i)\b(?:delivery\s+code|delivery\s+pin|order\s+pin|pin|code|otp)\s*(?:is|:|-)?\s*(\d{2,6})\b""")

    // Packages representing delivery or quick-commerce
    private val DELIVERY_PACKAGES = mapOf(
        "in.swiggy.android" to "Swiggy",
        "com.application.zomato" to "Zomato",
        "com.zeptoconsumerapp" to "Zepto",
        "com.grofers.customerapp" to "Blinkit",
        "com.ubercab" to "Uber",
        "com.olacabs.customer" to "Ola",
        "com.rapido.passenger" to "Rapido",
        "com.amazon.mShop.android.shopping" to "Amazon",
        "com.flipkart.android" to "Flipkart",
        "com.dunzo.user" to "Dunzo"
    )

    // Deduplication cache to prevent alert spam within 30 seconds
    private val recentProcessedHashes = ConcurrentHashMap<String, Long>()

    private fun isRecentlyProcessed(key: String, cooldownMs: Long = 30_000L): Boolean {
        val now = System.currentTimeMillis()
        val last = recentProcessedHashes[key]
        if (last != null && now - last < cooldownMs) {
            return true
        }
        recentProcessedHashes[key] = now
        return false
    }

    /**
     * Inspects every incoming notification stream for ambient value.
     */
    fun processNotification(
        context: Context,
        sbn: StatusBarNotification,
        title: String,
        text: String,
        subText: String?
    ) {
        val pkg = sbn.packageName
        val combined = "$title $text ${subText.orEmpty()}".trim()
        val combinedLower = combined.lowercase(Locale.ROOT)

        // 1. Check for OTP (SMS, WhatsApp, Banking, Auth)
        if (OTP_KEYWORDS.containsMatchIn(combinedLower)) {
            val match = OTP_DIGITS_REGEX.find(text) ?: OTP_DIGITS_REGEX.find(title)
            if (match != null) {
                val code = match.groupValues[1]
                val dedupeKey = "otp_${pkg}_$code"
                if (!isRecentlyProcessed(dedupeKey)) {
                    handleExtractedOtp(context, code, title, text)
                    return
                }
            }
        }

        // 2. Check for Delivery & Logistics Arrival
        val brandName = DELIVERY_PACKAGES[pkg] ?: resolveDeliveryBrandFromText(combinedLower)
        if (brandName != null && isDeliveryStatusUpdate(combinedLower)) {
            val dedupeKey = "delivery_${pkg}_${combined.hashCode()}"
            if (!isRecentlyProcessed(dedupeKey, cooldownMs = 60_000L)) {
                handleDeliveryUpdate(context, brandName, combined, sbn)
            }
        }
    }

    private fun resolveDeliveryBrandFromText(lower: String): String? {
        return when {
            lower.contains("swiggy") -> "Swiggy"
            lower.contains("zomato") -> "Zomato"
            lower.contains("zepto") -> "Zepto"
            lower.contains("blinkit") -> "Blinkit"
            lower.contains("uber") -> "Uber"
            lower.contains("ola") -> "Ola"
            lower.contains("rapido") -> "Rapido"
            lower.contains("amazon") -> "Amazon"
            lower.contains("flipkart") -> "Flipkart"
            else -> null
        }
    }

    private fun isDeliveryStatusUpdate(lower: String): Boolean {
        return lower.contains("arriving") ||
                lower.contains("arrived") ||
                lower.contains("at your gate") ||
                lower.contains("at the door") ||
                lower.contains("out for delivery") ||
                lower.contains("delivery partner") ||
                lower.contains("delivery pin") ||
                lower.contains("share code") ||
                lower.contains("driver is arriving") ||
                lower.contains("reaching in")
    }

    /**
     * Auto-copies the OTP to clipboard and posts an ambient HUD alert.
     */
    private fun handleExtractedOtp(context: Context, code: String, sourceTitle: String, sourceText: String) {
        try {
            val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as? ClipboardManager
            val clip = ClipData.newPlainText("OTP Code", code)
            clipboard?.setPrimaryClip(clip)
            Log.i(TAG, "⚡ Autopilot: Copied OTP '$code' to clipboard automatically!")

            Handler(Looper.getMainLooper()).post {
                Toast.makeText(context, "📋 OTP $code copied to clipboard!", Toast.LENGTH_SHORT).show()
            }

            val notificationManager = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            ensureChannels(notificationManager)

            // Intent to copy again
            val copyIntent = Intent(context, OtpCopyReceiver::class.java).apply {
                action = "com.zipper.app.ACTION_COPY_OTP"
                putExtra("otp_code", code)
            }
            val pCopy = PendingIntent.getBroadcast(
                context,
                code.hashCode(),
                copyIntent,
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
            )

            val notif = NotificationCompat.Builder(context, CHANNEL_OTP)
                .setSmallIcon(R.drawable.ic_check_circle)
                .setContentTitle("📋 OTP Auto-Copied: $code")
                .setContentText("$sourceTitle • Tap to use or paste into app")
                .setPriority(NotificationCompat.PRIORITY_MAX)
                .setDefaults(NotificationCompat.DEFAULT_ALL)
                .setAutoCancel(true)
                .addAction(R.drawable.ic_copy_clean, "Copy: $code", pCopy)
                .build()

            notificationManager.notify(70001, notif)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to copy OTP: ${e.message}")
        }
    }

    /**
     * Surfaces delivery PIN and quick actions on screen without app switching.
     */
    private fun handleDeliveryUpdate(
        context: Context,
        brandName: String,
        fullText: String,
        sbn: StatusBarNotification
    ) {
        try {
            val notificationManager = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            ensureChannels(notificationManager)

            // Extract PIN if present
            val pinMatch = DELIVERY_PIN_REGEX.find(fullText)
            val deliveryPin = pinMatch?.groupValues?.get(1)

            val title = if (!deliveryPin.isNullOrBlank()) {
                "🛵 $brandName: Delivery Code $deliveryPin"
            } else {
                "🛵 $brandName: Arrival Update"
            }

            val notifBuilder = NotificationCompat.Builder(context, CHANNEL_DELIVERY)
                .setSmallIcon(R.drawable.ic_bolt)
                .setContentTitle(title)
                .setContentText(fullText)
                .setPriority(NotificationCompat.PRIORITY_HIGH)
                .setAutoCancel(true)

            // Add 1-tap Copy PIN action
            if (!deliveryPin.isNullOrBlank()) {
                val copyIntent = Intent(context, OtpCopyReceiver::class.java).apply {
                    action = "com.zipper.app.ACTION_COPY_OTP"
                    putExtra("otp_code", deliveryPin)
                }
                val pCopy = PendingIntent.getBroadcast(
                    context,
                    deliveryPin.hashCode(),
                    copyIntent,
                    PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
                )
                notifBuilder.addAction(R.drawable.ic_copy_clean, "Copy PIN: $deliveryPin", pCopy)
            }

            // Add 1-tap Open Order / Chat action
            sbn.notification.contentIntent?.let { pi ->
                notifBuilder.setContentIntent(pi)
                notifBuilder.addAction(R.drawable.ic_arrow_back, "Open $brandName", pi)
            }

            notificationManager.notify(70002, notifBuilder.build())
            Log.i(TAG, "🛵 Autopilot: Posted Delivery HUD for $brandName (PIN: $deliveryPin)")
        } catch (e: Exception) {
            Log.w(TAG, "Failed to handle delivery update: ${e.message}")
        }
    }

    private fun ensureChannels(manager: NotificationManager) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            if (manager.getNotificationChannel(CHANNEL_OTP) == null) {
                val chanOtp = NotificationChannel(
                    CHANNEL_OTP,
                    "Zipper OTP Autopilot",
                    NotificationManager.IMPORTANCE_HIGH
                ).apply {
                    description = "Surfaces and auto-copies one-time passwords for effortless sign-in"
                }
                manager.createNotificationChannel(chanOtp)
            }
            if (manager.getNotificationChannel(CHANNEL_DELIVERY) == null) {
                val chanDelivery = NotificationChannel(
                    CHANNEL_DELIVERY,
                    "Zipper Delivery & Logistics HUD",
                    NotificationManager.IMPORTANCE_HIGH
                ).apply {
                    description = "Surfaces delivery PINs and driver tracking for Swiggy, Zomato, Zepto, Uber"
                }
                manager.createNotificationChannel(chanDelivery)
            }
        }
    }
}
