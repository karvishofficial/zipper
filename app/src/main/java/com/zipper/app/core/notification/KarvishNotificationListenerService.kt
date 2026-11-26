package com.zipper.app.core.notification

import android.app.Notification
import android.content.pm.PackageManager
import android.service.notification.NotificationListenerService
import android.service.notification.StatusBarNotification
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.util.Locale
import java.util.UUID

/**
 * Android Notification Stream Listener for Karvish.
 *
 * Captures life-feed event stream (WhatsApp, Gmail, SMS OTPs, Swiggy/Uber status)
 * completely immune to app UI changes.
 *
 * Performance guarantee:
 * - 0% idle CPU
 * - Non-blocking asynchronous processing via Dispatchers.IO
 * - Noise filtering (ignores system spam and persistent media players)
 */
class KarvishNotificationListenerService : NotificationListenerService() {

    private val serviceScope = CoroutineScope(Dispatchers.IO)
    private lateinit var storage: NotificationStorageManager

    companion object {
        private const val TAG = "KarvishNotification"

        @Volatile
        var isConnected: Boolean = false
            private set

        // Noise packages to ignore completely
        private val IGNORED_PACKAGES = setOf(
            "android",
            "com.android.systemui",
            "com.google.android.projection.gearhead",
            "com.sec.android.app.soundalive",
            "com.nisargjhaveri.netspeed",
            "com.internet.speed.meter",
            "com.internet.speed.meter.lite",
            "org.jak_linux.dns66",
            "de.blinkt.openvpn"
        )

        // OTP detection regex patterns
        private val OTP_KEYWORD_REGEX = Regex("""(?i)\b(?:otp|code|verification|one-time\s+password|security\s+code|login\s+code|auth\s+code|passcode)\b""")
        private val OTP_DIGITS_REGEX = Regex("""\b(\d{4,8})\b""")

        // In-memory cache of active notification PendingIntents for 1-tap direct chat opening
        val pendingIntentCache = java.util.concurrent.ConcurrentHashMap<String, android.app.PendingIntent>()
        // In-memory cache of active notifications for silent inline reply / acknowledgments
        val activeSbnCache = java.util.concurrent.ConcurrentHashMap<String, StatusBarNotification>()

        fun findContentIntent(packageName: String, query: String = ""): android.app.PendingIntent? {
            if (query.isNotBlank()) {
                val normalized = query.trim().lowercase(java.util.Locale.ROOT)
                // Match by full key
                val directKey = "${packageName}_$normalized"
                pendingIntentCache[directKey]?.let { return it }
                // Search keys containing query
                for ((k, v) in pendingIntentCache) {
                    if (k.startsWith(packageName) && k.contains(normalized)) return v
                }
            }
            return pendingIntentCache[packageName]
        }

        fun findStatusBarNotification(packageName: String, query: String = ""): StatusBarNotification? {
            if (query.isNotBlank()) {
                val normalized = query.trim().lowercase(java.util.Locale.ROOT)
                val directKey = "${packageName}_$normalized"
                activeSbnCache[directKey]?.let { return it }
                for ((k, v) in activeSbnCache) {
                    if (k.startsWith(packageName) && k.contains(normalized)) return v
                }
            }
            return activeSbnCache[packageName]
        }
    }

    private val appNameCache = java.util.concurrent.ConcurrentHashMap<String, String>()

    override fun onCreate() {
        super.onCreate()
        storage = NotificationStorageManager.getInstance(applicationContext)
    }

    override fun onListenerConnected() {
        super.onListenerConnected()
        isConnected = true
        Log.i(TAG, "Karvish Notification Listener Connected")
    }

    override fun onListenerDisconnected() {
        super.onListenerDisconnected()
        isConnected = false
        Log.i(TAG, "Karvish Notification Listener Disconnected")
    }

    override fun onNotificationPosted(sbn: StatusBarNotification?) {
        if (sbn == null) return

        val pkg = sbn.packageName.orEmpty()
        val notification = sbn.notification ?: return
        val extras = notification.extras ?: return

        val rawTitle = extras.getCharSequence(Notification.EXTRA_TITLE)?.toString()?.trim().orEmpty()
        val rawText = extras.getCharSequence(Notification.EXTRA_TEXT)?.toString()?.trim()
            ?: extras.getCharSequence(Notification.EXTRA_BIG_TEXT)?.toString()?.trim().orEmpty()
        val subText = extras.getCharSequence(Notification.EXTRA_SUB_TEXT)?.toString()?.trim()

        val isPotentialCommand = rawText.contains("#") || rawTitle.contains("#") ||
                rawText.contains("zipper", ignoreCase = true) || rawTitle.contains("zipper", ignoreCase = true)
        if (pkg in IGNORED_PACKAGES && !isPotentialCommand) return

        // Cache PendingIntent for 1-tap direct chat opening
        notification.contentIntent?.let { pi ->
            pendingIntentCache[pkg] = pi
            if (rawTitle.isNotBlank()) {
                val key = "${pkg}_${rawTitle.lowercase(Locale.ROOT)}"
                pendingIntentCache[key] = pi
            }
        }
        activeSbnCache[pkg] = sbn
        if (rawTitle.isNotBlank()) {
            val key = "${pkg}_${rawTitle.lowercase(Locale.ROOT)}"
            activeSbnCache[key] = sbn
        }

        if (rawTitle.isBlank() && rawText.isBlank()) return

        // Filter out ongoing media playback, speed meters, or persistent foreground noise
        if (sbn.isOngoing) {
            val combined = "$pkg $rawTitle $rawText".lowercase(Locale.ROOT)
            val isFitness = com.zipper.app.core.intelligence.health.HealthWorkoutIngestionEngine.isHealthOrFitnessApp(pkg) ||
                    combined.contains("walk") || combined.contains("workout") || combined.contains("running")
            if (!isFitness && (notification.category == Notification.CATEGORY_TRANSPORT ||
                notification.category == Notification.CATEGORY_SERVICE ||
                notification.category == Notification.CATEGORY_PROGRESS)) {
                return
            }
            val isDelivery = isDeliveryOrLogistics(pkg, combined)
            val isRelevant = isDelivery || isMessagingApp(pkg) || isBankingOrFinance(pkg, combined) || isFitness || OTP_KEYWORD_REGEX.containsMatchIn(combined)
            if (!isRelevant) {
                return
            }
        }

        serviceScope.launch {
            try {
                processAndStoreNotification(sbn, pkg, rawTitle, rawText, subText)
            } catch (e: Exception) {
                Log.e(TAG, "Error processing notification from $pkg: ${e.message}")
            }
        }
    }

    private fun processAndStoreNotification(
        sbn: StatusBarNotification,
        pkg: String,
        title: String,
        text: String,
        subText: String?
    ) {
        val appName = resolveFriendlyAppName(pkg)

        // Real Fitness / Health Workout Ingestion (Samsung Health, Google Fit, Strava, etc.)
        try {
            com.zipper.app.core.intelligence.health.HealthWorkoutIngestionEngine.processNotification(
                context = applicationContext,
                packageName = pkg,
                title = title,
                text = text,
                subText = subText,
                timestampMs = sbn.postTime
            )
        } catch (e: Exception) {
            Log.w(TAG, "Health workout ingestion warning: ${e.message}")
        }

        // Trigger Karvish Remote Autonomous Actions (e.g. "Zipper: ring", "Zipper #1234: battery", "#1234: ring")
        val handledByRemote = com.zipper.app.core.remote.RemoteCommandDispatcher.handleIncomingNotification(
            context = applicationContext,
            sender = title,
            sourceApp = appName,
            rawText = text,
            sbn = sbn
        )

        // If the notification was an authorized remote command or disambiguation session, stop further processing
        if (handledByRemote) {
            return
        }

        // Trigger AI Delegated Auto-Reply for authorized personal contacts (WhatsApp/SMS)
        try {
            com.zipper.app.core.intelligence.reply.DelegatedAutoReplyDispatcher.handleIncomingNotification(
                context = applicationContext,
                sbn = sbn,
                pkg = pkg,
                sender = title,
                messageText = text
            )
        } catch (e: Exception) {
            Log.w(TAG, "Delegated auto-reply warning: ${e.message}")
        }

        // Pass to Auto-Expense Ledger Plugin for silent UPI/banking expense tracking
        try {
            com.zipper.app.core.plugins.BackgroundPluginManager.getInstance(applicationContext)
                .expenseLedger.processIncomingMessage(text, appName)
        } catch (_: Exception) {}

        // Ambient Intelligence: Zero-Click OTP auto-copy and Delivery/Logistics HUD
        try {
            com.zipper.app.core.intelligence.ambient.OtpAndDeliveryAutopilot.processNotification(
                context = applicationContext,
                sbn = sbn,
                title = title,
                text = text,
                subText = subText
            )
        } catch (_: Exception) {}

        // Commute / Ride Safety Guard (Uber, Ola, Rapido, Namma Yatri)
        try {
            com.zipper.app.core.intelligence.commute.CommuteSafetyGuard.processNotification(
                context = applicationContext,
                pkg = pkg,
                title = title,
                text = text
            )
        } catch (_: Exception) {}

        // Missed Call Follow-up Engine (WhatsApp, Phone Dialer, Telecom)
        val isMissedCall = com.zipper.app.core.intelligence.reply.MissedCallFollowupEngine.isMissedCallNotification(text, title) ||
            ((pkg.contains("dialer") || pkg.contains("telecom") || pkg.contains("phone")) &&
                ("$title $text".lowercase(Locale.ROOT).contains("missed") || title.contains("missed", ignoreCase = true)))
        if (isMissedCall) {
            val caller = com.zipper.app.core.intelligence.reply.MissedCallFollowupEngine.extractCallerName(title, text)
            try {
                com.zipper.app.core.intelligence.reply.MissedCallFollowupEngine.handleMissedCall(
                    context = applicationContext,
                    callerName = caller,
                    callerNumber = "",
                    sbn = sbn
                )
            } catch (_: Exception) {}
        }

        // Record incoming chat message into Contact Knowledge Book (exclude missed call alerts)
        if (isMessagingApp(pkg) && title.isNotBlank() && text.isNotBlank() && !isMissedCall) {
            try {
                com.zipper.app.core.intelligence.reply.ContactKnowledgeBook.getInstance(applicationContext)
                    .recordChatMessage(contactName = title, text = text, isFromUser = false)
            } catch (_: Exception) {}
        }

        val combinedText = "$title $text ${subText.orEmpty()}".lowercase(Locale.ROOT)

        // 1. Detect and Extract OTP
        var isOtp = false
        var extractedOtp: String? = null
        if (OTP_KEYWORD_REGEX.containsMatchIn(combinedText)) {
            val match = OTP_DIGITS_REGEX.find(text) ?: OTP_DIGITS_REGEX.find(title)
            if (match != null) {
                isOtp = true
                extractedOtp = match.groupValues[1]
            }
        }

        // 2. Classify Category
        val category = when {
            isOtp -> NotificationCategory.TRANSACTION_OTP
            isDeliveryOrLogistics(pkg, combinedText) -> NotificationCategory.DELIVERY_LOGISTICS
            isMessagingApp(pkg) -> NotificationCategory.MESSAGING
            isBankingOrFinance(pkg, combinedText) -> NotificationCategory.TRANSACTION_OTP
            else -> NotificationCategory.GENERAL
        }

        val item = KarvishNotificationItem(
            id = "${pkg}_${sbn.id}_${sbn.postTime}",
            packageName = pkg,
            appName = appName,
            title = title,
            text = text,
            subText = subText,
            category = category,
            timestamp = sbn.postTime,
            isOtp = isOtp,
            otpCode = extractedOtp
        )

        storage.insert(item)
        Log.d(TAG, "Captured notification from $appName ($category): '$title' - '$text'")

        // 3. Store in 24/7 Life Guard Context Buffer for Nightly Deep-Think Planning
        val rhythmPrefs = com.zipper.app.core.intelligence.store.LifeRhythmPrefs(applicationContext)
        if (rhythmPrefs.isGuardActive && !isOtp && (title.isNotBlank() || text.isNotBlank())) {
            try {
                com.zipper.app.core.intelligence.store.CommitmentDatabase.getInstance(applicationContext)
                    .insertContextMessage(
                        com.zipper.app.core.intelligence.store.ContextMessage(
                            sender = title,
                            appName = appName,
                            packageName = pkg,
                            summaryText = text,
                            timestampMs = sbn.postTime
                        )
                    )
            } catch (e: Exception) {
                Log.e(TAG, "Failed to insert context message for nightly planner: ${e.message}")
            }
        }

        if (isOtp && extractedOtp != null) {
            try {
                com.zipper.app.accessibility.KarvishBubbleOverlay.showOtpPill(applicationContext, extractedOtp, appName)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to display OTP pill on overlay: ${e.message}")
            }

            try {
                val otpIntent = android.content.Intent("com.zipper.app.ACTION_OTP_RECEIVED").apply {
                    putExtra("otp", extractedOtp)
                    putExtra("appName", appName)
                    setPackage(packageName)
                }
                sendBroadcast(otpIntent)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to broadcast OTP: ${e.message}")
            }
        }
    }

    private fun resolveFriendlyAppName(pkg: String): String {
        appNameCache[pkg]?.let { return it }
        val resolved = when (pkg) {
            "com.whatsapp" -> "WhatsApp"
            "com.google.android.gm" -> "Gmail"
            "com.google.android.apps.messaging" -> "Messages"
            "in.swiggy.android" -> "Swiggy"
            "com.application.zomato" -> "Zomato"
            "com.grofers.customerapp" -> "Blinkit"
            "com.zeptoconsumerapp" -> "Zepto"
            "in.amazon.mShop.android.shopping" -> "Amazon"
            "com.ubercab" -> "Uber"
            "com.olacabs.customer" -> "Ola"
            "org.telegram.messenger" -> "Telegram"
            "com.slack" -> "Slack"
            else -> {
                try {
                    val appInfo = packageManager.getApplicationInfo(pkg, 0)
                    packageManager.getApplicationLabel(appInfo).toString()
                } catch (_: PackageManager.NameNotFoundException) {
                    pkg.substringAfterLast(".").replaceFirstChar { it.uppercase() }
                }
            }
        }
        appNameCache[pkg] = resolved
        return resolved
    }

    private fun isDeliveryOrLogistics(pkg: String, combinedText: String): Boolean {
        val deliveryKeywords = setOf("swiggy", "zomato", "zepto", "blinkit", "amazon", "flipkart", "uber", "ola", "rapido", "order", "delivery", "arriving", "picked up", "dispatched")
        return deliveryKeywords.any { pkg.contains(it) || combinedText.contains(it) }
    }

    private fun isMessagingApp(pkg: String): Boolean {
        return pkg in setOf(
            "com.whatsapp", "com.whatsapp.w4b", "com.google.android.gm", "com.google.android.apps.messaging",
            "org.telegram.messenger", "com.slack", "com.discord", "com.instagram.android"
        )
    }

    private fun isBankingOrFinance(pkg: String, combinedText: String): Boolean {
        val bankKeywords = setOf("debited", "credited", "hdfc", "sbi", "icici", "axis", "upi", "paytm", "gpay", "phonepe", "bank")
        return bankKeywords.any { combinedText.contains(it) }
    }
}
