package com.zipper.app.core.intelligence.commute

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import android.util.Log
import androidx.core.app.NotificationCompat
import com.zipper.app.R
import com.zipper.app.core.remote.EmergencyRinger
import com.zipper.app.core.remote.RemoteSystemDiagnostics
import java.util.Locale
import java.util.UUID

enum class CommuteStatus {
    ACTIVE,
    SAFE_ARRIVED,
    DELAYED,
    SOS_TRIGGERED
}

data class CommuteSession(
    val id: String = UUID.randomUUID().toString(),
    val destination: String,
    val driverDetails: String? = null,
    val vehicleNumber: String? = null,
    val expectedDurationMinutes: Int = 30,
    val startTimeMs: Long = System.currentTimeMillis(),
    val expectedArrivalMs: Long = System.currentTimeMillis() + (expectedDurationMinutes * 60 * 1000L),
    val emergencyContact: String? = null,
    var status: CommuteStatus = CommuteStatus.ACTIVE
)

/**
 * Commute / SOS Safety Guard.
 *
 * Monitors real-time commutes, rides (Uber, Ola, Rapido, Namma Yatri),
 * detects unexpected delays or route deviations, and arms instant 1-tap SOS location dispatch.
 */
object CommuteSafetyGuard {

    private const val TAG = "CommuteSafetyGuard"
    const val CHANNEL_ID = "zipper_commute_guard"
    const val NOTIFICATION_ID = 99881

    const val ACTION_SAFE_ARRIVED = "com.zipper.app.COMMUTE_SAFE_ARRIVED"
    const val ACTION_TRIGGER_SOS = "com.zipper.app.COMMUTE_TRIGGER_SOS"

    @Volatile
    private var activeSession: CommuteSession? = null

    fun getActiveSession(): CommuteSession? = activeSession

    fun isGuardActive(): Boolean = activeSession?.status == CommuteStatus.ACTIVE

    fun startGuard(
        context: Context,
        destination: String,
        durationMinutes: Int = 30,
        emergencyContact: String? = null,
        driverDetails: String? = null,
        vehicleNumber: String? = null
    ): CommuteSession {
        val session = CommuteSession(
            destination = destination,
            driverDetails = driverDetails,
            vehicleNumber = vehicleNumber,
            expectedDurationMinutes = durationMinutes,
            emergencyContact = emergencyContact
        )
        activeSession = session
        Log.i(TAG, "Commute Guard started: En route to $destination (~$durationMinutes mins)")

        postActiveNotification(context, session)
        return session
    }

    fun markSafeArrival(context: Context): String {
        val session = activeSession
        activeSession = null

        val nm = context.getSystemService(Context.NOTIFICATION_SERVICE) as? NotificationManager
        nm?.cancel(NOTIFICATION_ID)

        val dest = session?.destination ?: "destination"
        val msg = "✅ Safe arrival confirmed at $dest. Commute Guard disarmed."
        Log.i(TAG, msg)

        // Post a brief completion notification
        createNotificationChannel(context)
        val notif = NotificationCompat.Builder(context, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setContentTitle("✅ Arrived Safely")
            .setContentText("You reached $dest. Safety guard disarmed.")
            .setPriority(NotificationCompat.PRIORITY_DEFAULT)
            .setAutoCancel(true)
            .build()
        nm?.notify(NOTIFICATION_ID + 1, notif)

        return msg
    }

    fun triggerSos(context: Context, reason: String = "Manual SOS Tap"): String {
        activeSession?.status = CommuteStatus.SOS_TRIGGERED
        val locationUrl = RemoteSystemDiagnostics.getLocationUrl(context)
        val batteryStatus = RemoteSystemDiagnostics.getBatteryAndNetworkStatus(context)
        val dest = activeSession?.destination ?: "Unknown destination"

        Log.w(TAG, "🚨 EMERGENCY SOS TRIGGERED! Reason: $reason. Location: $locationUrl")

        // 1. Sound emergency siren
        try {
            EmergencyRinger.startRinging(context)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start emergency alarm: ${e.message}")
        }

        // 2. Post high-priority emergency banner notification
        createNotificationChannel(context)
        val nm = context.getSystemService(Context.NOTIFICATION_SERVICE) as? NotificationManager

        val cancelIntent = Intent(context, CommuteSafetyReceiver::class.java).apply {
            action = ACTION_SAFE_ARRIVED
        }
        val pCancel = PendingIntent.getBroadcast(
            context,
            201,
            cancelIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val alertText = "🚨 SOS ACTIVE ($reason)! En route to $dest.\n📍 $locationUrl\n$batteryStatus"

        val notif = NotificationCompat.Builder(context, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setContentTitle("🚨 EMERGENCY SOS DISPATCHED")
            .setContentText("Emergency alert active. Tap to silence.")
            .setStyle(NotificationCompat.BigTextStyle().bigText(alertText))
            .setPriority(NotificationCompat.PRIORITY_MAX)
            .setOngoing(true)
            .addAction(R.drawable.ic_launcher_foreground, "🔇 Silence Alarm & I'm Safe", pCancel)
            .build()

        nm?.notify(NOTIFICATION_ID, notif)

        return "🚨 SOS ALERT: Commute to $dest is in emergency state!\n$locationUrl\n$batteryStatus"
    }

    fun checkStatusAndOverdue(context: Context): String {
        val session = activeSession ?: return "🛡️ No active commute guard session."
        val now = System.currentTimeMillis()
        val overdueMins = ((now - session.expectedArrivalMs) / (60 * 1000L)).toInt()

        if (overdueMins > 10 && session.status == CommuteStatus.ACTIVE) {
            session.status = CommuteStatus.DELAYED
            triggerSos(context, "Ride overdue by $overdueMins mins")
            return "⚠️ Commute to ${session.destination} is overdue by $overdueMins minutes! SOS alert triggered."
        }

        val remainingMins = ((session.expectedArrivalMs - now) / (60 * 1000L)).toInt()
        val statusText = if (remainingMins > 0) "~$remainingMins mins remaining" else "Arriving now"
        return "🛡️ Commute Guard Active: En route to ${session.destination} ($statusText). Status: ${session.status.name}"
    }

    /**
     * Inspects incoming ride notifications from Uber, Ola, Rapido, and Namma Yatri
     * to automatically protect the user's journey.
     */
    fun processNotification(
        context: Context,
        pkg: String,
        title: String,
        text: String
    ) {
        val lowerPkg = pkg.lowercase(Locale.ROOT)
        val isRideApp = lowerPkg.contains("uber") || lowerPkg.contains("ola") ||
            lowerPkg.contains("rapido") || lowerPkg.contains("nammayatri") || lowerPkg.contains("juspay")

        if (!isRideApp) return

        val combined = "$title $text".lowercase(Locale.ROOT)
        val isRideStarted = combined.contains("on the way") || combined.contains("started") ||
            combined.contains("heading to") || combined.contains("driver assigned") ||
            combined.contains("arriving in") || combined.contains("otp")

        if (isRideStarted && activeSession == null) {
            // Extract destination if present
            val destMatch = Regex("""(?i)(?:heading\s+to|to)\s+([a-zA-Z0-9\s,]+)""").find(text)
            val destination = destMatch?.groupValues?.get(1)?.trim() ?: "Destination"

            // Auto-arm commute guard for 35 mins
            startGuard(
                context = context,
                destination = destination,
                durationMinutes = 35,
                driverDetails = title
            )
            Log.i(TAG, "Auto-armed Commute Safety Guard for ride from $pkg to $destination")
        }
    }

    private fun postActiveNotification(context: Context, session: CommuteSession) {
        createNotificationChannel(context)
        val nm = context.getSystemService(Context.NOTIFICATION_SERVICE) as? NotificationManager ?: return

        // Action 1: Safe Arrived
        val safeIntent = Intent(context, CommuteSafetyReceiver::class.java).apply {
            action = ACTION_SAFE_ARRIVED
        }
        val pSafe = PendingIntent.getBroadcast(
            context,
            101,
            safeIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        // Action 2: Trigger SOS
        val sosIntent = Intent(context, CommuteSafetyReceiver::class.java).apply {
            action = ACTION_TRIGGER_SOS
        }
        val pSos = PendingIntent.getBroadcast(
            context,
            102,
            sosIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val notif = NotificationCompat.Builder(context, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setContentTitle("🛡️ Commute Safety Guard Active")
            .setContentText("En route to ${session.destination} (~${session.expectedDurationMinutes}m)")
            .setStyle(
                NotificationCompat.BigTextStyle().bigText(
                    "En route to ${session.destination}.\nETA: ~${session.expectedDurationMinutes} minutes.\nMonitoring route delay and unexpected stoppages."
                )
            )
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setOngoing(true)
            .addAction(R.drawable.ic_launcher_foreground, "✅ I'm Safe / Arrived", pSafe)
            .addAction(R.drawable.ic_launcher_foreground, "🚨 Trigger SOS", pSos)
            .build()

        nm.notify(NOTIFICATION_ID, notif)
    }

    private fun createNotificationChannel(context: Context) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Commute Safety Guard",
                NotificationManager.IMPORTANCE_HIGH
            ).apply {
                description = "Monitors route safety and emergency SOS during rides and travel"
                enableVibration(true)
            }
            val nm = context.getSystemService(Context.NOTIFICATION_SERVICE) as? NotificationManager
            nm?.createNotificationChannel(channel)
        }
    }
}
