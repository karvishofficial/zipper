package com.zipper.app.telephony.voip

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import android.util.Log
import androidx.core.app.NotificationCompat
import com.zipper.app.MainActivity
import com.zipper.app.R
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

/**
 * Manages live synchronization and lifecycle of VoIPBin cloud calls on Android.
 * Handles autonomous calling, live status polling, transcript syncing, and notifications.
 */
object VoipBinCallSyncManager {

    private const val TAG = "VoipBinSyncMgr"
    private const val NOTIFICATION_CHANNEL_ID = "zipper_voip_calls"
    private const val NOTIFICATION_ID_BASE = 8800

    private val scope = CoroutineScope(Dispatchers.IO)
    private var activePollJob: Job? = null

    private val _currentCallState = MutableStateFlow<VoipCallDetails?>(null)
    val currentCallState: StateFlow<VoipCallDetails?> = _currentCallState.asStateFlow()

    data class ActiveCallSession(
        val callId: String,
        val contactName: String,
        val phoneNumber: String,
        val initialGoal: String,
        val startTimeMs: Long = System.currentTimeMillis()
    )

    @Volatile
    private var currentSession: ActiveCallSession? = null

    /**
     * Starts an autonomous conversational AI call through VoIPBin.
     */
    fun startAutonomousVoipCall(
        context: Context,
        contactName: String,
        targetNumber: String,
        goalMessage: String,
        onStatusUpdate: ((String) -> Unit)? = null
    ): Boolean {
        val configMgr = VoipConfigManager(context)
        val config = configMgr.getConfig()

        if (!config.isConfigured) {
            Log.e(TAG, "Cannot start VoIP call: API key or Assistant ID missing.")
            onStatusUpdate?.invoke("⚠️ VoIPBin not configured. Please add API Key in Settings.")
            return false
        }

        val client = VoipBinClient(config)

        scope.launch {
            try {
                onStatusUpdate?.invoke("Connecting to VoIPBin Cloud for $contactName ($targetNumber)...")
                showCallNotification(
                    context = context,
                    title = "📞 Zipper Calling $contactName",
                    message = "Dialing $targetNumber via Cloud AI Assistant...",
                    ongoing = true
                )

                val result = client.createCall(
                    phoneNumber = targetNumber,
                    goal = goalMessage,
                    targetName = contactName
                )

                result.onSuccess { response ->
                    val session = ActiveCallSession(
                        callId = response.callId,
                        contactName = contactName,
                        phoneNumber = targetNumber,
                        initialGoal = goalMessage
                    )
                    currentSession = session
                    onStatusUpdate?.invoke("✅ Call Initiated [ID: ${response.callId}]. Conversing...")

                    // Start live monitoring loop
                    startCallMonitoring(context, client, session, onStatusUpdate)
                }.onFailure { err ->
                    Log.e(TAG, "Failed to initiate VoIP call: ${err.message}", err)
                    onStatusUpdate?.invoke("❌ Call initiation failed: ${err.message}")
                    showCallNotification(
                        context = context,
                        title = "❌ Zipper Call Failed",
                        message = "Could not call $contactName: ${err.message}",
                        ongoing = false
                    )
                }
            } catch (e: Exception) {
                Log.e(TAG, "Exception during VoIP call initiation", e)
                onStatusUpdate?.invoke("❌ Error: ${e.message}")
            }
        }

        return true
    }

    private fun startCallMonitoring(
        context: Context,
        client: VoipBinClient,
        session: ActiveCallSession,
        onStatusUpdate: ((String) -> Unit)?
    ) {
        activePollJob?.cancel()
        activePollJob = scope.launch {
            var consecutiveErrors = 0
            val maxPollDurationMs = 5 * 60 * 1000L // Max 5 minutes timeout
            val startMs = System.currentTimeMillis()

            while (isActive && (System.currentTimeMillis() - startMs) < maxPollDurationMs) {
                delay(3000)

                val statusResult = client.getCall(session.callId)
                statusResult.onSuccess { details ->
                    _currentCallState.value = details
                    consecutiveErrors = 0

                    Log.d(TAG, "Call ${session.callId} status: ${details.status}, duration: ${details.durationSeconds}s")

                    if (details.status == "in-progress" || details.status == "active") {
                        showCallNotification(
                            context = context,
                            title = "📞 In Call with ${session.contactName}",
                            message = if (details.transcript.isNotBlank()) {
                                "Zipper: ${details.transcript.takeLast(80)}"
                            } else {
                                "Conversing via Groq Llama 3.3..."
                            },
                            ongoing = true
                        )
                        onStatusUpdate?.invoke("📞 Connected with ${session.contactName} (${details.durationSeconds}s)")
                    }

                    if (details.isEnded) {
                        Log.i(TAG, "Call ${session.callId} ended with status: ${details.status}")
                        val summaryText = details.summary.ifBlank {
                            details.transcript.ifBlank { "Call completed normally with ${session.contactName}." }
                        }

                        showCallNotification(
                            context = context,
                            title = "✅ Call Completed: ${session.contactName}",
                            message = summaryText,
                            ongoing = false
                        )
                        onStatusUpdate?.invoke("✅ Call Ended (${details.status}): $summaryText")
                        currentSession = null
                        return@launch
                    }
                }.onFailure { err ->
                    consecutiveErrors++
                    Log.w(TAG, "Polling call status error ($consecutiveErrors): ${err.message}")
                    if (consecutiveErrors >= 5) {
                        Log.e(TAG, "Stopping polling due to repeated errors.")
                        onStatusUpdate?.invoke("⚠️ Lost live connection with call status.")
                        return@launch
                    }
                }
            }
        }
    }

    /**
     * Terminates the current active VoIP call if one is in progress.
     */
    fun endCurrentCall(context: Context, onComplete: ((Boolean) -> Unit)? = null) {
        val session = currentSession ?: run {
            onComplete?.invoke(false)
            return
        }

        val config = VoipConfigManager(context).getConfig()
        val client = VoipBinClient(config)

        scope.launch {
            activePollJob?.cancel()
            val result = client.endCall(session.callId)
            currentSession = null
            _currentCallState.value = null

            showCallNotification(
                context = context,
                title = "📞 Call Disconnected",
                message = "Ended call with ${session.contactName}",
                ongoing = false
            )
            onComplete?.invoke(result.isSuccess)
        }
    }

    private fun showCallNotification(
        context: Context,
        title: String,
        message: String,
        ongoing: Boolean
    ) {
        try {
            val notificationManager = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                val channel = NotificationChannel(
                    NOTIFICATION_CHANNEL_ID,
                    "Zipper AI Phone Calls",
                    if (ongoing) NotificationManager.IMPORTANCE_LOW else NotificationManager.IMPORTANCE_DEFAULT
                ).apply {
                    description = "Live progress and summaries of Zipper conversational AI calls"
                    enableVibration(!ongoing)
                }
                notificationManager.createNotificationChannel(channel)
            }

            val intent = Intent(context, MainActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
            }
            val pendingIntent = PendingIntent.getActivity(
                context,
                0,
                intent,
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
            )

            val builder = NotificationCompat.Builder(context, NOTIFICATION_CHANNEL_ID)
                .setSmallIcon(R.drawable.ic_launcher_foreground)
                .setContentTitle(title)
                .setContentText(message)
                .setStyle(NotificationCompat.BigTextStyle().bigText(message))
                .setOngoing(ongoing)
                .setAutoCancel(!ongoing)
                .setContentIntent(pendingIntent)
                .setPriority(if (ongoing) NotificationCompat.PRIORITY_LOW else NotificationCompat.PRIORITY_DEFAULT)

            notificationManager.notify(NOTIFICATION_ID_BASE, builder.build())
        } catch (e: Exception) {
            Log.e(TAG, "Error displaying VoIP call notification: ${e.message}", e)
        }
    }
}
