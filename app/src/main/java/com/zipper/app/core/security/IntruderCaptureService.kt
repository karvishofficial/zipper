package com.zipper.app.core.security

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import androidx.core.app.NotificationCompat
import kotlinx.coroutines.*

/**
 * High-priority background foreground service that coordinates the silent capture of
 * the front-camera intruder photo, GPS location, and instant dispatch to Telegram.
 */
class IntruderCaptureService : Service() {

    private val serviceScope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private var wakeLock: PowerManager.WakeLock? = null

    companion object {
        private const val TAG = "IntruderCaptureService"
        private const val CHANNEL_ID = "zipper_security_alerts"
        private const val NOTIFICATION_ID = 8821
        const val EXTRA_TRIGGER_REASON = "extra_trigger_reason"
        const val ACTION_STOP_TRACKING = "com.zipper.app.core.security.ACTION_STOP_TRACKING"

        fun start(context: Context, reason: String) {
            val intent = Intent(context, IntruderCaptureService::class.java).apply {
                putExtra(EXTRA_TRIGGER_REASON, reason)
            }
            try {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    context.startForegroundService(intent)
                } else {
                    context.startService(intent)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to start IntruderCaptureService: ${e.message}")
            }
        }

        fun stopTracking(context: Context) {
            val intent = Intent(context, IntruderCaptureService::class.java).apply {
                action = ACTION_STOP_TRACKING
            }
            context.startService(intent)
        }
    }

    override fun onCreate() {
        super.onCreate()
        acquireWakeLock()
        startForegroundNotification()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP_TRACKING) {
            Log.i(TAG, "Stopping all active intruder telemetry and tracking loops.")
            SecurityConfig.setLiveTrackingActive(applicationContext, false)
            stopSelf()
            return START_NOT_STICKY
        }

        val reason = intent?.getStringExtra(EXTRA_TRIGGER_REASON) ?: "INTRUDER_DETECTED"
        Log.i(TAG, "IntruderCaptureService started with reason: $reason")

        serviceScope.launch {
            try {
                SecurityConfig.recordIntruderIncident(applicationContext)

                // 1. Immediately capture BOTH front (face) and back (environment) photos
                val dualPhotos = SilentCameraEngine.captureBothPhotos(applicationContext)
                val location = LocationSnapshotHelper.getBestLocation(applicationContext)

                Log.d(TAG, "Captured: front=${dualPhotos.frontPhoto != null} (${dualPhotos.frontPhoto?.size ?: 0}B), back=${dualPhotos.backPhoto != null} (${dualPhotos.backPhoto?.size ?: 0}B), loc=${location?.latitude},${location?.longitude}")

                // 2. Dispatch to configured alert email and save local blackbox evidence
                EmailAlertDispatcher.sendEmergencyAlert(
                    context = applicationContext,
                    triggerReason = reason,
                    dualPhotos = dualPhotos,
                    location = location
                )

                // 2b. Instantly push live telemetry and photos to Firebase Firestore for the Netlify Live Dashboard
                FirebaseTelemetryRelay.pushTelemetry(
                    context = applicationContext,
                    location = location,
                    photos = dualPhotos,
                    triggerReason = reason,
                    state = if (SecurityConfig.isStealthModeActive(applicationContext)) "MIMIC_ACTIVE" else "ALERT_ACTIVE"
                )

                // 3. If reason is MIMIC or PROTECT, start continuous adaptive tracking
                if (reason.contains("MIMIC") || reason.contains("PROTECT") || reason.contains("LOCKDOWN") || reason.contains("POWER_VOL_DOWN")) {
                    SecurityConfig.setLiveTrackingActive(applicationContext, true)
                    var lastLat = location?.latitude ?: 0.0
                    var lastLng = location?.longitude ?: 0.0
                    var loopCount = 0

                    while (isActive && SecurityConfig.isLiveTrackingActive(applicationContext) && SecurityConfig.isStealthModeActive(applicationContext)) {
                        delay(6_000L) // Real-time pulse every 6 seconds
                        if (!SecurityConfig.isLiveTrackingActive(applicationContext) || !SecurityConfig.isStealthModeActive(applicationContext)) break

                        val newLoc = LocationSnapshotHelper.getBestLocation(applicationContext, timeoutMs = 3000L)
                        if (newLoc != null) {
                            val dist = FloatArray(1)
                            android.location.Location.distanceBetween(lastLat, lastLng, newLoc.latitude, newLoc.longitude, dist)
                            val hasMoved = dist[0] > 40f // Moved > 40 meters
                            loopCount++

                            // Always stream telemetry every 6s to Firebase, and email if moved or every 60s
                            val freshPhotos = if (hasMoved) {
                                SilentCameraEngine.captureBothPhotos(applicationContext)
                            } else {
                                DualPhotoEvidence(null, null)
                            }

                            // Push to Firebase Firestore in real-time
                            FirebaseTelemetryRelay.pushTelemetry(
                                context = applicationContext,
                                location = newLoc,
                                photos = if (hasMoved) freshPhotos else null,
                                triggerReason = if (hasMoved) "IN_MOTION (${dist[0].toInt()}m moved)" else "STATIONARY_HEARTBEAT"
                            )

                            if (hasMoved || loopCount >= 10) { // Every 60s or upon significant movement send email
                                loopCount = 0
                                lastLat = newLoc.latitude
                                lastLng = newLoc.longitude

                                EmailAlertDispatcher.sendEmergencyAlert(
                                    context = applicationContext,
                                    triggerReason = if (hasMoved) "IN_MOTION (${dist[0].toInt()}m moved)" else "STATIONARY_HEARTBEAT",
                                    dualPhotos = freshPhotos,
                                    location = newLoc
                                )
                            }
                        }
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error during intruder capture routine: ${e.message}", e)
            } finally {
                stopSelf()
            }
        }

        return START_NOT_STICKY
    }

    override fun onDestroy() {
        super.onDestroy()
        serviceScope.cancel()
        releaseWakeLock()
        Log.d(TAG, "IntruderCaptureService stopped cleanly")
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun acquireWakeLock() {
        val pm = getSystemService(Context.POWER_SERVICE) as? PowerManager
        wakeLock = pm?.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "Zipper:IntruderCaptureWakeLock")?.apply {
            acquire(15000L) // 15 seconds max safety timeout
        }
    }

    private fun releaseWakeLock() {
        runCatching {
            if (wakeLock?.isHeld == true) {
                wakeLock?.release()
            }
        }
    }

    private fun startForegroundNotification() {
        val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Device Protection",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Active anti-theft telemetry"
                setShowBadge(false)
            }
            nm.createNotificationChannel(channel)
        }

        val notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Security Guard")
            .setContentText("System integrity active")
            .setSmallIcon(android.R.drawable.ic_lock_lock)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setOngoing(true)
            .build()

        runCatching {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                var serviceType = ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC

                val hasCamera = androidx.core.content.ContextCompat.checkSelfPermission(
                    this, android.Manifest.permission.CAMERA
                ) == android.content.pm.PackageManager.PERMISSION_GRANTED

                val hasLocation = androidx.core.content.ContextCompat.checkSelfPermission(
                    this, android.Manifest.permission.ACCESS_FINE_LOCATION
                ) == android.content.pm.PackageManager.PERMISSION_GRANTED ||
                androidx.core.content.ContextCompat.checkSelfPermission(
                    this, android.Manifest.permission.ACCESS_COARSE_LOCATION
                ) == android.content.pm.PackageManager.PERMISSION_GRANTED

                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R && hasCamera) {
                    serviceType = serviceType or ServiceInfo.FOREGROUND_SERVICE_TYPE_CAMERA
                }
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE && hasLocation) {
                    serviceType = serviceType or ServiceInfo.FOREGROUND_SERVICE_TYPE_LOCATION
                }

                try {
                    startForeground(NOTIFICATION_ID, notification, serviceType)
                } catch (e: Exception) {
                    Log.w(TAG, "Failed startForeground with type $serviceType (${e.message}); retrying with DATA_SYNC")
                    try {
                        startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
                    } catch (e2: Exception) {
                        Log.w(TAG, "Failed DATA_SYNC startForeground (${e2.message}); falling back to basic")
                        startForeground(NOTIFICATION_ID, notification)
                    }
                }
            } else {
                startForeground(NOTIFICATION_ID, notification)
            }
        }.onFailure { err ->
            Log.e(TAG, "Failed startForegroundNotification safely: ${err.message}")
        }
    }
}
