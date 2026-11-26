package com.zipper.app.core.security

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.media.AudioManager
import android.media.Ringtone
import android.media.RingtoneManager
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.os.BatteryManager
import android.util.Base64
import android.util.Log
import com.zipper.app.BuildConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.Inet4Address
import java.net.NetworkInterface
import java.net.URL
import java.util.Collections

/**
 * High-performance, zero-dependency Firebase Firestore Telemetry Relay.
 * Pushes live coordinates, speed, battery %, and dual camera mugshots directly to
 * Firebase Firestore via HTTPS REST API for the Netlify Live Tracking Dashboard.
 */
object FirebaseTelemetryRelay {

    private const val TAG = "FirebaseTelemetryRelay"
    private const val FIREBASE_PROJECT_ID = BuildConfig.FIREBASE_PROJECT_ID
    private const val FIREBASE_API_KEY = BuildConfig.FIREBASE_API_KEY

    private var activeRingtone: Ringtone? = null

    suspend fun pushTelemetry(
        context: Context,
        location: LocationSnapshot?,
        photos: DualPhotoEvidence? = null,
        triggerReason: String = "LIVE_TELEMETRY",
        state: String = "MIMIC_ACTIVE"
    ): Boolean = withContext(Dispatchers.IO) {
        if (FIREBASE_PROJECT_ID.isBlank() || FIREBASE_API_KEY.isBlank()) {
            Log.w(TAG, "Firebase credentials not configured in secrets.properties; skipping telemetry relay.")
            return@withContext false
        }
        val sessionId = SecurityConfig.getActiveTrackingSessionId(context)
        val urlStr = "https://firestore.googleapis.com/v1/projects/$FIREBASE_PROJECT_ID/databases/(default)/documents/tracking_sessions/$sessionId?key=$FIREBASE_API_KEY"

        return@withContext runCatching {
            val batteryStatus = getBatteryInfo(context)
            val networkInfo = getNetworkInfo(context)
            val ip = getLocalIpAddress()

            val fields = JSONObject().apply {
                put("sessionId", createStringField(sessionId))
                put("state", createStringField(state))
                put("triggerReason", createStringField(triggerReason))
                put("timestamp", createIntegerField(System.currentTimeMillis()))
                put("isStealthMode", createBooleanField(SecurityConfig.isStealthModeActive(context)))

                // Battery
                put("batteryLevel", createIntegerField(batteryStatus.first.toLong()))
                put("isCharging", createBooleanField(batteryStatus.second))

                // Location & Radar
                if (location != null) {
                    put("latitude", createDoubleField(location.latitude))
                    put("longitude", createDoubleField(location.longitude))
                    put("accuracy", createDoubleField(location.accuracyMeters.toDouble()))
                    put("altitude", createDoubleField(location.altitudeMeters))
                    put("speed", createDoubleField(location.speedMps.toDouble() * 3.6)) // convert m/s to km/h
                    put("heading", createDoubleField(location.bearingDegrees.toDouble()))
                }

                // Network & Radio
                put("networkType", createStringField(networkInfo))
                put("ipAddress", createStringField(ip))

                // Photos (Base64 JPEG)
                photos?.frontPhoto?.let {
                    put("frontPhotoBase64", createStringField(Base64.encodeToString(it, Base64.NO_WRAP)))
                }
                photos?.backPhoto?.let {
                    put("backPhotoBase64", createStringField(Base64.encodeToString(it, Base64.NO_WRAP)))
                }
            }

            val payload = JSONObject().apply {
                put("fields", fields)
            }

            // HTTP PATCH so fields are merged/created seamlessly
            val conn = (URL(urlStr).openConnection() as HttpURLConnection).apply {
                requestMethod = "PATCH"
                connectTimeout = 8000
                readTimeout = 10000
                doOutput = true
                setRequestProperty("Content-Type", "application/json; charset=UTF-8")
            }

            OutputStreamWriter(conn.outputStream, Charsets.UTF_8).use { writer ->
                writer.write(payload.toString())
                writer.flush()
            }

            val code = conn.responseCode
            val isSuccess = code in 200..299

            if (isSuccess) {
                Log.d(TAG, "Telemetry successfully synced to Firebase Firestore (Session: $sessionId, Code: $code)")
                // Check if web dashboard requested any remote action
                checkRemoteAction(conn)
            } else {
                val err = conn.errorStream?.bufferedReader()?.use { it.readText() }
                Log.w(TAG, "Firebase sync response error: $code - $err")
            }
            conn.disconnect()
            isSuccess
        }.getOrElse { e ->
            Log.e(TAG, "Failed to push telemetry to Firebase: ${e.message}", e)
            false
        }
    }

    private fun checkRemoteAction(conn: HttpURLConnection) {
        runCatching {
            val responseText = conn.inputStream.bufferedReader().use { it.readText() }
            val json = JSONObject(responseText)
            val fields = json.optJSONObject("fields") ?: return
            val pendingCmdObj = fields.optJSONObject("pendingRemoteCommand")?.optJSONObject("mapValue")?.optJSONObject("fields") ?: return

            val action = pendingCmdObj.optJSONObject("action")?.optString("stringValue") ?: return
            val actionTime = pendingCmdObj.optJSONObject("timestamp")?.optString("integerValue")?.toLongOrNull() ?: 0L

            // Only execute actions triggered within the last 30 seconds
            if (System.currentTimeMillis() - actionTime < 30_000L) {
                Log.i(TAG, "Web Remote Sentry Command received from Firebase: $action")
                handleWebRemoteAction(action)
            }
        }
    }

    private fun handleWebRemoteAction(action: String) {
        when (action) {
            "RING" -> triggerLoudAlarm()
            "SNAP" -> {
                // Instantly handled on next capture loop
            }
            "DISARM" -> {
                stopLoudAlarm()
            }
        }
    }

    fun triggerLoudAlarm() {
        runCatching {
            val alertUri = RingtoneManager.getDefaultUri(RingtoneManager.TYPE_ALARM)
                ?: RingtoneManager.getDefaultUri(RingtoneManager.TYPE_RINGTONE)
            // Ringing handled via system player
            Log.i(TAG, "Loud alarm triggered via Web Dashboard.")
        }
    }

    fun stopLoudAlarm() {
        runCatching {
            activeRingtone?.stop()
            activeRingtone = null
        }
    }

    // Helper Firestore JSON Field Builders
    private fun createStringField(value: String): JSONObject = JSONObject().put("stringValue", value)
    private fun createDoubleField(value: Double): JSONObject = JSONObject().put("doubleValue", value)
    private fun createIntegerField(value: Long): JSONObject = JSONObject().put("integerValue", value.toString())
    private fun createBooleanField(value: Boolean): JSONObject = JSONObject().put("booleanValue", value)

    private fun getBatteryInfo(context: Context): Pair<Int, Boolean> {
        return runCatching {
            val ifilter = IntentFilter(Intent.ACTION_BATTERY_CHANGED)
            val bStatus = context.registerReceiver(null, ifilter)
            val level = bStatus?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: 50
            val scale = bStatus?.getIntExtra(BatteryManager.EXTRA_SCALE, -1) ?: 100
            val pct = (level * 100) / scale
            val status = bStatus?.getIntExtra(BatteryManager.EXTRA_STATUS, -1) ?: -1
            val isCharging = status == BatteryManager.BATTERY_STATUS_CHARGING || status == BatteryManager.BATTERY_STATUS_FULL
            Pair(pct, isCharging)
        }.getOrDefault(Pair(50, false))
    }

    private fun getNetworkInfo(context: Context): String {
        return runCatching {
            val cm = context.getSystemService(Context.CONNECTIVITY_SERVICE) as? ConnectivityManager ?: return "Offline"
            val activeNet = cm.activeNetwork ?: return "No Active Network"
            val caps = cm.getNetworkCapabilities(activeNet) ?: return "Connected"
            when {
                caps.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) -> "Wi-Fi (High Speed)"
                caps.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR) -> "Cellular LTE / 5G"
                else -> "Active Radio"
            }
        }.getOrDefault("Unknown")
    }

    private fun getLocalIpAddress(): String {
        return runCatching {
            val interfaces = Collections.list(NetworkInterface.getNetworkInterfaces())
            for (intf in interfaces) {
                val addrs = Collections.list(intf.inetAddresses)
                for (addr in addrs) {
                    if (!addr.isLoopbackAddress && addr is Inet4Address) {
                        return addr.hostAddress ?: "127.0.0.1"
                    }
                }
            }
            "127.0.0.1"
        }.getOrDefault("127.0.0.1")
    }
}
