package com.zipper.app.core.security

import android.content.Context
import android.os.BatteryManager
import android.util.Base64
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.BufferedReader
import java.io.BufferedWriter
import java.io.DataOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.InputStreamReader
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL
import java.nio.charset.StandardCharsets
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.UUID
import javax.net.ssl.SSLSocket
import javax.net.ssl.SSLSocketFactory

/**
 * Dispatches emergency anti-theft telemetry, dual intruder mugshots (front & back camera),
 * and live GPS maps coordinates directly to the owner's configured destination email.
 *
 * Employs direct SSL SMTP (smtp.gmail.com:465), custom webhook relay, and local blackbox storage.
 */
object EmailAlertDispatcher {

    private const val TAG = "EmailAlertDispatcher"
    private const val LINE_FEED = "\r\n"

    suspend fun sendEmergencyAlert(
        context: Context,
        triggerReason: String,
        dualPhotos: DualPhotoEvidence,
        location: LocationSnapshot?
    ): Boolean = withContext(Dispatchers.IO) {
        val targetEmail = SecurityConfig.getAlertEmail(context)

        // 1. Always persist both photos to local device storage as emergency blackbox evidence
        dualPhotos.frontPhoto?.let { saveEvidenceLocally(context, it, "FRONT_${triggerReason}") }
        dualPhotos.backPhoto?.let { saveEvidenceLocally(context, it, "BACK_${triggerReason}") }

        val timeStr = SimpleDateFormat("dd MMM yyyy, hh:mm:ss a", Locale.getDefault()).format(Date())
        val battery = getBatteryLevel(context)
        val mapsUrl = location?.mapsUrl ?: "https://maps.google.com"

        Log.i(TAG, "Dispatching Emergency Alert to $targetEmail (Trigger: $triggerReason, Battery: $battery%, Loc: ${location?.latitude},${location?.longitude})")

        var delivered = false

        // 2. Try direct SSL SMTP if App Password is configured
        val appPassword = SecurityConfig.getSmtpAppPassword(context)
        if (appPassword.isNotBlank()) {
            val bodyText = buildString {
                appendLine("🚨 ZIPPER EMERGENCY ALERT: PHONE SHIELD ACTIVATED")
                appendLine("==================================================")
                appendLine("Trigger Reason : $triggerReason")
                appendLine("Timestamp      : $timeStr")
                appendLine("Battery Level  : $battery%")
                val netlifyBase = SecurityConfig.getNetlifyBaseUrl(context)
                val sessionId = SecurityConfig.getActiveTrackingSessionId(context)
                val liveRadarUrl = "$netlifyBase/?session=$sessionId#key=1234"

                if (location != null) {
                    appendLine("Live GPS Loc   : ${location.latitude}, ${location.longitude} (accuracy: ~${location.accuracyMeters.toInt()}m)")
                    appendLine("Google Maps    : $mapsUrl")
                } else {
                    appendLine("GPS Status     : Acquiring high-accuracy fix...")
                }
                appendLine("==================================================")
                appendLine("🔴 LIVE TRACKING RADAR (ACTIVATED):")
                appendLine(liveRadarUrl)
                appendLine("(Open link above in any browser to watch live moving dot, battery, nearby places & intruder photos)")
                appendLine("==================================================")
                appendLine("Photos attached: Front Camera (Intruder) & Back Camera (Surroundings).")
            }

            delivered = SmtpDirectMailer.sendEmail(
                username = SecurityConfig.DEFAULT_ALERT_EMAIL,
                appPassword = appPassword,
                to = targetEmail,
                subject = "🚨 EMERGENCY: Phone Stolen Alert — Location & Dual Mugshots",
                bodyText = bodyText,
                frontPhoto = dualPhotos.frontPhoto,
                backPhoto = dualPhotos.backPhoto
            )
            if (delivered) {
                Log.i(TAG, "Alert successfully sent directly via Gmail SSL SMTP to $targetEmail")
            }
        }

        // 3. Try custom webhook if configured
        val webhookUrl = SecurityConfig.getCustomAlertWebhook(context)
        if (!delivered && webhookUrl.isNotBlank()) {
            delivered = dispatchViaWebhook(
                webhookUrl = webhookUrl,
                targetEmail = targetEmail,
                triggerReason = triggerReason,
                timeStr = timeStr,
                battery = battery,
                location = location,
                frontBytes = dualPhotos.frontPhoto,
                backBytes = dualPhotos.backPhoto
            )
        }

        // 4. Fallback: also dispatch to Telegram if token is set
        TelegramAlertDispatcher.sendIntruderAlert(
            context = context,
            triggerReason = triggerReason,
            photoBytes = dualPhotos.frontPhoto ?: dualPhotos.backPhoto,
            location = location
        )

        return@withContext delivered
    }

    suspend fun sendDisarmNotice(context: Context, reason: String) = withContext(Dispatchers.IO) {
        val targetEmail = SecurityConfig.getAlertEmail(context)
        val timeStr = SimpleDateFormat("dd MMM yyyy, hh:mm:ss a", Locale.getDefault()).format(Date())

        Log.i(TAG, "Dispatching Disarm notice to $targetEmail ($reason)")

        val appPassword = SecurityConfig.getSmtpAppPassword(context)
        if (appPassword.isNotBlank()) {
            val bodyText = "✅ ZIPPER SYSTEM DISARMED\n\nReason: $reason\nTime: $timeStr\n\nAll tracking loops, fake shutdowns, and silent camera captures have ceased."
            SmtpDirectMailer.sendEmail(
                username = SecurityConfig.DEFAULT_ALERT_EMAIL,
                appPassword = appPassword,
                to = targetEmail,
                subject = "✅ ALL CLEAR: Zipper Stealth Tracking Disarmed",
                bodyText = bodyText,
                frontPhoto = null,
                backPhoto = null
            )
        }

        val webhookUrl = SecurityConfig.getCustomAlertWebhook(context)
        if (webhookUrl.isNotBlank()) {
            runCatching {
                val json = JSONObject().apply {
                    put("to", targetEmail)
                    put("status", "DISARMED")
                    put("reason", reason)
                    put("timestamp", timeStr)
                }
                val url = URL(webhookUrl)
                val conn = (url.openConnection() as HttpURLConnection).apply {
                    requestMethod = "POST"
                    connectTimeout = 8000
                    readTimeout = 10000
                    doOutput = true
                    setRequestProperty("Content-Type", "application/json; charset=UTF-8")
                }
                conn.outputStream.use { it.write(json.toString().toByteArray(Charsets.UTF_8)) }
                conn.responseCode
                conn.disconnect()
            }
        }
    }

    private fun dispatchViaWebhook(
        webhookUrl: String,
        targetEmail: String,
        triggerReason: String,
        timeStr: String,
        battery: Int,
        location: LocationSnapshot?,
        frontBytes: ByteArray?,
        backBytes: ByteArray?
    ): Boolean {
        return runCatching {
            val boundary = "ZipperAlertBoundary_" + UUID.randomUUID().toString()
            val url = URL(webhookUrl)
            val conn = (url.openConnection() as HttpURLConnection).apply {
                requestMethod = "POST"
                connectTimeout = 12000
                readTimeout = 18000
                doOutput = true
                doInput = true
                useCaches = false
                setRequestProperty("Content-Type", "multipart/form-data; boundary=$boundary")
                setRequestProperty("User-Agent", "ZipperSecurityAgent/2.0")
            }

            DataOutputStream(conn.outputStream).use { dos ->
                addFormField(dos, boundary, "to", targetEmail)
                addFormField(dos, boundary, "subject", "🚨 EMERGENCY: Phone Stolen Alert")
                addFormField(dos, boundary, "trigger_reason", triggerReason)
                addFormField(dos, boundary, "timestamp", timeStr)
                addFormField(dos, boundary, "battery", "$battery%")
                if (location != null) {
                    addFormField(dos, boundary, "coordinates", "${location.latitude}, ${location.longitude}")
                    addFormField(dos, boundary, "accuracy", "~${location.accuracyMeters.toInt()}m")
                    addFormField(dos, boundary, "google_maps_url", location.mapsUrl)
                }

                if (frontBytes != null && frontBytes.isNotEmpty()) {
                    dos.writeBytes("--$boundary$LINE_FEED")
                    dos.writeBytes("Content-Disposition: form-data; name=\"front_photo\"; filename=\"front_mugshot.jpg\"$LINE_FEED")
                    dos.writeBytes("Content-Type: image/jpeg$LINE_FEED")
                    dos.writeBytes(LINE_FEED)
                    dos.write(frontBytes)
                    dos.writeBytes(LINE_FEED)
                }

                if (backBytes != null && backBytes.isNotEmpty()) {
                    dos.writeBytes("--$boundary$LINE_FEED")
                    dos.writeBytes("Content-Disposition: form-data; name=\"back_photo\"; filename=\"back_environment.jpg\"$LINE_FEED")
                    dos.writeBytes("Content-Type: image/jpeg$LINE_FEED")
                    dos.writeBytes(LINE_FEED)
                    dos.write(backBytes)
                    dos.writeBytes(LINE_FEED)
                }

                dos.writeBytes("--$boundary--$LINE_FEED")
                dos.flush()
            }

            val code = conn.responseCode
            conn.disconnect()
            Log.i(TAG, "Webhook response: $code")
            code in 200..299
        }.getOrDefault(false)
    }

    private fun addFormField(dos: DataOutputStream, boundary: String, name: String, value: String) {
        dos.writeBytes("--$boundary$LINE_FEED")
        dos.writeBytes("Content-Disposition: form-data; name=\"$name\"$LINE_FEED")
        dos.writeBytes("Content-Type: text/plain; charset=UTF-8$LINE_FEED")
        dos.writeBytes(LINE_FEED)
        dos.write(value.toByteArray(Charsets.UTF_8))
        dos.writeBytes(LINE_FEED)
    }

    private const val MAX_EVIDENCE_AGE_MS = 24 * 60 * 60 * 1000L // 1 day auto-purge

    private fun getBatteryLevel(context: Context): Int {
        val bm = context.getSystemService(Context.BATTERY_SERVICE) as? BatteryManager
        return bm?.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY) ?: -1
    }

    private fun saveEvidenceLocally(context: Context, bytes: ByteArray, reason: String) {
        runCatching {
            val dir = File(context.filesDir, "security_evidence").apply { mkdirs() }

            // Automatically purge evidence older than 24 hours (1 day) to preserve storage
            val now = System.currentTimeMillis()
            dir.listFiles()?.forEach { file ->
                if (file.isFile && (now - file.lastModified() > MAX_EVIDENCE_AGE_MS)) {
                    val deleted = file.delete()
                    Log.d(TAG, "Purged expired security evidence (>24h): ${file.name} (deleted=$deleted)")
                }
            }

            val cleanReason = reason.replace(Regex("[^a-zA-Z0-9_-]"), "_")
            val fileName = "evidence_${System.currentTimeMillis()}_$cleanReason.jpg"
            val file = File(dir, fileName)
            FileOutputStream(file).use { it.write(bytes) }
            Log.d(TAG, "Saved emergency evidence: ${file.absolutePath} (${bytes.size} bytes)")
        }
    }
}

/**
 * Lightweight, direct SSL SMTP client for Gmail (smtp.gmail.com:465).
 * Sends multipart/mixed emails with Base64-encoded image attachments directly.
 * Zero external libraries required.
 */
object SmtpDirectMailer {

    private const val TAG = "SmtpDirectMailer"

    fun sendEmail(
        host: String = "smtp.gmail.com",
        port: Int = 465,
        username: String,
        appPassword: String,
        to: String,
        subject: String,
        bodyText: String,
        frontPhoto: ByteArray?,
        backPhoto: ByteArray?
    ): Boolean {
        var socket: SSLSocket? = null
        return try {
            val socketFactory = SSLSocketFactory.getDefault()
            socket = socketFactory.createSocket(host, port) as SSLSocket
            socket.soTimeout = 15000

            val reader = BufferedReader(InputStreamReader(socket.inputStream, StandardCharsets.UTF_8))
            val writer = BufferedWriter(OutputStreamWriter(socket.outputStream, StandardCharsets.UTF_8))

            fun readResponse(): String {
                val line = reader.readLine() ?: ""
                Log.d(TAG, "SMTP <- $line")
                return line
            }

            fun sendCmd(cmd: String) {
                writer.write(cmd + "\r\n")
                writer.flush()
            }

            val greeting = readResponse()
            if (!greeting.startsWith("220")) {
                Log.e(TAG, "SMTP greeting failed: $greeting")
                return false
            }

            sendCmd("EHLO zipper.security.guard")
            var line = readResponse()
            while (line.startsWith("250-")) {
                line = readResponse()
            }

            sendCmd("AUTH LOGIN")
            readResponse() // 334 Username
            sendCmd(Base64.encodeToString(username.toByteArray(StandardCharsets.UTF_8), Base64.NO_WRAP))
            readResponse() // 334 Password
            sendCmd(Base64.encodeToString(appPassword.replace(" ", "").toByteArray(StandardCharsets.UTF_8), Base64.NO_WRAP))
            val authResp = readResponse()
            if (!authResp.startsWith("235")) {
                Log.e(TAG, "SMTP Authentication failed: $authResp")
                sendCmd("QUIT")
                return false
            }

            sendCmd("MAIL FROM:<$username>")
            readResponse()
            sendCmd("RCPT TO:<$to>")
            readResponse()
            sendCmd("DATA")
            readResponse()

            val boundary = "====_Zipper_Alert_" + System.currentTimeMillis() + "_===="
            writer.write("From: \"Zipper Emergency Shield\" <$username>\r\n")
            writer.write("To: <$to>\r\n")
            writer.write("Subject: $subject\r\n")
            writer.write("MIME-Version: 1.0\r\n")
            writer.write("Content-Type: multipart/mixed; boundary=\"$boundary\"\r\n\r\n")

            // Body text part
            writer.write("--$boundary\r\n")
            writer.write("Content-Type: text/plain; charset=UTF-8\r\n\r\n")
            writer.write(bodyText + "\r\n\r\n")

            // Attach Front Photo
            if (frontPhoto != null && frontPhoto.isNotEmpty()) {
                writer.write("--$boundary\r\n")
                writer.write("Content-Type: image/jpeg; name=\"thief_face_front.jpg\"\r\n")
                writer.write("Content-Transfer-Encoding: base64\r\n")
                writer.write("Content-Disposition: attachment; filename=\"thief_face_front.jpg\"\r\n\r\n")
                writer.write(Base64.encodeToString(frontPhoto, Base64.DEFAULT))
                writer.write("\r\n\r\n")
            }

            // Attach Back Photo
            if (backPhoto != null && backPhoto.isNotEmpty()) {
                writer.write("--$boundary\r\n")
                writer.write("Content-Type: image/jpeg; name=\"surroundings_back.jpg\"\r\n")
                writer.write("Content-Transfer-Encoding: base64\r\n")
                writer.write("Content-Disposition: attachment; filename=\"surroundings_back.jpg\"\r\n\r\n")
                writer.write(Base64.encodeToString(backPhoto, Base64.DEFAULT))
                writer.write("\r\n\r\n")
            }

            writer.write("--$boundary--\r\n")
            writer.write("\r\n.\r\n")
            writer.flush()

            val dataResp = readResponse()
            sendCmd("QUIT")
            dataResp.startsWith("250")
        } catch (e: Exception) {
            Log.e(TAG, "SMTP delivery failed: ${e.message}", e)
            false
        } finally {
            runCatching { socket?.close() }
        }
    }
}
