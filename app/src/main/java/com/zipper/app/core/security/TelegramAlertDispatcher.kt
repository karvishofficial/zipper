package com.zipper.app.core.security

import android.content.Context
import android.os.BatteryManager
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.DataOutputStream
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.UUID

/**
 * Dispatches intruder mugshots and real-time GPS coordinates directly to the owner's
 * Telegram Bot channel using standard HTTPS multipart requests.
 * Zero external libraries needed; works entirely via Android standard HttpURLConnection.
 */
object TelegramAlertDispatcher {

    private const val TAG = "TelegramAlertDispatcher"
    private const val LINE_FEED = "\r\n"

    suspend fun sendIntruderAlert(
        context: Context,
        triggerReason: String,
        photoBytes: ByteArray?,
        location: LocationSnapshot?
    ): Boolean = withContext(Dispatchers.IO) {
        val botToken = SecurityConfig.getTelegramBotToken(context)
        val chatId = SecurityConfig.getTelegramChatId(context)

        // Also save photo to local device storage as emergency evidence
        if (photoBytes != null) {
            saveEvidenceLocally(context, photoBytes, triggerReason)
        }

        if (botToken.isBlank() || chatId.isBlank()) {
            Log.w(TAG, "Telegram Bot Token or Chat ID not configured. Photo saved locally.")
            return@withContext false
        }

        val caption = buildAlertCaption(context, triggerReason, location)

        return@withContext try {
            if (photoBytes != null && photoBytes.isNotEmpty()) {
                sendPhoto(botToken, chatId, photoBytes, caption)
            } else {
                sendMessage(botToken, chatId, caption)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to dispatch alert to Telegram: ${e.message}")
            false
        }
    }

    private fun buildAlertCaption(
        context: Context,
        triggerReason: String,
        location: LocationSnapshot?
    ): String {
        val timeStr = SimpleDateFormat("dd MMM yyyy, hh:mm:ss a", Locale.getDefault()).format(Date())
        val batteryLevel = getBatteryLevel(context)

        return buildString {
            append("🚨 *INTRUDER ALERT - ZIPPER SECURITY*\n\n")
            append("⚠️ *Trigger:* $triggerReason\n")
            append("🕒 *Time:* $timeStr\n")
            append("🔋 *Battery:* $batteryLevel%\n")

            if (location != null) {
                append("📍 *Coordinates:* `${location.latitude}, ${location.longitude}`\n")
                append("🎯 *Accuracy:* ~${location.accuracyMeters.toInt()}m\n")
                append("🗺️ *Live Map:* ${location.mapsUrl}\n")
            } else {
                append("📍 *Location:* Acquiring GPS fix in background...\n")
            }
            append("\n_Captured autonomously via Zipper Security Engine._")
        }
    }

    private fun sendPhoto(
        botToken: String,
        chatId: String,
        photoBytes: ByteArray,
        caption: String
    ): Boolean {
        val boundary = "ZipperSecBoundary" + UUID.randomUUID().toString()
        val url = URL("https://api.telegram.org/bot$botToken/sendPhoto")
        val conn = (url.openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = 10000
            readTimeout = 15000
            doOutput = true
            doInput = true
            useCaches = false
            setRequestProperty("Content-Type", "multipart/form-data; boundary=$boundary")
            setRequestProperty("User-Agent", "ZipperSecurity/1.0")
        }

        DataOutputStream(conn.outputStream).use { dos ->
            // chat_id field
            addFormField(dos, boundary, "chat_id", chatId)

            // caption field
            addFormField(dos, boundary, "caption", caption)

            // parse_mode field
            addFormField(dos, boundary, "parse_mode", "Markdown")

            // photo file
            dos.writeBytes("--$boundary$LINE_FEED")
            dos.writeBytes("Content-Disposition: form-data; name=\"photo\"; filename=\"intruder.jpg\"$LINE_FEED")
            dos.writeBytes("Content-Type: image/jpeg$LINE_FEED")
            dos.writeBytes(LINE_FEED)
            dos.write(photoBytes)
            dos.writeBytes(LINE_FEED)

            // End of multipart
            dos.writeBytes("--$boundary--$LINE_FEED")
            dos.flush()
        }

        val code = conn.responseCode
        conn.disconnect()
        Log.i(TAG, "Telegram sendPhoto response code: $code")
        return code in 200..299
    }

    private fun sendMessage(
        botToken: String,
        chatId: String,
        text: String
    ): Boolean {
        val boundary = "ZipperSecBoundary" + UUID.randomUUID().toString()
        val url = URL("https://api.telegram.org/bot$botToken/sendMessage")
        val conn = (url.openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = 8000
            readTimeout = 10000
            doOutput = true
            doInput = true
            useCaches = false
            setRequestProperty("Content-Type", "multipart/form-data; boundary=$boundary")
        }

        DataOutputStream(conn.outputStream).use { dos ->
            addFormField(dos, boundary, "chat_id", chatId)
            addFormField(dos, boundary, "text", text)
            addFormField(dos, boundary, "parse_mode", "Markdown")
            dos.writeBytes("--$boundary--$LINE_FEED")
            dos.flush()
        }

        val code = conn.responseCode
        conn.disconnect()
        Log.i(TAG, "Telegram sendMessage response code: $code")
        return code in 200..299
    }

    private fun addFormField(dos: DataOutputStream, boundary: String, name: String, value: String) {
        dos.writeBytes("--$boundary$LINE_FEED")
        dos.writeBytes("Content-Disposition: form-data; name=\"$name\"$LINE_FEED")
        dos.writeBytes("Content-Type: text/plain; charset=UTF-8$LINE_FEED")
        dos.writeBytes(LINE_FEED)
        dos.write(value.toByteArray(Charsets.UTF_8))
        dos.writeBytes(LINE_FEED)
    }

    private fun getBatteryLevel(context: Context): Int {
        val bm = context.getSystemService(Context.BATTERY_SERVICE) as? BatteryManager
        return bm?.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY) ?: -1
    }

    private fun saveEvidenceLocally(context: Context, bytes: ByteArray, reason: String) {
        runCatching {
            val dir = File(context.filesDir, "security_evidence").apply { mkdirs() }
            val fileName = "intruder_${System.currentTimeMillis()}_${reason.replace(" ", "_")}.jpg"
            val file = File(dir, fileName)
            FileOutputStream(file).use { it.write(bytes) }
            Log.d(TAG, "Evidence saved locally to: ${file.absolutePath}")
        }
    }
}
