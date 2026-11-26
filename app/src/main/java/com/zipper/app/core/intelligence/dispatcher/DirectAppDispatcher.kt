package com.zipper.app.core.intelligence.dispatcher

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.widget.Toast
import java.util.Locale

object DirectAppDispatcher {

    /**
     * Directly prepares and opens an email draft with pre-filled recipient, subject, body,
     * and pre-attached document/PDF with FileProvider read permissions.
     */
    fun dispatchEmailWithAttachment(
        context: Context,
        recipientEmail: String,
        subject: String,
        body: String,
        attachmentUri: Uri? = null
    ): Boolean {
        return try {
            val emailIntent = Intent(Intent.ACTION_SEND).apply {
                type = if (attachmentUri != null) "application/pdf" else "message/rfc822"
                if (recipientEmail.isNotBlank()) {
                    putExtra(Intent.EXTRA_EMAIL, arrayOf(recipientEmail))
                }
                putExtra(Intent.EXTRA_SUBJECT, subject.ifBlank { "Update from Karvish" })
                putExtra(Intent.EXTRA_TEXT, body)

                if (attachmentUri != null) {
                    putExtra(Intent.EXTRA_STREAM, attachmentUri)
                    addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                }

                // Prefer Gmail directly if installed
                setPackage("com.google.android.gm")
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }

            try {
                context.startActivity(emailIntent)
                true
            } catch (_: Exception) {
                // Fallback to system email chooser if Gmail package is missing
                emailIntent.setPackage(null)
                val chooser = Intent.createChooser(emailIntent, "Send Email via...").apply {
                    addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                }
                context.startActivity(chooser)
                true
            }
        } catch (e: Exception) {
            Toast.makeText(context, "Could not launch email app: ${e.message}", Toast.LENGTH_SHORT).show()
            false
        }
    }

    /**
     * Directly launches WhatsApp with pre-filled recipient and message.
     */
    fun dispatchWhatsApp(
        context: Context,
        recipientOrPhone: String,
        message: String
    ): Boolean {
        return try {
            val cleanPhone = recipientOrPhone.replace(Regex("[^0-9+]"), "")
            val sendIntent = Intent(Intent.ACTION_VIEW).apply {
                val uriStr = if (cleanPhone.length >= 10) {
                    "https://api.whatsapp.com/send?phone=$cleanPhone&text=${Uri.encode(message)}"
                } else {
                    "https://api.whatsapp.com/send?text=${Uri.encode(message)}"
                }
                data = Uri.parse(uriStr)
                setPackage("com.whatsapp")
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            context.startActivity(sendIntent)
            true
        } catch (_: Exception) {
            try {
                val shareIntent = Intent(Intent.ACTION_SEND).apply {
                    type = "text/plain"
                    putExtra(Intent.EXTRA_TEXT, message)
                    setPackage("com.whatsapp")
                    addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                }
                context.startActivity(shareIntent)
                true
            } catch (e: Exception) {
                Toast.makeText(context, "WhatsApp is not installed on this device", Toast.LENGTH_SHORT).show()
                false
            }
        }
    }

    /**
     * Launches the real physical fitness companion (Samsung Health or Google Fit).
     * Strictly avoids faking workouts or fabricating autonomous execution of human exercise.
     */
    fun launchSamsungHealth(context: Context): Boolean {
        val candidates = listOf(
            "com.sec.android.app.shealth",      // Samsung Health
            "com.google.android.apps.fitness",  // Google Fit
            "com.strava"                        // Strava
        )

        for (pkg in candidates) {
            val intent = context.packageManager.getLaunchIntentForPackage(pkg)
            if (intent != null) {
                intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                context.startActivity(intent)
                val appLabel = if (pkg.contains("shealth")) "Samsung Health" else if (pkg.contains("fitness")) "Google Fit" else "Strava"
                Toast.makeText(context, "🚀 Launching $appLabel for your workout...", Toast.LENGTH_SHORT).show()
                return true
            }
        }

        Toast.makeText(context, "Please open Samsung Health to track your workout.", Toast.LENGTH_LONG).show()
        return false
    }
}
