package com.zipper.app.core.local

import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.provider.ContactsContract
import android.util.Log

/**
 * 0ms Instant Local Action & Deep-Link Dispatcher.
 * Dispatches WhatsApp, Calls, SMS, Google Maps, YouTube, Quick Commerce, and App Launches
 * with zero UI latency using Android Intents and URL Schemes.
 */
object LocalActionDispatcher {

    private const val TAG = "LocalActionDispatcher"

    data class ContactCandidate(
        val name: String,
        val phoneNumber: String
    )

    data class DispatchResult(
        val handled: Boolean,
        val feedback: String? = null,
        val appName: String? = null,
        val disambiguationNeeded: Boolean = false,
        val contactCandidates: List<ContactCandidate> = emptyList(),
        val pendingAction: String? = null, // "WHATSAPP", "CALL", "SMS"
        val pendingMessage: String? = null
    )

    fun isPackageInstalled(context: Context, packageName: String): Boolean {
        return try {
            context.packageManager.getPackageInfo(packageName, 0)
            true
        } catch (_: PackageManager.NameNotFoundException) {
            false
        }
    }

    /**
     * Resolves contact name to all matching candidates in <10ms for disambiguation.
     */
    fun findContactCandidates(context: Context, contactName: String): List<ContactCandidate> {
        if (context.checkSelfPermission(android.Manifest.permission.READ_CONTACTS) != PackageManager.PERMISSION_GRANTED) {
            return emptyList()
        }
        val uri = ContactsContract.CommonDataKinds.Phone.CONTENT_URI
        val projection = arrayOf(
            ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME,
            ContactsContract.CommonDataKinds.Phone.NUMBER
        )
        val selection = "${ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME} LIKE ?"
        val selectionArgs = arrayOf("%$contactName%")
        val list = mutableListOf<ContactCandidate>()
        val seenNumbers = mutableSetOf<String>()

        try {
            context.contentResolver.query(uri, projection, selection, selectionArgs, null)?.use { cursor ->
                val nameIdx = cursor.getColumnIndex(ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME)
                val numberIdx = cursor.getColumnIndex(ContactsContract.CommonDataKinds.Phone.NUMBER)
                while (cursor.moveToNext()) {
                    val name = if (nameIdx != -1) cursor.getString(nameIdx) else contactName
                    val number = if (numberIdx != -1) cursor.getString(numberIdx) else ""
                    val clean = number.replace(Regex("[^0-9+]"), "")
                    if (clean.isNotBlank() && seenNumbers.add(clean)) {
                        list.add(ContactCandidate(name, number))
                        if (list.size >= 5) break
                    }
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Error looking up contact candidates: ${e.message}")
        }
        return list
    }

    /**
     * Resolves single contact name to phone number in <10ms from local SQLite if permission is granted.
     */
    fun findContactPhoneNumber(context: Context, contactName: String): String? {
        val candidates = findContactCandidates(context, contactName)
        return candidates.firstOrNull()?.phoneNumber
    }

    /**
     * Evaluates whether a user query is a 0ms local device action.
     * Returns DispatchResult(handled = true) if executed.
     */
    fun tryDispatch(context: Context, goal: String): DispatchResult {
        val trimmed = goal.trim()
        val lower = trimmed.lowercase()

        // 1. FLASHLIGHT / TORCH ("turn on flashlight", "torch on", "turn off flashlight")
        if (lower.contains("flashlight") || lower.contains("torch")) {
            val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as? android.hardware.camera2.CameraManager
            val cameraId = cameraManager?.cameraIdList?.firstOrNull()
            if (cameraManager != null && cameraId != null) {
                val enable = !lower.contains("off")
                return try {
                    cameraManager.setTorchMode(cameraId, enable)
                    DispatchResult(true, if (enable) "Flashlight turned ON 💡" else "Flashlight turned OFF 🌑", "System")
                } catch (e: Exception) {
                    DispatchResult(false)
                }
            }
        }

        // 2. ALARMS & TIMERS ("set alarm for 7:30 am", "set an alarm at 6 am", "alarm for 7", "set timer for 10 minutes")
        if (lower.contains("alarm") || lower.contains("alaram") || lower.contains("wake me up")) {
            val alarmResult = dispatchAlarm(context, lower)
            if (alarmResult.handled) return alarmResult
        }
        if (lower.contains("timer")) {
            val timerResult = dispatchTimer(context, lower)
            if (timerResult.handled) return timerResult
        }

        // 3. QUICK NOTES ("note: buy paneer tonight", "take note: meeting at 3", "add note:")
        if (lower.startsWith("note:") || lower.startsWith("note ") || lower.startsWith("take note") || lower.startsWith("add note")) {
            return dispatchNote(context, trimmed)
        }

        // 4. EMAIL ACTIONS ("email boss@example.com: I'll be late", "send email to john: update")
        if (lower.startsWith("email ") || lower.startsWith("send email to ") || lower.startsWith("mail to ") ||
            (lower.contains("@") && (lower.contains("email") || lower.contains("send email") || lower.contains("mail")))) {
            return dispatchEmail(context, trimmed)
        }

        // 5. WHATSAPP ACTIONS: Bypassed for full Karvish Autonomous Agent Mode
        // Let V62 Agent Policy handle WhatsApp navigation, contact search, typing, and sending!
        // (Direct background intent calls only if goal is not an agent goal)
        /*
        if (lower.contains("whatsapp") || (lower.startsWith("send message to") && lower.contains(":")) ||
            (lower.startsWith("message ") && lower.contains(":"))) {
            return dispatchWhatsApp(context, trimmed, lower)
        }
        */

        // 6. PHONE CALLS ("call Mom", "call 9876543210", "dial Rahul")
        if (lower.startsWith("call ") || lower.startsWith("dial ")) {
            val target = trimmed.substring(if (lower.startsWith("call ")) 5 else 5).trim()
            if (target.isNotBlank()) {
                if (target.all { it.isDigit() || it == '+' || it == '-' || it == ' ' }) {
                    return executeDirectCall(context, target)
                }
                val candidates = findContactCandidates(context, target)
                if (candidates.size > 1) {
                    return DispatchResult(
                        handled = true,
                        feedback = "Multiple contacts found for '$target'. Select who to call:",
                        appName = "Phone",
                        disambiguationNeeded = true,
                        contactCandidates = candidates,
                        pendingAction = "CALL"
                    )
                } else if (candidates.size == 1) {
                    return executeDirectCall(context, candidates[0].phoneNumber, candidates[0].name)
                } else {
                    val intent = Intent(Intent.ACTION_DIAL).apply { addFlags(Intent.FLAG_ACTIVITY_NEW_TASK) }
                    return try {
                        context.startActivity(intent)
                        DispatchResult(true, "Opening phone dialer...", "Phone")
                    } catch (_: Exception) {
                        DispatchResult(false)
                    }
                }
            }
        }

        // 7. SMS ("send sms to Mom: reaching soon", "text Rahul: hi")
        if (lower.startsWith("send sms to ") || lower.startsWith("sms ") || (lower.startsWith("text ") && lower.contains(":"))) {
            return dispatchSms(context, trimmed)
        }

        // 8. NAVIGATION / MAPS ("navigate to MG Road", "directions to airport", "open maps")
        if (lower.startsWith("navigate to ") || lower.startsWith("directions to ")) {
            val dest = trimmed.substringAfter("to ").trim()
            val uri = Uri.parse("google.navigation:q=${Uri.encode(dest)}&mode=d")
            val intent = Intent(Intent.ACTION_VIEW, uri).apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                if (isPackageInstalled(context, "com.google.android.apps.maps")) {
                    setPackage("com.google.android.apps.maps")
                }
            }
            return try {
                context.startActivity(intent)
                DispatchResult(true, "Starting navigation to $dest...", "Google Maps")
            } catch (e: Exception) {
                DispatchResult(false)
            }
        }

        // 9. YOUTUBE: Allow autonomous agent goals (search, play, open and search) to run in full Agent Mode!
        val isYouTubeAgentGoal = lower.contains(" and ") || lower.contains(" then ") || lower.startsWith("search ") || lower.startsWith("play ")
        if (!isYouTubeAgentGoal && lower.contains("youtube")) {
            val query = if (lower.contains("search ")) {
                trimmed.substringAfter("search ").substringBefore(" on youtube").trim()
            } else if (lower.startsWith("open youtube")) {
                ""
            } else {
                trimmed.replace(Regex("(?i)\\b(?:on|in|via)\\s+youtube\\b"), "").replace(Regex("(?i)\\byoutube\\b"), "").trim()
            }

            val intent = if (query.isNotBlank()) {
                Intent(Intent.ACTION_VIEW, Uri.parse("vnd.youtube://results?search_query=${Uri.encode(query)}")).apply {
                    addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                    if (isPackageInstalled(context, "com.google.android.youtube")) {
                        setPackage("com.google.android.youtube")
                    }
                }
            } else {
                context.packageManager.getLaunchIntentForPackage("com.google.android.youtube")?.apply {
                    addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                }
            }
            if (intent != null) {
                return try {
                    context.startActivity(intent)
                    DispatchResult(true, if (query.isNotBlank()) "Opening YouTube for '$query'..." else "Opening YouTube...", "YouTube")
                } catch (e: Exception) {
                    DispatchResult(false)
                }
            }
        }

        // 10. QUICK COMMERCE DIRECT SCHEMES ("open zepto cart", "open blinkit cart")
        if (lower.contains("zepto") && lower.contains("cart")) {
            val intent = Intent(Intent.ACTION_VIEW, Uri.parse("zepto://zeptonow.com/app/cart")).apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            return try {
                context.startActivity(intent)
                DispatchResult(true, "Opening Zepto Cart...", "Zepto")
            } catch (e: Exception) {
                DispatchResult(false)
            }
        }
        if (lower.contains("blinkit") && lower.contains("cart")) {
            val intent = Intent(Intent.ACTION_VIEW, Uri.parse("blinkit://cart")).apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            return try {
                context.startActivity(intent)
                DispatchResult(true, "Opening Blinkit Cart...", "Blinkit")
            } catch (e: Exception) {
                DispatchResult(false)
            }
        }

        // 11. GENERIC APP LAUNCH ("open calculator", "open spotify", "open settings")
        // If the query contains compound actions ("and", "then", "search", "send", "message", "find", "post"),
        // or targets Instagram, do NOT intercept—let Karvish Autonomous Agent handle it!
        val isCompoundAgentGoal = lower.contains(" and ") || lower.contains(" then ") ||
            lower.contains(" search ") || lower.contains(" send ") || lower.contains(" message ") ||
            lower.contains(" find ") || lower.contains(" click ") || lower.contains(" post ") ||
            lower.contains(" instagram ") || lower.startsWith("open instagram")
        if (!isCompoundAgentGoal && (lower.startsWith("open ") || lower.startsWith("launch "))) {
            val appTarget = trimmed.substring(5).trim().lowercase()
            val pkg = when {
                appTarget.contains("calculator") -> resolveCalculatorPackage(context)
                appTarget.contains("spotify") -> "com.spotify.music"
                appTarget.contains("settings") -> "com.android.settings"
                appTarget.contains("zepto") -> if (isPackageInstalled(context, "com.zeptoconsumerapp")) "com.zeptoconsumerapp" else "com.zepto.consumer"
                appTarget.contains("blinkit") -> "com.grofers.customerapp"
                appTarget.contains("swiggy") -> "in.swiggy.android"
                appTarget.contains("zomato") -> "com.application.zomato"
                appTarget.contains("amazon") -> if (isPackageInstalled(context, "in.amazon.mShop.android.shopping")) "in.amazon.mShop.android.shopping" else "com.amazon.mShop.android.shopping"
                appTarget.contains("flipkart") -> "com.flipkart.android"
                appTarget.contains("chrome") -> "com.android.chrome"
                else -> null
            }

            if (pkg != null) {
                val launchIntent = if (pkg == "com.android.settings") {
                    Intent(android.provider.Settings.ACTION_SETTINGS).apply { addFlags(Intent.FLAG_ACTIVITY_NEW_TASK) }
                } else {
                    context.packageManager.getLaunchIntentForPackage(pkg)?.apply { addFlags(Intent.FLAG_ACTIVITY_NEW_TASK) }
                }
                if (launchIntent != null) {
                    return try {
                        context.startActivity(launchIntent)
                        DispatchResult(true, "Opening $appTarget...", appTarget.replaceFirstChar { it.uppercase() })
                    } catch (e: Exception) {
                        DispatchResult(false)
                    }
                }
            }
        }

        return DispatchResult(false)
    }

    private fun dispatchWhatsApp(context: Context, goal: String, lower: String): DispatchResult {
        var recipient = ""
        var message = ""

        if (goal.contains(":")) {
            val parts = goal.split(":", limit = 2)
            val recipientPart = parts[0]
            message = parts[1].trim()
            recipient = recipientPart
                .replace(Regex("(?i)\\b(?:send|message|whatsapp|text|to|on)\\b"), "")
                .trim()
        } else {
            recipient = goal
                .replace(Regex("(?i)\\b(?:send|message|whatsapp|text|to|on|open)\\b"), "")
                .trim()
        }

        val isNumeric = recipient.isNotBlank() && recipient.all { it.isDigit() || it == '+' || it == '-' || it == ' ' }
        if (isNumeric) {
            val cleanPhone = recipient.replace(Regex("[^0-9]"), "")
            return executeDirectWhatsApp(context, cleanPhone, message)
        }

        if (recipient.isNotBlank()) {
            val candidates = findContactCandidates(context, recipient)
            if (candidates.size > 1) {
                return DispatchResult(
                    handled = true,
                    feedback = "Multiple contacts found for '$recipient'. Select who to message on WhatsApp:",
                    appName = "WhatsApp",
                    disambiguationNeeded = true,
                    contactCandidates = candidates,
                    pendingAction = "WHATSAPP",
                    pendingMessage = message
                )
            } else if (candidates.size == 1) {
                return executeDirectWhatsApp(context, candidates[0].phoneNumber, message, candidates[0].name)
            }
        }

        // Fallback: general WhatsApp launch or share
        return if (message.isNotBlank()) {
            val intent = Intent(Intent.ACTION_SEND).apply {
                type = "text/plain"
                putExtra(Intent.EXTRA_TEXT, message)
                setPackage("com.whatsapp")
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            try {
                context.startActivity(intent)
                DispatchResult(true, "Opening WhatsApp to send message...", "WhatsApp")
            } catch (e: Exception) {
                DispatchResult(false)
            }
        } else {
            val launchIntent = context.packageManager.getLaunchIntentForPackage("com.whatsapp")?.apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            if (launchIntent != null) {
                try {
                    context.startActivity(launchIntent)
                    DispatchResult(true, "Opening WhatsApp...", "WhatsApp")
                } catch (e: Exception) {
                    DispatchResult(false)
                }
            } else {
                DispatchResult(false)
            }
        }
    }

    private fun dispatchEmail(context: Context, goal: String): DispatchResult {
        var recipient = ""
        val subject = "Note from Zipper"
        var body = ""

        if (goal.contains(":")) {
            val parts = goal.split(":", limit = 2)
            val recipientPart = parts[0]
            body = parts[1].trim()
            recipient = recipientPart
                .replace(Regex("(?i)\\b(?:send|email|mail|to|on)\\b"), "")
                .trim()
        } else {
            recipient = goal
                .replace(Regex("(?i)\\b(?:send|email|mail|to|on)\\b"), "")
                .trim()
        }

        val uri = if (recipient.contains("@")) {
            Uri.parse("mailto:$recipient?subject=${Uri.encode(subject)}&body=${Uri.encode(body)}")
        } else {
            Uri.parse("mailto:?subject=${Uri.encode(subject)}&body=${Uri.encode(body)}")
        }

        val intent = Intent(Intent.ACTION_SENDTO, uri).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }

        return try {
            context.startActivity(intent)
            DispatchResult(
                true,
                if (recipient.isNotBlank()) "Opening email draft for $recipient..." else "Opening email compose...",
                "Email"
            )
        } catch (e: Exception) {
            val sendIntent = Intent(Intent.ACTION_SEND).apply {
                type = "message/rfc822"
                if (recipient.contains("@")) putExtra(Intent.EXTRA_EMAIL, arrayOf(recipient))
                putExtra(Intent.EXTRA_SUBJECT, subject)
                putExtra(Intent.EXTRA_TEXT, body)
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            try {
                context.startActivity(sendIntent)
                DispatchResult(true, "Opening email client...", "Email")
            } catch (_: Exception) {
                DispatchResult(false)
            }
        }
    }

    private fun dispatchNote(context: Context, goal: String): DispatchResult {
        val noteText = goal.substringAfter(":").ifBlank {
            goal.replace(Regex("(?i)\\b(?:take|add|write)?\\s*note\\b"), "").trim()
        }.trim()

        if (noteText.isBlank()) {
            return DispatchResult(false)
        }

        if (isPackageInstalled(context, "com.google.android.keep")) {
            val intent = Intent(Intent.ACTION_SEND).apply {
                type = "text/plain"
                putExtra(Intent.EXTRA_TEXT, noteText)
                setPackage("com.google.android.keep")
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            return try {
                context.startActivity(intent)
                DispatchResult(true, "Saved note in Google Keep: \"$noteText\" ✓", "Google Keep")
            } catch (e: Exception) {
                DispatchResult(false)
            }
        }

        // Generic text share to note apps
        val shareIntent = Intent(Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(Intent.EXTRA_TEXT, noteText)
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }
        return try {
            context.startActivity(shareIntent)
            DispatchResult(true, "Saved note: \"$noteText\" ✓", "Notes")
        } catch (e: Exception) {
            DispatchResult(false)
        }
    }

    private fun dispatchAlarm(context: Context, lower: String): DispatchResult {
        // Match expressions like "7:30 am", "7 am", "7pm", "at 7", "for 7", "6:00", etc.
        val timePattern = Regex("(?i)(?:at|for|set|alaram|alarm)?\\s*(\\d{1,2})(?::(\\d{2}))?\\s*(am|pm)?")
        val match = timePattern.findAll(lower).firstOrNull { m ->
            val h = m.groupValues[1].toIntOrNull() ?: -1
            h in 1..24
        }
        if (match != null) {
            var hour = match.groupValues[1].toIntOrNull() ?: 7
            val minute = match.groupValues[2].toIntOrNull() ?: 0
            val ampm = match.groupValues[3].lowercase()
            if (ampm == "pm" && hour < 12) hour += 12
            if (ampm == "am" && hour == 12) hour = 0

            val intent = Intent(android.provider.AlarmClock.ACTION_SET_ALARM).apply {
                putExtra(android.provider.AlarmClock.EXTRA_HOUR, hour)
                putExtra(android.provider.AlarmClock.EXTRA_MINUTES, minute)
                putExtra(android.provider.AlarmClock.EXTRA_MESSAGE, "Zipper Alarm")
                putExtra(android.provider.AlarmClock.EXTRA_SKIP_UI, true)
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            return try {
                context.startActivity(intent)
                val ampmLabel = if (hour >= 12) "PM" else "AM"
                val displayHour = if (hour % 12 == 0) 12 else hour % 12
                val formatted = String.format(java.util.Locale.getDefault(), "%d:%02d %s", displayHour, minute, ampmLabel)
                DispatchResult(true, "Alarm set silently for $formatted ✓", "Clock")
            } catch (e: Exception) {
                Log.w(TAG, "Failed to set alarm via AlarmClock: ${e.message}")
                DispatchResult(false)
            }
        }
        return DispatchResult(false)
    }

    private fun dispatchTimer(context: Context, lower: String): DispatchResult {
        val secPattern = Regex("(?i)(\\d+)\\s*(?:sec|second|secs|seconds)")
        val sec = secPattern.find(lower)?.groupValues?.get(1)?.toIntOrNull() ?: 0
        val explicitMinPattern = Regex("(?i)(\\d+)\\s*(?:min|minute|mins|minutes)")
        val min = explicitMinPattern.find(lower)?.groupValues?.get(1)?.toIntOrNull()
            ?: Regex("(?i)timer(?:\\s+for)?\\s+(\\d+)").find(lower)?.groupValues?.get(1)?.toIntOrNull()
            ?: 0
        val totalSec = (min * 60) + sec

        if (totalSec > 0) {
            val intent = Intent(android.provider.AlarmClock.ACTION_SET_TIMER).apply {
                putExtra(android.provider.AlarmClock.EXTRA_LENGTH, totalSec)
                putExtra(android.provider.AlarmClock.EXTRA_MESSAGE, "Zipper Timer")
                putExtra(android.provider.AlarmClock.EXTRA_SKIP_UI, true)
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            return try {
                context.startActivity(intent)
                val minDisplay = totalSec / 60
                val secDisplay = totalSec % 60
                val label = if (secDisplay > 0) "${minDisplay}m ${secDisplay}s" else "${minDisplay}m"
                DispatchResult(true, "Timer set for $label ✓", "Clock")
            } catch (e: Exception) {
                Log.w(TAG, "Failed to set timer: ${e.message}")
                DispatchResult(false)
            }
        }
        return DispatchResult(false)
    }

    private fun dispatchSms(context: Context, goal: String): DispatchResult {
        var recipient = ""
        var message = ""
        if (goal.contains(":")) {
            val parts = goal.split(":", limit = 2)
            recipient = parts[0].replace(Regex("(?i)\\b(?:send|sms|text|to|on)\\b"), "").trim()
            message = parts[1].trim()
        } else {
            recipient = goal.replace(Regex("(?i)\\b(?:send|sms|text|to|on)\\b"), "").trim()
        }

        if (recipient.isNotBlank()) {
            val isNumeric = recipient.all { it.isDigit() || it == '+' || it == '-' || it == ' ' }
            if (isNumeric) {
                return executeDirectSms(context, recipient, message)
            }
            val candidates = findContactCandidates(context, recipient)
            if (candidates.size > 1) {
                return DispatchResult(
                    handled = true,
                    feedback = "Multiple contacts found for '$recipient'. Select who to text via SMS:",
                    appName = "SMS",
                    disambiguationNeeded = true,
                    contactCandidates = candidates,
                    pendingAction = "SMS",
                    pendingMessage = message
                )
            } else if (candidates.size == 1) {
                return executeDirectSms(context, candidates[0].phoneNumber, message, candidates[0].name)
            }
        }

        val intent = Intent(Intent.ACTION_SENDTO, Uri.parse("smsto:")).apply {
            if (message.isNotBlank()) putExtra("sms_body", message)
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }
        return try {
            context.startActivity(intent)
            DispatchResult(true, "Opening SMS app...", "SMS")
        } catch (_: Exception) {
            DispatchResult(false)
        }
    }

    // Direct Execution Helpers (invoked directly or when user selects a candidate)
    fun executeDirectWhatsApp(context: Context, phoneNumber: String, message: String, contactName: String? = null): DispatchResult {
        val cleanPhone = phoneNumber.replace(Regex("[^0-9]"), "")
        val uri = Uri.parse("https://api.whatsapp.com/send?phone=$cleanPhone&text=${Uri.encode(message)}")
        val intent = Intent(Intent.ACTION_VIEW, uri).apply {
            setPackage("com.whatsapp")
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }
        return try {
            context.startActivity(intent)
            val target = contactName ?: phoneNumber
            DispatchResult(true, "Opening WhatsApp chat with $target...", "WhatsApp")
        } catch (e: Exception) {
            Log.w(TAG, "Failed direct WhatsApp send: ${e.message}")
            DispatchResult(false)
        }
    }

    fun executeDirectCall(context: Context, phoneNumber: String, contactName: String? = null): DispatchResult {
        val cleanPhone = phoneNumber.replace(" ", "")
        val intent = Intent(Intent.ACTION_DIAL, Uri.parse("tel:$cleanPhone")).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }
        return try {
            context.startActivity(intent)
            val target = contactName ?: phoneNumber
            DispatchResult(true, "Dialing $target...", "Phone")
        } catch (e: Exception) {
            DispatchResult(false)
        }
    }

    fun executeDirectSms(context: Context, phoneNumber: String, message: String, contactName: String? = null): DispatchResult {
        val cleanPhone = phoneNumber.replace(" ", "")
        val intent = Intent(Intent.ACTION_SENDTO, Uri.parse("smsto:$cleanPhone")).apply {
            putExtra("sms_body", message)
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }
        return try {
            context.startActivity(intent)
            val target = contactName ?: phoneNumber
            DispatchResult(true, "Drafting SMS to $target...", "SMS")
        } catch (e: Exception) {
            DispatchResult(false)
        }
    }

    private fun resolveCalculatorPackage(context: Context): String? {
        val calcPackages = listOf(
            "com.sec.android.app.popupcalculator", // Samsung
            "com.google.android.calculator",       // Pixel / Stock
            "com.miui.calculator",                 // Xiaomi
            "com.oneplus.calculator",              // OnePlus
            "com.coloros.calculator"               // Oppo / Realme
        )
        return calcPackages.firstOrNull { isPackageInstalled(context, it) }
    }
}
