package com.zipper.app.telephony

import android.app.Activity
import android.app.role.RoleManager
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.provider.ContactsContract
import android.telecom.TelecomManager
import android.util.Log

/**
 * Manages autonomous call initiation, contact lookup, and Default Dialer role requests.
 */
object KarvishCallManager {
    private const val TAG = "KarvishCallManager"

    /**
     * Checks if the app is currently the default phone/dialer app.
     */
    fun isDefaultDialer(context: Context): Boolean {
        val telecomManager = context.getSystemService(Context.TELECOM_SERVICE) as? TelecomManager
        return telecomManager?.defaultDialerPackage == context.packageName
    }

    /**
     * Requests the Default Dialer role from the user via standard Android dialog.
     */
    fun requestDefaultDialer(activity: Activity, requestCode: Int = 101) {
        if (isDefaultDialer(activity)) return

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val roleManager = activity.getSystemService(Context.ROLE_SERVICE) as? RoleManager
            if (roleManager?.isRoleAvailable(RoleManager.ROLE_DIALER) == true) {
                val intent = roleManager.createRequestRoleIntent(RoleManager.ROLE_DIALER)
                activity.startActivityForResult(intent, requestCode)
                return
            }
        }

        // Fallback for older Android / OEM standard prompt
        val intent = Intent(TelecomManager.ACTION_CHANGE_DEFAULT_DIALER).apply {
            putExtra(TelecomManager.EXTRA_CHANGE_DEFAULT_DIALER_PACKAGE_NAME, activity.packageName)
        }
        activity.startActivityForResult(intent, requestCode)
    }

    data class CallGoalInfo(
        val contactQuery: String,
        val message: String
    )

    /**
     * Parses ANY goal with ANY contact name or phone number, and ANY message.
     */
    fun parseCallGoal(goal: String): CallGoalInfo? {
        val clean = goal.trim()
        val patterns = listOf(
            Regex("""(?i)^(?:call|phone|dial)\s+([^,:]+?)\s+(?:and\s+say|to\s+say|saying|say)\s+(.+)$"""),
            Regex("""(?i)^(?:call|phone|dial)\s+([^,:]+?)\s*:\s*(.+)$"""),
            Regex("""(?i)^(?:call|phone|dial)\s+(.+)$""")
        )

        for (pattern in patterns) {
            val match = pattern.find(clean)
            if (match != null) {
                val contact = match.groupValues[1].trim()
                val msg = match.groupValues.getOrNull(2)?.trim().orEmpty()
                val finalMsg = if (msg.isNotBlank()) {
                    msg
                } else {
                    "Hello $contact, this is an automated message from Karthik: I will come in a few minutes. Thank you."
                }
                return CallGoalInfo(contact, finalMsg)
            }
        }
        return null
    }

    /**
     * Searches device contacts by name and returns the first matching phone number.
     * Supports direct phone numbers (+91..., 0..., digits) or names (Amma, Pooja, John, etc.).
     */
    fun lookupContactNumber(context: Context, contactQuery: String): String? {
        val cleanQuery = contactQuery.trim()

        // 1. Direct phone number check
        if (cleanQuery.matches(Regex("""^\+?[0-9\s\-()]{3,20}$"""))) {
            return cleanQuery.replace(Regex("[^0-9+]"), "")
        }

        // 2. Search device contacts
        val search = cleanQuery.lowercase()
        val uri = ContactsContract.CommonDataKinds.Phone.CONTENT_URI
        val projection = arrayOf(
            ContactsContract.CommonDataKinds.Phone.NUMBER,
            ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME
        )

        try {
            val cursor = context.contentResolver.query(
                uri,
                projection,
                null,
                null,
                null
            )

            var bestMatch: String? = null

            cursor?.use { c ->
                val numIdx = c.getColumnIndex(ContactsContract.CommonDataKinds.Phone.NUMBER)
                val nameIdx = c.getColumnIndex(ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME)

                while (c.moveToNext()) {
                    val name = c.getString(nameIdx) ?: ""
                    val number = c.getString(numIdx) ?: ""
                    val nameLower = name.lowercase().trim()

                    // Exact match takes priority
                    if (nameLower == search) {
                        Log.i(TAG, "Found exact contact match: '$name' -> $number")
                        return number.replace(Regex("[^0-9+]"), "")
                    }

                    // Partial / contains match as candidate
                    if (bestMatch == null && (nameLower.contains(search) || search.contains(nameLower))) {
                        Log.i(TAG, "Found candidate contact match: '$name' -> $number")
                        bestMatch = number.replace(Regex("[^0-9+]"), "")
                    }
                }
            }

            if (bestMatch != null) return bestMatch
        } catch (e: Exception) {
            Log.e(TAG, "Error querying contacts: ${e.message}", e)
        }

        Log.w(TAG, "Contact '$contactQuery' not found in device contacts.")
        return null
    }

    /**
     * Places the autonomous call to any contact or phone number.
     * Routes through VoIPBin Cloud AI if configured and enabled, else falls back to Normal SIM call.
     */
    fun placeAutonomousCall(
        context: Context,
        contactName: String,
        targetNumber: String,
        messagePayload: String,
        onStatusUpdate: ((String) -> Unit)? = null
    ): Boolean {
        val cleanNumber = targetNumber.replace(Regex("[^0-9+]"), "").trim()
        if (cleanNumber.isBlank()) {
            Log.e(TAG, "Cannot place call: Phone number is empty.")
            return false
        }

        // 1. Check if VoIPBin Cloud AI Calling is configured and enabled
        val voipConfig = com.zipper.app.telephony.voip.VoipConfigManager(context).getConfig()
        if (voipConfig.isEnabled && voipConfig.isConfigured) {
            Log.i(TAG, "Routing autonomous call through VoIPBin Cloud AI to $contactName ($cleanNumber)")
            return com.zipper.app.telephony.voip.VoipBinCallSyncManager.startAutonomousVoipCall(
                context = context,
                contactName = contactName,
                targetNumber = cleanNumber,
                goalMessage = messagePayload,
                onStatusUpdate = onStatusUpdate
            )
        }

        // 2. Fallback to standard cellular SIM call
        Log.i(TAG, "Placing Normal SIM call to $contactName ($cleanNumber): '$messagePayload'")
        return NormalCallController.startNormalCall(
            context = context,
            targetName = contactName,
            targetNumber = cleanNumber,
            messagePayload = messagePayload
        )
    }
}
