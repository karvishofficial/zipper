package com.zipper.app.core.intelligence.reply

import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import org.json.JSONArray
import java.util.Locale

/**
 * Manages storage and matching for contacts delegated to the AI auto-reply agent.
 * Thread-safe and persistent across device reboots.
 */
class DelegatedContactManager private constructor(context: Context) {

    private val prefs: SharedPreferences =
        context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    companion object {
        private const val TAG = "DelegatedContactManager"
        private const val PREFS_NAME = "karvish_delegated_contacts_prefs"
        private const val KEY_CONTACTS = "delegated_contacts_json"

        @Volatile
        private var instance: DelegatedContactManager? = null

        fun getInstance(context: Context): DelegatedContactManager {
            return instance ?: synchronized(this) {
                instance ?: DelegatedContactManager(context).also { instance = it }
            }
        }
    }

    @Synchronized
    fun getContacts(): List<DelegatedContact> {
        val raw = prefs.getString(KEY_CONTACTS, null) ?: return emptyList()
        return try {
            val array = JSONArray(raw)
            val list = mutableListOf<DelegatedContact>()
            for (i in 0 until array.length()) {
                val obj = array.getJSONObject(i)
                list.add(DelegatedContact.fromJson(obj))
            }
            list
        } catch (e: Exception) {
            Log.e(TAG, "Error deserializing delegated contacts: ${e.message}")
            emptyList()
        }
    }

    @Synchronized
    fun getContact(id: String): DelegatedContact? {
        return getContacts().firstOrNull { it.id == id }
    }

    @Synchronized
    fun saveContact(contact: DelegatedContact) {
        val current = getContacts().toMutableList()
        val existingIndex = current.indexOfFirst { it.id == contact.id }
        if (existingIndex >= 0) {
            current[existingIndex] = contact
        } else {
            // Check if contact with same name exists
            val nameMatchIndex = current.indexOfFirst {
                it.contactName.equals(contact.contactName, ignoreCase = true)
            }
            if (nameMatchIndex >= 0) {
                current[nameMatchIndex] = contact
            } else {
                current.add(contact)
            }
        }
        persistContacts(current)
        Log.i(TAG, "Saved delegated contact '${contact.contactName}' (Enabled: ${contact.isEnabled})")
    }

    @Synchronized
    fun deleteContact(id: String) {
        val current = getContacts().toMutableList()
        val removed = current.removeAll { it.id == id }
        if (removed) {
            persistContacts(current)
            Log.i(TAG, "Deleted delegated contact with id: $id")
        }
    }

    @Synchronized
    fun toggleContact(id: String, isEnabled: Boolean) {
        val current = getContacts().toMutableList()
        val idx = current.indexOfFirst { it.id == id }
        if (idx >= 0) {
            current[idx] = current[idx].copy(isEnabled = isEnabled)
            persistContacts(current)
            Log.i(TAG, "Toggled contact '${current[idx].contactName}' -> $isEnabled")
        }
    }

    /**
     * Finds an enabled delegated contact matching the given notification sender name or phone.
     * Matches via exact name, case-insensitive match, or normalized phone digits.
     */
    @Synchronized
    fun findMatching(senderTitle: String): DelegatedContact? {
        if (senderTitle.isBlank()) return null
        val cleanSender = senderTitle.trim().lowercase(Locale.ROOT)
        val senderDigits = senderTitle.replace(Regex("""[^0-9]"""), "")
        val senderLast10 = if (senderDigits.length >= 10) senderDigits.takeLast(10) else ""

        val all = getContacts().filter { it.isEnabled && it.contactName.isNotBlank() }

        // 1. Exact Name Match (case-insensitive, trims spaces/emojis)
        all.firstOrNull { it.contactName.trim().lowercase(Locale.ROOT) == cleanSender }?.let {
            return it
        }

        // 2. Substring Name Containment (e.g. "Rajamma" matches "Rajamma +91..." or "Rajamma (Home)")
        all.firstOrNull { contact ->
            val cleanContactName = contact.contactName.trim().lowercase(Locale.ROOT)
            cleanSender.contains(cleanContactName) || cleanContactName.contains(cleanSender)
        }?.let { return it }

        // 3. Normalized 10-Digit Phone Match (India standard: ignores +91, 0, spaces, dashes)
        if (senderLast10.isNotBlank()) {
            all.firstOrNull { contact ->
                val contactDigits = contact.phoneNumber.replace(Regex("""[^0-9]"""), "")
                val contactLast10 = if (contactDigits.length >= 10) contactDigits.takeLast(10) else ""
                contactLast10.isNotBlank() && contactLast10 == senderLast10
            }?.let { return it }
        }

        return null
    }

    /**
     * Records that an AI reply was dispatched for the given contact id.
     */
    @Synchronized
    fun recordReply(id: String) {
        val current = getContacts().toMutableList()
        val idx = current.indexOfFirst { it.id == id }
        if (idx >= 0) {
            current[idx] = current[idx].recordReplySent()
            persistContacts(current)
            Log.i(TAG, "Recorded reply sent to '${current[idx].contactName}' (Count: ${current[idx].sessionReplyCount})")
        }
    }

    /**
     * Resets the active session budget for a contact so they start completely fresh.
     */
    @Synchronized
    fun resetSession(id: String) {
        val current = getContacts().toMutableList()
        val idx = current.indexOfFirst { it.id == id }
        if (idx >= 0) {
            current[idx] = current[idx].copy(sessionReplyCount = 0, lastReplyTimestampMs = 0L)
            persistContacts(current)
            Log.i(TAG, "Reset session budget for '${current[idx].contactName}'")
        }
    }

    private fun persistContacts(contacts: List<DelegatedContact>) {
        val array = JSONArray()
        for (c in contacts) {
            array.put(c.toJson())
        }
        prefs.edit().putString(KEY_CONTACTS, array.toString()).apply()
    }
}
