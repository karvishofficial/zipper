package com.zipper.app.core.intelligence.reply

import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject

/**
 * Represents a single turn in a 2-way conversation between the contact and Karthik (AI).
 */
data class ChatTurn(
    val sender: String, // e.g. "Rajamma" or "Karthik"
    val message: String,
    val timestamp: Long = System.currentTimeMillis()
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("sender", sender)
        put("message", message)
        put("timestamp", timestamp)
    }

    companion object {
        fun fromJson(json: JSONObject): ChatTurn {
            return ChatTurn(
                sender = json.optString("sender", "Unknown"),
                message = json.optString("message", ""),
                timestamp = json.optLong("timestamp", System.currentTimeMillis())
            )
        }
    }
}

/**
 * Thread-safe persistent 2-way conversation memory for delegated contacts.
 * Remembers both incoming messages and outgoing replies so the LLM has continuous
 * contextual awareness across the entire chat session.
 */
class DelegatedConversationMemory private constructor(context: Context) {

    private val prefs: SharedPreferences =
        context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    companion object {
        private const val TAG = "DelegatedChatMemory"
        private const val PREFS_NAME = "karvish_delegated_chat_memory"
        private const val MAX_TURNS_PER_CONTACT = 30

        @Volatile
        private var instance: DelegatedConversationMemory? = null

        fun getInstance(context: Context): DelegatedConversationMemory {
            return instance ?: synchronized(this) {
                instance ?: DelegatedConversationMemory(context).also { instance = it }
            }
        }

        private fun sanitizeKey(contactKey: String): String {
            return "chat_" + contactKey.trim().lowercase().replace(Regex("""[^a-z0-9_]"""), "_")
        }
        fun pruneWithAnchor(list: List<ChatTurn>, maxTurns: Int): List<ChatTurn> {
            if (list.size <= maxTurns) return list
            if (maxTurns <= 2) return list.takeLast(maxTurns)
            val anchor = list.take(2)
            val rollingTail = list.drop(2).takeLast(maxTurns - 2)
            return anchor + rollingTail
        }
    }

    @Synchronized
    fun recordMessage(contactKey: String, sender: String, text: String) {
        val cleanText = text.trim()
        if (cleanText.isBlank()) return

        val key = sanitizeKey(contactKey)
        val history = getRawHistory(contactKey).toMutableList()

        // Deduplicate if identical to immediately preceding turn from same sender
        if (history.isNotEmpty()) {
            val last = history.last()
            if (last.sender == sender && last.message == cleanText) {
                return
            }
        }

        history.add(ChatTurn(sender = sender, message = cleanText))

        // Retain rolling window of turns with Genesis Anchor (Preserve turns 1 & 2 forever!)
        val trimmed = pruneWithAnchor(history, MAX_TURNS_PER_CONTACT)

        val array = JSONArray()
        for (turn in trimmed) {
            array.put(turn.toJson())
        }

        prefs.edit().putString(key, array.toString()).apply()
        Log.i(TAG, "Recorded chat turn in memory for '$contactKey' by $sender: '$cleanText'")
    }

    @Synchronized
    private fun getRawHistory(contactKey: String): List<ChatTurn> {
        val key = sanitizeKey(contactKey)
        val raw = prefs.getString(key, null) ?: return emptyList()
        return try {
            val array = JSONArray(raw)
            val list = mutableListOf<ChatTurn>()
            for (i in 0 until array.length()) {
                list.add(ChatTurn.fromJson(array.getJSONObject(i)))
            }
            list
        } catch (e: Exception) {
            Log.w(TAG, "Error parsing raw chat history for $contactKey: ${e.message}")
            emptyList()
        }
    }

    @Synchronized
    fun getHistory(contactKey: String, maxTurns: Int = 18): List<ChatTurn> {
        val list = getRawHistory(contactKey)
        return pruneWithAnchor(list, maxTurns)
    }

    @Synchronized
    fun clearHistory(contactKey: String) {
        val key = sanitizeKey(contactKey)
        prefs.edit().remove(key).apply()
        Log.i(TAG, "Cleared chat memory for '$contactKey'")
    }
}
