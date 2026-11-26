package com.zipper.app.core.intelligence.reply

import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.util.Locale

enum class LanguageStyle {
    CASUAL_SLANG,          // "bro", "macha", "scenes", Hinglish/Kannada slang
    BROKEN_BASIC_ENGLISH,  // Short unpolished English ("bro were u r", "u comming?")
    CASUAL_SHORT,          // 2-4 words ("k", "busy bro, ping in 15m")
    FORMAL_ENGLISH,        // Respectful, complete sentences (landlord, professor, client)
    KANNADA_REGIONAL       // Natural Kannada / Kanglish ("yavaga barthya?", "oorige barthini", "oota aaytha?")
}

data class ChatMessageSnippet(
    val sender: String,
    val text: String,
    val isFromUser: Boolean,
    val timestampMs: Long = System.currentTimeMillis()
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("sender", sender)
        put("text", text)
        put("isFromUser", isFromUser)
        put("timestampMs", timestampMs)
    }

    companion object {
        fun fromJson(json: JSONObject): ChatMessageSnippet {
            return ChatMessageSnippet(
                sender = json.optString("sender", "Contact"),
                text = json.optString("text", ""),
                isFromUser = json.optBoolean("isFromUser", false),
                timestampMs = json.optLong("timestampMs", System.currentTimeMillis())
            )
        }
    }
}

data class ContactDossier(
    val contactName: String,
    val phoneNumber: String = "",
    val relationship: String = "Friend", // "Close Friend", "College Project Partner", "Family", "Landlord"
    val memoryPoints: List<String> = emptyList(), // e.g. ["Did AI projects together", "Lives in Indiranagar"]
    val languageStyle: LanguageStyle = LanguageStyle.CASUAL_SLANG,
    val recentHistory: List<ChatMessageSnippet> = emptyList(),
    val doNotCommitWithoutApproval: Boolean = true,
    val lastUpdatedMs: Long = System.currentTimeMillis()
) {
    fun isElderOrFamily(): Boolean {
        val relLower = relationship.lowercase(Locale.ROOT)
        val isElderRel = relLower.contains("mother") || relLower.contains("mom") || relLower.contains("amma") ||
            relLower.contains("father") || relLower.contains("dad") || relLower.contains("appa") ||
            relLower.contains("uncle") || relLower.contains("aunty") || relLower.contains("aunt") ||
            relLower.contains("elder") || relLower.contains("landlord") || relLower.contains("professor") ||
            relLower.contains("family") || relLower.contains("relative")
        val hasElderMemory = memoryPoints.any {
            val p = it.lowercase(Locale.ROOT)
            p.contains("elder") || p.contains("mother") || p.contains("mom") || p.contains("amma") ||
                p.contains("father") || p.contains("dad") || p.contains("aunty") || p.contains("uncle") ||
                p.contains("landlord") || p.contains("relative")
        }
        return isElderRel || hasElderMemory
    }

    fun isKannadaPreferred(): Boolean {
        if (languageStyle == LanguageStyle.KANNADA_REGIONAL) return true
        return memoryPoints.any {
            val p = it.lowercase(Locale.ROOT)
            p.contains("kannada") || p.contains("only kannada") || p.contains("speaks kannada")
        }
    }

    /**
     * Resolves the proper culturally authentic honorific for this contact
     * without confusing aunts (Aunty) with mothers (Amma), etc.
     */
    fun resolveHonorific(): String {
        val rel = relationship.lowercase(Locale.ROOT)
        val name = contactName.lowercase(Locale.ROOT)
        val mem = memoryPoints.joinToString(" ").lowercase(Locale.ROOT)

        val isAunt = rel.contains("aunty") || rel.contains("aunt") || mem.contains("aunty") || mem.contains("aunt")
        val isMom = !isAunt && (rel.contains("mother") || rel.contains("mom") || rel == "amma" ||
            (name.endsWith("amma") && (rel.contains("parent") || mem.contains("mother"))))
        val isDad = rel.contains("father") || rel.contains("dad") || rel.contains("appa")
        val isUncle = rel.contains("uncle") || mem.contains("uncle")
        val isSir = rel.contains("landlord") || rel.contains("professor") || rel.contains("sir") || rel.contains("boss")

        return when {
            isAunt -> "Aunty"
            isMom -> "Amma"
            isDad -> "Appa"
            isUncle -> "Uncle"
            isSir -> "Sir"
            isElderOrFamily() -> "Aunty/Uncle"
            else -> ""
        }
    }

    fun toJson(): JSONObject = JSONObject().apply {
        put("contactName", contactName)
        put("phoneNumber", phoneNumber)
        put("relationship", relationship)
        put("memoryPoints", JSONArray(memoryPoints))
        put("languageStyle", languageStyle.name)
        val histArr = JSONArray()
        recentHistory.forEach { histArr.put(it.toJson()) }
        put("recentHistory", histArr)
        put("doNotCommitWithoutApproval", doNotCommitWithoutApproval)
        put("lastUpdatedMs", lastUpdatedMs)
    }

    companion object {
        fun fromJson(json: JSONObject): ContactDossier {
            val memArr = json.optJSONArray("memoryPoints")
            val points = mutableListOf<String>()
            if (memArr != null) {
                for (i in 0 until memArr.length()) {
                    points.add(memArr.optString(i))
                }
            }

            val histArr = json.optJSONArray("recentHistory")
            val history = mutableListOf<ChatMessageSnippet>()
            if (histArr != null) {
                for (i in 0 until histArr.length()) {
                    histArr.optJSONObject(i)?.let { history.add(ChatMessageSnippet.fromJson(it)) }
                }
            }

            val styleName = json.optString("languageStyle", LanguageStyle.CASUAL_SLANG.name)
            val style = try {
                LanguageStyle.valueOf(styleName)
            } catch (_: Exception) {
                LanguageStyle.CASUAL_SLANG
            }

            return ContactDossier(
                contactName = json.optString("contactName", "Contact"),
                phoneNumber = json.optString("phoneNumber", ""),
                relationship = json.optString("relationship", "Friend"),
                memoryPoints = points,
                languageStyle = style,
                recentHistory = history,
                doNotCommitWithoutApproval = json.optBoolean("doNotCommitWithoutApproval", true),
                lastUpdatedMs = json.optLong("lastUpdatedMs", System.currentTimeMillis())
            )
        }
    }
}

/**
 * 100% On-Device Contact Knowledge Book.
 * Maintains personal dossiers, shared history, and language level per contact.
 */
class ContactKnowledgeBook private constructor(context: Context) {

    private val prefs: SharedPreferences =
        context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    companion object {
        private const val TAG = "ContactKnowledgeBook"
        private const val PREFS_NAME = "zipper_contact_knowledge_prefs"
        private const val KEY_DOSSIERS = "contact_dossiers_json"

        @Volatile
        private var instance: ContactKnowledgeBook? = null

        fun getInstance(context: Context): ContactKnowledgeBook {
            return instance ?: synchronized(this) {
                instance ?: ContactKnowledgeBook(context).also { instance = it }
            }
        }

        fun getInstanceOrNull(): ContactKnowledgeBook? = instance

        fun detectLanguageStyle(texts: List<String>): LanguageStyle {
            if (texts.isEmpty()) return LanguageStyle.CASUAL_SLANG

            val combined = texts.joinToString(" ").lowercase(Locale.ROOT)

            // 1. Formal / Polite
            val isFormal = combined.contains("dear") || combined.contains("regards") ||
                combined.contains("kindly") || combined.contains("sir") || combined.contains("please share") ||
                combined.contains("convenient") || combined.contains("reschedule") || combined.contains("thank you")

            if (isFormal) return LanguageStyle.FORMAL_ENGLISH

            // 2. Broken / Basic English (Phonetic texting, non-native syntax)
            val isBrokenBasic = combined.contains("were u r") || combined.contains("u comming") ||
                combined.contains("call me fast") || combined.contains("wer r u") ||
                combined.contains("hw r u") || combined.contains("pls come") ||
                combined.contains("wat r u") || combined.contains("wat r u doin") ||
                combined.contains("whr r u") || combined.contains("tell na") ||
                combined.contains("why u not") || combined.contains("u there") ||
                combined.contains("were r u") || combined.contains("come fast") ||
                combined.contains(Regex("""\b(wr|wer|whr|hw|wat|wht|pls|plz|tel|bcoz|cum|cmng|gud|abt)\b"""))

            if (isBrokenBasic) return LanguageStyle.BROKEN_BASIC_ENGLISH

            // 3. Heavy Bangalore Casual Slang & Kanglish (macha, scenes, etc.)
            val hasHeavySlang = combined.contains("macha") || combined.contains("maga") ||
                combined.contains("yelli") || combined.contains("bhai") || combined.contains("kidhar") ||
                combined.contains("scenes") || combined.contains("broo") || combined.contains("da") ||
                combined.contains("yaar") || combined.contains("kya") || combined.contains("bol") ||
                combined.contains("guru")

            if (hasHeavySlang) return LanguageStyle.CASUAL_SLANG

            // 4. Kannada / Regional (Pure Kannada Script or Common Romanized Kannada Vocabulary)
            val hasKannadaScript = combined.any { it in '\u0C80'..'\u0CFF' }
            val isKannada = hasKannadaScript ||
                combined.contains("yavaga") || combined.contains("barthya") || combined.contains("baralva") ||
                combined.contains("barthini") || combined.contains("oorige") || combined.contains("mysore") ||
                combined.contains("hegiddira") || combined.contains("yaake") || combined.contains("elli iddiya") ||
                combined.contains("oota aaytha") || combined.contains("kelasa") ||
                combined.contains("maneli") || combined.contains("madthini") || combined.contains("sigana") ||
                combined.contains("haudu") || combined.contains("beda") || combined.contains("beku") ||
                combined.contains("banni") || combined.contains("bega")

            if (isKannada) return LanguageStyle.KANNADA_REGIONAL

            // 5. Ultra Short Texting (2-4 words per message)
            val isUltraShort = texts.all { it.trim().split(Regex("""\s+""")).size <= 4 }
            if (isUltraShort) return LanguageStyle.CASUAL_SHORT

            return LanguageStyle.CASUAL_SLANG
        }

        fun parseNaturalRememberCommand(rawText: String): Pair<String, String>? {
            val regex = Regex(
                """(?i)^(?:zipper\s*,?\s*|hey\s+zipper\s*,?\s*)?(?:remember\s+(?:about\s+)?|note\s+(?:down\s+)?about\s+)([a-zA-Z0-9\s]+?)[:,-]\s*(.+)"""
            )
            val match = regex.find(rawText.trim()) ?: return null
            val contact = match.groupValues[1].trim()
            val facts = match.groupValues[2].trim()
            if (contact.isNotBlank() && facts.isNotBlank()) {
                return Pair(contact, facts)
            }
            return null
        }
    }

    @Synchronized
    fun getAllDossiers(): List<ContactDossier> {
        val raw = prefs.getString(KEY_DOSSIERS, null) ?: return emptyList()
        return try {
            val arr = JSONArray(raw)
            val list = mutableListOf<ContactDossier>()
            for (i in 0 until arr.length()) {
                list.add(ContactDossier.fromJson(arr.getJSONObject(i)))
            }
            list
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse dossiers: ${e.message}")
            emptyList()
        }
    }

    @Synchronized
    fun getDossier(contactName: String): ContactDossier? {
        val clean = contactName.trim().lowercase(Locale.ROOT)
        if (clean.isBlank()) return null
        val all = getAllDossiers()
        // 1. Exact case-insensitive match
        all.firstOrNull { it.contactName.trim().equals(clean, ignoreCase = true) }?.let { return it }
        // 2. Alphanumeric match (ignoring spaces/punctuation)
        val cleanAlpha = clean.filter { it.isLetterOrDigit() }
        if (cleanAlpha.isNotBlank()) {
            all.firstOrNull { it.contactName.filter { c -> c.isLetterOrDigit() }.equals(cleanAlpha, ignoreCase = true) }?.let { return it }
        }
        // 3. Substring match
        return all.firstOrNull {
            val dClean = it.contactName.trim().lowercase(Locale.ROOT)
            clean.contains(dClean) || dClean.contains(clean)
        }
    }

    @Synchronized
    fun saveDossier(dossier: ContactDossier) {
        val current = getAllDossiers().toMutableList()
        val idx = current.indexOfFirst {
            it.contactName.equals(dossier.contactName, ignoreCase = true)
        }
        if (idx >= 0) {
            current[idx] = dossier.copy(lastUpdatedMs = System.currentTimeMillis())
        } else {
            current.add(dossier.copy(lastUpdatedMs = System.currentTimeMillis()))
        }
        persist(current)
        Log.i(TAG, "Saved Knowledge Dossier for '${dossier.contactName}' (${dossier.memoryPoints.size} memory points)")
    }

    @Synchronized
    fun addMemoryPoint(contactName: String, point: String, relationship: String? = null): ContactDossier {
        val existing = getDossier(contactName) ?: ContactDossier(
            contactName = contactName,
            relationship = relationship ?: "Friend"
        )
        val cleanPoint = point.trim()
        val updatedPoints = if (cleanPoint.isNotBlank() && !existing.memoryPoints.contains(cleanPoint)) {
            existing.memoryPoints + cleanPoint
        } else {
            existing.memoryPoints
        }

        val updated = existing.copy(
            relationship = relationship ?: existing.relationship,
            memoryPoints = updatedPoints
        )
        saveDossier(updated)
        return updated
    }

    @Synchronized
    fun recordChatMessage(contactName: String, text: String, isFromUser: Boolean) {
        val dossier = getDossier(contactName) ?: ContactDossier(contactName = contactName)
        val snippet = ChatMessageSnippet(
            sender = if (isFromUser) "Karthik" else contactName,
            text = text.trim(),
            isFromUser = isFromUser
        )

        // Keep rolling buffer of last 4 messages
        val updatedHistory = (dossier.recentHistory + snippet).takeLast(4)

        // Detect language style dynamically from contact messages only (never polluted by user's replies)
        val contactMessages = updatedHistory.filter { !it.isFromUser }.map { it.text }
        val detectedStyle = if (contactMessages.isNotEmpty()) {
            detectLanguageStyle(contactMessages)
        } else {
            dossier.languageStyle
        }

        val updated = dossier.copy(
            recentHistory = updatedHistory,
            languageStyle = detectedStyle
        )
        saveDossier(updated)
    }

    @Synchronized
    fun deleteMemoryPoint(contactName: String, point: String) {
        val existing = getDossier(contactName) ?: return
        val updated = existing.copy(memoryPoints = existing.memoryPoints.filterNot { it.equals(point, ignoreCase = true) })
        saveDossier(updated)
    }

    fun detectLanguageStyle(texts: List<String>): LanguageStyle =
        Companion.detectLanguageStyle(texts)

    fun parseNaturalRememberCommand(rawText: String): Pair<String, String>? =
        Companion.parseNaturalRememberCommand(rawText)

    private fun persist(dossiers: List<ContactDossier>) {
        val arr = JSONArray()
        dossiers.forEach { arr.put(it.toJson()) }
        prefs.edit().putString(KEY_DOSSIERS, arr.toString()).apply()
    }
}
