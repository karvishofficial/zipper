package com.zipper.app.core.intelligence.reply

import java.util.Locale

/**
 * Speech-Act & Directive-to-Persona Translator.
 *
 * Converts Karthik's raw, 3rd-person, or ultra-short directives (e.g. "say I am not available to that boy",
 * "tell landlord rent tomorrow", "say no") into natural, polished 1st-person WhatsApp/SMS messages
 * tailored to the specific recipient's relationship and English level.
 */
object SpeechActTranslator {

    private val DIRECTIVE_PREFIXES = listOf(
        Regex("""(?i)^(?:zipper\s*,?\s*)?(?:say|tell\s+(?:him|her|them|that\s+boy|that\s+guy|to\s+him|to\s+her)|reply\s+(?:that|saying)?|message\s+(?:him|her)?\s*(?:saying|that)?|text\s+(?:him|her)?\s*(?:saying|that)?)\s+"""),
        Regex("""(?i)\s+(?:to\s+(?:that\s+boy|that\s+guy|him|her))[\s.]*$""")
    )

    /**
     * Translates a user directive into a contextual 1st-person message.
     */
    fun translateDirective(
        rawDirective: String,
        dossier: ContactDossier? = null,
        incomingMessageText: String? = null
    ): String {
        var clean = rawDirective.trim()
        for (pattern in DIRECTIVE_PREFIXES) {
            clean = clean.replace(pattern, "").trim()
        }

        // Remove surrounding quotes if present
        clean = clean.removeSurrounding("\"").removeSurrounding("'").trim()
        val lower = clean.lowercase(Locale.ROOT)
        val style = dossier?.languageStyle ?: LanguageStyle.CASUAL_SLANG
        val rel = dossier?.relationship?.lowercase(Locale.ROOT) ?: "friend"

        val isLandlordOrElder = rel.contains("landlord") || rel.contains("elder") || rel.contains("uncle") || rel.contains("owner")
        val isFamily = rel.contains("mom") || rel.contains("dad") || rel.contains("family") || rel.contains("ma") || rel.contains("amma")
        val isColleague = rel.contains("colleague") || rel.contains("work") || rel.contains("client") || rel.contains("boss")

        // 1. Not Available / Busy Directives
        if (lower.contains("not available") || lower.contains("not free") || lower.contains("busy") ||
            lower.contains("occupied") || lower.contains("tied up") || lower.contains("cant talk") || lower.contains("can't talk")
        ) {
            return when {
                isLandlordOrElder -> "Hello Uncle, I am occupied with work right now. Will call you as soon as I'm free. Thank you."
                isFamily -> "Hey Ma, a bit occupied with work right now, will call you in a bit ❤️"
                isColleague -> "Hey, tied up with client work right now, will get back to you shortly."
                style == LanguageStyle.CASUAL_SHORT -> "Busy bro, ping in 15m"
                style == LanguageStyle.BROKEN_BASIC_ENGLISH -> "In work bro, can't talk now. Text me what happened."
                else -> "Hey bro, not around right now, caught up with something. Will ping you later 👍"
            }
        }

        // 2. Out / Stepped Out Directives
        if (lower.contains("went out") || lower.contains("stepped out") || lower.contains("outside") || lower.contains("not at home")) {
            return when {
                isLandlordOrElder -> "Hello Uncle, I have stepped out for some urgent work. Will connect with you once back."
                isFamily -> "Hey, stepped out for a bit, will be home soon!"
                isColleague -> "Hey, stepped out for an appointment today, won't be able to make it."
                style == LanguageStyle.CASUAL_SHORT -> "Out bro, ping later"
                else -> "Hey bro, stepped out right now. Will catch up once I'm back!"
            }
        }

        // 3. Rent / Payment Directives
        if (lower.contains("rent") || lower.contains("payment") || lower.contains("money")) {
            val timing = if (lower.contains("tomorrow")) "by tomorrow evening" else "shortly"
            return when {
                isLandlordOrElder -> "Hello Uncle, I will transfer the rent $timing. Thank you."
                else -> "Hey, will settle the payment $timing 👍"
            }
        }

        // 4. Rejection / "Say No" / Decline
        if (lower == "no" || lower == "say no" || lower == "decline" || lower == "reject" || lower.contains("cant make it")) {
            return when {
                isLandlordOrElder -> "Hello, I won't be able to take this up right now. Thank you."
                isColleague -> "Hey, won't be able to make it today, tied up with other priorities."
                style == LanguageStyle.CASUAL_SHORT -> "Can't today bro"
                else -> "Can't make it today bro, tied up with something. Let's catch up later!"
            }
        }

        // 5. Short "Will Call Later" / "Ping Later"
        if (lower.contains("call later") || lower.contains("ping later") || lower.contains("talk later") || lower == "later") {
            return when {
                isLandlordOrElder -> "Hello Uncle, I will call you back shortly."
                isFamily -> "Will call you back in 15 mins!"
                style == LanguageStyle.CASUAL_SHORT -> "Call in 15m bro"
                else -> "Tied up bro, will call you back in a bit 👍"
            }
        }

        // 6. At Home Directives
        if (lower.contains("at home") || lower.contains("reached home") || lower.contains("in home") || lower.contains("maneli")) {
            return when {
                isLandlordOrElder -> "Hello Uncle, I am currently at home. Please let me know what happened."
                isFamily -> "Hey, I'm at home!"
                style == LanguageStyle.CASUAL_SHORT -> "At home bro"
                style == LanguageStyle.BROKEN_BASIC_ENGLISH -> "I am at home bro. Tell what happened."
                else -> "Ha bro, I'm at home only! What happened, all good?"
            }
        }

        // 7. Generic Polish: If the user provided a full thought in shorthand, convert 3rd to 1st person
        val firstPerson = clean
            .replace(Regex("""(?i)\bhe\s+is\b"""), "I am")
            .replace(Regex("""(?i)\bhe\s+has\b"""), "I have")
            .replace(Regex("""(?i)\bhe\s+will\b"""), "I will")
            .replace(Regex("""(?i)\bhe\s+can't\b"""), "I can't")
            .replace(Regex("""(?i)\bhe\s+cant\b"""), "I can't")
            .replace(Regex("""(?i)\bhe\s+wants\b"""), "I want")
            .replace(Regex("""(?i)\bhis\b"""), "my")
            .replace(Regex("""(?i)\bhim\b"""), "me")

        return when {
            style == LanguageStyle.CASUAL_SHORT && firstPerson.split(" ").size <= 4 -> firstPerson
            style == LanguageStyle.CASUAL_SLANG && !firstPerson.lowercase().contains("bro") -> "$firstPerson bro"
            else -> firstPerson
        }
    }
}
