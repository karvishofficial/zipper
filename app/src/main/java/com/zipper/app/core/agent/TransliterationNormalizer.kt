package com.zipper.app.core.agent

/**
 * Lightweight On-Device Transliteration & Normalizer for Karvish V61.
 * Normalizes common Hinglish colloquialisms, Indian app nicknames, and frequent voice STT typos
 * before goal tokens are passed to the neural model.
 */
object TransliterationNormalizer {

    private val APP_NICKNAMES = mapOf(
        "yt" to "YouTube",
        "tube" to "YouTube",
        "insta" to "Instagram",
        "calc" to "Calculator",
        "whats" to "WhatsApp",
        "wa" to "WhatsApp",
        "msg" to "Messages",
        "zomto" to "Zomato",
        "swigy" to "Swiggy",
        "zepto" to "Zepto",
        "blnk" to "Blinkit",
        "blinkt" to "Blinkit",
        "instamrt" to "Instamart"
    )

    private val HINGLISH_TERM_MAPPINGS = mapOf(
        "doodh" to "milk",
        "dudhh" to "milk",
        "dood" to "milk",
        "dahi" to "curd",
        "atta" to "wheat flour",
        "aata" to "wheat flour",
        "chawal" to "rice",
        "tel" to "cooking oil",
        "pani" to "water",
        "paal" to "milk",
        "cheeni" to "sugar",
        "chini" to "sugar",
        "anda" to "eggs",
        "ande" to "eggs"
    )

    private val VERB_SLANG_MAPPINGS = mapOf(
        "serch" to "search",
        "look up" to "search",
        "find me" to "search",
        "grab" to "order",
        "lelo" to "order",
        "mangao" to "order",
        "kholo" to "open",
        "chalau" to "play",
        "lagao" to "play",
        "dekho" to "check",
        "fotography" to "photography",
        "spcex" to "SpaceX"
    )

    /**
     * Normalizes a raw user goal string while preserving the original intent.
     * Keeps words in natural lowercase, replacing known colloquial tokens.
     */
    fun normalize(rawGoal: String): String {
        if (rawGoal.isBlank()) return rawGoal
        val words = rawGoal.trim().split(Regex("\\s+"))
        val normalizedWords = words.map { word ->
            val clean = word.lowercase().trim { !it.isLetterOrDigit() }
            val suffix = word.substring(word.indexOfLast { it.isLetterOrDigit() } + 1)
            val prefix = word.substring(0, word.indexOfFirst { it.isLetterOrDigit() }.coerceAtLeast(0))

            val mapped = APP_NICKNAMES[clean]
                ?: HINGLISH_TERM_MAPPINGS[clean]
                ?: VERB_SLANG_MAPPINGS[clean]
                ?: clean

            if (prefix.isNotEmpty() || suffix.isNotEmpty()) {
                "$prefix$mapped$suffix"
            } else {
                mapped
            }
        }
        return normalizedWords.joinToString(" ")
    }
}
