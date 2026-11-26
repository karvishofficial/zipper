package com.zipper.app.core.voice

/**
 * EnglishPhonemizer
 *
 * Lightweight Grapheme-to-Phoneme (G2P) engine for Kokoro / Paradee-8M.
 * Replicates the exact text normalization (num2words) and General American G2P
 * (misaki british=False / af_heart voice) from the author's Hugging Face Space:
 * https://huggingface.co/spaces/sahilmahendrakar/paradee
 */
object EnglishPhonemizer {

    // Native Kokoro / Misaki phoneme dictionary (matching Kokoro af_heart / Paradee-8M)
    private val DICT = mapOf(
        "hey" to "hˈA",
        "hi" to "hˈI",
        "hello" to "həlˈO",
        "karthik" to "kˈɑɹθɪk",
        "kartik" to "kˈɑɹθɪk",
        "it" to "ˈɪt",
        "its" to "ˈɪts",
        "it's" to "ˈɪts",
        "is" to "ˈɪz",
        "time" to "tˈIm",
        "to" to "tu",
        "turn" to "tˈɜːɹn",
        "off" to "ˈɔf",
        "on" to "ˈɔn",
        "switch" to "swˈɪtʃ",
        "the" to "ði",
        "water" to "wˈɔɾəɹ",
        "motor" to "mˈOɾəɹ",
        "pump" to "pˈʌmp",
        "geyser" to "ɡˈIzəɹ",
        "stove" to "stˈOv",
        "gas" to "ɡˈæs",
        "before" to "bəfˈɔɹ",
        "overflows" to "ˌOvəɹflˈOz",
        "overflow" to "ˌOvəɹflˈO",
        "tank" to "tˈæŋk",
        "full" to "fˈʊl",
        "alert" to "əlˈɜːɹt",
        "reminder" to "ɹəmˈIndəɹ",
        "scheduled" to "skˈɛdʒuːld",
        "urgent" to "ˈɜːɹdʒənt",
        "please" to "plˈiz",
        "check" to "tʃˈɛk",
        "done" to "dˈʌn",
        "taken" to "tˈAkən",
        "medicine" to "mˈɛdəsən",
        "tablet" to "tˈæblət",
        "pills" to "pˈɪlz",
        "bill" to "bˈɪl",
        "pay" to "pˈA",
        "payment" to "pˈAmənt",
        "paid" to "pˈAd",
        "due" to "dˈu",
        "today" to "tədˈA",
        "tonight" to "tənˈIt",
        "tomorrow" to "təmˈɑɹO",
        "electricity" to "əlˌɛktɹˈɪsəɾi",
        "bescom" to "bˈɛskɑm",
        "wifi" to "wˈIfI",
        "recharge" to "ɹitʃˈɑɹdʒ",
        "credit" to "kɹˈɛdət",
        "card" to "kˈɑɹd",
        "rupees" to "ɹupˈiz",
        "rupee" to "ɹupˈi",
        "minutes" to "mˈɪnəts",
        "minute" to "mˈɪnət",
        "hours" to "ˈWəɹz",
        "hour" to "ˈWəɹ",
        "seconds" to "sˈɛkəndz",
        "have" to "hæv",
        "has" to "hæz",
        "had" to "hæd",
        "you" to "ju",
        "your" to "jʊɹ",
        "this" to "ðɪs",
        "that" to "ðæt",
        "for" to "fɔɹ",
        "from" to "fɹʌm",
        "in" to "ɪn",
        "at" to "æt",
        "be" to "bi",
        "been" to "bɪn",
        "running" to "ɹˈʌnɪŋ",
        "over" to "ˈOvəɹ",
        "now" to "nˈW",
        "later" to "lˈAɾəɹ",
        "snooze" to "snˈuz",
        "snoozed" to "snˈuzd",
        "call" to "kˈɔl",
        "calling" to "kˈɔlɪŋ",
        "all" to "ˈɔl",
        "back" to "bˈæk",
        "great" to "ɡɹˈAt",
        "good" to "ɡˈʊd",
        "day" to "dˈA",
        "night" to "nˈIt",
        "morning" to "mˈɔɹnɪŋ",
        "meeting" to "mˈiɾɪŋ",
        "standup" to "stˈændʌp",
        "zoom" to "zˈum",
        "gym" to "dʒˈɪm",
        "workout" to "wˈɜːɹkˌWt",
        "awesome" to "ˈɔsəm",
        "completed" to "kəmplˈiɾɪd",
        "marked" to "mˈɑɹkt",
        "starting" to "stˈɑɹɾɪŋ",
        "agent" to "ˈAdʒənt",
        "task" to "tˈæsk",
        "zipper" to "zˈɪpəɹ",
        "voice" to "vˈYs",
        "assistant" to "əsˈɪstənt",
        "test" to "tˈɛst",
        "succeeded" to "səksˈidəd",
        "perfectly" to "pˈɜːɹfəktli",
        "zero" to "zˈɪɹO",
        "one" to "wˈʌn",
        "two" to "tˈu",
        "three" to "θɹˈi",
        "four" to "fˈɔɹ",
        "five" to "fˈIv",
        "six" to "sˈɪks",
        "seven" to "sˈɛvən",
        "eight" to "ˈAt",
        "nine" to "nˈIn",
        "ten" to "tˈɛn",
        "eleven" to "əlˈɛvən",
        "twelve" to "twˈɛlv",
        "thirteen" to "θˌɜːɹtˈin",
        "fourteen" to "fˌɔɹtˈin",
        "fifteen" to "fˌɪftˈin",
        "sixteen" to "sˌɪkstˈin",
        "seventeen" to "sˌɛvəntˈin",
        "eighteen" to "ˌAtˈin",
        "nineteen" to "nˌIntˈin",
        "twenty" to "twˈɛnti",
        "thirty" to "θˈɜːɹɾi",
        "forty" to "fˈɔɹɾi",
        "fifty" to "fˈɪfti",
        "sixty" to "sˈɪksti",
        "seventy" to "sˈɛvənti",
        "eighty" to "ˈAɾi",
        "ninety" to "nˈIndi",
        "hundred" to "hˈʌndɹəd",
        "thousand" to "θˈWzənd"
    )

    /**
     * Expands all digits into plain English words (e.g. 25 -> twenty-five)
     * just like num2words does in the Hugging Face Space.
     */
    fun expandNumbers(text: String): String {
        return text.replace(Regex("""\b\d+\b""")) { match ->
            val num = match.value.toLongOrNull() ?: return@replace match.value
            numberToWords(num)
        }
    }

    private fun numberToWords(n: Long): String {
        if (n == 0L) return "zero"
        if (n < 0) return "minus " + numberToWords(-n)

        val ones = arrayOf(
            "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
            "seventeen", "eighteen", "nineteen"
        )
        val tens = arrayOf(
            "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"
        )

        return when {
            n < 20 -> ones[n.toInt()]
            n < 100 -> {
                val rem = (n % 10).toInt()
                if (rem == 0) tens[(n / 10).toInt()] else "${tens[(n / 10).toInt()]} ${ones[rem]}"
            }
            n < 1000 -> {
                val hundredPart = "${ones[(n / 100).toInt()]} hundred"
                val rem = n % 100
                if (rem == 0L) hundredPart else "$hundredPart and ${numberToWords(rem)}"
            }
            n < 1000000 -> {
                val thousandPart = "${numberToWords(n / 1000)} thousand"
                val rem = n % 1000
                if (rem == 0L) thousandPart else "$thousandPart ${numberToWords(rem)}"
            }
            else -> n.toString()
        }
    }

    /**
     * Converts an English sentence into Kokoro IPA phonemes.
     */
    fun phonemize(text: String): String {
        // Step 1: Expand digits to spoken English words (25 -> twenty five)
        val expanded = expandNumbers(text.trim())
        if (expanded.isEmpty()) return ""

        val words = expanded.split(Regex("""\s+"""))
        val result = StringBuilder()

        for (i in words.indices) {
            val rawWord = words[i]
            val phoneme = wordToPhoneme(rawWord)
            result.append(phoneme)
            if (i < words.size - 1) {
                result.append(" ")
            }
        }
        return result.toString()
    }

    private fun wordToPhoneme(word: String): String {
        // Handle hyphenated words like twenty-five
        if (word.contains("-")) {
            return word.split("-").joinToString(" ") { wordToPhoneme(it) }
        }

        // Strip punctuation
        var leadingPunct = ""
        var trailingPunct = ""
        var core = word.lowercase()

        while (core.isNotEmpty() && !core.first().isLetterOrDigit()) {
            leadingPunct += core.first()
            core = core.drop(1)
        }
        while (core.isNotEmpty() && !core.last().isLetterOrDigit()) {
            trailingPunct = core.last() + trailingPunct
            core = core.dropLast(1)
        }

        if (core.isEmpty()) return word

        // 1. Dictionary lookup (American General IPA)
        val dictMatch = DICT[core]
        if (dictMatch != null) {
            return leadingPunct + dictMatch + trailingPunct
        }

        // 2. Rule-based letter-to-sound (LTS) fallback
        val lts = applyPhoneticRules(core)
        return leadingPunct + lts + trailingPunct
    }

    private fun applyPhoneticRules(w: String): String {
        return w
            // Digraphs
            .replace("sh", "ʃ")
            .replace("ch", "tʃ")
            .replace("th", "θ")
            .replace("ph", "f")
            .replace("wh", "w")
            .replace("ng", "ŋ")
            .replace("ck", "k")
            .replace("qu", "kw")
            // Kokoro native diphthongs & vowel pairs
            .replace("ai", "A")
            .replace("ay", "A")
            .replace("ee", "i")
            .replace("ea", "i")
            .replace("oo", "u")
            .replace("oa", "O")
            .replace("ow", "W")
            .replace("ou", "W")
            .replace("oi", "Y")
            .replace("oy", "Y")
            .replace("igh", "I")
            .replace("er", "ɜːɹ")
            .replace("ir", "ɜːɹ")
            .replace("ur", "ɜːɹ")
            .replace("ar", "ɑɹ")
            .replace("or", "ɔɹ")
            // Basic vowels
            .replace("a", "æ")
            .replace("e", "ɛ")
            .replace("i", "ɪ")
            .replace("o", "ɑ")
            .replace("u", "ʌ")
            // Consonants
            .replace("c", "k")
            .replace("j", "dʒ")
            .replace("r", "ɹ")
    }
}
