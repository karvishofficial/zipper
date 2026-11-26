package com.zipper.app.core.web

import java.util.Locale

enum class ItemCategory {
    FRESH,        // Milk, bread, curd, eggs, veggies, fruits (best for 10-min Zepto/Blinkit)
    BULK_PANTRY,  // Atta, rice, oil, ghee, dal, sugar, detergents (best for Amazon/Flipkart bulk discounts)
    GENERAL       // Household, snacks, personal care, electronics
}

data class ParsedShoppingItem(
    val originalText: String,
    val normalizedQuery: String,
    val quantity: String? = null,
    val canonicalName: String,
    val category: ItemCategory,
    val hardAttributes: List<String> = emptyList(),
    val brand: String? = null
)

/**
 * Intelligent Multilingual Shopping List & Colloquial Intent Parser.
 *
 * Normalizes vernacular Hindi/Hinglish grocery lists (e.g. from WhatsApp, SMS, or Voice),
 * converts colloquial quantities ("aadha kilo" -> "500g", "pav kilo" -> "250g"),
 * maps regional ingredient names ("tamatar" -> "tomato", "pyaz" -> "onion"),
 * and strips conversational noise ("aate waqt ... le aana").
 */
object ShoppingListParser {

    private val BOILERPLATE_PREFIXES = listOf(
        Regex("""(?i)^[\s\S]*?\b(?:aate\s+waqt|aate\s+time|ghar\s+aate\s+huye|ghar\s+aate\s+time)\s+"""),
        Regex("""(?i)^[\s\S]*?\b(?:saman\s+le\s+aana|saman\s+lana|le\s+aana|le\s+aao|laana|lao)\s+"""),
        Regex("""(?i)^[\s\S]*?\b(?:please\s+bring|bring\s+following|shopping\s+list[:\s]*|grocery\s+list[:\s]*|items?\s+to\s+buy[:\s]*|buy[:\s]*|order[:\s]*)\s*"""),
        Regex("""(?i)^[\s\S]*?\b(?:ghar\s+ke\s+liye|kitchen\s+list[:\s]*)\s*""")
    )

    private val BOILERPLATE_SUFFIXES = listOf(
        Regex("""(?i)\s+(?:bhi\s+)?(?:le\s+aana|le\s+aao|laana|lao|de\s+dena|kharid\s+lena|kharidna)[\s.!,]*$"""),
        Regex("""(?i)\s+(?:please|asap|today|jaldi)[\s.!,]*$""")
    )

    // Colloquial Indian quantity mapping to standardized metrics
    private val QUANTITY_NORMALIZATIONS = listOf(
        Regex("""(?i)\b(?:aadha|adha|aadhe)\s+(?:kilo|kg)\b""") to "500g",
        Regex("""(?i)\b(?:half)\s*(?:a\s+)?(?:kg|kilo)\b""") to "500g",
        Regex("""(?i)\b(?:pav|pao|paav)\s+(?:kilo|kg)\b""") to "250g",
        Regex("""(?i)\b(?:quarter)\s*(?:a\s+)?(?:kg|kilo)\b""") to "250g",
        Regex("""(?i)\b(?:dhai\s+sau|dhai\s+so)\s*(?:gram|gm|g)?\b""") to "250g",
        Regex("""(?i)\b(?:ek|1)\s+(?:kilo)\b""") to "1kg",
        Regex("""(?i)\b(?:do|2)\s+(?:kilo)\b""") to "2kg",
        Regex("""(?i)\b(?:teen|3)\s+(?:kilo)\b""") to "3kg",
        Regex("""(?i)\b(?:paanch|5)\s+(?:kilo)\b""") to "5kg",
        Regex("""(?i)\b(?:aadha|adha)\s+(?:litre|liter|ltr|l)\b""") to "500ml",
        Regex("""(?i)\b(?:half)\s*(?:a\s+)?(?:litre|liter|ltr|l)\b""") to "500ml",
        Regex("""(?i)\b(?:ek|1)\s+(?:litre|liter|ltr)\b""") to "1L",
        Regex("""(?i)\b(?:do|2)\s+(?:litre|liter|ltr)\b""") to "2L",
        Regex("""(?i)\b(?:ek|1)\s+(?:packet|pkt|pack)\b""") to "1 pack",
        Regex("""(?i)\b(?:do|2)\s+(?:packet|pkt|packs?)\b""") to "2 pack",
        Regex("""(?i)\b(\d+(?:\.\d+)?)\s*(?:g|gms|gm|grams?)\b""") to "$1g",
        Regex("""(?i)\b(\d+(?:\.\d+)?)\s*(?:kg|kgs|kilos?)\b""") to "$1kg",
        Regex("""(?i)\b(\d+(?:\.\d+)?)\s*(?:l|litres?|liters?|ltr)\b""") to "$1L",
        Regex("""(?i)\b(\d+(?:\.\d+)?)\s*(?:ml|millilitres?|milliliters?)\b""") to "$1ml"
    )

    // Vernacular Hindi/Hinglish grocery item dictionary
    private val VERNACULAR_DICTIONARY = mapOf(
        "aloo" to "potato",
        "aalu" to "potato",
        "tamatar" to "tomato",
        "pyaz" to "onion",
        "pyaaz" to "onion",
        "kanda" to "onion",
        "doodh" to "milk",
        "dahi" to "curd",
        "paneer" to "paneer",
        "makhan" to "butter",
        "maska" to "butter",
        "adrak" to "ginger",
        "lahsun" to "garlic",
        "lehsun" to "garlic",
        "mirchi" to "green chilli",
        "hari mirch" to "green chilli",
        "dhaniya" to "coriander",
        "kothmir" to "coriander",
        "nimbu" to "lemon",
        "pudina" to "mint",
        "palak" to "spinach",
        "bhindi" to "lady finger",
        "gajar" to "carrot",
        "shimla mirch" to "capsicum",
        "matar" to "green peas",
        "chawal" to "rice",
        "cheeni" to "sugar",
        "shakkar" to "sugar",
        "namak" to "salt",
        "tel" to "cooking oil",
        "sarson ka tel" to "mustard oil",
        "besan" to "besan",
        "sooji" to "sooji",
        "suji" to "sooji",
        "atta" to "atta",
        "aata" to "atta",
        "ghee" to "desi ghee",
        "kaju" to "cashew",
        "badaam" to "almond",
        "badam" to "almond",
        "anda" to "eggs",
        "ande" to "eggs"
    )

    // Categories mapping for arbitrage basket splitting
    private val FRESH_KEYWORDS = setOf(
        "tomato", "potato", "onion", "milk", "curd", "paneer", "butter", "bread",
        "egg", "eggs", "vegetable", "vegetables", "fruit", "fruits", "banana", "apple",
        "ginger", "garlic", "green chilli", "chilli", "coriander", "lemon", "mint",
        "spinach", "carrot", "peas", "capsicum", "mushroom", "cucumber", "coriander leaves"
    )

    private val BULK_PANTRY_KEYWORDS = setOf(
        "atta", "aashirvaad", "rice", "basmati", "oil", "mustard oil", "sunflower oil",
        "fortune", "ghee", "dal", "toor dal", "moong dal", "chana dal", "pulses",
        "sugar", "salt", "besan", "sooji", "dry fruits", "cashew", "almond",
        "detergent", "surf excel", "ariel", "tide", "shampoo", "soap", "toothpaste",
        "colgate", "harpic", "lizol", "vim", "prill"
    )

    private val HARD_ATTRIBUTES = listOf(
        "organic", "cold-pressed", "cold pressed", "kachhi ghani", "sugar-free", "sugar free",
        "zero sugar", "diet", "lactose-free", "lactose free", "whole wheat", "brown",
        "multigrain", "gluten-free", "gluten free", "toned", "double toned", "full cream",
        "cow milk", "extra virgin", "virgin", "refined"
    )

    private val RECOGNIZED_BRANDS = listOf(
        "amul", "fortune", "aashirvaad", "tata", "nandini", "mother dairy", "britannia",
        "dettol", "surf excel", "vim", "colgate", "parle", "nestle", "cadbury", "lays",
        "kurkure", "saffola", "epigamia", "country delight", "akshayakalpa", "milky mist"
    )

    /**
     * Parses a raw text (e.g. from WhatsApp, SMS, note, or prompt) into structured shopping items.
     */
    fun parseShoppingList(rawText: String): List<ParsedShoppingItem> {
        if (rawText.isBlank()) return emptyList()

        var cleaned = rawText.trim()
        for (prefix in BOILERPLATE_PREFIXES) {
            cleaned = cleaned.replace(prefix, "")
        }
        for (suffix in BOILERPLATE_SUFFIXES) {
            cleaned = cleaned.replace(suffix, "")
        }

        // Split into candidate tokens by newlines, bullets, commas, or 'aur'/'and'/'+'
        val bulletRegex = Regex("""^(\d+[.)]\s*|[-*•]\s*)""")
        val rawTokens = mutableListOf<String>()

        val lines = cleaned.lines().map { it.replace(bulletRegex, "").trim() }.filter { it.isNotBlank() }
        if (lines.size > 1) {
            rawTokens.addAll(lines)
        } else {
            val parts = cleaned.split(Regex("""(?i),\s*(?:and\s+|aur\s+)?|\s+aur\s+|\s+and\s+|\s+\+\s+|\s+&\s+"""))
                .map { it.replace(bulletRegex, "").trim() }
                .filter { it.isNotBlank() }
            rawTokens.addAll(parts)
        }

        val parsedItems = mutableListOf<ParsedShoppingItem>()
        for (token in rawTokens) {
            val item = parseSingleItem(token)
            if (item != null) {
                parsedItems.add(item)
            }
        }
        return parsedItems
    }

    private fun parseSingleItem(rawToken: String): ParsedShoppingItem? {
        var text = rawToken.trim()
            .replace(Regex("""^(?i)(?:and|aur|&|\+)\s+"""), "")
            .replace(Regex("""(?i)\b(?:in|on|from|at|via)\s+(?:amazon|flipkart|zepto|blinkit)\b"""), "")
            .trim()

        if (text.length < 2) return null

        // 1. Normalize quantity in the string
        var extractedQty: String? = null
        for ((pattern, replacement) in QUANTITY_NORMALIZATIONS) {
            val match = pattern.find(text)
            if (match != null) {
                val resolved = if (replacement.contains("$1") && match.groupValues.size > 1) {
                    replacement.replace("$1", match.groupValues[1])
                } else {
                    replacement
                }
                extractedQty = resolved
                text = text.replaceRange(match.range, " $resolved ").replace(Regex("""\s+"""), " ").trim()
                break
            }
        }

        // 2. Identify and normalize vernacular Hindi names
        var canonical = text
        for ((hindiWord, englishWord) in VERNACULAR_DICTIONARY) {
            val wordPattern = Regex("""(?i)\b${Regex.escape(hindiWord)}\b""")
            if (wordPattern.containsMatchIn(text)) {
                canonical = text.replace(wordPattern, englishWord).replace(Regex("""\s+"""), " ").trim()
                break
            }
        }

        // 3. Category classification
        val lowerCanonical = canonical.lowercase(Locale.ROOT)
        val category = when {
            FRESH_KEYWORDS.any { lowerCanonical.contains(it) } -> ItemCategory.FRESH
            BULK_PANTRY_KEYWORDS.any { lowerCanonical.contains(it) } -> ItemCategory.BULK_PANTRY
            else -> ItemCategory.GENERAL
        }

        // 4. Extract hard attributes and brands
        val matchedAttributes = mutableListOf<String>()
        for (attr in HARD_ATTRIBUTES) {
            val attrPattern = Regex("""(?i)\b${Regex.escape(attr)}\b""")
            if (attrPattern.containsMatchIn(lowerCanonical)) {
                matchedAttributes.add(attr.replace(" ", "-"))
            }
        }

        var matchedBrand: String? = null
        for (b in RECOGNIZED_BRANDS) {
            val bPattern = Regex("""(?i)\b${Regex.escape(b)}\b""")
            if (bPattern.containsMatchIn(lowerCanonical)) {
                matchedBrand = b
                break
            }
        }

        return ParsedShoppingItem(
            originalText = rawToken.trim(),
            normalizedQuery = canonical,
            quantity = extractedQty,
            canonicalName = canonical.replace(Regex("""(?i)\b\d+\s*(?:g|kg|gm|l|ml|pack)\b"""), "").trim(),
            category = category,
            hardAttributes = matchedAttributes,
            brand = matchedBrand
        )
    }
}
