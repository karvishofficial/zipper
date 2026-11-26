package com.zipper.app.webmind.data

import java.util.Locale

enum class QueryDomain {
    QUICK_GROCERY,
    ELECTRONICS_TECH,
    GENERAL_RESEARCH
}

data class ClassifiedIntent(
    val domain: QueryDomain,
    val isComparison: Boolean,
    val cleanEntity: String,
    val rawQuery: String
)

/**
 * Intelligent Semantic Domain & Intent Classifier for Karvish Webmind.
 * Normalizes user queries into actionable domain entities with zero cloud tokens.
 */
object WebDomainClassifier {

    private val GROCERY_WORDS = setOf(
        "tomato", "tomatoes", "potato", "potatoes", "onion", "onions", "milk", "curd",
        "paneer", "egg", "eggs", "bread", "butter", "cheese", "vegetable", "vegetables",
        "fruit", "fruits", "apple", "apples", "banana", "bananas", "atta", "rice",
        "dal", "oil", "sugar", "salt", "coriander", "ginger", "garlic", "chilli",
        "lemon", "carrot", "cucumber", "capsicum", "grocery", "groceries", "biscuit",
        "tea", "coffee", "maggi", "noodles"
    )

    private val TECH_WORDS = setOf(
        "mobile", "mobiles", "phone", "phones", "smartphone", "smartphones",
        "iphone", "samsung", "oneplus", "realme", "redmi", "xiaomi", "pixel", "iqoo",
        "vivo", "oppo", "motorola", "moto", "laptop", "laptops", "macbook", "ipad",
        "tablet", "earphone", "earphones", "headphone", "headphones", "tws", "earbuds",
        "smartwatch", "watch", "gpu", "rtx", "camera", "monitor", "charger"
    )

    private val COMPARISON_PATTERNS = listOf(
        "compare", " vs ", "versus", "which is cheaper", "which is better",
        "which is best", "cheaper in", "cheapest in", "difference between",
        "better than", "across", "price difference"
    )

    fun classify(query: String): ClassifiedIntent {
        val trimmed = query.trim()
        val lower = trimmed.lowercase(Locale.ROOT)

        val isComparison = COMPARISON_PATTERNS.any { lower.contains(it) } ||
            (lower.contains("price") && (lower.contains("and") || lower.contains("or") || lower.contains("both")))

        val cleanEntity = extractEntity(trimmed)

        val words = cleanEntity.lowercase(Locale.ROOT).split(Regex("""[^a-z0-9]+""")).filter { it.isNotBlank() }

        val domain = when {
            words.any { it in GROCERY_WORDS } || GROCERY_WORDS.any { lower.contains(it) } -> QueryDomain.QUICK_GROCERY
            words.any { it in TECH_WORDS } || TECH_WORDS.any { lower.contains(it) } -> QueryDomain.ELECTRONICS_TECH
            else -> QueryDomain.GENERAL_RESEARCH
        }

        return ClassifiedIntent(
            domain = domain,
            isComparison = isComparison,
            cleanEntity = cleanEntity,
            rawQuery = trimmed
        )
    }

    /**
     * Determines whether a query is an interactive automation task (form filling, auth, cart)
     * versus an informational or research question.
     */
    fun isAutonomousActionGoal(query: String): Boolean {
        val lower = query.lowercase(Locale.ROOT).trim()

        val informationalPrefixes = listOf(
            "what is", "who is", "tell me", "explain", "about", "how does", "how to",
            "meaning of", "definition of", "history of", "news on", "latest news",
            "why is", "where is", "when did", "summary of", "pictures of", "photos of"
        )
        if (informationalPrefixes.any { lower.startsWith(it) }) return false
        if (lower.contains("what is") || lower.contains("who is") || lower.contains("tell me about")) return false

        val explicitActionPhrases = listOf(
            "fill ", "fill out", "fill the form", "login", "log in", "sign in", "add to cart",
            "checkout", "type into", "enter my", "submit form", "book ticket",
            "book flight", "register on", "sign up", "download pdf", "download file"
        )
        val hasExplicitAction = explicitActionPhrases.any { lower.contains(it) }

        val hasMultiStepChain = (lower.contains(" and ") || lower.contains(" then ")) &&
            (lower.contains("click") || lower.contains("type") || lower.contains("select") || lower.contains("cart"))

        return hasExplicitAction || hasMultiStepChain
    }

    private fun extractEntity(query: String): String {
        return query
            .replace(Regex("""(?i)\b(?:in|on|from|at|via)\s+(?:amazon|flipkart|zepto|blinkit|instamart|bigbasket|croma)\b"""), " ")
            .replace(Regex("""(?i)\b(?:amazon|flipkart|zepto|blinkit|instamart|bigbasket|croma)\b"""), " ")
            .replace(Regex("""(?i)\b(?:find|search|the|cheapest|lowest price|best offer|best deal|best price|which is cheaper|which is better|which is best|price of|price in|rate of|price|rate|across|both|and|vs|versus|compare|stores?|online|buy|order|get)\b"""), " ")
            .replace(Regex("""\s+"""), " ")
            .trim()
            .ifBlank { query.trim() }
    }
}
