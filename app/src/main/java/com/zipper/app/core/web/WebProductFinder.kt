package com.zipper.app.core.web

import com.zipper.app.core.model.DistilledNode
import com.zipper.app.core.model.DistilledUiState
import java.net.URLEncoder
import java.util.Locale

data class WebProductItem(
    val title: String,
    val price: Double,
    val priceDisplay: String,
    val url: String,
    val imageUrl: String? = null,
    val rating: String? = null,
    val store: String = "Web",
    val isSponsored: Boolean = false,
    val snippet: String? = null
)

/**
 * Intelligent Web Product Discovery and Deal Extractor for Karvish.
 * Parses e-commerce product grids, normalizes prices, prunes sponsored distractors,
 * and extracts canonical product deep links.
 */
object WebProductFinder {

    private val SHOPPING_KEYWORDS = setOf(
        "cheapest", "lowest price", "price of", "best price", "best offer", "best deal", "buy", "deal", "discount",
        "amazon", "flipkart", "zepto", "blinkit", "myntra", "earphone", "earphones",
        "headphones", "mobile", "phone", "iphone", "samsung", "tomato", "potato", "onion", "grocery"
    )

    fun isShoppingQuery(query: String): Boolean {
        val lower = query.lowercase(Locale.ROOT)
        return SHOPPING_KEYWORDS.any { lower.contains(it) } ||
            (lower.contains("find ") && (lower.contains("under ") || lower.contains("below ") || lower.contains("₹") || lower.contains("rs"))) ||
            lower.contains("price")
    }

    fun isGroceryQuery(cleanQuery: String): Boolean {
        val lower = cleanQuery.lowercase(Locale.ROOT)
        val groceryWords = setOf(
            "tomato", "potato", "onion", "milk", "bread", "egg", "eggs", "paneer", "curd",
            "butter", "cheese", "vegetable", "vegetables", "fruit", "fruits", "apple", "banana",
            "atta", "rice", "dal", "oil", "sugar", "salt"
        )
        return groceryWords.any { lower.contains(it) }
    }

    fun cleanSearchQuery(query: String): String {
        return query
            .replace(Regex("(?i)\\b(?:in|on|from|at|via)\\s+(?:amazon|flipkart|zepto|zzepto|blinkit|blibkit|blikit|blanket|blenkit|myntra)\\b"), " ")
            .replace(Regex("(?i)\\b(?:amazon|flipkart|zepto|zzepto|blinkit|blibkit|blikit|blanket|blenkit|myntra)\\b"), " ")
            .replace(Regex("(?i)\\b(?:find|search|the|cheapest|lowest price|best offer|best deal|best price|which is cheaper|which is better|which is best|price of|price in|rate of|price|rate|across|both|and|vs|versus|compare|stores?|online|buy|order|get)\\b"), " ")
            .replace(Regex("\\s+"), " ")
            .trim()
            .ifBlank { query.trim() }
    }

    /**
     * Parses multiple shopping items from list queries (e.g. 500g tomato, 1 boat earphone, 1kg besan).
     */
    fun parseMultiProductItems(query: String): List<String> {
        val trimmed = query.trim()
        val clean = trimmed
            .replace(Regex("(?i)^\\s*(?:order|buy|get|search|find)\\s+(?:the\\s+)?(?:cheapest|best|lowest\\s+price|top\\s+rated|items?|deal|and|\\s+)+\\s*"), "")
            .replace(Regex("(?i)\\s+(?:on|from|in|at)\\s+(?:zepto|zzepto|blinkit|blibkit|blikit|blanket|blenkit|amazon|flipkart|myntra)\\s*$"), "")
            .replace(Regex("(?i)^\\s*(?:on|from|in|at)\\s+(?:zepto|zzepto|blinkit|blibkit|blikit|blanket|blenkit|amazon|flipkart|myntra)\\s+"), "")
            .trim()

        val bulletRegex = Regex("^(\\d+[.)]\\s*|[-*•]\\s*)")
        val lines = clean.lines()
            .map { it.replace(bulletRegex, "").trim() }
            .map { cleanItemCandidate(it) }
            .filter { it.isNotBlank() }

        if (lines.size > 1) {
            val candidateLines = lines.filter { it.length >= 2 }
            if (candidateLines.size > 1) return candidateLines
        }

        val candidates = clean.split(Regex("(?i),\\s*|\\s+and\\s+"))
            .map { it.replace(bulletRegex, "").trim() }
            .map { cleanItemCandidate(it) }
            .filter { it.length >= 2 }

        if (candidates.size > 1) {
            return candidates
        }
        return emptyList()
    }

    fun cleanItemCandidate(item: String): String {
        return item
            .replace(Regex("(?i)\\b(?:in|on|from|at|via)\\s+(?:amazon|flipkart|zepto|zzepto|blinkit|blibkit|blikit|blanket|blenkit|myntra)\\b"), "")
            .replace(Regex("(?i)\\b(?:amazon|flipkart|zepto|zzepto|blinkit|blibkit|blikit|blanket|blenkit|myntra)\\b"), "")
            .replace(Regex("\\s+"), " ")
            .trim()
    }

    /**
     * Resolves an optimized e-commerce search URL.
     */
    fun buildSearchUrl(query: String, explicitStore: String? = null): Pair<String, String> {
        val lower = query.lowercase(Locale.ROOT)
        val cleanTerms = cleanSearchQuery(query)
        val encoded = URLEncoder.encode(cleanTerms, "UTF-8")
        val wantsCheapest = lower.contains("cheapest") || lower.contains("lowest price")

        val targetStore = explicitStore ?: when {
            lower.contains("amazon") -> "Amazon"
            lower.contains("flipkart") -> "Flipkart"
            lower.contains("zepto") -> "Zepto"
            lower.contains("blinkit") || lower.contains("blanket") || lower.contains("blkbkit") -> "Blinkit"
            isGroceryQuery(cleanTerms) -> "Zepto"
            else -> "Amazon"
        }

        return when (targetStore) {
            "Amazon" -> {
                val sortParam = if (wantsCheapest) "&s=price-asc-rank" else ""
                "https://www.amazon.in/s?k=$encoded$sortParam" to "Amazon"
            }
            "Flipkart" -> {
                val sortParam = if (wantsCheapest) "&sort=price_asc" else ""
                "https://www.flipkart.com/search?q=$encoded$sortParam" to "Flipkart"
            }
            "Zepto" -> {
                "https://www.zeptonow.com/search?q=$encoded" to "Zepto"
            }
            "Blinkit" -> {
                "https://blinkit.com/s/?q=$encoded" to "Blinkit"
            }
            else -> {
                val sortParam = if (wantsCheapest) "&s=price-asc-rank" else ""
                "https://www.amazon.in/s?k=$encoded$sortParam" to "Amazon"
            }
        }
    }

    /**
     * Extracts structured products from distilled DOM nodes.
     */
    fun extractProductsFromState(state: DistilledUiState, store: String = "Web"): List<WebProductItem> {
        val products = mutableListOf<WebProductItem>()
        val priceRegex = Regex("""(?:₹|Rs\.?|\$)\s*([\d,]+(?:\.\d+)?)""")

        // Group related candidate nodes by vertical & horizontal proximity
        val candidateList = state.nodes
        for (i in candidateList.indices) {
            val node = candidateList[i]
            val label = listOfNotNull(node.text, node.contentDescription).joinToString(" ")
            val priceMatch = priceRegex.find(label)

            if (priceMatch != null) {
                val priceValStr = priceMatch.groupValues[1].replace(",", "")
                val priceVal = priceValStr.toDoubleOrNull() ?: continue
                if (priceVal <= 0.0 || priceVal > 1000000.0) continue

                // Find associated title, link, and image near this price node, sorted by vertical proximity
                val nearbyNodes = candidateList.filter {
                    val vDist = Math.abs(it.bounds.top - node.bounds.top)
                    val hDist = Math.abs(it.bounds.left - node.bounds.left)
                    vDist < 350 && hDist < 500 && it.index != node.index
                }.sortedBy { Math.abs(it.bounds.top - node.bounds.top) }

                val ratingFilterRegex = Regex("""(?i)\b([1-5]\.[0-9]|\bstars?\b|out of 5|\bratings?\b|\breviews?\b)""")
                val titleNode = nearbyNodes.firstOrNull {
                    val t = it.text?.trim().orEmpty()
                    t.length >= 8 &&
                        !priceRegex.containsMatchIn(t) &&
                        !ratingFilterRegex.containsMatchIn(t) &&
                        !t.contains("Sponsored", true)
                } ?: nearbyNodes.firstOrNull {
                    val t = it.text?.trim().orEmpty()
                    t.length >= 8 && !priceRegex.containsMatchIn(t) && !t.contains("Sponsored", true)
                } ?: node

                val linkNode = (listOf(node) + nearbyNodes).firstOrNull {
                    val h = it.href.orEmpty()
                    h.isNotBlank() && !h.startsWith("javascript:", true) &&
                        (h.contains("/dp/") || h.contains("/p/") || h.contains("/prn/") || h.contains("/pn/") || h.length > 30)
                } ?: (listOf(node) + nearbyNodes).firstOrNull { !it.href.isNullOrBlank() }

                val imgNode = (listOf(node) + nearbyNodes).firstOrNull { !it.imageUrl.isNullOrBlank() }

                val canonicalUrl = linkNode?.href ?: state.activityName ?: "about:blank"
                val cleanTitle = listOfNotNull(titleNode.text, titleNode.contentDescription)
                    .firstOrNull { it.length >= 6 }
                    ?.replace(priceRegex, "")
                    ?.replace("Sponsored", "", ignoreCase = true)
                    ?.trim() ?: "Product Match"

                val isSponsored = label.contains("Sponsored", true) ||
                    nearbyNodes.any { (it.text ?: "").contains("Sponsored", true) }

                products.add(
                    WebProductItem(
                        title = cleanTitle,
                        price = priceVal,
                        priceDisplay = "₹${"%,.0f".format(priceVal)}",
                        url = canonicalUrl,
                        imageUrl = imgNode?.imageUrl,
                        rating = extractRating(nearbyNodes),
                        store = store,
                        isSponsored = isSponsored
                    )
                )
            }
        }

        return products.distinctBy { it.title }
    }

    private fun extractRating(nodes: List<DistilledNode>): String? {
        val ratingRegex = Regex("""\b([1-5]\.[0-9])\s*(?:★|stars?|out of 5)?\b""")
        for (n in nodes) {
            val txt = "${n.text.orEmpty()} ${n.contentDescription.orEmpty()}"
            val match = ratingRegex.find(txt)
            if (match != null) {
                return "⭐ ${match.groupValues[1]}"
            }
        }
        return null
    }

    /**
     * Finds the winning best/cheapest matching product.
     * Enforces strict keyword relevance and rejects accessories/unrelated items.
     */
    fun findBestMatch(
        products: List<WebProductItem>,
        targetKeywords: List<String>,
        wantsBestOffer: Boolean = false
    ): WebProductItem? {
        if (products.isEmpty()) return null

        val essentialKws = targetKeywords.filter {
            it.length >= 3 && it !in setOf("fresh", "organic", "item", "product", "online", "buy", "good")
        }

        // 1. Mandatory keyword validation: item MUST contain at least one essential search keyword
        val matching = if (essentialKws.isNotEmpty()) {
            products.filter { prod ->
                val titleLower = prod.title.lowercase()
                essentialKws.any { kw -> titleLower.contains(kw) }
            }
        } else {
            products
        }

        // If no products match the query (e.g. Amazon returned hookah plate for tomato), reject!
        if (matching.isEmpty()) return null

        // 2. Reject accessories if searching for core electronics/items
        val accessoryWords = setOf("case", "cover", "pouch", "seeds", "sachet", "protector", "tips", "skin")
        val withoutAccessories = matching.filter { prod ->
            val titleLower = prod.title.lowercase()
            accessoryWords.none { titleLower.contains(it) }
        }.ifEmpty { matching }

        // 3. Filter out sponsored unless all are sponsored
        val organic = withoutAccessories.filter { !it.isSponsored }.ifEmpty { withoutAccessories }

        // 4. Return based on goal type:
        return if (wantsBestOffer) {
            organic.maxWithOrNull(
                compareBy<WebProductItem> { it.rating != null }
                    .thenByDescending { -it.price }
            ) ?: organic.firstOrNull()
        } else {
            organic.minByOrNull { it.price } ?: organic.firstOrNull()
        }
    }
}
