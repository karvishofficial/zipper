package com.zipper.app.core.agent

import android.graphics.Rect
import com.zipper.app.core.model.DistilledNode
import com.zipper.app.core.model.DistilledUiState
import com.zipper.app.core.model.NormalizedBounds
import java.util.Locale
import kotlin.math.abs

enum class UnitType {
    WEIGHT,  // Normalized to kg (1 kg = 1.0, 500 g = 0.5)
    VOLUME,  // Normalized to L (1 L = 1.0, 500 ml = 0.5)
    COUNT,   // Normalized to count (pack of 6 = 6.0)
    UNKNOWN
}

enum class ComparativeMode {
    CHEAPEST,
    BEST_OFFER
}

data class ComparativeIntent(
    val mode: ComparativeMode,
    val targetEntity: String,
    val targetApp: String? = null,
    val requestedQuantity: Double? = null,
    val requestedQuantityText: String? = null,
    val requestedUnitType: UnitType? = null
)

data class ParsedQuantity(
    val normalizedQuantity: Double,
    val unitType: UnitType,
    val rawText: String
)

data class ParsedProductCard(
    val title: String,
    val quantityText: String?,
    val normalizedQuantity: Double,
    val unitType: UnitType,
    val price: Double,
    val originalPrice: Double? = null,
    val discountPercent: Double = 0.0,
    val rating: Double? = null,
    val actionNodeIndex: Int? = null,
    val actionNodeLabel: String? = null,
    val bounds: NormalizedBounds,
    val rawBounds: Rect? = null,
    val discoveredAtDepth: Int = 0
) {
    val unitPrice: Double
        get() = if (normalizedQuantity > 0.0) price / normalizedQuantity else price

    /**
     * Value score for "best offer" / "best deal":
     * Balances discount percentage, rating, and affordability.
     */
    val valueScore: Double
        get() {
            val discScore = discountPercent.coerceIn(0.0, 100.0) * 0.6
            val ratingScore = (rating ?: 3.5) * 10.0
            val pricePenalty = (unitPrice.coerceAtLeast(1.0) / 100.0) * 2.0
            return discScore + ratingScore - pricePenalty
        }
}

/**
 * Universal, app-agnostic e-commerce parser.
 * Extracts product cards, weights/quantities, prices (₹), and discounts
 * across native Android View, Jetpack Compose, and Flutter apps (Zepto, Blinkit, Amazon, etc.).
 */
object ProductCardParser {

    val WEIGHT_PATTERNS = listOf(
        Regex("""(?i)\b(\d+(?:\.\d+)?)\s*(?:kg|kgs|kilo|kilograms?)\b""") to (1.0 to UnitType.WEIGHT),
        Regex("""(?i)\b(\d+(?:\.\d+)?)\s*(?:g|gm|gms|grams?)\b""") to (0.001 to UnitType.WEIGHT),
        Regex("""(?i)\b(\d+(?:\.\d+)?)\s*(?:mg|milligrams?)\b""") to (0.000001 to UnitType.WEIGHT),
        Regex("""(?i)\b(\d+(?:\.\d+)?)\s*(?:l|ltr|litres?|liters?)\b""") to (1.0 to UnitType.VOLUME),
        Regex("""(?i)\b(\d+(?:\.\d+)?)\s*(?:ml|millilitres?|milliliters?)\b""") to (0.001 to UnitType.VOLUME),
        Regex("""(?i)\bpack\s*of\s*(\d+)\b""") to (1.0 to UnitType.COUNT),
        Regex("""(?i)\b(\d+)\s*(?:pcs|pc|pieces?|units?|packs?)\b""") to (1.0 to UnitType.COUNT),
        Regex("""(?i)\b(\d+)\s*(?:dozen)\b""") to (12.0 to UnitType.COUNT),
        Regex("""(?i)\b1\s*(?:bunch)\b""") to (1.0 to UnitType.COUNT)
    )

    fun extractQuantity(text: String): ParsedQuantity? {
        for ((pat, spec) in WEIGHT_PATTERNS) {
            val m = pat.find(text)
            if (m != null) {
                val rawVal = m.groupValues[1].toDoubleOrNull() ?: 1.0
                return ParsedQuantity(
                    normalizedQuantity = rawVal * spec.first,
                    unitType = spec.second,
                    rawText = m.value.trim()
                )
            }
        }
        return null
    }

    fun matchesRequestedQuantity(card: ParsedProductCard, reqQty: Double?, reqUnit: UnitType?): Boolean {
        if (reqQty == null || reqUnit == null) return true
        if (card.unitType != reqUnit) {
            return false
        }
        return abs(card.normalizedQuantity - reqQty) <= 0.05
    }

    private val PRICE_REGEX = Regex("""[₹Rs\.]\s*(\d+(?:,\d+)*(?:\.\d+)?)""")
    private val DISCOUNT_REGEX = Regex("""(?i)\b(\d+)%\s*(?:off|discount)\b""")
    private val RATING_REGEX = Regex("""\b([1-5]\.[0-9])\s*(?:★|\b)""")

    private val COMMON_DISTRACTORS = setOf(
        "ketchup", "sauce", "soup", "puree", "paste", "pickle", "chips",
        "flavour", "flavor", "flavored", "flavoured", "seed", "seeds",
        "plant", "powder", "chutney", "masala", "mix", "instant", "snack",
        "crisps", "wafer", "wafers"
    )

    /**
     * Parses all candidate product cards visible in the given screen state.
     * Uses spatial 2D proximity and hierarchical grouping to assemble product attributes.
     */
    fun parseCardsFromState(
        state: DistilledUiState,
        targetEntity: String,
        currentScrollDepth: Int = 0
    ): List<ParsedProductCard> {
        val cleanEntity = stemWord(targetEntity.trim().lowercase(Locale.ROOT))
        val nodes = state.nodes

        // 1. Identify price nodes as anchor points (strictly excluding promo/delivery banners and bottom bars)
        val priceNodes = nodes.filter { node ->
            val text = (node.text ?: node.contentDescription ?: "").lowercase(Locale.ROOT)
            val res = node.resourceId?.lowercase(Locale.ROOT).orEmpty()
            val isPromo = text.contains("shop for") || text.contains("unlock") || text.contains("delivery") ||
                text.contains("free delivery") || text.contains("offers for you") || text.contains("extra ₹") ||
                res.contains("cart") || res.contains("delivery") || res.contains("banner") || res.contains("promo")
            val isBottomNav = node.bounds.top > 880
            !isPromo && !isBottomNav &&
                (text.contains("₹") || text.contains("rs.") || text.contains("rs ")) &&
                PRICE_REGEX.containsMatchIn(text)
        }

        if (priceNodes.isEmpty()) {
            return emptyList()
        }

        val cards = mutableListOf<ParsedProductCard>()
        val usedPriceIndices = mutableSetOf<Int>()

        for (priceNode in priceNodes) {
            if (priceNode.index in usedPriceIndices) continue

            // Spatial cluster around the price node (same column: xDist <= 260, yDist <= 220)
            // Sorted by 2D proximity so elements closer to this price node are prioritized
            val cluster = nodes.filter { n ->
                val yDist = abs(n.bounds.centerY - priceNode.bounds.centerY)
                val xDist = abs(n.bounds.centerX - priceNode.bounds.centerX)
                yDist <= 220 && xDist <= 260
            }.sortedBy { n ->
                abs(n.bounds.centerY - priceNode.bounds.centerY) + abs(n.bounds.centerX - priceNode.bounds.centerX)
            }

            val card = assembleCardFromCluster(cluster, priceNode, cleanEntity, currentScrollDepth)
            if (card != null) {
                cards.add(card)
                usedPriceIndices.add(priceNode.index)
            }
        }

        // Deduplicate cards by title
        return cards.distinctBy { it.title.lowercase(Locale.ROOT).trim() }
    }

    private fun assembleCardFromCluster(
        cluster: List<DistilledNode>,
        primaryPriceNode: DistilledNode,
        targetEntityStem: String,
        currentScrollDepth: Int = 0
    ): ParsedProductCard? {
        val clusterTexts = cluster.mapNotNull { it.text ?: it.contentDescription }
        val combinedText = clusterTexts.joinToString(" ")

        // 1. Extract Price
        val priceMatches = PRICE_REGEX.findAll(combinedText).mapNotNull {
            it.groupValues[1].replace(",", "").toDoubleOrNull()
        }.toList()

        if (priceMatches.isEmpty()) return null
        val sellingPrice = priceMatches.minOrNull() ?: return null
        val originalPrice = if (priceMatches.size > 1) priceMatches.maxOrNull() else null

        // 2. Identify Product Title
        val titleNode = cluster.firstOrNull { n ->
            val txt = (n.text ?: n.contentDescription ?: "").trim()
            val lower = txt.lowercase(Locale.ROOT)
            val isSearchSuggestionHeader = lower.startsWith("show all results") ||
                lower.startsWith("search for") ||
                lower.startsWith("results for") ||
                lower.contains("autosuggest") ||
                lower.contains("history")
            val isActionLabel = lower == "add" || lower == "+" || lower == "add to cart" ||
                lower == "add item" || lower.startsWith("add to") || lower == "buy" ||
                lower == "order" || lower == "view" || lower == "cart"
            val isPromoOrBanner = lower.contains("unlock") || lower.contains("delivery") ||
                lower.contains("shop for") || lower.contains("offers for you") || lower.contains("extra ₹")
            if (txt.length < 3 || txt.contains("₹") || txt.contains("Rs") || txt.contains("% off", true) || isSearchSuggestionHeader || isActionLabel || isPromoOrBanner) {
                false
            } else {
                isEntityMatch(lower, targetEntityStem)
            }
        } ?: return null // Must match target entity!

        val title = (titleNode.text ?: titleNode.contentDescription ?: "").trim()

        // Distractor check: reject if contains distractor words and entity is not that distractor
        val titleWords = title.lowercase(Locale.ROOT).split(Regex("[^a-z0-9]+")).toSet()
        val hasDistractor = titleWords.any { it in COMMON_DISTRACTORS }
        if (hasDistractor && targetEntityStem !in COMMON_DISTRACTORS) {
            return null
        }
        if (targetEntityStem == "potato" && titleWords.contains("sweet")) {
            return null
        }

        // 3. Extract Quantity / Weight
        val parsedQty = extractQuantity(combinedText)
        val normQty = parsedQty?.normalizedQuantity ?: 1.0
        val uType = parsedQty?.unitType ?: UnitType.UNKNOWN
        val qtyStr = parsedQty?.rawText

        // 4. Extract Discount Percentage
        val discMatch = DISCOUNT_REGEX.find(combinedText)
        val discountPercent = discMatch?.groupValues?.get(1)?.toDoubleOrNull() ?: run {
            if (originalPrice != null && originalPrice > sellingPrice) {
                ((originalPrice - sellingPrice) / originalPrice) * 100.0
            } else 0.0
        }

        // 5. Extract Rating
        val rating = RATING_REGEX.find(combinedText)?.groupValues?.get(1)?.toDoubleOrNull()

        // 6. Identify Action Node ("ADD" / "+" / "Add to Cart")
        val cardLeft = titleNode.bounds.left - 60
        val cardRight = titleNode.bounds.right + 60
        val actionNode = cluster.firstOrNull { n ->
            val inCardColumn = n.bounds.centerX in cardLeft..cardRight
            val label = (n.text ?: n.contentDescription ?: n.resourceId ?: "").trim().lowercase(Locale.ROOT)
            val isAddLabel = label == "add" || label == "+" || label.contains("add to cart") ||
                label.contains("add item") || label.endsWith(" add") || label.startsWith("add ") ||
                label.contains("btn_add") || label.contains("add_button")
            val isActionable = n.isClickable || n.role in setOf("BUTTON", "VIEW", "TEXT", "IMAGE")
            inCardColumn && isAddLabel && isActionable
        } ?: cluster.firstOrNull { n ->
            val inCardColumn = n.bounds.centerX in cardLeft..cardRight
            val res = n.resourceId?.lowercase(Locale.ROOT).orEmpty()
            inCardColumn && (res.contains("add") || res.contains("cart")) && n.isClickable
        } ?: cluster.firstOrNull { n ->
            val label = (n.text ?: n.contentDescription ?: n.resourceId ?: "").trim().lowercase(Locale.ROOT)
            val isAddLabel = label == "add" || label == "+" || label.contains("add to cart") ||
                label.contains("add item") || label.endsWith(" add") || label.startsWith("add ") ||
                label.contains("btn_add") || label.contains("add_button")
            val isActionable = n.isClickable || n.role in setOf("BUTTON", "VIEW", "TEXT", "IMAGE")
            isAddLabel && isActionable
        } ?: cluster.firstOrNull { n ->
            val inCardColumn = n.bounds.centerX in cardLeft..cardRight
            val isBelowTitle = n.bounds.centerY > titleNode.bounds.centerY
            val label = (n.text ?: n.contentDescription ?: "").lowercase(Locale.ROOT)
            val isPromo = label.contains("delivery") || label.contains("shop for") || label.contains("unlock")
            inCardColumn && isBelowTitle && !isPromo && (n.isClickable || n.role in setOf("BUTTON", "VIEW"))
        } ?: titleNode.takeIf { it.isClickable }

        val hasRealAdd = actionNode != null && actionNode != titleNode &&
            (actionNode.text?.contains("add", true) == true ||
             actionNode.contentDescription?.contains("add", true) == true ||
             actionNode.resourceId?.contains("add", true) == true ||
             actionNode.text == "+" || actionNode.contentDescription == "+" ||
             actionNode.resourceId?.contains("cart", true) == true)

        if (parsedQty == null && !hasRealAdd) {
            return null
        }

        val cardBounds = titleNode.bounds
        val rawBounds = titleNode.rawBounds

        return ParsedProductCard(
            title = title,
            quantityText = qtyStr ?: "1 unit",
            normalizedQuantity = normQty,
            unitType = uType,
            price = sellingPrice,
            originalPrice = originalPrice,
            discountPercent = discountPercent,
            rating = rating,
            actionNodeIndex = actionNode?.index,
            actionNodeLabel = actionNode?.semanticLabel,
            bounds = cardBounds,
            rawBounds = rawBounds,
            discoveredAtDepth = currentScrollDepth
        )
    }

    private fun isEntityMatch(text: String, targetStem: String): Boolean {
        if (targetStem.isBlank()) return true
        val words = text.split(Regex("[^a-z0-9]+")).filter { it.isNotBlank() }
        return words.any { w ->
            val wStem = stemWord(w)
            wStem == targetStem || (wStem.length >= 4 && targetStem.length >= 4 && (wStem.contains(targetStem) || targetStem.contains(wStem)))
        }
    }

    fun stemWord(word: String): String {
        val w = word.lowercase(Locale.ROOT).trim()
        return when {
            w.endsWith("tomatoes") -> "tomato"
            w.endsWith("potatoes") -> "potato"
            w.endsWith("onions") -> "onion"
            w.endsWith("apples") -> "apple"
            w.endsWith("bananas") -> "banana"
            w.endsWith("oranges") -> "orange"
            w.endsWith("lemons") -> "lemon"
            w.endsWith("ies") -> w.dropLast(3) + "y"
            w.endsWith("shes") || w.endsWith("ches") || w.endsWith("xes") || w.endsWith("zes") || w.endsWith("sses") || w.endsWith("oes") -> w.dropLast(2)
            w.endsWith("s") && !w.endsWith("ss") && !w.endsWith("us") && w.length > 3 -> w.dropLast(1)
            else -> w
        }
    }
}
