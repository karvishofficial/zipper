package com.zipper.app.core.agent

import android.util.Log
import com.zipper.app.core.model.DistilledNode
import com.zipper.app.core.model.DistilledUiState
import java.util.Locale
import kotlin.math.abs

/**
 * Universal Comparative Exploration Engine.
 *
 * Implements bounded multi-turn scroll exploration and deterministic unit price / value
 * comparison across dynamic e-commerce catalogs (Zepto, Blinkit, Amazon, etc.).
 *
 * Zero heuristic spoonfeeding: uses pure math (Price / NormalizedQuantity) and
 * universal UI affordance grounding.
 */
class ProductExplorationEngine(val intent: ComparativeIntent) {

    companion object {
        private const val TAG = "ProductExplorationEngine"

        private val CHEAPEST_KEYWORDS = listOf(
            "cheapest", "lowest price", "least expensive", "cheaper", "lowest rate", "best price"
        )

        private val BEST_OFFER_KEYWORDS = listOf(
            "best offer", "best deal", "top deal", "max discount", "maximum discount",
            "highest discount", "best discount", "biggest discount"
        )

        private val KNOWN_APPS = listOf(
            "zepto", "blinkit", "amazon", "flipkart", "instamart", "zomato", "swiggy", "bigbasket", "dunzo"
        )

        /**
         * Detects comparative shopping intent from a user goal string.
         * Returns null for non-comparative tasks (YouTube, Settings, WhatsApp, Alarm, etc.).
         */
        fun detectComparativeIntent(goal: String): ComparativeIntent? {
            val lower = goal.lowercase(Locale.ROOT).trim()

            val isCheapest = CHEAPEST_KEYWORDS.any { lower.contains(it) }
            val isBestOffer = BEST_OFFER_KEYWORDS.any { lower.contains(it) }

            if (!isCheapest && !isBestOffer) {
                return null
            }

            val mode = if (isCheapest) ComparativeMode.CHEAPEST else ComparativeMode.BEST_OFFER

            // Extract App if mentioned
            val app = KNOWN_APPS.firstOrNull { lower.contains(it) }

            // Extract requested quantity (e.g. "500g", "1kg", "500 ml", "pack of 6")
            val parsedQuantity = ProductCardParser.extractQuantity(lower)

            // Extract Product Entity
            var cleaned = lower
            val allKeywords = CHEAPEST_KEYWORDS + BEST_OFFER_KEYWORDS + listOf(
                "find", "search", "get", "buy", "order", "add", "the", "to", "cart",
                "for", "of", "please", "me", "a", "an", "some", "fast", "quick",
                "lowest", "price", "deal", "offer", "best", "cheap", "item", "product"
            )

            for (kw in allKeywords) {
                cleaned = cleaned.replace(Regex("""\b${Regex.escape(kw)}\b"""), " ")
            }

            if (app != null) {
                cleaned = cleaned.replace(Regex("""\b(?:on|in|from|inside)?\s*${Regex.escape(app)}\b"""), " ")
            }

            if (parsedQuantity != null) {
                cleaned = cleaned.replace(Regex("""\b${Regex.escape(parsedQuantity.rawText.lowercase(Locale.ROOT))}\b"""), " ")
            }

            cleaned = cleaned.replace(Regex("""[^a-z0-9\s]"""), " ").trim()
            val tokens = cleaned.split(Regex("""\s+""")).filter { it.isNotBlank() }

            val rawEntity = tokens.firstOrNull() ?: ""
            val entityStem = ProductCardParser.stemWord(rawEntity)

            return ComparativeIntent(
                mode = mode,
                targetEntity = entityStem,
                targetApp = app,
                requestedQuantity = parsedQuantity?.normalizedQuantity,
                requestedQuantityText = parsedQuantity?.rawText,
                requestedUnitType = parsedQuantity?.unitType
            )
        }
    }

    private val accumulatedCards = mutableMapOf<String, ParsedProductCard>()
    private var scrollDownCount: Int = 0
    private val maxScrollDownBudget: Int = 2 // Explores Screen 1, Screen 2, Screen 3 (up to 3 screens)
    private var scrollUpCount: Int = 0
    private var currentScrollDepth: Int = 0
    private var selectedWinner: ParsedProductCard? = null
    var isWinnerActionDispatched: Boolean = false
        private set

    val totalAccumulatedCount: Int
        get() = accumulatedCards.size

    val winner: ParsedProductCard?
        get() = selectedWinner

    /**
     * Determines the next action in the exploration cycle based on the current screen state.
     *
     * Cycle:
     * 1. Screen 1: Extract visible cards -> Scroll down to explore catalog depth.
     * 2. Screen 2: Extract cards -> Scroll down if budget permits.
     * 3. Screen 3: Extract cards -> Run deterministic comparative math -> Select winner.
     * 4. Ground winner in viewport (or scroll back up if winner was on an earlier screen) -> Tap "ADD".
     */
    fun evaluateNextAction(state: DistilledUiState): AgentAction? {
        val visibleCards = ProductCardParser.parseCardsFromState(state, intent.targetEntity, currentScrollDepth)

        // Modal dismissal gate: if a promotional modal or bottom sheet covers the catalog, dismiss it
        val closeModalNode = state.nodes.firstOrNull { n ->
            val res = n.resourceId?.lowercase(Locale.ROOT).orEmpty()
            val desc = n.contentDescription?.lowercase(Locale.ROOT).orEmpty()
            val txt = n.text?.lowercase(Locale.ROOT).orEmpty()
            val isClose = res.contains("close") || res.contains("dismiss") || res.contains("cross") ||
                desc.contains("close") || desc.contains("dismiss") || txt == "close" || txt == "x" || txt == "✕"
            isClose && (n.isClickable || n.role in setOf("BUTTON", "IMAGE", "VIEW"))
        }
        if (closeModalNode != null && visibleCards.isEmpty()) {
            Log.i(TAG, "Promotional modal / bottom sheet detected without visible product cards. Tapping close button #${closeModalNode.index}.")
            return AgentAction.Tap(
                targetNodeIndex = closeModalNode.index,
                candidateId = "tap_dismiss_promo_modal_${closeModalNode.index}",
                semanticLabel = "Close Offer Modal"
            )
        }

        // Gate: If no product cards matching target entity are visible on screen, engine remains IDLE.
        // This ensures the agent is on the actual search results screen before exploration takes over.
        if (visibleCards.isEmpty() && accumulatedCards.isEmpty()) {
            Log.d(TAG, "No product cards for entity '${intent.targetEntity}' on current screen. Engine remains IDLE.")
            return null
        }

        for (card in visibleCards) {
            val key = card.title.lowercase(Locale.ROOT).trim()
            if (!accumulatedCards.containsKey(key)) {
                accumulatedCards[key] = card
                Log.d(TAG, "Accumulated card at depth $currentScrollDepth: '${card.title}' | ₹${card.price} for ${card.quantityText} (unit: ₹${"%.2f".format(card.unitPrice)})")
            }
        }

        Log.i(TAG, "Current screen visible cards: ${visibleCards.size}, Total buffer: ${accumulatedCards.size}, Scroll depth: $currentScrollDepth (down count: $scrollDownCount, up count: $scrollUpCount)")

        // Phase 1: Exploration - Scroll Down up to maxScrollDownBudget
        if (scrollDownCount < maxScrollDownBudget) {
            scrollDownCount++
            currentScrollDepth++
            Log.i(TAG, "Exploring catalog depth: Discovered ${accumulatedCards.size} products so far. Scrolling DOWN (step $scrollDownCount of $maxScrollDownBudget)...")
            return AgentAction.Scroll(
                direction = ScrollDirection.DOWN,
                candidateId = "scroll_explore_down"
            )
        }

        // Phase 2: Select the Winner using Deterministic Math
        if (selectedWinner == null) {
            selectedWinner = selectWinner()
            if (selectedWinner != null) {
                val w = selectedWinner!!
                Log.i(TAG, """
                ======================================================
                🏆 COMPARATIVE WINNER SELECTED:
                   Mode: ${intent.mode}
                   Requested: ${intent.requestedQuantityText ?: "Any quantity"}
                   Winner Title: '${w.title}'
                   Price: ₹${w.price} (${w.quantityText})
                   Unit Price: ₹${"%.2f".format(w.unitPrice)}
                   Discovered at Depth: ${w.discoveredAtDepth}
                   Total Compared: ${accumulatedCards.size}
                ======================================================
                """.trimIndent())
            }
        }

        val winningCard = selectedWinner ?: return null

        // Phase 3: Ground Winner in Active Viewport
        val liveMatchingCard = visibleCards.firstOrNull { card ->
            card.title.equals(winningCard.title, ignoreCase = true) ||
            (abs(card.price - winningCard.price) < 0.01 && abs(card.normalizedQuantity - winningCard.normalizedQuantity) < 0.01)
        }

        val targetTapNode = liveMatchingCard?.actionNodeIndex ?: run {
            if (liveMatchingCard != null) {
                state.nodes.firstOrNull { n ->
                    val inCol = n.bounds.centerX in (liveMatchingCard.bounds.left - 40)..(liveMatchingCard.bounds.right + 40)
                    val inRow = n.bounds.centerY in liveMatchingCard.bounds.top..(liveMatchingCard.bounds.bottom + 200)
                    inCol && inRow && (n.isClickable || n.role in setOf("BUTTON", "VIEW"))
                }?.index
            } else null
        }

        if (liveMatchingCard != null && targetTapNode != null) {
            val tapLabel = liveMatchingCard.actionNodeLabel ?: "Add to Cart"
            Log.i(TAG, "Winner '${winningCard.title}' is visible on screen! Action node #$targetTapNode ('$tapLabel'). Dispatching ADD.")
            isWinnerActionDispatched = true
            return AgentAction.Tap(
                targetNodeIndex = targetTapNode,
                candidateId = "tap_add_winner_$targetTapNode",
                semanticLabel = tapLabel
            )
        }

        // If winner was discovered on an earlier screen and is not visible, scroll back UP to re-ground
        if (scrollUpCount <= scrollDownCount) {
            scrollUpCount++
            if (currentScrollDepth > 0) currentScrollDepth--
            Log.i(TAG, "Winner '${winningCard.title}' was at depth ${winningCard.discoveredAtDepth}. Currently at depth $currentScrollDepth. Scrolling UP to re-ground target (up count: $scrollUpCount)...")
            return AgentAction.Scroll(
                direction = ScrollDirection.UP,
                candidateId = "scroll_back_to_winner"
            )
        }

        // Fallback: If scroll back happened and still can't find exact node, pick best matching visible card on screen
        val fallbackCard = visibleCards.filter {
            ProductCardParser.matchesRequestedQuantity(it, intent.requestedQuantity, intent.requestedUnitType)
        }.minByOrNull { it.unitPrice } ?: visibleCards.minByOrNull { it.unitPrice }

        if (fallbackCard?.actionNodeIndex != null) {
            val tapLabel = fallbackCard.actionNodeLabel ?: "Add to Cart"
            Log.w(TAG, "Winner not directly visible after scroll back; tapping best visible card #${fallbackCard.actionNodeIndex} ('${fallbackCard.title}')")
            isWinnerActionDispatched = true
            return AgentAction.Tap(
                targetNodeIndex = fallbackCard.actionNodeIndex,
                candidateId = "tap_add_fallback_${fallbackCard.actionNodeIndex}",
                semanticLabel = tapLabel
            )
        }

        return null
    }

    private fun selectWinner(): ParsedProductCard? {
        if (accumulatedCards.isEmpty()) return null

        val cards = accumulatedCards.values.toList()

        // 1. If user requested a specific quantity (e.g. "500g"), strictly filter for matching cards first!
        if (intent.requestedQuantity != null && intent.requestedUnitType != null) {
            val matchingCards = cards.filter {
                ProductCardParser.matchesRequestedQuantity(it, intent.requestedQuantity, intent.requestedUnitType)
            }
            if (matchingCards.isNotEmpty()) {
                Log.i(TAG, "Filtering by requested quantity '${intent.requestedQuantityText}': found ${matchingCards.size} matching items out of ${cards.size}")
                return when (intent.mode) {
                    ComparativeMode.CHEAPEST -> matchingCards.minByOrNull { it.price }
                    ComparativeMode.BEST_OFFER -> matchingCards.maxByOrNull { it.valueScore }
                }
            } else {
                Log.w(TAG, "No cards strictly matched requested quantity '${intent.requestedQuantityText}'; falling back to general unit price comparison.")
            }
        }

        // 2. General comparison (no quantity specified, or fallback):
        return when (intent.mode) {
            ComparativeMode.CHEAPEST -> {
                val weightCards = cards.filter { it.unitType == UnitType.WEIGHT }
                if (weightCards.isNotEmpty()) {
                    weightCards.minByOrNull { it.unitPrice }
                } else {
                    cards.minByOrNull { it.unitPrice }
                }
            }
            ComparativeMode.BEST_OFFER -> {
                cards.maxByOrNull { it.valueScore }
            }
        }
    }

    private fun findSortAffordance(state: DistilledUiState): DistilledNode? {
        return state.nodes.firstOrNull { n ->
            val label = (n.text ?: n.contentDescription ?: "").lowercase(Locale.ROOT)
            val res = n.resourceId?.lowercase(Locale.ROOT).orEmpty()
            (label.contains("sort") || label.contains("filter") || res.contains("sort") || res.contains("filter")) &&
                (n.isClickable || n.role in setOf("BUTTON", "TAB", "VIEW"))
        }
    }
}
