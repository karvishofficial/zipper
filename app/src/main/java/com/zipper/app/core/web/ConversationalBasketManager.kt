package com.zipper.app.core.web

import java.util.Locale

enum class BasketMutationType {
    NEW_BASKET,       // User specified a new list from scratch
    ITEMS_ADDED,      // Appended new items to active basket (e.g. "and add 1 pack bread")
    ITEM_MODIFIED,    // Adjusted quantity/unit of existing item (e.g. "make paneer 1kg")
    ITEM_REMOVED,     // Removed an item from active basket (e.g. "remove potato")
    STORE_SWITCHED,   // Switched target fulfillment store (e.g. "order on Blinkit instead")
    BASKET_CLEARED,   // Emptied basket
    NON_BASKET_QUERY  // Query is not related to shopping basket
}

data class ActiveBasketState(
    val items: List<ParsedShoppingItem> = emptyList(),
    val targetStore: String? = null,
    val lastUpdatedMs: Long = System.currentTimeMillis()
)

data class BasketMutationResult(
    val type: BasketMutationType,
    val explanation: String,
    val currentItems: List<ParsedShoppingItem>,
    val targetStore: String?
)

/**
 * Evergreen Conversational Basket Manager.
 *
 * Maintains multi-turn shopping basket session memory and intelligently resolves
 * natural language updates across conversation turns:
 * - Adding items: "and add 1 packet bread", "also add 1L milk"
 * - Modifying quantity: "make paneer 1kg", "change tomato to 500g"
 * - Removing items: "remove potato", "don't want aloo", "delete paneer"
 * - Switching stores: "order on Blinkit instead", "switch to Zepto"
 * - Clearing basket: "clear basket", "cancel order"
 */
object ConversationalBasketManager {

    private var state = ActiveBasketState()

    fun getActiveBasket(): ActiveBasketState = state

    fun clearBasket() {
        state = ActiveBasketState()
    }

    fun setExplicitBasket(items: List<ParsedShoppingItem>, store: String? = null): BasketMutationResult {
        state = ActiveBasketState(items = items, targetStore = store)
        return BasketMutationResult(
            type = BasketMutationType.NEW_BASKET,
            explanation = "Initialized basket with ${items.size} items",
            currentItems = state.items,
            targetStore = state.targetStore
        )
    }

    fun processUserInput(rawInput: String): BasketMutationResult {
        val input = rawInput.trim()
        val lower = input.lowercase(Locale.ROOT)

        // 1. Check for Reset / Clear Intent
        val clearRegex = Regex("""(?i)\b(?:clear\s+basket|clear\s+cart|empty\s+cart|empty\s+basket|cancel\s+order|reset\s+list|start\s+over)\b""")
        if (clearRegex.containsMatchIn(lower)) {
            clearBasket()
            return BasketMutationResult(
                type = BasketMutationType.BASKET_CLEARED,
                explanation = "Basket has been cleared.",
                currentItems = emptyList(),
                targetStore = null
            )
        }

        // 2. Check for Store Switching (if basket has active items)
        if (state.items.isNotEmpty()) {
            val storeSwitchRegex = Regex("""(?i)\b(?:order\s+on|switch\s+to|do\s+(?:it|this)\s+on|change\s+to|use)\s+(zepto|blinkit|amazon|flipkart)\b(?:\s+instead)?""")
            val switchMatch = storeSwitchRegex.find(lower)
            if (switchMatch != null) {
                val rawStore = switchMatch.groupValues[1].lowercase(Locale.ROOT)
                val resolvedStore = when {
                    rawStore.contains("zepto") -> "Zepto"
                    rawStore.contains("blinkit") -> "Blinkit"
                    rawStore.contains("amazon") -> "Amazon"
                    rawStore.contains("flipkart") -> "Flipkart"
                    else -> "Zepto"
                }
                state = state.copy(targetStore = resolvedStore, lastUpdatedMs = System.currentTimeMillis())
                return BasketMutationResult(
                    type = BasketMutationType.STORE_SWITCHED,
                    explanation = "Switched fulfillment store to $resolvedStore for all ${state.items.size} items.",
                    currentItems = state.items,
                    targetStore = state.targetStore
                )
            }
        }

        // 3. Check for Item Removal Intent
        if (state.items.isNotEmpty()) {
            val removeRegex = Regex("""(?i)\b(?:remove|delete|drop|don't\s+want|dont\s+want|cancel|hatao|hata\s+do)\s+([a-z0-9\s]+?)(?:$|[,.]|\s+and\s+)""")
            val removeMatch = removeRegex.find(lower)
            if (removeMatch != null) {
                val targetName = removeMatch.groupValues[1].trim()
                val remaining = state.items.filterNot { item ->
                    val norm = item.normalizedQuery.lowercase(Locale.ROOT)
                    val canon = item.canonicalName.lowercase(Locale.ROOT)
                    val orig = item.originalText.lowercase(Locale.ROOT)
                    norm.contains(targetName) || canon.contains(targetName) || orig.contains(targetName) ||
                        targetName.contains(canon)
                }

                if (remaining.size < state.items.size) {
                    val removedCount = state.items.size - remaining.size
                    state = state.copy(items = remaining, lastUpdatedMs = System.currentTimeMillis())
                    return BasketMutationResult(
                        type = BasketMutationType.ITEM_REMOVED,
                        explanation = "Removed '$targetName' from basket ($removedCount item removed, ${remaining.size} remaining).",
                        currentItems = state.items,
                        targetStore = state.targetStore
                    )
                }
            }
        }

        // 4. Check for Item Quantity Modification
        if (state.items.isNotEmpty()) {
            val modRegex = Regex("""(?i)\b(?:make|change|update|increase|decrease)\s+([a-z\s]+?)\s+(?:to\s+)?(\d+\s*(?:kg|kilo|g|gm|l|litre|liter|ml|pack|packets?))\b""")
            val modMatch = modRegex.find(lower)
            if (modMatch != null) {
                val itemTarget = modMatch.groupValues[1].trim()
                val newRawQty = modMatch.groupValues[2].trim()

                // Standardize quantity
                val standardizedQty = newRawQty.lowercase(Locale.ROOT)
                    .replace("kilo", "kg")
                    .replace("packets", "pack")
                    .replace("packet", "pack")
                    .replace("liter", "L")
                    .replace("litre", "L")
                    .replace("gm", "g")
                    .replace(" ", "")

                val updatedList = state.items.map { item ->
                    val norm = item.normalizedQuery.lowercase(Locale.ROOT)
                    val canon = item.canonicalName.lowercase(Locale.ROOT)
                    if (norm.contains(itemTarget) || canon.contains(itemTarget) || itemTarget.contains(canon)) {
                        val cleanName = item.canonicalName
                        val newQuery = "$standardizedQty $cleanName"
                        item.copy(
                            quantity = standardizedQty,
                            normalizedQuery = newQuery
                        )
                    } else {
                        item
                    }
                }

                state = state.copy(items = updatedList, lastUpdatedMs = System.currentTimeMillis())
                return BasketMutationResult(
                    type = BasketMutationType.ITEM_MODIFIED,
                    explanation = "Updated $itemTarget quantity to $standardizedQty.",
                    currentItems = state.items,
                    targetStore = state.targetStore
                )
            }
        }

        // 5. Check for Item Addition (Append to active basket)
        val isAdditionIntent = lower.startsWith("and add") || lower.startsWith("also add") ||
            lower.startsWith("add ") || lower.startsWith("+ ") || lower.contains("aur le aana") ||
            lower.contains("also include")

        if (state.items.isNotEmpty() && isAdditionIntent) {
            val cleanedAddition = input
                .replace(Regex("""(?i)^(?:and\s+add|also\s+add|add|\+|also\s+include)\s+"""), "")
                .trim()

            val newParsedItems = ShoppingListParser.parseShoppingList(cleanedAddition)
            if (newParsedItems.isNotEmpty()) {
                val combined = state.items.toMutableList()
                for (newItem in newParsedItems) {
                    val alreadyPresent = combined.indexOfFirst {
                        it.canonicalName.equals(newItem.canonicalName, ignoreCase = true)
                    }
                    if (alreadyPresent >= 0) {
                        combined[alreadyPresent] = newItem // update with newer quantity
                    } else {
                        combined.add(newItem)
                    }
                }
                state = state.copy(items = combined, lastUpdatedMs = System.currentTimeMillis())
                return BasketMutationResult(
                    type = BasketMutationType.ITEMS_ADDED,
                    explanation = "Added ${newParsedItems.size} item(s) to basket: ${newParsedItems.joinToString(", ") { it.normalizedQuery }}",
                    currentItems = state.items,
                    targetStore = state.targetStore
                )
            }
        }

        // 6. Check for a Brand New Shopping List
        val parsed = ShoppingListParser.parseShoppingList(input)
        val hasGrocery = parsed.any { it.category in setOf(ItemCategory.FRESH, ItemCategory.BULK_PANTRY) }
        val hasMultiple = parsed.size > 1
        val hasActionIntent = lower.contains("buy") || lower.contains("order") ||
            lower.contains("cart") || lower.contains("get")

        if (hasMultiple || (hasGrocery && hasActionIntent)) {
            // Extract store if explicitly stated
            val explicitStore = when {
                lower.contains("zepto") -> "Zepto"
                lower.contains("blinkit") -> "Blinkit"
                lower.contains("amazon") -> "Amazon"
                lower.contains("flipkart") -> "Flipkart"
                else -> null
            }
            state = ActiveBasketState(items = parsed, targetStore = explicitStore)
            return BasketMutationResult(
                type = BasketMutationType.NEW_BASKET,
                explanation = "Detected ${parsed.size} items in new shopping basket.",
                currentItems = state.items,
                targetStore = state.targetStore
            )
        }

        return BasketMutationResult(
            type = BasketMutationType.NON_BASKET_QUERY,
            explanation = "Query is not a shopping basket modification.",
            currentItems = state.items,
            targetStore = state.targetStore
        )
    }
}
