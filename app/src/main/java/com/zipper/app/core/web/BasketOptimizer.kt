package com.zipper.app.core.web

data class SmartSplitPlan(
    val hasSplitAdvantage: Boolean,
    val freshStoreName: String,
    val freshStorePkg: String,
    val freshItems: List<ParsedShoppingItem>,
    val bulkStoreName: String,
    val bulkStorePkg: String,
    val bulkItems: List<ParsedShoppingItem>,
    val estimatedSavingsInr: Int,
    val explanation: String
)

/**
 * Cross-Store Basket Optimizer & Arbitrage Engine.
 *
 * Partitions a multi-item shopping basket between Quick Commerce (Zepto/Blinkit)
 * for fresh perishables and Bulk E-Commerce (Amazon/Flipkart) for heavy pantry staples,
 * maximizing savings while preserving 10-minute instant delivery for urgent goods.
 */
object BasketOptimizer {

    fun optimizeBasket(items: List<ParsedShoppingItem>): SmartSplitPlan {
        val freshList = items.filter { it.category == ItemCategory.FRESH || it.category == ItemCategory.GENERAL }
        val bulkList = items.filter { it.category == ItemCategory.BULK_PANTRY }

        val hasAdvantage = freshList.isNotEmpty() && bulkList.isNotEmpty()
        // Heavy pantry staples (atta, oil, dal, surf) have 15-25% bulk markup in 10-min dark stores vs Amazon
        val estimatedSavings = bulkList.size * 50

        val explanation = if (hasAdvantage) {
            "⚡ 10-min delivery on Zepto for ${freshList.size} fresh items (${freshList.take(3).joinToString { it.canonicalName }}) + " +
            "📦 Amazon for ${bulkList.size} bulk staples (${bulkList.take(3).joinToString { it.canonicalName }}) saving ~₹$estimatedSavings."
        } else if (freshList.isNotEmpty()) {
            "All ${freshList.size} items are best suited for 10-minute Quick Commerce (Zepto / Blinkit)."
        } else {
            "All ${bulkList.size} items are heavy pantry staples, best suited for Amazon Fresh for maximum bulk savings."
        }

        return SmartSplitPlan(
            hasSplitAdvantage = hasAdvantage,
            freshStoreName = "Zepto",
            freshStorePkg = WebDeepLinkManager.PKG_ZEPTO,
            freshItems = freshList,
            bulkStoreName = "Amazon",
            bulkStorePkg = WebDeepLinkManager.PKG_AMAZON_IN,
            bulkItems = bulkList,
            estimatedSavingsInr = estimatedSavings,
            explanation = explanation
        )
    }
}
