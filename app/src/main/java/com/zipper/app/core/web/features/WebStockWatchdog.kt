package com.zipper.app.core.web.features

import java.util.Locale

data class WatchdogTarget(
    val id: String,
    val title: String,
    val url: String,
    val targetPrice: Double? = null,
    val alertOnStock: Boolean = true,
    val lastCheckedPrice: Double? = null,
    val inStock: Boolean = true
)

data class WatchdogCheckResult(
    val target: WatchdogTarget,
    val currentPrice: Double?,
    val isInStock: Boolean,
    val triggeredAlert: Boolean,
    val alertMessage: String
)

/**
 * Background Drop Watchdog.
 * Tracks product pages for flash discounts, lightning deals, and back-in-stock alerts.
 */
object WebStockWatchdog {

    private val OUT_OF_STOCK_PHRASES = setOf(
        "out of stock", "currently unavailable", "sold out",
        "item unavailable", "temporarily out of stock",
        "notify me", "coming soon", "back in stock soon"
    )

    fun detectStockStatus(pageHtmlOrText: String): Boolean {
        val lower = pageHtmlOrText.lowercase(Locale.ROOT)
        // If an explicit out of stock phrase is prominent, item is not in stock
        val isOutOfStock = OUT_OF_STOCK_PHRASES.any { lower.contains(it) }
        return !isOutOfStock
    }

    fun extractCurrentPrice(pageHtmlOrText: String): Double? {
        val regex = Regex("""(?:₹|Rs\.?|INR|\$)\s*([0-9,]+(?:\.[0-9]+)?)""")
        val matches = regex.findAll(pageHtmlOrText)
        for (m in matches) {
            val num = m.groupValues[1].replace(",", "").toDoubleOrNull()
            if (num != null && num > 0.0) {
                return num
            }
        }
        return null
    }

    fun evaluate(
        target: WatchdogTarget,
        currentPrice: Double?,
        isInStock: Boolean
    ): WatchdogCheckResult {
        var alert = false
        val reasons = mutableListOf<String>()

        // 1. Check price drop
        if (target.targetPrice != null && currentPrice != null) {
            if (currentPrice <= target.targetPrice) {
                alert = true
                reasons.add("Price dropped to ₹$currentPrice (Target: ₹${target.targetPrice})")
            }
        }

        // 2. Check back in stock
        if (target.alertOnStock && !target.inStock && isInStock) {
            alert = true
            reasons.add("Item is now BACK IN STOCK!")
        }

        val message = if (alert) {
            "🚨 Watchdog Alert for '${target.title}': " + reasons.joinToString("; ")
        } else {
            "No alert triggered for '${target.title}' (Price: ₹${currentPrice ?: 0.0}, InStock: $isInStock)"
        }

        return WatchdogCheckResult(
            target = target,
            currentPrice = currentPrice,
            isInStock = isInStock,
            triggeredAlert = alert,
            alertMessage = message
        )
    }
}
