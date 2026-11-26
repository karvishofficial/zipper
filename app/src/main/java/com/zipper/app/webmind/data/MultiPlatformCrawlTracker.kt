package com.zipper.app.webmind.data

import java.util.Locale

enum class CrawlPlatformStatus {
    CRAWLED,
    SEARCHING,
    INDEXED_OFFLINE,
    UNAVAILABLE
}

data class PlatformCrawlBadge(
    val storeName: String,
    val status: CrawlPlatformStatus,
    val productTitle: String,
    val price: Double,
    val priceDisplay: String,
    val deliveryTime: String,
    val badge: String,
    val targetUrl: String,
    val isLowestPrice: Boolean = false
)

data class MultiPlatformCrawlReport(
    val query: String,
    val entityName: String,
    val platformsCrawled: List<PlatformCrawlBadge>,
    val lowestPrice: Double,
    val highestPrice: Double,
    val savingsAmount: Double,
    val savingsPercentage: Double,
    val verdictText: String,
    val mathProof: String
)

object MultiPlatformCrawlTracker {

    fun buildReport(
        query: String,
        entityName: String,
        badges: List<PlatformCrawlBadge>
    ): MultiPlatformCrawlReport {
        val validPriced = badges.filter { it.price > 0.0 }
        val lowest = validPriced.minByOrNull { it.price }
        val highest = validPriced.maxByOrNull { it.price }

        val lowestPrice = lowest?.price ?: 0.0
        val highestPrice = highest?.price ?: 0.0
        val diff = if (highestPrice > lowestPrice) highestPrice - lowestPrice else 0.0
        val savingsPct = if (highestPrice > 0.0 && diff > 0.0) (diff / highestPrice) * 100.0 else 0.0

        val lowestStore = lowest?.storeName ?: "Top Store"
        val highestStore = highest?.storeName ?: "Competitor"

        val mathProof = if (diff > 0.0 && validPriced.size >= 2) {
            String.format(
                Locale.US,
                "₹%.0f (%s) vs ₹%.0f (%s) → Saves ₹%.0f (%.1f%% cheaper)",
                lowestPrice, lowestStore, highestPrice, highestStore, diff, savingsPct
            )
        } else {
            "Verified across ${badges.size} platforms."
        }

        val verdict = if (diff > 0.0 && lowest != null) {
            String.format(
                Locale.US,
                "🏆 %s offers the best price at %s, saving ₹%.0f (%.1f%%) compared to %s.",
                lowestStore, lowest.priceDisplay, diff, savingsPct, highestStore
            )
        } else if (lowest != null) {
            "Verified deal on ${lowest.storeName} at ${lowest.priceDisplay}."
        } else {
            "Multi-platform live search completed across ${badges.size} platforms."
        }

        val updatedBadges = badges.map { b ->
            b.copy(isLowestPrice = (b == lowest && diff > 0.0))
        }

        return MultiPlatformCrawlReport(
            query = query,
            entityName = entityName,
            platformsCrawled = updatedBadges,
            lowestPrice = lowestPrice,
            highestPrice = highestPrice,
            savingsAmount = diff,
            savingsPercentage = savingsPct,
            verdictText = verdict,
            mathProof = mathProof
        )
    }

    /**
     * Converts offline comparison items into live crawl badges with direct URLs.
     */
    fun fromOfflinePayload(query: String, payload: OfflineComparisonPayload): MultiPlatformCrawlReport {
        val badges = payload.items.map { item ->
            PlatformCrawlBadge(
                storeName = item.store,
                status = CrawlPlatformStatus.INDEXED_OFFLINE,
                productTitle = item.title,
                price = item.price,
                priceDisplay = item.priceDisplay,
                deliveryTime = item.deliverySpeed,
                badge = item.badge,
                targetUrl = item.url.ifBlank {
                    when (item.store.lowercase(Locale.ROOT)) {
                        "zepto" -> "https://www.zeptonow.com/search?q=" + java.net.URLEncoder.encode(item.title, "UTF-8")
                        "blinkit" -> "https://blinkit.com/s/?q=" + java.net.URLEncoder.encode(item.title, "UTF-8")
                        "amazon" -> "https://www.amazon.in/s?k=" + java.net.URLEncoder.encode(item.title, "UTF-8")
                        "flipkart" -> "https://www.flipkart.com/search?q=" + java.net.URLEncoder.encode(item.title, "UTF-8")
                        else -> "https://www.google.com/search?q=" + java.net.URLEncoder.encode("${item.store} ${item.title}", "UTF-8")
                    }
                }
            )
        }
        return buildReport(query, payload.title, badges)
    }

    /**
     * Converts a ComprehensiveComparisonPayload into a MultiPlatformCrawlReport.
     */
    fun fromComprehensivePayload(payload: ComprehensiveComparisonPayload): MultiPlatformCrawlReport {
        val badges = payload.products.map { p ->
            PlatformCrawlBadge(
                storeName = p.store,
                status = if (payload.isOfflineKnowledge) CrawlPlatformStatus.INDEXED_OFFLINE else CrawlPlatformStatus.CRAWLED,
                productTitle = p.title,
                price = p.price,
                priceDisplay = if (p.unitPriceDisplay.isNotBlank()) "${p.priceDisplay} (${p.unitPriceDisplay})" else p.priceDisplay,
                deliveryTime = p.deliverySpeed,
                badge = p.badge,
                targetUrl = p.productUrl.ifBlank {
                    when (p.store.lowercase(Locale.ROOT)) {
                        "zepto" -> "https://www.zeptonow.com/search?q=" + java.net.URLEncoder.encode(p.title, "UTF-8")
                        "blinkit" -> "https://blinkit.com/s/?q=" + java.net.URLEncoder.encode(p.title, "UTF-8")
                        "amazon" -> "https://www.amazon.in/s?k=" + java.net.URLEncoder.encode(p.title, "UTF-8")
                        "flipkart" -> "https://www.flipkart.com/search?q=" + java.net.URLEncoder.encode(p.title, "UTF-8")
                        else -> "https://www.google.com/search?q=" + java.net.URLEncoder.encode("${p.store} ${p.title}", "UTF-8")
                    }
                },
                isLowestPrice = p.isBestDeal
            )
        }
        val lowest = badges.firstOrNull { it.isLowestPrice } ?: badges.minByOrNull { it.price }
        val highest = badges.maxByOrNull { it.price }
        val lowestPrice = lowest?.price ?: 0.0
        val highestPrice = highest?.price ?: 0.0
        val diff = if (highestPrice > lowestPrice) highestPrice - lowestPrice else 0.0
        val savingsPct = if (highestPrice > 0.0 && diff > 0.0) (diff / highestPrice) * 100.0 else 0.0

        return MultiPlatformCrawlReport(
            query = payload.query,
            entityName = payload.title,
            platformsCrawled = badges,
            lowestPrice = lowestPrice,
            highestPrice = highestPrice,
            savingsAmount = diff,
            savingsPercentage = savingsPct,
            verdictText = payload.definitiveVerdict,
            mathProof = payload.savingsProofText
        )
    }
}
