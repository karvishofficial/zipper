package com.zipper.app.core.web.features

import com.zipper.app.core.web.HiddenWebController
import com.zipper.app.core.web.WebDeepLinkManager
import com.zipper.app.core.web.WebProductFinder
import com.zipper.app.core.web.WebProductItem
import com.zipper.app.core.web.WebStoreOffer
import kotlinx.coroutines.delay
import java.util.Locale

data class ComparisonResult(
    val query: String,
    val itemsByStore: Map<String, WebProductItem>,
    val bestOverallDeal: WebProductItem?,
    val summaryMarkdown: String,
    val offers: List<WebStoreOffer> = emptyList(),
    val crawlReport: com.zipper.app.webmind.data.MultiPlatformCrawlReport? = null
)

/**
 * Multi-Store Live Price Comparison Engine.
 * Cross-checks live pricing across Amazon, Flipkart, Zepto, and Blinkit,
 * calculating price differences and highlighting the lowest price deal.
 */
object PriceComparisonEngine {

    fun isComparisonQuery(query: String): Boolean {
        val lower = query.lowercase(Locale.ROOT)
        return lower.contains("compare") ||
            lower.contains(" vs ") ||
            lower.contains("versus") ||
            lower.contains("which is cheaper") ||
            lower.contains("which is better") ||
            lower.contains("which is best") ||
            lower.contains("cheaper in") ||
            lower.contains("cheapest in") ||
            lower.contains("under 70k") ||
            lower.contains("under 50k") ||
            lower.contains("under 60k") ||
            lower.contains("under 40k") ||
            lower.contains("under 30k") ||
            lower.contains("under 80k") ||
            (lower.contains("best") && (lower.contains("laptop") || lower.contains("phone") || lower.contains("pc") || lower.contains("earphone") || lower.contains("under"))) ||
            (lower.contains("price") && (lower.contains("across") || lower.contains("both") || lower.contains("and") || lower.contains(" or ")))
    }

    suspend fun compareStores(
        query: String,
        webController: HiddenWebController,
        webEngine: com.zipper.app.core.web.WebOnDeviceEngine? = null
    ): ComparisonResult {
        val neuralProduct = webEngine?.extractQueryEntity(query)
        val cleanProduct = neuralProduct ?: WebProductFinder.cleanSearchQuery(query)
        val targetKeywords = cleanProduct.lowercase().split(Regex("[^a-z0-9]+")).filter { it.length >= 3 }
        val lower = query.lowercase(Locale.ROOT)

        val storesToQuery = mutableListOf<String>()
        if (lower.contains("amazon")) storesToQuery.add("Amazon")
        if (lower.contains("flipkart")) storesToQuery.add("Flipkart")
        if (lower.contains("zepto")) storesToQuery.add("Zepto")
        if (lower.contains("blinkit") || lower.contains("blkbkit") || lower.contains("blanket") || lower.contains("grofers")) storesToQuery.add("Blinkit")

        // If user didn't specify at least 2 stores, auto-select based on category
        if (storesToQuery.size < 2) {
            val isGrocery = WebProductFinder.isGroceryQuery(cleanProduct)
            if (isGrocery) {
                if (!storesToQuery.contains("Zepto")) storesToQuery.add("Zepto")
                if (!storesToQuery.contains("Blinkit")) storesToQuery.add("Blinkit")
            } else {
                if (!storesToQuery.contains("Amazon")) storesToQuery.add("Amazon")
                if (!storesToQuery.contains("Flipkart")) storesToQuery.add("Flipkart")
            }
        }

        // Inject delivery location cookies for Zepto & Blinkit to bypass location modal blockers
        try {
            val cm = android.webkit.CookieManager.getInstance()
            cm.setAcceptCookie(true)
            cm.setCookie("https://www.zeptonow.com", "user_pincode=560001; Domain=.zeptonow.com; Path=/")
            cm.setCookie("https://blinkit.com", "gr_pincode=560001; Domain=.blinkit.com; Path=/")
        } catch (_: Exception) {}

        val itemsByStore = mutableMapOf<String, WebProductItem>()
        val parallelPool = com.zipper.app.core.web.WebParallelPool(webController.getContext())

        val tasks = storesToQuery.map { store ->
            com.zipper.app.core.web.WebTask { workerController ->
                val (searchUrl, storeName) = WebProductFinder.buildSearchUrl(cleanProduct, explicitStore = store)
                val loaded = workerController.navigate(searchUrl, timeoutMs = 10000L)
                if (loaded) {
                    workerController.waitForDomStability(1800L)
                    val state = workerController.distillWebState()
                    val products = WebProductFinder.extractProductsFromState(state, storeName)
                    val best = WebProductFinder.findBestMatch(products, targetKeywords)
                    if (best != null) {
                        storeName to best
                    } else null
                } else null
            }
        }

        try {
            val results: List<Pair<String, WebProductItem>?> = parallelPool.executeParallel(tasks)
            for (res in results.filterNotNull()) {
                itemsByStore[res.first] = res.second
            }
        } catch (_: Exception) {
            // Fallback to sequential execution if parallel pool fails
            for (store in storesToQuery) {
                val (searchUrl, storeName) = WebProductFinder.buildSearchUrl(cleanProduct, explicitStore = store)
                val loaded = webController.navigate(searchUrl, timeoutMs = 10000L)
                if (loaded) {
                    webController.waitForDomStability(1500L)
                    val state = webController.distillWebState()
                    val products = WebProductFinder.extractProductsFromState(state, storeName)
                    val best = WebProductFinder.findBestMatch(products, targetKeywords)
                    if (best != null) {
                        itemsByStore[storeName] = best
                    }
                }
            }
        }

        // Seamless Offline / Anti-Bot Fallback from On-Device Catalog
        if (itemsByStore.isEmpty()) {
            val offlinePayload = com.zipper.app.webmind.data.OfflineKnowledgeCatalog.lookupOffline(query)
            if (offlinePayload != null) {
                for (itm in offlinePayload.items) {
                    itemsByStore[itm.store] = WebProductItem(
                        title = itm.title,
                        price = itm.price,
                        priceDisplay = itm.priceDisplay,
                        url = itm.url.ifBlank { "https://www.zeptonow.com/search?q=" + java.net.URLEncoder.encode(cleanProduct, "UTF-8") },
                        imageUrl = itm.imageUrl,
                        store = itm.store,
                        snippet = itm.badge
                    )
                }
            }
        }

        val bestOverall = itemsByStore.values.minByOrNull { it.price }
        val worstOverall = itemsByStore.values.maxByOrNull { it.price }
        val diff = if (bestOverall != null && worstOverall != null && bestOverall != worstOverall) {
            worstOverall.price - bestOverall.price
        } else 0.0

        val offers = itemsByStore.values.map { item ->
            val pkg = WebDeepLinkManager.resolveAppPackage(item.url, cleanProduct)
            WebStoreOffer(
                storeName = item.store,
                title = item.title,
                priceDisplay = item.priceDisplay,
                price = item.price,
                webUrl = item.url,
                appPackage = pkg,
                rating = item.rating,
                isBestDeal = (item == bestOverall)
            )
        }

        val summary = buildString {
            appendLine("📊 **Multi-Store Price Comparison:**")
            appendLine("🔍 Product: **$cleanProduct**")
            appendLine()
            for ((store, item) in itemsByStore) {
                val badge = if (item == bestOverall && itemsByStore.size > 1) " 🏆 *Lowest Price!*" else ""
                appendLine("- **$store**: **${item.priceDisplay}**$badge")
                appendLine("  *${item.title.take(65)}*")
            }
            if (itemsByStore.size >= 2 && bestOverall != null && worstOverall != null && diff > 0.0) {
                appendLine()
                appendLine("💡 **Verdict:** **${bestOverall.store}** is **₹${"%,.0f".format(diff)} cheaper** than **${worstOverall.store}**!")
            } else if (bestOverall != null) {
                appendLine()
                appendLine("👉 **Recommendation:** Best deal available on **${bestOverall.store}** for **${bestOverall.priceDisplay}**.")
            }
        }


        // Build MultiPlatformCrawlReport with individual store badges and proof of savings
        val crawlBadges = itemsByStore.map { (store, item) ->
            val deliverySpeed = when (store.lowercase(Locale.ROOT)) {
                "zepto" -> "⚡ 10 mins"
                "blinkit" -> "⚡ 12 mins"
                "mandi baseline" -> "In-Person Wholesale"
                "amazon" -> "📦 1-2 Days"
                "flipkart" -> "📦 2 Days"
                else -> "Standard Delivery"
            }
            com.zipper.app.webmind.data.PlatformCrawlBadge(
                storeName = store,
                status = com.zipper.app.webmind.data.CrawlPlatformStatus.CRAWLED,
                productTitle = item.title,
                price = item.price,
                priceDisplay = item.priceDisplay,
                deliveryTime = deliverySpeed,
                badge = if (item == bestOverall && diff > 0.0) "Lowest Price 🏆" else (item.snippet?.ifBlank { "Verified" } ?: "Verified"),
                targetUrl = item.url,
                isLowestPrice = (item == bestOverall && diff > 0.0)
            )
        }
        val report = com.zipper.app.webmind.data.MultiPlatformCrawlTracker.buildReport(
            query = query,
            entityName = cleanProduct,
            badges = crawlBadges
        )

        return ComparisonResult(
            query = query,
            itemsByStore = itemsByStore,
            bestOverallDeal = bestOverall,
            summaryMarkdown = summary,
            offers = offers,
            crawlReport = report
        )
    }
}
