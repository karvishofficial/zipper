package com.zipper.app.webmind.data

import java.io.Serializable

data class WebProduct(
    val id: String,
    val title: String,
    val price: Double,
    val priceFormatted: String,
    val rating: Float,
    val reviewCount: String,
    val imageUrl: String,
    val productUrl: String,
    val store: String,
    val bestFor: String = "Best Overall",
    val specs: Map<String, String> = emptyMap()
) : Serializable

data class ResearchSession(
    val id: String,
    val query: String,
    val category: String, // "Search", "Compare", "Track", "Summarize"
    val timestamp: Long,
    val summaryText: String,
    val iconType: String = "search"
) : Serializable {
    fun getFormattedTime(): String {
        val diff = System.currentTimeMillis() - timestamp
        val mins = diff / (60 * 1000)
        val hours = diff / (60 * 60 * 1000)
        val days = diff / (24 * 60 * 60 * 1000)
        return when {
            mins < 1 -> "Just now"
            mins < 60 -> "$mins min ago"
            hours < 24 -> "$hours hour${if (hours > 1) "s" else ""} ago"
            days < 7 -> "$days day${if (days > 1) "s" else ""} ago"
            else -> "${days / 7} weeks ago"
        }
    }
}

data class TrackedProduct(
    val id: String,
    val title: String,
    val store: String,
    val currentPrice: Double,
    val targetPrice: Double,
    val thresholdType: String, // "BELOW", "DROPS_BY", "ABOVE"
    val productUrl: String,
    val imageUrl: String,
    val isActive: Boolean = true,
    val createdTime: Long = System.currentTimeMillis()
) : Serializable

data class PriceHistoryPoint(
    val productId: String,
    val price: Double,
    val timestamp: Long,
    val dateLabel: String
) : Serializable

data class SummaryResult(
    val sourceUrl: String,
    val title: String,
    val publisher: String,
    val dateStr: String,
    val thumbnailUrl: String,
    val takeaways: List<String>,
    val faqs: List<Pair<String, String>>
) : Serializable
