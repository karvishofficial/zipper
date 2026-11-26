package com.zipper.app.webmind.data

/**
 * Rich model representing a single product/store item in a multi-way comparison.
 */
data class MatrixProduct(
    val id: String,
    val title: String,
    val shortName: String,
    val brand: String,
    val category: String,
    val price: Double,
    val priceDisplay: String,
    val unitPriceDisplay: String = "",
    val store: String,
    val rating: Float = 4.5f,
    val reviewCount: String = "Verified",
    val imageUrl: String = "",
    val productUrl: String = "",
    val deliverySpeed: String = "Standard Delivery",
    val badge: String = "",
    val isBestDeal: Boolean = false,
    val specs: Map<String, String> = emptyMap(),
    val pros: List<String> = emptyList(),
    val cons: List<String> = emptyList()
)

/**
 * A group of related specifications (e.g., "Performance", "Display", "Battery & Charging").
 */
data class SpecCategoryGroup(
    val categoryName: String,
    val rows: List<SpecMatrixRow>
)

/**
 * A single row in the spec comparison table (e.g. "Processor: A18 vs Snapdragon 8 Gen 3").
 */
data class SpecMatrixRow(
    val label: String,
    val valuesPerProduct: Map<String, String>
)

/**
 * Synthesized buyer review sentiment derived from verified user reviews.
 */
data class ReviewSentimentSummary(
    val entityName: String,
    val overallRating: Float,
    val totalReviews: String,
    val topPraises: List<String>,
    val topComplaints: List<String>,
    val reviewerVerdict: String
)

/**
 * Full multi-platform comparison payload containing products, specs grid,
 * review sentiment, and mathematical proof of savings.
 */
data class ComprehensiveComparisonPayload(
    val query: String,
    val title: String,
    val headline: String,
    val category: String,
    val products: List<MatrixProduct>,
    val specGroups: List<SpecCategoryGroup>,
    val sentiments: List<ReviewSentimentSummary>,
    val definitiveVerdict: String,
    val savingsProofText: String,
    val isOfflineKnowledge: Boolean = false
)
