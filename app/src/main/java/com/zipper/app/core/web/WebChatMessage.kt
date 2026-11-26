package com.zipper.app.core.web

import java.util.UUID

enum class ChatMessageType {
    USER,
    PROGRESS_STEP,
    ASSISTANT
}

data class WebStoreOffer(
    val storeName: String,
    val title: String,
    val priceDisplay: String,
    val price: Double,
    val webUrl: String,
    val appPackage: String? = null,
    val rating: String? = null,
    val isBestDeal: Boolean = false
)

data class WebResultData(
    val title: String,
    val price: String? = null,
    val snippet: String? = null,
    val webUrl: String? = null,
    val appPackage: String? = null,
    val originalGoal: String? = null,
    val imageUrl: String? = null,
    val imageUrls: List<String> = emptyList(),
    val rating: String? = null,
    val storeName: String? = null,
    var feedbackMessage: String? = null,
    var isFeedbackSubmitted: Boolean = false,
    val comparisonOffers: List<WebStoreOffer> = emptyList()
)

data class WebChatMessage(
    val id: String = UUID.randomUUID().toString(),
    val type: ChatMessageType,
    val text: String,
    val resultData: WebResultData? = null,
    val timestamp: Long = System.currentTimeMillis()
)
