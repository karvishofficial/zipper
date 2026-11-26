package com.zipper.app.core.notification

enum class NotificationCategory {
    MESSAGING,          // WhatsApp, Telegram, Slack, Messages, Gmail
    TRANSACTION_OTP,    // Bank alerts, 2FA codes, OTPs
    DELIVERY_LOGISTICS, // Swiggy, Zomato, Zepto, Blinkit, Amazon, Uber
    GENERAL             // Other apps and system utilities
}

data class KarvishNotificationItem(
    val id: String,
    val packageName: String,
    val appName: String,
    val title: String,
    val text: String,
    val subText: String? = null,
    val category: NotificationCategory,
    val timestamp: Long = System.currentTimeMillis(),
    val isOtp: Boolean = false,
    val otpCode: String? = null
)
