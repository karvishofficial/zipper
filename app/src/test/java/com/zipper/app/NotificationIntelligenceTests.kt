package com.zipper.app

import com.zipper.app.core.notification.KarvishNotificationItem
import com.zipper.app.core.notification.NotificationCategory
import com.zipper.app.core.notification.NotificationQueryEngine
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class NotificationIntelligenceTests {

    @Test
    fun testNotificationQueryDetection() {
        // Trigger cases
        assertTrue(NotificationQueryEngine.isNotificationRelatedQuery("what was my last otp"))
        assertTrue(NotificationQueryEngine.isNotificationRelatedQuery("show otp code"))
        assertTrue(NotificationQueryEngine.isNotificationRelatedQuery("track swiggy order"))
        assertTrue(NotificationQueryEngine.isNotificationRelatedQuery("where is my uber"))
        assertTrue(NotificationQueryEngine.isNotificationRelatedQuery("summarize my notifications"))
        assertTrue(NotificationQueryEngine.isNotificationRelatedQuery("what did i miss"))
        assertTrue(NotificationQueryEngine.isNotificationRelatedQuery("any messages on whatsapp"))

        // Non-trigger cases (e.g. regular shopping or web search)
        assertFalse(NotificationQueryEngine.isNotificationRelatedQuery("buy 1kg tomato on zepto"))
        assertFalse(NotificationQueryEngine.isNotificationRelatedQuery("search best laptop on amazon"))
        assertFalse(NotificationQueryEngine.isNotificationRelatedQuery("rrce student login"))
    }

    @Test
    fun testAiSynthesisPromptFormatting() {
        val sampleItems = listOf(
            KarvishNotificationItem(
                id = "1",
                packageName = "com.whatsapp",
                appName = "WhatsApp",
                title = "Rahul",
                text = "Are we meeting at 5 PM?",
                category = NotificationCategory.MESSAGING,
                timestamp = System.currentTimeMillis()
            ),
            KarvishNotificationItem(
                id = "2",
                packageName = "in.swiggy.android",
                appName = "Swiggy",
                title = "Order Update",
                text = "Your food is arriving in 5 mins",
                category = NotificationCategory.DELIVERY_LOGISTICS,
                timestamp = System.currentTimeMillis()
            )
        )

        val prompt = NotificationQueryEngine.buildAiSynthesisPrompt(sampleItems)
        assertTrue(prompt.contains("WhatsApp"))
        assertTrue(prompt.contains("Rahul"))
        assertTrue(prompt.contains("Swiggy"))
        assertTrue(prompt.contains("executive summary"))
    }

    @Test
    fun testNotificationItemProperties() {
        val item = KarvishNotificationItem(
            id = "test_otp_1",
            packageName = "com.google.android.apps.messaging",
            appName = "Messages",
            title = "HDFC Bank",
            text = "Your OTP for transaction of Rs 500 is 849201. Do not share.",
            category = NotificationCategory.TRANSACTION_OTP,
            isOtp = true,
            otpCode = "849201"
        )

        assertEquals("849201", item.otpCode)
        assertTrue(item.isOtp)
        assertEquals(NotificationCategory.TRANSACTION_OTP, item.category)
    }
}
