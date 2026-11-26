package com.zipper.app

import com.zipper.app.core.web.WebSuiteManager
import com.zipper.app.core.web.features.CouponTestResult
import com.zipper.app.core.web.features.FareItem
import com.zipper.app.core.web.features.FareQuery
import com.zipper.app.core.web.features.MediaType
import com.zipper.app.core.web.features.UserProfile
import com.zipper.app.core.web.features.WatchdogTarget
import com.zipper.app.core.web.features.ZappedRule
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class WebSuiteTests {

    @Test
    fun testWebCheckoutEngine() {
        val checkout = WebSuiteManager.checkout

        // Price parsing
        assertEquals(1499.0, checkout.parseCheckoutPrice("₹ 1,499.00") ?: 0.0, 0.01)
        assertEquals(49.99, checkout.parseCheckoutPrice("$49.99") ?: 0.0, 0.01)
        assertEquals(350.0, checkout.parseCheckoutPrice("Rs. 350") ?: 0.0, 0.01)

        // Autofill JS generation
        val profile = UserProfile(
            fullName = "Karthik S",
            phone = "9876543210",
            email = "karthik@zipper.app",
            addressLine = "Flat 402, Sunshine Heights",
            city = "Bengaluru",
            pincode = "560001",
            upiId = "karthik@okaxis"
        )
        val js = checkout.generateAutofillJs(profile)
        assertTrue(js.contains("Karthik S"))
        assertTrue(js.contains("9876543210"))
        assertTrue(js.contains("karthik@okaxis"))

        // Best coupon computation
        val results = listOf(
            CouponTestResult("WELCOME50", true, 1000.0, 500.0, 500.0),
            CouponTestResult("FLAT100", true, 1000.0, 900.0, 100.0),
            CouponTestResult("INVALID", false, 1000.0, 1000.0, 0.0)
        )
        val best = checkout.computeBestCoupon(1000.0, results)
        assertNotNull(best)
        assertEquals("WELCOME50", best?.code)
        assertEquals(500.0, best?.savings ?: 0.0, 0.01)
    }

    @Test
    fun testWebShieldEngine() {
        val shield = WebSuiteManager.shield

        assertTrue(shield.isPaywalledDomain("https://www.nytimes.com/2026/10/tech-article.html"))
        assertTrue(shield.isPaywalledDomain("https://medium.com/@author/deep-learning-notes"))
        assertTrue(shield.isPaywalledDomain("https://indianexpress.com/article/technology"))
        assertFalse(shield.isPaywalledDomain("https://en.wikipedia.org/wiki/Kotlin"))

        val bypassJs = shield.generatePaywallBypassJs()
        assertTrue(bypassJs.contains("overflow"))
        assertTrue(bypassJs.contains("paywall"))

        val adCss = shield.generateAdBlockRulesCss()
        assertTrue(adCss.contains("doubleclick.net"))
    }

    @Test
    fun testWebToNativeEscalator() {
        val escalator = WebSuiteManager.escalator

        // When Flipkart app is installed
        val resInstalled = escalator.resolveEscalation("https://www.flipkart.com/apple-iphone-15/p/itm123") { pkg ->
            pkg == "com.flipkart.android"
        }
        assertTrue(resInstalled.canEscalate)
        assertEquals("com.flipkart.android", resInstalled.targetPackage)
        assertTrue(resInstalled.intentUri?.startsWith("flipkart://dl/") == true)

        // When YouTube app is installed
        val resYt = escalator.resolveEscalation("https://www.youtube.com/watch?v=dQw4w9WgXcQ") { pkg ->
            pkg == "com.google.android.youtube"
        }
        assertTrue(resYt.canEscalate)
        assertEquals("vnd.youtube:dQw4w9WgXcQ", resYt.intentUri)

        // When Swiggy app is not installed
        val resNotInstalled = escalator.resolveEscalation("https://www.swiggy.com/restaurants/pizza-blr") { _ ->
            false
        }
        assertFalse(resNotInstalled.canEscalate)
        assertNull(resNotInstalled.intentUri)
    }

    @Test
    fun testWebStealthEngine() {
        val stealth = WebSuiteManager.stealth

        assertTrue(stealth.isDynamicPricingTarget("https://www.makemytrip.com/flights"))
        assertTrue(stealth.isDynamicPricingTarget("https://www.goibibo.com/flights/blr-del"))
        assertTrue(stealth.isDynamicPricingTarget("https://www.booking.com/hotel/in/bangalore"))
        assertFalse(stealth.isDynamicPricingTarget("https://github.com/torvalds/linux"))

        val stealthJs = stealth.generateStealthShieldJs()
        assertTrue(stealthJs.contains("toDataURL"))
        assertTrue(stealthJs.contains("webdriver"))
        assertTrue(stealthJs.contains("getFloatFrequencyData"))
    }

    @Test
    fun testWebFareMatrixScanner() {
        val scanner = WebSuiteManager.fareMatrix
        val query = FareQuery("BLR", "DEL", "2026-10-15")

        val rawFares = listOf(
            FareItem("MakeMyTrip", "IndiGo", "6E-201", "06:00", "08:45", 5500.0, ""),
            FareItem("EaseMyTrip", "Air India", "AI-505", "07:30", "10:15", 4800.0, ""),
            FareItem("Cleartrip", "Akasa", "QP-110", "11:00", "13:50", 6200.0, "")
        )

        val matrix = scanner.buildMatrix(rawFares, query)
        assertEquals(4800.0, matrix.lowestPrice, 0.01)
        assertEquals(6200.0, matrix.highestPrice, 0.01)
        assertEquals(1400.0, matrix.totalSavings, 0.01)
        assertTrue(matrix.fares.first().isCheapest)
        assertEquals("Air India", matrix.fares.first().operatorOrAirline)

        val md = scanner.formatMatrixMarkdown(matrix)
        assertTrue(md.contains("BLR ➔ DEL"))
        assertTrue(md.contains("Air India"))
        assertTrue(md.contains("CHEAPEST"))
    }

    @Test
    fun testWebStockWatchdog() {
        val watchdog = WebSuiteManager.watchdog

        // Stock status detection
        assertTrue(watchdog.detectStockStatus("In Stock. Usually dispatched within 24 hours."))
        assertFalse(watchdog.detectStockStatus("Currently unavailable. We don't know when this item will be back in stock."))
        assertFalse(watchdog.detectStockStatus("Sold Out"))

        // Alert evaluation on price drop
        val target = WatchdogTarget(
            id = "w1",
            title = "Sony WH-1000XM5 Headphones",
            url = "https://amazon.in/dp/B09XS7JWHH",
            targetPrice = 24990.0,
            inStock = true
        )

        val dropResult = watchdog.evaluate(target, currentPrice = 22990.0, isInStock = true)
        assertTrue(dropResult.triggeredAlert)
        assertTrue(dropResult.alertMessage.contains("Price dropped"))

        // Alert evaluation when price is higher
        val noDropResult = watchdog.evaluate(target, currentPrice = 26990.0, isInStock = true)
        assertFalse(noDropResult.triggeredAlert)
    }

    @Test
    fun testWebMediaExtractor() {
        val extractor = WebSuiteManager.mediaExtractor
        val html = """
            <html>
                <body>
                    <h1>Lecture Notes</h1>
                    <a href="/downloads/lecture_week_1.pdf">Download Slides</a>
                    <a href="https://cdn.example.com/videos/lecture.mp4">HD Video</a>
                </body>
            </html>
        """.trimIndent()

        val intercepted = listOf("https://stream.server.com/live/stream.m3u8")
        val media = extractor.extractMediaFromPage(html, "https://university.edu/course/", intercepted)

        assertEquals(3, media.size)
        assertTrue(media.any { it.type == MediaType.STREAM_PLAYLIST })
        assertTrue(media.any { it.type == MediaType.PDF && it.url == "https://university.edu/downloads/lecture_week_1.pdf" })
        assertTrue(media.any { it.type == MediaType.VIDEO && it.url == "https://cdn.example.com/videos/lecture.mp4" })
    }

    @Test
    fun testWebElementZapper() {
        val zapper = WebSuiteManager.zapper

        val rules = listOf(
            ZappedRule("example.com", "#annoying-banner"),
            ZappedRule("example.com", ".newsletter-popup")
        )
        val css = zapper.generatePersistentCss(rules)
        assertTrue(css.contains("#annoying-banner"))
        assertTrue(css.contains(".newsletter-popup"))
        assertTrue(css.contains("display: none !important;"))

        val cleanerJs = zapper.generateObstructionCleanerJs()
        assertTrue(cleanerJs.contains("innerHeight"))
    }

    @Test
    fun testWebPortalAssistantConstraints() {
        val portal = WebSuiteManager.portalAssistant

        // Constraint parsing from text
        val text1 = "Upload photograph in JPG format between 20 KB and 50 KB."
        val constraint1 = portal.parseConstraintsFromPage(text1)
        assertEquals(20 * 1024L, constraint1.minSizeBytes)
        assertEquals(50 * 1024L, constraint1.maxSizeBytes)

        val text2 = "Upload scanned signature not exceeding 20kb."
        val constraint2 = portal.parseConstraintsFromPage(text2)
        assertEquals(20 * 1024L, constraint2.maxSizeBytes)
    }

    @Test
    fun testWebSnapshotVault() {
        val vault = WebSuiteManager.snapshotVault
        val html = """
            <html>
                <body>
                    <h1>Understanding Attention Mechanisms</h1>
                    <p>Attention mechanisms have revolutionized natural language processing and computer vision.</p>
                    <h2>Key Components</h2>
                    <ul>
                        <li>Query, Key, and Value vectors</li>
                        <li>Softmax scaling factor</li>
                    </ul>
                </body>
            </html>
        """.trimIndent()

        val markdown = vault.convertHtmlToCleanMarkdown(html, "Attention Mechanisms", "https://ai.example.com/attention")
        assertTrue(markdown.contains("# Understanding Attention Mechanisms"))
        assertTrue(markdown.contains("## Key Components"))
        assertTrue(markdown.contains("* Query, Key, and Value vectors"))

        val (execSummary, keyPoints) = vault.generateExecutiveSummary(markdown)
        assertTrue(execSummary.isNotBlank())
        assertTrue(keyPoints.isNotEmpty())
    }
}
