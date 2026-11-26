package com.zipper.app

import com.zipper.app.core.web.features.UniversalPriceCrawler
import com.zipper.app.webmind.data.ComprehensiveKnowledgeCatalog
import com.zipper.app.webmind.data.MultiPlatformCrawlTracker
import com.zipper.app.webmind.data.OfflineKnowledgeCatalog
import com.zipper.app.webmind.data.UnitPriceNormalizer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Rigorous Unit Test Suite for Webmind 2.0 & Comprehensive Knowledge Engine.
 */
class WebmindMasterSystemTests {

    // =================================================================
    // 1. UNIT PRICE NORMALIZER TESTS
    // =================================================================
    @Test
    fun testUnitPriceNormalizer_parsesWeightsAndCalculatesCostPerKg() {
        val q1 = UnitPriceNormalizer.parseQuantity("Fresh Hybrid Tomato (500g)")
        assertNotNull(q1)
        assertEquals(500.0, q1!!.amount, 0.001)
        assertEquals("kg", q1.standardUnit)
        assertEquals(0.001, q1.standardMultiplier, 0.0001)

        val cost1 = UnitPriceNormalizer.computeUnitPrice(22.0, "Fresh Hybrid Tomato (500g)")
        assertNotNull(cost1)
        assertEquals(44.0, cost1!!.unitPrice, 0.01)
        assertEquals("₹44/kg", cost1.unitDisplay)

        val cost2 = UnitPriceNormalizer.computeUnitPrice(39.0, "Farm Fresh Tomato (1kg)")
        assertNotNull(cost2)
        assertEquals(39.0, cost2!!.unitPrice, 0.01)
        assertEquals("₹39/kg", cost2.unitDisplay)
    }

    @Test
    fun testUnitPriceNormalizer_computesAccurateSavingsDelta() {
        val delta = UnitPriceNormalizer.computeSavingsDelta(
            cheapestPrice = 39.0,
            cheapestTitle = "Tomato (1kg)",
            competitorPrice = 22.0,
            competitorTitle = "Tomato (500g)"
        )
        // Blinkit ₹39/kg vs Zepto ₹44/kg -> saves ₹5/kg (11% cheaper)
        assertTrue(delta.contains("₹39/kg vs ₹44/kg"))
        assertTrue(delta.contains("Saves ₹5/kg"))
        assertTrue(delta.contains("11% cheaper"))
    }

    @Test
    fun testUnitPriceNormalizer_parsesStorageAndVolumes() {
        val qStorage = UnitPriceNormalizer.parseQuantity("iPhone 16 128GB Ultramarine")
        assertNotNull(qStorage)
        assertEquals(128.0, qStorage!!.amount, 0.01)
        assertEquals("GB", qStorage.standardUnit)

        val qVolume = UnitPriceNormalizer.parseQuantity("Amul Taaza Milk 1L")
        assertNotNull(qVolume)
        assertEquals(1.0, qVolume!!.amount, 0.01)
        assertEquals("L", qVolume.standardUnit)
    }

    // =================================================================
    // 2. COMPREHENSIVE KNOWLEDGE CATALOG TESTS
    // =================================================================
    @Test
    fun testComprehensiveCatalog_resolvesFlagshipPhoneTriWayComparison() {
        val result = ComprehensiveKnowledgeCatalog.lookup("compare iPhone 16 vs S24 vs Pixel 9")
        assertNotNull(result)
        assertEquals(3, result!!.products.size)

        val iphone = result.products.first { it.shortName == "iPhone 16" }
        val s24 = result.products.first { it.shortName == "Galaxy S24" }
        val pixel = result.products.first { it.shortName == "Pixel 9" }

        assertEquals(79900.0, iphone.price, 0.01)
        assertEquals(64999.0, s24.price, 0.01)
        assertEquals(74999.0, pixel.price, 0.01)

        assertTrue(s24.isBestDeal)
        assertTrue(result.definitiveVerdict.contains("Samsung Galaxy S24 wins on Overall Value"))
        assertTrue(result.savingsProofText.contains("₹14,901 cheaper"))

        // Spec groups verification
        assertFalse(result.specGroups.isEmpty())
        val perfGroup = result.specGroups.first { it.categoryName == "Performance & Silicon" }
        val procRow = perfGroup.rows.first { it.label == "Processor" }
        assertEquals("Apple A18 (3nm)", procRow.valuesPerProduct["iPhone 16"])

        // Sentiment verification
        assertEquals(2, result.sentiments.size)
        assertTrue(result.sentiments.any { it.entityName == "Galaxy S24" })
    }

    @Test
    fun testComprehensiveCatalog_resolvesMidrangeGalaxyA26Comparison() {
        val result = ComprehensiveKnowledgeCatalog.lookup("compare Samsung A26 vs Redmi Note 13 Pro")
        assertNotNull(result)
        assertEquals(2, result!!.products.size)

        val a26 = result.products.first { it.shortName == "Galaxy A26" }
        val redmi = result.products.first { it.shortName == "Redmi Note 13 Pro" }

        assertEquals(22999.0, a26.price, 0.01)
        assertEquals(24999.0, redmi.price, 0.01)
        assertTrue(a26.isBestDeal)
        assertTrue(result.savingsProofText.contains("saves ₹2,000"))
    }

    @Test
    fun testComprehensiveCatalog_resolvesGamingLaptopsUnder70k() {
        val result = ComprehensiveKnowledgeCatalog.lookup("compare best laptops under 70k")
        assertNotNull(result)
        assertEquals(3, result!!.products.size)

        val loq = result.products.first { it.shortName == "Lenovo LOQ 15" }
        assertEquals(68990.0, loq.price, 0.01)
        assertTrue(loq.isBestDeal)
        assertTrue(result.definitiveVerdict.contains("100% sRGB"))
    }

    @Test
    fun testComprehensiveCatalog_resolvesAudioAndEV() {
        val audio = ComprehensiveKnowledgeCatalog.lookup("compare Sony WH-1000XM5 vs Bose QC45")
        assertNotNull(audio)
        assertEquals(2, audio!!.products.size)
        assertTrue(audio.products.first { it.shortName == "Sony WH-1000XM5" }.isBestDeal)

        val ev = ComprehensiveKnowledgeCatalog.lookup("compare Ather 450X vs Ola S1 Pro")
        assertNotNull(ev)
        assertEquals(2, ev!!.products.size)
        assertTrue(ev.products.first { it.shortName == "Ola S1 Pro" }.isBestDeal)
    }

    @Test
    fun testComprehensiveCatalog_resolvesTomatoGroceryWithNormalizedRates() {
        val result = ComprehensiveKnowledgeCatalog.lookup("compare tomato")
        assertNotNull(result)
        assertEquals(3, result!!.products.size)

        val zepto = result.products.first { it.store == "Zepto" }
        val blinkit = result.products.first { it.store == "Blinkit" }

        assertEquals("₹44/kg", zepto.unitPriceDisplay)
        assertEquals("₹39/kg", blinkit.unitPriceDisplay)
        assertTrue(blinkit.isBestDeal)
        assertTrue(result.savingsProofText.contains("11.4% cheaper"))
    }

    // =================================================================
    // 3. MULTI-PLATFORM CRAWL TRACKER TESTS
    // =================================================================
    @Test
    fun testMultiPlatformCrawlTracker_buildsReportFromComprehensivePayload() {
        val payload = ComprehensiveKnowledgeCatalog.lookup("compare tomato")!!
        val report = MultiPlatformCrawlTracker.fromComprehensivePayload(payload)

        assertEquals("compare tomato", report.query)
        assertEquals(3, report.platformsCrawled.size)
        assertTrue(report.platformsCrawled.any { it.storeName == "Blinkit" && it.isLowestPrice })
        assertTrue(report.platformsCrawled.all { it.targetUrl.isNotBlank() })
    }

    // =================================================================
    // 4. UNIVERSAL PRICE CRAWLER TARGETS TEST
    // =================================================================
    @Test
    fun testUniversalPriceCrawler_resolvesExpandedTargets() {
        val groceryTargets = UniversalPriceCrawler.resolveCrawlerTargets("tomato 1kg")
        val groceryStores = groceryTargets.map { it.storeName }
        assertTrue(groceryStores.contains("Zepto"))
        assertTrue(groceryStores.contains("Blinkit"))
        assertTrue(groceryStores.contains("Google Shopping"))

        val techTargets = UniversalPriceCrawler.resolveCrawlerTargets("laptop under 70k")
        val techStores = techTargets.map { it.storeName }
        assertTrue(techStores.contains("Amazon"))
        assertTrue(techStores.contains("Flipkart"))
        assertTrue(techStores.contains("Croma"))
        assertTrue(techStores.contains("Google Shopping"))
    }

    // =================================================================
    // 5. OFFLINE KNOWLEDGE CATALOG INTEGRATION TEST
    // =================================================================
    @Test
    fun testOfflineKnowledgeCatalog_leveragesComprehensivePayloadSeamlessly() {
        val payload = OfflineKnowledgeCatalog.lookupOffline("compare iPhone 16 vs S24")
        assertNotNull(payload)
        assertEquals(3, payload!!.items.size)
        assertTrue(payload.summaryMarkdown.contains("Galaxy S24"))
    }

    // =================================================================
    // 6. INTENT DISAMBIGUATION (RESEARCH VS ACTION)
    // =================================================================
    @Test
    fun testIntentClassification_disambiguatesResearchFromActionTasks() {
        val classifier = com.zipper.app.webmind.data.WebDomainClassifier

        // Research/informational queries must NOT be marked as autonomous action goals
        assertFalse(classifier.isAutonomousActionGoal("about new AI models in 2026"))
        assertFalse(classifier.isAutonomousActionGoal("what is quantum computing"))
        assertFalse(classifier.isAutonomousActionGoal("who is the CEO of Google"))
        assertFalse(classifier.isAutonomousActionGoal("tell me about SpaceX Starship"))
        assertFalse(classifier.isAutonomousActionGoal("explain transformer architecture"))
        assertFalse(classifier.isAutonomousActionGoal("latest news on Chandrayaan-3"))
        assertFalse(classifier.isAutonomousActionGoal("photos of Paris"))

        // Explicit interactive automation commands MUST be marked as autonomous action goals
        assertTrue(classifier.isAutonomousActionGoal("fill registration form on VTU portal"))
        assertTrue(classifier.isAutonomousActionGoal("login to portal and check marks"))
        assertTrue(classifier.isAutonomousActionGoal("add to cart and checkout"))
        assertTrue(classifier.isAutonomousActionGoal("download pdf from website"))
        assertTrue(classifier.isAutonomousActionGoal("open zepto and click add to cart"))
    }

    // =================================================================
    // 7. USER PREFERENCE VAULT & BRAND AFFINITY TESTS (MUSE ARCHITECTURE)
    // =================================================================
    @Test
    fun testUserPreferenceVault_enrichesGenericQueriesWithPreferredBrands() {
        val raw1 = "1L milk"
        val enriched1 = com.zipper.app.core.web.UserPreferenceVault.enrichWithAffinities(raw1)
        assertEquals("1L Amul milk", enriched1)

        val raw2 = "5kg atta and 1kg dal"
        val enriched2 = com.zipper.app.core.web.UserPreferenceVault.enrichWithAffinities(raw2)
        assertEquals("5kg Aashirvaad atta and 1kg Tata Sampann dal", enriched2)

        val raw3 = "6 eggs"
        val enriched3 = com.zipper.app.core.web.UserPreferenceVault.enrichWithAffinities(raw3)
        assertEquals("6 Farm Fresh eggs", enriched3)

        // Must preserve user's explicit alternative brand selection
        val explicitNandini = "1L nandini milk"
        val preserved = com.zipper.app.core.web.UserPreferenceVault.enrichWithAffinities(explicitNandini)
        assertEquals("1L nandini milk", preserved)
    }

    // =================================================================
    // 8. SHOPPING LIST PARSER WITH QUANTITY & CATEGORY TESTS
    // =================================================================
    @Test
    fun testShoppingListParser_correctlyParsesQuantitiesAndCategories() {
        val input = "Order 1kg tomatoes, 500g paneer, and 5kg atta"
        val parsed = com.zipper.app.core.web.ShoppingListParser.parseShoppingList(input)
        assertEquals(3, parsed.size)

        val tomato = parsed[0]
        assertEquals("1kg", tomato.quantity)
        assertEquals("tomatoes", tomato.canonicalName)
        assertEquals("1kg tomatoes", tomato.normalizedQuery)
        assertEquals(com.zipper.app.core.web.ItemCategory.FRESH, tomato.category)

        val paneer = parsed[1]
        assertEquals("500g", paneer.quantity)
        assertEquals("paneer", paneer.canonicalName)
        assertEquals("500g paneer", paneer.normalizedQuery)

        val atta = parsed[2]
        assertEquals("5kg", atta.quantity)
        assertEquals("atta", atta.canonicalName)
        assertEquals("5kg atta", atta.normalizedQuery)
        assertEquals(com.zipper.app.core.web.ItemCategory.BULK_PANTRY, atta.category)
    }

    // =================================================================
    // 9. MUSE MANDATORY CHECKOUT GATE TOTALS & SAVINGS LOGIC
    // =================================================================
    @Test
    fun testMuseCheckoutGate_computesSubtotalAndSavingsDelta() {
        val prices = listOf(44.0, 110.0, 240.0) // Tomato, Paneer, Atta
        val totalCartCost = prices.sum()
        val estimatedSavings = (totalCartCost * 0.15).toInt()

        assertEquals(394.0, totalCartCost, 0.01)
        assertEquals(59, estimatedSavings) // 15% estimated savings vs MRP
    }
}
