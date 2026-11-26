package com.zipper.app.core.web

import android.content.Context
import com.zipper.app.core.web.features.ArticleReaderEngine
import com.zipper.app.core.web.features.PnrTrackerEngine
import com.zipper.app.core.web.features.PriceComparisonEngine
import com.zipper.app.core.web.features.WebCheckoutEngine
import com.zipper.app.core.web.features.WebElementZapper
import com.zipper.app.core.web.features.WebFareMatrixScanner
import com.zipper.app.core.web.features.WebMediaExtractor
import com.zipper.app.core.web.features.WebPortalAssistant
import com.zipper.app.core.web.features.WebShieldEngine
import com.zipper.app.core.web.features.WebSnapshotVault
import com.zipper.app.core.web.features.WebStealthEngine
import com.zipper.app.core.web.features.WebStockWatchdog
import com.zipper.app.core.web.features.WebToNativeEscalator

/**
 * Central Coordinator for the Karvish Web Super Suite.
 * Unifies all 10 dual-engine modules (AI-OFF deterministic rules & AI-ON neural hooks)
 * with existing web controllers and headless runners.
 */
object WebSuiteManager {

    // Feature 1: Ghost Checkout & Coupon Blitz
    val checkout = WebCheckoutEngine

    // Feature 2: Universal Paywall & Anti-Ad Shield
    val shield = WebShieldEngine

    // Feature 3: Web-to-Native App Escalator
    val escalator = WebToNativeEscalator

    // Feature 4: Anti-Dynamic-Pricing Stealth Container
    val stealth = WebStealthEngine

    // Feature 5: Multi-Store Fare & Ticket Matrix Scanner
    val fareMatrix = WebFareMatrixScanner

    // Feature 6: Background Drop Watchdog
    val watchdog = WebStockWatchdog

    // Feature 7: Direct Media & Clean Document Extractor
    val mediaExtractor = WebMediaExtractor

    // Feature 8: Permanent Element Zapper & Sticky HUD
    val zapper = WebElementZapper

    // Feature 9: Indian Government & Exam Portal Assistant
    val portalAssistant = WebPortalAssistant

    // Feature 10: Offline Web Snapshot & Knowledge Vault
    val snapshotVault = WebSnapshotVault

    // Auxiliary Specialized Engines
    val reader = ArticleReaderEngine
    val pnrTracker = PnrTrackerEngine
    val priceComparison = PriceComparisonEngine

    /**
     * Injects baseline protective scripts (Stealth + Paywall Bypass + Ad Block) into a WebView.
     */
    fun buildPageSanitizationScript(url: String, enableShield: Boolean = true, enableStealth: Boolean = true): String {
        val scripts = mutableListOf<String>()

        if (enableStealth && stealth.isDynamicPricingTarget(url)) {
            scripts.add(stealth.generateStealthShieldJs())
        }

        if (enableShield) {
            if (shield.isPaywalledDomain(url)) {
                scripts.add(shield.generatePaywallBypassJs())
            }
            scripts.add(zapper.generateObstructionCleanerJs())
        }

        return scripts.joinToString("\n\n")
    }
}
