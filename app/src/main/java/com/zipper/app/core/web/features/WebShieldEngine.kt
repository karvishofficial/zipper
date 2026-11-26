package com.zipper.app.core.web.features

import java.util.Locale

data class ShieldConfig(
    val bypassPaywall: Boolean = true,
    val blockAdsAndTrackers: Boolean = true,
    val spoofCrawlerUserAgent: Boolean = true
)

/**
 * Universal Paywall & Anti-Ad Shield.
 * Strips out paywall overlays, scrolling lockups, and intrusive ad-trackers
 * while allowing legitimate content hydration.
 */
object WebShieldEngine {

    const val GOOGLEBOT_USER_AGENT = "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
    const val BINGBOT_USER_AGENT = "Mozilla/5.0 (compatible; bingbot/2.0; +http://www.bing.com/bingbot.htm)"

    private val KNOWN_METERED_DOMAINS = setOf(
        "nytimes.com", "medium.com", "washingtonpost.com", "wsj.com",
        "thehindu.com", "indianexpress.com", "economictimes.indiatimes.com",
        "bloomberg.com", "ft.com", "theatlantic.com", "newyorker.com",
        "wired.com", "hbr.org", "businessinsider.com", "telegraph.co.uk"
    )

    fun isPaywalledDomain(url: String): Boolean {
        val lower = url.lowercase(Locale.ROOT)
        return KNOWN_METERED_DOMAINS.any { lower.contains(it) }
    }

    fun shouldSpoofCrawler(url: String): Boolean {
        return isPaywalledDomain(url)
    }

    /**
     * Generates a JavaScript injection payload to eradicate modal paywalls,
     * backdrop shades, and unlock body scrolling.
     */
    fun generatePaywallBypassJs(): String {
        return """
            (function() {
                try {
                    // 1. Restore scrollability
                    document.documentElement.style.setProperty('overflow', 'auto', 'important');
                    document.body.style.setProperty('overflow', 'auto', 'important');
                    document.body.style.setProperty('position', 'static', 'important');

                    // 2. Clear paywall backdrop overlays
                    const overlaySelectors = [
                        "div[class*='paywall' i]",
                        "div[id*='paywall' i]",
                        "div[class*='modal-backdrop' i]",
                        "div[class*='overlay' i][style*='fixed']",
                        "div[class*='subscription-prompt' i]",
                        "div[id*='regwall' i]",
                        "div[class*='gate' i][style*='fixed']",
                        "section[class*='metered-wall' i]"
                    ];

                    overlaySelectors.forEach(sel => {
                        document.querySelectorAll(sel).forEach(el => {
                            if (el) el.remove();
                        });
                    });

                    // 3. Remove blur & unhide truncated text
                    const blurred = document.querySelectorAll("[style*='blur'], [class*='blurred' i], [class*='locked-content' i]");
                    blurred.forEach(el => {
                        el.style.setProperty('filter', 'none', 'important');
                        el.style.setProperty('opacity', '1', 'important');
                        el.style.setProperty('pointer-events', 'auto', 'important');
                        el.style.setProperty('user-select', 'text', 'important');
                    });

                    // 4. Expand truncated article bodies
                    const truncated = document.querySelectorAll("[class*='truncate' i], [style*='max-height'], [class*='collapsed' i]");
                    truncated.forEach(el => {
                        el.style.setProperty('max-height', 'none', 'important');
                        el.style.setProperty('height', 'auto', 'important');
                        el.style.setProperty('overflow', 'visible', 'important');
                    });

                    return "SHIELD_PAYWALL_CLEARED";
                } catch(e) {
                    return "SHIELD_PAYWALL_ERROR: " + e.message;
                }
            })();
        """.trimIndent()
    }

    /**
     * Generates an aggressive CSS injection block to silence intrusive ad networks and trackers.
     */
    fun generateAdBlockRulesCss(): String {
        return """
            iframe[src*='doubleclick.net'],
            iframe[src*='googlesyndication.com'],
            iframe[src*='adservice.google'],
            div[class*='ad-banner' i],
            div[id*='ad-banner' i],
            div[class*='ad-slot' i],
            div[id*='google_ads' i],
            div[class*='taboola' i],
            div[class*='outbrain' i],
            ins.adsbygoogle,
            aside[class*='sponsored' i] {
                display: none !important;
                visibility: hidden !important;
                height: 0 !important;
                width: 0 !important;
                opacity: 0 !important;
                pointer-events: none !important;
            }
        """.trimIndent()
    }
}
