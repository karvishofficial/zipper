package com.zipper.app.core.web.features

import java.util.Locale

data class ZappedRule(
    val domain: String,
    val selector: String,
    val reason: String = "User Zapped",
    val createdAt: Long = System.currentTimeMillis()
)

/**
 * Permanent Element Zapper & Sticky HUD.
 * Allows interactive or automatic suppression of obstructing DOM banners,
 * permanently saving custom CSS selectors per domain.
 */
object WebElementZapper {

    fun generatePersistentCss(rules: List<ZappedRule>): String {
        if (rules.isEmpty()) return ""
        val selectors = rules.joinToString(",\n") { it.selector }
        return """
            $selectors {
                display: none !important;
                visibility: hidden !important;
                height: 0 !important;
                max-height: 0 !important;
                opacity: 0 !important;
                pointer-events: none !important;
            }
        """.trimIndent()
    }

    /**
     * Injects an interactive DOM inspector into the WebView.
     * When tapped, highlights the element and reports its unique CSS selector.
     */
    fun generateZapperHudJs(): String {
        return """
            (function() {
                if (window.__zipper_zapper_active) return "ALREADY_ACTIVE";
                window.__zipper_zapper_active = true;

                let hoveredEl = null;
                const overlay = document.createElement('div');
                overlay.id = '__zipper_zapper_highlight';
                overlay.style.position = 'fixed';
                overlay.style.pointerEvents = 'none';
                overlay.style.border = '2px solid #FF3B30';
                overlay.style.backgroundColor = 'rgba(255, 59, 48, 0.2)';
                overlay.style.zIndex = '2147483647';
                overlay.style.display = 'none';
                document.body.appendChild(overlay);

                function getSelector(el) {
                    if (el.id) return '#' + CSS.escape(el.id);
                    if (el.className && typeof el.className === 'string') {
                        const classes = el.className.trim().split(/\s+/).filter(c => c.length > 0 && !c.includes(':'));
                        if (classes.length > 0) return el.tagName.toLowerCase() + '.' + classes.map(c => CSS.escape(c)).join('.');
                    }
                    return el.tagName.toLowerCase();
                }

                document.addEventListener('mouseover', function(e) {
                    if (!window.__zipper_zapper_active) return;
                    hoveredEl = e.target;
                    const rect = hoveredEl.getBoundingClientRect();
                    overlay.style.top = rect.top + 'px';
                    overlay.style.left = rect.left + 'px';
                    overlay.style.width = rect.width + 'px';
                    overlay.style.height = rect.height + 'px';
                    overlay.style.display = 'block';
                }, true);

                document.addEventListener('click', function(e) {
                    if (!window.__zipper_zapper_active) return;
                    e.preventDefault();
                    e.stopPropagation();
                    if (hoveredEl) {
                        const sel = getSelector(hoveredEl);
                        hoveredEl.remove();
                        overlay.style.display = 'none';
                        window.__zipper_zapper_active = false;
                        if (window.ZipperNativeHost && window.ZipperNativeHost.onElementZapped) {
                            window.ZipperNativeHost.onElementZapped(sel);
                        }
                    }
                }, true);

                return "ZAPPER_ARMED";
            })();
        """.trimIndent()
    }

    /**
     * Automatic cleaner script that scans for fixed/sticky headers/footers
     * occupying >25% of the viewport height and suppresses them.
     */
    fun generateObstructionCleanerJs(): String {
        return """
            (function() {
                try {
                    let cleanedCount = 0;
                    const vh = window.innerHeight;
                    const elements = Array.from(document.querySelectorAll('div, header, footer, aside, section, nav'));

                    elements.forEach(el => {
                        const style = window.getComputedStyle(el);
                        if (style.position === 'fixed' || style.position === 'sticky') {
                            const rect = el.getBoundingClientRect();
                            if (rect.height > (vh * 0.25) || (parseInt(style.zIndex, 10) > 1000 && rect.height > (vh * 0.15))) {
                                el.style.setProperty('display', 'none', 'important');
                                cleanedCount++;
                            }
                        }
                    });

                    return "CLEANED_" + cleanedCount;
                } catch(e) {
                    return "CLEAN_ERROR: " + e.message;
                }
            })();
        """.trimIndent()
    }
}
