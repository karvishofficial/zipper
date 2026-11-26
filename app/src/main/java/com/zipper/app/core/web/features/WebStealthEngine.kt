package com.zipper.app.core.web.features

import java.util.Locale

/**
 * Anti-Dynamic-Pricing Stealth Container.
 * Neutralizes browser fingerprinting vectors (Canvas, AudioContext, WebGL, navigator overrides)
 * to prevent airline/hotel websites from tracking queries and artificially escalating ticket fares.
 */
object WebStealthEngine {

    private val SUSPICIOUS_DYNAMIC_PRICING_DOMAINS = setOf(
        "makemytrip.com", "goibibo.com", "cleartrip.com", "easemytrip.com",
        "indigo.in", "airindia.com", "airasia.com", "spicejet.com",
        "booking.com", "agoda.com", "expedia.com", "skyscanner.co.in",
        "ixigo.com", "kayak.co.in", "irctc.co.in"
    )

    fun isDynamicPricingTarget(url: String): Boolean {
        val lower = url.lowercase(Locale.ROOT)
        return SUSPICIOUS_DYNAMIC_PRICING_DOMAINS.any { lower.contains(it) }
    }

    /**
     * Injects anti-fingerprinting JS to randomize Canvas hashes,
     * perturbs audio frequency graphs, and normalizes navigator properties.
     */
    fun generateStealthShieldJs(): String {
        return """
            (function() {
                try {
                    // 1. Mask navigator.webdriver
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined,
                        configurable: true
                    });

                    // 2. Normalize languages & platform
                    Object.defineProperty(navigator, 'languages', {
                        get: () => ['en-US', 'en'],
                        configurable: true
                    });

                    // 3. Canvas Fingerprint Noise Perturbation
                    const origToDataURL = HTMLCanvasElement.prototype.toDataURL;
                    HTMLCanvasElement.prototype.toDataURL = function(type) {
                        try {
                            const ctx = this.getContext('2d');
                            if (ctx) {
                                // Add subtle 1-pixel imperceptible variation
                                const imgData = ctx.getImageData(0, 0, Math.min(this.width, 2), Math.min(this.height, 2));
                                if (imgData && imgData.data.length > 0) {
                                    imgData.data[0] = (imgData.data[0] ^ 1);
                                    ctx.putImageData(imgData, 0, 0);
                                }
                            }
                        } catch(e) {}
                        return origToDataURL.apply(this, arguments);
                    };

                    // 4. AudioContext Fingerprint Perturbation
                    if (window.AudioContext || window.webkitAudioContext) {
                        const AudioCtx = window.AudioContext || window.webkitAudioContext;
                        const origGetFloatFreqData = AnalyserNode.prototype.getFloatFrequencyData;
                        AnalyserNode.prototype.getFloatFrequencyData = function(array) {
                            origGetFloatFreqData.apply(this, arguments);
                            for (let i = 0; i < array.length; i = i + 10) {
                                array[i] += 0.0001 * Math.sin(i);
                            }
                        };
                    }

                    // 5. Spoof WebGL Vendor & Renderer
                    const getParameter = WebGLRenderingContext.prototype.getParameter;
                    WebGLRenderingContext.prototype.getParameter = function(parameter) {
                        // UNMASKED_VENDOR_WEBGL
                        if (parameter === 37445) return "Google Inc. (Qualcomm)";
                        // UNMASKED_RENDERER_WEBGL
                        if (parameter === 37446) return "ANGLE (Qualcomm, Adreno (TM) 640 Direct3D11 vs_5_0 ps_5_0)";
                        return getParameter.apply(this, arguments);
                    };

                    return "STEALTH_SHIELD_ACTIVE";
                } catch(e) {
                    return "STEALTH_SHIELD_ERROR: " + e.message;
                }
            })();
        """.trimIndent()
    }
}
