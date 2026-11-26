package com.zipper.app.core.web.features

import java.util.Locale

data class UserProfile(
    val fullName: String,
    val phone: String,
    val email: String,
    val addressLine: String,
    val city: String,
    val pincode: String,
    val state: String = "Karnataka",
    val upiId: String? = null
)

data class CouponCandidate(
    val code: String,
    val discountDescription: String,
    val minSpend: Double = 0.0
)

data class CouponTestResult(
    val code: String,
    val success: Boolean,
    val previousTotal: Double,
    val newTotal: Double,
    val savings: Double
)

data class CheckoutFormFields(
    val hasAddressFields: Boolean,
    val hasPaymentFields: Boolean,
    val hasCouponInput: Boolean,
    val detectedPrice: Double?
)

/**
 * Ghost Checkout & Coupon Blitz Engine.
 * Provides high-speed deterministic form autofill for addresses, payment tokens,
 * and automated headless coupon iteration to discover the lowest checkout total.
 */
object WebCheckoutEngine {

    val DEFAULT_GLOBAL_COUPONS = listOf(
        CouponCandidate("WELCOME50", "Flat 50% / Welcome Discount", 100.0),
        CouponCandidate("FLAT100", "Flat ₹100 Off", 500.0),
        CouponCandidate("SUPER20", "20% Instant Savings", 200.0),
        CouponCandidate("NEWUSER", "New Customer Discount", 0.0),
        CouponCandidate("FREESHIP", "Free Delivery / Zero Shipping", 0.0),
        CouponCandidate("UPI10", "10% Off on UPI Payment", 300.0),
        CouponCandidate("DISCOUNT10", "Standard 10% Coupon", 250.0)
    )

    fun parseCheckoutPrice(priceStr: String): Double? {
        val clean = priceStr.replace(",", "").trim()
        val regex = Regex("""(?:₹|Rs\.?|INR|\$|€|£)?\s*([0-9]+(?:\.[0-9]+)?)""", RegexOption.IGNORE_CASE)
        val match = regex.find(clean) ?: return null
        return match.groupValues[1].toDoubleOrNull()
    }

    fun generateAutofillJs(profile: UserProfile): String {
        return """
            (function() {
                try {
                    function setValue(selector, val) {
                        const inputs = document.querySelectorAll(selector);
                        inputs.forEach(el => {
                            if (el && !el.disabled && el.offsetParent !== null) {
                                el.focus();
                                el.value = val;
                                el.dispatchEvent(new Event('input', { bubbles: true }));
                                el.dispatchEvent(new Event('change', { bubbles: true }));
                                el.dispatchEvent(new Event('blur', { bubbles: true }));
                            }
                        });
                    }

                    // Full Name
                    setValue("input[autocomplete*='name'], input[name*='name' i], input[id*='name' i], input[placeholder*='name' i]", "${profile.fullName.replace("\"", "\\\"")}");
                    
                    // Phone Number
                    setValue("input[type='tel'], input[autocomplete*='tel'], input[name*='phone' i], input[name*='mobile' i], input[placeholder*='mobile' i]", "${profile.phone}");
                    
                    // Email
                    setValue("input[type='email'], input[autocomplete*='email'], input[name*='email' i], input[placeholder*='email' i]", "${profile.email}");
                    
                    // Pincode / Zipcode
                    setValue("input[autocomplete*='postal-code'], input[name*='pincode' i], input[name*='pin' i], input[name*='zip' i], input[placeholder*='pin' i]", "${profile.pincode}");
                    
                    // City
                    setValue("input[autocomplete*='address-level2'], input[name*='city' i], input[id*='city' i]", "${profile.city.replace("\"", "\\\"")}");
                    
                    // Address
                    setValue("input[autocomplete*='address-line1'], textarea[name*='address' i], input[name*='address' i], textarea[placeholder*='address' i]", "${profile.addressLine.replace("\"", "\\\"")}");

                    // UPI ID if available
                    ${if (profile.upiId != null) """
                    setValue("input[placeholder*='upi' i], input[name*='vpa' i], input[id*='vpa' i]", "${profile.upiId}");
                    """ else ""}

                    return "AUTOFILL_SUCCESS";
                } catch(e) {
                    return "AUTOFILL_ERROR: " + e.message;
                }
            })();
        """.trimIndent()
    }

    fun generateCouponTestJs(couponCode: String): String {
        return """
            (function() {
                try {
                    const couponInputs = Array.from(document.querySelectorAll("input[placeholder*='coupon' i], input[name*='coupon' i], input[id*='coupon' i], input[placeholder*='promo' i], input[name*='promo' i], input[placeholder*='voucher' i]"));
                    const targetInput = couponInputs.find(el => el.offsetParent !== null && !el.disabled);
                    if (!targetInput) return JSON.stringify({ found: false, error: "NO_COUPON_INPUT" });

                    targetInput.focus();
                    targetInput.value = "$couponCode";
                    targetInput.dispatchEvent(new Event('input', { bubbles: true }));
                    targetInput.dispatchEvent(new Event('change', { bubbles: true }));

                    // Find Apply button adjacent or within parent form
                    const parent = targetInput.closest('form') || targetInput.parentElement || document.body;
                    const buttons = Array.from(parent.querySelectorAll("button, input[type='button'], input[type='submit'], a, span"));
                    const applyBtn = buttons.find(b => {
                        const txt = (b.innerText || b.value || "").toLowerCase().trim();
                        return (txt === 'apply' || txt === 'redeem' || txt === 'use' || txt.includes('apply coupon')) && b.offsetParent !== null;
                    });

                    if (applyBtn) {
                        applyBtn.click();
                        return JSON.stringify({ found: true, applied: true, code: "$couponCode" });
                    }
                    return JSON.stringify({ found: true, applied: false, error: "NO_APPLY_BUTTON" });
                } catch(e) {
                    return JSON.stringify({ found: false, error: e.message });
                }
            })();
        """.trimIndent()
    }

    fun detectCheckoutFields(htmlText: String): CheckoutFormFields {
        val lower = htmlText.lowercase(Locale.ROOT)
        val hasAddress = lower.contains("pincode") || lower.contains("delivery address") ||
            lower.contains("shipping address") || lower.contains("postal code") || lower.contains("street address")
        val hasPayment = lower.contains("upi") || lower.contains("card number") ||
            lower.contains("cash on delivery") || lower.contains("cvv") || lower.contains("net banking")
        val hasCoupon = lower.contains("apply coupon") || lower.contains("promo code") ||
            lower.contains("discount code") || lower.contains("have a coupon")

        // Look for price totals
        val priceRegex = Regex("""(?:grand total|total amount|order total|total payable|to pay)\s*[:=]?\s*(?:₹|Rs\.?|INR|\$)?\s*([0-9,]+(?:\.[0-9]+)?)""", RegexOption.IGNORE_CASE)
        val priceMatch = priceRegex.find(htmlText)
        val detectedPrice = priceMatch?.groupValues?.get(1)?.let { parseCheckoutPrice(it) }

        return CheckoutFormFields(
            hasAddressFields = hasAddress,
            hasPaymentFields = hasPayment,
            hasCouponInput = hasCoupon,
            detectedPrice = detectedPrice
        )
    }

    fun computeBestCoupon(
        initialTotal: Double,
        testResults: List<CouponTestResult>
    ): CouponTestResult? {
        val successful = testResults.filter { it.success && it.savings > 0 }
        return successful.maxByOrNull { it.savings }
    }
}
