package com.zipper.app.core.web

import android.util.Log
import com.zipper.app.accessibility.KarvishAccessibilityService
import com.zipper.app.core.agent.OnDeviceRankerEngine
import com.zipper.app.core.model.ActionType
import com.zipper.app.core.model.DistilledNode
import com.zipper.app.core.model.DistilledUiState
import kotlinx.coroutines.delay
import java.util.Locale

/**
 * Autonomous Native Closer for Web-Scouted Multi-Product Relays.
 * Inspects live on-screen product cards, strictly filters out distractors (chips, wafers, ads),
 * matches requested units (500g, 1kg), selects the CHEAPEST and BEST genuine product,
 * and executes physical hardware gesture taps to guarantee addition to cart.
 */
object WebNativeRelayCloser {

    private const val TAG = "WebNativeRelayCloser"

    sealed class AddToCartResult {
        data class Success(
            val productTitle: String,
            val price: Double,
            val priceDisplay: String,
            val unit: String?,
            val viewsExplored: Int,
            val freeDeliveryPrompt: String? = null
        ) : AddToCartResult()

        data class OutOfStock(
            val reason: String,
            val viewsExplored: Int
        ) : AddToCartResult()

        data class Failed(
            val reason: String,
            val viewsExplored: Int
        ) : AddToCartResult()
    }

    data class LiveProductCard(
        val title: String,
        val price: Double,
        val unit: String?,
        val isAd: Boolean,
        val addNodeIndex: Int,
        val topBound: Int,
        val hasKeywordMatch: Boolean = false
    )

    data class ObservedCard(
        val title: String,
        val price: Double,
        val unit: String?,
        val isAd: Boolean,
        val viewportSeen: Int,
        val topBound: Int
    )

    suspend fun attemptAddToCart(
        targetAppPackage: String?,
        itemQuery: String = "",
        expectedKeywords: List<String> = emptyList(),
        maxScrolls: Int = 2,
        timeoutMs: Long = 25000L
    ): AddToCartResult {
        val a11y = KarvishAccessibilityService.instance
        if (a11y == null) {
            Log.w(TAG, "KarvishAccessibilityService instance is not connected.")
            return AddToCartResult.Failed("Accessibility service not connected", 0)
        }

        // 1. Wait for target app to enter foreground with live UI nodes
        val startWait = System.currentTimeMillis()
        var targetReady = false
        while (System.currentTimeMillis() - startWait < 8000L) {
            val state = a11y.captureCurrentState()
            val currentPkg = state.packageName?.lowercase(Locale.ROOT)
            val isTargetApp = if (!targetAppPackage.isNullOrBlank()) {
                currentPkg?.contains(targetAppPackage.lowercase(Locale.ROOT)) == true
            } else {
                currentPkg != null && currentPkg != "com.zipper.app"
            }
            if (isTargetApp && state.nodes.size >= 4) {
                targetReady = true
                break
            }
            delay(400L)
        }

        if (!targetReady) {
            Log.w(TAG, "Target app $targetAppPackage did not reach foreground with UI nodes.")
            return AddToCartResult.Failed("Store app not ready in foreground", 0)
        }

        // Pre-warm Karvish neural engine for spatial candidate understanding
        try {
            OnDeviceRankerEngine.preWarm(a11y.applicationContext, "models/karvish_v62.onnx")
        } catch (_: Exception) {}

        // 2. Mandatory Multi-Viewport Exploration Budget:
        // Strictly inspects Viewport 0, Viewport 1, Viewport 2...
        // Prevents skipping scrolling and exposes genuine local items below top-tier sponsored cards.
        val observedPool = mutableListOf<ObservedCard>()
        var outOfStockDetected = false

        for (viewport in 0..maxScrolls) {
            delay(if (viewport == 0) 1200L else 800L)
            val state = a11y.captureCurrentState()

            // Check for explicit out-of-stock indicators
            val hasOosText = state.nodes.any { n ->
                val lbl = n.semanticLabel.lowercase(Locale.ROOT)
                lbl.contains("out of stock") || lbl.contains("sold out") ||
                    lbl.contains("currently unavailable") || lbl.contains("no results found") ||
                    lbl.contains("item not found") || lbl.contains("not available")
            }
            if (hasOosText) {
                outOfStockDetected = true
            }

            // Extract all visible cards in this viewport
            val cardsInView = extractLiveProductCards(state.nodes, expectedKeywords)
            for (card in cardsInView) {
                val alreadyAdded = observedPool.any {
                    it.title.equals(card.title, ignoreCase = true) ||
                        (Math.abs(it.price - card.price) < 0.5 && it.title.take(15).equals(card.title.take(15), ignoreCase = true))
                }
                if (!alreadyAdded) {
                    observedPool.add(
                        ObservedCard(
                            title = card.title,
                            price = card.price,
                            unit = card.unit,
                            isAd = card.isAd,
                            viewportSeen = viewport,
                            topBound = card.topBound
                        )
                    )
                }
            }

            Log.i(TAG, "Viewport $viewport explored: found ${cardsInView.size} cards (pool total: ${observedPool.size})")

            // Execute scroll down if more exploration rounds remain
            if (viewport < maxScrolls) {
                Log.i(TAG, "Executing mandatory exploration scroll (${viewport + 1}/$maxScrolls)")
                a11y.executeAction(null, ActionType.SCROLL_DOWN, null)
                delay(700L)
            }
        }

        // 3. Strict Distractor & Keyword Filtering
        val queryLower = itemQuery.lowercase(Locale.ROOT)
        val isGrocery = WebProductFinder.isGroceryQuery(itemQuery)
        val groceryNegativeDistractors = setOf(
            "chips", "wafers", "wafer", "katori", "namkeen", "pakoda", "ketchup",
            "sauce", "seeds", "powder", "soup", "puree", "paste", "flavour", "flavoured",
            "canned", "tin", "seed", "oil", "juice", "chutney"
        )
        val electronicsNegativeDistractors = setOf(
            "case", "cover", "pouch", "skin", "adapter", "cable", "cleaner", "box", "dummy", "strap"
        )
        val rawDistractors = if (isGrocery) groceryNegativeDistractors else electronicsNegativeDistractors
        val activeDistractors = rawDistractors.filter { !queryLower.contains(it) }

        val coreKeywords = expectedKeywords.filter { kw ->
            kw.length >= 3 && kw !in setOf("item", "fresh", "best", "good", "order", "buy", "gram", "kilo")
        }

        // Semantic Attribute & Brand Guardrails (Never buy chemical/wrong variants)
        val requestedQualifiers = listOf(
            "organic", "cold-pressed", "cold pressed", "kachhi ghani", "sugar-free", "sugar free",
            "zero sugar", "diet", "lactose-free", "lactose free", "whole wheat", "brown",
            "multigrain", "gluten-free", "gluten free", "toned", "double toned", "full cream",
            "cow milk", "extra virgin", "virgin"
        ).filter { queryLower.contains(it) }

        val requestedBrands = listOf(
            "amul", "fortune", "aashirvaad", "tata", "nandini", "mother dairy", "britannia",
            "dettol", "surf excel", "vim", "colgate", "parle", "nestle", "cadbury", "lays",
            "kurkure", "saffola", "epigamia", "country delight", "akshayakalpa", "milky mist"
        ).filter { queryLower.contains(it) }

        val validMatches = observedPool.filter { card ->
            val t = card.title.lowercase(Locale.ROOT)
            val notAd = !card.isAd && !t.contains("sponsored") && !t.startsWith("ad ")
            val noDistractors = activeDistractors.none { t.contains(it) }
            val matchesKeyword = if (coreKeywords.isNotEmpty()) {
                coreKeywords.any { t.contains(it) }
            } else true

            val matchesQualifiers = if (requestedQualifiers.isNotEmpty()) {
                requestedQualifiers.all { t.contains(it.replace("-", " ")) || t.contains(it.replace(" ", "-")) || t.contains(it) }
            } else true

            val matchesBrands = if (requestedBrands.isNotEmpty()) {
                requestedBrands.any { t.contains(it) }
            } else true

            notAd && noDistractors && matchesKeyword && matchesQualifiers && matchesBrands
        }

        // 4. Product Detail Page (PDP) Mode Check
        // If no grid cards were found, check if the app opened directly into a single Product Detail Page (Amazon/Flipkart)
        if (validMatches.isEmpty() && !outOfStockDetected) {
            Log.i(TAG, "Checking if current screen is a Product Detail Page (PDP)...")
            var pdpAddBtn: DistilledNode? = null
            var pdpState = a11y.captureCurrentState()

            // Check current screen, or scroll down up to 2 times to find the buried PDP Add to Cart button (standard on Amazon/Flipkart)
            for (pdpScroll in 0..2) {
                pdpAddBtn = pdpState.nodes.firstOrNull { n ->
                    val lbl = n.semanticLabel.trim().lowercase(Locale.ROOT)
                    val res = n.resourceId?.lowercase(Locale.ROOT).orEmpty()
                    val isPdpCta = lbl == "add to cart" || lbl == "add to basket" || lbl == "add to fresh cart" ||
                        lbl.contains("add to cart") || lbl.contains("add to basket") || lbl.contains("add to fresh") ||
                        lbl == "buy now" || res.contains("add-to-cart-button") || res.contains("add_to_cart") ||
                        res.contains("btn_add_to_cart") || res.contains("buy_now")
                    isPdpCta && n.bounds.top > 250 && (n.isClickable || n.role in setOf("BUTTON", "VIEW", "TEXT"))
                }

                if (pdpAddBtn != null) break

                if (pdpScroll < 2) {
                    Log.i(TAG, "PDP Add button not visible yet, scrolling down to locate it ($pdpScroll/2)")
                    a11y.executeAction(null, ActionType.SCROLL_DOWN, null)
                    delay(800L)
                    pdpState = a11y.captureCurrentState()
                }
            }

            if (pdpAddBtn != null) {
                Log.i(TAG, "Found PDP purchase button '${pdpAddBtn.semanticLabel}' at index=${pdpAddBtn.index}. Executing physical TAP.")
                a11y.executeAction(pdpAddBtn.index, ActionType.TAP, null)
                delay(1200L)

                // Handle any warranty / accessory / replacement upsell sheet
                val postPdpState = a11y.captureCurrentState()
                handleUniversalModalOrSheet(a11y, postPdpState, itemQuery)
                delay(800L)

                // Verify cart state
                var pdpVerified = false
                for (chk in 1..4) {
                    val st = a11y.captureCurrentState()
                    val hasCartEvidence = st.nodes.any { n ->
                        val lbl = n.semanticLabel.trim().lowercase(Locale.ROOT)
                        val res = n.resourceId?.lowercase(Locale.ROOT).orEmpty()
                        lbl.contains("view cart") || lbl.contains("item in cart") || lbl.contains("added to cart") ||
                            lbl.contains("go to cart") || lbl.contains("checkout") || res.contains("cart_count") ||
                            res.contains("action_bar_cart") || res.contains("floating-cart") || res.contains("view_cart")
                    }
                    if (hasCartEvidence) {
                        pdpVerified = true
                        break
                    }
                    delay(500L)
                }

                if (pdpVerified) {
                    Log.i(TAG, "PDP Cart addition verified for '$itemQuery'")
                    return AddToCartResult.Success(
                        productTitle = itemQuery,
                        price = 0.0,
                        priceDisplay = "Added",
                        unit = null,
                        viewsExplored = maxScrolls + 1
                    )
                }
            }
        }

        // 5. Certified Out-of-Stock Check
        if (validMatches.isEmpty()) {
            if (outOfStockDetected || observedPool.isEmpty()) {
                Log.w(TAG, "Out of stock certified for '$itemQuery' across ${maxScrolls + 1} viewports")
                return AddToCartResult.OutOfStock(
                    reason = "Product is out of stock / unavailable",
                    viewsExplored = maxScrolls + 1
                )
            } else {
                Log.w(TAG, "No genuine matches found matching $coreKeywords across ${maxScrolls + 1} viewports")
                return AddToCartResult.OutOfStock(
                    reason = "No matching products available for $itemQuery",
                    viewsExplored = maxScrolls + 1
                )
            }
        }

        // 5. Quantity / Unit Matching (e.g. 500g, 1kg)
        val targetQuantityRegex = Regex("""(?i)\b(\d+\s*(?:g|kg|gm|ml|l|pc))\b""")
        val targetQuantity = targetQuantityRegex.find(queryLower)?.groupValues?.get(1)?.lowercase(Locale.ROOT)?.replace(" ", "")

        val candidatePool = if (targetQuantity != null) {
            val qtyMatches = validMatches.filter {
                val u = it.unit?.replace(" ", "").orEmpty()
                val t = it.title.lowercase(Locale.ROOT).replace(" ", "")
                u.contains(targetQuantity) || t.contains(targetQuantity)
            }
            if (qtyMatches.isNotEmpty()) qtyMatches else validMatches
        } else {
            validMatches
        }

        // 6. Strict Minimum Price Selection: CANNOT select non-cheapest genuine product!
        val winningCard = candidatePool.filter { it.price > 0.0 }.minByOrNull { it.price }
            ?: candidatePool.first()

        Log.i(TAG, "WINNER: Selected CHEAPEST genuine product '${winningCard.title}' at ₹${winningCard.price} (first seen at Viewport ${winningCard.viewportSeen})")

        // 7. Align screen to winning card & tap
        val scrollsUpNeeded = maxScrolls - winningCard.viewportSeen
        if (scrollsUpNeeded > 0) {
            Log.i(TAG, "Scrolling up $scrollsUpNeeded times to return to Viewport ${winningCard.viewportSeen} for cheapest item")
            for (s in 1..scrollsUpNeeded) {
                a11y.executeAction(null, ActionType.SCROLL_UP, null)
                delay(700L)
            }
            delay(500L)
        }

        // Capture live state and locate winning card's ADD button
        var currentTargetState = a11y.captureCurrentState()
        var targetLiveCards = extractLiveProductCards(currentTargetState.nodes, expectedKeywords)
        var matchedLiveCard = targetLiveCards.firstOrNull {
            it.title.equals(winningCard.title, ignoreCase = true) ||
                (Math.abs(it.price - winningCard.price) < 1.0 && it.title.take(15).equals(winningCard.title.take(15), ignoreCase = true))
        } ?: targetLiveCards.filter { !it.isAd && it.price > 0.0 }.minByOrNull { it.price }

        var addNode = matchedLiveCard?.let { card ->
            currentTargetState.nodes.firstOrNull { it.index == card.addNodeIndex }
        }

        if (addNode == null) {
            // Secondary alignment attempt
            delay(400L)
            currentTargetState = a11y.captureCurrentState()
            targetLiveCards = extractLiveProductCards(currentTargetState.nodes, expectedKeywords)
            matchedLiveCard = targetLiveCards.firstOrNull {
                it.title.contains(coreKeywords.firstOrNull().orEmpty(), ignoreCase = true) && !it.isAd
            }
            addNode = matchedLiveCard?.let { card ->
                currentTargetState.nodes.firstOrNull { it.index == card.addNodeIndex }
            }
        }

        if (addNode == null) {
            Log.w(TAG, "Could not find ADD button for '${winningCard.title}' on current aligned screen")
            return AddToCartResult.Failed("Could not locate ADD button for ${winningCard.title}", maxScrolls + 1)
        }

        // 8. Physical Gesture Tap on ADD
        Log.i(TAG, "Executing physical TAP on ADD button for '${winningCard.title}' (index=${addNode.index})")
        val tapped = a11y.executeAction(addNode.index, ActionType.TAP, null)
        if (!tapped) {
            return AddToCartResult.Failed("Physical tap gesture failed", maxScrolls + 1)
        }
        delay(1200L)

        // 9. Handle Universal Modal / Bottom Sheet / Upsell Popup if opened
        val postTapState = a11y.captureCurrentState()
        handleUniversalModalOrSheet(a11y, postTapState, itemQuery)
        delay(800L)

        // 10. Closed-Loop Physical Cart Verification (Delta S Gate)
        var verified = false
        val originalAddY = addNode.bounds.centerY
        for (attempt in 1..4) {
            val postState = a11y.captureCurrentState()

            // Check A: Stepper transition at or near button position
            val hasStepper = postState.nodes.any { n ->
                val l = n.semanticLabel.trim().lowercase(Locale.ROOT)
                val r = n.resourceId?.lowercase(Locale.ROOT).orEmpty()
                val isStepperIndicator = l == "-" || l == "+" || l == "1" || l.contains("decrement") ||
                    l.contains("quantity") || r.contains("stepper") || r.contains("qty") || r.contains("btn_minus")
                isStepperIndicator && Math.abs(n.bounds.centerY - originalAddY) < 180
            }

            // Check B: Cart badge / bottom pill updated
            val hasCartIndicator = postState.nodes.any { n ->
                val lbl = n.semanticLabel.trim().lowercase(Locale.ROOT)
                val res = n.resourceId?.lowercase(Locale.ROOT).orEmpty()
                lbl.contains("view cart") || lbl.contains("item in cart") || lbl.contains("items in cart") ||
                    lbl.contains("pay ₹") || lbl.contains("go to cart") || lbl.contains("checkout") ||
                    res.contains("floating-cart") || res.contains("view_cart") || res.contains("cart_strip") ||
                    res.contains("cart_button")
            }

            if (hasStepper || hasCartIndicator) {
                verified = true
                Log.i(TAG, "Cart delta verification PASS (hasStepper=$hasStepper, hasCartIndicator=$hasCartIndicator) on check $attempt")
                break
            }
            delay(500L)
        }

        return if (verified) {
            // Dynamic Live Free Delivery Sniffer (Zero hardcoded numbers!)
            var liveDeliveryTip: String? = null
            try {
                val postCartState = a11y.captureCurrentState()
                val deliveryBannerRegex = Regex("""(?i)\b(?:add|shop\s+for|worth)\s*(?:₹|rs\.?)\s*(\d+)\s*(?:more)?\s*(?:for|to get)\s*free\s*delivery\b""")
                for (node in postCartState.nodes) {
                    val lbl = node.semanticLabel
                    val match = deliveryBannerRegex.find(lbl)
                    if (match != null) {
                        val neededInr = match.groupValues[1]
                        liveDeliveryTip = "Add items worth ₹$neededInr more for FREE delivery"
                        Log.i(TAG, "Sniffed live free delivery tip: $liveDeliveryTip")
                        break
                    }
                }
            } catch (_: Exception) {}

            AddToCartResult.Success(
                productTitle = winningCard.title,
                price = winningCard.price,
                priceDisplay = "₹${winningCard.price.toInt()}",
                unit = winningCard.unit,
                viewsExplored = maxScrolls + 1,
                freeDeliveryPrompt = liveDeliveryTip
            )
        } else {
            Log.w(TAG, "Cart delta not verified after tap on '${winningCard.title}'")
            AddToCartResult.Failed("Cart delta could not be verified", maxScrolls + 1)
        }
    }

    private suspend fun handleUniversalModalOrSheet(
        a11y: KarvishAccessibilityService,
        state: DistilledUiState,
        itemQuery: String
    ): Boolean {
        val queryLower = itemQuery.lowercase(Locale.ROOT)
        val targetQuantity = Regex("""(?i)\b(\d+\s*(?:g|kg|gm|ml|l|pc))\b""")
            .find(queryLower)?.groupValues?.get(1)?.replace(" ", "")

        // 1. Check for Upsell / Promotional / Location / Coupon Popups ("People also bought", "Coupon", "Scratch card", "Enable location")
        val isUpsell = state.nodes.any { n ->
            val l = n.semanticLabel.lowercase(Locale.ROOT)
            l.contains("frequently bought") || l.contains("people also bought") ||
                l.contains("recommended") || l.contains("add-on") || l.contains("save more") ||
                l.contains("pair with") || l.contains("delivery tip") || l.contains("replace") ||
                l.contains("substitute") || l.contains("you won") || l.contains("coupon") ||
                l.contains("scratch card") || l.contains("rate us") || l.contains("feedback") ||
                l.contains("enable location")
        }

        if (isUpsell) {
            val skipBtn = state.nodes.firstOrNull { n ->
                val l = n.semanticLabel.trim().lowercase(Locale.ROOT)
                val r = n.resourceId?.lowercase(Locale.ROOT).orEmpty()
                l == "skip" || l == "no thanks" || l == "not now" || l == "dismiss" ||
                    l == "continue" || l == "close" || l == "cancel" || l == "\ue921" || l == "x" ||
                    l == "maybe later" || l == "remind later" || l == "got it" || l == "ok, got it" ||
                    r.contains("close") || r.contains("btn_close") || r.contains("skip")
            }
            if (skipBtn != null) {
                Log.i(TAG, "Dismissing popup via '${skipBtn.semanticLabel}' at index=${skipBtn.index}")
                a11y.executeAction(skipBtn.index, ActionType.TAP, null)
                delay(700L)
                return true
            } else {
                Log.i(TAG, "Dismissing popup via GLOBAL_ACTION_BACK")
                a11y.performGlobalAction(android.accessibilityservice.AccessibilityService.GLOBAL_ACTION_BACK)
                delay(700L)
                return true
            }
        }

        // 2. Check for Weight / Pack Size Bottom Sheet (e.g. Potato 1kg / 500g, Besan 500g / 1kg)
        val closeBtn = state.nodes.firstOrNull { n ->
            val lbl = n.semanticLabel.trim()
            val res = n.resourceId?.lowercase(Locale.ROOT).orEmpty()
            val isCloseSymbol = lbl == "\ue921" || lbl.contains("close", ignoreCase = true) ||
                lbl.equals("x", ignoreCase = true) || res.contains("outer_icon") || res.contains("btn_close") ||
                res.contains("close") || res.contains("iconcross") || res.contains("cross")
            isCloseSymbol && n.bounds.top in 150..800 && (n.isClickable || n.role in setOf("BUTTON", "VIEW", "IMAGE"))
        }

        val hasVariantText = state.nodes.any { n ->
            val l = n.semanticLabel.lowercase(Locale.ROOT)
            l.contains("pack of") || l.contains("packs") || l.contains("select size") ||
                l.contains("choose pack") || l.contains("options") || l.contains("pouch") ||
                l.contains("save extra") || l.contains("variants") || l.contains("select weight") ||
                l.contains("choose weight") || l.contains("select quantity") || l.contains("customise")
        }

        if (closeBtn == null && !hasVariantText) {
            return false
        }

        val sheetNodes = state.nodes.filter { it.bounds.top > 350 }

        // Step 2A: If user requested a specific unit (e.g. 1kg or 500g), select that option chip/card first!
        if (targetQuantity != null) {
            val matchingOptionNode = sheetNodes.firstOrNull { n ->
                val l = n.semanticLabel.lowercase(Locale.ROOT).replace(" ", "")
                l.contains(targetQuantity) && (n.isClickable || n.role in setOf("BUTTON", "VIEW", "TEXT", "RADIO_BUTTON"))
            }
            if (matchingOptionNode != null) {
                Log.i(TAG, "Selecting requested variant option '$targetQuantity' at index=${matchingOptionNode.index}")
                a11y.executeAction(matchingOptionNode.index, ActionType.TAP, null)
                delay(500L)
            }
        }

        // Step 2B: Tap the primary bottom sheet action button (Done / Confirm / Add / + Add / Add to Cart)
        val freshSheetState = a11y.captureCurrentState()
        val confirmBtn = freshSheetState.nodes.filter { it.bounds.top > 400 }.firstOrNull { n ->
            val lbl = n.semanticLabel.trim().lowercase(Locale.ROOT)
            val res = n.resourceId?.lowercase(Locale.ROOT).orEmpty()
            val isConfirmLabel = lbl == "done" || lbl == "confirm" || lbl == "select" || lbl == "apply" ||
                lbl == "add" || lbl == "+ add" || lbl == "add to cart" || lbl == "add item" || lbl == "+ add to cart" ||
                lbl.contains("add to cart") || lbl.contains("add item")
            val isConfirmRes = res.contains("btn_done") || res.contains("btn_add") || res.contains("cta") || res.contains("confirm")
            (isConfirmLabel || isConfirmRes) && (n.isClickable || n.role in setOf("BUTTON", "VIEW"))
        }

        if (confirmBtn != null) {
            Log.i(TAG, "Tapping confirmation CTA '${confirmBtn.semanticLabel}' on sheet at index=${confirmBtn.index}")
            a11y.executeAction(confirmBtn.index, ActionType.TAP, null)
            delay(800L)
        }

        // Step 2C: Dismiss sheet via close button if still visible
        val postCheckState = a11y.captureCurrentState()
        val remainingClose = postCheckState.nodes.firstOrNull { n ->
            val lbl = n.semanticLabel.trim()
            val res = n.resourceId?.lowercase(Locale.ROOT).orEmpty()
            val isCloseSymbol = lbl == "\ue921" || lbl.contains("close", ignoreCase = true) ||
                lbl.equals("x", ignoreCase = true) || res.contains("outer_icon") || res.contains("btn_close") ||
                res.contains("close") || res.contains("iconcross")
            isCloseSymbol && n.bounds.top in 150..800
        }
        if (remainingClose != null) {
            Log.i(TAG, "Dismissing sheet via remaining close button at index=${remainingClose.index}")
            a11y.executeAction(remainingClose.index, ActionType.TAP, null)
            delay(500L)
        }

        return true
    }

    private fun extractLiveProductCards(
        nodes: List<DistilledNode>,
        expectedKeywords: List<String>
    ): List<LiveProductCard> {
        val nonCartExcludeWords = setOf("address", "tip", "view cart", "go to cart", "checkout", "coupon", "promo", "recent searches", "trending")
        val priceRegex = Regex("""(?:₹|Rs\.?|\$)\s*([\d,]+(?:\.\d+)?)""")
        val unitRegex = Regex("""(?i)\b(\d+(?:\.\d+)?\s*(?:g|kg|gm|ml|l|ltr|pc|pcs|pack))\b""")
        val availableRegex = Regex("""^(.*?)\s+is available for\s+₹?([\d,]+(?:\.\d+)?)\s*,?\s*(.*)$""", RegexOption.IGNORE_CASE)

        val cards = mutableListOf<LiveProductCard>()

        // 1. Container-based extraction (Blinkit & Zepto self-contained product containers)
        for (cNode in nodes) {
            val label = cNode.semanticLabel
            val res = cNode.resourceId?.lowercase(Locale.ROOT).orEmpty()

            // 1A. Blinkit format: "X is available for ₹Y, Z"
            val bMatch = availableRegex.find(label)
            if (bMatch != null) {
                // Reject full-screen root containers that span across columns (single card width is 200..550)
                if (cNode.bounds.width > 550) continue

                val rawGroup = bMatch.groupValues[1].trim()
                val lines = rawGroup.lines().map { it.trim() }.filter { it.isNotBlank() }
                val rawTitle = lines.lastOrNull { line ->
                    !line.contains("showing results", ignoreCase = true) &&
                        !line.contains("results for", ignoreCase = true) &&
                        !line.contains("view cart", ignoreCase = true) &&
                        !line.contains("items", ignoreCase = true) &&
                        !line.equals("ad", ignoreCase = true)
                } ?: rawGroup
                val price = bMatch.groupValues[2].replace(",", "").toDoubleOrNull() ?: 0.0
                val unit = bMatch.groupValues[3].trim().ifBlank { null }

                val addBtn = nodes.firstOrNull { n ->
                    val l = n.semanticLabel.trim().lowercase(Locale.ROOT)
                    val isAdd = l == "add" || l == "+" || l == "+ add"
                    isAdd && n.bounds.top >= cNode.bounds.top - 20 &&
                        n.bounds.bottom <= cNode.bounds.bottom + 20 &&
                        n.bounds.left >= cNode.bounds.left - 20 &&
                        n.bounds.right <= cNode.bounds.right + 20
                }

                if (addBtn != null && rawTitle.isNotBlank()) {
                    val hasKeyword = expectedKeywords.any { kw -> rawTitle.contains(kw, ignoreCase = true) }
                    cards.add(
                        LiveProductCard(
                            title = rawTitle,
                            price = price,
                            unit = unit,
                            isAd = rawTitle.startsWith("ad", ignoreCase = true) || label.contains("sponsored", ignoreCase = true) || label.startsWith("ad ", ignoreCase = true),
                            addNodeIndex = addBtn.index,
                            topBound = addBtn.bounds.top,
                            hasKeywordMatch = hasKeyword
                        )
                    )
                }
                continue
            }

            // 1B. Zepto format: "₹MRP, ₹OFF, Product Title, Unit, ₹SellingPrice"
            val isZeptoContainer = res.contains("product-card-container") ||
                (label.contains("₹") && label.contains(",") && (label.contains("off", ignoreCase = true) || label.contains("₹\u200a")))
            if (isZeptoContainer) {
                val prices = priceRegex.findAll(label)
                    .mapNotNull { it.groupValues[1].replace(",", "").toDoubleOrNull() }
                    .toList()
                val sellingPrice = prices.lastOrNull() ?: 0.0
                val unit = unitRegex.find(label)?.groupValues?.get(1)?.lowercase(Locale.ROOT)

                val parts = label.split(",").map { it.trim() }
                val titlePart = parts.firstOrNull { part ->
                    !part.contains("₹") &&
                        !part.contains("off", ignoreCase = true) &&
                        !part.contains("add to cart", ignoreCase = true) &&
                        !unitRegex.containsMatchIn(part) &&
                        part.length >= 3
                }
                val rawTitle = titlePart.orEmpty()

                val addBtn = nodes.firstOrNull { n ->
                    val l = n.semanticLabel.trim().lowercase(Locale.ROOT)
                    val r = n.resourceId?.lowercase(Locale.ROOT).orEmpty()
                    val isAdd = l == "add" || l == "+" || l == "+ add" || l == "add to cart" || r.contains("product-card-add-button")
                    isAdd && n.bounds.top >= cNode.bounds.top - 20 &&
                        n.bounds.bottom <= cNode.bounds.bottom + 20 &&
                        n.bounds.left >= cNode.bounds.left - 20 &&
                        n.bounds.right <= cNode.bounds.right + 20
                }

                if (addBtn != null && rawTitle.isNotBlank() && sellingPrice > 0.0) {
                    val hasKeyword = expectedKeywords.any { kw -> rawTitle.contains(kw, ignoreCase = true) }
                    cards.add(
                        LiveProductCard(
                            title = rawTitle,
                            price = sellingPrice,
                            unit = unit,
                            isAd = label.contains("sponsored", ignoreCase = true),
                            addNodeIndex = addBtn.index,
                            topBound = addBtn.bounds.top,
                            hasKeywordMatch = hasKeyword
                        )
                    )
                }
            }
        }

        // 2. Spatial Grid grouping (for apps without container descriptions, works for 1, 2, or 3 columns)
        val addButtons = nodes.filter { node ->
            val label = node.semanticLabel.trim().lowercase(Locale.ROOT)
            val res = node.resourceId?.lowercase(Locale.ROOT).orEmpty()
            val isAddLabel = label == "add" || label == "+" || label == "+ add" ||
                label == "add to cart" || label == "add item" || label.contains("add to cart") ||
                label.contains("add to basket") || label == "buy now"
            val isAddRes = res.contains("add_to_cart") || res.contains("btn_add") ||
                res.contains("add_button") || res.contains("additem")
            val isActionable = node.isClickable || node.role in setOf("BUTTON", "VIEW", "TEXT", "IMAGE")
            val isExcluded = nonCartExcludeWords.any { label.contains(it) }

            // Do not duplicate buttons already found in container pass
            val alreadyMapped = cards.any { it.addNodeIndex == node.index }

            (isAddLabel || isAddRes) && isActionable && !isExcluded && !alreadyMapped
        }

        val excludeTitleKeywords = setOf(
            "decrease quantity", "increase quantity", "quantity", "mins", "lowest in",
            "free delivery", "% off", "mrp", "save extra", "per pack", "options",
            "view cart", "rating", "review", "trusted organic", "combo", "sponsored", "ad",
            "showing results", "results for", "search for", "deliver to", "items in cart",
            "cart", "coupon", "explore", "frequently bought"
        )

        for (btn in addButtons) {
            val cellNodes = nodes.filter {
                val hDist = Math.abs(it.bounds.centerX - btn.bounds.centerX)
                val vDist = Math.abs(it.bounds.centerY - btn.bounds.centerY)
                hDist < 200 && vDist < 300 && it.index != btn.index && it.bounds.top > 80 && it.bounds.bottom < 980
            }

            var cardPrice = 0.0
            for (cn in cellNodes) {
                // Strictly exclude unit rates (e.g. ₹11.7/100g, ₹26/pack)
                val isUnitRate = cn.semanticLabel.contains(Regex("""(?i)(?:/|per)\s*(?:pack|100\s*g|g|kg|pc|unit)"""))
                if (!isUnitRate) {
                    val pMatch = priceRegex.find(cn.semanticLabel)
                    if (pMatch != null) {
                        val pVal = pMatch.groupValues[1].replace(",", "").toDoubleOrNull() ?: 0.0
                        if (pVal >= 12.0) {
                            cardPrice = pVal
                            break
                        }
                    }
                }
            }

            var cardUnit: String? = null
            for (cn in cellNodes) {
                val uMatch = unitRegex.find(cn.semanticLabel)
                if (uMatch != null) {
                    cardUnit = uMatch.groupValues[1].lowercase(Locale.ROOT)
                    break
                }
            }

            val titleNode = cellNodes.filter {
                val lbl = it.semanticLabel.trim().lowercase(Locale.ROOT)
                lbl.length in 4..120 &&
                    !priceRegex.containsMatchIn(lbl) &&
                    lbl != "add" && lbl != "+" &&
                    excludeTitleKeywords.none { kw -> lbl.contains(kw) }
            }.maxByOrNull { it.semanticLabel.length }

            val rawTitle = titleNode?.semanticLabel?.trim().orEmpty()
            if (rawTitle.isBlank()) continue

            val isAd = cellNodes.any {
                val l = it.semanticLabel.trim().lowercase(Locale.ROOT)
                l == "ad" || l.contains("sponsored") || l.startsWith("ad ") || l.endsWith(" ad")
            }

            val hasKeyword = expectedKeywords.any { kw -> rawTitle.contains(kw, ignoreCase = true) }

            cards.add(
                LiveProductCard(
                    title = rawTitle,
                    price = cardPrice,
                    unit = cardUnit,
                    isAd = isAd,
                    addNodeIndex = btn.index,
                    topBound = btn.bounds.top,
                    hasKeywordMatch = hasKeyword
                )
            )
        }

        return cards
    }

    private fun findWinningAddToCartNode(
        nodes: List<DistilledNode>,
        itemQuery: String,
        expectedKeywords: List<String>
    ): DistilledNode? {
        val cards = extractLiveProductCards(nodes, expectedKeywords)
        val queryLower = itemQuery.lowercase(Locale.ROOT)

        val isGrocery = WebProductFinder.isGroceryQuery(itemQuery)
        val groceryNegativeDistractors = setOf(
            "chips", "wafers", "wafer", "katori", "namkeen", "pakoda", "ketchup",
            "sauce", "seeds", "powder", "soup", "puree", "paste", "flavour", "flavoured",
            "canned", "tin", "seed", "oil", "juice", "chutney"
        )
        val electronicsNegativeDistractors = setOf(
            "case", "cover", "pouch", "skin", "adapter", "cable", "cleaner", "box", "dummy", "strap"
        )
        val rawDistractors = if (isGrocery) groceryNegativeDistractors else electronicsNegativeDistractors

        // DYNAMIC DISTRACTOR FILTER: Never filter out a word that the user explicitly asked for!
        val activeDistractors = rawDistractors.filter { distractor ->
            !queryLower.contains(distractor)
        }

        val coreKeywords = expectedKeywords.filter { kw ->
            kw.length >= 3 && kw !in setOf("item", "fresh", "best", "good", "order", "buy", "gram", "kilo")
        }

        // Strict filtering: Ad exclusion, dynamic distractor exclusion, and keyword match in PRODUCT TITLE
        val validCards = cards.filter { card ->
            val titleLower = card.title.lowercase(Locale.ROOT)
            val notAd = !card.isAd && !titleLower.contains("sponsored") && !titleLower.startsWith("ad ")
            val noDistractors = activeDistractors.none { titleLower.contains(it) }
            val matchesKeyword = if (coreKeywords.isNotEmpty()) {
                coreKeywords.any { titleLower.contains(it) }
            } else true

            notAd && noDistractors && matchesKeyword
        }

        // CRITICAL: If the screen has not loaded any cards matching this query's keywords yet,
        // NEVER fall back to picking random cards from the previous search! Return null to let loop poll.
        if (validCards.isEmpty()) {
            if (coreKeywords.isNotEmpty()) {
                Log.d(TAG, "Waiting for screen to render products matching $coreKeywords...")
                return null
            }
        }

        val targetQuantityRegex = Regex("""(?i)\b(\d+\s*(?:g|kg|gm|ml|l|pc))\b""")
        val targetQuantity = targetQuantityRegex.find(queryLower)?.groupValues?.get(1)?.lowercase(Locale.ROOT)?.replace(" ", "")

        val pool = if (targetQuantity != null) {
            val qtyMatches = validCards.filter {
                val u = it.unit?.replace(" ", "").orEmpty()
                val t = it.title.lowercase(Locale.ROOT).replace(" ", "")
                u.contains(targetQuantity) || t.contains(targetQuantity)
            }
            if (qtyMatches.isNotEmpty()) qtyMatches else validCards
        } else {
            validCards
        }

        // STRICT PRICE COMPARISON: Pick the CHEAPEST genuine matching item
        val winningCard = pool.filter { it.price > 0.0 }.minByOrNull { it.price }
            ?: pool.firstOrNull()

        if (winningCard != null) {
            Log.i(TAG, "Selected CHEAPEST & BEST product card: '${winningCard.title}' price=₹${winningCard.price} unit=${winningCard.unit}")
            return nodes.firstOrNull { it.index == winningCard.addNodeIndex }
        }

        // Only fall back to Product Detail Page Add to Cart if the page title explicitly matches keywords!
        if (coreKeywords.isNotEmpty()) {
            val pdpTitleNode = nodes.firstOrNull { n ->
                val l = n.semanticLabel.lowercase(Locale.ROOT)
                coreKeywords.any { kw -> l.contains(kw) } && n.bounds.top < 600
            }
            if (pdpTitleNode != null) {
                return nodes.firstOrNull { node ->
                    val label = node.semanticLabel.trim().lowercase(Locale.ROOT)
                    (label == "add to cart" || label == "add to basket" || label == "buy now" || label == "add") &&
                        node.bounds.top > 700 && (node.isClickable || node.role == "BUTTON")
                }
            }
        }

        return null
    }
}
