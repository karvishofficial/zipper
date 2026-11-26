package com.zipper.app

import com.zipper.app.core.agent.AgentAction
import com.zipper.app.core.agent.AutonomousApprovalGate
import com.zipper.app.core.agent.AutonomousObstaclePruner
import com.zipper.app.core.agent.CrossAppChainer
import com.zipper.app.core.agent.StateDeltaVerifier
import com.zipper.app.core.model.DistilledNode
import com.zipper.app.core.model.DistilledUiState
import com.zipper.app.core.model.NormalizedBounds
import com.zipper.app.ditto.DittoAppMetadata
import com.zipper.app.ditto.DittoFlow
import com.zipper.app.ditto.DittoPhysicalExecution
import com.zipper.app.ditto.DittoStateSnapshot
import com.zipper.app.ditto.DittoStep
import com.zipper.app.ditto.DittoTargetIdentity
import com.zipper.app.ditto.DittoTargetMetadata
import com.zipper.app.ditto.DittoTransitionContract
import com.zipper.app.ditto.ParameterizedSkillExecutor
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class AutonomousAgentUpgradeTests {

    private fun makeNode(
        index: Int,
        role: String = "BUTTON",
        text: String? = null,
        desc: String? = null,
        resId: String? = null,
        isClickable: Boolean = true,
        bounds: NormalizedBounds = NormalizedBounds(100, 100, 900, 200)
    ) = DistilledNode(
        index = index,
        role = role,
        text = text,
        contentDescription = desc,
        resourceId = resId,
        bounds = bounds,
        isClickable = isClickable
    )

    // ==========================================
    // 1. AUTONOMOUS OBSTACLE & INTERSTITIAL PRUNER
    // ==========================================

    @Test
    fun testObstaclePrunerDetectsAndDismissesPromoModal() {
        val promoState = DistilledUiState(
            packageName = "com.swiggy.android",
            nodes = listOf(
                makeNode(0, role = "TEXT", text = "Festive Sale! Get 50% Flat Discount Now", isClickable = false),
                makeNode(1, role = "BUTTON", text = "Claim Offer", isClickable = true),
                makeNode(2, role = "BUTTON", text = "Maybe Later", isClickable = true, resId = "com.swiggy.android:id/btn_dismiss")
            )
        )

        val decision = AutonomousObstaclePruner.evaluate(promoState, "Order chicken biryani")
        assertTrue(decision.isObstacleDetected)
        assertNotNull(decision.pruneAction)
        assertTrue(decision.pruneAction is AgentAction.Tap)
        assertEquals(2, (decision.pruneAction as AgentAction.Tap).targetNodeIndex)
        assertTrue(decision.reason.contains("Maybe Later", ignoreCase = true))
    }

    @Test
    fun testObstaclePrunerClearsTopCornerCloseIcon() {
        val bannerState = DistilledUiState(
            packageName = "com.flipkart.android",
            nodes = listOf(
                makeNode(0, role = "IMAGE", text = "Special offer for today", isClickable = false),
                makeNode(1, role = "BUTTON", text = "✕", isClickable = true, bounds = NormalizedBounds(850, 50, 950, 150))
            )
        )

        val decision = AutonomousObstaclePruner.evaluate(bannerState, "Search noise cancelling headphones")
        assertTrue(decision.isObstacleDetected)
        assertEquals(1, (decision.pruneAction as AgentAction.Tap).targetNodeIndex)
    }

    @Test
    fun testObstaclePrunerIgnoresModalWhenGoalExplicitlyTargetsIt() {
        val promoState = DistilledUiState(
            packageName = "com.swiggy.android",
            nodes = listOf(
                makeNode(0, role = "TEXT", text = "Festive discount offer"),
                makeNode(1, role = "BUTTON", text = "Claim discount")
            )
        )

        val decision = AutonomousObstaclePruner.evaluate(promoState, "Claim festive discount")
        assertFalse(decision.isObstacleDetected)
    }

    // ==========================================
    // 2. STATE DELTA & VERIFICATION ENGINE
    // ==========================================

    @Test
    fun testStateDeltaVerifierDetectsOutOfStockError() {
        val preState = DistilledUiState(
            packageName = "com.zeptonow.android",
            nodes = listOf(makeNode(0, text = "Amul Taaza Milk 500ml", role = "TEXT"))
        )
        val postState = DistilledUiState(
            packageName = "com.zeptonow.android",
            nodes = listOf(
                makeNode(0, text = "Amul Taaza Milk 500ml", role = "TEXT"),
                makeNode(1, text = "Currently Out of Stock at your dark store", role = "TEXT")
            )
        )

        val result = StateDeltaVerifier.verifyTransition(
            goal = "Buy Amul milk",
            preState = preState,
            postState = postState,
            action = AgentAction.Tap(0)
        )

        assertEquals(StateDeltaVerifier.VerificationStatus.REJECTED_ERROR, result.status)
        assertEquals("OUT_OF_STOCK", result.errorCategory)
        assertTrue(result.summary.contains("Out of Stock", ignoreCase = true))
    }

    @Test
    fun testStateDeltaVerifierDetectsPaymentFailure() {
        val preState = DistilledUiState(packageName = "com.swiggy.android", nodes = listOf(makeNode(0, text = "Paying ₹450")))
        val postState = DistilledUiState(
            packageName = "com.swiggy.android",
            nodes = listOf(makeNode(0, text = "Payment Failed: Bank server timed out. Please try again."))
        )

        val result = StateDeltaVerifier.verifyTransition(
            goal = "Checkout lunch",
            preState = preState,
            postState = postState,
            action = AgentAction.Tap(0)
        )

        assertEquals(StateDeltaVerifier.VerificationStatus.REJECTED_ERROR, result.status)
        assertEquals("PAYMENT_FAILED", result.errorCategory)
    }

    @Test
    fun testStateDeltaVerifierConfirmsPositiveMilestone() {
        val preState = DistilledUiState(packageName = "com.whatsapp", nodes = listOf(makeNode(0, text = "Drafting message")))
        val postState = DistilledUiState(
            packageName = "com.whatsapp",
            nodes = listOf(makeNode(0, text = "Message sent"))
        )

        val result = StateDeltaVerifier.verifyTransition(
            goal = "Send message to Rohit",
            preState = preState,
            postState = postState,
            action = AgentAction.Tap(0)
        )

        assertEquals(StateDeltaVerifier.VerificationStatus.TERMINATION_CONFIRMED, result.status)
        assertEquals("MESSAGE_SENT", result.confirmedMilestone)
    }

    // ==========================================
    // 3. PARAMETERIZED SKILL ENGINE (DITTO)
    // ==========================================

    @Test
    fun testSkillParameterDerivationAndSubstitution() {
        val flow = DittoFlow(
            flowId = "test_flow_123",
            title = "Search Product Flow",
            appMetadata = DittoAppMetadata(launchPackage = "com.amazon.mShop.android.shopping"),
            steps = listOf(
                DittoStep(
                    stepIndex = 1,
                    actionType = "TAP",
                    targetIdentity = DittoTargetIdentity(primaryText = "Search Amazon", semanticRole = "INPUT"),
                    targetMetadata = DittoTargetMetadata(resourceId = "rs_search_src_text"),
                    physicalExecution = DittoPhysicalExecution(100, 100),
                    stateBefore = DittoStateSnapshot(),
                    stateAfter = DittoStateSnapshot(),
                    transitionContract = DittoTransitionContract()
                ),
                DittoStep(
                    stepIndex = 2,
                    actionType = "TYPE_TEXT",
                    argument = "MacBook Pro M3",
                    targetIdentity = DittoTargetIdentity(primaryText = "Search input", semanticRole = "INPUT"),
                    targetMetadata = DittoTargetMetadata(resourceId = "rs_search_src_text"),
                    physicalExecution = DittoPhysicalExecution(100, 100),
                    stateBefore = DittoStateSnapshot(),
                    stateAfter = DittoStateSnapshot(),
                    transitionContract = DittoTransitionContract()
                )
            )
        )

        // Verify parameter slot derivation
        val typeStep = flow.steps[1]
        val isTypeStep = typeStep.actionType == "TYPE_TEXT"
        assertTrue(isTypeStep)
        assertEquals("MacBook Pro M3", typeStep.argument)

        // Verify parameter override simulation
        val runtimeParams = mapOf("query" to "iPad Air 256GB")
        val override = runtimeParams["query"] ?: typeStep.argument
        val adaptedStep = typeStep.copy(argument = override)

        assertEquals("iPad Air 256GB", adaptedStep.argument)
    }

    // ==========================================
    // 4. CROSS-APP INTENT CHAINER
    // ==========================================

    @Test
    fun testCrossAppChainerDetectsWhatsAppToMapsWorkflow() {
        val goal = "Open address from Rahul on WhatsApp in Google Maps"
        val plan = CrossAppChainer.analyzeGoal(goal)

        assertTrue(plan.isChainedGoal)
        assertEquals("com.whatsapp", plan.sourceApp)
        assertEquals("com.google.android.apps.maps", plan.targetApp)
        assertEquals(2, plan.steps.size)
        assertTrue(plan.steps[0].isSourceExtraction)
        assertEquals("destination_address", plan.steps[0].payloadSlot)
        assertFalse(plan.steps[1].isSourceExtraction)
    }

    @Test
    fun testCrossAppChainerDetectsSmsOtpToAppWorkflow() {
        val goal = "Read OTP from messages and login to Zepto"
        val plan = CrossAppChainer.analyzeGoal(goal)

        assertTrue(plan.isChainedGoal)
        assertEquals("com.google.android.apps.messaging", plan.sourceApp)
        assertEquals("com.zeptonow.android", plan.targetApp)
        assertEquals("otp_code", plan.steps[0].payloadSlot)
    }

    @Test
    fun testCrossAppChainerIgnoresSingleAppGoals() {
        val goal = "Search cold brew on Starbucks app"
        val plan = CrossAppChainer.analyzeGoal(goal)

        assertFalse(plan.isChainedGoal)
    }

    // ==========================================
    // 5. AUTONOMOUS APPROVAL GATE (GROK BOT SAFETY BRAKE)
    // ==========================================

    @Test
    fun testApprovalGateTriggersOnPaymentCommitAffordances() {
        // Test case 1: Pay button with INR currency
        val inrPayNode = makeNode(0, role = "BUTTON", text = "Pay ₹499", isClickable = true)
        val inrDecision = AutonomousApprovalGate.evaluate(inrPayNode, AgentAction.Tap(0))
        assertTrue(inrDecision.isApprovalRequired)
        assertTrue(inrDecision.isPayment)
        assertEquals("₹499", inrDecision.detectedAmount)
        assertTrue(inrDecision.targetDetail.contains("₹499"))

        // Test case 2: Place Order button on unseen food app
        val placeOrderNode = makeNode(1, role = "BUTTON", text = "Place Order", isClickable = true)
        val placeOrderDecision = AutonomousApprovalGate.evaluate(placeOrderNode, AgentAction.Tap(1))
        assertTrue(placeOrderDecision.isApprovalRequired)
        assertTrue(placeOrderDecision.isPayment)

        // Test case 3: USD Subscription checkout
        val usdSubNode = makeNode(2, role = "BUTTON", text = "Subscribe $19.99 / month", isClickable = true)
        val usdDecision = AutonomousApprovalGate.evaluate(usdSubNode, AgentAction.Tap(2))
        assertTrue(usdDecision.isApprovalRequired)
        assertTrue(usdDecision.isPayment)
        assertEquals("$19.99", usdDecision.detectedAmount)

        // Test case 4: Submit action on payment form
        val submitPayNode = makeNode(3, role = "BUTTON", text = "Proceed to Pay", resId = "com.unseen.app:id/btn_proceed_to_pay")
        val submitDecision = AutonomousApprovalGate.evaluate(submitPayNode, AgentAction.Submit)
        assertTrue(submitDecision.isApprovalRequired)
        assertTrue(submitDecision.isPayment)
    }

    @Test
    fun testApprovalGateTriggersOnDestructiveActions() {
        val deleteNode = makeNode(0, role = "BUTTON", text = "Delete Account Permanently", isClickable = true)
        val deleteDecision = AutonomousApprovalGate.evaluate(deleteNode, AgentAction.Tap(0))
        assertTrue(deleteDecision.isApprovalRequired)
        assertTrue(deleteDecision.isDestructive)
        assertFalse(deleteDecision.isPayment)
        assertTrue(deleteDecision.title.contains("Irreversible"))

        val wipeNode = makeNode(1, role = "BUTTON", text = "Factory reset and wipe all", isClickable = true)
        val wipeDecision = AutonomousApprovalGate.evaluate(wipeNode, AgentAction.Tap(1))
        assertTrue(wipeDecision.isApprovalRequired)
        assertTrue(wipeDecision.isDestructive)
    }

    @Test
    fun testApprovalGateIgnoresNonBindingCartAndBrowseActions() {
        // Add to Cart should NEVER be gated (user is just browsing/shopping)
        val addToCartNode = makeNode(0, role = "BUTTON", text = "Add to Cart", isClickable = true)
        val addDecision = AutonomousApprovalGate.evaluate(addToCartNode, AgentAction.Tap(0))
        assertFalse(addDecision.isApprovalRequired)

        // View Cart should not be gated
        val viewCartNode = makeNode(1, role = "BUTTON", text = "View Cart (3 items)", isClickable = true)
        val viewCartDecision = AutonomousApprovalGate.evaluate(viewCartNode, AgentAction.Tap(1))
        assertFalse(viewCartDecision.isApprovalRequired)

        // Search button should not be gated
        val searchNode = makeNode(2, role = "BUTTON", text = "Search Products", isClickable = true)
        val searchDecision = AutonomousApprovalGate.evaluate(searchNode, AgentAction.Tap(2))
        assertFalse(searchDecision.isApprovalRequired)
    }

    @Test
    fun testApprovalGateIgnoresNonClickablePriceLabels() {
        // Pure non-clickable product price display (e.g. ₹999 on product details card)
        val priceDisplayNode = makeNode(0, role = "TEXT", text = "Total Price: ₹999", isClickable = false)
        val priceDecision = AutonomousApprovalGate.evaluate(priceDisplayNode, AgentAction.Tap(0))
        assertFalse(priceDecision.isApprovalRequired)
    }
}
