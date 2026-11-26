package com.zipper.app

import com.zipper.app.core.agent.AgentAction
import com.zipper.app.core.agent.ComparativeMode
import com.zipper.app.core.agent.ProductCardParser
import com.zipper.app.core.agent.ProductExplorationEngine
import com.zipper.app.core.agent.ScrollDirection
import com.zipper.app.core.agent.UnitType
import com.zipper.app.core.model.DistilledNode
import com.zipper.app.core.model.DistilledUiState
import com.zipper.app.core.model.NormalizedBounds
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ProductExplorationTests {

    @Test
    fun testIntentRecognitionForCheapestAndBestOffer() {
        val intent1 = ProductExplorationEngine.detectComparativeIntent("Cheapest tomatoes on Zepto")
        assertNotNull(intent1)
        assertEquals(ComparativeMode.CHEAPEST, intent1?.mode)
        assertEquals("tomato", intent1?.targetEntity)
        assertEquals("zepto", intent1?.targetApp)

        val intent2 = ProductExplorationEngine.detectComparativeIntent("find the cheapest milk in Blinkit")
        assertNotNull(intent2)
        assertEquals(ComparativeMode.CHEAPEST, intent2?.mode)
        assertEquals("milk", intent2?.targetEntity)
        assertEquals("blinkit", intent2?.targetApp)

        val intent3 = ProductExplorationEngine.detectComparativeIntent("best offer earphones on Amazon")
        assertNotNull(intent3)
        assertEquals(ComparativeMode.BEST_OFFER, intent3?.mode)
        assertEquals("earphone", intent3?.targetEntity)
        assertEquals("amazon", intent3?.targetApp)

        // Non-comparative goals should return null
        assertNull(ProductExplorationEngine.detectComparativeIntent("Search SpaceX on YouTube"))
        assertNull(ProductExplorationEngine.detectComparativeIntent("Set alarm for 7:00 AM"))
        assertNull(ProductExplorationEngine.detectComparativeIntent("Send hello to Mom"))
    }

    @Test
    fun testUnitPriceCalculationAndDistractorSuppression() {
        // Create simulated Zepto tomato search result nodes
        val nodes = listOf(
            // Product 1: Hybrid Tomato 1kg @ ₹36 (Unit price ₹36/kg)
            DistilledNode(
                index = 1,
                role = "TEXT",
                text = "Fresh Hybrid Tomato 1 kg",
                bounds = NormalizedBounds(50, 100, 450, 150)
            ),
            DistilledNode(
                index = 2,
                role = "TEXT",
                text = "₹ 36",
                bounds = NormalizedBounds(50, 160, 200, 200)
            ),
            DistilledNode(
                index = 3,
                role = "BUTTON",
                text = "ADD",
                bounds = NormalizedBounds(300, 160, 450, 210),
                isClickable = true
            ),

            // Product 2: Local Tomato 500g @ ₹22 (Unit price ₹44/kg)
            DistilledNode(
                index = 4,
                role = "TEXT",
                text = "Local Tomato 500 g",
                bounds = NormalizedBounds(550, 100, 950, 150)
            ),
            DistilledNode(
                index = 5,
                role = "TEXT",
                text = "₹ 22",
                bounds = NormalizedBounds(550, 160, 700, 200)
            ),
            DistilledNode(
                index = 6,
                role = "BUTTON",
                text = "ADD",
                bounds = NormalizedBounds(800, 160, 950, 210),
                isClickable = true
            ),

            // Distractor 1: Tomato Ketchup (should be suppressed!)
            DistilledNode(
                index = 7,
                role = "TEXT",
                text = "Kissan Fresh Tomato Ketchup 500g",
                bounds = NormalizedBounds(50, 300, 450, 350)
            ),
            DistilledNode(
                index = 8,
                role = "TEXT",
                text = "₹ 120",
                bounds = NormalizedBounds(50, 360, 200, 400)
            ),
            DistilledNode(
                index = 9,
                role = "BUTTON",
                text = "ADD",
                bounds = NormalizedBounds(300, 360, 450, 410),
                isClickable = true
            )
        )

        val state = DistilledUiState(
            packageName = "com.zepto",
            nodes = nodes
        )

        val parsedCards = ProductCardParser.parseCardsFromState(state, "tomato")
        assertEquals(2, parsedCards.size) // Ketchup distractor excluded!

        val card1kg = parsedCards.find { it.normalizedQuantity == 1.0 }
        val card500g = parsedCards.find { it.normalizedQuantity == 0.5 }

        assertNotNull(card1kg)
        assertNotNull(card500g)

        assertEquals(36.0, card1kg!!.price, 0.01)
        assertEquals(36.0, card1kg.unitPrice, 0.01) // ₹36 / 1.0kg

        assertEquals(22.0, card500g!!.price, 0.01)
        assertEquals(44.0, card500g.unitPrice, 0.01) // ₹22 / 0.5kg = ₹44/kg

        assertTrue("1kg @ ₹36 is cheaper per kg than 500g @ ₹22", card1kg.unitPrice < card500g.unitPrice)
    }

    @Test
    fun testMultiTurnScrollExplorationAndWinnerSelection() {
        val intent = ProductExplorationEngine.detectComparativeIntent("Cheapest tomatoes on Zepto")!!
        val engine = ProductExplorationEngine(intent)

        // Screen 1: Card A (500g @ ₹25 -> ₹50/kg)
        val screen1 = DistilledUiState(
            packageName = "com.zepto",
            nodes = listOf(
                DistilledNode(index = 1, role = "TEXT", text = "Tomato 500 g", bounds = NormalizedBounds(50, 100, 450, 150)),
                DistilledNode(index = 2, role = "TEXT", text = "₹ 25", bounds = NormalizedBounds(50, 160, 200, 200)),
                DistilledNode(index = 3, role = "BUTTON", text = "ADD", bounds = NormalizedBounds(300, 160, 450, 210), isClickable = true)
            )
        )

        // Turn 1 should decide to scroll DOWN to explore Screen 2 (step 1 of 2)
        val action1 = engine.evaluateNextAction(screen1)
        assertEquals(AgentAction.Scroll(direction = ScrollDirection.DOWN, candidateId = "scroll_explore_down"), action1)

        // Screen 2: Card B (1kg @ ₹36 -> ₹36/kg winner!)
        val screen2 = DistilledUiState(
            packageName = "com.zepto",
            nodes = listOf(
                DistilledNode(index = 4, role = "TEXT", text = "Hybrid Tomato 1 kg", bounds = NormalizedBounds(50, 100, 450, 150)),
                DistilledNode(index = 5, role = "TEXT", text = "₹ 36", bounds = NormalizedBounds(50, 160, 200, 200)),
                DistilledNode(index = 6, role = "BUTTON", text = "ADD", bounds = NormalizedBounds(300, 160, 450, 210), isClickable = true)
            )
        )

        // Turn 2 should scroll DOWN to explore Screen 3 (step 2 of 2)
        val action2 = engine.evaluateNextAction(screen2)
        assertEquals(AgentAction.Scroll(direction = ScrollDirection.DOWN, candidateId = "scroll_explore_down"), action2)

        // Screen 3: Card B is still visible or re-grounded
        val action3 = engine.evaluateNextAction(screen2)
        assertNotNull(action3)
        assertTrue(action3 is AgentAction.Tap)
        assertEquals(6, (action3 as AgentAction.Tap).targetNodeIndex)
        assertEquals("Hybrid Tomato 1 kg", engine.winner?.title)
        assertEquals(36.0, engine.winner?.unitPrice ?: 0.0, 0.01)
        assertTrue(engine.isWinnerActionDispatched)
    }

    @Test
    fun testSpecificQuantityExtractionAndMatching() {
        val intent = ProductExplorationEngine.detectComparativeIntent("Find the cheapest 500g tomato on Zepto")
        assertNotNull(intent)
        assertEquals(ComparativeMode.CHEAPEST, intent?.mode)
        assertEquals("tomato", intent?.targetEntity)
        assertEquals("zepto", intent?.targetApp)
        assertEquals(0.5, intent?.requestedQuantity ?: 0.0, 0.01)
        assertEquals("500g", intent?.requestedQuantityText)
        assertEquals(UnitType.WEIGHT, intent?.requestedUnitType)

        val intent1kg = ProductExplorationEngine.detectComparativeIntent("cheapest 1 kg onion")
        assertNotNull(intent1kg)
        assertEquals("onion", intent1kg?.targetEntity)
        assertEquals(1.0, intent1kg?.requestedQuantity ?: 0.0, 0.01)
        assertEquals(UnitType.WEIGHT, intent1kg?.requestedUnitType)
    }

    @Test
    fun testSpecificQuantityStrictFilteringExcludesOtherWeights() {
        val intent = ProductExplorationEngine.detectComparativeIntent("Find the cheapest 500g tomato on Zepto")!!
        val engine = ProductExplorationEngine(intent)

        // Screen with 1kg @ ₹30 (unit price ₹30/kg) and 500g @ ₹18 (unit price ₹36/kg)
        val screen = DistilledUiState(
            packageName = "com.zepto",
            nodes = listOf(
                // 1kg Tomato @ ₹30 (Lower unit price, but WRONG quantity!)
                DistilledNode(index = 1, role = "TEXT", text = "Fresh Tomato 1 kg", bounds = NormalizedBounds(50, 100, 450, 150)),
                DistilledNode(index = 2, role = "TEXT", text = "₹ 30", bounds = NormalizedBounds(50, 160, 200, 200)),
                DistilledNode(index = 3, role = "BUTTON", text = "ADD", bounds = NormalizedBounds(300, 160, 450, 210), isClickable = true),

                // 500g Tomato @ ₹18 (Requested quantity match!)
                DistilledNode(index = 4, role = "TEXT", text = "Local Tomato 500 g", bounds = NormalizedBounds(550, 100, 950, 150)),
                DistilledNode(index = 5, role = "TEXT", text = "₹ 18", bounds = NormalizedBounds(550, 160, 700, 200)),
                DistilledNode(index = 6, role = "BUTTON", text = "ADD", bounds = NormalizedBounds(800, 160, 950, 210), isClickable = true)
            )
        )

        // Turn 1 scrolls down (step 1)
        engine.evaluateNextAction(screen)
        // Turn 2 scrolls down (step 2)
        engine.evaluateNextAction(screen)

        // Turn 3 selects winner: MUST pick 500g Tomato @ ₹18 (node 6), NOT 1kg @ ₹30 (node 3)
        val action = engine.evaluateNextAction(screen)
        assertNotNull(action)
        assertTrue(action is AgentAction.Tap)
        assertEquals(6, (action as AgentAction.Tap).targetNodeIndex)
        assertEquals("Local Tomato 500 g", engine.winner?.title)
        assertEquals(18.0, engine.winner?.price ?: 0.0, 0.01)
    }

    @Test
    fun testReturnToTopWhenCheapestIsOnScreen1() {
        val intent = ProductExplorationEngine.detectComparativeIntent("Find the cheapest 500g tomato on Zepto")!!
        val engine = ProductExplorationEngine(intent)

        // Screen 1: Cheapest 500g Tomato @ ₹15
        val screen1 = DistilledUiState(
            packageName = "com.zepto",
            nodes = listOf(
                DistilledNode(index = 1, role = "TEXT", text = "Cheapest Tomato 500 g", bounds = NormalizedBounds(50, 100, 450, 150)),
                DistilledNode(index = 2, role = "TEXT", text = "₹ 15", bounds = NormalizedBounds(50, 160, 200, 200)),
                DistilledNode(index = 3, role = "BUTTON", text = "ADD", bounds = NormalizedBounds(300, 160, 450, 210), isClickable = true)
            )
        )

        // Screen 2: More expensive 500g Tomato @ ₹25
        val screen2 = DistilledUiState(
            packageName = "com.zepto",
            nodes = listOf(
                DistilledNode(index = 4, role = "TEXT", text = "Organic Tomato 500 g", bounds = NormalizedBounds(50, 100, 450, 150)),
                DistilledNode(index = 5, role = "TEXT", text = "₹ 25", bounds = NormalizedBounds(50, 160, 200, 200)),
                DistilledNode(index = 6, role = "BUTTON", text = "ADD", bounds = NormalizedBounds(300, 160, 450, 210), isClickable = true)
            )
        )

        // Screen 3: Even more expensive 500g Tomato @ ₹35
        val screen3 = DistilledUiState(
            packageName = "com.zepto",
            nodes = listOf(
                DistilledNode(index = 7, role = "TEXT", text = "Premium Tomato 500 g", bounds = NormalizedBounds(50, 100, 450, 150)),
                DistilledNode(index = 8, role = "TEXT", text = "₹ 35", bounds = NormalizedBounds(50, 160, 200, 200)),
                DistilledNode(index = 9, role = "BUTTON", text = "ADD", bounds = NormalizedBounds(300, 160, 450, 210), isClickable = true)
            )
        )

        // Turn 1 (Screen 1): Discovers Screen 1 card -> scrolls DOWN
        val action1 = engine.evaluateNextAction(screen1)
        assertEquals(AgentAction.Scroll(direction = ScrollDirection.DOWN, candidateId = "scroll_explore_down"), action1)

        // Turn 2 (Screen 2): Discovers Screen 2 card -> scrolls DOWN
        val action2 = engine.evaluateNextAction(screen2)
        assertEquals(AgentAction.Scroll(direction = ScrollDirection.DOWN, candidateId = "scroll_explore_down"), action2)

        // Turn 3 (Screen 3): Discovers Screen 3 card -> budget exhausted, selects Screen 1's card as winner!
        // But Screen 3 only has Premium Tomato (node 9). Winning card (Cheapest Tomato) is on Screen 1.
        // Engine MUST emit Scroll UP to re-ground!
        val action3 = engine.evaluateNextAction(screen3)
        assertEquals(AgentAction.Scroll(direction = ScrollDirection.UP, candidateId = "scroll_back_to_winner"), action3)

        // Turn 4 (Screen 2): Still on screen 2 -> Scroll UP again!
        val action4 = engine.evaluateNextAction(screen2)
        assertEquals(AgentAction.Scroll(direction = ScrollDirection.UP, candidateId = "scroll_back_to_winner"), action4)

        // Turn 5 (Screen 1): Now back on Screen 1 where Cheapest Tomato is visible -> Tap ADD (node #3)!
        val action5 = engine.evaluateNextAction(screen1)
        assertNotNull(action5)
        assertTrue(action5 is AgentAction.Tap)
        assertEquals(3, (action5 as AgentAction.Tap).targetNodeIndex)
        assertTrue(engine.isWinnerActionDispatched)
    }

    @Test
    fun testEngineRemainsIdleWhenNoProductCardsVisible() {
        val intent = ProductExplorationEngine.detectComparativeIntent("Find the cheapest 500g tomato on Zepto")!!
        val engine = ProductExplorationEngine(intent)

        // Home screen / Search input state (no product cards)
        val homeState = DistilledUiState(
            packageName = "com.zepto",
            nodes = listOf(
                DistilledNode(index = 1, role = "INPUT", text = "Search 'milk, curd, tomato'", bounds = NormalizedBounds(50, 50, 950, 120), isEditable = true),
                DistilledNode(index = 2, role = "BUTTON", text = "Categories", bounds = NormalizedBounds(50, 150, 300, 250))
            )
        )

        val action = engine.evaluateNextAction(homeState)
        assertNull("Engine must remain idle when no product cards are visible", action)
    }

    @Test
    fun testBestOfferSelectionWithDiscountsAndRatings() {
        val intent = ProductExplorationEngine.detectComparativeIntent("best offer earphones on Amazon")!!
        val engine = ProductExplorationEngine(intent)

        val screen = DistilledUiState(
            packageName = "in.amazon.mShop.android.shopping",
            nodes = listOf(
                // Earphone A: ₹999, 10% off, 3.5 rating
                DistilledNode(index = 1, role = "TEXT", text = "Boat Earphones Bass", bounds = NormalizedBounds(50, 100, 450, 150)),
                DistilledNode(index = 2, role = "TEXT", text = "₹ 999 10% off ★ 3.5", bounds = NormalizedBounds(50, 160, 300, 200)),
                DistilledNode(index = 3, role = "BUTTON", text = "Add to Cart", bounds = NormalizedBounds(350, 160, 450, 210), isClickable = true),

                // Earphone B: ₹799, 45% off, 4.4 rating -> Clear Winner!
                DistilledNode(index = 4, role = "TEXT", text = "Noise Earphones Pro", bounds = NormalizedBounds(550, 100, 950, 150)),
                DistilledNode(index = 5, role = "TEXT", text = "₹ 799 45% off ★ 4.4", bounds = NormalizedBounds(550, 160, 800, 200)),
                DistilledNode(index = 6, role = "BUTTON", text = "Add to Cart", bounds = NormalizedBounds(850, 160, 950, 210), isClickable = true)
            )
        )

        // Turn 1 scrolls to explore (step 1)
        engine.evaluateNextAction(screen)
        // Turn 2 scrolls to explore (step 2)
        engine.evaluateNextAction(screen)

        // Turn 3 selects winner
        val action3 = engine.evaluateNextAction(screen)
        assertNotNull(action3)
        assertTrue(action3 is AgentAction.Tap)
        assertEquals(6, (action3 as AgentAction.Tap).targetNodeIndex)
        assertEquals("Noise Earphones Pro", engine.winner?.title)
        assertEquals(45.0, engine.winner?.discountPercent ?: 0.0, 0.01)
    }
}
