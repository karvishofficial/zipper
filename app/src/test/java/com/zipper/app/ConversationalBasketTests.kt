package com.zipper.app

import com.zipper.app.core.web.BasketMutationType
import com.zipper.app.core.web.ConversationalBasketManager
import com.zipper.app.core.web.ShoppingListParser
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

class ConversationalBasketTests {

    @Before
    fun setup() {
        ConversationalBasketManager.clearBasket()
    }

    @Test
    fun testInitialBasketCreation() {
        val result = ConversationalBasketManager.processUserInput("1kg tomato, 500g paneer")
        assertEquals(BasketMutationType.NEW_BASKET, result.type)
        assertEquals(2, result.currentItems.size)
        assertEquals("1kg tomato", result.currentItems[0].normalizedQuery)
        assertEquals("500g paneer", result.currentItems[1].normalizedQuery)
    }

    @Test
    fun testQuantityModification() {
        // Step 1: Create initial basket
        ConversationalBasketManager.processUserInput("1kg tomato, 500g paneer")

        // Step 2: Modify paneer quantity to 1kg
        val modResult = ConversationalBasketManager.processUserInput("make paneer 1kg")
        assertEquals(BasketMutationType.ITEM_MODIFIED, modResult.type)
        assertEquals(2, modResult.currentItems.size)

        val updatedPaneer = modResult.currentItems.first { it.canonicalName.contains("paneer") }
        assertEquals("1kg", updatedPaneer.quantity)
        assertEquals("1kg paneer", updatedPaneer.normalizedQuery)

        // Tomato should still remain unaffected!
        val tomato = modResult.currentItems.first { it.canonicalName.contains("tomato") }
        assertEquals("1kg", tomato.quantity)
    }

    @Test
    fun testItemAddition() {
        // Step 1: Create initial basket
        ConversationalBasketManager.processUserInput("1kg tomato, 500g paneer")

        // Step 2: Append bread
        val addResult = ConversationalBasketManager.processUserInput("and add 1 packet bread")
        assertEquals(BasketMutationType.ITEMS_ADDED, addResult.type)
        assertEquals(3, addResult.currentItems.size)

        assertTrue(addResult.currentItems.any { it.canonicalName.contains("bread") })
    }

    @Test
    fun testItemRemoval() {
        // Step 1: Create 3-item basket
        ConversationalBasketManager.processUserInput("1kg tomato, 500g paneer, 1 pack bread")

        // Step 2: Remove tomato
        val removeResult = ConversationalBasketManager.processUserInput("remove tomato")
        assertEquals(BasketMutationType.ITEM_REMOVED, removeResult.type)
        assertEquals(2, removeResult.currentItems.size)
        assertTrue(removeResult.currentItems.none { it.canonicalName.contains("tomato") })
        assertTrue(removeResult.currentItems.any { it.canonicalName.contains("paneer") })
        assertTrue(removeResult.currentItems.any { it.canonicalName.contains("bread") })
    }

    @Test
    fun testStoreSwitching() {
        // Step 1: Create initial basket on Zepto
        ConversationalBasketManager.processUserInput("1kg tomato, 500g paneer on zepto")

        // Step 2: Switch to Blinkit
        val switchResult = ConversationalBasketManager.processUserInput("order on blinkit instead")
        assertEquals(BasketMutationType.STORE_SWITCHED, switchResult.type)
        assertEquals("Blinkit", switchResult.targetStore)
        assertEquals(2, switchResult.currentItems.size)
    }

    @Test
    fun testClearBasket() {
        ConversationalBasketManager.processUserInput("1kg tomato, 500g paneer")
        val clearResult = ConversationalBasketManager.processUserInput("clear basket")
        assertEquals(BasketMutationType.BASKET_CLEARED, clearResult.type)
        assertTrue(clearResult.currentItems.isEmpty())
    }

    @Test
    fun testSemanticAttributeExtraction() {
        val items = ShoppingListParser.parseShoppingList("500g organic tomato, 1L cold-pressed mustard oil, sugar-free biscuits")
        assertEquals(3, items.size)

        val organicTomato = items[0]
        assertTrue(organicTomato.hardAttributes.contains("organic"))

        val coldPressedOil = items[1]
        assertTrue(coldPressedOil.hardAttributes.any { it.contains("cold-pressed") })

        val sugarFreeBiscuits = items[2]
        assertTrue(sugarFreeBiscuits.hardAttributes.any { it.contains("sugar-free") })
    }

    @Test
    fun testBrandExtraction() {
        val items = ShoppingListParser.parseShoppingList("500g amul butter, 5kg aashirvaad atta")
        assertEquals(2, items.size)

        assertEquals("amul", items[0].brand)
        assertEquals("aashirvaad", items[1].brand)
    }
}
