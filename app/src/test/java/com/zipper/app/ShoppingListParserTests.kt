package com.zipper.app

import com.zipper.app.core.web.BasketOptimizer
import com.zipper.app.core.web.ItemCategory
import com.zipper.app.core.web.ShoppingListParser
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ShoppingListParserTests {

    @Test
    fun testColloquialWhatsAppMessageParsing() {
        val rawMessage = "aate waqt 1kg aloo, 500g paneer aur Amul doodh le aana"
        val items = ShoppingListParser.parseShoppingList(rawMessage)

        assertEquals(3, items.size)
        // 1kg aloo -> 1kg potato
        assertTrue(items[0].normalizedQuery.contains("potato", ignoreCase = true))
        assertTrue(items[0].normalizedQuery.contains("1kg"))
        assertEquals(ItemCategory.FRESH, items[0].category)

        // 500g paneer
        assertTrue(items[1].normalizedQuery.contains("paneer", ignoreCase = true))
        assertTrue(items[1].normalizedQuery.contains("500g"))
        assertEquals(ItemCategory.FRESH, items[1].category)

        // Amul doodh -> milk
        assertTrue(items[2].normalizedQuery.contains("milk", ignoreCase = true))
        assertEquals(ItemCategory.FRESH, items[2].category)
    }

    @Test
    fun testIndianColloquialQuantities() {
        val input = "aadha kilo tamatar, pav kilo mirchi, 2kg chawal"
        val items = ShoppingListParser.parseShoppingList(input)

        assertEquals(3, items.size)
        // aadha kilo -> 500g tomato
        assertTrue(items[0].normalizedQuery.contains("500g"))
        assertTrue(items[0].normalizedQuery.contains("tomato", ignoreCase = true))

        // pav kilo -> 250g green chilli
        assertTrue(items[1].normalizedQuery.contains("250g"))
        assertTrue(items[1].normalizedQuery.contains("green chilli", ignoreCase = true))

        // 2kg chawal -> 2kg rice
        assertTrue(items[2].normalizedQuery.contains("2kg"))
        assertTrue(items[2].normalizedQuery.contains("rice", ignoreCase = true))
    }

    @Test
    fun testBasketOptimizerSmartSplit() {
        val input = "500g tomato, 1kg potato, 5kg atta, 1L oil"
        val items = ShoppingListParser.parseShoppingList(input)
        val plan = BasketOptimizer.optimizeBasket(items)

        assertTrue("Should detect split advantage between Fresh and Bulk Pantry", plan.hasSplitAdvantage)
        assertEquals(2, plan.freshItems.size) // tomato, potato
        assertEquals(2, plan.bulkItems.size)  // atta, oil
        assertTrue(plan.estimatedSavingsInr >= 100)
    }

    @Test
    fun testMultilineNumberedList() {
        val input = """
            Grocery list:
            1. 1kg onion
            2. 500g curd
            3. 1 pack bread
        """.trimIndent()

        val items = ShoppingListParser.parseShoppingList(input)
        assertEquals(3, items.size)
        assertTrue(items[0].normalizedQuery.contains("onion"))
        assertTrue(items[1].normalizedQuery.contains("curd"))
        assertTrue(items[2].normalizedQuery.contains("bread"))
    }
}
