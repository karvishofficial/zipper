package com.zipper.app

import com.zipper.app.core.agent.GenericExpressionEvaluator
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

class GenericExpressionEvaluatorTests {

    @Test
    fun testEvaluatesAdditionGeneric() {
        val parsed = GenericExpressionEvaluator.parseAndEvaluate("Open Calculator and calculate 25 + 37")
        assertNotNull(parsed)
        assertEquals("62", parsed!!.expectedResult)
    }

    @Test
    fun testEvaluatesSubtractionGeneric() {
        val parsed = GenericExpressionEvaluator.parseAndEvaluate("Open Calculator and calculate 84 - 19")
        assertNotNull(parsed)
        assertEquals("65", parsed!!.expectedResult)
    }

    @Test
    fun testEvaluatesThreeDigitExpressionGeneric() {
        val parsed = GenericExpressionEvaluator.parseAndEvaluate("Open Calculator and calculate 123 + 456")
        assertNotNull(parsed)
        assertEquals("579", parsed!!.expectedResult)
    }

    @Test
    fun testEvaluatesMultiplicationGeneric() {
        val parsed = GenericExpressionEvaluator.parseAndEvaluate("Calculate 50 * 3")
        assertNotNull(parsed)
        assertEquals("150", parsed!!.expectedResult)
    }

    @Test
    fun testEvaluatesDivisionGeneric() {
        val parsed = GenericExpressionEvaluator.parseAndEvaluate("Calculate 100 / 4")
        assertNotNull(parsed)
        assertEquals("25", parsed!!.expectedResult)
    }

    @Test
    fun testReturnsNullForNonMathGoals() {
        val parsed = GenericExpressionEvaluator.parseAndEvaluate("Open Clock and switch to Stopwatch")
        assertNull(parsed)
    }
}
