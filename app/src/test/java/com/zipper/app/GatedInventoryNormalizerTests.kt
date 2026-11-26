package com.zipper.app

import com.zipper.app.core.agent.GatedInventoryNormalizer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Before
import org.junit.Test

class GatedInventoryNormalizerTests {

    private lateinit var normalizer: GatedInventoryNormalizer

    @Before
    fun setUp() {
        // Test with seeded contacts: "Mom", "Dad", "Rohit", "Rohan"
        normalizer = GatedInventoryNormalizer(
            context = null,
            initialContacts = listOf("Mom", "Dad", "Rohit", "Rohan")
        )
    }

    @Test
    fun testAppAliasesResolution() {
        assertEquals("YouTube", normalizer.extractTargetAppName("play doremon on yt"))
        assertEquals("YouTube", normalizer.extractTargetAppName("Search SpaceX on youtube"))
        assertEquals("WhatsApp", normalizer.extractTargetAppName("yo msg mom reached in wa"))
        assertEquals("WhatsApp", normalizer.extractTargetAppName("send hi in whatsapp"))
        assertEquals("Instagram", normalizer.extractTargetAppName("search photography on insta"))
        assertEquals("Instagram", normalizer.extractTargetAppName("check ig for new reels"))
        assertEquals("Calculator", normalizer.extractTargetAppName("calc 450 * 12"))
        assertEquals("Zepto", normalizer.extractTargetAppName("allo 1kg paneeerr 2kg order in zepto"))
        assertEquals("Play Store", normalizer.extractTargetAppName("open play store"))
        assertEquals("Spotify", normalizer.extractTargetAppName("play that weeknd song in spoti"))
    }

    @Test
    fun testEradicationOfBlindTheAppBug() {
        // In the old code: "open the chat" matched "open the" and extracted "The"
        val res1 = normalizer.extractTargetAppName("Can you open the chat with mom")
        assertNotEquals("The", res1)

        val res2 = normalizer.extractTargetAppName("open the settings")
        assertEquals("Settings", res2)

        val res3 = normalizer.extractTargetAppName("open a new tab")
        assertNotEquals("A", res3)
    }

    @Test
    fun testGoalNormalizationStringReplacement() {
        val raw1 = "Search SpaceX on yt"
        val norm1 = normalizer.normalizeGoal(raw1)
        assertEquals("Search SpaceX on YouTube", norm1)

        val raw2 = "yo msg mom reached in wa"
        val norm2 = normalizer.normalizeGoal(raw2)
        assertEquals("yo msg mom reached in WhatsApp", norm2)

        val raw3 = "search photography on insta"
        val norm3 = normalizer.normalizeGoal(raw3)
        assertEquals("search photography on Instagram", norm3)
    }

    @Test
    fun testGatedContactResolution() {
        // "momm" is edit distance 1 from "Mom", and next best contact is > 3 away -> resolves cleanly
        val resMom = normalizer.resolveContactGated("momm")
        assertEquals("Mom", resMom)

        // "roh" is edit distance 2 from both "Rohit" and "Rohan" (margin = 0) -> Gated! Refuses to guess!
        val resRoh = normalizer.resolveContactGated("roh")
        assertEquals("roh", resRoh)
    }
}
