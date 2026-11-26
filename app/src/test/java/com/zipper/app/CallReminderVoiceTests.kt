package com.zipper.app

import com.zipper.app.core.voice.CallReminder
import com.zipper.app.core.voice.CallReminderActionType
import com.zipper.app.core.voice.CallReminderScheduler
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Calendar

class CallReminderVoiceTests {

    @Test
    fun testCallReminderJsonRoundTrip() {
        val original = CallReminder(
            id = "test-reminder-123",
            title = "Water Motor",
            spokenPrompt = "Hey Karthik, please turn off the water motor.",
            triggerTimeEpochMs = 1790000000000L,
            actionType = CallReminderActionType.AGENT_HANDOFF,
            targetGoal = "Turn off motor switch",
            targetPackage = "com.smart.home",
            isCompleted = false,
            isSnoozed = true,
            snoozeCount = 2,
            createdAtEpochMs = 1789999000000L
        )

        val json = original.toJson()
        val deserialized = CallReminder.fromJson(json)

        assertEquals(original.id, deserialized.id)
        assertEquals(original.title, deserialized.title)
        assertEquals(original.spokenPrompt, deserialized.spokenPrompt)
        assertEquals(original.triggerTimeEpochMs, deserialized.triggerTimeEpochMs)
        assertEquals(original.actionType, deserialized.actionType)
        assertEquals(original.targetGoal, deserialized.targetGoal)
        assertEquals(original.targetPackage, deserialized.targetPackage)
        assertEquals(original.isCompleted, deserialized.isCompleted)
        assertEquals(original.isSnoozed, deserialized.isSnoozed)
        assertEquals(original.snoozeCount, deserialized.snoozeCount)
        assertEquals(original.createdAtEpochMs, deserialized.createdAtEpochMs)
    }

    @Test
    fun testNaturalReminderRelativeMinutes() {
        val now = System.currentTimeMillis()
        val parsed = CallReminderScheduler.parseNaturalReminder("call me in 20 minutes to switch off water motor")
        assertNotNull(parsed)
        parsed!!

        assertEquals("Switch off water motor", parsed.title)
        assertTrue(parsed.spokenPrompt.contains("switch off water motor"))
        val diffMinutes = (parsed.triggerTimeEpochMs - now) / (60 * 1000.0)
        assertTrue("Expected ~20 minutes but got $diffMinutes", diffMinutes >= 19.5 && diffMinutes <= 20.5)
        assertEquals(CallReminderActionType.VOICE_ONLY, parsed.actionType)
    }

    @Test
    fun testNaturalReminderRelativeHours() {
        val now = System.currentTimeMillis()
        val parsed = CallReminderScheduler.parseNaturalReminder("remind me in 2 hours: check oven roasting")
        assertNotNull(parsed)
        parsed!!

        assertTrue(parsed.title.contains("check oven roasting", ignoreCase = true))
        val diffHours = (parsed.triggerTimeEpochMs - now) / (60 * 60 * 1000.0)
        assertTrue("Expected ~2 hours but got $diffHours", diffHours >= 1.95 && diffHours <= 2.05)
    }

    @Test
    fun testNaturalReminderBillAutoHandoff() {
        val parsed = CallReminderScheduler.parseNaturalReminder("call me in 15 mins to pay BESCOM electricity bill")
        assertNotNull(parsed)
        parsed!!

        assertEquals(CallReminderActionType.AGENT_HANDOFF, parsed.actionType)
        assertNotNull(parsed.targetGoal)
        assertTrue(parsed.targetGoal!!.contains("electricity bill"))
    }

    @Test
    fun testNaturalReminderExactTime() {
        val parsed = CallReminderScheduler.parseNaturalReminder("call me at 8:30 pm to head to the gym")
        assertNotNull(parsed)
        parsed!!

        val cal = Calendar.getInstance().apply { timeInMillis = parsed.triggerTimeEpochMs }
        assertEquals(20, cal.get(Calendar.HOUR_OF_DAY))
        assertEquals(30, cal.get(Calendar.MINUTE))
        assertTrue(parsed.title.contains("head to the gym", ignoreCase = true))
    }

    @Test
    fun testEnglishPhonemizer() {
        val input = "hey karthik time to turn off the water motor"
        val phonemes = com.zipper.app.core.voice.EnglishPhonemizer.phonemize(input)
        assertTrue("Expected water phoneme in $phonemes", phonemes.contains("wˈɔɾəɹ"))
        assertTrue("Expected motor phoneme in $phonemes", phonemes.contains("mˈOɾəɹ"))
        assertTrue("Expected karthik phoneme in $phonemes", phonemes.contains("kˈɑɹθɪk"))

        val inputWithDigits = "it has been 25 minutes"
        val phonemesDigits = com.zipper.app.core.voice.EnglishPhonemizer.phonemize(inputWithDigits)
        assertTrue("Expected twenty in $phonemesDigits", phonemesDigits.contains("twˈɛnti"))
        assertTrue("Expected five in $phonemesDigits", phonemesDigits.contains("fˈIv"))
    }
}
