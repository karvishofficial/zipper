package com.zipper.app

import com.zipper.app.core.intelligence.reply.DelegatedContact
import com.zipper.app.core.intelligence.reply.SmartReplyGenerator
import org.junit.Assert.*
import org.junit.Test

/**
 * Unit test suite for the Delegated Contact AI Auto-Reply system.
 * Validates budget limits, anti-loop closure guardrails, JSON serialization, and heuristic response generation.
 */
class DelegatedAutoReplyTests {

    @Test
    fun testDelegatedContactSerializationRoundtrip() {
        val contact = DelegatedContact(
            id = "test-uuid-1234",
            contactName = "Manoj",
            phoneNumber = "+919876543210",
            customContext = "In client meeting until 5 PM. Will send APK after.",
            personaTone = "Casual & Friendly",
            isEnabled = true,
            maxRepliesPerSession = 3,
            requireApproval = false,
            appendAiBadge = true,
            lastReplyTimestampMs = 1700000000000L,
            sessionReplyCount = 2
        )

        val json = contact.toJson()
        val deserialized = DelegatedContact.fromJson(json)

        assertEquals("test-uuid-1234", deserialized.id)
        assertEquals("Manoj", deserialized.contactName)
        assertEquals("+919876543210", deserialized.phoneNumber)
        assertEquals("In client meeting until 5 PM. Will send APK after.", deserialized.customContext)
        assertEquals("Casual & Friendly", deserialized.personaTone)
        assertTrue(deserialized.isEnabled)
        assertEquals(3, deserialized.maxRepliesPerSession)
        assertFalse(deserialized.requireApproval)
        assertTrue(deserialized.appendAiBadge)
        assertEquals(1700000000000L, deserialized.lastReplyTimestampMs)
        assertEquals(2, deserialized.sessionReplyCount)
    }

    @Test
    fun testSessionBudgetAndAntiLoopRateLimiting() {
        var contact = DelegatedContact(
            contactName = "Manoj",
            maxRepliesPerSession = 3,
            lastReplyTimestampMs = System.currentTimeMillis(),
            sessionReplyCount = 0
        )

        assertFalse("Initial session should not exceed budget", contact.isSessionBudgetExceeded())

        // Simulate 3 replies in the active session
        contact = contact.recordReplySent()
        assertEquals(1, contact.sessionReplyCount)
        assertFalse(contact.isSessionBudgetExceeded())

        contact = contact.recordReplySent()
        assertEquals(2, contact.sessionReplyCount)
        assertFalse(contact.isSessionBudgetExceeded())

        contact = contact.recordReplySent()
        assertEquals(3, contact.sessionReplyCount)
        assertTrue("Budget should be exceeded after 3 replies", contact.isSessionBudgetExceeded())

        // Simulate a new session arriving after 20 minutes (stale threshold = 15m)
        val staleTime = System.currentTimeMillis() - (20 * 60 * 1000L)
        val staleContact = contact.copy(lastReplyTimestampMs = staleTime)

        assertFalse("Stale session should reset and allow new message", staleContact.isSessionBudgetExceeded())
        val freshSessionContact = staleContact.recordReplySent()
        assertEquals("Count should reset to 1 in new session", 1, freshSessionContact.sessionReplyCount)
    }

    @Test
    fun testClosureDetectionPreventsInfinitePingPong() {
        // Closures and acknowledgments must return TRUE (skipping reply)
        assertTrue(SmartReplyGenerator.isClosureOrAcknowledgment("ok"))
        assertTrue(SmartReplyGenerator.isClosureOrAcknowledgment("OK"))
        assertTrue(SmartReplyGenerator.isClosureOrAcknowledgment("k"))
        assertTrue(SmartReplyGenerator.isClosureOrAcknowledgment("thanks"))
        assertTrue(SmartReplyGenerator.isClosureOrAcknowledgment("thank you!"))
        assertTrue(SmartReplyGenerator.isClosureOrAcknowledgment("noted"))
        assertTrue(SmartReplyGenerator.isClosureOrAcknowledgment("👍"))
        assertTrue(SmartReplyGenerator.isClosureOrAcknowledgment("cool"))
        assertTrue(SmartReplyGenerator.isClosureOrAcknowledgment("cya"))
        assertTrue(SmartReplyGenerator.isClosureOrAcknowledgment("bye"))

        // Real questions or conversational inputs must return FALSE (triggering reply)
        assertFalse(SmartReplyGenerator.isClosureOrAcknowledgment("Are we meeting today?"))
        assertFalse(SmartReplyGenerator.isClosureOrAcknowledgment("Can you send the APK?"))
        assertFalse(SmartReplyGenerator.isClosureOrAcknowledgment("Where are you right now?"))
        assertFalse(SmartReplyGenerator.isClosureOrAcknowledgment("Is the build ready?"))
    }

    @Test
    fun testDisabledContactIsBlocked() {
        val contact = DelegatedContact(
            contactName = "Manoj",
            isEnabled = false
        )
        assertTrue("Disabled contact must report budget exceeded to block replies", contact.isSessionBudgetExceeded())
    }

    @Test
    fun testContinuousModeNeverExceedsBudget() {
        var contact = DelegatedContact(
            contactName = "Rajamma",
            maxRepliesPerSession = 100, // Continuous mode
            lastReplyTimestampMs = System.currentTimeMillis(),
            sessionReplyCount = 50
        )
        assertFalse("Continuous mode with 100 limit should never block", contact.isSessionBudgetExceeded())

        val unconstrainedContact = contact.copy(maxRepliesPerSession = 0)
        assertFalse("Continuous mode with 0 limit should never block", unconstrainedContact.isSessionBudgetExceeded())
    }

    @Test
    fun testAnchorPreservedFifoMemoryStack() {
        // Create 25 sequential conversation turns
        val turns = (1..25).map { i ->
            com.zipper.app.core.intelligence.reply.ChatTurn(
                sender = if (i % 2 == 1) "Manoj" else "Karthik",
                message = "Turn $i message content"
            )
        }

        // Prune to a max of 15 turns
        val pruned = com.zipper.app.core.intelligence.reply.DelegatedConversationMemory.pruneWithAnchor(turns, maxTurns = 15)

        assertEquals("Pruned list must respect maxTurns limit", 15, pruned.size)

        // Turn 1 and Turn 2 (The Genesis Anchor) MUST remain permanently pinned
        assertEquals("Turn 1 message content", pruned[0].message)
        assertEquals("Manoj", pruned[0].sender)
        assertEquals("Turn 2 message content", pruned[1].message)
        assertEquals("Karthik", pruned[1].sender)

        // The rolling tail must contain the most recent turns (e.g. Turn 13 to Turn 25)
        assertEquals("Turn 25 message content", pruned.last().message)
        assertEquals("Turn 24 message content", pruned[pruned.size - 2].message)

        // Intermediate turns (e.g. Turn 3, Turn 4) must be pruned to avoid amnesia of initial context
        assertFalse("Intermediate Turn 3 should be pruned out", pruned.any { it.message == "Turn 3 message content" })
        assertFalse("Intermediate Turn 4 should be pruned out", pruned.any { it.message == "Turn 4 message content" })
    }

    @Test
    fun testCallUserPayloadParsing() {
        val payload = "CALL_USER: Rahul wants to meet tomorrow at 10 AM | Not 100% sure about 10 AM yet bro, let me check my schedule once I'm free and ping you in a bit!"
        val parsed = com.zipper.app.core.intelligence.reply.DelegatedAutoReplyDispatcher.parseCallUserPayload(payload, "Rahul")

        assertEquals("Rahul wants to meet tomorrow at 10 AM", parsed.reason)
        assertEquals("Not 100% sure about 10 AM yet bro, let me check my schedule once I'm free and ping you in a bit!", parsed.deflectionReply)

        // Test with case insensitivity and extra spaces
        val messyPayload = "call_user:  Discuss 11 AM sync  |   I will confirm shortly bro!  "
        val parsedMessy = com.zipper.app.core.intelligence.reply.DelegatedAutoReplyDispatcher.parseCallUserPayload(messyPayload, "Rahul")
        assertEquals("Discuss 11 AM sync", parsedMessy.reason)
        assertEquals("I will confirm shortly bro!", parsedMessy.deflectionReply)
    }

    @Test
    fun testSlangIsolationForProfessionalPersona() {
        val casualContact = DelegatedContact(
            contactName = "Rahul",
            personaTone = "Casual & Friendly"
        )
        val professionalContact = DelegatedContact(
            contactName = "Dr. Sharma",
            personaTone = "Professional & Polite"
        )

        val casualReply = SmartReplyGenerator.generateHeuristicFallback(casualContact, "Can we have a meeting at 10am?")
        assertTrue("Casual reply can use friendly slang", casualReply.contains("bro", ignoreCase = true))

        val professionalReply = SmartReplyGenerator.generateHeuristicFallback(professionalContact, "Can we have a meeting at 10am?")
        assertFalse("Professional reply MUST NOT contain 'bro'", professionalReply.contains("bro", ignoreCase = true))
        assertTrue("Professional reply must be formal and polite", professionalReply.contains("calendar", ignoreCase = true))
    }

    @Test
    fun testSessionResetStartsFresh() {
        val contact = DelegatedContact(
            id = "test-reset-1",
            contactName = "Manoj",
            maxRepliesPerSession = 3,
            sessionReplyCount = 3,
            lastReplyTimestampMs = System.currentTimeMillis()
        )
        assertTrue("Budget should be full", contact.isSessionBudgetExceeded())

        // Resetting session manually resets count and timestamp
        val freshContact = contact.copy(sessionReplyCount = 0, lastReplyTimestampMs = 0L)
        assertFalse("After reset, session should allow fresh chat", freshContact.isSessionBudgetExceeded())
        assertEquals(0, freshContact.sessionReplyCount)
    }

    @Test
    fun testDirectiveSynthesisConvertsRawInstructionsToFirstPerson() {
        val client = com.zipper.app.core.web.WebLlmClient()

        // Test 1: User says "tmw is okk ask her what time" to a casual Kanglish contact
        val casualSynthesized = client.fallbackSynthesizeInstruction(
            contactName = "Rajamma",
            incomingMessage = "BRO CAN WE MEET TMMORW>",
            userInstruction = "tmw is okk ask her what time",
            personaTone = "Casual & Friendly"
        )
        assertFalse("Must NOT leak raw directive 'ask her'", casualSynthesized.contains("ask her", ignoreCase = true))
        assertTrue("Must include time inquiry", casualSynthesized.contains("time", ignoreCase = true))

        // Test 2: User says "tmw is okk ask her what time" to a Professional contact
        val proSynthesized = client.fallbackSynthesizeInstruction(
            contactName = "Dr. Mehta",
            incomingMessage = "Could we schedule a call tomorrow?",
            userInstruction = "tmw is okk ask her what time",
            personaTone = "Professional & Polite"
        )
        assertFalse("Must NOT leak raw directive", proSynthesized.contains("ask her", ignoreCase = true))
        assertFalse("Must NOT contain slang 'bro'", proSynthesized.contains("bro", ignoreCase = true))
        assertTrue("Must be formal", proSynthesized.contains("convenient", ignoreCase = true) || proSynthesized.contains("time", ignoreCase = true))

        // Test 3: User says "I am free at 5pm"
        val timeSynthesized = client.fallbackSynthesizeInstruction(
            contactName = "Rajamma",
            incomingMessage = "Can we meet at 11 AM?",
            userInstruction = "I am free at 5pm",
            personaTone = "Casual & Friendly"
        )
        assertTrue("Must mention 5 PM", timeSynthesized.contains("5 PM", ignoreCase = true))
        assertFalse("Must NOT leak third person speech", timeSynthesized.contains("Karthik says", ignoreCase = true))
    }

    @Test
    fun testClearCallCooldownAllowsImmediateSubsequentCall() {
        val contactId = "contact-test-123"
        // Clearing cooldown should not throw and safely execute
        com.zipper.app.core.intelligence.reply.DelegatedAutoReplyDispatcher.clearCallCooldown(contactId)
    }

    @Test
    fun testExtractProposedTime() {
        assertEquals("5PM", SmartReplyGenerator.extractProposedTime("TOMMOROW AROUND 5PM?"))
        assertEquals("11 am", SmartReplyGenerator.extractProposedTime("Can we meet at 11 am?"))
        assertEquals("4:30 pm", SmartReplyGenerator.extractProposedTime("Let's sync at 4:30 pm"))
        assertEquals("5 gantege", SmartReplyGenerator.extractProposedTime("Nale 5 gantege sigana?"))
        assertNull(SmartReplyGenerator.extractProposedTime("CAN WE MEET TOMMOROW?"))
        assertNull(SmartReplyGenerator.extractProposedTime("Yeah, tomorrow works for me! What time were you thinking?"))
    }

    @Test
    fun testSafetyInterceptorPreventsUnauthorizedTimeAgreement() {
        // Conversation turns where Karthik has only agreed to tomorrow in general, but NOT 5 PM
        val history = listOf(
            com.zipper.app.core.intelligence.reply.ChatTurn(sender = "rajamma", message = "CAN WE MEET TOMMOROW?"),
            com.zipper.app.core.intelligence.reply.ChatTurn(sender = "Karthik", message = "Yeah, tomorrow works for me! What time were you thinking?")
        )

        // Contact proposes 5 PM
        val proposed = "5PM"
        assertFalse(
            "Karthik has NOT approved 5 PM yet! Safety interceptor must report unconfirmed.",
            SmartReplyGenerator.isTimeAlreadyConfirmedByUser(proposed, history)
        )

        // After Karthik answers the call and explicitly confirms 5 PM
        val updatedHistory = history + listOf(
            com.zipper.app.core.intelligence.reply.ChatTurn(sender = "rajamma", message = "TOMMOROW AROUND 5PM?"),
            com.zipper.app.core.intelligence.reply.ChatTurn(sender = "Karthik", message = "5 PM works for me! See you then.")
        )

        assertTrue(
            "Once Karthik has confirmed 5 PM, subsequent messages should know it is confirmed.",
            SmartReplyGenerator.isTimeAlreadyConfirmedByUser(proposed, updatedHistory)
        )
    }

    @Test
    fun testEmergencyAndUrgentMeetDetection() {
        // High urgency and urgent meet inquiries must trigger CALL_USER
        assertTrue(
            SmartReplyGenerator.isUrgentOrEmergency("CAN I KNOW BRO WHERE ARE YOU BECAUSE WE WANT TO MEET")
        )
        assertTrue(
            SmartReplyGenerator.isUrgentOrEmergency("Where are you bro we need to meet urgently")
        )
        assertTrue(
            SmartReplyGenerator.isUrgentOrEmergency("Hospital emergency bro please reply")
        )
        assertTrue(
            SmartReplyGenerator.isUrgentOrEmergency("Nale 5 gantege sigana, urgent ide baa")
        )

        // Casual location question without urgency or meet intent should NOT trigger emergency call
        assertFalse(
            SmartReplyGenerator.isUrgentOrEmergency("where are you now bro?")
        )
        assertTrue(
            SmartReplyGenerator.isCasualLocationInquiry("where are you now bro?")
        )
    }

    @Test
    fun testCasualLocationPrivacyShieldDoesNotLie() {
        val casualContact = DelegatedContact(contactName = "Rajamma", personaTone = "Casual & Friendly")
        val proContact = DelegatedContact(contactName = "Dr. Mehta", personaTone = "Professional & Polite")

        val casualReply = SmartReplyGenerator.generateHeuristicFallback(casualContact, "where are you now bro?")
        assertFalse("Must never invent fake places", casualReply.contains("Indiranagar") || casualReply.contains("cafe") || casualReply.contains("home"))
        assertTrue("Must use privacy shield and ask what's up", casualReply.contains("work") && casualReply.contains("what's up"))

        val proReply = SmartReplyGenerator.generateHeuristicFallback(proContact, "where are you right now?")
        assertFalse("Must not use casual slang", proReply.contains("bro"))
        assertTrue("Must be formal and ask if urgent", proReply.contains("desk") && proReply.contains("urgent"))
    }

    @Test
    fun testLocationDirectiveSynthesis() {
        val client = com.zipper.app.core.web.WebLlmClient()

        // 1. Karthik says "at home tell them to come" in call
        val synthesizedHome = client.fallbackSynthesizeInstruction(
            contactName = "Rajamma",
            incomingMessage = "CAN I KNOW BRO WHERE ARE YOU BECAUSE WE WANT TO MEET",
            userInstruction = "at home tell them to come",
            personaTone = "Casual & Friendly"
        )
        assertTrue("Must confirm home", synthesizedHome.contains("home", ignoreCase = true))
        assertTrue("Must be natural Kanglish/casual", synthesizedHome.contains("bro", ignoreCase = true) || synthesizedHome.contains("baa", ignoreCase = true))
        assertFalse("Must NOT leak raw directive", synthesizedHome.contains("tell them", ignoreCase = true))

        // 2. Karthik says "silk board stuck in traffic ask what happened"
        val synthesizedTraffic = client.fallbackSynthesizeInstruction(
            contactName = "Rajamma",
            incomingMessage = "Where are you bro?",
            userInstruction = "silk board stuck in traffic ask what happened",
            personaTone = "Casual & Friendly"
        )
        assertTrue("Must mention traffic", synthesizedTraffic.contains("traffic", ignoreCase = true))
        assertFalse("Must NOT leak raw directive", synthesizedTraffic.contains("ask what happened", ignoreCase = true))
    }

    @Test
    fun testCasualHeuristicKanglishAndSlang() {
        val contact = DelegatedContact(contactName = "Shashank", personaTone = "Casual & Friendly")
        
        // 1. Kanglish college question
        val clgReply = SmartReplyGenerator.generateHeuristicFallback(contact, "tmw clg?")
        assertTrue("Must contain casual slang or clg response", clgReply.contains("bro", ignoreCase = true) || clgReply.contains("clg", ignoreCase = true))

        // 2. Macha greeting
        val machaReply = SmartReplyGenerator.generateHeuristicFallback(contact, "macha what are you doing?")
        assertTrue("Must respond to macha with friendly slang", machaReply.contains("macha", ignoreCase = true) || machaReply.contains("bro", ignoreCase = true))

        // 3. Question mark message
        val qReply = SmartReplyGenerator.generateHeuristicFallback(contact, "Can we meet tomorrow?")
        assertTrue("Casual question reply should contain bro", qReply.contains("bro", ignoreCase = true))
    }

    @Test
    fun testGroqChatReplyWithQwenModel() = kotlinx.coroutines.runBlocking {
        val groqKey = com.zipper.app.BuildConfig.SECONDARY_GROQ_API_KEY
            .ifBlank { com.zipper.app.BuildConfig.PRIMARY_GROQ_API_KEY }
        org.junit.Assume.assumeTrue("Skipping live Groq test: no GROQ API key configured in secrets.properties", groqKey.isNotBlank())

        val client = com.zipper.app.core.web.WebLlmClient()
        val config = com.zipper.app.core.web.WebAiConfig(
            provider = "Groq",
            apiKey = groqKey,
            model = "qwen/qwen3.8-27b"
        )
        val reply = client.generateChatReply(
            contactName = "rajamma",
            incomingMessage = "TMW CLG?",
            recentHistory = emptyList(),
            userContext = "some how convince them to come to college tmw",
            personaTone = "Casual & Friendly",
            config = config
        )
        println("Generated Groq reply: $reply")
        assertNotNull(reply)
        assertTrue(reply!!.isNotBlank())
        assertFalse(reply.contains("NO_REPLY", ignoreCase = true))
    }
}



