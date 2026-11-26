package com.zipper.app

import com.zipper.app.core.intelligence.reply.ChatMessageSnippet
import com.zipper.app.core.intelligence.reply.CommitmentAssessment
import com.zipper.app.core.intelligence.reply.CommitmentGuard
import com.zipper.app.core.intelligence.reply.CommitmentRiskType
import com.zipper.app.core.intelligence.reply.ContactDossier
import com.zipper.app.core.intelligence.reply.ContactKnowledgeBook
import com.zipper.app.core.intelligence.reply.LanguageStyle
import com.zipper.app.core.intelligence.reply.MissedCallFollowupEngine
import com.zipper.app.core.intelligence.reply.MultiPingDebouncer
import com.zipper.app.core.intelligence.reply.SpeechActTranslator
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

class GhostTwinSystemTests {

    // ============================================================
    // 1. MULTI-PING DEBOUNCER TESTS
    // ============================================================

    @Test
    fun testMultiPingDebouncerConcatenatesRapidPings() = runBlocking {
        MultiPingDebouncer.resetForTesting(debounceMs = 150L, maxAccMs = 500L)

        val latch = CountDownLatch(1)
        val unifiedResult = AtomicReference<String>()

        val onBatchReady: (String, String, String, android.service.notification.StatusBarNotification?) -> Unit = { _, _, text, _ ->
            unifiedResult.set(text)
            latch.countDown()
        }

        // Send 3 rapid pings
        MultiPingDebouncer.enqueueMessage("pkg_user1", "Manoj", "HELLO", null, onBatchReady)
        delay(30)
        MultiPingDebouncer.enqueueMessage("pkg_user1", "Manoj", "BROO", null, onBatchReady)
        delay(30)
        MultiPingDebouncer.enqueueMessage("pkg_user1", "Manoj", "WHATSUPP", null, onBatchReady)

        // Wait for 150ms debounce silence window to flush
        val completed = latch.await(1000, TimeUnit.MILLISECONDS)
        assertTrue("Debounce batch should flush within timeout", completed)

        val expected = "HELLO\nBROO\nWHATSUPP"
        assertEquals(expected, unifiedResult.get())
    }

    @Test
    fun testMultiPingDebouncerSinglePingFlushes() = runBlocking {
        MultiPingDebouncer.resetForTesting(debounceMs = 80L, maxAccMs = 300L)

        val latch = CountDownLatch(1)
        val unifiedResult = AtomicReference<String>()

        MultiPingDebouncer.enqueueMessage("pkg_user2", "Rahul", "Are you free today?", null) { _, _, text, _ ->
            unifiedResult.set(text)
            latch.countDown()
        }

        val completed = latch.await(800, TimeUnit.MILLISECONDS)
        assertTrue("Single ping should flush after debounce silence", completed)
        assertEquals("Are you free today?", unifiedResult.get())
    }

    // ============================================================
    // 2. SPEECH-ACT & DIRECTIVE-TO-PERSONA TRANSLATOR TESTS
    // ============================================================

    @Test
    fun testSpeechActTranslatorNotAvailableToFriend() {
        val dossier = ContactDossier(
            contactName = "Manoj",
            relationship = "Close Friend",
            languageStyle = LanguageStyle.CASUAL_SLANG
        )

        val translated = SpeechActTranslator.translateDirective(
            rawDirective = "say I am not available to that boy",
            dossier = dossier
        )

        assertEquals("Hey bro, not around right now, caught up with something. Will ping you later 👍", translated)
    }

    @Test
    fun testSpeechActTranslatorNotAvailableToLandlord() {
        val dossier = ContactDossier(
            contactName = "Landlord Uncle",
            relationship = "Landlord",
            languageStyle = LanguageStyle.FORMAL_ENGLISH
        )

        val translated = SpeechActTranslator.translateDirective(
            rawDirective = "say I am not available to that boy",
            dossier = dossier
        )

        assertEquals("Hello Uncle, I am occupied with work right now. Will call you as soon as I'm free. Thank you.", translated)
    }

    @Test
    fun testSpeechActTranslatorRentDirectiveToLandlord() {
        val dossier = ContactDossier(
            contactName = "Landlord",
            relationship = "Landlord",
            languageStyle = LanguageStyle.FORMAL_ENGLISH
        )

        val translated = SpeechActTranslator.translateDirective(
            rawDirective = "tell landlord rent tomorrow",
            dossier = dossier
        )

        assertEquals("Hello Uncle, I will transfer the rent by tomorrow evening. Thank you.", translated)
    }

    @Test
    fun testSpeechActTranslatorSayNoDirectives() {
        val friendDossier = ContactDossier(
            contactName = "Manoj",
            relationship = "Friend",
            languageStyle = LanguageStyle.CASUAL_SLANG
        )
        val shortDossier = ContactDossier(
            contactName = "Ravi",
            relationship = "Friend",
            languageStyle = LanguageStyle.CASUAL_SHORT
        )

        val replyFriend = SpeechActTranslator.translateDirective("say no", friendDossier)
        val replyShort = SpeechActTranslator.translateDirective("say no", shortDossier)

        assertEquals("Can't make it today bro, tied up with something. Let's catch up later!", replyFriend)
        assertEquals("Can't today bro", replyShort)
    }

    @Test
    fun testSpeechActTranslatorAtHomeDirective() {
        val friendDossier = ContactDossier(
            contactName = "Manoj",
            relationship = "Friend",
            languageStyle = LanguageStyle.CASUAL_SLANG
        )

        val reply = SpeechActTranslator.translateDirective("tell him I am at home", friendDossier)
        assertEquals("Ha bro, I'm at home only! What happened, all good?", reply)
    }

    // ============================================================
    // 3. COMMITMENT GUARD (ANTI-HALLUCINATION & LIABILITY SHIELD) TESTS
    // ============================================================

    @Test
    fun testCommitmentGuardBlocksFinancialRequests() {
        val assessment = CommitmentGuard.assessRisk("Can you send 500 rupees on GPay?", "Sure bro, sending now")
        assertTrue("Financial requests must be flagged as risky", assessment.isRisky)
        assertEquals(CommitmentRiskType.FINANCIAL, assessment.riskType)

        val sanitized = CommitmentGuard.sanitizeReply("Send money on UPI", "I will send right now")
        assertFalse(sanitized.contains("sending", ignoreCase = true))
        assertTrue(sanitized.contains("check", ignoreCase = true) || sanitized.contains("free", ignoreCase = true))
    }

    @Test
    fun testCommitmentGuardBlocksSchedulePromises() {
        val assessment = CommitmentGuard.assessRisk("Meet today?", "I promise I will definitely come at 5")
        assertTrue("Firm schedule guarantees must be blocked", assessment.isRisky)
        assertEquals(CommitmentRiskType.SCHEDULE_PROMISE, assessment.riskType)

        val sanitized = CommitmentGuard.sanitizeReply("Meet at 5?", "I guarantee I will reach at 5pm")
        assertFalse(sanitized.contains("guarantee", ignoreCase = true))
        assertTrue(sanitized.contains("schedule", ignoreCase = true) || sanitized.contains("free", ignoreCase = true))
    }

    @Test
    fun testCommitmentGuardBlocksFabricatedLies() {
        val assessment = CommitmentGuard.assessRisk("Where are you?", "Sorry bro admitted in hospital")
        assertTrue("Fabricated hospital/accident excuses must be blocked", assessment.isRisky)
        assertEquals(CommitmentRiskType.FABRICATED_LIE, assessment.riskType)

        val sanitized = CommitmentGuard.sanitizeReply("Where are you?", "Had an emergency accident")
        assertFalse(sanitized.contains("accident", ignoreCase = true))
    }

    @Test
    fun testCommitmentGuardAllowsSafeCasualConversation() {
        val assessment = CommitmentGuard.assessRisk("Did you finish lunch?", "Ha bro just finished")
        assertFalse("Safe conversation should not be flagged", assessment.isRisky)
        assertEquals(CommitmentRiskType.SAFE, assessment.riskType)
    }

    // ============================================================
    // 4. CONTACT KNOWLEDGE BOOK & COMMAND PARSER TESTS
    // ============================================================

    @Test
    fun testParseNaturalRememberCommand() {
        val parsed1 = ContactKnowledgeBook.parseNaturalRememberCommand("remember Manoj: college project partner, lives in Indiranagar")
        assertNotNull(parsed1)
        assertEquals("Manoj", parsed1?.first)
        assertEquals("college project partner, lives in Indiranagar", parsed1?.second)

        val parsed2 = ContactKnowledgeBook.parseNaturalRememberCommand("zipper, note down about Rahul: works at Google")
        assertNotNull(parsed2)
        assertEquals("Rahul", parsed2?.first)
        assertEquals("works at Google", parsed2?.second)

        val parsed3 = ContactKnowledgeBook.parseNaturalRememberCommand("hey zipper remember about Landlord - rent due on 5th")
        assertNotNull(parsed3)
        assertEquals("Landlord", parsed3?.first)
        assertEquals("rent due on 5th", parsed3?.second)
    }

    @Test
    fun testDetectLanguageStyle() {
        val slangStyle = ContactKnowledgeBook.detectLanguageStyle(listOf("macha scenes tonight?", "en samachara bro"))
        assertEquals(LanguageStyle.CASUAL_SLANG, slangStyle)

        val brokenStyle = ContactKnowledgeBook.detectLanguageStyle(listOf("bro were u r", "u comming?"))
        assertEquals(LanguageStyle.BROKEN_BASIC_ENGLISH, brokenStyle)

        val shortStyle = ContactKnowledgeBook.detectLanguageStyle(listOf("k", "busy bro", "ping later"))
        assertEquals(LanguageStyle.CASUAL_SHORT, shortStyle)

        val formalStyle = ContactKnowledgeBook.detectLanguageStyle(listOf("Dear Sir, kindly share the agreement", "Regards"))
        assertEquals(LanguageStyle.FORMAL_ENGLISH, formalStyle)
    }

    // ============================================================
    // 5. MISSED CALL CONTEXTUAL FOLLOW-UP ENGINE TESTS
    // ============================================================

    @Test
    fun testMissedCallFollowupProjectContext() {
        val dossier = ContactDossier(
            contactName = "Manoj",
            relationship = "College Project Partner",
            languageStyle = LanguageStyle.CASUAL_SLANG,
            recentHistory = listOf(
                ChatMessageSnippet("Manoj", "Did you submit the project code?", false),
                ChatMessageSnippet("Karthik", "Working on it bro", true)
            )
        )

        val reply = MissedCallFollowupEngine.synthesizeFollowupMessage("Manoj", dossier)
        assertTrue("Follow-up should mention project / notes context", reply.contains("project", ignoreCase = true) || reply.contains("notes", ignoreCase = true))
    }

    @Test
    fun testMissedCallFollowupMeetingPlansContext() {
        val dossier = ContactDossier(
            contactName = "Rahul",
            relationship = "Friend",
            languageStyle = LanguageStyle.CASUAL_SLANG,
            recentHistory = listOf(
                ChatMessageSnippet("Rahul", "Are we meeting in the evening at gym?", false)
            )
        )

        val reply = MissedCallFollowupEngine.synthesizeFollowupMessage("Rahul", dossier)
        assertTrue("Follow-up should mention meeting / gym plans", reply.contains("still on", ignoreCase = true) || reply.contains("later", ignoreCase = true))
    }

    @Test
    fun testMissedCallFollowupLandlordContext() {
        val dossier = ContactDossier(
            contactName = "Landlord",
            relationship = "Landlord",
            languageStyle = LanguageStyle.FORMAL_ENGLISH
        )

        val reply = MissedCallFollowupEngine.synthesizeFollowupMessage("Landlord", dossier)
        assertTrue("Follow-up should address Landlord politely as Uncle/Sir", reply.contains("Uncle", ignoreCase = true) || reply.contains("Hello", ignoreCase = true))
    }

    @Test
    fun testWhatsAppMissedCallDetection() {
        assertTrue("Missed voice call should be detected", MissedCallFollowupEngine.isMissedCallNotification("Missed voice call", "Rahul"))
        assertTrue("Missed video call should be detected", MissedCallFollowupEngine.isMissedCallNotification("Missed video call", "Rahul"))
        assertTrue("2 missed calls should be detected", MissedCallFollowupEngine.isMissedCallNotification("2 missed calls", "Rahul"))
        assertTrue("Title indicator should be detected", MissedCallFollowupEngine.isMissedCallNotification("from Rahul", "Missed voice call"))
        assertTrue("Group missed call should be detected", MissedCallFollowupEngine.isMissedCallNotification("Missed group call", "Study Group"))

        assertFalse("Normal chat message should NOT be detected as missed call", MissedCallFollowupEngine.isMissedCallNotification("Hey bro are you free?", "Rahul"))
    }

    @Test
    fun testWhatsAppCallerExtraction() {
        assertEquals("Rahul", MissedCallFollowupEngine.extractCallerName("Rahul", "Missed voice call"))
        assertEquals("Rahul", MissedCallFollowupEngine.extractCallerName("Missed voice call", "from Rahul"))
        assertEquals("Rahul", MissedCallFollowupEngine.extractCallerName("Missed voice call", "Rahul at 12:45 PM"))
        assertEquals("Rahul", MissedCallFollowupEngine.extractCallerName("WhatsApp", "Missed voice call from Rahul"))
        assertEquals("Rahul", MissedCallFollowupEngine.extractCallerName("Rahul (2)", "2 missed calls"))
        assertEquals("Bhavani", MissedCallFollowupEngine.extractCallerName("Bhavani", "Missed video call"))
    }

    @Test
    fun testMeetingInquiryDetection() {
        assertTrue(CommitmentGuard.isMeetingOrHangoutInquiry("Where can we meet?"))
        assertTrue(CommitmentGuard.isMeetingOrHangoutInquiry("shall we meet today"))
        assertTrue(CommitmentGuard.isMeetingOrHangoutInquiry("Can we meet bro?"))
        assertTrue(CommitmentGuard.isMeetingOrHangoutInquiry("are you free today to catch up"))
        assertTrue(CommitmentGuard.isMeetingOrHangoutInquiry("where to meet bro"))
        assertTrue(CommitmentGuard.isMeetingOrHangoutInquiry("scenes enu"))
        assertTrue(CommitmentGuard.isMeetingOrHangoutInquiry("sigana nale"))

        assertFalse(CommitmentGuard.isMeetingOrHangoutInquiry("Send me the project notes"))
        assertFalse(CommitmentGuard.isMeetingOrHangoutInquiry("Did you finish assignment?"))
        assertFalse(CommitmentGuard.isMeetingOrHangoutInquiry("How are you?"))
        assertFalse(CommitmentGuard.isMeetingOrHangoutInquiry("Ok bro"))
    }

    @Test
    fun testLocationFabricationPrevention() {
        val incoming = "Where can we meet?"
        val hallucinatedReply = "Yeah bro let's meet in Indiranagar at 5 PM!"

        val assessment = CommitmentGuard.assessRisk(incoming, hallucinatedReply)
        assertTrue("Hallucinated venue/location should be flagged risky", assessment.isRisky)
        assertEquals(CommitmentRiskType.LOCATION_FABRICATION, assessment.riskType)
        assertFalse("Sanitized reply must NOT contain Indiranagar", assessment.safeAlternativeReply.contains("Indiranagar", ignoreCase = true))

        val sanitized = CommitmentGuard.sanitizeReply(incoming, hallucinatedReply)
        assertFalse("Sanitized reply must strip hallucinated location", sanitized.contains("Indiranagar", ignoreCase = true))
    }

    @Test
    fun testBrokenEnglishStyleDetection() {
        val brokenTexts = listOf("bro were u r", "wat r u doin", "pls come fast")
        val style = ContactKnowledgeBook.detectLanguageStyle(brokenTexts)
        assertEquals(LanguageStyle.BROKEN_BASIC_ENGLISH, style)
    }

    @Test
    fun testCasualShortStyleDetection() {
        val shortTexts = listOf("wr u", "free?", "busy bro")
        val style = ContactKnowledgeBook.detectLanguageStyle(shortTexts)
        assertTrue("Short texts should detect as CASUAL_SHORT or BROKEN_BASIC_ENGLISH",
            style == LanguageStyle.CASUAL_SHORT || style == LanguageStyle.BROKEN_BASIC_ENGLISH)
    }

    @Test
    fun testKannadaRegionalStyleDetection() {
        val kannadaTexts = listOf("yavaga barthya karthi mysore ge?", "baralva?")
        val style = ContactKnowledgeBook.detectLanguageStyle(kannadaTexts)
        assertEquals(LanguageStyle.KANNADA_REGIONAL, style)
    }

    @Test
    fun testElderKannadaDeflectionZeroBro() {
        val dossier = ContactDossier(
            contactName = "Rajamma",
            relationship = "Mother",
            memoryPoints = listOf("she knows only kannada", "lives in mysore"),
            languageStyle = LanguageStyle.KANNADA_REGIONAL
        )
        assertTrue("Dossier should be recognized as elder/family", dossier.isElderOrFamily())
        assertTrue("Dossier should have Kannada preference", dossier.isKannadaPreferred())

        val contact = com.zipper.app.core.intelligence.reply.DelegatedContact(
            contactName = "Rajamma",
            personaTone = "Warm friendly"
        )
        val deflection = com.zipper.app.core.intelligence.reply.SmartReplyGenerator.buildDeflectionHoldMessage(
            contact = contact,
            dossier = dossier,
            incomingMessage = "yavaga barthya karthi mysore ge?"
        )
        assertFalse("Deflection to mother/elder must NEVER contain 'bro'", deflection.contains("bro", ignoreCase = true))
        assertTrue("Deflection to mother/elder in Kannada must be respectful",
            deflection.contains("Amma", ignoreCase = true) || deflection.contains("kelasa", ignoreCase = true))
    }

    @Test
    fun testMultiPingDebounceDefaultDelayIs7Seconds() {
        assertEquals("Debounce delay should be 7000ms (7s) for fast responsiveness", 7000L, MultiPingDebouncer.DEFAULT_DEBOUNCE_DELAY_MS)
    }

    // ============================================================
    // 6. UNIVERSAL MISSED CALL FOLLOWUP & WEBLINK TESTS
    // ============================================================

    @Test
    fun testSynthesizeFollowupMessageUniversalVariants() {
        // Mom in Kannada
        val momDossierKannada = ContactDossier(
            contactName = "Mom",
            relationship = "Mother",
            languageStyle = LanguageStyle.KANNADA_REGIONAL
        )
        val momReplyKannada = MissedCallFollowupEngine.synthesizeFollowupMessage("Mom", momDossierKannada)
        assertTrue("Mom in Kannada should use Amma and respect", momReplyKannada.startsWith("Amma, call nod-de"))

        // Dad in English
        val dadDossierEnglish = ContactDossier(
            contactName = "Dad",
            relationship = "Father",
            languageStyle = LanguageStyle.FORMAL_ENGLISH
        )
        val dadReply = MissedCallFollowupEngine.synthesizeFollowupMessage("Dad", dadDossierEnglish)
        assertTrue("Dad in English should be respectful", dadReply.contains("Hey Dad, saw your missed call"))

        // Landlord
        val landlordDossier = ContactDossier(
            contactName = "Suresh",
            relationship = "Landlord",
            languageStyle = LanguageStyle.FORMAL_ENGLISH
        )
        val landlordReply = MissedCallFollowupEngine.synthesizeFollowupMessage("Suresh", landlordDossier)
        assertTrue("Landlord should be addressed respectfully without slang", landlordReply.startsWith("Hello Sir, saw your call"))

        // Friend in Casual Slang
        val friendDossier = ContactDossier(
            contactName = "Naveen",
            relationship = "Friend",
            languageStyle = LanguageStyle.CASUAL_SLANG
        )
        val friendReply = MissedCallFollowupEngine.synthesizeFollowupMessage("Naveen", friendDossier)
        assertFalse("Friend should not be addressed as Sir or Uncle", friendReply.contains("Sir") || friendReply.contains("Uncle"))
        assertTrue("Friend should receive friendly note", friendReply.contains("Hey Naveen"))
    }

    @Test
    fun testFallbackSynthesizeInstructionTunesTimeAvailability() {
        val client = com.zipper.app.core.web.WebLlmClient()

        // 1. Raw decision "I can be in 4" to Aunt in Kannada
        val tunedKannada = client.fallbackSynthesizeInstruction(
            contactName = "Rajamma",
            incomingMessage = "WILL YOU COME TOMORROW",
            userInstruction = "I can be in 4",
            personaTone = "Warm & respectful"
        )
        assertFalse("Should never output raw 'I can be in 4'", tunedKannada.contains("I can be in 4"))
        assertTrue("Should tune into 4 PM Kannada proposal", tunedKannada.contains("4 PM"))

        // 2. Raw decision "3pm" (must NOT leak raw "3pm"!)
        val tuned3pm = client.fallbackSynthesizeInstruction(
            contactName = "Rajamma",
            incomingMessage = "YEST GANTEGE BARTYA?",
            userInstruction = "3pm",
            personaTone = "Warm & respectful"
        )
        assertFalse("Should never output raw '3pm'", tuned3pm.equals("3pm", ignoreCase = true))
        assertTrue("Should tune into 3 PM proposal", tuned3pm.contains("3 PM"))

        // 3. Raw decision "clg" to friend
        val tunedFriend = client.fallbackSynthesizeInstruction(
            contactName = "Rahul",
            incomingMessage = "bro are you coming tomorrow?",
            userInstruction = "clg",
            personaTone = "Casual slang"
        )
        assertFalse("Should never output raw 'clg' alone", tunedFriend.equals("clg", ignoreCase = true))
        assertTrue("Should tune into natural college message", tunedFriend.contains("clg") || tunedFriend.contains("college"))
    }

    @Test
    fun testResolveHonorificAcrossDiverseRelationships() {
        // 1. Aunt (even if name ends in 'amma' like Rajamma)
        val aunt = ContactDossier(
            contactName = "Rajamma",
            relationship = "Aunt",
            languageStyle = LanguageStyle.KANNADA_REGIONAL
        )
        assertEquals("Aunty", aunt.resolveHonorific())

        // 2. Mother
        val mom = ContactDossier(
            contactName = "Amma",
            relationship = "Mother",
            languageStyle = LanguageStyle.KANNADA_REGIONAL
        )
        assertEquals("Amma", mom.resolveHonorific())

        // 3. Father
        val dad = ContactDossier(
            contactName = "Appa",
            relationship = "Father",
            languageStyle = LanguageStyle.FORMAL_ENGLISH
        )
        assertEquals("Appa", dad.resolveHonorific())

        // 4. Landlord
        val landlord = ContactDossier(
            contactName = "Suresh Landlord",
            relationship = "Landlord",
            languageStyle = LanguageStyle.FORMAL_ENGLISH
        )
        assertEquals("Sir", landlord.resolveHonorific())

        // 5. Friend / Peer
        val friend = ContactDossier(
            contactName = "Rahul",
            relationship = "Close Friend",
            languageStyle = LanguageStyle.CASUAL_SLANG
        )
        assertEquals("", friend.resolveHonorific())
    }

    @Test
    fun testObjectionAndCounterProposalDetection() {
        val dispatcher = com.zipper.app.core.intelligence.reply.DelegatedAutoReplyDispatcher

        // Objections & Urgency
        assertTrue("IVATHE BA should be detected as objection/counter", dispatcher.isObjectionOrCounterProposal("IVATHE BA"))
        assertTrue("bega baa should be detected as counter", dispatcher.isObjectionOrCounterProposal("bega baa"))
        assertTrue("No come at 2 PM should be counter", dispatcher.isObjectionOrCounterProposal("No 4 PM won't work, come at 2 PM"))
        assertTrue("Urgent ide should be counter", dispatcher.isObjectionOrCounterProposal("urgent ide baa"))
        assertTrue("beda should be counter", dispatcher.isObjectionOrCounterProposal("Nale beda"))

        // Pure agreement (not an objection)
        assertFalse("Sari should not be counter", dispatcher.isObjectionOrCounterProposal("Sari 4 PM ge"))
        assertFalse("Ok done should not be counter", dispatcher.isObjectionOrCounterProposal("Ok done see you"))
    }

    @Test
    fun testFallbackChainPrioritizesGroqDualLayer() {
        val config = com.zipper.app.core.web.WebAiConfig(
            provider = "ChatGPT Web (Free)",
            fallbackProvider = "Groq",
            fallbackApiKey = "gsk_primaryKey123",
            secondaryGroqApiKey = "gsk_secondaryKey456"
        )
        val chain = config.getFallbackChain()
        assertTrue("Chain must have at least 2 tiers", chain.size >= 2)
        assertEquals("Groq", chain[0].provider)
        assertEquals("gsk_primaryKey123", chain[0].apiKey)
        assertEquals("Groq", chain[1].provider)
        assertEquals("gsk_secondaryKey456", chain[1].apiKey)
    }

    @Test
    fun testFallbackSpokenBriefingTurnsRawTextIntoCleanExecutiveBriefing() {
        val client = com.zipper.app.core.web.WebLlmClient()

        // 1. All-caps objection + time question from Rajamma
        val briefing1 = client.fallbackSpokenBriefing(
            contactName = "rajamma",
            incomingMessage = "AGALA KANO,YESHT GANTE GE FREE AGTYA NALE NINUN?"
        )
        assertTrue("Name must be capitalized", briefing1.startsWith("Rajamma"))
        assertTrue("Must explain that it won't work and asks time", briefing1.contains("won't work", ignoreCase = true) && briefing1.contains("free", ignoreCase = true))
        assertFalse("Must NOT dump raw all-caps text", briefing1.contains("AGALA KANO"))

        // 2. Urgent demand
        val briefing2 = client.fallbackSpokenBriefing(
            contactName = "Rahul",
            incomingMessage = "IVATHE BA"
        )
        assertTrue("Must state urgent", briefing2.contains("Urgent", ignoreCase = true))

        // 3. College inquiry
        val briefing3 = client.fallbackSpokenBriefing(
            contactName = "Manoj",
            incomingMessage = "clg bartya nale?"
        )
        assertTrue("Must mention college", briefing3.contains("college", ignoreCase = true))
    }
}


