package com.zipper.app

import com.zipper.app.core.remote.RemoteActionType
import com.zipper.app.core.remote.RemoteCommandParser
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class RemoteCommandTests {

    @Test
    fun testTriggerDetection() {
        assertTrue(RemoteCommandParser.isTriggerCommand("zipper: open whatsapp and send hi to shobha"))
        assertTrue(RemoteCommandParser.isTriggerCommand("Zipper: send hi to shobha on whatsapp"))
        assertTrue(RemoteCommandParser.isTriggerCommand("hey zipper, order 1kg tomato on zepto"))
        assertTrue(RemoteCommandParser.isTriggerCommand("Zipper buy potato on blinkit"))
        assertTrue(RemoteCommandParser.isTriggerCommand("zipper what was my last otp"))
        assertTrue(RemoteCommandParser.isTriggerCommand("zipper #1234: ring"))
        assertTrue(RemoteCommandParser.isTriggerCommand("hey zipper 9988: battery"))
        assertTrue(RemoteCommandParser.isTriggerCommand("#1234: ring"))
        assertTrue(RemoteCommandParser.isTriggerCommand("#1234 ring"))
        assertTrue(RemoteCommandParser.isTriggerCommand("#1234: battery"))
        assertTrue(RemoteCommandParser.isTriggerCommand("#1234: at 2:30 send hello to bhavani"))

        assertFalse(RemoteCommandParser.isTriggerCommand("Hey are you coming to party?"))
        assertFalse(RemoteCommandParser.isTriggerCommand("Your Swiggy order has arrived"))
    }

    @Test
    fun testPinExtraction() {
        val cmdWithPin = RemoteCommandParser.parse(
            rawText = "zipper #1234: battery",
            sender = "Unknown +919876543210",
            sourceApp = "WhatsApp"
        )
        assertNotNull(cmdWithPin)
        assertEquals("1234", cmdWithPin?.pinProvided)
        assertEquals(RemoteActionType.GET_BATTERY_STATUS, cmdWithPin?.actionType)

        val cmdDirectPin = RemoteCommandParser.parse(
            rawText = "#1234: ring",
            sender = "Unknown",
            sourceApp = "WhatsApp"
        )
        assertNotNull(cmdDirectPin)
        assertEquals("1234", cmdDirectPin?.pinProvided)
        assertEquals(RemoteActionType.FIND_MY_PHONE_RING, cmdDirectPin?.actionType)

        val cmdDirectSpacePin = RemoteCommandParser.parse(
            rawText = "#1234 battery status",
            sender = "Unknown",
            sourceApp = "SMS"
        )
        assertNotNull(cmdDirectSpacePin)
        assertEquals("1234", cmdDirectSpacePin?.pinProvided)
        assertEquals(RemoteActionType.GET_BATTERY_STATUS, cmdDirectSpacePin?.actionType)

        val cmdWithRawPin = RemoteCommandParser.parse(
            rawText = "hey zipper 5566: ring phone",
            sender = "Friend",
            sourceApp = "SMS"
        )
        assertNotNull(cmdWithRawPin)
        assertEquals("5566", cmdWithRawPin?.pinProvided)
        assertEquals(RemoteActionType.FIND_MY_PHONE_RING, cmdWithRawPin?.actionType)

        val cmdNoPin = RemoteCommandParser.parse(
            rawText = "zipper: location",
            sender = "Mom",
            sourceApp = "WhatsApp"
        )
        assertNotNull(cmdNoPin)
        assertNull(cmdNoPin?.pinProvided)
        assertEquals(RemoteActionType.GET_LOCATION, cmdNoPin?.actionType)
    }

    @Test
    fun testScheduledMessageParsing() {
        // 1. Pattern: "#1234: at 2:30 send hello bhavani to bhavani"
        val cmd1 = RemoteCommandParser.parse(
            rawText = "#1234: at 2:30 send hello bhavani to bhavani",
            sender = "Self",
            sourceApp = "WhatsApp"
        )
        assertNotNull(cmd1)
        assertEquals("1234", cmd1?.pinProvided)
        assertEquals(RemoteActionType.SCHEDULE_MESSAGE, cmd1?.actionType)
        assertEquals("bhavani", cmd1?.recipient)
        assertEquals("hello bhavani", cmd1?.messageBody)
        assertNotNull(cmd1?.scheduledTimeMs)

        // 2. Trailing at time: "zipper #1234: send hello to rahul at 2:30pm"
        val cmd2 = RemoteCommandParser.parse(
            rawText = "zipper #1234: send hello to rahul at 2:30pm",
            sender = "Self",
            sourceApp = "SMS"
        )
        assertNotNull(cmd2)
        assertEquals("1234", cmd2?.pinProvided)
        assertEquals(RemoteActionType.SCHEDULE_MESSAGE, cmd2?.actionType)
        assertEquals("rahul", cmd2?.recipient)
        assertEquals("hello", cmd2?.messageBody)
        assertNotNull(cmd2?.scheduledTimeMs)

        // 3. Relative minutes: "#1234: in 20 mins send hi to rahul"
        val cmd3 = RemoteCommandParser.parse(
            rawText = "#1234: in 20 mins send hi to rahul",
            sender = "Self",
            sourceApp = "WhatsApp"
        )
        assertNotNull(cmd3)
        assertEquals(RemoteActionType.SCHEDULE_MESSAGE, cmd3?.actionType)
        assertEquals("rahul", cmd3?.recipient)
        assertEquals("hi", cmd3?.messageBody)
        assertTrue(cmd3!!.scheduledTimeMs!! > System.currentTimeMillis() + (18 * 60 * 1000L))
    }

    @Test
    fun testEmergencyRingAndStopParsing() {
        val ring1 = RemoteCommandParser.parse("zipper: ring", "Friend", "WhatsApp")
        assertEquals(RemoteActionType.FIND_MY_PHONE_RING, ring1?.actionType)

        val ring2 = RemoteCommandParser.parse("zipper: find my phone", "Friend", "WhatsApp")
        assertEquals(RemoteActionType.FIND_MY_PHONE_RING, ring2?.actionType)

        val ring3 = RemoteCommandParser.parse("zipper #1234: alarm", "Friend", "SMS")
        assertEquals(RemoteActionType.FIND_MY_PHONE_RING, ring3?.actionType)

        val stop1 = RemoteCommandParser.parse("zipper: stop", "Friend", "WhatsApp")
        assertEquals(RemoteActionType.FIND_MY_PHONE_STOP, stop1?.actionType)

        val stop2 = RemoteCommandParser.parse("zipper: stop ringing", "Friend", "SMS")
        assertEquals(RemoteActionType.FIND_MY_PHONE_STOP, stop2?.actionType)

        val stop3 = RemoteCommandParser.parse("zipper: silent", "Friend", "WhatsApp")
        assertEquals(RemoteActionType.FIND_MY_PHONE_STOP, stop3?.actionType)
    }

    @Test
    fun testBatteryAndLocationParsing() {
        val batt1 = RemoteCommandParser.parse("zipper: battery", "Self", "WhatsApp")
        assertEquals(RemoteActionType.GET_BATTERY_STATUS, batt1?.actionType)

        val batt2 = RemoteCommandParser.parse("zipper #1234: battery status", "Self", "SMS")
        assertEquals(RemoteActionType.GET_BATTERY_STATUS, batt2?.actionType)

        val loc1 = RemoteCommandParser.parse("zipper: location", "Dad", "WhatsApp")
        assertEquals(RemoteActionType.GET_LOCATION, loc1?.actionType)

        val loc2 = RemoteCommandParser.parse("zipper: where is my phone", "Dad", "WhatsApp")
        assertEquals(RemoteActionType.GET_LOCATION, loc2?.actionType)
    }

    @Test
    fun testOtpCommandParsing() {
        val cmd1 = RemoteCommandParser.parse("zipper: last otp", "Self", "SMS")
        assertEquals(RemoteActionType.GET_LAST_OTP, cmd1?.actionType)

        val cmd2 = RemoteCommandParser.parse("zipper #1234: get otp", "Self", "WhatsApp")
        assertEquals(RemoteActionType.GET_LAST_OTP, cmd2?.actionType)
    }

    @Test
    fun testWhatsAppMessageParsing() {
        val cmd1 = RemoteCommandParser.parse(
            rawText = "zipper: open whatsapp and send hi to shobha",
            sender = "Self",
            sourceApp = "WhatsApp"
        )
        assertNotNull(cmd1)
        assertEquals(RemoteActionType.SEND_WHATSAPP_MESSAGE, cmd1?.actionType)
        assertEquals("shobha", cmd1?.recipient?.lowercase())
        assertEquals("hi", cmd1?.messageBody?.lowercase())

        val cmd2 = RemoteCommandParser.parse(
            rawText = "zipper: send 'meeting at 5pm' to Rahul on whatsapp",
            sender = "Self",
            sourceApp = "SMS"
        )
        assertNotNull(cmd2)
        assertEquals(RemoteActionType.SEND_WHATSAPP_MESSAGE, cmd2?.actionType)
        assertEquals("rahul", cmd2?.recipient?.lowercase())
        assertEquals("meeting at 5pm", cmd2?.messageBody)
    }

    @Test
    fun testShoppingCommandParsing() {
        val cmd = RemoteCommandParser.parse(
            rawText = "zipper: order 1kg tomato, 500g potato on zepto",
            sender = "Mom",
            sourceApp = "WhatsApp"
        )
        assertNotNull(cmd)
        assertEquals(RemoteActionType.EXECUTE_SHOPPING, cmd?.actionType)
        assertEquals("1kg tomato, 500g potato", cmd?.shoppingQuery)
        assertEquals("zepto", cmd?.targetApp?.lowercase())
    }

    @Test
    fun testFileSharingCommandParsing() {
        // Direct file to self
        val cmd1 = RemoteCommandParser.parse("zipper: send 12th marks card pdf", "Self", "WhatsApp")
        assertNotNull(cmd1)
        assertEquals(RemoteActionType.SEND_FILE, cmd1?.actionType)
        assertEquals("12th marks card pdf", cmd1?.fileQuery)
        assertNull(cmd1?.recipient)

        // File with PIN to specific recipient
        val cmd2 = RemoteCommandParser.parse("zipper #1234: send resume pdf to Rahul on whatsapp", "Self", "WhatsApp")
        assertNotNull(cmd2)
        assertEquals(RemoteActionType.SEND_FILE, cmd2?.actionType)
        assertEquals("resume pdf", cmd2?.fileQuery)
        assertEquals("Rahul", cmd2?.recipient)
        assertEquals("1234", cmd2?.pinProvided)

        // Share format
        val cmd3 = RemoteCommandParser.parse("zipper: share 12th marks card with Manoj", "Self", "WhatsApp")
        assertNotNull(cmd3)
        assertEquals(RemoteActionType.SEND_FILE, cmd3?.actionType)
        assertEquals("12th marks card", cmd3?.fileQuery)
        assertEquals("Manoj", cmd3?.recipient)

        // Spaced extension format (e.g. "send screen_wa. png")
        val cmd4 = RemoteCommandParser.parse("Zipper #1234:send screen_wa. png", "Rajamma", "WhatsApp")
        assertNotNull(cmd4)
        assertEquals(RemoteActionType.SEND_FILE, cmd4?.actionType)
        assertEquals("screen_wa.png", cmd4?.fileQuery)
        assertEquals("1234", cmd4?.pinProvided)

        // Compound word query without space
        val cmd5 = RemoteCommandParser.parse("Zipper #1234:send karthikresume.pdf", "Rajamma", "WhatsApp")
        assertNotNull(cmd5)
        assertEquals(RemoteActionType.SEND_FILE, cmd5?.actionType)
        assertEquals("karthikresume.pdf", cmd5?.fileQuery)
    }

    @Test
    fun testWhatsAppMessageParsingWithoutAnd() {
        val cmd = RemoteCommandParser.parse(
            rawText = "zipper #1234: open whatsapp send 'meeting at 5pm' to manoj",
            sender = "Rajamma",
            sourceApp = "WhatsApp"
        )
        assertNotNull(cmd)
        assertEquals(RemoteActionType.SEND_WHATSAPP_MESSAGE, cmd?.actionType)
        assertEquals("manoj", cmd?.recipient?.lowercase())
        assertEquals("meeting at 5pm", cmd?.messageBody)
    }

    @Test
    fun testFileDisambiguationSelection() {
        val session = com.zipper.app.core.remote.PendingDisambiguationSession(
            senderId = "Karthik",
            fileCandidates = listOf(
                com.zipper.app.core.remote.ResolvedFile("12th_marks_final.pdf", "/storage/12th_marks_final.pdf", 1024, "application/pdf"),
                com.zipper.app.core.remote.ResolvedFile("12th_marks_draft.pdf", "/storage/12th_marks_draft.pdf", 2048, "application/pdf")
            )
        )
        com.zipper.app.core.remote.PendingRemoteSessionStore.setPending("Karthik", session)

        val idx1 = RemoteCommandParser.parseSelectionIndex("1", session)
        assertEquals(0, idx1)

        val idx2 = RemoteCommandParser.parseSelectionIndex("zipper: 2", session)
        assertEquals(1, idx2)

        val parsed = RemoteCommandParser.parse("2", "Karthik", "WhatsApp")
        assertNotNull(parsed)
        assertEquals(RemoteActionType.CONFIRM_FILE_SELECTION, parsed?.actionType)
        assertEquals("12th_marks_draft.pdf", parsed?.resolvedFile?.name)

        com.zipper.app.core.remote.PendingRemoteSessionStore.removePending("Karthik")
    }

    @Test
    fun testShortPinAndWhereIsMyPhone() {
        // Test short 2-character PIN
        val cmdShort = RemoteCommandParser.parse("#12: ring", "Unknown", "WhatsApp")
        assertNotNull(cmdShort)
        assertEquals("12", cmdShort?.pinProvided)
        assertEquals(RemoteActionType.FIND_MY_PHONE_RING, cmdShort?.actionType)

        // Test colon-separated with zipper prefix
        val cmdColon = RemoteCommandParser.parse("zipper: #12: ring", "Unknown", "SMS")
        assertNotNull(cmdColon)
        assertEquals("12", cmdColon?.pinProvided)
        assertEquals(RemoteActionType.FIND_MY_PHONE_RING, cmdColon?.actionType)

        // Test natural language find triggers
        val cmdWhere = RemoteCommandParser.parse("#1234 where is my phone", "Friend", "SMS")
        assertNotNull(cmdWhere)
        assertEquals("1234", cmdWhere?.pinProvided)
        assertEquals(RemoteActionType.GET_LOCATION, cmdWhere?.actionType)

        val cmdRing = RemoteCommandParser.parse("zipper #1234: ring my phone", "Self", "WhatsApp")
        assertNotNull(cmdRing)
        assertEquals(RemoteActionType.FIND_MY_PHONE_RING, cmdRing?.actionType)
    }

    @Test
    fun testScreenMemoryRecallCommand() {
        val cmdRecall = RemoteCommandParser.parse("#1234: recall shoe price", "Self", "WhatsApp")
        assertNotNull(cmdRecall)
        assertEquals("1234", cmdRecall?.pinProvided)
        assertEquals(RemoteActionType.RECALL_SCREEN_MEMORY, cmdRecall?.actionType)
        assertEquals("shoe price", cmdRecall?.rawGoal)

        val cmdWhatDidISee = RemoteCommandParser.parse("zipper #1234: what did i see on amazon", "Self", "SMS")
        assertNotNull(cmdWhatDidISee)
        assertEquals(RemoteActionType.RECALL_SCREEN_MEMORY, cmdWhatDidISee?.actionType)
    }

    @Test
    fun testCartArbitrageCommand() {
        val cmdArbitrage = RemoteCommandParser.parse("#1234: compare cart", "Self", "WhatsApp")
        assertNotNull(cmdArbitrage)
        assertEquals(RemoteActionType.EVALUATE_CART_ARBITRAGE, cmdArbitrage?.actionType)

        val report = com.zipper.app.core.web.features.CartArbitrageEngine.evaluateBasketArbitrage(
            fallbackText = "1kg tomato, 500g paneer, 1 pack bread"
        )
        assertNotNull(report.cheapestStore)
        assertTrue(report.storeEvaluations.size >= 3)
        assertTrue(report.summaryMarkdown.contains("Cross-App Cart Arbitrage Report"))
    }

    @Test
    fun testCommuteAndSosCommands() {
        val cmdSos = RemoteCommandParser.parse("#1234: sos", "Sister", "WhatsApp")
        assertNotNull(cmdSos)
        assertEquals(RemoteActionType.TRIGGER_SOS, cmdSos?.actionType)

        val cmdCommute = RemoteCommandParser.parse("#1234: commute to Koramangala 30m", "Self", "SMS")
        assertNotNull(cmdCommute)
        assertEquals(RemoteActionType.START_COMMUTE_GUARD, cmdCommute?.actionType)
        assertTrue(cmdCommute?.rawGoal?.contains("Koramangala") == true)

        val cmdSafe = RemoteCommandParser.parse("#1234: arrived safely", "Self", "WhatsApp")
        assertNotNull(cmdSafe)
        assertEquals(RemoteActionType.STOP_COMMUTE_GUARD, cmdSafe?.actionType)
    }
}
