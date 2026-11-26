package com.zipper.app

import com.zipper.app.telephony.KarvishCallManager
import com.zipper.app.telephony.voip.VoipCallDetails
import com.zipper.app.telephony.voip.VoipCallResponse
import com.zipper.app.telephony.voip.VoipConfig
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Unit tests for Zipper's VoIPBin CPaaS cloud telephony and AI calling integration.
 */
class VoipBinTelephonyTests {

    @Test
    fun testVoipConfigDefaults() {
        val config = VoipConfig()
        assertEquals("https://api.voipbin.net/v1.0", config.apiUrl)
        assertEquals("8caad583-40ce-4374-a656-09c56b06571b", config.assistantId)
        assertEquals("sip:direct.c95b295237fe@sip.voipbin.net", config.sipUri)
        assertFalse("Should be disabled by default until user activates", config.isEnabled)
        assertFalse("Should not be configured without API key", config.isConfigured)
    }

    @Test
    fun testVoipConfigWithApiKey() {
        val config = VoipConfig(apiKey = "test_voipbin_key_abc123", isEnabled = true)
        assertTrue("Should be configured when API key is present", config.isConfigured)
        assertTrue(config.isEnabled)
    }

    @Test
    fun testVoipCallResponseParsing() {
        val mockJson = """
            {
                "id": "call_987654321",
                "status": "initiated",
                "phone_number": "+919876543210",
                "created_at": "2026-10-15T18:00:00Z"
            }
        """.trimIndent()

        val json = JSONObject(mockJson)
        val response = VoipCallResponse(
            callId = json.optString("id"),
            status = json.optString("status"),
            phoneNumber = json.optString("phone_number"),
            createdAt = json.optString("created_at"),
            rawResponse = mockJson
        )

        assertEquals("call_987654321", response.callId)
        assertEquals("initiated", response.status)
        assertEquals("+919876543210", response.phoneNumber)
    }

    @Test
    fun testVoipCallDetailsActiveStatus() {
        val details = VoipCallDetails(
            callId = "call_123",
            status = "in-progress",
            durationSeconds = 42,
            transcript = "Zipper: Hello Rajamma! Rajamma: Hey, can we meet at 6:30?",
            summary = "",
            isEnded = false
        )

        assertFalse(details.isEnded)
        assertEquals("in-progress", details.status)
        assertEquals(42, details.durationSeconds)
        assertTrue(details.transcript.contains("Rajamma"))
    }

    @Test
    fun testVoipCallDetailsCompletedStatus() {
        val details = VoipCallDetails(
            callId = "call_123",
            status = "completed",
            durationSeconds = 65,
            transcript = "Zipper: Confirmed for 7:15 PM.",
            summary = "Meeting rescheduled to 7:15 PM.",
            isEnded = true
        )

        assertTrue(details.isEnded)
        assertEquals("completed", details.status)
        assertEquals("Meeting rescheduled to 7:15 PM.", details.summary)
    }

    @Test
    fun testCallGoalParsing() {
        val goal1 = "Call Rajamma and say meeting at 6:30 PM"
        val parsed1 = KarvishCallManager.parseCallGoal(goal1)
        assertNotNull(parsed1)
        assertEquals("Rajamma", parsed1?.contactQuery)
        assertEquals("meeting at 6:30 PM", parsed1?.message)

        val goal2 = "Phone Amma: coming home now"
        val parsed2 = KarvishCallManager.parseCallGoal(goal2)
        assertNotNull(parsed2)
        assertEquals("Amma", parsed2?.contactQuery)
        assertEquals("coming home now", parsed2?.message)

        val goal3 = "Dial +919449213667 and say what time works"
        val parsed3 = KarvishCallManager.parseCallGoal(goal3)
        assertNotNull(parsed3)
        assertEquals("+919449213667", parsed3?.contactQuery)
        assertEquals("what time works", parsed3?.message)
    }

    @Test
    fun testVoipOutboundRequestBodyStructure() {
        val phoneNumber = "+919876543210"
        val assistantId = "8caad583-40ce-4374-a656-09c56b06571b"
        val goal = "Ask Rajamma if 6:30 PM works"
        val targetName = "Rajamma"

        val body = JSONObject().apply {
            put("source", JSONObject().apply {
                put("type", "ai")
                put("target", assistantId)
            })
            put("destinations", org.json.JSONArray().apply {
                put(JSONObject().apply {
                    put("type", "tel")
                    put("target", phoneNumber)
                })
            })
            put("variables", JSONObject().apply {
                put("goal", goal)
                put("contact_name", targetName)
                put("user_name", "Karthik")
            })
        }

        val source = body.getJSONObject("source")
        assertEquals("ai", source.getString("type"))
        assertEquals(assistantId, source.getString("target"))

        val dest = body.getJSONArray("destinations").getJSONObject(0)
        assertEquals("tel", dest.getString("type"))
        assertEquals(phoneNumber, dest.getString("target"))

        val vars = body.getJSONObject("variables")
        assertEquals(goal, vars.getString("goal"))
        assertEquals(targetName, vars.getString("contact_name"))
        assertEquals("Karthik", vars.getString("user_name"))
    }
}
