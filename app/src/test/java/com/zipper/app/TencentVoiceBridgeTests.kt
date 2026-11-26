package com.zipper.app

import com.zipper.app.core.intelligence.voice.TencentCloudSigner
import com.zipper.app.core.intelligence.voice.TencentRtcConfig
import com.zipper.app.core.intelligence.voice.TrtcUserSigGenerator
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class TencentVoiceBridgeTests {

    @Test
    fun testTencentCloudSignerHeaderGeneration() {
        val headers = TencentCloudSigner.signRequest(
            secretId = "TEST_TENCENT_SECRET_ID",
            secretKey = "TEST_TENCENT_SECRET_KEY",
            service = "trtc",
            host = "trtc.intl.tencentcloudapi.com",
            action = "StartAIConversation",
            version = "2019-07-22",
            region = "ap-singapore",
            payloadJson = "{\"SdkAppId\":20049543}",
            timestampSeconds = 1700000000L
        )

        assertNotNull(headers["Authorization"])
        assertTrue("Authorization header must use TC3-HMAC-SHA256", headers["Authorization"]!!.startsWith("TC3-HMAC-SHA256 Credential=TEST_TENCENT_SECRET_ID/"))
        assertEquals("StartAIConversation", headers["X-TC-Action"])
        assertEquals("2019-07-22", headers["X-TC-Version"])
        assertEquals("ap-singapore", headers["X-TC-Region"])
        assertEquals("1700000000", headers["X-TC-Timestamp"])
        assertEquals("trtc.intl.tencentcloudapi.com", headers["Host"])
    }

    @Test
    fun testTrtcUserSigGeneratorProducesValidUrlSafeToken() {
        val userSig = TrtcUserSigGenerator.genUserSig(
            sdkAppId = 20049543L,
            secretKey = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            userId = "zipper_voice_twin",
            expireSeconds = 86400L
        )

        assertNotNull(userSig)
        assertTrue("UserSig must not be empty", userSig.isNotBlank())
        assertFalse("UserSig must be URL-safe (no '+' characters)", userSig.contains("+"))
        assertFalse("UserSig must be URL-safe (no '/' characters)", userSig.contains("/"))
    }

    @Test
    fun testTencentRtcConfigDefaults() {
        val unconfigured = TencentRtcConfig(sdkAppId = 0L, sdkSecretKey = "", secretId = "", secretKey = "")
        assertFalse("Blank credentials must mark config as unconfigured", unconfigured.isConfigured)

        val configured = TencentRtcConfig(
            sdkAppId = 20049543L,
            sdkSecretKey = "test_sdk_secret",
            secretId = "test_secret_id",
            secretKey = "test_secret_key"
        )
        assertEquals(20049543L, configured.sdkAppId)
        assertEquals("ap-singapore", configured.region)
        assertTrue("Credentials must mark config as configured", configured.isConfigured)
        assertTrue("Tencent RTC must be enabled by default", configured.isEnabled)
    }
}
