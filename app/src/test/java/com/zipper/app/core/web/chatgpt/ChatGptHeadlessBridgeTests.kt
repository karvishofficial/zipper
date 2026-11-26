package com.zipper.app.core.web.chatgpt

import com.zipper.app.core.web.WebAiConfig
import com.zipper.app.core.web.WebConfigManager
import com.zipper.app.core.web.WebLlmClient
import com.zipper.app.core.web.VerificationResult
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ChatGptHeadlessBridgeTests {

    @Test
    fun testEmptyPromptFailsFast() = runBlocking {
        // Calling ask with an empty string should fail fast without network interaction
        val result = ChatGptHeadlessBridge.ask(
            context = android.content.ContextWrapper(null),
            prompt = "   "
        )
        assertTrue(result.isFailure)
        assertEquals("Prompt cannot be empty.", result.exceptionOrNull()?.message)
    }

    @Test
    fun testWebConfigIncludesChatGptProvider() {
        assertTrue(WebConfigManager.SUPPORTED_PROVIDERS.contains("ChatGPT Web (Free)"))
        assertEquals(
            "gpt-4o-guest",
            WebConfigManager.getDefaultModelForProvider("ChatGPT Web (Free)")
        )
    }

    @Test
    fun testChatGptWebConfigIsConfiguredWithoutApiKey() {
        val config = WebAiConfig(
            provider = "ChatGPT Web (Free)",
            apiKey = "",
            model = "gpt-4o-guest"
        )
        // ChatGPT Web mode requires NO API key
        assertTrue(config.isConfigured)
    }

    @Test
    fun testVerifyKeySucceedsInstantlyForChatGptWeb() = runBlocking {
        val client = WebLlmClient()
        val config = WebAiConfig(
            provider = "ChatGPT Web (Free)",
            apiKey = "",
            model = "gpt-4o-guest"
        )
        val result = client.verifyKey(config)
        assertTrue(result is VerificationResult.Success)
        val success = result as VerificationResult.Success
        assertEquals("ChatGPT Web (Free)", success.provider)
        assertTrue(success.model.contains("No Login Required"))
    }
}
