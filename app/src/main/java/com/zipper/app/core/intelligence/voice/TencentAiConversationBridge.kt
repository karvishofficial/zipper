package com.zipper.app.core.intelligence.voice

import android.content.Context
import android.util.Log
import com.zipper.app.core.intelligence.reply.ContactKnowledgeBook
import com.zipper.app.core.intelligence.reply.LanguageStyle
import com.zipper.app.core.web.WebConfigManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.util.Locale

/**
 * High-level Conversational Voice Bridge.
 * Binds Contact Knowledge Dossiers, Commitment Guard, and LLM providers to live Tencent RTC voice calls.
 */
object TencentAiConversationBridge {

    private const val TAG = "TencentVoiceBridge"
    private val scope = CoroutineScope(Dispatchers.IO)

    data class VoiceBridgeSession(
        val roomId: String,
        val callerName: String,
        val webLink: String,
        val deflectionText: String,
        val createdAtMs: Long = System.currentTimeMillis()
    )

    /**
     * Prepares and initiates a live voice AI session for a caller who missed or declined a call.
     */
    suspend fun prepareCallBridge(
        context: Context,
        callerName: String,
        callerNumber: String = ""
    ): VoiceBridgeSession? {
        val cleanName = callerName.trim().ifBlank { "Friend" }
        val rtcManager = TencentRtcManager.getInstance(context)
        val config = rtcManager.getConfig()
        if (!config.isConfigured || !config.isEnabled) {
            Log.w(TAG, "Tencent RTC is not configured or disabled.")
            return null
        }

        val book = ContactKnowledgeBook.getInstance(context)
        val dossier = book.getDossier(cleanName)

        val roomId = "zipper_${cleanName.lowercase(Locale.ROOT).replace(Regex("[^a-z0-9]"), "_").take(10)}_${System.currentTimeMillis() % 100000}"

        // 1. Determine Welcome Message & Tone based on Dossier
        val style = dossier?.languageStyle ?: LanguageStyle.CASUAL_SLANG
        val rel = dossier?.relationship ?: "Friend"
        val isLandlordOrElder = rel.contains("landlord", true) || rel.contains("elder", true) || rel.contains("uncle", true)
        val isFamily = rel.contains("mom", true) || rel.contains("dad", true) || rel.contains("family", true)

        val welcomeMessage = when {
            isLandlordOrElder -> "Hello Uncle, Karthik is currently occupied with some work. He asked me to take your call. How may I assist you?"
            isFamily -> "Hey, Karthik is tied up with work right now and asked me to take your call. Is everything okay?"
            style == LanguageStyle.BROKEN_BASIC_ENGLISH -> "Hey bro, Karthik in work now. Tell me what happened."
            style == LanguageStyle.CASUAL_SHORT -> "Hey bro, Karthik busy right now. What's up?"
            else -> "Hey $cleanName, what's up bro! Karthik is caught up with some work right now and asked me to take your call. What's happening?"
        }

        // 2. Build Personalized LLM System Prompt
        val memorySummary = if (dossier != null && dossier.memoryPoints.isNotEmpty()) {
            "KNOWN FACTS ABOUT $cleanName: " + dossier.memoryPoints.joinToString("; ")
        } else {
            "KNOWN FACTS: Close acquaintance of Karthik."
        }

        val systemPrompt = """
            You are Karthik's AI Voice Twin speaking on a LIVE, real-time audio phone call with $cleanName.
            Speak naturally in the FIRST PERSON as Karthik's voice assistant representing him.
            
            RECIPIENT DOSSIER:
            - Name: $cleanName
            - Relationship with Karthik: $rel
            - $memorySummary
            
            VOICE TONE RULES:
            ${if (isLandlordOrElder) {
                "- Maintain a formal, courteous, respectful tone. Never use slang or 'bro'."
            } else {
                "- Casual, warm Bangalore friendly slang ('bro', 'yeah bro', 'done bro', 'chill'). Speak with energetic, authentic friend vibes."
            }}
            
            STRICT COMMITMENT GUARDRAILS (CRITICAL):
            1. NEVER promise money, UPI, GPay, PhonePe, cash, or financial transfers. Say: "Can't access GPay or banking right now bro, caught up with work. Ping Karthik directly later!"
            2. NEVER fabricate lies or fake hospital/accident excuses.
            3. NEVER confirm a meeting time or schedule guarantee on your own. Say: "Not 100% sure about that time yet bro, let me check my schedule once I'm free and ping you in a bit!"
            4. Keep responses concise and conversational (1 to 2 sentences per turn), optimized for speech.
        """.trimIndent()

        // 3. Resolve LLM Provider Keys
        val aiConfig = WebConfigManager(context).getConfig()
        val llmKey = aiConfig.apiKey.ifBlank { aiConfig.fallbackApiKey }
        val llmUrl = if (aiConfig.provider.contains("Groq", true)) {
            "https://api.groq.com/openai/v1/chat/completions"
        } else if (aiConfig.provider.contains("OpenAI", true)) {
            "https://api.openai.com/v1/chat/completions"
        } else {
            "https://api.groq.com/openai/v1/chat/completions"
        }
        val llmModel = if (aiConfig.provider.contains("Groq", true)) {
            "llama-3.3-70b-versatile"
        } else {
            "gpt-4o-mini"
        }

        // 4. Launch AI Agent in TRTC Room
        val result = rtcManager.startAiConversation(
            roomId = roomId,
            aiUserId = "zipper_voice_twin",
            welcomeMessage = welcomeMessage,
            systemPrompt = systemPrompt,
            llmApiKey = llmKey,
            llmApiUrl = llmUrl,
            llmModel = llmModel
        )

        if (result.isFailure) {
            Log.e(TAG, "Failed to spin up cloud AI voice agent: ${result.exceptionOrNull()?.message}")
        } else {
            Log.i(TAG, "⚡ AI Voice Agent joined TRTC room '$roomId' for '$cleanName'")
        }

        // 5. Generate Caller Web Link
        val webLink = rtcManager.generateCallerWebLink(roomId, cleanName)

        val deflection = when {
            isLandlordOrElder -> "Hello Uncle, I am occupied with work right now. You can speak directly to my AI assistant here: $webLink"
            isFamily -> "Hey, tied up right now! Talk to my voice assistant here: $webLink"
            else -> "Hey $cleanName! Saw your call, in the middle of something right now. Click here to talk directly to my AI voice twin: $webLink"
        }

        return VoiceBridgeSession(
            roomId = roomId,
            callerName = cleanName,
            webLink = webLink,
            deflectionText = deflection
        )
    }
}
