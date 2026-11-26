package com.zipper.app.core.intelligence.reply

import android.content.Context
import android.util.Log
import com.zipper.app.core.notification.NotificationStorageManager
import com.zipper.app.core.web.WebConfigManager
import com.zipper.app.core.web.WebLlmClient
import java.util.Locale

/**
 * Intelligent Reply Generator.
 * Synthesizes contact conversation context, user instructions, and calls the connected LLM
 * (Gemini, Groq, OpenAI, etc.) with strict anti-bot and closure guardrails.
 */
object SmartReplyGenerator {

    private const val TAG = "SmartReplyGenerator"
    private val webLlmClient = WebLlmClient()

    private val CLOSURE_PATTERNS = setOf(
        "ok", "k", "okay", "okk", "thx", "thanks", "thank you", "ty", "cya", "bye", "good night", "gn",
        "gm", "cool", "sure", "noted", "done", "alright", "super", "got it", "nice", "sounds good",
        "👍", "👌", "🙏", "❤️", "😊", "tc"
    )

    /**
     * Inspects incoming message text to determine if it is a simple closure/acknowledgment
     * where an auto-reply would be annoying or create an infinite loop.
     */
    fun isClosureOrAcknowledgment(text: String): Boolean {
        val clean = text.trim().lowercase(Locale.ROOT).replace(Regex("""[.,!?;:]+"""), "").trim()
        if (clean.isBlank()) return true
        if (clean in CLOSURE_PATTERNS) return true
        if (clean.length <= 4 && CLOSURE_PATTERNS.any { clean == it }) return true
        return false
    }

    /**
     * Generates an intelligent, human-like reply for a delegated contact.
     * Returns null if no reply is appropriate (e.g. closure message or error).
     */
    suspend fun generateReply(
        context: Context,
        contact: DelegatedContact,
        incomingMessage: String
    ): String? {
        val cleanMessage = incomingMessage.trim()
        if (cleanMessage.isBlank()) return null

        // 1. Check Closure Guardrail
        if (isClosureOrAcknowledgment(cleanMessage)) {
            Log.i(TAG, "Incoming message '$cleanMessage' from ${contact.contactName} is a closure acknowledgment. Skipping reply to avoid loop.")
            return null
        }

        val memory = DelegatedConversationMemory.getInstance(context)

        // 2. Record incoming message into persistent 2-way chat memory
        memory.recordMessage(contact.contactName, contact.contactName, cleanMessage)

        // 3. Fetch 2-Way Conversation Thread History (Anchor-preserved FIFO)
        val turns = memory.getHistory(contact.contactName, maxTurns = 18)
        val formattedHistory = turns.map { "${it.sender}: ${it.message}" }

        // 4. Attempt LLM Generation for Ghost-Writer
        // Real-time WhatsApp/SMS auto-replies require instant sub-second response times (<300ms).
        // If the user enabled headless ChatGPT Web for Webmind browsing, prioritize the fast direct API (Groq/Gemini) for messaging.
        val aiConfig = WebConfigManager(context).getConfig()
        var selectedReply: String? = null

        val (firstConfig, secondConfig) = if (aiConfig.provider.contains("ChatGPT Web", true)) {
            val fb = aiConfig.getFallbackConfig()
            if (fb != null && fb.apiKey.isNotBlank()) {
                Pair(fb, aiConfig)
            } else {
                Pair(aiConfig, null)
            }
        } else {
            Pair(aiConfig, aiConfig.getFallbackConfig())
        }

        val knowledgeBook = ContactKnowledgeBook.getInstance(context)
        val dossier = knowledgeBook.getDossier(contact.contactName)

        val memoryContext = if (dossier != null && dossier.memoryPoints.isNotEmpty()) {
            val mems = dossier.memoryPoints.joinToString("; ")
            val rel = dossier.relationship
            if (contact.customContext.isNotBlank()) {
                "${contact.customContext} | Relationship: $rel | Memory Dossier: $mems"
            } else {
                "Relationship: $rel | Memory Dossier: $mems"
            }
        } else {
            contact.customContext
        }

        val enrichedTone = if (dossier != null) {
            when {
                dossier.isElderOrFamily() && (dossier.isKannadaPreferred() || dossier.languageStyle == LanguageStyle.KANNADA_REGIONAL) ->
                    "${contact.personaTone} (MANDATORY: Contact is family/elder who speaks Kannada. Strictly respectful, polite Kannada. NEVER use 'bro', 'macha', or casual youth slang!)"
                dossier.isElderOrFamily() ->
                    "${contact.personaTone} (MANDATORY: Contact is an elder/family member. Strictly respectful, polite tone. NEVER use 'bro', 'macha', or casual youth slang!)"
                dossier.languageStyle == LanguageStyle.KANNADA_REGIONAL || dossier.isKannadaPreferred() ->
                    "${contact.personaTone} (Match Romanized Kannada / Kanglish, natural friendly Bengaluru regional vibe)"
                dossier.languageStyle == LanguageStyle.FORMAL_ENGLISH ->
                    "${contact.personaTone} (Strictly formal and respectful English, no slang)"
                dossier.languageStyle == LanguageStyle.BROKEN_BASIC_ENGLISH ->
                    "${contact.personaTone} (Match simple basic English, short common words, no complex grammar)"
                dossier.languageStyle == LanguageStyle.CASUAL_SHORT ->
                    "${contact.personaTone} (Ultra short 2-4 words texting, very brief)"
                else ->
                    "${contact.personaTone} (Casual friendly texting, Bangalore slang / Kanglish)"
            }
        } else {
            contact.personaTone
        }

        if (firstConfig.apiKey.isNotBlank()) {
            Log.i(TAG, "Generating chat reply via Primary Engine: ${firstConfig.provider} (${firstConfig.model}) for ${contact.contactName}...")
            val llmReply = webLlmClient.generateChatReply(
                contactName = contact.contactName,
                incomingMessage = cleanMessage,
                recentHistory = formattedHistory,
                userContext = memoryContext,
                personaTone = enrichedTone,
                config = firstConfig
            )
            if (!llmReply.isNullOrBlank() && !llmReply.equals("NO_REPLY", ignoreCase = true)) {
                Log.i(TAG, "Primary Engine generated reply for ${contact.contactName}: '$llmReply'")
                selectedReply = llmReply
            }
        }

        // 5. Dual-Engine Multi-Provider Fallback
        if (selectedReply == null && secondConfig != null && (secondConfig.apiKey.isNotBlank() || secondConfig.provider.contains("ChatGPT Web", true))) {
            Log.i(TAG, "Primary failed or unconfigured. Trying Secondary Fallback Engine: ${secondConfig.provider} (${secondConfig.model}) for ${contact.contactName}...")
            val fallbackReply = webLlmClient.generateChatReply(
                contactName = contact.contactName,
                incomingMessage = cleanMessage,
                recentHistory = formattedHistory,
                userContext = memoryContext,
                personaTone = enrichedTone,
                config = secondConfig
            )
            if (!fallbackReply.isNullOrBlank() && !fallbackReply.equals("NO_REPLY", ignoreCase = true)) {
                Log.i(TAG, "Secondary Fallback Engine generated reply for ${contact.contactName}: '$fallbackReply'")
                selectedReply = fallbackReply
            }
        }

        // 6. Natural On-Device Heuristic Fallback (Only if both LLMs are offline/unreachable)
        if (selectedReply == null) {
            Log.i(TAG, "Using on-device heuristic generator for ${contact.contactName}...")
            selectedReply = generateHeuristicFallback(contact, cleanMessage, turns)
        }

        // 7. Check if LLM requested a live confirmation call
        if (selectedReply.startsWith("CALL_USER:", ignoreCase = true)) {
            val parts = selectedReply.split("|", limit = 2)
            val reason = parts[0].trim()
            val cleanDeflection = buildDeflectionHoldMessage(contact, dossier, cleanMessage)
            return "$reason | $cleanDeflection"
        }

        // 7b. Meeting Proposal & Hangout Interceptor: Force CALL_USER so Karthik decides, ZERO hallucinated spots
        if (CommitmentGuard.isMeetingOrHangoutInquiry(cleanMessage) && !isMeetingAlreadyConfirmedByUser(cleanMessage, turns)) {
            Log.i(TAG, "🛡️ Meeting Inquiry Interceptor triggered: '${contact.contactName}' asked to meet / coordinate ('$cleanMessage'). Overriding to force CALL_USER.")
            val deflection = buildDeflectionHoldMessage(contact, dossier, cleanMessage)
            return "CALL_USER: ${contact.contactName} is asking to meet / coordinate plans | $deflection"
        }

        // 7c. Safety Interceptor: Prevent unauthorized commitment to meeting times
        val proposedTime = extractProposedTime(cleanMessage)
        if (proposedTime != null && !isTimeAlreadyConfirmedByUser(proposedTime, turns)) {
            Log.i(TAG, "🛡️ Safety Interceptor triggered: '${contact.contactName}' proposed unconfirmed time '$proposedTime'. Overriding LLM to force CALL_USER.")
            val deflection = buildDeflectionHoldMessage(contact, dossier, cleanMessage)
            return "CALL_USER: ${contact.contactName} wants to meet at $proposedTime | $deflection"
        }

        // 7d. Emergency & Urgent Meet Interceptor: Trigger CALL_USER immediately if emergency or urgent meet
        if (isUrgentOrEmergency(cleanMessage)) {
            Log.i(TAG, "🚨 Urgency Interceptor triggered: '${contact.contactName}' sent emergency/urgent meet request ('$cleanMessage'). Overriding to force CALL_USER.")
            val deflection = buildDeflectionHoldMessage(contact, dossier, cleanMessage)
            return "CALL_USER: ${contact.contactName} says: \"$cleanMessage\" (Urgent) | $deflection"
        }

        // 7e. Casual Location Inquiry Shield: Ensure zero hallucination / lying about location
        if (isCasualLocationInquiry(cleanMessage)) {
            val isElder = dossier?.isElderOrFamily() == true
            val isKan = dossier?.isKannadaPreferred() == true || dossier?.languageStyle == LanguageStyle.KANNADA_REGIONAL
            val isPro = contact.personaTone.contains("Professional", true) || contact.personaTone.contains("Formal", true)
            val honorific = dossier?.resolveHonorific() ?: ""
            val tag = if (honorific.isNotBlank()) " $honorific" else ""
            val privacyShieldReply = when {
                isElder && isKan -> "Swalpa kelasa meele horagade iddini$tag, enu aagilla."
                isElder -> "Outside for some work right now$tag, is everything okay?"
                isKan -> "Horagade idini bro, en samachara?"
                isPro -> "Currently away from my desk attending to some work. Is there something urgent?"
                else -> "Outside for some work bro, why what's up? En samachara?"
            }
            Log.i(TAG, "🛡️ Location Privacy Shield applied for '${contact.contactName}': '$privacyShieldReply'")
            selectedReply = privacyShieldReply
        }

        // 7f. Final CommitmentGuard Sanitization: Ensure zero hallucinated places (Indiranagar, etc.) or promises
        selectedReply = CommitmentGuard.sanitizeReply(cleanMessage, selectedReply)

        // 8. Record outgoing reply into persistent 2-way chat memory (clean message)
        memory.recordMessage(contact.contactName, "Karthik", selectedReply)

        // 9. Transparent AI badge option
        val finalReply = if (contact.appendAiBadge) {
            "$selectedReply\n\n— Sent via Zipper AI on behalf of Karthik"
        } else {
            selectedReply
        }

        return finalReply
    }

    fun buildDeflectionHoldMessage(contact: DelegatedContact, dossier: ContactDossier?, incomingMessage: String = ""): String {
        val msgLower = incomingMessage.lowercase(Locale.ROOT)
        val isKannada = dossier?.isKannadaPreferred() == true ||
            dossier?.languageStyle == LanguageStyle.KANNADA_REGIONAL ||
            msgLower.contains("yavaga") ||
            msgLower.contains("barthya") ||
            msgLower.contains("baralva") ||
            msgLower.contains("oorige") ||
            msgLower.contains("mysore")
        val isElder = dossier?.isElderOrFamily() == true
        val isPro = contact.personaTone.contains("Professional", true) || contact.personaTone.contains("Formal", true)
        val honorific = dossier?.resolveHonorific() ?: ""

        return when {
            isElder && isKannada -> "${if (honorific.isNotBlank()) "$honorific, " else ""}swalpa kelasa ide, schedule check maadi message/call madthini."
            isElder -> "${if (honorific.isNotBlank()) "$honorific, " else ""}swalpa busy iddini, check maadi call/message madthini."
            isKannada -> "Swalpa kelasa ide bro, schedule check maadi ping madthini!"
            isPro -> "I will check my calendar and get back to you shortly regarding our schedule."
            dossier?.languageStyle == LanguageStyle.BROKEN_BASIC_ENGLISH -> "Checking schedule bro. Ping in 10 min."
            dossier?.languageStyle == LanguageStyle.CASUAL_SHORT -> "Checking schedule, ping soon."
            else -> "Checking my schedule bro, let me see and ping you in a bit!"
        }
    }

    fun isMeetingAlreadyConfirmedByUser(message: String, history: List<ChatTurn>): Boolean {
        val userTurns = history.filter { it.sender.equals("Karthik", ignoreCase = true) }
        val confirmedRegex = Regex("""(?i)\b(i am free|i'm free|let's meet at|we can meet at|confirmed|see you at|works for me)\b""")
        return userTurns.any { confirmedRegex.containsMatchIn(it.message) }
    }

    internal fun generateHeuristicFallback(
        contact: DelegatedContact,
        incomingMessage: String,
        history: List<ChatTurn> = emptyList()
    ): String {
        val msgLower = incomingMessage.lowercase(Locale.ROOT)
        val ctx = contact.customContext.trim()
        val alreadySentContext = history.any { it.sender == "Karthik" && ctx.isNotBlank() && it.message.contains(ctx.take(12), ignoreCase = true) }
        val isProfessional = contact.personaTone.contains("Professional", ignoreCase = true) ||
            contact.personaTone.contains("Formal", ignoreCase = true)

        val knowledgeBook = ContactKnowledgeBook.getInstanceOrNull()
        val dossier = knowledgeBook?.getDossier(contact.contactName)
        val style = dossier?.languageStyle ?: LanguageStyle.CASUAL_SLANG

        // 1. Elder / Family Member Adaptation (Strictly respectful, no "bro")
        if (dossier?.isElderOrFamily() == true) {
            val honorific = dossier.resolveHonorific()
            val tag = if (honorific.isNotBlank()) " $honorific" else ""
            val isKan = dossier.isKannadaPreferred() || style == LanguageStyle.KANNADA_REGIONAL ||
                msgLower.contains("yavaga") || msgLower.contains("barthya") || msgLower.contains("oota") || msgLower.contains("mysore") || msgLower.contains("baralva")
            if (isKan) {
                return when {
                    msgLower.contains("oota") -> "Ha aaythu$tag, nindhu aaytha?"
                    msgLower.contains("yavaga") || msgLower.contains("barthya") || msgLower.contains("baralva") -> "Swalpa kelasa ide$tag, mugisi barthini."
                    msgLower.contains("elli") || msgLower.contains("where") -> "Swalpa kelasa meele horagade iddini$tag."
                    else -> "Swalpa kelasa ide$tag, free aagi call madthini."
                }
            } else {
                return when {
                    msgLower.contains("where") -> "Currently out attending to some work$tag. Is everything alright?"
                    msgLower.contains("when") -> "Occupied with work right now$tag, will follow up with you shortly."
                    else -> "Currently in the middle of some work$tag, will call you as soon as I am free."
                }
            }
        }

        // 2. Kannada / Regional Slang Adaptation
        if (style == LanguageStyle.KANNADA_REGIONAL || dossier?.isKannadaPreferred() == true) {
            return when {
                msgLower.contains("oota") -> "Ha aaythu bro, needhu aaytha?"
                msgLower.contains("madti") || msgLower.contains("doing") || msgLower.contains("wyd") -> if (ctx.isNotBlank() && !alreadySentContext) "$ctx bro, en samachara?" else "Kelasa madtha idini bro, en samachara?"
                msgLower.contains("yavaga") || msgLower.contains("barthya") || msgLower.contains("oorige") || msgLower.contains("baralva") -> "Ha barthini bro, swalpa work mugiskondu barthini."
                msgLower.contains("elli") || msgLower.contains("where") -> "Horagade idini bro, en samachara?"
                msgLower.contains("sigana") || msgLower.contains("meet") -> "Ha done bro, nale sigana 👍"
                msgLower.contains("call") -> "Call madoke aagalla bro, illi text madi."
                incomingMessage.endsWith("?") -> "Ha bro, check maadi helthini."
                else -> "Kelasa ide bro, aamele text madthini."
            }
        }

        // 3. Broken / Basic English Adaptation
        if (style == LanguageStyle.BROKEN_BASIC_ENGLISH) {
            return when {
                msgLower.contains("doing") || msgLower.contains("wyd") || msgLower.contains("up to") -> "In work bro. What happened?"
                msgLower.contains("where") || msgLower.contains("wr") || msgLower.contains("wer") || msgLower.contains("whr") -> "In work bro. Why what happened?"
                msgLower.contains("when") || msgLower.contains("free") || msgLower.contains("time") -> "Free in 20 min bro. Wait."
                msgLower.contains("call") -> "Cannot talk now bro, text me here."
                msgLower.contains("clg") || msgLower.contains("college") -> "Coming to clg bro. You coming?"
                msgLower.contains("notes") || msgLower.contains("project") -> "Sending notes in 15 mins bro."
                incomingMessage.endsWith("?") -> "Ha bro, checking and tell you."
                else -> "In work bro. Text me what happened."
            }
        }

        // 4. Casual Short Texting (2-4 words)
        if (style == LanguageStyle.CASUAL_SHORT) {
            return when {
                msgLower.contains("doing") || msgLower.contains("wyd") -> "Working bro, what's up?"
                msgLower.contains("where") || msgLower.contains("wr") -> "Outside bro, why?"
                msgLower.contains("when") || msgLower.contains("free") -> "Free in 20m."
                msgLower.contains("call") -> "Busy bro, text."
                incomingMessage.endsWith("?") -> "Yeah bro, ping later."
                else -> "Tied up bro."
            }
        }

        if (isProfessional) {
            return when {
                msgLower.contains("10am") || msgLower.contains("10 am") || msgLower.contains("11am") || msgLower.contains("11 am") || msgLower.contains("meeting") -> {
                    "I cannot confirm that specific time just yet. I will check my calendar and get back to you shortly."
                }
                msgLower.contains("who") -> {
                    "Hello, this is Karthik replying directly from my phone."
                }
                msgLower.contains("where are you") || msgLower.contains("where r u") || msgLower.contains("where ru") || msgLower.contains("elli iddiya") -> {
                    "Currently away from my desk attending to some work. Is there something urgent?"
                }
                msgLower.contains("gpay") || msgLower.contains("phonepe") || msgLower.contains("money") || msgLower.contains("transfer") -> {
                    "I am unable to process payments or transfers right now. I will check once available."
                }
                msgLower.contains("when") || msgLower.contains("reach") || msgLower.contains("free") -> {
                    if (ctx.isNotBlank() && !alreadySentContext) {
                        "Hello, I am currently occupied ($ctx). I will get back to you once available."
                    } else {
                        "I will be available shortly and will get back to you."
                    }
                }
                msgLower.contains("urgent") || msgLower.contains("important") -> {
                    "Received your message. I am currently attending to something and will review this as soon as possible."
                }
                ctx.isNotBlank() && !alreadySentContext -> {
                    "Hello, just to update you: $ctx"
                }
                incomingMessage.endsWith("?") -> {
                    "Understood. Let me check the details and update you shortly."
                }
                else -> {
                    "Understood. I will follow up with you as soon as I am free."
                }
            }
        }

        return when {
            msgLower.contains("nale sigana") || msgLower.contains("sigana") || msgLower.contains("sari agadre") -> {
                "Ha done bro, nale sigana 👍"
            }
            msgLower.contains("where are you") || msgLower.contains("where r u") || msgLower.contains("where ru") || msgLower.contains("elli iddiya") -> {
                "Outside for some work bro, why what's up? En samachara?"
            }
            msgLower.contains("gpay") || msgLower.contains("phonepe") || msgLower.contains("paytm") || msgLower.contains("send money") || (msgLower.contains("send") && (msgLower.contains("rs") || msgLower.contains("rupees"))) -> {
                "Can't access GPay/banking right now bro, caught up with work. Ping you later!"
            }
            msgLower.contains("oota") -> {
                "Ha aaythu bro, needhu aaytha?"
            }
            msgLower.contains("en samachara") || msgLower.contains("enu samachara") -> {
                "Enu illa bro, regular work anthe. Nindhu en samachara?"
            }
            msgLower.contains("10am") || msgLower.contains("10 am") || msgLower.contains("11am") || msgLower.contains("11 am") || msgLower.contains("meeting") -> {
                "Not 100% sure about that time yet bro, let me check my schedule once I'm free and ping you in a bit!"
            }
            msgLower.contains("clg") || msgLower.contains("college") -> {
                if (ctx.isNotBlank() && !alreadySentContext) {
                    "Ha bro, $ctx"
                } else {
                    "Ha bro, coming to clg! You also coming?"
                }
            }
            msgLower.contains("macha") || msgLower.contains("guru") || msgLower.contains("maga") -> {
                "En samachara macha, what's up? In the middle of some work, will ping you soon!"
            }
            msgLower.contains("who") -> {
                "Haha it's me Karthik! Just replying from my phone."
            }
            msgLower.contains("think") -> {
                "Yeah looks good bro, will check and catch up in a bit!"
            }
            msgLower.contains("when") || msgLower.contains("come") || msgLower.contains("reach") || msgLower.contains("free") -> {
                if (ctx.isNotBlank() && !alreadySentContext) {
                    "Hey bro, I'm tied up right now ($ctx). Will ping you once free!"
                } else {
                    "Should be free in a bit bro, will ping you soon!"
                }
            }
            msgLower.contains("call") -> {
                "Can't pick up right now bro, will call you in a few minutes!"
            }
            msgLower.contains("urgent") || msgLower.contains("important") -> {
                "Got your message bro! In the middle of something right now, will check this as soon as possible."
            }
            ctx.isNotBlank() && !alreadySentContext -> {
                "Hey bro! Just to let you know: $ctx"
            }
            incomingMessage.endsWith("?") -> {
                "Yeah bro, let me check and let you know in a bit!"
            }
            else -> {
                "Got it bro! In the middle of something right now, will ping you soon."
            }
        }
    }

    /**
     * Extracts proposed meeting times from message text (e.g. "5pm", "11:30 AM", "around 5", "5 gantege").
     */
    fun extractProposedTime(text: String): String? {
        val clean = text.trim()

        // 1. Matches 5pm, 5:30pm, 11am, 11:30 am, 5 PM, etc.
        val amPmRegex = Regex("""(?i)\b(?:around|at|by)?\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm))\b""")
        amPmRegex.find(clean)?.let { return it.groupValues[1].trim() }

        // 2. Matches "5 o'clock", "5 oclock"
        val oclockRegex = Regex("""(?i)\b(?:around|at|by)?\s*(\d{1,2}\s*(?:o['’]?clock))\b""")
        oclockRegex.find(clean)?.let { return it.groupValues[1].trim() }

        // 3. Matches Kannada/Kanglish "5 gantege", "5 gante", "5pm ge", "5:00 ge"
        val kanglishRegex = Regex("""(?i)\b(\d{1,2}(?::\d{2})?\s*(?:pm|am)?\s*(?:gantege|gante|ge))\b""")
        kanglishRegex.find(clean)?.let { return it.groupValues[1].trim() }

        // 4. Matches "around 5", "at 5" followed by question mark or end of proposal (e.g. "tomorrow at 5?", "around 5?")
        val aroundAtRegex = Regex("""(?i)\b(?:around|at)\s+([1-9]|1[0-2])\s*(?:\?|$|\b)""")
        aroundAtRegex.find(clean)?.let { return "${it.groupValues[1].trim()} (approx)" }

        return null
    }

    /**
     * Checks if Karthik has already explicitly stated or agreed to this time in conversation history.
     */
    fun isTimeAlreadyConfirmedByUser(proposedTime: String, history: List<ChatTurn>): Boolean {
        val digitMatch = Regex("""(\d{1,2})""").find(proposedTime)?.groupValues?.get(1) ?: return false
        val cleanProposed = proposedTime.lowercase(Locale.ROOT).replace(" ", "").replace("(approx)", "")

        return history.any { turn ->
            if (turn.sender != "Karthik") return@any false
            val rawLower = turn.message.lowercase(Locale.ROOT)
            val stripped = rawLower.replace(" ", "")

            val hasTime = stripped.contains(cleanProposed) ||
                stripped.contains("${digitMatch}pm") ||
                stripped.contains("${digitMatch}am") ||
                stripped.contains("at$digitMatch") ||
                stripped.contains("${digitMatch}works") ||
                stripped.contains("${digitMatch}is") ||
                stripped.contains("${digitMatch}:")

            val isPositive = rawLower.contains("work") ||
                rawLower.contains("fine") ||
                rawLower.contains("free") ||
                rawLower.contains("see you") ||
                rawLower.contains("confirm") ||
                rawLower.contains("done") ||
                rawLower.contains("ok") ||
                rawLower.contains("sure")

            val isNegative = rawLower.contains("can't") ||
                rawLower.contains("cant") ||
                rawLower.contains("not free") ||
                rawLower.contains("busy") ||
                rawLower.contains("tough") ||
                rawLower.contains("unable")

            hasTime && isPositive && !isNegative
        }
    }

    /**
     * Detects emergencies or urgent meet inquiries requiring a live phone call to Karthik.
     */
    fun isUrgentOrEmergency(text: String): Boolean {
        val lower = text.lowercase(Locale.ROOT)
        val hasEmergencyWord = lower.contains("emergency") ||
            lower.contains("accident") ||
            lower.contains("hospital") ||
            lower.contains("urgent") ||
            lower.contains("urgently") ||
            lower.contains("critical") ||
            lower.contains("help me") ||
            lower.contains("bega baa") ||
            lower.contains("bega ba") ||
            lower.contains("urgent ide") ||
            lower.contains("ivathe ba") ||
            lower.contains("ivathe") ||
            lower.contains("ivatte")

        val isWhereAreYou = lower.contains("where are you") ||
            lower.contains("where r u") ||
            lower.contains("where ru") ||
            lower.contains("elli iddiya") ||
            lower.contains("elli idira")

        val hasMeetIntent = lower.contains("meet") ||
            lower.contains("sigana") ||
            lower.contains("coming") ||
            lower.contains("reach") ||
            lower.contains("outside") ||
            lower.contains("gate") ||
            lower.contains("flat") ||
            lower.contains("house") ||
            lower.contains("home") ||
            lower.contains("see you now")

        return hasEmergencyWord || (isWhereAreYou && hasMeetIntent)
    }

    /**
     * Detects casual location questions where the privacy shield should be applied instead of inventing fake places.
     */
    fun isCasualLocationInquiry(text: String): Boolean {
        val lower = text.lowercase(Locale.ROOT)
        return (lower.contains("where are you") ||
            lower.contains("where r u") ||
            lower.contains("where ru") ||
            lower.contains("elli iddiya") ||
            lower.contains("elli idira")) && !isUrgentOrEmergency(text)
    }
}

