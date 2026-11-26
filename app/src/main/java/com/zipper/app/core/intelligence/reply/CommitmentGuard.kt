package com.zipper.app.core.intelligence.reply

import java.util.Locale

enum class CommitmentRiskType {
    FINANCIAL,            // Money, UPI, GPay, borrow, lend, bank transfer
    LEGAL_CONTRACT,       // Sign, contract, agreement, approve deal
    SCHEDULE_PROMISE,     // Promising a specific time/meeting without user confirmation
    LOCATION_FABRICATION, // Hallucinating meeting places (e.g. Indiranagar) without user approval
    FABRICATED_LIE,       // Hospital, accident, fake disaster excuses
    SAFE                  // Neutral inquiry or safe conversation
}

data class CommitmentAssessment(
    val riskType: CommitmentRiskType,
    val isRisky: Boolean,
    val explanation: String,
    val safeAlternativeReply: String
)

/**
 * Commitment Guard.
 *
 * Strict anti-hallucination and liability barrier for Zipper's Ghost Twin.
 * Ensures the AI never promises money, never makes unauthorized legal/calendar commitments,
 * and never fabricates lies or fake excuses.
 */
object CommitmentGuard {

    private val FINANCIAL_KEYWORDS = setOf(
        "transfer", "send money", "lend", "borrow", "gpay", "phonepe", "paytm", "upi",
        "rupees", "rs", "₹", "account number", "account", "bank", "split"
    )

    private val SCHEDULE_PROMISE_KEYWORDS = setOf(
        "promise", "definitely come", "guarantee", "will reach at", "i will pay"
    )

    private val FABRICATED_EXCUSE_KEYWORDS = setOf(
        "hospital", "admitted", "accident", "funeral", "police", "stolen", "fracture", "emergency"
    )

    private val FABRICATED_LOCATION_PATTERNS = listOf(
        Regex("""(?i)\b(meet|hang\s*out|catch\s*up)\s+(at|in|near|around)\s+([a-zA-Z\s]+)"""),
        Regex("""(?i)\b(let'?s\s+meet|lets\s+meet|we\s+can\s+meet|shall\s+we\s+meet)\s+(at|in|near)\s+([a-zA-Z\s]+)"""),
        Regex("""(?i)\b(indiranagar|koramangala|mg\s*road|hsr|whitefield|church\s*street|starbucks|third\s*wave|ccd|cafe)\b""")
    )

    /**
     * Detects if an incoming message is proposing a meeting, asking where/when to meet,
     * or inquiring about hanging out.
     */
    fun isMeetingOrHangoutInquiry(text: String): Boolean {
        val lower = text.lowercase(Locale.ROOT)
        return lower.contains("where can we meet") ||
            lower.contains("where to meet") ||
            lower.contains("shall we meet") ||
            lower.contains("can we meet") ||
            lower.contains("let's meet") ||
            lower.contains("lets meet") ||
            lower.contains("want to meet") ||
            lower.contains("we need to meet") ||
            lower.contains("catch up") ||
            lower.contains("hang out") ||
            lower.contains("hangout") ||
            lower.contains("sigana") ||
            lower.contains("meet agana") ||
            lower.contains("free today") ||
            lower.contains("free tomorrow") ||
            lower.contains("are you free") ||
            lower.contains("r u free") ||
            lower.contains("ur free") ||
            lower.contains("scenes enu") ||
            lower.contains("what scenes") ||
            lower.contains("any plans") ||
            lower.contains("yavaga barthya") ||
            lower.contains("yavaga barodu") ||
            lower.contains("oorige barthya") ||
            lower.contains("when are you coming") ||
            lower.contains("when r u coming") ||
            lower.contains("when you coming") ||
            lower.contains("baralva") ||
            lower.contains("ivathe ba") ||
            lower.contains("bega ba") ||
            lower.contains("bega baa") ||
            lower.contains("banni") ||
            lower.contains("barthira")
    }

    fun assessRisk(
        incomingMessage: String,
        proposedReply: String? = null
    ): CommitmentAssessment {
        val lowerIn = incomingMessage.lowercase(Locale.ROOT)
        val lowerProp = proposedReply?.lowercase(Locale.ROOT).orEmpty()

        // 1. Check for Financial Requests
        val hasFinancialIn = FINANCIAL_KEYWORDS.any { lowerIn.contains(it) }
        val hasFinancialProp = FINANCIAL_KEYWORDS.any { lowerProp.contains(it) }
        if (hasFinancialIn || hasFinancialProp) {
            return CommitmentAssessment(
                riskType = CommitmentRiskType.FINANCIAL,
                isRisky = true,
                explanation = "Financial or money transfer discussion detected.",
                safeAlternativeReply = "In between something right now, will check my account and ping you once I'm free."
            )
        }

        // 2. Check for Fabricated Excuses / Lies
        val hasLie = FABRICATED_EXCUSE_KEYWORDS.any { lowerProp.contains(it) }
        if (hasLie) {
            return CommitmentAssessment(
                riskType = CommitmentRiskType.FABRICATED_LIE,
                isRisky = true,
                explanation = "Proposed reply contained unverified critical excuse.",
                safeAlternativeReply = "Tied up with some urgent work right now, will ping you once free."
            )
        }

        // 3. Check for Fabricated Meeting Locations / Venues (e.g. Indiranagar hallucination)
        if (isMeetingOrHangoutInquiry(incomingMessage)) {
            val hasLocationProposal = FABRICATED_LOCATION_PATTERNS.any { it.containsMatchIn(lowerProp) }
            if (hasLocationProposal) {
                return CommitmentAssessment(
                    riskType = CommitmentRiskType.LOCATION_FABRICATION,
                    isRisky = true,
                    explanation = "Proposed reply contained unverified meeting location or venue commitment.",
                    safeAlternativeReply = "Checking my schedule bro, let me see and ping you in a bit!"
                )
            }
        }

        // 4. Check for Schedule Commitments
        val hasPromise = SCHEDULE_PROMISE_KEYWORDS.any { lowerProp.contains(it) }
        if (hasPromise) {
            return CommitmentAssessment(
                riskType = CommitmentRiskType.SCHEDULE_PROMISE,
                isRisky = true,
                explanation = "Proposed reply contained firm schedule or payment guarantee.",
                safeAlternativeReply = "Not 100% sure about that time yet bro, will check my schedule and ping you shortly."
            )
        }

        return CommitmentAssessment(
            riskType = CommitmentRiskType.SAFE,
            isRisky = false,
            explanation = "Conversation is safe and non-committal.",
            safeAlternativeReply = proposedReply ?: "Hey, got your message. Will get back to you shortly."
        )
    }

    /**
     * Sanitizes any drafted reply before it is dispatched to the network.
     */
    fun sanitizeReply(incomingMessage: String, draftedReply: String): String {
        val assessment = assessRisk(incomingMessage, draftedReply)
        return if (assessment.isRisky) {
            assessment.safeAlternativeReply
        } else {
            draftedReply
        }
    }

    /**
     * Formats clarification text for the user voice HUD when someone asks for a commitment.
     */
    fun buildClarificationPrompt(contactName: String, incomingText: String): String {
        val clean = incomingText.take(60).trim()
        return "$contactName is asking: \"$clean\". What should I tell them?"
    }
}
