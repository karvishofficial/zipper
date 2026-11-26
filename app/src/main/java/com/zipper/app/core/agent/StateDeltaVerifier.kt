package com.zipper.app.core.agent

import android.util.Log
import com.zipper.app.core.model.DistilledUiState
import java.util.Locale

/**
 * State Delta & Execution Verification Engine.
 *
 * Implements senior closed-loop evaluation:
 * - Detects negative/rejected UI transitions (out of stock, invalid coupon, payment failed,
 *   server error, location not serviceable, invalid credentials) and prevents false-positive
 *   task completions.
 * - Confirms observable positive milestone evidence (order placed, message sent, added to cart,
 *   alarm scheduled) so the agent terminates with genuine physical evidence.
 *
 * 100% On-Device, ₹0 Cloud APIs.
 */
object StateDeltaVerifier {

    private const val TAG = "StateDeltaVerifier"

    enum class VerificationStatus {
        POSITIVE_PROGRESS,
        REJECTED_ERROR,
        UNCHANGED_NOOP,
        TERMINATION_CONFIRMED
    }

    data class VerificationResult(
        val status: VerificationStatus,
        val summary: String,
        val errorCategory: String? = null,
        val confirmedMilestone: String? = null,
        val rawErrorMessage: String? = null
    )

    // Negative state patterns that indicate action rejection or operational failure
    private val NEGATIVE_ERROR_PATTERNS = listOf(
        "out of stock" to "OUT_OF_STOCK",
        "currently unavailable" to "OUT_OF_STOCK",
        "item unavailable" to "OUT_OF_STOCK",
        "sold out" to "OUT_OF_STOCK",
        "payment failed" to "PAYMENT_FAILED",
        "transaction declined" to "PAYMENT_FAILED",
        "payment declined" to "PAYMENT_FAILED",
        "insufficient balance" to "PAYMENT_FAILED",
        "invalid coupon" to "PROMO_REJECTED",
        "coupon expired" to "PROMO_REJECTED",
        "promo code not valid" to "PROMO_REJECTED",
        "invalid promo code" to "PROMO_REJECTED",
        "location not serviceable" to "LOCATION_UNSERVICEABLE",
        "we do not deliver to this location" to "LOCATION_UNSERVICEABLE",
        "delivery unavailable" to "LOCATION_UNSERVICEABLE",
        "server error" to "SERVER_ERROR",
        "something went wrong" to "SERVER_ERROR",
        "please try again later" to "SERVER_ERROR",
        "connection timed out" to "NETWORK_ERROR",
        "no internet connection" to "NETWORK_ERROR",
        "incorrect password" to "AUTH_FAILED",
        "invalid credentials" to "AUTH_FAILED",
        "invalid otp" to "AUTH_FAILED",
        "wrong otp" to "AUTH_FAILED"
    )

    // Positive completion milestone signatures
    private val POSITIVE_MILESTONE_PATTERNS = listOf(
        "order placed" to "ORDER_PLACED",
        "order confirmed" to "ORDER_CONFIRMED",
        "booking confirmed" to "BOOKING_CONFIRMED",
        "payment successful" to "PAYMENT_SUCCESSFUL",
        "transaction successful" to "TRANSACTION_SUCCESSFUL",
        "added to cart" to "CART_UPDATED",
        "item added" to "CART_UPDATED",
        "alarm set for" to "ALARM_SCHEDULED",
        "message sent" to "MESSAGE_SENT"
    )

    /**
     * Evaluates the observation transition from preState to postState given the attempted action.
     */
    fun verifyTransition(
        goal: String,
        preState: DistilledUiState,
        postState: DistilledUiState,
        action: AgentAction
    ): VerificationResult {
        val lowerGoal = goal.lowercase(Locale.ROOT).trim()

        val postTexts = postState.nodes.mapNotNull {
            it.text?.takeIf { t -> t.isNotBlank() } ?: it.contentDescription?.takeIf { c -> c.isNotBlank() }
        }
        val postCombined = postTexts.joinToString(" ").lowercase(Locale.ROOT)

        val preTexts = preState.nodes.mapNotNull {
            it.text?.takeIf { t -> t.isNotBlank() } ?: it.contentDescription?.takeIf { c -> c.isNotBlank() }
        }
        val preCombined = preTexts.joinToString(" ").lowercase(Locale.ROOT)

        // 1. Detect Negative Errors that were NOT already present before the action
        for ((pattern, category) in NEGATIVE_ERROR_PATTERNS) {
            if (postCombined.contains(pattern) && !preCombined.contains(pattern)) {
                val matchedSnippet = postTexts.firstOrNull { it.contains(pattern, ignoreCase = true) } ?: pattern
                Log.w(TAG, "[REJECTED_TRANSITION] Detected error '$pattern' ($category): '$matchedSnippet'")
                return VerificationResult(
                    status = VerificationStatus.REJECTED_ERROR,
                    summary = "Action failed with environment error: $matchedSnippet",
                    errorCategory = category,
                    rawErrorMessage = matchedSnippet
                )
            }
        }

        // 2. Detect Positive Termination Milestones
        for ((pattern, milestone) in POSITIVE_MILESTONE_PATTERNS) {
            if (postCombined.contains(pattern)) {
                Log.i(TAG, "[MILESTONE_CONFIRMED] Observable physical evidence confirmed '$milestone'")
                return VerificationResult(
                    status = VerificationStatus.TERMINATION_CONFIRMED,
                    summary = "Observable physical evidence confirmed: $pattern",
                    confirmedMilestone = milestone
                )
            }
        }

        // 3. Detect Effective UI Mutation
        val nodeCountDelta = kotlin.math.abs(postState.nodes.size - preState.nodes.size)
        val packageChanged = preState.packageName != postState.packageName
        val hasTextChanged = postCombined != preCombined

        return if (packageChanged || hasTextChanged || nodeCountDelta > 0) {
            VerificationResult(
                status = VerificationStatus.POSITIVE_PROGRESS,
                summary = "UI state evolved (+${nodeCountDelta} node delta)"
            )
        } else {
            VerificationResult(
                status = VerificationStatus.UNCHANGED_NOOP,
                summary = "Action caused zero observable UI mutations (potential stale click or dead touch)"
            )
        }
    }
}
