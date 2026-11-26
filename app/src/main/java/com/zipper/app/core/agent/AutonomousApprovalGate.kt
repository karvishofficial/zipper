package com.zipper.app.core.agent

import com.zipper.app.core.model.DistilledNode

/**
 * AUTONOMOUS APPROVAL GATE (Grok Bot Safety Architecture)
 *
 * Implements an Autonomous Safety Brake (AGENTS.md Section 4.3 & 5.1):
 * Halts autonomous execution before irreversible, sensitive actions
 * (financial checkout, payments, transfers, or permanent account deletions)
 * and prompts for human authorization.
 *
 * 100% universal across ANY unseen app:
 * - Does NOT use package name whitelists or app-specific rules.
 * - Inspects generic currency symbols, payment commitment semantics, and destructive action roles.
 * - Distinguishes non-binding actions (browsing, add-to-cart, view-cart) from binding commitments (pay now, place order).
 * - Zero cloud API cost (₹0), purely on-device semantic evaluation.
 */
data class ApprovalDecision(
    val isApprovalRequired: Boolean,
    val isPayment: Boolean = false,
    val isDestructive: Boolean = false,
    val detectedAmount: String? = null,
    val title: String = "",
    val prompt: String = "",
    val targetDetail: String = ""
)

object AutonomousApprovalGate {

    // Universal currency pattern (₹, $, €, £, ¥, INR, USD, EUR, GBP followed or preceded by numbers)
    private val CURRENCY_REGEX = Regex(
        """(?:[₹\$€£¥]|INR|USD|EUR|GBP)\s*([0-9]+(?:[,.][0-9]+)?)""",
        RegexOption.IGNORE_CASE
    )

    // Irreversible payment/checkout commitment affordances (strictly commit triggers, not general cart/browse)
    private val PAYMENT_COMMIT_REGEX = Regex(
        """\b(pay|checkout|place order|order now|proceed to pay|make payment|upi pay|pay now|complete purchase|subscribe|buy now|transfer money|send money|confirm payment)\b""",
        RegexOption.IGNORE_CASE
    )

    // Non-binding browsing/cart actions that should NOT trigger the approval gate
    private val NON_BINDING_CART_REGEX = Regex(
        """\b(add to cart|add item|view cart|go to cart|cart \(\d+\)|cart|see details|view details|menu|browse|search)\b""",
        RegexOption.IGNORE_CASE
    )

    // Irreversible destructive keywords
    private val DESTRUCTIVE_REGEX = Regex(
        """\b(delete account|format device|erase all|factory reset|wipe all|permanently delete|clear all data)\b""",
        RegexOption.IGNORE_CASE
    )

    /**
     * Evaluates whether a proposed agent action targets an irreversible or financial commitment.
     */
    fun evaluate(
        targetNode: DistilledNode?,
        action: AgentAction,
        packageName: String? = ""
    ): ApprovalDecision {
        if (targetNode == null) return ApprovalDecision(isApprovalRequired = false)
        if (action !is AgentAction.Tap && action !is AgentAction.Submit) {
            return ApprovalDecision(isApprovalRequired = false)
        }

        val text = targetNode.text.orEmpty()
        val desc = targetNode.contentDescription.orEmpty()
        val label = targetNode.semanticLabel.orEmpty()
        val resId = targetNode.resourceId.orEmpty().lowercase()
        val combinedText = "$text $desc $label".trim().lowercase()

        // 1. Skip if this is explicitly a non-binding browsing or add-to-cart action
        if (NON_BINDING_CART_REGEX.containsMatchIn(combinedText) && !combinedText.contains("proceed") && !combinedText.contains("place")) {
            return ApprovalDecision(isApprovalRequired = false)
        }

        // 2. Detect Destructive Actions
        val hasDestructiveKeyword = DESTRUCTIVE_REGEX.containsMatchIn(combinedText) ||
            DESTRUCTIVE_REGEX.containsMatchIn(resId)
        if (hasDestructiveKeyword) {
            val targetLabel = listOfNotNull(targetNode.text, targetNode.contentDescription, targetNode.semanticLabel)
                .firstOrNull { it.isNotBlank() } ?: "Destructive Action"

            return ApprovalDecision(
                isApprovalRequired = true,
                isDestructive = true,
                title = "Irreversible Action Gate",
                prompt = "The agent is about to execute a destructive or irreversible action. Human confirmation is required.",
                targetDetail = "Action: TAP on '$targetLabel'"
            )
        }

        // 3. Detect Financial & Payment Commitments
        val currencyMatch = CURRENCY_REGEX.find("$text $desc $label")
        val hasPaymentCommit = PAYMENT_COMMIT_REGEX.containsMatchIn(combinedText) ||
            PAYMENT_COMMIT_REGEX.containsMatchIn(resId)

        val isClickableOrButton = targetNode.isClickable ||
            targetNode.role.contains("BUTTON", ignoreCase = true) ||
            targetNode.className?.contains("Button", ignoreCase = true) == true

        if (hasPaymentCommit || (currencyMatch != null && isClickableOrButton && (combinedText.contains("pay") || combinedText.contains("order") || combinedText.contains("buy")))) {
            val amountStr = currencyMatch?.value?.trim() ?: ""
            val targetLabel = listOfNotNull(targetNode.text, targetNode.contentDescription, targetNode.semanticLabel)
                .firstOrNull { it.isNotBlank() } ?: "Payment / Order Button"

            return ApprovalDecision(
                isApprovalRequired = true,
                isPayment = true,
                detectedAmount = amountStr.ifBlank { null },
                title = "Payment Authorization Gate",
                prompt = "The agent is about to execute a financial checkout or order commitment. Autonomous Approval Gate requires human authorization before proceeding.",
                targetDetail = "Action: TAP on '$targetLabel'${if (amountStr.isNotBlank()) " (Amount: $amountStr)" else ""}"
            )
        }

        return ApprovalDecision(isApprovalRequired = false)
    }
}
