package com.zipper.app.core.agent

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.util.Log
import com.zipper.app.core.remote.RemoteFileResolver
import java.util.Locale

/**
 * Cross-App Intent & Workflow Chaining Engine.
 *
 * Real-world human tasks span multiple applications (e.g., getting an address
 * from WhatsApp and opening Google Maps, or reading an OTP from SMS and entering
 * it into a banking/delivery app).
 *
 * This engine segments compound cross-app goals, extracts source entities,
 * and passes the context to the target app's execution session.
 *
 * 100% on-device, ₹0 cloud APIs.
 */
object CrossAppChainer {

    private const val TAG = "CrossAppChainer"

    data class ChainedStep(
        val stepOrder: Int,
        val appPackage: String,
        val actionDescription: String,
        val isSourceExtraction: Boolean,
        val payloadSlot: String? = null
    )

    data class ChainedPlan(
        val isChainedGoal: Boolean,
        val sourceApp: String?,
        val targetApp: String?,
        val extractedEntity: String?,
        val targetIntentUri: Uri?,
        val steps: List<ChainedStep>,
        val summary: String
    )

    /**
     * Analyzes a user goal to check if it represents a cross-app workflow.
     */
    fun analyzeGoal(goal: String): ChainedPlan {
        val lower = goal.lowercase(Locale.ROOT).trim()

        // 1. WhatsApp -> Google Maps Address Chaining
        // E.g.: "Open address from WhatsApp in Google Maps" or "Navigate to address from Rohit on WhatsApp"
        val isAddressToMaps = (lower.contains("address") || lower.contains("location")) &&
            (lower.contains("whatsapp") || lower.contains("chat") || lower.contains("message")) &&
            (lower.contains("maps") || lower.contains("navigate") || lower.contains("route") || lower.contains("uber") || lower.contains("ola"))

        if (isAddressToMaps) {
            val contactMatch = Regex("""(?:from|of)\s+([a-zA-Z]+)""").find(goal)
            val sender = contactMatch?.groupValues?.get(1)?.trim() ?: "Contact"

            val steps = listOf(
                ChainedStep(
                    stepOrder = 1,
                    appPackage = "com.whatsapp",
                    actionDescription = "Extract physical address or location link from chat with $sender",
                    isSourceExtraction = true,
                    payloadSlot = "destination_address"
                ),
                ChainedStep(
                    stepOrder = 2,
                    appPackage = "com.google.android.apps.maps",
                    actionDescription = "Launch Maps and plot turn-by-turn route to destination_address",
                    isSourceExtraction = false,
                    payloadSlot = "destination_address"
                )
            )

            return ChainedPlan(
                isChainedGoal = true,
                sourceApp = "com.whatsapp",
                targetApp = "com.google.android.apps.maps",
                extractedEntity = null,
                targetIntentUri = Uri.parse("geo:0,0?q="),
                steps = steps,
                summary = "Extract address from WhatsApp ($sender) ➔ Navigate on Google Maps"
            )
        }

        // 2. SMS/Notifications -> App OTP Injection Chaining
        // E.g.: "Read OTP from messages and login to Blinkit"
        val isOtpToApp = (lower.contains("otp") || lower.contains("code") || lower.contains("verification")) &&
            (lower.contains("message") || lower.contains("sms")) &&
            (lower.contains("zepto") || lower.contains("blinkit") || lower.contains("swiggy") || lower.contains("zomato") || lower.contains("amazon") || lower.contains("flipkart"))

        if (isOtpToApp) {
            val targetAppPkg = when {
                lower.contains("zepto") -> "com.zeptonow.android"
                lower.contains("blinkit") -> "com.grofers.customerapp"
                lower.contains("swiggy") -> "in.swiggy.android"
                lower.contains("zomato") -> "com.application.zomato"
                lower.contains("amazon") -> "in.amazon.mShop.android.shopping"
                else -> "com.flipkart.android"
            }

            val steps = listOf(
                ChainedStep(
                    stepOrder = 1,
                    appPackage = "com.google.android.apps.messaging",
                    actionDescription = "Read latest 4-6 digit numeric OTP from recent bank/service SMS",
                    isSourceExtraction = true,
                    payloadSlot = "otp_code"
                ),
                ChainedStep(
                    stepOrder = 2,
                    appPackage = targetAppPkg,
                    actionDescription = "Inject otp_code into verification pin fields and verify login",
                    isSourceExtraction = false,
                    payloadSlot = "otp_code"
                )
            )

            return ChainedPlan(
                isChainedGoal = true,
                sourceApp = "com.google.android.apps.messaging",
                targetApp = targetAppPkg,
                extractedEntity = null,
                targetIntentUri = null,
                steps = steps,
                summary = "Extract OTP from SMS ➔ Inject into target app ($targetAppPkg)"
            )
        }

        return ChainedPlan(
            isChainedGoal = false,
            sourceApp = null,
            targetApp = null,
            extractedEntity = null,
            targetIntentUri = null,
            steps = emptyList(),
            summary = "Single-app goal"
        )
    }

    /**
     * Generates a direct Android intent to jumpstart the target app with the pre-filled entity.
     */
    fun createTargetLaunchIntent(context: Context, plan: ChainedPlan, resolvedValue: String): Intent? {
        val targetPkg = plan.targetApp ?: return null

        if (targetPkg == "com.google.android.apps.maps") {
            val encodedQuery = Uri.encode(resolvedValue)
            val uri = Uri.parse("geo:0,0?q=$encodedQuery")
            return Intent(Intent.ACTION_VIEW, uri).apply {
                setPackage("com.google.android.apps.maps")
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
        }

        return context.packageManager.getLaunchIntentForPackage(targetPkg)?.apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }
    }
}
