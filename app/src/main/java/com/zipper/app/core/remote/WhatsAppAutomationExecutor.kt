package com.zipper.app.core.remote

import android.content.Context
import android.os.Bundle
import android.util.Log
import android.view.accessibility.AccessibilityNodeInfo
import com.zipper.app.accessibility.KarvishAccessibilityService
import kotlinx.coroutines.delay
import kotlinx.coroutines.withTimeoutOrNull

/**
 * Autonomous WhatsApp Motor for Karvish.
 *
 * Operates WhatsApp end-to-end:
 * 1. Launches WhatsApp foreground session.
 * 2. Clicks Search affordance.
 * 3. Enters recipient name and selects matching chat.
 * 4. Types target message into input field.
 * 5. Clicks Send CTA.
 */
object WhatsAppAutomationExecutor {

    private const val TAG = "WhatsAppAutomation"
    private const val WHATSAPP_PKG = "com.whatsapp"

    /**
     * Directly sends a message to a specific phone number via WhatsApp deep-link.
     * This bypasses UI searching, pre-fills the message text, taps send, and restores home.
     */
    suspend fun executeSendToPhone(
        context: Context,
        normalizedPhone: String,
        message: String,
        restoreHomeAfterSend: Boolean = true
    ): Boolean {
        val a11y = KarvishAccessibilityService.instance
        val cleanPhone = normalizedPhone.replace("+", "").replace(" ", "").replace("-", "")

        try {
            val uri = android.net.Uri.parse(
                "https://api.whatsapp.com/send?phone=$cleanPhone&text=${android.net.Uri.encode(message)}"
            )
            val intent = android.content.Intent(android.content.Intent.ACTION_VIEW, uri).apply {
                setPackage(WHATSAPP_PKG)
                addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK or android.content.Intent.FLAG_ACTIVITY_CLEAR_TOP)
            }
            context.startActivity(intent)
        } catch (e: Exception) {
            Log.e(TAG, "Error starting WhatsApp direct intent: ${e.message}")
            return false
        }

        if (a11y == null) {
            Log.w(TAG, "KarvishAccessibilityService not connected; launched WhatsApp direct intent.")
            return true
        }

        // Wait up to 3.5s for chat screen to render with Send button
        val sendSuccess = withTimeoutOrNull(3500L) {
            var sent = false
            while (!sent) {
                delay(400L)
                sent = findAndClickSendButton(a11y)
            }
            sent
        } ?: false

        if (sendSuccess) {
            delay(500L)
            if (restoreHomeAfterSend) {
                a11y.performGlobalAction(android.accessibilityservice.AccessibilityService.GLOBAL_ACTION_HOME)
            }
        }

        return sendSuccess
    }

    /**
     * Directly shares a file to a specific recipient or phone number via WhatsApp ACTION_SEND intent.
     * Uses FileProvider to grant safe content URI permissions, targets WhatsApp,
     * triggers auto-send via Accessibility, and restores the home screen.
     */
    suspend fun executeSendFile(
        context: Context,
        targetPhone: String?,
        file: java.io.File,
        mimeType: String = "application/pdf",
        restoreHomeAfterSend: Boolean = true
    ): Boolean {
        val a11y = KarvishAccessibilityService.instance
        if (!file.exists() || !file.canRead()) {
            Log.e(TAG, "File does not exist or cannot be read: ${file.absolutePath}")
            return false
        }

        try {
            val authority = "${context.packageName}.fileprovider"
            val contentUri = androidx.core.content.FileProvider.getUriForFile(context, authority, file)

            val shareIntent = android.content.Intent(android.content.Intent.ACTION_SEND).apply {
                type = mimeType
                putExtra(android.content.Intent.EXTRA_STREAM, contentUri)
                if (!targetPhone.isNullOrBlank()) {
                    val cleanPhone = targetPhone.replace("+", "").replace(" ", "").replace("-", "")
                    putExtra("jid", "$cleanPhone@s.whatsapp.net")
                }
                setPackage(WHATSAPP_PKG)
                addFlags(
                    android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION or
                    android.content.Intent.FLAG_ACTIVITY_NEW_TASK or
                    android.content.Intent.FLAG_ACTIVITY_CLEAR_TOP
                )
            }
            context.startActivity(shareIntent)
            Log.i(TAG, "Dispatched WhatsApp file share intent for ${file.name}")
        } catch (e: Exception) {
            Log.e(TAG, "Error starting WhatsApp file share intent: ${e.message}")
            return false
        }

        if (a11y == null) {
            Log.w(TAG, "KarvishAccessibilityService not connected; file share intent dispatched.")
            return true
        }

        // Wait up to 4s for WhatsApp send preview screen to render with Send button
        val sendSuccess = withTimeoutOrNull(4000L) {
            var sent = false
            while (!sent) {
                delay(400L)
                sent = findAndClickSendButton(a11y)
            }
            sent
        } ?: false

        if (sendSuccess) {
            delay(600L)
            if (restoreHomeAfterSend) {
                a11y.performGlobalAction(android.accessibilityservice.AccessibilityService.GLOBAL_ACTION_HOME)
            }
        }

        return sendSuccess
    }

    suspend fun executeSendMessage(
        context: Context,
        recipient: String,
        message: String
    ): Boolean {
        val a11y = KarvishAccessibilityService.instance
        if (a11y == null) {
            Log.e(TAG, "KarvishAccessibilityService is not connected")
            return false
        }

        Log.i(TAG, "Initiating autonomous WhatsApp message to '$recipient': \"$message\"")

        // 0. Show Karvish Control Overlay (Floating HUD bottom card) on whatever screen the phone is on!
        com.zipper.app.accessibility.KarvishControlOverlay.show(
            context = a11y,
            goal = "Send WhatsApp: \"$message\" to $recipient"
        )

        // 1. Launch WhatsApp
        com.zipper.app.accessibility.KarvishControlOverlay.updateTurnProgress(
            context = a11y,
            turn = 1,
            maxTurns = 6,
            actionName = "Open App",
            statusText = "Waking WhatsApp in foreground..."
        )
        val launched = a11y.launchApp(WHATSAPP_PKG)
        if (!launched) {
            Log.e(TAG, "Failed to launch WhatsApp")
            com.zipper.app.accessibility.KarvishControlOverlay.hide()
            return false
        }

        // Wait up to 3s for WhatsApp to appear in foreground
        val inForeground: Boolean = withTimeoutOrNull(3000L) {
            var found = false
            while (!found) {
                val root = a11y.rootInActiveWindow
                if (root?.packageName?.toString() == WHATSAPP_PKG) {
                    found = true
                    break
                }
                delay(200L)
            }
            found
        } ?: false

        if (!inForeground) {
            Log.w(TAG, "WhatsApp did not become foreground in time, proceeding anyway...")
        }
        delay(600L)

        // 2. Locate and Click Search Button
        com.zipper.app.accessibility.KarvishControlOverlay.updateTurnProgress(
            context = a11y,
            turn = 2,
            maxTurns = 6,
            actionName = "Click Search",
            statusText = "Opening contact search bar..."
        )
        val searchClicked = findAndClickSearchButton(a11y)
        if (!searchClicked) {
            Log.e(TAG, "Could not find or click WhatsApp search button")
            com.zipper.app.accessibility.KarvishControlOverlay.hide()
            return false
        }
        delay(600L)

        // 3. Type Recipient Name into Search Input
        com.zipper.app.accessibility.KarvishControlOverlay.updateTurnProgress(
            context = a11y,
            turn = 3,
            maxTurns = 6,
            actionName = "Type Text",
            statusText = "Searching for contact '$recipient'..."
        )
        val searchTyped = findAndTypeText(a11y, recipient)
        if (!searchTyped) {
            Log.e(TAG, "Could not type recipient '$recipient' into search")
            com.zipper.app.accessibility.KarvishControlOverlay.hide()
            return false
        }
        delay(1200L) // Wait for contact search results to load

        // 4. Click the matching contact row
        com.zipper.app.accessibility.KarvishControlOverlay.updateTurnProgress(
            context = a11y,
            turn = 4,
            maxTurns = 6,
            actionName = "Select Contact",
            statusText = "Opening chat conversation with '$recipient'..."
        )
        val contactClicked = findAndClickContactRow(a11y, recipient)
        if (!contactClicked) {
            Log.e(TAG, "Could not find matching contact row for '$recipient'")
            com.zipper.app.accessibility.KarvishControlOverlay.showBlockedBanner(
                message = "Could not find '$recipient' in WhatsApp contacts",
                onResume = { com.zipper.app.accessibility.KarvishControlOverlay.hide() },
                onAbort = { com.zipper.app.accessibility.KarvishControlOverlay.hide() }
            )
            return false
        }
        delay(1000L) // Wait for chat window to open

        // 5. Type the Message in Chat Entry
        com.zipper.app.accessibility.KarvishControlOverlay.updateTurnProgress(
            context = a11y,
            turn = 5,
            maxTurns = 6,
            actionName = "Type Message",
            statusText = "Typing message: \"$message\"..."
        )
        val messageTyped = findAndTypeChatMessage(a11y, message)
        if (!messageTyped) {
            Log.e(TAG, "Could not type message into WhatsApp chat entry")
            com.zipper.app.accessibility.KarvishControlOverlay.hide()
            return false
        }
        delay(400L)

        // 6. Click Send Button
        com.zipper.app.accessibility.KarvishControlOverlay.updateTurnProgress(
            context = a11y,
            turn = 6,
            maxTurns = 6,
            actionName = "Send Message",
            statusText = "Tapping Send button..."
        )
        val sendClicked = findAndClickSendButton(a11y)
        if (!sendClicked) {
            Log.e(TAG, "Could not click WhatsApp send button")
            com.zipper.app.accessibility.KarvishControlOverlay.hide()
            return false
        }

        Log.i(TAG, "🎉 Successfully sent WhatsApp message to '$recipient'!")
        com.zipper.app.accessibility.KarvishControlOverlay.showCompleted(
            milestones = listOf(
                "Woke WhatsApp",
                "Found '$recipient'",
                "Sent: \"$message\""
            )
        ) {
            com.zipper.app.accessibility.KarvishControlOverlay.hide()
        }
        delay(2000L)
        com.zipper.app.accessibility.KarvishControlOverlay.hide()
        return true
    }

    private suspend fun findAndClickSearchButton(a11y: KarvishAccessibilityService): Boolean {
        val root = a11y.rootInActiveWindow ?: return false
        val allNodes = a11y.collectAllNodes(root)

        // Find by content description "Search" or id containing "search"
        val searchBtn = allNodes.firstOrNull { node ->
            val desc = node.contentDescription?.toString()?.lowercase().orEmpty()
            val resId = node.viewIdResourceName?.lowercase().orEmpty()
            desc == "search" || resId.contains("menuitem_search") || resId.contains("search_button")
        }

        return if (searchBtn != null) {
            searchBtn.performAction(AccessibilityNodeInfo.ACTION_CLICK) ||
                clickNodeCenter(a11y, searchBtn)
        } else {
            false
        }
    }

    private fun findAndTypeText(a11y: KarvishAccessibilityService, text: String): Boolean {
        val root = a11y.rootInActiveWindow ?: return false
        val allNodes = a11y.collectAllNodes(root)

        val searchInput = allNodes.firstOrNull { node ->
            node.isEditable ||
                node.viewIdResourceName?.contains("search_src_text", ignoreCase = true) == true ||
                node.viewIdResourceName?.contains("search_input", ignoreCase = true) == true
        }

        if (searchInput != null) {
            val args = Bundle().apply {
                putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, text)
            }
            return searchInput.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args)
        }
        return false
    }

    private suspend fun findAndClickContactRow(a11y: KarvishAccessibilityService, recipient: String): Boolean {
        val root = a11y.rootInActiveWindow ?: return false
        val allNodes = a11y.collectAllNodes(root)
        val targetLower = recipient.lowercase()

        // Find node containing recipient name
        val match = allNodes.firstOrNull { node ->
            val text = node.text?.toString()?.lowercase().orEmpty()
            val desc = node.contentDescription?.toString()?.lowercase().orEmpty()
            (text.contains(targetLower) || desc.contains(targetLower)) &&
                !node.isEditable &&
                !node.viewIdResourceName?.contains("search", ignoreCase = true).orFalse()
        }

        if (match != null) {
            var curr: AccessibilityNodeInfo? = match
            while (curr != null) {
                if (curr.isClickable) {
                    if (curr.performAction(AccessibilityNodeInfo.ACTION_CLICK)) return true
                    break
                }
                curr = curr.parent
            }
            return clickNodeCenter(a11y, match)
        }
        return false
    }

    private fun findAndTypeChatMessage(a11y: KarvishAccessibilityService, message: String): Boolean {
        val root = a11y.rootInActiveWindow ?: return false
        val allNodes = a11y.collectAllNodes(root)

        val entryNode = allNodes.firstOrNull { node ->
            node.viewIdResourceName?.contains("entry", ignoreCase = true) == true ||
                (node.isEditable && node.viewIdResourceName?.contains("search", ignoreCase = true) != true)
        }

        if (entryNode != null) {
            val args = Bundle().apply {
                putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, message)
            }
            return entryNode.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args)
        }
        return false
    }

    private suspend fun findAndClickSendButton(a11y: KarvishAccessibilityService): Boolean {
        val root = a11y.rootInActiveWindow ?: return false
        val allNodes = a11y.collectAllNodes(root)

        val sendBtn = allNodes.firstOrNull { node ->
            val desc = node.contentDescription?.toString()?.lowercase().orEmpty()
            val resId = node.viewIdResourceName?.lowercase().orEmpty()
            desc == "send" || resId.endsWith(":id/send") || resId.contains("send_btn")
        }

        return if (sendBtn != null) {
            sendBtn.performAction(AccessibilityNodeInfo.ACTION_CLICK) ||
                clickNodeCenter(a11y, sendBtn)
        } else {
            false
        }
    }

    private suspend fun clickNodeCenter(a11y: KarvishAccessibilityService, node: AccessibilityNodeInfo): Boolean {
        val rect = android.graphics.Rect()
        node.getBoundsInScreen(rect)
        if (rect.width() > 0 && rect.height() > 0) {
            return a11y.clickAtCoordinates(rect.centerX().toFloat(), rect.centerY().toFloat())
        }
        return false
    }

    private fun Boolean?.orFalse(): Boolean = this ?: false
}
