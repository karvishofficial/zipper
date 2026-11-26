package com.zipper.app.core.plugins

import android.content.Context
import android.util.Log
import com.zipper.app.core.plugins.clipboard.ClipboardActionEngine
import com.zipper.app.core.plugins.expense.AutoExpenseLedgerEngine
import com.zipper.app.core.plugins.sentry.SentryGuardService
/**
 * Coordinates the zero-screen background plugins:
 * 1. Ghost Clipboard Intelligence Engine
 * 2. Silent UPI & Auto-Expense Ledger
 * 3. Library & Train Sentry Anti-Theft Guard
 *
 * Completely isolated from the native Autonomous Action Agent (karvish_v57.onnx).
 */
class BackgroundPluginManager private constructor(private val context: Context) {

    companion object {
        private const val TAG = "BackgroundPluginManager"

        @Volatile
        private var instance: BackgroundPluginManager? = null

        fun getInstance(context: Context): BackgroundPluginManager {
            return instance ?: synchronized(this) {
                instance ?: BackgroundPluginManager(context.applicationContext).also { instance = it }
            }
        }
    }

    val clipboardEngine: ClipboardActionEngine by lazy { ClipboardActionEngine(context) }
    val expenseLedger: AutoExpenseLedgerEngine by lazy { AutoExpenseLedgerEngine(context) }
    val sentryGuard: SentryGuardService by lazy { SentryGuardService(context) }

    private var isClipboardActive = false

    fun startClipboardIntelligence(onAction: ((com.zipper.app.core.plugins.clipboard.ClipboardAction) -> Unit)? = null) {
        if (!isClipboardActive) {
            clipboardEngine.onActionDetected = onAction
            clipboardEngine.startListening()
            isClipboardActive = true
            Log.i(TAG, "Ghost Clipboard plugin activated")
        }
    }

    fun stopClipboardIntelligence() {
        if (isClipboardActive) {
            clipboardEngine.stopListening()
            isClipboardActive = false
            Log.i(TAG, "Ghost Clipboard plugin stopped")
        }
    }

    fun armSentryGuard(gracePeriodMs: Long = 5000L, onAlarm: ((String) -> Unit)? = null) {
        sentryGuard.onAlarmTriggered = onAlarm
        sentryGuard.arm(gracePeriodMs)
        Log.i(TAG, "Sentry Guard armed")
    }

    fun disarmSentryGuard() {
        sentryGuard.disarm()
        Log.i(TAG, "Sentry Guard disarmed")
    }

    fun isSentryArmed(): Boolean {
        val st = sentryGuard.state
        return st == com.zipper.app.core.plugins.sentry.SentryState.ARMED || st == com.zipper.app.core.plugins.sentry.SentryState.ARMING
    }
}
