package com.zipper.app.core.voice

import android.content.Context
import android.media.AudioManager
import android.os.Handler
import android.os.Looper
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import java.util.Locale

data class ScreeningDecisionPill(
    val id: String,
    val label: String,
    val spokenResponse: String,
    val actionType: ScreeningAction = ScreeningAction.SPEAK_AND_HANGUP
)

enum class ScreeningAction {
    SPEAK_AND_HANGUP,
    RING_LOUDLY,
    TRANSFER_TO_USER
}

/**
 * Pixel-Style Autonomous Call Screener & Whisperer.
 *
 * Answers incoming unknown/busy calls via natural TTS, explains that Karthik is occupied,
 * asks the caller for their identity and reason, and surfaces 1-tap decision pills for instant disposal.
 */
object CallScreeningEngine {

    private const val TAG = "CallScreeningEngine"

    private var tts: TextToSpeech? = null
    private var isTtsReady = false

    val DEFAULT_DECISION_PILLS = listOf(
        ScreeningDecisionPill(
            id = "LEAVE_AT_DOOR",
            label = "📦 Leave at door / guard",
            spokenResponse = "Please leave the package at the door or with building security. Thank you!",
            actionType = ScreeningAction.SPEAK_AND_HANGUP
        ),
        ScreeningDecisionPill(
            id = "CALL_BACK_15M",
            label = "⏱️ Call back in 15 min",
            spokenResponse = "Karthik is currently occupied and will call you back in 15 minutes. Thank you!",
            actionType = ScreeningAction.SPEAK_AND_HANGUP
        ),
        ScreeningDecisionPill(
            id = "NOT_INTERESTED",
            label = "🚫 Not interested",
            spokenResponse = "We are not interested in this service or offer. Please remove this number. Thank you.",
            actionType = ScreeningAction.SPEAK_AND_HANGUP
        ),
        ScreeningDecisionPill(
            id = "RING_URGENT",
            label = "🚨 Urgent: Ring loudly",
            spokenResponse = "Connecting you now with high urgency.",
            actionType = ScreeningAction.RING_LOUDLY
        )
    )

    fun initializeTts(context: Context, onReady: (() -> Unit)? = null) {
        if (tts != null && isTtsReady) {
            onReady?.invoke()
            return
        }
        tts = TextToSpeech(context.applicationContext) { status ->
            if (status == TextToSpeech.SUCCESS) {
                tts?.language = Locale.US
                isTtsReady = true
                Log.i(TAG, "Call screening TTS initialized successfully.")
                onReady?.invoke()
            } else {
                Log.w(TAG, "Failed to initialize call screening TTS.")
            }
        }
    }

    /**
     * Speaks the opening greeting when the call is answered by Zipper.
     */
    fun speakOpeningGreeting(
        context: Context,
        onGreetingFinished: () -> Unit
    ) {
        initializeTts(context) {
            val greeting = "Hello. I am Karthik's Zipper AI assistant. He is currently occupied. Who is calling and what is this regarding?"
            speakUtterance(greeting, onGreetingFinished)
        }
    }

    /**
     * Executes the user's tapped decision pill.
     */
    fun executeDecision(
        context: Context,
        pill: ScreeningDecisionPill,
        onFinished: () -> Unit
    ) {
        Log.i(TAG, "Executing call screening decision: ${pill.id}")
        when (pill.actionType) {
            ScreeningAction.SPEAK_AND_HANGUP -> {
                speakUtterance(pill.spokenResponse) {
                    Handler(Looper.getMainLooper()).postDelayed({
                        onFinished()
                    }, 500)
                }
            }
            ScreeningAction.RING_LOUDLY -> {
                speakUtterance(pill.spokenResponse) {
                    try {
                        com.zipper.app.core.remote.EmergencyRinger.startRinging(context)
                    } catch (e: Exception) {
                        Log.w(TAG, "Alarm error: ${e.message}")
                    }
                    onFinished()
                }
            }
            ScreeningAction.TRANSFER_TO_USER -> {
                onFinished()
            }
        }
    }

    private fun speakUtterance(text: String, onDone: () -> Unit) {
        val activeTts = tts
        if (activeTts == null || !isTtsReady) {
            onDone()
            return
        }

        val utteranceId = "screen_${System.currentTimeMillis()}"
        activeTts.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) {}
            override fun onDone(utteranceId: String?) {
                Handler(Looper.getMainLooper()).post { onDone() }
            }
            override fun onError(utteranceId: String?) {
                Handler(Looper.getMainLooper()).post { onDone() }
            }
        })

        activeTts.speak(text, TextToSpeech.QUEUE_FLUSH, null, utteranceId)
    }

    fun release() {
        try {
            tts?.stop()
            tts?.shutdown()
            tts = null
            isTtsReady = false
        } catch (_: Exception) {}
    }
}
