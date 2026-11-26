package com.zipper.app.telephony

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioDeviceInfo
import android.media.AudioManager
import android.os.Build
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.telecom.Call
import android.telecom.InCallService
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.util.Locale

/**
 * Telecom InCallService for autonomous cellular calling.
 * Detects STATE_ACTIVE (John answering), enables speakerphone,
 * plays carrier VAD wakeup chime, speaks the message via local TTS,
 * and disconnects the call upon completion.
 */
class KarvishInCallService : InCallService(), TextToSpeech.OnInitListener {

    companion object {
        private const val TAG = "KarvishInCallService"
        private const val UTTERANCE_ID_KARVISH = "karvish_call_utterance"

        private val _callState = MutableStateFlow<KarvishCallState>(KarvishCallState.Idle)
        val callState: StateFlow<KarvishCallState> = _callState.asStateFlow()

        // Configuration injected before placing the call
        @Volatile var pendingContactName: String = "Contact"
        @Volatile var pendingMessage: String = ""
        @Volatile var speechRate: Float = 0.92f
        @Volatile var targetVolumePercent: Float = 0.78f
        @Volatile var useWakeupChime: Boolean = true

        fun reset() {
            _callState.value = KarvishCallState.Idle
            pendingContactName = "Contact"
            pendingMessage = ""
        }
    }

    private var activeCall: Call? = null
    private var tts: TextToSpeech? = null
    private var isTtsReady = false
    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
    private lateinit var audioManager: AudioManager

    private val callCallback = object : Call.Callback() {
        override fun onStateChanged(call: Call, state: Int) {
            super.onStateChanged(call, state)
            val contact = pendingContactName
            val number = call.details?.handle?.schemeSpecificPart.orEmpty()

            Log.i(TAG, "Call state changed: $state for $contact ($number)")

            when (state) {
                Call.STATE_DIALING, Call.STATE_CONNECTING -> {
                    _callState.value = KarvishCallState.Dialing(contact, number)
                }
                Call.STATE_RINGING -> {
                    _callState.value = KarvishCallState.Ringing(contact, number)
                }
                Call.STATE_ACTIVE -> {
                    _callState.value = KarvishCallState.Connected(contact, pendingMessage)
                    onCallConnected(call)
                }
                Call.STATE_DISCONNECTED -> {
                    val cause = call.details?.disconnectCause?.reason ?: "Call Ended"
                    Log.i(TAG, "Call disconnected: $cause")
                    if (_callState.value !is KarvishCallState.Completed) {
                        _callState.value = KarvishCallState.Completed(contact, "Call disconnected: $cause")
                    }
                    teardownAudio()
                }
            }
        }
    }

    override fun onCreate() {
        super.onCreate()
        audioManager = getSystemService(Context.AUDIO_SERVICE) as AudioManager
        tts = TextToSpeech(applicationContext, this)
        Log.i(TAG, "KarvishInCallService created and initializing TTS.")
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            tts?.let { engine ->
                val result = engine.setLanguage(Locale.US)
                if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                    Log.w(TAG, "Default US language not supported for TTS, trying default Locale")
                    engine.language = Locale.getDefault()
                }
                engine.setSpeechRate(speechRate)
                engine.setPitch(1.0f)

                val audioAttributes = AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_ALARM)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
                engine.setAudioAttributes(audioAttributes)

                engine.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                    override fun onStart(utteranceId: String?) {
                        Log.i(TAG, "TTS utterance started.")
                        _callState.value = KarvishCallState.Speaking(pendingContactName, pendingMessage, 0)
                    }

                    override fun onDone(utteranceId: String?) {
                        Log.i(TAG, "TTS utterance finished successfully.")
                        _callState.value = KarvishCallState.Speaking(pendingContactName, pendingMessage, 100)
                        serviceScope.launch {
                            // Brief pause after speech before disconnecting
                            delay(1200L)
                            disconnectCall()
                        }
                    }

                    @Deprecated("Deprecated in Java")
                    override fun onError(utteranceId: String?) {
                        Log.e(TAG, "TTS utterance failed.")
                        serviceScope.launch {
                            delay(1000L)
                            disconnectCall()
                        }
                    }

                    override fun onError(utteranceId: String?, errorCode: Int) {
                        Log.e(TAG, "TTS utterance error code: $errorCode")
                        serviceScope.launch {
                            delay(1000L)
                            disconnectCall()
                        }
                    }
                })
            }
            isTtsReady = true
            Log.i(TAG, "Local TextToSpeech engine initialized and ready.")
        } else {
            Log.e(TAG, "Failed to initialize local TextToSpeech engine.")
        }
    }

    override fun onCallAdded(call: Call) {
        super.onCallAdded(call)
        activeCall = call
        call.registerCallback(callCallback)
        Log.i(TAG, "Call added to KarvishInCallService: ${call.details?.handle}")
    }

    override fun onCallRemoved(call: Call) {
        super.onCallRemoved(call)
        call.unregisterCallback(callCallback)
        if (activeCall == call) {
            activeCall = null
        }
        Log.i(TAG, "Call removed from KarvishInCallService.")
    }

    private fun onCallConnected(call: Call) {
        serviceScope.launch {
            try {
                // 1. Configure audio routing & speakerphone
                audioManager.mode = AudioManager.MODE_IN_CALL

                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                    val speakerDevice = audioManager.availableCommunicationDevices.firstOrNull {
                        it.type == AudioDeviceInfo.TYPE_BUILTIN_SPEAKER
                    }
                    if (speakerDevice != null) {
                        audioManager.setCommunicationDevice(speakerDevice)
                    }
                }
                @Suppress("DEPRECATION")
                audioManager.isSpeakerphoneOn = true

                // Lock all relevant hardware streams to 100% MAX to flood primary mic at 1.5cm distance
                val maxAlarmVol = audioManager.getStreamMaxVolume(AudioManager.STREAM_ALARM)
                audioManager.setStreamVolume(AudioManager.STREAM_ALARM, maxAlarmVol, 0)

                val maxCallVol = audioManager.getStreamMaxVolume(AudioManager.STREAM_VOICE_CALL)
                audioManager.setStreamVolume(AudioManager.STREAM_VOICE_CALL, maxCallVol, 0)

                val maxVol = audioManager.getStreamMaxVolume(AudioManager.STREAM_MUSIC)
                audioManager.setStreamVolume(AudioManager.STREAM_MUSIC, maxVol, 0)

                // 2. Buffer delay (400ms) for recipient to say "Hello" & hardware stabilization
                delay(450L)

                // 3. Play Carrier VAD Wakeup Chime (forces carrier voice gate open)
                if (useWakeupChime) {
                    CarrierWakeupChime.play()
                    delay(150L)
                }

                // 4. Speak the autonomous payload
                val messageText = pendingMessage.ifBlank {
                    "Hello ${pendingContactName}, this is an automated message from Karthik: I will come in a few minutes. Thank you."
                }

                speakAutonomousMessage(messageText)
            } catch (e: Exception) {
                Log.e(TAG, "Error during in-call speech flow: ${e.message}", e)
                disconnectCall()
            }
        }
    }

    private fun speakAutonomousMessage(text: String) {
        if (!isTtsReady || tts == null) {
            Log.e(TAG, "TTS not ready when trying to speak autonomous message.")
            disconnectCall()
            return
        }

        val params = Bundle().apply {
            putFloat(TextToSpeech.Engine.KEY_PARAM_VOLUME, 1.0f)
            putInt(TextToSpeech.Engine.KEY_PARAM_STREAM, AudioManager.STREAM_ALARM)
        }

        val result = tts?.speak(text, TextToSpeech.QUEUE_FLUSH, params, UTTERANCE_ID_KARVISH)
        Log.i(TAG, "TTS speak dispatched on STREAM_ALARM: '$text' (Result: $result)")
    }

    private fun disconnectCall() {
        try {
            activeCall?.let { c ->
                Log.i(TAG, "Disconnecting cellular call upon milestone completion...")
                c.disconnect()
            }
            _callState.value = KarvishCallState.Completed(
                pendingContactName,
                "Message successfully delivered to $pendingContactName."
            )
        } catch (e: Exception) {
            Log.w(TAG, "Error disconnecting call: ${e.message}")
        } finally {
            teardownAudio()
        }
    }

    private fun teardownAudio() {
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                audioManager.clearCommunicationDevice()
            }
            @Suppress("DEPRECATION")
            audioManager.isSpeakerphoneOn = false
            audioManager.mode = AudioManager.MODE_NORMAL
        } catch (_: Exception) {}
    }

    override fun onDestroy() {
        super.onDestroy()
        activeCall?.unregisterCallback(callCallback)
        teardownAudio()
        tts?.stop()
        tts?.shutdown()
        serviceScope.cancel()
        Log.i(TAG, "KarvishInCallService destroyed.")
    }
}
