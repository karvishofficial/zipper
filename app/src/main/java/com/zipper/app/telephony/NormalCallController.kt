package com.zipper.app.telephony

import android.content.Context
import android.content.Intent
import android.media.AudioAttributes
import android.media.AudioDeviceInfo
import android.media.AudioManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.telecom.TelecomManager
import android.util.Log
import android.view.accessibility.AccessibilityNodeInfo
import com.zipper.app.accessibility.KarvishAccessibilityService
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.util.Locale

/**
 * NormalCallController executes autonomous cellular calls using standard Android
 * ACTION_CALL intent without requiring Zipper to be the default dialer.
 *
 * Guaranteed Behaviors:
 * 1. NEVER speaks while the phone is ringing — waits for the recipient to answer (verified via active call timer).
 * 2. Articulates slowly (0.85x speech rate) for maximum clarity over cellular networks.
 * 3. Repeats the message 3 times with 1.8s pauses before auto-disconnecting.
 */
object NormalCallController : TextToSpeech.OnInitListener {
    private const val TAG = "NormalCallController"
    private const val UTTERANCE_ID = "karvish_normal_call"
    private const val TOTAL_REPEATS = 3
    private const val SLOW_SPEECH_RATE = 0.85f

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
    private var monitorJob: Job? = null

    private val _callState = MutableStateFlow<KarvishCallState>(KarvishCallState.Idle)
    val callState: StateFlow<KarvishCallState> = _callState.asStateFlow()

    private var tts: TextToSpeech? = null
    private var isTtsReady = false

    private var activeTargetName: String = "Contact"
    private var activeMessage: String = ""
    private var isCallInProgress = false
    private var hasSpokenForThisCall = false
    private var currentRepeatCount = 0

    fun init(context: Context) {
        if (tts == null) {
            tts = TextToSpeech(context.applicationContext, this)
        }
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            tts?.let { engine ->
                val result = engine.setLanguage(Locale.US)
                if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                    engine.language = Locale.getDefault()
                }
                engine.setSpeechRate(SLOW_SPEECH_RATE)
                engine.setPitch(1.0f)

                val audioAttributes = AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_ALARM)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
                engine.setAudioAttributes(audioAttributes)

                engine.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                    override fun onStart(utteranceId: String?) {
                        Log.i(TAG, "TTS utterance started for: $activeTargetName (Repeat ${currentRepeatCount + 1}/$TOTAL_REPEATS)")
                        val progress = ((currentRepeatCount + 1) * 100) / TOTAL_REPEATS
                        _callState.value = KarvishCallState.Speaking(
                            activeTargetName,
                            "[Repeat ${currentRepeatCount + 1}/$TOTAL_REPEATS] $activeMessage",
                            progress
                        )
                    }

                    override fun onDone(utteranceId: String?) {
                        Log.i(TAG, "TTS utterance completed (Repeat ${currentRepeatCount + 1}/$TOTAL_REPEATS).")
                        onUtteranceFinished()
                    }

                    @Deprecated("Deprecated in Java")
                    override fun onError(utteranceId: String?) {
                        Log.e(TAG, "TTS utterance error.")
                        onUtteranceFinished()
                    }

                    override fun onError(utteranceId: String?, errorCode: Int) {
                        Log.e(TAG, "TTS utterance error code: $errorCode")
                        onUtteranceFinished()
                    }
                })
            }
            isTtsReady = true
            Log.i(TAG, "TextToSpeech initialized with slow speech rate ($SLOW_SPEECH_RATE).")
        }
    }

    private fun onUtteranceFinished() {
        currentRepeatCount++
        Log.i(TAG, "Completed repetition $currentRepeatCount of $TOTAL_REPEATS")

        if (currentRepeatCount < TOTAL_REPEATS && isCallInProgress) {
            scope.launch {
                // Pause 1.8 seconds between repetitions so listener clearly digests every sentence
                delay(1800L)
                if (isCallInProgress) {
                    val nextPayload = when (currentRepeatCount) {
                        1 -> "Repeating message: $activeMessage."
                        2 -> "Final repeat: $activeMessage. Thank you."
                        else -> "$activeMessage."
                    }
                    speakUtterance(nextPayload)
                }
            }
        } else {
            // All 3 repetitions finished
            Log.i(TAG, "All $TOTAL_REPEATS repetitions completed. Disconnecting call in 2 seconds...")
            _callState.value = KarvishCallState.Speaking(activeTargetName, activeMessage, 100)
            scope.launch {
                delay(2000L)
                endCurrentCall()
            }
        }
    }

    /**
     * Initiates a normal cellular SIM call to any contact or phone number.
     */
    fun startNormalCall(
        context: Context,
        targetName: String,
        targetNumber: String,
        messagePayload: String
    ): Boolean {
        init(context)
        activeTargetName = targetName
        activeMessage = messagePayload
        isCallInProgress = true
        hasSpokenForThisCall = false
        currentRepeatCount = 0

        _callState.value = KarvishCallState.Dialing(targetName, targetNumber)

        val cleanNumber = targetNumber.replace(Regex("[^0-9+]"), "").trim()
        val callIntent = Intent(Intent.ACTION_CALL).apply {
            data = Uri.parse("tel:$cleanNumber")
            flags = Intent.FLAG_ACTIVITY_NEW_TASK
        }

        return try {
            Log.i(TAG, "Launching normal ACTION_CALL to $cleanNumber ($targetName)")
            context.startActivity(callIntent)

            // Start monitoring call progress
            monitorCallProgress(context)
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start normal call: ${e.message}", e)
            _callState.value = KarvishCallState.Failed(targetName, "Could not place call: ${e.message}")
            isCallInProgress = false
            false
        }
    }

    private fun monitorCallProgress(context: Context) {
        monitorJob?.cancel()
        monitorJob = scope.launch {
            val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager

            Log.i(TAG, "Waiting for recipient to answer... (Requires strictly incrementing call duration timer)")
            var connected = false
            var lastTimerSec: Int? = null
            var consecutiveTimerCount = 0

            // Wait up to 90 seconds (standard cellular ring timeout before carrier voicemail)
            for (attempt in 1..180) {
                delay(500L)
                if (!isCallInProgress) break

                val a11y = KarvishAccessibilityService.instance
                val root = a11y?.rootInActiveWindow

                if (root != null) {
                    // Check if call was ended or declined by recipient
                    if (isCallEndedScreen(root)) {
                        Log.i(TAG, "Call was declined, busy, or ended before pickup.")
                        _callState.value = KarvishCallState.Failed(activeTargetName, "Call was declined or unanswered.")
                        endCurrentCall()
                        return@launch
                    }

                    // Extract actual call duration seconds (ignoring clocks and timestamps like "11:28 am")
                    val currentSec = extractActiveCallTimerSeconds(root)
                    if (currentSec != null) {
                        Log.d(TAG, "Detected potential call duration timer: ${currentSec}s (last was $lastTimerSec)")
                        if (lastTimerSec != null && currentSec >= lastTimerSec!! && (currentSec - lastTimerSec!! <= 3)) {
                            consecutiveTimerCount++
                            if (consecutiveTimerCount >= 2) {
                                Log.i(TAG, "RECIPIENT ANSWERED! Confirmed ticking call duration timer (${currentSec}s). Proceeding to audio pipeline.")
                                connected = true
                                break
                            }
                        } else {
                            consecutiveTimerCount = 1
                        }
                        lastTimerSec = currentSec
                    } else {
                        consecutiveTimerCount = 0
                        lastTimerSec = null
                        if (attempt % 6 == 0) {
                            Log.d(TAG, "Phone is still ringing... (attempt $attempt)")
                            _callState.value = KarvishCallState.Ringing(activeTargetName, "Ringing...")
                        }
                    }
                }
            }

            if (connected && isCallInProgress && !hasSpokenForThisCall) {
                onCallConnected(audioManager)
            }
        }
    }

    private fun isCallEndedScreen(root: AccessibilityNodeInfo): Boolean {
        return hasExactOrStatusText(root, listOf("call ended", "call ended.", "call declined", "line busy", "call failed", "busy", "not answered"))
    }

    private fun hasExactOrStatusText(node: AccessibilityNodeInfo?, keywords: List<String>): Boolean {
        if (node == null) return false
        val text = node.text?.toString().orEmpty().trim().lowercase()
        val desc = node.contentDescription?.toString().orEmpty().trim().lowercase()

        for (kw in keywords) {
            if (text == kw || desc == kw) {
                return true
            }
        }

        for (i in 0 until node.childCount) {
            val child = node.getChild(i)
            if (hasExactOrStatusText(child, keywords)) {
                return true
            }
        }
        return false
    }

    /**
     * Extracts call duration in seconds from the active in-call UI.
     * Strictly rejects clocks (e.g. "11:28 am", "11:35"), requiring pure M:SS or MM:SS format.
     */
    private fun extractActiveCallTimerSeconds(node: AccessibilityNodeInfo?): Int? {
        if (node == null) return null
        val rawText = node.text?.toString().orEmpty().trim()
        val rawDesc = node.contentDescription?.toString().orEmpty().trim()
        val candidate = if (rawText.isNotBlank()) rawText else rawDesc

        // Reject any candidate containing AM/PM or non-timer text
        if (candidate.isNotBlank() &&
            !candidate.contains("am", ignoreCase = true) &&
            !candidate.contains("pm", ignoreCase = true) &&
            candidate.matches(Regex("""^0{0,2}:[0-5]\d$|^[0-5]?\d:[0-5]\d$"""))
        ) {
            val parts = candidate.split(":")
            val min = parts[0].toIntOrNull() ?: 0
            val sec = parts[1].toIntOrNull() ?: 0
            // Call duration on pickup is always under 300 seconds (5 minutes)
            if (min <= 5) {
                return min * 60 + sec
            }
        }

        for (i in 0 until node.childCount) {
            val found = extractActiveCallTimerSeconds(node.getChild(i))
            if (found != null) return found
        }
        return null
    }

    private suspend fun onCallConnected(audioManager: AudioManager) {
        if (hasSpokenForThisCall) return
        hasSpokenForThisCall = true

        Log.i(TAG, "Recipient answered! Engaging speakerphone & VAD chime.")
        _callState.value = KarvishCallState.Connected(activeTargetName, activeMessage)

        // 1. Route to speakerphone & lock 100% maximum hardware volume
        try {
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

            val maxMusicVol = audioManager.getStreamMaxVolume(AudioManager.STREAM_MUSIC)
            audioManager.setStreamVolume(AudioManager.STREAM_MUSIC, maxMusicVol, 0)

            Log.i(TAG, "Hardware volumes locked to 100% [ALARM=$maxAlarmVol, VOICE_CALL=$maxCallVol, MUSIC=$maxMusicVol].")
        } catch (e: Exception) {
            Log.w(TAG, "Audio routing adjustment warning: ${e.message}")
        }

        // Tap on-screen Speaker button via Accessibility if present
        clickSpeakerButtonIfAvailable()

        // 2. Buffer pause (600ms) for recipient to say "Hello?"
        delay(600L)

        // 3. Play Carrier VAD Wakeup Chime (forces open carrier voice gate)
        CarrierWakeupChime.play()
        delay(200L)

        // 4. Dispatch initial message (Repeat 1)
        val initialPayload = "Hello $activeTargetName. This is an automated message from Karthik: $activeMessage."
        speakUtterance(initialPayload)
    }

    private fun speakUtterance(payload: String) {
        if (!isTtsReady || tts == null) {
            Log.e(TAG, "TTS not ready for speakUtterance.")
            endCurrentCall()
            return
        }

        tts?.setSpeechRate(SLOW_SPEECH_RATE)
        val params = Bundle().apply {
            putFloat(TextToSpeech.Engine.KEY_PARAM_VOLUME, 1.0f)
            putInt(TextToSpeech.Engine.KEY_PARAM_STREAM, AudioManager.STREAM_ALARM)
        }
        tts?.speak(payload, TextToSpeech.QUEUE_FLUSH, params, UTTERANCE_ID)
        Log.i(TAG, "TTS speech dispatched on STREAM_ALARM: '$payload' at rate $SLOW_SPEECH_RATE")
    }

    private fun clickSpeakerButtonIfAvailable() {
        val a11y = KarvishAccessibilityService.instance ?: return
        val root = a11y.rootInActiveWindow ?: return

        // 1. Search by Samsung / AOSP resource ID
        val resIds = listOf(
            "com.samsung.android.incallui:id/speaker_button",
            "com.samsung.android.incallui:id/audio_output_button",
            "com.samsung.android.incallui:id/speaker",
            "com.google.android.dialer:id/incall_speaker_button",
            "com.android.dialer:id/speakerButton",
            "speaker_button"
        )
        for (resId in resIds) {
            val nodes = root.findAccessibilityNodeInfosByViewId(resId)
            for (node in nodes) {
                if (node.isChecked || node.isSelected) {
                    Log.i(TAG, "Speakerphone is already verified ON via viewId ($resId).")
                    return
                }
                if (node.isClickable) {
                    node.performAction(AccessibilityNodeInfo.ACTION_CLICK)
                    Log.i(TAG, "Clicked on-screen Speaker button via viewId ($resId).")
                    return
                }
                node.parent?.let { parent ->
                    if (parent.isClickable) {
                        parent.performAction(AccessibilityNodeInfo.ACTION_CLICK)
                        Log.i(TAG, "Clicked on-screen Speaker parent via viewId ($resId).")
                        return
                    }
                }
            }
        }

        // 2. Search recursively by content description and text
        val found = clickSpeakerRecursive(root)
        if (found) {
            Log.i(TAG, "Clicked on-screen Speaker button via recursive search.")
        }
    }

    private fun clickSpeakerRecursive(node: AccessibilityNodeInfo?): Boolean {
        if (node == null) return false
        val text = node.text?.toString().orEmpty().lowercase()
        val desc = node.contentDescription?.toString().orEmpty().lowercase()

        val isSpeakerLabel = desc.contains("speaker") || text.contains("speaker") ||
                             desc.contains("speakerphone") || text.contains("speakerphone") ||
                             desc.contains("turn on speaker") || desc.contains("audio output")

        if (isSpeakerLabel) {
            if (node.isChecked || node.isSelected) {
                Log.i(TAG, "Speakerphone is already verified ON.")
                return true
            }
            if (node.isClickable) {
                node.performAction(AccessibilityNodeInfo.ACTION_CLICK)
                return true
            }
            val parent = node.parent
            if (parent != null && parent.isClickable) {
                parent.performAction(AccessibilityNodeInfo.ACTION_CLICK)
                return true
            }
        }

        for (i in 0 until node.childCount) {
            if (clickSpeakerRecursive(node.getChild(i))) return true
        }
        return false
    }

    fun endCurrentCall() {
        Log.i(TAG, "Ending current cellular call...")
        isCallInProgress = false
        monitorJob?.cancel()

        val a11y = KarvishAccessibilityService.instance
        var hungUp = false

        // 1. Try TelecomManager.endCall() if permission available (Android 9+)
        if (a11y != null && Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            val telecom = a11y.getSystemService(Context.TELECOM_SERVICE) as? TelecomManager
            try {
                @Suppress("DEPRECATION")
                telecom?.let {
                    it.endCall()
                    hungUp = true
                    Log.i(TAG, "Ended call via TelecomManager.endCall().")
                }
            } catch (e: Exception) {
                Log.w(TAG, "TelecomManager.endCall() failed: ${e.message}")
            }
        }

        // 2. Failsafe: Tap red End Call button via Accessibility
        if (!hungUp && a11y != null) {
            val root = a11y.rootInActiveWindow
            if (root != null) {
                val endNodes = root.findAccessibilityNodeInfosByText("End call")
                    .ifEmpty { root.findAccessibilityNodeInfosByText("End") }
                    .ifEmpty { root.findAccessibilityNodeInfosByText("Hang up") }

                for (node in endNodes) {
                    if (node.isClickable) {
                        node.performAction(AccessibilityNodeInfo.ACTION_CLICK)
                        hungUp = true
                        Log.i(TAG, "Ended call via Accessibility click.")
                        break
                    }
                    node.parent?.let { parent ->
                        if (parent.isClickable) {
                            parent.performAction(AccessibilityNodeInfo.ACTION_CLICK)
                            hungUp = true
                            Log.i(TAG, "Ended call via Accessibility parent click.")
                            break
                        }
                    }
                }
            }
        }

        // Teardown audio
        if (a11y != null) {
            val audioManager = a11y.getSystemService(Context.AUDIO_SERVICE) as AudioManager
            try {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                    audioManager.clearCommunicationDevice()
                }
                @Suppress("DEPRECATION")
                audioManager.isSpeakerphoneOn = false
                audioManager.mode = AudioManager.MODE_NORMAL
            } catch (_: Exception) {}
        }

        _callState.value = KarvishCallState.Completed(
            activeTargetName,
            "Message repeated $TOTAL_REPEATS times and delivered to $activeTargetName."
        )
    }

    fun reset() {
        isCallInProgress = false
        hasSpokenForThisCall = false
        currentRepeatCount = 0
        monitorJob?.cancel()
        _callState.value = KarvishCallState.Idle
    }
}
