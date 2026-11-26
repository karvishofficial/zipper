package com.zipper.app.core.voice

import android.animation.ObjectAnimator
import android.animation.PropertyValuesHolder
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.floatingactionbutton.FloatingActionButton
import com.zipper.app.R
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * VoiceAssistantActivity
 *
 * Fullscreen / lockscreen-capable conversational Voice Assistant interface.
 * Operates 100% on-device and offline with zero audio files saved to disk.
 *
 * Clean one-shot lifecycle:
 * 1. Wakes up either via "Zipper" wake-word or manual trigger.
 * 2. If wake-word included a command ("Zipper set an alarm for 2:41"), executes directly.
 * 3. Otherwise greets with Paradee-8M, listens once to user speech, executes the task,
 *    speaks natural confirmation, and cleanly auto-dismisses.
 * 4. Never loops or keeps repeating recognition in the background.
 */
class VoiceAssistantActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "VoiceAssistantActivity"
        const val EXTRA_INITIAL_PROMPT = "extra_initial_prompt"

        fun start(context: Context, initialPrompt: String? = null) {
            val intent = Intent(context, VoiceAssistantActivity::class.java).apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP)
                putExtra(EXTRA_INITIAL_PROMPT, initialPrompt)
            }
            context.startActivity(intent)
        }
    }

    private lateinit var tvTranscriptUser: TextView
    private lateinit var tvAssistantReply: TextView
    private lateinit var tvVoiceEngineBadge: TextView
    private lateinit var vOrbPulse: View
    private lateinit var tvOrbIcon: TextView
    private lateinit var btnCloseAssistant: FloatingActionButton
    private lateinit var chipWaterMotor: Button
    private lateinit var chipActiveReminders: Button

    private var pulseAnimator: ObjectAnimator? = null
    private var speechRecognizer: SpeechRecognizer? = null
    private val mainHandler = Handler(Looper.getMainLooper())
    private var isFinishingFlow = false
    private var hasExecutedCommand = false

    private val silenceTimeoutRunnable = Runnable {
        if (!isFinishingFlow && !hasExecutedCommand) {
            tvTranscriptUser.text = "No speech detected."
            tvVoiceEngineBadge.text = "Closing..."
            mainHandler.postDelayed({ stopAllAndFinish() }, 1500L)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        WakeWordService.notifyAssistantStarted()
        setupWindowFlags()
        setContentView(R.layout.activity_voice_assistant)

        initViews()
        ParadeeVoiceEngine.init(this)
        startOrbPulse()

        val initial = intent.getStringExtra(EXTRA_INITIAL_PROMPT)?.trim()
        if (!initial.isNullOrBlank()) {
            // Wake word arrived with direct command (e.g. "Zipper set alarm for 2:41")
            tvTranscriptUser.text = "\"$initial\""
            handleCommand(initial)
        } else {
            // General wake word (e.g. "Zipper" / "Hey Zipper")
            greetAndListen()
        }
    }

    private fun setupWindowFlags() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
            setShowWhenLocked(true)
            setTurnScreenOn(true)
        } else {
            @Suppress("DEPRECATION")
            window.addFlags(
                WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED or
                WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON or
                WindowManager.LayoutParams.FLAG_DISMISS_KEYGUARD
            )
        }
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        window.setBackgroundDrawable(android.graphics.drawable.ColorDrawable(android.graphics.Color.TRANSPARENT))
    }

    private fun initViews() {
        tvTranscriptUser = findViewById(R.id.tvTranscriptUser)
        tvAssistantReply = findViewById(R.id.tvAssistantReply)
        tvVoiceEngineBadge = findViewById(R.id.tvVoiceEngineBadge)
        vOrbPulse = findViewById(R.id.vOrbPulse)
        tvOrbIcon = findViewById(R.id.tvOrbIcon)
        btnCloseAssistant = findViewById(R.id.btnCloseAssistant)
        chipWaterMotor = findViewById(R.id.chipWaterMotor)
        chipActiveReminders = findViewById(R.id.chipActiveReminders)

        btnCloseAssistant.setOnClickListener {
            stopAllAndFinish()
        }

        chipWaterMotor.setOnClickListener {
            handleCommand("call me in 25 minutes to turn off water motor")
        }

        chipActiveReminders.setOnClickListener {
            handleCommand("what are my active reminders")
        }
    }

    private fun startOrbPulse() {
        pulseAnimator = ObjectAnimator.ofPropertyValuesHolder(
            vOrbPulse,
            PropertyValuesHolder.ofFloat(View.SCALE_X, 1.0f, 1.35f),
            PropertyValuesHolder.ofFloat(View.SCALE_Y, 1.0f, 1.35f),
            PropertyValuesHolder.ofFloat(View.ALPHA, 0.35f, 0.95f)
        ).apply {
            duration = 1000
            repeatCount = ObjectAnimator.INFINITE
            repeatMode = ObjectAnimator.REVERSE
            start()
        }
    }

    private fun greetAndListen() {
        tvTranscriptUser.text = "Hey Karthik, I'm listening..."
        tvAssistantReply.text = "Say a command, alarm, or reminder..."
        ParadeeVoiceEngine.speak(
            text = "Hey Karthik, I'm listening.",
            onStart = {
                tvVoiceEngineBadge.text = "🎙️ Speaking via Paradee-8M..."
            },
            onDone = {
                if (!isFinishingFlow) {
                    tvVoiceEngineBadge.text = "👂 Listening..."
                    startListening()
                }
            }
        )
    }

    /**
     * One-shot speech listening: listens once for user's command,
     * then immediately executes and auto-dismisses. Never loops.
     */
    private fun startListening() {
        if (isFinishingFlow) return
        if (androidx.core.content.ContextCompat.checkSelfPermission(
                this,
                android.Manifest.permission.RECORD_AUDIO
            ) != android.content.pm.PackageManager.PERMISSION_GRANTED
        ) {
            androidx.core.app.ActivityCompat.requestPermissions(
                this,
                arrayOf(android.Manifest.permission.RECORD_AUDIO),
                1001
            )
            return
        }
        if (!SpeechRecognizer.isRecognitionAvailable(this)) {
            tvTranscriptUser.text = "Speech recognition unavailable on this device."
            mainHandler.postDelayed({ stopAllAndFinish() }, 2000L)
            return
        }

        mainHandler.removeCallbacks(silenceTimeoutRunnable)
        // 6-second timeout if no speech is detected
        mainHandler.postDelayed(silenceTimeoutRunnable, 6000L)

        speechRecognizer?.destroy()
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this).apply {
            setRecognitionListener(object : RecognitionListener {
                override fun onReadyForSpeech(params: Bundle?) {
                    tvTranscriptUser.text = "Listening for your voice..."
                }

                override fun onBeginningOfSpeech() {
                    mainHandler.removeCallbacks(silenceTimeoutRunnable)
                    tvTranscriptUser.text = "Hearing speech..."
                }

                override fun onRmsChanged(rmsdB: Float) {}
                override fun onBufferReceived(buffer: ByteArray?) {}
                override fun onEndOfSpeech() {
                    tvTranscriptUser.text = "Processing..."
                }

                override fun onError(error: Int) {
                    mainHandler.removeCallbacks(silenceTimeoutRunnable)
                    Log.w(TAG, "SpeechRecognizer error: $error")
                    if (!isFinishingFlow && !hasExecutedCommand) {
                        tvTranscriptUser.text = "Didn't catch that."
                        tvVoiceEngineBadge.text = "Closing..."
                        mainHandler.postDelayed({ stopAllAndFinish() }, 2000L)
                    }
                }

                override fun onResults(results: Bundle?) {
                    mainHandler.removeCallbacks(silenceTimeoutRunnable)
                    val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    val spoken = matches?.firstOrNull()?.trim()
                    if (!spoken.isNullOrBlank()) {
                        handleCommand(spoken)
                    } else if (!isFinishingFlow && !hasExecutedCommand) {
                        tvTranscriptUser.text = "No speech detected."
                        mainHandler.postDelayed({ stopAllAndFinish() }, 1500L)
                    }
                }

                override fun onPartialResults(partialResults: Bundle?) {
                    val matches = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    val partial = matches?.firstOrNull()?.trim()
                    if (!partial.isNullOrBlank()) {
                        tvTranscriptUser.text = partial
                    }
                }

                override fun onEvent(eventType: Int, params: Bundle?) {}
            })
        }

        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
        }

        try {
            speechRecognizer?.startListening(intent)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start speech listening: ${e.message}")
            stopAllAndFinish()
        }
    }

    private fun handleCommand(rawCommand: String) {
        if (hasExecutedCommand) return
        hasExecutedCommand = true
        mainHandler.removeCallbacks(silenceTimeoutRunnable)

        // Clean command string: remove redundant wake prefixes if present
        val command = rawCommand
            .replace(Regex("""(?i)^(?:hey|hi|ok|okay)?\s*zipper(?:,)?\s*"""), "")
            .trim()
            .ifBlank { rawCommand.trim() }

        tvTranscriptUser.text = "\"$command\""
        val lower = command.lowercase(Locale.ROOT)

        // Case 0: If Karvish Agent is currently running in an app, forward live command or argument correction
        if (com.zipper.app.core.agent.AgentRuntimeService.isRunning &&
            !lower.contains("alarm") && !lower.contains("reminder") && !lower.contains("call me in")
        ) {
            Log.i(TAG, "Forwarding live command/correction to active Karvish agent: '$command'")
            val serviceIntent = Intent(applicationContext, com.zipper.app.core.agent.AgentRuntimeService::class.java).apply {
                putExtra(com.zipper.app.core.agent.AgentRuntimeService.EXTRA_GOAL, command)
            }
            androidx.core.content.ContextCompat.startForegroundService(applicationContext, serviceIntent)
            speakResponse("Updating: $command")
            return
        }

        when {
            // Case 1: Check active reminders
            lower.contains("active") || lower.contains("what reminder") || lower.contains("any reminder") || lower.contains("list reminder") || lower.contains("show reminder") || lower.contains("pending") -> {
                val all = CallReminderScheduler.getAllReminders(this).filter { !it.isCompleted }
                val reply = if (all.isEmpty()) {
                    "You have no active reminders right now. All caught up!"
                } else {
                    val count = all.size
                    val first = all.first()
                    "You have $count active reminder. Next is ${first.title}."
                }
                speakResponse(reply)
            }

            // Case 2: Mark done / completed
            lower.contains("done") || lower.contains("mark done") || lower.contains("completed") || lower.contains("finish reminder") -> {
                val all = CallReminderScheduler.getAllReminders(this).filter { !it.isCompleted }
                if (all.isNotEmpty()) {
                    CallReminderScheduler.markComplete(this, all.first().id)
                    speakResponse("Marked done! Have a great day.")
                } else {
                    speakResponse("No active reminders to mark complete.")
                }
            }

            // Case 3: Schedule reminder or alarm
            lower.contains("alarm") || lower.contains("call me") || lower.contains("remind me") || lower.contains("wake me") || lower.contains("set reminder") || lower.contains("motor") || lower.contains("minute") || lower.contains("timer") || lower.contains("schedule") -> {
                val parsed = CallReminderScheduler.parseNaturalReminder(command)
                if (parsed != null) {
                    CallReminderScheduler.schedule(this, parsed)
                    val reply = "Scheduled! I will call you for ${parsed.title}."
                    speakResponse(reply)
                } else {
                    // Fallback create 20 min reminder
                    val fallback = CallReminder(
                        title = command.take(30),
                        spokenPrompt = "Hey Karthik, time to: $command",
                        triggerTimeEpochMs = System.currentTimeMillis() + 20 * 60 * 1000L,
                        actionType = CallReminderActionType.VOICE_ONLY
                    )
                    CallReminderScheduler.schedule(this, fallback)
                    speakResponse("Reminder scheduled for 20 minutes from now.")
                }
            }

            // Case 4: Time Query
            lower.contains("what time") || lower.contains("what's the time") || lower.contains("current time") || lower.contains("time is it") -> {
                val timeStr = SimpleDateFormat("h:mm a", Locale.US).format(Date())
                speakResponse("It's currently $timeStr.")
            }

            // Case 5: Battery Query
            lower.contains("battery") -> {
                val batteryStatus: Intent? = registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
                val level: Int = batteryStatus?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
                val scale: Int = batteryStatus?.getIntExtra(BatteryManager.EXTRA_SCALE, -1) ?: -1
                val pct = if (level >= 0 && scale > 0) (level * 100) / scale else 50
                speakResponse("Your battery is at $pct percent.")
            }

            // Case 6: General Greeting or Identity
            lower == "hello" || lower == "hi" || lower == "hey" || lower.contains("who are you") || lower.contains("what can you do") -> {
                speakResponse("Hello Karthik! I'm Zipper, your on-device offline voice assistant and autonomous screen copilot.")
            }

            // Case 7: All phone actions & Karvish autonomous agent commands (Search, Fill, Type, Tap, Open, Order, etc.)
            else -> {
                val goal = command.replaceFirstChar { it.uppercase() }
                Log.i(TAG, "Dispatching autonomous Karvish agent goal: '$goal'")
                val serviceIntent = Intent(applicationContext, com.zipper.app.core.agent.AgentRuntimeService::class.java).apply {
                    putExtra(com.zipper.app.core.agent.AgentRuntimeService.EXTRA_GOAL, goal)
                    putExtra(com.zipper.app.core.agent.AgentRuntimeService.EXTRA_RECOVERY, true)
                    putExtra(com.zipper.app.core.agent.AgentRuntimeService.EXTRA_USE_REMOTE_RANKER, false)
                }
                androidx.core.content.ContextCompat.startForegroundService(applicationContext, serviceIntent)
                Toast.makeText(applicationContext, "🚀 Karvish Agent: $goal", Toast.LENGTH_SHORT).show()
                // Immediately finish voice overlay so active app returns to foreground before agent senses screen
                stopAllAndFinish()
            }
        }
    }

    private fun speakResponse(replyText: String) {
        tvAssistantReply.text = replyText
        tvOrbIcon.text = "💬"
        ParadeeVoiceEngine.speak(
            text = replyText,
            onStart = {
                tvVoiceEngineBadge.text = "🎙️ Speaking via Paradee-8M..."
            },
            onDone = {
                tvVoiceEngineBadge.text = "✓ Finished"
                mainHandler.postDelayed({
                    stopAllAndFinish()
                }, 1800L)
            }
        )
    }

    private fun stopAllAndFinish() {
        if (isFinishingFlow) return
        isFinishingFlow = true
        mainHandler.removeCallbacksAndMessages(null)
        try {
            speechRecognizer?.destroy()
            speechRecognizer = null
        } catch (_: Exception) {}
        ParadeeVoiceEngine.stop()
        pulseAnimator?.cancel()
        WakeWordService.notifyAssistantFinished()
        finish()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 1001 && grantResults.isNotEmpty() && grantResults[0] == android.content.pm.PackageManager.PERMISSION_GRANTED) {
            startListening()
        } else {
            tvTranscriptUser.text = "Microphone permission is required."
            tvVoiceEngineBadge.text = "Permission Denied"
            mainHandler.postDelayed({ stopAllAndFinish() }, 2000L)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        isFinishingFlow = true
        mainHandler.removeCallbacksAndMessages(null)
        try {
            speechRecognizer?.destroy()
            speechRecognizer = null
        } catch (_: Exception) {}
        ParadeeVoiceEngine.stop()
        pulseAnimator?.cancel()
        WakeWordService.notifyAssistantFinished()
    }
}
