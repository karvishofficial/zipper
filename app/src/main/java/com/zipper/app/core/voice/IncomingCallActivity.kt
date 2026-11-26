package com.zipper.app.core.voice

import android.animation.ObjectAnimator
import android.animation.PropertyValuesHolder
import android.content.Context
import android.content.Intent
import android.media.AudioAttributes
import android.media.AudioManager
import android.media.Ringtone
import android.media.RingtoneManager
import android.os.Build
import android.os.Bundle
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log
import android.view.View
import android.view.WindowManager
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.zipper.app.R
import com.zipper.app.core.agent.AgentRuntimeService
import com.zipper.app.core.intelligence.reply.DelegatedCallSessionManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Calendar
import java.util.Locale

/**
 * IncomingCallActivity
 *
 * Daylight iOS/OneUI 7 aesthetic voice reminder HUD featuring:
 * - 3D Mascot Avatar with dual concentric pulsing radar rings
 * - Calling About goal card with dynamic category & due date grounding
 * - Spoken speech transcript bubble with voice waveform
 * - 3 Quick Action tactile pills: Start now, Snooze 30m, Reschedule
 * - Dual floating call controls with soft ambient glowing auras
 */
class IncomingCallActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "IncomingCallActivity"

        const val EXTRA_REMINDER_TITLE = "extra_reminder_title"
        const val EXTRA_REMINDER_PROMPT = "extra_reminder_prompt"
        const val EXTRA_ACTION_TYPE = "extra_action_type"
        const val EXTRA_TARGET_GOAL = "extra_target_goal"
        const val EXTRA_COMMITMENT_ID = "extra_commitment_id"
        const val EXTRA_CATEGORY = "extra_category"
        const val EXTRA_CANDIDATE_PACKAGES = "extra_candidate_packages"
        const val EXTRA_SOURCE_PERSON = "extra_source_person"
        const val EXTRA_CALL_SESSION_ID = "extra_call_session_id"
        const val EXTRA_DEFLECTION_REPLY = "extra_deflection_reply"
        const val EXTRA_ORIGINAL_MESSAGE = "extra_original_message"
    }

    private var reminderId: String? = null
    private var reminderTitle: String = "Reminder"
    private var reminderPrompt: String = "You have a scheduled reminder."
    private var originalIncomingMessage: String? = null
    private var actionType: CallReminderActionType = CallReminderActionType.VOICE_ONLY
    private var targetGoal: String? = null
    private var commitmentId: Long? = null
    private var categoryName: String? = null
    private var candidatePackages: List<String> = emptyList()
    private var sourcePerson: String = ""
    private var callSessionId: String? = null
    private var deflectionReply: String? = null

    // UI Elements
    private lateinit var tvCallStatusHeader: TextView
    private lateinit var tvCallDuration: TextView
    private lateinit var vAvatarPulse: View
    private lateinit var vAvatarPulse2: View
    private lateinit var tvReminderTitle: TextView
    private lateinit var tvReminderDue: TextView
    private lateinit var tvReminderPrompt: TextView
    private lateinit var tvVoiceEngineBadge: TextView
    private lateinit var ivVoiceWaveform: ImageView

    // Action Controls
    private lateinit var btnQuickStartNow: View
    private lateinit var btnQuickSnooze: View
    private lateinit var btnQuickReschedule: View
    private lateinit var btnDeclineCall: View
    private lateinit var btnAnswerCall: View
    private lateinit var vGlowDecline: View
    private lateinit var vGlowAnswer: View
    private lateinit var tvAnswerLabel: TextView
    private lateinit var tvDeclineLabel: TextView

    // Call Screening HUD
    private var llScreeningDecisionsContainer: View? = null
    private var btnScreenCall: View? = null
    private var llScreenCallContainer: View? = null

    // Candidate Apps HUD
    private lateinit var llCandidateAppsContainer: View
    private lateinit var tvCandidateAppsLabel: TextView
    private lateinit var llCandidateAppIcons: LinearLayout

    private var ringtone: Ringtone? = null
    private var vibrator: Vibrator? = null
    private var pulseAnimator1: ObjectAnimator? = null
    private var pulseAnimator2: ObjectAnimator? = null
    private var answerPulseAnimator: ObjectAnimator? = null
    private var answerGlowAnimator: ObjectAnimator? = null
    private var waveformAnimator: ObjectAnimator? = null

    private var audioManager: AudioManager? = null
    private var originalAudioMode: Int = AudioManager.MODE_NORMAL
    private var originalSpeakerphoneOn: Boolean = false

    private val activityScope = CoroutineScope(Dispatchers.Main)
    private var callDurationJob: Job? = null
    private var isCallConnected = false

    private var speechRecognizer: SpeechRecognizer? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setupWindowFlags()

        setContentView(R.layout.activity_incoming_call)

        // Parse extras
        reminderId = intent.getStringExtra(CallReminderScheduler.EXTRA_REMINDER_ID)
        reminderTitle = intent.getStringExtra(EXTRA_REMINDER_TITLE) ?: "Reminder"
        reminderPrompt = intent.getStringExtra(EXTRA_REMINDER_PROMPT) ?: "You have a scheduled reminder."
        val typeName = intent.getStringExtra(EXTRA_ACTION_TYPE) ?: CallReminderActionType.VOICE_ONLY.name
        actionType = try { CallReminderActionType.valueOf(typeName) } catch (_: Exception) { CallReminderActionType.VOICE_ONLY }
        targetGoal = intent.getStringExtra(EXTRA_TARGET_GOAL)

        val cid = intent.getLongExtra(EXTRA_COMMITMENT_ID, -1L)
        if (cid != -1L) commitmentId = cid
        categoryName = intent.getStringExtra(EXTRA_CATEGORY)
        candidatePackages = intent.getStringArrayExtra(EXTRA_CANDIDATE_PACKAGES)?.toList() ?: emptyList()
        sourcePerson = intent.getStringExtra(EXTRA_SOURCE_PERSON).orEmpty()
        callSessionId = intent.getStringExtra(EXTRA_CALL_SESSION_ID)
        deflectionReply = intent.getStringExtra(EXTRA_DEFLECTION_REPLY)
        originalIncomingMessage = intent.getStringExtra(EXTRA_ORIGINAL_MESSAGE)

        initViews()
        ParadeeVoiceEngine.init(this)
        startRinging()
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
    }

    private fun initViews() {
        tvCallStatusHeader = findViewById(R.id.tvCallStatusHeader)
        tvCallDuration = findViewById(R.id.tvCallDuration)
        vAvatarPulse = findViewById(R.id.vAvatarPulse)
        vAvatarPulse2 = findViewById(R.id.vAvatarPulse2)
        tvReminderTitle = findViewById(R.id.tvReminderTitle)
        tvReminderDue = findViewById(R.id.tvReminderDue)
        tvReminderPrompt = findViewById(R.id.tvReminderPrompt)
        tvVoiceEngineBadge = findViewById(R.id.tvVoiceEngineBadge)
        ivVoiceWaveform = findViewById(R.id.ivVoiceWaveform)

        btnQuickStartNow = findViewById(R.id.btnQuickStartNow)
        btnQuickSnooze = findViewById(R.id.btnQuickSnooze)
        btnQuickReschedule = findViewById(R.id.btnQuickReschedule)

        btnDeclineCall = findViewById(R.id.btnDeclineCall)
        btnAnswerCall = findViewById(R.id.btnAnswerCall)
        vGlowDecline = findViewById(R.id.vGlowDecline)
        vGlowAnswer = findViewById(R.id.vGlowAnswer)
        tvAnswerLabel = findViewById(R.id.tvAnswerLabel)
        tvDeclineLabel = findViewById(R.id.tvDeclineLabel)

        llCandidateAppsContainer = findViewById(R.id.llCandidateAppsContainer)
        tvCandidateAppsLabel = findViewById(R.id.tvCandidateAppsLabel)
        llCandidateAppIcons = findViewById(R.id.llCandidateAppIcons)

        // Set initial texts
        tvReminderTitle.text = reminderTitle
        val cleanPrompt = reminderPrompt.trim().removeSurrounding("\"").removeSurrounding("“").removeSurrounding("”")
        tvReminderPrompt.text = "“$cleanPrompt”"

        val categoryClean = categoryName?.lowercase()?.replaceFirstChar { it.uppercase() } ?: "Task"
        tvReminderDue.text = "Due soon • $categoryClean"

        // Load full commitment details from DB if available
        commitmentId?.let { cid ->
            activityScope.launch(Dispatchers.IO) {
                val db = com.zipper.app.core.intelligence.store.CommitmentDatabase.getInstance(applicationContext)
                val commitment = db.getCommitmentById(cid)
                if (commitment != null) {
                    withContext(Dispatchers.Main) {
                        tvReminderTitle.text = commitment.title

                        val dueStr = if (commitment.deadlineMs != null && commitment.deadlineMs > 0) {
                            val cal = Calendar.getInstance().apply { timeInMillis = commitment.deadlineMs }
                            val timeStr = String.format(Locale.US, "%02d:%02d", cal.get(Calendar.HOUR_OF_DAY), cal.get(Calendar.MINUTE))
                            "Due today at $timeStr • ${commitment.category.name.lowercase().replaceFirstChar { it.uppercase() }}"
                        } else if (commitment.scheduledTimeMs > 0) {
                            val cal = Calendar.getInstance().apply { timeInMillis = commitment.scheduledTimeMs }
                            val timeStr = String.format(Locale.US, "%02d:%02d", cal.get(Calendar.HOUR_OF_DAY), cal.get(Calendar.MINUTE))
                            "Scheduled at $timeStr • ${commitment.category.name.lowercase().replaceFirstChar { it.uppercase() }}"
                        } else {
                            "Priority Task • ${commitment.category.name.lowercase().replaceFirstChar { it.uppercase() }}"
                        }
                        tvReminderDue.text = dueStr

                        // Enhance spoken dialogue with person/details if available
                        if (cleanPrompt.isBlank() || cleanPrompt == "You have a scheduled reminder.") {
                            tvReminderPrompt.text = "“Hey Karthik, your ${commitment.title} is scheduled now. Do you want to work on it now or should I reschedule it?”"
                        }
                    }
                }
            }
        }

        // Setup Quick Action Pills
        callSessionId?.let { sid ->
            // In delegated call session mode: Customize HUD & pills for schedule confirmation
            tvCallStatusHeader.text = "Incoming Schedule Call"
            tvReminderDue.text = "Schedule Confirmation • ${sourcePerson.ifBlank { "WhatsApp" }}"

            // Make hero avatar compact in schedule confirmation mode so all cards and typing input fit on screen
            val flHeroAvatar = findViewById<View>(R.id.flHeroAvatarContainer)
            flHeroAvatar?.let { hero ->
                val lp = hero.layoutParams
                val density = resources.displayMetrics.density
                lp.height = (110 * density).toInt()
                lp.width = (110 * density).toInt()
                hero.layoutParams = lp
            }

            val session = DelegatedCallSessionManager.getSession(sid)
            val contactName = session?.contact?.contactName ?: sourcePerson
            val effectiveOriginalMsg = originalIncomingMessage.takeIf { !it.isNullOrBlank() }
                ?: session?.incomingMessage

            val cardOriginalMessage = findViewById<View>(R.id.cardOriginalMessage)
            val tvOriginalMessageHeader = findViewById<TextView>(R.id.tvOriginalMessageHeader)
            val tvOriginalMessageText = findViewById<TextView>(R.id.tvOriginalMessageText)

            if (!effectiveOriginalMsg.isNullOrBlank()) {
                cardOriginalMessage?.visibility = View.VISIBLE
                val headerName = contactName.ifBlank { "Contact" }.uppercase(Locale.ROOT)
                tvOriginalMessageHeader?.text = "MESSAGE FROM $headerName"
                tvOriginalMessageText?.text = "“$effectiveOriginalMsg”"
            }

            findViewById<TextView>(R.id.tvQuickStartNowText)?.text = "Confirm"
            findViewById<TextView>(R.id.tvQuickSnoozeTitle)?.text = "Busy"
            findViewById<TextView>(R.id.tvQuickSnoozeSubtitle)?.text = "Ping in 30m"
            findViewById<TextView>(R.id.tvQuickRescheduleText)?.text = "4 PM instead"

            // Show interactive in-call custom message typing input
            val llCallCustomInputContainer = findViewById<View>(R.id.llCallCustomInputContainer)
            val etCallCustomInput = findViewById<android.widget.EditText>(R.id.etCallCustomInput)
            val btnCallCustomSend = findViewById<View>(R.id.btnCallCustomSend)
            llCallCustomInputContainer?.visibility = View.VISIBLE
            findViewById<View>(R.id.btnDontCallToday)?.visibility = View.GONE

            val svCallContent = findViewById<androidx.core.widget.NestedScrollView>(R.id.svCallContent)
            etCallCustomInput?.setOnFocusChangeListener { _, hasFocus ->
                if (hasFocus) {
                    svCallContent?.postDelayed({
                        svCallContent.smoothScrollTo(0, llCallCustomInputContainer?.bottom ?: 0)
                    }, 250)
                }
            }

            val sendCustomReply = {
                val typed = etCallCustomInput?.text?.toString()?.trim().orEmpty()
                if (typed.isNotBlank()) {
                    stopRinging()
                    tvVoiceEngineBadge.text = "💬 Sending your message..."
                    DelegatedCallSessionManager.handleUserDecision(applicationContext, sid, typed)
                    finishAndDismiss()
                }
            }

            btnCallCustomSend?.setOnClickListener {
                sendCustomReply()
            }

            etCallCustomInput?.setOnEditorActionListener { _, actionId, _ ->
                if (actionId == android.view.inputmethod.EditorInfo.IME_ACTION_SEND) {
                    sendCustomReply()
                    true
                } else {
                    false
                }
            }

            val dossier = com.zipper.app.core.intelligence.reply.ContactKnowledgeBook.getInstance(applicationContext).getDossier(contactName)
            val isElder = dossier?.isElderOrFamily() == true
            val isKan = dossier?.isKannadaPreferred() == true || dossier?.languageStyle == com.zipper.app.core.intelligence.reply.LanguageStyle.KANNADA_REGIONAL

            if (isElder && isKan) {
                findViewById<TextView>(R.id.tvQuickStartNowText)?.text = "Sari (Confirm)"
                findViewById<TextView>(R.id.tvQuickSnoozeTitle)?.text = "Swalpa busy"
                findViewById<TextView>(R.id.tvQuickSnoozeSubtitle)?.text = "Aamele call"
                findViewById<TextView>(R.id.tvQuickRescheduleText)?.text = "4 PM ge"
            } else if (isElder) {
                findViewById<TextView>(R.id.tvQuickStartNowText)?.text = "Confirm"
                findViewById<TextView>(R.id.tvQuickSnoozeTitle)?.text = "Busy"
                findViewById<TextView>(R.id.tvQuickSnoozeSubtitle)?.text = "Follow up later"
                findViewById<TextView>(R.id.tvQuickRescheduleText)?.text = "4 PM instead"
            }

            btnQuickStartNow.setOnClickListener {
                stopRinging()
                DelegatedCallSessionManager.handleUserDecision(applicationContext, sid, "Confirm: that time works for me.")
                finishAndDismiss()
            }

            btnQuickSnooze.setOnClickListener {
                stopRinging()
                DelegatedCallSessionManager.handleUserDecision(applicationContext, sid, "Busy with work right now, will follow up once free.")
                finishAndDismiss()
            }

            btnQuickReschedule.setOnClickListener {
                stopRinging()
                DelegatedCallSessionManager.handleUserDecision(applicationContext, sid, "Can we meet at 4 PM instead?")
                finishAndDismiss()
            }
        } ?: run {
            btnQuickStartNow.setOnClickListener {
                stopRinging()
                if (!targetGoal.isNullOrBlank()) {
                    handleAgentExecution(targetGoal!!)
                } else if (candidatePackages.isNotEmpty()) {
                    val pkg = candidatePackages.first()
                    val launchIntent = packageManager.getLaunchIntentForPackage(pkg)
                    if (launchIntent != null) startActivity(launchIntent)
                    finishAndDismiss()
                } else {
                    handleDone()
                }
            }

            btnQuickSnooze.setOnClickListener {
                handleSnooze(minutes = 30)
            }

            btnQuickReschedule.setOnClickListener {
                handleSnooze(minutes = 120)
            }
        }

        // Setup "Don't Call Today" Pill
        findViewById<View>(R.id.btnDontCallToday)?.setOnClickListener {
            handleMuteToday()
        }
        findViewById<View>(R.id.btnCallScreenDontCallToday)?.setOnClickListener {
            handleMuteToday()
        }

        // Setup Bottom Call Controls: Decline cleanly dismisses and cancels alarm
        btnDeclineCall.setOnClickListener {
            handleDismiss()
        }

        btnAnswerCall.setOnClickListener {
            answerCall()
        }

        // Setup Pixel-Style Autonomous Call Screener Controls
        llScreeningDecisionsContainer = findViewById(R.id.llScreeningDecisionsContainer)
        btnScreenCall = findViewById(R.id.btnScreenCall)
        llScreenCallContainer = findViewById(R.id.llScreenCallContainer)

        btnScreenCall?.setOnClickListener {
            startCallScreening()
        }

        findViewById<View>(R.id.btnScreenPillLeaveAtDoor)?.setOnClickListener {
            executeScreeningPill("LEAVE_AT_DOOR")
        }
        findViewById<View>(R.id.btnScreenPillCallBack)?.setOnClickListener {
            executeScreeningPill("CALL_BACK_15M")
        }
        findViewById<View>(R.id.btnScreenPillNotInterested)?.setOnClickListener {
            executeScreeningPill("NOT_INTERESTED")
        }
        findViewById<View>(R.id.btnScreenPillUrgentRing)?.setOnClickListener {
            executeScreeningPill("RING_URGENT")
        }
    }

    private fun startCallScreening() {
        if (isCallConnected) return
        isCallConnected = true
        stopRinging()

        // 1. Audio Routing to Loudspeaker
        audioManager = getSystemService(Context.AUDIO_SERVICE) as? AudioManager
        audioManager?.let { am ->
            originalAudioMode = am.mode
            originalSpeakerphoneOn = am.isSpeakerphoneOn
            am.mode = AudioManager.MODE_NORMAL
            @Suppress("DEPRECATION")
            am.isSpeakerphoneOn = true
        }

        // 2. Transition UI to Screening State
        tvCallStatusHeader.text = "Screening Call with Zipper AI"
        tvCallStatusHeader.setTextColor(ContextCompat.getColor(this, R.color.brand_blue))
        tvCallDuration.text = "00:00"

        // Hide answer and screen call buttons, change decline to End Call
        (btnAnswerCall.parent as? View)?.visibility = View.GONE
        tvAnswerLabel.visibility = View.GONE
        llScreenCallContainer?.visibility = View.GONE
        tvDeclineLabel.text = "End Call"
        btnDeclineCall.setOnClickListener {
            handleDismiss()
        }

        // Show decision pills HUD
        llScreeningDecisionsContainer?.visibility = View.VISIBLE

        // Start Call Timer
        startCallTimer()

        // Animate voice waveform
        waveformAnimator = ObjectAnimator.ofFloat(ivVoiceWaveform, View.SCALE_Y, 0.7f, 1.3f).apply {
            duration = 350
            repeatCount = ObjectAnimator.INFINITE
            repeatMode = ObjectAnimator.REVERSE
            start()
        }

        tvVoiceEngineBadge.text = "🤖 Zipper AI: Asking caller's purpose..."
        CallScreeningEngine.speakOpeningGreeting(applicationContext) {
            waveformAnimator?.cancel()
            ivVoiceWaveform.scaleY = 1.0f
            tvVoiceEngineBadge.text = "👂 Waiting for caller / tap decision pill above"
        }
    }

    private fun executeScreeningPill(pillId: String) {
        val pill = CallScreeningEngine.DEFAULT_DECISION_PILLS.firstOrNull { it.id == pillId } ?: return
        tvVoiceEngineBadge.text = "💬 Whispering to caller: '${pill.label}'..."
        waveformAnimator = ObjectAnimator.ofFloat(ivVoiceWaveform, View.SCALE_Y, 0.7f, 1.3f).apply {
            duration = 350
            repeatCount = ObjectAnimator.INFINITE
            repeatMode = ObjectAnimator.REVERSE
            start()
        }
        CallScreeningEngine.executeDecision(applicationContext, pill) {
            waveformAnimator?.cancel()
            ivVoiceWaveform.scaleY = 1.0f
            if (pill.actionType != ScreeningAction.RING_LOUDLY) {
                finishAndDismiss()
            }
        }
    }

    private fun startRinging() {
        // 1. Ringtone
        try {
            val ringtoneUri = RingtoneManager.getDefaultUri(RingtoneManager.TYPE_RINGTONE)
            ringtone = RingtoneManager.getRingtone(applicationContext, ringtoneUri)
            ringtone?.audioAttributes = AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_NOTIFICATION_RINGTONE)
                .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                .build()
            ringtone?.play()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to play ringtone: ${e.message}")
        }

        // 2. Vibration
        try {
            vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                val vm = getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as? VibratorManager
                vm?.defaultVibrator
            } else {
                @Suppress("DEPRECATION")
                getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator
            }

            val pattern = longArrayOf(0, 1000, 800, 1000, 800)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                vibrator?.vibrate(VibrationEffect.createWaveform(pattern, 0))
            } else {
                @Suppress("DEPRECATION")
                vibrator?.vibrate(pattern, 0)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to vibrate: ${e.message}")
        }

        // 3. Avatar Concentric Ripple Animations
        pulseAnimator1 = ObjectAnimator.ofPropertyValuesHolder(
            vAvatarPulse,
            PropertyValuesHolder.ofFloat(View.SCALE_X, 1.0f, 1.25f),
            PropertyValuesHolder.ofFloat(View.SCALE_Y, 1.0f, 1.25f),
            PropertyValuesHolder.ofFloat(View.ALPHA, 0.4f, 0.9f)
        ).apply {
            duration = 1000
            repeatCount = ObjectAnimator.INFINITE
            repeatMode = ObjectAnimator.REVERSE
            start()
        }

        pulseAnimator2 = ObjectAnimator.ofPropertyValuesHolder(
            vAvatarPulse2,
            PropertyValuesHolder.ofFloat(View.SCALE_X, 1.0f, 1.4f),
            PropertyValuesHolder.ofFloat(View.SCALE_Y, 1.0f, 1.4f),
            PropertyValuesHolder.ofFloat(View.ALPHA, 0.2f, 0.7f)
        ).apply {
            duration = 1200
            repeatCount = ObjectAnimator.INFINITE
            repeatMode = ObjectAnimator.REVERSE
            start()
        }

        // 4. Answer Button Breathing / Heartbeat Pulse
        answerPulseAnimator = ObjectAnimator.ofPropertyValuesHolder(
            btnAnswerCall,
            PropertyValuesHolder.ofFloat(View.SCALE_X, 1.0f, 1.08f),
            PropertyValuesHolder.ofFloat(View.SCALE_Y, 1.0f, 1.08f)
        ).apply {
            duration = 1100
            repeatCount = ObjectAnimator.INFINITE
            repeatMode = ObjectAnimator.REVERSE
            start()
        }

        answerGlowAnimator = ObjectAnimator.ofPropertyValuesHolder(
            vGlowAnswer,
            PropertyValuesHolder.ofFloat(View.SCALE_X, 1.0f, 1.22f),
            PropertyValuesHolder.ofFloat(View.SCALE_Y, 1.0f, 1.22f),
            PropertyValuesHolder.ofFloat(View.ALPHA, 0.4f, 0.95f)
        ).apply {
            duration = 1100
            repeatCount = ObjectAnimator.INFINITE
            repeatMode = ObjectAnimator.REVERSE
            start()
        }
    }

    private fun stopRinging() {
        try {
            ringtone?.stop()
            ringtone = null
        } catch (_: Exception) {}

        try {
            vibrator?.cancel()
            vibrator = null
        } catch (_: Exception) {}

        try {
            pulseAnimator1?.cancel()
            pulseAnimator2?.cancel()
            answerPulseAnimator?.cancel()
            answerGlowAnimator?.cancel()

            vAvatarPulse.scaleX = 1.0f
            vAvatarPulse.scaleY = 1.0f
            vAvatarPulse2.scaleX = 1.0f
            vAvatarPulse2.scaleY = 1.0f
            btnAnswerCall.scaleX = 1.0f
            btnAnswerCall.scaleY = 1.0f
            vGlowAnswer.scaleX = 1.0f
            vGlowAnswer.scaleY = 1.0f
        } catch (_: Exception) {}
    }

    private fun answerCall() {
        if (isCallConnected) return
        isCallConnected = true
        stopRinging()

        // 1. Audio Routing to Loudspeaker
        audioManager = getSystemService(Context.AUDIO_SERVICE) as? AudioManager
        audioManager?.let { am ->
            originalAudioMode = am.mode
            originalSpeakerphoneOn = am.isSpeakerphoneOn
            am.mode = AudioManager.MODE_NORMAL
            @Suppress("DEPRECATION")
            am.isSpeakerphoneOn = true
        }

        // 2. Transition UI to In-Call State
        tvCallStatusHeader.text = "Connected"
        tvCallStatusHeader.setTextColor(ContextCompat.getColor(this, R.color.brand_blue))
        tvCallDuration.text = "00:00"

        // Transform call controls: Hide Answer button, switch Decline button to red "End Call"
        (btnAnswerCall.parent as? View)?.visibility = View.GONE
        tvAnswerLabel.visibility = View.GONE
        tvDeclineLabel.text = "End Call"
        btnDeclineCall.setOnClickListener {
            handleDismiss()
        }

        // Animate voice waveform
        waveformAnimator = ObjectAnimator.ofFloat(ivVoiceWaveform, View.SCALE_Y, 0.7f, 1.3f).apply {
            duration = 350
            repeatCount = ObjectAnimator.INFINITE
            repeatMode = ObjectAnimator.REVERSE
            start()
        }

        // Populate Candidate Apps HUD
        populateCandidateApps()

        // 3. Start In-Call Timer
        startCallTimer()

        // 4. Synthesize Reminder via Paradee-8M
        val speechText = tvReminderPrompt.text.toString().removeSurrounding("“").removeSurrounding("”")
        tvVoiceEngineBadge.text = "🎙️ Speaking via Paradee-8M..."
        ParadeeVoiceEngine.speak(
            text = speechText,
            onStart = {
                tvVoiceEngineBadge.text = "🎙️ Paradee-8M Speaking..."
            },
            onDone = {
                waveformAnimator?.cancel()
                ivVoiceWaveform.scaleY = 1.0f
                if (callSessionId != null) {
                    tvVoiceEngineBadge.text = "👂 Listening ('Yes' / 'Busy' / 'Decline')..."
                } else {
                    tvVoiceEngineBadge.text = "👂 Listening ('Done' / 'Snooze' / 'Start')..."
                }
                startVoiceRecognition()
            }
        )
    }

    private fun populateCandidateApps() {
        val category = try {
            categoryName?.let { com.zipper.app.core.intelligence.store.CommitmentCategory.valueOf(it) }
                ?: com.zipper.app.core.intelligence.store.CommitmentCategory.GENERAL
        } catch (_: Exception) {
            com.zipper.app.core.intelligence.store.CommitmentCategory.GENERAL
        }

        val resolver = com.zipper.app.core.intelligence.resolver.InstalledAppResolver(this)
        val candidates = resolver.resolveCandidates(
            category = category,
            suggestedPackages = candidatePackages,
            sourcePerson = sourcePerson
        )

        if (candidates.isEmpty()) {
            llCandidateAppsContainer.visibility = View.GONE
            return
        }

        llCandidateAppIcons.removeAllViews()
        llCandidateAppsContainer.visibility = View.VISIBLE

        when (category) {
            com.zipper.app.core.intelligence.store.CommitmentCategory.FITNESS,
            com.zipper.app.core.intelligence.store.CommitmentCategory.HEALTH -> {
                tvCandidateAppsLabel.text = "CHOOSE WORKOUT APP"
            }
            com.zipper.app.core.intelligence.store.CommitmentCategory.EDUCATION,
            com.zipper.app.core.intelligence.store.CommitmentCategory.WORK -> {
                tvCandidateAppsLabel.text = "CHOOSE STUDY APP"
            }
            com.zipper.app.core.intelligence.store.CommitmentCategory.COMMUNICATION -> {
                tvCandidateAppsLabel.text = "CHOOSE MESSAGING APP"
            }
            else -> {
                tvCandidateAppsLabel.text = "OPEN WITH APP"
            }
        }

        for (candidate in candidates) {
            val itemView = layoutInflater.inflate(R.layout.item_candidate_app_pill, llCandidateAppIcons, false)
            val ivIcon = itemView.findViewById<ImageView>(R.id.ivCandidateAppIcon)
            val tvName = itemView.findViewById<TextView>(R.id.tvCandidateAppName)

            candidate.icon?.let { ivIcon.setImageDrawable(it) }
            tvName.text = candidate.appName

            itemView.setOnClickListener {
                handleCandidateAppLaunch(candidate)
            }

            llCandidateAppIcons.addView(itemView)
        }
    }

    private fun handleCandidateAppLaunch(candidate: com.zipper.app.core.intelligence.resolver.ResolvedAppCandidate) {
        commitmentId?.let { cid ->
            activityScope.launch(Dispatchers.IO) {
                com.zipper.app.core.intelligence.store.CommitmentDatabase.getInstance(applicationContext)
                    .updateCommitmentStatus(cid, com.zipper.app.core.intelligence.store.CommitmentStatus.IN_PROGRESS)
            }
        }
        reminderId?.let { CallReminderScheduler.markComplete(this, it) }

        ParadeeVoiceEngine.speak("Opening ${candidate.appName} for you. Have a great session!") {
            candidate.launchIntent?.let { intent ->
                try {
                    startActivity(intent)
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to launch ${candidate.packageName}: ${e.message}")
                }
            }
            finishAndDismiss()
        }
    }

    private fun startCallTimer() {
        callDurationJob = activityScope.launch {
            var seconds = 0
            while (isActive) {
                delay(1000L)
                seconds++
                val mins = seconds / 60
                val secs = seconds % 60
                tvCallDuration.text = String.format(Locale.US, "%02d:%02d", mins, secs)
            }
        }
    }

    private fun startVoiceRecognition() {
        if (!SpeechRecognizer.isRecognitionAvailable(this)) return

        activityScope.launch {
            try {
                speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this@IncomingCallActivity)
                val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                    putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                    putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
                    putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 3)
                }

                speechRecognizer?.setRecognitionListener(object : RecognitionListener {
                    override fun onResults(results: Bundle?) {
                        val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                        val text = matches?.firstOrNull()?.lowercase().orEmpty()
                        Log.i(TAG, "Recognized in-call voice response: '$text'")

                        val sid = callSessionId
                        if (sid != null) {
                            val session = DelegatedCallSessionManager.getSession(sid)
                            val contactName = session?.contact?.contactName ?: sourcePerson
                            val dossier = com.zipper.app.core.intelligence.reply.ContactKnowledgeBook.getInstance(applicationContext).getDossier(contactName)
                            val isElder = dossier?.isElderOrFamily() == true
                            val isKan = dossier?.isKannadaPreferred() == true || dossier?.languageStyle == com.zipper.app.core.intelligence.reply.LanguageStyle.KANNADA_REGIONAL

                            if (text.contains("decline") || text.contains("no") || text.contains("cancel") || text.contains("dismiss") || text.contains("beda") || text.contains("illa")) {
                                handleDismiss()
                            } else if (text.contains("yes") || text.contains("confirm") || text.contains("ok") || text.contains("sure") || text.contains("haudu") || text.contains("sari") || text.contains("works") || text.contains("fine")) {
                                stopRinging()
                                DelegatedCallSessionManager.handleUserDecision(applicationContext, sid, "Confirm: that time works for me.")
                                finishAndDismiss()
                            } else if (text.contains("later") || text.contains("snooze") || text.contains("busy")) {
                                stopRinging()
                                DelegatedCallSessionManager.handleUserDecision(applicationContext, sid, "Busy with work right now, will follow up once free.")
                                finishAndDismiss()
                            } else if (text.isNotBlank()) {
                                stopRinging()
                                DelegatedCallSessionManager.handleUserDecision(applicationContext, sid, text)
                                finishAndDismiss()
                            }
                            return
                        }

                        // Interactive Conversational AI Voice Twin loop
                        if (sourcePerson.contains("Voice Twin", true) || reminderTitle.contains("Voice Twin", true)) {
                            if (text.contains("bye") || text.contains("end call") || text.contains("dismiss") || text.contains("exit") || text.contains("close")) {
                                ParadeeVoiceEngine.speak("Catch you in a bit bro! Talk later.") {
                                    finishAndDismiss()
                                }
                                return
                            }

                            if (text.isNotBlank()) {
                                tvVoiceEngineBadge.text = "🧠 Voice Twin thinking..."
                                activityScope.launch(Dispatchers.IO) {
                                    val aiConfig = com.zipper.app.core.web.WebConfigManager(applicationContext).getConfig()
                                    val webLlm = com.zipper.app.core.web.WebLlmClient()
                                    val answer = webLlm.generateChatReply(
                                        contactName = "Karthik",
                                        incomingMessage = text,
                                        recentHistory = emptyList(),
                                        userContext = "You are Karthik's AI Voice Twin speaking on a live voice test with Karthik himself. Speak as a close Bangalore friend with authentic warm slang ('bro', 'ha bro', 'done bro', 'chill'). Keep response to 1-2 spoken sentences.",
                                        personaTone = "Warm, lively Bangalore slang friend, conversational voice persona",
                                        config = aiConfig
                                    ) ?: "I'm right here bro! Mic and brain are fully connected. What's on your mind?"

                                    val cleanSpoken = answer.replace(Regex("""(?i)CALL_USER:[^|]+\|\s*"""), "").trim()
                                    withContext(Dispatchers.Main) {
                                        tvVoiceEngineBadge.text = "🎙️ Speaking..."
                                        tvReminderPrompt.text = "“$cleanSpoken”"
                                        waveformAnimator?.start()
                                        ParadeeVoiceEngine.speak(cleanSpoken) {
                                            waveformAnimator?.cancel()
                                            ivVoiceWaveform.scaleY = 1.0f
                                            tvVoiceEngineBadge.text = "👂 Listening... (speak or say 'bye')"
                                            startVoiceRecognition()
                                        }
                                    }
                                }
                                return
                            }
                        }

                        if (text.contains("done") || text.contains("taken") || text.contains("complete") || text.contains("finish")) {
                            handleDone()
                        } else if (text.contains("don't call") || text.contains("dont call") || text.contains("mute")) {
                            handleMuteToday()
                        } else if (text.contains("dismiss") || text.contains("cancel") || text.contains("no")) {
                            handleDismiss()
                        } else if (text.contains("snooze") || text.contains("later") || text.contains("10 min")) {
                            handleSnooze(minutes = 10)
                        } else if (text.contains("pay") || text.contains("do it") || text.contains("start") || text.contains("run")) {
                            targetGoal?.let { handleAgentExecution(it) } ?: handleDone()
                        }
                    }
                    override fun onReadyForSpeech(params: Bundle?) {}
                    override fun onBeginningOfSpeech() {}
                    override fun onRmsChanged(rmsdB: Float) {}
                    override fun onBufferReceived(buffer: ByteArray?) {}
                    override fun onEndOfSpeech() {}
                    override fun onError(error: Int) {}
                    override fun onPartialResults(partialResults: Bundle?) {}
                    override fun onEvent(eventType: Int, params: Bundle?) {}
                })
                speechRecognizer?.startListening(intent)
            } catch (e: Exception) {
                Log.w(TAG, "Voice recognition init skipped: ${e.message}")
            }
        }
    }

    private fun handleDone() {
        commitmentId?.let { cid ->
            activityScope.launch(Dispatchers.IO) {
                com.zipper.app.core.intelligence.store.CommitmentDatabase.getInstance(applicationContext)
                    .updateCommitmentStatus(cid, com.zipper.app.core.intelligence.store.CommitmentStatus.COMPLETED)
            }
        }
        reminderId?.let { CallReminderScheduler.markComplete(this, it) }
        ParadeeVoiceEngine.speak("Marked done! Have a great day.") {
            finishAndDismiss()
        }
    }

    private fun handleDismiss() {
        callSessionId?.let { sid ->
            DelegatedCallSessionManager.handleDecline(applicationContext, sid)
        }
        commitmentId?.let { cid ->
            activityScope.launch(Dispatchers.IO) {
                com.zipper.app.core.intelligence.store.CommitmentDatabase.getInstance(applicationContext)
                    .updateCommitmentStatus(cid, com.zipper.app.core.intelligence.store.CommitmentStatus.COMPLETED)
            }
        }
        reminderId?.let { CallReminderScheduler.markComplete(this, it) }
        finishAndDismiss()
    }

    private fun handleMuteToday() {
        stopRinging()
        com.zipper.app.core.intelligence.scheduler.LifePlanCallScheduler.muteAllCallsToday(applicationContext)
        commitmentId?.let { cid ->
            activityScope.launch(Dispatchers.IO) {
                com.zipper.app.core.intelligence.store.CommitmentDatabase.getInstance(applicationContext)
                    .updateCommitmentStatus(cid, com.zipper.app.core.intelligence.store.CommitmentStatus.COMPLETED)
            }
        }
        reminderId?.let { CallReminderScheduler.markComplete(this, it) }
        try {
            ParadeeVoiceEngine.speak("Muted all calls for today. Have a peaceful day!") {
                finishAndDismiss()
            }
            activityScope.launch {
                delay(1200L)
                finishAndDismiss()
            }
        } catch (_: Exception) {
            finishAndDismiss()
        }
    }

    private fun handleSnooze(minutes: Int) {
        commitmentId?.let { cid ->
            activityScope.launch(Dispatchers.IO) {
                com.zipper.app.core.intelligence.store.CommitmentDatabase.getInstance(applicationContext)
                    .updateCommitmentStatus(cid, com.zipper.app.core.intelligence.store.CommitmentStatus.SNOOZED)
            }
        }
        reminderId?.let { CallReminderScheduler.snooze(this, it, minutes) }
        ParadeeVoiceEngine.speak("Snoozed for $minutes minutes.") {
            finishAndDismiss()
        }
    }

    private fun handleAgentExecution(goal: String) {
        reminderId?.let { CallReminderScheduler.markComplete(this, it) }
        ParadeeVoiceEngine.speak("Starting Karvish autonomous agent to complete this.") {
            try {
                val serviceIntent = Intent(this@IncomingCallActivity, AgentRuntimeService::class.java).apply {
                    putExtra(AgentRuntimeService.EXTRA_GOAL, goal)
                    putExtra(AgentRuntimeService.EXTRA_RECOVERY, true)
                    putExtra(AgentRuntimeService.EXTRA_USE_REMOTE_RANKER, false)
                }
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    startForegroundService(serviceIntent)
                } else {
                    startService(serviceIntent)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to start AgentRuntimeService: ${e.message}")
            }
            finishAndDismiss()
        }
    }

    private fun finishAndDismiss() {
        stopRinging()
        ParadeeVoiceEngine.stop()
        finish()
    }

    override fun onDestroy() {
        super.onDestroy()
        stopRinging()
        callDurationJob?.cancel()

        callSessionId?.let { sid ->
            DelegatedCallSessionManager.handleDecline(applicationContext, sid)
        }

        try {
            speechRecognizer?.destroy()
        } catch (_: Exception) {}

        try {
            CallScreeningEngine.release()
        } catch (_: Exception) {}

        audioManager?.let { am ->
            try {
                am.mode = originalAudioMode
                @Suppress("DEPRECATION")
                am.isSpeakerphoneOn = originalSpeakerphoneOn
            } catch (_: Exception) {}
        }
    }
}
