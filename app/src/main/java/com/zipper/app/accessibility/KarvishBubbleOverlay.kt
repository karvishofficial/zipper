package com.zipper.app.accessibility

import android.animation.Animator
import android.animation.AnimatorListenerAdapter
import android.animation.ValueAnimator
import android.content.Context
import android.content.Intent
import android.graphics.*
import android.graphics.Outline
import android.graphics.drawable.Drawable
import android.graphics.drawable.GradientDrawable
import android.graphics.drawable.StateListDrawable
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.VibrationEffect
import android.os.Vibrator
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.ViewOutlineProvider
import android.view.WindowManager
import android.view.animation.AccelerateInterpolator
import android.view.animation.DecelerateInterpolator
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.core.content.ContextCompat
import com.zipper.app.core.agent.AgentRuntimeService
import com.zipper.app.core.graphics.Zipper3DCoreView
import com.zipper.app.core.graphics.ZipperCoreState
import com.zipper.app.core.voice.WakeWordService
import java.util.Locale
import kotlin.math.abs
import kotlin.math.sin

/**
 * KarvishBubbleOverlay
 *
 * Universal Movable 3D GLB Cyber Orb Assistant with Dual Soundwave Wings & Floating Thinking Capsule.
 *
 * Architectural Features:
 * 1. Real 3D Liquid GLB Core: Uses the hardware-accelerated Zipper3DCoreView OpenGL ES mesh.
 * 2. 100% Clean Idle Screen: Never stays on screen during normal phone usage.
 * 3. Bottom Wake-Up Entrance: Glides smoothly up from below the bottom edge when "Hey Zipper" is said.
 * 4. Magical Celestial Chime: In-RAM synthesized 210ms C5 -> G5 shimmer chime on wake.
 * 5. Symmetrical Soundwave Wings: Left & Right audio equalizer wings flanking the 3D orb.
 * 6. Floating Thinking / Selecting Capsule: Sits above the 3D ball, streaming live transcription
 *    and agent candidate/step decisions.
 * 7. Anywhere Placable Drag Physics: Drag the orb or capsule to reposition anywhere on screen.
 * 8. Submerge Exit: Smoothly submerges down beneath the bezel on completion.
 */
object KarvishBubbleOverlay {

    private const val TAG = "KarvishBubbleOverlay"
    private const val PREFS_NAME = "karvish_bubble_prefs"
    private const val KEY_ENABLED = "bubble_enabled"

    private var windowManager: WindowManager? = null
    private var containerView: FrameLayout? = null
    private var currentParams: WindowManager.LayoutParams? = null

    var core3dView: Zipper3DCoreView? = null
        private set
    private var leftWaveWing: SoundwaveWingView? = null
    private var rightWaveWing: SoundwaveWingView? = null
    private var speechRecognizer: SpeechRecognizer? = null

    // UI elements
    private var thinkingCapsule: LinearLayout? = null
    private var inspectorCard: LinearLayout? = null
    private var interactiveCard: LinearLayout? = null
    private var ivAppIcon: ImageView? = null
    private var ivInspectorAppIcon: ImageView? = null
    private var tvInspectorHeaderBadge: TextView? = null
    private var tvBrainIcon: TextView? = null
    private var tvThinkingStatus: TextView? = null
    private var tvStatusSubtext: TextView? = null
    private var tvExpandHint: TextView? = null
    private var tvCardGoal: TextView? = null
    private var tvCardAction: TextView? = null
    private var tvCardReasoning: TextView? = null
    private var btnPauseAction: TextView? = null
    private var btnStopAction: TextView? = null

    private val mainHandler = Handler(Looper.getMainLooper())
    private var submergeAnimator: ValueAnimator? = null

    var isShowing: Boolean = false
        private set
    var isListening: Boolean = false
        private set
    var isExecutingGoal: Boolean = false
        private set
    var isExpanded: Boolean = false
        private set

    var activeGoal: String = ""
        private set

    private var onPauseClickListener: ((Boolean) -> Unit)? = null
    private var onCancelClickListener: (() -> Unit)? = null
    private var isAgentPaused: Boolean = false

    private var pendingDismissRunnable: Runnable? = null
    private var activeCardVoiceRecognizer: SpeechRecognizer? = null

    fun cancelPendingDismiss() {
        pendingDismissRunnable?.let {
            mainHandler.removeCallbacks(it)
            pendingDismissRunnable = null
        }
        mainHandler.removeCallbacks(silenceTimeoutRunnable)
    }

    fun scheduleDismiss(delayMs: Long, force: Boolean = false) {
        cancelPendingDismiss()
        if (!force && (isExecutingGoal || interactiveCard?.visibility == View.VISIBLE)) {
            Log.i(TAG, "scheduleDismiss ignored: goal executing ($isExecutingGoal) or card visible (${interactiveCard?.visibility == View.VISIBLE})")
            return
        }
        val r = Runnable {
            pendingDismissRunnable = null
            dismissWithSubmerge(force = force)
        }
        pendingDismissRunnable = r
        mainHandler.postDelayed(r, delayMs)
    }

    private val silenceTimeoutRunnable = Runnable {
        if (isExecutingGoal || interactiveCard?.visibility == View.VISIBLE) {
            Log.i(TAG, "Speech timeout ignored: goal is executing ($isExecutingGoal) or card is visible (${interactiveCard?.visibility == View.VISIBLE})")
            return@Runnable
        }
        Log.i(TAG, "Speech timeout: no voice detected")
        tvThinkingStatus?.text = "No speech detected"
        scheduleDismiss(2500L, force = true)
    }

    fun isEnabled(context: Context): Boolean {
        return context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .getBoolean(KEY_ENABLED, true)
    }

    fun setEnabled(context: Context, enabled: Boolean) {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putBoolean(KEY_ENABLED, enabled)
            .apply()
    }

    fun setAgentControls(onPause: ((Boolean) -> Unit)?, onCancel: (() -> Unit)?) {
        onPauseClickListener = onPause
        onCancelClickListener = onCancel
    }

    fun setCoreState(state: ZipperCoreState) {
        mainHandler.post {
            core3dView?.currentState = state
            when (state) {
                ZipperCoreState.WORKING -> {
                    tvThinkingStatus?.text = "Agent active..."
                }
                ZipperCoreState.COMPLETED -> {
                    tvThinkingStatus?.text = "Task completed ✓"
                    scheduleDismiss(2000L, force = true)
                }
                else -> {}
            }
        }
    }

    fun setAudioAmplitude(amplitude: Float) {
        mainHandler.post {
            core3dView?.audioAmplitude = amplitude
            leftWaveWing?.setAmplitude(amplitude)
            rightWaveWing?.setAmplitude(amplitude)
        }
    }

    /**
     * Starts or connects the 3D Cyber Orb directly for an autonomous agent session.
     */
    fun startAutonomousSession(context: Context, goal: String) {
        mainHandler.post {
            cancelPendingDismiss()
            activeGoal = goal
            isExecutingGoal = true
            if (!isShowing || containerView == null) {
                wakeUpFromBottom(context)
                mainHandler.postDelayed({
                    transitionToAutonomousExecution(context, goal)
                }, 380L)
            } else {
                transitionToAutonomousExecution(context, goal)
            }
        }
    }

    /**
     * Transitions from voice listening to autonomous agent execution:
     * - Keeps the 3D orb ALIVE on screen
     * - Fades out the microphone soundwave wings
     * - Switches the capsule into translucent frosted glass showing thoughts & actions
     */
    fun transitionToAutonomousExecution(context: Context, goal: String) {
        mainHandler.post {
            cancelPendingDismiss()
            isListening = false
            isExecutingGoal = true
            activeGoal = goal

            try {
                speechRecognizer?.stopListening()
                speechRecognizer?.destroy()
                speechRecognizer = null
            } catch (_: Exception) {}
            WakeWordService.notifyAssistantFinished()

            // Smoothly collapse soundwave wings
            leftWaveWing?.animate()?.alpha(0f)?.setDuration(220L)?.withEndAction {
                leftWaveWing?.visibility = View.GONE
            }?.start()
            rightWaveWing?.animate()?.alpha(0f)?.setDuration(220L)?.withEndAction {
                rightWaveWing?.visibility = View.GONE
            }?.start()

            // Update floating capsule to translucent action mode
            val currentPkg = KarvishAccessibilityService.instance?.rootInActiveWindow?.packageName?.toString()
            updateCapsuleIcon(context, currentPkg)

            tvBrainIcon?.text = "🧠"
            tvThinkingStatus?.text = "Planning: $goal"
            tvStatusSubtext?.text = "Analyzing screen state..."
            tvStatusSubtext?.visibility = View.VISIBLE
            tvExpandHint?.visibility = View.VISIBLE

            tvCardGoal?.text = "🎯 $goal"
            tvCardAction?.text = "Initializing plan..."
            tvCardReasoning?.text = "Capturing screen candidates..."

            core3dView?.currentState = ZipperCoreState.WORKING
            core3dView?.resumeRendering()
        }
    }

    private fun updateCapsuleIcon(context: Context, packageName: String?) {
        val pm = context.packageManager
        val isSystemOrSelf = packageName.isNullOrBlank() ||
                packageName == context.packageName ||
                packageName.contains("launcher", ignoreCase = true)

        val iconDrawable: Drawable? = if (!isSystemOrSelf) {
            try {
                pm.getApplicationIcon(packageName!!)
            } catch (_: Exception) {
                null
            }
        } else null

        val appLabel: String? = if (!isSystemOrSelf) {
            try {
                pm.getApplicationLabel(pm.getApplicationInfo(packageName!!, 0)).toString().uppercase()
            } catch (_: Exception) {
                null
            }
        } else null

        if (iconDrawable != null) {
            ivAppIcon?.setImageDrawable(iconDrawable)
            ivAppIcon?.visibility = View.VISIBLE
            tvBrainIcon?.visibility = View.GONE

            ivInspectorAppIcon?.setImageDrawable(iconDrawable)
            ivInspectorAppIcon?.visibility = View.VISIBLE
            tvInspectorHeaderBadge?.text = "${appLabel ?: "APP"} • AUTONOMOUS AGENT"
        } else {
            ivAppIcon?.visibility = View.GONE
            tvBrainIcon?.visibility = View.VISIBLE

            ivInspectorAppIcon?.visibility = View.GONE
            tvInspectorHeaderBadge?.text = "KARVISH • AUTONOMOUS AGENT"
        }
    }

    /**
     * Updates the persistent 3D overlay with live turn & candidate decisions.
     */
    fun updateAgentTurn(
        turn: Int,
        maxTurns: Int,
        actionName: String,
        targetLabel: String,
        statusText: String?,
        subtext: String?,
        packageName: String? = null
    ) {
        mainHandler.post {
            cancelPendingDismiss()
            isExecutingGoal = true
            isListening = false

            leftWaveWing?.visibility = View.GONE
            rightWaveWing?.visibility = View.GONE

            core3dView?.currentState = ZipperCoreState.WORKING

            val resolvedPkg = packageName?.takeIf { it.isNotBlank() }
                ?: KarvishAccessibilityService.instance?.rootInActiveWindow?.packageName?.toString()

            val ctx = containerView?.context
            if (ctx != null) {
                updateCapsuleIcon(ctx, resolvedPkg)
            }

            val icon = when {
                actionName.contains("TYPE", ignoreCase = true) -> "⌨"
                actionName.contains("SCROLL", ignoreCase = true) -> "📜"
                actionName.contains("TAP", ignoreCase = true) -> "⚡"
                actionName.contains("SUBMIT", ignoreCase = true) -> "⏎"
                actionName.contains("BACK", ignoreCase = true) -> "◀"
                else -> "🧠"
            }
            tvBrainIcon?.text = icon

            val displayAction = when {
                statusText != null && statusText.isNotBlank() -> statusText
                targetLabel.isNotBlank() -> "$actionName: $targetLabel"
                else -> actionName
            }
            tvThinkingStatus?.text = displayAction

            val cleanSubtext = when {
                subtext.isNullOrBlank() -> if (targetLabel.isNotBlank()) "Target: $targetLabel" else "Evaluating screen candidates..."
                subtext.contains("Delta:", ignoreCase = true) || subtext.contains("Nodes:", ignoreCase = true) -> "Advancing goal state..."
                else -> subtext
            }
            tvStatusSubtext?.text = cleanSubtext
            tvStatusSubtext?.visibility = View.VISIBLE
            tvExpandHint?.visibility = View.VISIBLE

            // Update expanded card views
            tvCardAction?.text = "Step $turn • $displayAction"
            tvCardReasoning?.text = cleanSubtext
        }
    }

    /**
     * Notifies completion: displays success status, plays haptics, and submerges smoothly.
     */
    fun onGoalCompleted(goal: String, summary: String) {
        mainHandler.post {
            cancelPendingDismiss()
            isExecutingGoal = false
            ivAppIcon?.visibility = View.GONE
            tvBrainIcon?.visibility = View.VISIBLE
            tvBrainIcon?.text = "✓"
            tvThinkingStatus?.text = "Completed: $goal"
            tvStatusSubtext?.text = summary.ifBlank { "All steps successfully verified" }
            tvStatusSubtext?.visibility = View.VISIBLE
            core3dView?.currentState = ZipperCoreState.COMPLETED

            val ctx = containerView?.context
            if (ctx != null) vibrateHeartbeat(ctx)

            scheduleDismiss(2600L, force = true)
        }
    }

    /**
     * Notifies agent cancellation or failure.
     */
    fun onGoalStopped(summary: String) {
        mainHandler.post {
            cancelPendingDismiss()
            isExecutingGoal = false
            ivAppIcon?.visibility = View.GONE
            tvBrainIcon?.visibility = View.VISIBLE
            tvBrainIcon?.text = "⏹"
            tvThinkingStatus?.text = "Agent Stopped"
            tvStatusSubtext?.text = summary
            tvStatusSubtext?.visibility = View.VISIBLE
            core3dView?.currentState = ZipperCoreState.IDLE

            scheduleDismiss(1800L, force = true)
        }
    }

    /**
     * Instant Floating OTP Alert Pill:
     * Appears smoothly when an SMS or WhatsApp OTP is received, copies it to the clipboard,
     * and shows a 1-tap card above the 3D orb with auto-dismiss.
     */
    fun showOtpPill(context: Context, otp: String, appName: String) {
        mainHandler.post {
            cancelPendingDismiss()
            if (!isShowing || containerView == null) {
                wakeUpFromBottom(context)
            }
            isExecutingGoal = false
            ivAppIcon?.visibility = View.GONE
            tvBrainIcon?.visibility = View.VISIBLE
            tvBrainIcon?.text = "🔐"
            tvThinkingStatus?.text = "OTP: $otp ($appName)"
            tvStatusSubtext?.text = "📋 Copied to clipboard • Tap to paste"
            tvStatusSubtext?.visibility = View.VISIBLE
            core3dView?.currentState = ZipperCoreState.WORKING

            try {
                val cm = context.getSystemService(Context.CLIPBOARD_SERVICE) as android.content.ClipboardManager
                cm.setPrimaryClip(android.content.ClipData.newPlainText("OTP", otp))
                vibrateHeartbeat(context)
            } catch (_: Exception) {}

            thinkingCapsule?.setOnClickListener {
                try {
                    val a11y = KarvishAccessibilityService.instance
                    val focused = a11y?.rootInActiveWindow?.findFocus(android.view.accessibility.AccessibilityNodeInfo.FOCUS_INPUT)
                    if (focused != null && focused.isEditable) {
                        val args = android.os.Bundle().apply {
                            putCharSequence(android.view.accessibility.AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, otp)
                        }
                        focused.performAction(android.view.accessibility.AccessibilityNodeInfo.ACTION_SET_TEXT, args)
                        android.widget.Toast.makeText(context, "⚡ Injected OTP: $otp", android.widget.Toast.LENGTH_SHORT).show()
                    } else {
                        android.widget.Toast.makeText(context, "📋 OTP $otp copied to clipboard", android.widget.Toast.LENGTH_SHORT).show()
                    }
                } catch (_: Exception) {}
            }

            scheduleDismiss(18000L, force = true)
        }
    }

    /**
     * Toggles between the compact translucent capsule and the expanded inspector card.
     */
    fun toggleInspectorExpanded() {
        mainHandler.post {
            val root = containerView ?: return@post
            val wm = windowManager ?: return@post
            val params = currentParams ?: return@post
            val density = root.resources.displayMetrics.density

            isExpanded = !isExpanded

            val collapsedHeight = (165 * density).toInt()
            val expandedHeight = (280 * density).toInt()
            val heightDiff = expandedHeight - collapsedHeight

            if (isExpanded) {
                thinkingCapsule?.visibility = View.GONE
                inspectorCard?.visibility = View.VISIBLE
                tvCardGoal?.text = if (activeGoal.isNotBlank()) "🎯 $activeGoal" else "🎯 Active Task"
                params.height = expandedHeight
                params.y = (params.y - heightDiff).coerceAtLeast((24 * density).toInt())
            } else {
                inspectorCard?.visibility = View.GONE
                thinkingCapsule?.visibility = View.VISIBLE
                params.height = collapsedHeight
                params.y = (params.y + heightDiff)
            }

            try {
                wm.updateViewLayout(root, params)
            } catch (_: Exception) {}
        }
    }

    /**
     * Human-in-the-Loop Multi-Choice Disambiguation Card:
     * Displays numbered option buttons directly above the 3D orb with live voice listener.
     */
    fun showDisambiguationCard(
        title: String,
        prompt: String,
        options: List<String>,
        onSelect: (Int) -> Unit,
        onCancel: () -> Unit
    ) {
        mainHandler.post {
            cancelPendingDismiss()
            isExecutingGoal = true

            val root = containerView ?: return@post
            val wm = windowManager ?: return@post
            val params = currentParams ?: return@post
            val card = interactiveCard ?: return@post
            val ctx = card.context
            val density = ctx.resources.displayMetrics.density

            thinkingCapsule?.visibility = View.GONE
            inspectorCard?.visibility = View.GONE
            card.removeAllViews()

            // Header Row: 👥 Icon + Title + Close Button
            val headerRow = LinearLayout(ctx).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply {
                    bottomMargin = (4 * density).toInt()
                }
            }

            val iconTv = TextView(ctx).apply {
                text = "👥 "
                textSize = 14f
            }
            headerRow.addView(iconTv)

            val titleTv = TextView(ctx).apply {
                text = title
                textSize = 13.5f
                setTextColor(Color.WHITE)
                typeface = Typeface.create("sans-serif-bold", Typeface.BOLD)
                layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
            }
            headerRow.addView(titleTv)

            val cancelHeaderBtn = TextView(ctx).apply {
                text = "✕"
                textSize = 14f
                setTextColor(Color.parseColor("#94A3B8"))
                setPadding((8 * density).toInt(), (4 * density).toInt(), (6 * density).toInt(), (4 * density).toInt())
                isClickable = true
                isFocusable = true
                setOnClickListener {
                    try {
                        activeCardVoiceRecognizer?.stopListening()
                        activeCardVoiceRecognizer?.destroy()
                    } catch (_: Exception) {}
                    activeCardVoiceRecognizer = null
                    dismissInteractiveCard()
                    onCancel()
                }
            }
            headerRow.addView(cancelHeaderBtn)
            card.addView(headerRow)

            // Prompt Subtitle
            val promptTv = TextView(ctx).apply {
                text = "$prompt (Say 1, 2, 3 or tap below)"
                textSize = 11.5f
                setTextColor(Color.parseColor("#38BDF8"))
                typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply {
                    bottomMargin = (10 * density).toInt()
                }
            }
            card.addView(promptTv)

            fun choose(selectedIdx: Int) {
                try {
                    activeCardVoiceRecognizer?.stopListening()
                    activeCardVoiceRecognizer?.destroy()
                } catch (_: Exception) {}
                activeCardVoiceRecognizer = null
                dismissInteractiveCard()
                onSelect(selectedIdx)
            }

            fun abort() {
                try {
                    activeCardVoiceRecognizer?.stopListening()
                    activeCardVoiceRecognizer?.destroy()
                } catch (_: Exception) {}
                activeCardVoiceRecognizer = null
                dismissInteractiveCard()
                onCancel()
            }

            // Options Stack
            val optionsStack = LinearLayout(ctx).apply {
                orientation = LinearLayout.VERTICAL
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                )
            }

            options.take(5).forEachIndexed { idx, optText ->
                val optBtn = TextView(ctx).apply {
                    text = "[${idx + 1}]  $optText"
                    textSize = 12.5f
                    setTextColor(Color.WHITE)
                    typeface = Typeface.create("sans-serif-bold", Typeface.BOLD)
                    setPadding((14 * density).toInt(), (10 * density).toInt(), (14 * density).toInt(), (10 * density).toInt())
                    isClickable = true
                    isFocusable = true
                    background = StateListDrawable().apply {
                        val pressedBg = GradientDrawable().apply {
                            shape = GradientDrawable.RECTANGLE
                            cornerRadius = 14f * density
                            setColor(Color.parseColor("#0284C7"))
                            setStroke((1.5f * density).toInt(), Color.parseColor("#38BDF8"))
                        }
                        val normalBg = GradientDrawable().apply {
                            shape = GradientDrawable.RECTANGLE
                            cornerRadius = 14f * density
                            setColor(Color.parseColor("#1E293B"))
                            setStroke((1.2f * density).toInt(), Color.parseColor("#38BDF8"))
                        }
                        addState(intArrayOf(android.R.attr.state_pressed), pressedBg)
                        addState(intArrayOf(), normalBg)
                    }
                    layoutParams = LinearLayout.LayoutParams(
                        LinearLayout.LayoutParams.MATCH_PARENT,
                        LinearLayout.LayoutParams.WRAP_CONTENT
                    ).apply {
                        bottomMargin = (7 * density).toInt()
                    }
                    setOnClickListener {
                        Log.i(TAG, "[DISAMBIGUATION] Option $idx tapped: $optText")
                        choose(idx)
                    }
                }
                optionsStack.addView(optBtn)
            }
            card.addView(optionsStack)

            // Voice Choice Recognizer - Started after brief delay so Paradee TTS finishes prompt speaking
            try {
                activeCardVoiceRecognizer?.destroy()
            } catch (_: Exception) {}
            activeCardVoiceRecognizer = null

            if (SpeechRecognizer.isRecognitionAvailable(ctx)) {
                mainHandler.postDelayed({
                    if (card.visibility != View.VISIBLE) return@postDelayed
                    try {
                        val vr = SpeechRecognizer.createSpeechRecognizer(ctx).apply {
                            setRecognitionListener(object : RecognitionListener {
                                override fun onReadyForSpeech(params: Bundle?) {}
                                override fun onBeginningOfSpeech() {}
                                override fun onRmsChanged(rmsdB: Float) {}
                                override fun onBufferReceived(buffer: ByteArray?) {}
                                override fun onEndOfSpeech() {}
                                override fun onError(error: Int) {
                                    Log.d(TAG, "Disambiguation voice recognition error=$error (options card remains active)")
                                }
                                override fun onResults(results: Bundle?) {
                                    val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                                    val spoken = matches?.firstOrNull()?.trim()?.lowercase(Locale.ROOT).orEmpty()
                                    Log.i(TAG, "[DISAMBIGUATION] Spoken: '$spoken'")
                                    when {
                                        spoken.contains("1") || spoken.contains("one") || spoken.contains("first") -> choose(0)
                                        spoken.contains("2") || spoken.contains("two") || spoken.contains("second") -> choose(1)
                                        spoken.contains("3") || spoken.contains("three") || spoken.contains("third") -> choose(2)
                                        spoken.contains("4") || spoken.contains("four") || spoken.contains("fourth") -> choose(3)
                                        spoken.contains("5") || spoken.contains("five") || spoken.contains("fifth") -> choose(4)
                                        spoken == "cancel" || spoken == "stop" || spoken == "abort" || spoken == "cancel that" || spoken == "close" -> abort()
                                        else -> {
                                            val matchIdx = options.indexOfFirst { opt ->
                                                val cleanOpt = opt.lowercase(Locale.ROOT).trim()
                                                spoken.contains(cleanOpt) || cleanOpt.contains(spoken)
                                            }
                                            if (matchIdx != -1) choose(matchIdx)
                                        }
                                    }
                                }
                                override fun onPartialResults(partialResults: Bundle?) {}
                                override fun onEvent(eventType: Int, params: Bundle?) {}
                            })
                        }
                        activeCardVoiceRecognizer = vr
                        val vIntent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
                        }
                        vr.startListening(vIntent)
                    } catch (e: Exception) {
                        Log.w(TAG, "Voice disambiguation error: ${e.message}")
                    }
                }, 1200L)
            }

            // Expand height dynamically while keeping 3D orb anchored
            val baseCardHeight = (135 + (options.size.coerceAtMost(5) * 44)) * density
            val targetHeight = baseCardHeight.toInt()
            val heightDiff = targetHeight - params.height
            params.height = targetHeight
            params.y = (params.y - heightDiff).coerceAtLeast((24 * density).toInt())

            card.visibility = View.VISIBLE
            try {
                wm.updateViewLayout(root, params)
            } catch (_: Exception) {}
        }
    }

    /**
     * Confirmation Card: Displays explicit approve/cancel buttons above 3D orb.
     */
    fun showConfirmationCard(
        title: String,
        prompt: String,
        confirmText: String = "Confirm",
        cancelText: String = "Cancel",
        onConfirm: () -> Unit,
        onCancel: () -> Unit
    ) {
        mainHandler.post {
            cancelPendingDismiss()
            isExecutingGoal = true

            val root = containerView ?: return@post
            val wm = windowManager ?: return@post
            val params = currentParams ?: return@post
            val card = interactiveCard ?: return@post
            val ctx = card.context
            val density = ctx.resources.displayMetrics.density

            thinkingCapsule?.visibility = View.GONE
            inspectorCard?.visibility = View.GONE
            card.removeAllViews()

            val titleTv = TextView(ctx).apply {
                text = "🛡️ $title"
                textSize = 13.5f
                setTextColor(Color.WHITE)
                typeface = Typeface.create("sans-serif-bold", Typeface.BOLD)
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply {
                    bottomMargin = (4 * density).toInt()
                }
            }
            card.addView(titleTv)

            val promptTv = TextView(ctx).apply {
                text = prompt
                textSize = 12f
                setTextColor(Color.parseColor("#FDE68A"))
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply {
                    bottomMargin = (12 * density).toInt()
                }
            }
            card.addView(promptTv)

            val btnRow = LinearLayout(ctx).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                )
            }

            val btnOk = TextView(ctx).apply {
                text = confirmText
                textSize = 12f
                setTextColor(Color.WHITE)
                typeface = Typeface.create("sans-serif-bold", Typeface.BOLD)
                gravity = Gravity.CENTER
                isClickable = true
                isFocusable = true
                background = StateListDrawable().apply {
                    val pressedBg = GradientDrawable().apply {
                        shape = GradientDrawable.RECTANGLE
                        cornerRadius = 14f * density
                        setColor(Color.parseColor("#059669"))
                    }
                    val normalBg = GradientDrawable().apply {
                        shape = GradientDrawable.RECTANGLE
                        cornerRadius = 14f * density
                        setColor(Color.parseColor("#10B981"))
                    }
                    addState(intArrayOf(android.R.attr.state_pressed), pressedBg)
                    addState(intArrayOf(), normalBg)
                }
                setPadding((12 * density).toInt(), (8 * density).toInt(), (12 * density).toInt(), (8 * density).toInt())
                layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply {
                    marginEnd = (8 * density).toInt()
                }
                setOnClickListener {
                    dismissInteractiveCard()
                    onConfirm()
                }
            }
            btnRow.addView(btnOk)

            val btnCancel = TextView(ctx).apply {
                text = cancelText
                textSize = 12f
                setTextColor(Color.WHITE)
                typeface = Typeface.create("sans-serif-bold", Typeface.BOLD)
                gravity = Gravity.CENTER
                isClickable = true
                isFocusable = true
                background = StateListDrawable().apply {
                    val pressedBg = GradientDrawable().apply {
                        shape = GradientDrawable.RECTANGLE
                        cornerRadius = 14f * density
                        setColor(Color.parseColor("#334155"))
                    }
                    val normalBg = GradientDrawable().apply {
                        shape = GradientDrawable.RECTANGLE
                        cornerRadius = 14f * density
                        setColor(Color.parseColor("#475569"))
                    }
                    addState(intArrayOf(android.R.attr.state_pressed), pressedBg)
                    addState(intArrayOf(), normalBg)
                }
                setPadding((12 * density).toInt(), (8 * density).toInt(), (12 * density).toInt(), (8 * density).toInt())
                layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
                setOnClickListener {
                    dismissInteractiveCard()
                    onCancel()
                }
            }
            btnRow.addView(btnCancel)
            card.addView(btnRow)

            val targetHeight = (185 * density).toInt()
            val heightDiff = targetHeight - params.height
            params.height = targetHeight
            params.y = (params.y - heightDiff).coerceAtLeast((24 * density).toInt())

            card.visibility = View.VISIBLE
            try {
                wm.updateViewLayout(root, params)
            } catch (_: Exception) {}
        }
    }

    /**
     * Smoothly collapses any open disambiguation or confirmation card.
     */
    fun dismissInteractiveCard() {
        mainHandler.post {
            try {
                activeCardVoiceRecognizer?.stopListening()
                activeCardVoiceRecognizer?.destroy()
            } catch (_: Exception) {}
            activeCardVoiceRecognizer = null

            val root = containerView ?: return@post
            val wm = windowManager ?: return@post
            val params = currentParams ?: return@post
            val card = interactiveCard ?: return@post
            val density = root.resources.displayMetrics.density

            card.removeAllViews()
            card.visibility = View.GONE
            thinkingCapsule?.visibility = View.VISIBLE

            val collapsedHeight = (165 * density).toInt()
            val heightDiff = params.height - collapsedHeight
            params.height = collapsedHeight
            params.y = (params.y + heightDiff)

            try {
                wm.updateViewLayout(root, params)
            } catch (_: Exception) {}
        }
    }

    /**
     * Wakes up the Real 3D GLB Cyber Orb from below the bottom edge.
     */
    fun wakeUpFromBottom(context: Context) {
        mainHandler.post {
            if (isShowing && containerView != null) {
                core3dView?.resumeRendering()
                startListening(context)
                return@post
            }

            try {
                val a11y = KarvishAccessibilityService.instance
                val targetContext = a11y ?: context
                val wm = targetContext.getSystemService(Context.WINDOW_SERVICE) as? WindowManager
                windowManager = wm
                if (wm == null) {
                    Log.e(TAG, "WindowManager is null, cannot wake up 3D cyber orb")
                    return@post
                }

                val overlayType = if (a11y != null) {
                    WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY
                } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
                } else {
                    @Suppress("DEPRECATION")
                    WindowManager.LayoutParams.TYPE_PHONE
                }

                val metrics = targetContext.resources.displayMetrics
                val density = metrics.density
                val screenWidth = metrics.widthPixels
                val screenHeight = metrics.heightPixels

                val overlayWidth = minOf((340 * density).toInt(), screenWidth - (16 * density).toInt())
                val overlayHeight = (165 * density).toInt()

                val targetX = (screenWidth - overlayWidth) / 2
                val targetY = screenHeight - overlayHeight - (48 * density).toInt()

                val params = WindowManager.LayoutParams(
                    overlayWidth,
                    overlayHeight,
                    overlayType,
                    WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                            WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
                    PixelFormat.TRANSLUCENT
                ).apply {
                    gravity = Gravity.TOP or Gravity.START
                    x = targetX
                    y = screenHeight // Start hidden below screen edge
                }
                currentParams = params

                // Main root container
                val root = FrameLayout(targetContext).apply {
                    clipChildren = false
                    clipToPadding = false
                }

                // Vertical column: [Thinking Capsule / Inspector Card] above, [Wings + 3D GLB Core] below
                val mainColumn = LinearLayout(targetContext).apply {
                    orientation = LinearLayout.VERTICAL
                    gravity = Gravity.CENTER_HORIZONTAL
                    clipChildren = false
                    clipToPadding = false
                }

                // -------------------------------------------------------------
                // 1. FLOATING TRANSLUCENT CAPSULE (Above the Orb)
                // -------------------------------------------------------------
                val capsule = LinearLayout(targetContext).apply {
                    orientation = LinearLayout.HORIZONTAL
                    gravity = Gravity.CENTER_VERTICAL
                    setPadding((14 * density).toInt(), (8 * density).toInt(), (12 * density).toInt(), (8 * density).toInt())

                    // Translucent Obsidian Frosted Glass (83% opacity with subtle neon cyan micro-border)
                    background = GradientDrawable().apply {
                        shape = GradientDrawable.RECTANGLE
                        cornerRadius = 20f * density
                        setColor(Color.parseColor("#D40B0F19"))
                        setStroke((1.2f * density).toInt(), Color.parseColor("#4438BDF8"))
                    }
                    elevation = 10f * density

                    layoutParams = LinearLayout.LayoutParams(
                        LinearLayout.LayoutParams.WRAP_CONTENT,
                        LinearLayout.LayoutParams.WRAP_CONTENT
                    ).apply {
                        bottomMargin = (8 * density).toInt()
                    }

                    // Tapping capsule expands details if executing
                    setOnClickListener {
                        if (isExecutingGoal) {
                            toggleInspectorExpanded()
                        }
                    }
                }
                thinkingCapsule = capsule

                // App Icon (Rounded adaptive app icon of current foreground app)
                val appIconView = ImageView(targetContext).apply {
                    val iconSize = (18 * density).toInt()
                    layoutParams = LinearLayout.LayoutParams(iconSize, iconSize).apply {
                        marginEnd = (6 * density).toInt()
                    }
                    scaleType = ImageView.ScaleType.FIT_CENTER
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
                        clipToOutline = true
                        outlineProvider = object : ViewOutlineProvider() {
                            override fun getOutline(view: View, outline: Outline) {
                                outline.setRoundRect(0, 0, view.width, view.height, 4f * density)
                            }
                        }
                    }
                    visibility = View.GONE
                }
                ivAppIcon = appIconView
                capsule.addView(appIconView)

                // Indicator Icon (✨ Listening, 🧠 Thinking, ⚡ Tap, ⌨ Type, ✓ Done)
                val brainIcon = TextView(targetContext).apply {
                    text = "✨"
                    textSize = 13f
                    layoutParams = LinearLayout.LayoutParams(
                        LinearLayout.LayoutParams.WRAP_CONTENT,
                        LinearLayout.LayoutParams.WRAP_CONTENT
                    ).apply {
                        marginEnd = (6 * density).toInt()
                    }
                }
                tvBrainIcon = brainIcon
                capsule.addView(brainIcon)

                // Vertical Text Block: Main Action + Sub-goal Thought
                val textCol = LinearLayout(targetContext).apply {
                    orientation = LinearLayout.VERTICAL
                    gravity = Gravity.START
                    layoutParams = LinearLayout.LayoutParams(
                        LinearLayout.LayoutParams.WRAP_CONTENT,
                        LinearLayout.LayoutParams.WRAP_CONTENT
                    ).apply {
                        marginEnd = (8 * density).toInt()
                    }
                }

                val tvStatus = TextView(targetContext).apply {
                    text = "Listening for goal..."
                    textSize = 12f
                    setTextColor(Color.parseColor("#F8FAFC"))
                    typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
                    maxWidth = (210 * density).toInt()
                    isSingleLine = true
                    ellipsize = android.text.TextUtils.TruncateAt.END
                }
                tvThinkingStatus = tvStatus
                textCol.addView(tvStatus)

                val tvSubtext = TextView(targetContext).apply {
                    text = "Analyzing screen state..."
                    textSize = 10f
                    setTextColor(Color.parseColor("#94A3B8"))
                    typeface = Typeface.create("sans-serif", Typeface.NORMAL)
                    maxWidth = (210 * density).toInt()
                    isSingleLine = true
                    ellipsize = android.text.TextUtils.TruncateAt.END
                    visibility = View.GONE
                }
                tvStatusSubtext = tvSubtext
                textCol.addView(tvSubtext)

                capsule.addView(textCol)

                // Expand Hint indicator (▲)
                val tvExpand = TextView(targetContext).apply {
                    text = "▲"
                    textSize = 10f
                    setTextColor(Color.parseColor("#64748B"))
                    setPadding((3 * density).toInt(), (2 * density).toInt(), (4 * density).toInt(), (2 * density).toInt())
                    visibility = View.GONE
                }
                tvExpandHint = tvExpand
                capsule.addView(tvExpand)

                // Minimalist Close Button (✕)
                val tvClose = TextView(targetContext).apply {
                    text = "✕"
                    textSize = 13f
                    setTextColor(Color.parseColor("#94A3B8"))
                    setPadding((6 * density).toInt(), (2 * density).toInt(), (6 * density).toInt(), (2 * density).toInt())
                    setOnClickListener {
                        if (isExecutingGoal) {
                            onCancelClickListener?.invoke()
                        }
                        dismissWithSubmerge(force = true)
                    }
                }
                capsule.addView(tvClose)

                mainColumn.addView(capsule)

                // -------------------------------------------------------------
                // 1B. EXPANDABLE INSPECTOR CARD (Shown when user taps the Orb)
                // -------------------------------------------------------------
                val card = LinearLayout(targetContext).apply {
                    orientation = LinearLayout.VERTICAL
                    setPadding((14 * density).toInt(), (12 * density).toInt(), (14 * density).toInt(), (12 * density).toInt())

                    background = GradientDrawable().apply {
                        shape = GradientDrawable.RECTANGLE
                        cornerRadius = 20f * density
                        setColor(Color.parseColor("#EB0B0F19")) // 92% obsidian glass
                        setStroke((1.4f * density).toInt(), Color.parseColor("#5538BDF8")) // Neon cyan glow border
                    }
                    elevation = 12f * density

                    layoutParams = LinearLayout.LayoutParams(
                        (310 * density).toInt(),
                        LinearLayout.LayoutParams.WRAP_CONTENT
                    ).apply {
                        bottomMargin = (8 * density).toInt()
                    }
                    visibility = View.GONE
                }
                inspectorCard = card

                // Card Header Row: Badge + Minimize Button
                val cardHeaderRow = LinearLayout(targetContext).apply {
                    orientation = LinearLayout.HORIZONTAL
                    gravity = Gravity.CENTER_VERTICAL
                    layoutParams = LinearLayout.LayoutParams(
                        LinearLayout.LayoutParams.MATCH_PARENT,
                        LinearLayout.LayoutParams.WRAP_CONTENT
                    ).apply {
                        bottomMargin = (6 * density).toInt()
                    }
                }

                val inspectorIcon = ImageView(targetContext).apply {
                    val iconSize = (14 * density).toInt()
                    layoutParams = LinearLayout.LayoutParams(iconSize, iconSize).apply {
                        marginEnd = (5 * density).toInt()
                    }
                    scaleType = ImageView.ScaleType.FIT_CENTER
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
                        clipToOutline = true
                        outlineProvider = object : ViewOutlineProvider() {
                            override fun getOutline(view: View, outline: Outline) {
                                outline.setRoundRect(0, 0, view.width, view.height, 3f * density)
                            }
                        }
                    }
                    visibility = View.GONE
                }
                ivInspectorAppIcon = inspectorIcon
                cardHeaderRow.addView(inspectorIcon)

                val headerBadge = TextView(targetContext).apply {
                    text = "KARVISH • AUTONOMOUS AGENT"
                    textSize = 10f
                    setTextColor(Color.parseColor("#38BDF8"))
                    typeface = Typeface.create("sans-serif-bold", Typeface.BOLD)
                    letterSpacing = 0.06f
                    layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
                }
                tvInspectorHeaderBadge = headerBadge
                cardHeaderRow.addView(headerBadge)

                val btnCollapse = TextView(targetContext).apply {
                    text = "▼ MINIMIZE"
                    textSize = 10f
                    setTextColor(Color.parseColor("#94A3B8"))
                    typeface = Typeface.create("sans-serif-bold", Typeface.BOLD)
                    setPadding((4 * density).toInt(), (2 * density).toInt(), (4 * density).toInt(), (2 * density).toInt())
                    setOnClickListener {
                        toggleInspectorExpanded()
                    }
                }
                cardHeaderRow.addView(btnCollapse)
                card.addView(cardHeaderRow)

                // Goal Title
                val cardGoal = TextView(targetContext).apply {
                    text = "🎯 Active Task"
                    textSize = 12.5f
                    setTextColor(Color.parseColor("#F8FAFC"))
                    typeface = Typeface.create("sans-serif-bold", Typeface.BOLD)
                    maxLines = 2
                    ellipsize = android.text.TextUtils.TruncateAt.END
                    layoutParams = LinearLayout.LayoutParams(
                        LinearLayout.LayoutParams.MATCH_PARENT,
                        LinearLayout.LayoutParams.WRAP_CONTENT
                    ).apply {
                        bottomMargin = (4 * density).toInt()
                    }
                }
                tvCardGoal = cardGoal
                card.addView(cardGoal)

                // Action & Thought Details
                val cardAction = TextView(targetContext).apply {
                    text = "⚡ Initializing..."
                    textSize = 11.5f
                    setTextColor(Color.parseColor("#E2E8F0"))
                    typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
                    maxLines = 1
                    ellipsize = android.text.TextUtils.TruncateAt.END
                }
                tvCardAction = cardAction
                card.addView(cardAction)

                val cardReasoning = TextView(targetContext).apply {
                    text = "Reasoning: Evaluating screen state..."
                    textSize = 10.5f
                    setTextColor(Color.parseColor("#94A3B8"))
                    maxLines = 2
                    ellipsize = android.text.TextUtils.TruncateAt.END
                    layoutParams = LinearLayout.LayoutParams(
                        LinearLayout.LayoutParams.MATCH_PARENT,
                        LinearLayout.LayoutParams.WRAP_CONTENT
                    ).apply {
                        bottomMargin = (10 * density).toInt()
                    }
                }
                tvCardReasoning = cardReasoning
                card.addView(cardReasoning)

                // Action Controls Row: Pause / Resume & Stop
                val controlsRow = LinearLayout(targetContext).apply {
                    orientation = LinearLayout.HORIZONTAL
                    gravity = Gravity.CENTER_VERTICAL
                    layoutParams = LinearLayout.LayoutParams(
                        LinearLayout.LayoutParams.MATCH_PARENT,
                        LinearLayout.LayoutParams.WRAP_CONTENT
                    )
                }

                val btnPause = TextView(targetContext).apply {
                    text = "⏸ Pause"
                    textSize = 11f
                    setTextColor(Color.parseColor("#38BDF8"))
                    typeface = Typeface.create("sans-serif-bold", Typeface.BOLD)
                    gravity = Gravity.CENTER
                    background = GradientDrawable().apply {
                        shape = GradientDrawable.RECTANGLE
                        cornerRadius = 14f * density
                        setColor(Color.parseColor("#1E293B"))
                        setStroke((1f * density).toInt(), Color.parseColor("#38BDF8"))
                    }
                    setPadding((12 * density).toInt(), (6 * density).toInt(), (12 * density).toInt(), (6 * density).toInt())
                    layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply {
                        marginEnd = (8 * density).toInt()
                    }
                    setOnClickListener {
                        isAgentPaused = !isAgentPaused
                        text = if (isAgentPaused) "▶ Resume" else "⏸ Pause"
                        onPauseClickListener?.invoke(isAgentPaused)
                    }
                }
                btnPauseAction = btnPause
                controlsRow.addView(btnPause)

                val btnStop = TextView(targetContext).apply {
                    text = "⏹ Stop"
                    textSize = 11f
                    setTextColor(Color.WHITE)
                    typeface = Typeface.create("sans-serif-bold", Typeface.BOLD)
                    gravity = Gravity.CENTER
                    background = GradientDrawable().apply {
                        shape = GradientDrawable.RECTANGLE
                        cornerRadius = 14f * density
                        setColor(Color.parseColor("#DC2626"))
                    }
                    setPadding((14 * density).toInt(), (6 * density).toInt(), (14 * density).toInt(), (6 * density).toInt())
                    layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
                    setOnClickListener {
                        onCancelClickListener?.invoke()
                        val stopIntent = Intent(targetContext, AgentRuntimeService::class.java).apply {
                            action = AgentRuntimeService.ACTION_STOP_AGENT
                        }
                        targetContext.startService(stopIntent)
                        onGoalStopped("Cancelled by user")
                    }
                }
                btnStopAction = btnStop
                controlsRow.addView(btnStop)

                card.addView(controlsRow)
                mainColumn.addView(card)

                // -------------------------------------------------------------
                // 1C. INTERACTIVE MODAL SELECTION CARD (Disambiguation / Choices)
                // -------------------------------------------------------------
                val modalContainer = LinearLayout(targetContext).apply {
                    orientation = LinearLayout.VERTICAL
                    setPadding((14 * density).toInt(), (12 * density).toInt(), (14 * density).toInt(), (12 * density).toInt())
                    background = GradientDrawable().apply {
                        shape = GradientDrawable.RECTANGLE
                        cornerRadius = 20f * density
                        setColor(Color.parseColor("#F00B0F19")) // 94% obsidian glass
                        setStroke((1.4f * density).toInt(), Color.parseColor("#6638BDF8")) // Neon cyan glow border
                    }
                    elevation = 14f * density
                    layoutParams = LinearLayout.LayoutParams(
                        (320 * density).toInt(),
                        LinearLayout.LayoutParams.WRAP_CONTENT
                    ).apply {
                        bottomMargin = (8 * density).toInt()
                    }
                    visibility = View.GONE
                }
                interactiveCard = modalContainer
                mainColumn.addView(modalContainer)

                // -------------------------------------------------------------
                // 2. 3D GLB ORB ROW: [LEFT WING] + [REAL 3D GLB CORE] + [RIGHT WING]
                // -------------------------------------------------------------
                val orbRow = LinearLayout(targetContext).apply {
                    orientation = LinearLayout.HORIZONTAL
                    gravity = Gravity.CENTER_VERTICAL
                    clipChildren = false
                    clipToPadding = false
                    layoutParams = LinearLayout.LayoutParams(
                        LinearLayout.LayoutParams.WRAP_CONTENT,
                        LinearLayout.LayoutParams.WRAP_CONTENT
                    )
                }

                // Left Symmetrical Soundwave Wing
                val leftWing = SoundwaveWingView(targetContext, isLeft = true).apply {
                    layoutParams = LinearLayout.LayoutParams((40 * density).toInt(), (34 * density).toInt()).apply {
                        marginEnd = (6 * density).toInt()
                    }
                }
                leftWaveWing = leftWing
                orbRow.addView(leftWing)

                // REAL 3D GLB LIQUID-METAL CORE (Zipper3DCoreView)
                val core = Zipper3DCoreView(targetContext).apply {
                    val coreSize = (88 * density).toInt()
                    layoutParams = LinearLayout.LayoutParams(coreSize, coreSize)
                    currentState = ZipperCoreState.LISTENING
                    isDockedIdle = false

                    // Tapping the 3D core toggles the expanded inspection view!
                    setOnClickListener {
                        if (isExecutingGoal) {
                            toggleInspectorExpanded()
                        }
                    }
                }
                core3dView = core
                orbRow.addView(core)

                // Right Symmetrical Soundwave Wing
                val rightWing = SoundwaveWingView(targetContext, isLeft = false).apply {
                    layoutParams = LinearLayout.LayoutParams((40 * density).toInt(), (34 * density).toInt()).apply {
                        marginStart = (6 * density).toInt()
                    }
                }
                rightWaveWing = rightWing
                orbRow.addView(rightWing)

                mainColumn.addView(orbRow)

                // -------------------------------------------------------------
                // 3. MOVABLE TOUCH & DRAG PHYSICS (Attached to 3D Orb Handle)
                // -------------------------------------------------------------
                orbRow.setOnTouchListener(object : View.OnTouchListener {
                    private var downX = 0
                    private var downY = 0
                    private var touchRawX = 0f
                    private var touchRawY = 0f
                    private var isDragging = false
                    private var downTime = 0L

                    override fun onTouch(v: View, event: MotionEvent): Boolean {
                        val currentWm = windowManager ?: return false
                        val currentP = currentParams ?: return false

                        when (event.action) {
                            MotionEvent.ACTION_DOWN -> {
                                downX = currentP.x
                                downY = currentP.y
                                touchRawX = event.rawX
                                touchRawY = event.rawY
                                isDragging = false
                                downTime = System.currentTimeMillis()
                                cancelPendingDismiss()
                                return true
                            }
                            MotionEvent.ACTION_MOVE -> {
                                val dx = (event.rawX - touchRawX).toInt()
                                val dy = (event.rawY - touchRawY).toInt()

                                if (abs(dx) > 8 || abs(dy) > 8) {
                                    isDragging = true
                                    val sw = targetContext.resources.displayMetrics.widthPixels
                                    val sh = targetContext.resources.displayMetrics.heightPixels

                                    currentP.x = (downX + dx).coerceIn(8, sw - overlayWidth - 8)
                                    currentP.y = (downY + dy).coerceIn(24, sh - currentP.height - 24)
                                    try {
                                        currentWm.updateViewLayout(containerView, currentP)
                                    } catch (_: Exception) {}
                                }
                                return true
                            }
                            MotionEvent.ACTION_UP -> {
                                val pressDuration = System.currentTimeMillis() - downTime
                                val moveDist = abs(event.rawX - touchRawX) + abs(event.rawY - touchRawY)
                                if (!isDragging && moveDist < 16 && pressDuration < 450) {
                                    if (isExecutingGoal) {
                                        toggleInspectorExpanded()
                                    } else {
                                        dismissWithSubmerge(force = true)
                                    }
                                } else {
                                    if (isListening) {
                                        mainHandler.removeCallbacks(silenceTimeoutRunnable)
                                        mainHandler.postDelayed(silenceTimeoutRunnable, 9000L)
                                    }
                                }
                                return true
                            }
                        }
                        return false
                    }
                })

                root.addView(mainColumn, FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.WRAP_CONTENT))

                wm.addView(root, params)
                containerView = root
                isShowing = true

                // 4. Play Magical Celestial Harmonic Chime + Heartbeat Haptics
                MagicalChimePlayer.play()
                vibrateHeartbeat(targetContext)

                // 5. Entrance Slide-Up Animation: Glides smoothly up from bottom into targetY
                submergeAnimator?.cancel()
                submergeAnimator = ValueAnimator.ofInt(screenHeight, targetY).apply {
                    duration = 340L
                    interpolator = DecelerateInterpolator(2.2f)
                    addUpdateListener {
                        params.y = it.animatedValue as Int
                        try {
                            wm.updateViewLayout(root, params)
                        } catch (_: Exception) {}
                    }
                    addListener(object : AnimatorListenerAdapter() {
                        override fun onAnimationEnd(animation: Animator) {
                            if (!isExecutingGoal) {
                                startListening(targetContext)
                            }
                        }
                    })
                    start()
                }

                Log.i(TAG, "Karvish 3D GLB Cyber Orb awakened at bottom")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to wake up 3D cyber orb assistant: ${e.message}")
            }
        }
    }

    fun show(context: Context) {
        wakeUpFromBottom(context)
    }

    fun dockToEdge(animated: Boolean = true) {}
    fun undockFromEdge(animated: Boolean = true, onComplete: (() -> Unit)? = null) {
        onComplete?.invoke()
    }

    private fun startListening(context: Context) {
        cancelPendingDismiss()
        isListening = true
        isExecutingGoal = false
        WakeWordService.notifyAssistantStarted()
        core3dView?.currentState = ZipperCoreState.LISTENING
        core3dView?.resumeRendering()
        leftWaveWing?.visibility = View.VISIBLE
        rightWaveWing?.visibility = View.VISIBLE
        leftWaveWing?.alpha = 1f
        rightWaveWing?.alpha = 1f
        leftWaveWing?.startActiveAnimation()
        rightWaveWing?.startActiveAnimation()
        ivAppIcon?.visibility = View.GONE
        tvBrainIcon?.visibility = View.VISIBLE
        tvBrainIcon?.text = "✨"
        tvThinkingStatus?.text = "Listening for goal..."
        tvStatusSubtext?.visibility = View.GONE
        tvExpandHint?.visibility = View.GONE

        if (!SpeechRecognizer.isRecognitionAvailable(context)) {
            tvThinkingStatus?.text = "Speech recognizer unavailable"
            scheduleDismiss(2500L, force = true)
            return
        }

        mainHandler.removeCallbacks(silenceTimeoutRunnable)
        mainHandler.postDelayed(silenceTimeoutRunnable, 9000L)

        try {
            speechRecognizer?.destroy()
            speechRecognizer = SpeechRecognizer.createSpeechRecognizer(context).apply {
                setRecognitionListener(object : RecognitionListener {
                    override fun onReadyForSpeech(params: Bundle?) {
                        tvThinkingStatus?.text = "Listening (e.g. search SpaceX)..."
                    }

                    override fun onBeginningOfSpeech() {
                        cancelPendingDismiss()
                        tvThinkingStatus?.text = "Hearing voice..."
                    }

                    override fun onRmsChanged(rmsdB: Float) {
                        val amp = (rmsdB / 10f).coerceIn(0f, 1f)
                        setAudioAmplitude(amp)
                    }

                    override fun onBufferReceived(buffer: ByteArray?) {}
                    override fun onEndOfSpeech() {
                        if (isExecutingGoal || interactiveCard?.visibility == View.VISIBLE) return
                        tvThinkingStatus?.text = "Processing goal..."
                        core3dView?.currentState = ZipperCoreState.THINKING
                    }

                    override fun onError(error: Int) {
                        cancelPendingDismiss()
                        if (isExecutingGoal || interactiveCard?.visibility == View.VISIBLE) {
                            Log.i(TAG, "startListening onError($error) ignored because goal is active")
                            return
                        }
                        Log.w(TAG, "3D Orb speech recognition error: $error")
                        tvThinkingStatus?.text = "Didn't catch that (Tap to close)"
                        scheduleDismiss(3500L, force = true)
                    }

                    override fun onPartialResults(partialResults: Bundle?) {
                        if (isExecutingGoal || interactiveCard?.visibility == View.VISIBLE) return
                        val matches = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                        val partial = matches?.firstOrNull()?.trim()
                        if (!partial.isNullOrBlank()) {
                            tvThinkingStatus?.text = "\"$partial\""
                        }
                    }

                    override fun onResults(results: Bundle?) {
                        cancelPendingDismiss()
                        if (isExecutingGoal || interactiveCard?.visibility == View.VISIBLE) return
                        val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                        val spoken = matches?.firstOrNull()?.trim()
                        if (!spoken.isNullOrBlank()) {
                            onSpeechGoalReceived(context, spoken)
                        } else {
                            scheduleDismiss(1500L, force = true)
                        }
                    }

                    override fun onEvent(eventType: Int, params: Bundle?) {}
                })
            }

            val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
                putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
                putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 3)
            }
            speechRecognizer?.startListening(intent)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start speech recognizer in 3D cyber orb: ${e.message}")
            scheduleDismiss(1500L, force = true)
        }
    }

    private fun onSpeechGoalReceived(context: Context, rawSpoken: String) {
        val cleanGoal = rawSpoken
            .replace(Regex("""(?i)^(?:hey|hi|ok|okay)?\s*zipper(?:,)?\s*"""), "")
            .trim()
            .ifBlank { rawSpoken.trim() }

        activeGoal = cleanGoal
        tvThinkingStatus?.text = "🚀 $cleanGoal"
        vibrateHeartbeat(context)

        // Smoothly transition directly into autonomous execution without vanishing!
        transitionToAutonomousExecution(context, cleanGoal)

        mainHandler.postDelayed({
            Log.i(TAG, "Dispatching goal to Karvish Agent: '$cleanGoal'")
            val serviceIntent = Intent(context.applicationContext, AgentRuntimeService::class.java).apply {
                putExtra(AgentRuntimeService.EXTRA_GOAL, cleanGoal.replaceFirstChar { it.uppercase() })
                putExtra(AgentRuntimeService.EXTRA_RECOVERY, true)
                putExtra(AgentRuntimeService.EXTRA_USE_REMOTE_RANKER, false)
            }
            ContextCompat.startForegroundService(context.applicationContext, serviceIntent)
        }, 500L)
    }

    /**
     * Submerges the 3D cyber orb and capsule smoothly down off the bottom bezel.
     */
    fun dismissWithSubmerge(force: Boolean = false, onEnd: (() -> Unit)? = null) {
        mainHandler.post {
            cancelPendingDismiss()

            if (!force && (isExecutingGoal || interactiveCard?.visibility == View.VISIBLE)) {
                Log.w(TAG, "dismissWithSubmerge BLOCKED: isExecutingGoal=$isExecutingGoal, cardVisible=${interactiveCard?.visibility == View.VISIBLE}")
                return@post
            }

            try {
                speechRecognizer?.stopListening()
                speechRecognizer?.destroy()
                speechRecognizer = null
            } catch (_: Exception) {}

            try {
                activeCardVoiceRecognizer?.stopListening()
                activeCardVoiceRecognizer?.destroy()
                activeCardVoiceRecognizer = null
            } catch (_: Exception) {}

            isListening = false
            WakeWordService.notifyAssistantFinished()

            val root = containerView ?: run {
                hide()
                onEnd?.invoke()
                return@post
            }
            val wm = windowManager ?: run {
                hide()
                onEnd?.invoke()
                return@post
            }
            val params = currentParams ?: run {
                hide()
                onEnd?.invoke()
                return@post
            }

            val screenHeight = root.resources.displayMetrics.heightPixels

            submergeAnimator?.cancel()
            submergeAnimator = ValueAnimator.ofInt(params.y, screenHeight + 50).apply {
                duration = 260L
                interpolator = AccelerateInterpolator(2.0f)
                addUpdateListener {
                    params.y = it.animatedValue as Int
                    try {
                        wm.updateViewLayout(root, params)
                    } catch (_: Exception) {}
                }
                addListener(object : AnimatorListenerAdapter() {
                    override fun onAnimationEnd(animation: Animator) {
                        hide()
                        onEnd?.invoke()
                    }
                })
                start()
            }
        }
    }

    fun dismiss() = dismissWithSubmerge(force = true)

    fun hide() {
        mainHandler.post {
            submergeAnimator?.cancel()
            cancelPendingDismiss()
            try {
                speechRecognizer?.destroy()
                speechRecognizer = null
            } catch (_: Exception) {}
            try {
                activeCardVoiceRecognizer?.destroy()
                activeCardVoiceRecognizer = null
            } catch (_: Exception) {}
            isListening = false
            isExecutingGoal = false
            isExpanded = false

            core3dView?.pauseRendering()

            if (!isShowing || containerView == null) return@post
            try {
                windowManager?.removeView(containerView)
                Log.i(TAG, "Karvish 3D cyber orb cleanly submerged and removed")
            } catch (e: Exception) {
                Log.w(TAG, "Error removing 3D cyber orb: ${e.message}")
            } finally {
                containerView = null
                currentParams = null
                core3dView = null
                leftWaveWing = null
                rightWaveWing = null
                thinkingCapsule = null
                inspectorCard = null
                interactiveCard = null
                ivAppIcon = null
                ivInspectorAppIcon = null
                tvInspectorHeaderBadge = null
                tvBrainIcon = null
                tvThinkingStatus = null
                tvStatusSubtext = null
                tvExpandHint = null
                tvCardGoal = null
                tvCardAction = null
                tvCardReasoning = null
                btnPauseAction = null
                btnStopAction = null
                isShowing = false
            }
        }
    }

    private fun vibrateHeartbeat(context: Context) {
        try {
            val vibrator = context.getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator ?: return
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                vibrator.vibrate(
                    VibrationEffect.createWaveform(
                        longArrayOf(0, 30, 60, 45),
                        intArrayOf(0, 160, 0, 220),
                        -1
                    )
                )
            } else {
                @Suppress("DEPRECATION")
                vibrator.vibrate(50)
            }
        } catch (_: Exception) {}
    }

    // =========================================================================
    // IN-RAM MAGICAL CHIME PLAYER (Instant C5 -> G5 Crystal Harmonic Chord)
    // =========================================================================
    object MagicalChimePlayer {
        private const val SAMPLE_RATE = 44100

        fun play() {
            Thread({
                try {
                    val durationMs = 210
                    val numSamples = (SAMPLE_RATE * durationMs / 1000)
                    val buffer = ShortArray(numSamples)

                    for (i in 0 until numSamples) {
                        val t = i.toDouble() / SAMPLE_RATE
                        // Celestial envelope: 8ms fast linear attack + smooth exponential decay
                        val attack = (i.toDouble() / (SAMPLE_RATE * 0.008)).coerceAtMost(1.0)
                        val decay = kotlin.math.exp(-t * 15.0)
                        val envelope = attack * decay

                        // Dual harmonic shimmer chord (C5 523.25Hz, G5 783.99Hz, C6 1046.5Hz)
                        val s1 = sin(2.0 * Math.PI * 523.25 * t)
                        val s2 = sin(2.0 * Math.PI * 783.99 * t) * 0.72
                        val s3 = sin(2.0 * Math.PI * 1046.50 * t) * 0.35

                        val sample = ((s1 + s2 + s3) / 2.07 * envelope * 24000.0).toInt().coerceIn(-32767, 32767)
                        buffer[i] = sample.toShort()
                    }

                    val audioTrack = AudioTrack(
                        AudioAttributes.Builder()
                            .setUsage(AudioAttributes.USAGE_ASSISTANCE_SONIFICATION)
                            .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                            .build(),
                        AudioFormat.Builder()
                            .setSampleRate(SAMPLE_RATE)
                            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                            .build(),
                        buffer.size * 2,
                        AudioTrack.MODE_STATIC,
                        AudioManager.AUDIO_SESSION_ID_GENERATE
                    )

                    audioTrack.write(buffer, 0, buffer.size)
                    audioTrack.play()

                    Thread.sleep(durationMs.toLong() + 30L)
                    audioTrack.stop()
                    audioTrack.release()
                } catch (e: Exception) {
                    Log.w(TAG, "Magical chime play error: ${e.message}")
                }
            }, "Zipper-MagicalChime").apply {
                isDaemon = true
                start()
            }
        }
    }

    // =========================================================================
    // SYMMETRICAL SOUNDWAVE WING VIEW (Left & Right Flanking Bars)
    // =========================================================================
    private class SoundwaveWingView(context: Context, private val isLeft: Boolean) : View(context) {

        private val density = context.resources.displayMetrics.density
        private val barWidth = 3.2f * density
        private val barRadius = 1.6f * density
        private val barGap = 3.5f * density

        private val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
            color = Color.parseColor("#00F0FF")
        }

        private var amplitude = 0f
        private var phase = 0f
        private var isAnimatingActive = false
        private var animator: ValueAnimator? = null

        init {
            animator = ValueAnimator.ofFloat(0f, 6.28f).apply {
                duration = 1400L
                repeatCount = ValueAnimator.INFINITE
                addUpdateListener {
                    phase = it.animatedValue as Float
                    invalidate()
                }
                start()
            }
        }

        fun startActiveAnimation() {
            isAnimatingActive = true
            invalidate()
        }

        fun stopActiveAnimation() {
            isAnimatingActive = false
            amplitude = 0f
            invalidate()
        }

        fun setAmplitude(amp: Float) {
            amplitude = amp.coerceIn(0f, 1f)
            invalidate()
        }

        override fun onDraw(canvas: Canvas) {
            super.onDraw(canvas)
            val centerY = height / 2f
            val maxHeight = height * 0.90f
            val minHeight = 4f * density

            val totalWidth = 4 * barWidth + 3 * barGap
            val startX = (width - totalWidth) / 2f

            for (i in 0 until 4) {
                val barIndex = if (isLeft) i else (3 - i)
                val rampFactor = 0.40f + 0.15f * barIndex // Ramps height towards the 3D orb

                val waveOffset = (phase + i * 1.15f)
                val baseNorm = 0.22f + 0.18f * abs(sin(waveOffset))
                val currentNorm = if (isAnimatingActive) {
                    (baseNorm + amplitude * 0.85f * rampFactor).coerceIn(0.12f, 1.0f)
                } else {
                    baseNorm * rampFactor
                }

                val barH = minHeight + (maxHeight - minHeight) * currentNorm
                val top = centerY - barH / 2f
                val bottom = centerY + barH / 2f

                val xPos = startX + i * (barWidth + barGap)
                canvas.drawRoundRect(
                    xPos,
                    top,
                    xPos + barWidth,
                    bottom,
                    barRadius,
                    barRadius,
                    paint
                )
            }
        }

        override fun onDetachedFromWindow() {
            super.onDetachedFromWindow()
            animator?.cancel()
        }
    }
}
