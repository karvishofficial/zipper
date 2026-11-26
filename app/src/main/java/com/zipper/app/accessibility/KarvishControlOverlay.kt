package com.zipper.app.accessibility

import android.animation.ValueAnimator
import android.content.Context
import android.content.Intent
import android.graphics.*
import android.graphics.drawable.ClipDrawable
import android.graphics.drawable.Drawable
import android.graphics.drawable.GradientDrawable
import android.graphics.drawable.LayerDrawable
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.util.LruCache
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import android.view.animation.AccelerateDecelerateInterpolator
import android.view.animation.DecelerateInterpolator
import android.view.animation.LinearInterpolator
import android.widget.*
import com.zipper.app.core.agent.AgentAction
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale
import kotlin.math.abs

/**
 * KARVISH CONTROL OVERLAY
 * Universal Floating Master Control HUD and Target Highlighter for autonomous on-device actions.
 *
 * Implements the futuristic cyber-neon interface:
 * 1. Full-screen transparent canvas drawing glowing neon bounding boxes & target badges (1, 2, 3..).
 * 2. Glassmorphic floating HUD card showing live Turn X/N, dynamic app icon, radial progress ring,
 *    4-phase pipeline stepper (Understand -> Execute -> Verify -> Complete), 3-column telemetry grid
 *    (Action / Target / Confidence), thinking mode trace accordion, and hard stop / pause / more controls.
 * 3. Emerald green completion card upon goal completion with verified milestone checklist.
 *
 * 100% universal across any Android app (Instagram, WhatsApp, Zepto, YouTube, Settings, etc.)
 * Does not modify or interfere with the main Zipper app UI.
 */
object KarvishControlOverlay {
    private const val TAG = "KarvishControlOverlay"

    private var windowManager: WindowManager? = null
    private var highlightView: HighlightCanvasView? = null
    private var hudView: View? = null
    private var currentHudParams: WindowManager.LayoutParams? = null

    var isShowing = false
        private set
    var isMinimized = false
        private set
    private var isPaused = false
    private var onPauseClickListener: ((Boolean) -> Unit)? = null
    private var onCancelClickListener: (() -> Unit)? = null
    private var activeGoal: String? = null

    private val mainHandler = Handler(Looper.getMainLooper())
    private val iconCache = LruCache<String, Drawable>(64)
    private val overlayIoScope = CoroutineScope(Dispatchers.IO)

    fun initialize(service: Context) {
        mainHandler.post {
            windowManager = service.getSystemService(Context.WINDOW_SERVICE) as? WindowManager
            val goal = activeGoal
            if (goal != null && !isShowing) {
                Log.i(TAG, "Rebinding Karvish Control Overlay after accessibility service connection")
                show(service, goal, onPauseClickListener, onCancelClickListener)
            }
        }
    }

    fun minimize() {
        mainHandler.post {
            if (!isShowing || isMinimized) return@post
            isMinimized = true
            hudView?.animate()
                ?.alpha(0f)
                ?.scaleX(0.85f)
                ?.scaleY(0.85f)
                ?.setDuration(220L)
                ?.withEndAction {
                    hudView?.visibility = View.GONE
                }
                ?.start()
        }
    }

    fun restore() {
        mainHandler.post {
            if (!isShowing || !isMinimized) return@post
            isMinimized = false
            hudView?.let { hud ->
                hud.visibility = View.VISIBLE
                hud.alpha = 0f
                hud.scaleX = 0.85f
                hud.scaleY = 0.85f
                val collapsedPill = hud.findViewWithTag<View>("collapsed_pill_view")
                collapsedPill?.visibility = View.VISIBLE
                val card = hud.findViewWithTag<View>("hud_card_container")
                card?.visibility = View.GONE
                val density = hud.context.resources.displayMetrics.density
                val screenWidth = hud.context.resources.displayMetrics.widthPixels
                val pillWidth = (screenWidth * 0.90f).toInt().coerceAtMost((420 * density).toInt())
                currentHudParams?.width = pillWidth
                try {
                    windowManager?.updateViewLayout(hud, currentHudParams)
                } catch (_: Exception) {}
                hud.animate()
                    ?.alpha(1f)
                    ?.scaleX(1f)
                    ?.scaleY(1f)
                    ?.setDuration(240L)
                    ?.start()
            }
        }
    }

    /**
     * Start/Show the Karvish Control Overlay for a goal.
     */
    fun show(
        context: Context,
        goal: String,
        onPause: ((Boolean) -> Unit)? = null,
        onCancel: (() -> Unit)? = null
    ) {
        activeGoal = goal
        onPauseClickListener = onPause
        onCancelClickListener = onCancel
        isPaused = false
        isMinimized = false

        // Connect controls to persistent 3D Cyber Orb overlay
        KarvishBubbleOverlay.setAgentControls(onPause, onCancel)

        mainHandler.post {
            try {
                val a11y = KarvishAccessibilityService.instance
                val useA11y = a11y != null
                val targetContext = a11y ?: context
                val wm = (targetContext.getSystemService(Context.WINDOW_SERVICE) as? WindowManager)
                windowManager = wm
                if (wm == null) {
                    Log.e(TAG, "WindowManager is null, cannot show Karvish Control Overlay")
                    return@post
                }

                val overlayType = if (useA11y) {
                    WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY
                } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
                } else {
                    @Suppress("DEPRECATION")
                    WindowManager.LayoutParams.TYPE_PHONE
                }

                if (!isShowing) {
                    // 1. Add Full-screen transparent highlight view
                    val highlight = HighlightCanvasView(targetContext)
                    val highlightParams = WindowManager.LayoutParams(
                        WindowManager.LayoutParams.MATCH_PARENT,
                        WindowManager.LayoutParams.MATCH_PARENT,
                        overlayType,
                        WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE or
                                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                                WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN or
                                WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
                        PixelFormat.TRANSLUCENT
                    ).apply {
                        gravity = Gravity.TOP or Gravity.START
                        x = 0
                        y = 0
                    }
                    wm.addView(highlight, highlightParams)
                    highlightView = highlight

                    // 2. Add Floating Draggable Masterclass HUD Card (Suppressed if 3D Cyber Orb is active)
                    if (!KarvishBubbleOverlay.isShowing) {
                        val hud = createHudCard(targetContext, goal)
                        val density = targetContext.resources.displayMetrics.density
                        val screenWidth = targetContext.resources.displayMetrics.widthPixels
                        val pillWidth = (screenWidth * 0.90f).toInt().coerceAtMost((420 * density).toInt())
                        val hudParams = WindowManager.LayoutParams(
                            pillWidth,
                            WindowManager.LayoutParams.WRAP_CONTENT,
                            overlayType,
                            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                                    WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
                            PixelFormat.TRANSLUCENT
                        ).apply {
                            gravity = Gravity.BOTTOM or Gravity.CENTER_HORIZONTAL
                            y = (24 * density).toInt() // Positioned comfortably in lower screen above nav bar
                        }
                        currentHudParams = hudParams
                        wm.addView(hud, hudParams)
                        hudView = hud
                    }

                    isShowing = true
                    isMinimized = false
                    Log.i(TAG, "Karvish Master Control Overlay attached to WindowManager with overlayType=$overlayType")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to attach Karvish Control Overlay: ${e.message}")
            }
        }
    }

    /**
     * Update Turn info, target bounding box, and live model confidence.
     */
    fun updateTurn(
        context: Context,
        turn: Int,
        maxTurns: Int,
        action: AgentAction?,
        targetLabel: String,
        confidence: Float,
        bounds: Rect?,
        statusText: String? = null,
        isVisionActive: Boolean = false,
        visionIconClass: String? = null
    ) {
        mainHandler.post {
            try {
                if (!isShowing) {
                    show(context, statusText ?: "Executing task...")
                }

                // 1. Update Highlight View
                highlightView?.setTarget(bounds, turn)

                // 2. Update Master HUD
                hudView?.let { hud ->
                    val turnText = hud.findViewWithTag<TextView>("turn_text")
                    turnText?.text = "$turn / $maxTurns"

                    val headerTitle = hud.findViewWithTag<TextView>("header_title")
                    if (isVisionActive) {
                        headerTitle?.text = "KARVISH V61 • 👁️ VISION"
                        headerTitle?.setTextColor(Color.parseColor("#00E5FF"))
                    } else {
                        headerTitle?.text = "KARVISH"
                        headerTitle?.setTextColor(Color.WHITE)
                    }

                    val statusTv = hud.findViewWithTag<TextView>("status_text")
                    val computedStatus = statusText ?: when (action) {
                        is AgentAction.OpenApp -> "Opening ${action.appName}..."
                        is AgentAction.Tap -> {
                            if (isVisionActive) {
                                "👁️ Grounded: ${visionIconClass ?: targetLabel}"
                            } else {
                                "Targeting: $targetLabel"
                            }
                        }
                        is AgentAction.TypeText -> "Typing: \"${action.text}\""
                        is AgentAction.Submit -> "Submitting query..."
                        is AgentAction.Scroll -> "Scrolling screen..."
                        else -> "Processing next action..."
                    }
                    statusTv?.text = computedStatus

                    val statusSubtextTv = hud.findViewWithTag<TextView>("status_subtext")
                    val computedSubtext = when (action) {
                        is AgentAction.OpenApp -> "Launching target app from current screen"
                        is AgentAction.Tap -> if (isVisionActive) "Interacting with verified visual element" else "Interacting with verified screen element"
                        is AgentAction.TypeText -> "Entering input argument into active text field"
                        is AgentAction.Submit -> "Committing search query via IME/keyboard"
                        is AgentAction.Scroll -> "Navigating viewport to reveal target content"
                        else -> "Executing autonomous policy action"
                    }
                    statusSubtextTv?.text = computedSubtext

                    // Forward turn and reasoning to persistent 3D Cyber Orb overlay
                    val actionLabel = when (action) {
                        is AgentAction.OpenApp -> "TAP (Open App)"
                        is AgentAction.Tap -> if (isVisionActive) "TAP (Vision)" else "TAP"
                        is AgentAction.TypeText -> "TYPE_TEXT"
                        is AgentAction.Submit -> "SUBMIT"
                        is AgentAction.Scroll -> "SCROLL"
                        else -> action?.javaClass?.simpleName ?: "ACTION"
                    }
                    KarvishBubbleOverlay.updateAgentTurn(
                        turn = turn,
                        maxTurns = maxTurns,
                        actionName = actionLabel,
                        targetLabel = targetLabel,
                        statusText = computedStatus,
                        subtext = computedSubtext
                    )

                    // Update Bottom Thinking Pill Views
                    val pillThinkingTv = hud.findViewWithTag<TextView>("pill_thinking_text")
                    pillThinkingTv?.text = computedStatus

                    val pillSubtextTv = hud.findViewWithTag<TextView>("pill_subtext")
                    pillSubtextTv?.text = computedSubtext

                    val pillTurnTv = hud.findViewWithTag<TextView>("pill_turn_text")
                    pillTurnTv?.text = "$turn / $maxTurns"

                    // Resolve Dynamic App Icon
                    val appIconView = hud.findViewWithTag<ImageView>("app_icon_view")
                    val targetAppName = when (action) {
                        is AgentAction.OpenApp -> action.appName
                        else -> {
                            val textToCheck = "$statusText $targetLabel $computedStatus"
                            when {
                                textToCheck.contains("Instagram", ignoreCase = true) -> "Instagram"
                                textToCheck.contains("WhatsApp", ignoreCase = true) -> "WhatsApp"
                                textToCheck.contains("YouTube", ignoreCase = true) -> "YouTube"
                                textToCheck.contains("Zepto", ignoreCase = true) -> "Zepto"
                                textToCheck.contains("Calculator", ignoreCase = true) -> "Calculator"
                                textToCheck.contains("Settings", ignoreCase = true) -> "Settings"
                                textToCheck.contains("LinkedIn", ignoreCase = true) -> "LinkedIn"
                                textToCheck.contains("Play Store", ignoreCase = true) -> "Play Store"
                                else -> null
                            }
                        }
                    }
                    val resolvedIcon = resolveAppIcon(context, targetAppName, targetView = appIconView)
                    if (resolvedIcon != null) {
                        appIconView?.setImageDrawable(resolvedIcon)
                    }

                    // Progress calculations
                    val progressPercent = ((turn.toFloat() / maxTurns.coerceAtLeast(1).toFloat()) * 100)
                        .toInt().coerceIn(0, 100)

                    val radialProgress = hud.findViewWithTag<RadialProgressView>("radial_progress_view")
                    radialProgress?.setProgress(progressPercent, animated = true)

                    val progressBar = hud.findViewWithTag<ProgressBar>("task_progress_bar")
                    progressBar?.progress = progressPercent

                    // Pipeline Stepper Phase
                    val stepper = hud.findViewWithTag<PipelineStepperView>("pipeline_stepper_view")
                    val phase = when {
                        action == null && turn == 1 -> 1 // Understand
                        action != null -> 2              // Execute
                        else -> 3                        // Verify
                    }
                    stepper?.setPhase(phase)

                    // Action Name
                    val actionTv = hud.findViewWithTag<TextView>("action_val")
                    val actionName = when (action) {
                        is AgentAction.OpenApp -> "TAP (Open App)"
                        is AgentAction.Tap -> if (isVisionActive) "TAP (Vision)" else "TAP"
                        is AgentAction.TypeText -> "TYPE_TEXT"
                        is AgentAction.Submit -> "SUBMIT"
                        is AgentAction.Scroll -> "SCROLL"
                        else -> action?.javaClass?.simpleName ?: "TAP"
                    }
                    actionTv?.text = actionName

                    // Target Name
                    val targetTv = hud.findViewWithTag<TextView>("target_val")
                    targetTv?.text = if (isVisionActive && visionIconClass != null) {
                        "$targetLabel ($visionIconClass)"
                    } else {
                        targetLabel.ifBlank { "Screen element" }
                    }

                    // Confidence Percentage & Bar
                    val confPercent = (confidence.coerceIn(0.01f, 1.0f) * 100).toInt()
                    val confTv = hud.findViewWithTag<TextView>("conf_val")
                    confTv?.text = "$confPercent%"

                    val confBar = hud.findViewWithTag<ProgressBar>("conf_bar")
                    confBar?.let { bar ->
                        val anim = ValueAnimator.ofInt(bar.progress, confPercent)
                        anim.duration = 400
                        anim.interpolator = AccelerateDecelerateInterpolator()
                        anim.addUpdateListener { bar.progress = it.animatedValue as Int }
                        anim.start()
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error updating turn overlay: ${e.message}")
            }
        }
    }

    fun updateTurnProgress(
        context: Context,
        turn: Int,
        maxTurns: Int = 6,
        actionName: String,
        statusText: String
    ) {
        updateTurn(
            context = context,
            turn = turn,
            maxTurns = maxTurns,
            action = null,
            targetLabel = actionName,
            confidence = 0.99f,
            bounds = null,
            statusText = statusText
        )
    }

    /**
     * Show emerald green Task Completed View with verified milestones.
     */
    fun showCompleted(
        milestones: List<String>,
        onDone: () -> Unit
    ) {
        mainHandler.post {
            try {
                KarvishBubbleOverlay.onGoalCompleted(activeGoal ?: "Task", "All steps verified ✓")
                highlightView?.clearTarget()
                hudView?.let { hud ->
                    val collapsedPill = hud.findViewWithTag<View>("collapsed_pill_view")
                    collapsedPill?.visibility = View.GONE
                    val container = hud.findViewWithTag<LinearLayout>("hud_card_container") ?: return@let
                    container.visibility = View.VISIBLE
                    container.removeAllViews()

                    val ctx = hud.context
                    val density = ctx.resources.displayMetrics.density
                    val screenWidth = ctx.resources.displayMetrics.widthPixels
                    val cardWidth = (screenWidth * 0.94f).toInt().coerceAtMost((480 * density).toInt())
                    currentHudParams?.width = cardWidth
                    try {
                        windowManager?.updateViewLayout(hud, currentHudParams)
                    } catch (_: Exception) {}

                    // Green emerald gradient card background
                    val greenBg = GradientDrawable(
                        GradientDrawable.Orientation.TOP_BOTTOM,
                        intArrayOf(Color.parseColor("#064E3B"), Color.parseColor("#022C22"))
                    ).apply {
                        cornerRadius = 24 * density
                        setStroke((1.5f * density).toInt(), Color.parseColor("#10B981"))
                    }
                    container.background = greenBg
                    container.setPadding(
                        (20 * density).toInt(),
                        (18 * density).toInt(),
                        (20 * density).toInt(),
                        (18 * density).toInt()
                    )

                    // Header row: Checkmark + "Task completed!"
                    val headerRow = LinearLayout(ctx).apply {
                        orientation = LinearLayout.HORIZONTAL
                        gravity = Gravity.CENTER_VERTICAL
                    }
                    val checkCircle = TextView(ctx).apply {
                        text = "✓"
                        textSize = 18f
                        setTextColor(Color.WHITE)
                        gravity = Gravity.CENTER
                        val circleBg = GradientDrawable().apply {
                            shape = GradientDrawable.OVAL
                            setColor(Color.parseColor("#10B981"))
                        }
                        background = circleBg
                        layoutParams = LinearLayout.LayoutParams((28 * density).toInt(), (28 * density).toInt()).apply {
                            rightMargin = (12 * density).toInt()
                        }
                    }
                    val titleTv = TextView(ctx).apply {
                        text = "Task completed!"
                        textSize = 18f
                        setTypeface(null, Typeface.BOLD)
                        setTextColor(Color.WHITE)
                    }
                    headerRow.addView(checkCircle)
                    headerRow.addView(titleTv)
                    container.addView(headerRow)

                    // Milestone Checklist items
                    val checklistLayout = LinearLayout(ctx).apply {
                        orientation = LinearLayout.VERTICAL
                        setPadding(0, (12 * density).toInt(), 0, (16 * density).toInt())
                    }
                    val displayMilestones = if (milestones.isNotEmpty()) milestones else listOf(
                        "Opened target app",
                        "Executed autonomous actions",
                        "Verified state transitions",
                        "Completed goal requirements"
                    )
                    for (m in displayMilestones) {
                        val itemRow = LinearLayout(ctx).apply {
                            orientation = LinearLayout.HORIZONTAL
                            gravity = Gravity.CENTER_VERTICAL
                            setPadding(0, (4 * density).toInt(), 0, (4 * density).toInt())
                        }
                        val itemCheck = TextView(ctx).apply {
                            text = "✓"
                            textSize = 14f
                            setTextColor(Color.parseColor("#34D399"))
                            layoutParams = LinearLayout.LayoutParams(
                                LinearLayout.LayoutParams.WRAP_CONTENT,
                                LinearLayout.LayoutParams.WRAP_CONTENT
                            ).apply { rightMargin = (8 * density).toInt() }
                        }
                        val itemText = TextView(ctx).apply {
                            text = m
                            textSize = 14f
                            setTextColor(Color.parseColor("#E2E8F0"))
                        }
                        itemRow.addView(itemCheck)
                        itemRow.addView(itemText)
                        checklistLayout.addView(itemRow)
                    }
                    container.addView(checklistLayout)

                    // Done Button
                    val doneBtn = Button(ctx).apply {
                        text = "✓ Done"
                        textSize = 15f
                        setTypeface(null, Typeface.BOLD)
                        setTextColor(Color.WHITE)
                        val btnBg = GradientDrawable().apply {
                            cornerRadius = 16 * density
                            setColor(Color.parseColor("#059669"))
                        }
                        background = btnBg
                        setOnClickListener {
                            hide()
                            onDone()
                        }
                    }
                    container.addView(doneBtn)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error rendering completion card: ${e.message}")
            }
        }
    }

    /**
     * Module 3 Escalation: Terminal Ceiling BLOCKED Banner with Human-in-the-Loop Handshake.
     * Displays an Amber banner with [Resume] and [Abort] options.
     */
    fun showBlockedBanner(
        message: String,
        onResume: () -> Unit,
        onAbort: () -> Unit
    ) {
        mainHandler.post {
            try {
                highlightView?.clearTarget()
                hudView?.let { hud ->
                    val collapsedPill = hud.findViewWithTag<View>("collapsed_pill_view")
                    collapsedPill?.visibility = View.GONE
                    val container = hud.findViewWithTag<LinearLayout>("hud_card_container") ?: return@let
                    container.visibility = View.VISIBLE
                    container.removeAllViews()

                    val ctx = hud.context
                    val density = ctx.resources.displayMetrics.density
                    val screenWidth = ctx.resources.displayMetrics.widthPixels
                    val cardWidth = (screenWidth * 0.94f).toInt().coerceAtMost((480 * density).toInt())
                    currentHudParams?.width = cardWidth
                    try {
                        windowManager?.updateViewLayout(hud, currentHudParams)
                    } catch (_: Exception) {}

                    // Amber warning card background (#78350F -> #451A03)
                    val amberBg = GradientDrawable(
                        GradientDrawable.Orientation.TOP_BOTTOM,
                        intArrayOf(Color.parseColor("#78350F"), Color.parseColor("#451A03"))
                    ).apply {
                        cornerRadius = 24 * density
                        setStroke((1.5f * density).toInt(), Color.parseColor("#F59E0B"))
                    }
                    container.background = amberBg
                    container.setPadding(
                        (20 * density).toInt(),
                        (18 * density).toInt(),
                        (20 * density).toInt(),
                        (18 * density).toInt()
                    )

                    // Header row: Warning icon + "Action Blocked"
                    val headerRow = LinearLayout(ctx).apply {
                        orientation = LinearLayout.HORIZONTAL
                        gravity = Gravity.CENTER_VERTICAL
                    }
                    val warnCircle = TextView(ctx).apply {
                        text = "!"
                        textSize = 18f
                        setTextColor(Color.WHITE)
                        setTypeface(null, Typeface.BOLD)
                        gravity = Gravity.CENTER
                        val circleBg = GradientDrawable().apply {
                            shape = GradientDrawable.OVAL
                            setColor(Color.parseColor("#F59E0B"))
                        }
                        background = circleBg
                        layoutParams = LinearLayout.LayoutParams((28 * density).toInt(), (28 * density).toInt()).apply {
                            rightMargin = (12 * density).toInt()
                        }
                    }
                    val titleTv = TextView(ctx).apply {
                        text = "Action Blocked (Ceiling 3)"
                        textSize = 18f
                        setTypeface(null, Typeface.BOLD)
                        setTextColor(Color.WHITE)
                    }
                    headerRow.addView(warnCircle)
                    headerRow.addView(titleTv)
                    container.addView(headerRow)

                    // Message text
                    val msgTv = TextView(ctx).apply {
                        text = message
                        textSize = 14f
                        setTextColor(Color.parseColor("#FDE68A"))
                        setPadding(0, (10 * density).toInt(), 0, (14 * density).toInt())
                    }
                    container.addView(msgTv)

                    // Action buttons row: [Resume] and [Abort]
                    val btnRow = LinearLayout(ctx).apply {
                        orientation = LinearLayout.HORIZONTAL
                        gravity = Gravity.CENTER
                    }

                    val resumeBtn = Button(ctx).apply {
                        text = "Resume"
                        setTextColor(Color.parseColor("#064E3B"))
                        textSize = 14f
                        setTypeface(null, Typeface.BOLD)
                        background = GradientDrawable().apply {
                            cornerRadius = 14 * density
                            setColor(Color.parseColor("#10B981"))
                        }
                        layoutParams = LinearLayout.LayoutParams(0, (42 * density).toInt(), 1f).apply {
                            rightMargin = (8 * density).toInt()
                        }
                        setOnClickListener {
                            hide()
                            onResume()
                        }
                    }

                    val abortBtn = Button(ctx).apply {
                        text = "Abort"
                        setTextColor(Color.WHITE)
                        textSize = 14f
                        setTypeface(null, Typeface.BOLD)
                        background = GradientDrawable().apply {
                            cornerRadius = 14 * density
                            setColor(Color.parseColor("#EF4444"))
                        }
                        layoutParams = LinearLayout.LayoutParams(0, (42 * density).toInt(), 1f).apply {
                            leftMargin = (8 * density).toInt()
                        }
                        setOnClickListener {
                            hide()
                            onAbort()
                        }
                    }

                    btnRow.addView(resumeBtn)
                    btnRow.addView(abortBtn)
                    container.addView(btnRow)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error rendering blocked card: ${e.message}")
            }
        }
    }

    /**
     * Human-in-the-Loop Disambiguation Card:
     * When multiple matching contacts or choices are found on screen,
     * renders interactive neon badges [1], [2], [3] on screen and accepts
     * voice choices ("1", "2", "first one") or on-screen taps.
     */
    fun showDisambiguationCard(
        title: String,
        prompt: String,
        options: List<String>,
        boundsList: List<Rect?>? = null,
        onSelect: (Int) -> Unit,
        onCancel: () -> Unit
    ) {
        mainHandler.post {
            try {
                // 1. Draw glowing neon badges on screen for each option
                val candidates = options.mapIndexed { idx, optText ->
                    NumberedCandidate(
                        index = idx + 1,
                        label = optText,
                        rect = boundsList?.getOrNull(idx)
                    )
                }
                highlightView?.setNumberedCandidates(candidates)

                // 2. Speak the prompt naturally
                val spokenPrompt = if (options.size > 1) {
                    "Found ${options.size} options for $title. Say 1, 2, or 3."
                } else {
                    "Found option for $title. Say 1 to confirm."
                }
                com.zipper.app.core.voice.ParadeeVoiceEngine.speak(spokenPrompt)

                // 3. If KarvishBubbleOverlay is active, display the interactive selection card directly above the 3D Orb!
                if (KarvishBubbleOverlay.isShowing) {
                    KarvishBubbleOverlay.showDisambiguationCard(
                        title = title,
                        prompt = prompt,
                        options = options,
                        onSelect = { selectedIdx ->
                            highlightView?.clearNumberedCandidates()
                            onSelect(selectedIdx)
                        },
                        onCancel = {
                            highlightView?.clearNumberedCandidates()
                            onCancel()
                        }
                    )
                    return@post
                }

                hudView?.let { hud ->
                    val collapsedPill = hud.findViewWithTag<View>("collapsed_pill_view")
                    collapsedPill?.visibility = View.GONE
                    val container = hud.findViewWithTag<LinearLayout>("hud_card_container")
                    container?.visibility = View.VISIBLE
                    val mainPanel = hud.findViewWithTag<LinearLayout>("main_hud_panel")
                    val cardSlot = hud.findViewWithTag<LinearLayout>("custom_card_slot") ?: hud.findViewWithTag<LinearLayout>("hud_card_container") ?: return@let

                    mainPanel?.visibility = View.GONE
                    cardSlot.removeAllViews()
                    cardSlot.visibility = View.VISIBLE

                    val ctx = hud.context
                    val density = ctx.resources.displayMetrics.density
                    val screenWidth = ctx.resources.displayMetrics.widthPixels
                    val cardWidth = (screenWidth * 0.94f).toInt().coerceAtMost((480 * density).toInt())
                    currentHudParams?.width = cardWidth
                    try { windowManager?.updateViewLayout(hud, currentHudParams) } catch (_: Exception) {}

                    var disambiguationRecognizer: android.speech.SpeechRecognizer? = null

                    fun selectOption(selectedIdx: Int) {
                        try {
                            disambiguationRecognizer?.destroy()
                            disambiguationRecognizer = null
                        } catch (_: Exception) {}
                        highlightView?.clearNumberedCandidates()
                        cardSlot.removeAllViews()
                        cardSlot.visibility = View.GONE
                        mainPanel?.visibility = View.VISIBLE
                        // Return to collapsed pill state
                        val pillWidth = (screenWidth * 0.90f).toInt().coerceAtMost((420 * density).toInt())
                        collapsedPill?.visibility = View.VISIBLE
                        container?.visibility = View.GONE
                        currentHudParams?.width = pillWidth
                        try { windowManager?.updateViewLayout(hud, currentHudParams) } catch (_: Exception) {}
                        onSelect(selectedIdx)
                    }

                    fun cancelDisambiguation() {
                        try {
                            disambiguationRecognizer?.destroy()
                            disambiguationRecognizer = null
                        } catch (_: Exception) {}
                        highlightView?.clearNumberedCandidates()
                        cardSlot.removeAllViews()
                        cardSlot.visibility = View.GONE
                        mainPanel?.visibility = View.VISIBLE
                        // Return to collapsed pill state
                        val pillWidth = (screenWidth * 0.90f).toInt().coerceAtMost((420 * density).toInt())
                        collapsedPill?.visibility = View.VISIBLE
                        container?.visibility = View.GONE
                        currentHudParams?.width = pillWidth
                        try { windowManager?.updateViewLayout(hud, currentHudParams) } catch (_: Exception) {}
                        onCancel()
                    }

                    // Start Voice Choice Listener
                    if (android.speech.SpeechRecognizer.isRecognitionAvailable(ctx)) {
                        try {
                            disambiguationRecognizer = android.speech.SpeechRecognizer.createSpeechRecognizer(ctx).apply {
                                setRecognitionListener(object : android.speech.RecognitionListener {
                                    override fun onReadyForSpeech(params: Bundle?) {}
                                    override fun onBeginningOfSpeech() {}
                                    override fun onRmsChanged(rmsdB: Float) {}
                                    override fun onBufferReceived(buffer: ByteArray?) {}
                                    override fun onEndOfSpeech() {}
                                    override fun onError(error: Int) {
                                        // Silently keep dialog open for manual tap if voice times out
                                    }
                                    override fun onResults(results: Bundle?) {
                                        val matches = results?.getStringArrayList(android.speech.SpeechRecognizer.RESULTS_RECOGNITION)
                                        val spoken = matches?.firstOrNull()?.lowercase(Locale.ROOT).orEmpty()
                                        Log.i(TAG, "[Disambiguation Voice] Spoken: '$spoken'")

                                        when {
                                            spoken.contains("1") || spoken.contains("one") || spoken.contains("first") -> selectOption(0)
                                            spoken.contains("2") || spoken.contains("two") || spoken.contains("second") -> selectOption(1)
                                            spoken.contains("3") || spoken.contains("three") || spoken.contains("third") -> selectOption(2)
                                            spoken.contains("4") || spoken.contains("four") || spoken.contains("fourth") -> selectOption(3)
                                            spoken.contains("cancel") || spoken.contains("stop") || spoken.contains("abort") -> cancelDisambiguation()
                                            else -> {
                                                // Check for exact name match in options
                                                val matchIdx = options.indexOfFirst { opt ->
                                                    spoken.contains(opt.lowercase(Locale.ROOT)) || opt.lowercase(Locale.ROOT).contains(spoken)
                                                }
                                                if (matchIdx != -1) {
                                                    selectOption(matchIdx)
                                                }
                                            }
                                        }
                                    }
                                    override fun onPartialResults(partialResults: Bundle?) {}
                                    override fun onEvent(eventType: Int, params: Bundle?) {}
                                })
                            }
                            val vIntent = Intent(android.speech.RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                                putExtra(android.speech.RecognizerIntent.EXTRA_LANGUAGE_MODEL, android.speech.RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                                putExtra(android.speech.RecognizerIntent.EXTRA_MAX_RESULTS, 1)
                            }
                            disambiguationRecognizer?.startListening(vIntent)
                        } catch (e: Exception) {
                            Log.w(TAG, "Could not start disambiguation voice listener: ${e.message}")
                        }
                    }

                    // Header row: Contact Icon + Title
                    val headerRow = LinearLayout(ctx).apply {
                        orientation = LinearLayout.HORIZONTAL
                        gravity = Gravity.CENTER_VERTICAL
                    }
                    val iconCircle = TextView(ctx).apply {
                        text = "👤"
                        textSize = 18f
                        gravity = Gravity.CENTER
                        layoutParams = LinearLayout.LayoutParams((32 * density).toInt(), (32 * density).toInt()).apply {
                            rightMargin = (10 * density).toInt()
                        }
                    }
                    val titleTv = TextView(ctx).apply {
                        text = title
                        textSize = 17f
                        setTypeface(null, Typeface.BOLD)
                        setTextColor(Color.WHITE)
                    }
                    headerRow.addView(iconCircle)
                    headerRow.addView(titleTv)
                    cardSlot.addView(headerRow)

                    // Subtitle / Prompt
                    val promptTv = TextView(ctx).apply {
                        text = "$prompt (Say 1, 2, 3 or tap below)"
                        textSize = 13f
                        setTextColor(Color.parseColor("#38BDF8"))
                        setPadding(0, (6 * density).toInt(), 0, (12 * density).toInt())
                    }
                    cardSlot.addView(promptTv)

                    // Stacked options list
                    val optionsContainer = LinearLayout(ctx).apply {
                        orientation = LinearLayout.VERTICAL
                    }

                    options.forEachIndexed { idx, optText ->
                        val optBtn = Button(ctx).apply {
                            text = "[${idx + 1}]  $optText"
                            setTextColor(Color.WHITE)
                            textSize = 14f
                            setTypeface(null, Typeface.BOLD)
                            gravity = Gravity.START or Gravity.CENTER_VERTICAL
                            setPadding((16 * density).toInt(), (10 * density).toInt(), (16 * density).toInt(), (10 * density).toInt())
                            background = GradientDrawable().apply {
                                cornerRadius = 12 * density
                                setColor(Color.parseColor("#1E293B"))
                                setStroke((1.5f * density).toInt(), Color.parseColor("#00E5FF"))
                            }
                            layoutParams = LinearLayout.LayoutParams(
                                LinearLayout.LayoutParams.MATCH_PARENT,
                                LinearLayout.LayoutParams.WRAP_CONTENT
                            ).apply {
                                bottomMargin = (8 * density).toInt()
                            }
                            setOnClickListener {
                                selectOption(idx)
                            }
                        }
                        optionsContainer.addView(optBtn)
                    }
                    cardSlot.addView(optionsContainer)

                    // Cancel button
                    val cancelBtn = Button(ctx).apply {
                        text = "Cancel"
                        setTextColor(Color.parseColor("#94A3B8"))
                        textSize = 12f
                        background = null
                        layoutParams = LinearLayout.LayoutParams(
                            LinearLayout.LayoutParams.WRAP_CONTENT,
                            LinearLayout.LayoutParams.WRAP_CONTENT
                        ).apply {
                            gravity = Gravity.CENTER_HORIZONTAL
                            topMargin = (4 * density).toInt()
                        }
                        setOnClickListener {
                            cancelDisambiguation()
                        }
                    }
                    cardSlot.addView(cancelBtn)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error rendering disambiguation card: ${e.message}")
            }
        }
    }

    /**
     * Human-in-the-Loop Pre-Send Confirmation Card:
     * When the agent is about to send a message, prompts the user to explicitly confirm or abort.
     */
    fun showConfirmationCard(
        title: String,
        prompt: String,
        confirmText: String = "Send",
        cancelText: String = "Don't Send",
        onConfirm: () -> Unit,
        onCancel: () -> Unit
    ) {
        mainHandler.post {
            try {
                highlightView?.clearTarget()

                // If KarvishBubbleOverlay is active, display the confirmation card directly above the 3D Orb!
                if (KarvishBubbleOverlay.isShowing) {
                    KarvishBubbleOverlay.showConfirmationCard(
                        title = title,
                        prompt = prompt,
                        confirmText = confirmText,
                        cancelText = cancelText,
                        onConfirm = onConfirm,
                        onCancel = onCancel
                    )
                    return@post
                }

                hudView?.let { hud ->
                    val collapsedPill = hud.findViewWithTag<View>("collapsed_pill_view")
                    collapsedPill?.visibility = View.GONE
                    val container = hud.findViewWithTag<LinearLayout>("hud_card_container")
                    container?.visibility = View.VISIBLE
                    val mainPanel = hud.findViewWithTag<LinearLayout>("main_hud_panel")
                    val cardSlot = hud.findViewWithTag<LinearLayout>("custom_card_slot") ?: hud.findViewWithTag<LinearLayout>("hud_card_container") ?: return@let

                    mainPanel?.visibility = View.GONE
                    cardSlot.removeAllViews()
                    cardSlot.visibility = View.VISIBLE

                    val ctx = hud.context
                    val density = ctx.resources.displayMetrics.density
                    val screenWidth = ctx.resources.displayMetrics.widthPixels
                    val cardWidth = (screenWidth * 0.94f).toInt().coerceAtMost((480 * density).toInt())
                    currentHudParams?.width = cardWidth
                    try { windowManager?.updateViewLayout(hud, currentHudParams) } catch (_: Exception) {}

                    // Header row: Message Icon + Title
                    val headerRow = LinearLayout(ctx).apply {
                        orientation = LinearLayout.HORIZONTAL
                        gravity = Gravity.CENTER_VERTICAL
                    }
                    val iconCircle = TextView(ctx).apply {
                        text = "✉️"
                        textSize = 18f
                        gravity = Gravity.CENTER
                        layoutParams = LinearLayout.LayoutParams((32 * density).toInt(), (32 * density).toInt()).apply {
                            rightMargin = (10 * density).toInt()
                        }
                    }
                    val titleTv = TextView(ctx).apply {
                        text = title
                        textSize = 17f
                        setTypeface(null, Typeface.BOLD)
                        setTextColor(Color.WHITE)
                    }
                    headerRow.addView(iconCircle)
                    headerRow.addView(titleTv)
                    cardSlot.addView(headerRow)

                    // Subtitle / Prompt
                    val promptTv = TextView(ctx).apply {
                        text = prompt
                        textSize = 14f
                        setTextColor(Color.parseColor("#FDE68A"))
                        setPadding(0, (8 * density).toInt(), 0, (14 * density).toInt())
                    }
                    cardSlot.addView(promptTv)

                    // Action buttons row: [Send] and [Don't Send]
                    val btnRow = LinearLayout(ctx).apply {
                        orientation = LinearLayout.HORIZONTAL
                        gravity = Gravity.CENTER
                    }

                    val sendBtn = Button(ctx).apply {
                        text = confirmText
                        setTextColor(Color.parseColor("#064E3B"))
                        textSize = 14f
                        setTypeface(null, Typeface.BOLD)
                        background = GradientDrawable().apply {
                            cornerRadius = 14 * density
                            setColor(Color.parseColor("#10B981"))
                        }
                        layoutParams = LinearLayout.LayoutParams(0, (42 * density).toInt(), 1f).apply {
                            rightMargin = (8 * density).toInt()
                        }
                        setOnClickListener {
                            cardSlot.removeAllViews()
                            cardSlot.visibility = View.GONE
                            mainPanel?.visibility = View.VISIBLE
                            val pillWidth = (screenWidth * 0.90f).toInt().coerceAtMost((420 * density).toInt())
                            collapsedPill?.visibility = View.VISIBLE
                            container?.visibility = View.GONE
                            currentHudParams?.width = pillWidth
                            try { windowManager?.updateViewLayout(hud, currentHudParams) } catch (_: Exception) {}
                            onConfirm()
                        }
                    }

                    val cancelBtn = Button(ctx).apply {
                        text = cancelText
                        setTextColor(Color.WHITE)
                        textSize = 14f
                        setTypeface(null, Typeface.BOLD)
                        background = GradientDrawable().apply {
                            cornerRadius = 14 * density
                            setColor(Color.parseColor("#EF4444"))
                        }
                        layoutParams = LinearLayout.LayoutParams(0, (42 * density).toInt(), 1f).apply {
                            leftMargin = (8 * density).toInt()
                        }
                        setOnClickListener {
                            cardSlot.removeAllViews()
                            cardSlot.visibility = View.GONE
                            mainPanel?.visibility = View.VISIBLE
                            val pillWidth = (screenWidth * 0.90f).toInt().coerceAtMost((420 * density).toInt())
                            collapsedPill?.visibility = View.VISIBLE
                            container?.visibility = View.GONE
                            currentHudParams?.width = pillWidth
                            try { windowManager?.updateViewLayout(hud, currentHudParams) } catch (_: Exception) {}
                            onCancel()
                        }
                    }

                    btnRow.addView(sendBtn)
                    btnRow.addView(cancelBtn)
                    cardSlot.addView(btnRow)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error rendering confirmation card: ${e.message}")
            }
        }
    }

    /**
     * Autonomous Approval Gate Card (Grok Bot Safety Architecture):
     * When the agent is about to execute an irreversible action (e.g. Pay, Place Order, Delete),
     * prompts the user to explicitly authorize or abort the action.
     */
    fun showApprovalGateCard(
        title: String,
        prompt: String,
        targetDetail: String,
        onApprove: () -> Unit,
        onAbort: () -> Unit
    ) {
        mainHandler.post {
            try {
                highlightView?.clearTarget()
                hudView?.let { hud ->
                    val collapsedPill = hud.findViewWithTag<View>("collapsed_pill_view")
                    collapsedPill?.visibility = View.GONE
                    val container = hud.findViewWithTag<LinearLayout>("hud_card_container")
                    container?.visibility = View.VISIBLE
                    val mainPanel = hud.findViewWithTag<LinearLayout>("main_hud_panel")
                    val cardSlot = hud.findViewWithTag<LinearLayout>("custom_card_slot") ?: hud.findViewWithTag<LinearLayout>("hud_card_container") ?: return@let

                    mainPanel?.visibility = View.GONE
                    cardSlot.removeAllViews()
                    cardSlot.visibility = View.VISIBLE

                    val ctx = hud.context
                    val density = ctx.resources.displayMetrics.density
                    val screenWidth = ctx.resources.displayMetrics.widthPixels
                    val cardWidth = (screenWidth * 0.94f).toInt().coerceAtMost((480 * density).toInt())
                    currentHudParams?.width = cardWidth
                    try { windowManager?.updateViewLayout(hud, currentHudParams) } catch (_: Exception) {}

                    // Header row: Shield/Lock Icon + Title
                    val headerRow = LinearLayout(ctx).apply {
                        orientation = LinearLayout.HORIZONTAL
                        gravity = Gravity.CENTER_VERTICAL
                    }
                    val iconCircle = TextView(ctx).apply {
                        text = "🛡️"
                        textSize = 18f
                        gravity = Gravity.CENTER
                        layoutParams = LinearLayout.LayoutParams((32 * density).toInt(), (32 * density).toInt()).apply {
                            rightMargin = (10 * density).toInt()
                        }
                    }
                    val titleTv = TextView(ctx).apply {
                        text = title
                        textSize = 17f
                        setTypeface(null, Typeface.BOLD)
                        setTextColor(Color.parseColor("#FBBF24")) // Neon Amber
                    }
                    headerRow.addView(iconCircle)
                    headerRow.addView(titleTv)
                    cardSlot.addView(headerRow)

                    // Prompt text
                    val promptTv = TextView(ctx).apply {
                        text = prompt
                        textSize = 13f
                        setTextColor(Color.parseColor("#E2E8F0"))
                        setPadding(0, (6 * density).toInt(), 0, (8 * density).toInt())
                    }
                    cardSlot.addView(promptTv)

                    // Target detail box (high-contrast card)
                    val detailBox = TextView(ctx).apply {
                        text = targetDetail
                        textSize = 13f
                        setTypeface(null, Typeface.BOLD)
                        setTextColor(Color.parseColor("#38BDF8"))
                        setPadding((12 * density).toInt(), (8 * density).toInt(), (12 * density).toInt(), (8 * density).toInt())
                        background = GradientDrawable().apply {
                            cornerRadius = 10 * density
                            setColor(Color.parseColor("#0F172A"))
                            setStroke((1 * density).toInt(), Color.parseColor("#334155"))
                        }
                        layoutParams = LinearLayout.LayoutParams(
                            LinearLayout.LayoutParams.MATCH_PARENT,
                            LinearLayout.LayoutParams.WRAP_CONTENT
                        ).apply {
                            bottomMargin = (12 * density).toInt()
                        }
                    }
                    cardSlot.addView(detailBox)

                    // Action buttons row: [Approve & Proceed] and [Abort & Halt]
                    val btnRow = LinearLayout(ctx).apply {
                        orientation = LinearLayout.HORIZONTAL
                        gravity = Gravity.CENTER
                    }

                    val approveBtn = Button(ctx).apply {
                        text = "Approve & Proceed"
                        setTextColor(Color.WHITE)
                        textSize = 13f
                        setTypeface(null, Typeface.BOLD)
                        background = GradientDrawable().apply {
                            cornerRadius = 14 * density
                            setColor(Color.parseColor("#059669")) // Emerald
                        }
                        layoutParams = LinearLayout.LayoutParams(0, (42 * density).toInt(), 1f).apply {
                            rightMargin = (8 * density).toInt()
                        }
                        setOnClickListener {
                            cardSlot.removeAllViews()
                            cardSlot.visibility = View.GONE
                            mainPanel?.visibility = View.VISIBLE
                            val pillWidth = (screenWidth * 0.90f).toInt().coerceAtMost((420 * density).toInt())
                            collapsedPill?.visibility = View.VISIBLE
                            container?.visibility = View.GONE
                            currentHudParams?.width = pillWidth
                            try { windowManager?.updateViewLayout(hud, currentHudParams) } catch (_: Exception) {}
                            onApprove()
                        }
                    }

                    val abortBtn = Button(ctx).apply {
                        text = "Abort & Halt"
                        setTextColor(Color.WHITE)
                        textSize = 13f
                        setTypeface(null, Typeface.BOLD)
                        background = GradientDrawable().apply {
                            cornerRadius = 14 * density
                            setColor(Color.parseColor("#DC2626")) // Crimson
                        }
                        layoutParams = LinearLayout.LayoutParams(0, (42 * density).toInt(), 1f).apply {
                            leftMargin = (8 * density).toInt()
                        }
                        setOnClickListener {
                            cardSlot.removeAllViews()
                            cardSlot.visibility = View.GONE
                            mainPanel?.visibility = View.VISIBLE
                            val pillWidth = (screenWidth * 0.90f).toInt().coerceAtMost((420 * density).toInt())
                            collapsedPill?.visibility = View.VISIBLE
                            container?.visibility = View.GONE
                            currentHudParams?.width = pillWidth
                            try { windowManager?.updateViewLayout(hud, currentHudParams) } catch (_: Exception) {}
                            onAbort()
                        }
                    }

                    btnRow.addView(approveBtn)
                    btnRow.addView(abortBtn)
                    cardSlot.addView(btnRow)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error rendering approval gate card: ${e.message}")
            }
        }
    }

    /**
     * MuSpark Thinking Mode Telemetry Streamer:
     * Streams live neural reasoning trace, target evaluation, and state mutation deltas.
     */
    fun updateThinkingTrace(trace: String) {
        mainHandler.post {
            try {
                hudView?.let { hud ->
                    val traceTv = hud.findViewWithTag<TextView>("thinking_trace_text")
                    traceTv?.text = trace

                    val pillThinkingTv = hud.findViewWithTag<TextView>("pill_thinking_text")
                    pillThinkingTv?.text = trace
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error updating thinking trace: ${e.message}")
            }
        }
    }

    /**
     * Dismiss and remove all overlay views.
     */
    fun hide() {
        activeGoal = null
        isMinimized = false
        mainHandler.post {
            try {
                highlightView?.let { windowManager?.removeViewImmediate(it) }
                highlightView = null

                hudView?.let { windowManager?.removeViewImmediate(it) }
                hudView = null

                isShowing = false
                isPaused = false
                currentHudParams = null
                Log.i(TAG, "Karvish Master Control Overlay removed")
            } catch (e: Exception) {
                Log.e(TAG, "Error hiding overlay: ${e.message}")
            }
        }
    }

    // =========================================================================
    // DYNAMIC APP ICON RESOLVER
    // =========================================================================

    fun resolveAppIcon(context: Context, appNameOrPkg: String?, targetView: ImageView? = null): Drawable? {
        if (appNameOrPkg.isNullOrBlank()) return null
        val pm = context.packageManager
        val cleanName = appNameOrPkg.trim()
        val cacheKey = cleanName.lowercase(Locale.ROOT)

        // 1. In-memory L1 Cache (0.001 ms)
        iconCache.get(cacheKey)?.let { return it }

        // 2. Direct Package Lookup (0.1 ms)
        try {
            val icon = pm.getApplicationIcon(cleanName)
            iconCache.put(cacheKey, icon)
            return icon
        } catch (_: Exception) {}

        // 3. Fast Known App Package Mapping (0.01 ms)
        val lower = cacheKey
        val knownPkg = when {
            lower.contains("instagram") || lower == "insta" -> "com.instagram.android"
            lower.contains("whatsapp") -> "com.whatsapp"
            lower.contains("youtube") -> "com.google.android.youtube"
            lower.contains("zepto") -> "com.zeptonow.android"
            lower.contains("calculator") || lower == "calc" -> "com.sec.android.app.popupcalculator"
            lower.contains("settings") -> "com.android.settings"
            lower.contains("linkedin") -> "com.linkedin.android"
            lower.contains("play store") || lower.contains("google play") -> "com.android.vending"
            lower.contains("chrome") -> "com.android.chrome"
            lower.contains("maps") -> "com.google.android.apps.maps"
            lower.contains("photos") -> "com.google.android.apps.photos"
            lower.contains("clock") || lower.contains("alarm") || lower.contains("timer") -> "com.sec.android.app.clockpackage"
            lower.contains("camera") -> "com.sec.android.app.camera"
            lower.contains("swiggy") -> "in.swiggy.android"
            lower.contains("zomato") -> "com.application.zomato"
            lower.contains("blinkit") || lower.contains("grofers") -> "com.grofers.customerapp"
            lower.contains("amazon") -> "in.amazon.mShop.android.shopping"
            lower.contains("flipkart") -> "com.flipkart.android"
            lower.contains("uber") -> "com.ubercab"
            lower.contains("ola") -> "com.olacabs.customer"
            lower.contains("rapido") -> "com.rapido.passenger"
            lower.contains("gmail") -> "com.google.android.gm"
            lower.contains("messages") || lower.contains("sms") -> "com.google.android.apps.messaging"
            lower.contains("phone") || lower.contains("dialer") -> "com.samsung.android.dialer"
            lower.contains("telegram") -> "org.telegram.messenger"
            lower.contains("slack") -> "com.slack"
            lower.contains("discord") -> "com.discord"
            lower.contains("spotify") -> "com.spotify.music"
            lower.contains("keep") || lower.contains("notes") -> "com.google.android.keep"
            else -> null
        }
        if (knownPkg != null) {
            try {
                val icon = pm.getApplicationIcon(knownPkg)
                iconCache.put(cacheKey, icon)
                return icon
            } catch (_: Exception) {}
        }

        // 4. Asynchronous Non-Blocking Fallback (Never freezes the Main Thread Looper!)
        if (targetView != null) {
            overlayIoScope.launch {
                try {
                    val installedApps = pm.getInstalledApplications(0)
                    for (appInfo in installedApps) {
                        val label = pm.getApplicationLabel(appInfo).toString()
                        if (label.equals(cleanName, ignoreCase = true) ||
                            (cleanName.length >= 3 && label.contains(cleanName, ignoreCase = true))) {
                            val resolvedIcon = pm.getApplicationIcon(appInfo)
                            iconCache.put(cacheKey, resolvedIcon)
                            withContext(Dispatchers.Main) {
                                targetView.setImageDrawable(resolvedIcon)
                            }
                            break
                        }
                    }
                } catch (_: Exception) {}
            }
        }

        return null
    }

    // =========================================================================
    // CUSTOM MASTER HUD VIEWS
    // =========================================================================

    class KarvishAvatarBadgeView @JvmOverloads constructor(
        context: Context,
        attrs: android.util.AttributeSet? = null
    ) : View(context, attrs) {
        private val density = resources.displayMetrics.density
        private var angle = 0f

        private val bgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
            color = Color.parseColor("#0C192E")
        }

        private val glowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = 3.5f * density
            color = Color.parseColor("#4400E5FF")
        }

        private val ringPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = 1.8f * density
            color = Color.parseColor("#00E5FF")
        }

        private val anim = ValueAnimator.ofFloat(0f, 360f).apply {
            duration = 5000L
            repeatCount = ValueAnimator.INFINITE
            interpolator = LinearInterpolator()
            addUpdateListener {
                angle = it.animatedValue as Float
                invalidate()
            }
        }

        init {
            anim.start()
        }

        override fun onDraw(canvas: Canvas) {
            super.onDraw(canvas)
            val w = width.toFloat()
            val h = height.toFloat()
            val cx = w / 2f
            val cy = h / 2f
            val r = (Math.min(w, h) / 2f) - (3f * density)

            canvas.drawCircle(cx, cy, r, bgPaint)
            canvas.drawCircle(cx, cy, r, glowPaint)
            canvas.drawCircle(cx, cy, r, ringPaint)

            canvas.save()
            canvas.rotate(angle, cx, cy)
            val swirlPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                style = Paint.Style.FILL
                shader = RadialGradient(
                    cx, cy, r * 0.9f,
                    intArrayOf(
                        Color.parseColor("#38BDF8"),
                        Color.parseColor("#0284C7"),
                        Color.parseColor("#0A192F"),
                        Color.TRANSPARENT
                    ),
                    floatArrayOf(0f, 0.45f, 0.8f, 1f),
                    Shader.TileMode.CLAMP
                )
            }
            canvas.drawCircle(cx, cy, r * 0.85f, swirlPaint)
            canvas.restore()
        }
    }

    class RadialProgressView @JvmOverloads constructor(
        context: Context,
        attrs: android.util.AttributeSet? = null
    ) : View(context, attrs) {
        private var progressPercent: Int = 25
        private val density = resources.displayMetrics.density

        private val trackPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = 3.8f * density
            color = Color.parseColor("#1E293B")
            strokeCap = Paint.Cap.ROUND
        }

        private val glowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = 7f * density
            color = Color.parseColor("#3300E5FF")
            strokeCap = Paint.Cap.ROUND
        }

        private val arcPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = 3.8f * density
            color = Color.parseColor("#00E5FF")
            strokeCap = Paint.Cap.ROUND
        }

        private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = 10.5f * density
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
            textAlign = Paint.Align.CENTER
        }

        fun setProgress(percent: Int, animated: Boolean = true) {
            val clamped = percent.coerceIn(0, 100)
            if (animated) {
                val anim = ValueAnimator.ofInt(progressPercent, clamped)
                anim.duration = 380L
                anim.interpolator = DecelerateInterpolator(1.5f)
                anim.addUpdateListener {
                    progressPercent = it.animatedValue as Int
                    invalidate()
                }
                anim.start()
            } else {
                progressPercent = clamped
                invalidate()
            }
        }

        override fun onDraw(canvas: Canvas) {
            super.onDraw(canvas)
            val w = width.toFloat()
            val h = height.toFloat()
            val cx = w / 2f
            val cy = h / 2f
            val stroke = 3.8f * density
            val radius = (Math.min(w, h) / 2f) - stroke - (2f * density)
            val rect = RectF(cx - radius, cy - radius, cx + radius, cy + radius)

            canvas.drawArc(rect, 0f, 360f, false, trackPaint)

            val sweep = (progressPercent / 100f) * 360f
            if (sweep > 0f) {
                canvas.drawArc(rect, -90f, sweep, false, glowPaint)
                canvas.drawArc(rect, -90f, sweep, false, arcPaint)
            }

            val textY = cy - (textPaint.descent() + textPaint.ascent()) / 2f
            canvas.drawText("$progressPercent%", cx, textY, textPaint)
        }
    }

    class PipelineStepperView @JvmOverloads constructor(
        context: Context,
        attrs: android.util.AttributeSet? = null
    ) : View(context, attrs) {
        private var currentPhase: Int = 1 // 1=Understand, 2=Execute, 3=Verify, 4=Complete
        private val density = resources.displayMetrics.density

        private val trackPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = 2f * density
            color = Color.parseColor("#1E293B")
        }

        private val activeLinePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = 2f * density
            color = Color.parseColor("#00E5FF")
        }

        private val lineGlowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = 5f * density
            color = Color.parseColor("#4400E5FF")
        }

        private val inactiveCirclePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
            color = Color.parseColor("#111827")
        }

        private val inactiveBorderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = 1.2f * density
            color = Color.parseColor("#334155")
        }

        private val activeCirclePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
            color = Color.parseColor("#00E5FF")
        }

        private val circleGlowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = 3.5f * density
            color = Color.parseColor("#4400E5FF")
        }

        private val nodeTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            textSize = 9.5f * density
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
            textAlign = Paint.Align.CENTER
        }

        private val labelTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            textSize = 9f * density
            textAlign = Paint.Align.CENTER
        }

        private val phases = listOf("Understand", "Execute", "Verify", "Complete")

        fun setPhase(phase: Int) {
            currentPhase = phase.coerceIn(1, 4)
            invalidate()
        }

        override fun onDraw(canvas: Canvas) {
            super.onDraw(canvas)
            val w = width.toFloat()
            val h = height.toFloat()
            if (w <= 0f || h <= 0f) return

            val nodeRadius = 10f * density
            val topY = 14f * density
            val labelY = topY + nodeRadius + (13f * density)

            val stepCount = 4
            val paddingHorizontal = 28f * density
            val totalSpacing = w - (paddingHorizontal * 2f)
            val stepX = FloatArray(stepCount) { idx ->
                paddingHorizontal + (idx * (totalSpacing / (stepCount - 1)))
            }

            for (i in 0 until stepCount - 1) {
                val startX = stepX[i] + nodeRadius
                val endX = stepX[i + 1] - nodeRadius
                val isLineActive = (i + 1) < currentPhase

                if (isLineActive) {
                    canvas.drawLine(startX, topY, endX, topY, lineGlowPaint)
                    canvas.drawLine(startX, topY, endX, topY, activeLinePaint)
                } else {
                    canvas.drawLine(startX, topY, endX, topY, trackPaint)
                }
            }

            for (i in 0 until stepCount) {
                val cx = stepX[i]
                val cy = topY
                val stepNum = i + 1
                val isCurrent = stepNum == currentPhase
                val isCompleted = stepNum < currentPhase

                when {
                    isCompleted -> {
                        canvas.drawCircle(cx, cy, nodeRadius, activeCirclePaint)
                        nodeTextPaint.color = Color.parseColor("#080D1A")
                        val ty = cy - (nodeTextPaint.descent() + nodeTextPaint.ascent()) / 2f
                        canvas.drawText("✓", cx, ty, nodeTextPaint)
                    }
                    isCurrent -> {
                        canvas.drawCircle(cx, cy, nodeRadius + (3f * density), circleGlowPaint)
                        canvas.drawCircle(cx, cy, nodeRadius, activeCirclePaint)
                        nodeTextPaint.color = Color.parseColor("#080D1A")
                        val ty = cy - (nodeTextPaint.descent() + nodeTextPaint.ascent()) / 2f
                        canvas.drawText("$stepNum", cx, ty, nodeTextPaint)
                    }
                    else -> {
                        canvas.drawCircle(cx, cy, nodeRadius, inactiveCirclePaint)
                        canvas.drawCircle(cx, cy, nodeRadius, inactiveBorderPaint)
                        nodeTextPaint.color = Color.parseColor("#94A3B8")
                        val ty = cy - (nodeTextPaint.descent() + nodeTextPaint.ascent()) / 2f
                        canvas.drawText("$stepNum", cx, ty, nodeTextPaint)
                    }
                }

                if (isCurrent || isCompleted) {
                    labelTextPaint.color = Color.WHITE
                    labelTextPaint.typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
                } else {
                    labelTextPaint.color = Color.parseColor("#64748B")
                    labelTextPaint.typeface = Typeface.create(Typeface.DEFAULT, Typeface.NORMAL)
                }
                canvas.drawText(phases[i], cx, labelY, labelTextPaint)
            }
        }
    }

    // =========================================================================
    // UI BUILDER: MASTER CONTROL HUD CARD (MATCHES USER ATTACHED SCREENSHOT)
    // =========================================================================

    private fun createHudCard(context: Context, initialGoal: String): View {
        val density = context.resources.displayMetrics.density
        val screenWidth = context.resources.displayMetrics.widthPixels
        val screenHeight = context.resources.displayMetrics.heightPixels
        val pillWidth = (screenWidth * 0.90f).toInt().coerceAtMost((420 * density).toInt())
        val cardWidth = (screenWidth * 0.94f).toInt().coerceAtMost((480 * density).toInt())

        // Root wrapper with padding from screen edges
        val root = FrameLayout(context).apply {
            setPadding((12 * density).toInt(), 0, (12 * density).toInt(), 0)
        }

        // -------------------------------------------------------------------------
        // STATE 1: COLLAPSED BOTTOM THINKING PILL (Default Active)
        // Highly compact (~46dp height), non-intrusive, floating above navigation bar
        // -------------------------------------------------------------------------
        val collapsedPill = LinearLayout(context).apply {
            tag = "collapsed_pill_view"
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            visibility = View.VISIBLE
            layoutParams = FrameLayout.LayoutParams(
                pillWidth,
                (46 * density).toInt(),
                Gravity.CENTER_HORIZONTAL
            )
            val pillBg = GradientDrawable(
                GradientDrawable.Orientation.LEFT_RIGHT,
                intArrayOf(Color.parseColor("#0F172A"), Color.parseColor("#090D18"))
            ).apply {
                cornerRadius = 23 * density
            }
            background = pillBg
            elevation = 10 * density
            setPadding(
                (8 * density).toInt(),
                (4 * density).toInt(),
                (8 * density).toInt(),
                (4 * density).toInt()
            )
        }

        // 1. Mini Rotating 3D Thinking Orb ("small ball thinking kind")
        val pillAvatar = KarvishAvatarBadgeView(context).apply {
            layoutParams = LinearLayout.LayoutParams((30 * density).toInt(), (30 * density).toInt()).apply {
                rightMargin = (8 * density).toInt()
            }
        }
        collapsedPill.addView(pillAvatar)

        // 2. Soft Opacity Streaming Text Column ("some opacity text going")
        val pillTextCol = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER_VERTICAL
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply {
                rightMargin = (6 * density).toInt()
            }
        }
        val pillThinkingText = TextView(context).apply {
            tag = "pill_thinking_text"
            text = initialGoal
            textSize = 12f
            setTypeface(null, Typeface.BOLD)
            setTextColor(Color.parseColor("#F1F5F9"))
            alpha = 0.92f
            maxLines = 1
            ellipsize = android.text.TextUtils.TruncateAt.END
        }
        val pillSubtext = TextView(context).apply {
            tag = "pill_subtext"
            text = "Starting autonomous execution..."
            textSize = 9.5f
            setTextColor(Color.parseColor("#64748B"))
            alpha = 0.75f
            maxLines = 1
            ellipsize = android.text.TextUtils.TruncateAt.END
        }
        pillTextCol.addView(pillThinkingText)
        pillTextCol.addView(pillSubtext)
        collapsedPill.addView(pillTextCol)

        // 3. Compact Turn Badge Pill ("1/4")
        val pillTurnBadge = LinearLayout(context).apply {
            tag = "pill_turn_badge"
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            background = GradientDrawable().apply {
                setColor(Color.parseColor("#162032"))
                cornerRadius = 8 * density
                setStroke((1 * density).toInt(), Color.parseColor("#1E293B"))
            }
            setPadding((6 * density).toInt(), (2 * density).toInt(), (6 * density).toInt(), (2 * density).toInt())
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { rightMargin = (6 * density).toInt() }
        }
        val pillTurnText = TextView(context).apply {
            tag = "pill_turn_text"
            text = "1 / 4"
            textSize = 10f
            setTypeface(null, Typeface.BOLD)
            setTextColor(Color.parseColor("#38BDF8"))
        }
        pillTurnBadge.addView(pillTurnText)
        collapsedPill.addView(pillTurnBadge)

        // 4. Expand Chevron ("▲")
        val pillExpandBtn = TextView(context).apply {
            tag = "pill_btn_expand"
            text = "▲"
            textSize = 11f
            gravity = Gravity.CENTER
            setTextColor(Color.parseColor("#94A3B8"))
            background = GradientDrawable().apply {
                setColor(Color.parseColor("#111827"))
                cornerRadius = 8 * density
                setStroke((1 * density).toInt(), Color.parseColor("#1F2937"))
            }
            layoutParams = LinearLayout.LayoutParams((26 * density).toInt(), (26 * density).toInt()).apply {
                rightMargin = (6 * density).toInt()
            }
        }
        collapsedPill.addView(pillExpandBtn)

        // 5. Compact Instant Stop Button ("■")
        val pillStopBtn = TextView(context).apply {
            tag = "pill_btn_stop"
            text = "■"
            textSize = 11f
            gravity = Gravity.CENTER
            setTextColor(Color.parseColor("#EF4444"))
            background = GradientDrawable().apply {
                setColor(Color.parseColor("#3B1219"))
                cornerRadius = 8 * density
                setStroke((1 * density).toInt(), Color.parseColor("#7F1D1D"))
            }
            layoutParams = LinearLayout.LayoutParams((26 * density).toInt(), (26 * density).toInt())
            setOnClickListener {
                onCancelClickListener?.invoke()
                hide()
            }
        }
        collapsedPill.addView(pillStopBtn)

        // -------------------------------------------------------------------------
        // STATE 2: EXPANDED FULL TELEMETRY CARD (Default Hidden)
        // Opens upward when user taps the pill or expand arrow
        // -------------------------------------------------------------------------
        val card = LinearLayout(context).apply {
            tag = "hud_card_container"
            orientation = LinearLayout.VERTICAL
            visibility = View.GONE
            layoutParams = FrameLayout.LayoutParams(
                cardWidth,
                FrameLayout.LayoutParams.WRAP_CONTENT,
                Gravity.CENTER_HORIZONTAL
            )
            val cardBg = GradientDrawable(
                GradientDrawable.Orientation.TOP_BOTTOM,
                intArrayOf(Color.parseColor("#0F172A"), Color.parseColor("#080D1A"))
            ).apply {
                cornerRadius = 24 * density
            }
            background = cardBg
            setPadding(
                (16 * density).toInt(),
                (14 * density).toInt(),
                (16 * density).toInt(),
                (14 * density).toInt()
            )
        }

        fun expandToFullCard() {
            collapsedPill.visibility = View.GONE
            card.visibility = View.VISIBLE
            card.alpha = 0f
            card.scaleY = 0.92f
            card.animate().alpha(1f).scaleY(1f).setDuration(220L).start()
            currentHudParams?.width = cardWidth
            try {
                windowManager?.updateViewLayout(root, currentHudParams)
            } catch (_: Exception) {}
        }

        fun collapseToPill() {
            card.visibility = View.GONE
            collapsedPill.visibility = View.VISIBLE
            collapsedPill.alpha = 0f
            collapsedPill.scaleY = 0.92f
            collapsedPill.animate().alpha(1f).scaleY(1f).setDuration(220L).start()
            currentHudParams?.width = pillWidth
            try {
                windowManager?.updateViewLayout(root, currentHudParams)
            } catch (_: Exception) {}
        }

        pillExpandBtn.setOnClickListener { expandToFullCard() }

        // Touch & Drag on Collapsed Pill (drag moves vertically; tap expands)
        var pillInitialY = (24 * density).toInt()
        var pillInitialTouchY = 0f
        var pillIsDragging = false

        collapsedPill.setOnTouchListener { _, event ->
            val wm = windowManager
            val params = currentHudParams
            if (wm == null || params == null) return@setOnTouchListener false

            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    pillInitialY = params.y
                    pillInitialTouchY = event.rawY
                    pillIsDragging = false
                    true
                }
                MotionEvent.ACTION_MOVE -> {
                    val dy = (pillInitialTouchY - event.rawY).toInt()
                    if (abs(dy) > (5 * density).toInt()) {
                        pillIsDragging = true
                        params.y = (pillInitialY + dy).coerceIn((12 * density).toInt(), screenHeight - (100 * density).toInt())
                        try {
                            wm.updateViewLayout(root, params)
                        } catch (_: Exception) {}
                    }
                    true
                }
                MotionEvent.ACTION_UP -> {
                    if (!pillIsDragging) {
                        expandToFullCard()
                    }
                    pillIsDragging = false
                    true
                }
                MotionEvent.ACTION_CANCEL -> {
                    pillIsDragging = false
                    true
                }
                else -> false
            }
        }

        // Dedicated Main HUD Panel (holds all live telemetry views)
        val mainPanel = LinearLayout(context).apply {
            tag = "main_hud_panel"
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
        }

        // --- ROW 1: HEADER (Avatar + "KARVISH CONTROL" + Running Badge + Window Controls) ---
        val headerRow = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { bottomMargin = (12 * density).toInt() }
        }

        // Drag handle on header for expanded card
        var initialY = (24 * density).toInt()
        var initialTouchY = 0f
        var isDragging = false

        headerRow.setOnTouchListener { _, event ->
            val wm = windowManager
            val params = currentHudParams
            if (wm == null || params == null) return@setOnTouchListener false

            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    initialY = params.y
                    initialTouchY = event.rawY
                    isDragging = false
                    true
                }
                MotionEvent.ACTION_MOVE -> {
                    val dy = (initialTouchY - event.rawY).toInt()
                    if (abs(dy) > 4) {
                        isDragging = true
                        params.y = (initialY + dy).coerceIn((12 * density).toInt(), screenHeight - (100 * density).toInt())
                        try {
                            wm.updateViewLayout(hudView, params)
                        } catch (_: Exception) {}
                    }
                    true
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    isDragging = false
                    true
                }
                else -> false
            }
        }

        // 1. Avatar Badge (Rotating 3D Liquid Core)
        val avatarBadge = KarvishAvatarBadgeView(context).apply {
            layoutParams = LinearLayout.LayoutParams((36 * density).toInt(), (36 * density).toInt()).apply {
                rightMargin = (10 * density).toInt()
            }
        }

        // 2. Title Block: "KARVISH" + "CONTROL"
        val titleBlock = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { rightMargin = (10 * density).toInt() }
        }
        val headerTitle = TextView(context).apply {
            tag = "header_title"
            text = "KARVISH"
            textSize = 13.5f
            setTypeface(null, Typeface.BOLD)
            setTextColor(Color.WHITE)
            letterSpacing = 0.12f
        }
        val headerSubtitle = TextView(context).apply {
            text = "C O N T R O L"
            textSize = 8.5f
            setTypeface(null, Typeface.BOLD)
            setTextColor(Color.parseColor("#64748B"))
            letterSpacing = 0.22f
        }
        titleBlock.addView(headerTitle)
        titleBlock.addView(headerSubtitle)

        // 3. Status Badge: "● Running"
        val statusBadge = LinearLayout(context).apply {
            tag = "status_badge_pill"
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            background = GradientDrawable().apply {
                setColor(Color.parseColor("#0F223A"))
                cornerRadius = 12 * density
                setStroke((1 * density).toInt(), Color.parseColor("#1E3A8A"))
            }
            setPadding((8 * density).toInt(), (3 * density).toInt(), (10 * density).toInt(), (3 * density).toInt())
        }
        val statusBadgeDot = TextView(context).apply {
            tag = "status_badge_dot"
            text = "●"
            textSize = 10f
            setTextColor(Color.parseColor("#38BDF8"))
        }
        val statusBadgeText = TextView(context).apply {
            tag = "status_badge_text"
            text = "Running"
            textSize = 10.5f
            setTypeface(null, Typeface.BOLD)
            setTextColor(Color.parseColor("#38BDF8"))
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { leftMargin = (4 * density).toInt() }
        }
        statusBadge.addView(statusBadgeDot)
        statusBadge.addView(statusBadgeText)

        // Spacer to push window controls right
        val headerSpacer = View(context).apply {
            layoutParams = LinearLayout.LayoutParams(0, 0, 1f)
        }

        // 4. Window Action Buttons [📌] [➖] [✕]
        val windowControls = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }

        fun makePillButton(textStr: String, onClick: () -> Unit): TextView {
            return TextView(context).apply {
                text = textStr
                textSize = 12f
                gravity = Gravity.CENTER
                setTextColor(Color.parseColor("#CBD5E1"))
                background = GradientDrawable().apply {
                    setColor(Color.parseColor("#111827"))
                    cornerRadius = 8 * density
                    setStroke((1 * density).toInt(), Color.parseColor("#1F2937"))
                }
                layoutParams = LinearLayout.LayoutParams((28 * density).toInt(), (28 * density).toInt()).apply {
                    leftMargin = (6 * density).toInt()
                }
                setOnClickListener { onClick() }
            }
        }

        val btnCollapse = makePillButton("▼") {
            collapseToPill()
        }
        val btnPin = makePillButton("📌") {
            Toast.makeText(context, "KARVISH HUD: Floating Pin Active", Toast.LENGTH_SHORT).show()
        }
        val btnMinimize = makePillButton("➖") {
            minimize()
        }
        val btnClose = makePillButton("✕") {
            onCancelClickListener?.invoke()
            hide()
        }

        windowControls.addView(btnCollapse)
        windowControls.addView(btnPin)
        windowControls.addView(btnMinimize)
        windowControls.addView(btnClose)

        headerRow.addView(avatarBadge)
        headerRow.addView(titleBlock)
        headerRow.addView(statusBadge)
        headerRow.addView(headerSpacer)
        headerRow.addView(windowControls)
        mainPanel.addView(headerRow)

        // --- ROW 2: ACTIVE TASK CARD (Dynamic App Icon + Progress + Radial Donut) ---
        val taskCard = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            background = GradientDrawable().apply {
                setColor(Color.parseColor("#0C1424"))
                cornerRadius = 16 * density
                setStroke((1 * density).toInt(), Color.parseColor("#1E293B"))
            }
            setPadding((12 * density).toInt(), (10 * density).toInt(), (12 * density).toInt(), (10 * density).toInt())
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { bottomMargin = (12 * density).toInt() }
        }

        // Dynamic App Icon (Squircle)
        val iconContainer = FrameLayout(context).apply {
            background = GradientDrawable().apply {
                setColor(Color.parseColor("#162032"))
                cornerRadius = 12 * density
            }
            layoutParams = LinearLayout.LayoutParams((44 * density).toInt(), (44 * density).toInt()).apply {
                rightMargin = (10 * density).toInt()
            }
        }
        val appIconView = ImageView(context).apply {
            tag = "app_icon_view"
            scaleType = ImageView.ScaleType.FIT_CENTER
            setPadding((4 * density).toInt(), (4 * density).toInt(), (4 * density).toInt(), (4 * density).toInt())
            layoutParams = FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            )
            val initialIcon = resolveAppIcon(context, initialGoal, targetView = this)
            if (initialIcon != null) {
                setImageDrawable(initialIcon)
            } else {
                setImageResource(android.R.drawable.sym_def_app_icon)
            }
        }
        iconContainer.addView(appIconView)
        taskCard.addView(iconContainer)

        // Center Details: Title + Subtitle + Linear Cyan Progress Bar
        val taskDetails = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }

        val topDetailRow = RelativeLayout(context).apply {
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
        }
        val statusText = TextView(context).apply {
            tag = "status_text"
            text = initialGoal
            textSize = 13.5f
            setTypeface(null, Typeface.BOLD)
            setTextColor(Color.WHITE)
            maxLines = 1
            ellipsize = android.text.TextUtils.TruncateAt.END
            layoutParams = RelativeLayout.LayoutParams(
                RelativeLayout.LayoutParams.WRAP_CONTENT,
                RelativeLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                addRule(RelativeLayout.ALIGN_PARENT_LEFT)
                addRule(RelativeLayout.LEFT_OF, 1001)
            }
        }
        val turnText = TextView(context).apply {
            id = 1001
            tag = "turn_text"
            text = "1 / 4"
            textSize = 11f
            setTextColor(Color.parseColor("#64748B"))
            layoutParams = RelativeLayout.LayoutParams(
                RelativeLayout.LayoutParams.WRAP_CONTENT,
                RelativeLayout.LayoutParams.WRAP_CONTENT
            ).apply { addRule(RelativeLayout.ALIGN_PARENT_RIGHT) }
        }
        topDetailRow.addView(statusText)
        topDetailRow.addView(turnText)
        taskDetails.addView(topDetailRow)

        val statusSubtext = TextView(context).apply {
            tag = "status_subtext"
            text = "Executing autonomous action on screen"
            textSize = 10.5f
            setTextColor(Color.parseColor("#64748B"))
            maxLines = 1
            ellipsize = android.text.TextUtils.TruncateAt.END
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                topMargin = (2 * density).toInt()
                bottomMargin = (6 * density).toInt()
            }
        }
        taskDetails.addView(statusSubtext)

        // Linear Progress Bar (Glowing Cyan Gradient)
        val linearProgressBar = ProgressBar(context, null, android.R.attr.progressBarStyleHorizontal).apply {
            tag = "task_progress_bar"
            max = 100
            progress = 25
            val barHeight = (4 * density).toInt()
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                barHeight
            )
            val bgDrawable = GradientDrawable().apply {
                cornerRadius = 2 * density
                setColor(Color.parseColor("#1E293B"))
            }
            val progDrawable = GradientDrawable(
                GradientDrawable.Orientation.LEFT_RIGHT,
                intArrayOf(Color.parseColor("#00E5FF"), Color.parseColor("#38BDF8"))
            ).apply {
                cornerRadius = 2 * density
            }
            progressDrawable = LayerDrawable(
                arrayOf(
                    bgDrawable,
                    ClipDrawable(progDrawable, Gravity.START, ClipDrawable.HORIZONTAL)
                )
            ).apply {
                setId(0, android.R.id.background)
                setId(1, android.R.id.progress)
            }
        }
        taskDetails.addView(linearProgressBar)
        taskCard.addView(taskDetails)

        // Right: Radial Donut Progress Ring
        val radialProgress = RadialProgressView(context).apply {
            tag = "radial_progress_view"
            layoutParams = LinearLayout.LayoutParams((46 * density).toInt(), (46 * density).toInt()).apply {
                leftMargin = (10 * density).toInt()
            }
            setProgress(25, animated = false)
        }
        taskCard.addView(radialProgress)
        mainPanel.addView(taskCard)

        // --- ROW 3: PIPELINE STEPPER (Understand -> Execute -> Verify -> Complete) ---
        val pipelineStepper = PipelineStepperView(context).apply {
            tag = "pipeline_stepper_view"
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                (44 * density).toInt()
            ).apply { bottomMargin = (12 * density).toInt() }
            setPhase(1)
        }
        mainPanel.addView(pipelineStepper)

        // --- ROW 4: 3-COLUMN TELEMETRY GRID (Action / Target / Confidence) ---
        val telemetryRow = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { bottomMargin = (10 * density).toInt() }
        }

        fun makeTelemetryCard(
            iconText: String,
            iconColor: Int,
            labelStr: String,
            valTag: String,
            defaultVal: String,
            valColor: Int,
            includeBar: Boolean = false
        ): View {
            return LinearLayout(context).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                background = GradientDrawable().apply {
                    setColor(Color.parseColor("#0C1424"))
                    cornerRadius = 12 * density
                    setStroke((1 * density).toInt(), Color.parseColor("#1E293B"))
                }
                setPadding((8 * density).toInt(), (8 * density).toInt(), (8 * density).toInt(), (8 * density).toInt())

                val iconBox = TextView(context).apply {
                    text = iconText
                    textSize = 11.5f
                    setTextColor(iconColor)
                    gravity = Gravity.CENTER
                    background = GradientDrawable().apply {
                        setColor(Color.parseColor("#162238"))
                        cornerRadius = 6 * density
                    }
                    layoutParams = LinearLayout.LayoutParams((26 * density).toInt(), (26 * density).toInt()).apply {
                        rightMargin = (6 * density).toInt()
                    }
                }
                addView(iconBox)

                val textCol = LinearLayout(context).apply {
                    orientation = LinearLayout.VERTICAL
                    layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
                }
                val lbl = TextView(context).apply {
                    text = labelStr
                    textSize = 8.5f
                    setTextColor(Color.parseColor("#64748B"))
                }
                val vl = TextView(context).apply {
                    tag = valTag
                    text = defaultVal
                    textSize = 11f
                    setTypeface(null, Typeface.BOLD)
                    setTextColor(valColor)
                    maxLines = 1
                    ellipsize = android.text.TextUtils.TruncateAt.END
                }
                textCol.addView(lbl)
                textCol.addView(vl)

                if (includeBar) {
                    val cBar = ProgressBar(context, null, android.R.attr.progressBarStyleHorizontal).apply {
                        tag = "conf_bar"
                        max = 100
                        progress = 100
                        val barHeight = (3 * density).toInt()
                        layoutParams = LinearLayout.LayoutParams(
                            LinearLayout.LayoutParams.MATCH_PARENT,
                            barHeight
                        ).apply { topMargin = (2 * density).toInt() }

                        val bg = GradientDrawable().apply {
                            cornerRadius = 1.5f * density
                            setColor(Color.parseColor("#1E293B"))
                        }
                        val fg = GradientDrawable().apply {
                            cornerRadius = 1.5f * density
                            setColor(Color.parseColor("#10B981"))
                        }
                        progressDrawable = LayerDrawable(
                            arrayOf(
                                bg,
                                ClipDrawable(fg, Gravity.START, ClipDrawable.HORIZONTAL)
                            )
                        ).apply {
                            setId(0, android.R.id.background)
                            setId(1, android.R.id.progress)
                        }
                    }
                    textCol.addView(cBar)
                }

                addView(textCol)
            }
        }

        val cardAction = makeTelemetryCard("↖", Color.parseColor("#38BDF8"), "Action", "action_val", "TAP (Open App)", Color.parseColor("#38BDF8")).apply {
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply {
                rightMargin = (6 * density).toInt()
            }
        }
        val cardTarget = makeTelemetryCard("🎯", Color.parseColor("#38BDF8"), "Target", "target_val", "Instagram", Color.WHITE).apply {
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply {
                rightMargin = (6 * density).toInt()
            }
        }
        val cardConfidence = makeTelemetryCard("📊", Color.parseColor("#10B981"), "Confidence", "conf_val", "100%", Color.parseColor("#10B981"), includeBar = true).apply {
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }

        telemetryRow.addView(cardAction)
        telemetryRow.addView(cardTarget)
        telemetryRow.addView(cardConfidence)
        mainPanel.addView(telemetryRow)

        // --- ROW 5: THINKING MODE (LIVE TRACE) ACCORDION ---
        val thinkingAccordion = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            background = GradientDrawable().apply {
                setColor(Color.parseColor("#0C1424"))
                cornerRadius = 12 * density
                setStroke((1 * density).toInt(), Color.parseColor("#1E293B"))
            }
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { bottomMargin = (10 * density).toInt() }
        }

        val thinkingHeader = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding((10 * density).toInt(), (8 * density).toInt(), (10 * density).toInt(), (8 * density).toInt())
        }
        val waveIcon = TextView(context).apply {
            text = "〰️"
            textSize = 12f
            setTextColor(Color.parseColor("#818CF8"))
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { rightMargin = (6 * density).toInt() }
        }
        val thinkingTitle = TextView(context).apply {
            text = "Thinking Mode (Live Trace)"
            textSize = 11.5f
            setTextColor(Color.parseColor("#CBD5E1"))
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }
        val chevron = TextView(context).apply {
            tag = "thinking_chevron"
            text = "∨"
            textSize = 12f
            setTypeface(null, Typeface.BOLD)
            setTextColor(Color.parseColor("#64748B"))
        }
        thinkingHeader.addView(waveIcon)
        thinkingHeader.addView(thinkingTitle)
        thinkingHeader.addView(chevron)
        thinkingAccordion.addView(thinkingHeader)

        val thinkingContent = LinearLayout(context).apply {
            tag = "thinking_content_container"
            orientation = LinearLayout.VERTICAL
            visibility = View.GONE
            setPadding((10 * density).toInt(), 0, (10 * density).toInt(), (8 * density).toInt())
        }
        val thinkingText = TextView(context).apply {
            tag = "thinking_trace_text"
            text = "Neural policy evaluating screen affordances..."
            textSize = 10f
            typeface = Typeface.MONOSPACE
            setTextColor(Color.parseColor("#94A3B8"))
            maxLines = 6
            ellipsize = android.text.TextUtils.TruncateAt.END
            background = GradientDrawable().apply {
                setColor(Color.parseColor("#060B14"))
                cornerRadius = 8 * density
            }
            setPadding((8 * density).toInt(), (6 * density).toInt(), (8 * density).toInt(), (6 * density).toInt())
        }
        thinkingContent.addView(thinkingText)
        thinkingAccordion.addView(thinkingContent)

        var isThinkingExpanded = false
        thinkingHeader.setOnClickListener {
            isThinkingExpanded = !isThinkingExpanded
            thinkingContent.visibility = if (isThinkingExpanded) View.VISIBLE else View.GONE
            chevron.text = if (isThinkingExpanded) "∧" else "∨"
        }
        mainPanel.addView(thinkingAccordion)

        // --- ROW 6: FOOTER ACTIONS ([⏹ Stop (Hard Stop)], [⏸ Pause], [••• More ∨]) ---
        val footerRow = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
        }

        // Button 1: [⏹ Stop (Hard Stop)]
        val btnStop = LinearLayout(context).apply {
            tag = "btn_stop_hard"
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            background = GradientDrawable(
                GradientDrawable.Orientation.TOP_BOTTOM,
                intArrayOf(Color.parseColor("#991B1B"), Color.parseColor("#7F1D1D"))
            ).apply {
                cornerRadius = 14 * density
                setStroke((1 * density).toInt(), Color.parseColor("#DC2626"))
            }
            layoutParams = LinearLayout.LayoutParams(0, (40 * density).toInt(), 1.25f).apply {
                rightMargin = (8 * density).toInt()
            }
            val stopIcon = TextView(context).apply {
                text = "■"
                textSize = 13f
                setTextColor(Color.WHITE)
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply { rightMargin = (6 * density).toInt() }
            }
            val stopTextCol = LinearLayout(context).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER_HORIZONTAL
            }
            val stopTitle = TextView(context).apply {
                text = "Stop"
                textSize = 12f
                setTypeface(null, Typeface.BOLD)
                setTextColor(Color.WHITE)
            }
            val stopSub = TextView(context).apply {
                text = "(Hard Stop)"
                textSize = 8f
                setTextColor(Color.parseColor("#FCA5A5"))
            }
            stopTextCol.addView(stopTitle)
            stopTextCol.addView(stopSub)
            addView(stopIcon)
            addView(stopTextCol)

            setOnClickListener {
                onCancelClickListener?.invoke()
                hide()
            }
        }

        // Button 2: [⏸ Pause]
        val btnPause = LinearLayout(context).apply {
            tag = "btn_pause"
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            background = GradientDrawable().apply {
                setColor(Color.parseColor("#111827"))
                cornerRadius = 14 * density
                setStroke((1 * density).toInt(), Color.parseColor("#1F2937"))
            }
            layoutParams = LinearLayout.LayoutParams(0, (40 * density).toInt(), 1f).apply {
                rightMargin = (8 * density).toInt()
            }
            val pauseIcon = TextView(context).apply {
                tag = "pause_icon"
                text = "⏸"
                textSize = 12f
                setTextColor(Color.WHITE)
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply { rightMargin = (4 * density).toInt() }
            }
            val pauseText = TextView(context).apply {
                tag = "pause_text"
                text = "Pause"
                textSize = 12f
                setTypeface(null, Typeface.BOLD)
                setTextColor(Color.WHITE)
            }
            addView(pauseIcon)
            addView(pauseText)

            setOnClickListener {
                isPaused = !isPaused
                pauseText.text = if (isPaused) "Resume" else "Pause"
                pauseIcon.text = if (isPaused) "▶" else "⏸"
                statusBadgeText.text = if (isPaused) "Paused" else "Running"
                statusBadgeText.setTextColor(if (isPaused) Color.parseColor("#F59E0B") else Color.parseColor("#38BDF8"))
                statusBadgeDot.setTextColor(if (isPaused) Color.parseColor("#F59E0B") else Color.parseColor("#38BDF8"))
                onPauseClickListener?.invoke(isPaused)
            }
        }

        // Button 3: [••• More ∨]
        val btnMore = TextView(context).apply {
            tag = "btn_more"
            text = "••• More ∨"
            textSize = 11.5f
            setTypeface(null, Typeface.BOLD)
            setTextColor(Color.WHITE)
            gravity = Gravity.CENTER
            background = GradientDrawable().apply {
                setColor(Color.parseColor("#111827"))
                cornerRadius = 14 * density
                setStroke((1 * density).toInt(), Color.parseColor("#1F2937"))
            }
            layoutParams = LinearLayout.LayoutParams(0, (40 * density).toInt(), 1f)
            setOnClickListener {
                Toast.makeText(context, "KARVISH V61 Policy • Single GLThread Active • 0-Spoonfeeding", Toast.LENGTH_SHORT).show()
            }
        }

        footerRow.addView(btnStop)
        footerRow.addView(btnPause)
        footerRow.addView(btnMore)
        mainPanel.addView(footerRow)

        card.addView(mainPanel)

        // Custom Card Slot (for Disambiguation, Blocked banner, Approval Gate)
        val customCardSlot = LinearLayout(context).apply {
            tag = "custom_card_slot"
            orientation = LinearLayout.VERTICAL
            visibility = View.GONE
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
        }
        card.addView(customCardSlot)

        root.addView(collapsedPill)
        root.addView(card)
        return root
    }

    // =========================================================================
    // CANVAS VIEW: NEON TARGET BOUNDING BOX & STEP BADGE
    // =========================================================================

    data class NumberedCandidate(
        val index: Int,
        val label: String,
        val rect: Rect? = null
    )

    private class HighlightCanvasView(context: Context) : View(context) {
        private var targetRect: Rect? = null
        private var stepIndex: Int = 1
        private var pulseFraction = 0f
        private var numberedCandidates: List<NumberedCandidate> = emptyList()

        private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = 3.5f * resources.displayMetrics.density
            color = Color.parseColor("#00E5FF")
        }

        private val glowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = 8f * resources.displayMetrics.density
            color = Color.parseColor("#3300E5FF")
        }

        private val badgeBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
            color = Color.parseColor("#00E5FF")
        }

        private val badgeTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#0A1128")
            textSize = 12f * resources.displayMetrics.density
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
            textAlign = Paint.Align.CENTER
        }

        private val animator = ValueAnimator.ofFloat(0f, 1f).apply {
            duration = 1000
            repeatMode = ValueAnimator.REVERSE
            repeatCount = ValueAnimator.INFINITE
            addUpdateListener {
                pulseFraction = it.animatedValue as Float
                invalidate()
            }
        }

        init {
            animator.start()
        }

        fun setTarget(rect: Rect?, step: Int) {
            targetRect = rect
            stepIndex = step
            invalidate()
        }

        fun clearTarget() {
            targetRect = null
            invalidate()
        }

        fun setNumberedCandidates(candidates: List<NumberedCandidate>) {
            numberedCandidates = candidates
            invalidate()
        }

        fun clearNumberedCandidates() {
            numberedCandidates = emptyList()
            invalidate()
        }

        override fun onDraw(canvas: Canvas) {
            super.onDraw(canvas)
            val density = resources.displayMetrics.density
            val loc = IntArray(2)
            getLocationOnScreen(loc)

            // 1. Draw Multi-Candidate Numbered Badges & Neon Boxes (Disambiguation [1], [2], [3]...)
            if (numberedCandidates.isNotEmpty()) {
                for (cand in numberedCandidates) {
                    val rect = cand.rect ?: continue
                    val rF = RectF(
                        (rect.left - loc[0]).toFloat(),
                        (rect.top - loc[1]).toFloat(),
                        (rect.right - loc[0]).toFloat(),
                        (rect.bottom - loc[1]).toFloat()
                    )
                    rF.inset(-2f * density, -2f * density)
                    val radius = 10f * density

                    // Neon Box & Glow
                    glowPaint.alpha = (60 + pulseFraction * 70).toInt()
                    canvas.drawRoundRect(rF, radius, radius, glowPaint)

                    boxPaint.alpha = (210 + pulseFraction * 45).toInt()
                    canvas.drawRoundRect(rF, radius, radius, boxPaint)

                    // Top-Left Neon Badge [1], [2], [3]...
                    val badgeRadius = 12f * density
                    val badgeCx = rF.left + (16f * density)
                    val badgeCy = rF.top + (16f * density)
                    canvas.drawCircle(badgeCx, badgeCy, badgeRadius + 2f * density, glowPaint)
                    canvas.drawCircle(badgeCx, badgeCy, badgeRadius, badgeBgPaint)

                    val textY = badgeCy - (badgeTextPaint.descent() + badgeTextPaint.ascent()) / 2f
                    canvas.drawText("${cand.index}", badgeCx, textY, badgeTextPaint)
                }
                return
            }

            // 2. Draw Single Target Neon Box
            val rect = targetRect ?: return

            // Convert screen-absolute bounds to this View's local canvas coordinates
            val rF = RectF(
                (rect.left - loc[0]).toFloat(),
                (rect.top - loc[1]).toFloat(),
                (rect.right - loc[0]).toFloat(),
                (rect.bottom - loc[1]).toFloat()
            )

            // Add subtle padding (-2dp) so bounding box comfortably frames the element
            rF.inset(-2f * density, -2f * density)

            val radius = 10f * density

            // Draw glowing outer halo
            glowPaint.alpha = (50 + pulseFraction * 60).toInt()
            canvas.drawRoundRect(rF, radius, radius, glowPaint)

            // Draw bright cyan rounded bounding box
            boxPaint.alpha = (200 + pulseFraction * 55).toInt()
            canvas.drawRoundRect(rF, radius, radius, boxPaint)

            // Draw numbered badge at top-right corner
            val badgeRadius = 10f * density
            val badgeCx = rF.right - (4f * density)
            val badgeCy = rF.top + (4f * density)
            canvas.drawCircle(badgeCx, badgeCy, badgeRadius + 2f * density, glowPaint)
            canvas.drawCircle(badgeCx, badgeCy, badgeRadius, badgeBgPaint)

            val textY = badgeCy - (badgeTextPaint.descent() + badgeTextPaint.ascent()) / 2f
            canvas.drawText("$stepIndex", badgeCx, textY, badgeTextPaint)
        }
    }

    // =========================================================================
    // ANIMATED ROTATING SPINNER DOT VIEW
    // =========================================================================

    private class SpinnerDotView(context: Context) : View(context) {
        private var angle = 0f
        private val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = 2.5f * resources.displayMetrics.density
            color = Color.parseColor("#00E5FF")
        }

        private val anim = ValueAnimator.ofFloat(0f, 360f).apply {
            duration = 1200
            repeatCount = ValueAnimator.INFINITE
            interpolator = android.view.animation.LinearInterpolator()
            addUpdateListener {
                angle = it.animatedValue as Float
                invalidate()
            }
        }

        init {
            anim.start()
        }

        override fun onDraw(canvas: Canvas) {
            super.onDraw(canvas)
            val w = width.toFloat()
            val h = height.toFloat()
            val r = (Math.min(w, h) / 2f) - paint.strokeWidth
            val rect = RectF(w / 2f - r, h / 2f - r, w / 2f + r, h / 2f + r)
            canvas.drawArc(rect, angle, 270f, false, paint)
        }
    }
}
