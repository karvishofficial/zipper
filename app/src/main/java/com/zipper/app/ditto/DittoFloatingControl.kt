package com.zipper.app.ditto

import android.animation.ValueAnimator
import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Color
import android.graphics.PixelFormat
import android.graphics.drawable.GradientDrawable
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.ViewConfiguration
import android.view.WindowManager
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import com.zipper.app.accessibility.KarvishAccessibilityService
import kotlin.math.abs

/**
 * DITTO FLOATING CONTROL OVERLAY (Crash-Proof V2.2)
 *
 * Movable, Collapsible/Unhidable on-screen HUD for Ditto demonstration recording.
 * Stays on top of third-party apps via WindowManager.
 *
 * Fully protected with crash-proof WindowManager error handling to prevent BadTokenException
 * and ViewNotAttachedException across app transitions.
 */
object DittoFloatingControl {
    private const val TAG = "DittoFloatingControl"

    private var windowManager: WindowManager? = null
    private var controlView: View? = null
    private var layoutParams: WindowManager.LayoutParams? = null

    private var isShowing = false
    private var isCollapsed = false
    private var isPaused = false
    private var currentStepCount = 0

    private var onFinishClickListener: (() -> Unit)? = null
    private var onPauseToggleListener: ((Boolean) -> Unit)? = null
    private var onCancelClickListener: (() -> Unit)? = null

    private val mainHandler = Handler(Looper.getMainLooper())

    fun isShowing(): Boolean = isShowing

    /**
     * Shows the movable, collapsible floating control HUD.
     */
    fun show(
        context: Context,
        onFinish: () -> Unit,
        onPauseToggle: ((Boolean) -> Unit)? = null,
        onCancel: (() -> Unit)? = null
    ) {
        onFinishClickListener = onFinish
        onPauseToggleListener = onPauseToggle
        onCancelClickListener = onCancel
        currentStepCount = 0
        isPaused = false
        isCollapsed = false

        mainHandler.post {
            try {
                if (isShowing && controlView != null) {
                    updateStep(0, "Ready")
                    return@post
                }

                val a11y = KarvishAccessibilityService.instance
                val targetContext = a11y ?: context.applicationContext
                val wm = targetContext.getSystemService(Context.WINDOW_SERVICE) as? WindowManager
                windowManager = wm
                if (wm == null) {
                    Log.e(TAG, "WindowManager unavailable")
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

                val density = targetContext.resources.displayMetrics.density
                val screenHeight = targetContext.resources.displayMetrics.heightPixels
                val defaultY = maxOf((400 * density).toInt(), screenHeight - (150 * density).toInt())

                val params = WindowManager.LayoutParams(
                    WindowManager.LayoutParams.WRAP_CONTENT,
                    WindowManager.LayoutParams.WRAP_CONTENT,
                    overlayType,
                    WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                            WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN or
                            WindowManager.LayoutParams.FLAG_WATCH_OUTSIDE_TOUCH,
                    PixelFormat.TRANSLUCENT
                ).apply {
                    gravity = Gravity.TOP or Gravity.START
                    x = (16 * density).toInt()
                    y = defaultY // Placed near bottom edge so it NEVER covers top search bars / app headers!
                }
                layoutParams = params

                val view = createControlLayout(targetContext, params)
                safeWmCall("addView") {
                    wm.addView(view, params)
                }
                controlView = view
                isShowing = true
                Log.i(TAG, "DittoFloatingControl safely attached to WindowManager")
            } catch (e: Throwable) {
                Log.e(TAG, "Failed to show DittoFloatingControl: ${e.message}", e)
            }
        }
    }

    /**
     * Updates the step counter and status message.
     */
    fun updateStep(stepCount: Int, summary: String = "") {
        currentStepCount = stepCount
        mainHandler.post {
            try {
                controlView?.let { root ->
                    val stepTv = root.findViewById<TextView>(ID_STEP_TEXT)
                    stepTv?.text = "Step $stepCount"
                    val sumTv = root.findViewById<TextView>(ID_SUMMARY_TEXT)
                    if (summary.isNotBlank()) {
                        sumTv?.text = summary
                        sumTv?.visibility = View.VISIBLE
                    } else {
                        sumTv?.visibility = View.GONE
                    }
                }
            } catch (e: Throwable) {
                Log.w(TAG, "Failed to updateStep: ${e.message}")
            }
        }
    }

    /**
     * Dismisses the floating HUD.
     */
    fun hide() {
        mainHandler.post {
            try {
                val v = controlView
                val wm = windowManager
                if (v != null && wm != null) {
                    safeWmCall("removeView") {
                        wm.removeView(v)
                    }
                    Log.i(TAG, "DittoFloatingControl safely removed from WindowManager")
                }
            } catch (e: Throwable) {
                Log.w(TAG, "Error removing DittoFloatingControl: ${e.message}")
            } finally {
                controlView = null
                isShowing = false
                isCollapsed = false
                onFinishClickListener = null
                onPauseToggleListener = null
                onCancelClickListener = null
            }
        }
    }

    private inline fun safeWmCall(actionName: String, block: () -> Unit) {
        try {
            block()
        } catch (e: Throwable) {
            Log.w(TAG, "WindowManager safe guard caught on $actionName: ${e.message}")
        }
    }

    // =========================================================================
    // View Construction & Draggable Logic
    // =========================================================================

    private const val ID_EXPANDED_CONTAINER = 1001
    private const val ID_COLLAPSED_CONTAINER = 1002
    private const val ID_STEP_TEXT = 1003
    private const val ID_SUMMARY_TEXT = 1004
    private const val ID_REC_DOT = 1005
    private const val ID_PAUSE_BTN = 1006

    @SuppressLint("ClickableViewAccessibility")
    private fun createControlLayout(context: Context, params: WindowManager.LayoutParams): View {
        val density = context.resources.displayMetrics.density
        val touchSlop = ViewConfiguration.get(context).scaledTouchSlop

        val root = FrameLayout(context)

        // 1. Expanded Panel (Full Controls)
        val expandedPanel = LinearLayout(context).apply {
            id = ID_EXPANDED_CONTAINER
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            val padH = (12 * density).toInt()
            val padV = (8 * density).toInt()
            setPadding(padH, padV, padH, padV)
            background = GradientDrawable().apply {
                shape = GradientDrawable.RECTANGLE
                cornerRadius = 24 * density
                setColor(Color.parseColor("#EE10141E")) // Deep cyber dark
                setStroke((1.5f * density).toInt(), Color.parseColor("#00E676")) // Neon emerald border
            }
            elevation = 12 * density
        }

        // Pulse dot
        val dotView = View(context).apply {
            id = ID_REC_DOT
            val dotSize = (10 * density).toInt()
            layoutParams = LinearLayout.LayoutParams(dotSize, dotSize).apply {
                marginEnd = (8 * density).toInt()
            }
            background = GradientDrawable().apply {
                shape = GradientDrawable.OVAL
                setColor(Color.parseColor("#FF1744")) // Vibrant Red
            }
        }
        startPulseAnimation(dotView)
        expandedPanel.addView(dotView)

        // Text Info (Step count + Status) - Constrained to prevent pushing buttons off screen
        val maxTextWidthPx = (100 * density).toInt()
        val textContainer = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                marginEnd = (8 * density).toInt()
            }
        }

        val stepTextView = TextView(context).apply {
            id = ID_STEP_TEXT
            text = "Step 0"
            setTextColor(Color.WHITE)
            textSize = 12f
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            maxLines = 1
            maxWidth = maxTextWidthPx
            ellipsize = android.text.TextUtils.TruncateAt.END
        }
        textContainer.addView(stepTextView)

        val summaryTextView = TextView(context).apply {
            id = ID_SUMMARY_TEXT
            text = ""
            setTextColor(Color.parseColor("#B0BEC5"))
            textSize = 10f
            maxLines = 1
            maxWidth = maxTextWidthPx
            ellipsize = android.text.TextUtils.TruncateAt.END
            visibility = View.GONE
        }
        textContainer.addView(summaryTextView)
        expandedPanel.addView(textContainer)

        // Emerald Finish Button - Placed immediately after text so it is NEVER pushed off-screen
        val finishBtn = Button(context).apply {
            text = "✔ FINISH"
            textSize = 12f
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            setTextColor(Color.BLACK)
            val hPad = (10 * density).toInt()
            val vPad = (5 * density).toInt()
            setPadding(hPad, vPad, hPad, vPad)
            background = GradientDrawable().apply {
                shape = GradientDrawable.RECTANGLE
                cornerRadius = 14 * density
                setColor(Color.parseColor("#00E676")) // High-contrast neon emerald
            }
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                (32 * density).toInt()
            ).apply {
                marginEnd = (6 * density).toInt()
            }
            setOnClickListener {
                Log.i(TAG, "User tapped FINISH on DittoFloatingControl")
                onFinishClickListener?.invoke()
            }
        }
        expandedPanel.addView(finishBtn)

        // Pause / Resume Button
        val pauseBtn = Button(context).apply {
            id = ID_PAUSE_BTN
            text = "⏸"
            textSize = 12f
            setTextColor(Color.WHITE)
            val btnPad = (6 * density).toInt()
            setPadding(btnPad, btnPad, btnPad, btnPad)
            background = GradientDrawable().apply {
                shape = GradientDrawable.RECTANGLE
                cornerRadius = 14 * density
                setColor(Color.parseColor("#37474F"))
            }
            layoutParams = LinearLayout.LayoutParams((32 * density).toInt(), (32 * density).toInt()).apply {
                marginEnd = (6 * density).toInt()
            }
            setOnClickListener {
                isPaused = !isPaused
                text = if (isPaused) "▶" else "⏸"
                val dot = expandedPanel.findViewById<View>(ID_REC_DOT)
                dot?.setBackgroundColor(if (isPaused) Color.YELLOW else Color.parseColor("#FF1744"))
                onPauseToggleListener?.invoke(isPaused)
            }
        }
        expandedPanel.addView(pauseBtn)

        // Minimize / Collapse Button
        val collapseBtn = Button(context).apply {
            text = "−"
            textSize = 14f
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            setTextColor(Color.parseColor("#CFD8DC"))
            val btnPad = (4 * density).toInt()
            setPadding(btnPad, btnPad, btnPad, btnPad)
            background = GradientDrawable().apply {
                shape = GradientDrawable.RECTANGLE
                cornerRadius = 12 * density
                setColor(Color.parseColor("#263238"))
            }
            layoutParams = LinearLayout.LayoutParams((28 * density).toInt(), (28 * density).toInt())
            setOnClickListener {
                toggleCollapsedState(root)
            }
        }
        expandedPanel.addView(collapseBtn)

        root.addView(expandedPanel)

        // 2. Collapsed Tiny Pill (Contains quick-finish ✔ and expand ▶)
        val collapsedPill = LinearLayout(context).apply {
            id = ID_COLLAPSED_CONTAINER
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            visibility = View.GONE
            val padH = (8 * density).toInt()
            val padV = (6 * density).toInt()
            setPadding(padH, padV, padH, padV)
            background = GradientDrawable().apply {
                shape = GradientDrawable.RECTANGLE
                cornerRadius = 20 * density
                setColor(Color.parseColor("#EE10141E"))
                setStroke((1.5f * density).toInt(), Color.parseColor("#00E676"))
            }
            elevation = 12 * density

            val pillDot = View(context).apply {
                val dotSize = (8 * density).toInt()
                layoutParams = LinearLayout.LayoutParams(dotSize, dotSize).apply {
                    marginEnd = (6 * density).toInt()
                }
                background = GradientDrawable().apply {
                    shape = GradientDrawable.OVAL
                    setColor(Color.parseColor("#FF1744"))
                }
            }
            startPulseAnimation(pillDot)
            addView(pillDot)

            // Quick Finish in Pill
            val pillFinish = Button(context).apply {
                text = "✔"
                textSize = 11f
                typeface = android.graphics.Typeface.DEFAULT_BOLD
                setTextColor(Color.BLACK)
                setPadding(0, 0, 0, 0)
                background = GradientDrawable().apply {
                    shape = GradientDrawable.RECTANGLE
                    cornerRadius = 10 * density
                    setColor(Color.parseColor("#00E676"))
                }
                layoutParams = LinearLayout.LayoutParams((24 * density).toInt(), (24 * density).toInt()).apply {
                    marginEnd = (6 * density).toInt()
                }
                setOnClickListener {
                    Log.i(TAG, "User tapped pill FINISH on DittoFloatingControl")
                    onFinishClickListener?.invoke()
                }
            }
            addView(pillFinish)

            val pillLabel = TextView(context).apply {
                text = "▶"
                textSize = 11f
                setTextColor(Color.parseColor("#00E676"))
                typeface = android.graphics.Typeface.DEFAULT_BOLD
            }
            addView(pillLabel)

            setOnClickListener {
                toggleCollapsedState(root)
            }
        }

        root.addView(collapsedPill)

        // Touch Dragging with Safe Updates & Edge Snapping
        var initialX = 0
        var initialY = 0
        var initialTouchX = 0f
        var initialTouchY = 0f
        var isDragging = false

        root.setOnTouchListener { v, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    initialX = params.x
                    initialY = params.y
                    initialTouchX = event.rawX
                    initialTouchY = event.rawY
                    isDragging = false
                    false // Allow child buttons to click
                }
                MotionEvent.ACTION_MOVE -> {
                    val dx = (event.rawX - initialTouchX).toInt()
                    val dy = (event.rawY - initialTouchY).toInt()
                    if (abs(dx) > touchSlop || abs(dy) > touchSlop) {
                        isDragging = true
                        val screenWidth = context.resources.displayMetrics.widthPixels
                        val screenHeight = context.resources.displayMetrics.heightPixels
                        val viewW = if (v.width > 0) v.width else (260 * density).toInt()
                        val maxX = maxOf((8 * density).toInt(), screenWidth - viewW - (8 * density).toInt())
                        params.x = (initialX + dx).coerceIn((8 * density).toInt(), maxX)
                        params.y = (initialY + dy).coerceIn((40 * density).toInt(), screenHeight - (100 * density).toInt())
                        safeWmCall("updateViewLayout") {
                            windowManager?.updateViewLayout(v, params)
                        }
                        true
                    } else {
                        false
                    }
                }
                MotionEvent.ACTION_UP -> {
                    if (isDragging) {
                        val screenWidth = context.resources.displayMetrics.widthPixels
                        val targetX = if (params.x + v.width / 2 < screenWidth / 2) {
                            (8 * density).toInt()
                        } else {
                            screenWidth - v.width - (8 * density).toInt()
                        }
                        params.x = targetX
                        safeWmCall("updateViewLayout") {
                            windowManager?.updateViewLayout(v, params)
                        }
                        true
                    } else {
                        false
                    }
                }
                else -> false
            }
        }

        return root
    }

    private fun toggleCollapsedState(root: View) {
        isCollapsed = !isCollapsed
        val expanded = root.findViewById<View>(ID_EXPANDED_CONTAINER)
        val collapsed = root.findViewById<View>(ID_COLLAPSED_CONTAINER)
        if (isCollapsed) {
            expanded?.visibility = View.GONE
            collapsed?.visibility = View.VISIBLE
        } else {
            collapsed?.visibility = View.GONE
            expanded?.visibility = View.VISIBLE
        }
        safeWmCall("toggleCollapsedState") {
            windowManager?.updateViewLayout(root, layoutParams)
        }
    }

    private fun startPulseAnimation(view: View) {
        val anim = ValueAnimator.ofFloat(0.4f, 1.0f).apply {
            duration = 800
            repeatMode = ValueAnimator.REVERSE
            repeatCount = ValueAnimator.INFINITE
            addUpdateListener {
                view.alpha = it.animatedValue as Float
            }
        }
        anim.start()
    }
}
