package com.zipper.app.core.graphics

import android.content.Context
import android.content.Intent
import android.graphics.PixelFormat
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import com.zipper.app.accessibility.KarvishAccessibilityService
import com.zipper.app.core.agent.AgentRuntimeService
import com.zipper.app.core.agent.AgentStepRecord

/**
 * LiveOverlayManager
 * Coordinates the full Antigravity Live Agent Overlay experience over target apps.
 * Manages the touch-safe corner illumination, floating 3D Cyber Bubble,
 * top step card, and expanded control panel.
 */
object LiveOverlayManager {

    private const val TAG = "LiveOverlayManager"

    private var windowManager: WindowManager? = null
    private var cornerOverlayView: AntigravityCornerOverlayView? = null
    private var topCardView: LiveStepCardView? = null
    private var bubble3dView: Zipper3DCoreView? = null
    private var controlSheetView: TaskControlSheetView? = null

    private var isOverlayActive = false
    private var isSheetExpanded = false
    private val mainHandler = Handler(Looper.getMainLooper())

    private var activeGoal: String = ""
    private var currentStepCount: Int = 1
    private var estimatedTotalSteps: Int = 4

    fun startExecutionOverlay(context: Context, goal: String) {
        activeGoal = goal
        currentStepCount = 1
        mainHandler.post {
            try {
                val a11y = KarvishAccessibilityService.instance
                val targetContext = a11y ?: context
                val wm = targetContext.getSystemService(Context.WINDOW_SERVICE) as? WindowManager
                windowManager = wm
                if (wm == null) return@post

                val overlayType = if (a11y != null) {
                    WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY
                } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
                } else {
                    @Suppress("DEPRECATION")
                    WindowManager.LayoutParams.TYPE_PHONE
                }

                val density = targetContext.resources.displayMetrics.density
                val screenWidth = targetContext.resources.displayMetrics.widthPixels
                val screenHeight = targetContext.resources.displayMetrics.heightPixels

                // Screen perimeter border / corner illumination disabled per user request
                // for a completely clean, borderless mobile screen experience.

                // WINDOW 2 & WINDOW 3 (Duplicate top card and bubble) are superseded by
                // the unified KarvishControlOverlay bottom thinking pill to eliminate duplicate overlays.
                if (com.zipper.app.accessibility.KarvishBubbleOverlay.isShowing) {
                    com.zipper.app.accessibility.KarvishBubbleOverlay.setCoreState(ZipperCoreState.WORKING)
                }

                isOverlayActive = true
                Log.i(TAG, "Live Agent Overlay successfully attached (corner glow active; HUD handled by KarvishControlOverlay)")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to attach Live Agent Overlay", e)
            }
        }
    }

    private fun setupDraggableBubble(
        bubble: View,
        params: WindowManager.LayoutParams,
        wm: WindowManager,
        context: Context
    ) {
        bubble.setOnTouchListener(object : View.OnTouchListener {
            private var initialX = 0
            private var initialY = 0
            private var initialTouchX = 0f
            private var initialTouchY = 0f
            private var isMoving = false
            private var downTime = 0L

            override fun onTouch(v: View, event: MotionEvent): Boolean {
                when (event.action) {
                    MotionEvent.ACTION_DOWN -> {
                        initialX = params.x
                        initialY = params.y
                        initialTouchX = event.rawX
                        initialTouchY = event.rawY
                        isMoving = false
                        downTime = System.currentTimeMillis()
                        return true
                    }
                    MotionEvent.ACTION_MOVE -> {
                        val dx = (event.rawX - initialTouchX).toInt()
                        val dy = (event.rawY - initialTouchY).toInt()
                        if (Math.abs(dx) > 10 || Math.abs(dy) > 10) {
                            isMoving = true
                            params.x = initialX + dx
                            params.y = initialY + dy
                            try { wm.updateViewLayout(bubble, params) } catch (_: Exception) {}
                        }
                        return true
                    }
                    MotionEvent.ACTION_UP -> {
                        val elapsed = System.currentTimeMillis() - downTime
                        if (!isMoving && elapsed < 300) {
                            toggleTaskControlSheet(context)
                        } else {
                            // Magnetic snap to screen edges
                            val screenWidth = context.resources.displayMetrics.widthPixels
                            params.x = if (params.x < screenWidth / 2) 16 else (screenWidth - bubble.width - 16)
                            try { wm.updateViewLayout(bubble, params) } catch (_: Exception) {}
                        }
                        return true
                    }
                }
                return false
            }
        })
    }

    private fun toggleTaskControlSheet(context: Context) {
        val wm = windowManager ?: return
        if (isSheetExpanded && controlSheetView != null) {
            try {
                wm.removeView(controlSheetView)
                controlSheetView = null
                isSheetExpanded = false
            } catch (_: Exception) {}
        } else {
            val a11y = KarvishAccessibilityService.instance
            val targetContext = a11y ?: context
            val overlayType = if (a11y != null) {
                WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY
            } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
            } else {
                @Suppress("DEPRECATION")
                WindowManager.LayoutParams.TYPE_PHONE
            }

            val density = targetContext.resources.displayMetrics.density
            val screenWidth = targetContext.resources.displayMetrics.widthPixels
            val sheet = TaskControlSheetView(targetContext).apply {
                val demoSteps = listOf(
                    StepItem("Open application", isCompleted = currentStepCount > 1, isCurrent = currentStepCount == 1),
                    StepItem("Locate target affordance", isCompleted = currentStepCount > 2, isCurrent = currentStepCount == 2),
                    StepItem("Execute task action", isCompleted = currentStepCount > 3, isCurrent = currentStepCount == 3),
                    StepItem("Verify completion outcome", isCompleted = currentStepCount > 4, isCurrent = currentStepCount >= 4)
                )
                updateSteps("Step $currentStepCount of $estimatedTotalSteps", demoSteps, ZipperCoreState.WORKING)
                onCollapseClickListener = { toggleTaskControlSheet(targetContext) }
                onStopClickListener = {
                    val stopIntent = Intent(targetContext, AgentRuntimeService::class.java).apply {
                        action = AgentRuntimeService.ACTION_STOP_AGENT
                    }
                    targetContext.startService(stopIntent)
                    dismissOverlay()
                }
            }

            val sheetParams = WindowManager.LayoutParams(
                (screenWidth * 0.92f).toInt(),
                WindowManager.LayoutParams.WRAP_CONTENT,
                overlayType,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                        WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
                PixelFormat.TRANSLUCENT
            ).apply {
                gravity = Gravity.BOTTOM or Gravity.CENTER_HORIZONTAL
                y = (40 * density).toInt()
            }
            wm.addView(sheet, sheetParams)
            controlSheetView = sheet
            isSheetExpanded = true
        }
    }

    private fun toggleTopCardVisibility() {
        topCardView?.visibility = if (topCardView?.visibility == View.VISIBLE) View.GONE else View.VISIBLE
    }

    fun onStepUpdated(turn: Int, actionText: String, coreState: ZipperCoreState = ZipperCoreState.WORKING) {
        currentStepCount = turn
        mainHandler.post {
            topCardView?.updateProgress(turn, estimatedTotalSteps, actionText, coreState)
            bubble3dView?.currentState = coreState
            controlSheetView?.core3dView?.currentState = coreState
        }
    }

    fun onGoalCompleted(summary: String) {
        mainHandler.post {
            topCardView?.setCompletedMode(summary)
            cornerOverlayView?.currentIntensity = AntigravityCornerOverlayView.Intensity.IDLE
            bubble3dView?.currentState = ZipperCoreState.COMPLETED
            com.zipper.app.accessibility.KarvishBubbleOverlay.setCoreState(ZipperCoreState.COMPLETED)

            // Smoothly auto-dismiss after 4 seconds
            mainHandler.postDelayed({
                dismissOverlay()
            }, 4000L)
        }
    }

    fun dismissOverlay() {
        mainHandler.post {
            val wm = windowManager ?: return@post
            try {
                topCardView?.core3dView?.pauseRendering()
                bubble3dView?.pauseRendering()
                controlSheetView?.core3dView?.pauseRendering()
                cornerOverlayView?.let { wm.removeView(it) }
                topCardView?.let { wm.removeView(it) }
                bubble3dView?.let { wm.removeView(it) }
                controlSheetView?.let { wm.removeView(it) }
                com.zipper.app.accessibility.KarvishBubbleOverlay.setCoreState(ZipperCoreState.IDLE)
            } catch (e: Exception) {
                Log.w(TAG, "Error cleaning up overlay views", e)
            } finally {
                cornerOverlayView = null
                topCardView = null
                bubble3dView = null
                controlSheetView = null
                isOverlayActive = false
                isSheetExpanded = false
            }
        }
    }
}
