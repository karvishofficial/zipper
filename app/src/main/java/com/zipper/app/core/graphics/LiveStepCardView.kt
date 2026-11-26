package com.zipper.app.core.graphics

import android.content.Context
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.util.AttributeSet
import android.view.Gravity
import android.view.View
import android.widget.*

/**
 * Top Compact Live Step Indicator Card.
 * Houses the mini 3D liquid-metal core, live step counter, action description,
 * dual-tone gradient progress bar, and minimize toggle.
 */
class LiveStepCardView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : LinearLayout(context, attrs) {

    val core3dView: Zipper3DCoreView
    private val tvStepTitle: TextView
    private val tvActionDesc: TextView
    private val progressBar: ProgressBar
    private val btnClose: TextView

    var onMinimizeClickListener: (() -> Unit)? = null

    init {
        orientation = HORIZONTAL
        gravity = Gravity.CENTER_VERTICAL

        val density = resources.displayMetrics.density
        setPadding(
            (12 * density).toInt(),
            (8 * density).toInt(),
            (14 * density).toInt(),
            (8 * density).toInt()
        )

        // Glassmorphic Cyber Card Background
        background = GradientDrawable().apply {
            shape = GradientDrawable.RECTANGLE
            cornerRadius = 22 * density
            setColor(Color.parseColor("#E60B0F19")) // 90% opacity deep obsidian
            setStroke((1.5f * density).toInt(), Color.parseColor("#4D00F0FF")) // 30% cyan glow border
        }

        // 1. Mini 3D Core View
        val coreSize = (60 * density).toInt()
        core3dView = Zipper3DCoreView(context).apply {
            currentState = ZipperCoreState.WORKING
        }
        addView(core3dView, LayoutParams(coreSize, coreSize).apply {
            marginEnd = (12 * density).toInt()
        })

        // 2. Middle Text & Progress Layout
        val textContainer = LinearLayout(context).apply {
            orientation = VERTICAL
            layoutParams = LayoutParams(0, LayoutParams.WRAP_CONTENT, 1.0f)
        }

        tvStepTitle = TextView(context).apply {
            text = "Step 1 of 4"
            textSize = 12f
            setTextColor(Color.parseColor("#00F0FF"))
            typeface = android.graphics.Typeface.DEFAULT_BOLD
        }
        textContainer.addView(tvStepTitle)

        tvActionDesc = TextView(context).apply {
            text = "Executing action..."
            textSize = 13f
            setTextColor(Color.WHITE)
            maxLines = 1
            ellipsize = android.text.TextUtils.TruncateAt.END
        }
        textContainer.addView(tvActionDesc)

        progressBar = ProgressBar(context, null, android.R.attr.progressBarStyleHorizontal).apply {
            isIndeterminate = false
            max = 100
            progress = 30
            layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, (4 * density).toInt()).apply {
                topMargin = (4 * density).toInt()
            }
            progressDrawable = GradientDrawable().apply {
                shape = GradientDrawable.RECTANGLE
                cornerRadius = 2 * density
                setColor(Color.parseColor("#00F0FF"))
            }
        }
        textContainer.addView(progressBar)

        addView(textContainer)

        // 3. Minimize / Close Button
        btnClose = TextView(context).apply {
            text = "✕"
            textSize = 15f
            setTextColor(Color.parseColor("#94A3B8"))
            gravity = Gravity.CENTER
            setPadding((8 * density).toInt(), (4 * density).toInt(), 0, (4 * density).toInt())
            setOnClickListener {
                onMinimizeClickListener?.invoke()
            }
        }
        addView(btnClose)
    }

    fun updateProgress(stepIndex: Int, totalSteps: Int, actionText: String, coreState: ZipperCoreState = ZipperCoreState.WORKING) {
        post {
            tvStepTitle.text = if (totalSteps > 0) "Step $stepIndex of $totalSteps" else "Zipper Active"
            tvActionDesc.text = actionText
            core3dView.currentState = coreState
            val pct = if (totalSteps > 0) ((stepIndex.toFloat() / totalSteps.toFloat()) * 100).toInt() else 50
            progressBar.progress = pct.coerceIn(5, 100)
        }
    }

    fun setListeningMode(goalText: String) {
        post {
            tvStepTitle.text = "Zipper Listening..."
            tvActionDesc.text = goalText.ifBlank { "Speak naturally..." }
            core3dView.currentState = ZipperCoreState.LISTENING
            progressBar.isIndeterminate = true
        }
    }

    fun setThinkingMode(planText: String) {
        post {
            tvStepTitle.text = "Zipper Thinking..."
            tvActionDesc.text = planText.ifBlank { "Understanding request..." }
            core3dView.currentState = ZipperCoreState.THINKING
            progressBar.isIndeterminate = true
        }
    }

    fun setVerifyingMode(verifyText: String) {
        post {
            tvStepTitle.text = "Verifying..."
            tvActionDesc.text = verifyText
            core3dView.currentState = ZipperCoreState.VERIFYING
            progressBar.progress = 95
        }
    }

    fun setCompletedMode(summaryText: String) {
        post {
            tvStepTitle.text = "Goal Completed ✓"
            tvActionDesc.text = summaryText
            core3dView.currentState = ZipperCoreState.COMPLETED
            progressBar.progress = 100
        }
    }
}
