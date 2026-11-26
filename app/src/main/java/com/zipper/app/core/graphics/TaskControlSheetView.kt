package com.zipper.app.core.graphics

import android.content.Context
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.util.AttributeSet
import android.view.Gravity
import android.widget.*

/**
 * Step checklist item model.
 */
data class StepItem(
    val title: String,
    val isCompleted: Boolean = false,
    val isCurrent: Boolean = false
)

/**
 * Expanded Task Control Panel (User Controls Tray).
 * Opens when the user taps the floating Cyber Bubble.
 * Displays the real live milestone checklist, step progress, and Pause/Stop controls.
 */
class TaskControlSheetView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : LinearLayout(context, attrs) {

    val core3dView: Zipper3DCoreView
    private val tvSubtitle: TextView
    private val tvCurrentStep: TextView
    private val stepsLayout: LinearLayout
    private val btnPause: Button
    private val btnStop: Button

    var onPauseClickListener: ((Boolean) -> Unit)? = null
    var onStopClickListener: (() -> Unit)? = null
    var onCollapseClickListener: (() -> Unit)? = null

    private var isPaused = false

    init {
        orientation = VERTICAL
        val density = resources.displayMetrics.density
        setPadding((16 * density).toInt(), (16 * density).toInt(), (16 * density).toInt(), (16 * density).toInt())

        // Obsidian Glassmorphic Background with Cyan Rim
        background = GradientDrawable().apply {
            shape = GradientDrawable.RECTANGLE
            cornerRadius = 24 * density
            setColor(Color.parseColor("#F20B0F19")) // 95% obsidian
            setStroke((1.5f * density).toInt(), Color.parseColor("#4D00F0FF"))
        }

        // 1. Header Row
        val headerRow = LinearLayout(context).apply {
            orientation = HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }

        val coreSize = (38 * density).toInt()
        core3dView = Zipper3DCoreView(context).apply {
            currentState = ZipperCoreState.WORKING
        }
        headerRow.addView(core3dView, LayoutParams(coreSize, coreSize).apply {
            marginEnd = (10 * density).toInt()
        })

        val titleContainer = LinearLayout(context).apply {
            orientation = VERTICAL
            layoutParams = LayoutParams(0, LayoutParams.WRAP_CONTENT, 1.0f)
        }
        val tvTitle = TextView(context).apply {
            text = "Zipper"
            textSize = 15f
            setTextColor(Color.WHITE)
            typeface = android.graphics.Typeface.DEFAULT_BOLD
        }
        tvSubtitle = TextView(context).apply {
            text = "Working on your request..."
            textSize = 12f
            setTextColor(Color.parseColor("#94A3B8"))
        }
        titleContainer.addView(tvTitle)
        titleContainer.addView(tvSubtitle)
        headerRow.addView(titleContainer)

        val btnCollapse = TextView(context).apply {
            text = "⌄"
            textSize = 22f
            setTextColor(Color.parseColor("#94A3B8"))
            setPadding((8 * density).toInt(), 0, (4 * density).toInt(), 0)
            setOnClickListener { onCollapseClickListener?.invoke() }
        }
        headerRow.addView(btnCollapse)
        addView(headerRow)

        // 2. Current Step Subheader
        tvCurrentStep = TextView(context).apply {
            text = "Step 2 of 4: Executing..."
            textSize = 13f
            setTextColor(Color.parseColor("#00F0FF"))
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            setPadding(0, (12 * density).toInt(), 0, (8 * density).toInt())
        }
        addView(tvCurrentStep)

        // 3. Dynamic Steps Container
        stepsLayout = LinearLayout(context).apply {
            orientation = VERTICAL
            setPadding(0, (4 * density).toInt(), 0, (14 * density).toInt())
        }
        addView(stepsLayout)

        // 4. Action Buttons (Pause & Stop)
        val buttonRow = LinearLayout(context).apply {
            orientation = HORIZONTAL
        }

        btnPause = Button(context).apply {
            text = "⏸ Pause"
            textSize = 13f
            setTextColor(Color.WHITE)
            background = GradientDrawable().apply {
                shape = GradientDrawable.RECTANGLE
                cornerRadius = 14 * density
                setColor(Color.parseColor("#1E293B"))
                setStroke((1f * density).toInt(), Color.parseColor("#334155"))
            }
            layoutParams = LayoutParams(0, (42 * density).toInt(), 1.0f).apply {
                marginEnd = (8 * density).toInt()
            }
            setOnClickListener {
                isPaused = !isPaused
                text = if (isPaused) "▶ Resume" else "⏸ Pause"
                onPauseClickListener?.invoke(isPaused)
            }
        }
        buttonRow.addView(btnPause)

        btnStop = Button(context).apply {
            text = "🛑 Stop"
            textSize = 13f
            setTextColor(Color.WHITE)
            background = GradientDrawable().apply {
                shape = GradientDrawable.RECTANGLE
                cornerRadius = 14 * density
                setColor(Color.parseColor("#450A0A"))
                setStroke((1f * density).toInt(), Color.parseColor("#DC2626"))
            }
            layoutParams = LayoutParams(0, (42 * density).toInt(), 1.0f)
            setOnClickListener {
                onStopClickListener?.invoke()
            }
        }
        buttonRow.addView(btnStop)

        addView(buttonRow)
    }

    fun updateSteps(stepTitle: String, steps: List<StepItem>, coreState: ZipperCoreState = ZipperCoreState.WORKING) {
        post {
            tvCurrentStep.text = stepTitle
            core3dView.currentState = coreState
            stepsLayout.removeAllViews()

            val density = resources.displayMetrics.density
            for (step in steps) {
                val row = LinearLayout(context).apply {
                    orientation = HORIZONTAL
                    gravity = Gravity.CENTER_VERTICAL
                    setPadding(0, (4 * density).toInt(), 0, (4 * density).toInt())
                }

                val icon = TextView(context).apply {
                    text = when {
                        step.isCompleted -> "✓ "
                        step.isCurrent -> "⊙ "
                        else -> "○ "
                    }
                    textSize = 13f
                    setTextColor(
                        when {
                            step.isCompleted -> Color.parseColor("#34D399")
                            step.isCurrent -> Color.parseColor("#00F0FF")
                            else -> Color.parseColor("#64748B")
                        }
                    )
                }

                val label = TextView(context).apply {
                    text = step.title
                    textSize = 13f
                    setTextColor(
                        when {
                            step.isCompleted -> Color.parseColor("#94A3B8")
                            step.isCurrent -> Color.WHITE
                            else -> Color.parseColor("#64748B")
                        }
                    )
                }

                row.addView(icon)
                row.addView(label)
                stepsLayout.addView(row)
            }
        }
    }
}
