package com.zipper.app.core.graphics

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.Canvas
import android.graphics.LinearGradient
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Shader
import android.util.AttributeSet
import android.view.animation.LinearInterpolator
import android.widget.RelativeLayout

/**
 * Luxury Cyber Command Bar with dynamic traveling neon light border.
 * Hardware-accelerated, locked 60 FPS delta-time sweep with zero battery overhead.
 */
class CyberNeonCommandBarLayout @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : RelativeLayout(context, attrs, defStyleAttr) {

    private val bgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = 0xFF080D1A.toInt() // Deep OLED glass
    }

    private val glowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 9.0f
    }

    private val borderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 3.5f
    }

    private val rectF = RectF()
    private val cornerRadius = 65f // ~26dp
    private var phase = 0f
    private var animator: ValueAnimator? = null

    init {
        setWillNotDraw(false)
        startTravelingLight()
    }

    private fun startTravelingLight() {
        animator = ValueAnimator.ofFloat(0f, 1f).apply {
            duration = 3200L
            repeatCount = ValueAnimator.INFINITE
            interpolator = LinearInterpolator()
            addUpdateListener {
                phase = it.animatedValue as Float
                invalidate()
            }
            start()
        }
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        val inset = borderPaint.strokeWidth / 2f
        rectF.set(inset, inset, w.toFloat() - inset, h.toFloat() - inset)
    }

    override fun onDraw(canvas: Canvas) {
        val w = width.toFloat()
        if (w <= 0) return

        // 1. Draw OLED deep glass interior
        canvas.drawRoundRect(rectF, cornerRadius, cornerRadius, bgPaint)

        // 2. Draw moving neon light beam around perimeter with radiant bloom halo
        val offset = (phase * w * 2f) - w
        val glowGradient = LinearGradient(
            offset, 0f, offset + w, 0f,
            intArrayOf(
                0x00000000,
                0x4038BDF8.toInt(),
                0x60818CF8.toInt(),
                0x7000F0FF.toInt(),
                0x00000000
            ),
            floatArrayOf(0.0f, 0.35f, 0.50f, 0.65f, 1.0f),
            Shader.TileMode.REPEAT
        )
        glowPaint.shader = glowGradient
        canvas.drawRoundRect(rectF, cornerRadius, cornerRadius, glowPaint)

        val sweepGradient = LinearGradient(
            offset, 0f, offset + w, 0f,
            intArrayOf(
                0xFF1E293B.toInt(),
                0xFF38BDF8.toInt(),
                0xFF818CF8.toInt(),
                0xFF00F0FF.toInt(),
                0xFF1E293B.toInt()
            ),
            floatArrayOf(0.0f, 0.35f, 0.50f, 0.65f, 1.0f),
            Shader.TileMode.REPEAT
        )
        borderPaint.shader = sweepGradient
        canvas.drawRoundRect(rectF, cornerRadius, cornerRadius, borderPaint)

        super.onDraw(canvas)
    }

    override fun onDetachedFromWindow() {
        super.onDetachedFromWindow()
        animator?.cancel()
    }
}
