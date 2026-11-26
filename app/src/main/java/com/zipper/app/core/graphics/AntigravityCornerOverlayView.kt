package com.zipper.app.core.graphics

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import android.view.animation.LinearInterpolator
import kotlin.math.sin

/**
 * Antigravity Corner & Edge Illumination Overlay View.
 * Renders the non-intrusive luminous cyber corners, Hoysala temple filigree,
 * and soft iridescent cloud wisps around the screen perimeter.
 *
 * ZERO SCREEN BLUR - ZERO TOUCH INTERCEPTION.
 */
class AntigravityCornerOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    enum class Intensity {
        IDLE,       // Very subtle ambient glow
        ACTIVE,     // Soft pulsing glow during listening/thinking
        EXECUTING   // Stronger luminous neon glow during active working
    }

    var currentIntensity: Intensity = Intensity.IDLE
        set(value) {
            field = value
            invalidate()
        }

    private var pulsePhase = 0.0f
    private val animator = ValueAnimator.ofFloat(0f, 1f).apply {
        duration = 3000L
        repeatCount = ValueAnimator.INFINITE
        interpolator = LinearInterpolator()
        addUpdateListener {
            pulsePhase = it.animatedFraction
            invalidate()
        }
    }

    // Paint Objects
    private val topCornerGlowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
    }

    private val bottomCornerGlowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
    }

    private val filigreePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 1.5f
    }

    private val cloudPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }

    init {
        animator.start()
    }

    override fun onAttachedToWindow() {
        super.onAttachedToWindow()
        if (!animator.isRunning) animator.start()
    }

    override fun onDetachedFromWindow() {
        super.onDetachedFromWindow()
        animator.cancel()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val w = width.toFloat()
        val h = height.toFloat()
        if (w <= 0f || h <= 0f) return

        val density = resources.displayMetrics.density
        val cornerRadius = 36f * density
        val strokeWidth = 3.5f * density
        val cornerArcLength = 120f * density

        val baseAlpha = when (currentIntensity) {
            Intensity.IDLE -> 0.25f + 0.08f * sin(pulsePhase * Math.PI.toFloat() * 2f)
            Intensity.ACTIVE -> 0.55f + 0.15f * sin(pulsePhase * Math.PI.toFloat() * 2f)
            Intensity.EXECUTING -> 0.85f + 0.12f * sin(pulsePhase * Math.PI.toFloat() * 4f)
        }

        // 1. TOP CORNERS: Electric Blue & Vibrant Cyan
        topCornerGlowPaint.strokeWidth = strokeWidth
        val cyanShader = LinearGradient(
            0f, 0f, cornerArcLength, cornerArcLength,
            Color.argb((baseAlpha * 255).toInt(), 0, 240, 255),
            Color.argb((baseAlpha * 120).toInt(), 37, 99, 235),
            Shader.TileMode.CLAMP
        )
        topCornerGlowPaint.shader = cyanShader

        // Top-Left Arc
        val tlPath = Path().apply {
            moveTo(0f, cornerArcLength)
            lineTo(0f, cornerRadius)
            quadTo(0f, 0f, cornerRadius, 0f)
            lineTo(cornerArcLength, 0f)
        }
        canvas.drawPath(tlPath, topCornerGlowPaint)

        // Top-Right Arc
        val trShader = LinearGradient(
            w, 0f, w - cornerArcLength, cornerArcLength,
            Color.argb((baseAlpha * 255).toInt(), 0, 240, 255),
            Color.argb((baseAlpha * 120).toInt(), 37, 99, 235),
            Shader.TileMode.CLAMP
        )
        topCornerGlowPaint.shader = trShader
        val trPath = Path().apply {
            moveTo(w, cornerArcLength)
            lineTo(w, cornerRadius)
            quadTo(w, 0f, w - cornerRadius, 0f)
            lineTo(w - cornerArcLength, 0f)
        }
        canvas.drawPath(trPath, topCornerGlowPaint)

        // 2. BOTTOM CORNERS: Iridescent Liquid Gold & Amber Copper
        bottomCornerGlowPaint.strokeWidth = strokeWidth
        val blShader = LinearGradient(
            0f, h, cornerArcLength, h - cornerArcLength,
            Color.argb((baseAlpha * 240).toInt(), 245, 158, 11),
            Color.argb((baseAlpha * 110).toInt(), 217, 119, 6),
            Shader.TileMode.CLAMP
        )
        bottomCornerGlowPaint.shader = blShader
        val blPath = Path().apply {
            moveTo(0f, h - cornerArcLength)
            lineTo(0f, h - cornerRadius)
            quadTo(0f, h, cornerRadius, h)
            lineTo(cornerArcLength, h)
        }
        canvas.drawPath(blPath, bottomCornerGlowPaint)

        // Bottom-Right Arc
        val brShader = LinearGradient(
            w, h, w - cornerArcLength, h - cornerArcLength,
            Color.argb((baseAlpha * 240).toInt(), 245, 158, 11),
            Color.argb((baseAlpha * 110).toInt(), 217, 119, 6),
            Shader.TileMode.CLAMP
        )
        bottomCornerGlowPaint.shader = brShader
        val brPath = Path().apply {
            moveTo(w, h - cornerArcLength)
            lineTo(w, h - cornerRadius)
            quadTo(w, h, w - cornerRadius, h)
            lineTo(w - cornerArcLength, h)
        }
        canvas.drawPath(brPath, bottomCornerGlowPaint)

        // 3. HOYSALA VINTAGE FILIGREE & SACRED YANTRA ARCS (Subtle inner geometry)
        filigreePaint.color = Color.argb((baseAlpha * 85).toInt(), 245, 200, 100)
        drawHoysalaCornerDetail(canvas, cornerRadius, 12f * density, 0f, 0f, 1f, 1f)
        drawHoysalaCornerDetail(canvas, cornerRadius, 12f * density, w, 0f, -1f, 1f)
        drawHoysalaCornerDetail(canvas, cornerRadius, 12f * density, 0f, h, 1f, -1f)
        drawHoysalaCornerDetail(canvas, cornerRadius, 12f * density, w, h, -1f, -1f)

        // 4. CLOUD-LIKE EDGES: Soft, airy translucent iridescent mist in corners
        val cloudRadius = 70f * density
        val cloudAlpha = (baseAlpha * 35).toInt().coerceIn(0, 255)
        cloudPaint.shader = RadialGradient(
            0f, 0f, cloudRadius,
            Color.argb(cloudAlpha, 0, 240, 255),
            Color.TRANSPARENT,
            Shader.TileMode.CLAMP
        )
        canvas.drawCircle(0f, 0f, cloudRadius, cloudPaint)

        cloudPaint.shader = RadialGradient(
            w, 0f, cloudRadius,
            Color.argb(cloudAlpha, 139, 92, 246),
            Color.TRANSPARENT,
            Shader.TileMode.CLAMP
        )
        canvas.drawCircle(w, 0f, cloudRadius, cloudPaint)

        cloudPaint.shader = RadialGradient(
            w, h, cloudRadius,
            Color.argb(cloudAlpha, 245, 158, 11),
            Color.TRANSPARENT,
            Shader.TileMode.CLAMP
        )
        canvas.drawCircle(w, h, cloudRadius, cloudPaint)
    }

    private fun drawHoysalaCornerDetail(
        canvas: Canvas,
        radius: Float,
        offset: Float,
        originX: Float,
        originY: Float,
        dirX: Float,
        dirY: Float
    ) {
        val path = Path().apply {
            val r1 = radius + offset
            val r2 = radius + offset * 1.8f
            // Yantra concentric balance arcs
            moveTo(originX + dirX * r1, originY)
            quadTo(originX + dirX * r1, originY + dirY * r1, originX, originY + dirY * r1)
            moveTo(originX + dirX * r2, originY)
            quadTo(originX + dirX * r2, originY + dirY * r2, originX, originY + dirY * r2)
        }
        canvas.drawPath(path, filigreePaint)
    }
}
