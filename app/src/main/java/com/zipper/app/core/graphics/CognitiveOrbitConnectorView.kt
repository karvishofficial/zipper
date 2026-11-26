package com.zipper.app.core.graphics

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View

/**
 * High-performance hardware-accelerated cyber circuit connectors linking
 * the 4 cognitive loop badges to the center 3D energy pedestal.
 */
class CognitiveOrbitConnectorView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val linePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 1.5f * resources.displayMetrics.density
        color = Color.parseColor("#4438BDF8")
    }

    private val dotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.parseColor("#38BDF8")
    }

    private val dotGlowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.parseColor("#3338BDF8")
    }

    private val path = Path()

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat()
        val h = height.toFloat()
        if (w <= 0 || h <= 0) return

        val density = resources.displayMetrics.density
        val centerX = w / 2f
        val centerY = h * 0.65f

        val pedestalRadiusX = 75f * density
        val pedestalRadiusY = 28f * density

        // Badge Anchor Coordinates (relative to connector container)
        val leftX = w * 0.28f
        val rightX = w * 0.72f

        val topBadgeY = h * 0.24f
        val botBadgeY = h * 0.62f

        path.reset()

        // 1. Top-Left: Understand -> Platform
        val p1StartX = leftX
        val p1StartY = topBadgeY
        val p1EndX = centerX - pedestalRadiusX * 0.85f
        val p1EndY = centerY - pedestalRadiusY * 0.4f
        path.moveTo(p1StartX, p1StartY)
        path.cubicTo(
            p1StartX + 20f * density, p1StartY,
            p1EndX - 15f * density, p1EndY - 10f * density,
            p1EndX, p1EndY
        )

        // 2. Bottom-Left: Find -> Platform
        val p2StartX = leftX
        val p2StartY = botBadgeY
        val p2EndX = centerX - pedestalRadiusX * 0.7f
        val p2EndY = centerY + pedestalRadiusY * 0.5f
        path.moveTo(p2StartX, p2StartY)
        path.cubicTo(
            p2StartX + 18f * density, p2StartY,
            p2EndX - 15f * density, p2EndY + 8f * density,
            p2EndX, p2EndY
        )

        // 3. Top-Right: Plan -> Platform
        val p3StartX = rightX
        val p3StartY = topBadgeY
        val p3EndX = centerX + pedestalRadiusX * 0.85f
        val p3EndY = centerY - pedestalRadiusY * 0.4f
        path.moveTo(p3StartX, p3StartY)
        path.cubicTo(
            p3StartX - 20f * density, p3StartY,
            p3EndX + 15f * density, p3EndY - 10f * density,
            p3EndX, p3EndY
        )

        // 4. Bottom-Right: Verify -> Platform
        val p4StartX = rightX
        val p4StartY = botBadgeY
        val p4EndX = centerX + pedestalRadiusX * 0.7f
        val p4EndY = centerY + pedestalRadiusY * 0.5f
        path.moveTo(p4StartX, p4StartY)
        path.cubicTo(
            p4StartX - 18f * density, p4StartY,
            p4EndX + 15f * density, p4EndY + 8f * density,
            p4EndX, p4EndY
        )

        canvas.drawPath(path, linePaint)

        // Draw Glowing Connector Terminals
        val dotRadius = 2.5f * density
        val glowRadius = 5.5f * density

        drawGlowingDot(canvas, p1StartX, p1StartY, dotRadius, glowRadius)
        drawGlowingDot(canvas, p1EndX, p1EndY, dotRadius, glowRadius)

        drawGlowingDot(canvas, p2StartX, p2StartY, dotRadius, glowRadius)
        drawGlowingDot(canvas, p2EndX, p2EndY, dotRadius, glowRadius)

        drawGlowingDot(canvas, p3StartX, p3StartY, dotRadius, glowRadius)
        drawGlowingDot(canvas, p3EndX, p3EndY, dotRadius, glowRadius)

        drawGlowingDot(canvas, p4StartX, p4StartY, dotRadius, glowRadius)
        drawGlowingDot(canvas, p4EndX, p4EndY, dotRadius, glowRadius)
    }

    private fun drawGlowingDot(canvas: Canvas, x: Float, y: Float, radius: Float, glowRadius: Float) {
        canvas.drawCircle(x, y, glowRadius, dotGlowPaint)
        canvas.drawCircle(x, y, radius, dotPaint)
    }
}
