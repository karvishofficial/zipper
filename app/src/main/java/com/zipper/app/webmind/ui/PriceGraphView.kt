package com.zipper.app.webmind.ui

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import com.zipper.app.webmind.data.PriceHistoryPoint

class PriceGraphView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var points: List<PriceHistoryPoint> = emptyList()

    private val linePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#2563EB")
        strokeWidth = 6f
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
    }

    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }

    private val gridPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#F3F4F6")
        strokeWidth = 2f
        style = Paint.Style.STROKE
    }

    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#9CA3AF")
        textSize = 28f
        typeface = Typeface.create(Typeface.SANS_SERIF, Typeface.NORMAL)
    }

    private val pointDotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#2563EB")
        style = Paint.Style.FILL
    }

    private val pointDotOuterPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        style = Paint.Style.FILL
    }

    private val pinBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#111827")
        style = Paint.Style.FILL
    }

    private val pinTextTitlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 28f
        typeface = Typeface.create(Typeface.SANS_SERIF, Typeface.BOLD)
        textAlign = Paint.Align.CENTER
    }

    private val pinTextSubPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#D1D5DB")
        textSize = 22f
        typeface = Typeface.create(Typeface.SANS_SERIF, Typeface.NORMAL)
        textAlign = Paint.Align.CENTER
    }

    fun setData(newPoints: List<PriceHistoryPoint>) {
        points = newPoints
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (points.isEmpty()) return

        val paddingLeft = 120f
        val paddingRight = 180f
        val paddingTop = 60f
        val paddingBottom = 70f

        val graphWidth = width - paddingLeft - paddingRight
        val graphHeight = height - paddingTop - paddingBottom
        if (graphWidth <= 0 || graphHeight <= 0) return

        // 1. Draw horizontal grid lines and Y-axis labels
        val yLabels = listOf(
            Pair("₹80K", paddingTop),
            Pair("₹70K", paddingTop + graphHeight * 0.33f),
            Pair("₹65K", paddingTop + graphHeight * 0.66f),
            Pair("₹60K", paddingTop + graphHeight)
        )

        for ((label, yPos) in yLabels) {
            canvas.drawLine(paddingLeft, yPos, width - 40f, yPos, gridPaint)
            canvas.drawText(label, 20f, yPos + 10f, textPaint)
        }

        val minPrice = 58000.0
        val maxPrice = 82000.0

        val coords = points.mapIndexed { index, point ->
            val x = paddingLeft + (index.toFloat() / (points.size - 1).coerceAtLeast(1)) * graphWidth
            val normalized = ((point.price - minPrice) / (maxPrice - minPrice)).coerceIn(0.0, 1.0)
            val y = (paddingTop + graphHeight) - (normalized * graphHeight).toFloat()
            PointF(x, y)
        }

        // 2. Draw smooth Bezier curve
        val path = Path()
        val fillPath = Path()

        path.moveTo(coords[0].x, coords[0].y)
        fillPath.moveTo(coords[0].x, paddingTop + graphHeight)
        fillPath.lineTo(coords[0].x, coords[0].y)

        for (i in 0 until coords.size - 1) {
            val p0 = coords[i]
            val p1 = coords[i + 1]
            val controlX1 = (p0.x + p1.x) / 2f
            val controlY1 = p0.y
            val controlX2 = (p0.x + p1.x) / 2f
            val controlY2 = p1.y
            path.cubicTo(controlX1, controlY1, controlX2, controlY2, p1.x, p1.y)
            fillPath.cubicTo(controlX1, controlY1, controlX2, controlY2, p1.x, p1.y)
        }

        fillPath.lineTo(coords.last().x, paddingTop + graphHeight)
        fillPath.close()

        // Gradient fill
        fillPaint.shader = LinearGradient(
            0f, paddingTop, 0f, paddingTop + graphHeight,
            Color.parseColor("#402563EB"), Color.TRANSPARENT, Shader.TileMode.CLAMP
        )
        canvas.drawPath(fillPath, fillPaint)
        canvas.drawPath(path, linePaint)

        // 3. Draw X-axis labels
        for (i in points.indices) {
            val label = points[i].dateLabel
            val x = coords[i].x
            textPaint.textAlign = Paint.Align.CENTER
            canvas.drawText(label, x, height - 15f, textPaint)
        }

        // 4. Draw callout pin on last point
        val lastPoint = coords.last()
        canvas.drawCircle(lastPoint.x, lastPoint.y, 14f, pointDotOuterPaint)
        canvas.drawCircle(lastPoint.x, lastPoint.y, 8f, pointDotPaint)

        // Black Callout Badge
        val badgeW = 210f
        val badgeH = 88f
        val badgeLeft = (lastPoint.x - badgeW / 2f).coerceAtMost(width - badgeW - 10f)
        val badgeTop = lastPoint.y - badgeH - 24f

        val badgeRect = RectF(badgeLeft, badgeTop, badgeLeft + badgeW, badgeTop + badgeH)
        canvas.drawRoundRect(badgeRect, 18f, 18f, pinBgPaint)

        canvas.drawText("₹62,990", badgeRect.centerX(), badgeTop + 36f, pinTextTitlePaint)
        canvas.drawText("Oct 11, 2026", badgeRect.centerX(), badgeTop + 68f, pinTextSubPaint)
    }
}
