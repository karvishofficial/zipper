package com.zipper.app.core.web.features

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import java.io.ByteArrayOutputStream
import java.util.Locale

data class UploadConstraint(
    val minSizeBytes: Long = 0L,
    val maxSizeBytes: Long,
    val targetMimeType: String = "image/jpeg",
    val maxDimensionPx: Int = 1000
)

data class ResizeResult(
    val compressedBytes: ByteArray,
    val originalSizeBytes: Int,
    val finalSizeBytes: Int,
    val width: Int,
    val height: Int,
    val meetsConstraint: Boolean
)

/**
 * Universal Government & Exam Portal Assistant.
 * Auto-detects upload restrictions (e.g. UPSC/SSC/GATE 20KB-50KB limits)
 * and resizes/compresses photos and signatures to guarantee portal acceptance.
 */
object WebPortalAssistant {

    val DEFAULT_PHOTO_CONSTRAINT = UploadConstraint(
        minSizeBytes = 20 * 1024L,
        maxSizeBytes = 50 * 1024L,
        targetMimeType = "image/jpeg",
        maxDimensionPx = 600
    )

    val DEFAULT_SIGNATURE_CONSTRAINT = UploadConstraint(
        minSizeBytes = 10 * 1024L,
        maxSizeBytes = 20 * 1024L,
        targetMimeType = "image/jpeg",
        maxDimensionPx = 400
    )

    fun parseConstraintsFromPage(pageText: String): UploadConstraint {
        val lower = pageText.lowercase(Locale.ROOT)

        // Check for KB mentions: "between 20 kb and 50 kb", "max 50kb", "not exceeding 100kb"
        val rangeRegex = Regex("""(?:between|from)\s*([0-9]+)\s*(?:kb)?\s*(?:to|and)\s*([0-9]+)\s*kb""", RegexOption.IGNORE_CASE)
        val rangeMatch = rangeRegex.find(lower)
        if (rangeMatch != null) {
            val minKb = rangeMatch.groupValues[1].toLongOrNull() ?: 20L
            val maxKb = rangeMatch.groupValues[2].toLongOrNull() ?: 50L
            return UploadConstraint(
                minSizeBytes = minKb * 1024L,
                maxSizeBytes = maxKb * 1024L
            )
        }

        val maxRegex = Regex("""(?:max|maximum|under|less than|not exceeding)\s*([0-9]+)\s*kb""", RegexOption.IGNORE_CASE)
        val maxMatch = maxRegex.find(lower)
        if (maxMatch != null) {
            val maxKb = maxMatch.groupValues[1].toLongOrNull() ?: 50L
            return UploadConstraint(
                minSizeBytes = 0L,
                maxSizeBytes = maxKb * 1024L
            )
        }

        return if (lower.contains("signature")) {
            DEFAULT_SIGNATURE_CONSTRAINT
        } else {
            DEFAULT_PHOTO_CONSTRAINT
        }
    }

    /**
     * Resizes and compresses an image byte array to strictly conform to upload constraints.
     */
    fun resizeAndCompressImage(
        originalBytes: ByteArray,
        constraint: UploadConstraint
    ): ResizeResult {
        val originalSize = originalBytes.size
        val bitmap = BitmapFactory.decodeByteArray(originalBytes, 0, originalSize)
            ?: return ResizeResult(
                compressedBytes = originalBytes,
                originalSizeBytes = originalSize,
                finalSizeBytes = originalSize,
                width = 0,
                height = 0,
                meetsConstraint = false
            )

        // 1. Scale down if dimensions exceed maxDimensionPx
        var width = bitmap.width
        var height = bitmap.height
        val maxDim = constraint.maxDimensionPx
        val scaledBitmap = if (width > maxDim || height > maxDim) {
            val scale = maxDim.toFloat() / maxOf(width, height)
            width = (width * scale).toInt()
            height = (height * scale).toInt()
            Bitmap.createScaledBitmap(bitmap, width, height, true)
        } else {
            bitmap
        }

        // 2. Binary search for JPEG quality to fit under maxSizeBytes
        var quality = 90
        var outBytes: ByteArray = originalBytes
        val targetMax = constraint.maxSizeBytes

        for (q in listOf(90, 80, 65, 50, 35, 20)) {
            val baos = ByteArrayOutputStream()
            scaledBitmap.compress(Bitmap.CompressFormat.JPEG, q, baos)
            outBytes = baos.toByteArray()
            quality = q
            if (outBytes.size <= targetMax) {
                break
            }
        }

        return ResizeResult(
            compressedBytes = outBytes,
            originalSizeBytes = originalSize,
            finalSizeBytes = outBytes.size,
            width = width,
            height = height,
            meetsConstraint = outBytes.size <= constraint.maxSizeBytes
        )
    }
}
