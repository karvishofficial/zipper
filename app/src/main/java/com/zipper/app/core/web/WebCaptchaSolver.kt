package com.zipper.app.core.web

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Base64
import android.util.Log
import com.google.android.gms.tasks.Task
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import org.json.JSONObject
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/**
 * Universal Web Alphanumeric CAPTCHA Solver.
 * Detects captcha image & input element pairings in the DOM,
 * decodes base64/canvas pixels, and runs on-device Google ML Kit OCR
 * with sanitization and validation.
 */
class WebCaptchaSolver {

    companion object {
        private const val TAG = "WebCaptchaSolver"
    }

    private val recognizer by lazy {
        TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
    }

    data class CaptchaInfo(
        val inputKarvishId: String,
        val base64Data: String
    )

    /**
     * Inspects the active WebView DOM for any security code / captcha input and paired image.
     */
    suspend fun detectCaptcha(controller: WebController): CaptchaInfo? {
        if (controller !is HiddenWebController) return null
        return withContext(Dispatchers.Main) {
            val script = """
            (function() {
                try {
                    // 1. Find the captcha text input field
                    const inputs = Array.from(document.querySelectorAll('input[type="text"], input:not([type])'));
                    let input = inputs.find(el => {
                        const id = (el.id || '').toLowerCase();
                        const name = (el.name || '').toLowerCase();
                        const ph = (el.placeholder || '').toLowerCase();
                        const aria = (el.getAttribute('aria-label') || '').toLowerCase();
                        return id.includes('captcha') || name.includes('captcha') || ph.includes('captcha') ||
                               ph.includes('code') || aria.includes('captcha') || id.includes('security') || name.includes('security');
                    });

                    if (!input) return JSON.stringify({ found: false });

                    // Ensure karvish id
                    let inputId = input.id || input.name || input.getAttribute('data-karvish-id');
                    if (!inputId) {
                        inputId = 'k_cap_input';
                        input.setAttribute('data-karvish-id', inputId);
                    }

                    // 2. Find associated image
                    let img = document.getElementById('cn-cap-img')
                           || document.querySelector('img[src*="captcha" i], img[id*="captcha" i], img[id*="cap" i], img[alt*="captcha" i]');

                    if (!img) {
                        const container = input.closest('form, div, tr, p') || document.body;
                        img = container.querySelector('img');
                    }

                    if (!img) return JSON.stringify({ found: false });

                    // Auto-trigger tap if captcha requires initial click/load
                    if (img.naturalWidth === 0 || !img.src || img.src.length < 50) {
                        if (typeof refreshCaptcha === 'function') refreshCaptcha();
                        else img.click();
                    }

                    let src = img.src || '';
                    if (src.startsWith('data:image/')) {
                        return JSON.stringify({ found: true, inputId: inputId, base64: src });
                    }

                    // Render to canvas if external URL
                    const canvas = document.createElement('canvas');
                    canvas.width = img.naturalWidth || img.width || 140;
                    canvas.height = img.naturalHeight || img.height || 45;
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(img, 0, 0);
                    const canvasSrc = canvas.toDataURL('image/jpeg');

                    return JSON.stringify({
                        found: true,
                        inputId: inputId,
                        base64: canvasSrc
                    });
                } catch(e) {
                    return JSON.stringify({ found: false, error: e.message });
                }
            })()
        """.trimIndent()

            val wv = controller.getWebView() ?: return@withContext null
            val rawResult = suspendCancellableCoroutine<String> { continuation ->
                wv.evaluateJavascript(script) { res ->
                    if (continuation.isActive) continuation.resume(res ?: "")
                }
            }
            parseDetectionResult(rawResult)
        }
    }

    /**
     * Triggers a refresh of the web CAPTCHA image via DOM click or native function call.
     */
    suspend fun refreshCaptcha(controller: WebController): Boolean {
        if (controller !is HiddenWebController) return false
        return withContext(Dispatchers.Main) {
            val script = """
            (function() {
                try {
                    const btn = document.querySelector('.cn-cap-refresh')
                        || document.querySelector('button[title*="refresh" i], [onclick*="refreshCaptcha"]')
                        || document.getElementById('cn-cap-img');
                    if (btn) {
                        btn.click();
                        return 'true';
                    }
                    if (typeof refreshCaptcha === 'function') {
                        refreshCaptcha();
                        return 'true';
                    }
                    return 'false';
                } catch(e) {
                    return 'error:' + e.message;
                }
            })()
        """.trimIndent()

            val wv = controller.getWebView() ?: return@withContext false
            val res = suspendCancellableCoroutine<String> { continuation ->
                wv.evaluateJavascript(script) { r ->
                    if (continuation.isActive) continuation.resume(r ?: "")
                }
            }
            res.contains("true")
        }
    }

    private fun parseDetectionResult(raw: String): CaptchaInfo? {
        val clean = raw.trim().removeSurrounding("\"")
            .replace("\\\"", "\"")
            .replace("\\\\", "\\")
        return try {
            val json = JSONObject(clean)
            if (json.optBoolean("found", false)) {
                val inputId = json.optString("inputId")
                val b64 = json.optString("base64")
                if (inputId.isNotBlank() && b64.isNotBlank()) {
                    CaptchaInfo(inputId, b64)
                } else null
            } else null
        } catch (_: Exception) {
            null
        }
    }

    /**
     * Solves the given base64 captcha image using on-device ML Kit OCR.
     */
    suspend fun solve(base64Data: String): String? = withContext(Dispatchers.IO) {
        try {
            val cleanBase64 = if (base64Data.contains(",")) {
                base64Data.substringAfter(",")
            } else {
                base64Data
            }

            val decodedBytes = Base64.decode(cleanBase64, Base64.DEFAULT)
            val bitmap = BitmapFactory.decodeByteArray(decodedBytes, 0, decodedBytes.size)
                ?: return@withContext null
            solveBitmap(bitmap)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to solve captcha via ML Kit", e)
            null
        }
    }

    /**
     * Solves a raw Bitmap captcha using on-device ML Kit OCR.
     */
    suspend fun solveBitmap(bitmap: Bitmap): String? = withContext(Dispatchers.IO) {
        try {
            // Upscale 2.5x to improve ML Kit OCR clarity on small CAPTCHA fonts
            val scaledBitmap = Bitmap.createScaledBitmap(
                bitmap,
                (bitmap.width * 2.5).toInt(),
                (bitmap.height * 2.5).toInt(),
                true
            )

            val inputImage = InputImage.fromBitmap(scaledBitmap, 0)
            val visionText = recognizer.process(inputImage).await()

            val rawChars = visionText.text.replace(Regex("[^A-Za-z0-9]"), "").trim()
            Log.i(TAG, "ML Kit detected captcha raw text: '${visionText.text}' -> sanitized: '$rawChars'")

            if (rawChars.length in 4..6) {
                rawChars
            } else {
                Log.w(TAG, "Detected captcha text length (${rawChars.length}) not in standard 4-6 range: '$rawChars'")
                rawChars.takeIf { it.isNotBlank() }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to solve bitmap captcha via ML Kit", e)
            null
        }
    }

    private suspend fun <T> Task<T>.await(): T = suspendCancellableCoroutine { cont ->
        addOnSuccessListener { result -> cont.resume(result) }
        addOnFailureListener { ex -> cont.resumeWithException(ex) }
    }
}
