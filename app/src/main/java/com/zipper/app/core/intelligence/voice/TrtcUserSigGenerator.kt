package com.zipper.app.core.intelligence.voice

import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.nio.charset.StandardCharsets
import java.util.zip.Deflater
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec

/**
 * TRTC UserSig Token Generator.
 * Computes standard Tencent TRTC UserSig signatures (TLS 2.0 HMAC-SHA256 with Deflate compression).
 */
object TrtcUserSigGenerator {

    /**
     * Generates a valid TRTC UserSig token for joining Tencent RTC rooms.
     *
     * @param sdkAppId Your TRTC application ID (e.g. 20049543)
     * @param secretKey Your TRTC SDKSecretKey
     * @param userId The user ID joining the room (e.g. "zipper_voice_twin" or "caller_manoj")
     * @param expireSeconds Validity duration in seconds (default 7 days)
     */
    fun genUserSig(
        sdkAppId: Long,
        secretKey: String,
        userId: String,
        expireSeconds: Long = 86400 * 7
    ): String {
        val currTime = System.currentTimeMillis() / 1000

        val baseString = "TLS.identifier:$userId\nTLS.sdkappid:$sdkAppId\nTLS.time:$currTime\nTLS.expire:$expireSeconds\n"
        val mac = Mac.getInstance("HmacSHA256")
        mac.init(SecretKeySpec(secretKey.toByteArray(StandardCharsets.UTF_8), "HmacSHA256"))
        val sigBytes = mac.doFinal(baseString.toByteArray(StandardCharsets.UTF_8))
        val base64Sig = encodeBase64(sigBytes)

        val doc = JSONObject().apply {
            put("TLS.ver", "2.0")
            put("TLS.identifier", userId)
            put("TLS.sdkappid", sdkAppId)
            put("TLS.expire", expireSeconds)
            put("TLS.time", currTime)
            put("TLS.sig", base64Sig)
        }

        val jsonBytes = doc.toString().toByteArray(StandardCharsets.UTF_8)
        val compressor = Deflater()
        compressor.setInput(jsonBytes)
        compressor.finish()

        val outputStream = ByteArrayOutputStream(jsonBytes.size)
        val buffer = ByteArray(1024)
        while (!compressor.finished()) {
            val count = compressor.deflate(buffer)
            outputStream.write(buffer, 0, count)
        }
        compressor.end()

        val compressedBytes = outputStream.toByteArray()
        val standardBase64 = encodeBase64(compressedBytes)

        // Convert to URL-safe Base64 as required by Tencent TRTC
        return standardBase64
            .replace('+', '*')
            .replace('/', '-')
            .replace('=', '_')
    }

    private fun encodeBase64(bytes: ByteArray): String {
        return try {
            val clazz = Class.forName("android.util.Base64")
            val method = clazz.getMethod("encodeToString", ByteArray::class.java, Int::class.javaPrimitiveType)
            (method.invoke(null, bytes, 2 /* NO_WRAP */) as String).trim()
        } catch (_: Throwable) {
            java.util.Base64.getEncoder().encodeToString(bytes)
        }
    }
}
