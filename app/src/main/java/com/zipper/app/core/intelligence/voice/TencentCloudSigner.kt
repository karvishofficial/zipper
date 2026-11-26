package com.zipper.app.core.intelligence.voice

import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec

/**
 * Tencent Cloud API 3.0 (TC3-HMAC-SHA256) Signature Generator.
 * Used for secure REST communication with Tencent Cloud TRTC server APIs.
 */
object TencentCloudSigner {

    private const val ALGORITHM = "TC3-HMAC-SHA256"

    fun sha256Hex(data: ByteArray): String {
        val md = MessageDigest.getInstance("SHA-256")
        return bytesToHex(md.digest(data))
    }

    fun sha256Hex(str: String): String = sha256Hex(str.toByteArray(StandardCharsets.UTF_8))

    fun hmacSha256(key: ByteArray, data: String): ByteArray {
        val mac = Mac.getInstance("HmacSHA256")
        mac.init(SecretKeySpec(key, "HmacSHA256"))
        return mac.doFinal(data.toByteArray(StandardCharsets.UTF_8))
    }

    fun bytesToHex(bytes: ByteArray): String {
        val hexChars = "0123456789abcdef"
        val result = StringBuilder(bytes.size * 2)
        for (b in bytes) {
            val i = b.toInt() and 0xff
            result.append(hexChars[i shr 4])
            result.append(hexChars[i and 0x0f])
        }
        return result.toString()
    }

    /**
     * Signs a Tencent Cloud API 3.0 request and returns all required HTTP headers.
     */
    fun signRequest(
        secretId: String,
        secretKey: String,
        service: String, // e.g. "trtc"
        host: String,    // e.g. "trtc.intl.tencentcloudapi.com"
        action: String,  // e.g. "StartAIConversation"
        version: String, // e.g. "2019-07-22"
        region: String,  // e.g. "ap-singapore"
        payloadJson: String,
        timestampSeconds: Long = System.currentTimeMillis() / 1000
    ): Map<String, String> {
        val sdf = SimpleDateFormat("yyyy-MM-dd", Locale.US).apply {
            timeZone = TimeZone.getTimeZone("UTC")
        }
        val date = sdf.format(Date(timestampSeconds * 1000))

        val canonicalHeaders = "content-type:application/json; charset=utf-8\nhost:$host\nx-tc-action:${action.lowercase(Locale.US)}\n"
        val signedHeaders = "content-type;host;x-tc-action"
        val hashedPayload = sha256Hex(payloadJson)

        val canonicalRequest = "POST\n/\n\n$canonicalHeaders\n$signedHeaders\n$hashedPayload"

        val credentialScope = "$date/$service/tc3_request"
        val hashedCanonicalRequest = sha256Hex(canonicalRequest)
        val stringToSign = "$ALGORITHM\n$timestampSeconds\n$credentialScope\n$hashedCanonicalRequest"

        val kDate = hmacSha256(("TC3$secretKey").toByteArray(StandardCharsets.UTF_8), date)
        val kService = hmacSha256(kDate, service)
        val kSigning = hmacSha256(kService, "tc3_request")
        val signatureBytes = hmacSha256(kSigning, stringToSign)
        val signature = bytesToHex(signatureBytes)

        val authorization = "$ALGORITHM Credential=$secretId/$credentialScope, SignedHeaders=$signedHeaders, Signature=$signature"

        return mapOf(
            "Host" to host,
            "Content-Type" to "application/json; charset=utf-8",
            "X-TC-Action" to action,
            "X-TC-Version" to version,
            "X-TC-Timestamp" to timestampSeconds.toString(),
            "X-TC-Region" to region,
            "Authorization" to authorization
        )
    }
}
