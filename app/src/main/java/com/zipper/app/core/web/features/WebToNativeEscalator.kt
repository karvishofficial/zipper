package com.zipper.app.core.web.features

import android.net.Uri
import java.util.Locale

data class NativeAppMapping(
    val domainPattern: String,
    val packageName: String,
    val appName: String,
    val schemePrefix: String
)

data class EscalationResult(
    val canEscalate: Boolean,
    val targetPackage: String?,
    val appName: String?,
    val intentUri: String?,
    val webUrl: String
)

/**
 * Web-to-Native App Escalator.
 * Converts e-commerce and media web URLs into exact native Android app deep-links,
 * allowing instant handoff to native apps with stored payment methods and app-only deals.
 */
object WebToNativeEscalator {

    val SUPPORTED_MAPPINGS = listOf(
        NativeAppMapping("flipkart.com", "com.flipkart.android", "Flipkart", "flipkart://dl/"),
        NativeAppMapping("swiggy.com", "in.swiggy.android", "Swiggy", "swiggy://"),
        NativeAppMapping("zomato.com", "com.application.zomato", "Zomato", "zomato://"),
        NativeAppMapping("amazon.in", "in.amazon.mShop.android.shopping", "Amazon", "com.amazon.mobile.shopping.web://"),
        NativeAppMapping("amazon.com", "com.amazon.mShop.android.shopping", "Amazon US", "com.amazon.mobile.shopping.web://"),
        NativeAppMapping("myntra.com", "com.myntra.android", "Myntra", "myntra://"),
        NativeAppMapping("youtube.com", "com.google.android.youtube", "YouTube", "vnd.youtube://"),
        NativeAppMapping("youtu.be", "com.google.android.youtube", "YouTube", "vnd.youtube://"),
        NativeAppMapping("zeptonow.com", "com.zepto.android", "Zepto", "zepto://"),
        NativeAppMapping("blinkit.com", "com.grofers.customerapp", "Blinkit", "blinkit://")
    )

    fun findMappingForUrl(url: String): NativeAppMapping? {
        val lower = url.lowercase(Locale.ROOT)
        return SUPPORTED_MAPPINGS.firstOrNull { mapping ->
            lower.contains(mapping.domainPattern)
        }
    }

    fun synthesizeDeepLink(webUrl: String, mapping: NativeAppMapping): String {
        val queryParamV = runCatching {
            val query = webUrl.substringAfter('?', "")
            query.split('&').firstOrNull { it.startsWith("v=") }?.substringAfter("v=")
        }.getOrNull()?.takeIf { it.isNotBlank() }

        val pathSegment = webUrl.substringBefore('?').substringAfterLast('/')

        val pathAndQuery = runCatching {
            webUrl.substringAfter("://").substringAfter('/', "")
        }.getOrDefault("")

        return when (mapping.packageName) {
            "com.flipkart.android" -> "flipkart://dl/$pathAndQuery"
            "in.swiggy.android" -> "swiggy://$pathAndQuery"
            "com.application.zomato" -> "zomato://$pathAndQuery"
            "com.google.android.youtube" -> {
                val videoId = queryParamV ?: pathSegment.takeIf { it.isNotBlank() && it != "watch" }
                if (!videoId.isNullOrBlank()) "vnd.youtube:$videoId" else "vnd.youtube://"
            }
            "in.amazon.mShop.android.shopping", "com.amazon.mShop.android.shopping" -> {
                // Direct universal link or intent scheme
                webUrl
            }
            else -> "${mapping.schemePrefix}$pathAndQuery"
        }
    }

    /**
     * Resolves whether the current URL can be escalated to an installed native Android app.
     * @param isPackageInstalled Lambda to query PackageManager on device (or mock in unit tests).
     */
    fun resolveEscalation(
        webUrl: String,
        isPackageInstalled: (packageName: String) -> Boolean
    ): EscalationResult {
        val mapping = findMappingForUrl(webUrl)
            ?: return EscalationResult(
                canEscalate = false,
                targetPackage = null,
                appName = null,
                intentUri = null,
                webUrl = webUrl
            )

        val installed = isPackageInstalled(mapping.packageName)
        if (!installed) {
            return EscalationResult(
                canEscalate = false,
                targetPackage = mapping.packageName,
                appName = mapping.appName,
                intentUri = null,
                webUrl = webUrl
            )
        }

        val deepLink = synthesizeDeepLink(webUrl, mapping)
        return EscalationResult(
            canEscalate = true,
            targetPackage = mapping.packageName,
            appName = mapping.appName,
            intentUri = deepLink,
            webUrl = webUrl
        )
    }
}
