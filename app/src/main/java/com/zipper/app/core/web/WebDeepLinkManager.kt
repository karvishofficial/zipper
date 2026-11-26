package com.zipper.app.core.web

import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.util.Log

/**
 * Universal Deep-Link and App Handoff Manager for Karvish Web Agent.
 * Dispatches canonical product and web URLs directly into native installed applications
 * (Amazon, Flipkart, Zepto, Blinkit, etc.) or falls back to system browser.
 */
object WebDeepLinkManager {

    private const val TAG = "WebDeepLinkManager"

    // Supported Native App Packages
    const val PKG_AMAZON_IN = "in.amazon.mShop.android.shopping"
    const val PKG_AMAZON_GLOBAL = "com.amazon.mShop.android.shopping"
    const val PKG_FLIPKART = "com.flipkart.android"
    const val PKG_ZEPTO = "com.zeptoconsumerapp"
    const val PKG_ZEPTO_ALT = "com.zepto.consumer"
    const val PKG_BLINKIT = "com.grofers.customerapp"
    const val PKG_MYNTRA = "com.myntra.android"

    fun isPackageInstalled(context: Context, packageName: String): Boolean {
        return try {
            context.packageManager.getPackageInfo(packageName, 0)
            true
        } catch (_: PackageManager.NameNotFoundException) {
            false
        }
    }

    /**
     * Resolves the primary native app package for a given URL or query.
     */
    fun resolveAppPackage(url: String?, query: String? = null, context: Context? = null): String? {
        val u = (url ?: "").lowercase()
        val q = (query ?: "").lowercase()

        return when {
            u.contains("amazon") || q.contains("amazon") -> {
                if (context != null) {
                    if (isPackageInstalled(context, PKG_AMAZON_IN)) PKG_AMAZON_IN
                    else if (isPackageInstalled(context, PKG_AMAZON_GLOBAL)) PKG_AMAZON_GLOBAL
                    else PKG_AMAZON_IN
                } else {
                    PKG_AMAZON_GLOBAL
                }
            }
            u.contains("flipkart") || q.contains("flipkart") -> PKG_FLIPKART
            u.contains("zepto") || q.contains("zepto") -> {
                if (context != null && !isPackageInstalled(context, PKG_ZEPTO) && isPackageInstalled(context, PKG_ZEPTO_ALT)) {
                    PKG_ZEPTO_ALT
                } else PKG_ZEPTO
            }
            u.contains("blinkit") || q.contains("blinkit") || q.contains("blanket") || q.contains("blenkit") || u.contains("grofers") -> PKG_BLINKIT
            u.contains("myntra") || q.contains("myntra") -> PKG_MYNTRA
            else -> null
        }
    }

    /**
     * Returns a human-friendly store brand name for badges and buttons.
     */
    fun resolveStoreName(url: String?, appPackage: String? = null): String {
        val u = (url ?: "").lowercase()
        val p = (appPackage ?: "").lowercase()
        return when {
            p.contains("amazon") || u.contains("amazon") -> "Amazon"
            p.contains("flipkart") || u.contains("flipkart") -> "Flipkart"
            p.contains("zepto") || u.contains("zepto") -> "Zepto"
            p.contains("blinkit") || p.contains("grofers") || u.contains("blinkit") -> "Blinkit"
            p.contains("myntra") || u.contains("myntra") -> "Myntra"
            else -> "Store"
        }
    }

    /**
     * Builds canonical search URLs across all major e-commerce platforms.
     */
    fun buildStoreSearchUrl(store: String, query: String, forApp: Boolean = false): String {
        val encoded = try {
            java.net.URLEncoder.encode(query.trim(), "UTF-8")
        } catch (_: Exception) {
            query.trim()
        }
        val s = store.lowercase()
        return when {
            s.contains("blinkit") || s.contains("grofers") -> "https://blinkit.com/s/?q=$encoded"
            s.contains("zepto") -> if (forApp) "zepto://zeptonow.com/app/search?query=$encoded" else "https://www.zeptonow.com/search?q=$encoded"
            s.contains("amazon") -> "https://www.amazon.in/s?k=$encoded"
            s.contains("flipkart") -> "https://www.flipkart.com/search?q=$encoded"
            s.contains("myntra") -> "https://www.myntra.com/$encoded"
            else -> "https://www.google.com/search?q=$encoded"
        }
    }

    /**
     * Resolves the installed native package for a designated store brand.
     */
    fun resolveInstalledPackageForStore(store: String, context: Context): String? {
        val s = store.lowercase()
        return when {
            s.contains("blinkit") || s.contains("grofers") -> {
                if (isPackageInstalled(context, PKG_BLINKIT)) PKG_BLINKIT else null
            }
            s.contains("zepto") -> {
                if (isPackageInstalled(context, PKG_ZEPTO)) PKG_ZEPTO
                else if (isPackageInstalled(context, PKG_ZEPTO_ALT)) PKG_ZEPTO_ALT
                else null
            }
            s.contains("amazon") -> {
                if (isPackageInstalled(context, PKG_AMAZON_IN)) PKG_AMAZON_IN
                else if (isPackageInstalled(context, PKG_AMAZON_GLOBAL)) PKG_AMAZON_GLOBAL
                else null
            }
            s.contains("flipkart") -> {
                if (isPackageInstalled(context, PKG_FLIPKART)) PKG_FLIPKART else null
            }
            s.contains("myntra") -> {
                if (isPackageInstalled(context, PKG_MYNTRA)) PKG_MYNTRA else null
            }
            else -> null
        }
    }

    /**
     * Auto-detects the best store platform based on user intent and installed applications on device.
     */
    fun resolveBestStoreForQuery(query: String, multiItems: List<String>, context: Context): String {
        val q = query.lowercase()
        if (q.contains("zepto") || q.contains("zept") || q.contains("zzepto")) return "Zepto"
        if (q.contains("blinkit") || q.contains("blibkit") || q.contains("blikit") || q.contains("blanket") || q.contains("blenkit") || q.contains("grofers")) return "Blinkit"
        if (q.contains("amazon") || q.contains("amzn")) return "Amazon"
        if (q.contains("flipkart") || q.contains("fk")) return "Flipkart"
        if (q.contains("myntra")) return "Myntra"

        val hasGrocery = multiItems.any { WebProductFinder.isGroceryQuery(it) } || WebProductFinder.isGroceryQuery(query)
        return if (hasGrocery) {
            when {
                isPackageInstalled(context, PKG_BLINKIT) -> "Blinkit"
                isPackageInstalled(context, PKG_ZEPTO) || isPackageInstalled(context, PKG_ZEPTO_ALT) -> "Zepto"
                isPackageInstalled(context, PKG_AMAZON_IN) || isPackageInstalled(context, PKG_AMAZON_GLOBAL) -> "Amazon"
                else -> "Blinkit"
            }
        } else {
            when {
                isPackageInstalled(context, PKG_AMAZON_IN) || isPackageInstalled(context, PKG_AMAZON_GLOBAL) -> "Amazon"
                isPackageInstalled(context, PKG_FLIPKART) -> "Flipkart"
                isPackageInstalled(context, PKG_BLINKIT) -> "Blinkit"
                isPackageInstalled(context, PKG_ZEPTO) || isPackageInstalled(context, PKG_ZEPTO_ALT) -> "Zepto"
                else -> "Amazon"
            }
        }
    }

    /**
     * Builds canonical direct-to-cart URLs across major e-commerce platforms.
     */
    fun buildStoreCartUrl(store: String, forApp: Boolean = false): String {
        val s = store.lowercase()
        return when {
            s.contains("blinkit") || s.contains("grofers") -> "https://blinkit.com/cart"
            s.contains("zepto") -> if (forApp) "zepto://zeptonow.com/app/cart" else "https://www.zeptonow.com/cart"
            s.contains("amazon") -> "https://www.amazon.in/gp/aw/c"
            s.contains("flipkart") -> "https://www.flipkart.com/viewcart"
            else -> "https://google.com"
        }
    }

    /**
     * Opens the native store shopping cart directly for 1-tap checkout.
     */
    fun openStoreCart(context: Context, storeName: String): Boolean {
        val targetPkg = resolveInstalledPackageForStore(storeName, context)
        val cartUrl = buildStoreCartUrl(storeName, forApp = targetPkg != null)
        return openDeepLink(
            context = context,
            url = cartUrl,
            preferredPackage = targetPkg,
            storeName = storeName
        )
    }

    /**
     * Opens a store search query directly in the native application with fallback to browser.
     */
    fun openStoreSearch(context: Context, storeName: String, query: String): Boolean {
        val targetPkg = resolveInstalledPackageForStore(storeName, context)
        val searchUrl = buildStoreSearchUrl(storeName, query, forApp = targetPkg != null)
        return openDeepLink(
            context = context,
            url = searchUrl,
            preferredPackage = targetPkg,
            fallbackSearchQuery = query,
            storeName = storeName
        )
    }

    /**
     * Dispatches a 1-tap deep link into the native installed app on the exact product page,
     * or gracefully falls back to the system browser.
     */
    fun openDeepLink(
        context: Context,
        url: String,
        preferredPackage: String? = null,
        fallbackSearchQuery: String? = null,
        storeName: String? = null
    ): Boolean {
        val isUrlInvalid = url.isBlank() ||
            url.equals("null", ignoreCase = true) ||
            url.startsWith("https://null", ignoreCase = true) ||
            url.startsWith("http://null", ignoreCase = true) ||
            url.equals("about:blank", ignoreCase = true)

        var fixedUrl = if (isUrlInvalid) {
            if (!fallbackSearchQuery.isNullOrBlank() && !storeName.isNullOrBlank()) {
                val targetPkg = preferredPackage ?: resolveAppPackage(null, fallbackSearchQuery, context = context)
                buildStoreSearchUrl(storeName, fallbackSearchQuery, forApp = targetPkg != null)
            } else {
                ""
            }
        } else if (!url.startsWith("http://", ignoreCase = true) && !url.startsWith("https://", ignoreCase = true) && !url.startsWith("zepto://", ignoreCase = true)) {
            "https://$url"
        } else url

        // Android intent filters for Zepto specifically require www.zepto.com or www.zeptonow.com
        if (fixedUrl.contains("://zepto.com", ignoreCase = true)) {
            fixedUrl = fixedUrl.replace("://zepto.com", "://www.zepto.com")
        }

        // 1. Try launching specifically into the native app if installed
        val targetPkg = preferredPackage ?: resolveAppPackage(fixedUrl, fallbackSearchQuery, context = context)
        if (!targetPkg.isNullOrBlank()) {
            val effectivePkg = when (targetPkg) {
                PKG_AMAZON_IN, PKG_AMAZON_GLOBAL -> {
                    if (isPackageInstalled(context, PKG_AMAZON_IN)) PKG_AMAZON_IN
                    else if (isPackageInstalled(context, PKG_AMAZON_GLOBAL)) PKG_AMAZON_GLOBAL
                    else null
                }
                PKG_ZEPTO, PKG_ZEPTO_ALT -> {
                    if (isPackageInstalled(context, PKG_ZEPTO)) PKG_ZEPTO
                    else if (isPackageInstalled(context, PKG_ZEPTO_ALT)) PKG_ZEPTO_ALT
                    else null
                }
                else -> {
                    if (isPackageInstalled(context, targetPkg)) targetPkg else null
                }
            }

            if (effectivePkg != null) {
                // Zepto native intent routing: Must use zepto:// scheme
                if ((effectivePkg == PKG_ZEPTO || effectivePkg == PKG_ZEPTO_ALT) && !fixedUrl.startsWith("zepto://")) {
                    val queryParam = try {
                        Uri.parse(fixedUrl).getQueryParameter("q") ?: Uri.parse(fixedUrl).getQueryParameter("query")
                    } catch (_: Exception) { null } ?: fallbackSearchQuery

                    if (!queryParam.isNullOrBlank()) {
                        fixedUrl = "zepto://zeptonow.com/app/search?query=${Uri.encode(queryParam.trim())}"
                    } else if (fixedUrl.contains("cart", ignoreCase = true)) {
                        fixedUrl = "zepto://zeptonow.com/app/cart"
                    } else {
                        fixedUrl = "zepto://zeptonow.com/app"
                    }
                }

                if (fixedUrl.isNotBlank()) {
                    val uri = Uri.parse(fixedUrl)
                    try {
                        val appIntent = Intent(Intent.ACTION_VIEW, uri).apply {
                            setPackage(effectivePkg)
                            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP)
                        }
                        context.startActivity(appIntent)
                        Log.i(TAG, "Successfully launched deep link in native app '$effectivePkg': $fixedUrl")
                        return true
                    } catch (e: Exception) {
                        Log.w(TAG, "Targeted app launch failed for '$effectivePkg', trying unconstrained intent: ${e.message}")
                        try {
                            val unconstrainedIntent = Intent(Intent.ACTION_VIEW, uri).apply {
                                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                            }
                            context.startActivity(unconstrainedIntent)
                            return true
                        } catch (_: Exception) {}
                    }
                }

                // If URL failed or was blank, launch app root cleanly
                try {
                    val launchIntent = context.packageManager.getLaunchIntentForPackage(effectivePkg)?.apply {
                        addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                    }
                    if (launchIntent != null) {
                        context.startActivity(launchIntent)
                        Log.i(TAG, "Launched native app root for '$effectivePkg' as fallback")
                        return true
                    }
                } catch (_: Exception) {}
            }
        }

        if (fixedUrl.isBlank()) return false

        // 2. Fallback to default browser / system resolver
        return try {
            val uri = Uri.parse(fixedUrl)
            val browserIntent = Intent(Intent.ACTION_VIEW, uri).apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            context.startActivity(browserIntent)
            Log.i(TAG, "Launched URL in default browser: $fixedUrl")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to launch URL anywhere: ${e.message}", e)
            false
        }
    }

    /**
     * Muse-Style Direct Cart Opener.
     * Deep-links directly to the native shopping cart or checkout screen of the retailer.
     */
    fun openCartScreen(context: Context, storeName: String): Boolean {
        val s = storeName.lowercase(java.util.Locale.ROOT)
        val cartUri = when {
            s.contains("zepto") -> Uri.parse("zepto://zeptonow.com/app/cart")
            s.contains("blinkit") || s.contains("grofers") -> Uri.parse("blinkit://cart")
            s.contains("amazon") -> Uri.parse("https://www.amazon.in/gp/cart/view.html")
            s.contains("flipkart") -> Uri.parse("https://www.flipkart.com/viewcart")
            else -> null
        }
        val pkg = resolveInstalledPackageForStore(storeName, context)
        if (pkg != null) {
            try {
                if (cartUri != null) {
                    val intent = Intent(Intent.ACTION_VIEW, cartUri).apply {
                        setPackage(pkg)
                        addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                    }
                    context.startActivity(intent)
                    return true
                }
            } catch (_: Exception) {}

            try {
                val launchIntent = context.packageManager.getLaunchIntentForPackage(pkg)?.apply {
                    addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                }
                if (launchIntent != null) {
                    context.startActivity(launchIntent)
                    return true
                }
            } catch (_: Exception) {}
        }
        val webFallback = when {
            s.contains("zepto") -> "https://www.zeptonow.com/cart"
            s.contains("blinkit") -> "https://blinkit.com/cart"
            s.contains("amazon") -> "https://www.amazon.in/gp/cart/view.html"
            s.contains("flipkart") -> "https://www.flipkart.com/viewcart"
            else -> "https://www.google.com"
        }
        return openDeepLink(
            context = context,
            url = webFallback,
            preferredPackage = pkg,
            storeName = storeName
        )
    }
}

