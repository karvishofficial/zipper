package com.zipper.app.core.web

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.os.ParcelFileDescriptor
import android.print.PageRange
import android.print.PrintAttributes
import android.print.PrintDocumentAdapter
import android.print.PrintDocumentInfo
import android.util.Log
import android.webkit.CookieManager
import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import com.zipper.app.core.agent.ScrollDirection
import com.zipper.app.core.model.DistilledUiState
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import org.json.JSONTokener
import java.io.BufferedReader
import java.io.ByteArrayInputStream
import java.io.File
import java.io.InputStreamReader
import kotlin.coroutines.resume

/**
 * Single-worker Hidden WebView Controller for Karvish.
 * Enables silent background web exploration, product discovery, and DOM state extraction.
 *
 * Implements:
 * - Offscreen execution without screen hijacking
 * - Memory-friendly single reusable WebView lifecycle (protects Samsung A26 RAM)
 * - Closed-loop React prototype value setters for controlled form fields
 * - Normalized coordinate and candidate extraction matching Karvish V50+ contracts
 */
/**
 * Interface abstraction for web controllers, enabling testing and modularity.
 */
interface WebController {
    suspend fun navigate(url: String, timeoutMs: Long = 12000L): Boolean
    suspend fun distillWebState(): DistilledUiState
    suspend fun clickElement(karvishId: String): Boolean
    suspend fun typeText(karvishId: String, text: String): Boolean
    suspend fun selectOption(karvishId: String, value: String): Boolean
    suspend fun scroll(direction: ScrollDirection, pixels: Int = 800): Boolean
    suspend fun submit(): Boolean
    suspend fun goBack(): Boolean
    suspend fun clear()
    suspend fun destroy()
}

class HiddenWebController(private val context: Context) : WebController {

    companion object {
        private const val TAG = "HiddenWebController"
        private const val USER_AGENT =
            "Mozilla/5.0 (Linux; Android 14; SM-A266B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36"

        private val AD_AND_TRACKER_HOSTS = setOf(
            "doubleclick.net",
            "google-analytics.com",
            "googlesyndication.com",
            "adservice.google",
            "adnxs.com",
            "facebook.net",
            "criteo.net",
            "criteo.com",
            "taboola.com",
            "outbrain.com",
            "amazon-adsystem.com",
            "moatads.com",
            "scorecardresearch.com",
            "hotjar.com",
            "clarity.ms",
            "adroll.com",
            "pubmatic.com",
            "rubiconproject.com",
            "smartadserver.com"
        )

        private val EMPTY_RESOURCE = WebResourceResponse("text/plain", "UTF-8", ByteArrayInputStream(ByteArray(0)))

        fun isAdOrTracker(url: String?): Boolean {
            if (url.isNullOrBlank()) return false
            val host = try {
                val withoutScheme = if (url.contains("://")) url.substringAfter("://") else url
                withoutScheme.substringBefore("/").substringBefore("?").substringBefore(":").lowercase()
            } catch (_: Exception) {
                null
            } ?: return false
            return AD_AND_TRACKER_HOSTS.any { host.contains(it) }
        }

        const val COOKIE_CRUSHER_JS = """
            (function() {
                try {
                    const sel = [
                        'button#accept-cookie', 'button#onetrust-accept-btn-handler',
                        'button.cookie-accept', 'button[aria-label*="accept" i]',
                        'button[aria-label*="agree" i]', 'a[aria-label*="accept" i]',
                        'button[id*="cookie" i]', 'button[class*="cookie" i]'
                    ];
                    for (const s of sel) {
                        const b = document.querySelector(s);
                        if (b && b.offsetParent !== null) { b.click(); break; }
                    }
                    const buttons = Array.from(document.querySelectorAll('button, a.btn, [role="button"]'));
                    for (const btn of buttons) {
                        const txt = (btn.innerText || '').trim().toLowerCase();
                        if (txt === 'accept all' || txt === 'accept all cookies' || txt === 'i accept' || txt === 'i agree' || txt === 'got it' || txt === 'agree') {
                            if (btn.offsetParent !== null) { btn.click(); break; }
                        }
                    }
                } catch(e) {}
            })();
        """
    }

    private var webView: WebView? = null
    private var mutationBridge: WebMutationBridge? = null
    private val distiller = WebStateDistiller()
    private val mainHandler = Handler(Looper.getMainLooper())
    private var cachedDistillerScript: String? = null

    var onUrlChanged: ((String) -> Unit)? = null
    var isTurboBackgroundMode: Boolean = false

    fun getWebView(): WebView? = webView
    fun getContext(): Context = context

    fun flushCookies() {
        try {
            CookieManager.getInstance().flush()
        } catch (e: Exception) {
            Log.w(TAG, "Failed to flush cookies", e)
        }
    }

    fun saveCurrentPageAsPdf(destinationFile: File, onComplete: (Boolean) -> Unit) {
        val wv = webView ?: run {
            onComplete(false)
            return
        }
        mainHandler.post {
            try {
                destinationFile.parentFile?.mkdirs()
                val printAdapter = wv.createPrintDocumentAdapter("KarvishWebPage")
                val printAttributes = PrintAttributes.Builder()
                    .setColorMode(PrintAttributes.COLOR_MODE_COLOR)
                    .setMediaSize(PrintAttributes.MediaSize.ISO_A4)
                    .setResolution(PrintAttributes.Resolution("pdf", "pdf", 300, 300))
                    .setMinMargins(PrintAttributes.Margins.NO_MARGINS)
                    .build()

                val pfd = ParcelFileDescriptor.open(
                    destinationFile,
                    ParcelFileDescriptor.MODE_CREATE or ParcelFileDescriptor.MODE_READ_WRITE or ParcelFileDescriptor.MODE_TRUNCATE
                )

                android.print.PdfPrintHelper.printAdapterToPdf(
                    printAdapter,
                    printAttributes,
                    pfd,
                    object : android.print.PdfPrintHelper.Callback {
                        override fun onSuccess() {
                            onComplete(true)
                        }

                        override fun onFailure() {
                            onComplete(false)
                        }
                    }
                )
            } catch (e: Exception) {
                Log.e(TAG, "Failed to export PDF", e)
                onComplete(false)
            }
        }
    }

    private suspend fun getOrCreateWebView(): WebView = withContext(Dispatchers.Main) {
        webView?.let { return@withContext it }
        val wv = WebView(context).apply {
            settings.apply {
                javaScriptEnabled = true
                domStorageEnabled = true
                databaseEnabled = true
                useWideViewPort = true
                loadWithOverviewMode = true
                userAgentString = USER_AGENT
                cacheMode = WebSettings.LOAD_DEFAULT
                setSupportZoom(true)
                builtInZoomControls = true
                displayZoomControls = false
            }
            val cookieManager = CookieManager.getInstance()
            cookieManager.setAcceptCookie(true)
            cookieManager.setAcceptThirdPartyCookies(this, true)
            val bridge = WebMutationBridge()
            mutationBridge = bridge
            addJavascriptInterface(bridge, WebMutationBridge.JS_INTERFACE_NAME)

            webViewClient = object : WebViewClient() {
                override fun onPageFinished(view: WebView?, finishedUrl: String?) {
                    super.onPageFinished(view, finishedUrl)
                    finishedUrl?.let { onUrlChanged?.invoke(it) }
                    view?.evaluateJavascript(loadNetworkObserverScript(), null)
                    view?.evaluateJavascript(WebMutationBridge.OBSERVER_INJECTION_JS, null)
                    view?.evaluateJavascript(COOKIE_CRUSHER_JS, null)
                    flushCookies()
                }
                override fun shouldInterceptRequest(view: WebView?, request: WebResourceRequest?): WebResourceResponse? {
                    val reqUrl = request?.url?.toString()
                    if (isAdOrTracker(reqUrl)) {
                        return EMPTY_RESOURCE
                    }
                    if (isTurboBackgroundMode) {
                        val lower = reqUrl?.lowercase() ?: ""
                        if (lower.endsWith(".jpg") || lower.endsWith(".jpeg") || lower.endsWith(".png") ||
                            lower.endsWith(".webp") || lower.endsWith(".gif") || lower.endsWith(".svg") ||
                            lower.endsWith(".woff") || lower.endsWith(".woff2") || lower.endsWith(".ttf") ||
                            lower.endsWith(".mp4") || lower.contains("/image") || lower.contains("/media")
                        ) {
                            return EMPTY_RESOURCE
                        }
                    }
                    return super.shouldInterceptRequest(view, request)
                }
                override fun shouldOverrideUrlLoading(view: WebView?, request: WebResourceRequest?): Boolean {
                    val reqUrl = request?.url?.toString() ?: return false
                    return !(reqUrl.startsWith("http://") || reqUrl.startsWith("https://"))
                }
            }
            layout(0, 0, 1080, 2400)
        }
        webView = wv
        wv
    }

    private var cachedNetworkObserverScript: String? = null

    private fun loadNetworkObserverScript(): String {
        cachedNetworkObserverScript?.let { return it }
        return try {
            context.assets.open("web_network_observer.js").use { inputStream ->
                BufferedReader(InputStreamReader(inputStream)).readText()
            }
        } catch (_: Exception) {
            ""
        }.also { cachedNetworkObserverScript = it }
    }

    suspend fun waitForDomStability(timeoutMs: Long = 4000L): Boolean = withContext(Dispatchers.Main) {
        val bridge = mutationBridge
        if (bridge != null) {
            bridge.awaitSettled(timeoutMs)
        } else {
            true
        }
    }

    suspend fun evaluateJs(script: String): String = withContext(Dispatchers.Main) {
        val wv = getOrCreateWebView()
        evaluateJs(wv, script)
    }

    fun dispatchPhysicalTouch(normalizedX: Float, normalizedY: Float): Boolean {
        val wv = webView ?: return false
        val density = context.resources.displayMetrics.density
        val targetWidth = if (wv.width > 0) wv.width.toFloat() else 1080f * density
        val targetHeight = if (wv.height > 0) wv.height.toFloat() else 2400f * density

        val physicalX = (normalizedX / 1000f) * targetWidth
        val physicalY = (normalizedY / 1000f) * targetHeight

        val downTime = android.os.SystemClock.uptimeMillis()
        val eventTime = downTime + 50
        val downEvent = android.view.MotionEvent.obtain(downTime, downTime, android.view.MotionEvent.ACTION_DOWN, physicalX, physicalY, 0)
        val upEvent = android.view.MotionEvent.obtain(downTime, eventTime, android.view.MotionEvent.ACTION_UP, physicalX, physicalY, 0)

        val downOk = wv.dispatchTouchEvent(downEvent)
        val upOk = wv.dispatchTouchEvent(upEvent)
        downEvent.recycle()
        upEvent.recycle()
        return downOk && upOk
    }

    private fun loadDistillerScript(): String {
        cachedDistillerScript?.let { return it }
        return try {
            context.assets.open("web_distiller.js").use { inputStream ->
                BufferedReader(InputStreamReader(inputStream)).readText()
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to load web_distiller.js asset, using fallback", e)
            FALLBACK_DISTILLER_SCRIPT
        }.also { cachedDistillerScript = it }
    }

    override suspend fun navigate(url: String, timeoutMs: Long): Boolean = withContext(Dispatchers.Main) {
        val wv = getOrCreateWebView()
        val pageLoaded = CompletableDeferred<Boolean>()

        wv.webViewClient = object : WebViewClient() {
            override fun onPageFinished(view: WebView?, finishedUrl: String?) {
                super.onPageFinished(view, finishedUrl)
                if (!pageLoaded.isCompleted) {
                    pageLoaded.complete(true)
                }
                finishedUrl?.let { onUrlChanged?.invoke(it) }
                view?.evaluateJavascript(COOKIE_CRUSHER_JS, null)
                flushCookies()
            }

            override fun shouldInterceptRequest(view: WebView?, request: WebResourceRequest?): WebResourceResponse? {
                val reqUrl = request?.url?.toString()
                if (isAdOrTracker(reqUrl)) {
                    return EMPTY_RESOURCE
                }
                return super.shouldInterceptRequest(view, request)
            }

            override fun onReceivedError(view: WebView?, request: WebResourceRequest?, error: WebResourceError?) {
                super.onReceivedError(view, request, error)
                if (request?.isForMainFrame == true && !pageLoaded.isCompleted) {
                    Log.w(TAG, "Page load error for $url: ${error?.description}")
                    pageLoaded.complete(false)
                }
            }

            override fun shouldOverrideUrlLoading(view: WebView?, request: WebResourceRequest?): Boolean {
                val reqUrl = request?.url?.toString() ?: return false
                return !(reqUrl.startsWith("http://") || reqUrl.startsWith("https://"))
            }
        }

        wv.loadUrl(url)
        val success = withTimeoutOrNull(timeoutMs) { pageLoaded.await() } ?: false
        if (!success) {
            Log.w(TAG, "Navigation timed out for $url after ${timeoutMs}ms")
        }
        success
    }

    override suspend fun distillWebState(): DistilledUiState = withContext(Dispatchers.Main) {
        val wv = getOrCreateWebView()
        val script = loadDistillerScript()
        val rawJson = evaluateJs(wv, script)
        val unescaped = unescapeJsString(rawJson)
        distiller.distill(unescaped)
    }

    override suspend fun clickElement(karvishId: String): Boolean = withContext(Dispatchers.Main) {
        val wv = getOrCreateWebView()
        val script = """
            (function() {
                try {
                    const el = document.querySelector('[data-karvish-id="' + '$karvishId' + '"]')
                        || document.getElementById('$karvishId')
                        || document.querySelector('[name="' + '$karvishId' + '"]');
                    if (!el) return 'false';
                    el.scrollIntoView({ behavior: 'instant', block: 'center' });
                    el.focus();
                    el.click();
                    return 'true';
                } catch(e) {
                    return 'error:' + e.message;
                }
            })()
        """.trimIndent()
        val result = evaluateJs(wv, script)
        result.contains("true")
    }

    override suspend fun typeText(karvishId: String, text: String): Boolean = withContext(Dispatchers.Main) {
        val wv = getOrCreateWebView()
        val escapedText = text.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")
        val script = """
            (function() {
                try {
                    const el = document.querySelector('[data-karvish-id="' + '$karvishId' + '"]')
                        || document.getElementById('$karvishId')
                        || document.querySelector('[name="' + '$karvishId' + '"]');
                    if (!el) return 'false';
                    el.scrollIntoView({ behavior: 'instant', block: 'center' });
                    el.focus();
                    
                    // Check if part of split OTP / PIN inputs (e.g. 4 separate boxes with gaps)
                    if ('$escapedText'.length > 1) {
                        const parent = el.parentElement || el;
                        const container = parent.closest('form, [class*="otp" i], [class*="pin" i], [id*="otp" i], [id*="pin" i], [class*="grid" i], [class*="row" i]')
                            || parent.parentElement || parent;
                        let splitInputs = [];
                        if (container) {
                            splitInputs = Array.from(container.querySelectorAll('input[type="text"], input[type="tel"], input[type="number"], input[type="password"], input:not([type])')).filter(i => {
                                const maxL = i.getAttribute('maxlength') || i.maxLength;
                                const cls = ((i.className || '') + ' ' + (i.id || '') + ' ' + (i.name || '')).toLowerCase();
                                return maxL == 1 || i.size == 1 || cls.includes('otp') || cls.includes('pin') || cls.includes('digit') || cls.includes('code');
                            });
                        }
                        if (splitInputs.length === 0 && el.parentElement) {
                            splitInputs = Array.from(el.parentElement.querySelectorAll('input')).filter(i => (i.getAttribute('maxlength') || i.maxLength) == 1);
                        }
                        if (splitInputs.length >= '$escapedText'.length) {
                            splitInputs.sort((a, b) => (a.compareDocumentPosition(b) & 2) ? 1 : -1);
                            const chars = '$escapedText'.split('');
                            chars.forEach((c, idx) => {
                                const targetInp = splitInputs[idx];
                                if (targetInp) {
                                    targetInp.focus();
                                    const proto = Object.getPrototypeOf(targetInp);
                                    const desc = Object.getOwnPropertyDescriptor(proto, 'value')
                                        || Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value');
                                    if (desc && desc.set) {
                                        desc.set.call(targetInp, c);
                                    } else {
                                        targetInp.value = c;
                                    }
                                    targetInp.dispatchEvent(new KeyboardEvent('keydown', { key: c, bubbles: true }));
                                    targetInp.dispatchEvent(new KeyboardEvent('keypress', { key: c, bubbles: true }));
                                    targetInp.dispatchEvent(new Event('input', { bubbles: true }));
                                    targetInp.dispatchEvent(new KeyboardEvent('keyup', { key: c, bubbles: true }));
                                    targetInp.dispatchEvent(new Event('change', { bubbles: true }));
                                }
                            });
                            return 'true';
                        }
                    }

                    // React/Vue/Angular controlled input prototype setter trick
                    const prototype = Object.getPrototypeOf(el);
                    const descriptor = Object.getOwnPropertyDescriptor(prototype, 'value') 
                        || Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value');
                    if (descriptor && descriptor.set) {
                        descriptor.set.call(el, '$escapedText');
                    } else {
                        el.value = '$escapedText';
                    }
                    el.dispatchEvent(new Event('input', { bubbles: true }));
                    el.dispatchEvent(new Event('change', { bubbles: true }));
                    return 'true';
                } catch(e) {
                    return 'error:' + e.message;
                }
            })()
        """.trimIndent()
        val result = evaluateJs(wv, script)
        result.contains("true")
    }

    override suspend fun selectOption(karvishId: String, value: String): Boolean = withContext(Dispatchers.Main) {
        val wv = getOrCreateWebView()
        val escapedValue = value.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")
        val script = """
            (function() {
                try {
                    const el = document.querySelector('[data-karvish-id="' + '$karvishId' + '"]')
                        || document.getElementById('$karvishId')
                        || document.querySelector('[name="' + '$karvishId' + '"]');
                    if (!el) return 'false';
                    el.scrollIntoView({ behavior: 'instant', block: 'center' });
                    el.focus();

                    if (el.tagName === 'SELECT') {
                        const target = '$escapedValue'.trim().toLowerCase();
                        const months = {'jan':1,'january':1,'feb':2,'february':2,'mar':3,'march':3,'apr':4,'april':4,'may':5,'jun':6,'june':6,'jul':7,'july':7,'aug':8,'august':8,'sep':9,'september':9,'oct':10,'october':10,'nov':11,'november':11,'dec':12,'december':12};
                        const targetMonth = months[target] || null;
                        const targetInt = parseInt(target, 10);

                        let foundIndex = -1;
                        for (let i = 0; i < el.options.length; i++) {
                            const opt = el.options[i];
                            const val = (opt.value || '').trim().toLowerCase();
                            const txt = (opt.text || '').trim().toLowerCase();
                            if (val === target || txt === target) {
                                foundIndex = i;
                                break;
                            }
                            if (targetMonth !== null && (months[val] === targetMonth || months[txt] === targetMonth || parseInt(val, 10) === targetMonth || parseInt(txt, 10) === targetMonth)) {
                                foundIndex = i;
                                break;
                            }
                            if (!isNaN(targetInt) && targetInt > 0 && (parseInt(val, 10) === targetInt || parseInt(txt, 10) === targetInt)) {
                                foundIndex = i;
                                break;
                            }
                        }
                        if (foundIndex === -1) {
                            for (let i = 0; i < el.options.length; i++) {
                                const opt = el.options[i];
                                const val = (opt.value || '').trim().toLowerCase();
                                const txt = (opt.text || '').trim().toLowerCase();
                                if (txt.includes(target) || target.includes(txt) || val.includes(target)) {
                                    foundIndex = i;
                                    break;
                                }
                            }
                        }
                        if (foundIndex !== -1) {
                            el.selectedIndex = foundIndex;
                            el.value = el.options[foundIndex].value;
                            el.dispatchEvent(new Event('input', { bubbles: true }));
                            el.dispatchEvent(new Event('change', { bubbles: true }));
                            if (typeof el.onchange === 'function') {
                                try { el.onchange(); } catch(e) {}
                            }
                            return 'true';
                        }
                        return 'false';
                    }

                    // Fallback if applied to input or custom dropdown
                    const prototype = Object.getPrototypeOf(el);
                    const descriptor = Object.getOwnPropertyDescriptor(prototype, 'value') 
                        || Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value');
                    if (descriptor && descriptor.set) {
                        descriptor.set.call(el, '$escapedValue');
                    } else {
                        el.value = '$escapedValue';
                    }
                    el.dispatchEvent(new Event('input', { bubbles: true }));
                    el.dispatchEvent(new Event('change', { bubbles: true }));
                    return 'true';
                } catch(e) {
                    return 'error:' + e.message;
                }
            })()
        """.trimIndent()
        val result = evaluateJs(wv, script)
        result.contains("true")
    }

    override suspend fun scroll(direction: ScrollDirection, pixels: Int): Boolean = withContext(Dispatchers.Main) {
        val wv = getOrCreateWebView()
        val deltaY = if (direction == ScrollDirection.DOWN) pixels else -pixels
        val script = """
            (function() {
                window.scrollBy({ top: $deltaY, behavior: 'instant' });
                return 'true';
            })()
        """.trimIndent()
        val result = evaluateJs(wv, script)
        result.contains("true")
    }

    override suspend fun submit(): Boolean = withContext(Dispatchers.Main) {
        val wv = getOrCreateWebView()
        val script = """
            (function() {
                try {
                    const active = document.activeElement;
                    if (active && active.form) {
                        active.form.submit();
                        return 'true';
                    }
                    const submitBtn = document.querySelector('button[type="submit"], input[type="submit"]');
                    if (submitBtn) {
                        submitBtn.click();
                        return 'true';
                    }
                    return 'false';
                } catch(e) {
                    return 'false';
                }
            })()
        """.trimIndent()
        val result = evaluateJs(wv, script)
        result.contains("true")
    }

    override suspend fun goBack(): Boolean = withContext(Dispatchers.Main) {
        val wv = getOrCreateWebView()
        if (wv.canGoBack()) {
            wv.goBack()
            true
        } else {
            false
        }
    }

    override suspend fun clear(): Unit = withContext(Dispatchers.Main) {
        webView?.apply {
            clearHistory()
            clearCache(true)
            loadUrl("about:blank")
        }
    }

    override suspend fun destroy(): Unit = withContext(Dispatchers.Main) {
        webView?.apply {
            (parent as? android.view.ViewGroup)?.removeView(this)
            stopLoading()
            clearHistory()
            clearCache(true)
            destroy()
        }
        webView = null
    }

    suspend fun getHtml(): String = withContext(Dispatchers.Main) {
        val wv = getOrCreateWebView()
        val raw = evaluateJs(wv, "document.documentElement ? document.documentElement.outerHTML : ''")
        unescapeJsString(raw)
    }

    private suspend fun evaluateJs(wv: WebView, script: String): String =
        suspendCancellableCoroutine { continuation ->
            wv.evaluateJavascript(script) { result ->
                if (continuation.isActive) {
                    continuation.resume(result ?: "")
                }
            }
        }

    private fun unescapeJsString(raw: String): String {
        val trimmed = raw.trim()
        if (trimmed.startsWith("\"") && trimmed.endsWith("\"") && trimmed.length >= 2) {
            return try {
                JSONTokener(trimmed).nextValue().toString()
            } catch (_: Exception) {
                trimmed.substring(1, trimmed.length - 1)
                    .replace("\\\"", "\"")
                    .replace("\\\\", "\\")
                    .replace("\\n", "\n")
            }
        }
        return trimmed
    }
}

private const val FALLBACK_DISTILLER_SCRIPT = """
(function() {
    return JSON.stringify({
        url: window.location.href,
        title: document.title,
        candidates: []
    });
})();
"""
