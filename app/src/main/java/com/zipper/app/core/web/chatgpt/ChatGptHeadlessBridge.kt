package com.zipper.app.core.web.chatgpt

import android.annotation.SuppressLint
import android.app.Activity
import android.content.Context
import android.graphics.Color
import android.graphics.Typeface
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.webkit.CookieManager
import android.webkit.JavascriptInterface
import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import com.google.android.material.bottomsheet.BottomSheetBehavior
import com.google.android.material.bottomsheet.BottomSheetDialog
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import java.lang.ref.WeakReference
import kotlin.coroutines.resume

/**
 * ChatGPT Guest & Verification Bridge for Zipper & Webmind.
 * Enables zero-key, zero-login AI reasoning directly via a background WebView.
 *
 * Smart Verification Flow:
 * 1. Background Execution by Default: Runs queries invisibly whenever possible.
 * 2. Cloudflare & Bot Challenge Detection: Detects "Just a moment", Cloudflare Turnstile,
 *    or login gates in real-time.
 * 3. Smart BottomSheet Popup: Pops up an interactive verification sheet on the active Activity
 *    so the user can tap "Verify you are human" or log in directly.
 * 4. Auto-Resume & Auto-Dismiss: Once verified, automatically detects the chat input,
 *    flushes cookies, dismisses the sheet, and completes the pending prompt.
 * 5. Multi-Turn Session Continuity: Reuses the active WebView instance in memory to preserve
 *    continuous conversation context and cookies across queries.
 */
object ChatGptHeadlessBridge {

    private const val TAG = "ChatGptHeadlessBridge"
    private const val CHATGPT_URL = "https://chatgpt.com"
    private const val CHROME_USER_AGENT =
        "Mozilla/5.0 (Linux; Android 15; SM-A266B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Mobile Safari/537.36"

    private var appContext: Context? = null
    private var headlessWebView: WebView? = null
    private var isPageLoaded = false
    private var isEngineReady = false
    private var pendingPrompt: String? = null
    private val mainHandler = Handler(Looper.getMainLooper())
    private val executionMutex = Mutex()

    private var activeContinuation: ((Result<String>) -> Unit)? = null
    private var weakActivity: WeakReference<Activity>? = null
    private var activeDialog: BottomSheetDialog? = null

    /**
     * Associates the current foreground Activity for interactive bot verification when needed.
     */
    fun setVerificationActivity(activity: Activity) {
        weakActivity = WeakReference(activity)
    }

    /**
     * Initializes and preloads the ChatGPT session in the background.
     * Safe to call multiple times; only initializes once.
     */
    @SuppressLint("SetJavaScriptEnabled")
    fun init(context: Context) {
        appContext = context.applicationContext
        if (context is Activity) {
            setVerificationActivity(context)
        }
        if (headlessWebView != null) return

        mainHandler.post {
            try {
                val ctx = appContext ?: context.applicationContext
                headlessWebView = WebView(ctx).apply {
                    settings.apply {
                        javaScriptEnabled = true
                        domStorageEnabled = true
                        databaseEnabled = true
                        cacheMode = WebSettings.LOAD_DEFAULT
                        userAgentString = CHROME_USER_AGENT
                        mediaPlaybackRequiresUserGesture = false
                    }

                    val cookieManager = CookieManager.getInstance()
                    cookieManager.setAcceptCookie(true)
                    cookieManager.setAcceptThirdPartyCookies(this, true)

                    addJavascriptInterface(BridgeInterface(), "ZipperBridge")

                    webViewClient = object : WebViewClient() {
                        override fun onPageFinished(view: WebView?, url: String?) {
                            super.onPageFinished(view, url)
                            val title = view?.title ?: ""
                            Log.i(TAG, "ChatGPT web page finished loading: $url (Title: $title)")
                            isPageLoaded = true
                            CookieManager.getInstance().flush()

                            val isCfTitle = title.contains("Just a moment", true) ||
                                    title.contains("Cloudflare", true) ||
                                    title.contains("Attention Required", true) ||
                                    title.contains("Security Check", true)
                            val isCfUrl = url?.contains("challenges.cloudflare.com") == true

                            if (isCfTitle || isCfUrl) {
                                Log.w(TAG, "Cloudflare challenge detected by title/url: $title / $url")
                                triggerVerificationPrompt("Cloudflare Bot Challenge Detected")
                            }

                            injectReadyWatcher(view)
                        }

                        override fun onReceivedError(
                            view: WebView?,
                            request: WebResourceRequest?,
                            error: WebResourceError?
                        ) {
                            super.onReceivedError(view, request, error)
                            if (request?.isForMainFrame == true) {
                                Log.w(TAG, "Main frame error loading ChatGPT: ${error?.description}")
                            }
                        }
                    }

                    Log.i(TAG, "Preloading background ChatGPT guest session: $CHATGPT_URL")
                    loadUrl(CHATGPT_URL)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize ChatGPT WebView: ${e.message}", e)
            }
        }
    }

    /**
     * Displays an interactive BottomSheet with the active ChatGPT WebView so the user
     * can solve Cloudflare Turnstile, CAPTCHA, or log in.
     */
    fun showVerificationSheet(activity: Activity, reason: String = "Verification Required") {
        mainHandler.post {
            try {
                if (activity.isFinishing || activity.isDestroyed) return@post
                if (activeDialog?.isShowing == true) return@post

                val wv = headlessWebView ?: run {
                    init(activity)
                    headlessWebView
                } ?: return@post

                // Ensure WebView is detached from any previous parent before attaching to dialog
                (wv.parent as? ViewGroup)?.removeView(wv)

                val dialog = BottomSheetDialog(activity)
                activeDialog = dialog

                val density = activity.resources.displayMetrics.density
                fun dp(px: Int) = (px * density).toInt()

                // Root Layout
                val root = LinearLayout(activity).apply {
                    orientation = LinearLayout.VERTICAL
                    layoutParams = ViewGroup.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT
                    )
                    setPadding(dp(16), dp(12), dp(16), dp(16))
                    setBackgroundColor(Color.WHITE)
                }

                // Header Row
                val headerRow = LinearLayout(activity).apply {
                    orientation = LinearLayout.HORIZONTAL
                    gravity = Gravity.CENTER_VERTICAL
                    layoutParams = LinearLayout.LayoutParams(
                        LinearLayout.LayoutParams.MATCH_PARENT,
                        LinearLayout.LayoutParams.WRAP_CONTENT
                    )
                }

                val titleLayout = LinearLayout(activity).apply {
                    orientation = LinearLayout.VERTICAL
                    layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1.0f)
                }

                val tvTitle = TextView(activity).apply {
                    text = "🤖 ChatGPT Verification & Session"
                    textSize = 15.5f
                    setTypeface(null, Typeface.BOLD)
                    setTextColor(Color.parseColor("#111827"))
                }

                val tvSubtitle = TextView(activity).apply {
                    text = "Solve Cloudflare check or log in below. Zipper will auto-close and continue once verified."
                    textSize = 12f
                    setTextColor(Color.parseColor("#4B5563"))
                    setPadding(0, dp(2), 0, 0)
                }

                titleLayout.addView(tvTitle)
                titleLayout.addView(tvSubtitle)
                headerRow.addView(titleLayout)

                // Reload Button
                val btnRefresh = Button(activity).apply {
                    text = "🔄"
                    textSize = 14f
                    setBackgroundColor(Color.TRANSPARENT)
                    setTextColor(Color.parseColor("#2563EB"))
                    setOnClickListener {
                        wv.loadUrl(CHATGPT_URL)
                        Toast.makeText(activity, "Reloading ChatGPT...", Toast.LENGTH_SHORT).show()
                    }
                }
                headerRow.addView(btnRefresh)

                // Close Button
                val btnClose = Button(activity).apply {
                    text = "✕"
                    textSize = 16f
                    setTypeface(null, Typeface.BOLD)
                    setBackgroundColor(Color.TRANSPARENT)
                    setTextColor(Color.parseColor("#6B7280"))
                    setOnClickListener {
                        dialog.dismiss()
                    }
                }
                headerRow.addView(btnClose)

                root.addView(headerRow)

                val spacer = View(activity).apply {
                    layoutParams = LinearLayout.LayoutParams(
                        LinearLayout.LayoutParams.MATCH_PARENT,
                        dp(8)
                    )
                }
                root.addView(spacer)

                // WebView Container Frame
                val displayMetrics = activity.resources.displayMetrics
                val targetHeight = (displayMetrics.heightPixels * 0.70).toInt()

                val webViewContainer = FrameLayout(activity).apply {
                    layoutParams = LinearLayout.LayoutParams(
                        LinearLayout.LayoutParams.MATCH_PARENT,
                        targetHeight
                    )
                    setBackgroundColor(Color.WHITE)
                }

                webViewContainer.addView(
                    wv,
                    FrameLayout.LayoutParams(
                        FrameLayout.LayoutParams.MATCH_PARENT,
                        FrameLayout.LayoutParams.MATCH_PARENT
                    )
                )

                root.addView(webViewContainer)
                dialog.setContentView(root)

                dialog.behavior.apply {
                    state = BottomSheetBehavior.STATE_EXPANDED
                    skipCollapsed = true
                    isDraggable = true
                }

                dialog.setOnDismissListener {
                    (wv.parent as? ViewGroup)?.removeView(wv)
                    activeDialog = null
                    Log.i(TAG, "ChatGPT verification bottom sheet dismissed.")
                }

                dialog.show()
                Log.i(TAG, "Displayed ChatGPT verification bottom sheet for: $reason")
            } catch (e: Exception) {
                Log.e(TAG, "Error displaying verification bottom sheet: ${e.message}", e)
            }
        }
    }

    /**
     * Dismisses the verification BottomSheet if currently active.
     */
    fun dismissVerificationSheet() {
        mainHandler.post {
            try {
                if (activeDialog?.isShowing == true) {
                    val activity = weakActivity?.get()
                    if (activity != null && !activity.isFinishing && !activity.isDestroyed) {
                        Toast.makeText(activity, "✓ ChatGPT Verified! Resuming...", Toast.LENGTH_SHORT).show()
                    }
                    activeDialog?.dismiss()
                    activeDialog = null
                }
                CookieManager.getInstance().flush()
            } catch (e: Exception) {
                Log.w(TAG, "Error dismissing verification bottom sheet: ${e.message}")
            }
        }
    }

    private fun triggerVerificationPrompt(reason: String) {
        mainHandler.post {
            val act = weakActivity?.get()
            if (act != null && !act.isFinishing && !act.isDestroyed) {
                showVerificationSheet(act, reason)
            } else {
                Log.w(TAG, "Verification needed ($reason) but no active Activity is attached.")
            }
        }
    }

    /**
     * Sends a natural language query or research goal to ChatGPT and suspends
     * until the full streamed response is completed.
     */
    suspend fun ask(
        context: Context? = null,
        prompt: String,
        timeoutMs: Long = 60000L
    ): Result<String> = executionMutex.withLock {
        val cleanPrompt = prompt.trim()
        if (cleanPrompt.isBlank()) {
            return Result.failure(IllegalArgumentException("Prompt cannot be empty."))
        }

        if (context is Activity) {
            setVerificationActivity(context)
        }

        val ctx = context?.applicationContext ?: appContext
        if (ctx != null && headlessWebView == null) {
            init(ctx)
        }

        return withTimeoutOrNull(timeoutMs) {
            suspendCancellableCoroutine { continuation ->
                activeContinuation = { result ->
                    if (continuation.isActive) {
                        continuation.resume(result)
                    }
                }

                continuation.invokeOnCancellation {
                    activeContinuation = null
                    pendingPrompt = null
                }

                mainHandler.post {
                    val wv = headlessWebView
                    if (wv == null) {
                        dispatchResult(Result.failure(IllegalStateException("WebView not initialized. Context required.")))
                        return@post
                    }

                    if (!isEngineReady) {
                        pendingPrompt = cleanPrompt
                        if (!isPageLoaded) {
                            Log.i(TAG, "Page not yet ready, navigating to $CHATGPT_URL...")
                            wv.loadUrl(CHATGPT_URL)
                        } else {
                            injectReadyWatcher(wv)
                        }
                    } else {
                        executePromptInjection(wv, cleanPrompt)
                    }
                }
            }
        } ?: Result.failure(Exception("ChatGPT response timed out after ${timeoutMs / 1000}s."))
    }

    private fun injectReadyWatcher(view: WebView?) {
        val watcherScript = """
            (function() {
                let attempts = 0;
                const checkReady = setInterval(function() {
                    attempts++;
                    const title = document.title || "";
                    const isCf = title.includes("Just a moment") ||
                                 title.includes("Cloudflare") ||
                                 title.includes("Attention Required") ||
                                 !!document.querySelector('.challenge-running') ||
                                 !!document.querySelector('#challenge-stage') ||
                                 !!document.querySelector('iframe[src*="cloudflare"]') ||
                                 !!document.querySelector('iframe[src*="turnstile"]') ||
                                 !!document.querySelector('#cf-turnstile') ||
                                 !!document.querySelector('.cf-turnstile');

                    if (isCf && attempts % 6 === 1) {
                        window.ZipperBridge.onVerificationNeeded("Cloudflare / Bot Challenge Detected");
                    }

                    const input = document.getElementById("prompt-textarea") ||
                                  document.querySelector('textarea[data-id="root"]') ||
                                  document.querySelector('div[contenteditable="true"]') ||
                                  document.querySelector('div[role="textbox"]');

                    if (input) {
                        clearInterval(checkReady);
                        window.ZipperBridge.onEngineReady();
                    } else if (attempts > 30) {
                        clearInterval(checkReady);
                        const text = (document.body ? document.body.innerText || "" : "");
                        if (text.includes("Log in") || text.includes("Sign up") || isCf) {
                            window.ZipperBridge.onVerificationNeeded("Login or Cloudflare Check");
                        }
                    }
                }, 300);
            })();
        """.trimIndent()
        view?.evaluateJavascript(watcherScript, null)
    }

    private fun executePromptInjection(webView: WebView, prompt: String) {
        val escapedPrompt = prompt
            .replace("\\", "\\\\")
            .replace("\"", "\\\"")
            .replace("\'", "\\\'")
            .replace("\n", "\\n")
            .replace("\r", "")

        val script = """
            (function() {
                try {
                    let attempts = 0;
                    const maxAttempts = 60;

                    function findInput() {
                        return document.getElementById("prompt-textarea") ||
                               document.querySelector('textarea[data-id="root"]') ||
                               document.querySelector('div[contenteditable="true"]') ||
                               document.querySelector('div[role="textbox"]') ||
                               document.querySelector('p[data-placeholder]') ||
                               document.querySelector('textarea');
                    }

                    function findAssistantMessages() {
                        let nodes = document.querySelectorAll('div[data-message-author-role="assistant"]');
                        if (nodes.length > 0) return nodes;

                        nodes = document.querySelectorAll('article div.markdown, article div.prose, [class*="agent-turn"] .markdown, [class*="agent-turn"] .prose');
                        if (nodes.length > 0) return nodes;

                        nodes = document.querySelectorAll('article[data-testid^="conversation-turn-"]');
                        if (nodes.length > 1) {
                            return [nodes[nodes.length - 1]];
                        }

                        nodes = document.querySelectorAll('main div.markdown, main [class*="markdown"]');
                        if (nodes.length > 0) return nodes;

                        return [];
                    }

                    function dismissGuestModals() {
                        try {
                            const buttons = Array.from(document.querySelectorAll('button, a'));
                            const stayLoggedOut = buttons.find(b => b.textContent && (
                                b.textContent.toLowerCase().includes('stay logged out') ||
                                b.textContent.toLowerCase().includes('maybe later') ||
                                b.textContent.toLowerCase().includes('dismiss')
                            ));
                            if (stayLoggedOut) {
                                stayLoggedOut.click();
                                window.ZipperBridge.onLog("Dismissed guest modal: " + stayLoggedOut.textContent.trim());
                            }
                            const cookieAccept = document.getElementById('onetrust-accept-btn-handler')
                                              || buttons.find(b => b.textContent && (b.textContent.trim().toLowerCase() === 'accept all' || b.textContent.trim().toLowerCase() === 'accept'));
                            if (cookieAccept) {
                                cookieAccept.click();
                                window.ZipperBridge.onLog("Dismissed cookie consent");
                            }
                        } catch(e) {}
                    }

                    function typeAndSend() {
                        dismissGuestModals();
                        const input = findInput();
                        if (!input) {
                            attempts++;
                            if (attempts < maxAttempts) {
                                setTimeout(typeAndSend, 300);
                            } else {
                                window.ZipperBridge.onVerificationNeeded("Input field not found after timeout");
                                window.ZipperBridge.onResponseError("TIMEOUT_FINDING_INPUT");
                            }
                            return;
                        }

                        window.ZipperBridge.onLog("Found input: " + input.tagName + " (id=" + input.id + ")");
                        input.focus();

                        const isEditableDiv = input.isContentEditable || input.getAttribute('contenteditable') === 'true' || input.tagName.toLowerCase() === 'div';

                        if (isEditableDiv) {
                            try {
                                document.execCommand('selectAll', false, null);
                                document.execCommand('insertText', false, "$escapedPrompt");
                            } catch(e) {}
                            if (!input.innerText || !input.innerText.includes("$escapedPrompt")) {
                                input.innerText = "$escapedPrompt";
                            }
                            input.dispatchEvent(new Event('input', { bubbles: true }));
                            input.dispatchEvent(new Event('change', { bubbles: true }));
                        } else {
                            const proto = Object.getPrototypeOf(input);
                            const descriptor = Object.getOwnPropertyDescriptor(proto, 'value')
                                || Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value');

                            if (descriptor && descriptor.set) {
                                descriptor.set.call(input, "$escapedPrompt");
                            } else {
                                input.value = "$escapedPrompt";
                                try {
                                    document.execCommand('insertText', false, "$escapedPrompt");
                                } catch(e) {}
                            }
                            input.dispatchEvent(new Event('input', { bubbles: true }));
                            input.dispatchEvent(new Event('change', { bubbles: true }));
                        }

                        let sendAttempts = 0;
                        function trySend() {
                            sendAttempts++;
                            const sendBtn = document.querySelector('button[data-testid="send-button"]')
                                         || document.querySelector('button[aria-label="Send prompt"]')
                                         || document.querySelector('button[aria-label="Send message"]')
                                         || document.querySelector('button[data-testid="fruitjuice-send-button"]')
                                         || document.querySelector('form button[type="submit"]');

                            const isSendDisabled = sendBtn ? sendBtn.disabled : true;
                            window.ZipperBridge.onLog("Dispatching send attempt " + sendAttempts + ": btn=" + !!sendBtn + " (disabled=" + isSendDisabled + ")");

                            if (sendBtn && !sendBtn.disabled) {
                                sendBtn.click();
                                startResponseObserver();
                            } else if (sendAttempts < 6) {
                                setTimeout(trySend, 250);
                            } else {
                                const enterDown = new KeyboardEvent('keydown', { key: 'Enter', code: 'Enter', keyCode: 13, which: 13, bubbles: true });
                                input.dispatchEvent(enterDown);
                                startResponseObserver();
                            }
                        }

                        setTimeout(trySend, 300);
                    }

                    function startResponseObserver() {
                        let lastText = "";
                        let stableCount = 0;
                        const startTime = Date.now();
                        let loggedFirst = false;

                        const pollTimer = setInterval(function() {
                            const stopBtn = document.querySelector('button[data-testid="stop-button"]')
                                         || document.querySelector('button[aria-label="Stop streaming"]')
                                         || document.querySelector('button[aria-label="Stop generating"]');
                            const streamingEl = document.querySelector('.result-streaming');
                            const answers = findAssistantMessages();

                            if (!loggedFirst) {
                                window.ZipperBridge.onLog("ResponseObserver active. Answers found=" + answers.length);
                                loggedFirst = true;
                            }

                            if (answers.length > 0) {
                                const currentAnswer = answers[answers.length - 1].innerText.trim();
                                if (currentAnswer.length > 0 && currentAnswer === lastText) {
                                    stableCount++;
                                } else {
                                    stableCount = 0;
                                    lastText = currentAnswer;
                                }

                                const isDone = (!stopBtn && !streamingEl && currentAnswer.length > 0 && stableCount >= 2);

                                if (isDone) {
                                    clearInterval(pollTimer);
                                    window.ZipperBridge.onResponseSuccess(currentAnswer);
                                    return;
                                }
                            }

                            // 55-second inner safety timeout
                            if (Date.now() - startTime > 55000) {
                                clearInterval(pollTimer);
                                if (lastText.length > 0) {
                                    window.ZipperBridge.onResponseSuccess(lastText);
                                } else {
                                    const bodySnip = (document.body.innerText || "").substring(0, 200).replace(/\n/g, ' ');
                                    window.ZipperBridge.onLog("TIMEOUT body preview: " + bodySnip);
                                    window.ZipperBridge.onResponseError("TIMEOUT_NO_RESPONSE");
                                }
                            }
                        }, 350);
                    }

                    typeAndSend();
                } catch(err) {
                    window.ZipperBridge.onResponseError("EXCEPTION: " + err.message);
                }
            })();
        """.trimIndent()

        webView.evaluateJavascript(script, null)
    }

    private fun dispatchResult(result: Result<String>) {
        val cont = activeContinuation
        activeContinuation = null
        cont?.invoke(result)
    }

    /**
     * Clears cookies and resets the background session.
     */
    fun resetSession() {
        mainHandler.post {
            headlessWebView?.loadUrl("about:blank")
            isPageLoaded = false
            isEngineReady = false
            pendingPrompt = null
        }
    }

    /**
     * Android JavascriptInterface bridge receiving direct callbacks from the webpage.
     */
    private class BridgeInterface {
        @JavascriptInterface
        fun onLog(msg: String) {
            Log.d(TAG, "DOM: $msg")
        }

        @JavascriptInterface
        fun onVerificationNeeded(reason: String) {
            Log.w(TAG, "Bridge reports verification needed: $reason")
            mainHandler.post {
                triggerVerificationPrompt(reason)
            }
        }

        @JavascriptInterface
        fun onEngineReady() {
            Log.i(TAG, "ChatGPT guest engine is ready for input.")
            isEngineReady = true
            mainHandler.post {
                dismissVerificationSheet()
                val prompt = pendingPrompt
                val wv = headlessWebView
                if (prompt != null && wv != null) {
                    pendingPrompt = null
                    Log.i(TAG, "Executing deferred prompt injection after verification ready: ${prompt.take(30)}...")
                    executePromptInjection(wv, prompt)
                }
            }
        }

        @JavascriptInterface
        fun onResponseSuccess(answer: String) {
            Log.i(TAG, "ChatGPT successfully returned answer (${answer.length} chars)")
            dispatchResult(Result.success(answer))
        }

        @JavascriptInterface
        fun onResponseError(errorMessage: String) {
            Log.e(TAG, "ChatGPT bridge reported error: $errorMessage")
            dispatchResult(Result.failure(Exception("ChatGPT Web Error: $errorMessage")))
        }
    }
}
