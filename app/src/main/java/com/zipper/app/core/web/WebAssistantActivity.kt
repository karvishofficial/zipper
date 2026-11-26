package com.zipper.app.core.web

import android.content.Intent
import android.content.res.ColorStateList
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.view.View
import android.view.ViewGroup
import android.view.inputmethod.EditorInfo
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.activity.OnBackPressedCallback
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.zipper.app.R
import com.zipper.app.core.web.features.ArticleReaderEngine
import com.zipper.app.core.web.features.CartArbitrageEngine
import com.zipper.app.core.web.features.PnrTrackerEngine
import com.zipper.app.core.web.features.PriceComparisonEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.io.File
import kotlinx.coroutines.withContext
import java.net.URLEncoder
import java.util.Locale
import android.util.Log

class WebAssistantActivity : AppCompatActivity() {

    private val TAG = "WebAssistantActivity"
    private lateinit var btnBack: ImageButton
    private lateinit var tvWebStatus: TextView
    private lateinit var btnToggleAiMode: Button
    private lateinit var btnTogglePeek: Button
    private lateinit var btnApiSettings: Button
    private lateinit var flPeekContainer: FrameLayout
    private lateinit var flWebViewPlaceholder: FrameLayout
    private lateinit var rvChatMessages: RecyclerView
    private lateinit var etWebPrompt: EditText
    private lateinit var btnWebVoice: ImageButton
    private lateinit var btnWebSend: ImageButton
    private lateinit var chipReplay: TextView

    private lateinit var llFullscreenWebOverlay: LinearLayout
    private lateinit var flFullscreenPlaceholder: FrameLayout
    private lateinit var tvBrowserUrl: TextView
    private lateinit var btnBrowserClose: ImageButton
    private lateinit var btnBrowserRefresh: ImageButton
    private lateinit var btnBrowserExternal: ImageButton
    private lateinit var btnBrowserSavePdf: Button
    private lateinit var btnBrowserNavBack: ImageButton
    private lateinit var btnBrowserNavForward: ImageButton
    private lateinit var btnBackToChat: Button
    private lateinit var btnPeekSavePdf: Button
    private lateinit var btnExpandFullscreen: Button
    private lateinit var btnClosePeek: ImageButton
    private lateinit var tvPeekPreviewUrl: TextView

    private lateinit var chatAdapter: WebChatAdapter
    private lateinit var webController: HiddenWebController
    private lateinit var configManager: WebConfigManager
    private lateinit var replayStore: WebReplayStore
    private val webLlmClient = WebLlmClient()
    private val webOnDeviceEngine by lazy { WebOnDeviceEngine(this) }
    private var speechRecognizer: SpeechRecognizer? = null
    private var textToSpeech: TextToSpeech? = null
    private var isSpeaking = false
    private var isPeekVisible = false
    private var isAutoReplayMode = false
    private var isAiModeEnabled = true

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_web_assistant)

        isAiModeEnabled = getSharedPreferences("karvish_web_prefs", MODE_PRIVATE)
            .getBoolean("is_ai_mode_enabled", true)

        textToSpeech = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                textToSpeech?.language = Locale.ENGLISH
            }
        }

        webController = HiddenWebController(this)
        configManager = WebConfigManager(this)
        replayStore = WebReplayStore(this)
        com.zipper.app.core.web.chatgpt.ChatGptHeadlessBridge.init(this)

        initViews()
        setupChat()
        setupListeners()
        setupSpeechRecognizer()
        updateApiButtonUi()
        updateAiModeButtonUi()
        refreshReplayChip()

        webController.onUrlChanged = { newUrl ->
            runOnUiThread {
                tvBrowserUrl.text = newUrl
                tvPeekPreviewUrl.text = newUrl
            }
        }

        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (llFullscreenWebOverlay.visibility == View.VISIBLE) {
                    val wv = webController.getWebView()
                    if (wv != null && wv.canGoBack()) {
                        wv.goBack()
                    } else {
                        hideFullscreenWeb()
                    }
                } else if (isPeekVisible) {
                    hidePeekView()
                } else {
                    isEnabled = false
                    onBackPressedDispatcher.onBackPressed()
                }
            }
        })

        // Welcome message
        chatAdapter.addMessage(
            WebChatMessage(
                type = ChatMessageType.ASSISTANT,
                text = "Hello! I am Karvish Web Assistant. I can research products, compare prices across stores, check train PNRs, or lookup live web information silently in the background without hijacking your phone screen.\n\nTry tapping a suggestion below or ask me anything!"
            )
        )

        handleIncomingIntent(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleIncomingIntent(intent)
    }

    private fun handleIncomingIntent(intent: Intent?) {
        if (intent == null) return
        val autoReplay = intent.getBooleanExtra("auto_replay", false)
        val sharedText = if (intent.action == Intent.ACTION_SEND && intent.type?.startsWith("text/") == true) {
            intent.getStringExtra(Intent.EXTRA_TEXT)
        } else null
        val query = sharedText
            ?: intent.getStringExtra("query")
            ?: intent.getStringExtra("auto_replay_query")
            ?: intent.getStringExtra("flow_title")
        if (autoReplay || !query.isNullOrBlank()) {
            isAutoReplayMode = autoReplay
            val execQuery = if (!query.isNullOrBlank()) query else replayStore.getLatestFlow()?.originalQuery
            if (!execQuery.isNullOrBlank()) {
                executeWebQuery(execQuery)
            }
        }
    }

    private fun initViews() {
        btnBack = findViewById(R.id.btnBack)
        tvWebStatus = findViewById(R.id.tvWebStatus)
        btnToggleAiMode = findViewById(R.id.btnToggleAiMode)
        btnTogglePeek = findViewById(R.id.btnTogglePeek)
        btnApiSettings = findViewById(R.id.btnApiSettings)
        flPeekContainer = findViewById(R.id.flPeekContainer)
        flWebViewPlaceholder = findViewById(R.id.flWebViewPlaceholder)
        rvChatMessages = findViewById(R.id.rvChatMessages)
        etWebPrompt = findViewById(R.id.etWebPrompt)
        btnWebVoice = findViewById(R.id.btnWebVoice)
        btnWebSend = findViewById(R.id.btnWebSend)
        chipReplay = findViewById(R.id.chipReplay)

        llFullscreenWebOverlay = findViewById(R.id.llFullscreenWebOverlay)
        flFullscreenPlaceholder = findViewById(R.id.flFullscreenPlaceholder)
        tvBrowserUrl = findViewById(R.id.tvBrowserUrl)
        btnBrowserClose = findViewById(R.id.btnBrowserClose)
        btnBrowserRefresh = findViewById(R.id.btnBrowserRefresh)
        btnBrowserExternal = findViewById(R.id.btnBrowserExternal)
        btnBrowserSavePdf = findViewById(R.id.btnBrowserSavePdf)
        btnBrowserNavBack = findViewById(R.id.btnBrowserNavBack)
        btnBrowserNavForward = findViewById(R.id.btnBrowserNavForward)
        btnBackToChat = findViewById(R.id.btnBackToChat)
        btnPeekSavePdf = findViewById(R.id.btnPeekSavePdf)
        btnExpandFullscreen = findViewById(R.id.btnExpandFullscreen)
        btnClosePeek = findViewById(R.id.btnClosePeek)
        tvPeekPreviewUrl = findViewById(R.id.tvPeekPreviewUrl)
    }

    private fun setupChat() {
        chatAdapter = WebChatAdapter(
            onOpenWeb = { url ->
                if (url.startsWith("zipper://relay")) {
                    try {
                        val uri = Uri.parse(url)
                        val store = uri.getQueryParameter("store") ?: "Zepto"
                        val q = uri.getQueryParameter("q") ?: ""
                        val parsed = ShoppingListParser.parseShoppingList(q)
                        val items = if (parsed.size > 1) parsed.map { it.normalizedQuery } else WebProductFinder.parseMultiProductItems(q)
                        lifecycleScope.launch {
                            executeMultiProductRelay(q, items, store)
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Failed to parse zipper relay URL: ${e.message}", e)
                    }
                } else if (url.startsWith("zipper://settings/notifications")) {
                    try {
                        startActivity(com.zipper.app.core.notification.NotificationQueryEngine.getPermissionSettingsIntent())
                    } catch (e: Exception) {
                        Toast.makeText(this, "Could not open notification settings: ${e.message}", Toast.LENGTH_SHORT).show()
                    }
                } else if (url.startsWith("zipper://copy")) {
                    try {
                        val textToCopy = Uri.parse(url).getQueryParameter("text") ?: ""
                        val clipboard = getSystemService(android.content.Context.CLIPBOARD_SERVICE) as android.content.ClipboardManager
                        clipboard.setPrimaryClip(android.content.ClipData.newPlainText("OTP", textToCopy))
                        Toast.makeText(this, "📋 Copied '$textToCopy' to clipboard!", Toast.LENGTH_SHORT).show()
                    } catch (e: Exception) {
                        Toast.makeText(this, "Failed to copy: ${e.message}", Toast.LENGTH_SHORT).show()
                    }
                } else if (url.startsWith("zipper://split")) {
                    try {
                        val uri = Uri.parse(url)
                        val freshStore = uri.getQueryParameter("freshStore") ?: "Zepto"
                        val bulkStore = uri.getQueryParameter("bulkStore") ?: "Amazon"
                        val freshItems = uri.getQueryParameter("freshItems")?.split(";")?.filter { it.isNotBlank() } ?: emptyList()
                        val bulkItems = uri.getQueryParameter("bulkItems")?.split(";")?.filter { it.isNotBlank() } ?: emptyList()
                        lifecycleScope.launch {
                            executeSmartSplitRelay(freshStore, freshItems, bulkStore, bulkItems)
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Failed to parse zipper split URL: ${e.message}", e)
                    }
                } else {
                    showFullscreenWeb(url)
                }
            },
            onOpenApp = { pkg ->
                val launchIntent = packageManager.getLaunchIntentForPackage(pkg)
                if (launchIntent != null) {
                    startActivity(launchIntent)
                } else {
                    Toast.makeText(this, "App '$pkg' is not installed", Toast.LENGTH_SHORT).show()
                }
            },
            onFeedbackSuccess = { msg ->
                val query = msg.resultData?.originalGoal.orEmpty()
                val targetUrl = msg.resultData?.webUrl.orEmpty()
                val flowTitle = if (targetUrl.contains("rrce", true) || query.contains("rrce", true)) {
                    "RRCE Student Login"
                } else {
                    "Saved Web Flow"
                }
                replayStore.saveFlow(flowTitle, targetUrl, query, 10)
                WebDittoBridge.saveToDittoAndDemonstrations(
                    context = this,
                    title = flowTitle,
                    goal = query,
                    url = targetUrl
                )
                Toast.makeText(this, "🎉 Flow saved to Quick Replay & DITTO! Re-run anytime.", Toast.LENGTH_LONG).show()
                refreshReplayChip()
            },
            onFeedbackFail = { msg ->
                val query = msg.resultData?.originalGoal
                if (!query.isNullOrBlank()) {
                    Toast.makeText(this, "🔄 Retrying autonomous execution with refreshed CAPTCHA...", Toast.LENGTH_SHORT).show()
                    executeWebQuery(query)
                }
            },
            onReadAloud = { text ->
                toggleReadAloud(text)
            }
        )
        val layoutManager = LinearLayoutManager(this).apply {
            stackFromEnd = true
        }
        rvChatMessages.layoutManager = layoutManager
        rvChatMessages.adapter = chatAdapter
    }

    private fun refreshReplayChip() {
        val latest = replayStore.getLatestFlow()
        if (latest != null) {
            chipReplay.visibility = View.VISIBLE
            chipReplay.text = "▶ Replay: ${latest.title}"
            chipReplay.setOnClickListener {
                executeWebQuery(latest.originalQuery)
            }
        } else {
            chipReplay.visibility = View.GONE
        }
    }

    private fun showPeekView() {
        isPeekVisible = true
        val wv = webController.getWebView()
        if (wv != null) {
            (wv.parent as? ViewGroup)?.removeView(wv)
            flWebViewPlaceholder.removeAllViews()
            flWebViewPlaceholder.addView(
                wv,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT
                )
            )
            tvPeekPreviewUrl.text = wv.url ?: "LIVE PREVIEW"
        }
        flPeekContainer.visibility = View.VISIBLE
        btnTogglePeek.text = "❌ Hide"
    }

    private fun hidePeekView() {
        isPeekVisible = false
        flPeekContainer.visibility = View.GONE
        btnTogglePeek.text = "👁️ Peek"
    }

    private fun showFullscreenWeb(url: String? = null) {
        val wv = webController.getWebView()
        if (wv != null) {
            (wv.parent as? ViewGroup)?.removeView(wv)
            flFullscreenPlaceholder.removeAllViews()
            flFullscreenPlaceholder.addView(
                wv,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT
                )
            )
            tvBrowserUrl.text = wv.url ?: url ?: "Live Session"
            webController.flushCookies()
            val currentUrl = wv.url ?: url ?: ""
            if (currentUrl.isNotBlank()) {
                val sanitizeJs = WebSuiteManager.buildPageSanitizationScript(currentUrl)
                if (sanitizeJs.isNotBlank()) {
                    wv.evaluateJavascript(sanitizeJs, null)
                }
            }
        } else if (!url.isNullOrBlank()) {
            lifecycleScope.launch {
                webController.navigate(url)
                showFullscreenWeb(url)
            }
            return
        }
        llFullscreenWebOverlay.visibility = View.VISIBLE
    }

    private fun hideFullscreenWeb() {
        llFullscreenWebOverlay.visibility = View.GONE
        val wv = webController.getWebView()
        (wv?.parent as? ViewGroup)?.removeView(wv)
        if (isPeekVisible) {
            showPeekView()
        }
    }

    private fun setupListeners() {
        btnBack.setOnClickListener { finish() }

        btnTogglePeek.setOnClickListener {
            if (isPeekVisible) hidePeekView() else showPeekView()
        }

        btnPeekSavePdf.setOnClickListener {
            exportCurrentPagePdf()
        }

        btnBrowserSavePdf.setOnClickListener {
            exportCurrentPagePdf()
        }

        btnExpandFullscreen.setOnClickListener {
            showFullscreenWeb()
        }

        btnClosePeek.setOnClickListener {
            hidePeekView()
        }

        btnBrowserClose.setOnClickListener {
            hideFullscreenWeb()
        }

        btnBackToChat.setOnClickListener {
            hideFullscreenWeb()
        }

        btnBrowserRefresh.setOnClickListener {
            webController.getWebView()?.reload()
        }

        btnBrowserNavBack.setOnClickListener {
            val wv = webController.getWebView()
            if (wv != null && wv.canGoBack()) {
                wv.goBack()
            } else {
                Toast.makeText(this, "No previous page in history", Toast.LENGTH_SHORT).show()
            }
        }

        btnBrowserNavForward.setOnClickListener {
            val wv = webController.getWebView()
            if (wv != null && wv.canGoForward()) {
                wv.goForward()
            } else {
                Toast.makeText(this, "No next page in history", Toast.LENGTH_SHORT).show()
            }
        }

        btnBrowserExternal.setOnClickListener {
            val currentUrl = webController.getWebView()?.url
            if (!currentUrl.isNullOrBlank()) {
                try {
                    startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(currentUrl)))
                    Toast.makeText(this, "Opening in external browser...", Toast.LENGTH_SHORT).show()
                } catch (e: Exception) {
                    Toast.makeText(this, "Could not open browser: ${e.message}", Toast.LENGTH_SHORT).show()
                }
            }
        }

        btnToggleAiMode.setOnClickListener {
            isAiModeEnabled = !isAiModeEnabled
            getSharedPreferences("karvish_web_prefs", MODE_PRIVATE)
                .edit()
                .putBoolean("is_ai_mode_enabled", isAiModeEnabled)
                .apply()
            updateAiModeButtonUi()
            val modeMsg = if (isAiModeEnabled) {
                "⚡ AI Mode: ON (Groq LLM for multi-step reasoning)"
            } else {
                "⚡ AI Mode: OFF (Fast Local Mode, <2s, 0 tokens)"
            }
            Toast.makeText(this, modeMsg, Toast.LENGTH_SHORT).show()
        }

        btnApiSettings.setOnClickListener {
            showApiSettingsDialog()
        }

        btnWebSend.setOnClickListener {
            val query = etWebPrompt.text.toString().trim()
            if (query.isNotBlank()) {
                executeWebQuery(query)
                etWebPrompt.setText("")
            }
        }

        etWebPrompt.setOnEditorActionListener { _, actionId, _ ->
            if (actionId == EditorInfo.IME_ACTION_SEND) {
                val query = etWebPrompt.text.toString().trim()
                if (query.isNotBlank()) {
                    executeWebQuery(query)
                    etWebPrompt.setText("")
                }
                true
            } else {
                false
            }
        }

        btnWebVoice.setOnClickListener {
            toggleVoiceInput()
        }

        // Suggestion Chips
        findViewById<TextView>(R.id.chipTomato).setOnClickListener {
            executeWebQuery("Find the price of 1kg fresh tomatoes")
        }
        findViewById<TextView>(R.id.chipIPhone).setOnClickListener {
            executeWebQuery("Find the lowest price for iPhone 15 128GB")
        }
        findViewById<TextView>(R.id.chipFlight).setOnClickListener {
            executeWebQuery("Flights from Bangalore to Delhi tomorrow")
        }
        findViewById<TextView>(R.id.chipGold).setOnClickListener {
            executeWebQuery("What is 22k gold rate in Bangalore today?")
        }
        findViewById<TextView>(R.id.chipCompare)?.setOnClickListener {
            executeWebQuery("Compare iPhone 15 price across Amazon and Flipkart")
        }
        findViewById<TextView>(R.id.chipPNR).setOnClickListener {
            executeWebQuery("Check IRCTC PNR status")
        }
    }

    private fun executeWebQuery(rawQuery: String) {
        val query = rawQuery.trim('\'', '"', ' ').trim()
        chatAdapter.addMessage(WebChatMessage(type = ChatMessageType.USER, text = query))
        scrollToBottom()

        tvWebStatus.text = "🔄 BROWSING"
        tvWebStatus.setTextColor(android.graphics.Color.parseColor("#F59E0B"))

        lifecycleScope.launch {
            try {
                // A. Encrypted Hardware Vault Command
                if (query.startsWith("vault", ignoreCase = true)) {
                    val vault = WebVaultManager(this@WebAssistantActivity)
                    val lowerQ = query.lowercase()
                    if (lowerQ.startsWith("vault set ")) {
                        val parts = query.removePrefix("vault set ").removePrefix("Vault set ").trim().split(" ", limit = 2)
                        if (parts.size == 2) {
                            vault.saveField(parts[0].trim().lowercase(), parts[1].trim())
                            chatAdapter.addMessage(
                                WebChatMessage(
                                    type = ChatMessageType.ASSISTANT,
                                    text = "🔐 **Encrypted Vault Updated!**\nSaved key **`${parts[0]}`** securely in hardware Keystore."
                                )
                            )
                        } else {
                            chatAdapter.addMessage(
                                WebChatMessage(
                                    type = ChatMessageType.ASSISTANT,
                                    text = "⚠️ Usage: `vault set <key> <value>` (e.g., `vault set usn 1RR22CS001` or `vault set dob_day 29`)"
                                )
                            )
                        }
                    } else {
                        val configured = vault.getAllConfiguredKeys()
                        val msg = if (configured.isEmpty()) {
                            "🔐 **Encrypted Form Vault is empty.**\nTo save credentials, type: `vault set usn 1RR22CS001` or `vault set dob_day 29`"
                        } else {
                            "🔐 **Encrypted Form Vault Active Keys:**\n" + configured.joinToString("\n") { "• **`$it`** (Stored securely in Keystore)" } + "\n\n*These fields auto-fill web forms in <50ms with 0 tokens in both AI: ON and AI: OFF modes.*"
                        }
                        chatAdapter.addMessage(WebChatMessage(type = ChatMessageType.ASSISTANT, text = msg))
                    }
                    tvWebStatus.text = "🟢 READY"
                    tvWebStatus.setTextColor(android.graphics.Color.parseColor("#34D399"))
                    scrollToBottom()
                    return@launch
                }

                // B. Background Passive Watcher Command
                if (query.startsWith("watch", ignoreCase = true)) {
                    val watcher = WebWatcherManager(this@WebAssistantActivity)
                    val qClean = query.substring(5).trim()
                    val urlMatch = Regex("""https?://[^\s]+""").find(qClean)?.value
                    if (urlMatch != null) {
                        val numMatch = Regex("""\b\d+(?:\.\d+)?\b""").find(qClean.replace(urlMatch, ""))?.value?.toDoubleOrNull()
                        val item = WebWatchItem(
                            id = java.util.UUID.randomUUID().toString().take(8),
                            title = if (numMatch != null) "Price Watcher" else "Status Watcher",
                            url = urlMatch,
                            watchType = if (numMatch != null) WatchType.PRICE_DROP else WatchType.STATUS_CHANGE,
                            targetThreshold = numMatch
                        )
                        watcher.addWatchItem(item)
                        val ack = if (numMatch != null) {
                            "👁️ **Background Price Watcher Activated!**\nWatching `$urlMatch` for price drop to **₹$numMatch**.\nRuns silently in background every 2h without waking screen."
                        } else {
                            "👁️ **Background Status Watcher Activated!**\nWatching `$urlMatch` for updates/confirmations.\nRuns silently in background every 2h without waking screen."
                        }
                        chatAdapter.addMessage(WebChatMessage(type = ChatMessageType.ASSISTANT, text = ack))
                    } else if (qClean.contains("list") || qClean.contains("show") || qClean.isBlank()) {
                        val allWatches = watcher.getAllWatchItems()
                        val msg = if (allWatches.isEmpty()) "👁️ No active background watches."
                        else "👁️ **Active Background Watches:**\n" + allWatches.joinToString("\n") { "• **${it.title}**: ${it.url} (Threshold: ${it.targetThreshold ?: "Status Change"})" }
                        chatAdapter.addMessage(WebChatMessage(type = ChatMessageType.ASSISTANT, text = msg))
                    } else {
                        chatAdapter.addMessage(WebChatMessage(type = ChatMessageType.ASSISTANT, text = "⚠️ Usage: `watch <url>` or `watch <url> 1499`"))
                    }
                    tvWebStatus.text = "🟢 READY"
                    tvWebStatus.setTextColor(android.graphics.Color.parseColor("#34D399"))
                    scrollToBottom()
                    return@launch
                }



                // 0B. Direct Media & Clean Document Extractor
                if (query.startsWith("download", ignoreCase = true) || query.startsWith("extract media", ignoreCase = true) || query.contains(".pdf", ignoreCase = true)) {
                    val url = ArticleReaderEngine.extractUrlFromQuery(query)
                    if (!url.isNullOrBlank()) {
                        chatAdapter.addMessage(
                            WebChatMessage(
                                type = ChatMessageType.PROGRESS_STEP,
                                text = "📥 Probing underlying streams, video playlists, and clean PDF documents..."
                            )
                        )
                        scrollToBottom()

                        webController.navigate(url)
                        val html = webController.getHtml()
                        val media = WebSuiteManager.mediaExtractor.extractMediaFromPage(html, url)

                        if (media.isNotEmpty()) {
                            val sb = StringBuilder()
                            sb.appendLine("### 📥 Extracted Media & Clean Files (${media.size})")
                            for (med in media) {
                                sb.appendLine("• **[${med.type}]** [${med.title}](${med.url}) (`${med.mimeType}`)")
                            }
                            chatAdapter.addMessage(
                                WebChatMessage(
                                    type = ChatMessageType.ASSISTANT,
                                    text = sb.toString(),
                                    resultData = WebResultData(
                                        title = media.first().title,
                                        snippet = "Direct download ready",
                                        webUrl = media.first().url,
                                        originalGoal = query
                                    )
                                )
                            )
                        } else {
                            chatAdapter.addMessage(
                                WebChatMessage(
                                    type = ChatMessageType.ASSISTANT,
                                    text = "ℹ️ No direct standalone media or PDF links discovered on this page."
                                )
                            )
                        }
                        tvWebStatus.text = "🟢 READY"
                        tvWebStatus.setTextColor(android.graphics.Color.parseColor("#34D399"))
                        scrollToBottom()
                        return@launch
                    }
                }

                // 0C. Offline Knowledge Vault
                if (query.equals("snapshots", true) || query.equals("vault snapshots", true) || query.equals("offline notes", true)) {
                    val snapshots = WebSuiteManager.snapshotVault.listSnapshots(this@WebAssistantActivity)
                    val text = if (snapshots.isEmpty()) {
                        "📚 **Offline Vault is empty.**\nTo save an offline snapshot, type: `save snapshot <url>`"
                    } else {
                        "### 📚 Offline Web Vault (${snapshots.size} Saved Documents)\n" +
                            snapshots.joinToString("\n\n") { snap ->
                                "• **${snap.title}**\n  *${snap.executiveSummary}*\n  Source: `${snap.sourceUrl}`"
                            }
                    }
                    chatAdapter.addMessage(WebChatMessage(type = ChatMessageType.ASSISTANT, text = text))
                    tvWebStatus.text = "🟢 READY"
                    tvWebStatus.setTextColor(android.graphics.Color.parseColor("#34D399"))
                    scrollToBottom()
                    return@launch
                }
                if (query.startsWith("save snapshot", ignoreCase = true) || query.startsWith("snapshot", ignoreCase = true)) {
                    val url = ArticleReaderEngine.extractUrlFromQuery(query) ?: webController.getWebView()?.url
                    if (!url.isNullOrBlank()) {
                        chatAdapter.addMessage(WebChatMessage(type = ChatMessageType.PROGRESS_STEP, text = "📸 Capturing offline Markdown snapshot into local storage..."))
                        scrollToBottom()

                        webController.navigate(url)
                        val html = webController.getHtml()
                        val title = webController.getWebView()?.title ?: "Web Snapshot"
                        val snap = WebSuiteManager.snapshotVault.saveSnapshot(this@WebAssistantActivity, title, url, html)

                        chatAdapter.addMessage(
                            WebChatMessage(
                                type = ChatMessageType.ASSISTANT,
                                text = "✅ **Snapshot Saved Offline!**\n**${snap.title}**\n\n*${snap.executiveSummary}*\n\nKey Points:\n" +
                                    snap.keyPoints.joinToString("\n") { "• $it" }
                            )
                        )
                        tvWebStatus.text = "🟢 READY"
                        tvWebStatus.setTextColor(android.graphics.Color.parseColor("#34D399"))
                        scrollToBottom()
                        return@launch
                    }
                }

                // 0D. Universal Indian Railways PNR & Courier Shipment Tracker
                if (PnrTrackerEngine.isTrackingQuery(query)) {
                    chatAdapter.addMessage(
                        WebChatMessage(
                            type = ChatMessageType.PROGRESS_STEP,
                            text = "🚆 Fetching live status for \"$query\"..."
                        )
                    )
                    scrollToBottom()

                    val trackResult = PnrTrackerEngine.track(query, webController)
                    chatAdapter.addMessage(
                        WebChatMessage(
                            type = ChatMessageType.ASSISTANT,
                            text = trackResult.detailsMarkdown,
                            resultData = WebResultData(
                                title = trackResult.title,
                                snippet = "Status: ${trackResult.status}",
                                webUrl = trackResult.webUrl,
                                originalGoal = query
                            )
                        )
                    )
                    tvWebStatus.text = "🟢 READY"
                    tvWebStatus.setTextColor(android.graphics.Color.parseColor("#34D399"))
                    scrollToBottom()
                    return@launch
                }

                // 0E. Cross-App Cart Arbitrageur (Zepto vs Blinkit vs Instamart)
                if (CartArbitrageEngine.isArbitrageQuery(query)) {
                    chatAdapter.addMessage(
                        WebChatMessage(
                            type = ChatMessageType.PROGRESS_STEP,
                            text = "🛒 Analyzing basket arbitrage across Zepto, Blinkit & Swiggy Instamart..."
                        )
                    )
                    scrollToBottom()

                    val report = CartArbitrageEngine.evaluateBasketArbitrage(fallbackText = query)
                    chatAdapter.addMessage(
                        WebChatMessage(
                            type = ChatMessageType.ASSISTANT,
                            text = report.summaryMarkdown,
                            resultData = report.cheapestStore?.let { store ->
                                WebResultData(
                                    title = "${store.storeName} (Cheapest Cart)",
                                    snippet = "Total: ₹${"%.0f".format(store.totalCost)} (Save ₹${"%.0f".format(report.maxSingleStoreSavings)})",
                                    webUrl = store.checkoutUrl,
                                    originalGoal = query
                                )
                            }
                        )
                    )
                    tvWebStatus.text = "🟢 READY"
                    tvWebStatus.setTextColor(android.graphics.Color.parseColor("#34D399"))
                    scrollToBottom()
                    return@launch
                }

                // 1. Multi-Store Price Comparison Engine
                if (PriceComparisonEngine.isComparisonQuery(query)) {
                    chatAdapter.addMessage(
                        WebChatMessage(
                            type = ChatMessageType.PROGRESS_STEP,
                            text = "📊 Comparing prices across stores for \"$query\"..."
                        )
                    )
                    scrollToBottom()

                    val compResult = PriceComparisonEngine.compareStores(query, webController, webOnDeviceEngine)
                    val bestDeal = compResult.bestOverallDeal
                    val appPkg = bestDeal?.let { WebDeepLinkManager.resolveAppPackage(it.url, query, context = this@WebAssistantActivity) }

                    chatAdapter.addMessage(
                        WebChatMessage(
                            type = ChatMessageType.ASSISTANT,
                            text = compResult.summaryMarkdown,
                            resultData = WebResultData(
                                title = bestDeal?.title ?: "Multi-Store Price Comparison",
                                price = bestDeal?.priceDisplay,
                                snippet = if (bestDeal != null) "Best deal on ${bestDeal.store}: ${bestDeal.title}" else "Compared across ${compResult.itemsByStore.size} stores",
                                webUrl = bestDeal?.url,
                                appPackage = appPkg,
                                rating = bestDeal?.rating,
                                storeName = bestDeal?.store,
                                originalGoal = query,
                                comparisonOffers = compResult.offers
                            )
                        )
                    )
                    tvWebStatus.text = "🟢 READY"
                    tvWebStatus.setTextColor(android.graphics.Color.parseColor("#34D399"))
                    scrollToBottom()
                    return@launch
                }

                // 2. Ad-Free Article Reader & Summarizer
                if (ArticleReaderEngine.isArticleQuery(query)) {
                    val targetUrl = ArticleReaderEngine.extractUrlFromQuery(query)
                    if (!targetUrl.isNullOrBlank()) {
                        chatAdapter.addMessage(
                            WebChatMessage(
                                type = ChatMessageType.PROGRESS_STEP,
                                text = "📰 Extracting clean article content and generating TL;DR..."
                            )
                        )
                        scrollToBottom()

                        val summaryResult = ArticleReaderEngine.readAndSummarize(targetUrl, webController)
                        chatAdapter.addMessage(
                            WebChatMessage(
                                type = ChatMessageType.ASSISTANT,
                                text = summaryResult.fullMarkdown,
                                resultData = WebResultData(
                                    title = summaryResult.title,
                                    snippet = summaryResult.executiveSummary,
                                    webUrl = targetUrl,
                                    originalGoal = query
                                )
                            )
                        )
                        tvWebStatus.text = "🟢 READY"
                        tvWebStatus.setTextColor(android.graphics.Color.parseColor("#34D399"))
                        scrollToBottom()
                        return@launch
                    }
                }

                // 2.5 Notification Life-Feed Intelligence Gate (Immune to UI Changes)
                if (com.zipper.app.core.notification.NotificationQueryEngine.isNotificationRelatedQuery(query)) {
                    if (!com.zipper.app.core.notification.NotificationQueryEngine.isPermissionGranted(this@WebAssistantActivity)) {
                        chatAdapter.addMessage(
                            WebChatMessage(
                                type = ChatMessageType.ASSISTANT,
                                text = "🔔 **Notification Access Required**\n\nTo view live OTPs, track deliveries, and generate executive summaries, Karvish needs Notification Listener permission.\n\n👉 [Tap Here to Enable Notification Access](zipper://settings/notifications)",
                                resultData = WebResultData(
                                    title = "Enable Notification Access",
                                    snippet = "Grant notification listener permission in Android Settings",
                                    webUrl = "zipper://settings/notifications",
                                    originalGoal = query
                                )
                            )
                        )
                        tvWebStatus.text = "🟢 READY"
                        tvWebStatus.setTextColor(android.graphics.Color.parseColor("#34D399"))
                        scrollToBottom()
                        return@launch
                    }

                    val localResult = com.zipper.app.core.notification.NotificationQueryEngine.handleLocalNotificationQuery(this@WebAssistantActivity, query)
                    if (localResult != null) {
                        if (isAiModeEnabled && localResult.requiresAiSynthesis && localResult.items.isNotEmpty()) {
                            chatAdapter.addMessage(
                                WebChatMessage(
                                    type = ChatMessageType.PROGRESS_STEP,
                                    text = "🧠 Synthesizing executive briefing from ${localResult.items.size} notifications via AI..."
                                )
                            )
                            scrollToBottom()

                            val aiPrompt = com.zipper.app.core.notification.NotificationQueryEngine.buildAiSynthesisPrompt(localResult.items)
                            val aiResponse = webLlmClient.generateCompletion(aiPrompt, configManager.getConfig()) ?: localResult.markdownResponse
                            chatAdapter.addMessage(
                                WebChatMessage(
                                    type = ChatMessageType.ASSISTANT,
                                    text = aiResponse,
                                    resultData = WebResultData(
                                        title = "AI Executive Notification Briefing",
                                        snippet = "Synthesized from ${localResult.items.size} recent notifications",
                                        originalGoal = query
                                    )
                                )
                            )
                        } else {
                            val copyOffer = if (!localResult.copyableOtp.isNullOrBlank()) {
                                listOf(
                                    WebStoreOffer(
                                        storeName = "Clipboard",
                                        title = "📋 Copy OTP Code (${localResult.copyableOtp})",
                                        priceDisplay = "1-Tap Copy",
                                        price = 0.0,
                                        webUrl = "zipper://copy?text=${localResult.copyableOtp}",
                                        appPackage = "",
                                        isBestDeal = true
                                    )
                                )
                            } else emptyList()

                            chatAdapter.addMessage(
                                WebChatMessage(
                                    type = ChatMessageType.ASSISTANT,
                                    text = localResult.markdownResponse,
                                    resultData = WebResultData(
                                        title = localResult.title,
                                        snippet = "Notification Life-Stream Engine",
                                        originalGoal = query,
                                        comparisonOffers = copyOffer
                                    )
                                )
                            )
                        }
                        tvWebStatus.text = "🟢 READY"
                        tvWebStatus.setTextColor(android.graphics.Color.parseColor("#34D399"))
                        scrollToBottom()
                        return@launch
                    }
                }

                // 3. Conversational Multi-Turn Basket Manager (Session State & Delta Mutations)
                val qLower = query.lowercase(Locale.ROOT)
                val basketMutation = ConversationalBasketManager.processUserInput(query)

                if (basketMutation.type == BasketMutationType.BASKET_CLEARED) {
                    chatAdapter.addMessage(
                        WebChatMessage(
                            type = ChatMessageType.ASSISTANT,
                            text = "🗑️ **Shopping Basket Cleared.** Your active list has been reset."
                        )
                    )
                    tvWebStatus.text = "🟢 READY"
                    tvWebStatus.setTextColor(android.graphics.Color.parseColor("#34D399"))
                    scrollToBottom()
                    return@launch
                }

                if (basketMutation.type in setOf(
                        BasketMutationType.ITEMS_ADDED,
                        BasketMutationType.ITEM_MODIFIED,
                        BasketMutationType.ITEM_REMOVED,
                        BasketMutationType.STORE_SWITCHED
                    )
                ) {
                    val activeItems = basketMutation.currentItems
                    if (activeItems.isEmpty()) {
                        chatAdapter.addMessage(
                            WebChatMessage(
                                type = ChatMessageType.ASSISTANT,
                                text = "🛒 **Basket is now empty.** (${basketMutation.explanation})"
                            )
                        )
                        tvWebStatus.text = "🟢 READY"
                        tvWebStatus.setTextColor(android.graphics.Color.parseColor("#34D399"))
                        scrollToBottom()
                        return@launch
                    }

                    val targetStore = basketMutation.targetStore
                    if (targetStore != null && (qLower.contains("now") || qLower.contains("order") || qLower.contains("go") || qLower.contains("buy"))) {
                        executeMultiProductRelay("Multi-item list on $targetStore", activeItems.map { it.normalizedQuery }, targetStore)
                        return@launch
                    }

                    renderBasketFulfillmentOptions(
                        query = query,
                        items = activeItems,
                        headerPrefix = "🛒 **Basket Updated!**\n• *${basketMutation.explanation}*\n\n"
                    )
                    return@launch
                }

                val parsedItems = if (basketMutation.type == BasketMutationType.NEW_BASKET) {
                    basketMutation.currentItems
                } else {
                    ShoppingListParser.parseShoppingList(query)
                }

                val multiItems = if (parsedItems.size > 1) {
                    parsedItems.map { it.normalizedQuery }
                } else {
                    WebProductFinder.parseMultiProductItems(query)
                }

                val hasExplicitStore = qLower.contains("zepto") || qLower.contains("zzepto") ||
                    qLower.contains("blinkit") || qLower.contains("blibkit") || qLower.contains("blikit") ||
                    qLower.contains("amazon") || qLower.contains("amzn") ||
                    qLower.contains("flipkart") || qLower.contains("fk") ||
                    qLower.contains("myntra")

                val hasActionIntent = qLower.contains("buy") || qLower.contains("order") ||
                    qLower.contains("add to cart") || qLower.contains("cart") ||
                    qLower.contains("get") || qLower.contains("checkout") ||
                    parsedItems.isNotEmpty() ||
                    multiItems.any { WebProductFinder.isGroceryQuery(it) }

                if (multiItems.size > 1) {
                    if (hasExplicitStore) {
                        // User explicitly requested a store (e.g. "500g tomato, 1kg potato on zepto") -> Execute immediately!
                        val targetStore = basketMutation.targetStore
                            ?: WebDeepLinkManager.resolveBestStoreForQuery(query, multiItems, this@WebAssistantActivity)
                        executeMultiProductRelay(query, multiItems, targetStore)
                        return@launch
                    } else if (hasActionIntent && !isAiModeEnabled) {
                        renderBasketFulfillmentOptions(query, parsedItems)
                        return@launch
                    }
                    // When isAiModeEnabled is true and no store is explicitly named, do NOT intercept!
                    // Let the query flow directly to the Cloud AI / Groq LLM web runner below!
                }

                if (WebProductFinder.isShoppingQuery(query)) {
                    val (searchUrl, storeName) = WebProductFinder.buildSearchUrl(query)
                    chatAdapter.addMessage(
                        WebChatMessage(
                            type = ChatMessageType.PROGRESS_STEP,
                            text = "🛒 Searching $storeName for \"$query\"..."
                        )
                    )
                    scrollToBottom()

                    val loaded = webController.navigate(searchUrl, timeoutMs = 12000L)
                    if (loaded) {
                        delay(1500L)
                        val state = webController.distillWebState()
                        val products = WebProductFinder.extractProductsFromState(state, storeName)
                        val cleanTerms = WebProductFinder.cleanSearchQuery(query)
                        val targetKeywords = cleanTerms.lowercase().split(Regex("[^a-z0-9]+")).filter { it.length >= 3 }
                        val wantsBestOffer = query.contains("best offer", ignoreCase = true) || query.contains("best deal", ignoreCase = true)
                        val bestProduct = WebProductFinder.findBestMatch(products, targetKeywords, wantsBestOffer = wantsBestOffer)

                        if (bestProduct != null) {
                            val appPkg = WebDeepLinkManager.resolveAppPackage(bestProduct.url, query, context = this@WebAssistantActivity)
                            val reply = "🛒 **Best Match on $storeName:**\n\n**${bestProduct.title}**\n💰 Price: **${bestProduct.priceDisplay}**${if (bestProduct.rating != null) " (${bestProduct.rating})" else ""}\n\n👉 *Tap **🛒 Open in $storeName** to view and purchase directly inside the app!*"

                            chatAdapter.addMessage(
                                WebChatMessage(
                                    type = ChatMessageType.ASSISTANT,
                                    text = reply,
                                    resultData = WebResultData(
                                        title = bestProduct.title,
                                        price = bestProduct.priceDisplay,
                                        snippet = "Found on $storeName: ${bestProduct.title}",
                                        webUrl = bestProduct.url,
                                        appPackage = appPkg,
                                        rating = bestProduct.rating,
                                        storeName = storeName,
                                        originalGoal = query,
                                        imageUrl = bestProduct.imageUrl,
                                        imageUrls = listOfNotNull(bestProduct.imageUrl)
                                    )
                                )
                            )
                            tvWebStatus.text = "🟢 READY"
                            tvWebStatus.setTextColor(android.graphics.Color.parseColor("#34D399"))
                            scrollToBottom()
                            return@launch
                        }
                    }
                }

                chatAdapter.addMessage(
                    WebChatMessage(
                        type = ChatMessageType.PROGRESS_STEP,
                        text = if (isAiModeEnabled) {
                            "🌐 Initializing autonomous web runner (Cloud LLM)..."
                        } else {
                            "⚡ Initializing Karvish-Web On-Device Model (AI: OFF, 0 Cloud Tokens)..."
                        }
                    )
                )
                scrollToBottom()

                val runner = WebAutonomousRunner(
                    webController = webController,
                    actionExecutor = WebActionExecutor(webController),
                    webLlmClient = if (isAiModeEnabled) webLlmClient else null,
                    aiConfig = if (isAiModeEnabled) configManager.getConfig() else null,
                    webOnDeviceEngine = webOnDeviceEngine,
                    onProgressUpdate = { step ->
                        runOnUiThread {
                            chatAdapter.updateLastProgress(step)
                            scrollToBottom()
                        }
                    }
                )

                val result = runner.runEpisode(query)

                val targetPackage = WebDeepLinkManager.resolveAppPackage(result.finalUrl, query)
                val storeName = WebDeepLinkManager.resolveStoreName(result.finalUrl, targetPackage)

                val primaryPrice = if (result.extractedContent?.contains("₹") == true) {
                    Regex("""₹\s*\d+(?:,\d+)*(?:\.\d+)?""").find(result.extractedContent)?.value
                } else null

                val replyText = if (result.success) {
                    if (isAutoReplayMode) {
                        showFullscreenWeb(result.finalUrl)
                    }
                    if (result.fieldsFilled.isNotEmpty()) {
                        if (!isAutoReplayMode) {
                            showPeekView()
                        }
                        val solvedCaptcha = result.fieldsFilled.any { it.first.contains("security", true) || it.first.contains("captcha", true) }
                        if (solvedCaptcha) {
                            "✅ **Authentication & Verification Completed!**\n\n${result.summary}\n\n👉 *The live active session is shown in the preview above (or tap **👁️ Peek**)!*"
                        } else {
                            "✅ **Form Filled!**\n\n${result.summary}\n\n👉 *The live filled page is shown in the preview above (or tap **👁️ Peek**)!*"
                        }
                    } else {
                        "🌐 **Web Task Completed (${result.turnsExecuted} turns):**\n\n${result.summary}"
                    }
                } else {
                    "⚠️ ${result.summary}"
                }

                chatAdapter.addMessage(
                    WebChatMessage(
                        type = ChatMessageType.ASSISTANT,
                        text = replyText,
                        resultData = WebResultData(
                            title = "Web Target: " + result.finalUrl,
                            price = primaryPrice,
                            snippet = result.extractedContent?.take(200),
                            webUrl = result.finalUrl,
                            appPackage = targetPackage,
                            storeName = storeName,
                            originalGoal = query,
                            imageUrl = result.imageUrls.firstOrNull(),
                            imageUrls = result.imageUrls
                        )
                    )
                )

                tvWebStatus.text = "🟢 READY"
                tvWebStatus.setTextColor(android.graphics.Color.parseColor("#34D399"))
                scrollToBottom()

            } catch (e: Exception) {
                chatAdapter.addMessage(
                    WebChatMessage(
                        type = ChatMessageType.ASSISTANT,
                        text = "Encountered an issue executing web episode: ${e.message}\nPlease check your network connection and try again."
                    )
                )
                tvWebStatus.text = "🟢 READY"
                tvWebStatus.setTextColor(android.graphics.Color.parseColor("#34D399"))
                scrollToBottom()
            }
        }
    }

    private fun renderBasketFulfillmentOptions(
        query: String,
        items: List<ParsedShoppingItem>,
        headerPrefix: String = ""
    ) {
        val splitPlan = if (items.size > 1) BasketOptimizer.optimizeBasket(items) else null
        val storeOffers = mutableListOf<WebStoreOffer>()
        val multiItems = items.map { it.normalizedQuery }

        // 1. Smart Split Arbitrage Option (if both fresh and bulk items are present)
        if (splitPlan != null && splitPlan.hasSplitAdvantage) {
            val freshParam = Uri.encode(splitPlan.freshItems.joinToString(";") { it.normalizedQuery })
            val bulkParam = Uri.encode(splitPlan.bulkItems.joinToString(";") { it.normalizedQuery })
            val splitUrl = "zipper://split?freshStore=${splitPlan.freshStoreName}&bulkStore=${splitPlan.bulkStoreName}&freshItems=$freshParam&bulkItems=$bulkParam"

            storeOffers.add(
                WebStoreOffer(
                    storeName = "Smart Split (Zepto + Amazon)",
                    title = "⚡ ${splitPlan.freshItems.size} Fresh in Zepto (10m) + 📦 ${splitPlan.bulkItems.size} Bulk in Amazon",
                    priceDisplay = "💡 Save ~₹${splitPlan.estimatedSavingsInr}",
                    price = 0.0,
                    webUrl = splitUrl,
                    appPackage = splitPlan.freshStorePkg,
                    isBestDeal = true
                )
            )
        }

        // 2. Individual Store Options
        val encodedItemsQuery = Uri.encode(multiItems.joinToString(", "))
        if (WebDeepLinkManager.isPackageInstalled(this@WebAssistantActivity, WebDeepLinkManager.PKG_ZEPTO) || WebDeepLinkManager.isPackageInstalled(this@WebAssistantActivity, WebDeepLinkManager.PKG_ZEPTO_ALT)) {
            storeOffers.add(
                WebStoreOffer(
                    storeName = "Zepto",
                    title = "Queue all ${multiItems.size} items in Zepto (10-min delivery)",
                    priceDisplay = "⚡ 10 mins",
                    price = 0.0,
                    webUrl = "zipper://relay?store=Zepto&q=$encodedItemsQuery",
                    appPackage = WebDeepLinkManager.PKG_ZEPTO,
                    isBestDeal = (splitPlan == null || !splitPlan.hasSplitAdvantage)
                )
            )
        }
        if (WebDeepLinkManager.isPackageInstalled(this@WebAssistantActivity, WebDeepLinkManager.PKG_BLINKIT)) {
            storeOffers.add(
                WebStoreOffer(
                    storeName = "Blinkit",
                    title = "Queue all ${multiItems.size} items in Blinkit (10-min delivery)",
                    priceDisplay = "⚡ 10 mins",
                    price = 0.0,
                    webUrl = "zipper://relay?store=Blinkit&q=$encodedItemsQuery",
                    appPackage = WebDeepLinkManager.PKG_BLINKIT,
                    isBestDeal = false
                )
            )
        }
        if (WebDeepLinkManager.isPackageInstalled(this@WebAssistantActivity, WebDeepLinkManager.PKG_AMAZON_IN) || WebDeepLinkManager.isPackageInstalled(this@WebAssistantActivity, WebDeepLinkManager.PKG_AMAZON_GLOBAL)) {
            storeOffers.add(
                WebStoreOffer(
                    storeName = "Amazon",
                    title = "Queue all ${multiItems.size} items in Amazon / Fresh",
                    priceDisplay = "📦 Standard",
                    price = 0.0,
                    webUrl = "zipper://relay?store=Amazon&q=$encodedItemsQuery",
                    appPackage = WebDeepLinkManager.PKG_AMAZON_IN,
                    isBestDeal = false
                )
            )
        }

        val promptMessage = buildString {
            if (headerPrefix.isNotBlank()) {
                append(headerPrefix)
            }
            append("🛒 **Active Shopping Basket (${items.size} items):**\n\n")
            items.forEachIndexed { i, item ->
                val icon = when (item.category) {
                    ItemCategory.FRESH -> "🥬"
                    ItemCategory.BULK_PANTRY -> "📦"
                    ItemCategory.GENERAL -> "🛍️"
                }
                append("${i + 1}. $icon **${item.normalizedQuery}**")
                if (item.hardAttributes.isNotEmpty()) {
                    append(" `[${item.hardAttributes.joinToString(", ")}]`")
                }
                if (item.originalText != item.normalizedQuery && item.originalText.isNotBlank()) {
                    append(" *(from \"${item.originalText}\")*")
                }
                append("\n")
            }
            if (splitPlan != null && splitPlan.hasSplitAdvantage) {
                append("\n💡 **Smart Arbitrage Opportunity:**\n")
                append("${splitPlan.explanation}\n")
            }
            append("\n👉 *Tap your preferred fulfillment option below:*")
        }

        chatAdapter.addMessage(
            WebChatMessage(
                type = ChatMessageType.ASSISTANT,
                text = promptMessage,
                resultData = WebResultData(
                    title = "Smart Shopping Basket",
                    snippet = "Select store or Smart Split to order ${items.size} items",
                    originalGoal = query,
                    comparisonOffers = storeOffers
                )
            )
        )
        tvWebStatus.text = "🟢 READY"
        tvWebStatus.setTextColor(android.graphics.Color.parseColor("#34D399"))
        scrollToBottom()
    }

    private suspend fun executeMultiProductRelay(
        query: String,
        multiItems: List<String>,
        targetStore: String,
        returnToActivity: Boolean = true
    ) {
        val targetPkg = WebDeepLinkManager.resolveInstalledPackageForStore(targetStore, this@WebAssistantActivity)

        chatAdapter.addMessage(
            WebChatMessage(
                type = ChatMessageType.PROGRESS_STEP,
                text = "🛒 Initiating Multi-Product Relay on $targetStore (${multiItems.size} items)..."
            )
        )
        scrollToBottom()

        val queuedProducts = mutableListOf<WebProductItem>()
        for ((idx, itemQuery) in multiItems.withIndex()) {
            val cleanTerms = WebProductFinder.cleanSearchQuery(itemQuery)
            val targetKeywords = cleanTerms.lowercase(Locale.ROOT).split(Regex("[^a-z0-9]+")).filter { it.length >= 3 }

            chatAdapter.updateLastProgress("🛒 [${idx + 1}/${multiItems.size}] Scouting '$itemQuery' on $targetStore...")
            scrollToBottom()

            chatAdapter.updateLastProgress("👆 [${idx + 1}/${multiItems.size}] Adding '$itemQuery' to cart in $targetStore...")
            scrollToBottom()

            // Dispatch query directly into native store search (clean, genuine search results page)
            val opened = WebDeepLinkManager.openStoreSearch(
                context = this@WebAssistantActivity,
                storeName = targetStore,
                query = itemQuery
            )

            if (opened) {
                // Autonomous Native Closer with mandatory multi-viewport exploration,
                // strict cheapest genuine item selection, and closed-loop cart delta verification
                val result = WebNativeRelayCloser.attemptAddToCart(
                    targetAppPackage = targetPkg,
                    itemQuery = itemQuery,
                    expectedKeywords = targetKeywords,
                    maxScrolls = 2
                )
                when (result) {
                    is WebNativeRelayCloser.AddToCartResult.Success -> {
                        chatAdapter.updateLastProgress("✅ [${idx + 1}/${multiItems.size}] Added '${result.productTitle}' (${result.priceDisplay}) to cart")
                        scrollToBottom()
                        if (result.freeDeliveryPrompt != null) {
                            chatAdapter.addMessage(
                                WebChatMessage(
                                    type = ChatMessageType.PROGRESS_STEP,
                                    text = "💡 Live Store Alert: ${result.freeDeliveryPrompt} on $targetStore!"
                                )
                            )
                            scrollToBottom()
                        }
                        queuedProducts.add(WebProductItem(
                            title = result.productTitle,
                            price = result.price,
                            priceDisplay = result.priceDisplay,
                            snippet = result.freeDeliveryPrompt,
                            url = WebDeepLinkManager.buildStoreSearchUrl(targetStore, itemQuery),
                            store = targetStore
                        ))
                    }
                    is WebNativeRelayCloser.AddToCartResult.OutOfStock -> {
                        chatAdapter.updateLastProgress("⚠️ [${idx + 1}/${multiItems.size}] '${itemQuery}' is OUT OF STOCK. Skipping...")
                        scrollToBottom()
                        queuedProducts.add(WebProductItem(
                            title = "$itemQuery (Out of Stock)",
                            price = 0.0,
                            priceDisplay = "Out of Stock",
                            url = WebDeepLinkManager.buildStoreSearchUrl(targetStore, itemQuery),
                            store = targetStore
                        ))
                    }
                    is WebNativeRelayCloser.AddToCartResult.Failed -> {
                        chatAdapter.updateLastProgress("❌ [${idx + 1}/${multiItems.size}] Could not verify addition for '${itemQuery}' (${result.reason})")
                        scrollToBottom()
                        queuedProducts.add(WebProductItem(
                            title = "$itemQuery (Not added)",
                            price = 0.0,
                            priceDisplay = "Not added",
                            url = WebDeepLinkManager.buildStoreSearchUrl(targetStore, itemQuery),
                            store = targetStore
                        ))
                    }
                }
            }
        }

        if (queuedProducts.isNotEmpty()) {
            val addedCount = queuedProducts.count { it.priceDisplay.startsWith("₹") || it.priceDisplay == "Added" }
            val cartSummary = buildString {
                append("🛒 **Multi-Product Order Queue Completed ($addedCount/${multiItems.size} items added on $targetStore):**\n\n")
                queuedProducts.forEachIndexed { i, p ->
                    val icon = if (p.priceDisplay == "Out of Stock") "⚠️" else if (p.priceDisplay == "Not added") "❌" else "✅"
                    append("$icon ${i + 1}. **${p.title}** — ${p.priceDisplay}\n")
                }
                val liveTip = queuedProducts.firstOrNull { !it.snippet.isNullOrBlank() }?.snippet
                if (liveTip != null) {
                    append("\n\n💡 **Live Store Update:** $liveTip")
                }
                append("\n👉 *Cheapest genuine items selected with verified cart addition in your native $targetStore app!*")
            }

            if (returnToActivity) {
                // Bring Zipper back to foreground so user sees completion summary
                try {
                    val returnIntent = Intent(this@WebAssistantActivity, WebAssistantActivity::class.java).apply {
                        addFlags(Intent.FLAG_ACTIVITY_REORDER_TO_FRONT or Intent.FLAG_ACTIVITY_SINGLE_TOP)
                    }
                    startActivity(returnIntent)
                } catch (_: Exception) {}
            }

            val cartUrl = WebDeepLinkManager.buildStoreCartUrl(targetStore)

            chatAdapter.addMessage(
                WebChatMessage(
                    type = ChatMessageType.ASSISTANT,
                    text = cartSummary,
                    resultData = WebResultData(
                        title = "$targetStore Multi-Item Cart (${queuedProducts.size} items)",
                        price = if (queuedProducts.any { it.price > 0 }) "₹" + queuedProducts.sumOf { it.price }.toInt().toString() else "In Cart",
                        snippet = queuedProducts.joinToString(", ") { it.title },
                        webUrl = cartUrl,
                        appPackage = targetPkg,
                        storeName = targetStore,
                        originalGoal = query,
                        imageUrls = queuedProducts.mapNotNull { it.imageUrl }.take(3)
                    )
                )
            )
            tvWebStatus.text = "🟢 READY"
            tvWebStatus.setTextColor(android.graphics.Color.parseColor("#34D399"))
            scrollToBottom()
        }
    }

    private suspend fun executeSmartSplitRelay(
        freshStore: String,
        freshItems: List<String>,
        bulkStore: String,
        bulkItems: List<String>
    ) {
        chatAdapter.addMessage(
            WebChatMessage(
                type = ChatMessageType.PROGRESS_STEP,
                text = "💡 **Initiating Smart Split Cart Optimization:**\n" +
                       "• Phase 1: Staging ${freshItems.size} fresh items in $freshStore (10-min delivery)\n" +
                       "• Phase 2: Staging ${bulkItems.size} bulk staples in $bulkStore (bulk savings)"
            )
        )
        scrollToBottom()

        // Phase 1: Fresh items in Quick Commerce (e.g. Zepto)
        if (freshItems.isNotEmpty()) {
            executeMultiProductRelay("Fresh items on $freshStore", freshItems, freshStore, returnToActivity = false)
        }

        delay(1200L)

        // Phase 2: Bulk items in Amazon Fresh
        if (bulkItems.isNotEmpty()) {
            executeMultiProductRelay("Bulk staples on $bulkStore", bulkItems, bulkStore, returnToActivity = true)
        }

        chatAdapter.addMessage(
            WebChatMessage(
                type = ChatMessageType.ASSISTANT,
                text = "🎉 **Smart Split Optimization Complete!**\n\n" +
                       "✅ **$freshStore:** ${freshItems.size} Fresh items staged for 10-minute delivery.\n" +
                       "✅ **$bulkStore:** ${bulkItems.size} Bulk staples staged for maximum bulk savings.\n\n" +
                       "👉 *Both carts have been loaded with lowest genuine prices. Open each app to complete payment!*"
            )
        )
        tvWebStatus.text = "🟢 READY"
        tvWebStatus.setTextColor(android.graphics.Color.parseColor("#34D399"))
        scrollToBottom()
    }

    private fun scrollToBottom() {
        if (chatAdapter.itemCount > 0) {
            rvChatMessages.smoothScrollToPosition(chatAdapter.itemCount - 1)
        }
    }

    private fun setupSpeechRecognizer() {
        if (SpeechRecognizer.isRecognitionAvailable(this)) {
            speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this).apply {
                setRecognitionListener(object : RecognitionListener {
                    override fun onReadyForSpeech(params: Bundle?) {
                        tvWebStatus.text = "🎙️ LISTENING"
                        tvWebStatus.setTextColor(android.graphics.Color.parseColor("#38BDF8"))
                    }
                    override fun onResults(results: Bundle?) {
                        val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                        if (!matches.isNullOrEmpty()) {
                            val recognized = matches[0]
                            etWebPrompt.setText(recognized)
                            executeWebQuery(recognized)
                        }
                        tvWebStatus.text = "🟢 READY"
                        tvWebStatus.setTextColor(android.graphics.Color.parseColor("#34D399"))
                    }
                    override fun onError(error: Int) {
                        tvWebStatus.text = "🟢 READY"
                        tvWebStatus.setTextColor(android.graphics.Color.parseColor("#34D399"))
                    }
                    override fun onBeginningOfSpeech() {}
                    override fun onRmsChanged(rmsdB: Float) {}
                    override fun onBufferReceived(buffer: ByteArray?) {}
                    override fun onEndOfSpeech() {}
                    override fun onPartialResults(partialResults: Bundle?) {}
                    override fun onEvent(eventType: Int, params: Bundle?) {}
                })
            }
        }
    }

    private fun toggleVoiceInput() {
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
        }
        speechRecognizer?.startListening(intent)
    }

    private fun updateApiButtonUi() {
        val config = configManager.getConfig()
        if (config.isConfigured) {
            val shortName = config.provider.split(" ").first()
            btnApiSettings.text = "⚡ AI: $shortName"
            btnApiSettings.setTextColor(Color.parseColor("#34D399"))
            btnApiSettings.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#064E3B"))
        } else {
            btnApiSettings.text = "⚙️ Config"
            btnApiSettings.setTextColor(Color.parseColor("#38BDF8"))
            btnApiSettings.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#1E293B"))
        }
    }

    private fun updateAiModeButtonUi() {
        if (!::btnToggleAiMode.isInitialized) return
        if (isAiModeEnabled) {
            btnToggleAiMode.text = "⚡ AI: ON"
            btnToggleAiMode.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#059669"))
            btnToggleAiMode.setTextColor(Color.WHITE)
        } else {
            btnToggleAiMode.text = "⚡ AI: OFF"
            btnToggleAiMode.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#334155"))
            btnToggleAiMode.setTextColor(Color.parseColor("#94A3B8"))
        }
    }

    private fun showApiSettingsDialog() {
        val dialogView = layoutInflater.inflate(R.layout.dialog_web_api_settings, null)
        val spnProvider = dialogView.findViewById<Spinner>(R.id.spnProvider)
        val etModelName = dialogView.findViewById<EditText>(R.id.etModelName)
        val layoutCustomUrl = dialogView.findViewById<LinearLayout>(R.id.layoutCustomUrl)
        val etCustomBaseUrl = dialogView.findViewById<EditText>(R.id.etCustomBaseUrl)
        val etApiKey = dialogView.findViewById<EditText>(R.id.etApiKey)
        val tvVerifyStatus = dialogView.findViewById<TextView>(R.id.tvVerifyStatus)
        val btnCancelApi = dialogView.findViewById<Button>(R.id.btnCancelApi)
        val btnClearApi = dialogView.findViewById<Button>(R.id.btnClearApi)
        val btnVerifyAndSave = dialogView.findViewById<Button>(R.id.btnVerifyAndSave)

        val currentConfig = configManager.getConfig()

        val adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_dropdown_item,
            WebConfigManager.SUPPORTED_PROVIDERS
        )
        spnProvider.adapter = adapter
        val providerIndex = WebConfigManager.SUPPORTED_PROVIDERS.indexOf(currentConfig.provider).coerceAtLeast(0)
        spnProvider.setSelection(providerIndex)

        etModelName.setText(currentConfig.model)
        etApiKey.setText(currentConfig.apiKey)
        etCustomBaseUrl.setText(currentConfig.customUrl)

        fun updateCustomUrlVisibility(provider: String) {
            val isCustom = provider.contains("Custom", true)
            layoutCustomUrl.visibility = if (isCustom) View.VISIBLE else View.GONE
            val isChatGptWeb = provider.contains("ChatGPT Web", true)
            if (isChatGptWeb) {
                etApiKey.hint = "No API key required (Public Web Guest Mode)"
                etApiKey.isEnabled = false
                tvVerifyStatus.text = "🟢 Zero-config Guest Mode. Tap 'Verify & Save' to activate."
                tvVerifyStatus.setTextColor(Color.parseColor("#10B981"))
            } else {
                etApiKey.hint = "Enter API Key"
                etApiKey.isEnabled = true
            }
        }
        updateCustomUrlVisibility(currentConfig.provider)

        spnProvider.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                val selectedProvider = WebConfigManager.SUPPORTED_PROVIDERS[position]
                updateCustomUrlVisibility(selectedProvider)
                val currentModel = etModelName.text?.toString().orEmpty()
                if (currentModel.isBlank() || WebConfigManager.SUPPORTED_PROVIDERS.any { it != selectedProvider && currentModel == WebConfigManager.getDefaultModelForProvider(it) }) {
                    etModelName.setText(WebConfigManager.getDefaultModelForProvider(selectedProvider))
                }
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }

        if (currentConfig.isConfigured) {
            tvVerifyStatus.text = "🟢 Verified & Ready (${currentConfig.provider} - ${currentConfig.model})"
            tvVerifyStatus.setTextColor(Color.parseColor("#10B981"))
        } else {
            tvVerifyStatus.text = "⚪ Not verified. Connect your API key to enable autonomous form-filling."
            tvVerifyStatus.setTextColor(Color.parseColor("#94A3B8"))
        }

        val dialog = AlertDialog.Builder(this)
            .setView(dialogView)
            .create()

        dialog.window?.setBackgroundDrawableResource(android.R.color.transparent)

        btnVerifyAndSave.setOnClickListener {
            val selectedProvider = spnProvider.selectedItem.toString()
            val model = etModelName.text.toString().trim()
            val apiKey = etApiKey.text.toString().trim()
            val customUrl = etCustomBaseUrl.text.toString().trim()
            val isChatGptWeb = selectedProvider.contains("ChatGPT Web", true)

            if (apiKey.isBlank() && !selectedProvider.contains("Custom", true) && !isChatGptWeb) {
                tvVerifyStatus.text = "🔴 Error: Please enter your API key."
                tvVerifyStatus.setTextColor(Color.parseColor("#EF4444"))
                return@setOnClickListener
            }

            btnVerifyAndSave.isEnabled = false
            tvVerifyStatus.text = "🟡 Verifying key with $selectedProvider..."
            tvVerifyStatus.setTextColor(Color.parseColor("#F59E0B"))

            val testConfig = WebAiConfig(
                provider = selectedProvider,
                apiKey = apiKey,
                model = model.ifBlank { WebConfigManager.getDefaultModelForProvider(selectedProvider) },
                customUrl = customUrl,
                isVerified = false
            )

            lifecycleScope.launch {
                val result = webLlmClient.verifyKey(testConfig)
                withContext(Dispatchers.Main) {
                    btnVerifyAndSave.isEnabled = true
                    when (result) {
                        is VerificationResult.Success -> {
                            val verifiedConfig = testConfig.copy(
                                isVerified = true,
                                lastVerified = System.currentTimeMillis()
                            )
                            configManager.saveConfig(verifiedConfig)
                            updateApiButtonUi()

                            tvVerifyStatus.text = "🟢 OK / Verified! API key is active and ready."
                            tvVerifyStatus.setTextColor(Color.parseColor("#10B981"))
                            Toast.makeText(this@WebAssistantActivity, "AI Provider verified and saved!", Toast.LENGTH_SHORT).show()
                        }
                        is VerificationResult.Failure -> {
                            tvVerifyStatus.text = "🔴 Error: ${result.errorMessage}"
                            tvVerifyStatus.setTextColor(Color.parseColor("#EF4444"))
                        }
                    }
                }
            }
        }

        btnClearApi.setOnClickListener {
            configManager.clearConfig()
            etApiKey.setText("")
            etModelName.setText(WebConfigManager.getDefaultModelForProvider(spnProvider.selectedItem.toString()))
            tvVerifyStatus.text = "⚪ Cleared. Heuristic fallback enabled."
            tvVerifyStatus.setTextColor(Color.parseColor("#94A3B8"))
            updateApiButtonUi()
            Toast.makeText(this, "API configuration cleared.", Toast.LENGTH_SHORT).show()
        }

        btnCancelApi.setOnClickListener {
            dialog.dismiss()
        }

        dialog.show()
    }

    private fun toggleReadAloud(text: String) {
        val tts = textToSpeech ?: run {
            Toast.makeText(this, "Text-to-speech initializing...", Toast.LENGTH_SHORT).show()
            return
        }
        if (isSpeaking) {
            tts.stop()
            isSpeaking = false
            Toast.makeText(this, "⏹️ Audio stopped", Toast.LENGTH_SHORT).show()
        } else {
            val clean = text
                .replace(Regex("""[*#_`>~]"""), "")
                .replace(Regex("""\[.*?\]\(.*?\)"""), "")
                .replace(Regex("""\n+"""), " ")
                .trim()
            tts.speak(clean, TextToSpeech.QUEUE_FLUSH, null, "read_aloud")
            isSpeaking = true
            Toast.makeText(this, "▶️ Reading aloud...", Toast.LENGTH_SHORT).show()
        }
    }

    private fun exportCurrentPagePdf() {
        val destDir = File(getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS), "Karvish")
        destDir.mkdirs()
        val destFile = File(destDir, "Karvish_Page_${System.currentTimeMillis()}.pdf")
        Toast.makeText(this, "Generating clean PDF...", Toast.LENGTH_SHORT).show()
        webController.saveCurrentPageAsPdf(destFile) { success ->
            runOnUiThread {
                if (success) {
                    Toast.makeText(this, "✅ PDF Saved: ${destFile.name}", Toast.LENGTH_LONG).show()
                    try {
                        val uri = androidx.core.content.FileProvider.getUriForFile(
                            this@WebAssistantActivity,
                            "${packageName}.fileprovider",
                            destFile
                        )
                        val intent = Intent(Intent.ACTION_VIEW).apply {
                            setDataAndType(uri, "application/pdf")
                            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                        }
                        startActivity(Intent.createChooser(intent, "Open PDF with"))
                    } catch (e: Exception) {
                        Log.e(TAG, "Error opening PDF chooser", e)
                    }
                } else {
                    Toast.makeText(this, "Could not generate PDF from this page", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        speechRecognizer?.destroy()
        textToSpeech?.stop()
        textToSpeech?.shutdown()
        lifecycleScope.launch {
            webController.destroy()
        }
    }
}
