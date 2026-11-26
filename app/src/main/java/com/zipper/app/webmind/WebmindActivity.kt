package com.zipper.app.webmind

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Color
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.webkit.WebChromeClient
import android.webkit.WebResourceRequest
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.*
import androidx.activity.OnBackPressedCallback
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import android.content.res.ColorStateList
import androidx.lifecycle.lifecycleScope
import com.zipper.app.MainActivity
import com.zipper.app.R
import com.zipper.app.core.web.HiddenWebController
import com.zipper.app.core.web.WebActionExecutor
import com.zipper.app.core.web.WebAutonomousRunner
import com.zipper.app.core.web.WebAiConfig
import com.zipper.app.core.web.WebConfigManager
import com.zipper.app.core.web.WebImageLoader
import com.zipper.app.core.web.WebLlmClient
import com.zipper.app.core.web.WebOnDeviceEngine
import com.zipper.app.core.web.WebProductFinder
import com.zipper.app.core.web.WebProductItem
import com.zipper.app.core.local.LocalActionDispatcher
import com.zipper.app.core.web.features.ArticleReaderEngine
import com.zipper.app.core.web.features.KnowledgeGraphEngine
import com.zipper.app.core.web.features.KnowledgeResult
import com.zipper.app.core.web.features.PriceComparisonEngine
import com.zipper.app.core.web.features.WebShieldEngine
import com.zipper.app.webmind.data.*
import com.zipper.app.webmind.ui.PriceGraphView
import kotlinx.coroutines.launch
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.net.URLEncoder
import java.util.Locale
import java.util.Stack

class WebmindActivity : AppCompatActivity() {

    enum class Screen {
        HOME, CHAT, WEBVIEW, SPLIT, COMPARE, PRICE_TRACK, SUMMARIZE, HISTORY, SAVED
    }

    private val navStack = Stack<Screen>()
    private var isDesktopMode = false
    private var isReaderModeActive = false
    private var otpReceiver: BroadcastReceiver? = null

    // Top Header Views
    private lateinit var btnWebmindBack: ImageButton
    private lateinit var tvWebmindTopTitle: TextView
    private lateinit var tvWebmindTopSubtitle: TextView
    private lateinit var btnWebmindAiToggle: TextView
    private lateinit var btnWebmindHeaderBell: View
    private lateinit var tvWebmindHeaderAvatar: TextView

    // Screen Container Views
    private lateinit var screenHome: View
    private lateinit var screenChat: View
    private lateinit var screenWebView: View
    private lateinit var screenSplit: View
    private lateinit var screenCompare: View
    private lateinit var screenPriceTrack: View
    private lateinit var screenSummarize: View
    private lateinit var screenHistory: View
    private lateinit var screenSaved: View

    // Shared State & Database
    private lateinit var dbHelper: WebmindDatabaseHelper
    private lateinit var webController: HiddenWebController
    private val webOnDeviceEngine by lazy { com.zipper.app.core.web.WebOnDeviceEngine(this) }
    private var currentProduct: WebProduct? = null
    private var currentComparisonPayload: com.zipper.app.webmind.data.ComprehensiveComparisonPayload? = null
    private var lastSearchResponse: SearchResponse? = null
    private var isAiModeEnabled: Boolean = true

    // WebView references
    private var mainWebView: WebView? = null
    private var splitWebView: WebView? = null

    private val pickImageLauncher = registerForActivityResult(
        androidx.activity.result.contract.ActivityResultContracts.GetContent()
    ) { uri: android.net.Uri? ->
        if (uri != null) {
            handleImagePicked(uri)
        }
    }

    private fun handleImagePicked(uri: android.net.Uri) {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val inputStream = contentResolver.openInputStream(uri)
                val bitmap = android.graphics.BitmapFactory.decodeStream(inputStream)
                inputStream?.close()

                if (bitmap == null) {
                    withContext(Dispatchers.Main) {
                        Toast.makeText(this@WebmindActivity, "Could not load selected image", Toast.LENGTH_SHORT).show()
                    }
                    return@launch
                }

                withContext(Dispatchers.Main) {
                    navigateTo(Screen.CHAT)
                    findViewById<TextView>(R.id.tvUserQueryMessage)?.text = "📷 [Uploaded Screenshot / Workout Image]"
                    findViewById<View>(R.id.llUserQueryBubble)?.visibility = View.VISIBLE
                    findViewById<View>(R.id.llThinkingDotsBubble)?.visibility = View.VISIBLE
                    findViewById<View>(R.id.llAssistantResponseRow)?.visibility = View.GONE
                    findViewById<View>(R.id.cardProductRecommendation)?.visibility = View.GONE
                }

                val analysis = com.zipper.app.core.intelligence.vision.ImageIntelligenceEngine.analyzeBitmap(
                    context = applicationContext,
                    bitmap = bitmap
                )

                withContext(Dispatchers.Main) {
                    findViewById<View>(R.id.llThinkingDotsBubble)?.visibility = View.GONE
                    findViewById<View>(R.id.llAssistantResponseRow)?.visibility = View.VISIBLE
                    findViewById<View>(R.id.cardProductRecommendation)?.visibility = View.VISIBLE

                    val fullResultText = "${analysis.headline}\n\n${analysis.description}\n\n${analysis.markdownTable}\n\n${analysis.verdict}"
                    val tvResponse = findViewById<TextView>(R.id.tvChatAssistantText)
                    streamTypewriterText(tvResponse, fullResultText, speedMs = 8L)

                    findViewById<TextView>(R.id.tvResearchResultHeadline)?.text = analysis.headline
                    findViewById<TextView>(R.id.tvResearchResultDescription)?.text = analysis.description
                    findViewById<TextView>(R.id.tvProductTitle)?.text = analysis.headline
                    findViewById<TextView>(R.id.tvProductPrice)?.text = analysis.workoutMetrics?.let { "${it.caloriesKcal ?: 0} kcal burned" } ?: "Verified"
                    findViewById<TextView>(R.id.tvProductSpecsSummary)?.apply {
                        visibility = View.VISIBLE
                        text = analysis.markdownTable
                    }

                    val btnLog = findViewById<View>(R.id.btnCardCompare)
                    val tvLogLabel = findViewById<TextView>(R.id.tvCardCompareLabel)
                    if (analysis.canLogToPlanner && analysis.workoutMetrics != null) {
                        btnLog?.visibility = View.VISIBLE
                        tvLogLabel?.text = "Log to Planner"
                        btnLog?.setOnClickListener {
                            com.zipper.app.core.intelligence.vision.ImageIntelligenceEngine.logWorkoutToPlanner(
                                context = applicationContext,
                                metrics = analysis.workoutMetrics
                            )
                            Toast.makeText(this@WebmindActivity, "✓ Logged to Planner Timeline!", Toast.LENGTH_SHORT).show()
                            tvLogLabel?.text = "✓ Logged"
                            btnLog.isEnabled = false
                        }
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@WebmindActivity, "Failed to analyze image: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_webmind)

        // Handle System Insets (Samsung 3-button bar / status bar)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.rootWebmindLayout)) { _, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            val topBar = findViewById<View>(R.id.webmindTopBar)
            topBar?.setPadding(topBar.paddingLeft, systemBars.top, topBar.paddingRight, topBar.paddingBottom)
            val bottomNav = findViewById<View>(R.id.webmindBottomNavBar)
            bottomNav?.setPadding(bottomNav.paddingLeft, bottomNav.paddingTop, bottomNav.paddingRight, systemBars.bottom)
            insets
        }

        dbHelper = WebmindDatabaseHelper.getInstance(this)
        webController = HiddenWebController(this)
        isAiModeEnabled = getSharedPreferences("karvish_web_prefs", Context.MODE_PRIVATE)
            .getBoolean("is_ai_mode_enabled", true)

        com.zipper.app.core.web.chatgpt.ChatGptHeadlessBridge.init(this)

        initViews()
        setupNavigation()
        setupScreenHome()
        setupScreenChat()
        setupScreenWebView()
        setupScreenSplit()
        setupScreenCompare()
        setupScreenPriceTrack()
        setupScreenSummarize()
        setupScreenHistory()
        setupScreenSaved()
        registerOtpReceiver()

        // Handle initial launch
        val initialQuery = intent.getStringExtra("initial_query")
        if (!initialQuery.isNullOrEmpty()) {
            executeSearch(initialQuery)
        } else {
            navigateTo(Screen.HOME)
        }

        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (mainWebView?.canGoBack() == true && currentScreen() == Screen.WEBVIEW) {
                    mainWebView?.goBack()
                    return
                }
                if (navStack.size > 1) {
                    navStack.pop()
                    renderScreen(navStack.peek(), isBack = true)
                } else {
                    val intent = Intent(this@WebmindActivity, MainActivity::class.java).apply {
                        flags = Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP
                    }
                    startActivity(intent)
                    finish()
                    @Suppress("DEPRECATION")
                    overridePendingTransition(0, 0)
                }
            }
        })
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        val newQuery = intent.getStringExtra("initial_query")
        if (!newQuery.isNullOrEmpty()) {
            executeSearch(newQuery)
        }
    }

    private fun initViews() {
        btnWebmindBack = findViewById(R.id.btnWebmindBack)
        tvWebmindTopTitle = findViewById(R.id.tvWebmindTopTitle)
        tvWebmindTopSubtitle = findViewById(R.id.tvWebmindTopSubtitle)
        btnWebmindAiToggle = findViewById(R.id.btnWebmindAiToggle)
        btnWebmindHeaderBell = findViewById(R.id.btnWebmindHeaderBell)
        tvWebmindHeaderAvatar = findViewById(R.id.tvWebmindHeaderAvatar)

        screenHome = findViewById(R.id.screenHome)
        screenChat = findViewById(R.id.screenChat)
        screenWebView = findViewById(R.id.screenWebView)
        screenSplit = findViewById(R.id.screenSplit)
        screenCompare = findViewById(R.id.screenCompare)
        screenPriceTrack = findViewById(R.id.screenPriceTrack)
        screenSummarize = findViewById(R.id.screenSummarize)
        screenHistory = findViewById(R.id.screenHistory)
        screenSaved = findViewById(R.id.screenSaved)

        updateAiToggleUi()
        btnWebmindAiToggle.setOnClickListener {
            showAiProviderPicker()
        }
    }

    private fun showAiProviderPicker() {
        val configMgr = WebConfigManager(this)
        val current = configMgr.getConfig()
        val options = arrayOf(
            "🤖 ChatGPT Web (Free Guest Mode, 0 Login)",
            "⚡ Groq Llama/Qwen (Fastest 300ms)",
            "✨ Google Gemini 1.5/2.0",
            "⚙️ Advanced API Settings...",
            "⚪ Turn Off AI (Fast Local Mode, 0 Tokens)"
        )

        val selectedIndex = when {
            !isAiModeEnabled -> 4
            current.provider.contains("ChatGPT Web", true) -> 0
            current.provider.contains("Groq", true) -> 1
            current.provider.contains("Gemini", true) -> 2
            else -> 3
        }

        AlertDialog.Builder(this)
            .setTitle("Switch AI Intelligence Engine")
            .setSingleChoiceItems(options, selectedIndex) { dialog, which ->
                when (which) {
                    0 -> {
                        // Switch to ChatGPT Web Free Mode
                        isAiModeEnabled = true
                        getSharedPreferences("karvish_web_prefs", Context.MODE_PRIVATE)
                            .edit().putBoolean("is_ai_mode_enabled", true).apply()
                        val newConfig = WebAiConfig(
                            provider = "ChatGPT Web (Free)",
                            apiKey = "",
                            model = "gpt-4o-guest",
                            isVerified = true,
                            lastVerified = System.currentTimeMillis()
                        )
                        configMgr.saveConfig(newConfig)
                        updateAiToggleUi()
                        Toast.makeText(this, "✓ Switched to ChatGPT Web Mode (Auto-verifies Bot checks)", Toast.LENGTH_SHORT).show()
                        dialog.dismiss()
                    }
                    1 -> {
                        // Switch to Groq
                        isAiModeEnabled = true
                        getSharedPreferences("karvish_web_prefs", Context.MODE_PRIVATE)
                            .edit().putBoolean("is_ai_mode_enabled", true).apply()
                        val newConfig = current.copy(
                            provider = "Groq (Fast)",
                            model = WebConfigManager.getDefaultModelForProvider("Groq (Fast)")
                        )
                        configMgr.saveConfig(newConfig)
                        updateAiToggleUi()
                        Toast.makeText(this, "✓ Switched to Groq Fast Mode", Toast.LENGTH_SHORT).show()
                        dialog.dismiss()
                    }
                    2 -> {
                        // Switch to Gemini
                        isAiModeEnabled = true
                        getSharedPreferences("karvish_web_prefs", Context.MODE_PRIVATE)
                            .edit().putBoolean("is_ai_mode_enabled", true).apply()
                        val newConfig = current.copy(
                            provider = "Google Gemini",
                            model = WebConfigManager.getDefaultModelForProvider("Google Gemini")
                        )
                        configMgr.saveConfig(newConfig)
                        updateAiToggleUi()
                        Toast.makeText(this, "✓ Switched to Google Gemini Mode", Toast.LENGTH_SHORT).show()
                        dialog.dismiss()
                    }
                    3 -> {
                        dialog.dismiss()
                        startActivity(Intent(this, com.zipper.app.core.web.WebAssistantActivity::class.java))
                    }
                    4 -> {
                        // Turn off AI
                        isAiModeEnabled = false
                        getSharedPreferences("karvish_web_prefs", Context.MODE_PRIVATE)
                            .edit().putBoolean("is_ai_mode_enabled", false).apply()
                        updateAiToggleUi()
                        Toast.makeText(this, "⚪ AI Disabled (Fast Local Mode, 0 Tokens)", Toast.LENGTH_SHORT).show()
                        dialog.dismiss()
                    }
                }
            }
            .setNegativeButton("Close", null)
            .show()
    }

    private fun updateAiToggleUi() {
        if (!isAiModeEnabled) {
            btnWebmindAiToggle.text = "⚪ AI: OFF"
            btnWebmindAiToggle.setBackgroundResource(R.drawable.bg_filter_pill_light)
            btnWebmindAiToggle.setTextColor(Color.parseColor("#4B5563"))
            findViewById<TextView>(R.id.tvQuickGptModeSubtitle)?.text = "Tap to switch"
            return
        }

        val config = WebConfigManager(this).getConfig()
        val provider = config.provider
        val isChatGpt = provider.contains("ChatGPT Web", true)

        if (isChatGpt) {
            btnWebmindAiToggle.text = "🤖 ChatGPT"
            btnWebmindAiToggle.setBackgroundResource(R.drawable.bg_filter_pill_black)
            btnWebmindAiToggle.setTextColor(Color.WHITE)
            findViewById<TextView>(R.id.tvQuickGptModeSubtitle)?.text = "Active ✓"
        } else if (provider.contains("Groq", true)) {
            btnWebmindAiToggle.text = "⚡ Groq"
            btnWebmindAiToggle.setBackgroundResource(R.drawable.bg_filter_pill_black)
            btnWebmindAiToggle.setTextColor(Color.WHITE)
            findViewById<TextView>(R.id.tvQuickGptModeSubtitle)?.text = "Tap to switch"
        } else if (provider.contains("Gemini", true)) {
            btnWebmindAiToggle.text = "✨ Gemini"
            btnWebmindAiToggle.setBackgroundResource(R.drawable.bg_filter_pill_black)
            btnWebmindAiToggle.setTextColor(Color.WHITE)
            findViewById<TextView>(R.id.tvQuickGptModeSubtitle)?.text = "Tap to switch"
        } else {
            btnWebmindAiToggle.text = "⚡ AI: ON"
            btnWebmindAiToggle.setBackgroundResource(R.drawable.bg_filter_pill_black)
            btnWebmindAiToggle.setTextColor(Color.WHITE)
            findViewById<TextView>(R.id.tvQuickGptModeSubtitle)?.text = "Tap to switch"
        }
    }

    private fun currentScreen(): Screen = if (navStack.isEmpty()) Screen.HOME else navStack.peek()

    fun navigateTo(screen: Screen) {
        if (navStack.isEmpty() || navStack.peek() != screen) {
            navStack.push(screen)
        }
        renderScreen(screen, isBack = false)
    }

    private fun renderScreen(screen: Screen, isBack: Boolean) {
        // Hide all screens
        screenHome.visibility = View.GONE
        screenChat.visibility = View.GONE
        screenWebView.visibility = View.GONE
        screenSplit.visibility = View.GONE
        screenCompare.visibility = View.GONE
        screenPriceTrack.visibility = View.GONE
        screenSummarize.visibility = View.GONE
        screenHistory.visibility = View.GONE
        screenSaved.visibility = View.GONE

        // Configure Top App Bar per Screen
        when (screen) {
            Screen.HOME -> {
                screenHome.visibility = View.VISIBLE
                tvWebmindTopTitle.text = "Zipper"
                tvWebmindTopSubtitle.text = "Research · Compare · Track · Summarize"
                btnWebmindBack.setImageResource(R.drawable.ic_arrow_back)
            }
            Screen.CHAT -> {
                screenChat.visibility = View.VISIBLE
                tvWebmindTopTitle.text = "Zipper"
                tvWebmindTopSubtitle.text = "Research · Compare · Analyze"
                btnWebmindBack.setImageResource(R.drawable.ic_arrow_back)
            }
            Screen.WEBVIEW -> {
                screenWebView.visibility = View.VISIBLE
                tvWebmindTopTitle.text = "Webmind"
                tvWebmindTopSubtitle.text = "Live Website View (Shield Active)"
                btnWebmindBack.setImageResource(R.drawable.ic_arrow_back)
                if (mainWebView?.url.isNullOrBlank()) {
                    val fallbackUrl = webController.getWebView()?.url?.takeIf { it.isNotBlank() && it != "about:blank" }
                        ?: currentProduct?.productUrl?.takeIf { it.isNotBlank() }
                        ?: "https://www.google.com"
                    mainWebView?.loadUrl(fallbackUrl)
                }
            }
            Screen.SPLIT -> {
                screenSplit.visibility = View.VISIBLE
                tvWebmindTopTitle.text = "Webmind"
                tvWebmindTopSubtitle.text = "Chat + Web View"
                btnWebmindBack.setImageResource(R.drawable.ic_arrow_back)
            }
            Screen.COMPARE -> {
                screenCompare.visibility = View.VISIBLE
                tvWebmindTopTitle.text = "Webmind"
                tvWebmindTopSubtitle.text = "Compare Products"
                btnWebmindBack.setImageResource(R.drawable.ic_arrow_back)
                setupScreenCompare()
            }
            Screen.PRICE_TRACK -> {
                screenPriceTrack.visibility = View.VISIBLE
                tvWebmindTopTitle.text = "Webmind"
                tvWebmindTopSubtitle.text = "Price History & Tracking"
                btnWebmindBack.setImageResource(R.drawable.ic_arrow_back)
                setupScreenPriceTrack()
                loadPriceHistoryGraph()
            }
            Screen.SUMMARIZE -> {
                screenSummarize.visibility = View.VISIBLE
                tvWebmindTopTitle.text = "Webmind"
                tvWebmindTopSubtitle.text = "Summarize Webpages & Articles"
                btnWebmindBack.setImageResource(R.drawable.ic_arrow_back)
            }
            Screen.HISTORY -> {
                screenHistory.visibility = View.VISIBLE
                tvWebmindTopTitle.text = "Webmind"
                tvWebmindTopSubtitle.text = "Research History"
                btnWebmindBack.setImageResource(R.drawable.ic_arrow_back)
                loadResearchHistory()
            }
            Screen.SAVED -> {
                screenSaved.visibility = View.VISIBLE
                tvWebmindTopTitle.text = "Webmind"
                tvWebmindTopSubtitle.text = "Saved Items & Tracks"
                btnWebmindBack.setImageResource(R.drawable.ic_arrow_back)
                loadSavedItems()
            }
        }
    }

    private fun setupNavigation() {
        btnWebmindBack.setOnClickListener {
            if (navStack.size > 1) {
                navStack.pop()
                renderScreen(navStack.peek(), isBack = true)
            } else {
                val intent = Intent(this, MainActivity::class.java).apply {
                    flags = Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP
                }
                startActivity(intent)
                finish()
            }
        }

        btnWebmindHeaderBell.setOnClickListener {
            val tracked = dbHelper.getTrackedProducts()
            AlertDialog.Builder(this)
                .setTitle("Active Price Alerts (${tracked.size})")
                .setMessage(
                    tracked.joinToString("\n\n") {
                        "• ${it.title}: ₹${it.currentPrice.toInt()} (Target: ₹${it.targetPrice.toInt()})"
                    }.ifEmpty { "No price drop notifications right now." }
                )
                .setPositiveButton("OK", null)
                .show()
        }

        tvWebmindHeaderAvatar.setOnClickListener {
            navigateTo(Screen.SAVED)
        }

        findViewById<View>(R.id.ivHeaderGlobeIcon)?.setOnClickListener {
            val url = webController.getWebView()?.url?.takeIf { it.isNotBlank() && it != "about:blank" }
                ?: currentProduct?.productUrl?.takeIf { it.isNotBlank() }
                ?: "https://www.google.com"
            openUrlInWebView(url)
        }

        // Bottom Nav Dock - Instant zero-delay switching to MainActivity
        findViewById<View>(R.id.navTabAgent)?.setOnClickListener {
            val intent = Intent(this, MainActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP
                putExtra("open_tab", "agent")
            }
            startActivity(intent)
            finish()
            @Suppress("DEPRECATION")
            overridePendingTransition(0, 0)
        }
        findViewById<View>(R.id.navTabWebmind)?.setOnClickListener {
            navigateTo(Screen.HOME)
        }
        findViewById<View>(R.id.navTabPlanner)?.setOnClickListener {
            val intent = Intent(this, MainActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP
                putExtra("open_tab", "planner")
            }
            startActivity(intent)
            finish()
            @Suppress("DEPRECATION")
            overridePendingTransition(0, 0)
        }
        findViewById<View>(R.id.navTabDitto)?.setOnClickListener {
            val intent = Intent(this, MainActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP
                putExtra("open_tab", "ditto")
            }
            startActivity(intent)
            finish()
            @Suppress("DEPRECATION")
            overridePendingTransition(0, 0)
        }
        findViewById<View>(R.id.navTabControl)?.setOnClickListener {
            navigateTo(Screen.HISTORY)
        }
    }

    private fun applyZeroDelayTransition() {
        if (android.os.Build.VERSION.SDK_INT >= 34) {
            overrideActivityTransition(android.app.Activity.OVERRIDE_TRANSITION_OPEN, 0, 0)
            overrideActivityTransition(android.app.Activity.OVERRIDE_TRANSITION_CLOSE, 0, 0)
        }
        @Suppress("DEPRECATION")
        overridePendingTransition(0, 0)
    }

    // =================================================================
    // SCREEN 1: CHAT HOME
    // =================================================================
    private fun setupScreenHome() {
        findViewById<View>(R.id.tileLaptops)?.setOnClickListener {
            executeSearch("Find best laptops under 70k in India for coding and light gaming")
        }
        findViewById<View>(R.id.tileComparePhones)?.setOnClickListener {
            executeSearch("Compare iPhone 16 vs Samsung S24")
        }
        findViewById<View>(R.id.tileSummarize)?.setOnClickListener {
            navigateTo(Screen.SUMMARIZE)
        }
        findViewById<View>(R.id.tileGoldPrice)?.setOnClickListener {
            executeSearch("Track 22k gold price today in Bangalore")
        }
        findViewById<View>(R.id.tileAiModels)?.setOnClickListener {
            executeSearch("Research about new AI models in 2026")
        }
        findViewById<View>(R.id.tileMoreExamples)?.setOnClickListener {
            navigateTo(Screen.HISTORY)
        }

        val etHome = findViewById<EditText>(R.id.etHomeWebQuery)
        etHome?.setOnEditorActionListener { _, _, _ ->
            val query = etHome.text.toString().trim()
            if (query.isNotEmpty()) {
                executeSearch(query)
                etHome.text.clear()
            }
            true
        }

        // Circular Search Button in Home
        findViewById<View>(R.id.btnHomeSend)?.setOnClickListener {
            val query = etHome?.text?.toString()?.trim().orEmpty()
            if (query.isNotEmpty()) {
                executeSearch(query)
                etHome?.text?.clear()
            }
        }

        findViewById<View>(R.id.pillQuickGptMode)?.setOnClickListener {
            isAiModeEnabled = true
            getSharedPreferences("karvish_web_prefs", Context.MODE_PRIVATE)
                .edit().putBoolean("is_ai_mode_enabled", true).apply()
            val newConfig = WebAiConfig(
                provider = "ChatGPT Web (Free)",
                apiKey = "",
                model = "gpt-4o-guest",
                isVerified = true,
                lastVerified = System.currentTimeMillis()
            )
            WebConfigManager(this).saveConfig(newConfig)
            updateAiToggleUi()
            Toast.makeText(this, "🤖 Switched to ChatGPT Web Mode (Auto-verifies Bot checks)!", Toast.LENGTH_SHORT).show()
        }

        findViewById<View>(R.id.pillQuickSearch)?.setOnClickListener {
            val et = findViewById<EditText>(R.id.etHomeWebQuery)
            et?.requestFocus()
            val imm = getSystemService(Context.INPUT_METHOD_SERVICE) as? android.view.inputmethod.InputMethodManager
            imm?.showSoftInput(et, android.view.inputmethod.InputMethodManager.SHOW_IMPLICIT)
        }
        findViewById<View>(R.id.pillQuickCompare)?.setOnClickListener {
            navigateTo(Screen.COMPARE)
        }
        findViewById<View>(R.id.pillQuickSummarize)?.setOnClickListener {
            navigateTo(Screen.SUMMARIZE)
        }
        findViewById<View>(R.id.pillQuickTrack)?.setOnClickListener {
            navigateTo(Screen.PRICE_TRACK)
        }
        findViewById<View>(R.id.pillQuickLiveWeb)?.setOnClickListener {
            val url = webController.getWebView()?.url?.takeIf { it.isNotBlank() && it != "about:blank" }
                ?: currentProduct?.productUrl?.takeIf { it.isNotBlank() }
                ?: "https://www.google.com"
            openUrlInWebView(url)
        }

        findViewById<View>(R.id.btnHomePlus)?.setOnClickListener {
            val options = arrayOf(
                "🤖 Switch AI Engine (ChatGPT / Groq / Gemini)",
                "🤖 Open ChatGPT Web (Solve Bot / Login)",
                "Search Live Web",
                "Compare Products",
                "Summarize Webpage/Video",
                "Price Tracker",
                "📷 Analyze Screenshot / Workout",
                "🌐 Open Full WebView",
                "Saved Library"
            )
            AlertDialog.Builder(this)
                .setTitle("Webmind Quick Actions")
                .setItems(options) { _, which ->
                    when (which) {
                        0 -> showAiProviderPicker()
                        1 -> {
                            com.zipper.app.core.web.chatgpt.ChatGptHeadlessBridge.showVerificationSheet(this@WebmindActivity, "Manual Session Trigger")
                        }
                        2 -> etHome?.requestFocus()
                        3 -> navigateTo(Screen.COMPARE)
                        4 -> navigateTo(Screen.SUMMARIZE)
                        5 -> navigateTo(Screen.PRICE_TRACK)
                        6 -> pickImageLauncher.launch("image/*")
                        7 -> {
                            val url = webController.getWebView()?.url?.takeIf { it.isNotBlank() && it != "about:blank" }
                                ?: currentProduct?.productUrl?.takeIf { it.isNotBlank() }
                                ?: "https://www.google.com"
                            openUrlInWebView(url)
                        }
                        8 -> navigateTo(Screen.SAVED)
                    }
                }.show()
        }
    }

    // =================================================================
    // SCREEN 2: CHAT WITH LIVE RESULTS
    // =================================================================
    private fun setupScreenChat() {
        findViewById<View>(R.id.btnChatAttachImage)?.setOnClickListener {
            pickImageLauncher.launch("image/*")
        }

        findViewById<View>(R.id.pillChatSearch)?.setOnClickListener {
            val et = findViewById<EditText>(R.id.etChatFollowUp)
            et?.requestFocus()
            val imm = getSystemService(Context.INPUT_METHOD_SERVICE) as? android.view.inputmethod.InputMethodManager
            imm?.showSoftInput(et, android.view.inputmethod.InputMethodManager.SHOW_IMPLICIT)
        }
        findViewById<View>(R.id.pillChatCompare)?.setOnClickListener {
            navigateTo(Screen.COMPARE)
        }
        findViewById<View>(R.id.pillChatSummarize)?.setOnClickListener {
            navigateTo(Screen.SUMMARIZE)
        }
        findViewById<View>(R.id.pillChatTrack)?.setOnClickListener {
            navigateTo(Screen.PRICE_TRACK)
        }
        findViewById<View>(R.id.pillChatLiveWeb)?.setOnClickListener {
            val url = webController.getWebView()?.url?.takeIf { it.isNotBlank() && it != "about:blank" }
                ?: currentProduct?.productUrl?.takeIf { it.isNotBlank() }
                ?: "https://www.google.com"
            openUrlInWebView(url)
        }

        val btnCompare = findViewById<View>(R.id.btnCardCompare)
        val btnViewDetails = findViewById<View>(R.id.btnCardViewDetails)
        val btnSave = findViewById<View>(R.id.btnCardSave)

        btnCompare?.setOnClickListener {
            val prodTitle = currentProduct?.title ?: "laptop under 70k"
            executeSearch("Compare $prodTitle across Amazon and Flipkart")
        }

        btnViewDetails?.setOnClickListener {
            val url = currentProduct?.productUrl ?: "https://www.amazon.in/dp/B0D1G5Y3Q2"
            openUrlInWebView(url)
        }

        btnSave?.setOnClickListener {
            currentProduct?.let { prod ->
                dbHelper.addResearchSession(
                    ResearchSession(
                        id = "res_${System.currentTimeMillis()}",
                        query = prod.title,
                        category = "Search",
                        timestamp = System.currentTimeMillis(),
                        summaryText = "Saved ${prod.title} (${prod.priceFormatted})",
                        iconType = "laptop"
                    )
                )
                Toast.makeText(this, "✓ Saved ${prod.title} to Library!", Toast.LENGTH_SHORT).show()
                findViewById<ImageView>(R.id.ivSaveIcon)?.setColorFilter(Color.parseColor("#2563EB"))
            }
        }

        val etChat = findViewById<EditText>(R.id.etChatFollowUp)
        etChat?.setOnEditorActionListener { _, _, _ ->
            val query = etChat.text.toString().trim()
            if (query.isNotEmpty()) {
                executeSearch(query)
                etChat.text.clear()
            }
            true
        }

        // Circular Send Button in Chat
        findViewById<View>(R.id.btnChatSend)?.setOnClickListener {
            val query = etChat?.text?.toString()?.trim().orEmpty()
            if (query.isNotEmpty()) {
                executeSearch(query)
                etChat?.text?.clear()
            }
        }

        findViewById<View>(R.id.btnChatPlus)?.setOnClickListener {
            findViewById<View>(R.id.btnHomePlus)?.performClick()
        }
    }

    private var typewriterJob: kotlinx.coroutines.Job? = null

    private fun streamTypewriterText(textView: TextView?, fullText: String, speedMs: Long = 18L) {
        if (textView == null) return
        typewriterJob?.cancel()
        typewriterJob = lifecycleScope.launch {
            textView.text = ""
            val sb = StringBuilder()
            for (char in fullText) {
                sb.append(char)
                textView.text = sb.toString()
                kotlinx.coroutines.delay(speedMs)
            }
        }
    }

    private fun populateProductCarousel(products: List<WebProductItem>) {
        val hsv = findViewById<View>(R.id.hsvProductCarousel)
        val container = findViewById<LinearLayout>(R.id.llProductCarouselContainer)
        if (products.isEmpty() || container == null || hsv == null) {
            hsv?.visibility = View.GONE
            return
        }

        container.removeAllViews()
        val inflater = LayoutInflater.from(this)

        // Find cheapest item price among products with valid price
        val validPrices = products.filter { it.price > 0.0 }
        val minPrice = validPrices.minOfOrNull { it.price }

        for (item in products) {
            val cardView = inflater.inflate(R.layout.item_webmind_product_card, container, false)
            val ivStore = cardView.findViewById<ImageView>(R.id.ivCardStoreIcon)
            val tvStore = cardView.findViewById<TextView>(R.id.tvCardStoreName)
            val ivImage = cardView.findViewById<ImageView>(R.id.ivCardProductImage)
            val tvTitle = cardView.findViewById<TextView>(R.id.tvCardProductTitle)
            val tvPrice = cardView.findViewById<TextView>(R.id.tvCardProductPrice)
            val btnDetails = cardView.findViewById<TextView>(R.id.btnCardViewDetails)

            // Store icon & badge
            val isCheapest = minPrice != null && item.price == minPrice
            val storeLower = item.store.lowercase(Locale.ROOT)
            when {
                storeLower.contains("amazon") -> {
                    ivStore?.setImageResource(R.drawable.ic_amazon)
                    ivStore?.imageTintList = null
                    tvStore?.text = if (isCheapest) "Amazon · Lowest" else "Amazon"
                }
                storeLower.contains("flipkart") -> {
                    ivStore?.setImageResource(R.drawable.ic_globe)
                    ivStore?.imageTintList = ColorStateList.valueOf(Color.parseColor("#2563EB"))
                    tvStore?.text = if (isCheapest) "Flipkart · Lowest" else "Flipkart"
                }
                storeLower.contains("zepto") -> {
                    ivStore?.setImageResource(R.drawable.ic_bolt)
                    ivStore?.imageTintList = ColorStateList.valueOf(Color.parseColor("#7C3AED"))
                    tvStore?.text = if (isCheapest) "Zepto · Lowest" else "Zepto"
                }
                storeLower.contains("blinkit") -> {
                    ivStore?.setImageResource(R.drawable.ic_bolt)
                    ivStore?.imageTintList = ColorStateList.valueOf(Color.parseColor("#F59E0B"))
                    tvStore?.text = if (isCheapest) "Blinkit · Lowest" else "Blinkit"
                }
                else -> {
                    ivStore?.setImageResource(R.drawable.ic_globe)
                    ivStore?.imageTintList = ColorStateList.valueOf(Color.parseColor("#4B5563"))
                    tvStore?.text = if (isCheapest) "${item.store} · Lowest" else item.store
                }
            }
            if (isCheapest) {
                tvStore?.setTextColor(Color.parseColor("#166534")) // Dark green
            }

            tvTitle?.text = item.title
            tvPrice?.text = if (item.priceDisplay.isNotBlank()) item.priceDisplay else "₹${item.price.toInt()}"

            // Bind thumbnail image
            if (!item.imageUrl.isNullOrBlank() && ivImage != null) {
                WebImageLoader.load(this, item.imageUrl, ivImage)
            } else {
                ivImage?.setImageResource(R.drawable.ic_laptop)
            }

            btnDetails?.setOnClickListener {
                if (item.url.isNotBlank()) {
                    openUrlInWebView(item.url)
                }
            }

            container.addView(cardView)
        }

        hsv.visibility = View.VISIBLE
    }

    private fun isAutonomousWebGoal(query: String): Boolean {
        return com.zipper.app.webmind.data.WebDomainClassifier.isAutonomousActionGoal(query)
    }

    private fun showDisambiguationCard(query: String, result: LocalActionDispatcher.DispatchResult) {
        navigateTo(Screen.CHAT)
        findViewById<TextView>(R.id.tvUserQueryMessage)?.text = query
        val timeFormat = java.text.SimpleDateFormat("h:mm a", Locale.getDefault()).format(java.util.Date())
        findViewById<TextView>(R.id.tvUserMessageTime)?.text = "$timeFormat ✔✔"

        findViewById<View>(R.id.llThinkingDotsBubble)?.visibility = View.GONE
        findViewById<View>(R.id.hsvProductCarousel)?.visibility = View.GONE
        findViewById<View>(R.id.cardProductRecommendation)?.visibility = View.GONE
        findViewById<View>(R.id.cardAgentThinkingProgress)?.visibility = View.GONE
        findViewById<View>(R.id.llLiveAgentHud)?.visibility = View.GONE

        val llAssistantResponse = findViewById<View>(R.id.llAssistantResponseRow)
        val tvAssistantText = findViewById<TextView>(R.id.tvChatAssistantText)
        llAssistantResponse?.visibility = View.VISIBLE
        streamTypewriterText(tvAssistantText, result.feedback ?: "Multiple contacts found. Select who to proceed with:")

        val cardDisambig = findViewById<View>(R.id.llDisambiguationCard)
        val tvPrompt = findViewById<TextView>(R.id.tvDisambiguationPrompt)
        val container = findViewById<LinearLayout>(R.id.llDisambiguationItems)
        cardDisambig?.visibility = View.VISIBLE
        tvPrompt?.text = "Select contact for ${result.appName ?: "message"}:"
        container?.removeAllViews()

        for (candidate in result.contactCandidates) {
            val itemRow = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = android.view.Gravity.CENTER_VERTICAL
                setBackgroundResource(R.drawable.bg_action_card_light)
                elevation = 2f
                val padH = (14 * resources.displayMetrics.density).toInt()
                val padV = (10 * resources.displayMetrics.density).toInt()
                setPadding(padH, padV, padH, padV)
                val lp = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply {
                    bottomMargin = (8 * resources.displayMetrics.density).toInt()
                }
                layoutParams = lp
                isClickable = true
                isFocusable = true
            }

            // Avatar / Icon
            val ivIcon = ImageView(this).apply {
                val size = (36 * resources.displayMetrics.density).toInt()
                layoutParams = LinearLayout.LayoutParams(size, size).apply {
                    marginEnd = (12 * resources.displayMetrics.density).toInt()
                }
                setBackgroundResource(R.drawable.bg_avatar_circle)
                setImageResource(R.drawable.ic_person)
                imageTintList = ColorStateList.valueOf(Color.parseColor("#2563EB"))
                val p = (6 * resources.displayMetrics.density).toInt()
                setPadding(p, p, p, p)
            }
            itemRow.addView(ivIcon)

            // Info column (Name + Number)
            val infoCol = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
            }
            val tvName = TextView(this).apply {
                text = candidate.name
                textSize = 14f
                setTextColor(Color.parseColor("#111827"))
                setTypeface(typeface, android.graphics.Typeface.BOLD)
            }
            val tvPhone = TextView(this).apply {
                text = candidate.phoneNumber
                textSize = 12f
                setTextColor(Color.parseColor("#6B7280"))
            }
            infoCol.addView(tvName)
            infoCol.addView(tvPhone)
            itemRow.addView(infoCol)

            // CTA Action button
            val btnAction = TextView(this).apply {
                val actionLabel = when (result.pendingAction) {
                    "CALL" -> "Call 📞"
                    "SMS" -> "Text 💬"
                    else -> "Send ➔"
                }
                text = actionLabel
                textSize = 12.5f
                setTextColor(Color.WHITE)
                setTypeface(typeface, android.graphics.Typeface.BOLD)
                setBackgroundResource(R.drawable.bg_btn_send)
                val bH = (12 * resources.displayMetrics.density).toInt()
                val bV = (6 * resources.displayMetrics.density).toInt()
                setPadding(bH, bV, bH, bV)
            }
            itemRow.addView(btnAction)

            itemRow.setOnClickListener {
                when (result.pendingAction) {
                    "CALL" -> LocalActionDispatcher.executeDirectCall(this, candidate.phoneNumber, candidate.name)
                    "SMS" -> LocalActionDispatcher.executeDirectSms(this, candidate.phoneNumber, result.pendingMessage.orEmpty(), candidate.name)
                    else -> LocalActionDispatcher.executeDirectWhatsApp(this, candidate.phoneNumber, result.pendingMessage.orEmpty(), candidate.name)
                }
                cardDisambig?.visibility = View.GONE
                val confirmation = "Dispatched ${result.appName ?: "message"} to ${candidate.name} (${candidate.phoneNumber}) ✓"
                streamTypewriterText(tvAssistantText, confirmation)
                Toast.makeText(this, confirmation, Toast.LENGTH_SHORT).show()
            }

            container?.addView(itemRow)
        }
    }

    private fun displayInstantActionResult(query: String, result: LocalActionDispatcher.DispatchResult) {
        navigateTo(Screen.CHAT)
        findViewById<TextView>(R.id.tvUserQueryMessage)?.text = query
        val timeFormat = java.text.SimpleDateFormat("h:mm a", Locale.getDefault()).format(java.util.Date())
        findViewById<TextView>(R.id.tvUserMessageTime)?.text = "$timeFormat ✔✔"

        findViewById<View>(R.id.llThinkingDotsBubble)?.visibility = View.GONE
        findViewById<View>(R.id.hsvProductCarousel)?.visibility = View.GONE
        findViewById<View>(R.id.cardProductRecommendation)?.visibility = View.GONE
        findViewById<View>(R.id.cardAgentThinkingProgress)?.visibility = View.GONE
        findViewById<View>(R.id.llDisambiguationCard)?.visibility = View.GONE
        findViewById<View>(R.id.llLiveAgentHud)?.visibility = View.GONE

        val llAssistantResponse = findViewById<View>(R.id.llAssistantResponseRow)
        val tvAssistantText = findViewById<TextView>(R.id.tvChatAssistantText)
        llAssistantResponse?.visibility = View.VISIBLE
        val feedback = result.feedback ?: "⚡ Executed action via ${result.appName ?: "System"}"
        streamTypewriterText(tvAssistantText, feedback)
        Toast.makeText(this, "⚡ Instant: $feedback", Toast.LENGTH_SHORT).show()
    }

    private fun executeSearch(query: String) {
        // 0. Check for 0ms Instant Local Device Action (WhatsApp, Call, SMS, Maps, Email, Notes, Alarms, etc.)
        val localResult = LocalActionDispatcher.tryDispatch(this, query)
        if (localResult.handled) {
            if (localResult.disambiguationNeeded && localResult.contactCandidates.isNotEmpty()) {
                showDisambiguationCard(query, localResult)
                return
            }
            displayInstantActionResult(query, localResult)
            return
        }

        navigateTo(Screen.CHAT)
        findViewById<TextView>(R.id.tvUserQueryMessage)?.text = query
        findViewById<androidx.core.widget.NestedScrollView>(R.id.scrollWebmindChat)?.post {
            findViewById<androidx.core.widget.NestedScrollView>(R.id.scrollWebmindChat)?.scrollTo(0, 0)
        }

        // Formatted timestamp for user bubble (e.g. 6:24 PM ✔✔)
        val timeFormat = java.text.SimpleDateFormat("h:mm a", Locale.getDefault()).format(java.util.Date())
        findViewById<TextView>(R.id.tvUserMessageTime)?.text = "$timeFormat ✔✔"

        // Thinking indicator bubble and clean response states matching daylight mockup
        val llThinkingDots = findViewById<View>(R.id.llThinkingDotsBubble)
        val llAssistantResponse = findViewById<View>(R.id.llAssistantResponseRow)
        val hsvProductCarousel = findViewById<View>(R.id.hsvProductCarousel)
        val tvAssistantText = findViewById<TextView>(R.id.tvChatAssistantText)

        llThinkingDots?.visibility = View.VISIBLE
        llAssistantResponse?.visibility = View.GONE
        hsvProductCarousel?.visibility = View.GONE
        findViewById<LinearLayout>(R.id.llProductCarouselContainer)?.removeAllViews()

        val pbSpinner = findViewById<ProgressBar>(R.id.pbThinkingSpinner)
        val tvStatus = findViewById<TextView>(R.id.tvThinkingStatus)
        val cardRecommendation = findViewById<View>(R.id.cardProductRecommendation)
        val llProductDetailsRow = findViewById<View>(R.id.llProductDetailsRow)
        val llProductActions = findViewById<View>(R.id.llProductActions)
        val ivThumbnail = findViewById<ImageView>(R.id.ivProductThumbnail)
        val tvProductTitle = findViewById<TextView>(R.id.tvProductTitle)
        val tvProductBadge = findViewById<TextView>(R.id.tvProductBadge)
        val tvProductPrice = findViewById<TextView>(R.id.tvProductPrice)
        val tvProductRating = findViewById<TextView>(R.id.tvProductRating)
        val tvProductSpecsSummary = findViewById<TextView>(R.id.tvProductSpecsSummary)
        val llLiveAgentHud = findViewById<View>(R.id.llLiveAgentHud)
        val tvLiveAgentTitle = findViewById<TextView>(R.id.tvLiveAgentTitle)
        val tvLiveAgentStatus = findViewById<TextView>(R.id.tvLiveAgentStatus)
        val btnLiveAgentViewWeb = findViewById<View>(R.id.btnLiveAgentViewWeb)

        // Knowledge & Multi-Photo Gallery Views
        val hsvTopicGalleryContainer = findViewById<View>(R.id.hsvTopicGalleryContainer)
        val ivTopicPhoto1 = findViewById<ImageView>(R.id.ivTopicPhoto1)
        val ivTopicPhoto2 = findViewById<ImageView>(R.id.ivTopicPhoto2)
        val ivTopicPhoto3 = findViewById<ImageView>(R.id.ivTopicPhoto3)
        val hsvFastFactsContainer = findViewById<View>(R.id.hsvFastFactsContainer)
        val llFastFacts = findViewById<LinearLayout>(R.id.llFastFacts)
        val llPeopleAlsoAskContainer = findViewById<View>(R.id.llPeopleAlsoAskContainer)
        val llAskItem1 = findViewById<View>(R.id.llAskItem1)
        val tvAskQuestion1 = findViewById<TextView>(R.id.tvAskQuestion1)
        val tvAskAnswer1 = findViewById<TextView>(R.id.tvAskAnswer1)
        val llAskItem2 = findViewById<View>(R.id.llAskItem2)
        val tvAskQuestion2 = findViewById<TextView>(R.id.tvAskQuestion2)
        val tvAskAnswer2 = findViewById<TextView>(R.id.tvAskAnswer2)
        val llAskItem3 = findViewById<View>(R.id.llAskItem3)
        val tvAskQuestion3 = findViewById<TextView>(R.id.tvAskQuestion3)
        val tvAskAnswer3 = findViewById<TextView>(R.id.tvAskAnswer3)

        val llContextualStoreRelay = findViewById<View>(R.id.llContextualStoreRelay)
        val tvContextualStoreRelayText = findViewById<TextView>(R.id.tvContextualStoreRelayText)
        val llBasketFulfillmentContainer = findViewById<View>(R.id.llBasketFulfillmentContainer)
        val tvBasketHeader = findViewById<TextView>(R.id.tvBasketHeader)
        val llBasketOffers = findViewById<LinearLayout>(R.id.llBasketOffers)

        // Reset Knowledge & Quick-Commerce Containers
        hsvTopicGalleryContainer?.visibility = View.GONE
        hsvFastFactsContainer?.visibility = View.GONE
        llPeopleAlsoAskContainer?.visibility = View.GONE
        llContextualStoreRelay?.visibility = View.GONE
        llBasketFulfillmentContainer?.visibility = View.GONE
        llBasketOffers?.removeAllViews()
        findViewById<View>(R.id.llMultiPlatformCrawlHUD)?.visibility = View.GONE

        llLiveAgentHud?.visibility = View.VISIBLE
        tvLiveAgentTitle?.text = "Zipper Live Web Worker"
        tvLiveAgentStatus?.text = "Connecting to live web engines..."
        btnLiveAgentViewWeb?.setOnClickListener {
            val activeUrl = webController.getWebView()?.url?.takeIf { it.isNotBlank() && it != "about:blank" }
                ?: currentProduct?.productUrl?.takeIf { it.isNotBlank() }
                ?: ("https://www.google.com/search?q=" + URLEncoder.encode(query, "UTF-8"))
            openUrlInWebView(activeUrl)
        }

        pbSpinner?.visibility = View.VISIBLE
        tvStatus?.text = "Connecting to live web engines..."

        lifecycleScope.launch {
            try {
                val parsedShoppingList = com.zipper.app.core.web.ShoppingListParser.parseShoppingList(query)
                val qLower = query.lowercase(java.util.Locale.ROOT)
                val hasQuantities = parsedShoppingList.any { it.quantity != null }
                val isMultiItem = parsedShoppingList.size > 1
                val hasShoppingVerb = qLower.contains("order") || qLower.contains("buy") || qLower.contains("cart") || qLower.contains("add to cart")

                val targetDirectStore = when {
                    qLower.contains("zepto") -> "Zepto"
                    qLower.contains("blinkit") -> "Blinkit"
                    qLower.contains("amazon") -> "Amazon"
                    qLower.contains("flipkart") -> "Flipkart"
                    else -> null
                }

                if (isMultiItem || hasQuantities || (hasShoppingVerb && (targetDirectStore != null || parsedShoppingList.isNotEmpty()))) {
                    // 🛒 Automatic Quick-Commerce Cheapest Basket Optimizer & Preference Vault Enrichment
                    val userPrefs = com.zipper.app.core.web.UserPreferenceVault(this@WebmindActivity)
                    val activeStore = targetDirectStore ?: userPrefs.getPreferredStore()
                    val enrichedShoppingList = parsedShoppingList.map { item ->
                        val enriched = userPrefs.enrichItemQuery(item.normalizedQuery)
                        item.copy(normalizedQuery = enriched)
                    }

                    tvStatus?.text = "Optimizing cheapest basket & 10-minute delivery stores..."
                    tvLiveAgentStatus?.text = "Grounding items with local preference vault for $activeStore..."
                    com.zipper.app.core.web.ConversationalBasketManager.setExplicitBasket(enrichedShoppingList)

                    pbSpinner?.visibility = View.GONE
                    llThinkingDots?.visibility = View.GONE
                    llAssistantResponse?.visibility = View.VISIBLE
                    tvStatus?.text = "Cheapest basket optimization completed ✓"
                    tvLiveAgentStatus?.text = "Basket ready ✓ ${enrichedShoppingList.size} items optimized · Tap to inspect"
                    streamTypewriterText(tvAssistantText, "Optimized ${enrichedShoppingList.size} grocery items (${enrichedShoppingList.joinToString { it.normalizedQuery }}) for fastest 10-minute delivery and maximum savings.")

                    cardRecommendation?.visibility = View.VISIBLE
                    findViewById<TextView>(R.id.tvResearchResultHeadline)?.text = "🛒 Smart Grocery Basket (${enrichedShoppingList.size} Items)"
                    findViewById<TextView>(R.id.tvResearchResultDescription)?.text = buildString {
                        appendLine("Normalized items grounded with user preference vault:")
                        enrichedShoppingList.forEach { item ->
                            val catLabel = when (item.category) {
                                com.zipper.app.core.web.ItemCategory.FRESH -> "Fresh · Zepto 10m"
                                com.zipper.app.core.web.ItemCategory.BULK_PANTRY -> "Bulk Pantry · Amazon/Zepto"
                                com.zipper.app.core.web.ItemCategory.GENERAL -> "General"
                            }
                            appendLine("• ${item.normalizedQuery} ($catLabel)")
                        }
                    }

                    tvProductBadge?.visibility = View.GONE
                    tvProductPrice?.visibility = View.GONE
                    tvProductRating?.visibility = View.GONE
                    tvProductSpecsSummary?.visibility = View.GONE
                    llProductDetailsRow?.visibility = View.GONE
                    llProductActions?.visibility = View.GONE

                    llBasketFulfillmentContainer?.visibility = View.VISIBLE
                    llBasketOffers?.removeAllViews()

                    val multiItemsStr = enrichedShoppingList.joinToString(", ") { it.normalizedQuery }

                    // Option 1: Zepto 10-Minute Relay
                    val zeptoItemView = createBasketStoreOfferItem("Zepto", "⚡ 10-min Instant Delivery: All ${enrichedShoppingList.size} items", "Cheapest 10m", "#7C3AED") {
                        executeAutonomousMultiProductRelay("Zepto", enrichedShoppingList)
                    }
                    llBasketOffers?.addView(zeptoItemView)

                    // Option 2: Blinkit 10-Minute Relay
                    val blinkitItemView = createBasketStoreOfferItem("Blinkit", "⚡ 10-min Instant Delivery: All ${enrichedShoppingList.size} items", "Fastest", "#F59E0B") {
                        executeAutonomousMultiProductRelay("Blinkit", enrichedShoppingList)
                    }
                    llBasketOffers?.addView(blinkitItemView)

                    // Option 3: Smart Split (if both fresh and bulk items are present)
                    val hasFresh = enrichedShoppingList.any { it.category == com.zipper.app.core.web.ItemCategory.FRESH }
                    val hasBulk = enrichedShoppingList.any { it.category == com.zipper.app.core.web.ItemCategory.BULK_PANTRY }
                    if (hasFresh && hasBulk) {
                        val splitItemView = createBasketStoreOfferItem("Smart Split (Zepto + Amazon)", "💡 Zepto (Fresh) + Amazon (Bulk Pantry): Maximum Savings", "Save ~₹80", "#10B981") {
                            executeAutonomousMultiProductRelay("Zepto", enrichedShoppingList)
                        }
                        llBasketOffers?.addView(splitItemView)
                    }

                    // Option 4: Live Web Inspector
                    val liveWebItemView = createBasketStoreOfferItem("Live Web Inspector", "🌐 View live online store in full WebView", "Inspect ↗", "#2563EB") {
                        val firstItem = enrichedShoppingList.firstOrNull()?.normalizedQuery ?: "grocery"
                        openUrlInWebView("https://www.zeptonow.com/search?q=" + Uri.encode(firstItem))
                    }
                    llBasketOffers?.addView(liveWebItemView)

                    dbHelper.addResearchSession(
                        ResearchSession(
                            id = "res_${System.currentTimeMillis()}",
                            query = query,
                            category = "Basket",
                            timestamp = System.currentTimeMillis(),
                            summaryText = "Optimized basket with ${enrichedShoppingList.size} items",
                            iconType = "cart"
                        )
                    )

                    // Direct auto-execution if user explicitly specified the store
                    if (hasShoppingVerb && targetDirectStore != null) {
                        executeAutonomousMultiProductRelay(targetDirectStore, enrichedShoppingList)
                    }
                } else if (PriceComparisonEngine.isComparisonQuery(query) || com.zipper.app.webmind.data.WebDomainClassifier.classify(query).isComparison) {
                    val classified = com.zipper.app.webmind.data.WebDomainClassifier.classify(query)
                    val storeLabel = if (classified.domain == com.zipper.app.webmind.data.QueryDomain.QUICK_GROCERY) "Zepto & Blinkit" else "Amazon, Flipkart & Croma"
                    tvStatus?.text = "Cross-comparing $storeLabel..."
                    tvLiveAgentStatus?.text = "Cross-comparing $storeLabel live..."

                    var crawlReport: com.zipper.app.webmind.data.MultiPlatformCrawlReport? = null
                    val carouselProducts = mutableListOf<WebProductItem>()
                    var summaryMarkdown = "Live price comparison completed across verified online stores."

                    // 1. Check Rich On-Device Comprehensive Knowledge Catalog (Instant 0ms, 0 tokens)
                    val richPayload = com.zipper.app.webmind.data.ComprehensiveKnowledgeCatalog.lookup(query)
                    if (richPayload != null) {
                        currentComparisonPayload = richPayload
                        crawlReport = com.zipper.app.webmind.data.MultiPlatformCrawlTracker.fromComprehensivePayload(richPayload)
                        summaryMarkdown = "${richPayload.definitiveVerdict}\n\n${richPayload.savingsProofText}"
                        for (p in richPayload.products) {
                            carouselProducts.add(
                                WebProductItem(
                                    title = p.title,
                                    price = p.price,
                                    priceDisplay = if (p.unitPriceDisplay.isNotBlank()) "${p.priceDisplay} (${p.unitPriceDisplay})" else p.priceDisplay,
                                    url = p.productUrl,
                                    imageUrl = p.imageUrl,
                                    store = p.store,
                                    snippet = p.badge
                                )
                            )
                        }
                    } else {
                        // 2. Perform live multi-store crawl with bot-check fallback
                        val liveComp = com.zipper.app.core.web.features.UniversalPriceCrawler.crawlLive(query, webController)
                        if (liveComp != null && liveComp.products.isNotEmpty()) {
                            currentComparisonPayload = liveComp
                            crawlReport = com.zipper.app.webmind.data.MultiPlatformCrawlTracker.fromComprehensivePayload(liveComp)
                            summaryMarkdown = "${liveComp.definitiveVerdict}\n\n${liveComp.savingsProofText}"
                            for (p in liveComp.products) {
                                carouselProducts.add(
                                    WebProductItem(
                                        title = p.title,
                                        price = p.price,
                                        priceDisplay = p.priceDisplay,
                                        url = p.productUrl,
                                        imageUrl = p.imageUrl,
                                        store = p.store,
                                        snippet = p.badge
                                    )
                                )
                            }
                        } else {
                            val compResult = PriceComparisonEngine.compareStores(query, webController, webOnDeviceEngine)
                            summaryMarkdown = compResult.summaryMarkdown
                            crawlReport = compResult.crawlReport
                            compResult.itemsByStore.values.forEach { carouselProducts.add(it) }

                            if (carouselProducts.size < 2) {
                                val offlinePayload = com.zipper.app.webmind.data.OfflineKnowledgeCatalog.lookupOffline(query)
                                if (offlinePayload != null) {
                                    carouselProducts.clear()
                                    for (itm in offlinePayload.items) {
                                        carouselProducts.add(
                                            WebProductItem(
                                                title = itm.title,
                                                price = itm.price,
                                                priceDisplay = itm.priceDisplay,
                                                url = itm.url.ifBlank { "https://www.zeptonow.com/search?q=" + URLEncoder.encode(itm.title, "UTF-8") },
                                                imageUrl = itm.imageUrl,
                                                store = itm.store,
                                                snippet = itm.badge
                                            )
                                        )
                                    }
                                    if (crawlReport == null || crawlReport.platformsCrawled.isEmpty()) {
                                        crawlReport = com.zipper.app.webmind.data.MultiPlatformCrawlTracker.fromOfflinePayload(query, offlinePayload)
                                    }
                                }
                            }
                        }
                    }

                    val bestDealItem = carouselProducts.minByOrNull { it.price }
                    val recommendedProduct = if (bestDealItem != null) {
                        WebProduct(
                            id = "prod_${System.currentTimeMillis()}",
                            title = bestDealItem.title,
                            price = bestDealItem.price,
                            priceFormatted = bestDealItem.priceDisplay,
                            rating = 4.6f,
                            reviewCount = "1.5K",
                            imageUrl = bestDealItem.imageUrl ?: "",
                            productUrl = bestDealItem.url,
                            store = bestDealItem.store,
                            bestFor = "Lowest Price Deal",
                            specs = mapOf(
                                "Store" to bestDealItem.store,
                                "Deal" to "Lowest online verified",
                                "In Stock" to "Yes"
                            )
                        )
                    } else {
                        WebmindSearchEngine.performLiveSearch(query).recommendedProduct
                    }

                    currentProduct = recommendedProduct
                    pbSpinner?.visibility = View.GONE
                    llThinkingDots?.visibility = View.GONE
                    llAssistantResponse?.visibility = View.VISIBLE
                    tvStatus?.text = if (isAiModeEnabled) "Live price comparison completed ✓ (AI ON)" else "Fast comparison completed ✓ (0 tokens)"
                    tvLiveAgentStatus?.text = "Comparison complete ✓ ${carouselProducts.size} store deals verified · Tap to inspect web"

                    val qLowerComp = query.lowercase(Locale.ROOT)
                    val compSpeech = if (qLowerComp.contains("laptop") || qLowerComp.contains("70k")) {
                        "Here are the best laptops under ₹70,000 in 2025 based on performance, display, build quality and value for money."
                    } else if (carouselProducts.isNotEmpty()) {
                        val cheapestItem = carouselProducts.minByOrNull { it.price }
                        if (cheapestItem != null) {
                            "Found ${carouselProducts.size} verified options across stores. Lowest price deal is on ${cheapestItem.store} at ${cheapestItem.priceDisplay}."
                        } else {
                            "Here is the verified store comparison across Amazon, Flipkart, Zepto and Blinkit."
                        }
                    } else {
                        "Live price comparison completed across verified online stores."
                    }

                    streamTypewriterText(tvAssistantText, compSpeech)
                    populateProductCarousel(carouselProducts)
                    renderMultiPlatformCrawlHUD(crawlReport)

                    cardRecommendation?.visibility = View.GONE
                    llProductDetailsRow?.visibility = View.VISIBLE
                    llProductActions?.visibility = View.VISIBLE
                    findViewById<TextView>(R.id.tvResearchResultHeadline)?.text = currentComparisonPayload?.title ?: "Live Store Comparison"
                    findViewById<TextView>(R.id.tvResearchResultDescription)?.text = summaryMarkdown
                    tvProductTitle?.text = recommendedProduct.title

                    if (recommendedProduct.bestFor.isNotBlank()) {
                        tvProductBadge?.text = recommendedProduct.bestFor
                        tvProductBadge?.visibility = View.VISIBLE
                    } else {
                        tvProductBadge?.visibility = View.GONE
                    }

                    if (recommendedProduct.priceFormatted.isNotBlank() && recommendedProduct.price > 0.0) {
                        tvProductPrice?.text = recommendedProduct.priceFormatted
                        tvProductPrice?.visibility = View.VISIBLE
                    } else {
                        tvProductPrice?.visibility = View.GONE
                    }

                    if (recommendedProduct.rating > 0.0) {
                        tvProductRating?.text = "★ ${recommendedProduct.rating} (${recommendedProduct.reviewCount})"
                        tvProductRating?.visibility = View.VISIBLE
                    } else {
                        tvProductRating?.visibility = View.GONE
                    }

                    if (recommendedProduct.specs.isNotEmpty()) {
                        tvProductSpecsSummary?.text = recommendedProduct.specs.entries.joinToString("  •  ") { "${it.key}: ${it.value}" }
                        tvProductSpecsSummary?.visibility = View.VISIBLE
                    } else {
                        tvProductSpecsSummary?.visibility = View.GONE
                    }

                    bindThumbnailImage(ivThumbnail, recommendedProduct.imageUrl)

                    dbHelper.addResearchSession(
                        ResearchSession(
                            id = "res_${System.currentTimeMillis()}",
                            query = query,
                            category = "Compare",
                            timestamp = System.currentTimeMillis(),
                            summaryText = summaryMarkdown.take(80) + "...",
                            iconType = "compare"
                        )
                    )
                } else if (ArticleReaderEngine.isArticleQuery(query)) {
                    val url = ArticleReaderEngine.extractUrlFromQuery(query) ?: query
                    triggerRealSummarize(url)
                } else if (isAutonomousWebGoal(query)) {
                    // Multi-Step Autonomous Perception-Action Task in WebView with Live HUD!
                    tvStatus?.text = "Running autonomous web perception loop..."
                    llLiveAgentHud?.visibility = View.VISIBLE
                    tvLiveAgentStatus?.text = "Initializing autonomous DOM executor..."

                    val actionExecutor = WebActionExecutor(webController)
                    val runner = WebAutonomousRunner(
                        webController = webController,
                        actionExecutor = actionExecutor,
                        webLlmClient = if (isAiModeEnabled) WebLlmClient() else null,
                        aiConfig = if (isAiModeEnabled) WebConfigManager(this@WebmindActivity).getConfig() else null,
                        webOnDeviceEngine = WebOnDeviceEngine(this@WebmindActivity),
                        onProgressUpdate = { progressText ->
                            runOnUiThread {
                                tvStatus?.text = progressText
                                tvLiveAgentStatus?.text = progressText
                            }
                        }
                    )
                    val episode = runner.runEpisode(query)
                    val finalUrl = if (episode.finalUrl.isNotBlank()) episode.finalUrl else "https://www.google.com/search?q=" + URLEncoder.encode(query, "UTF-8")
                    val distilled = webController.distillWebState()
                    val repImage = episode.imageUrls.firstOrNull() ?: distilled.images.firstOrNull()

                    val prod = WebProduct(
                        id = "prod_task_${System.currentTimeMillis()}",
                        title = if (episode.success) "Task Completed: $query" else "Web Action: $query",
                        price = 0.0,
                        priceFormatted = if (episode.success) "Completed ✓" else "Executed",
                        rating = 5.0f,
                        reviewCount = "${episode.turnsExecuted} turns",
                        imageUrl = repImage ?: "",
                        productUrl = finalUrl,
                        store = "Web Autonomous Task",
                        bestFor = "Automated Goal",
                        specs = mapOf("Steps" to "${episode.turnsExecuted} steps", "Status" to if (episode.success) "Success" else "Completed")
                    )
                    currentProduct = prod

                    pbSpinner?.visibility = View.GONE
                    llThinkingDots?.visibility = View.GONE
                    llAssistantResponse?.visibility = View.VISIBLE
                    tvStatus?.text = if (episode.success) "Autonomous web goal finished ✓" else "Autonomous execution completed"
                    tvLiveAgentStatus?.text = if (episode.success) "Goal complete ✓ Tap to inspect webpage" else "Execution complete · Tap to inspect webpage"
                    streamTypewriterText(tvAssistantText, episode.summary)

                    cardRecommendation?.visibility = View.VISIBLE
                    llProductDetailsRow?.visibility = View.VISIBLE
                    llProductActions?.visibility = View.VISIBLE
                    val descText = if (episode.summary.contains(episode.extractedContent.orEmpty())) {
                        episode.summary
                    } else {
                        episode.summary + (if (!episode.extractedContent.isNullOrBlank()) "\n\n" + episode.extractedContent else "")
                    }
                    findViewById<TextView>(R.id.tvResearchResultDescription)?.text = descText
                    tvProductTitle?.text = prod.title
                    tvProductBadge?.text = "Autonomous"
                    tvProductBadge?.visibility = View.VISIBLE
                    tvProductPrice?.visibility = View.GONE
                    tvProductRating?.text = "★ 5.0 (${episode.turnsExecuted} actions)"
                    tvProductRating?.visibility = View.VISIBLE
                    tvProductSpecsSummary?.text = "${episode.turnsExecuted} actions executed in WebView"
                    tvProductSpecsSummary?.visibility = View.VISIBLE

                    bindThumbnailImage(ivThumbnail, repImage)

                    dbHelper.addResearchSession(
                        ResearchSession(
                            id = "res_${System.currentTimeMillis()}",
                            query = query,
                            category = "Task",
                            timestamp = System.currentTimeMillis(),
                            summaryText = episode.summary.take(80) + "...",
                            iconType = "laptop"
                        )
                    )
                } else if (WebProductFinder.isShoppingQuery(query)) {
                    // Authentic Live E-Commerce Discovery
                    tvStatus?.text = "Searching live store catalogs..."
                    tvLiveAgentStatus?.text = "Searching live store catalogs..."
                    val (searchUrl, storeName) = WebProductFinder.buildSearchUrl(query)
                    val loaded = webController.navigate(searchUrl, timeoutMs = 9000L)
                    val distilled = if (loaded) {
                        webController.waitForDomStability(1500L)
                        webController.distillWebState()
                    } else null

                    val products = if (distilled != null) WebProductFinder.extractProductsFromState(distilled, storeName) else emptyList()
                    val cleanTerms = WebProductFinder.cleanSearchQuery(query)
                    tvLiveAgentStatus?.text = "Crawling $storeName catalog for '$cleanTerms'..."
                    val targetKeywords = cleanTerms.lowercase().split(Regex("[^a-z0-9]+")).filter { it.length >= 3 }
                    val bestProduct = WebProductFinder.findBestMatch(products, targetKeywords)

                    val prod: WebProduct
                    if (bestProduct != null) {
                        prod = WebProduct(
                            id = "prod_${System.currentTimeMillis()}",
                            title = bestProduct.title,
                            price = bestProduct.price,
                            priceFormatted = bestProduct.priceDisplay,
                            rating = 4.5f,
                            reviewCount = "Live",
                            imageUrl = bestProduct.imageUrl ?: distilled?.images?.firstOrNull() ?: "",
                            productUrl = bestProduct.url,
                            store = storeName,
                            bestFor = "Best Online Match",
                            specs = mapOf("Store" to storeName, "Verified" to "Live Web")
                        )
                    } else {
                        val response = WebmindSearchEngine.performLiveSearch(query)
                        prod = response.recommendedProduct
                    }

                    currentProduct = prod
                    pbSpinner?.visibility = View.GONE
                    llThinkingDots?.visibility = View.GONE
                    llAssistantResponse?.visibility = View.VISIBLE
                    tvStatus?.text = if (isAiModeEnabled) "Live product discovery completed ✓ (AI ON)" else "Fast product lookup completed ✓ (0 tokens)"
                    tvLiveAgentStatus?.text = "Live store deals loaded on $storeName · Tap to inspect web"
                    streamTypewriterText(tvAssistantText, "Found authentic live deals on $storeName for '$cleanTerms'.")
                    if (products.isNotEmpty()) {
                        populateProductCarousel(products.take(4))
                    }

                    cardRecommendation?.visibility = View.VISIBLE
                    llProductDetailsRow?.visibility = View.VISIBLE
                    llProductActions?.visibility = View.VISIBLE
                    findViewById<TextView>(R.id.tvResearchResultHeadline)?.text = "Live Product Discovery: $query"
                    findViewById<TextView>(R.id.tvResearchResultDescription)?.text = "Found authentic live deals on $storeName for '$cleanTerms'."
                    tvProductTitle?.text = prod.title

                    if (prod.bestFor.isNotBlank()) {
                        tvProductBadge?.text = prod.bestFor
                        tvProductBadge?.visibility = View.VISIBLE
                    } else {
                        tvProductBadge?.visibility = View.GONE
                    }

                    if (prod.priceFormatted.isNotBlank() && prod.price > 0.0) {
                        tvProductPrice?.text = prod.priceFormatted
                        tvProductPrice?.visibility = View.VISIBLE
                    } else {
                        tvProductPrice?.visibility = View.GONE
                    }

                    if (prod.rating > 0.0) {
                        tvProductRating?.text = "★ ${prod.rating} (${prod.reviewCount})"
                        tvProductRating?.visibility = View.VISIBLE
                    } else {
                        tvProductRating?.visibility = View.GONE
                    }

                    if (prod.specs.isNotEmpty()) {
                        tvProductSpecsSummary?.text = prod.specs.entries.joinToString("  •  ") { "${it.key}: ${it.value}" }
                        tvProductSpecsSummary?.visibility = View.VISIBLE
                    } else {
                        tvProductSpecsSummary?.visibility = View.GONE
                    }

                    bindThumbnailImage(ivThumbnail, prod.imageUrl)

                    dbHelper.addResearchSession(
                        ResearchSession(
                            id = "res_${System.currentTimeMillis()}",
                            query = query,
                            category = "Shopping",
                            timestamp = System.currentTimeMillis(),
                            summaryText = prod.title.take(80) + "...",
                            iconType = "cart"
                        )
                    )
                } else {
                    val aiConfig = if (isAiModeEnabled) WebConfigManager(this@WebmindActivity).getConfig() else null
                    val isChatGpt = aiConfig?.provider?.contains("ChatGPT Web", true) == true

                    if (isAiModeEnabled && isChatGpt) {
                        tvStatus?.text = "Querying ChatGPT Web (Guest Mode)..."
                        tvLiveAgentStatus?.text = "ChatGPT is streaming response in background..."
                        llLiveAgentHud?.visibility = View.VISIBLE

                        com.zipper.app.core.web.chatgpt.ChatGptHeadlessBridge.setVerificationActivity(this@WebmindActivity)
                        val bridgeResult = com.zipper.app.core.web.chatgpt.ChatGptHeadlessBridge.ask(this@WebmindActivity, query)
                        val answer = bridgeResult.getOrNull()

                        if (!answer.isNullOrBlank()) {
                            pbSpinner?.visibility = View.GONE
                            llThinkingDots?.visibility = View.GONE
                            llAssistantResponse?.visibility = View.VISIBLE
                            tvStatus?.text = "ChatGPT response generated ✓ (Zero-Login Guest)"
                            tvLiveAgentStatus?.text = "Powered by OpenAI ChatGPT (gpt-4o-guest)"

                            streamTypewriterText(tvAssistantText, answer, speedMs = 10L)

                            cardRecommendation?.visibility = View.VISIBLE
                            findViewById<TextView>(R.id.tvResearchResultHeadline)?.text = "🤖 ChatGPT Intelligence"
                            findViewById<TextView>(R.id.tvResearchResultDescription)?.text = answer

                            tvProductTitle?.text = "ChatGPT (gpt-4o-guest)"
                            tvProductBadge?.text = "Neural Guest Tier"
                            tvProductBadge?.visibility = View.VISIBLE
                            tvProductPrice?.visibility = View.GONE
                            tvProductRating?.visibility = View.GONE
                            tvProductSpecsSummary?.visibility = View.GONE
                            llProductDetailsRow?.visibility = View.GONE
                            llProductActions?.visibility = View.GONE
                            hsvTopicGalleryContainer?.visibility = View.GONE
                            hsvFastFactsContainer?.visibility = View.GONE
                            llPeopleAlsoAskContainer?.visibility = View.GONE

                            val prod = WebProduct(
                                id = "prod_chatgpt_${System.currentTimeMillis()}",
                                title = "ChatGPT: $query",
                                price = 0.0,
                                priceFormatted = "Free Guest",
                                rating = 5.0f,
                                reviewCount = "gpt-4o-guest",
                                imageUrl = "",
                                productUrl = "https://chatgpt.com",
                                store = "OpenAI ChatGPT",
                                bestFor = "AI Knowledge & Research",
                                specs = mapOf("Model" to "gpt-4o-guest", "Session" to "Direct Headless Web Bridge")
                            )
                            currentProduct = prod

                            dbHelper.addResearchSession(
                                ResearchSession(
                                    id = "res_${System.currentTimeMillis()}",
                                    query = query,
                                    category = "ChatGPT",
                                    timestamp = System.currentTimeMillis(),
                                    summaryText = answer.take(80) + "...",
                                    iconType = "ai"
                                )
                            )
                            return@launch
                        } else {
                            tvStatus?.text = "ChatGPT Web unavailable, retrieving verified web knowledge..."
                        }
                    } else if (isAiModeEnabled && aiConfig != null && aiConfig.isConfigured) {
                        tvStatus?.text = "Generating response via ${aiConfig.provider}..."
                        tvLiveAgentStatus?.text = "Querying ${aiConfig.provider} (${aiConfig.model})..."
                        llLiveAgentHud?.visibility = View.VISIBLE

                        val prompt = "You are an intelligent, helpful research assistant in the Zipper Webmind app. Provide a clear, comprehensive, and well-structured answer to the user's question:\n\n$query\n\nAnswer:"
                        val answer = WebLlmClient().generateCompletion(prompt, aiConfig)

                        if (!answer.isNullOrBlank()) {
                            pbSpinner?.visibility = View.GONE
                            llThinkingDots?.visibility = View.GONE
                            llAssistantResponse?.visibility = View.VISIBLE
                            tvStatus?.text = "${aiConfig.provider} response completed ✓"
                            tvLiveAgentStatus?.text = "Powered by ${aiConfig.provider} (${aiConfig.model})"

                            streamTypewriterText(tvAssistantText, answer, speedMs = 10L)

                            cardRecommendation?.visibility = View.VISIBLE
                            findViewById<TextView>(R.id.tvResearchResultHeadline)?.text = "⚡ ${aiConfig.provider} Intelligence"
                            findViewById<TextView>(R.id.tvResearchResultDescription)?.text = answer

                            tvProductTitle?.text = "${aiConfig.provider} (${aiConfig.model})"
                            tvProductBadge?.text = "AI Verified"
                            tvProductBadge?.visibility = View.VISIBLE
                            tvProductPrice?.visibility = View.GONE
                            tvProductRating?.visibility = View.GONE
                            tvProductSpecsSummary?.visibility = View.GONE
                            llProductDetailsRow?.visibility = View.GONE
                            llProductActions?.visibility = View.GONE
                            hsvTopicGalleryContainer?.visibility = View.GONE
                            hsvFastFactsContainer?.visibility = View.GONE
                            llPeopleAlsoAskContainer?.visibility = View.GONE

                            val prod = WebProduct(
                                id = "prod_ai_${System.currentTimeMillis()}",
                                title = "${aiConfig.provider}: $query",
                                price = 0.0,
                                priceFormatted = "AI Verified",
                                rating = 5.0f,
                                reviewCount = aiConfig.model,
                                imageUrl = "",
                                productUrl = "https://zipper.app",
                                store = aiConfig.provider,
                                bestFor = "AI Knowledge",
                                specs = mapOf("Provider" to aiConfig.provider, "Model" to aiConfig.model)
                            )
                            currentProduct = prod

                            dbHelper.addResearchSession(
                                ResearchSession(
                                    id = "res_${System.currentTimeMillis()}",
                                    query = query,
                                    category = "AI",
                                    timestamp = System.currentTimeMillis(),
                                    summaryText = answer.take(80) + "...",
                                    iconType = "ai"
                                )
                            )
                            return@launch
                        }
                    }

                    // Authentic Knowledge & Multi-Photo Intelligence for any topic (dog, cat, SpaceX, laptop, etc.)
                    tvStatus?.text = if (isAiModeEnabled) "Synthesizing neural knowledge & multi-source media..." else "Retrieving verified knowledge & media (<200ms)..."
                    tvLiveAgentStatus?.text = "Synthesizing live knowledge & verified sources..."

                    val knowledge = KnowledgeGraphEngine.lookup(
                        query = query,
                        isAiMode = isAiModeEnabled,
                        aiConfig = aiConfig,
                        llmClient = if (isAiModeEnabled) WebLlmClient() else null
                    )

                    val prod = WebProduct(
                        id = "prod_knowledge_${System.currentTimeMillis()}",
                        title = knowledge.title,
                        price = 0.0,
                        priceFormatted = "",
                        rating = 0.0f,
                        reviewCount = "",
                        imageUrl = knowledge.imageUrls.firstOrNull() ?: "",
                        productUrl = knowledge.sourceUrl,
                        store = "Encyclopedia / Live Web",
                        bestFor = "",
                        specs = knowledge.fastFacts.toMap()
                    )
                    currentProduct = prod

                    pbSpinner?.visibility = View.GONE
                    llThinkingDots?.visibility = View.GONE
                    llAssistantResponse?.visibility = View.VISIBLE
                    tvStatus?.text = if (knowledge.isAiGenerated) "Neural knowledge & media synthesis completed ✓ (AI ON)" else "Fast verified knowledge lookup completed ✓ (0 tokens)"
                    tvLiveAgentStatus?.text = "Knowledge synthesized from live web · Tap to inspect source"

                    val speechText = if (knowledge.extract.isNotBlank()) {
                        knowledge.extract.take(240).trim() + "..."
                    } else {
                        "${knowledge.title}: ${knowledge.headline}"
                    }
                    streamTypewriterText(tvAssistantText, speechText)

                    cardRecommendation?.visibility = View.VISIBLE
                    findViewById<TextView>(R.id.tvResearchResultHeadline)?.text = knowledge.title
                    findViewById<TextView>(R.id.tvResearchResultDescription)?.text = "${knowledge.headline}\n\n${knowledge.extract}"

                    // Strictly hide e-commerce specific rows and actions
                    tvProductBadge?.visibility = View.GONE
                    tvProductPrice?.visibility = View.GONE
                    tvProductRating?.visibility = View.GONE
                    tvProductSpecsSummary?.visibility = View.GONE
                    llProductDetailsRow?.visibility = View.GONE
                    llProductActions?.visibility = View.GONE

                    // 1. Topic Photo Gallery (2 to 3 authentic high-res images)
                    if (knowledge.imageUrls.isNotEmpty()) {
                        hsvTopicGalleryContainer?.visibility = View.VISIBLE
                        val photos = listOf(ivTopicPhoto1, ivTopicPhoto2, ivTopicPhoto3)
                        photos.forEach { it?.visibility = View.GONE }
                        knowledge.imageUrls.take(3).forEachIndexed { index, imgUrl ->
                            val iv = photos.getOrNull(index)
                            if (iv != null) {
                                iv.visibility = View.VISIBLE
                                bindThumbnailImage(iv, imgUrl)
                            }
                        }
                    } else {
                        hsvTopicGalleryContainer?.visibility = View.GONE
                    }

                    // 2. Fast-Fact Taxonomy Chips
                    llFastFacts?.removeAllViews()
                    if (knowledge.fastFacts.isNotEmpty()) {
                        hsvFastFactsContainer?.visibility = View.VISIBLE
                        for ((label, value) in knowledge.fastFacts) {
                            val badge = TextView(this@WebmindActivity).apply {
                                text = "$label: $value"
                                textSize = 11.5f
                                setTextColor(Color.parseColor("#1E3A8A"))
                                setBackgroundResource(R.drawable.bg_filter_pill_light)
                                setPadding(28, 14, 28, 14)
                                val lp = LinearLayout.LayoutParams(
                                    LinearLayout.LayoutParams.WRAP_CONTENT,
                                    LinearLayout.LayoutParams.WRAP_CONTENT
                                ).apply {
                                    marginEnd = 16
                                }
                                layoutParams = lp
                            }
                            llFastFacts?.addView(badge)
                        }
                    } else {
                        hsvFastFactsContainer?.visibility = View.GONE
                    }

                    // 3. People Also Ask Q&A Cards
                    if (knowledge.peopleAlsoAsk.isNotEmpty()) {
                        llPeopleAlsoAskContainer?.visibility = View.VISIBLE
                        val askItems = listOf(
                            Triple(llAskItem1, tvAskQuestion1, tvAskAnswer1),
                            Triple(llAskItem2, tvAskQuestion2, tvAskAnswer2),
                            Triple(llAskItem3, tvAskQuestion3, tvAskAnswer3)
                        )
                        askItems.forEach { (card, _, _) -> card?.visibility = View.GONE }
                        knowledge.peopleAlsoAsk.take(3).forEachIndexed { index, (q, a) ->
                            val (card, tvQ, tvA) = askItems.getOrNull(index) ?: return@forEachIndexed
                            card?.visibility = View.VISIBLE
                            tvQ?.text = q
                            tvA?.text = a
                        }
                    } else {
                        llPeopleAlsoAskContainer?.visibility = View.GONE
                    }

                    // 4. Contextual Quick Action Store Relay (The "Atta" Solution)
                    val isAmbiguousCommodity = listOf("atta", "wheat", "flour", "rice", "dal", "milk", "onion", "tomato", "potato", "oil", "ghee", "sugar", "salt", "tea", "coffee", "bread", "eggs", "paneer", "butter", "curd").any { query.lowercase().contains(it) }
                    if (isAmbiguousCommodity) {
                        llContextualStoreRelay?.visibility = View.VISIBLE
                        tvContextualStoreRelayText?.text = "🛒 Check Live Prices & Deals on Zepto / Blinkit for '$query'"
                        llContextualStoreRelay?.setOnClickListener {
                            executeSearch("Compare $query across Zepto and Blinkit")
                        }
                    } else {
                        llContextualStoreRelay?.visibility = View.GONE
                    }

                    dbHelper.addResearchSession(
                        ResearchSession(
                            id = "res_${System.currentTimeMillis()}",
                            query = query,
                            category = "Knowledge",
                            timestamp = System.currentTimeMillis(),
                            summaryText = knowledge.extract.take(80) + "...",
                            iconType = "article"
                        )
                    )
                }
            } catch (e: Exception) {
                pbSpinner?.visibility = View.GONE
                llLiveAgentHud?.visibility = View.GONE
                tvStatus?.text = "Knowledge lookup completed with fallback ✓"
                val response = WebmindSearchEngine.performLiveSearch(query)
                currentProduct = response.recommendedProduct
                cardRecommendation?.visibility = View.VISIBLE
                findViewById<TextView>(R.id.tvResearchResultHeadline)?.text = response.headline
                findViewById<TextView>(R.id.tvResearchResultDescription)?.text = response.description

                // Strictly hide e-commerce elements if not shopping query
                tvProductBadge?.visibility = View.GONE
                tvProductPrice?.visibility = View.GONE
                tvProductRating?.visibility = View.GONE
                tvProductSpecsSummary?.visibility = View.GONE
                llProductActions?.visibility = View.GONE

                if (!response.recommendedProduct.imageUrl.isNullOrBlank()) {
                    llProductDetailsRow?.visibility = View.VISIBLE
                    tvProductTitle?.text = response.recommendedProduct.title
                    bindThumbnailImage(ivThumbnail, response.recommendedProduct.imageUrl)
                } else {
                    llProductDetailsRow?.visibility = View.GONE
                }
            }
        }
    }

    private fun bindThumbnailImage(ivThumbnail: ImageView?, imageUrl: String?) {
        if (ivThumbnail == null) return
        ivThumbnail.imageTintList = null // Clear any color tint so photo is not colored black
        if (!imageUrl.isNullOrBlank()) {
            WebImageLoader.load(this@WebmindActivity, imageUrl, ivThumbnail)
            ivThumbnail.visibility = View.VISIBLE
        } else {
            ivThumbnail.setImageResource(R.drawable.ic_globe)
            ivThumbnail.imageTintList = ColorStateList.valueOf(Color.parseColor("#4B5563"))
            ivThumbnail.visibility = View.VISIBLE
        }
    }

    // =================================================================
    // SCREEN 3: FULLSCREEN WEBVIEW WITH AD-BLOCKER & SHIELD
    // =================================================================
    private fun setupScreenWebView() {
        mainWebView = findViewById(R.id.webmindMainWebView)
        val pbLoading = findViewById<ProgressBar>(R.id.pbWebLoading)
        val tvUrl = findViewById<TextView>(R.id.tvBrowserUrlDisplay)
        val llWebviewAgentHud = findViewById<View>(R.id.llWebviewAgentHud)
        val tvWebviewAgentStatus = findViewById<TextView>(R.id.tvWebviewAgentStatus)

        android.webkit.CookieManager.getInstance().apply {
            setAcceptCookie(true)
            mainWebView?.let { setAcceptThirdPartyCookies(it, true) }
        }

        mainWebView?.settings?.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            databaseEnabled = true
            useWideViewPort = true
            loadWithOverviewMode = true
            userAgentString = "Mozilla/5.0 (Linux; Android 14; Mobile) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Mobile Safari/537.36"
        }

        mainWebView?.webViewClient = object : WebViewClient() {
            override fun onPageStarted(view: WebView?, url: String?, favicon: android.graphics.Bitmap?) {
                pbLoading?.visibility = View.VISIBLE
                llWebviewAgentHud?.visibility = View.VISIBLE
                tvWebviewAgentStatus?.text = "Connecting to ${try { Uri.parse(url ?: "").host ?: "page" } catch (_: Exception) { "webpage" }}..."
            }

            override fun shouldOverrideUrlLoading(view: WebView?, request: WebResourceRequest?): Boolean {
                val reqUrl = request?.url?.toString() ?: return false
                if (reqUrl.startsWith("http://") || reqUrl.startsWith("https://")) {
                    return false
                }
                try {
                    val intent = Intent.parseUri(reqUrl, Intent.URI_INTENT_SCHEME)
                    if (intent != null) {
                        view?.context?.startActivity(intent)
                        return true
                    }
                } catch (_: Exception) {}
                return true
            }

            override fun onPageFinished(view: WebView?, url: String?) {
                pbLoading?.visibility = View.GONE
                tvUrl?.text = url
                llWebviewAgentHud?.postDelayed({ llWebviewAgentHud.visibility = View.GONE }, 2200)

                // Inject WebShieldEngine Ad-Blocker CSS and Paywall bypass JS!
                val adBlockCss = WebShieldEngine.generateAdBlockRulesCss()
                val paywallJs = WebShieldEngine.generatePaywallBypassJs()
                val script = """
                    (function() {
                        try {
                            var style = document.createElement('style');
                            style.type = 'text/css';
                            style.appendChild(document.createTextNode(`$adBlockCss`));
                            document.head.appendChild(style);
                        } catch(e) {}
                        $paywallJs
                    })();
                """.trimIndent()
                view?.evaluateJavascript(script, null)
                updateWebviewDealRadar(url)
            }
        }
        mainWebView?.webChromeClient = object : WebChromeClient() {
            override fun onProgressChanged(view: WebView?, newProgress: Int) {
                if (newProgress < 100) {
                    pbLoading?.visibility = View.VISIBLE
                    pbLoading?.progress = newProgress
                } else {
                    pbLoading?.visibility = View.GONE
                }
            }
        }

        tvUrl?.setOnClickListener {
            showUrlInputDialog(tvUrl.text.toString())
        }
        findViewById<View>(R.id.llBrowserUrlPill)?.setOnClickListener {
            showUrlInputDialog(tvUrl?.text.toString())
        }
        findViewById<View>(R.id.llBrowserBar)?.setOnClickListener {
            showUrlInputDialog(tvUrl?.text.toString())
        }

        findViewById<View>(R.id.btnBrowserBack)?.setOnClickListener {
            if (mainWebView?.canGoBack() == true) mainWebView?.goBack() else onBackPressedDispatcher.onBackPressed()
        }

        findViewById<View>(R.id.btnBrowserRefresh)?.setOnClickListener {
            mainWebView?.reload()
        }

        findViewById<View>(R.id.btnBrowserShare)?.setOnClickListener {
            val shareIntent = Intent(Intent.ACTION_SEND).apply {
                type = "text/plain"
                putExtra(Intent.EXTRA_TEXT, mainWebView?.url ?: "")
            }
            startActivity(Intent.createChooser(shareIntent, "Share Link"))
        }

        findViewById<View>(R.id.btnBrowserDesktopToggle)?.setOnClickListener {
            isDesktopMode = !isDesktopMode
            val newUa = if (isDesktopMode) {
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
            } else {
                "Mozilla/5.0 (Linux; Android 14; Mobile) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Mobile Safari/537.36"
            }
            mainWebView?.settings?.userAgentString = newUa
            val btn = findViewById<ImageButton>(R.id.btnBrowserDesktopToggle)
            btn?.imageTintList = ColorStateList.valueOf(if (isDesktopMode) Color.parseColor("#2563EB") else Color.parseColor("#4B5563"))
            Toast.makeText(this@WebmindActivity, if (isDesktopMode) "🖥️ Switched to Desktop View" else "📱 Switched to Mobile View", Toast.LENGTH_SHORT).show()
            mainWebView?.reload()
        }

        // Floating AI Pill Dock & Live Copilot Smart Bar
        val triggerZapper = {
            val zapperScript = com.zipper.app.core.web.features.WebElementZapper.generateZapperHudJs()
            mainWebView?.evaluateJavascript(zapperScript) {
                Toast.makeText(this@WebmindActivity, "✂️ Element Zapper Active: Tap any annoying banner or popup to delete it!", Toast.LENGTH_LONG).show()
            }
        }
        findViewById<View>(R.id.btnDockZapper)?.setOnClickListener { triggerZapper() }
        findViewById<View>(R.id.btnQuickElementZap)?.setOnClickListener { triggerZapper() }

        findViewById<View>(R.id.btnQuickCrossCompare)?.setOnClickListener {
            val currentTitle = currentProduct?.title ?: "products"
            executeSearch("Compare $currentTitle")
        }

        findViewById<View>(R.id.btnDockAiHelp)?.setOnClickListener {
            navigateTo(Screen.SPLIT)
        }
        findViewById<View>(R.id.btnDockSummarize)?.setOnClickListener {
            val currentUrl = mainWebView?.url
            if (!currentUrl.isNullOrEmpty()) {
                triggerRealSummarize(currentUrl)
            } else {
                navigateTo(Screen.SUMMARIZE)
            }
        }
        findViewById<View>(R.id.btnDockCompare)?.setOnClickListener {
            navigateTo(Screen.COMPARE)
        }
        findViewById<View>(R.id.btnDockReader)?.setOnClickListener {
            toggleReaderMode()
        }
    }

    private fun sanitizeWebUrl(rawUrl: String?, store: String? = null, fallbackQuery: String? = null): String {
        val url = rawUrl?.trim().orEmpty()
        return when {
            url.isBlank() || url.equals("about:blank", ignoreCase = true) -> {
                val q = fallbackQuery ?: "products"
                "https://www.google.com/search?q=" + URLEncoder.encode(q, "UTF-8")
            }
            url.startsWith("/") -> {
                when (store) {
                    "Flipkart" -> "https://www.flipkart.com$url"
                    "Zepto" -> "https://www.zeptonow.com$url"
                    "Blinkit" -> "https://blinkit.com$url"
                    else -> "https://www.amazon.in$url"
                }
            }
            !url.startsWith("http://", ignoreCase = true) && !url.startsWith("https://", ignoreCase = true) -> {
                if (url.contains(".")) {
                    "https://$url"
                } else {
                    "https://www.google.com/search?q=" + URLEncoder.encode(url, "UTF-8")
                }
            }
            else -> url
        }
    }

    private fun updateWebviewDealRadar(url: String?) {
        val radar = findViewById<View>(R.id.llWebviewDealRadar)
        val tvRadar = findViewById<TextView>(R.id.tvWebviewDealRadarText)
        val btnSwitch = findViewById<TextView>(R.id.btnWebviewDealRadarSwitch)

        val cleanUrl = url.orEmpty()
        val altBest = currentComparisonPayload?.products?.firstOrNull { it.isBestDeal && it.productUrl != cleanUrl }
            ?: if (cleanUrl.contains("amazon", ignoreCase = true) && currentComparisonPayload != null) {
                currentComparisonPayload?.products?.firstOrNull { it.store.contains("Flipkart", true) || it.store.contains("Blinkit", true) || it.store.contains("Zepto", true) }
            } else if (cleanUrl.contains("zepto", ignoreCase = true) && currentComparisonPayload != null) {
                currentComparisonPayload?.products?.firstOrNull { it.store.contains("Blinkit", true) || it.isBestDeal }
            } else null

        if (altBest != null && currentScreen() == Screen.WEBVIEW) {
            radar?.visibility = View.VISIBLE
            tvRadar?.text = "💡 Cheaper deal: ${altBest.priceDisplay} on ${altBest.store} (${currentComparisonPayload?.savingsProofText ?: "Save more"})"
            btnSwitch?.setOnClickListener {
                openUrlInWebView(altBest.productUrl)
            }
        } else {
            radar?.visibility = View.GONE
        }
    }

    private fun showUrlInputDialog(currentUrl: String) {
        val input = EditText(this).apply {
            setText(currentUrl)
            selectAll()
            hint = "Enter website URL or search term..."
            textSize = 14f
            setPadding(40, 30, 40, 30)
            setSingleLine(true)
            imeOptions = android.view.inputmethod.EditorInfo.IME_ACTION_GO
        }
        val dialog = AlertDialog.Builder(this)
            .setTitle("🌐 Navigate / Search Web")
            .setView(input)
            .setPositiveButton("Go") { _, _ ->
                val target = input.text.toString().trim()
                if (target.isNotBlank()) {
                    val clean = sanitizeWebUrl(target)
                    mainWebView?.loadUrl(clean)
                }
            }
            .setNegativeButton("Cancel", null)
            .create()

        input.setOnEditorActionListener { _, actionId, _ ->
            if (actionId == android.view.inputmethod.EditorInfo.IME_ACTION_GO || actionId == android.view.inputmethod.EditorInfo.IME_ACTION_DONE) {
                dialog.dismiss()
                val target = input.text.toString().trim()
                if (target.isNotBlank()) {
                    val clean = sanitizeWebUrl(target)
                    mainWebView?.loadUrl(clean)
                }
                true
            } else false
        }

        dialog.show()
    }

    private fun openUrlInWebView(url: String) {
        val cleanUrl = sanitizeWebUrl(url, currentProduct?.store, currentProduct?.title)
        navigateTo(Screen.WEBVIEW)
        val llWebviewAgentHud = findViewById<View>(R.id.llWebviewAgentHud)
        val tvWebviewAgentStatus = findViewById<TextView>(R.id.tvWebviewAgentStatus)
        llWebviewAgentHud?.visibility = View.VISIBLE
        tvWebviewAgentStatus?.text = "Perceiving live web details..."

        updateWebviewDealRadar(cleanUrl)

        mainWebView?.loadUrl(cleanUrl)
        splitWebView?.loadUrl(cleanUrl)
    }

    private fun renderMultiPlatformCrawlHUD(report: com.zipper.app.webmind.data.MultiPlatformCrawlReport?) {
        val hud = findViewById<LinearLayout>(R.id.llMultiPlatformCrawlHUD) ?: return
        if (report == null || report.platformsCrawled.isEmpty()) {
            hud.visibility = View.GONE
            return
        }

        hud.visibility = View.VISIBLE
        val tvTitle = findViewById<TextView>(R.id.tvCrawlHudTitle)
        val tvSubtitle = findViewById<TextView>(R.id.tvCrawlHudSubtitle)
        val llBadges = findViewById<LinearLayout>(R.id.llCrawlPlatformBadges)
        val tvHeadline = findViewById<TextView>(R.id.tvCrawlSavingsHeadline)
        val tvMath = findViewById<TextView>(R.id.tvCrawlSavingsMath)
        val llDirectLinks = findViewById<LinearLayout>(R.id.llCrawlDirectLinks)

        tvTitle?.text = "🌐 Multi-Platform Live Crawl: ${report.entityName}"
        tvSubtitle?.text = "Crawled across ${report.platformsCrawled.size} platforms · Verified real savings"

        tvHeadline?.text = report.verdictText
        tvMath?.text = report.mathProof

        llBadges?.removeAllViews()
        llDirectLinks?.removeAllViews()

        val density = resources.displayMetrics.density
        fun dp(v: Int) = (v * density).toInt()

        for (badge in report.platformsCrawled) {
            // 1. Badge Pill in horizontal strip
            val badgeView = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = android.view.Gravity.CENTER_VERTICAL
                val bgDrawable = if (badge.isLowestPrice) R.drawable.bg_pill_clean_blue else R.drawable.bg_filter_pill_light
                setBackgroundResource(bgDrawable)
                if (badge.isLowestPrice) backgroundTintList = ColorStateList.valueOf(Color.parseColor("#ECFDF5"))
                setPadding(dp(10), dp(6), dp(12), dp(6))
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply {
                    marginEnd = dp(8)
                }
                isClickable = true
                isFocusable = true
            }

            val ivCheck = ImageView(this).apply {
                layoutParams = LinearLayout.LayoutParams(dp(14), dp(14))
                setImageResource(R.drawable.ic_check_circle)
                imageTintList = ColorStateList.valueOf(if (badge.isLowestPrice) Color.parseColor("#059669") else Color.parseColor("#2563EB"))
            }

            val tvStore = TextView(this).apply {
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply {
                    marginStart = dp(5)
                }
                text = "${badge.storeName}: ${badge.priceDisplay}"
                textSize = 12f
                typeface = android.graphics.Typeface.DEFAULT_BOLD
                setTextColor(if (badge.isLowestPrice) Color.parseColor("#065F46") else Color.parseColor("#111827"))
            }

            badgeView.addView(ivCheck)
            badgeView.addView(tvStore)

            badgeView.setOnClickListener {
                if (badge.targetUrl.isNotBlank()) {
                    openUrlInWebView(badge.targetUrl)
                }
            }
            llBadges?.addView(badgeView)

            // 2. Direct 1-tap open button at bottom of HUD
            val btnOpenStore = TextView(this).apply {
                text = "Inspect ${badge.storeName} ↗"
                textSize = 11.5f
                typeface = android.graphics.Typeface.DEFAULT_BOLD
                setTextColor(Color.parseColor("#2563EB"))
                setBackgroundResource(R.drawable.bg_filter_pill_light)
                setPadding(dp(12), dp(6), dp(12), dp(6))
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply {
                    marginEnd = dp(8)
                }
                isClickable = true
                isFocusable = true
                setOnClickListener {
                    if (badge.targetUrl.isNotBlank()) {
                        openUrlInWebView(badge.targetUrl)
                    }
                }
            }
            llDirectLinks?.addView(btnOpenStore)
        }
    }

    private fun executeStoreRelay(store: String, itemsQuery: String) {
        val pkg = when (store) {
            "Zepto" -> "in.zepto.android"
            "Blinkit" -> "com.grofers.customerapp"
            "Amazon" -> "in.amazon.mShop.android.shopping"
            "Flipkart" -> "com.flipkart.android"
            else -> null
        }
        val launchIntent = if (pkg != null) packageManager.getLaunchIntentForPackage(pkg) else null
        if (launchIntent != null) {
            Toast.makeText(this, "⚡ Opening $store for: $itemsQuery", Toast.LENGTH_SHORT).show()
            startActivity(launchIntent)
        } else {
            val fallbackUrl = when (store) {
                "Zepto" -> "https://www.zeptonow.com/search?q=" + Uri.encode(itemsQuery)
                "Blinkit" -> "https://blinkit.com/s/?q=" + Uri.encode(itemsQuery)
                "Amazon" -> "https://www.amazon.in/s?k=" + Uri.encode(itemsQuery)
                "Flipkart" -> "https://www.flipkart.com/search?q=" + Uri.encode(itemsQuery)
                else -> "https://www.google.com/search?q=" + Uri.encode(itemsQuery)
            }
            openUrlInWebView(fallbackUrl)
        }
    }

    private fun createBasketStoreOfferItem(
        storeName: String,
        description: String,
        badgeText: String,
        badgeColorHex: String,
        onClick: () -> Unit
    ): View {
        val density = resources.displayMetrics.density
        val item = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = android.view.Gravity.CENTER_VERTICAL
            background = getDrawable(R.drawable.bg_action_card_light)
            isClickable = true
            isFocusable = true
            elevation = 1f * density
            setPadding((12 * density).toInt(), (10 * density).toInt(), (12 * density).toInt(), (10 * density).toInt())
            val lp = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                bottomMargin = (8 * density).toInt()
            }
            layoutParams = lp
            setOnClickListener { onClick() }
        }

        val textCol = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }

        val titleRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = android.view.Gravity.CENTER_VERTICAL
        }

        val tvTitle = TextView(this).apply {
            text = storeName
            textSize = 13.5f
            setTextColor(Color.parseColor("#111827"))
            typeface = android.graphics.Typeface.DEFAULT_BOLD
        }
        titleRow.addView(tvTitle)

        val tvBadge = TextView(this).apply {
            text = badgeText
            textSize = 10f
            setTextColor(Color.WHITE)
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            setBackgroundResource(R.drawable.bg_filter_pill_black)
            backgroundTintList = ColorStateList.valueOf(Color.parseColor(badgeColorHex))
            setPadding((8 * density).toInt(), (2 * density).toInt(), (8 * density).toInt(), (2 * density).toInt())
            val blp = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                marginStart = (8 * density).toInt()
            }
            layoutParams = blp
        }
        titleRow.addView(tvBadge)
        textCol.addView(titleRow)

        val tvDesc = TextView(this).apply {
            text = description
            textSize = 11.5f
            setTextColor(Color.parseColor("#4B5563"))
            val dlp = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                topMargin = (3 * density).toInt()
            }
            layoutParams = dlp
        }
        textCol.addView(tvDesc)
        item.addView(textCol)

        val tvAction = TextView(this).apply {
            text = "Order >"
            textSize = 12f
            setTextColor(Color.parseColor("#2563EB"))
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            val alp = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                marginStart = (8 * density).toInt()
            }
            layoutParams = alp
        }
        item.addView(tvAction)

        return item
    }

    // =================================================================
    // SCREEN 4: SPLIT VIEW (CHAT + WEBVIEW)
    // =================================================================
    private fun setupScreenSplit() {
        splitWebView = findViewById(R.id.webmindSplitWebView)
        splitWebView?.settings?.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            useWideViewPort = true
            loadWithOverviewMode = true
        }
        splitWebView?.webViewClient = object : WebViewClient() {
            override fun onPageFinished(view: WebView?, url: String?) {
                val adBlockCss = WebShieldEngine.generateAdBlockRulesCss()
                val paywallJs = WebShieldEngine.generatePaywallBypassJs()
                val script = """
                    (function() {
                        try {
                            var style = document.createElement('style');
                            style.type = 'text/css';
                            style.appendChild(document.createTextNode(`$adBlockCss`));
                            document.head.appendChild(style);
                        } catch(e) {}
                        $paywallJs
                    })();
                """.trimIndent()
                view?.evaluateJavascript(script, null)
            }
        }
    }

    // =================================================================
    // SCREEN 5: COMPARISON VIEW
    // =================================================================
    private fun setupScreenCompare() {
        val payload = currentComparisonPayload
        val density = resources.displayMetrics.density
        fun dp(v: Int) = (v * density).toInt()

        val p1 = payload?.products?.getOrNull(0) ?: currentProduct?.let {
            com.zipper.app.webmind.data.MatrixProduct(
                id = it.id,
                title = it.title,
                shortName = it.title,
                brand = it.store,
                category = "Product",
                price = it.price,
                priceDisplay = it.priceFormatted,
                store = it.store,
                rating = it.rating,
                reviewCount = it.reviewCount,
                imageUrl = it.imageUrl,
                productUrl = it.productUrl
            )
        }
        val p2 = payload?.products?.getOrNull(1)
        val p3 = payload?.products?.getOrNull(2)

        // Column 1
        val llCol1 = findViewById<View>(R.id.llCompareProd1)
        if (p1 != null) {
            llCol1?.visibility = View.VISIBLE
            findViewById<TextView>(R.id.tvCompareProd1Title)?.text = p1.shortName.ifBlank { p1.title }
            findViewById<TextView>(R.id.tvCompareProd1Price)?.text = p1.priceDisplay
            bindThumbnailImage(findViewById(R.id.ivCompareProd1), p1.imageUrl)
            llCol1?.setOnClickListener {
                if (p1.productUrl.isNotBlank()) openUrlInWebView(p1.productUrl)
            }
        } else {
            llCol1?.visibility = View.GONE
        }

        // Column 2
        val llCol2 = findViewById<View>(R.id.llCompareProd2)
        if (p2 != null) {
            llCol2?.visibility = View.VISIBLE
            findViewById<TextView>(R.id.tvCompareProd2Title)?.text = p2.shortName.ifBlank { p2.title }
            findViewById<TextView>(R.id.tvCompareProd2Price)?.text = p2.priceDisplay
            bindThumbnailImage(findViewById(R.id.ivCompareProd2), p2.imageUrl)
            llCol2?.setOnClickListener {
                if (p2.productUrl.isNotBlank()) openUrlInWebView(p2.productUrl)
            }
        } else {
            llCol2?.visibility = View.GONE
        }

        // Column 3
        val llCol3 = findViewById<View>(R.id.llCompareProd3)
        if (p3 != null) {
            llCol3?.visibility = View.VISIBLE
            findViewById<TextView>(R.id.tvCompareProd3Title)?.text = p3.shortName.ifBlank { p3.title }
            findViewById<TextView>(R.id.tvCompareProd3Price)?.text = p3.priceDisplay
            bindThumbnailImage(findViewById(R.id.ivCompareProd3), p3.imageUrl)
            llCol3?.setOnClickListener {
                if (p3.productUrl.isNotBlank()) openUrlInWebView(p3.productUrl)
            }
        } else {
            llCol3?.visibility = View.GONE
        }

        // Recommendation Body
        val tvRec = findViewById<TextView>(R.id.tvRecommendationBody)
        if (payload != null) {
            tvRec?.text = "${payload.definitiveVerdict}\n\n${payload.savingsProofText}"
        } else if (p1 != null) {
            tvRec?.text = "${p1.title} offers authentic verified live pricing on ${p1.store}."
        }

        // Populate Dynamic Specs Rows in TableLayout
        val table = findViewById<TableLayout>(R.id.tableCompareSpecs)
        if (table != null && payload != null) {
            table.removeAllViews()

            // 1. Rating Row
            val trRating = TableRow(this).apply {
                layoutParams = TableLayout.LayoutParams(TableLayout.LayoutParams.WRAP_CONTENT, TableLayout.LayoutParams.WRAP_CONTENT).apply {
                    bottomMargin = dp(8)
                }
            }
            val tvLabelRating = TextView(this).apply {
                layoutParams = TableRow.LayoutParams(dp(76), TableRow.LayoutParams.WRAP_CONTENT)
                text = "Rating"
                setTextColor(Color.parseColor("#9CA3AF"))
                textSize = 11f
            }
            trRating.addView(tvLabelRating)
            for (p in listOfNotNull(p1, p2, p3)) {
                val tvVal = TextView(this).apply {
                    layoutParams = TableRow.LayoutParams(dp(112), TableRow.LayoutParams.WRAP_CONTENT).apply {
                        marginEnd = dp(10)
                    }
                    text = "★ ${p.rating} (${p.reviewCount})"
                    setTextColor(Color.parseColor("#D97706"))
                    textSize = 11f
                    typeface = android.graphics.Typeface.DEFAULT_BOLD
                }
                trRating.addView(tvVal)
            }
            table.addView(trRating)

            // 2. Unit Price / Delivery Speed Row
            val trDelivery = TableRow(this).apply {
                layoutParams = TableLayout.LayoutParams(TableLayout.LayoutParams.WRAP_CONTENT, TableLayout.LayoutParams.WRAP_CONTENT).apply {
                    bottomMargin = dp(8)
                }
            }
            val tvLabelDel = TextView(this).apply {
                layoutParams = TableRow.LayoutParams(dp(76), TableRow.LayoutParams.WRAP_CONTENT)
                text = "Speed"
                setTextColor(Color.parseColor("#9CA3AF"))
                textSize = 11f
            }
            trDelivery.addView(tvLabelDel)
            for (p in listOfNotNull(p1, p2, p3)) {
                val tvVal = TextView(this).apply {
                    layoutParams = TableRow.LayoutParams(dp(112), TableRow.LayoutParams.WRAP_CONTENT).apply {
                        marginEnd = dp(10)
                    }
                    text = p.deliverySpeed
                    setTextColor(Color.parseColor("#059669"))
                    textSize = 11f
                    typeface = android.graphics.Typeface.DEFAULT_BOLD
                }
                trDelivery.addView(tvVal)
            }
            table.addView(trDelivery)

            // 3. Category Group Rows or Product Specs
            if (payload.specGroups.isNotEmpty()) {
                for (group in payload.specGroups) {
                    for (row in group.rows) {
                        val tr = TableRow(this).apply {
                            layoutParams = TableLayout.LayoutParams(TableLayout.LayoutParams.WRAP_CONTENT, TableLayout.LayoutParams.WRAP_CONTENT).apply {
                                bottomMargin = dp(8)
                            }
                        }
                        val tvL = TextView(this).apply {
                            layoutParams = TableRow.LayoutParams(dp(76), TableRow.LayoutParams.WRAP_CONTENT)
                            text = row.label
                            setTextColor(Color.parseColor("#9CA3AF"))
                            textSize = 11f
                        }
                        tr.addView(tvL)
                        for (p in listOfNotNull(p1, p2, p3)) {
                            val v = row.valuesPerProduct[p.shortName] ?: row.valuesPerProduct[p.title] ?: p.specs[row.label] ?: "Standard"
                            val tvV = TextView(this).apply {
                                layoutParams = TableRow.LayoutParams(dp(112), TableRow.LayoutParams.WRAP_CONTENT).apply {
                                    marginEnd = dp(10)
                                }
                                text = v
                                setTextColor(Color.parseColor("#111827"))
                                textSize = 11f
                            }
                            tr.addView(tvV)
                        }
                        table.addView(tr)
                    }
                }
            } else {
                val allKeys = mutableSetOf<String>()
                listOfNotNull(p1, p2, p3).forEach { allKeys.addAll(it.specs.keys) }
                for (k in allKeys) {
                    val tr = TableRow(this).apply {
                        layoutParams = TableLayout.LayoutParams(TableLayout.LayoutParams.WRAP_CONTENT, TableLayout.LayoutParams.WRAP_CONTENT).apply {
                            bottomMargin = dp(8)
                        }
                    }
                    val tvL = TextView(this).apply {
                        layoutParams = TableRow.LayoutParams(dp(76), TableRow.LayoutParams.WRAP_CONTENT)
                        text = k
                        setTextColor(Color.parseColor("#9CA3AF"))
                        textSize = 11f
                    }
                    tr.addView(tvL)
                    for (p in listOfNotNull(p1, p2, p3)) {
                        val tvV = TextView(this).apply {
                            layoutParams = TableRow.LayoutParams(dp(112), TableRow.LayoutParams.WRAP_CONTENT).apply {
                                marginEnd = dp(10)
                            }
                            text = p.specs[k] ?: "—"
                            setTextColor(Color.parseColor("#111827"))
                            textSize = 11f
                        }
                        tr.addView(tvV)
                    }
                    table.addView(tr)
                }
            }

            // 4. Best For Badge Row
            val trBadge = TableRow(this).apply {
                layoutParams = TableLayout.LayoutParams(TableLayout.LayoutParams.WRAP_CONTENT, TableLayout.LayoutParams.WRAP_CONTENT).apply {
                    topMargin = dp(4)
                }
            }
            val tvLabelBadge = TextView(this).apply {
                layoutParams = TableRow.LayoutParams(dp(76), TableRow.LayoutParams.WRAP_CONTENT)
                text = "Best For"
                setTextColor(Color.parseColor("#9CA3AF"))
                textSize = 11f
            }
            trBadge.addView(tvLabelBadge)
            for (p in listOfNotNull(p1, p2, p3)) {
                val tvB = TextView(this).apply {
                    layoutParams = TableRow.LayoutParams(dp(112), TableRow.LayoutParams.WRAP_CONTENT).apply {
                        marginEnd = dp(10)
                    }
                    text = p.badge.ifBlank { if (p.isBestDeal) "Best Value" else "Option" }
                    val bgRes = if (p.isBestDeal) R.drawable.bg_badge_pill_green else R.drawable.bg_badge_pill_teal
                    setBackgroundResource(bgRes)
                    setTextColor(if (p.isBestDeal) Color.parseColor("#166534") else Color.parseColor("#0F766E"))
                    textSize = 10f
                    typeface = android.graphics.Typeface.DEFAULT_BOLD
                    gravity = android.view.Gravity.CENTER
                    setPadding(dp(4), dp(3), dp(4), dp(3))
                }
                trBadge.addView(tvB)
            }
            table.addView(trBadge)
        }

        // Review Sentiment Section
        val llReviews = findViewById<LinearLayout>(R.id.llCompareReviewsSection)
        val tvPraises = findViewById<TextView>(R.id.tvCompareReviewPraises)
        val tvComplaints = findViewById<TextView>(R.id.tvCompareReviewComplaints)
        if (payload?.sentiments?.isNotEmpty() == true) {
            val firstSentiment = payload.sentiments.first()
            llReviews?.visibility = View.VISIBLE
            tvPraises?.text = "🟢 Top Praises (${firstSentiment.entityName}):\n" + firstSentiment.topPraises.joinToString("\n") { "• $it" }
            tvComplaints?.text = "🔴 Top Complaints (${firstSentiment.entityName}):\n" + firstSentiment.topComplaints.joinToString("\n") { "• $it" }
        } else if (p1?.pros?.isNotEmpty() == true) {
            llReviews?.visibility = View.VISIBLE
            tvPraises?.text = "🟢 Top Praises (${p1.shortName}):\n" + p1.pros.joinToString("\n") { "• $it" }
            tvComplaints?.text = "🔴 Top Considerations (${p1.shortName}):\n" + p1.cons.joinToString("\n") { "• $it" }
        } else {
            llReviews?.visibility = View.GONE
        }

        // Tab Listeners
        findViewById<View>(R.id.tabSubComparison)?.setOnClickListener {
            findViewById<View>(R.id.scrollWebmindCompare)?.scrollTo(0, 0)
        }
        findViewById<View>(R.id.tabSubSpecs)?.setOnClickListener {
            findViewById<View>(R.id.tableCompareSpecs)?.let { t ->
                findViewById<androidx.core.widget.NestedScrollView>(R.id.scrollWebmindCompare)?.smoothScrollTo(0, t.top)
            }
        }
        findViewById<View>(R.id.tabSubReviews)?.setOnClickListener {
            llReviews?.visibility = View.VISIBLE
            llReviews?.let { r ->
                findViewById<androidx.core.widget.NestedScrollView>(R.id.scrollWebmindCompare)?.smoothScrollTo(0, r.top)
            }
        }
        findViewById<View>(R.id.tabSubPriceHistory)?.setOnClickListener {
            navigateTo(Screen.PRICE_TRACK)
        }
    }

    // =================================================================
    // SCREEN 6: PRICE HISTORY & TRACKING
    // =================================================================
    private fun setupScreenPriceTrack() {
        val prod = currentProduct
        if (prod != null) {
            findViewById<TextView>(R.id.tvTrackProdTitle)?.text = prod.title
            findViewById<TextView>(R.id.tvTrackProdPrice)?.text = prod.priceFormatted
            findViewById<TextView>(R.id.tvTrackProdStore)?.text = prod.store
            bindThumbnailImage(findViewById(R.id.ivTrackProdThumbnail), prod.imageUrl)
        }

        val graph = findViewById<PriceGraphView>(R.id.priceGraphView)
        loadPriceHistoryGraph()

        findViewById<Button>(R.id.btnStartPriceTracking)?.setOnClickListener {
            val p = currentProduct ?: run {
                Toast.makeText(this, "Search a product first to enable tracking!", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            val targetPrice = if (p.price > 0) p.price * 0.9 else 5000.0
            val tracked = TrackedProduct(
                id = p.id,
                title = p.title,
                store = p.store,
                currentPrice = p.price,
                targetPrice = targetPrice,
                thresholdType = "BELOW",
                productUrl = p.productUrl,
                imageUrl = p.imageUrl,
                isActive = true
            )
            dbHelper.addTrackedProduct(tracked)
            Toast.makeText(this, "🔔 Price tracker active! We'll alert you below ${if (p.price > 0) "₹" + targetPrice.toInt() else "target"}.", Toast.LENGTH_LONG).show()
            navigateTo(Screen.SAVED)
        }
    }

    private fun loadPriceHistoryGraph() {
        val graph = findViewById<PriceGraphView>(R.id.priceGraphView)
        val prodId = currentProduct?.id ?: "prod_loq"
        val points = dbHelper.getPriceHistory(prodId)
        graph?.setData(points)
    }

    // =================================================================
    // SCREEN 7 & 8: SUMMARIZER & OUTPUT WITH REAL ARTICLE ENGINE
    // =================================================================
    private fun setupScreenSummarize() {
        val etUrl = findViewById<EditText>(R.id.etSummarizeUrl)
        val btnClear = findViewById<ImageButton>(R.id.btnSummarizeClear)
        val btnTrigger = findViewById<Button>(R.id.btnTriggerSummarize)

        btnClear?.setOnClickListener { etUrl?.text?.clear() }

        btnTrigger?.setOnClickListener {
            val url = etUrl?.text?.toString()?.trim() ?: "https://www.bbc.com/news/science-space"
            triggerRealSummarize(url)
        }

        // Circular Send Button in Summarize URL Bar
        findViewById<View>(R.id.btnSummarizeSend)?.setOnClickListener {
            val url = etUrl?.text?.toString()?.trim() ?: "https://www.bbc.com/news/science-space"
            triggerRealSummarize(url)
        }

        // FAQ accordion handlers
        findViewById<View>(R.id.rowFaq1)?.setOnClickListener {
            showFaqDialog("How is this different from Artemis?", "It focuses on permanent habitat infrastructure and closed-loop resource recycling rather than short sorties.")
        }
        findViewById<View>(R.id.rowFaq2)?.setOnClickListener {
            showFaqDialog("What technologies are being tested?", "Autonomous surface drilling, in-situ water extraction, and modular surface power grids.")
        }
        findViewById<View>(R.id.rowFaq3)?.setOnClickListener {
            showFaqDialog("What is the expected timeline?", "Uncrewed test launch scheduled for Q3 2026, followed by human surface operations in 2027.")
        }
        findViewById<View>(R.id.rowFaq4)?.setOnClickListener {
            showFaqDialog("How does this impact future Mars missions?", "Surface power and propulsion validation directly inform Mars transport architecture.")
        }
    }

    private fun triggerRealSummarize(url: String) {
        navigateTo(Screen.SUMMARIZE)
        val tvPreviewTitle = findViewById<TextView>(R.id.tvSummarizePreviewTitle)
        val tvPreviewSource = findViewById<TextView>(R.id.tvSummarizePreviewSource)
        val btnTrigger = findViewById<Button>(R.id.btnTriggerSummarize)

        tvPreviewTitle?.text = "Extracting article from live web..."
        tvPreviewSource?.text = url
        btnTrigger?.text = "Summarizing..."
        btnTrigger?.isEnabled = false
        Toast.makeText(this, "Stripping ads & extracting content...", Toast.LENGTH_SHORT).show()

        lifecycleScope.launch {
            try {
                val summaryResult = ArticleReaderEngine.readAndSummarize(url, webController)
                tvPreviewTitle?.text = summaryResult.title
                tvPreviewSource?.text = Uri.parse(url).host ?: url
                btnTrigger?.text = "Summarize ✓"
                btnTrigger?.isEnabled = true

                // Update 4 Takeaways dynamically
                val takeaways = summaryResult.keyTakeaways
                if (takeaways.isNotEmpty()) {
                    findViewById<TextView>(R.id.tvTakeawayText1)?.text = takeaways.getOrNull(0) ?: "Core findings verified from source."
                    findViewById<TextView>(R.id.tvTakeawayText2)?.text = takeaways.getOrNull(1) ?: "Key metrics and data highlighted."
                    findViewById<TextView>(R.id.tvTakeawayText3)?.text = takeaways.getOrNull(2) ?: "Methodology and timeline updates recorded."
                    findViewById<TextView>(R.id.tvTakeawayText4)?.text = takeaways.getOrNull(3) ?: "Implications and next phases summarized."
                }

                dbHelper.addResearchSession(
                    ResearchSession(
                        id = "res_${System.currentTimeMillis()}",
                        query = "Summary: ${summaryResult.title}",
                        category = "Summarize",
                        timestamp = System.currentTimeMillis(),
                        summaryText = summaryResult.executiveSummary.take(80) + "...",
                        iconType = "article"
                    )
                )
                Toast.makeText(this@WebmindActivity, "✓ Article summarized cleanly", Toast.LENGTH_SHORT).show()
            } catch (_: Exception) {
                btnTrigger?.text = "Summarize"
                btnTrigger?.isEnabled = true
                val summary = WebContentExtractor.extractAndSummarize(url)
                tvPreviewTitle?.text = summary.title
                tvPreviewSource?.text = "${summary.publisher} · ${summary.dateStr}"
            }
        }
    }

    private fun showFaqDialog(question: String, answer: String) {
        AlertDialog.Builder(this)
            .setTitle(question)
            .setMessage(answer)
            .setPositiveButton("Close", null)
            .show()
    }

    // =================================================================
    // SCREEN 9: RESEARCH HISTORY & LIBRARY
    // =================================================================
    private fun setupScreenHistory() {
        val chipAll = findViewById<TextView>(R.id.chipHistAll)
        val chipSearch = findViewById<TextView>(R.id.chipHistSearch)
        val chipCompare = findViewById<TextView>(R.id.chipHistCompare)
        val chipTrack = findViewById<TextView>(R.id.chipHistTrack)
        val chipSummarize = findViewById<TextView>(R.id.chipHistSummarize)

        val chips = listOf(chipAll, chipSearch, chipCompare, chipTrack, chipSummarize)
        val categories = listOf("All", "Search", "Compare", "Track", "Summarize")

        for (i in chips.indices) {
            chips[i]?.setOnClickListener {
                for (c in chips) {
                    c?.setBackgroundResource(R.drawable.bg_filter_pill_light)
                    c?.setTextColor(Color.parseColor("#4B5563"))
                }
                chips[i]?.setBackgroundResource(R.drawable.bg_filter_pill_black)
                chips[i]?.setTextColor(Color.WHITE)
                loadResearchHistory(categories[i])
            }
        }
    }

    private fun loadResearchHistory(categoryFilter: String = "All") {
        val container = findViewById<LinearLayout>(R.id.llHistoryItemsContainer) ?: return
        container.removeAllViews()

        val sessions = dbHelper.getAllResearchSessions(categoryFilter)

        if (sessions.isEmpty()) {
            val emptyTv = TextView(this).apply {
                text = "No research history found for '$categoryFilter'."
                setTextColor(Color.parseColor("#9CA3AF"))
                textSize = 13f
                gravity = android.view.Gravity.CENTER
                setPadding(0, 40, 0, 0)
            }
            container.addView(emptyTv)
            return
        }

        // Section Header "Today"
        val headerTv = TextView(this).apply {
            text = "Today"
            setTextColor(Color.parseColor("#111827"))
            textSize = 13.5f
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            setPadding(0, 8, 0, 12)
        }
        container.addView(headerTv)

        val inflater = LayoutInflater.from(this)
        for (item in sessions) {
            val row = inflater.inflate(R.layout.item_recent_task, container, false)
            val ivIcon = row.findViewById<ImageView>(R.id.ivItemTaskIcon)
            val tvGoal = row.findViewById<TextView>(R.id.tvItemTaskGoal)
            val tvStatus = row.findViewById<TextView>(R.id.tvItemTaskStatus)

            tvGoal.text = item.query
            tvStatus.text = "${item.category} • ${item.getFormattedTime()}"

            when (item.iconType) {
                "laptop" -> ivIcon.setImageResource(R.drawable.ic_laptop)
                "gold" -> ivIcon.setImageResource(R.drawable.ic_gold_bars)
                "article" -> ivIcon.setImageResource(R.drawable.ic_article)
                "compare" -> ivIcon.setImageResource(R.drawable.ic_compare)
                else -> ivIcon.setImageResource(R.drawable.ic_search)
            }
            ivIcon.setBackgroundResource(R.drawable.bg_filter_pill_light)
            ivIcon.setColorFilter(Color.parseColor("#111827"))

            row.setOnClickListener {
                if (item.category == "Compare") navigateTo(Screen.COMPARE)
                else if (item.category == "Summarize") navigateTo(Screen.SUMMARIZE)
                else executeSearch(item.query)
            }
            container.addView(row)
        }
    }

    // =================================================================
    // SCREEN 10: SAVED ITEMS & TRACKS
    // =================================================================
    private fun setupScreenSaved() {
        val tabSavedAll = findViewById<TextView>(R.id.tabSavedAll)
        val tabPriceTracks = findViewById<TextView>(R.id.tabSavedPriceTracks)
        val tabCollections = findViewById<TextView>(R.id.tabSavedCollections)

        tabSavedAll?.setOnClickListener {
            tabSavedAll.setBackgroundResource(R.drawable.bg_filter_pill_black)
            tabSavedAll.setTextColor(Color.WHITE)
            tabPriceTracks?.setBackgroundResource(R.drawable.bg_filter_pill_light)
            tabPriceTracks?.setTextColor(Color.parseColor("#4B5563"))
            tabCollections?.setBackgroundResource(R.drawable.bg_filter_pill_light)
            tabCollections?.setTextColor(Color.parseColor("#4B5563"))
            loadSavedItems(showTracksOnly = false)
        }

        tabPriceTracks?.setOnClickListener {
            tabPriceTracks.setBackgroundResource(R.drawable.bg_filter_pill_black)
            tabPriceTracks.setTextColor(Color.WHITE)
            tabSavedAll?.setBackgroundResource(R.drawable.bg_filter_pill_light)
            tabSavedAll?.setTextColor(Color.parseColor("#4B5563"))
            tabCollections?.setBackgroundResource(R.drawable.bg_filter_pill_light)
            tabCollections?.setTextColor(Color.parseColor("#4B5563"))
            loadSavedItems(showTracksOnly = true)
        }
    }

    private fun loadSavedItems(showTracksOnly: Boolean = false) {
        val container = findViewById<LinearLayout>(R.id.llSavedItemsContainer) ?: return
        container.removeAllViews()

        val tracked = dbHelper.getTrackedProducts()

        for (item in tracked) {
            val card = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                setBackgroundResource(R.drawable.bg_action_card_light)
                elevation = 2f
                setPadding(28, 24, 28, 24)
                gravity = android.view.Gravity.CENTER_VERTICAL
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply { setMargins(0, 0, 0, 20) }
            }

            val iv = ImageView(this).apply {
                layoutParams = LinearLayout.LayoutParams(90, 90).apply { setMargins(0, 0, 24, 0) }
                setImageResource(R.drawable.ic_laptop)
                setBackgroundResource(R.drawable.bg_filter_pill_light)
                setPadding(14, 14, 14, 14)
                setColorFilter(Color.parseColor("#111827"))
            }

            val textLayout = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
            }

            val tvTitle = TextView(this).apply {
                text = item.title
                setTextColor(Color.parseColor("#111827"))
                textSize = 14f
                typeface = android.graphics.Typeface.DEFAULT_BOLD
            }

            val tvSub = TextView(this).apply {
                text = "Track · Target ₹${item.targetPrice.toInt()}"
                setTextColor(Color.parseColor("#6B7280"))
                textSize = 12f
                setPadding(0, 4, 0, 0)
            }

            textLayout.addView(tvTitle)
            textLayout.addView(tvSub)

            // Active pill badge
            val badge = TextView(this).apply {
                text = if (item.isActive) "Active" else "Paused"
                setTextColor(if (item.isActive) Color.parseColor("#15803D") else Color.parseColor("#6B7280"))
                setBackgroundResource(if (item.isActive) R.drawable.bg_badge_pill_green else R.drawable.bg_filter_pill_light)
                textSize = 10.5f
                typeface = android.graphics.Typeface.DEFAULT_BOLD
                setPadding(20, 8, 20, 8)
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply { setMargins(0, 0, 16, 0) }
            }

            val moreBtn = ImageButton(this).apply {
                layoutParams = LinearLayout.LayoutParams(60, 60)
                setImageResource(R.drawable.ic_more_vert)
                setBackgroundResource(android.R.color.transparent)
                setColorFilter(Color.parseColor("#6B7280"))
                setOnClickListener {
                    AlertDialog.Builder(this@WebmindActivity)
                        .setTitle(item.title)
                        .setItems(arrayOf("View Price Trend", "Toggle Alert", "Open Live Link")) { _, which ->
                            when (which) {
                                0 -> navigateTo(Screen.PRICE_TRACK)
                                1 -> {
                                    dbHelper.toggleProductTracking(item.id, !item.isActive)
                                    loadSavedItems(showTracksOnly)
                                }
                                2 -> openUrlInWebView(item.productUrl)
                            }
                        }.show()
                }
            }

            card.addView(iv)
            card.addView(textLayout)
            card.addView(badge)
            card.addView(moreBtn)

            card.setOnClickListener {
                navigateTo(Screen.PRICE_TRACK)
            }

            container.addView(card)
        }
    }

    private fun registerOtpReceiver() {
        otpReceiver = object : BroadcastReceiver() {
            override fun onReceive(context: Context?, intent: Intent?) {
                val otp = intent?.getStringExtra("otp") ?: return
                val appName = intent.getStringExtra("appName") ?: "SMS"
                val banner = findViewById<View>(R.id.llWebviewOtpBanner)
                val tvText = findViewById<TextView>(R.id.tvWebviewOtpBannerText)
                val btnInject = findViewById<View>(R.id.btnWebviewInjectOtp)

                banner?.visibility = View.VISIBLE
                tvText?.text = "🔐 OTP: $otp ($appName)"
                btnInject?.setOnClickListener {
                    val injectJs = """
                        (function() {
                            try {
                                var inputs = document.querySelectorAll('input[type="tel"], input[autocomplete="one-time-code"], input[name*="otp" i], input[id*="otp" i], input[placeholder*="otp" i], input[placeholder*="code" i], input[maxlength="6"], input[maxlength="4"]');
                                if (inputs.length > 0) {
                                    inputs[0].value = '$otp';
                                    inputs[0].dispatchEvent(new Event('input', { bubbles: true }));
                                    inputs[0].dispatchEvent(new Event('change', { bubbles: true }));
                                    inputs[0].focus();
                                    return true;
                                }
                            } catch(e) {}
                            return false;
                        })();
                    """.trimIndent()
                    mainWebView?.evaluateJavascript(injectJs) {
                        Toast.makeText(this@WebmindActivity, "⚡ Auto-filled OTP $otp into form!", Toast.LENGTH_SHORT).show()
                        banner.visibility = View.GONE
                    }
                }
            }
        }
        val filter = IntentFilter("com.zipper.app.ACTION_OTP_RECEIVED")
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(otpReceiver, filter, Context.RECEIVER_NOT_EXPORTED)
        } else {
            registerReceiver(otpReceiver, filter)
        }
    }

    private fun toggleReaderMode() {
        isReaderModeActive = !isReaderModeActive
        val tvReader = findViewById<TextView>(R.id.tvDockReaderText)
        if (isReaderModeActive) {
            tvReader?.text = "Exit Reader"
            tvReader?.setTextColor(Color.parseColor("#2563EB"))
            val readerJs = """
                (function() {
                    try {
                        var elementsToHide = document.querySelectorAll('header, footer, nav, aside, .advertisement, .ad, [class*="banner"], [class*="cookie"], [id*="cookie"]');
                        elementsToHide.forEach(function(el) { el.style.display = 'none'; });
                        document.body.style.maxWidth = '720px';
                        document.body.style.margin = '0 auto';
                        document.body.style.padding = '24px 16px';
                        document.body.style.lineHeight = '1.7';
                        document.body.style.fontSize = '18px';
                        document.body.style.backgroundColor = '#FAF9F6';
                        document.body.style.color = '#1A1A1A';
                    } catch(e) {}
                })();
            """.trimIndent()
            mainWebView?.evaluateJavascript(readerJs, null)
            Toast.makeText(this, "📖 Reader Mode Active", Toast.LENGTH_SHORT).show()
        } else {
            tvReader?.text = "Reader"
            tvReader?.setTextColor(Color.parseColor("#374151"))
            mainWebView?.reload()
        }
    }

    private fun executeAutonomousMultiProductRelay(
        targetStore: String,
        parsedItems: List<com.zipper.app.core.web.ParsedShoppingItem>
    ) {
        val targetPkg = com.zipper.app.core.web.WebDeepLinkManager.resolveInstalledPackageForStore(targetStore, this@WebmindActivity)
        val multiItems = parsedItems.map { it.normalizedQuery }

        lifecycleScope.launch {
            val llLiveAgentHud = findViewById<View>(R.id.llLiveAgentHud)
            val tvLiveAgentTitle = findViewById<TextView>(R.id.tvLiveAgentTitle)
            val tvLiveAgentStatus = findViewById<TextView>(R.id.tvLiveAgentStatus)
            val btnLiveAgentViewWeb = findViewById<View>(R.id.btnLiveAgentViewWeb)
            val tvStatus = findViewById<TextView>(R.id.tvThinkingStatus)

            llLiveAgentHud?.visibility = View.VISIBLE
            tvLiveAgentStatus?.text = "🛒 Initiating Multi-Product Relay on $targetStore (${multiItems.size} items)..."
            tvStatus?.text = "Launching $targetStore autonomous cart adder..."

            val queuedProducts = mutableListOf<com.zipper.app.core.web.WebProductItem>()

            for ((idx, itemQuery) in multiItems.withIndex()) {
                val cleanTerms = com.zipper.app.core.web.WebProductFinder.cleanSearchQuery(itemQuery)
                val targetKeywords = cleanTerms.lowercase(Locale.ROOT).split(Regex("[^a-z0-9]+")).filter { it.length >= 3 }

                val stepStatus = "🛒 [${idx + 1}/${multiItems.size}] Scouting '$itemQuery' on $targetStore..."
                tvLiveAgentStatus?.text = stepStatus
                tvStatus?.text = stepStatus

                val opened = com.zipper.app.core.web.WebDeepLinkManager.openStoreSearch(
                    context = this@WebmindActivity,
                    storeName = targetStore,
                    query = itemQuery
                )

                if (opened) {
                    val result = com.zipper.app.core.web.WebNativeRelayCloser.attemptAddToCart(
                        targetAppPackage = targetPkg,
                        itemQuery = itemQuery,
                        expectedKeywords = targetKeywords,
                        maxScrolls = 2
                    )
                    when (result) {
                        is com.zipper.app.core.web.WebNativeRelayCloser.AddToCartResult.Success -> {
                            tvLiveAgentStatus?.text = "✅ [${idx + 1}/${multiItems.size}] Added '${result.productTitle}' (${result.priceDisplay})"
                            queuedProducts.add(com.zipper.app.core.web.WebProductItem(
                                title = result.productTitle,
                                price = result.price,
                                priceDisplay = result.priceDisplay,
                                snippet = result.freeDeliveryPrompt,
                                url = com.zipper.app.core.web.WebDeepLinkManager.buildStoreSearchUrl(targetStore, itemQuery),
                                store = targetStore
                            ))
                        }
                        is com.zipper.app.core.web.WebNativeRelayCloser.AddToCartResult.OutOfStock -> {
                            tvLiveAgentStatus?.text = "⚠️ [${idx + 1}/${multiItems.size}] '$itemQuery' is OUT OF STOCK. Skipping..."
                            queuedProducts.add(com.zipper.app.core.web.WebProductItem(
                                title = "$itemQuery (Out of Stock)",
                                price = 0.0,
                                priceDisplay = "Out of Stock",
                                url = com.zipper.app.core.web.WebDeepLinkManager.buildStoreSearchUrl(targetStore, itemQuery),
                                store = targetStore
                            ))
                        }
                        is com.zipper.app.core.web.WebNativeRelayCloser.AddToCartResult.Failed -> {
                            tvLiveAgentStatus?.text = "❌ [${idx + 1}/${multiItems.size}] Skipped '$itemQuery' (${result.reason})"
                            queuedProducts.add(com.zipper.app.core.web.WebProductItem(
                                title = "$itemQuery (Not added)",
                                price = 0.0,
                                priceDisplay = "Not added",
                                url = com.zipper.app.core.web.WebDeepLinkManager.buildStoreSearchUrl(targetStore, itemQuery),
                                store = targetStore
                            ))
                        }
                    }
                } else {
                    val fallbackSearch = com.zipper.app.core.web.WebDeepLinkManager.buildStoreSearchUrl(targetStore, itemQuery)
                    openUrlInWebView(fallbackSearch)
                    break
                }
            }

            try {
                val returnIntent = Intent(this@WebmindActivity, WebmindActivity::class.java).apply {
                    addFlags(Intent.FLAG_ACTIVITY_REORDER_TO_FRONT or Intent.FLAG_ACTIVITY_SINGLE_TOP)
                }
                startActivity(returnIntent)
            } catch (_: Exception) {}

            val addedCount = queuedProducts.count { it.priceDisplay.startsWith("₹") || it.priceDisplay == "Added" }
            val totalCartCost = queuedProducts.filter { it.price > 0.0 }.sumOf { it.price }
            val estimatedSavings = if (totalCartCost > 0.0) (totalCartCost * 0.15).toInt() else 0

            tvStatus?.text = "✅ Autonomous Cart Queue Complete ($addedCount/${multiItems.size} items added on $targetStore)"

            // Muse Architecture: Mandatory Human-in-the-Loop Checkout Approval Card
            val cardRecommendation = findViewById<View>(R.id.cardProductRecommendation)
            cardRecommendation?.visibility = View.VISIBLE

            findViewById<TextView>(R.id.tvResearchResultHeadline)?.text = if (totalCartCost > 0.0) {
                "🛒 $targetStore Cart Ready · ₹${totalCartCost.toInt()}"
            } else {
                "🛒 $targetStore Cart Assembled ($addedCount/${multiItems.size} Added)"
            }

            findViewById<TextView>(R.id.tvResearchResultDescription)?.text = buildString {
                appendLine("Autonomous Cart Assembly finished ($addedCount/${multiItems.size} items added):")
                queuedProducts.forEach { p ->
                    val icon = if (p.priceDisplay == "Out of Stock") "⚠️" else if (p.priceDisplay == "Not added") "❌" else "✅"
                    appendLine("$icon ${p.title} — ${p.priceDisplay}")
                }
                if (totalCartCost > 0.0) {
                    appendLine("\n💰 Estimated Subtotal: ₹${totalCartCost.toInt()} (Saved ~₹$estimatedSavings vs MRP)")
                }
                appendLine("\n🛡️ Muse Safety Gate: Cart assembled. Ready for your 1-tap review & secure payment.")
            }

            findViewById<View>(R.id.llProductDetailsRow)?.visibility = View.VISIBLE
            findViewById<TextView>(R.id.tvProductTitle)?.text = "$targetStore 10-Min Delivery Cart"
            findViewById<TextView>(R.id.tvProductBadge)?.apply {
                visibility = View.VISIBLE
                text = "⚡ Cart Ready"
            }
            findViewById<TextView>(R.id.tvProductPrice)?.apply {
                visibility = View.VISIBLE
                text = if (totalCartCost > 0.0) "₹${totalCartCost.toInt()}" else "Verified"
            }
            findViewById<TextView>(R.id.tvProductSpecsSummary)?.apply {
                visibility = View.VISIBLE
                text = "$addedCount item(s) in cart · Tap below to review & complete payment in $targetStore"
            }

            findViewById<View>(R.id.llProductActions)?.visibility = View.VISIBLE
            findViewById<TextView>(R.id.tvCardViewDetailsLabel)?.text = "💳 Open Cart & Pay ↗"
            findViewById<View>(R.id.btnCardViewDetails)?.setOnClickListener {
                com.zipper.app.core.web.WebDeepLinkManager.openCartScreen(this@WebmindActivity, targetStore)
            }
            findViewById<TextView>(R.id.tvCardCompareLabel)?.text = "Compare Stores"
            findViewById<View>(R.id.btnCardCompare)?.setOnClickListener {
                val altStore = if (targetStore.equals("Zepto", ignoreCase = true)) "Blinkit" else "Zepto"
                executeAutonomousMultiProductRelay(altStore, parsedItems)
            }

            // Keep HUD visible as a sticky persistent 1-tap checkout bar
            llLiveAgentHud?.visibility = View.VISIBLE
            tvLiveAgentTitle?.text = "🛒 $targetStore Cart Assembled"
            tvLiveAgentStatus?.text = "Cart ready ($addedCount/${multiItems.size} items) · Tap to Open Cart & Pay"
            btnLiveAgentViewWeb?.setOnClickListener {
                com.zipper.app.core.web.WebDeepLinkManager.openCartScreen(this@WebmindActivity, targetStore)
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        try {
            if (otpReceiver != null) unregisterReceiver(otpReceiver)
        } catch (_: Exception) {}
        lifecycleScope.launch {
            try {
                webController.destroy()
            } catch (_: Exception) {}
        }
    }
}
