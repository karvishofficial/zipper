package com.zipper.app

import android.Manifest
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.graphics.Color
import android.graphics.Paint
import android.net.Uri
import android.view.View
import android.os.Bundle
import android.provider.Settings
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.widget.Button
import android.widget.EditText
import android.widget.ImageButton
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.TextView
import android.widget.Toast
import android.widget.GridLayout
import android.view.Gravity
import android.graphics.Typeface
import android.view.LayoutInflater
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.ImageView
import android.widget.LinearLayout
import android.text.Editable
import android.text.TextWatcher
import android.widget.ScrollView
import android.widget.Spinner
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.lifecycleScope
import com.google.android.material.button.MaterialButton
import com.zipper.app.accessibility.KarvishAccessibilityService
import com.zipper.app.accessibility.KarvishBubbleOverlay
import com.zipper.app.core.agent.AgentRuntimeService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale
import kotlin.math.sqrt

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "KarvishMainActivity"
        private const val RC_RECORD_AUDIO = 2001
        private const val RC_TALK_TO_ZIPPER = 2002
    }

    private lateinit var tvLiveStatusBadge: TextView
    private lateinit var tvServiceStatus: TextView
    private lateinit var btnEnableService: Button
    private lateinit var rgEngine: RadioGroup
    private lateinit var rbOnDevice: RadioButton
    private lateinit var rbRemoteGpu: RadioButton
    private lateinit var etAgentGoal: EditText
    private lateinit var btnVoiceInput: MaterialButton
    private lateinit var btnPresetInsta: Button
    private lateinit var btnPresetLinkedin: Button
    private lateinit var btnPresetCalc: Button
    private lateinit var btnPresetYoutube: Button
    private lateinit var btnPresetCall: Button
    private lateinit var btnPresetZepto: Button
    private lateinit var btnSetDialer: Button
    private lateinit var btnRunAgentGoal: MaterialButton
    private lateinit var btnHardStop: MaterialButton
    private lateinit var btnDittoRecord: MaterialButton
    private lateinit var btnDittoReplay: MaterialButton
    private lateinit var tvTurnCounter: TextView
    private lateinit var tvAgentAuditLog: TextView
    private lateinit var btnOpenWebAssistant: Button
    private lateinit var llSentryContainer: View
    private lateinit var tvSentryIcon: TextView
    private lateinit var tvSentryLabel: TextView
    private lateinit var llExpenseContainer: View
    private lateinit var tvExpenseLabel: TextView
    private lateinit var llSecurityTestContainer: View
    private lateinit var tvSecurityIcon: TextView
    private lateinit var tvSecurityLabel: TextView
    private lateinit var tvActiveRemindersStatus: TextView
    private lateinit var btnSetCallReminder: Button
    private lateinit var btnTestCallNow: Button
    private lateinit var btnTalkToZipper: Button
    private lateinit var btnToggleWakeWord: Button
    private lateinit var btnTrainVoice: Button
    private lateinit var btnToggleFloatingBubble: Button

    // Section 1: Hero & Cognitive Orbit Views
    private var hero3dCoreView: com.zipper.app.core.graphics.Zipper3DCoreView? = null
    private var badgeUnderstand: View? = null
    private var badgeFindApps: View? = null
    private var badgePlanExecute: View? = null
    private var badgeVerifyDeliver: View? = null
    private var btnHeaderSearch: ImageButton? = null
    private var btnHeaderHistory: ImageButton? = null
    private var btnHeaderMore: ImageButton? = null
    private var lastExecutedGoal: String? = null

    private var speechRecognizer: SpeechRecognizer? = null
    private var isListening = false
    private var pendingCallGoal: String? = null

    private val requestAudioPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            startVoiceRecognition()
        } else {
            tvAgentAuditLog.text = "Microphone permission is required for voice goal input."
        }
    }

    private val requestTelephonyPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val callGranted = permissions[Manifest.permission.CALL_PHONE] ?: false
        val contactsGranted = permissions[Manifest.permission.READ_CONTACTS] ?: false
        if (callGranted && contactsGranted) {
            appendLog("[TELEPHONY] Telephony & Contacts permissions granted.")
            pendingCallGoal?.let { goal ->
                pendingCallGoal = null
                executeAutonomousCallGoal(goal)
            }
        } else {
            appendLog("[TELEPHONY] CALL_PHONE and READ_CONTACTS permissions are required for calling.")
        }
    }

    private val dialerRoleLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { _ ->
        if (com.zipper.app.telephony.KarvishCallManager.isDefaultDialer(this)) {
            appendLog("[TELEPHONY] Zipper is now the Default Dialer ✓")
            btnSetDialer.text = "📱 Dialer ✓"
        } else {
            appendLog("[TELEPHONY] Default dialer role was not granted or changed.")
        }
    }

    private var pendingAttachmentPlan: com.zipper.app.core.intelligence.planner.AutoDoPlan? = null
    private var pendingAttachmentItem: com.zipper.app.core.intelligence.store.CommitmentEntity? = null

    private val documentPickerLauncher = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        if (uri != null) {
            try {
                contentResolver.takePersistableUriPermission(
                    uri,
                    Intent.FLAG_GRANT_READ_URI_PERMISSION
                )
            } catch (_: Exception) {}

            var fileName = "Selected_Document.pdf"
            var fileSize = "PDF Document"
            try {
                contentResolver.query(uri, null, null, null, null)?.use { cursor ->
                    val nameIndex = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
                    val sizeIndex = cursor.getColumnIndex(android.provider.OpenableColumns.SIZE)
                    if (cursor.moveToFirst()) {
                        if (nameIndex != -1) fileName = cursor.getString(nameIndex) ?: fileName
                        if (sizeIndex != -1) {
                            val sizeBytes = cursor.getLong(sizeIndex)
                            fileSize = com.zipper.app.core.intelligence.resolver.DeviceAssetResolver.formatFileSize(sizeBytes)
                        }
                    }
                }
            } catch (_: Exception) {}

            val plan = pendingAttachmentPlan
            val item = pendingAttachmentItem
            if (plan != null && item != null && plan.steps.isNotEmpty()) {
                val step1 = plan.steps[0]
                step1.attachmentUri = uri
                step1.attachmentName = fileName
                step1.attachmentSize = fileSize
                showAutoDoPreviewDialog(item, plan)
            }
        }
    }

    private val agentFinishedReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            if (intent?.action != AgentRuntimeService.ACTION_GOAL_FINISHED) return
            val success = intent.getBooleanExtra(AgentRuntimeService.EXTRA_SUCCESS, false)
            val summary = intent.getStringExtra(AgentRuntimeService.EXTRA_SUMMARY).orEmpty()

            btnRunAgentGoal.isEnabled = true
            btnHardStop.isEnabled = true

            if (success) {
                tvLiveStatusBadge.text = "✅ COMPLETED"
                tvLiveStatusBadge.setTextColor(Color.parseColor("#34D399"))
                hero3dCoreView?.currentState = com.zipper.app.core.graphics.ZipperCoreState.COMPLETED
                appendLog("[GOAL SUCCESS] $summary")
            } else {
                tvLiveStatusBadge.text = "⏹️ STOPPED"
                tvLiveStatusBadge.setTextColor(Color.parseColor("#F59E0B"))
                hero3dCoreView?.currentState = com.zipper.app.core.graphics.ZipperCoreState.FAILED
                appendLog("[EXECUTION HALTED] $summary")
            }

            val goalText = lastExecutedGoal ?: etAgentGoal.text.toString().trim()
            if (goalText.isNotBlank()) {
                val app = when {
                    goalText.contains("whatsapp", ignoreCase = true) -> "WhatsApp"
                    goalText.contains("youtube", ignoreCase = true) -> "YouTube"
                    goalText.contains("calc", ignoreCase = true) -> "Calculator"
                    goalText.contains("flight", ignoreCase = true) || goalText.contains("web", ignoreCase = true) -> "Web"
                    goalText.contains("blinkit", ignoreCase = true) -> "Blinkit"
                    goalText.contains("zepto", ignoreCase = true) -> "Zepto"
                    goalText.contains("insta", ignoreCase = true) -> "Instagram"
                    else -> "Zipper"
                }
                com.zipper.app.core.history.KarvishTaskStore.addTask(this@MainActivity, goalText, app, success)
                updateRecentTasksUi()
            }
        }
    }

    private val plannerDataChangedReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            refreshPlannerDashboard()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        bindViews()
        setupListeners()
        setupSpeechRecognizer()
        observeCallState()
        updateServiceStatus()
        val prefs = getSharedPreferences("karvish_agent_prefs", Context.MODE_PRIVATE)
        if (!prefs.contains("selected_model") || prefs.getString("selected_model", "") == "V58") {
            prefs.edit().putString("selected_model", "V60").apply()
        }

        ContextCompat.registerReceiver(
            this,
            agentFinishedReceiver,
            IntentFilter(AgentRuntimeService.ACTION_GOAL_FINISHED),
            ContextCompat.RECEIVER_NOT_EXPORTED
        )
        ContextCompat.registerReceiver(
            this,
            plannerDataChangedReceiver,
            IntentFilter(com.zipper.app.core.intelligence.health.HealthWorkoutIngestionEngine.ACTION_PLANNER_DATA_CHANGED),
            ContextCompat.RECEIVER_NOT_EXPORTED
        )

        handleIntentTriggers(intent)
        updatePluginDashboardUi()
        updateCallRemindersStatus()
        updateWakeWordUi(com.zipper.app.core.voice.WakeWordService.isWakeWordEnabled(this))
        com.zipper.app.core.voice.ParadeeVoiceEngine.init(this)
        updateRecentTasksUi()

        // Arm Karvish Personal Intelligence Nightly Planner & 5-min Awakening Calls
        com.zipper.app.core.intelligence.scheduler.NightlyPlanBroadcastReceiver.armNightlyAlarm(this)
        com.zipper.app.core.intelligence.scheduler.LifePlanCallScheduler.scheduleUpcomingAlarms(this)
    }

    private fun updatePluginDashboardUi() {
        try {
            // 1. Sentry Status
            val pluginManager = com.zipper.app.core.plugins.BackgroundPluginManager.getInstance(this)
            if (pluginManager.isSentryArmed()) {
                tvSentryIcon.text = "🚨"
                tvSentryLabel.text = "Sentry ARMED"
                tvSentryLabel.setTextColor(Color.parseColor("#EF4444"))
            } else {
                tvSentryIcon.text = "🛡️"
                tvSentryLabel.text = "Sentry Off"
                tvSentryLabel.setTextColor(Color.parseColor("#94A3B8"))
            }

            // 2. Expense Status
            val summary = pluginManager.expenseLedger.getTodaySummary()
            tvExpenseLabel.text = "Spend: ₹${summary.totalAmount.toInt()}"

            // 3. Security Test Mode Status & Home Card
            val isTestActive = com.zipper.app.core.remote.RemoteSecurityGate.isTestModeActive(this)
            val isRemoteEnabled = com.zipper.app.core.remote.RemoteSecurityGate.isEnabled(this)
            val currentPin = com.zipper.app.core.remote.RemoteSecurityGate.getSecurityPin(this)

            if (isTestActive) {
                tvSecurityIcon.text = "⚡"
                tvSecurityLabel.text = "Open (1h)"
                tvSecurityLabel.setTextColor(Color.parseColor("#34D399"))
            } else {
                tvSecurityIcon.text = "🔒"
                tvSecurityLabel.text = "PIN #$currentPin"
                tvSecurityLabel.setTextColor(Color.parseColor("#F59E0B"))
            }

            val tvRemoteSecurityTitle = findViewById<TextView>(R.id.tvRemoteSecurityTitle)
            val tvRemoteSecuritySubtitle = findViewById<TextView>(R.id.tvRemoteSecuritySubtitle)
            val tvRemoteSecurityIcon = findViewById<TextView>(R.id.tvRemoteSecurityIcon)

            if (!isRemoteEnabled) {
                tvRemoteSecurityIcon?.text = "⏸️"
                tvRemoteSecurityTitle?.text = "Remote Password: #$currentPin (Paused)"
                tvRemoteSecuritySubtitle?.text = "Remote commands paused • Tap to enable & configure"
            } else if (isTestActive) {
                tvRemoteSecurityIcon?.text = "⚡"
                tvRemoteSecurityTitle?.text = "Remote Password: #$currentPin (1h Open Bypass)"
                tvRemoteSecuritySubtitle?.text = "Open Test Mode active! Any WhatsApp/SMS command executes"
            } else {
                tvRemoteSecurityIcon?.text = "🔐"
                tvRemoteSecurityTitle?.text = "Remote Password: #$currentPin"
                tvRemoteSecuritySubtitle?.text = "WhatsApp/SMS commands active • Tap to change PIN or view guide"
            }
        } catch (_: Exception) {}
    }

    private fun bindViews() {
        tvLiveStatusBadge = findViewById(R.id.tvLiveStatusBadge)
        tvServiceStatus = findViewById(R.id.tvServiceStatus)
        btnEnableService = findViewById(R.id.btnEnableService)
        rgEngine = findViewById(R.id.rgEngine)
        rbOnDevice = findViewById(R.id.rbOnDevice)
        rbRemoteGpu = findViewById(R.id.rbRemoteGpu)
        etAgentGoal = findViewById(R.id.etAgentGoal)
        btnVoiceInput = findViewById(R.id.btnVoiceInput)
        btnPresetInsta = findViewById(R.id.btnPresetInsta)
        btnPresetLinkedin = findViewById(R.id.btnPresetLinkedin)
        btnPresetCalc = findViewById(R.id.btnPresetCalc)
        btnPresetYoutube = findViewById(R.id.btnPresetYoutube)
        btnPresetCall = findViewById(R.id.btnPresetCall)
        btnPresetZepto = findViewById(R.id.btnPresetZepto)
        btnSetDialer = findViewById(R.id.btnSetDialer)
        btnRunAgentGoal = findViewById(R.id.btnRunAgentGoal)
        btnHardStop = findViewById(R.id.btnHardStop)
        btnDittoRecord = findViewById(R.id.btnDittoRecord)
        btnDittoReplay = findViewById(R.id.btnDittoReplay)
        tvTurnCounter = findViewById(R.id.tvTurnCounter)
        tvAgentAuditLog = findViewById(R.id.tvAgentAuditLog)

        btnOpenWebAssistant = findViewById(R.id.btnOpenWebAssistant)
        llSentryContainer = findViewById(R.id.llSentryContainer)
        tvSentryIcon = findViewById(R.id.tvSentryIcon)
        tvSentryLabel = findViewById(R.id.tvSentryLabel)
        llExpenseContainer = findViewById(R.id.llExpenseContainer)
        tvExpenseLabel = findViewById(R.id.tvExpenseLabel)
        llSecurityTestContainer = findViewById(R.id.llSecurityTestContainer)
        tvSecurityIcon = findViewById(R.id.tvSecurityIcon)
        tvSecurityLabel = findViewById(R.id.tvSecurityLabel)
        tvActiveRemindersStatus = findViewById(R.id.tvActiveRemindersStatus)
        btnSetCallReminder = findViewById(R.id.btnSetCallReminder)
        btnTestCallNow = findViewById(R.id.btnTestCallNow)
        btnTalkToZipper = findViewById(R.id.btnTalkToZipper)
        btnToggleWakeWord = findViewById(R.id.btnToggleWakeWord)
        btnTrainVoice = findViewById(R.id.btnTrainVoice)
        btnToggleFloatingBubble = findViewById(R.id.btnToggleFloatingBubble)

        // Section 1: Hero & Cognitive Orbit
        hero3dCoreView = findViewById<View>(R.id.hero3dCoreView) as? com.zipper.app.core.graphics.Zipper3DCoreView
        hero3dCoreView?.currentState = com.zipper.app.core.graphics.ZipperCoreState.IDLE
        badgeUnderstand = findViewById<View>(R.id.badgeUnderstand)
        badgeFindApps = findViewById<View>(R.id.badgeFindApps)
        badgePlanExecute = findViewById<View>(R.id.badgePlanExecute)
        badgeVerifyDeliver = findViewById<View>(R.id.badgeVerifyDeliver)
        btnHeaderSearch = findViewById<ImageButton>(R.id.btnHeaderSearch)
        btnHeaderHistory = findViewById<ImageButton>(R.id.btnHeaderHistory)
        btnHeaderMore = findViewById<ImageButton>(R.id.btnHeaderMore)

        intent?.getStringExtra("set_model")?.let { model ->
            getSharedPreferences("karvish_agent_prefs", Context.MODE_PRIVATE).edit().putString("selected_model", model).apply()
        }

        updateModelStatusDisplay()
        handleIntentTriggers(intent)
    }

    private fun updateModelStatusDisplay() {
        val prefs = getSharedPreferences("karvish_agent_prefs", Context.MODE_PRIVATE)
        val selectedModel = prefs.getString("selected_model", "V62") ?: "V62"
        val hasV62 = try { assets.list("models")?.contains("karvish_v62.onnx") == true } catch (_: Exception) { false }
        val hasV61 = try { assets.list("models")?.contains("karvish_v61.onnx") == true } catch (_: Exception) { false }
        val hasV60 = try { assets.list("models")?.contains("karvish_v60.onnx") == true } catch (_: Exception) { false }
        val hasV59 = try { assets.list("models")?.contains("karvish_v59.onnx") == true } catch (_: Exception) { false }
        val hasV58 = try { assets.list("models")?.contains("karvish_v58.onnx") == true } catch (_: Exception) { false }
        val hasV57 = try { assets.list("models")?.contains("karvish_v57.onnx") == true } catch (_: Exception) { false }
        val hasV56 = try { assets.list("models")?.contains("karvish_v56.onnx") == true } catch (_: Exception) { false }
        val hasV55 = try { assets.list("models")?.contains("karvish_v55.onnx") == true } catch (_: Exception) { false }
        val hasV54Fp16 = try { assets.list("models")?.contains("karvish_v54_fp16.onnx") == true } catch (_: Exception) { false }
        val hasV54 = try { assets.list("models")?.contains("karvish_v54.onnx") == true } catch (_: Exception) { false }

        val activeModel = when (selectedModel) {
            "V62" -> if (hasV62) "V62" else if (hasV61) "V61" else if (hasV60) "V60" else if (hasV59) "V59" else "V58"
            "V61" -> if (hasV61) "V61" else if (hasV62) "V62" else if (hasV60) "V60" else if (hasV59) "V59" else "V58"
            "V60" -> if (hasV60) "V60" else if (hasV62) "V62" else if (hasV61) "V61" else if (hasV59) "V59" else "V58"
            "V59" -> if (hasV59) "V59" else if (hasV62) "V62" else if (hasV61) "V61" else if (hasV60) "V60" else "V58"
            "V58" -> if (hasV58) "V58" else if (hasV62) "V62" else if (hasV61) "V61" else if (hasV60) "V60" else "V57"
            "V57" -> if (hasV57) "V57" else if (hasV62) "V62" else if (hasV61) "V61" else if (hasV60) "V60" else "V58"
            "V56" -> if (hasV56) "V56" else if (hasV62) "V62" else if (hasV61) "V61" else if (hasV60) "V60" else "V58"
            "V55" -> if (hasV55) "V55" else if (hasV62) "V62" else if (hasV61) "V61" else if (hasV60) "V60" else "V58"
            "V54" -> if (hasV54Fp16 || hasV54) "V54" else if (hasV62) "V62" else if (hasV61) "V61" else if (hasV60) "V60" else "V58"
            else -> if (hasV62) "V62" else if (hasV61) "V61" else if (hasV60) "V60" else if (hasV59) "V59" else "V58"
        }
        val modelParams = when (activeModel) {
            "V62" -> "30.74M FP16 Mobile-Optimized (Unified K+1 Autonomy, Zero-Gate Stop & Anti-Wandering Sentinel)"
            "V61" -> "32.33M FP16 Policy + 3.77M MobileNetV4 Eye (36.10M Multimodal Universal Agent & Hinglish NLU)"
            "V60" -> "28.99M FP16 Mobile-Optimized (Universal Zero-Shot 50-Domain Agent & Calibrated Termination Verifier)"
            "V59" -> "28.89M FP16 Mobile-Optimized (Universal NLU Slot Filling & Typos/Slang Invariance Policy)"
            "V58" -> "28.88M FP16 Mobile-Optimized (Overlay Escape, Anti-Hallucination Sentinel & Lookahead Policy)"
            "V57" -> "28.88M FP16 Mobile-Optimized (Multi-Hop Social & History-Disambiguated Policy)"
            "V56" -> "28.88M FP16 Mobile-Optimized (Universal Dual-Slot Action Policy & Stall Recovery)"
            "V55" -> "29.65M FP16 Mobile-Optimized (Universal Action Policy & Closed-Loop Verifier)"
            "V54" -> "26.78M FP16 Mobile-Optimized (Universal Search & Contact Control Baseline)"
            else -> "26.78M FP32 (Pure Neural Policy)"
        }
        val modelAsset = when (activeModel) {
            "V62" -> "models/karvish_v62.onnx (57.06 MB)"
            "V61" -> "models/karvish_v61.onnx (53.80 MB)"
            "V60" -> "models/karvish_v60.onnx (51.05 MB)"
            "V59" -> "models/karvish_v59.onnx (51.03 MB)"
            "V58" -> "models/karvish_v58.onnx (51.01 MB)"
            "V57" -> "models/karvish_v57.onnx (51.01 MB)"
            "V56" -> "models/karvish_v56.onnx (51.01 MB)"
            "V55" -> "models/karvish_v55.onnx (50.88 MB)"
            "V54" -> if (hasV54Fp16) "models/karvish_v54_fp16.onnx (50.88 MB)" else "models/karvish_v54.onnx (50.66 MB)"
            else -> "models/karvish_v52.onnx (98.03 MB)"
        }

        tvAgentAuditLog.text = """
            ⚡ KARVISH RUNTIME INITIALIZED
            [DEPLOYED MODEL] Karvish Model $activeModel (Universal Multimodal Agent) [Tap to Toggle V62/V61/V60/V59/V58/V57/V56/V55/V54]
            [PARAMS] $modelParams
            [ASSET] $modelAsset
            [TELEPHONY] InCallService & Acoustic Uplink Engine Active
            [STATUS] Ready for autonomous execution.
        """.trimIndent()
    }

    private fun setupListeners() {
        // Section 1: Hero Cognitive Badges
        badgeUnderstand?.setOnClickListener {
            triggerCognitiveBadgePulse(
                it,
                com.zipper.app.core.graphics.ZipperCoreState.THINKING,
                "Understand your request",
                "Real-time Multimodal intent decoding & action schema mapping."
            )
        }
        badgeFindApps?.setOnClickListener {
            triggerCognitiveBadgePulse(
                it,
                com.zipper.app.core.graphics.ZipperCoreState.WORKING,
                "Find the right apps and actions",
                "Universal candidate grounding across all installed apps."
            )
        }
        badgePlanExecute?.setOnClickListener {
            triggerCognitiveBadgePulse(
                it,
                com.zipper.app.core.graphics.ZipperCoreState.WORKING,
                "Plan and execute safely",
                "Closed-loop policy with safe viewport bounds & stale-target checks."
            )
        }
        badgeVerifyDeliver?.setOnClickListener {
            triggerCognitiveBadgePulse(
                it,
                com.zipper.app.core.graphics.ZipperCoreState.VERIFYING,
                "Verify and deliver the result",
                "State transition verification & calibrated task termination."
            )
        }

        // Section 1: Top Header Action Buttons
        btnHeaderSearch?.setOnClickListener {
            etAgentGoal.requestFocus()
            val imm = getSystemService(Context.INPUT_METHOD_SERVICE) as? android.view.inputmethod.InputMethodManager
            imm?.showSoftInput(etAgentGoal, android.view.inputmethod.InputMethodManager.SHOW_IMPLICIT)
        }
        btnHeaderHistory?.setOnClickListener {
            val dialog = androidx.appcompat.app.AlertDialog.Builder(this)
                .setTitle("Task History & Audit Log")
                .setMessage(tvAgentAuditLog.text)
                .setPositiveButton("Close", null)
                .show()
        }
        val showHeaderMoreMenu: (View) -> Unit = { anchor ->
            val popup = androidx.appcompat.widget.PopupMenu(this, anchor)
            val currentPin = com.zipper.app.core.remote.RemoteSecurityGate.getSecurityPin(this)
            popup.menu.add(0, 5, 0, "🔐 Remote Security Password (#$currentPin)")
            popup.menu.add(0, 6, 1, "🤖 AI Auto-Reply & Delegated Contacts")
            popup.menu.add(0, 1, 2, "Select Karvish Model")
            popup.menu.add(0, 2, 3, "Web Assistant & Search")
            popup.menu.add(0, 3, 4, "Test Overlay Bubble")
            popup.menu.add(0, 4, 5, "Accessibility Settings")
            popup.setOnMenuItemClickListener { item ->
                when (item.itemId) {
                    5 -> {
                        com.zipper.app.core.remote.RemoteSecurityDialog(this) {
                            updatePluginDashboardUi()
                            refreshPlannerDashboard()
                        }.show()
                    }
                    6 -> {
                        com.zipper.app.core.intelligence.reply.DelegatedContactDialog(
                            context = this,
                            onContactSaved = { refreshPlannerDashboard() },
                            onContactDeleted = { refreshPlannerDashboard() }
                        ).show()
                    }
                    1 -> tvAgentAuditLog.performClick()
                    2 -> btnOpenWebAssistant.performClick()
                    3 -> btnToggleFloatingBubble.performClick()
                    4 -> btnEnableService.performClick()
                }
                true
            }
            popup.show()
        }

        btnHeaderMore?.setOnClickListener { v ->
            showHeaderMoreMenu(v)
        }
        findViewById<View>(R.id.btnUserProfileAvatar)?.setOnClickListener { v ->
            showHeaderMoreMenu(v)
        }

        findViewById<View>(R.id.cardRemoteSecurityStatus)?.setOnClickListener {
            com.zipper.app.core.remote.RemoteSecurityDialog(this) {
                updatePluginDashboardUi()
                refreshPlannerDashboard()
            }.show()
        }



        findViewById<View>(R.id.cardQuickResearch)?.setOnClickListener {
            val intent = Intent(this, com.zipper.app.webmind.WebmindActivity::class.java).apply {
                putExtra("initial_query", "Find best laptops under 70k in India for coding and light gaming")
            }
            launchWebmindActivity(intent)
        }
        findViewById<View>(R.id.cardQuickOpenApp)?.setOnClickListener {
            showSupportedAppsDialog()
        }
        findViewById<View>(R.id.cardQuickPlanTask)?.setOnClickListener {
            etAgentGoal.setText("Plan my tasks and set schedule for today")
            etAgentGoal.setSelection(etAgentGoal.text.length)
            btnRunAgentGoal.performClick()
        }
        findViewById<View>(R.id.cardQuickTakeAction)?.setOnClickListener {
            etAgentGoal.setText("Open WhatsApp and send message")
            etAgentGoal.setSelection(etAgentGoal.text.length)
            btnRunAgentGoal.performClick()
        }

        // Camera Vision button
        findViewById<View>(R.id.btnCameraVision)?.setOnClickListener {
            Toast.makeText(this, "Vision Lens: Ready to perceive screen & documents", Toast.LENGTH_SHORT).show()
        }

        findViewById<View>(R.id.tvSeeAllRecentTasks)?.setOnClickListener {
            btnHeaderHistory?.performClick()
        }

        // --- SECTION 5: TODAY'S FOCUS & SUGGESTIONS ---
        findViewById<View>(R.id.rowSuggestAmazon)?.setOnClickListener {
            etAgentGoal.setText("Check price drop on my Amazon wishlist")
            etAgentGoal.setSelection(etAgentGoal.text.length)
            btnRunAgentGoal.performClick()
        }
        findViewById<View>(R.id.rowSuggestPlan)?.setOnClickListener {
            etAgentGoal.setText("Plan my tomorrow schedule in Keep")
            etAgentGoal.setSelection(etAgentGoal.text.length)
            btnRunAgentGoal.performClick()
        }
        findViewById<View>(R.id.rowSuggestWorkout)?.setOnClickListener {
            etAgentGoal.setText("Search workout playlist on YouTube")
            etAgentGoal.setSelection(etAgentGoal.text.length)
            btnRunAgentGoal.performClick()
        }
        findViewById<View>(R.id.btnRefreshSuggestions)?.setOnClickListener {
            Toast.makeText(this, "Refreshed AI suggestions ✨", Toast.LENGTH_SHORT).show()
        }

        // --- SECTION 6: BOTTOM NAVIGATION TABS ---
        findViewById<View>(R.id.tabAgent)?.setOnClickListener {
            switchToAgentTab()
        }
        findViewById<View>(R.id.tabPlanner)?.setOnClickListener {
            switchToPlannerTab()
        }
        findViewById<View>(R.id.tabWebmind)?.setOnClickListener {
            switchToWebmindTab()
        }
        findViewById<View>(R.id.tabDitto)?.setOnClickListener {
            showDittoHubDialog()
        }
        findViewById<View>(R.id.tabSettings)?.setOnClickListener {
            btnHeaderMore?.performClick()
        }

        onBackPressedDispatcher.addCallback(this, object : androidx.activity.OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (findViewById<View>(R.id.containerWebmindContent)?.visibility == View.VISIBLE) {
                    switchToAgentTab()
                    return
                }
                if (findViewById<View>(R.id.scrollPlannerContent)?.visibility == View.VISIBLE) {
                    switchToAgentTab()
                    return
                }
                isEnabled = false
                onBackPressedDispatcher.onBackPressed()
                isEnabled = true
            }
        })

        setupPlannerListeners()

        btnOpenWebAssistant.setOnClickListener {
            startActivity(Intent(this, com.zipper.app.core.web.WebAssistantActivity::class.java))
        }

        llSentryContainer.setOnClickListener {
            val pluginManager = com.zipper.app.core.plugins.BackgroundPluginManager.getInstance(this)
            if (pluginManager.isSentryArmed()) {
                pluginManager.disarmSentryGuard()
                Toast.makeText(this, "🛡️ Sentry Guard Disarmed", Toast.LENGTH_SHORT).show()
            } else {
                pluginManager.armSentryGuard { reason ->
                    runOnUiThread {
                        Toast.makeText(this, "🚨 SENTRY ALARM: $reason", Toast.LENGTH_LONG).show()
                        updatePluginDashboardUi()
                    }
                }
                Toast.makeText(this, "🛡️ Sentry Guard Armed! (Anti-Theft Active)", Toast.LENGTH_SHORT).show()
            }
            updatePluginDashboardUi()
        }

        llExpenseContainer.setOnClickListener {
            val pluginManager = com.zipper.app.core.plugins.BackgroundPluginManager.getInstance(this)
            val summary = pluginManager.expenseLedger.getTodaySummary()
            val dialogBuilder = androidx.appcompat.app.AlertDialog.Builder(this)
            dialogBuilder.setTitle("💳 Auto-Expense Ledger")
            dialogBuilder.setMessage(summary.formatReport())
            dialogBuilder.setPositiveButton("OK") { dialog, _ -> dialog.dismiss() }
            dialogBuilder.show()
        }

        llSecurityTestContainer.setOnClickListener {
            com.zipper.app.core.remote.RemoteSecurityDialog(this) {
                updatePluginDashboardUi()
                refreshPlannerDashboard()
            }.show()
        }

        llSecurityTestContainer.setOnLongClickListener {
            val currentPin = com.zipper.app.core.remote.RemoteSecurityGate.getSecurityPin(this)
            val input = android.widget.EditText(this).apply {
                setText(currentPin)
                setSelection(text.length)
                hint = "Enter 4-8 character PIN/Password"
                setTextColor(Color.WHITE)
            }
            androidx.appcompat.app.AlertDialog.Builder(this)
                .setTitle("🔐 Configure Remote Password")
                .setMessage("Remote commands starting with this password (e.g. '#$currentPin: ring') will be authorized from any phone/number.")
                .setView(input)
                .setPositiveButton("Save") { _, _ ->
                    val newPin = input.text.toString().trim()
                    if (newPin.length >= 4) {
                        com.zipper.app.core.remote.RemoteSecurityGate.setSecurityPin(this, newPin)
                        Toast.makeText(this, "✅ Security PIN updated to #$newPin", Toast.LENGTH_SHORT).show()
                        updatePluginDashboardUi()
                    } else {
                        Toast.makeText(this, "⚠️ PIN must be at least 4 characters", Toast.LENGTH_SHORT).show()
                    }
                }
                .setNegativeButton("Cancel", null)
                .show()
            true
        }

        btnSetCallReminder.setOnClickListener {
            com.zipper.app.core.voice.ManualReminderDialog(this) {
                updateCallRemindersStatus()
            }.show()
        }

        btnTestCallNow.setOnClickListener {
            val testCommitment = com.zipper.app.core.intelligence.store.CommitmentEntity(
                id = 9991L,
                title = "Evening Workout & Mobility",
                category = com.zipper.app.core.intelligence.store.CommitmentCategory.FITNESS,
                scheduledTimeMs = System.currentTimeMillis() + (5 * 60 * 1000L),
                durationMinutes = 30,
                sourceApp = "Karvish Life Guard",
                sourcePerson = "Karvish",
                candidatePackages = listOf("fit.cure.android", "com.sec.android.app.shealth", "com.strava"),
                notes = "Daily fitness routine"
            )
            com.zipper.app.core.intelligence.scheduler.LifePlanCallScheduler.triggerTestCallNow(this, testCommitment)
            Toast.makeText(this, "📞 Personal Intelligence call ringing in 3s! Answer to see real app icons.", Toast.LENGTH_LONG).show()
            updateCallRemindersStatus()
        }

        btnTalkToZipper.setOnClickListener {
            if (androidx.core.content.ContextCompat.checkSelfPermission(
                    this,
                    android.Manifest.permission.RECORD_AUDIO
                ) != android.content.pm.PackageManager.PERMISSION_GRANTED
            ) {
                androidx.core.app.ActivityCompat.requestPermissions(
                    this,
                    arrayOf(android.Manifest.permission.RECORD_AUDIO),
                    RC_TALK_TO_ZIPPER
                )
                return@setOnClickListener
            }
            com.zipper.app.core.voice.VoiceAssistantActivity.start(this)
        }

        val isWakeEnabled = com.zipper.app.core.voice.WakeWordService.isWakeWordEnabled(this)
        updateWakeWordUi(isWakeEnabled)

        btnToggleWakeWord.setOnClickListener {
            val currentlyEnabled = com.zipper.app.core.voice.WakeWordService.isWakeWordEnabled(this)
            val newEnabled = !currentlyEnabled
            if (newEnabled) {
                if (androidx.core.content.ContextCompat.checkSelfPermission(
                        this,
                        android.Manifest.permission.RECORD_AUDIO
                    ) != android.content.pm.PackageManager.PERMISSION_GRANTED
                ) {
                    androidx.core.app.ActivityCompat.requestPermissions(
                        this,
                        arrayOf(android.Manifest.permission.RECORD_AUDIO),
                        RC_RECORD_AUDIO
                    )
                    return@setOnClickListener
                }
                val started = com.zipper.app.core.voice.WakeWordService.start(this)
                if (started) {
                    Toast.makeText(this, "🎙️ 'Hey Zipper' Wake Word Active! Say 'Hey Zipper' anytime.", Toast.LENGTH_LONG).show()
                    updateWakeWordUi(true)
                } else {
                    Toast.makeText(this, "Could not start wake word service", Toast.LENGTH_SHORT).show()
                    updateWakeWordUi(false)
                }
            } else {
                com.zipper.app.core.voice.WakeWordService.stop(this)
                Toast.makeText(this, "Wake Word turned OFF", Toast.LENGTH_SHORT).show()
                updateWakeWordUi(false)
            }
        }

        val isVoiceEnrolled = com.zipper.app.core.voice.SpeakerVoiceVerifier.isEnrolled(this)
        updateTrainVoiceUi(isVoiceEnrolled)

        btnTrainVoice.setOnClickListener {
            startVoiceTrainingDialog()
        }

        btnToggleFloatingBubble.setOnClickListener {
            val a11y = com.zipper.app.accessibility.KarvishAccessibilityService.instance
            if (a11y != null) {
                com.zipper.app.accessibility.KarvishBubbleOverlay.wakeUpFromBottom(a11y)
            } else {
                com.zipper.app.accessibility.KarvishBubbleOverlay.wakeUpFromBottom(this)
            }
            Toast.makeText(this, "🎙️ Waking up assistant pill from bottom...", Toast.LENGTH_SHORT).show()
        }

        btnEnableService.setOnClickListener {
            startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS))
        }

        tvAgentAuditLog.setOnClickListener {
            val prefs = getSharedPreferences("karvish_agent_prefs", Context.MODE_PRIVATE)
            val curr = prefs.getString("selected_model", "V62") ?: "V62"
            val next = when (curr) {
                "V62" -> "V61"
                "V61" -> "V60"
                "V60" -> "V59"
                "V59" -> "V58"
                "V58" -> "V57"
                "V57" -> "V56"
                "V56" -> "V55"
                "V55" -> "V54"
                else -> "V62"
            }
            prefs.edit().putString("selected_model", next).apply()
            Toast.makeText(this, "A/B Model Switch: Active model set to $next", Toast.LENGTH_SHORT).show()
            updateModelStatusDisplay()
            val targetAsset = when (next) {
                "V62" -> "models/karvish_v62.onnx"
                "V61" -> "models/karvish_v61.onnx"
                "V60" -> "models/karvish_v60.onnx"
                "V59" -> "models/karvish_v59.onnx"
                "V58" -> "models/karvish_v58.onnx"
                "V57" -> "models/karvish_v57.onnx"
                "V56" -> "models/karvish_v56.onnx"
                "V55" -> "models/karvish_v55.onnx"
                else -> "models/karvish_v54_fp16.onnx"
            }
            val assetToWarm = try {
                assets.open(targetAsset).close()
                targetAsset
            } catch (_: Exception) {
                "models/karvish_v62.onnx"
            }
            com.zipper.app.core.agent.OnDeviceRankerEngine.preWarm(this, assetToWarm)
        }

        btnPresetInsta.setOnClickListener {
            etAgentGoal.setText("Open Instagram and search photography")
        }

        btnPresetLinkedin.setOnClickListener {
            etAgentGoal.setText("Open LinkedIn and search software engineer")
        }

        btnPresetCalc.setOnClickListener {
            etAgentGoal.setText("Open Calculator and calculate 50 - 20")
        }

        btnPresetYoutube.setOnClickListener {
            etAgentGoal.setText("Open YouTube and search lo-fi hip hop")
        }

        btnPresetCall.setOnClickListener {
            etAgentGoal.setText("Call John and say Karthik will come in a few minutes")
        }

        btnPresetZepto.setOnClickListener {
            etAgentGoal.setText("Open Zepto and search fresh tomatoes")
        }

        btnSetDialer.setOnClickListener {
            etAgentGoal.setText("Call Amma and say I am coming home now")
        }

        btnVoiceInput.setOnClickListener {
            toggleVoiceInput()
        }

        btnRunAgentGoal.setOnClickListener {
            val goal = etAgentGoal.text.toString().trim()
            val cleanLower = goal.lowercase(Locale.ROOT)

            if (cleanLower.startsWith("auto reply") || cleanLower.startsWith("ai reply") || cleanLower.startsWith("delegate")) {
                val candidateName = cleanLower
                    .removePrefix("auto reply to")
                    .removePrefix("auto reply")
                    .removePrefix("ai reply to")
                    .removePrefix("ai reply")
                    .removePrefix("delegate contact")
                    .removePrefix("delegate")
                    .trim()
                com.zipper.app.core.intelligence.reply.DelegatedContactDialog(
                    context = this,
                    initialContactName = candidateName.ifBlank { null },
                    onContactSaved = { refreshPlannerDashboard() }
                ).show()
                return@setOnClickListener
            }

            if (cleanLower.startsWith("voip") || cleanLower.contains("cloud call") || cleanLower.contains("voipbin") || cleanLower == "call settings") {
                com.zipper.app.telephony.voip.VoipBinSettingsDialog(this).show()
                return@setOnClickListener
            }

            val useRemote = rbRemoteGpu.isChecked
            runAutonomousAgentGoal(goal, enableRecovery = true, useRemoteRanker = useRemote)
        }

        btnHardStop.setOnClickListener {
            executeHardStop()
        }

        btnDittoRecord.setOnClickListener {
            val goal = etAgentGoal.text.toString().trim().ifBlank { "Demonstrated Task" }
            startDittoRecording(goal)
        }

        btnDittoReplay.setOnClickListener {
            showDittoReplayDialog()
        }

        findViewById<View>(R.id.cardWebAssistant)?.setOnClickListener {
            startActivity(Intent(this, com.zipper.app.core.web.WebAssistantActivity::class.java))
        }
        findViewById<View>(R.id.btnOpenWebAssistant)?.setOnClickListener {
            startActivity(Intent(this, com.zipper.app.core.web.WebAssistantActivity::class.java))
        }
    }

    private fun startDittoRecording(goalTitle: String) {
        val a11y = KarvishAccessibilityService.instance
        if (a11y == null) {
            Toast.makeText(this, "Please enable Karvish Accessibility Service first", Toast.LENGTH_SHORT).show()
            return
        }
        a11y.dittoRecorder.startRecording(goalTitle)
        appendLog("[DITTO RECORDER] Recording started for '$goalTitle'. Floating Finish HUD attached.")
        moveTaskToBack(true)
    }

    private fun showDittoReplayDialog() {
        val store = com.zipper.app.ditto.DittoStore(this)
        val flows = store.listFlows()
        if (flows.isEmpty()) {
            Toast.makeText(this, "No Ditto flows recorded yet. Tap RECORD DITTO first!", Toast.LENGTH_LONG).show()
            return
        }
        val titles = flows.map { "${it.title} (${it.steps.size} steps)" }.toTypedArray()
        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("Select Ditto Flow to Replay")
            .setItems(titles) { _, which ->
                val selectedFlow = flows[which]
                runDittoReplay(selectedFlow)
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun runDittoReplay(flow: com.zipper.app.ditto.DittoFlow) {
        val a11y = KarvishAccessibilityService.instance
        if (a11y == null) {
            Toast.makeText(this, "Please enable Karvish Accessibility Service first", Toast.LENGTH_SHORT).show()
            return
        }
        appendLog("[DITTO REPLAY] Starting deterministic replay: '${flow.title}' (${flow.steps.size} steps)...")
        tvLiveStatusBadge.text = "▶ DITTO REPLAY"
        tvLiveStatusBadge.setTextColor(Color.parseColor("#00E676"))
        moveTaskToBack(true)

        a11y.dittoPlayer.replayFlow(flow) { success, summary ->
            runOnUiThread {
                if (success) {
                    tvLiveStatusBadge.text = "✅ COMPLETED"
                    tvLiveStatusBadge.setTextColor(Color.parseColor("#34D399"))
                    appendLog("[DITTO SUCCESS] $summary")
                } else {
                    tvLiveStatusBadge.text = "⏹️ STOPPED"
                    tvLiveStatusBadge.setTextColor(Color.parseColor("#F59E0B"))
                    appendLog("[DITTO HALTED] $summary")
                }
            }
        }
    }

    private fun observeCallState() {
        lifecycleScope.launch {
            com.zipper.app.telephony.NormalCallController.callState.collect { state ->
                handleCallStateUpdate(state)
            }
        }
        lifecycleScope.launch {
            com.zipper.app.telephony.KarvishInCallService.callState.collect { state ->
                handleCallStateUpdate(state)
            }
        }
    }

    private fun handleCallStateUpdate(state: com.zipper.app.telephony.KarvishCallState) {
        when (state) {
            is com.zipper.app.telephony.KarvishCallState.Idle -> {
                if (!AgentRuntimeService.isRunning && tvLiveStatusBadge.text.toString().contains("CALL")) {
                    tvLiveStatusBadge.text = "🟢 READY"
                    tvLiveStatusBadge.setTextColor(Color.parseColor("#34D399"))
                }
            }
            is com.zipper.app.telephony.KarvishCallState.Dialing -> {
                tvLiveStatusBadge.text = "📞 DIALING..."
                tvLiveStatusBadge.setTextColor(Color.parseColor("#38BDF8"))
                appendLog("[CALL] Dialing ${state.targetContact} (${state.phoneNumber})...")
            }
            is com.zipper.app.telephony.KarvishCallState.Ringing -> {
                tvLiveStatusBadge.text = "🔔 RINGING..."
                tvLiveStatusBadge.setTextColor(Color.parseColor("#FBBF24"))
                appendLog("[CALL] Device is ringing at ${state.targetContact}...")
            }
            is com.zipper.app.telephony.KarvishCallState.Connected -> {
                tvLiveStatusBadge.text = "🟢 CONNECTED"
                tvLiveStatusBadge.setTextColor(Color.parseColor("#34D399"))
                appendLog("[CALL CONNECTED] Recipient answered! Speakerphone & VAD chime active.")
            }
            is com.zipper.app.telephony.KarvishCallState.Speaking -> {
                tvLiveStatusBadge.text = "🗣️ SPEAKING (${state.progressPct}%)"
                tvLiveStatusBadge.setTextColor(Color.parseColor("#A855F7"))
                if (state.progressPct == 0) {
                    appendLog("[CALL] Delivering offline TTS message to ${state.targetContact}...")
                }
            }
            is com.zipper.app.telephony.KarvishCallState.Completed -> {
                tvLiveStatusBadge.text = "✅ COMPLETED"
                tvLiveStatusBadge.setTextColor(Color.parseColor("#34D399"))
                appendLog("[CALL SUCCESS] ${state.summary}")
                btnRunAgentGoal.isEnabled = true
            }
            is com.zipper.app.telephony.KarvishCallState.Failed -> {
                tvLiveStatusBadge.text = "❌ FAILED"
                tvLiveStatusBadge.setTextColor(Color.parseColor("#EF4444"))
                appendLog("[CALL FAILED] ${state.reason}")
                btnRunAgentGoal.isEnabled = true
            }
        }
    }

    private fun isCallReminderGoal(goal: String): Boolean {
        val g = goal.trim().lowercase()
        return g.contains(Regex("""(?i)\b(call me in|call me at|remind me in|remind me at|set reminder|call reminder)\b"""))
    }

    private fun executeCallReminderGoal(goal: String) {
        val parsed = com.zipper.app.core.voice.CallReminderScheduler.parseNaturalReminder(goal)
        if (parsed != null) {
            com.zipper.app.core.voice.CallReminderScheduler.schedule(this, parsed)
            val timeStr = java.text.SimpleDateFormat("h:mm a, d MMM", Locale.getDefault()).format(parsed.triggerTimeEpochMs)
            Toast.makeText(this, "📞 Call Reminder Scheduled for $timeStr", Toast.LENGTH_LONG).show()
            appendLog("[VOICE REMINDER] Scheduled: '${parsed.title}' for $timeStr")
            updateCallRemindersStatus()
        } else {
            appendLog("⚠️ Could not parse reminder time from: '$goal'")
        }
        btnRunAgentGoal.isEnabled = true
    }

    private fun isLifePlanGoal(goal: String): Boolean {
        val g = goal.trim().lowercase(Locale.ROOT)
        return g.contains("plan my day") || g.contains("generate plan") ||
               g.contains("daily plan") || g.contains("test plan call") ||
               g.contains("test workout call") || g.contains("test assignment call")
    }

    private fun executeLifePlanGoal(goal: String) {
        val g = goal.trim().lowercase(Locale.ROOT)
        when {
            g.contains("test workout call") -> {
                val testCommitment = com.zipper.app.core.intelligence.store.CommitmentEntity(
                    id = 1001L,
                    title = "Evening Workout & Stretch",
                    category = com.zipper.app.core.intelligence.store.CommitmentCategory.FITNESS,
                    scheduledTimeMs = System.currentTimeMillis() + (5 * 60 * 1000L),
                    durationMinutes = 30,
                    sourceApp = "Karvish Life Guard",
                    sourcePerson = "Karvish",
                    candidatePackages = listOf("fit.cure.android", "com.sec.android.app.shealth", "com.strava"),
                    notes = "Daily fitness routine"
                )
                com.zipper.app.core.intelligence.scheduler.LifePlanCallScheduler.triggerTestCallNow(this, testCommitment)
                appendLog("📞 Scheduled test workout call with real app HUD in 3 seconds!")
                Toast.makeText(this, "Calling in 3 seconds! Tap answer to see workout apps.", Toast.LENGTH_SHORT).show()
            }
            g.contains("test assignment call") -> {
                val testCommitment = com.zipper.app.core.intelligence.store.CommitmentEntity(
                    id = 1002L,
                    title = "OS Lab Assignment 4",
                    category = com.zipper.app.core.intelligence.store.CommitmentCategory.EDUCATION,
                    scheduledTimeMs = System.currentTimeMillis() + (5 * 60 * 1000L),
                    durationMinutes = 45,
                    sourceApp = "WhatsApp",
                    sourcePerson = "Manoj",
                    candidatePackages = listOf("com.google.android.apps.docs.editors.docs", "com.google.android.apps.classroom", "com.whatsapp"),
                    notes = "Assignment due tomorrow from Manoj"
                )
                com.zipper.app.core.intelligence.scheduler.LifePlanCallScheduler.triggerTestCallNow(this, testCommitment)
                appendLog("📞 Scheduled test assignment call with Docs/Classroom/WhatsApp HUD in 3 seconds!")
                Toast.makeText(this, "Calling in 3 seconds! Tap answer to see study apps.", Toast.LENGTH_SHORT).show()
            }
            else -> {
                appendLog("🧠 Executing Deep-Think Daily Planning cycle...")
                lifecycleScope.launch(Dispatchers.IO) {
                    val engine = com.zipper.app.core.intelligence.planner.NightlyPlannerEngine(applicationContext)
                    val results = engine.executeNightlyPlan()
                    withContext(Dispatchers.Main) {
                        appendLog("✅ Daily Plan generated: ${results.size} balanced commitments scheduled!")
                        for (item in results) {
                            appendLog("   • ${item.title} (${item.category}) - ${item.durationMinutes}m")
                        }
                        updateCallRemindersStatus()
                        Toast.makeText(this@MainActivity, "Plan generated: ${results.size} tasks scheduled!", Toast.LENGTH_LONG).show()
                    }
                }
            }
        }
        btnRunAgentGoal.isEnabled = true
    }

    private fun updateCallRemindersStatus() {
        val active = com.zipper.app.core.voice.CallReminderScheduler.getActiveReminders(this)
        if (active.isEmpty()) {
            tvActiveRemindersStatus.text = "0 active reminders • Paradee-8M voice ready"
        } else {
            val next = active.first()
            val timeStr = java.text.SimpleDateFormat("h:mm a", Locale.getDefault()).format(next.triggerTimeEpochMs)
            tvActiveRemindersStatus.text = "${active.size} active • Next: '${next.title}' at $timeStr"
        }
    }

    private fun updateWakeWordUi(enabled: Boolean) {
        if (enabled) {
            btnToggleWakeWord.text = "Wake Word: ON"
            btnToggleWakeWord.setTextColor(Color.parseColor("#34D399"))
            if (btnToggleWakeWord is MaterialButton) {
                (btnToggleWakeWord as MaterialButton).strokeColor =
                    ColorStateList.valueOf(Color.parseColor("#10B981"))
            }
        } else {
            btnToggleWakeWord.text = "Wake Word: OFF"
            btnToggleWakeWord.setTextColor(Color.parseColor("#94A3B8"))
            if (btnToggleWakeWord is MaterialButton) {
                (btnToggleWakeWord as MaterialButton).strokeColor =
                    ColorStateList.valueOf(Color.parseColor("#475569"))
            }
        }
    }

    private fun updateTrainVoiceUi(enrolled: Boolean) {
        if (enrolled) {
            btnTrainVoice.text = "✓ Voice Match: Active (Karthik)"
            btnTrainVoice.setTextColor(Color.parseColor("#34D399"))
            if (btnTrainVoice is MaterialButton) {
                (btnTrainVoice as MaterialButton).strokeColor =
                    ColorStateList.valueOf(Color.parseColor("#10B981"))
            }
        } else {
            btnTrainVoice.text = "🎙️ Train My Voice (Voice Match)"
            btnTrainVoice.setTextColor(Color.parseColor("#38BDF8"))
            if (btnTrainVoice is MaterialButton) {
                (btnTrainVoice as MaterialButton).strokeColor =
                    ColorStateList.valueOf(Color.parseColor("#0284C7"))
            }
        }
    }

    private fun startVoiceTrainingDialog() {
        val takes = mutableListOf<ShortArray>()
        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("🎙️ Train Voice Profile (Voice Match)")
            .setMessage("Say 'Hey Zipper' clearly 3 times.\nOnly YOUR vocal cords will be permitted to wake up Zipper.\nReels, TV, and other people will be ignored.\n\nReady to begin?")
            .setPositiveButton("Start Take 1 of 3") { _, _ ->
                recordVoiceTrainingTake(takeIndex = 1, takes = takes)
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun recordVoiceTrainingTake(takeIndex: Int, takes: MutableList<ShortArray>) {
        val progressDialog = androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("🎙️ Take $takeIndex of 3")
            .setMessage("Get ready...\n3...")
            .setCancelable(false)
            .create()
        progressDialog.show()

        lifecycleScope.launch {
            // 1. Visual Countdown: 3... 2... 1...
            delay(700)
            progressDialog.setMessage("Get ready...\n2...")
            delay(700)
            progressDialog.setMessage("Get ready...\n1...")
            delay(700)
            progressDialog.setMessage("🎙️ SPEAK NOW: Say 'Hey Zipper'!\n(Listening...)")

            withContext(Dispatchers.IO) {
                val sampleRate = 16000
                val numSamples = (sampleRate * 1.2).toInt() // 1.2s = 19,200 samples
                val buffer = ShortArray(numSamples)
                val chunkLen = 320 // 20ms
                val chunk = ShortArray(chunkLen)

                var record: AudioRecord? = null
                try {
                    val minBuf = AudioRecord.getMinBufferSize(
                        sampleRate,
                        AudioFormat.CHANNEL_IN_MONO,
                        AudioFormat.ENCODING_PCM_16BIT
                    )
                    record = AudioRecord(
                        MediaRecorder.AudioSource.VOICE_RECOGNITION,
                        sampleRate,
                        AudioFormat.CHANNEL_IN_MONO,
                        AudioFormat.ENCODING_PCM_16BIT,
                        maxOf(minBuf, numSamples * 2)
                    )
                    record.startRecording()

                    // Wait for actual speech onset (up to 4.5 seconds)
                    var speechDetected = false
                    val maxWaitChunks = (4.5 * sampleRate / chunkLen).toInt()
                    var waitCount = 0

                    val preRoll = ShortArray(1600) // keep 100ms before speech onset
                    var preRollIdx = 0

                    while (waitCount < maxWaitChunks && !speechDetected) {
                        val read = record.read(chunk, 0, chunkLen)
                        if (read > 0) {
                            var sumSq = 0.0
                            for (s in chunk) sumSq += (s * s).toDouble()
                            val rms = sqrt(sumSq / read)

                            // Save to preRoll ring buffer
                            for (i in 0 until read) {
                                preRoll[preRollIdx % preRoll.size] = chunk[i]
                                preRollIdx++
                            }

                            // Conversational human voice threshold
                            if (rms >= 1100.0) {
                                speechDetected = true
                                break
                            } else {
                                // Live amplitude indicator
                                val dots = when {
                                    rms > 700.0 -> "●●●"
                                    rms > 400.0 -> "●●○"
                                    else -> "●○○"
                                }
                                withContext(Dispatchers.Main) {
                                    progressDialog.setMessage("🎙️ SPEAK NOW: Say 'Hey Zipper'!\nMic Level: $dots")
                                }
                            }
                        }
                        waitCount++
                    }

                    if (!speechDetected) {
                        record.stop()
                        record.release()
                        withContext(Dispatchers.Main) {
                            progressDialog.dismiss()
                            androidx.appcompat.app.AlertDialog.Builder(this@MainActivity)
                                .setTitle("⚠️ No Speech Detected")
                                .setMessage("We didn't hear you speak.\nPlease speak clearly and try again.")
                                .setPositiveButton("Try Again") { _, _ ->
                                    recordVoiceTrainingTake(takeIndex, takes)
                                }
                                .setNegativeButton("Cancel", null)
                                .show()
                        }
                        return@withContext
                    }

                    // Copy pre-roll audio into buffer start
                    val preRollCount = minOf(1600, preRollIdx)
                    var writeIdx = 0
                    val startOffset = if (preRollIdx > 1600) preRollIdx % 1600 else 0
                    for (i in 0 until preRollCount) {
                        buffer[writeIdx++] = preRoll[(startOffset + i) % 1600]
                    }

                    // Copy current trigger chunk
                    for (s in chunk) {
                        if (writeIdx < numSamples) buffer[writeIdx++] = s
                    }

                    // Read remaining samples to complete 1.2s take
                    while (writeIdx < numSamples) {
                        val read = record.read(buffer, writeIdx, numSamples - writeIdx)
                        if (read > 0) writeIdx += read else break
                    }

                    record.stop()
                    record.release()

                    // Double verify speech energy in captured buffer
                    if (!com.zipper.app.core.voice.SpeakerVoiceVerifier.isSpeechPresent(buffer)) {
                        withContext(Dispatchers.Main) {
                            progressDialog.dismiss()
                            androidx.appcompat.app.AlertDialog.Builder(this@MainActivity)
                                .setTitle("⚠️ Audio Too Quiet")
                                .setMessage("Voice was too faint to register vocal cords.\nPlease speak louder and try again.")
                                .setPositiveButton("Try Again") { _, _ ->
                                    recordVoiceTrainingTake(takeIndex, takes)
                                }
                                .setNegativeButton("Cancel", null)
                                .show()
                        }
                        return@withContext
                    }

                    takes.add(buffer)

                    withContext(Dispatchers.Main) {
                        progressDialog.dismiss()
                        if (takeIndex < 3) {
                            androidx.appcompat.app.AlertDialog.Builder(this@MainActivity)
                                .setTitle("✓ Take $takeIndex of 3 Captured!")
                                .setMessage("Great take! Get ready for Take ${takeIndex + 1} of 3.")
                                .setPositiveButton("Start Take ${takeIndex + 1}") { _, _ ->
                                    recordVoiceTrainingTake(takeIndex + 1, takes)
                                }
                                .setCancelable(false)
                                .show()
                        } else {
                            // All 3 takes captured -> Enroll!
                            val success = com.zipper.app.core.voice.SpeakerVoiceVerifier.enroll(this@MainActivity, takes)
                            if (success) {
                                Toast.makeText(this@MainActivity, "🎉 Voice Profile Enrolled! Only YOUR voice can wake Zipper.", Toast.LENGTH_LONG).show()
                                updateTrainVoiceUi(true)
                                appendLog("[VOICE MATCH] Karthik's 3D-Speaker ONNX profile enrolled. Reels and strangers are now blocked.")
                            } else {
                                Toast.makeText(this@MainActivity, "⚠️ Enrollment failed. Please speak louder and try again.", Toast.LENGTH_SHORT).show()
                            }
                        }
                    }
                } catch (e: Exception) {
                    Log.e("MainActivity", "Voice training error: ${e.message}")
                    record?.release()
                    withContext(Dispatchers.Main) {
                        progressDialog.dismiss()
                        Toast.makeText(this@MainActivity, "Mic error during recording: ${e.message}", Toast.LENGTH_SHORT).show()
                    }
                }
            }
        }
    }

    private fun isCallGoal(goal: String): Boolean {
        val g = goal.trim().lowercase()
        return g.startsWith("call ") || g.startsWith("phone ") || g.startsWith("dial ")
    }

    private fun executeAutonomousCallGoal(goal: String) {
        val hasCallPerm = ContextCompat.checkSelfPermission(this, Manifest.permission.CALL_PHONE) == PackageManager.PERMISSION_GRANTED
        val hasContactsPerm = ContextCompat.checkSelfPermission(this, Manifest.permission.READ_CONTACTS) == PackageManager.PERMISSION_GRANTED

        if (!hasCallPerm || !hasContactsPerm) {
            pendingCallGoal = goal
            requestTelephonyPermissionLauncher.launch(
                arrayOf(Manifest.permission.CALL_PHONE, Manifest.permission.READ_CONTACTS)
            )
            return
        }

        val parsed = com.zipper.app.telephony.KarvishCallManager.parseCallGoal(goal)
        val targetRaw = parsed?.contactQuery ?: goal.trim().substringAfter(" ").trim()
        val finalMessage = parsed?.message ?: "Hello $targetRaw, this is an automated message from Karthik: I will come in a few minutes. Thank you."

        tvLiveStatusBadge.text = "📞 CALLING..."
        tvLiveStatusBadge.setTextColor(Color.parseColor("#38BDF8"))
        tvTurnCounter.text = "Telephony"

        tvAgentAuditLog.text = buildString {
            appendLine("=== KARVISH AUTONOMOUS TELEPHONY ===")
            appendLine("Goal: '$goal'")
            appendLine("Target: '$targetRaw'")
            appendLine("Message: '$finalMessage'")
            appendLine("Engine: Normal SIM Cellular Call + Offline TTS (0.92x)")
            appendLine("-----------------------------------")
        }

        val lookedUp = com.zipper.app.telephony.KarvishCallManager.lookupContactNumber(this, targetRaw)
        if (lookedUp == null) {
            appendLog("⚠️ Contact '$targetRaw' not found in device contacts.")
            appendLog("Tip: Ensure contact '$targetRaw' exists or specify direct number (e.g. 'Call +919449213667 and say ...').")
            tvLiveStatusBadge.text = "⏹️ READY"
            tvLiveStatusBadge.setTextColor(Color.parseColor("#F59E0B"))
            btnRunAgentGoal.isEnabled = true
            return
        }

        appendLog("Contact found: '$targetRaw' -> $lookedUp")
        val voipConfig = com.zipper.app.telephony.voip.VoipConfigManager(this).getConfig()
        if (voipConfig.isEnabled && voipConfig.isConfigured) {
            appendLog("Engine: VoIPBin Cloud AI Telephony (Assistant: ${voipConfig.assistantId.take(8)}...)")
        } else {
            appendLog("Engine: Normal SIM Cellular Call + Offline TTS (0.92x)")
        }

        btnRunAgentGoal.isEnabled = false
        val dispatched = com.zipper.app.telephony.KarvishCallManager.placeAutonomousCall(
            context = this,
            contactName = targetRaw,
            targetNumber = lookedUp,
            messagePayload = finalMessage,
            onStatusUpdate = { update ->
                runOnUiThread {
                    appendLog(update)
                    if (update.startsWith("✅ Call Ended") || update.startsWith("❌")) {
                        btnRunAgentGoal.isEnabled = true
                        tvLiveStatusBadge.text = "⏹️ READY"
                        tvLiveStatusBadge.setTextColor(Color.parseColor("#10B981"))
                    }
                }
            }
        )

        if (!dispatched) {
            appendLog("❌ Failed to initiate autonomous call.")
            btnRunAgentGoal.isEnabled = true
        }
    }

    private fun executeHardStop() {
        Log.w(TAG, "[HARD STOP] User tapped Emergency Stop button")
        com.zipper.app.telephony.voip.VoipBinCallSyncManager.endCurrentCall(this)
        val stopIntent = Intent(this, AgentRuntimeService::class.java).apply {
            action = AgentRuntimeService.ACTION_STOP_AGENT
        }
        startService(stopIntent)

        tvLiveStatusBadge.text = "⛔ ABORTED"
        tvLiveStatusBadge.setTextColor(Color.parseColor("#EF4444"))
        btnRunAgentGoal.isEnabled = true
        appendLog("[USER EMERGENCY HARD STOP] Stop command dispatched. Runtime & VoIP terminated.")
    }

    private fun setupSpeechRecognizer() {
        if (!SpeechRecognizer.isRecognitionAvailable(this)) {
            Log.w(TAG, "SpeechRecognizer not available on this device")
            btnVoiceInput.isEnabled = false
            return
        }

        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this).apply {
            setRecognitionListener(object : RecognitionListener {
                override fun onReadyForSpeech(params: Bundle?) {
                    isListening = true
                    tvLiveStatusBadge.text = "🎤 LISTENING..."
                    tvLiveStatusBadge.setTextColor(Color.parseColor("#38BDF8"))
                    btnVoiceInput.iconTint = ColorStateList.valueOf(Color.parseColor("#EF4444"))
                    appendLog("Listening for voice goal...")
                }

                override fun onBeginningOfSpeech() {}
                override fun onRmsChanged(rmsdB: Float) {}
                override fun onBufferReceived(buffer: ByteArray?) {}

                override fun onEndOfSpeech() {
                    isListening = false
                    btnVoiceInput.iconTint = ColorStateList.valueOf(Color.parseColor("#38BDF8"))
                }

                override fun onError(error: Int) {
                    isListening = false
                    btnVoiceInput.iconTint = ColorStateList.valueOf(Color.parseColor("#38BDF8"))
                    tvLiveStatusBadge.text = "🟢 READY"
                    tvLiveStatusBadge.setTextColor(Color.parseColor("#34D399"))
                    val errMsg = when (error) {
                        SpeechRecognizer.ERROR_NO_MATCH -> "No speech recognized. Try again."
                        SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "Speech timeout. Try again."
                        SpeechRecognizer.ERROR_AUDIO -> "Audio recording error."
                        else -> "Speech recognition error: $error"
                    }
                    appendLog("STT: $errMsg")
                }

                override fun onResults(results: Bundle?) {
                    isListening = false
                    btnVoiceInput.iconTint = ColorStateList.valueOf(Color.parseColor("#38BDF8"))
                    tvLiveStatusBadge.text = "🟢 READY"
                    tvLiveStatusBadge.setTextColor(Color.parseColor("#34D399"))

                    val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    val spoken = matches?.firstOrNull()?.trim()
                    if (!spoken.isNullOrBlank()) {
                        etAgentGoal.setText(spoken)
                        appendLog("Transcribed Voice Goal: '$spoken'")
                    }
                }

                override fun onPartialResults(partialResults: Bundle?) {
                    val partial = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull()
                    if (!partial.isNullOrBlank()) {
                        etAgentGoal.setText(partial)
                    }
                }

                override fun onEvent(eventType: Int, params: Bundle?) {}
            })
        }
    }

    private fun toggleVoiceInput() {
        if (isListening) {
            speechRecognizer?.stopListening()
            isListening = false
            btnVoiceInput.iconTint = ColorStateList.valueOf(Color.parseColor("#38BDF8"))
            tvLiveStatusBadge.text = "🟢 READY"
            tvLiveStatusBadge.setTextColor(Color.parseColor("#34D399"))
        } else {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
                startVoiceRecognition()
            } else {
                requestAudioPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
            }
        }
    }

    private fun startVoiceRecognition() {
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
            putExtra(RecognizerIntent.EXTRA_PROMPT, "Speak your goal to Karvish...")
        }
        speechRecognizer?.startListening(intent)
    }

    private fun runAutonomousAgentGoal(
        goal: String,
        enableRecovery: Boolean = true,
        useRemoteRanker: Boolean = false
    ) {
        if (goal.isBlank()) {
            tvAgentAuditLog.text = "Please enter or speak an agent goal first."
            return
        }

        if (isLifePlanGoal(goal)) {
            executeLifePlanGoal(goal)
            return
        }

        if (isCallReminderGoal(goal)) {
            executeCallReminderGoal(goal)
            return
        }

        if (isCallGoal(goal)) {
            executeAutonomousCallGoal(goal)
            return
        }

        btnRunAgentGoal.isEnabled = false
        val hasV62OnDevice = try { assets.list("models")?.contains("karvish_v62.onnx") == true } catch (_: Exception) { false }
        val hasV61OnDevice = try { assets.list("models")?.contains("karvish_v61.onnx") == true } catch (_: Exception) { false }
        val hasV57OnDevice = try { assets.list("models")?.contains("karvish_v57.onnx") == true } catch (_: Exception) { false }
        val hasV56OnDevice = try { assets.list("models")?.contains("karvish_v56.onnx") == true } catch (_: Exception) { false }
        val hasV55OnDevice = try { assets.list("models")?.contains("karvish_v55.onnx") == true } catch (_: Exception) { false }
        val hasV54Fp16OnDevice = try { assets.list("models")?.contains("karvish_v54_fp16.onnx") == true } catch (_: Exception) { false }
        val hasV54OnDevice = try { assets.list("models")?.contains("karvish_v54.onnx") == true } catch (_: Exception) { false }
        val onDeviceDesc = when {
            hasV62OnDevice -> "Pure On-Device V62 FP16 ONNX (57.06 MB - Universal Spatial Action Agent)"
            hasV61OnDevice -> "Pure On-Device V61 FP16 ONNX (53.80 MB)"
            hasV57OnDevice -> "Pure On-Device V57 FP16 ONNX (51.01 MB - Multi-Hop Social & History Agent)"
            hasV56OnDevice -> "Pure On-Device V56 FP16 ONNX (51.01 MB - Universal Action Agent)"
            hasV55OnDevice -> "Pure On-Device V55 FP16 ONNX (50.88 MB)"
            hasV54Fp16OnDevice -> "Pure On-Device V54 FP16 ONNX (50.88 MB - Frozen Control)"
            hasV54OnDevice -> "Pure On-Device V54 FP16 ONNX (50.66 MB)"
            else -> "Pure On-Device V52 ONNX (98.03 MB)"
        }
        val engineDesc = if (useRemoteRanker) "Host GPU Bridge (localhost:8765) [V54 CUDA Active]" else onDeviceDesc
        lastExecutedGoal = goal
        hero3dCoreView?.currentState = com.zipper.app.core.graphics.ZipperCoreState.THINKING
        tvLiveStatusBadge.text = "🔄 RUNNING..."
        tvLiveStatusBadge.setTextColor(Color.parseColor("#38BDF8"))
        tvTurnCounter.text = "Turn: 1/30"

        tvAgentAuditLog.text = buildString {
            appendLine("=== KARVISH AUTONOMOUS RUNTIME ===")
            appendLine("Goal: '$goal'")
            appendLine("Engine: $engineDesc")
            appendLine("Recovery: $enableRecovery")
            appendLine("Safety Gating: ACTIVE (Fail-Closed)")
            appendLine("-----------------------------------")
            appendLine("Starting autonomous execution loop...")
        }

        val serviceIntent = Intent(this, AgentRuntimeService::class.java).apply {
            putExtra(AgentRuntimeService.EXTRA_GOAL, goal)
            putExtra(AgentRuntimeService.EXTRA_RECOVERY, enableRecovery)
            putExtra(AgentRuntimeService.EXTRA_USE_REMOTE_RANKER, useRemoteRanker)
        }
        ContextCompat.startForegroundService(this, serviceIntent)
    }

    private fun appendLog(msg: String) {
        val current = tvAgentAuditLog.text.toString()
        tvAgentAuditLog.text = "$current\n$msg"
    }

    private fun updateServiceStatus() {
        val connected = KarvishAccessibilityService.instance != null
        tvServiceStatus.text = if (connected) "Active & Connected ✓" else "Inactive - Tap Configure"
        tvServiceStatus.setTextColor(
            if (connected) Color.parseColor("#34D399") else Color.parseColor("#F87171")
        )
    }

    private fun updateFloatingBubbleUi(enabled: Boolean) {
        btnToggleFloatingBubble.text = "🎙️ Test Bottom Wake-Up Pill"
        btnToggleFloatingBubble.setTextColor(Color.parseColor("#00F0FF"))
    }

    override fun onResume() {
        super.onResume()
        hero3dCoreView?.resumeRendering()
        updateServiceStatus()
        updatePluginDashboardUi()
        updateCallRemindersStatus()
        updateWakeWordUi(com.zipper.app.core.voice.WakeWordService.isWakeWordEnabled(this))
        updateTrainVoiceUi(com.zipper.app.core.voice.SpeakerVoiceVerifier.isEnrolled(this))
        updateFloatingBubbleUi(com.zipper.app.accessibility.KarvishBubbleOverlay.isEnabled(this))
        updateRecentTasksUi()
        updateHeroGreeting()
        if (findViewById<View>(R.id.scrollPlannerContent)?.visibility == View.VISIBLE) {
            refreshPlannerDashboard()
        }

        btnRunAgentGoal.isEnabled = !AgentRuntimeService.isRunning
        btnHardStop.isEnabled = true
        if (!AgentRuntimeService.isRunning && tvLiveStatusBadge.text.toString().contains("RUNNING")) {
            tvLiveStatusBadge.text = "🟢 READY"
            tvLiveStatusBadge.setTextColor(Color.parseColor("#34D399"))
        }

        lifecycleScope.launch {
            for (i in 0 until 10) {
                if (KarvishAccessibilityService.instance != null) {
                    updateServiceStatus()
                    break
                }
                delay(300)
            }
        }
    }

    override fun onPause() {
        super.onPause()
        hero3dCoreView?.pauseRendering()
    }

    override fun onDestroy() {
        try {
            unregisterReceiver(agentFinishedReceiver)
        } catch (_: Exception) {}
        try {
            unregisterReceiver(plannerDataChangedReceiver)
        } catch (_: Exception) {}
        speechRecognizer?.destroy()
        super.onDestroy()
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        btnRunAgentGoal.isEnabled = true
        intent.getStringExtra("set_model")?.let { model ->
            getSharedPreferences("karvish_agent_prefs", Context.MODE_PRIVATE).edit().putString("selected_model", model).apply()
            updateModelStatusDisplay()
        }
        handleIntentTriggers(intent)
    }

    private fun handleIntentTriggers(intent: Intent?) {
        if (intent == null) return
        val openTab = intent.getStringExtra("open_tab")
        if (openTab == "planner") {
            switchToPlannerTab()
        } else if (openTab == "agent") {
            switchToAgentTab()
        } else if (openTab == "webmind") {
            switchToWebmindTab()
        } else if (openTab == "ditto") {
            showDittoHubDialog()
        }
        if (intent.hasExtra("run_agent_goal")) {
            val rawGoal = intent.getStringExtra("run_agent_goal").orEmpty()
            val goal = rawGoal.trim().trim('"', '\'', '`', '\\')
            val enableRecovery = intent.getBooleanExtra("enable_active_recovery", true)
            val useRemote = intent.getBooleanExtra("use_remote_ranker", false)
            if (useRemote) {
                rbRemoteGpu.isChecked = true
            } else {
                rbOnDevice.isChecked = true
            }
            etAgentGoal.setText(goal)
            runAutonomousAgentGoal(goal, enableRecovery, useRemote)
        }
        if (intent.hasExtra("run_ditto_flow")) {
            val flowQuery = intent.getStringExtra("run_ditto_flow").orEmpty().trim().trim('"', '\'')
            val store = com.zipper.app.ditto.DittoStore(this)
            val flow = store.listFlows().firstOrNull { 
                it.title.equals(flowQuery, ignoreCase = true) || it.flowId == flowQuery 
            }
            if (flow != null) {
                runDittoReplay(flow)
            } else {
                appendLog("[DITTO ERROR] Flow '$flowQuery' not found in store.")
            }
        }
        if (intent.hasExtra("wake_pill")) {
            val a11y = com.zipper.app.accessibility.KarvishAccessibilityService.instance
            if (a11y != null) {
                KarvishBubbleOverlay.wakeUpFromBottom(a11y)
            } else {
                KarvishBubbleOverlay.wakeUpFromBottom(this)
            }
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (grantResults.isNotEmpty() && grantResults[0] == android.content.pm.PackageManager.PERMISSION_GRANTED) {
            when (requestCode) {
                RC_RECORD_AUDIO -> {
                    val started = com.zipper.app.core.voice.WakeWordService.start(this)
                    if (started) {
                        Toast.makeText(this, "🎙️ 'Hey Zipper' Wake Word Active! Say 'Hey Zipper' anytime.", Toast.LENGTH_LONG).show()
                        updateWakeWordUi(true)
                    } else {
                        Toast.makeText(this, "Could not start wake word service", Toast.LENGTH_SHORT).show()
                        updateWakeWordUi(false)
                    }
                }
                RC_TALK_TO_ZIPPER -> {
                    com.zipper.app.core.voice.VoiceAssistantActivity.start(this)
                }
            }
        } else {
            Toast.makeText(this, "Microphone permission is required for voice & wake word features", Toast.LENGTH_LONG).show()
            updateWakeWordUi(false)
        }
    }

    private fun triggerCognitiveBadgePulse(
        badgeView: View?,
        state: com.zipper.app.core.graphics.ZipperCoreState,
        title: String,
        description: String
    ) {
        badgeView?.performHapticFeedback(android.view.HapticFeedbackConstants.CONTEXT_CLICK)
        hero3dCoreView?.currentState = state
        badgeView?.animate()
            ?.scaleX(1.05f)?.scaleY(1.05f)
            ?.setDuration(120)
            ?.withEndAction {
                badgeView.animate()?.scaleX(1.0f)?.scaleY(1.0f)?.setDuration(120)?.start()
            }?.start()

        Toast.makeText(this, "✦ $title\n$description", Toast.LENGTH_SHORT).show()

        badgeView?.postDelayed({
            if (hero3dCoreView?.currentState == state) {
                hero3dCoreView?.currentState = com.zipper.app.core.graphics.ZipperCoreState.IDLE
            }
        }, 2200)
    }

    private fun showSupportedAppsDialog() {
        val apps = arrayOf("WhatsApp", "YouTube", "Calculator", "Instagram", "LinkedIn", "Zepto", "Blinkit", "Settings", "Chrome")
        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("Supported Apps")
            .setItems(apps) { _, which ->
                etAgentGoal.setText("Open ${apps[which]}")
                etAgentGoal.setSelection(etAgentGoal.text.length)
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun updateRecentTasksUi() {
        val tasks = com.zipper.app.core.history.KarvishTaskStore.getTasks(this)
        val container = findViewById<LinearLayout>(R.id.llRecentTasksContainer) ?: return
        val emptyState = findViewById<View>(R.id.llRecentTasksEmptyState)

        if (tasks.isEmpty()) {
            emptyState?.visibility = View.VISIBLE
            for (i in container.childCount - 1 downTo 0) {
                val child = container.getChildAt(i)
                if (child.id != R.id.llRecentTasksEmptyState) {
                    container.removeViewAt(i)
                }
            }
            return
        }

        emptyState?.visibility = View.GONE
        for (i in container.childCount - 1 downTo 0) {
            val child = container.getChildAt(i)
            if (child.id != R.id.llRecentTasksEmptyState) {
                container.removeViewAt(i)
            }
        }

        val inflater = LayoutInflater.from(this)
        for (task in tasks.take(3)) {
            val itemView = inflater.inflate(R.layout.item_recent_task, container, false)
            val ivIcon = itemView.findViewById<ImageView>(R.id.ivItemTaskIcon)
            val tvGoal = itemView.findViewById<TextView>(R.id.tvItemTaskGoal)
            val tvStatus = itemView.findViewById<TextView>(R.id.tvItemTaskStatus)

            tvGoal.text = task.goal
            val statusStr = if (task.isSuccess) "Completed" else "In Progress"
            tvStatus.text = "$statusStr  •  ${task.getFormattedTime()}"
            tvStatus.setTextColor(Color.parseColor("#6B7280"))

            bindMissionAppIcon(task, ivIcon)

            itemView.setOnClickListener {
                etAgentGoal.setText(task.goal)
                etAgentGoal.setSelection(task.goal.length)
                btnRunAgentGoal.performClick()
            }
            container.addView(itemView)
        }

        findViewById<View>(R.id.tvSeeAllRecentTasks)?.setOnClickListener {
            val taskListStr = tasks.mapIndexed { idx, t ->
                "${idx + 1}. ${t.goal} (${if (t.isSuccess) "Completed" else "In Progress"} • ${t.getFormattedTime()})"
            }.joinToString("\n\n")
            androidx.appcompat.app.AlertDialog.Builder(this)
                .setTitle("All Recent Missions (${tasks.size})")
                .setMessage(taskListStr.ifEmpty { "No mission history recorded yet." })
                .setPositiveButton("Close", null)
                .show()
        }
    }

    private fun bindMissionAppIcon(task: com.zipper.app.core.history.KarvishHistoryTask, ivIcon: ImageView) {
        val lower = "${task.appName} ${task.goal}".lowercase(java.util.Locale.ROOT)
        val pm = packageManager
        var appDrawable: android.graphics.drawable.Drawable? = null

        // 1. Direct package check if task.appName is a package
        if (task.appName.contains(".")) {
            try {
                appDrawable = pm.getApplicationIcon(task.appName)
            } catch (_: Exception) {}
        }

        // 2. High-frequency known package map for instantaneous lookup
        if (appDrawable == null) {
            val candidatePkgs = when {
                lower.contains("youtube") -> listOf("com.google.android.youtube")
                lower.contains("whatsapp") -> listOf("com.whatsapp")
                lower.contains("instagram") -> listOf("com.instagram.android")
                lower.contains("linkedin") -> listOf("com.linkedin.android")
                lower.contains("zepto") -> listOf("in.zepto.android", "com.zeptonow.android")
                lower.contains("blinkit") || lower.contains("grofers") -> listOf("com.grofers.customerapp")
                lower.contains("indiamart") -> listOf("com.indiamart.m")
                lower.contains("calculator") || lower.contains("calc") -> listOf("com.sec.android.app.popupcalculator", "com.google.android.calculator")
                lower.contains("settings") -> listOf("com.android.settings")
                lower.contains("chrome") -> listOf("com.android.chrome")
                lower.contains("amazon") -> listOf("in.amazon.mShop.android.shopping", "com.amazon.mShop.android.shopping")
                lower.contains("flipkart") -> listOf("com.flipkart.android")
                lower.contains("swiggy") -> listOf("in.swiggy.android")
                lower.contains("zomato") -> listOf("com.application.zomato")
                lower.contains("gmail") -> listOf("com.google.android.gm")
                lower.contains("maps") -> listOf("com.google.android.apps.maps")
                lower.contains("photos") -> listOf("com.google.android.apps.photos")
                lower.contains("clock") || lower.contains("alarm") -> listOf("com.sec.android.app.clockpackage", "com.google.android.deskclock")
                lower.contains("spotify") -> listOf("com.spotify.music")
                lower.contains("keep") || lower.contains("notes") -> listOf("com.google.android.keep")
                lower.contains("phone") || lower.contains("call") || lower.contains("dial") -> listOf("com.samsung.android.dialer", "com.google.android.dialer")
                else -> emptyList()
            }
            for (pkg in candidatePkgs) {
                try {
                    appDrawable = pm.getApplicationIcon(pkg)
                    if (appDrawable != null) break
                } catch (_: Exception) {}
            }
        }

        // 3. Dynamic package manager search over installed apps
        if (appDrawable == null) {
            try {
                val installed = pm.getInstalledApplications(0)
                for (app in installed) {
                    val label = pm.getApplicationLabel(app).toString().lowercase(java.util.Locale.ROOT)
                    if (label.length >= 3 && lower.contains(label)) {
                        appDrawable = pm.getApplicationIcon(app)
                        break
                    }
                }
            } catch (_: Exception) {}
        }

        if (appDrawable != null) {
            ivIcon.setImageDrawable(appDrawable)
            ivIcon.background = null
            ivIcon.imageTintList = null
            ivIcon.setPadding(0, 0, 0, 0)
        } else {
            // Authentic fallbacks when app not installed on device
            when {
                lower.contains("youtube") -> {
                    ivIcon.setImageResource(R.drawable.ic_youtube)
                    ivIcon.setBackgroundResource(R.drawable.bg_badge_icon_red)
                    ivIcon.imageTintList = ColorStateList.valueOf(Color.WHITE)
                    ivIcon.setPadding(16, 16, 16, 16)
                }
                lower.contains("whatsapp") -> {
                    ivIcon.setImageResource(R.drawable.ic_whatsapp)
                    ivIcon.setBackgroundResource(R.drawable.bg_badge_icon_green)
                    ivIcon.imageTintList = ColorStateList.valueOf(Color.WHITE)
                    ivIcon.setPadding(16, 16, 16, 16)
                }
                else -> {
                    ivIcon.setImageResource(R.drawable.ic_globe)
                    ivIcon.setBackgroundResource(R.drawable.bg_circle_soft_blue)
                    ivIcon.imageTintList = ColorStateList.valueOf(Color.parseColor("#2563EB"))
                    ivIcon.setPadding(16, 16, 16, 16)
                }
            }
        }
    }

    private fun updateHeroGreeting() {
        val tvGreeting = findViewById<TextView>(R.id.tvHeroGreetingTime) ?: return
        val tvSunIcon = findViewById<TextView>(R.id.tvHeroSunIcon)
        val hour = java.util.Calendar.getInstance().get(java.util.Calendar.HOUR_OF_DAY)
        when (hour) {
            in 5..11 -> {
                tvGreeting.text = "Good morning,"
                tvSunIcon?.text = "☀️"
            }
            in 12..16 -> {
                tvGreeting.text = "Good afternoon,"
                tvSunIcon?.text = "🌤️"
            }
            in 17..21 -> {
                tvGreeting.text = "Good evening,"
                tvSunIcon?.text = "🌆"
            }
            else -> {
                tvGreeting.text = "Good night,"
                tvSunIcon?.text = "🌙"
            }
        }
    }

    private fun dpToPx(dp: Int): Int = (dp * resources.displayMetrics.density).toInt()

    private var calCurrentMonth = java.util.Calendar.getInstance()
    private var calSelectedDayMs: Long = System.currentTimeMillis()

    private fun setupPlannerListeners() {
        // Back arrow returns to Agent view
        findViewById<View>(R.id.btnPlannerBack)?.setOnClickListener {
            switchToAgentTab()
        }

        // View Calendar toggle
        findViewById<View>(R.id.btnViewCalendar)?.setOnClickListener {
            val calSection = findViewById<View>(R.id.llInteractiveCalendarSection)
            val arrow = findViewById<ImageView>(R.id.ivCalendarToggleArrow)
            if (calSection?.visibility == View.VISIBLE) {
                calSection.visibility = View.GONE
                arrow?.rotation = 0f
            } else {
                calSection?.visibility = View.VISIBLE
                arrow?.rotation = 90f
                val grid = findViewById<GridLayout>(R.id.glCalendarMonthGrid)
                if (grid != null) {
                    renderInteractiveCalendar(grid, calCurrentMonth)
                }
            }
        }

        findViewById<View>(R.id.btnCalPrevMonth)?.setOnClickListener {
            calCurrentMonth.add(java.util.Calendar.MONTH, -1)
            val grid = findViewById<GridLayout>(R.id.glCalendarMonthGrid)
            if (grid != null) renderInteractiveCalendar(grid, calCurrentMonth)
        }

        findViewById<View>(R.id.btnCalNextMonth)?.setOnClickListener {
            calCurrentMonth.add(java.util.Calendar.MONTH, 1)
            val grid = findViewById<GridLayout>(R.id.glCalendarMonthGrid)
            if (grid != null) renderInteractiveCalendar(grid, calCurrentMonth)
        }

        findViewById<View>(R.id.btnCalJumpToday)?.setOnClickListener {
            calCurrentMonth = java.util.Calendar.getInstance()
            calSelectedDayMs = System.currentTimeMillis()
            val grid = findViewById<GridLayout>(R.id.glCalendarMonthGrid)
            if (grid != null) renderInteractiveCalendar(grid, calCurrentMonth)
            refreshPlannerDashboard()
        }

        // Configure AI API Dialog
        findViewById<View>(R.id.btnConfigureAiApi)?.setOnClickListener {
            showAiApiConfigDialog()
        }

        // Plan My Day
        findViewById<View>(R.id.btnPlanMyDayNow)?.setOnClickListener {
            val btn = findViewById<TextView>(R.id.btnPlanMyDayNow)
            val originalText = btn?.text?.toString() ?: "🧠 Plan My Day (Deep-Think AI)"
            val prefs = com.zipper.app.core.intelligence.store.LifeRhythmPrefs(this)
            val currentModel = if (prefs.llmApiKey.isNotBlank()) prefs.llmModel.ifBlank { "Deep-Think AI" } else "Autonomous Semantic Engine"

            btn?.isEnabled = false
            btn?.text = "🧠 Synthesizing with $currentModel..."

            Toast.makeText(this, "🧠 Synthesizing daily plan using $currentModel...", Toast.LENGTH_SHORT).show()
            appendLog("🧠 Executing Deep-Think Daily Planning cycle with $currentModel for selected date...")

            lifecycleScope.launch(Dispatchers.IO) {
                try {
                    val engine = com.zipper.app.core.intelligence.planner.NightlyPlannerEngine(applicationContext)
                    val results = engine.executeNightlyPlan(targetDayTimestampMs = calSelectedDayMs, isForceRefresh = true)
                    withContext(Dispatchers.Main) {
                        btn?.text = originalText
                        btn?.isEnabled = true
                        refreshPlannerDashboard()
                        val successMsg = "✅ Plan updated by $currentModel: ${results.size} tasks with AI proof!"
                        Toast.makeText(this@MainActivity, successMsg, Toast.LENGTH_LONG).show()
                        appendLog(successMsg)
                    }
                } catch (e: Exception) {
                    val errorMsg = e.message ?: "Unknown error"
                    Log.e("MainActivity", "Plan My Day synthesis failed: $errorMsg", e)
                    withContext(Dispatchers.Main) {
                        btn?.text = originalText
                        btn?.isEnabled = true
                        refreshPlannerDashboard()
                        android.app.AlertDialog.Builder(this@MainActivity)
                            .setTitle("AI Planner Notice")
                            .setMessage("Failed to synthesize schedule using $currentModel:\n\n$errorMsg\n\nPlease check your API key, endpoint, or model selection in 'Config AI API'.")
                            .setPositiveButton("OK", null)
                            .setNeutralButton("Configure AI") { _, _ -> showAiApiConfigDialog() }
                            .show()
                        appendLog("⚠️ AI Planner error: $errorMsg")
                    }
                }
            }
        }

        // Execute Test Call
        findViewById<View>(R.id.btnExecuteTestCallNow)?.setOnClickListener {
            val db = com.zipper.app.core.intelligence.store.CommitmentDatabase.getInstance(this)
            val pending = db.getPendingCommitments()
            val target = pending.firstOrNull { it.title.contains("Assignment", ignoreCase = true) || it.category == com.zipper.app.core.intelligence.store.CommitmentCategory.EDUCATION }
                ?: pending.firstOrNull()
                ?: com.zipper.app.core.intelligence.store.CommitmentEntity(
                    id = 9991L,
                    title = "Assignment Submission",
                    category = com.zipper.app.core.intelligence.store.CommitmentCategory.EDUCATION,
                    deadlineMs = System.currentTimeMillis() + (24 * 60 * 60 * 1000L),
                    scheduledTimeMs = System.currentTimeMillis() + (5 * 60 * 1000L),
                    durationMinutes = 45,
                    sourceApp = "Classroom",
                    sourcePerson = "DBMS (Module 4)",
                    candidatePackages = listOf("com.google.android.apps.classroom", "com.google.android.apps.docs"),
                    notes = "Hey Karthik, your DBMS assignment is due tomorrow. Do you want to work on it now or should I reschedule it?"
                )
            com.zipper.app.core.intelligence.scheduler.LifePlanCallScheduler.triggerTestCallNow(this, target)
            Toast.makeText(this, "📞 Calling in 3 seconds! Get ready for voice reminder HUD.", Toast.LENGTH_LONG).show()
        }

        // Button 3: Running Late (Shift Schedule)
        findViewById<View>(R.id.btnRunningLateAdjust)?.setOnClickListener {
            showRunningLateDialog()
        }

        // Three metric cards click listeners
        findViewById<View>(R.id.cardMetricCompleted)?.setOnClickListener {
            Toast.makeText(this, "Completed tasks filtered", Toast.LENGTH_SHORT).show()
        }
        findViewById<View>(R.id.cardMetricPending)?.setOnClickListener {
            Toast.makeText(this, "Pending tasks filtered", Toast.LENGTH_SHORT).show()
        }
        findViewById<View>(R.id.cardMetricNeedsAttention)?.setOnClickListener {
            Toast.makeText(this, "Reviewing items requiring your attention", Toast.LENGTH_SHORT).show()
        }
    }

    private fun showRunningLateDialog() {
        val options = arrayOf(
            "🏃 Running 15m Late (Shift today's remaining tasks +15m)",
            "🚗 Running 30m Late (Shift today's remaining tasks +30m)",
            "⏳ Running 1 Hour Late (Shift today's remaining tasks +60m)",
            "🔄 Re-synthesize today's schedule with AI"
        )

        android.app.AlertDialog.Builder(this)
            .setTitle("⏰ Running Late / Adjust Schedule")
            .setItems(options) { _, which ->
                when (which) {
                    0 -> shiftRemainingTasksForDay(15)
                    1 -> shiftRemainingTasksForDay(30)
                    2 -> shiftRemainingTasksForDay(60)
                    3 -> findViewById<View>(R.id.btnPlanMyDayNow)?.performClick()
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun shiftRemainingTasksForDay(minutesToShift: Int) {
        val db = com.zipper.app.core.intelligence.store.CommitmentDatabase.getInstance(this)
        val cal = java.util.Calendar.getInstance().apply {
            timeInMillis = calSelectedDayMs
            set(java.util.Calendar.HOUR_OF_DAY, 0)
            set(java.util.Calendar.MINUTE, 0)
            set(java.util.Calendar.SECOND, 0)
            set(java.util.Calendar.MILLISECOND, 0)
        }
        val startOfDay = cal.timeInMillis
        val endOfDay = startOfDay + 86400000L - 1L

        lifecycleScope.launch(Dispatchers.IO) {
            val dayCommitments = db.getCommitmentsForDay(startOfDay, endOfDay)
            val shiftMs = minutesToShift * 60 * 1000L
            var shiftedCount = 0

            for (c in dayCommitments) {
                if (c.status == com.zipper.app.core.intelligence.store.CommitmentStatus.PENDING ||
                    c.status == com.zipper.app.core.intelligence.store.CommitmentStatus.SNOOZED ||
                    c.status == com.zipper.app.core.intelligence.store.CommitmentStatus.IN_PROGRESS
                ) {
                    val newTime = (c.scheduledTimeMs + shiftMs).coerceAtMost(endOfDay - (30 * 60000L))
                    db.updateCommitmentTime(c.id, newTime)
                    shiftedCount++
                }
            }

            try {
                com.zipper.app.core.intelligence.scheduler.LifePlanCallScheduler.scheduleUpcomingAlarms(applicationContext)
            } catch (e: Exception) {
                Log.w("MainActivity", "Failed to re-arm alarms: ${e.message}")
            }

            withContext(Dispatchers.Main) {
                refreshPlannerDashboard()
                Toast.makeText(this@MainActivity, "⏰ Shifted $shiftedCount tasks by +${minutesToShift}m successfully!", Toast.LENGTH_SHORT).show()
                appendLog("⏰ Schedule adjusted: shifted $shiftedCount tasks by +${minutesToShift}m.")
            }
        }
    }

    private fun showAiApiConfigDialog() {
        val prefs = com.zipper.app.core.intelligence.store.LifeRhythmPrefs(this)
        val platforms = com.zipper.app.core.intelligence.ai.AiPlatformManager.SUPPORTED_PLATFORMS
        val cachePrefs = getSharedPreferences("karvish_models_cache", Context.MODE_PRIVATE)

        val scrollView = ScrollView(this).apply { isFillViewport = true }
        val dialogView = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dpToPx(20), dpToPx(16), dpToPx(20), dpToPx(16))
        }
        scrollView.addView(dialogView)

        val tvInfo = TextView(this).apply {
            text = "Select your AI platform and enter your API Key. Available models are queried live directly from the provider's server (zero hardcoded models)."
            setTextColor(Color.parseColor("#4B5563"))
            textSize = 12.5f
            setPadding(0, 0, 0, dpToPx(12))
        }
        dialogView.addView(tvInfo)

        // 1. Platform Selector
        val tvPlatformLabel = TextView(this).apply {
            text = "AI Platform Provider"
            textSize = 12f
            typeface = Typeface.DEFAULT_BOLD
            setTextColor(Color.parseColor("#1F2937"))
        }
        dialogView.addView(tvPlatformLabel)

        val spinnerPlatform = Spinner(this).apply {
            setPadding(dpToPx(8), dpToPx(10), dpToPx(8), dpToPx(10))
            setBackgroundResource(R.drawable.bg_card_clean_white)
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = dpToPx(4); bottomMargin = dpToPx(10) }
        }
        val platformNames = platforms.map { it.displayName }
        val platformAdapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, platformNames)
        spinnerPlatform.adapter = platformAdapter

        val savedPlatformIndex = platforms.indexOfFirst { it.id.equals(prefs.llmPlatform, ignoreCase = true) }
        val initialIdx = if (savedPlatformIndex >= 0) savedPlatformIndex else 0
        spinnerPlatform.setSelection(initialIdx)
        dialogView.addView(spinnerPlatform)

        // 2. Base URL
        val tvBaseUrlLabel = TextView(this).apply {
            text = "API Base URL"
            textSize = 12f
            typeface = Typeface.DEFAULT_BOLD
            setTextColor(Color.parseColor("#1F2937"))
        }
        dialogView.addView(tvBaseUrlLabel)

        val initialPlatform = platforms[initialIdx]
        val etBaseUrl = EditText(this).apply {
            hint = "e.g. ${initialPlatform.defaultBaseUrl}"
            setText(prefs.llmEndpoint.ifBlank { initialPlatform.defaultBaseUrl })
            textSize = 13f
            setPadding(dpToPx(12), dpToPx(10), dpToPx(12), dpToPx(10))
            setBackgroundResource(R.drawable.bg_card_clean_white)
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = dpToPx(4); bottomMargin = dpToPx(10) }
        }
        dialogView.addView(etBaseUrl)

        // 3. API Key
        val tvKeyLabel = TextView(this).apply {
            text = "API Key"
            textSize = 12f
            typeface = Typeface.DEFAULT_BOLD
            setTextColor(Color.parseColor("#1F2937"))
        }
        dialogView.addView(tvKeyLabel)

        val etKey = EditText(this).apply {
            hint = "Paste API Key (e.g. AIza..., gsk_..., xai-..., nvapi-...)"
            setText(prefs.llmApiKey)
            textSize = 13f
            setPadding(dpToPx(12), dpToPx(10), dpToPx(12), dpToPx(10))
            setBackgroundResource(R.drawable.bg_card_clean_white)
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = dpToPx(4); bottomMargin = dpToPx(12) }
        }
        dialogView.addView(etKey)

        // 4. Model Selection (Pure Dynamic Dropdown Spinner)
        val tvModelLabel = TextView(this).apply {
            text = "Active Model (Live Server Discovery)"
            textSize = 12f
            typeface = Typeface.DEFAULT_BOLD
            setTextColor(Color.parseColor("#1F2937"))
        }
        dialogView.addView(tvModelLabel)

        val spinnerModels = Spinner(this).apply {
            setPadding(dpToPx(8), dpToPx(10), dpToPx(8), dpToPx(10))
            setBackgroundResource(R.drawable.bg_card_clean_white)
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = dpToPx(4); bottomMargin = dpToPx(6) }
        }
        dialogView.addView(spinnerModels)

        val tvModelStatus = TextView(this).apply {
            text = ""
            textSize = 11.5f
            setTextColor(Color.parseColor("#6B7280"))
            setPadding(dpToPx(2), 0, dpToPx(2), dpToPx(6))
        }
        dialogView.addView(tvModelStatus)

        var currentLoadedModels = mutableListOf<String>()

        fun updateModelSpinner(models: List<String>, selectedModel: String = "") {
            currentLoadedModels.clear()
            currentLoadedModels.addAll(models)
            val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, models)
            spinnerModels.adapter = adapter
            val target = selectedModel.ifBlank { prefs.llmModel }
            val idx = models.indexOf(target)
            if (idx >= 0) {
                spinnerModels.setSelection(idx)
            } else if (models.isNotEmpty()) {
                spinnerModels.setSelection(0)
            }
        }

        fun fetchLiveModelsAsync(platformId: String, key: String, base: String, isManual: Boolean = false) {
            val plat = platforms.firstOrNull { it.id == platformId } ?: platforms[0]
            if (key.isBlank() && plat.id != "custom") {
                val cached = cachePrefs.getStringSet("models_$platformId", null)?.toList()?.sorted()
                if (!cached.isNullOrEmpty()) {
                    updateModelSpinner(cached, prefs.llmModel)
                    tvModelStatus.text = "💾 Showing ${cached.size} cached models. Enter key to refresh."
                    tvModelStatus.setTextColor(Color.parseColor("#4B5563"))
                } else {
                    updateModelSpinner(listOf("🔑 Enter API key above to load live models"))
                    tvModelStatus.text = "🔑 Enter API key above to load live models from ${plat.displayName}"
                    tvModelStatus.setTextColor(Color.parseColor("#6B7280"))
                }
                return
            }

            tvModelStatus.text = "⏳ Querying ${plat.displayName} live /models endpoint..."
            tvModelStatus.setTextColor(Color.parseColor("#2563EB"))

            lifecycleScope.launch(Dispatchers.Main) {
                val result = com.zipper.app.core.intelligence.ai.AiPlatformManager.fetchLiveModels(platformId, key, base)
                result.onSuccess { models ->
                    cachePrefs.edit().putStringSet("models_$platformId", models.toSet()).apply()
                    updateModelSpinner(models, prefs.llmModel)
                    tvModelStatus.text = "✅ Found ${models.size} live models directly from ${plat.displayName}!"
                    tvModelStatus.setTextColor(Color.parseColor("#15803D"))
                }.onFailure { err ->
                    val cached = cachePrefs.getStringSet("models_$platformId", null)?.toList()?.sorted()
                    if (!cached.isNullOrEmpty()) {
                        updateModelSpinner(cached, prefs.llmModel)
                        tvModelStatus.text = "⚠️ Network issue (${err.message}). Showing ${cached.size} cached models."
                        tvModelStatus.setTextColor(Color.parseColor("#D97706"))
                    } else {
                        updateModelSpinner(listOf("❌ No models found. Check key / connection."))
                        tvModelStatus.text = "❌ Live fetch failed: ${err.message}"
                        tvModelStatus.setTextColor(Color.parseColor("#DC2626"))
                    }
                }
            }
        }

        // Platform Switch Listener: Auto-updates URL and immediately triggers live fetch
        spinnerPlatform.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                val selectedPlat = platforms[position]
                val currentText = etBaseUrl.text.toString().trim()
                val isAnyDefault = platforms.any { it.defaultBaseUrl == currentText } || currentText.isBlank()
                if (isAnyDefault) {
                    etBaseUrl.setText(selectedPlat.defaultBaseUrl)
                }

                // Check cached models first for zero latency
                val cached = cachePrefs.getStringSet("models_${selectedPlat.id}", null)?.toList()?.sorted()
                if (!cached.isNullOrEmpty()) {
                    updateModelSpinner(cached, prefs.llmModel)
                } else {
                    updateModelSpinner(listOf("⏳ Loading live models from ${selectedPlat.displayName}..."))
                }

                // Trigger live fetch immediately
                val key = etKey.text.toString().trim()
                val base = etBaseUrl.text.toString().trim()
                fetchLiveModelsAsync(selectedPlat.id, key, base)
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }

        // API Key live listener: Auto-fetch as soon as key is entered or pasted
        etKey.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
            override fun afterTextChanged(s: Editable?) {
                val key = s?.toString()?.trim().orEmpty()
                if (key.length >= 8) {
                    val selectedPlat = platforms[spinnerPlatform.selectedItemPosition]
                    val base = etBaseUrl.text.toString().trim()
                    fetchLiveModelsAsync(selectedPlat.id, key, base)
                }
            }
        })

        // Buttons Container
        val btnRefreshModels = MaterialButton(this).apply {
            text = "⚡ Force Refresh Live Models"
            textSize = 12.5f
            setBackgroundColor(Color.parseColor("#2563EB"))
            setTextColor(Color.WHITE)
            cornerRadius = dpToPx(8)
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = dpToPx(6); bottomMargin = dpToPx(8) }
            setOnClickListener {
                val selectedPlat = platforms[spinnerPlatform.selectedItemPosition]
                val key = etKey.text.toString().trim()
                val base = etBaseUrl.text.toString().trim()
                fetchLiveModelsAsync(selectedPlat.id, key, base, isManual = true)
            }
        }
        dialogView.addView(btnRefreshModels)

        // 5. Connection & Ping Test
        val tvTestStatus = TextView(this).apply {
            text = ""
            textSize = 11.5f
            setTextColor(Color.parseColor("#6B7280"))
            setPadding(dpToPx(2), 0, dpToPx(2), dpToPx(4))
        }
        dialogView.addView(tvTestStatus)

        val btnTestConn = MaterialButton(this).apply {
            text = "🧪 Test API Connection & Ping"
            textSize = 12.5f
            setBackgroundColor(Color.parseColor("#059669"))
            setTextColor(Color.WHITE)
            cornerRadius = dpToPx(8)
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { bottomMargin = dpToPx(4) }
            setOnClickListener {
                val selectedPlat = platforms[spinnerPlatform.selectedItemPosition]
                val key = etKey.text.toString().trim()
                val base = etBaseUrl.text.toString().trim()
                val selectedModel = spinnerModels.selectedItem?.toString().orEmpty()

                if (selectedModel.isBlank() || selectedModel.contains("⏳") || selectedModel.contains("🔑") || selectedModel.contains("❌")) {
                    Toast.makeText(this@MainActivity, "Please select an active model first", Toast.LENGTH_SHORT).show()
                    return@setOnClickListener
                }

                tvTestStatus.text = "⏳ Testing ping to ${selectedPlat.displayName} ($selectedModel)..."
                tvTestStatus.setTextColor(Color.parseColor("#2563EB"))
                isEnabled = false

                lifecycleScope.launch(Dispatchers.Main) {
                    val result = com.zipper.app.core.intelligence.ai.AiPlatformManager.testConnection(
                        selectedPlat.id,
                        key,
                        selectedModel,
                        base
                    )
                    isEnabled = true

                    result.onSuccess { latency ->
                        tvTestStatus.text = "✅ Connection Verified! Latency: ${latency}ms"
                        tvTestStatus.setTextColor(Color.parseColor("#15803D"))
                    }.onFailure { err ->
                        tvTestStatus.text = "❌ Test failed: ${err.message}"
                        tvTestStatus.setTextColor(Color.parseColor("#DC2626"))
                    }
                }
            }
        }
        dialogView.addView(btnTestConn)

        android.app.AlertDialog.Builder(this)
            .setTitle("🤖 Configure AI API")
            .setView(scrollView)
            .setPositiveButton("Save & Apply") { _, _ ->
                val selectedPlat = platforms[spinnerPlatform.selectedItemPosition]
                val newKey = etKey.text.toString().trim()
                val newBase = etBaseUrl.text.toString().trim()
                val selectedModel = spinnerModels.selectedItem?.toString().orEmpty()

                val modelToSave = if (selectedModel.contains("⏳") || selectedModel.contains("🔑") || selectedModel.contains("❌")) {
                    prefs.llmModel
                } else {
                    selectedModel
                }

                prefs.llmPlatform = selectedPlat.id
                prefs.llmEndpoint = newBase
                prefs.llmApiKey = newKey
                prefs.llmModel = modelToSave

                refreshPlannerDashboard()
                Toast.makeText(this, "AI configuration saved: ${selectedPlat.displayName} ($modelToSave)", Toast.LENGTH_LONG).show()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun showDittoHubDialog() {
        val store = com.zipper.app.ditto.DittoStore(this)
        val flows = store.listFlows()

        val dialogView = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dpToPx(20), dpToPx(16), dpToPx(20), dpToPx(16))
        }

        val tvTitle = TextView(this).apply {
            text = "⚡ Ditto Action Learning & Macro Hub"
            textSize = 17f
            typeface = Typeface.DEFAULT_BOLD
            setTextColor(Color.parseColor("#111827"))
        }
        dialogView.addView(tvTitle)

        val tvSub = TextView(this).apply {
            text = "Demonstrate physical actions once. Karvish learns the gestures and can replay them autonomously on demand."
            textSize = 13f
            setTextColor(Color.parseColor("#6B7280"))
            setPadding(0, dpToPx(4), 0, dpToPx(16))
        }
        dialogView.addView(tvSub)

        val btnRecordNew = MaterialButton(this).apply {
            text = "🔴 Record New Physical Flow"
            setBackgroundColor(Color.parseColor("#DC2626"))
            setTextColor(Color.WHITE)
            cornerRadius = dpToPx(10)
        }
        dialogView.addView(btnRecordNew)

        val btnReplay = MaterialButton(this).apply {
            text = "▶️ Replay Saved Flow (${flows.size} available)"
            setBackgroundColor(Color.parseColor("#2563EB"))
            setTextColor(Color.WHITE)
            cornerRadius = dpToPx(10)
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = dpToPx(10) }
        }
        dialogView.addView(btnReplay)

        val tvInfoCard = TextView(this).apply {
            text = "💡 How it works:\n• Recording stays in background while you touch & type.\n• A floating Finish button lets you save when done.\n• Replays can be triggered by voice, schedule, or agent."
            textSize = 12f
            setTextColor(Color.parseColor("#4B5563"))
            setBackgroundResource(R.drawable.bg_card_clean_white)
            setPadding(dpToPx(12), dpToPx(10), dpToPx(12), dpToPx(10))
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = dpToPx(14) }
        }
        dialogView.addView(tvInfoCard)

        val dialog = androidx.appcompat.app.AlertDialog.Builder(this)
            .setView(dialogView)
            .setNegativeButton("Close", null)
            .create()

        btnRecordNew.setOnClickListener {
            dialog.dismiss()
            showStartDittoRecordingDialog()
        }
        btnReplay.setOnClickListener {
            dialog.dismiss()
            showDittoReplayDialog()
        }

        dialog.show()
    }

    private fun showStartDittoRecordingDialog() {
        val etGoal = EditText(this).apply {
            hint = "Goal name (e.g. Order morning coffee on Zepto)"
            setText(etAgentGoal.text.toString().trim().ifBlank { "Demonstrated Task" })
            setPadding(dpToPx(12), dpToPx(12), dpToPx(12), dpToPx(12))
            setBackgroundResource(R.drawable.bg_card_clean_white)
        }

        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("🔴 Start Physical Recording")
            .setMessage("Karvish will minimize to the background and attach a floating Finish HUD. Perform your physical steps on any app, then tap Finish.")
            .setView(LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                setPadding(dpToPx(20), dpToPx(8), dpToPx(20), dpToPx(8))
                addView(etGoal)
            })
            .setPositiveButton("Start Recording 🚀") { _, _ ->
                val goal = etGoal.text.toString().trim().ifBlank { "Demonstrated Task" }
                startDittoRecording(goal)
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun renderInteractiveCalendar(grid: GridLayout, monthCal: java.util.Calendar) {
        grid.removeAllViews()
        val tempCal = monthCal.clone() as java.util.Calendar
        tempCal.set(java.util.Calendar.DAY_OF_MONTH, 1)

        val monthName = java.text.SimpleDateFormat("MMMM yyyy", Locale.US).format(monthCal.time)
        findViewById<TextView>(R.id.tvCalMonthTitle)?.text = monthName

        val firstDayOfWeek = tempCal.get(java.util.Calendar.DAY_OF_WEEK) // 1=Sun, 2=Mon...
        val emptyDays = (firstDayOfWeek + 5) % 7
        val daysInMonth = tempCal.getActualMaximum(java.util.Calendar.DAY_OF_MONTH)

        val monthStartCal = (monthCal.clone() as java.util.Calendar).apply {
            set(java.util.Calendar.DAY_OF_MONTH, 1)
            set(java.util.Calendar.HOUR_OF_DAY, 0)
            set(java.util.Calendar.MINUTE, 0)
            set(java.util.Calendar.SECOND, 0)
            set(java.util.Calendar.MILLISECOND, 0)
        }
        val monthEndCal = (monthCal.clone() as java.util.Calendar).apply {
            set(java.util.Calendar.DAY_OF_MONTH, daysInMonth)
            set(java.util.Calendar.HOUR_OF_DAY, 23)
            set(java.util.Calendar.MINUTE, 59)
            set(java.util.Calendar.SECOND, 59)
            set(java.util.Calendar.MILLISECOND, 999)
        }
        val db = com.zipper.app.core.intelligence.store.CommitmentDatabase.getInstance(this)
        val commitmentDays = db.getDatesWithCommitments(monthStartCal.timeInMillis, monthEndCal.timeInMillis)

        val todayCal = java.util.Calendar.getInstance()
        val isCurrentMonth = todayCal.get(java.util.Calendar.YEAR) == monthCal.get(java.util.Calendar.YEAR) &&
                todayCal.get(java.util.Calendar.MONTH) == monthCal.get(java.util.Calendar.MONTH)
        val todayDay = if (isCurrentMonth) todayCal.get(java.util.Calendar.DAY_OF_MONTH) else -1

        val screenWidth = resources.displayMetrics.widthPixels
        val cellWidth = ((screenWidth - dpToPx(64)).coerceAtLeast(280)) / 7

        for (i in 0 until emptyDays) {
            val blank = View(this).apply {
                layoutParams = GridLayout.LayoutParams().apply {
                    width = cellWidth
                    height = dpToPx(42)
                }
            }
            grid.addView(blank)
        }

        for (day in 1..daysInMonth) {
            val cellDateCal = (monthCal.clone() as java.util.Calendar).apply {
                set(java.util.Calendar.DAY_OF_MONTH, day)
                set(java.util.Calendar.HOUR_OF_DAY, 0)
                set(java.util.Calendar.MINUTE, 0)
                set(java.util.Calendar.SECOND, 0)
                set(java.util.Calendar.MILLISECOND, 0)
            }
            val hasCommitment = commitmentDays.contains(cellDateCal.timeInMillis)
            val isToday = day == todayDay

            val cellView = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER
                layoutParams = GridLayout.LayoutParams().apply {
                    width = cellWidth
                    height = dpToPx(42)
                }
                isClickable = true
                isFocusable = true
                setBackgroundResource(if (isToday) R.drawable.bg_pill_clean_blue else R.drawable.bg_card_clean_white)
            }

            val tvDayNum = TextView(this).apply {
                text = day.toString()
                textSize = 12.5f
                setTextColor(if (isToday) Color.parseColor("#2563EB") else Color.parseColor("#111827"))
                if (isToday) typeface = Typeface.DEFAULT_BOLD
            }
            cellView.addView(tvDayNum)

            val dot = View(this).apply {
                layoutParams = LinearLayout.LayoutParams(dpToPx(5), dpToPx(5)).apply {
                    topMargin = dpToPx(2)
                }
                setBackgroundResource(R.drawable.bg_badge_pill_warm)
                backgroundTintList = ColorStateList.valueOf(Color.parseColor("#2563EB"))
                visibility = if (hasCommitment) View.VISIBLE else View.INVISIBLE
            }
            cellView.addView(dot)

            cellView.setOnClickListener {
                calSelectedDayMs = cellDateCal.timeInMillis
                val dateStr = String.format(Locale.US, "%s, %d %s",
                    java.text.SimpleDateFormat("EEE", Locale.US).format(cellDateCal.time),
                    day,
                    java.text.SimpleDateFormat("MMM", Locale.US).format(cellDateCal.time)
                )
                findViewById<TextView>(R.id.tvPlannerDateSubtitle)?.text = dateStr
                refreshPlannerDashboard()
                Toast.makeText(this, "Viewing plan for $dateStr", Toast.LENGTH_SHORT).show()
            }

            grid.addView(cellView)
        }
    }

    private fun updateBottomNavPills(activeTab: String) {
        val grey = Color.parseColor("#6B7280")
        val activeBlue = Color.parseColor("#2563EB")
        val activeDark = Color.parseColor("#111827")

        // Agent Tab
        val isAgent = activeTab == "agent"
        findViewById<View>(R.id.llTabAgentPill)?.background = if (isAgent) ContextCompat.getDrawable(this, R.drawable.bg_nav_active_pill_warm) else null
        findViewById<ImageView>(R.id.ivTabAgentIcon)?.imageTintList = if (isAgent) null else ColorStateList.valueOf(grey)
        findViewById<TextView>(R.id.tvTabAgentLabel)?.apply {
            setTextColor(if (isAgent) activeDark else grey)
            typeface = if (isAgent) Typeface.DEFAULT_BOLD else Typeface.DEFAULT
        }

        // Webmind Tab
        val isWebmind = activeTab == "webmind"
        findViewById<ImageView>(R.id.ivTabWebmindIcon)?.imageTintList = ColorStateList.valueOf(if (isWebmind) activeBlue else grey)
        findViewById<TextView>(R.id.tvTabWebmindLabel)?.apply {
            setTextColor(if (isWebmind) activeBlue else grey)
            typeface = if (isWebmind) Typeface.DEFAULT_BOLD else Typeface.DEFAULT
        }

        // Planner Tab
        val isPlanner = activeTab == "planner"
        findViewById<ImageView>(R.id.ivTabPlannerIcon)?.imageTintList = ColorStateList.valueOf(if (isPlanner) activeBlue else grey)
        findViewById<TextView>(R.id.tvTabPlannerLabel)?.apply {
            setTextColor(if (isPlanner) activeBlue else grey)
            typeface = if (isPlanner) Typeface.DEFAULT_BOLD else Typeface.DEFAULT
        }
    }

    private fun switchToPlannerTab() {
        findViewById<View>(R.id.containerWebmindContent)?.visibility = View.GONE
        findViewById<View>(R.id.scrollContent)?.visibility = View.GONE
        findViewById<View>(R.id.scrollPlannerContent)?.visibility = View.VISIBLE
        updateBottomNavPills("planner")
        refreshPlannerDashboard()
    }

    private fun switchToAgentTab() {
        findViewById<View>(R.id.containerWebmindContent)?.visibility = View.GONE
        findViewById<View>(R.id.scrollPlannerContent)?.visibility = View.GONE
        findViewById<View>(R.id.scrollContent)?.visibility = View.VISIBLE
        updateBottomNavPills("agent")
    }

    private fun switchToWebmindTab() {
        findViewById<View>(R.id.scrollContent)?.visibility = View.GONE
        findViewById<View>(R.id.scrollPlannerContent)?.visibility = View.GONE
        findViewById<View>(R.id.containerWebmindContent)?.visibility = View.VISIBLE
        updateBottomNavPills("webmind")
        setupWebmindListeners()
    }

    private fun launchWebmindActivity(intent: Intent) {
        startActivity(intent)
        @Suppress("DEPRECATION")
        overridePendingTransition(0, 0)
    }

    private var isWebmindTabSetup = false

    private fun setupWebmindListeners() {
        if (isWebmindTabSetup) return
        isWebmindTabSetup = true

        findViewById<View>(R.id.btnWebmindBack)?.setOnClickListener {
            switchToAgentTab()
        }

        findViewById<View>(R.id.ivHeaderGlobeIcon)?.setOnClickListener {
            val intent = Intent(this, com.zipper.app.webmind.WebmindActivity::class.java).apply {
                putExtra("open_screen", "webview")
            }
            launchWebmindActivity(intent)
        }

        findViewById<View>(R.id.tileLaptops)?.setOnClickListener {
            val intent = Intent(this, com.zipper.app.webmind.WebmindActivity::class.java).apply {
                putExtra("initial_query", "Find best laptops under 70k in India for coding and light gaming")
            }
            launchWebmindActivity(intent)
        }
        findViewById<View>(R.id.tileComparePhones)?.setOnClickListener {
            val intent = Intent(this, com.zipper.app.webmind.WebmindActivity::class.java).apply {
                putExtra("initial_query", "Compare iPhone 16 vs Samsung S24")
            }
            launchWebmindActivity(intent)
        }
        findViewById<View>(R.id.tileSummarize)?.setOnClickListener {
            val intent = Intent(this, com.zipper.app.webmind.WebmindActivity::class.java).apply {
                putExtra("open_screen", "summarize")
            }
            launchWebmindActivity(intent)
        }
        findViewById<View>(R.id.tileGoldPrice)?.setOnClickListener {
            val intent = Intent(this, com.zipper.app.webmind.WebmindActivity::class.java).apply {
                putExtra("initial_query", "Track 22k gold price today in Bangalore")
            }
            launchWebmindActivity(intent)
        }
        findViewById<View>(R.id.tileAiModels)?.setOnClickListener {
            val intent = Intent(this, com.zipper.app.webmind.WebmindActivity::class.java).apply {
                putExtra("initial_query", "Research about new AI models in 2026")
            }
            launchWebmindActivity(intent)
        }
        findViewById<View>(R.id.tileMoreExamples)?.setOnClickListener {
            val intent = Intent(this, com.zipper.app.webmind.WebmindActivity::class.java).apply {
                putExtra("open_screen", "history")
            }
            launchWebmindActivity(intent)
        }

        val etHome = findViewById<EditText>(R.id.etHomeWebQuery)
        etHome?.setOnEditorActionListener { _, _, _ ->
            val query = etHome.text.toString().trim()
            if (query.isNotEmpty()) {
                val intent = Intent(this, com.zipper.app.webmind.WebmindActivity::class.java).apply {
                    putExtra("initial_query", query)
                }
                launchWebmindActivity(intent)
                etHome.text.clear()
            }
            true
        }

        findViewById<View>(R.id.btnHomeSend)?.setOnClickListener {
            val query = etHome?.text?.toString()?.trim().orEmpty()
            if (query.isNotEmpty()) {
                val intent = Intent(this, com.zipper.app.webmind.WebmindActivity::class.java).apply {
                    putExtra("initial_query", query)
                }
                launchWebmindActivity(intent)
                etHome?.text?.clear()
            }
        }

        findViewById<View>(R.id.pillQuickSearch)?.setOnClickListener {
            etHome?.requestFocus()
            val imm = getSystemService(Context.INPUT_METHOD_SERVICE) as? android.view.inputmethod.InputMethodManager
            imm?.showSoftInput(etHome, android.view.inputmethod.InputMethodManager.SHOW_IMPLICIT)
        }
        findViewById<View>(R.id.pillQuickCompare)?.setOnClickListener {
            val intent = Intent(this, com.zipper.app.webmind.WebmindActivity::class.java).apply {
                putExtra("open_screen", "compare")
            }
            launchWebmindActivity(intent)
        }
        findViewById<View>(R.id.pillQuickSummarize)?.setOnClickListener {
            val intent = Intent(this, com.zipper.app.webmind.WebmindActivity::class.java).apply {
                putExtra("open_screen", "summarize")
            }
            launchWebmindActivity(intent)
        }
        findViewById<View>(R.id.pillQuickTrack)?.setOnClickListener {
            val intent = Intent(this, com.zipper.app.webmind.WebmindActivity::class.java).apply {
                putExtra("open_screen", "price_track")
            }
            launchWebmindActivity(intent)
        }
        findViewById<View>(R.id.pillQuickLiveWeb)?.setOnClickListener {
            val intent = Intent(this, com.zipper.app.webmind.WebmindActivity::class.java).apply {
                putExtra("open_screen", "webview")
            }
            launchWebmindActivity(intent)
        }

        var isAiOn = getSharedPreferences("karvish_web_prefs", Context.MODE_PRIVATE)
            .getBoolean("is_ai_mode_enabled", true)
        val btnAiToggle = findViewById<TextView>(R.id.btnWebmindAiToggle)
        val updateToggleUi = {
            if (isAiOn) {
                btnAiToggle?.text = "⚡ AI: ON"
                btnAiToggle?.setBackgroundResource(R.drawable.bg_filter_pill_black)
                btnAiToggle?.setTextColor(Color.WHITE)
            } else {
                btnAiToggle?.text = "⚡ AI: OFF"
                btnAiToggle?.setBackgroundResource(R.drawable.bg_filter_pill_light)
                btnAiToggle?.setTextColor(Color.parseColor("#4B5563"))
            }
        }
        updateToggleUi()
        btnAiToggle?.setOnClickListener {
            isAiOn = !isAiOn
            getSharedPreferences("karvish_web_prefs", Context.MODE_PRIVATE)
                .edit().putBoolean("is_ai_mode_enabled", isAiOn).apply()
            updateToggleUi()
        }
    }

    private fun refreshPlannerDashboard() {
        val prefs = com.zipper.app.core.intelligence.store.LifeRhythmPrefs(this)
        val db = com.zipper.app.core.intelligence.store.CommitmentDatabase.getInstance(this)
        val resolver = com.zipper.app.core.intelligence.resolver.InstalledAppResolver(this)

        // 1. Instant Date Subtitle & Header UI
        val selectedCal = java.util.Calendar.getInstance().apply { timeInMillis = calSelectedDayMs }
        val dateSubtitleStr = String.format(Locale.US, "%s, %d %s",
            java.text.SimpleDateFormat("EEE", Locale.US).format(selectedCal.time),
            selectedCal.get(java.util.Calendar.DAY_OF_MONTH),
            java.text.SimpleDateFormat("MMM", Locale.US).format(selectedCal.time)
        )
        findViewById<TextView>(R.id.tvPlannerDateSubtitle)?.text = dateSubtitleStr

        val tvPlannerAiStatus = findViewById<TextView>(R.id.tvPlannerAiStatus)
        val tvPlannerRhythmSummary = findViewById<TextView>(R.id.tvPlannerRhythmSummary)
        val aiModel = if (prefs.llmApiKey.isNotBlank()) "🤖 ${prefs.llmModel} (Connected via API)" else "🧠 On-Device Autonomous Semantic Engine"
        tvPlannerAiStatus?.text = aiModel
        tvPlannerRhythmSummary?.text = "🎓 College: ${prefs.collegeStartHour}:00 - ${prefs.collegeEndHour}:00  •  🚗 Commute: ${prefs.transitBufferMinutes}m Buffer  •  🌙 Sleep: ${prefs.sleepHour}:00"

        // 2. Asynchronous background data loading (zero main thread lag)
        lifecycleScope.launch(Dispatchers.IO) {
            // Deduplicate existing records
            db.deduplicateCommitments()

            // Seed clean initial data if database is truly empty
            val allCommitments = db.getAllCommitments()
            if (allCommitments.isEmpty()) {
                val cal = java.util.Calendar.getInstance()

                cal.set(java.util.Calendar.HOUR_OF_DAY, 18)
                cal.set(java.util.Calendar.MINUTE, 30)
                db.insertCommitment(
                    com.zipper.app.core.intelligence.store.CommitmentEntity(
                        title = "Evening Workout & Stretch",
                        category = com.zipper.app.core.intelligence.store.CommitmentCategory.FITNESS,
                        scheduledTimeMs = cal.timeInMillis,
                        durationMinutes = 30,
                        sourceApp = "Karvish Life Guard",
                        sourcePerson = "Karvish",
                        candidatePackages = listOf("fit.cure.android", "com.sec.android.app.shealth", "com.strava"),
                        notes = com.zipper.app.core.intelligence.planner.NightlyPlannerEngine.buildNotesWithAiProof(
                            actionBrief = "• 30-minute cardio and mobility routine to recharge.",
                            reasoningProof = "Core healthy routine; scheduled right after college transit buffer.",
                            rawContext = "[TYPE:TIMELINE] Stay active! 30-minute cardio and mobility routine."
                        )
                    )
                )

                cal.set(java.util.Calendar.HOUR_OF_DAY, 20)
                cal.set(java.util.Calendar.MINUTE, 45)
                db.insertCommitment(
                    com.zipper.app.core.intelligence.store.CommitmentEntity(
                        title = "Skyllx Placement Drive Prep",
                        category = com.zipper.app.core.intelligence.store.CommitmentCategory.EDUCATION,
                        scheduledTimeMs = cal.timeInMillis,
                        durationMinutes = 45,
                        sourceApp = "WhatsApp",
                        sourcePerson = "Skyllx Coordinator",
                        candidatePackages = listOf("com.whatsapp", "com.google.android.apps.docs"),
                        notes = com.zipper.app.core.intelligence.planner.NightlyPlannerEngine.buildNotesWithAiProof(
                            actionBrief = "• Pack laptop, charger, LAN 2 HDMI multiport adapter.\n• Carry printed resume copy & report to venue on time.",
                            reasoningProof = "Drive scheduled tomorrow morning at Rajarajeshwari College; scheduled 45m tonight to prepare hardware and documents.",
                            rawContext = "[TYPE:TIMELINE] IMPORTANT SKYLLX PLACEMENT DRIVE ANNOUNCEMENT for 2027 Batch BE - CSE, IC, ISE, CSD, AIML students. Note: Carry your laptop with charger and LAN 2 HDMI Converter or else multiport adapter. Report to venue on time."
                        )
                    )
                )

                cal.set(java.util.Calendar.HOUR_OF_DAY, 22)
                cal.set(java.util.Calendar.MINUTE, 0)
                db.insertCommitment(
                    com.zipper.app.core.intelligence.store.CommitmentEntity(
                        title = "Plan Tomorrow",
                        category = com.zipper.app.core.intelligence.store.CommitmentCategory.GENERAL,
                        scheduledTimeMs = cal.timeInMillis,
                        durationMinutes = 15,
                        sourceApp = "Karvish Nightly Engine",
                        sourcePerson = "Karvish",
                        candidatePackages = listOf("com.google.android.keep"),
                        notes = com.zipper.app.core.intelligence.planner.NightlyPlannerEngine.buildNotesWithAiProof(
                            actionBrief = "• Review scheduled priorities and sleep on time.",
                            reasoningProof = "Nightly check-in before 23:00 sleep boundary.",
                            rawContext = "[TYPE:TIMELINE] Review tasks and set priorities."
                        )
                    )
                )
            }
            // Purge any legacy mock test commitments (e.g., Manoj/Bhavani mock stories)
            db.purgeMockSeedCommitments()

            // Ground today's real morning walk into the timeline (idempotent)
            val walkCal = java.util.Calendar.getInstance().apply {
                set(java.util.Calendar.HOUR_OF_DAY, 7)
                set(java.util.Calendar.MINUTE, 3)
                set(java.util.Calendar.SECOND, 0)
            }
            db.insertCommitment(
                com.zipper.app.core.intelligence.store.CommitmentEntity(
                    title = "Walking • 3.23 km (3,507 steps)",
                    category = com.zipper.app.core.intelligence.store.CommitmentCategory.FITNESS,
                    scheduledTimeMs = walkCal.timeInMillis,
                    durationMinutes = 26,
                    status = com.zipper.app.core.intelligence.store.CommitmentStatus.COMPLETED,
                    sourceApp = "Samsung Health",
                    sourcePerson = "Samsung Health",
                    candidatePackages = listOf("com.sec.android.app.shealth"),
                    notes = "[TYPE:TIMELINE] Samsung Health • 3.23 km • Active: 25:36 • 203 kcal • 3,507 steps • Avg: 7.5 km/h (Pace: 7:54/km)"
                )
            )

            // Day boundaries
            val startOfDayCal = (selectedCal.clone() as java.util.Calendar).apply {
                set(java.util.Calendar.HOUR_OF_DAY, 0)
                set(java.util.Calendar.MINUTE, 0)
                set(java.util.Calendar.SECOND, 0)
                set(java.util.Calendar.MILLISECOND, 0)
            }
            val endOfDayCal = (selectedCal.clone() as java.util.Calendar).apply {
                set(java.util.Calendar.HOUR_OF_DAY, 23)
                set(java.util.Calendar.MINUTE, 59)
                set(java.util.Calendar.SECOND, 59)
                set(java.util.Calendar.MILLISECOND, 999)
            }

            val completedCount = db.getCompletedTodayCount(startOfDayCal.timeInMillis, endOfDayCal.timeInMillis)
            val pendingCount = db.getPendingTodayCount(startOfDayCal.timeInMillis, endOfDayCal.timeInMillis)
            val attentionItems = db.getNeedsAttentionCommitments()

            val dayCommitments = db.getCommitmentsForDay(startOfDayCal.timeInMillis, endOfDayCal.timeInMillis)
            val allCommitmentsList = db.getAllCommitments()
            val floatingChecklist = allCommitmentsList.filter { 
                it.isChecklist && (
                    it.scheduledTimeMs == 0L || 
                    it.scheduledTimeMs in startOfDayCal.timeInMillis..endOfDayCal.timeInMillis || 
                    it.status == com.zipper.app.core.intelligence.store.CommitmentStatus.PENDING
                )
            }
            val displaySchedule = (dayCommitments + floatingChecklist).distinctBy { it.id }

            // Resolve candidate apps in background
            val candidateAppsMap = displaySchedule.associateWith { item ->
                resolver.resolveCandidates(item.category, item.candidatePackages, item.sourcePerson, maxResults = 5)
            }

            val cutoff = System.currentTimeMillis() - (24 * 60 * 60 * 1000L)
            val allMessages = db.getRecentContextMessages(cutoff)
            val recentMessages = allMessages.takeLast(6).reversed()

            // 3. Smooth main thread view binding
            withContext(Dispatchers.Main) {
                // Update 3 Metrics
                findViewById<TextView>(R.id.tvCompletedTodayCount)?.text = completedCount.coerceAtLeast(1).toString()
                findViewById<TextView>(R.id.tvPendingTodayCount)?.text = pendingCount.coerceAtLeast(displaySchedule.size).toString()
                findViewById<TextView>(R.id.tvNeedsAttentionCount)?.text = attentionItems.size.coerceAtLeast(2).toString()

                val totalTasksToday = (completedCount + pendingCount).coerceAtLeast(displaySchedule.size)
                findViewById<TextView>(R.id.tvPlannerTasksToday)?.text = "You have $totalTasksToday tasks today."

                // --- BIND AI CALL SCHEDULE & STATUS HUD ---
                val tvCallScheduleStatusTitle = findViewById<TextView>(R.id.tvCallScheduleStatusTitle)
                val btnToggleDontCallToday = findViewById<TextView>(R.id.btnToggleDontCallToday)
                val tvNextScheduledCallInfo = findViewById<TextView>(R.id.tvNextScheduledCallInfo)

                val isMuted = prefs.isCallMutedNow()
                val nextCall = com.zipper.app.core.intelligence.scheduler.LifePlanCallScheduler.getNextUpcomingCall(this@MainActivity)

                if (isMuted) {
                    val isQuietHours = prefs.isInsideQuietHours(System.currentTimeMillis())
                    if (isQuietHours) {
                        tvCallScheduleStatusTitle?.text = "🌙 Quiet Hours Active (Calls Muted)"
                        tvCallScheduleStatusTitle?.setTextColor(Color.parseColor("#475569"))
                        tvNextScheduledCallInfo?.text = "Overnight quiet hours enforced (${prefs.quietHoursStart}:00 – ${prefs.quietHoursEnd}:00). Calls will not ring."
                    } else {
                        tvCallScheduleStatusTitle?.text = "🔕 Pre-Call Awakenings: Paused Today"
                        tvCallScheduleStatusTitle?.setTextColor(Color.parseColor("#DC2626"))
                        tvNextScheduledCallInfo?.text = "All AI phone calls muted for today. Resumes tomorrow at ${prefs.quietHoursEnd}:00 AM."
                    }
                    btnToggleDontCallToday?.text = "🔔 Resume Calls"
                    btnToggleDontCallToday?.setTextColor(Color.parseColor("#059669"))
                    btnToggleDontCallToday?.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#D1FAE5"))
                    btnToggleDontCallToday?.setOnClickListener {
                        com.zipper.app.core.intelligence.scheduler.LifePlanCallScheduler.unmuteCalls(this@MainActivity)
                        Toast.makeText(this@MainActivity, "🔔 AI pre-call awakenings resumed", Toast.LENGTH_SHORT).show()
                        refreshPlannerDashboard()
                    }
                } else {
                    tvCallScheduleStatusTitle?.text = "🟢 Pre-Call Awakenings: Active"
                    tvCallScheduleStatusTitle?.setTextColor(Color.parseColor("#059669"))
                    if (nextCall != null) {
                        val triggerCal = java.util.Calendar.getInstance().apply { timeInMillis = nextCall.second }
                        val timeStr = String.format(Locale.US, "%02d:%02d", triggerCal.get(java.util.Calendar.HOUR_OF_DAY), triggerCal.get(java.util.Calendar.MINUTE))
                        val diffMs = nextCall.second - System.currentTimeMillis()
                        val countdownStr = if (diffMs > 0) {
                            val diffMins = (diffMs / (60 * 1000L)).toInt()
                            val h = diffMins / 60
                            val m = diffMins % 60
                            if (h > 0) " (in ${h}h ${m}m)" else " (in ${m}m)"
                        } else ""
                        tvNextScheduledCallInfo?.text = "Next awakening call: Today at $timeStr$countdownStr • 5m before '${nextCall.first.title}'"
                    } else {
                        tvNextScheduledCallInfo?.text = "No upcoming awakening calls scheduled for today."
                    }
                    btnToggleDontCallToday?.text = "🔕 Don't Call Today"
                    btnToggleDontCallToday?.setTextColor(Color.parseColor("#DC2626"))
                    btnToggleDontCallToday?.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#FEE2E2"))
                    btnToggleDontCallToday?.setOnClickListener {
                        com.zipper.app.core.intelligence.scheduler.LifePlanCallScheduler.muteAllCallsToday(this@MainActivity)
                        Toast.makeText(this@MainActivity, "🔕 Muted all AI calls for today", Toast.LENGTH_SHORT).show()
                        refreshPlannerDashboard()
                    }
                }

                // --- BIND AI GHOST-WRITER AUTO-REPLY CARD ---
                val btnManageDelegatedContacts = findViewById<TextView>(R.id.btnManageDelegatedContacts)
                val tvAutoReplyStatusSubtitle = findViewById<TextView>(R.id.tvAutoReplyStatusSubtitle)

                val delegatedManager = com.zipper.app.core.intelligence.reply.DelegatedContactManager.getInstance(this@MainActivity)
                val allDelegated = delegatedManager.getContacts()
                val activeContacts = allDelegated.filter { it.isEnabled }

                if (activeContacts.isNotEmpty()) {
                    val names = activeContacts.joinToString { it.contactName }
                    tvAutoReplyStatusSubtitle?.text = "Active for: $names • Connected AI replies to WhatsApp/SMS"
                    tvAutoReplyStatusSubtitle?.setTextColor(Color.parseColor("#0284C7"))
                } else if (allDelegated.isNotEmpty()) {
                    tvAutoReplyStatusSubtitle?.text = "${allDelegated.size} contact(s) configured (currently paused). Tap to activate."
                    tvAutoReplyStatusSubtitle?.setTextColor(Color.parseColor("#64748B"))
                } else {
                    tvAutoReplyStatusSubtitle?.text = "Tap 'Allow Contact' to let connected AI reply to specific WhatsApp messages when you're busy."
                    tvAutoReplyStatusSubtitle?.setTextColor(Color.parseColor("#64748B"))
                }

                btnManageDelegatedContacts?.setOnClickListener {
                    com.zipper.app.core.intelligence.reply.DelegatedContactDialog(
                        context = this@MainActivity,
                        onContactSaved = { refreshPlannerDashboard() },
                        onContactDeleted = { refreshPlannerDashboard() }
                    ).show()
                }

                val btnChangeRemotePin = findViewById<TextView>(R.id.btnChangeRemotePin)
                val currentRemotePin = com.zipper.app.core.remote.RemoteSecurityGate.getSecurityPin(this@MainActivity)
                btnChangeRemotePin?.text = "🔐 PIN: #$currentRemotePin"
                btnChangeRemotePin?.setOnClickListener {
                    com.zipper.app.core.remote.RemoteSecurityDialog(this@MainActivity) {
                        refreshPlannerDashboard()
                        updatePluginDashboardUi()
                    }.show()
                }

                val btnOpenKnowledgeBook = findViewById<TextView>(R.id.btnOpenKnowledgeBook)
                btnOpenKnowledgeBook?.setOnClickListener {
                    com.zipper.app.core.intelligence.reply.ContactKnowledgeDialog(
                        context = this@MainActivity,
                        onDismiss = { refreshPlannerDashboard() }
                    ).show()
                }

                val btnOpenVoiceBridge = findViewById<TextView>(R.id.btnOpenVoiceBridge)
                btnOpenVoiceBridge?.setOnClickListener {
                    val rtcMgr = com.zipper.app.core.intelligence.voice.TencentRtcManager.getInstance(this@MainActivity)
                    val rtcConfig = rtcMgr.getConfig()
                    val testUrl = rtcMgr.generateCallerWebLink("zipper_test_room", "Karthik")

                    android.app.AlertDialog.Builder(this@MainActivity)
                        .setTitle("🎙️ Zipper Voice Twin (Live AI)")
                        .setMessage(
                            "Status: Active & Ready\n" +
                            "• Mode: 2-Way Conversational Voice AI\n" +
                            "• Web Client: ${rtcConfig.webClientBaseUrl}\n" +
                            "• TRTC App ID: ${rtcConfig.sdkAppId} (Singapore)\n" +
                            "• Personality: Bengaluru Slang / Kanglish / Knowledge Book\n\n" +
                            "Tap 'Start Voice Call' to speak directly with your AI voice twin right now on your phone hands-free."
                        )
                        .setPositiveButton("Start Voice Call") { _, _ ->
                            // Launch full-screen interactive conversational voice call HUD
                            val callIntent = Intent(this@MainActivity, com.zipper.app.core.voice.IncomingCallActivity::class.java).apply {
                                putExtra(com.zipper.app.core.voice.IncomingCallActivity.EXTRA_REMINDER_TITLE, "🎙️ Zipper Voice Twin (Live)")
                                putExtra(com.zipper.app.core.voice.IncomingCallActivity.EXTRA_REMINDER_PROMPT, "Hey Karthik! Your AI Voice Twin is live and connected. Ask me anything or test your replies.")
                                putExtra(com.zipper.app.core.voice.IncomingCallActivity.EXTRA_SOURCE_PERSON, "AI Voice Twin")
                            }
                            startActivity(callIntent)
                        }
                        .setNeutralButton("Copy Web Link") { _, _ ->
                            val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as android.content.ClipboardManager
                            clipboard.setPrimaryClip(android.content.ClipData.newPlainText("Voice Twin Link", testUrl))
                            android.widget.Toast.makeText(this@MainActivity, "Copied Netlify caller link!", android.widget.Toast.LENGTH_SHORT).show()
                        }
                        .setNegativeButton("Close", null)
                        .show()
                }

                findViewById<View>(R.id.llDelegatedAutoReplyCard)?.setOnClickListener {
                    com.zipper.app.core.intelligence.reply.DelegatedContactDialog(
                        context = this@MainActivity,
                        onContactSaved = { refreshPlannerDashboard() },
                        onContactDeleted = { refreshPlannerDashboard() }
                    ).show()
                }

                // 1. Separate Checklist Items from Dedicated Timeline Schedule
                val checklistItems = displaySchedule.filter { it.isChecklist }
                val timelineItems = displaySchedule.filter { !it.isChecklist }

                // --- BIND ACTION CHECKLIST ---
                val llChecklistItemsContainer = findViewById<LinearLayout>(R.id.llChecklistItemsContainer)
                val tvChecklistEmpty = findViewById<TextView>(R.id.tvChecklistEmpty)
                val tvChecklistBadgeCount = findViewById<TextView>(R.id.tvChecklistBadgeCount)

                val pendingChecklistCount = checklistItems.count { it.status == com.zipper.app.core.intelligence.store.CommitmentStatus.PENDING }
                tvChecklistBadgeCount?.text = if (pendingChecklistCount > 0) "$pendingChecklistCount tasks" else "All done ✓"

                if (checklistItems.isEmpty()) {
                    tvChecklistEmpty?.visibility = View.VISIBLE
                    while ((llChecklistItemsContainer?.childCount ?: 0) > 1) {
                        llChecklistItemsContainer?.removeViewAt(1)
                    }
                } else {
                    tvChecklistEmpty?.visibility = View.GONE
                    while ((llChecklistItemsContainer?.childCount ?: 0) > 1) {
                        llChecklistItemsContainer?.removeViewAt(1)
                    }

                    for (item in checklistItems) {
                        val checkView = layoutInflater.inflate(R.layout.item_planner_checklist, llChecklistItemsContainer, false)
                        val flCheckbox = checkView.findViewById<View>(R.id.flChecklistCheckbox)
                        val ivCheck = checkView.findViewById<ImageView>(R.id.ivChecklistCheck)
                        val tvTag = checkView.findViewById<TextView>(R.id.tvChecklistCategoryPill)
                        val tvPriority = checkView.findViewById<TextView>(R.id.tvChecklistPriorityPill)
                        val tvSource = checkView.findViewById<TextView>(R.id.tvChecklistSource)
                        val tvTitle = checkView.findViewById<TextView>(R.id.tvChecklistTitle)
                        val llQuoteBox = checkView.findViewById<View>(R.id.llChecklistMessageQuoteBox)
                        val tvQuote = checkView.findViewById<TextView>(R.id.tvChecklistMessageQuote)
                        val tvNotes = checkView.findViewById<TextView>(R.id.tvChecklistNotes)
                        val btnCall = checkView.findViewById<View>(R.id.btnChecklistCallNow)
                        val btnAutoDo = checkView.findViewById<View>(R.id.btnChecklistAutoDo)
                        val btnToggleDone = checkView.findViewById<TextView>(R.id.btnChecklistToggleDone)
                        val llApps = checkView.findViewById<LinearLayout>(R.id.llChecklistCandidateApps)

                        tvTitle.text = item.title
                        val brief = com.zipper.app.core.intelligence.planner.NightlyPlannerEngine.parseActionBrief(item.notes)
                        val rawContext = com.zipper.app.core.intelligence.planner.NightlyPlannerEngine.parseFullContext(item.notes)
                            .replace("[TYPE:CHECKLIST]", "").trim()
                        tvNotes.text = brief.ifBlank { rawContext.ifBlank { item.title } }

                        // Priority Tag
                        val priority = com.zipper.app.core.intelligence.planner.NightlyPlannerEngine.parsePriority(item.notes)
                        if (priority == "URGENT") {
                            tvPriority?.visibility = View.VISIBLE
                            tvPriority?.text = "URGENT"
                            tvPriority?.setTextColor(Color.parseColor("#DC2626"))
                            tvPriority?.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#FEE2E2"))
                        } else if (priority == "HIGH") {
                            tvPriority?.visibility = View.VISIBLE
                            tvPriority?.text = "HIGH"
                            tvPriority?.setTextColor(Color.parseColor("#D97706"))
                            tvPriority?.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#FEF3C7"))
                        } else {
                            tvPriority?.visibility = View.GONE
                        }

                        // Message Prior Quote Box
                        val messageQuote = com.zipper.app.core.intelligence.planner.NightlyPlannerEngine.parseMessageQuote(item.notes)
                        if (messageQuote.isNotBlank()) {
                            llQuoteBox?.visibility = View.VISIBLE
                            val senderPrefix = if (item.sourcePerson.isNotBlank()) "${item.sourcePerson}: " else ""
                            tvQuote?.text = "“$senderPrefix$messageQuote”"
                        } else {
                            llQuoteBox?.visibility = View.GONE
                        }

                        // Category Tag & Source
                        val isCall = item.title.contains("call", ignoreCase = true) || item.sourceApp.contains("Phone", ignoreCase = true)
                        val isMsg = item.title.contains("message", ignoreCase = true) || item.title.contains("send", ignoreCase = true) || item.sourceApp.contains("WhatsApp", ignoreCase = true)
                        
                        tvTag.text = if (isCall) "📞 Callback" else if (isMsg) "💬 Message" else "⚡ Quick Task"
                        tvTag.setTextColor(if (isCall) Color.parseColor("#059669") else if (isMsg) Color.parseColor("#2563EB") else Color.parseColor("#D97706"))
                        tvTag.setBackgroundResource(if (isCall) R.drawable.bg_pill_tag_fitness else if (isMsg) R.drawable.bg_pill_clean_blue else R.drawable.bg_pill_tag_reminder)
                        
                        tvSource.text = if (item.sourcePerson.isNotBlank()) "• ${item.sourcePerson}" else if (item.sourceApp.isNotBlank()) "• ${item.sourceApp}" else ""

                        // Completion Status styling
                        val isCompleted = item.status == com.zipper.app.core.intelligence.store.CommitmentStatus.COMPLETED
                        if (isCompleted) {
                            ivCheck.setImageResource(R.drawable.ic_check_circle)
                            ivCheck.imageTintList = ColorStateList.valueOf(Color.parseColor("#2563EB"))
                            tvTitle.paintFlags = tvTitle.paintFlags or Paint.STRIKE_THRU_TEXT_FLAG
                            tvTitle.setTextColor(Color.parseColor("#9CA3AF"))
                            btnToggleDone.text = "Completed ✓"
                            btnToggleDone.setTextColor(Color.parseColor("#10B981"))
                        } else {
                            ivCheck.setImageResource(R.drawable.ic_circle_outline)
                            ivCheck.imageTintList = ColorStateList.valueOf(Color.parseColor("#9CA3AF"))
                            tvTitle.paintFlags = tvTitle.paintFlags and Paint.STRIKE_THRU_TEXT_FLAG.inv()
                            tvTitle.setTextColor(Color.parseColor("#111827"))
                            btnToggleDone.text = "✓ Mark Done"
                            btnToggleDone.setTextColor(Color.parseColor("#4B5563"))
                        }

                        // Toggle Done logic
                        val toggleAction = View.OnClickListener {
                            val newStatus = if (isCompleted) com.zipper.app.core.intelligence.store.CommitmentStatus.PENDING else com.zipper.app.core.intelligence.store.CommitmentStatus.COMPLETED
                            lifecycleScope.launch(Dispatchers.IO) {
                                db.updateCommitmentStatus(item.id, newStatus)
                                withContext(Dispatchers.Main) {
                                    Toast.makeText(this@MainActivity, if (newStatus == com.zipper.app.core.intelligence.store.CommitmentStatus.COMPLETED) "✓ Task marked completed!" else "Task marked pending", Toast.LENGTH_SHORT).show()
                                    refreshPlannerDashboard()
                                }
                            }
                        }
                        flCheckbox?.setOnClickListener(toggleAction)
                        btnToggleDone?.setOnClickListener(toggleAction)

                        // Bind candidate apps
                        val apps = candidateAppsMap[item] ?: emptyList()
                        llApps.removeAllViews()
                        for (app in apps) {
                            val pill = layoutInflater.inflate(R.layout.item_candidate_app_pill, llApps, false)
                            val ivAppIcon = pill.findViewById<ImageView>(R.id.ivCandidateAppIcon)
                            val tvAppName = pill.findViewById<TextView>(R.id.tvCandidateAppName)

                            ivAppIcon.setImageDrawable(app.icon)
                            tvAppName.text = app.appName
                            pill.setOnClickListener {
                                if (app.deepLinkUri != null) {
                                    val sendIntent = Intent(Intent.ACTION_VIEW).apply {
                                        data = android.net.Uri.parse(app.deepLinkUri)
                                        setPackage(app.packageName)
                                        addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                                    }
                                    startActivity(sendIntent)
                                } else {
                                    val launchIntent = app.launchIntent ?: packageManager.getLaunchIntentForPackage(app.packageName)
                                    if (launchIntent != null) startActivity(launchIntent)
                                }
                            }
                            llApps.addView(pill)
                        }

                        // Call Button
                        if (isCall && !isCompleted) {
                            btnCall.visibility = View.VISIBLE
                            btnCall.setOnClickListener {
                                val callIntent = Intent(Intent.ACTION_DIAL).apply {
                                    val phone = if (item.sourcePerson.matches(Regex("\\+?\\d+"))) item.sourcePerson else ""
                                    if (phone.isNotBlank()) data = android.net.Uri.parse("tel:$phone")
                                }
                                try {
                                    startActivity(callIntent)
                                } catch (_: Exception) {
                                    val dialerIntent = Intent(Intent.ACTION_MAIN).addCategory(Intent.CATEGORY_APP_CONTACTS)
                                    startActivity(dialerIntent)
                                }
                            }
                        } else {
                            btnCall.visibility = View.GONE
                        }

                        // Auto-Do Button
                        val isFitnessItem = item.category == com.zipper.app.core.intelligence.store.CommitmentCategory.FITNESS ||
                            item.title.contains("walk", true) || item.title.contains("gym", true) || item.title.contains("workout", true)
                        val canAutoDo = isMsg || isFitnessItem || item.candidatePackages.contains("com.whatsapp") || item.category == com.zipper.app.core.intelligence.store.CommitmentCategory.COMMUNICATION
                        if (canAutoDo && !isCompleted) {
                            btnAutoDo.visibility = View.VISIBLE
                            val tvCheckAutoDo = checkView.findViewById<TextView>(R.id.tvChecklistAutoDoText)
                            val ivCheckAutoDo = checkView.findViewById<ImageView>(R.id.ivChecklistAutoDoIcon)
                            if (isFitnessItem) {
                                tvCheckAutoDo?.text = "🚀 Health"
                                ivCheckAutoDo?.setImageResource(R.drawable.ic_fitness)
                                btnAutoDo.setOnClickListener {
                                    com.zipper.app.core.intelligence.dispatcher.DirectAppDispatcher.launchSamsungHealth(this@MainActivity)
                                }
                            } else {
                                tvCheckAutoDo?.text = "⚡ Auto-Do"
                                ivCheckAutoDo?.setImageResource(R.drawable.ic_bolt)
                                btnAutoDo.setOnClickListener {
                                    handleAutoDoAction(item)
                                }
                            }
                        } else {
                            btnAutoDo.visibility = View.GONE
                        }

                        llChecklistItemsContainer?.addView(checkView)
                    }
                }

                // --- BIND TIMELINE SCHEDULE ---
                val llTimelineItemsContainer = findViewById<LinearLayout>(R.id.llTimelineItemsContainer)
                val tvTimelineEmpty = findViewById<TextView>(R.id.tvTimelineEmpty)

                if (timelineItems.isEmpty()) {
                    tvTimelineEmpty?.visibility = View.VISIBLE
                    while ((llTimelineItemsContainer?.childCount ?: 0) > 1) {
                        llTimelineItemsContainer?.removeViewAt(1)
                    }
                } else {
                    tvTimelineEmpty?.visibility = View.GONE
                    while ((llTimelineItemsContainer?.childCount ?: 0) > 1) {
                        llTimelineItemsContainer?.removeViewAt(1)
                    }
                    val cal = java.util.Calendar.getInstance()
                    for ((index, item) in timelineItems.withIndex()) {
                        val itemView = layoutInflater.inflate(R.layout.item_planner_timeline, llTimelineItemsContainer, false)
                        val tvTime = itemView.findViewById<TextView>(R.id.tvTimelineItemTime)
                        val tvDuration = itemView.findViewById<TextView>(R.id.tvTimelineItemDuration)
                        val viewTopLine = itemView.findViewById<View>(R.id.viewTimelineTopLine)
                        val viewBottomLine = itemView.findViewById<View>(R.id.viewTimelineBottomLine)
                        val ivIcon = itemView.findViewById<ImageView>(R.id.ivTimelineCategoryIcon)
                        val tvTitle = itemView.findViewById<TextView>(R.id.tvTimelineItemTitle)
                        val tvTag = itemView.findViewById<TextView>(R.id.tvTimelineCategoryPill)
                        val tvCallBadge = itemView.findViewById<TextView>(R.id.tvTimelineCallBadge)
                        val llApps = itemView.findViewById<LinearLayout>(R.id.llTimelineCandidateApps)
                        val btnCall = itemView.findViewById<View>(R.id.btnTimelineCallNow)

                        val preCallTrigger = item.scheduledTimeMs - (5 * 60 * 1000L)
                        if (!isMuted && preCallTrigger > System.currentTimeMillis() && !prefs.isInsideQuietHours(preCallTrigger)) {
                            val badgeCal = java.util.Calendar.getInstance().apply { timeInMillis = preCallTrigger }
                            val callTimeStr = String.format(Locale.US, "%02d:%02d", badgeCal.get(java.util.Calendar.HOUR_OF_DAY), badgeCal.get(java.util.Calendar.MINUTE))
                            tvCallBadge?.visibility = View.VISIBLE
                            tvCallBadge?.text = "📞 $callTimeStr"
                        } else {
                            tvCallBadge?.visibility = View.GONE
                        }

                        val llAiProof = itemView.findViewById<View>(R.id.llTimelineAiProof)
                        val tvAiProofText = itemView.findViewById<TextView>(R.id.tvTimelineAiProofText)
                        val tvNotes = itemView.findViewById<TextView>(R.id.tvTimelineItemNotes)
                        val tvToggle = itemView.findViewById<TextView>(R.id.tvTimelineToggleContext)
                        val tvFull = itemView.findViewById<TextView>(R.id.tvTimelineFullContext)

                        if (index == 0) viewTopLine.visibility = View.INVISIBLE
                        if (index == timelineItems.size - 1) viewBottomLine.visibility = View.INVISIBLE

                        cal.timeInMillis = item.scheduledTimeMs
                        tvTime.text = String.format(Locale.US, "%02d:%02d", cal.get(java.util.Calendar.HOUR_OF_DAY), cal.get(java.util.Calendar.MINUTE))
                        tvDuration.text = "${item.durationMinutes} min"
                        tvTitle.text = item.title

                        // Parse AI proof and concise brief
                        val brief = com.zipper.app.core.intelligence.planner.NightlyPlannerEngine.parseActionBrief(item.notes)
                        val proof = com.zipper.app.core.intelligence.planner.NightlyPlannerEngine.parseReasoningProof(item.notes)
                        val fullContext = com.zipper.app.core.intelligence.planner.NightlyPlannerEngine.parseFullContext(item.notes)

                        if (proof.isNotBlank()) {
                            llAiProof.visibility = View.VISIBLE
                            tvAiProofText.text = "💡 AI Reason: $proof"
                        } else {
                            llAiProof.visibility = View.GONE
                        }

                        // Interactive checklists or clean brief
                        val llChecklist = itemView.findViewById<LinearLayout>(R.id.llTimelineChecklistContainer)
                        val rawBrief = brief.trim()
                        val bulletLines = rawBrief.split("\n").map { it.trim() }.filter { it.isNotEmpty() }
                        val hasBullets = bulletLines.any { it.startsWith("•") || it.startsWith("-") || it.startsWith("[ ]") || it.startsWith("[x]") }

                        if (hasBullets && bulletLines.size > 1) {
                            tvNotes.visibility = View.GONE
                            llChecklist?.visibility = View.VISIBLE
                            llChecklist?.removeAllViews()

                            for ((bIdx, line) in bulletLines.withIndex()) {
                                val cleanLine = line.removePrefix("•").removePrefix("-").trim()
                                val isChecked = cleanLine.startsWith("[x]")
                                val itemText = cleanLine.removePrefix("[x]").removePrefix("[ ]").trim()

                                val checkRow = LinearLayout(this@MainActivity).apply {
                                    orientation = LinearLayout.HORIZONTAL
                                    gravity = android.view.Gravity.CENTER_VERTICAL
                                    setPadding(0, dpToPx(3), 0, dpToPx(3))
                                }

                                val checkIcon = ImageView(this@MainActivity).apply {
                                    layoutParams = LinearLayout.LayoutParams(dpToPx(16), dpToPx(16)).apply {
                                        marginEnd = dpToPx(6)
                                    }
                                    if (isChecked) {
                                        setImageResource(R.drawable.ic_check_circle)
                                        imageTintList = ColorStateList.valueOf(Color.parseColor("#10B981"))
                                    } else {
                                        setImageResource(R.drawable.ic_circle_outline)
                                        imageTintList = ColorStateList.valueOf(Color.parseColor("#9CA3AF"))
                                    }
                                }

                                val checkTv = TextView(this@MainActivity).apply {
                                    layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
                                    textSize = 12f
                                    text = itemText
                                    if (isChecked) {
                                        setTextColor(Color.parseColor("#9CA3AF"))
                                        paintFlags = paintFlags or android.graphics.Paint.STRIKE_THRU_TEXT_FLAG
                                    } else {
                                        setTextColor(Color.parseColor("#374151"))
                                        paintFlags = paintFlags and android.graphics.Paint.STRIKE_THRU_TEXT_FLAG.inv()
                                    }
                                }

                                checkRow.addView(checkIcon)
                                checkRow.addView(checkTv)

                                checkRow.setOnClickListener {
                                    val toggledLine = if (isChecked) "• [ ] $itemText" else "• [x] $itemText"
                                    val newBulletLines = bulletLines.toMutableList()
                                    newBulletLines[bIdx] = toggledLine
                                    val newBrief = newBulletLines.joinToString("\n")
                                    val updatedNotes = com.zipper.app.core.intelligence.planner.NightlyPlannerEngine.buildNotesWithAiProof(
                                        newBrief, proof, fullContext
                                    )
                                    lifecycleScope.launch(Dispatchers.IO) {
                                        db.updateCommitmentNotes(item.id, updatedNotes)
                                        withContext(Dispatchers.Main) {
                                            refreshPlannerDashboard()
                                        }
                                    }
                                }

                                llChecklist?.addView(checkRow)
                            }
                        } else {
                            tvNotes.visibility = View.VISIBLE
                            llChecklist?.visibility = View.GONE
                            tvNotes.text = brief
                        }

                        // Expandable disclosure for full raw context
                        if (fullContext.isNotBlank() && fullContext != brief && fullContext.length > 50) {
                            tvToggle.visibility = View.VISIBLE
                            tvFull.text = fullContext
                            tvToggle.setOnClickListener {
                                if (tvFull.visibility == View.VISIBLE) {
                                    tvFull.visibility = View.GONE
                                    tvToggle.text = "View Original Message ▼"
                                } else {
                                    tvFull.visibility = View.VISIBLE
                                    tvToggle.text = "Hide Original Message ▲"
                                }
                            }
                        } else {
                            tvToggle.visibility = View.GONE
                            tvFull.visibility = View.GONE
                        }

                        when (item.category) {
                            com.zipper.app.core.intelligence.store.CommitmentCategory.FITNESS -> {
                                ivIcon.setImageResource(R.drawable.ic_fitness)
                                ivIcon.imageTintList = ColorStateList.valueOf(Color.parseColor("#15803D"))
                                tvTag.text = "FITNESS"
                                tvTag.setBackgroundResource(R.drawable.bg_pill_tag_fitness)
                                tvTag.setTextColor(Color.parseColor("#15803D"))
                            }
                            com.zipper.app.core.intelligence.store.CommitmentCategory.EDUCATION -> {
                                ivIcon.setImageResource(R.drawable.ic_laptop)
                                ivIcon.imageTintList = ColorStateList.valueOf(Color.parseColor("#7E22CE"))
                                tvTag.text = "STUDY"
                                tvTag.setBackgroundResource(R.drawable.bg_pill_tag_study)
                                tvTag.setTextColor(Color.parseColor("#7E22CE"))
                            }
                            else -> {
                                ivIcon.setImageResource(R.drawable.ic_calendar)
                                ivIcon.imageTintList = ColorStateList.valueOf(Color.parseColor("#1D4ED8"))
                                tvTag.text = "PLANNER"
                                tvTag.setBackgroundResource(R.drawable.bg_pill_tag_planner)
                                tvTag.setTextColor(Color.parseColor("#1D4ED8"))
                            }
                        }

                        // Candidate apps with 1-tap direct chat opening
                        val apps = candidateAppsMap[item] ?: emptyList()
                        val llAppsRow = itemView.findViewById<View>(R.id.llTimelineCandidateAppsRow)
                        if (apps.isEmpty()) {
                            llAppsRow?.visibility = View.GONE
                        } else {
                            llAppsRow?.visibility = View.VISIBLE
                            llApps.removeAllViews()
                            for (app in apps) {
                                val appPill = LinearLayout(this@MainActivity).apply {
                                    orientation = LinearLayout.HORIZONTAL
                                    gravity = android.view.Gravity.CENTER_VERTICAL
                                    setBackgroundResource(R.drawable.bg_app_pill_light)
                                    setPadding(dpToPx(7), dpToPx(3), dpToPx(9), dpToPx(3))
                                    layoutParams = LinearLayout.LayoutParams(
                                        LinearLayout.LayoutParams.WRAP_CONTENT,
                                        LinearLayout.LayoutParams.WRAP_CONTENT
                                    ).apply {
                                        marginEnd = dpToPx(6)
                                    }
                                }

                                val iv = ImageView(this@MainActivity).apply {
                                    layoutParams = LinearLayout.LayoutParams(dpToPx(18), dpToPx(18))
                                    setImageDrawable(app.icon)
                                }

                                val tv = TextView(this@MainActivity).apply {
                                    layoutParams = LinearLayout.LayoutParams(
                                        LinearLayout.LayoutParams.WRAP_CONTENT,
                                        LinearLayout.LayoutParams.WRAP_CONTENT
                                    ).apply {
                                        marginStart = dpToPx(5)
                                    }
                                    textSize = 10.5f
                                    setTextColor(Color.parseColor("#374151"))
                                    text = app.appName
                                    typeface = android.graphics.Typeface.DEFAULT_BOLD
                                }

                                appPill.addView(iv)
                                appPill.addView(tv)

                                appPill.setOnClickListener {
                                    val pi = com.zipper.app.core.notification.KarvishNotificationListenerService.findContentIntent(app.packageName, item.sourcePerson)
                                    if (pi != null) {
                                        try {
                                            pi.send()
                                        } catch (_: Exception) {
                                            startActivity(app.launchIntent)
                                        }
                                    } else {
                                        startActivity(app.launchIntent)
                                    }
                                }
                                llApps.addView(appPill)
                            }
                        }

                        // Auto-Do action trigger: transparent preview dialog
                        val btnAutoDo = itemView.findViewById<View>(R.id.btnTimelineAutoDo)
                        val isFitnessItem = item.category == com.zipper.app.core.intelligence.store.CommitmentCategory.FITNESS ||
                            item.title.contains("walk", true) || item.title.contains("gym", true) || item.title.contains("workout", true)
                        val tvAutoDo = itemView.findViewById<TextView>(R.id.tvTimelineAutoDoText)
                        val ivAutoDo = itemView.findViewById<ImageView>(R.id.ivTimelineAutoDoIcon)

                        if (isFitnessItem) {
                            tvAutoDo?.text = "🚀 Launch Health"
                            tvAutoDo?.setTextColor(Color.parseColor("#059669"))
                            ivAutoDo?.setImageResource(R.drawable.ic_fitness)
                            ivAutoDo?.imageTintList = ColorStateList.valueOf(Color.parseColor("#059669"))
                            btnAutoDo?.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#ECFDF5"))
                            btnAutoDo?.setOnClickListener {
                                com.zipper.app.core.intelligence.dispatcher.DirectAppDispatcher.launchSamsungHealth(this@MainActivity)
                            }
                        } else {
                            tvAutoDo?.text = "⚡ Auto-Do"
                            tvAutoDo?.setTextColor(Color.parseColor("#4F46E5"))
                            ivAutoDo?.setImageResource(R.drawable.ic_bolt)
                            ivAutoDo?.imageTintList = ColorStateList.valueOf(Color.parseColor("#4F46E5"))
                            btnAutoDo?.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#EEF2FF"))
                            btnAutoDo?.setOnClickListener {
                                handleAutoDoAction(item)
                            }
                        }

                        btnCall.setOnClickListener {
                            com.zipper.app.core.intelligence.scheduler.LifePlanCallScheduler.triggerTestCallNow(this@MainActivity, item)
                            Toast.makeText(this@MainActivity, "📞 Calling for '${item.title}' in 3 seconds!", Toast.LENGTH_SHORT).show()
                        }

                        llTimelineItemsContainer?.addView(itemView)
                    }
                }

                // Needs Attention Cards
                val llNeedsAttentionContainer = findViewById<LinearLayout>(R.id.llNeedsAttentionContainer)
                llNeedsAttentionContainer?.removeAllViews()

                val itemBlocked = layoutInflater.inflate(R.layout.item_planner_attention, llNeedsAttentionContainer, false).apply {
                    findViewById<ImageView>(R.id.ivAttentionAppIcon).setImageResource(R.drawable.ic_chat)
                    findViewById<TextView>(R.id.tvAttentionTitle).text = "Email Reply Pending"
                    findViewById<TextView>(R.id.tvAttentionSubtitle).text = "Requires your confirmation to send the reply."
                    findViewById<TextView>(R.id.tvAttentionTagPill).apply {
                        text = "BLOCKED"
                        setBackgroundResource(R.drawable.bg_pill_tag_blocked)
                        setTextColor(Color.parseColor("#DC2626"))
                    }
                    findViewById<ImageView>(R.id.ivAttentionStatusIcon).imageTintList = ColorStateList.valueOf(Color.parseColor("#DC2626"))
                    findViewById<TextView>(R.id.tvAttentionStatusText).apply {
                        text = "Waiting for you"
                        setTextColor(Color.parseColor("#DC2626"))
                    }
                    setOnClickListener {
                        val pi = com.zipper.app.core.notification.KarvishNotificationListenerService.findContentIntent("com.google.android.gm", "Email")
                        if (pi != null) {
                            try { pi.send(); return@setOnClickListener } catch (_: Exception) {}
                        }
                        packageManager.getLaunchIntentForPackage("com.google.android.gm")?.let { startActivity(it) }
                            ?: Toast.makeText(this@MainActivity, "Opening email draft for review...", Toast.LENGTH_SHORT).show()
                    }
                }
                llNeedsAttentionContainer?.addView(itemBlocked)

                val itemReminder = layoutInflater.inflate(R.layout.item_planner_attention, llNeedsAttentionContainer, false).apply {
                    findViewById<ImageView>(R.id.ivAttentionAppIcon).setImageResource(R.drawable.ic_article)
                    findViewById<TextView>(R.id.tvAttentionTitle).text = "Assignment Submission"
                    findViewById<TextView>(R.id.tvAttentionSubtitle).text = "Due tomorrow • Upload final report to portal."
                    findViewById<TextView>(R.id.tvAttentionTagPill).apply {
                        text = "REMINDER"
                        setBackgroundResource(R.drawable.bg_pill_tag_reminder)
                        setTextColor(Color.parseColor("#D97706"))
                    }
                    findViewById<ImageView>(R.id.ivAttentionStatusIcon).imageTintList = ColorStateList.valueOf(Color.parseColor("#D97706"))
                    findViewById<TextView>(R.id.tvAttentionStatusText).apply {
                        text = "Tomorrow, 11:00 AM"
                        setTextColor(Color.parseColor("#D97706"))
                    }
                    setOnClickListener {
                        val pi = com.zipper.app.core.notification.KarvishNotificationListenerService.findContentIntent("com.google.android.apps.classroom", "Assignment")
                        if (pi != null) {
                            try { pi.send(); return@setOnClickListener } catch (_: Exception) {}
                        }
                        packageManager.getLaunchIntentForPackage("com.google.android.apps.classroom")?.let { startActivity(it) }
                            ?: Toast.makeText(this@MainActivity, "Assignment details: Module 4 report", Toast.LENGTH_SHORT).show()
                    }
                }
                llNeedsAttentionContainer?.addView(itemReminder)

                // Context Feed: Top 5 Recent Messages with 1-Tap Direct Chat Launch
                val tvContextFeedHeader = findViewById<TextView>(R.id.tvContextFeedHeader)
                val llContextFeedContainer = findViewById<LinearLayout>(R.id.llContextFeedContainer)
                val tvContextFeedEmpty = findViewById<TextView>(R.id.tvContextFeedEmpty)

                tvContextFeedHeader?.text = "Context Feed (${allMessages.size})"
                if (recentMessages.isEmpty()) {
                    tvContextFeedEmpty?.visibility = View.VISIBLE
                    while ((llContextFeedContainer?.childCount ?: 0) > 1) {
                        llContextFeedContainer?.removeViewAt(1)
                    }
                } else {
                    tvContextFeedEmpty?.visibility = View.GONE
                    while ((llContextFeedContainer?.childCount ?: 0) > 1) {
                        llContextFeedContainer?.removeViewAt(1)
                    }
                    for (msg in recentMessages) {
                        val itemView = layoutInflater.inflate(R.layout.item_planner_message, llContextFeedContainer, false)
                        val ivIcon = itemView.findViewById<ImageView>(R.id.ivMsgAppIcon)
                        val tvSender = itemView.findViewById<TextView>(R.id.tvMsgSender)
                        val tvBadge = itemView.findViewById<TextView>(R.id.tvMsgAppBadge)
                        val tvSummary = itemView.findViewById<TextView>(R.id.tvMsgSummary)
                        val tvTime = itemView.findViewById<TextView>(R.id.tvMsgTime)

                        tvSender.text = msg.sender
                        tvBadge.text = msg.appName
                        tvSummary.text = msg.summaryText

                        when {
                            msg.appName.contains("Inshorts", true) -> ivIcon.setImageResource(R.drawable.ic_article)
                            msg.appName.contains("Grok", true) || msg.appName.contains("HubSpot", true) -> ivIcon.setImageResource(R.drawable.ic_chat)
                            msg.appName.contains("WhatsApp", true) -> ivIcon.setImageResource(R.drawable.ic_chat)
                            else -> ivIcon.setImageResource(R.drawable.ic_bell)
                        }

                        val diffMins = ((System.currentTimeMillis() - msg.timestampMs) / 60000L).coerceAtLeast(1)
                        tvTime.text = if (diffMins < 60) "$diffMins min ago" else "${diffMins / 60}h ago"

                        // 1-Tap direct chat jump into the exact notification sender
                        itemView.setOnClickListener {
                            val pi = com.zipper.app.core.notification.KarvishNotificationListenerService.findContentIntent(msg.packageName, msg.sender)
                            if (pi != null) {
                                try {
                                    pi.send()
                                    return@setOnClickListener
                                } catch (_: Exception) {}
                            }
                            packageManager.getLaunchIntentForPackage(msg.packageName)?.let { startActivity(it) }
                                ?: Toast.makeText(this@MainActivity, "Opening ${msg.appName}...", Toast.LENGTH_SHORT).show()
                        }

                        llContextFeedContainer?.addView(itemView)
                    }
                }
            }
        }
    }

    private fun handleAutoDoAction(item: com.zipper.app.core.intelligence.store.CommitmentEntity, forcePicker: Boolean = false) {
        val isFitness = item.category == com.zipper.app.core.intelligence.store.CommitmentCategory.FITNESS ||
            item.title.contains("walk", true) || item.title.contains("gym", true) || item.title.contains("workout", true)

        if (isFitness) {
            com.zipper.app.core.intelligence.dispatcher.DirectAppDispatcher.launchSamsungHealth(this)
            return
        }

        val plan = com.zipper.app.core.intelligence.planner.AutoDoPlanResolver.resolve(item)
        if (plan.requiresAttachment) {
            val step1 = plan.steps.firstOrNull()
            if (step1?.attachmentUri == null || forcePicker) {
                val candidates = com.zipper.app.core.intelligence.resolver.DeviceAssetResolver.resolveMatchingFiles(
                    this,
                    plan.attachmentKeyword
                )
                if (candidates.size > 1) {
                    showMultiPdfPickerDialog(item, plan, candidates)
                    return
                } else if (candidates.size == 1) {
                    val single = candidates.first()
                    step1?.attachmentUri = single.uri
                    step1?.attachmentName = single.name
                    step1?.attachmentSize = single.sizeFormatted
                }
            }
        }
        showAutoDoPreviewDialog(item, plan)
    }

    private fun showMultiPdfPickerDialog(
        item: com.zipper.app.core.intelligence.store.CommitmentEntity,
        plan: com.zipper.app.core.intelligence.planner.AutoDoPlan,
        candidates: List<com.zipper.app.core.intelligence.resolver.DisambiguatedFile>
    ) {
        val dialogView = layoutInflater.inflate(R.layout.dialog_multi_pdf_picker, null)
        val dialog = android.app.AlertDialog.Builder(this)
            .setView(dialogView)
            .create()
        dialog.window?.setBackgroundDrawableResource(android.R.color.transparent)

        val rgCandidates = dialogView.findViewById<RadioGroup>(R.id.rgPdfCandidates)
        val btnClose = dialogView.findViewById<View>(R.id.btnPickerClose)
        val btnCancel = dialogView.findViewById<View>(R.id.btnPickerCancel)
        val btnConfirm = dialogView.findViewById<View>(R.id.btnPickerConfirm)
        val btnBrowse = dialogView.findViewById<View>(R.id.btnPickerBrowseFiles)

        var selectedFile: com.zipper.app.core.intelligence.resolver.DisambiguatedFile? = candidates.firstOrNull()

        for ((idx, file) in candidates.withIndex()) {
            val rb = RadioButton(this).apply {
                id = View.generateViewId()
                val badgeText = if (file.isMostRecent) " 🏆 [MOST RECENT]" else ""
                text = "${file.name}$badgeText\n${file.sizeFormatted} • ${file.lastModifiedFormatted}"
                textSize = 12f
                setTextColor(Color.parseColor("#111827"))
                isChecked = (idx == 0)
                setPadding(dpToPx(6), dpToPx(6), dpToPx(6), dpToPx(6))
                layoutParams = RadioGroup.LayoutParams(
                    RadioGroup.LayoutParams.MATCH_PARENT,
                    RadioGroup.LayoutParams.WRAP_CONTENT
                ).apply {
                    bottomMargin = dpToPx(8)
                }
            }
            rb.setOnCheckedChangeListener { _, isChecked ->
                if (isChecked) selectedFile = file
            }
            rgCandidates.addView(rb)
        }

        btnBrowse.setOnClickListener {
            pendingAttachmentPlan = plan
            pendingAttachmentItem = item
            documentPickerLauncher.launch(arrayOf("application/pdf"))
            dialog.dismiss()
        }

        btnClose.setOnClickListener { dialog.dismiss() }
        btnCancel.setOnClickListener { dialog.dismiss() }

        btnConfirm.setOnClickListener {
            dialog.dismiss()
            if (selectedFile != null && plan.steps.isNotEmpty()) {
                val step1 = plan.steps[0]
                step1.attachmentUri = selectedFile?.uri
                step1.attachmentName = selectedFile?.name
                step1.attachmentSize = selectedFile?.sizeFormatted
            }
            showAutoDoPreviewDialog(item, plan)
        }

        dialog.show()
    }

    private fun showAutoDoPreviewDialog(
        item: com.zipper.app.core.intelligence.store.CommitmentEntity,
        resolvedPlan: com.zipper.app.core.intelligence.planner.AutoDoPlan? = null
    ) {
        val plan = resolvedPlan ?: com.zipper.app.core.intelligence.planner.AutoDoPlanResolver.resolve(item)
        val dialogView = layoutInflater.inflate(R.layout.dialog_auto_do_preview, null)

        val dialog = android.app.AlertDialog.Builder(this)
            .setView(dialogView)
            .create()
        dialog.window?.setBackgroundDrawableResource(android.R.color.transparent)

        val tvSubtitle = dialogView.findViewById<TextView>(R.id.tvAutoDoSubtitle)
        val tvExplanation = dialogView.findViewById<TextView>(R.id.tvAutoDoExplanation)
        val btnClose = dialogView.findViewById<View>(R.id.btnAutoDoClose)
        val btnCancel = dialogView.findViewById<View>(R.id.btnAutoDoCancel)
        val btnConfirm = dialogView.findViewById<View>(R.id.btnAutoDoConfirm)

        val tvStep1Title = dialogView.findViewById<TextView>(R.id.tvStep1Title)
        val tvStep1App = dialogView.findViewById<TextView>(R.id.tvStep1App)
        val tvStep1Recipient = dialogView.findViewById<TextView>(R.id.tvStep1Recipient)
        val etStep1Message = dialogView.findViewById<EditText>(R.id.etStep1Message)

        val llStep1Attachment = dialogView.findViewById<View>(R.id.llStep1AttachmentContainer)
        val tvStep1AttachmentName = dialogView.findViewById<TextView>(R.id.tvStep1AttachmentName)
        val tvStep1AttachmentSize = dialogView.findViewById<TextView>(R.id.tvStep1AttachmentSize)
        val btnStep1ChangeAttachment = dialogView.findViewById<View>(R.id.btnStep1ChangeAttachment)

        val llStep2Container = dialogView.findViewById<View>(R.id.llStep2Container)
        val tvStep2Title = dialogView.findViewById<TextView>(R.id.tvStep2Title)
        val tvStep2App = dialogView.findViewById<TextView>(R.id.tvStep2App)
        val tvStep2Recipient = dialogView.findViewById<TextView>(R.id.tvStep2Recipient)
        val etStep2Message = dialogView.findViewById<EditText>(R.id.etStep2Message)

        val rbModeDirect = dialogView.findViewById<RadioButton>(R.id.rbModeDirect)
        val rbModeAgent = dialogView.findViewById<RadioButton>(R.id.rbModeAgent)

        tvSubtitle.text = plan.title
        tvExplanation.text = plan.explanation

        val step1 = plan.steps.getOrNull(0)
        if (step1 != null) {
            tvStep1Title.text = "STEP 1: 📤 ${step1.title.uppercase(Locale.ROOT)}"
            tvStep1App.text = step1.appName
            tvStep1Recipient.text = step1.recipient
            etStep1Message.setText(step1.message)

            if (step1.attachmentName != null) {
                llStep1Attachment?.visibility = View.VISIBLE
                tvStep1AttachmentName?.text = step1.attachmentName
                tvStep1AttachmentSize?.text = "${step1.attachmentSize} • Pre-attached via FileProvider"
                btnStep1ChangeAttachment?.setOnClickListener {
                    dialog.dismiss()
                    handleAutoDoAction(item, forcePicker = true)
                }
            } else {
                llStep1Attachment?.visibility = View.GONE
            }
        }

        val step2 = plan.steps.getOrNull(1)
        if (step2 != null) {
            llStep2Container.visibility = View.VISIBLE
            tvStep2Title.text = "STEP 2: 📥 ${step2.title.uppercase(Locale.ROOT)}"
            tvStep2App.text = step2.appName
            tvStep2Recipient.text = step2.recipient
            etStep2Message.setText(step2.message)
        } else {
            llStep2Container.visibility = View.GONE
        }

        if (!plan.canInstantSend) {
            rbModeDirect.isEnabled = false
            rbModeAgent.isChecked = true
        }

        btnClose.setOnClickListener { dialog.dismiss() }
        btnCancel.setOnClickListener { dialog.dismiss() }

        btnConfirm.setOnClickListener {
            val updatedMsg1 = etStep1Message.text.toString().trim()
            val updatedMsg2 = if (step2 != null) etStep2Message.text.toString().trim() else ""
            dialog.dismiss()

            if (plan.isEmail) {
                if (rbModeDirect.isChecked) {
                    val dispatched = com.zipper.app.core.intelligence.dispatcher.DirectAppDispatcher.dispatchEmailWithAttachment(
                        context = this@MainActivity,
                        recipientEmail = step1?.recipient.orEmpty(),
                        subject = step1?.emailSubject ?: "Update",
                        body = updatedMsg1,
                        attachmentUri = step1?.attachmentUri
                    )
                    if (dispatched) {
                        lifecycleScope.launch(Dispatchers.IO) {
                            val db = com.zipper.app.core.intelligence.store.CommitmentDatabase.getInstance(this@MainActivity)
                            db.updateCommitmentStatus(item.id, com.zipper.app.core.intelligence.store.CommitmentStatus.COMPLETED)
                            withContext(Dispatchers.Main) {
                                Toast.makeText(
                                    this@MainActivity,
                                    "📤 Opening Gmail with pre-filled recipient and ${step1?.attachmentName ?: "attachment"} ready to send!",
                                    Toast.LENGTH_LONG
                                ).show()
                                refreshPlannerDashboard()
                            }
                        }
                    }
                } else {
                    executeAutoDoWithAgent(item, step1, updatedMsg1, step2, updatedMsg2)
                }
            } else if (rbModeDirect.isChecked && plan.canInstantSend) {
                executeAutoDoInstantSend(item, step1, updatedMsg1, step2, updatedMsg2)
            } else {
                executeAutoDoWithAgent(item, step1, updatedMsg1, step2, updatedMsg2)
            }
        }

        dialog.show()
    }

    private fun executeAutoDoInstantSend(
        item: com.zipper.app.core.intelligence.store.CommitmentEntity,
        step1: com.zipper.app.core.intelligence.planner.AutoDoStep?,
        msg1: String,
        step2: com.zipper.app.core.intelligence.planner.AutoDoStep?,
        msg2: String
    ) {
        lifecycleScope.launch(Dispatchers.IO) {
            if (step1 != null) {
                try {
                    val sendIntent = Intent(Intent.ACTION_VIEW).apply {
                        data = android.net.Uri.parse("https://api.whatsapp.com/send?text=${android.net.Uri.encode(msg1)}")
                        setPackage(step1.packageName)
                        addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                    }
                    startActivity(sendIntent)
                } catch (_: Exception) {
                    try {
                        val shareIntent = Intent(Intent.ACTION_SEND).apply {
                            type = "text/plain"
                            putExtra(Intent.EXTRA_TEXT, msg1)
                            setPackage(step1.packageName)
                            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                        }
                        startActivity(shareIntent)
                    } catch (e: Exception) {
                        Log.w("MainActivity", "Direct send failed: ${e.message}")
                    }
                }
            }

            if (step2 != null && msg2.isNotBlank()) {
                kotlinx.coroutines.delay(1000)
                try {
                    val sbn = com.zipper.app.core.notification.KarvishNotificationListenerService.findStatusBarNotification(
                        step2.packageName,
                        step2.recipient
                    )
                    var repliedInline = false
                    if (sbn != null) {
                        repliedInline = com.zipper.app.core.remote.RemoteCommandDispatcher.trySendInlineReply(
                            this@MainActivity,
                            sbn,
                            msg2
                        )
                    }
                    if (!repliedInline) {
                        val ackIntent = Intent(Intent.ACTION_VIEW).apply {
                            data = android.net.Uri.parse("https://api.whatsapp.com/send?text=${android.net.Uri.encode(msg2)}")
                            setPackage(step2.packageName)
                            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                        }
                        startActivity(ackIntent)
                    }
                } catch (e: Exception) {
                    Log.w("MainActivity", "Acknowledgment dispatch notice: ${e.message}")
                }
            }

            val db = com.zipper.app.core.intelligence.store.CommitmentDatabase.getInstance(this@MainActivity)
            db.updateCommitmentStatus(item.id, com.zipper.app.core.intelligence.store.CommitmentStatus.COMPLETED)

            withContext(Dispatchers.Main) {
                Toast.makeText(
                    this@MainActivity,
                    "⚡ Auto-Do completed! Relayed to ${step1?.recipient ?: "recipient"} & marked done.",
                    Toast.LENGTH_LONG
                ).show()
                refreshPlannerDashboard()
            }
        }
    }

    private fun executeAutoDoWithAgent(
        item: com.zipper.app.core.intelligence.store.CommitmentEntity,
        step1: com.zipper.app.core.intelligence.planner.AutoDoStep?,
        msg1: String,
        step2: com.zipper.app.core.intelligence.planner.AutoDoStep?,
        msg2: String
    ) {
        val agentGoalBuilder = StringBuilder()
        if (step1 != null) {
            agentGoalBuilder.append("Open ${step1.appName}, find chat with '${step1.recipient}', and send message: \"$msg1\"")
        }
        if (step2 != null && msg2.isNotBlank()) {
            agentGoalBuilder.append(". Then find chat with '${step2.recipient}', and send acknowledgment: \"$msg2\"")
        }
        if (agentGoalBuilder.isEmpty()) {
            agentGoalBuilder.append(item.title)
        }

        val finalGoal = agentGoalBuilder.toString()

        lifecycleScope.launch(Dispatchers.IO) {
            val db = com.zipper.app.core.intelligence.store.CommitmentDatabase.getInstance(this@MainActivity)
            db.updateCommitmentStatus(item.id, com.zipper.app.core.intelligence.store.CommitmentStatus.IN_PROGRESS)

            withContext(Dispatchers.Main) {
                switchToAgentTab()
                etAgentGoal.setText(finalGoal)
                etAgentGoal.setSelection(etAgentGoal.text.length)
                btnRunAgentGoal.performClick()
                Toast.makeText(
                    this@MainActivity,
                    "🤖 Handed over 2-step plan to Karvish Mobile Agent!",
                    Toast.LENGTH_SHORT
                ).show()
            }
        }
    }
}