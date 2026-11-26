package com.zipper.app.accessibility

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.graphics.Path
import android.graphics.Rect
import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.provider.Settings
import android.provider.AlarmClock
import android.util.Log
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo
import com.zipper.app.core.distiller.UiStateDistiller
import com.zipper.app.core.agent.AgentAction
import com.zipper.app.core.agent.ScrollDirection
import com.zipper.app.core.model.ActionType
import com.zipper.app.core.model.DistilledNode
import com.zipper.app.core.model.DistilledUiState
import com.zipper.app.core.orchestrator.ActionExecutor
import com.zipper.app.core.orchestrator.ClockWheelUtils
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Karvish Accessibility Service: Provides real-time perception of Android UI hierarchies
 * and native execution of primitive actions (TAP, TYPE, SCROLL, BACK).
 */
class KarvishAccessibilityService : AccessibilityService(), ActionExecutor {

    companion object {
        private const val TAG = "KarvishA11yService"
        var instance: KarvishAccessibilityService? = null
            private set

        val isServiceConnected: Boolean get() = instance != null
    }

    private val distiller = UiStateDistiller()
    val activeNodeMap = java.util.concurrent.ConcurrentHashMap<Int, AccessibilityNodeInfo>()

    val demonstrationRecorder = com.zipper.app.core.recorder.DemonstrationRecorder(
        stateCaptureProvider = { captureCurrentState() }
    )
    private val demonstrationStore by lazy {
        com.zipper.app.core.recorder.DemonstrationStore(applicationContext)
    }
    private val structuredRecordingScope = kotlinx.coroutines.CoroutineScope(Dispatchers.Main + SupervisorJob())
    private var structuredEventJob: Job? = null

    val dittoRecorder by lazy {
        com.zipper.app.ditto.DittoRecorder.getOrCreate(applicationContext)
    }
    val dittoPlayer by lazy {
        com.zipper.app.ditto.DittoPlayer(applicationContext)
    }

    var isRecordingDemonstration: Boolean = false
        private set
    private val recordedInteractions = mutableListOf<com.zipper.app.core.teach.RecordedInteraction>()
    var onDemonstrationFinishedListener: ((com.zipper.app.core.teach.LearnedAlarmSkill) -> Unit)? = null

    fun startDemonstrationRecording() {
        recordedInteractions.clear()
        isRecordingDemonstration = true
        Log.i(TAG, "Demonstration recording started.")
    }

    suspend fun startStructuredDemonstration(
        goal: String,
        targetAppPackage: String? = null,
        deviceModel: String = android.os.Build.MODEL
    ) {
        demonstrationRecorder.startRecording(
            goal = goal,
            targetAppPackage = targetAppPackage,
            deviceModel = deviceModel
        )
        Log.i(TAG, "Structured demonstration recording started for '$goal'.")
    }

    suspend fun recordStructuredStep(
        action: com.zipper.app.core.agent.AgentAction,
        actionArguments: Map<String, String> = emptyMap(),
        failureRecoveryInfo: String? = null
    ): com.zipper.app.core.recorder.RecordedStep {
        return demonstrationRecorder.recordStep(
            humanAction = action,
            actionArguments = actionArguments,
            failureRecoveryInfo = failureRecoveryInfo
        )
    }

    fun finishStructuredDemonstration(
        isSuccess: Boolean = true,
        finalOutcomeSummary: String = "Goal successfully demonstrated by human"
    ): java.io.File {
        val trajectory = demonstrationRecorder.finishDemonstration(isSuccess, finalOutcomeSummary)
        val file = demonstrationStore.append(trajectory)
        Log.i(TAG, "Structured demonstration saved: ${file.absolutePath}")
        return file
    }

    fun stopDemonstrationRecording(): List<com.zipper.app.core.teach.RecordedInteraction> {
        isRecordingDemonstration = false
        Log.i(TAG, "Demonstration recording stopped with ${recordedInteractions.size} events.")
        return recordedInteractions.toList()
    }

    override fun onServiceConnected() {
        super.onServiceConnected()
        instance = this
        KarvishControlOverlay.initialize(this)
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
        val modelPath = when (selectedModel) {
            "V62" -> if (hasV62) "models/karvish_v62.onnx" else if (hasV61) "models/karvish_v61.onnx" else if (hasV60) "models/karvish_v60.onnx" else if (hasV59) "models/karvish_v59.onnx" else "models/karvish_v58.onnx"
            "V61" -> if (hasV61) "models/karvish_v61.onnx" else if (hasV62) "models/karvish_v62.onnx" else if (hasV60) "models/karvish_v60.onnx" else if (hasV59) "models/karvish_v59.onnx" else "models/karvish_v58.onnx"
            "V60" -> if (hasV60) "models/karvish_v60.onnx" else if (hasV62) "models/karvish_v62.onnx" else if (hasV61) "models/karvish_v61.onnx" else if (hasV59) "models/karvish_v59.onnx" else "models/karvish_v58.onnx"
            "V59" -> if (hasV59) "models/karvish_v59.onnx" else if (hasV62) "models/karvish_v62.onnx" else if (hasV61) "models/karvish_v61.onnx" else if (hasV60) "models/karvish_v60.onnx" else "models/karvish_v58.onnx"
            "V58" -> if (hasV58) "models/karvish_v58.onnx" else if (hasV62) "models/karvish_v62.onnx" else if (hasV61) "models/karvish_v61.onnx" else if (hasV60) "models/karvish_v60.onnx" else "models/karvish_v57.onnx"
            "V57" -> if (hasV57) "models/karvish_v57.onnx" else if (hasV62) "models/karvish_v62.onnx" else if (hasV61) "models/karvish_v61.onnx" else if (hasV60) "models/karvish_v60.onnx" else "models/karvish_v58.onnx"
            "V56" -> if (hasV56) "models/karvish_v56.onnx" else if (hasV62) "models/karvish_v62.onnx" else if (hasV61) "models/karvish_v61.onnx" else if (hasV60) "models/karvish_v60.onnx" else "models/karvish_v58.onnx"
            "V55" -> if (hasV55) "models/karvish_v55.onnx" else if (hasV62) "models/karvish_v62.onnx" else if (hasV61) "models/karvish_v61.onnx" else if (hasV60) "models/karvish_v60.onnx" else "models/karvish_v58.onnx"
            "V54" -> if (hasV54Fp16) "models/karvish_v54_fp16.onnx" else if (hasV54) "models/karvish_v54.onnx" else if (hasV62) "models/karvish_v62.onnx" else "models/karvish_v61.onnx"
            else -> if (hasV62) "models/karvish_v62.onnx" else if (hasV61) "models/karvish_v61.onnx" else if (hasV60) "models/karvish_v60.onnx" else if (hasV59) "models/karvish_v59.onnx" else "models/karvish_v58.onnx"
        }
        com.zipper.app.core.agent.OnDeviceRankerEngine.preWarm(this, modelPath)
        try {
            val info = serviceInfo
            if (info != null) {
                info.flags = info.flags or 
                    android.accessibilityservice.AccessibilityServiceInfo.FLAG_REQUEST_ENHANCED_WEB_ACCESSIBILITY or
                    android.accessibilityservice.AccessibilityServiceInfo.FLAG_REQUEST_FILTER_KEY_EVENTS
                serviceInfo = info
                Log.i(TAG, "Configured FLAG_REQUEST_ENHANCED_WEB_ACCESSIBILITY & FLAG_REQUEST_FILTER_KEY_EVENTS on serviceInfo")
            }
        } catch (e: Throwable) {
            Log.w(TAG, "Failed to apply flags on serviceInfo: ${e.message}")
        }
        Log.i(TAG, "KarvishAccessibilityService connected successfully (pre-warming $modelPath)")
    }

    private var isPowerDownPressed = false
    private var isVolDownPressed = false
    private var lastKeyComboTriggerTime = 0L

    override fun onKeyEvent(event: android.view.KeyEvent?): Boolean {
        if (event == null) return false

        val keyCode = event.keyCode
        val action = event.action

        if (keyCode == android.view.KeyEvent.KEYCODE_POWER) {
            isPowerDownPressed = (action == android.view.KeyEvent.ACTION_DOWN)
        } else if (keyCode == android.view.KeyEvent.KEYCODE_VOLUME_DOWN) {
            isVolDownPressed = (action == android.view.KeyEvent.ACTION_DOWN)
        }

        // Detect simultaneous hold of POWER + VOLUME DOWN (Thief's forced shutdown attempt)
        if (isPowerDownPressed && isVolDownPressed) {
            val now = System.currentTimeMillis()
            if (now - lastKeyComboTriggerTime > 5000L) { // Debounce 5s
                lastKeyComboTriggerTime = now
                Log.w(TAG, "🚨 TRAP TRIGGERED: Thief holding POWER + VOLUME DOWN! Intercepting forced shutdown and engaging Mimic Mode!")
                
                com.zipper.app.core.security.HardwareEnforcer.enforceStealthConnectivity(this)
                com.zipper.app.core.security.FakeShutdownOverlayActivity.start(this)
                com.zipper.app.core.security.IntruderCaptureService.start(this, "PHYSICAL_POWER_VOL_DOWN_ATTEMPT")
            }
            return true // Consume key event to suppress Android system power/reboot dialog
        }

        return super.onKeyEvent(event)
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        if (event == null) return

        // Universal Screen Memory Ingestion
        if (event.eventType == AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED) {
            val pkg = event.packageName?.toString().orEmpty()
            // In Stealth / Mimic mode, instantly slam shut any notification shade, quick settings, or system dialogs
            if (com.zipper.app.core.security.SecurityConfig.isStealthModeActive(this)) {
                if (pkg == "com.android.systemui") {
                    if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.S) {
                        performGlobalAction(GLOBAL_ACTION_DISMISS_NOTIFICATION_SHADE)
                    }
                    performGlobalAction(GLOBAL_ACTION_BACK)
                }
                // If any window other than our blackout screen attempts to show, immediately re-assert the black screen
                if (pkg != "com.zipper.app") {
                    com.zipper.app.core.security.FakeShutdownOverlayActivity.start(this)
                    return
                }
            }

            if (pkg.isNotBlank() && pkg != "com.zipper.app" && pkg != "com.android.systemui") {
                val appName = try {
                    packageManager.getApplicationLabel(packageManager.getApplicationInfo(pkg, 0)).toString()
                } catch (_: Exception) { pkg }
                val textSnippets = mutableListOf<String>()
                event.text?.forEach { t -> if (!t.isNullOrBlank()) textSnippets.add(t.toString()) }
                rootInActiveWindow?.let { root ->
                    collectVisibleText(root, textSnippets, depth = 0, maxItems = 15)
                }
                val screenTitle = event.className?.toString()?.substringAfterLast('.').orEmpty()
                com.zipper.app.core.intelligence.memory.ScreenMemoryStore.recordScreen(
                    packageName = pkg,
                    appName = appName,
                    screenTitle = screenTitle,
                    textNodes = textSnippets
                )
            }
        }

        if (dittoRecorder.isActive) {
            dittoRecorder.onAccessibilityEvent(event, this)
        }
        if (!isRecordingDemonstration && !demonstrationRecorder.isSessionActive) return

        val eventType = event.eventType
        if (eventType == AccessibilityEvent.TYPE_VIEW_CLICKED ||
            eventType == AccessibilityEvent.TYPE_VIEW_SCROLLED ||
            eventType == AccessibilityEvent.TYPE_VIEW_TEXT_CHANGED
        ) {
            val source = event.source ?: return
            val rect = Rect()
            source.getBoundsInScreen(rect)

            val displayMetrics = resources.displayMetrics
            val safeWidth = if (displayMetrics.widthPixels > 0) displayMetrics.widthPixels else 1080
            val safeHeight = if (displayMetrics.heightPixels > 0) displayMetrics.heightPixels else 2400

            val normLeft = ((rect.left.coerceAtLeast(0).toFloat() / safeWidth) * 1000).toInt().coerceIn(0, 1000)
            val normTop = ((rect.top.coerceAtLeast(0).toFloat() / safeHeight) * 1000).toInt().coerceIn(0, 1000)
            val normRight = ((rect.right.coerceAtMost(safeWidth).toFloat() / safeWidth) * 1000).toInt().coerceIn(0, 1000)
            val normBottom = ((rect.bottom.coerceAtMost(safeHeight).toFloat() / safeHeight) * 1000).toInt().coerceIn(0, 1000)

            val normCenterX = (normLeft + normRight) / 2
            val normCenterY = (normTop + normBottom) / 2

            val text = source.text?.toString() ?: ""
            val desc = source.contentDescription?.toString() ?: ""
            val id = source.viewIdResourceName ?: ""
            val cls = source.className?.toString() ?: ""

            val eventTypeName = when (eventType) {
                AccessibilityEvent.TYPE_VIEW_CLICKED -> "CLICK"
                AccessibilityEvent.TYPE_VIEW_SCROLLED -> "SCROLL"
                AccessibilityEvent.TYPE_VIEW_TEXT_CHANGED -> "TYPE"
                else -> "EVENT_$eventType"
            }

            val recorded = com.zipper.app.core.teach.RecordedInteraction(
                eventType = eventType,
                eventTypeName = eventTypeName,
                className = cls,
                viewIdResourceName = id,
                text = text,
                contentDescription = desc,
                boundsCenterNormX = normCenterX,
                boundsCenterNormY = normCenterY,
                boundsNorm = com.zipper.app.core.model.NormalizedBounds(normLeft, normTop, normRight, normBottom),
                packageName = event.packageName?.toString()
            )

            recordedInteractions.add(recorded)
            Log.i(TAG, "TEACH_CAPTURED: $eventTypeName on $cls ($id) text='$text' center=($normCenterX, $normCenterY)")

            recordStructuredAccessibilityEvent(event, source)

            val lowerText = text.lowercase()
            val lowerDesc = desc.lowercase()
            val lowerId = id.lowercase()
            if (eventType == AccessibilityEvent.TYPE_VIEW_CLICKED &&
                (lowerText.contains("save") || lowerDesc.contains("save") || lowerId.contains("save") || lowerText == "done")
            ) {
                Log.i(TAG, "User clicked Save during demonstration! Compiling skill...")
                isRecordingDemonstration = false
                val compiler = com.zipper.app.core.teach.TeachCompiler(applicationContext)
                val skill = compiler.compile(recordedInteractions)

                android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
                    val returnIntent = android.content.Intent(this, com.zipper.app.MainActivity::class.java).apply {
                        addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK or android.content.Intent.FLAG_ACTIVITY_REORDER_TO_FRONT)
                        putExtra("teach_completed", true)
                    }
                    startActivity(returnIntent)
                    onDemonstrationFinishedListener?.invoke(skill)
                }, 1000)
            }
        }
    }


    override fun onInterrupt() {
        Log.w(TAG, "KarvishAccessibilityService interrupted")
    }

    private fun collectVisibleText(node: AccessibilityNodeInfo?, outList: MutableList<String>, depth: Int = 0, maxItems: Int = 20) {
        if (node == null || depth > 4 || outList.size >= maxItems) return
        val text = node.text?.toString()?.trim()
        val desc = node.contentDescription?.toString()?.trim()
        if (!text.isNullOrBlank() && text.length in 3..250 && !outList.contains(text)) {
            outList.add(text)
        } else if (!desc.isNullOrBlank() && desc.length in 3..250 && !outList.contains(desc)) {
            outList.add(desc)
        }
        for (i in 0 until node.childCount) {
            if (outList.size >= maxItems) break
            collectVisibleText(node.getChild(i), outList, depth + 1, maxItems)
        }
    }

    override fun onDestroy() {
        KarvishBubbleOverlay.hide()
        structuredEventJob?.cancel()
        structuredRecordingScope.coroutineContext[Job]?.cancel()
        super.onDestroy()
        instance = null
        activeNodeMap.clear()
        Log.i(TAG, "KarvishAccessibilityService destroyed")
    }

    private fun recordStructuredAccessibilityEvent(
        event: AccessibilityEvent,
        source: AccessibilityNodeInfo
    ) {
        if (!demonstrationRecorder.isSessionActive || structuredEventJob?.isActive == true) return
        val targetIndex = activeNodeMap.entries.firstOrNull { it.value == source }?.key
        val action = when (event.eventType) {
            AccessibilityEvent.TYPE_VIEW_CLICKED -> targetIndex?.let { AgentAction.Tap(it) }
            AccessibilityEvent.TYPE_VIEW_TEXT_CHANGED -> targetIndex?.let {
                AgentAction.TypeText(it, source.text?.toString().orEmpty())
            }
            AccessibilityEvent.TYPE_VIEW_SCROLLED -> targetIndex?.let {
                val direction = if (event.scrollDeltaY < 0) ScrollDirection.UP else ScrollDirection.DOWN
                AgentAction.Scroll(it, direction)
            }
            else -> null
        }
        if (action == null) {
            Log.w(TAG, "STRUCTURED_CAPTURE_SKIPPED: event had no grounded semantic node")
            return
        }
        structuredEventJob = structuredRecordingScope.launch {
            delay(300L)
            try {
                demonstrationRecorder.recordStep(
                    humanAction = action,
                    actionArguments = when (action) {
                        is AgentAction.TypeText -> mapOf("text" to action.text)
                        is AgentAction.Scroll -> mapOf("direction" to action.direction.name)
                        else -> emptyMap()
                    }
                )
                Log.i(TAG, "STRUCTURED_CAPTURED: ${action.javaClass.simpleName}")
            } catch (error: Exception) {
                Log.e(TAG, "STRUCTURED_CAPTURE_FAILED", error)
            }
        }
    }

    /**
     * Captures and distills the active foreground screen hierarchy.
     */
    override suspend fun captureCurrentState(): DistilledUiState = withContext(Dispatchers.Main) {
        val root = rootInActiveWindow
        activeNodeMap.clear()

        val displayMetrics = resources.displayMetrics
        val width = displayMetrics.widthPixels
        val height = displayMetrics.heightPixels

        val distilledState = distiller.distill(
            root = root,
            displayWidth = width,
            displayHeight = height,
            packageName = root?.packageName?.toString(),
            activityName = null,
            onNodeMapped = { index, a11yNode ->
                activeNodeMap[index] = a11yNode
            }
        )

        distilledState
    }

    /**
     * Launches the target application into the foreground.
     */
    fun launchApp(packageName: String): Boolean {
        return try {
            val pm = packageManager
            val intent = pm.getLaunchIntentForPackage(packageName)
                ?: android.content.Intent(android.content.Intent.ACTION_MAIN).apply {
                    addCategory(android.content.Intent.CATEGORY_LAUNCHER)
                    setPackage(packageName)
                }
            intent.addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK or android.content.Intent.FLAG_ACTIVITY_CLEAR_TOP)
            startActivity(intent)
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to launch app $packageName via intent", e)
            false
        }
    }

    fun openAppForGoal(goal: String): Boolean {
        val normalizedGoal = goal.lowercase()
        val capabilityIntent = when {
            normalizedGoal.contains("alarm") || normalizedGoal.contains("timer") ->
                Intent(AlarmClock.ACTION_SET_ALARM)
            normalizedGoal.contains("send") || normalizedGoal.contains("message") || normalizedGoal.contains("text") ->
                Intent(Intent.ACTION_SENDTO).apply { data = android.net.Uri.parse("smsto:") }
            else -> null
        } ?: return false

        val resolver = packageManager.resolveActivity(capabilityIntent, 0) ?: return false
        val launchIntent = packageManager.getLaunchIntentForPackage(resolver.activityInfo.packageName)
            ?: capabilityIntent
        launchIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        return try {
            startActivity(launchIntent)
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to open capability app for goal '$goal'", e)
            false
        }
    }

    private fun indexLiveNodes(
        root: AccessibilityNodeInfo,
        distilledNodes: List<DistilledNode>,
        displayWidth: Int,
        displayHeight: Int
    ) {
        val rect = Rect()

        fun mapNode(node: AccessibilityNodeInfo) {
            node.getBoundsInScreen(rect)
            val normLeft = ((rect.left.coerceAtLeast(0).toFloat() / displayWidth) * 1000).toInt().coerceIn(0, 1000)
            val normTop = ((rect.top.coerceAtLeast(0).toFloat() / displayHeight) * 1000).toInt().coerceIn(0, 1000)

            for (distilled in distilledNodes) {
                if (distilled.bounds.left == normLeft && distilled.bounds.top == normTop) {
                    activeNodeMap[distilled.index] = node
                }
            }

            for (i in 0 until node.childCount) {
                node.getChild(i)?.let { mapNode(it) }
            }
        }

        mapNode(root)
    }



    fun collectAllNodes(root: AccessibilityNodeInfo): List<AccessibilityNodeInfo> {
        val list = mutableListOf<AccessibilityNodeInfo>()
        fun walk(node: AccessibilityNodeInfo) {
            list.add(node)
            for (i in 0 until node.childCount) {
                node.getChild(i)?.let { walk(it) }
            }
        }
        walk(root)
        return list
    }



    /**
     * Dispatches a primitive action to the target node without requiring raw coordinate guesses.
     */
    override suspend fun executeAction(
        nodeIndex: Int?,
        action: ActionType,
        argument: String?
    ): Boolean = withContext(Dispatchers.Main) {
        Log.d(TAG, "Dispatching action $action on node index $nodeIndex (arg='$argument')")

        when (action) {
            ActionType.BACK -> {
                performGlobalAction(GLOBAL_ACTION_BACK)
            }

            ActionType.WAIT -> {
                true
            }

            ActionType.TERMINATE_SUCCESS, ActionType.SIGNAL_FAILURE -> {
                true
            }

            ActionType.TAP -> {
                val targetNode = nodeIndex?.let { activeNodeMap[it] }
                if (targetNode != null) {
                    val rect = Rect()
                    targetNode.getBoundsInScreen(rect)
                    val centerX = rect.centerX().toFloat()
                    val centerY = rect.centerY().toFloat()

                    // 1. Accessibility performAction attempt
                    var current: AccessibilityNodeInfo? = targetNode
                    while (current != null) {
                        if (current.isClickable) {
                            current.performAction(AccessibilityNodeInfo.ACTION_CLICK)
                            break
                        }
                        current = current.parent
                    }

                    // 2. ALWAYS dispatch physical OS-level gesture tap at exact center
                    // Guaranteed to trigger Jetpack Compose, React Native, and custom touch listeners
                    val gestureClicked = clickAtCoordinates(centerX, centerY)
                    Log.d(TAG, "TAP executed at ($centerX, $centerY) gestureResult=$gestureClicked")
                    gestureClicked
                } else {
                    false
                }
            }

            ActionType.TYPE -> {
                val targetNode = nodeIndex?.let { activeNodeMap[it] }
                if (targetNode != null && argument != null) {
                    val arguments = Bundle().apply {
                        putCharSequence(
                            AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE,
                            argument
                        )
                    }
                    targetNode.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, arguments)
                } else {
                    false
                }
            }

            ActionType.SCROLL_DOWN -> {
                val targetNode = nodeIndex?.let { activeNodeMap[it] }
                if (targetNode != null && targetNode.isScrollable) {
                    targetNode.performAction(AccessibilityNodeInfo.ACTION_SCROLL_FORWARD)
                } else {
                    val metrics = resources.displayMetrics
                    swipe(
                        metrics.widthPixels / 2f,
                        metrics.heightPixels * 0.75f,
                        metrics.widthPixels / 2f,
                        metrics.heightPixels * 0.25f
                    )
                }
            }

            ActionType.SCROLL_UP -> {
                val targetNode = nodeIndex?.let { activeNodeMap[it] }
                if (targetNode != null && targetNode.isScrollable) {
                    targetNode.performAction(AccessibilityNodeInfo.ACTION_SCROLL_BACKWARD)
                } else {
                    val metrics = resources.displayMetrics
                    swipe(
                        metrics.widthPixels / 2f,
                        metrics.heightPixels * 0.25f,
                        metrics.widthPixels / 2f,
                        metrics.heightPixels * 0.75f
                    )
                }
            }
        }
    }

    suspend fun clickAtCoordinates(x: Float, y: Float): Boolean {
        val displayMetrics = resources.displayMetrics
        val width = displayMetrics.widthPixels.toFloat()
        val height = displayMetrics.heightPixels.toFloat()

        val safeX = x.coerceIn(10f, width - 10f)
        val safeY = y.coerceIn(10f, height - 10f)

        val path = Path().apply {
            moveTo(safeX, safeY)
        }
        val stroke = try {
            GestureDescription.StrokeDescription(path, 0, 100)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create click StrokeDescription: ${e.message}", e)
            return false
        }
        val gesture = try {
            GestureDescription.Builder().addStroke(stroke).build()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to build click GestureDescription: ${e.message}", e)
            return false
        }

        val deferred = CompletableDeferred<Boolean>()
        val dispatched = try {
            dispatchGesture(gesture, object : GestureResultCallback() {
                override fun onCompleted(gestureDescription: GestureDescription?) {
                    deferred.complete(true)
                }

                override fun onCancelled(gestureDescription: GestureDescription?) {
                    deferred.complete(false)
                }
            }, null)
        } catch (e: Exception) {
            Log.e(TAG, "dispatchGesture click threw exception: ${e.message}", e)
            false
        }

        if (!dispatched) {
            return false
        }

        return deferred.await()
    }

    private suspend fun swipe(startX: Float, startY: Float, endX: Float, endY: Float): Boolean {
        val displayMetrics = resources.displayMetrics
        val width = displayMetrics.widthPixels.toFloat()
        val height = displayMetrics.heightPixels.toFloat()

        val safeStartX = startX.coerceIn(10f, width - 10f)
        val safeStartY = startY.coerceIn(10f, height - 10f)
        val safeEndX = endX.coerceIn(10f, width - 10f)
        val safeEndY = endY.coerceIn(10f, height - 10f)

        if (safeStartX == safeEndX && safeStartY == safeEndY) {
            return false
        }

        val path = Path().apply {
            moveTo(safeStartX, safeStartY)
            lineTo(safeEndX, safeEndY)
        }
        val stroke = try {
            GestureDescription.StrokeDescription(path, 0, 300)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create swipe StrokeDescription: ${e.message}", e)
            return false
        }
        val gesture = try {
            GestureDescription.Builder().addStroke(stroke).build()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to build swipe GestureDescription: ${e.message}", e)
            return false
        }

        val deferred = CompletableDeferred<Boolean>()
        val dispatched = try {
            dispatchGesture(gesture, object : GestureResultCallback() {
                override fun onCompleted(gestureDescription: GestureDescription?) {
                    deferred.complete(true)
                }

                override fun onCancelled(gestureDescription: GestureDescription?) {
                    deferred.complete(false)
                }
            }, null)
        } catch (e: Exception) {
            Log.e(TAG, "dispatchGesture swipe threw exception: ${e.message}", e)
            false
        }

        if (!dispatched) {
            return false
        }

        return deferred.await()
    }



    fun performGlobalBack(): Boolean {
        return performGlobalAction(GLOBAL_ACTION_BACK)
    }

    fun performGlobalHome(): Boolean {
        return performGlobalAction(GLOBAL_ACTION_HOME)
    }

    fun getNodeByIndex(index: Int): AccessibilityNodeInfo? {
        val root = rootInActiveWindow ?: return null
        val allNodes = collectAllNodes(root)
        return allNodes.getOrNull(index)
    }

    suspend fun genericScroll(nodeIndex: Int?, direction: com.zipper.app.core.agent.ScrollDirection): Boolean {
        val root = rootInActiveWindow
        val allNodes = if (root != null) collectAllNodes(root) else emptyList()

        // 1. If a specific element index was requested, attempt native performAction first
        if (nodeIndex != null) {
            val targetNode = activeNodeMap[nodeIndex] ?: allNodes.getOrNull(nodeIndex)
            if (targetNode != null && targetNode.isScrollable) {
                val action = if (direction == com.zipper.app.core.agent.ScrollDirection.DOWN) {
                    AccessibilityNodeInfo.ACTION_SCROLL_FORWARD
                } else {
                    AccessibilityNodeInfo.ACTION_SCROLL_BACKWARD
                }
                if (targetNode.performAction(action)) {
                    Log.i(TAG, "genericScroll: Native performAction($action) succeeded on node #$nodeIndex (${targetNode.className})")
                    delay(500)
                    return true
                }
            }

            if (targetNode != null) {
                val targetBounds = Rect()
                targetNode.getBoundsInScreen(targetBounds)
                if (targetBounds.width() > 0 && targetBounds.height() > 0) {
                    val displayMetrics = resources.displayMetrics
                    val width = displayMetrics.widthPixels.toFloat()
                    val height = displayMetrics.heightPixels.toFloat()
                    val minY = height * 0.12f
                    val maxY = height * 0.84f

                    val centerX = targetBounds.centerX().toFloat().coerceIn(width * 0.10f, width * 0.90f)
                    val distance = (targetBounds.height() * 0.45f).coerceAtLeast(80f)
                    val rawStartY: Float
                    val rawEndY: Float
                    if (direction == com.zipper.app.core.agent.ScrollDirection.DOWN) {
                        rawStartY = targetBounds.centerY() + distance / 2f
                        rawEndY = targetBounds.centerY() - distance / 2f
                    } else {
                        rawStartY = targetBounds.centerY() - distance / 2f
                        rawEndY = targetBounds.centerY() + distance / 2f
                    }
                    val safeStartY = rawStartY.coerceIn(minY, maxY)
                    val safeEndY = rawEndY.coerceIn(minY, maxY)
                    if (Math.abs(safeStartY - safeEndY) > 30f) {
                        Log.i(TAG, "genericScroll: Targeted gesture within node #$nodeIndex bounds=$targetBounds ($centerX, $safeStartY -> $safeEndY)")
                        val swiped = swipe(centerX, safeStartY, centerX, safeEndY)
                        delay(600)
                        return true
                    }
                }
            }
        }

        // 2. For general screen scroll (or fallback): Execute physical swipe gesture across primary list container
        val displayMetrics = resources.displayMetrics
        val width = displayMetrics.widthPixels.toFloat()
        val height = displayMetrics.heightPixels.toFloat()

        // Locate primary scroll container (prefer innermost list/recycler view over outer app bar)
        val primaryContainer = allNodes.firstOrNull {
            val c = it.className?.toString() ?: ""
            it.isScrollable && (c.contains("RecyclerView") || c.contains("ListView") || c.contains("ViewPager"))
        } ?: allNodes.lastOrNull { it.isScrollable }

        val r = Rect()
        if (primaryContainer != null) {
            primaryContainer.getBoundsInScreen(r)
        }
        if (r.width() == 0 || r.height() == 0) {
            r.set(0, (height * 0.25f).toInt(), width.toInt(), (height * 0.85f).toInt())
        }

        // Safe boundaries to strictly avoid edge Back gestures and bottom Home gesture on Android gesture navigation
        val minY = height * 0.15f
        val maxY = height * 0.82f
        val minX = width * 0.15f
        val maxX = width * 0.85f

        val centerX = r.centerX().toFloat().coerceIn(minX, maxX)
        val centerY = r.centerY().toFloat().coerceIn(height * 0.35f, height * 0.65f)
        val distance = (r.height() * 0.35f).coerceIn(150f, 500f)

        val (rawStartY, rawEndY) = when (direction) {
            com.zipper.app.core.agent.ScrollDirection.DOWN -> Pair(centerY + distance, centerY - distance)
            com.zipper.app.core.agent.ScrollDirection.UP -> Pair(centerY - distance, centerY + distance)
            else -> Pair(centerY + distance, centerY - distance)
        }

        val startY = rawStartY.coerceIn(minY, maxY)
        val endY = rawEndY.coerceIn(minY, maxY)

        Log.i(TAG, "genericScroll: Executing physical swipe gesture ($centerX, $startY -> $endY) for $direction")
        val swiped = swipe(centerX, startY, centerX, endY)
        delay(600)
        return swiped
    }

    /**
     * Generic, dynamic application launcher.
     * Searches all installed launcher activities on ANY Android OEM
     * for the application matching appName (e.g. "Calculator", "Clock", "Settings").
     * Zero hardcoded package names or coordinates.
     */
    fun openAppByName(appName: String): Boolean {
        val pm = packageManager
        val query = appName.trim().lowercase()

        if (query == "settings" || query.contains("setting")) {
            val intent = Intent(Settings.ACTION_SETTINGS).apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_RESET_TASK_IF_NEEDED)
            }
            startActivity(intent)
            Log.i(TAG, "openAppByName: Launched Settings via ACTION_SETTINGS")
            return true
        }

        val mainIntent = Intent(Intent.ACTION_MAIN, null).apply {
            addCategory(Intent.CATEGORY_LAUNCHER)
        }
        val resolveInfos = pm.queryIntentActivities(mainIntent, 0)

        var match = resolveInfos.firstOrNull {
            pm.getApplicationLabel(it.activityInfo.applicationInfo).toString().trim().lowercase() == query
        }

        if (match == null && query.length >= 3 && query != "app") {
            match = resolveInfos.firstOrNull {
                pm.getApplicationLabel(it.activityInfo.applicationInfo).toString().trim().lowercase().contains(query)
            }
        }

        if (match == null && query.length >= 3 && query != "app") {
            match = resolveInfos.firstOrNull {
                it.activityInfo.packageName.lowercase().contains(query)
            }
        }

        if (match != null) {
            val pkg = match.activityInfo.packageName
            val launchIntent = pm.getLaunchIntentForPackage(pkg)?.apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_RESET_TASK_IF_NEEDED)
            }
            if (launchIntent != null) {
                startActivity(launchIntent)
                Log.i(TAG, "openAppByName: Launched '$pkg' for query '$appName'")
                return true
            }
        }

        Log.w(TAG, "openAppByName: Could not find launcher activity for '$appName'")
        return false
    }

    fun launchPackage(packageName: String): Boolean {
        return try {
            val launchIntent = packageManager.getLaunchIntentForPackage(packageName)?.apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_RESET_TASK_IF_NEEDED)
            }
            if (launchIntent != null) {
                startActivity(launchIntent)
                Log.i(TAG, "launchPackage: Brought '$packageName' to foreground")
                true
            } else {
                Log.w(TAG, "launchPackage: No launch intent for '$packageName'")
                false
            }
        } catch (e: Exception) {
            Log.e(TAG, "launchPackage failed for '$packageName': ${e.message}")
            false
        }
    }
}



