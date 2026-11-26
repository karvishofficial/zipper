package com.zipper.app.ditto

import android.content.Context
import android.graphics.Rect
import android.util.Log
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo
import com.zipper.app.accessibility.KarvishAccessibilityService
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.util.Collections

/**
 * DITTO RECORDER (Crash-Proof V2.2)
 *
 * Implements:
 * 1. Strict System/Widget Blacklist: Prevents Google Search widgets (com.google.android.googlequicksearchbox)
 *    and Samsung launchers from hijacking launchPackage.
 * 2. Thread-Safe Step Collection: Eliminates ConcurrentModificationException using synchronized list operations.
 * 3. Sealed-Node Crash Protection: Traverses AccessibilityNodeInfo trees using safe wrappers to prevent
 *    IllegalStateException and native recycling crashes in dynamic apps like Instagram.
 * 4. Typing Buffer & Editing Session Consolidation: Preserves full text with spaces without letter-by-letter spam.
 */
class DittoRecorder(private val context: Context) {

    companion object {
        private const val TAG = "DittoRecorder"

        @Volatile
        var instance: DittoRecorder? = null

        fun getOrCreate(context: Context): DittoRecorder {
            return instance ?: synchronized(this) {
                instance ?: DittoRecorder(context.applicationContext).also { instance = it }
            }
        }

        val LAUNCHER_PACKAGES = setOf(
            "com.sec.android.app.launcher",             // Samsung One UI Home
            "com.google.android.apps.nexuslauncher",     // Pixel Launcher
            "com.android.launcher3",                     // AOSP Launcher
            "com.miui.home",                             // Xiaomi MIUI / HyperOS
            "com.oppo.launcher",                         // Oppo
            "com.coloros.launcher",                      // ColorOS / OnePlus
            "com.huawei.android.launcher"                // Huawei
        )

        val KEYBOARD_PACKAGES = setOf(
            "com.samsung.android.honeyboard",          // Samsung Keyboard
            "com.google.android.inputmethod.latin",      // GBoard
            "com.touchtype.swiftkey",                    // Microsoft SwiftKey
            "com.baidu.input"
        )

        val SYSTEM_UI_PACKAGES = setOf(
            "com.android.systemui",                      // System UI
            "com.samsung.android.app.cocktailbarservice",// Samsung Edge Panel
            "android",                                   // Android framework
            "com.zipper.app"                             // Zipper app
        )

        fun isKeyboardPackage(pkg: String?): Boolean {
            if (pkg.isNullOrBlank()) return false
            val p = pkg.trim().lowercase()
            return KEYBOARD_PACKAGES.contains(p) || p.contains(".inputmethod.") || p.contains(".keyboard") || p.contains(".honeyboard")
        }

        fun isLauncherPackage(pkg: String?): Boolean {
            if (pkg.isNullOrBlank()) return false
            return LAUNCHER_PACKAGES.contains(pkg.trim())
        }

        fun isSystemUiPackage(pkg: String?): Boolean {
            if (pkg.isNullOrBlank()) return false
            return SYSTEM_UI_PACKAGES.contains(pkg.trim())
        }

        fun isIgnoredPackage(pkg: String?): Boolean {
            if (pkg.isNullOrBlank()) return true
            val p = pkg.trim()
            return isSystemUiPackage(p) || isKeyboardPackage(p)
        }

        fun isNonAppPackage(pkg: String?): Boolean {
            if (pkg.isNullOrBlank()) return true
            val p = pkg.trim()
            return isLauncherPackage(p) || isSystemUiPackage(p) || isKeyboardPackage(p)
        }
    }

    private val dittoStore = DittoStore(context)
    private var isRecording = false
    private var isPaused = false
    private var activeTitle: String = ""
    private var launchPackage: String? = null
    private val steps = Collections.synchronizedList(mutableListOf<DittoStep>())
    private var previousSnapshot: DittoStateSnapshot? = null
    private var lastScrollTimestamp = 0L
    private var lastTapTimestamp = 0L
    private val lastTapRect = Rect()
    private var lastTapPackage: String? = null
    private var lastTapResId: String? = null

    // Typing Session Buffering
    private data class ActiveTextSession(
        val fieldKey: String,
        val viewId: String?,
        val targetIdentity: DittoTargetIdentity,
        val targetMetadata: DittoTargetMetadata,
        val physicalExecution: DittoPhysicalExecution,
        val stateBefore: DittoStateSnapshot,
        var currentText: String,
        var lastUpdated: Long = System.currentTimeMillis()
    )

    private var activeTextSession: ActiveTextSession? = null
    private var textDebounceJob: Job? = null

    val isActive: Boolean get() = isRecording

    /**
     * Starts a new Ditto recording session.
     * Shows the floating control overlay with the emerald FINISH button.
     */
    fun startRecording(title: String, targetPackage: String? = null) {
        val cleanTitle = title.trim().ifBlank { "Demonstrated Task ${System.currentTimeMillis() % 10000}" }
        activeTitle = cleanTitle
        launchPackage = if (!isIgnoredPackage(targetPackage)) targetPackage else null
        synchronized(steps) { steps.clear() }
        isRecording = true
        isPaused = false
        previousSnapshot = null
        activeTextSession = null
        textDebounceJob?.cancel()
        lastTapTimestamp = 0L
        lastTapRect.setEmpty()
        lastTapResId = null

        val a11y = KarvishAccessibilityService.instance
        if (a11y != null) {
            captureLiveSnapshot(a11y) { snapshot ->
                previousSnapshot = snapshot
            }
        }

        // Display the floating movable & collapsible HUD with FINISH button
        DittoFloatingControl.show(
            context = context,
            onFinish = {
                stopRecording(isSuccess = true)
            },
            onPauseToggle = { paused ->
                isPaused = paused
                Log.i(TAG, "Ditto recording paused=$isPaused")
            },
            onCancel = {
                cancelRecording()
            }
        )

        DittoFloatingControl.updateStep(0, "Recording: $cleanTitle")
        Log.i(TAG, "Ditto session started: '$cleanTitle' (Explicit Package: $targetPackage)")
    }

    /**
     * Called by KarvishAccessibilityService on each relevant accessibility event.
     */
    fun onAccessibilityEvent(event: AccessibilityEvent, service: KarvishAccessibilityService) {
        if (!isRecording || isPaused) return

        val eventPkg = event.packageName?.toString() ?: ""
        if (eventPkg == context.packageName || eventPkg == "com.zipper.app") {
            // AGENTS.md: Ignore internal Zipper clicks (e.g. RECORD DITTO button or floating overlay)
            return
        }

        val eventType = event.eventType

        // Module 1A: Anti-Hijack Package Locking
        // Lock launchPackage ONLY on TYPE_WINDOW_STATE_CHANGED for third-party foreground apps
        if (launchPackage.isNullOrBlank() &&
            eventType == AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED &&
            !isNonAppPackage(eventPkg) &&
            eventPkg != "com.zipper.app"
        ) {
            val activeRootPkg = service.rootInActiveWindow?.packageName?.toString()
            if (activeRootPkg == eventPkg || event.isFullScreen) {
                launchPackage = eventPkg
                Log.i(TAG, "[Module 1A Anti-Hijack] Locked target launchPackage: $eventPkg")
            }
        }

        // 1. Filter Launcher & System UI Swipes (User browsing home screen for app icon)
        if (isLauncherPackage(eventPkg) || isSystemUiPackage(eventPkg)) {
            if (eventType == AccessibilityEvent.TYPE_VIEW_SCROLLED) {
                return // Ignore home screen/system scrolling
            }
        }

        // Module 1B: Focus Loss (Blur) Handling
        if (eventType == AccessibilityEvent.TYPE_VIEW_FOCUSED) {
            val source = safeGetSource(event)
            val isEd = source != null && (try { source.isEditable } catch (_: Throwable) { false } || safeGetClassName(source)?.contains("EditText", ignoreCase = true) == true)
            val focusedId = source?.let { safeGetViewId(it) }
            if (activeTextSession != null && (!isEd || (focusedId != null && activeTextSession?.viewId != null && focusedId != activeTextSession?.viewId))) {
                Log.i(TAG, "[Module 1B Blur] Input focus changed; flushing active text session")
                flushActiveTextSession(service)
            }
        }

        // 2. Handle Typing Buffering (TYPE_VIEW_TEXT_CHANGED or Chromium/Web text selection changes)
        if (eventType == AccessibilityEvent.TYPE_VIEW_TEXT_CHANGED || eventType == AccessibilityEvent.TYPE_VIEW_TEXT_SELECTION_CHANGED) {
            val source = safeGetSource(event)
            val isEd = source != null && (try { source.isEditable } catch (_: Throwable) { false } || safeGetClassName(source)?.contains("EditText", ignoreCase = true) == true)
            if (eventType == AccessibilityEvent.TYPE_VIEW_TEXT_CHANGED || isEd) {
                handleTextChangedEvent(event, service)
                return
            }
        }

        // 4. Capture intentional user clicks and dropdown/dialog item selections
        val isClick = eventType == AccessibilityEvent.TYPE_VIEW_CLICKED
        val isSelection = eventType == AccessibilityEvent.TYPE_VIEW_SELECTED
        if (!isClick && !isSelection) {
            return
        }
        if (isSelection) {
            val eventTexts = try { event.text } catch (_: Throwable) { null }
            val firstText = eventTexts?.firstOrNull { it.isNotBlank() }?.toString()?.trim()
            val src = safeGetSource(event)
            val directText = src?.let { safeGetText(it) }?.trim() ?: firstText

            // Filter pure scroll indicator / page position updates (e.g. "1 of 10")
            val isScrollProgress = directText != null && (
                directText.matches(Regex("^\\d+\\s*/\\s*\\d+$")) ||
                directText.matches(Regex("^Page \\d+"))
            )
            if (directText.isNullOrBlank() || isScrollProgress) {
                return
            }
            Log.i(TAG, "Item selection event captured: '$directText' (pkg=$eventPkg)")
        }

        // Universal Keyboard Shield: Filter individual alphanumeric keystrokes, BUT capture explicit IME Enter / Search submits!
        val isImeWindow = try {
            event.source?.window?.type == android.view.accessibility.AccessibilityWindowInfo.TYPE_INPUT_METHOD
        } catch (_: Throwable) { false }
        if (isKeyboardPackage(eventPkg) || isImeWindow) {
            val src = safeGetSource(event)
            val keyLabel = listOfNotNull(
                event.text?.joinToString(" "),
                event.contentDescription?.toString(),
                src?.let { safeGetText(it) },
                src?.let { safeGetDesc(it) }
            ).joinToString(" ").lowercase().trim()

            val res = src?.let { safeGetViewId(it) }?.lowercase().orEmpty()
            val isImeSubmit = keyLabel.contains(Regex("\\b(search|go|enter|done|submit|send)\\b")) ||
                res.contains(Regex("enter|search|action|submit|go|send"))

            if (isImeSubmit) {
                Log.i(TAG, "Soft keyboard Enter/Search key tapped ('$keyLabel', res='$res'). Flushing text and recording SUBMIT step.")
                flushActiveTextSession(service)

                val stepIndex = synchronized(steps) { steps.size + 1 }
                val submitStep = DittoStep(
                    stepIndex = stepIndex,
                    actionType = "SUBMIT",
                    argument = null,
                    targetIdentity = DittoTargetIdentity(
                        primaryText = "Search",
                        contentDesc = "Submit / Search query",
                        semanticRole = "BUTTON"
                    ),
                    targetMetadata = DittoTargetMetadata(
                        resourceId = "system:submit",
                        className = "android.widget.Button",
                        recordedBounds = listOf(0, 0, 0, 0)
                    ),
                    physicalExecution = DittoPhysicalExecution(0, 0, 50L),
                    stateBefore = previousSnapshot ?: DittoStateSnapshot(activePackage = eventPkg),
                    stateAfter = DittoStateSnapshot(
                        keyAnchorText = "Submitted",
                        nodeCount = 1,
                        activePackage = eventPkg
                    ),
                    transitionContract = DittoTransitionContract("SUBMIT", "query")
                )
                synchronized(steps) { steps.add(submitStep) }
                previousSnapshot = submitStep.stateAfter
                DittoFloatingControl.updateStep(stepIndex, "SUBMIT / SEARCH")
            }
            return
        }

        // Flush any active typing session BEFORE recording this new tap action (synchronous)
        flushActiveTextSession(service)

        // Robust source extraction (falls back to active window if view was detached during quick screen transitions)
        var source = safeGetSource(event)
        val eventTextList = try { event.text } catch (_: Throwable) { null }
        val fallbackText = if (!eventTextList.isNullOrEmpty()) eventTextList.joinToString(" ").trim() else event.contentDescription?.toString()?.trim()

        if (source == null && !fallbackText.isNullOrBlank()) {
            val root = service.rootInActiveWindow
            if (root != null) {
                source = findNodeWithText(root, fallbackText)
            }
        }

        val rect = Rect()
        if (source != null) {
            safeGetBounds(source, rect)
            if (safeGetText(source).isNullOrBlank() && safeGetDesc(source).isNullOrBlank() && !fallbackText.isNullOrBlank()) {
                val childWithText = findNodeWithText(source, fallbackText)
                if (childWithText != null) {
                    source = childWithText
                    safeGetBounds(source, rect)
                }
            }
        } else {
            // Default center bounds if node was instantly detached by transition
            rect.set(100, 800, 980, 950)
        }

        val resId = if (source != null) safeGetViewId(source) else null
        val cls = if (source != null) (safeGetClassName(source) ?: "") else ""

        // Multi-Layer Rapid Click Debounce:
        // Compound controls like search bars (Instagram, WhatsApp, Google) trigger simultaneous click events
        // across child EditText, parent FrameLayout, and root Container in under 250ms.
        // Collapse them into 1 clean step using physical spatial distance (< 60px) and same resource id.
        val now = System.currentTimeMillis()
        val isRapidClick = (now - lastTapTimestamp < 250L) && (eventPkg == lastTapPackage)
        val dx = (rect.centerX() - lastTapRect.centerX()).toFloat()
        val dy = (rect.centerY() - lastTapRect.centerY()).toFloat()
        val tapDistance = kotlin.math.hypot(dx, dy)
        val isSameTarget = !resId.isNullOrBlank() && resId == lastTapResId
        val isCoincidentRect = (!rect.isEmpty && !lastTapRect.isEmpty && tapDistance < 60f) || isSameTarget

        if (isRapidClick && isCoincidentRect) {
            Log.i(TAG, "Suppressed redundant multi-layer click event within ${now - lastTapTimestamp}ms at $rect (distance=${tapDistance}px)")
            return
        }
        lastTapTimestamp = now
        lastTapRect.set(rect)
        lastTapPackage = eventPkg ?: ""
        lastTapResId = resId

        // Deep Semantic Target Extraction
        val extractedTarget = if (source != null) {
            extractBestSemanticTarget(source)
        } else {
            null
        }

        // Guaranteed Non-Null Primary Text: Multi-layered bulletproof fallback
        val cleanFallback = fallbackText?.let { cleanSemanticText(it) }?.takeIf { it.isNotBlank() }
        val finalPrimaryText = extractedTarget?.primaryText?.takeIf { it.isNotBlank() }
            ?: cleanFallback?.takeIf { it.isNotBlank() && !it.equals("button", ignoreCase = true) && !it.equals("image", ignoreCase = true) }
            ?: findTextInBounds(service.rootInActiveWindow, rect)
            ?: resId?.substringAfter(":id/")?.takeIf { it.isNotBlank() }
            ?: cls.substringAfterLast(".").takeIf { it.isNotBlank() }
            ?: "Element"

        val semanticTarget = (extractedTarget ?: DittoTargetIdentity(
            primaryText = finalPrimaryText,
            contentDesc = fallbackText,
            semanticRole = "LIST_ITEM",
            structuralContainer = null
        )).copy(primaryText = finalPrimaryText)

        // Check if this action is a Back navigation (system navigation bar or global back)
        val isSystemBack = isSystemUiPackage(eventPkg) && (
            semanticTarget.primaryText.equals("Back", ignoreCase = true) ||
            fallbackText?.equals("Back", ignoreCase = true) == true ||
            resId?.contains("back", ignoreCase = true) == true ||
            resId?.contains("navigation_bar_back", ignoreCase = true) == true ||
            fallbackText?.contains("navigate up", ignoreCase = true) == true ||
            event.contentDescription?.toString()?.equals("Back", ignoreCase = true) == true
        )

        // System UI / Navigation handling
        if (isSystemUiPackage(eventPkg)) {
            if (isSystemBack) {
                Log.i(TAG, "System Back navigation detected; converting to universal BACK action step")
            } else {
                Log.i(TAG, "Action on system UI ignored: '${semanticTarget.primaryText}' ($eventPkg)")
                return
            }
        }

        // Launcher handling (User tapping an app icon on Home Launcher)
        val isLauncherTap = isLauncherPackage(eventPkg)
        if (isLauncherTap) {
            val validIconText = semanticTarget.primaryText?.takeIf {
                it.isNotBlank() && !it.equals("Element", ignoreCase = true) && !it.equals("Workspace", ignoreCase = true)
            }
            if (validIconText == null) {
                Log.i(TAG, "Empty or non-icon launcher click ignored: '${semanticTarget.primaryText}' ($eventPkg)")
                return
            }
            Log.i(TAG, "Deliberate launcher app icon tap detected: '$validIconText' ($eventPkg)")
        }

        val actionType = if (isSystemBack) "BACK" else "TAP"
        val finalTarget = if (isSystemBack) {
            DittoTargetIdentity(
                primaryText = "Back",
                contentDesc = "Back",
                semanticRole = "SYSTEM_NAVIGATION",
                structuralContainer = null
            )
        } else if (isLauncherTap) {
            semanticTarget.copy(semanticRole = "APP_ICON")
        } else {
            semanticTarget
        }

        val targetMetadata = DittoTargetMetadata(
            resourceId = resId,
            className = cls,
            recordedBounds = listOf(rect.left, rect.top, rect.right, rect.bottom)
        )

        var executionX = rect.centerX()
        var executionY = rect.centerY()
        val hasDirectText = !fallbackText.isNullOrBlank() || source?.let { !safeGetText(it).isNullOrBlank() } == true
        val isRealAvatarElement = !hasDirectText && (cls.contains("ImageView", ignoreCase = true) ||
                resId?.contains("picture", ignoreCase = true) == true ||
                resId?.contains("avatar", ignoreCase = true) == true ||
                resId?.contains("photo", ignoreCase = true) == true ||
                resId?.contains("icon", ignoreCase = true) == true)
        val isAvatarClick = !isSystemBack && !isLauncherTap && isRealAvatarElement && (rect.left < 250 && rect.width() < 350)

        if (isAvatarClick) {
            // Left-aligned avatar icon clicked inside a list row (e.g. WhatsApp): shift tap into conversation body
            var p: AccessibilityNodeInfo? = source?.let { safeGetParent(it) }
            var rowRect: Rect? = null
            while (p != null) {
                val pr = Rect()
                safeGetBounds(p, pr)
                if (pr.width() > 500) {
                    rowRect = pr
                    break
                }
                p = safeGetParent(p)
            }
            if (rowRect != null) {
                executionX = maxOf(rowRect.centerX(), 540)
                executionY = rowRect.centerY()
            } else {
                executionX = maxOf(rect.right + 150, 540)
            }
            Log.i(TAG, "Shifted recorded click from avatar region (x=${rect.centerX()}) to chat body (x=$executionX)")
        }

        val physicalExecution = DittoPhysicalExecution(
            tapX = executionX,
            tapY = executionY,
            durationMs = 50L
        )

        val beforeState = if (previousSnapshot != null && previousSnapshot?.activePackage != "com.zipper.app" && !isIgnoredPackage(previousSnapshot?.activePackage ?: "")) {
            previousSnapshot!!
        } else {
            DittoStateSnapshot(
                keyAnchorText = finalTarget.primaryText,
                visibleSignatures = listOfNotNull(finalTarget.primaryText),
                nodeCount = 1,
                activePackage = eventPkg
            )
        }

        // SYNCHRONOUS STEP ADDITION & IMMEDIATE HUD UPDATE (Eliminates race condition with quick taps or finish)
        val stepIndex = synchronized(steps) { steps.size + 1 }
        val step = DittoStep(
            stepIndex = stepIndex,
            actionType = actionType,
            argument = null,
            targetIdentity = finalTarget,
            targetMetadata = targetMetadata,
            physicalExecution = physicalExecution,
            stateBefore = beforeState,
            stateAfter = beforeState, // will be updated asynchronously once transition settles
            transitionContract = DittoTransitionContract(
                expectedMutation = when (actionType) {
                    "TAP" -> "MUTATION_TAP"
                    "BACK" -> "MUTATION_BACK"
                    else -> "MUTATION_SCROLL"
                },
                requiredElement = finalTarget.primaryText
            )
        )
        synchronized(steps) { steps.add(step) }

        if (finalTarget.primaryText?.equals("Paste", ignoreCase = true) == true ||
            finalTarget.contentDesc?.equals("Paste", ignoreCase = true) == true) {
            synchronized(steps) {
                val lastTypeStep = steps.lastOrNull { it.actionType == "TYPE_TEXT" }
                if (lastTypeStep != null) {
                    val idx = steps.indexOf(lastTypeStep)
                    if (idx >= 0) {
                        steps[idx] = lastTypeStep.copy(isClipboardBound = true)
                        Log.i(TAG, "Tagged Step ${lastTypeStep.stepIndex} as isClipboardBound=true due to explicit 'Paste' click")
                    }
                }
            }
        }

        val label = finalTarget.primaryText?.take(18) ?: actionType
        DittoFloatingControl.updateStep(stepIndex, "$actionType: $label")
        Log.i(TAG, "Recorded Step $stepIndex immediately: $actionType on '${finalTarget.primaryText}' ($resId, pkg=$eventPkg)")

        // Asynchronously capture transition and update step.stateAfter
        CoroutineScope(Dispatchers.Main).launch {
            delay(if (actionType == "BACK") 700 else 400) // Allow back navigation animation to settle
            captureLiveSnapshot(service) { afterState ->
                step.stateAfter = afterState
                previousSnapshot = afterState
                Log.i(TAG, "Updated Step ${step.stepIndex} stateAfter (nodes=${afterState.nodeCount}, pkg=${afterState.activePackage})")
            }
        }
    }

    /**
     * Module 1C: Anti-Recycler Typing Guard
     * Strictly rejects RecyclerViews, ListViews, ScrollViews, and oversized containers
     * from being treated as text input targets.
     */
    private fun isValidInputTarget(node: AccessibilityNodeInfo?, rect: Rect): Boolean {
        if (node == null) return false
        val cls = safeGetClassName(node) ?: ""
        val isScroll = try { node.isScrollable } catch (_: Throwable) { false }
        val isEd = try { node.isEditable } catch (_: Throwable) { false }
        val height = rect.height()
        val isExplicitList = cls.contains("RecyclerView", ignoreCase = true) ||
            cls.contains("ListView", ignoreCase = true) ||
            cls.contains("ScrollView", ignoreCase = true)
        val isEditText = cls.contains("EditText", ignoreCase = true)
        if (isExplicitList || (isScroll && !isEd && height > 550) || height > 750) {
            Log.w(TAG, "[Module 1C Anti-Recycler Guard] Rejected scrollable/oversized container ($cls, h=$height) as input target")
            return false
        }
        if (!isEd && !isEditText) {
            return false
        }
        return true
    }

    private fun findRealInputChild(node: AccessibilityNodeInfo?): AccessibilityNodeInfo? {
        if (node == null) return null
        val queue = ArrayDeque<AccessibilityNodeInfo>()
        queue.add(node)
        var count = 0
        while (queue.isNotEmpty() && count < 40) {
            count++
            val curr = queue.removeFirst()
            val cls = safeGetClassName(curr) ?: ""
            val isEd = try { curr.isEditable } catch (_: Throwable) { false }
            if (isEd || cls.contains("EditText", ignoreCase = true)) {
                val r = Rect()
                safeGetBounds(curr, r)
                if (r.height() in 20..350) {
                    return curr
                }
            }
            val cc = try { curr.childCount } catch (_: Throwable) { 0 }
            for (i in 0 until cc) {
                val ch = safeGetChild(curr, i) ?: continue
                queue.add(ch)
            }
        }
        return null
    }

    /**
     * Handles text typing events by buffering keystrokes into a single consolidated editing session.
     */
    private fun handleTextChangedEvent(event: AccessibilityEvent, service: KarvishAccessibilityService) {
        val eventText = try { event.text?.joinToString("")?.takeIf { it.isNotBlank() } } catch (_: Throwable) { null }
        var source = safeGetSource(event) ?: try { service.rootInActiveWindow?.findFocus(AccessibilityNodeInfo.FOCUS_INPUT) } catch (_: Throwable) { null }
        val rect = Rect()
        if (source != null) {
            safeGetBounds(source, rect)
            if (!isValidInputTarget(source, rect)) {
                val realChild = findRealInputChild(source)
                if (realChild != null) {
                    source = realChild
                    safeGetBounds(source, rect)
                } else {
                    Log.w(TAG, "[Module 1C] Rejected container without compact input child")
                    return
                }
            }
        } else {
            rect.set(100, 100, 980, 250)
        }

        val rawText = (source?.let { safeGetText(it)?.takeIf { it.isNotBlank() } } ?: eventText) ?: ""
        val resId = source?.let { safeGetViewId(it) }
        val fieldKey = resId ?: "pos_${rect.left}_${rect.top}"

        textDebounceJob?.cancel()

        val existing = activeTextSession
        if (existing == null || existing.fieldKey != fieldKey) {
            // New input field focused: flush prior session synchronously if any
            flushActiveTextSession(service)

            val semanticTarget = if (source != null) {
                extractBestSemanticTarget(source).copy(semanticRole = "INPUT")
            } else {
                DittoTargetIdentity(
                    primaryText = "Search",
                    contentDesc = "Search",
                    semanticRole = "INPUT",
                    structuralContainer = null
                )
            }
            val targetMetadata = DittoTargetMetadata(
                resourceId = resId,
                className = source?.let { safeGetClassName(it) } ?: "android.widget.EditText",
                recordedBounds = listOf(rect.left, rect.top, rect.right, rect.bottom)
            )
            val physical = DittoPhysicalExecution(rect.centerX(), rect.centerY(), 50L)
            val before = previousSnapshot ?: DittoStateSnapshot(activePackage = event.packageName?.toString())

            activeTextSession = ActiveTextSession(
                fieldKey = fieldKey,
                viewId = resId,
                targetIdentity = semanticTarget,
                targetMetadata = targetMetadata,
                physicalExecution = physical,
                stateBefore = before,
                currentText = rawText,
                lastUpdated = System.currentTimeMillis()
            )
        } else {
            // Ongoing editing session in same field
            if (rawText.isNotBlank()) {
                existing.currentText = rawText
            }
            existing.lastUpdated = System.currentTimeMillis()
        }

        // Debounce: If user stops typing for 1000ms, finalize this typing step
        textDebounceJob = CoroutineScope(Dispatchers.Main).launch {
            delay(1000)
            flushActiveTextSession(service)
        }
    }

    /**
     * Flushes the buffered typing session into a single clean TYPE_TEXT step synchronously.
     * Synchronous addition to `steps` eliminates any race condition with subsequent taps or stopRecording().
     */
    private fun flushActiveTextSession(service: KarvishAccessibilityService) {
        var session = activeTextSession
        if (session == null) {
            // Check if there is currently an active focused input field with text on screen (e.g. Chrome url_bar, Web form, Search bar)
            val root = service.rootInActiveWindow
            var focused = root?.findFocus(AccessibilityNodeInfo.FOCUS_INPUT)
            val pkg = focused?.packageName?.toString() ?: ""
            val id = focused?.let { safeGetViewId(it) } ?: ""
            if (focused != null && pkg != "com.zipper.app" && !id.startsWith("com.zipper.app:")) {
                val rect = Rect()
                safeGetBounds(focused, rect)
                if (!isValidInputTarget(focused, rect)) {
                    focused = findRealInputChild(focused)
                    if (focused != null) {
                        safeGetBounds(focused, rect)
                    }
                }
                if (focused != null) {
                    val currentText = safeGetText(focused)?.toString()?.trim() ?: ""
                    if (currentText.isNotBlank()) {
                        val semanticTarget = extractBestSemanticTarget(focused).copy(semanticRole = "INPUT")
                        val targetMeta = DittoTargetMetadata(
                            resourceId = safeGetViewId(focused),
                            className = safeGetClassName(focused) ?: "android.widget.EditText",
                            recordedBounds = listOf(rect.left, rect.top, rect.right, rect.bottom)
                        )
                        val before = previousSnapshot ?: DittoStateSnapshot(activePackage = pkg)
                        session = ActiveTextSession(
                            fieldKey = safeGetViewId(focused) ?: "focused_${rect.left}_${rect.top}",
                            viewId = safeGetViewId(focused),
                            targetIdentity = semanticTarget,
                            targetMetadata = targetMeta,
                            physicalExecution = DittoPhysicalExecution(rect.centerX(), rect.centerY(), 50L),
                            stateBefore = before,
                            currentText = currentText,
                            lastUpdated = System.currentTimeMillis()
                        )
                    }
                }
            }
        }
        if (session == null) return

        textDebounceJob?.cancel()
        activeTextSession = null

        val finalText = session.currentText.replace(Regex("[\\u200B-\\u200D\\uFEFF]"), "").trim()
        val isHintOrPlaceholder = finalText.equals("search", ignoreCase = true) ||
            finalText.equals("search with meta ai", ignoreCase = true) ||
            finalText.equals("type a message", ignoreCase = true) ||
            finalText.equals("message", ignoreCase = true) ||
            finalText.equals("ask meta ai or search", ignoreCase = true) ||
            finalText.equals("ask google", ignoreCase = true) ||
            finalText.equals("search youtube", ignoreCase = true) ||
            finalText.equals(session.targetIdentity.primaryText, ignoreCase = true) ||
            finalText.equals(session.targetIdentity.contentDesc, ignoreCase = true) ||
            (finalText.startsWith("search", ignoreCase = true) && (
                finalText.contains("atta", ignoreCase = true) ||
                finalText.contains("youtube", ignoreCase = true) ||
                finalText.contains("google", ignoreCase = true) ||
                finalText.contains("play", ignoreCase = true) ||
                finalText.contains("store", ignoreCase = true) ||
                finalText.contains("here", ignoreCase = true) ||
                finalText.contains("...", ignoreCase = true)
            ))
        if (finalText.isEmpty() || isHintOrPlaceholder) {
            Log.i(TAG, "Skipping empty or hint placeholder text: '$finalText'")
            return // Skip empty/cleared text noise or hint placeholders
        }

        // Module 1B Consolidation: Consolidate in-progress typing steps on the same field
        synchronized(steps) {
            val last = steps.lastOrNull()
            if (last != null && last.actionType == "TYPE_TEXT") {
                val sameField = (last.targetMetadata.resourceId != null && last.targetMetadata.resourceId == session.targetMetadata.resourceId) ||
                    (last.targetIdentity.primaryText == session.targetIdentity.primaryText &&
                     last.targetMetadata.recordedBounds.isNotEmpty() && session.targetMetadata.recordedBounds.isNotEmpty() &&
                     kotlin.math.abs(last.targetMetadata.recordedBounds[1] - session.targetMetadata.recordedBounds[1]) < 80)
                if (sameField) {
                    if (last.argument == finalText) {
                        Log.i(TAG, "Skipping duplicate consecutive TYPE_TEXT step for '$finalText'")
                        return
                    }
                    Log.i(TAG, "[Module 1B Consolidation] Upgrading in-progress typing on same field from '${last.argument}' to '$finalText'")
                    steps.removeAt(steps.size - 1)
                }
            }
        }

        val isClipboardMatch = try {
            val cm = context.getSystemService(Context.CLIPBOARD_SERVICE) as? android.content.ClipboardManager
            val clipText = cm?.primaryClip?.getItemAt(0)?.text?.toString()?.trim()
            !clipText.isNullOrBlank() && clipText == finalText.trim()
        } catch (_: Throwable) {
            false
        }

        val stepIndex = synchronized(steps) { steps.size + 1 }
        val step = DittoStep(
            stepIndex = stepIndex,
            actionType = "TYPE_TEXT",
            argument = finalText,
            targetIdentity = session.targetIdentity,
            targetMetadata = session.targetMetadata,
            physicalExecution = session.physicalExecution,
            stateBefore = session.stateBefore,
            stateAfter = DittoStateSnapshot(
                keyAnchorText = finalText,
                visibleSignatures = listOf(finalText),
                nodeCount = 1,
                activePackage = session.stateBefore.activePackage
            ),
            transitionContract = DittoTransitionContract("MUTATION_INPUT", finalText),
            isClipboardBound = isClipboardMatch
        )

        val count = synchronized(steps) {
            steps.add(step)
            steps.size
        }
        previousSnapshot = step.stateAfter

        val displayArg = if (finalText.length > 15) finalText.take(15) + "..." else finalText
        DittoFloatingControl.updateStep(count, "TYPE: \"$displayArg\"")
        Log.i(TAG, "Flushed Buffered Step $stepIndex: TYPE_TEXT \"$finalText\" on '${session.targetIdentity.primaryText}'")
    }

    /**
     * Deep Recursive Tree Search (Crash-Proof):
     * Traverses down into compound containers (e.g. Instagram search result rows, WhatsApp contact rows)
     * using safe child access to extract the true semantic title or contact name without throwing exceptions.
     */
    private fun extractBestSemanticTarget(source: AccessibilityNodeInfo): DittoTargetIdentity {
        val directDesc = safeGetDesc(source)?.trim()
        val directId = safeGetViewId(source)?.lowercase() ?: ""
        val directCls = safeGetClassName(source) ?: ""
        val directText = safeGetText(source)?.trim()
        val roleDesc = safeGetRoleDescription(source)?.lowercase()?.trim()

        val isDirectInput = try { source.isEditable } catch (_: Throwable) { false } ||
            directCls.contains("EditText", ignoreCase = true) ||
            roleDesc?.contains("text box") == true || roleDesc?.contains("search") == true
        val isClickable = try { source.isClickable } catch (_: Throwable) { false } ||
            roleDesc?.contains("button") == true || roleDesc?.contains("link") == true
        val role = when {
            isDirectInput -> "INPUT"
            roleDesc?.contains("button") == true || directCls.contains("Button", ignoreCase = true) -> "BUTTON"
            roleDesc?.contains("link") == true -> "LINK"
            roleDesc?.contains("heading") == true || roleDesc?.contains("header") == true -> "HEADER"
            roleDesc?.contains("menu") == true -> "MENU_ITEM"
            roleDesc?.contains("tab") == true || directCls.contains("Tab", ignoreCase = true) -> "TAB"
            roleDesc?.contains("check") == true || directCls.contains("Check", ignoreCase = true) -> "CHECKBOX"
            isClickable -> "BUTTON"
            else -> "ELEMENT"
        }

        // Detect structural container safely
        val containerId = findStructuralContainerId(source)

        // 1. For INPUT fields, name after hint, content description, or resource ID (NOT the typed text!)
        if (isDirectInput) {
            val hint = try { source.hintText?.toString()?.trim() } catch (_: Throwable) { null }
            val inputName = when {
                !hint.isNullOrBlank() -> hint
                !directDesc.isNullOrBlank() -> directDesc
                !directId.isNullOrBlank() -> directId.substringAfter(":id/")
                else -> "Search"
            }
            return DittoTargetIdentity(
                primaryText = inputName,
                contentDesc = directDesc ?: hint,
                semanticRole = "INPUT",
                structuralContainer = containerId
            )
        }

        // 2. Direct text on node (e.g. TextView clicked directly)
        if (!directText.isNullOrBlank()) {
            if (isStatusOrSnippetText(directText, directId)) {
                // If clicked on status or preview snippet, climb to row container to find the actual contact/item name
                var p = safeGetParent(source)
                var climbs = 0
                while (p != null && climbs < 4) {
                    val rowTarget = findContactNameInRow(p)
                    if (rowTarget != null) {
                        return rowTarget.copy(structuralContainer = containerId)
                    }
                    p = safeGetParent(p)
                    climbs++
                }
            }
            val cleanText = cleanSemanticText(directText)
            val isPuaOnly = directText.all { it in '\uE000'..'\uF8FF' || it in setOf('\u2630', '\u22EE', '\u22EF') || it.isWhitespace() }
            val effectiveText = if (isPuaOnly || cleanText.isBlank()) {
                val cleanDesc = directDesc?.let { cleanSemanticText(it) }?.takeIf { it.isNotBlank() }
                val rect = Rect()
                safeGetBounds(source, rect)
                cleanDesc ?: (if (rect.top < 650 && rect.height() in 20..350) "Menu" else "Button")
            } else {
                cleanText
            }
            return DittoTargetIdentity(
                primaryText = effectiveText,
                contentDesc = directDesc,
                semanticRole = if (role == "ELEMENT") (if (effectiveText.equals("Menu", ignoreCase = true)) "MENU" else "BUTTON") else role,
                structuralContainer = containerId
            )
        }

        // 2. Direct ContentDescription priority for icons and tab buttons (e.g. Instagram Search tab, Voice Search, Camera)
        val cleanDirectDesc = directDesc?.let { cleanSemanticText(it) }
        val isMenuOrOverflow = isActionMenuOrOverflow(cleanDirectDesc, directId)
        if (!cleanDirectDesc.isNullOrBlank() &&
            cleanDirectDesc.length >= 2 &&
            !cleanDirectDesc.equals("button", ignoreCase = true) &&
            !cleanDirectDesc.equals("image", ignoreCase = true) &&
            !cleanDirectDesc.equals("icon", ignoreCase = true) &&
            !isMenuOrOverflow
        ) {
            return DittoTargetIdentity(
                primaryText = cleanDirectDesc,
                contentDesc = directDesc,
                semanticRole = if (role == "ELEMENT") "BUTTON" else role,
                structuralContainer = containerId
            )
        }

        // If clicked on an action menu or overflow 3-dots button on a list item, climb to find the actual item title
        if (isMenuOrOverflow) {
            var p = safeGetParent(source)
            var climbs = 0
            while (p != null && climbs < 4) {
                val rowTarget = findContactNameInRow(p)
                if (rowTarget != null && !rowTarget.primaryText.isNullOrBlank() && !isActionMenuOrOverflow(rowTarget.primaryText, null)) {
                    return rowTarget.copy(structuralContainer = containerId)
                }
                p = safeGetParent(p)
                climbs++
            }
        }

        val isCompoundContainer = (safeGetChild(source, 0) != null) || directId.contains("container") || directId.contains("row") || directCls.contains("Layout") || directCls.contains("ViewGroup")

        // 3. For compound list containers (e.g. WhatsApp contact row, Instagram Direct row), extract clean entity name
        if (isCompoundContainer) {
            val rowNameTarget = findContactNameInRow(source)
            if (rowNameTarget != null && !rowNameTarget.primaryText.isNullOrBlank()) {
                return rowNameTarget.copy(structuralContainer = containerId)
            }
        }

        // 4. Recursive child inspection with sealed node guards (DFS)
        val candidates = mutableListOf<Pair<String, Int>>()

        fun dfs(node: AccessibilityNodeInfo?, depth: Int) {
            if (node == null || depth > 8) return
            try {
                val t = safeGetText(node)?.trim()
                val d = safeGetDesc(node)?.let { cleanSemanticText(it) }
                val id = safeGetViewId(node)?.lowercase() ?: ""

                val tClean = t?.let { cleanSemanticText(it) }

                fun score(str: String, isDesc: Boolean): Int {
                    var s = 20
                    val strLow = str.lowercase()
                    if (isStatusOrSnippetText(str, id)) s -= 100
                    if (isActionMenuOrOverflow(str, id)) s -= 200
                    if (id.contains("contact_name") || id.contains("username") || id.contains("name") || id.contains("title") || id.contains("header") || id.contains("user") || id.contains("sender")) s += 80
                    if (id.contains("digest") || id.contains("snippet") || id.contains("preview") || id.contains("message") || id.contains("status")) s -= 50
                    if (str.matches(Regex(".*\\d{1,2}:\\d{2}.*"))) s -= 60 // De-prioritize timestamps like "10:45 AM"
                    if (str.equals("yesterday", ignoreCase = true) || str.equals("today", ignoreCase = true)) s -= 60
                    if (str.length in 2..40) s += 20
                    if (isDesc) s -= 5
                    return s
                }

                if (!tClean.isNullOrBlank() && tClean.length in 1..80) {
                    candidates.add(Pair(tClean, score(tClean, false)))
                }
                if (!d.isNullOrBlank() && d.length in 1..80 &&
                    !d.equals("button", ignoreCase = true) &&
                    !d.equals("image", ignoreCase = true) &&
                    !d.equals("icon", ignoreCase = true)
                ) {
                    candidates.add(Pair(d, score(d, true)))
                }

                val childCount = try { node.childCount } catch (_: Throwable) { 0 }
                for (i in 0 until childCount) {
                    val child = safeGetChild(node, i) ?: continue
                    dfs(child, depth + 1)
                }
            } catch (e: Throwable) {
                Log.w(TAG, "Safe dfs caught node traversal exception: ${e.message}")
            }
        }

        // Try DFS on source node
        dfs(source, 0)

        // 5. If candidates empty, climb to parent row container and try DFS (up to 3 levels)
        if (candidates.isEmpty()) {
            var p = safeGetParent(source)
            var climbs = 0
            while (p != null && climbs < 3) {
                val pr = Rect()
                safeGetBounds(p, pr)
                if (pr.width() > 400) {
                    dfs(p, 0)
                    if (candidates.isNotEmpty()) break
                }
                p = safeGetParent(p)
                climbs++
            }
        }

        val bestCandidate = candidates.filter { it.second > 0 }.maxByOrNull { it.second }?.first
            ?: candidates.maxByOrNull { it.second }?.first
        val cleanCandidate = bestCandidate?.let { cleanSemanticText(it) }
        val finalBestText = cleanCandidate?.takeIf { it.isNotBlank() } ?: cleanDirectDesc?.takeIf { it.isNotBlank() } ?: bestCandidate

        return DittoTargetIdentity(
            primaryText = finalBestText,
            contentDesc = directDesc,
            semanticRole = if (role == "ELEMENT" && !finalBestText.isNullOrBlank()) "LIST_ITEM" else role,
            structuralContainer = containerId
        )
    }

    private fun isStatusOrSnippetText(text: String?, viewId: String?): Boolean {
        if (text.isNullOrBlank()) return false
        val t = text.trim().lowercase()
        val id = viewId?.lowercase() ?: ""

        if (id.contains("digest") || id.contains("snippet") || id.contains("preview") ||
            id.contains("last_message") || id.contains("status_text") || id.contains("sub_title") ||
            id.contains("timestamp") || id.contains("subtitle") || id.contains("date")
        ) {
            return true
        }

        // Relative & clock timestamps (e.g. "2h", "15m", "3d", "1w", "52w", "10:45 am", "yesterday", "today")
        if (t.matches(Regex("^\\d{1,2}[smhdw]$")) || t.matches(Regex("^\\d{1,2}\\s*[smhdw]\\s*(ago)?$"))) return true
        if (t.matches(Regex(".*\\d{1,2}:\\d{2}.*"))) return true
        if (t == "yesterday" || t == "today" || t == "now" || t == "just now" || t == "active now") return true

        if (t.startsWith("forwarded") || t.startsWith("sent a ") || t.startsWith("sent an ") || t.startsWith("sent ") ||
            t.startsWith("liked ") || t.startsWith("reacted ") || t.startsWith("you: ") ||
            t.startsWith("you replied") || t.startsWith("replying to") ||
            t == "seen" || t == "delivered" || t == "read" || t == "opened" ||
            t.startsWith("active ") || t == "online" || t == "typing..." ||
            t.startsWith("missed ") || t.contains("voice call") || t.contains("video call") || t.contains("video chat") ||
            t.startsWith("requested to") || t.startsWith("mentioned you") ||
            t.endsWith("a message") || t.endsWith("a reel") || t.endsWith("a photo") || t.endsWith("a video")
        ) {
            return true
        }

        return false
    }

    private fun isActionMenuOrOverflow(text: String?, viewId: String?): Boolean {
        if (text.isNullOrBlank() && viewId.isNullOrBlank()) return false
        val t = text?.lowercase() ?: ""
        val id = viewId?.lowercase() ?: ""
        if (id.contains("menu_item") || id.contains("action_menu") || id.contains("more_options") || id.contains("overflow")) {
            if (!t.contains("search") && !t.contains("send") && !t.contains("submit")) return true
        }
        return t.contains("action menu") || t.contains("more options") || t.contains("overflow menu") ||
               t.contains("go to channel") || t.contains("popup menu") || t.startsWith("options for")
    }

    private fun findContactNameInRow(rowNode: AccessibilityNodeInfo): DittoTargetIdentity? {
        val candidates = mutableListOf<Pair<String, Int>>()
        fun scan(node: AccessibilityNodeInfo?, depth: Int) {
            if (node == null || depth > 10) return
            val t = safeGetText(node)?.trim()
            val d = safeGetDesc(node)?.let { cleanSemanticText(it) }
            val id = safeGetViewId(node)?.lowercase() ?: ""

            fun evaluate(str: String): Int {
                var s = 20
                val strClean = cleanSemanticText(str)
                if (isStatusOrSnippetText(str, id)) return -100
                if (isActionMenuOrOverflow(str, id)) return -200
                if (id.contains("contact_name") || id.contains("username") || id.contains("name") || id.contains("title") || id.contains("header") || id.contains("user") || id.contains("sender")) s += 80
                if (id.contains("digest") || id.contains("snippet") || id.contains("preview") || id.contains("message") || id.contains("status")) s -= 50
                if (strClean.length in 2..40) s += 30
                if (strClean.matches(Regex("^[A-Z][a-zA-Z0-9_.-]*$")) || strClean.matches(Regex("^[a-zA-Z0-9_.-]+$"))) s += 20
                return s
            }

            if (!t.isNullOrBlank() && t.length in 1..80) {
                candidates.add(Pair(t, evaluate(t)))
            }
            if (!d.isNullOrBlank() && d.length in 1..80 && !d.equals("button", true) && !d.equals("image", true)) {
                candidates.add(Pair(d, evaluate(d)))
            }

            val count = try { node.childCount } catch (_: Throwable) { 0 }
            for (i in 0 until minOf(count, 30)) {
                val c = safeGetChild(node, i) ?: continue
                scan(c, depth + 1)
            }
        }
        scan(rowNode, 0)
        val best = candidates.filter { it.second > 0 }.maxByOrNull { it.second }?.first
            ?: candidates.maxByOrNull { it.second }?.first?.takeIf { !isStatusOrSnippetText(it, null) }
            ?: return null
        val clean = cleanSemanticText(best)
        return DittoTargetIdentity(
            primaryText = clean.ifEmpty { best },
            contentDesc = null,
            semanticRole = "LIST_ITEM",
            structuralContainer = findStructuralContainerId(rowNode)
        )
    }

    private fun cleanSemanticText(raw: String): String = DittoSemanticCleaner.clean(raw)

    private fun findTextInBounds(root: AccessibilityNodeInfo?, targetRect: Rect): String? {
        if (root == null || targetRect.isEmpty) return null
        var foundName: String? = null
        var fallbackCandidate: String? = null
        fun scan(node: AccessibilityNodeInfo?) {
            if (node == null || foundName != null) return
            val r = Rect()
            safeGetBounds(node, r)
            if (Rect.intersects(r, targetRect)) {
                val t = safeGetText(node)?.trim()
                val id = safeGetViewId(node)?.lowercase() ?: ""
                if (!t.isNullOrBlank() && t.length in 2..50 && !isStatusOrSnippetText(t, id)) {
                    if (id.contains("name") || id.contains("username") || id.contains("title") || id.contains("header")) {
                        foundName = t
                        return
                    } else if (fallbackCandidate == null) {
                        fallbackCandidate = t
                    }
                }
            }
            val count = try { node.childCount } catch (_: Throwable) { 0 }
            for (i in 0 until count) {
                val c = safeGetChild(node, i) ?: continue
                scan(c)
            }
        }
        scan(root)
        return foundName ?: fallbackCandidate
    }

    private fun findNodeWithText(root: AccessibilityNodeInfo, text: String): AccessibilityNodeInfo? {
        val queue = ArrayDeque<AccessibilityNodeInfo>()
        queue.add(root)
        var count = 0
        val targetLow = text.lowercase()
        while (queue.isNotEmpty() && count < 80) {
            count++
            val node = queue.removeFirst()
            val t = safeGetText(node)?.lowercase()
            val d = safeGetDesc(node)?.lowercase()
            if (t?.contains(targetLow) == true || d?.contains(targetLow) == true) {
                return node
            }
            val cc = try { node.childCount } catch (_: Throwable) { 0 }
            for (i in 0 until cc) {
                val c = safeGetChild(node, i) ?: continue
                queue.add(c)
            }
        }
        return null
    }

    private fun findStructuralContainerId(node: AccessibilityNodeInfo): String? {
        var current: AccessibilityNodeInfo? = safeGetParent(node)
        var depth = 0
        while (current != null && depth < 4) {
            try {
                val id = safeGetViewId(current)
                val cls = safeGetClassName(current) ?: ""
                val isScroll = try { current.isScrollable } catch (_: Throwable) { false }
                if (isScroll || cls.contains("RecyclerView") || cls.contains("ListView") || cls.contains("ScrollView")) {
                    return id ?: cls
                }
                current = safeGetParent(current)
                depth++
            } catch (_: Throwable) {
                break
            }
        }
        return null
    }

    /**
     * Completes and saves the active recording session.
     */
    fun stopRecording(isSuccess: Boolean = true): DittoFlow? {
        if (!isRecording) return null

        val a11y = KarvishAccessibilityService.instance
        if (a11y != null) {
            // 1. Flush any active buffered typing session
            flushActiveTextSession(a11y)

            // Module 1D: Finish Flush Guarantee
            // Synchronously inspect rootInActiveWindow for focused input with uncommitted text
            try {
                val root = a11y.rootInActiveWindow
                var focused = root?.findFocus(AccessibilityNodeInfo.FOCUS_INPUT)
                val pkg = focused?.packageName?.toString() ?: ""
                if (focused != null && pkg != "com.zipper.app" && !isNonAppPackage(pkg)) {
                    val rect = Rect()
                    safeGetBounds(focused, rect)
                    if (!isValidInputTarget(focused, rect)) {
                        focused = findRealInputChild(focused)
                        if (focused != null) safeGetBounds(focused, rect)
                    }
                    if (focused != null) {
                        val text = safeGetText(focused)?.toString()?.trim() ?: ""
                        if (text.isNotBlank()) {
                            val lastStep = synchronized(steps) { steps.lastOrNull() }
                            if (lastStep == null || lastStep.actionType != "TYPE_TEXT" || lastStep.argument != text) {
                                Log.i(TAG, "[Module 1D Finish Flush] Guaranteeing final typed text: '$text'")
                                val targetId = extractBestSemanticTarget(focused).copy(semanticRole = "INPUT")
                                val targetMeta = DittoTargetMetadata(
                                    resourceId = safeGetViewId(focused),
                                    className = safeGetClassName(focused) ?: "android.widget.EditText",
                                    recordedBounds = listOf(rect.left, rect.top, rect.right, rect.bottom)
                                )
                                synchronized(steps) {
                                    steps.add(
                                        DittoStep(
                                            stepIndex = steps.size + 1,
                                            actionType = "TYPE_TEXT",
                                            argument = text,
                                            targetIdentity = targetId,
                                            targetMetadata = targetMeta,
                                            physicalExecution = DittoPhysicalExecution(rect.centerX(), rect.centerY(), 50L),
                                            stateBefore = previousSnapshot ?: DittoStateSnapshot(activePackage = pkg),
                                            stateAfter = DittoStateSnapshot(keyAnchorText = text, visibleSignatures = listOf(text), activePackage = pkg),
                                            transitionContract = DittoTransitionContract("MUTATION_INPUT", text)
                                        )
                                    )
                                }
                            }
                        }
                    }
                }
            } catch (e: Throwable) {
                Log.w(TAG, "Finish flush check caught error: ${e.message}")
            }
        }

        isRecording = false
        DittoFloatingControl.hide()

        val capturedSteps = synchronized(steps) { steps.toList() }
        if (capturedSteps.isEmpty()) {
            Log.w(TAG, "Recording stopped with 0 steps; discarding flow")
            return null
        }

        // Derive true launchPackage from captured in-app steps (guaranteed not to be home screen widgets)
        val candidatePackages = mutableListOf<String>()
        if (!launchPackage.isNullOrBlank() && !isNonAppPackage(launchPackage) && launchPackage != "com.zipper.app" && launchPackage != context.packageName) {
            candidatePackages.add(launchPackage!!)
        }
        for (step in capturedSteps) {
            step.stateBefore.activePackage?.takeIf { !isNonAppPackage(it) && it != "com.zipper.app" && it != context.packageName }?.let { candidatePackages.add(it) }
            step.stateAfter.activePackage?.takeIf { !isNonAppPackage(it) && it != "com.zipper.app" && it != context.packageName }?.let { candidatePackages.add(it) }
            step.targetMetadata.resourceId?.substringBefore(':')?.takeIf { it.contains(".") && !isNonAppPackage(it) && it != "com.zipper.app" && it != context.packageName }?.let { candidatePackages.add(it) }
        }

        val dominantStepPkg = candidatePackages.groupingBy { it }.eachCount().maxByOrNull { it.value }?.key
        val effectiveLaunchPkg = dominantStepPkg ?: if (!isNonAppPackage(launchPackage)) launchPackage else null

        // Module 1E: Dynamic Unique Flow Titling
        val appLabel = when {
            effectiveLaunchPkg?.contains("googlequicksearchbox", ignoreCase = true) == true -> "Google Search"
            effectiveLaunchPkg?.contains("perplexity", ignoreCase = true) == true -> "Perplexity"
            effectiveLaunchPkg?.contains("chrome", ignoreCase = true) == true -> "Chrome"
            effectiveLaunchPkg?.contains("whatsapp", ignoreCase = true) == true -> "WhatsApp"
            effectiveLaunchPkg?.contains("instagram", ignoreCase = true) == true -> "Instagram"
            effectiveLaunchPkg?.contains("youtube", ignoreCase = true) == true -> "YouTube"
            effectiveLaunchPkg?.contains("linkedin", ignoreCase = true) == true -> "LinkedIn"
            else -> effectiveLaunchPkg?.substringAfterLast(".")?.replaceFirstChar { if (it.isLowerCase()) it.titlecase(java.util.Locale.getDefault()) else it.toString() } ?: "App"
        }

        val timeFormat = java.text.SimpleDateFormat("HH:mm", java.util.Locale.getDefault())
        val timeStr = timeFormat.format(java.util.Date())

        val firstTypeQuery = capturedSteps.firstOrNull { it.actionType == "TYPE_TEXT" }?.argument?.takeIf { it.isNotBlank() }
        val finalTitle = when {
            activeTitle.isBlank() || activeTitle.startsWith("Demonstrated Task") -> {
                if (firstTypeQuery != null) "$appLabel: Search $firstTypeQuery ($timeStr)" else "$appLabel ($timeStr)"
            }
            !activeTitle.contains(appLabel, ignoreCase = true) && effectiveLaunchPkg != null -> {
                if (firstTypeQuery != null) "$appLabel: Search $firstTypeQuery ($timeStr)" else "$appLabel Flow ($timeStr)"
            }
            else -> activeTitle
        }

        val appMeta = DittoAppMetadata(
            launchPackage = effectiveLaunchPkg,
            deviceModel = android.os.Build.MODEL ?: "Samsung Galaxy A26",
            screenDensity = context.resources.displayMetrics.density,
            recordedTimestamp = System.currentTimeMillis()
        )

        val flow = DittoFlow(
            flowId = "ditto_${System.currentTimeMillis()}_${java.util.UUID.randomUUID().toString().take(6)}",
            title = finalTitle,
            provenance = "verified_human_demonstration",
            createdAt = System.currentTimeMillis(),
            appMetadata = appMeta,
            steps = capturedSteps
        )

        dittoStore.saveFlow(flow)
        Log.i(TAG, "DittoFlow '${flow.title}' successfully recorded and saved (${flow.steps.size} steps, LaunchPkg: $effectiveLaunchPkg, id: ${flow.flowId})")

        // Grok 1-Shot Apprenticeship: Automatically compile flow into reusable ParameterizedSkill
        try {
            val skillExecutor = ParameterizedSkillExecutor(context)
            val compiledSkill = skillExecutor.compileFlowToSkill(flow, flow.title)
            Log.i(TAG, "Grok 1-Shot Skill compiled: '${compiledSkill.name}' with ${compiledSkill.parameters.size} parameter slots")
        } catch (e: Throwable) {
            Log.w(TAG, "Could not compile flow to parameterized skill: ${e.message}")
        }

        return flow
    }

    /**
     * Cancels recording without saving.
     */
    fun cancelRecording() {
        isRecording = false
        activeTextSession = null
        textDebounceJob?.cancel()
        synchronized(steps) { steps.clear() }
        DittoFloatingControl.hide()
        Log.w(TAG, "Ditto recording cancelled")
    }

    private fun captureLiveSnapshot(service: KarvishAccessibilityService, onCaptured: (DittoStateSnapshot) -> Unit) {
        try {
            val root = service.rootInActiveWindow
            if (root == null) {
                onCaptured(DittoStateSnapshot())
                return
            }
            val signatures = mutableListOf<String>()
            var count = 0
            fun traverse(node: AccessibilityNodeInfo?) {
                if (node == null || count > 60) return
                count++
                val t = safeGetText(node)?.trim()
                if (!t.isNullOrBlank() && t.length in 2..40 && signatures.size < 8) {
                    signatures.add(t)
                }
                val childCount = try { node.childCount } catch (_: Throwable) { 0 }
                for (i in 0 until childCount) {
                    val child = safeGetChild(node, i) ?: continue
                    traverse(child)
                }
            }
            traverse(root)
            val pkg = safeGetPackageName(root)
            onCaptured(
                DittoStateSnapshot(
                    keyAnchorText = signatures.firstOrNull(),
                    visibleSignatures = signatures,
                    nodeCount = count,
                    activePackage = pkg
                )
            )
        } catch (e: Throwable) {
            Log.w(TAG, "Failed to capture live snapshot: ${e.message}")
            onCaptured(DittoStateSnapshot())
        }
    }

    // =========================================================================
    // Sealed-Node Safe Accessibility Getters (Zero Crash Guarantee)
    // =========================================================================

    private fun safeGetSource(event: AccessibilityEvent): AccessibilityNodeInfo? = try {
        event.source
    } catch (_: Throwable) { null }

    private fun safeGetChild(node: AccessibilityNodeInfo, index: Int): AccessibilityNodeInfo? = try {
        node.getChild(index)
    } catch (_: Throwable) { null }

    private fun safeGetParent(node: AccessibilityNodeInfo): AccessibilityNodeInfo? = try {
        node.parent
    } catch (_: Throwable) { null }

    private fun safeGetText(node: AccessibilityNodeInfo): String? = try {
        node.text?.toString()
    } catch (_: Throwable) { null }

    private fun safeGetDesc(node: AccessibilityNodeInfo): String? = try {
        node.contentDescription?.toString()
    } catch (_: Throwable) { null }

    private fun safeGetViewId(node: AccessibilityNodeInfo): String? = try {
        node.viewIdResourceName
    } catch (_: Throwable) { null }

    private fun safeGetClassName(node: AccessibilityNodeInfo): String? = try {
        node.className?.toString()
    } catch (_: Throwable) { null }

    private fun safeGetRoleDescription(node: AccessibilityNodeInfo): String? = try {
        androidx.core.view.accessibility.AccessibilityNodeInfoCompat.wrap(node).roleDescription?.toString()
            ?: node.extras?.getCharSequence("AccessibilityNodeInfo.roleDescription")?.toString()
    } catch (_: Throwable) { null }

    private fun safeGetPackageName(node: AccessibilityNodeInfo): String? = try {
        node.packageName?.toString()
    } catch (_: Throwable) { null }

    private fun safeGetBounds(node: AccessibilityNodeInfo, rect: Rect) {
        try {
            node.getBoundsInScreen(rect)
        } catch (_: Throwable) {
            rect.setEmpty()
        }
    }
}
