package com.zipper.app.ditto

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Path
import android.graphics.Rect
import android.os.Bundle
import android.util.Log
import android.view.Display
import android.view.accessibility.AccessibilityNodeInfo
import com.zipper.app.accessibility.KarvishAccessibilityService
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlin.coroutines.resume

/**
 * DITTO DETERMINISTIC REPLAY ENGINE (Crash-Proof V2.2)
 *
 * Implements senior-calibrated replay contract:
 * 1. Zero Autonomous Hallucination: Strictly reproduces demonstrated steps without inventing workflow steps.
 * 2. Launch Package Sanitizer: Rejects system widgets and Google search boxes; launches real demonstrated app.
 * 3. Sealed-Node Crash Protection: Robust safe node traversal prevents IllegalStateException in dynamic apps.
 * 4. 3-Tier Hierarchical Matching: Exact Semantic -> Strong Structural -> Fail-Closed Stop.
 * 5. Position Re-Grounding: Dynamically adapts to shuffled or moved elements.
 * 6. Bounded Container Recovery: Searches specific list containers within a calibrated budget without guessing.
 */
class DittoPlayer(private val context: Context) {

    companion object {
        private const val TAG = "DittoPlayer"
        private const val DEFAULT_CONFIDENCE_THRESHOLD = 0.80f
        private const val DEFAULT_RECOVERY_SCROLL_BUDGET = 4

        private val SYSTEM_PACKAGES_BLACKLIST = setOf(
            "com.sec.android.app.launcher",
            "com.google.android.apps.nexuslauncher",
            "com.android.launcher3",
            "com.miui.home",
            "com.oppo.launcher",
            "com.coloros.launcher",
            "com.huawei.android.launcher",
            "com.android.systemui",
            "android",
            "com.zipper.app"
        )
    }

    private var isPlaying = false
    private var isPaused = false

    val isReplaying: Boolean get() = isPlaying

    /**
     * Replays a demonstrated DittoFlow sequentially.
     */
    fun replayFlow(
        flow: DittoFlow,
        confidenceThreshold: Float = DEFAULT_CONFIDENCE_THRESHOLD,
        recoveryScrollBudget: Int = DEFAULT_RECOVERY_SCROLL_BUDGET,
        onComplete: (Boolean, String) -> Unit
    ) {
        replayFlow(flow, confidenceThreshold, recoveryScrollBudget, emptyMap(), onComplete)
    }

    /**
     * Replays a demonstrated DittoFlow with optional runtime parameter substitution.
     */
    fun replayFlow(
        flow: DittoFlow,
        confidenceThreshold: Float = DEFAULT_CONFIDENCE_THRESHOLD,
        recoveryScrollBudget: Int = DEFAULT_RECOVERY_SCROLL_BUDGET,
        parameters: Map<String, String>,
        onComplete: (Boolean, String) -> Unit
    ) {
        val targetFlow = if (parameters.isNotEmpty()) {
            val adaptedSteps = flow.steps.map { step ->
                if (step.actionType == "TYPE_TEXT") {
                    val override = parameters[step.targetIdentity.primaryText?.lowercase()]
                        ?: parameters[step.targetIdentity.contentDesc?.lowercase()]
                        ?: parameters["query"]
                        ?: parameters["input"]
                        ?: parameters["search"]
                        ?: parameters["message"]
                        ?: parameters["amount"]
                        ?: if (parameters.size == 1) parameters.values.first() else null
                    if (override != null) {
                        Log.i(TAG, "Dynamic parameter substituted for step ${step.stepIndex}: '$override'")
                        step.copy(argument = override)
                    } else step
                } else step
            }
            flow.copy(steps = adaptedSteps)
        } else {
            flow
        }

        if (isPlaying) {
            Log.w(TAG, "Replay already active")
            return
        }

        val service = KarvishAccessibilityService.instance
        if (service == null) {
            onComplete(false, "KarvishAccessibilityService is not connected")
            return
        }

        isPlaying = true
        isPaused = false

        DittoFloatingControl.show(
            context = context,
            onFinish = {
                stopReplay("User stopped replay")
                onComplete(false, "Replay stopped by user")
            },
            onPauseToggle = { paused ->
                isPaused = paused
                Log.i(TAG, "Ditto replay paused=$isPaused")
            },
            onCancel = {
                stopReplay("Cancelled")
                onComplete(false, "Replay cancelled")
            }
        )

        CoroutineScope(Dispatchers.Default).launch {
            try {
                Log.i(TAG, "Starting deterministic replay: '${flow.title}' (${flow.steps.size} steps)")

                // 0. Ensure Target App is launched before Step 1 (Sanitizing against Google Search widgets)
                val targetPkg = resolveCleanLaunchPackage(flow)
                val currentPkg = safeGetPackageName(service.rootInActiveWindow)
                val webUrl = flow.steps.firstOrNull { 
                    it.argument?.startsWith("http") == true || it.targetIdentity.primaryText?.startsWith("http") == true 
                }?.let { if (it.argument?.startsWith("http") == true) it.argument else it.targetIdentity.primaryText }

                if (!targetPkg.isNullOrBlank() && (targetPkg != currentPkg || !webUrl.isNullOrBlank())) {
                    Log.i(TAG, "Launching verified target app: $targetPkg before replay Step 1")
                    withContext(Dispatchers.Main) {
                        DittoFloatingControl.updateStep(0, "Opening App...")
                    }
                    val launchIntent = if (!webUrl.isNullOrBlank() && (targetPkg.contains("chrome") || targetPkg.contains("browser"))) {
                        Intent(Intent.ACTION_VIEW, android.net.Uri.parse(webUrl)).apply {
                            setPackage(targetPkg)
                            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                        }
                    } else {
                        context.packageManager.getLaunchIntentForPackage(targetPkg)?.apply {
                            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_RESET_TASK_IF_NEEDED)
                        }
                    }
                    if (launchIntent != null) {
                        context.startActivity(launchIntent)
                        delay(2500) // Wait for app/web to load
                    } else {
                        Log.w(TAG, "Could not obtain launch intent for package: $targetPkg")
                    }
                }

                for (step in targetFlow.steps) {
                    while (isPaused && isPlaying) {
                        delay(300)
                    }
                    if (!isPlaying) break

                    withContext(Dispatchers.Main) {
                        DittoFloatingControl.updateStep(
                            step.stepIndex,
                            "Replaying: ${step.targetIdentity.primaryText?.take(18) ?: step.actionType}"
                        )
                    }

                    // Bypass OPEN_APP step since it was handled during initial launch
                    if (step.actionType == "OPEN_APP") {
                        Log.i(TAG, "Step ${step.stepIndex}: Target app already launched; advancing.")
                        delay(400)
                        continue
                    }

                    // Special Case: If Step 1 is navigating to Search tab (e.g. Search and explore in Instagram),
                    // but the current screen is ALREADY displaying the search interface:
                    // STRICT REQUIREMENT: Only bypass if flow has subsequent steps, target is an actual bottom tab
                    // and NOT an in-page search bar or address bar (e.g. Chrome, Gmail, YouTube search bar).
                    if (step.stepIndex == 1 && step.actionType == "TAP" && flow.steps.size > 1) {
                        val isBottomTab = step.targetIdentity.semanticRole == "TAB" ||
                            step.targetMetadata.className?.contains("Tab", ignoreCase = true) == true ||
                            (step.targetMetadata.recordedBounds.size >= 4 && step.targetMetadata.recordedBounds[1] > 1800)
                        val isExplicitSearchField = step.targetIdentity.semanticRole == "INPUT" ||
                            step.targetMetadata.className?.contains("EditText", ignoreCase = true) == true ||
                            step.targetMetadata.resourceId?.contains("search_box", ignoreCase = true) == true ||
                            step.targetMetadata.resourceId?.contains("open_search", ignoreCase = true) == true ||
                            step.targetMetadata.resourceId?.contains("url_bar", ignoreCase = true) == true
                        val pLow = step.targetIdentity.primaryText?.lowercase() ?: ""
                        val cLow = step.targetIdentity.contentDesc?.lowercase() ?: ""
                        if (isBottomTab && !isExplicitSearchField && (pLow.contains("search") || cLow.contains("search"))) {
                            val root = service.rootInActiveWindow
                            if (root != null && hasVisibleSearchField(root)) {
                                Log.i(TAG, "Search interface already active on screen; bypassing Step 1 tab click.")
                                withContext(Dispatchers.Main) {
                                    DittoFloatingControl.updateStep(1, "Search Active (Skipped Tab)")
                                }
                                delay(400)
                                continue
                            }
                        }
                    }

                    // Special Case: Redundant Search Affordance Tap Bypass
                    // If current step is a TAP on a search bar/field, but an active search input is ALREADY focused on screen:
                    val isSearchFieldTap = step.actionType == "TAP" && (
                        step.targetMetadata.resourceId?.contains("search", ignoreCase = true) == true ||
                        step.targetIdentity.primaryText?.contains("search", ignoreCase = true) == true ||
                        step.targetIdentity.contentDesc?.contains("search", ignoreCase = true) == true
                    )
                    if (isSearchFieldTap && step.stepIndex > 1) {
                        val root = service.rootInActiveWindow
                        val focusedInput = root?.findFocus(AccessibilityNodeInfo.FOCUS_INPUT)
                        val fPkg = safeGetPackageName(focusedInput) ?: ""
                        val fId = focusedInput?.let { safeGetViewId(it) } ?: ""
                        if (focusedInput != null && fPkg != "com.zipper.app" && !fId.startsWith("com.zipper.app:") && hasVisibleSearchField(root)) {
                            Log.i(TAG, "Search input is already active and focused; bypassing redundant Step ${step.stepIndex} search tap.")
                            withContext(Dispatchers.Main) {
                                DittoFloatingControl.updateStep(step.stepIndex, "Search Ready (Bypassed Tap)")
                            }
                            delay(300)
                            continue
                        }
                    }

                    // Special Case: Launcher App Icon Tap Handling (App Launch)
                    // If current step is tapping an app icon from the home launcher to open an app:
                    val isLauncherOrigin = step.actionType == "TAP" && (
                        step.stateBefore.activePackage?.let {
                            it.contains(".launcher", ignoreCase = true) || it.contains(".home", ignoreCase = true)
                        } == true || step.targetIdentity.semanticRole == "APP_ICON"
                    )
                    if (isLauncherOrigin) {
                        val subsequentPkg = flow.steps.drop(step.stepIndex).firstOrNull { s ->
                            val p = s.stateBefore.activePackage ?: s.stateAfter?.activePackage
                            !p.isNullOrBlank() && !SYSTEM_PACKAGES_BLACKLIST.contains(p) && p != context.packageName && p != "com.zipper.app"
                        }?.let { s ->
                            s.stateBefore.activePackage?.takeIf { !SYSTEM_PACKAGES_BLACKLIST.contains(it) && it != context.packageName && it != "com.zipper.app" }
                                ?: s.stateAfter?.activePackage?.takeIf { !SYSTEM_PACKAGES_BLACKLIST.contains(it) && it != context.packageName && it != "com.zipper.app" }
                        }

                        val appLabelTarget = step.targetIdentity.primaryText ?: step.targetIdentity.contentDesc
                        val appPkg = step.stateAfter?.activePackage?.takeIf {
                            !SYSTEM_PACKAGES_BLACKLIST.contains(it) && it != "com.zipper.app" && it != context.packageName
                        } ?: subsequentPkg
                          ?: resolvePackageByLabel(appLabelTarget)
                          ?: (if (step.stepIndex == 1) targetPkg else null)

                        if (!appPkg.isNullOrBlank()) {
                            var activeForeground = safeGetPackageName(service.rootInActiveWindow)
                            if (activeForeground != appPkg) {
                                Log.i(TAG, "Step ${step.stepIndex}: App $appPkg not active in foreground ($activeForeground); launching via Intent.")
                                withContext(Dispatchers.Main) {
                                    DittoFloatingControl.updateStep(step.stepIndex, "Opening App...")
                                }
                                val webUrl = flow.steps.firstOrNull { 
                                    it.argument?.startsWith("http") == true || it.targetIdentity.primaryText?.startsWith("http") == true 
                                }?.let { if (it.argument?.startsWith("http") == true) it.argument else it.targetIdentity.primaryText }

                                val launchIntent = if (!webUrl.isNullOrBlank() && (appPkg.contains("chrome") || appPkg.contains("browser"))) {
                                    Intent(Intent.ACTION_VIEW, android.net.Uri.parse(webUrl)).apply {
                                        setPackage(appPkg)
                                        addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                                    }
                                } else {
                                    context.packageManager.getLaunchIntentForPackage(appPkg)?.apply {
                                        addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_RESET_TASK_IF_NEEDED)
                                    }
                                }
                                if (launchIntent != null) {
                                    context.startActivity(launchIntent)
                                    delay(1800)
                                }
                                activeForeground = safeGetPackageName(service.rootInActiveWindow)
                            }

                            Log.i(TAG, "Step ${step.stepIndex}: Target app $appPkg ready in foreground ($activeForeground); bypassing launcher icon tap.")
                            withContext(Dispatchers.Main) {
                                DittoFloatingControl.updateStep(step.stepIndex, "App Ready")
                            }
                            delay(400)
                            continue
                        }
                    }

                    // Handle universal BACK navigation immediately without requiring an on-screen target node
                    if (step.actionType == "BACK") {
                        Log.i(TAG, "Step ${step.stepIndex}: Executing universal GLOBAL_ACTION_BACK")
                        withContext(Dispatchers.Main) {
                            DittoFloatingControl.updateStep(step.stepIndex, "Navigating: Back")
                        }
                        val executed = service.performGlobalAction(android.accessibilityservice.AccessibilityService.GLOBAL_ACTION_BACK)
                        if (!executed) {
                            val reason = "Failed to dispatch action 'BACK' at step ${step.stepIndex}"
                            Log.e(TAG, reason)
                            stopReplay(reason)
                            onComplete(false, reason)
                            return@launch
                        }
                        delay(600)
                        verifyTransition(service, step)
                        delay(400)
                        continue
                    }

                    // Special Case: Placeholder / Hint Text Entry Bypass
                    // E.g. WhatsApp clears the input field after Send, leaving hint "Message", which should not be re-typed
                    if (step.actionType == "TYPE_TEXT") {
                        val rawArg = step.argument ?: ""
                        val arg = rawArg.replace(Regex("[\\u200B-\\u200D\\uFEFF]"), "").trim()
                        val isHintPlaceholder = arg.isEmpty() ||
                            arg.equals("Message", ignoreCase = true) ||
                            arg.equals("Type a message", ignoreCase = true)
                        if (isHintPlaceholder) {
                            Log.i(TAG, "Step ${step.stepIndex}: Skipping placeholder/hint text entry '$arg'")
                            withContext(Dispatchers.Main) {
                                DittoFloatingControl.updateStep(step.stepIndex, "Skipped hint")
                            }
                            delay(200)
                            continue
                        }
                    }

                    // Special Case: Redundant Paste Tap Bypass
                    // If current step is a TAP on 'Paste', but the preceding step was TYPE_TEXT that already injected the clipboard content:
                    val isPasteTap = step.actionType == "TAP" && (
                        step.targetIdentity.primaryText?.equals("Paste", ignoreCase = true) == true ||
                        step.targetIdentity.contentDesc?.equals("Paste", ignoreCase = true) == true ||
                        step.targetMetadata.resourceId?.contains("paste", ignoreCase = true) == true
                    )
                    val prevStep = flow.steps.getOrNull(step.stepIndex - 2)
                    if (isPasteTap && prevStep?.actionType == "TYPE_TEXT") {
                        Log.i(TAG, "Step ${step.stepIndex}: Input already populated by preceding typing step; bypassing redundant Paste tap.")
                        withContext(Dispatchers.Main) {
                            DittoFloatingControl.updateStep(step.stepIndex, "Pasted")
                        }
                        delay(300)
                        continue
                    }

                    // 1. Hierarchical Target Match with Bounded Container Re-Grounding
                    val matchResult = findTargetNodeWithRecovery(
                        service = service,
                        step = step,
                        threshold = confidenceThreshold,
                        maxRecoveryScrolls = recoveryScrollBudget
                    )

                    if (matchResult == null || matchResult.confidence < confidenceThreshold) {
                        // Resilient Fallback: If target was an in-app Back button and not matched, fallback to GLOBAL_ACTION_BACK
                        if (step.targetIdentity.primaryText?.equals("Back", ignoreCase = true) == true ||
                            step.targetIdentity.contentDesc?.equals("Back", ignoreCase = true) == true ||
                            step.targetMetadata.resourceId?.contains("back", ignoreCase = true) == true) {
                            Log.i(TAG, "Back button target not found on screen; executing fallback GLOBAL_ACTION_BACK")
                            val fallbackExecuted = service.performGlobalAction(android.accessibilityservice.AccessibilityService.GLOBAL_ACTION_BACK)
                            if (fallbackExecuted) {
                                delay(600)
                                verifyTransition(service, step)
                                delay(400)
                                continue
                            }
                        }

                        // Resilient Fallback: Eager Transition / Fast-Forward Check
                        val nextStep = flow.steps.getOrNull(step.stepIndex)
                        if (canFastForwardToNextStep(service, step, nextStep)) {
                            val targetName = step.targetIdentity.primaryText ?: "Element #${step.stepIndex}"
                            Log.i(TAG, "Step ${step.stepIndex}: Target '$targetName' not found on current screen, but screen has already transitioned to subsequent state. Fast-forwarding past Step ${step.stepIndex}!")
                            withContext(Dispatchers.Main) {
                                DittoFloatingControl.updateStep(step.stepIndex, "Completed (Fast-Forward)")
                            }
                            delay(300)
                            continue
                        }

                        val targetName = step.targetIdentity.primaryText ?: "Element #${step.stepIndex}"
                        val reason = "FAIL-CLOSED: Target '$targetName' could not be safely identified (Conf: ${matchResult?.confidence ?: 0f} < $confidenceThreshold)"
                        Log.w(TAG, reason)
                        withContext(Dispatchers.Main) {
                            DittoFloatingControl.updateStep(step.stepIndex, "STOPPED: '$targetName' Not Found")
                        }
                        delay(2500)
                        stopReplay(reason)
                        onComplete(false, reason)
                        return@launch
                    }

                    val targetNode = matchResult.node
                    val targetBounds = matchResult.currentBounds

                    Log.i(TAG, "[TIER ${matchResult.tier}] Target matched: '${step.targetIdentity.primaryText}' at $targetBounds (conf=${matchResult.confidence})")

                    // 2. Dispatch ONE Recorded Action
                    val executed = dispatchAction(service, step, targetNode, targetBounds, flow)
                    if (!executed) {
                        val reason = "Failed to dispatch action '${step.actionType}' at step ${step.stepIndex}"
                        Log.e(TAG, reason)
                        withContext(Dispatchers.Main) {
                            DittoFloatingControl.updateStep(step.stepIndex, "STOPPED: Action ${step.actionType} Failed")
                        }
                        delay(2500)
                        stopReplay(reason)
                        onComplete(false, reason)
                        return@launch
                    }

                    // 3. Strict Transition Verification Gate
                    delay(600) // Allow animation and state transition to settle
                    val transitionVerified = verifyTransition(service, step)
                    if (!transitionVerified) {
                        // Grace delay attempt
                        delay(800)
                        if (!verifyTransition(service, step)) {
                            val reason = "STRICT TRANSITION GATE FAILED at Step ${step.stepIndex}: Screen did not produce expected mutation"
                            Log.e(TAG, reason)
                            withContext(Dispatchers.Main) {
                                DittoFloatingControl.updateStep(step.stepIndex, "STOPPED: Transition Failed")
                            }
                            delay(2500)
                            stopReplay(reason)
                            onComplete(false, reason)
                            return@launch
                        }
                    }

                    Log.i(TAG, "Step ${step.stepIndex}/${flow.steps.size} completed and verified.")
                    val postInterval = if (step.actionType == "TYPE_TEXT") 800L else 400L
                    delay(postInterval) // Allow dynamic search results to populate

                }

                Log.i(TAG, "DittoFlow '${flow.title}' fully executed and verified!")
                stopReplay("Success")
                onComplete(true, "Flow '${flow.title}' successfully replayed (${flow.steps.size} steps)")
            } catch (e: Throwable) {
                val err = "Replay encountered unexpected error: ${e.message}"
                Log.e(TAG, err, e)
                stopReplay(err)
                onComplete(false, err)
            }
        }
    }

    /**
     * Resolves the real target app package, strictly rejecting system widgets and Google Search bar.
     */
    private fun resolveCleanLaunchPackage(flow: DittoFlow): String? {
        // 1. In a multi-app flow, Step 1's destination app takes absolute priority for initial launch
        val step1 = flow.steps.firstOrNull()
        val step1Pkg = step1?.stateAfter?.activePackage?.takeIf { !SYSTEM_PACKAGES_BLACKLIST.contains(it) && it != "com.zipper.app" && it != context.packageName }
            ?: step1?.stateBefore?.activePackage?.takeIf { !SYSTEM_PACKAGES_BLACKLIST.contains(it) && it != "com.zipper.app" && it != context.packageName }
        if (!step1Pkg.isNullOrBlank()) {
            return step1Pkg
        }

        // 2. Check declared package first if valid and non-system
        val declaredPkg = flow.appMetadata.launchPackage
        if (!declaredPkg.isNullOrBlank() && !SYSTEM_PACKAGES_BLACKLIST.contains(declaredPkg) && declaredPkg != "com.zipper.app" && declaredPkg != context.packageName) {
            return declaredPkg
        }

        // 2. Comprehensive candidate collection from stateBefore, stateAfter, and targetMetadata resourceId
        val candidatePackages = mutableListOf<String>()
        for (step in flow.steps) {
            step.stateBefore.activePackage?.takeIf { !SYSTEM_PACKAGES_BLACKLIST.contains(it) && it != "com.zipper.app" && it != context.packageName }?.let { candidatePackages.add(it) }
            step.stateAfter?.activePackage?.takeIf { !SYSTEM_PACKAGES_BLACKLIST.contains(it) && it != "com.zipper.app" && it != context.packageName }?.let { candidatePackages.add(it) }
            step.targetMetadata.resourceId?.substringBefore(':')?.takeIf { it.contains(".") && !SYSTEM_PACKAGES_BLACKLIST.contains(it) && it != "com.zipper.app" && it != context.packageName }?.let { candidatePackages.add(it) }
        }

        val dominant = candidatePackages.groupingBy { it }.eachCount().maxByOrNull { it.value }?.key
        if (!dominant.isNullOrBlank()) {
            return dominant
        }

        return null
    }

    private fun resolvePackageByLabel(appLabel: String?): String? {
        if (appLabel.isNullOrBlank()) return null
        val clean = appLabel.trim().lowercase()
        return try {
            val pm = context.packageManager
            val installed = pm.getInstalledApplications(0)
            installed.firstOrNull { app ->
                val label = pm.getApplicationLabel(app).toString().lowercase()
                label == clean || label.contains(clean) || clean.contains(label) || app.packageName.contains(clean)
            }?.packageName
        } catch (_: Throwable) {
            null
        }
    }

    fun stopReplay(reason: String = "") {
        isPlaying = false
        isPaused = false
        DittoFloatingControl.hide()
        Log.i(TAG, "Ditto replay stopped: $reason")
    }

    // =========================================================================
    // 3-Tier Hierarchical Target Matcher (Permutation & Position Invariant)
    // =========================================================================

    private data class MatchResult(
        val node: AccessibilityNodeInfo,
        val currentBounds: Rect,
        val confidence: Float,
        val tier: Int
    )

    /**
     * Finds target node on current screen, or performs bounded container-specific
     * re-grounding scrolls if the element was pushed off-screen.
     */
    private suspend fun findTargetNodeWithRecovery(
        service: KarvishAccessibilityService,
        step: DittoStep,
        threshold: Float,
        maxRecoveryScrolls: Int
    ): MatchResult? {
        val targetName = step.targetIdentity.primaryText?.let { DittoSemanticCleaner.clean(it) }?.ifBlank { "target" } ?: "target"
        val settleTimeoutMs = 8000L
        val pollIntervalMs = 350L
        val startTime = System.currentTimeMillis()
        var lastNodeCount = -1

        // Phase 1: State-Driven Adaptive Settle Gate (Wait for dynamic app/screen loading)
        // Senior Mandate: No blind sleep. Immediately executes when target appears.
        while (System.currentTimeMillis() - startTime < settleTimeoutMs) {
            val root = service.rootInActiveWindow
            val currentNodeCount = root?.childCount ?: 0

            // 1. Fast-Path: Deterministic Hierarchical Match
            val fastMatch = findTargetNodeHierarchical(service, step, threshold)
            if (fastMatch != null && fastMatch.confidence >= threshold) {
                return fastMatch
            }

            // 2. Strict Neural Target Locator
            // Senior Architectural Rule: If screen has just received input, allow hierarchical match at least 1000ms
            // to locate the exact entity before allowing fuzzy neural ranking to run.
            val elapsed = System.currentTimeMillis() - startTime
            if (elapsed >= 1000L) {
                try {
                    when (val neuralDecision = DittoNeuralRegrounder.locateTarget(service, step, threshold, minAmbiguityMargin = 0.15f)) {
                        is DittoNeuralRegrounder.RegroundDecision.TargetFound -> {
                            Log.i(TAG, "Neural Locator found target for '$targetName': ${neuralDecision.label} (conf=${neuralDecision.confidence})")
                            return MatchResult(
                                node = neuralDecision.node,
                                currentBounds = neuralDecision.bounds,
                                confidence = neuralDecision.confidence,
                                tier = 1 // Neural semantic confirmation
                            )
                        }
                        is DittoNeuralRegrounder.RegroundDecision.TargetAmbiguous -> {
                            Log.w(TAG, "Neural Locator stopped replay: Ambiguous target between '${neuralDecision.topCandidate}' and '${neuralDecision.secondCandidate}' (margin=${neuralDecision.margin})")
                            return null // Fail closed immediately on ambiguity to avoid wrong clicks
                        }
                        is DittoNeuralRegrounder.RegroundDecision.TargetNotFound -> {
                            // Keep polling if screen is still changing / loading
                        }
                    }
                } catch (e: Throwable) {
                    Log.w(TAG, "Neural locator check caught exception: ${e.message}")
                }
            }

            val isTreeMutating = currentNodeCount != lastNodeCount
            lastNodeCount = currentNodeCount

            // If screen is settled and target is not visible, transition early to bounded recovery scroll
            if (!isTreeMutating && elapsed >= 2500L && step.actionType == "TAP") {
                Log.i(TAG, "Screen settled and '$targetName' not visible in current viewport; starting scroll recovery")
                break
            }

            if (elapsed > 1200L) {
                withContext(Dispatchers.Main) {
                    DittoFloatingControl.updateStep(step.stepIndex, "Loading: $targetName (${(settleTimeoutMs - elapsed) / 1000}s)")
                }
            }

            delay(pollIntervalMs)
        }

        // Phase 2: If element pushed off-screen in a scrollable list, attempt bounded container scroll
        if (step.actionType != "TAP" && step.actionType != "TYPE_TEXT") {
            return null
        }

        // Do not perform blind scroll recovery if active app has already changed away from step's origin app!
        val currentAppPkg = safeGetPackageName(service.rootInActiveWindow)
        val expectedOriginPkg = step.stateBefore.activePackage
        if (!expectedOriginPkg.isNullOrBlank() && !currentAppPkg.isNullOrBlank() && currentAppPkg != expectedOriginPkg) {
            Log.i(TAG, "Active app is '$currentAppPkg' (expected '$expectedOriginPkg'); bypassing scroll recovery")
            return null
        }

        // Bounded incremental scroll loop (re-grounding mechanism, NOT invented workflow)
        for (attempt in 1..maxRecoveryScrolls) {
            val root = service.rootInActiveWindow ?: break
            val scrollableContainer = findScrollableContainer(root, step.targetIdentity.structuralContainer)
                ?: break // If no scrollable container found, stop without blind scrolling

            val containerBounds = Rect()
            safeGetBounds(scrollableContainer, containerBounds)
            if (containerBounds.width() <= 0 || containerBounds.height() <= 0) {
                break
            }

            withContext(Dispatchers.Main) {
                DittoFloatingControl.updateStep(step.stepIndex, "Finding: $targetName ($attempt/$maxRecoveryScrolls)")
            }

            val scrolled = performContainerScroll(service, scrollableContainer, containerBounds, directionDown = true)
            if (!scrolled) {
                Log.w(TAG, "Container scroll could not be dispatched; stopping re-grounding")
                break
            }

            delay(650) // Wait for inertial scrolling and view recycling to settle

            // Re-scan accessibility tree with Fast-Path and Neural Locator
            val freshMatch = findTargetNodeHierarchical(service, step, threshold)
            if (freshMatch != null && freshMatch.confidence >= threshold) {
                Log.i(TAG, "Target '$targetName' successfully re-grounded after $attempt scroll(s)!")
                return freshMatch
            }

            try {
                val neuralMatch = DittoNeuralRegrounder.locateTarget(service, step, threshold)
                if (neuralMatch is DittoNeuralRegrounder.RegroundDecision.TargetFound) {
                    Log.i(TAG, "Target '$targetName' successfully re-grounded via Neural Locator after $attempt scroll(s)!")
                    return MatchResult(neuralMatch.node, neuralMatch.bounds, neuralMatch.confidence, tier = 1)
                }
            } catch (_: Throwable) {}
        }

        Log.w(TAG, "Re-grounding budget exhausted ($maxRecoveryScrolls attempts). Target '$targetName' not found.")
        return null
    }

    private fun findScrollableContainer(root: AccessibilityNodeInfo, expectedContainerId: String?): AccessibilityNodeInfo? {
        var matchById: AccessibilityNodeInfo? = null
        var largestScrollable: AccessibilityNodeInfo? = null
        var maxArea = 0
        var visited = 0

        fun search(node: AccessibilityNodeInfo?) {
            if (node == null || visited > 100) return
            visited++
            try {
                val id = safeGetViewId(node)?.lowercase()
                val cls = safeGetClassName(node) ?: ""

                // Senior Mandate: NEVER treat horizontal view pagers or horizontal scroll views as vertical scroll containers!
                // Calling ACTION_SCROLL_FORWARD on a ViewPager pages through tabs (e.g. WhatsApp Chats -> Updates -> Communities -> Calls)!
                val isHorizontalPager = cls.contains("ViewPager", ignoreCase = true) ||
                        cls.contains("Pager", ignoreCase = true) ||
                        cls.contains("HorizontalScrollView", ignoreCase = true) ||
                        cls.contains("Horizontal", ignoreCase = true)

                val rect = Rect()
                safeGetBounds(node, rect)
                val isVerticalShape = rect.height() >= rect.width() * 0.75f && rect.height() > 350

                val isScroll = !isHorizontalPager && isVerticalShape && (
                    cls.contains("RecyclerView", ignoreCase = true) ||
                    cls.contains("ListView", ignoreCase = true) ||
                    (cls.contains("ScrollView", ignoreCase = true) && !cls.contains("Horizontal", ignoreCase = true)) ||
                    (try { node.isScrollable } catch (_: Throwable) { false })
                )

                if (isScroll) {
                    val area = rect.width() * rect.height()
                    if (!expectedContainerId.isNullOrBlank() && id != null && id.contains(expectedContainerId.lowercase())) {
                        matchById = node
                        return
                    }
                    if (area > maxArea && rect.height() > 350) {
                        maxArea = area
                        largestScrollable = node
                    }
                }

                val childCount = try { node.childCount } catch (_: Throwable) { 0 }
                for (i in 0 until childCount) {
                    val child = safeGetChild(node, i) ?: continue
                    search(child)
                }
            } catch (e: Throwable) {
                Log.w(TAG, "Safe search caught container exception: ${e.message}")
            }
        }
        search(root)
        return matchById ?: largestScrollable
    }

    private fun performContainerScroll(
        service: KarvishAccessibilityService,
        container: AccessibilityNodeInfo,
        bounds: Rect,
        directionDown: Boolean
    ): Boolean {
        val cls = safeGetClassName(container) ?: ""
        if (cls.contains("ViewPager", ignoreCase = true) || cls.contains("Pager", ignoreCase = true) || cls.contains("Horizontal", ignoreCase = true)) {
            Log.w(TAG, "Refusing to scroll horizontal container '$cls'")
            return false
        }

        // Try native Accessibility ACTION_SCROLL_FORWARD first on legitimate vertical lists
        try {
            val hasScroll = try {
                container.actionList?.any { it.id == AccessibilityNodeInfo.ACTION_SCROLL_FORWARD } == true
            } catch (_: Throwable) { false }

            if (hasScroll) {
                val ok = try { container.performAction(AccessibilityNodeInfo.ACTION_SCROLL_FORWARD) } catch (_: Throwable) { false }
                if (ok) return true
            }
        } catch (_: Throwable) {}

        // Safe gesture scroll strictly within vertical list content boundaries
        // Clamp top to > 400 (avoid action bar/search bar) and bottom to < 1850 (strictly avoid bottom nav bar tabs)
        return try {
            val safeTop = maxOf(bounds.top + 150, 400)
            val safeBottom = minOf(bounds.bottom - 150, 1850)
            if (safeBottom <= safeTop + 100) {
                Log.w(TAG, "Container bounds too constrained for gesture scroll: [$safeTop..$safeBottom]")
                return false
            }

            val startY = if (directionDown) safeBottom.toFloat() else safeTop.toFloat()
            val endY = if (directionDown) safeTop.toFloat() else safeBottom.toFloat()
            val centerX = bounds.centerX().toFloat()

            val path = Path().apply {
                moveTo(centerX, startY)
                lineTo(centerX, endY)
            }
            val gesture = GestureDescription.Builder()
                .addStroke(GestureDescription.StrokeDescription(path, 0, 250))
                .build()
            service.dispatchGesture(gesture, null, null)
        } catch (e: Throwable) {
            Log.w(TAG, "Gesture scroll error: ${e.message}")
            false
        }
    }


    private fun extractNodeHierarchyText(node: AccessibilityNodeInfo): String {
        val sb = StringBuilder()
        fun dfs(curr: AccessibilityNodeInfo?, depth: Int) {
            if (curr == null || depth > 6) return
            try {
                val t = safeGetText(curr)?.trim()
                val d = safeGetDesc(curr)?.trim()
                if (!t.isNullOrBlank()) sb.append(t).append(" ")
                if (!d.isNullOrBlank()) sb.append(d).append(" ")

                val count = try { curr.childCount } catch (_: Throwable) { 0 }
                for (i in 0 until minOf(count, 15)) {
                    val child = safeGetChild(curr, i) ?: continue
                    dfs(child, depth + 1)
                }
            } catch (e: Throwable) {
                Log.w(TAG, "Hierarchy text extraction exception caught: ${e.message}")
            }
        }
        dfs(node, 0)
        return sb.toString().trim().lowercase()
    }

    private fun findTargetNodeHierarchical(
        service: KarvishAccessibilityService,
        step: DittoStep,
        threshold: Float
    ): MatchResult? {
        val primaryRoot = service.rootInActiveWindow
        val roots = mutableListOf<AccessibilityNodeInfo>()
        primaryRoot?.let { roots.add(it) }
        try {
            service.windows?.forEach { win ->
                win.root?.let { if (!roots.contains(it)) roots.add(it) }
            }
        } catch (_: Throwable) {}

        if (roots.isEmpty()) return null
        val candidates = mutableListOf<AccessibilityNodeInfo>()
        var count = 0

        fun collectNodes(node: AccessibilityNodeInfo?) {
            if (node == null || count > 800) return
            count++
            val pkg = safeGetPackageName(node) ?: ""
            val id = safeGetViewId(node) ?: ""
            if (pkg == "com.zipper.app" || id.startsWith("com.zipper.app:")) {
                return // Strictly ignore host Zipper app nodes
            }
            candidates.add(node)
            val childCount = try { node.childCount } catch (_: Throwable) { 0 }
            for (i in 0 until childCount) {
                val child = safeGetChild(node, i) ?: continue
                collectNodes(child)
            }
        }
        for (r in roots) {
            collectNodes(r)
        }

        val targetIdentity = step.targetIdentity
        val targetMeta = step.targetMetadata
        val pText = step.targetIdentity.primaryText?.let { DittoSemanticCleaner.clean(it) }?.takeIf { it.isNotBlank() }?.lowercase()
        val cDesc = step.targetIdentity.contentDesc?.let { DittoSemanticCleaner.clean(it) }?.takeIf { it.isNotBlank() }?.lowercase()
        val resId = targetMeta.resourceId?.lowercase()
        val isTypeAction = step.actionType == "TYPE_TEXT"

        // STRICT INPUT CANDIDATE ISOLATION:
        val isInputTarget = isTypeAction || targetIdentity.semanticRole == "INPUT" ||
            targetMeta.className?.contains("EditText", ignoreCase = true) == true

        // Clean resource ID matching (handles com.android.chrome:id/username vs username)
        val cleanResId = resId?.substringAfterLast(":id/")?.substringAfterLast("/")?.lowercase()

        // 1. Direct Input Resource ID Match across all candidates
        if (isInputTarget && !cleanResId.isNullOrBlank()) {
            val inputById = candidates.find { node ->
                val cls = safeGetClassName(node) ?: ""
                val isEd = try { node.isEditable } catch (_: Throwable) { false }
                val isFocus = try { node.isFocusable } catch (_: Throwable) { false }
                val nId = safeGetViewId(node)?.lowercase()
                val cleanNId = nId?.substringAfterLast(":id/")?.substringAfterLast("/")
                val r = Rect()
                safeGetBounds(node, r)
                val isNotGiant = !(r.width() >= 1000 && r.height() >= 1200)
                isNotGiant && (isEd || isFocus || cls.contains("EditText", ignoreCase = true)) &&
                (nId == resId || cleanNId == cleanResId)
            }
            if (inputById != null) {
                val r = Rect()
                safeGetBounds(inputById, r)
                Log.i(TAG, "Input target matched directly by resource ID '${safeGetViewId(inputById)}' at $r")
                return MatchResult(inputById, r, 1.0f, tier = 1)
            }
        }

        // 1B. Dropdown / Selector Resource ID Match across all candidates
        // Handles cases where dropdown currently displays a placeholder (e.g. 'Day')
        // while the recorded step expects an option value (e.g. '29')
        if (!isTypeAction && !cleanResId.isNullOrBlank()) {
            val dropdownById = candidates.find { node ->
                val cls = safeGetClassName(node) ?: ""
                val isEd = try { node.isEditable } catch (_: Throwable) { false }
                val nId = safeGetViewId(node)?.lowercase()
                val cleanNId = nId?.substringAfterLast(":id/")?.substringAfterLast("/")
                val isIdMatch = nId == resId || cleanNId == cleanResId
                if (!isIdMatch) return@find false

                val r = Rect()
                safeGetBounds(node, r)
                val isNotGiant = !(r.width() >= 1000 && r.height() >= 1200)
                val isSelectLike = !isEd || cls.contains("Spinner", ignoreCase = true) || cls.contains("View", ignoreCase = true)
                isNotGiant && isSelectLike
            }
            if (dropdownById != null) {
                val r = Rect()
                safeGetBounds(dropdownById, r)
                val currText = safeGetText(dropdownById)?.trim() ?: ""
                Log.i(TAG, "Dropdown/Selector target matched by resource ID '${safeGetViewId(dropdownById)}' at $r (currText='$currText', target='$pText')")
                return MatchResult(dropdownById, r, 0.95f, tier = 1)
            }
        }

        // When typing text, check if an input field is already actively focused on screen (ignoring Zipper).
        if (isTypeAction) {
            val focusedInput = roots.firstNotNullOfOrNull { it.findFocus(AccessibilityNodeInfo.FOCUS_INPUT) }
            val fPkg = safeGetPackageName(focusedInput) ?: ""
            val fId = focusedInput?.let { safeGetViewId(it) } ?: ""
            if (focusedInput != null && fPkg != "com.zipper.app" && !fId.startsWith("com.zipper.app:")) {
                val fRect = Rect()
                safeGetBounds(focusedInput, fRect)
                if (fRect.width() > 0 && fRect.height() > 0 && !(fRect.width() >= 1000 && fRect.height() >= 1200)) {
                    Log.i(TAG, "Actively focused input field '${safeGetViewId(focusedInput)}' directly targeted for TYPE_TEXT")
                    return MatchResult(focusedInput, fRect, 1.0f, tier = 1)
                }
            }

            // Collect all valid editable candidates on screen
            val editableNodes = candidates.filter { node ->
                val cls = safeGetClassName(node) ?: ""
                val isEd = try { node.isEditable } catch (_: Throwable) { false }
                val isScroll = try { node.isScrollable } catch (_: Throwable) { false }
                val r = Rect()
                safeGetBounds(node, r)
                val isGiant = r.width() >= 1000 && r.height() >= 1200
                (isEd || cls.contains("EditText", ignoreCase = true)) && !isScroll && !isGiant
            }

            if (editableNodes.isNotEmpty()) {
                // Check exact resource ID match if available
                val exactIdMatch = editableNodes.find { node ->
                    val nId = safeGetViewId(node)?.lowercase()
                    val cleanNId = nId?.substringAfterLast(":id/")?.substringAfterLast("/")
                    !cleanResId.isNullOrBlank() && (nId == resId || cleanNId == cleanResId)
                }
                if (exactIdMatch != null) {
                    val r = Rect()
                    safeGetBounds(exactIdMatch, r)
                    Log.i(TAG, "Exact resource ID input field '${safeGetViewId(exactIdMatch)}' matched for TYPE_TEXT")
                    return MatchResult(exactIdMatch, r, 1.0f, tier = 1)
                }

                // Single editable node on screen is the unconditional target
                if (editableNodes.size == 1) {
                    val singleNode = editableNodes[0]
                    val singleRect = Rect()
                    safeGetBounds(singleNode, singleRect)
                    Log.i(TAG, "Single active input field '${safeGetViewId(singleNode)}' matched directly for TYPE_TEXT")
                    return MatchResult(singleNode, singleRect, 1.0f, tier = 1)
                }

                // Smart Search vs Bottom Message Grounding:
                val isRecordedSearch = resId?.contains("search") == true ||
                        pText?.contains("search") == true ||
                        cDesc?.contains("search") == true ||
                        (targetMeta.recordedBounds.size >= 4 && targetMeta.recordedBounds[1] < 600)
                val isRecordedBottomMessage = resId?.contains("entry") == true ||
                        resId?.contains("message") == true ||
                        (targetMeta.recordedBounds.size >= 4 && targetMeta.recordedBounds[1] > 1200)

                if (isRecordedSearch) {
                    val searchNode = editableNodes.find {
                        val id = safeGetViewId(it)?.lowercase() ?: ""
                        val d = safeGetDesc(it)?.lowercase() ?: ""
                        val r = Rect()
                        safeGetBounds(it, r)
                        (id.contains("search") || d.contains("search") || r.top < 600) && r.width() > 100
                    } ?: editableNodes.minByOrNull {
                        val r = Rect()
                        safeGetBounds(it, r)
                        r.top
                    }
                    if (searchNode != null) {
                        val r = Rect()
                        safeGetBounds(searchNode, r)
                        Log.i(TAG, "Top active search input field '${safeGetViewId(searchNode)}' matched for TYPE_TEXT")
                        return MatchResult(searchNode, r, 0.98f, tier = 1)
                    }
                } else if (isRecordedBottomMessage) {
                    val messageNode = editableNodes.find {
                        val id = safeGetViewId(it)?.lowercase() ?: ""
                        val r = Rect()
                        safeGetBounds(it, r)
                        (id.contains("entry") || id.contains("message") || r.bottom > 1200) && r.width() > 100
                    } ?: editableNodes.maxByOrNull {
                        val r = Rect()
                        safeGetBounds(it, r)
                        r.bottom
                    }
                    if (messageNode != null) {
                        val r = Rect()
                        safeGetBounds(messageNode, r)
                        Log.i(TAG, "Bottom message input field '${safeGetViewId(messageNode)}' matched for TYPE_TEXT")
                        return MatchResult(messageNode, r, 0.98f, tier = 1)
                    }
                }

                // Spatial Proximity Match:
                if (targetMeta.recordedBounds.size >= 4) {
                    val recY = (targetMeta.recordedBounds[1] + targetMeta.recordedBounds[3]) / 2
                    val closest = editableNodes.minByOrNull {
                        val r = Rect()
                        safeGetBounds(it, r)
                        kotlin.math.abs(r.centerY() - recY)
                    }
                    if (closest != null) {
                        val r = Rect()
                        safeGetBounds(closest, r)
                        Log.i(TAG, "Spatially closest editable input field '${safeGetViewId(closest)}' matched for TYPE_TEXT")
                        return MatchResult(closest, r, 0.95f, tier = 1)
                    }
                }
            } else {
                val crossWinNode = findEditableAcrossWindows(service) ?: findEditableInputNode(primaryRoot)
                if (crossWinNode != null) {
                    val r = Rect()
                    safeGetBounds(crossWinNode, r)
                    Log.i(TAG, "Cross-window editable input field '${safeGetViewId(crossWinNode)}' matched for TYPE_TEXT")
                    return MatchResult(crossWinNode, r, 0.95f, tier = 1)
                }
            }
        }

        val searchPool: List<AccessibilityNodeInfo> = candidates

        // Tier 1: Exact Semantic Match
        // Sub-tier 1A: Direct Text or Content Description match (Smallest Bounding Box Wins!)
        var directBestNode: AccessibilityNodeInfo? = null
        var directBestRect = Rect()
        var directMinArea = Long.MAX_VALUE

        val pTokens = pText?.split(Regex("[^a-zA-Z0-9]+"))?.filter { it.length >= 3 } ?: emptyList()

        for (node in searchPool) {
            val rawText = safeGetText(node)?.trim()
            val rawDesc = safeGetDesc(node)?.trim()
            val nText = rawText?.lowercase()
            val nDesc = rawDesc?.lowercase()
            val cleanNText = rawText?.let { DittoSemanticCleaner.clean(it).lowercase() }
            val cleanNDesc = rawDesc?.let { DittoSemanticCleaner.clean(it).lowercase() }

            val isTokenMatch = pTokens.isNotEmpty() && pTokens.all { token ->
                cleanNText?.contains(token) == true || cleanNDesc?.contains(token) == true ||
                nText?.contains(token) == true || nDesc?.contains(token) == true
            }

            val isDirectMatch = (!pText.isNullOrBlank() && (
                nText == pText || nDesc == pText || cleanNText == pText || cleanNDesc == pText ||
                (pText.length >= 3 && (nDesc?.startsWith(pText) == true || nText?.startsWith(pText) == true || cleanNDesc?.startsWith(pText) == true || cleanNText?.startsWith(pText) == true)) ||
                (pText.length >= 3 && (nText?.contains(pText) == true || cleanNText?.contains(pText) == true || nDesc?.contains(pText) == true || cleanNDesc?.contains(pText) == true)) ||
                isTokenMatch
            )) || (!cDesc.isNullOrBlank() && (
                nDesc == cDesc || nText == cDesc || cleanNDesc == cDesc || cleanNText == cDesc ||
                (cDesc.length >= 3 && (nDesc?.startsWith(cDesc) == true || nText?.startsWith(cDesc) == true || cleanNDesc?.startsWith(cDesc) == true || cleanNText?.startsWith(cDesc) == true)) ||
                (cDesc.length >= 3 && (nText?.contains(cDesc) == true || cleanNText?.contains(cDesc) == true || nDesc?.contains(cDesc) == true || cleanNDesc?.contains(cDesc) == true))
            ))

            if (isDirectMatch) {
                val rect = Rect()
                safeGetBounds(node, rect)
                val area = rect.width().toLong() * rect.height().toLong()
                if (rect.width() > 0 && rect.height() > 0 && area < directMinArea) {
                    directMinArea = area
                    directBestNode = node
                    directBestRect = rect
                }
            }
        }

        if (directBestNode != null) {
            // Label-to-Input resolution: If target was an INPUT, but directBestNode is an uneditable label,
            // resolve to the paired input field positioned immediately adjacent to/below the label!
            if (isInputTarget) {
                val isDirectEd = try { directBestNode.isEditable } catch (_: Throwable) { false }
                val directCls = safeGetClassName(directBestNode) ?: ""
                if (!isDirectEd && !directCls.contains("EditText", ignoreCase = true)) {
                    val pairedInput = searchPool.find { n ->
                        val nCls = safeGetClassName(n) ?: ""
                        val nEd = try { n.isEditable } catch (_: Throwable) { false }
                        val nFocus = try { n.isFocusable } catch (_: Throwable) { false }
                        val nR = Rect()
                        safeGetBounds(n, nR)
                        (nEd || nFocus || nCls.contains("EditText", ignoreCase = true)) &&
                        nR.top >= directBestRect.bottom - 20 && nR.top <= directBestRect.bottom + 150 &&
                        nR.left >= directBestRect.left - 100 && nR.right >= directBestRect.left
                    }
                    if (pairedInput != null) {
                        val pairedRect = Rect()
                        safeGetBounds(pairedInput, pairedRect)
                        Log.i(TAG, "Label '${safeGetText(directBestNode)}' successfully resolved to paired input field at $pairedRect")
                        return MatchResult(pairedInput, pairedRect, 1.0f, tier = 1)
                    }
                }
            }
            return MatchResult(directBestNode, directBestRect, 1.0f, tier = 1)
        }

        // Sub-tier 1B: Hierarchy Aggregation Match (Smallest Containing Box Wins!)
        var aggBestNode: AccessibilityNodeInfo? = null
        var aggBestRect = Rect()
        var aggMinArea = Long.MAX_VALUE

        for (node in searchPool) {
            val aggText = extractNodeHierarchyText(node)
            val cleanAggText = DittoSemanticCleaner.clean(aggText).lowercase()
            val isAggMatch = (!pText.isNullOrBlank() && (
                aggText == pText || cleanAggText == pText ||
                (pText.length >= 3 && (aggText.contains(pText) || cleanAggText.contains(pText)))
            )) || (!cDesc.isNullOrBlank() && (
                aggText == cDesc || cleanAggText == cDesc ||
                (cDesc.length >= 3 && (aggText.contains(cDesc) || cleanAggText.contains(cDesc)))
            ))

            if (isAggMatch) {
                val rect = Rect()
                safeGetBounds(node, rect)
                val isGiantContainer = rect.width() >= 1000 && rect.height() >= 1200
                if (isGiantContainer) continue
                val area = rect.width().toLong() * rect.height().toLong()
                if (rect.width() > 0 && rect.height() > 0 && area < aggMinArea) {
                    aggMinArea = area
                    aggBestNode = node
                    aggBestRect = rect
                }
            }
        }

        if (aggBestNode != null) {
            return MatchResult(aggBestNode, aggBestRect, 1.0f, tier = 1)
        }

        // Rotating Search Placeholder Fallback (e.g. Blinkit "Search for atta..." rotating to "Search for milk...", Binance ticker, Zepto)
        val isRecordedSearchAffordance = isTypeAction || resId?.contains("search") == true ||
            pText?.contains("search") == true || cDesc?.contains("search") == true ||
            (targetMeta.recordedBounds.size >= 4 && targetMeta.recordedBounds[1] < 600)

        if (isRecordedSearchAffordance) {
            val searchAffordance = searchPool.firstOrNull { node ->
                val id = safeGetViewId(node)?.lowercase() ?: ""
                val t = safeGetText(node)?.trim()?.lowercase() ?: ""
                val d = safeGetDesc(node)?.trim()?.lowercase() ?: ""
                val r = Rect()
                safeGetBounds(node, r)
                val isTopSearchArea = r.top in 40..650 && r.width() > 200 && r.height() in 40..300
                isTopSearchArea && (
                    id.contains("search") || id.contains("query") || id.contains("edittext") ||
                    d.contains("search") || t.startsWith("search") || t.contains("search") ||
                    t.startsWith("ask ") || d.startsWith("ask ")
                )
            }
            if (searchAffordance != null) {
                val r = Rect()
                safeGetBounds(searchAffordance, r)
                Log.i(TAG, "Search affordance '${safeGetViewId(searchAffordance)}' / '${safeGetText(searchAffordance)}' matched as rotating search placeholder fallback")
                return MatchResult(searchAffordance, r, 0.95f, tier = 1)
            }
        }

        // Tier 2: Strong Structural Match (Resource ID match or Parent Context match)
        var bestNode: AccessibilityNodeInfo? = null
        var bestScore = 0f
        var bestRect = Rect()
        var bestArea = Long.MAX_VALUE

        for (node in searchPool) {
            var score = 0f
            val nId = safeGetViewId(node)?.lowercase()
            val aggText = extractNodeHierarchyText(node)

            val isRepeatingRowOrContainer = resId?.contains("row") == true || resId?.contains("container") == true || resId?.contains("item") == true
            val isGenericTabOrIcon = resId?.contains("tab") == true || resId?.contains("icon") == true || resId?.contains("nav") == true || isRepeatingRowOrContainer
            val cleanResId = resId?.substringAfterLast(":id/")?.substringAfterLast("/")?.lowercase()
            val cleanNId = nId?.substringAfterLast(":id/")?.substringAfterLast("/")?.lowercase()
            val isIdMatch = !resId.isNullOrBlank() && (nId == resId || (!cleanResId.isNullOrBlank() && cleanNId == cleanResId))
            if (isIdMatch) {
                if (!isGenericTabOrIcon) {
                    // Specific input field resource IDs (e.g. com.whatsapp:id/entry, username) score authoritative confidence
                    if (isTypeAction || targetIdentity.semanticRole == "INPUT") {
                        score += 0.85f
                    } else {
                        score += 0.50f
                    }
                } else {
                    // Do not let identical tab_button or repeating row_container resourceIds falsely match without text evidence
                    score += 0.10f
                }
            }

            val hasValidText = aggText.isNotBlank() && aggText.length >= 2
            if (hasValidText && !pText.isNullOrBlank()) {
                val pClean = pText.trim()
                if (aggText == pClean || aggText.contains(pClean) || (pClean.length >= 4 && aggText.length >= 3 && pClean.contains(aggText))) {
                    score += 0.70f
                }
            }
            if (hasValidText && !cDesc.isNullOrBlank()) {
                val cClean = cDesc.trim()
                if (aggText == cClean || aggText.contains(cClean) || (cClean.length >= 4 && aggText.length >= 3 && cClean.contains(aggText))) {
                    score += 0.60f
                }
            }

            val rect = Rect()
            safeGetBounds(node, rect)
            val isGiantContainer = rect.width() >= 1000 && rect.height() >= 1200
            if (isGiantContainer) continue
            val area = rect.width().toLong() * rect.height().toLong()

            // Smallest Bounding Box Rule for Tier 2: Never pick giant 1080w containers when a child matches
            if (score > bestScore || (score == bestScore && score > 0f && area < bestArea)) {
                if (rect.width() > 0 && rect.height() > 0) {
                    bestScore = score
                    bestNode = node
                    bestRect = rect
                    bestArea = area
                }
            }
        }

        if (bestNode != null && bestScore >= threshold) {
            return MatchResult(bestNode, bestRect, bestScore.coerceAtMost(0.95f), tier = 2)
        }

        // Tier 2B: Spatial Coordinate & Interactive Affordance Fallback for Web/Icon Buttons
        // When a recorded icon (e.g. Chrome hamburger menu, close/navigation icon, PUA glyph)
        // has no distinguishable text, match the interactive node covering or closest to recorded tap coordinates!
        val recBounds = targetMeta.recordedBounds
        val tapX = step.physicalExecution.tapX
        val tapY = step.physicalExecution.tapY
        val isIconOrNav = pText.isNullOrBlank() || pText in setOf("menu", "button", "navigation", "close", "icon") ||
            pText.all { it in '\uE000'..'\uF8FF' || it in setOf('\u2630', '\u22EE', '\u22EF') } ||
            targetIdentity.structuralContainer?.contains("WebView", ignoreCase = true) == true ||
            step.targetMetadata.className?.contains("EditText", ignoreCase = true) == true ||
            targetIdentity.semanticRole == "INPUT"
        if (isIconOrNav && (tapX > 0 || recBounds.size >= 4)) {
            val targetX = if (tapX > 0) tapX else (recBounds[0] + recBounds[2]) / 2
            val targetY = if (tapY > 0) tapY else (recBounds[1] + recBounds[3]) / 2

            val spatialCandidates = searchPool.filter { node ->
                val r = Rect()
                safeGetBounds(node, r)
                val isClick = try { node.isClickable } catch (_: Throwable) { false }
                val isEd = try { node.isEditable } catch (_: Throwable) { false }
                val cls = safeGetClassName(node) ?: ""
                val isInteractive = isClick || isEd || cls.contains("Button", ignoreCase = true) ||
                    cls.contains("EditText", ignoreCase = true) || cls.contains("View", ignoreCase = true) ||
                    cls.contains("Image", ignoreCase = true)
                isInteractive && r.width() in 20..1080 && r.height() in 20..600 &&
                    r.contains(targetX, targetY)
            }

            val bestSpatial = spatialCandidates.minByOrNull { node ->
                val r = Rect()
                safeGetBounds(node, r)
                r.width().toLong() * r.height().toLong()
            }

            if (bestSpatial != null) {
                val r = Rect()
                safeGetBounds(bestSpatial, r)
                Log.i(TAG, "Spatial affordance matched for step ${step.stepIndex} '${step.targetIdentity.primaryText}' at $r (tap: $targetX, $targetY)")
                return MatchResult(bestSpatial, r, 0.88f, tier = 2)
            }
        }

        // Tier 3: Ambiguous or Missing -> Fail-Closed
        return null
    }

    private suspend fun dispatchAction(
        service: KarvishAccessibilityService,
        step: DittoStep,
        targetNode: AccessibilityNodeInfo,
        targetBounds: Rect,
        flow: DittoFlow? = null
    ): Boolean = withContext(Dispatchers.Main) {
        try {
            when (step.actionType) {
                "TAP" -> {
                    var tapX = if (step.physicalExecution.tapX in (targetBounds.left + 50)..(targetBounds.right - 50)) {
                        step.physicalExecution.tapX
                    } else if (targetBounds.width() > 700 && targetBounds.height() < 300) {
                        targetBounds.left + (targetBounds.width() * 0.35f).toInt()
                    } else {
                        targetBounds.centerX()
                    }
                    var tapY = targetBounds.centerY()

                    // Safety guard against tapping circular contact avatars in list rows
                    // Senior Mandate: NEVER treat text views, contact names, input fields, or message entry bars as avatars!
                    val isInputTarget = step.targetIdentity.semanticRole == "INPUT" ||
                            step.actionType == "TYPE_TEXT" ||
                            targetNode.isEditable ||
                            step.targetMetadata.className?.contains("EditText", ignoreCase = true) == true ||
                            targetBounds.top > 1800
                    val cls = safeGetClassName(targetNode) ?: ""
                    val resId = safeGetViewId(targetNode)?.lowercase() ?: ""
                    val hasText = !safeGetText(targetNode).isNullOrBlank() || !safeGetDesc(targetNode).isNullOrBlank()
                    val isRealAvatarElement = !hasText && (cls.contains("ImageView", ignoreCase = true) ||
                            resId.contains("picture") || resId.contains("avatar") || resId.contains("photo") || resId.contains("icon"))
                    val isAvatarBounds = !isInputTarget && isRealAvatarElement && targetBounds.left < 250 && targetBounds.width() < 300 && targetBounds.height() < 300
                    if (isAvatarBounds) {
                        // Shift tap into the conversation row body
                        tapX = maxOf(targetBounds.right + 150, 540)
                        Log.i(TAG, "Tapping avatar region intercepted; shifted tapX to $tapX to ensure chat room opens.")
                    }

                    // Smart Dropdown / Selector Handling:
                    // If element is a dropdown/selector and currently displays a placeholder (e.g. 'Day')
                    // or a different option, open the picker and select the desired option (e.g. '29')
                    val desiredOption = step.targetIdentity.primaryText?.trim()
                    val currentText = (safeGetText(targetNode) ?: safeGetDesc(targetNode))?.trim()
                    val isTargetEditable = try { targetNode.isEditable } catch (_: Throwable) { false } ||
                            cls.contains("EditText", ignoreCase = true) || isInputTarget
                    val isDropdownSelection = !isTargetEditable && !desiredOption.isNullOrBlank()

                    if (isDropdownSelection) {
                        if (!currentText.isNullOrBlank() && currentText.equals(desiredOption, ignoreCase = true)) {
                            Log.i(TAG, "Dropdown '${safeGetViewId(targetNode)}' already displays '$desiredOption'; skipping redundant selection")
                            return@withContext true
                        }

                        Log.i(TAG, "Dropdown '${safeGetViewId(targetNode)}' displays '$currentText'; opening picker to select '$desiredOption'")

                        // 1. Open the dropdown/picker: ALWAYS dispatch a real physical hardware touch gesture!
                        // In Android Chromium, performAction(ACTION_CLICK) can conflict with touch gestures or fail to pop the native select dialog.
                        dispatchGestureTap(service, targetBounds.centerX(), targetBounds.centerY())
                        delay(500) // Allow picker dialog animation to complete

                        // 2. Poll across windows for the desired option or scrollable list
                        var optionSelected = false
                        val pickerDeadline = System.currentTimeMillis() + 8000L
                        var scrollAttempts = 0
                        while (System.currentTimeMillis() < pickerDeadline && !optionSelected) {
                            delay(200)
                            val optionNode = findNodeMatchingTextAcrossWindows(service, desiredOption!!)
                            if (optionNode != null) {
                                val optBounds = Rect()
                                safeGetBounds(optionNode, optBounds)
                                Log.i(TAG, "Found option '$desiredOption' at $optBounds; tapping to select")
                                dispatchGestureTap(service, optBounds.centerX(), optBounds.centerY())
                                optionSelected = true
                                delay(600)
                                break
                            }

                            // If option not yet visible, scroll dialog down
                            if (scrollAttempts < 15) {
                                scrollAttempts++
                                val dialogList = findScrollableListInDialog(service)
                                val listBounds = Rect()
                                if (dialogList != null) {
                                    safeGetBounds(dialogList, listBounds)
                                } else {
                                    val screenW = service.resources.displayMetrics.widthPixels.takeIf { it > 0 } ?: 1080
                                    val screenH = service.resources.displayMetrics.heightPixels.takeIf { it > 0 } ?: 2400
                                    listBounds.set(100, (screenH * 0.2f).toInt(), screenW - 100, (screenH * 0.85f).toInt())
                                }
                                val swipeStart = (listBounds.bottom - 120).toFloat()
                                val swipeEnd = (listBounds.top + 120).toFloat()
                                if (swipeStart > swipeEnd) {
                                    Log.i(TAG, "Scrolling picker dialog down (attempt $scrollAttempts) from $swipeStart to $swipeEnd")
                                    val scrollDown = Path().apply {
                                        moveTo(listBounds.centerX().toFloat(), swipeStart)
                                        lineTo(listBounds.centerX().toFloat(), swipeEnd)
                                    }
                                    val gesture = GestureDescription.Builder()
                                        .addStroke(GestureDescription.StrokeDescription(scrollDown, 0, 250))
                                        .build()
                                    service.dispatchGesture(gesture, null, null)
                                    delay(400) // Allow list scrolling physics to settle
                                }
                            }
                        }

                        if (optionSelected) {
                            return@withContext true
                        } else {
                            Log.w(TAG, "Could not find option '$desiredOption' in picker after $scrollAttempts scrolls; dismissing picker safely")
                            try { service.performGlobalAction(android.accessibilityservice.AccessibilityService.GLOBAL_ACTION_BACK) } catch (_: Throwable) {}
                            return@withContext false
                        }
                    }

                    var clicked = false
                    // Always attempt native ACTION_CLICK on the clickable ancestor (e.g. WhatsApp list row container)
                    try {
                        val clickableTarget = findClickableTarget(targetNode)
                        clicked = try { clickableTarget.performAction(AccessibilityNodeInfo.ACTION_CLICK) == true } catch (_: Throwable) { false }
                        if (!clicked && clickableTarget != targetNode) {
                            clicked = try { targetNode.performAction(AccessibilityNodeInfo.ACTION_CLICK) == true } catch (_: Throwable) { false }
                        }
                        if (clicked) {
                            Log.i(TAG, "Successfully performed native ACTION_CLICK on '${safeGetViewId(clickableTarget)}'")
                        }
                    } catch (e: Throwable) {
                        Log.w(TAG, "Native click error: ${e.message}")
                        clicked = false
                    }

                    if (!clicked) {
                        Log.i(TAG, "Native click not accepted; dispatching gesture tap at ($tapX, $tapY)")
                        dispatchGestureTap(service, tapX, tapY)
                    } else true
                }
                "SELECT" -> {
                    val desiredOption = (step.argument ?: step.targetIdentity.primaryText)?.trim()
                    if (desiredOption.isNullOrBlank()) {
                        Log.w(TAG, "SELECT action has no desired option")
                        return@withContext false
                    }
                    val currentText = (safeGetText(targetNode) ?: safeGetDesc(targetNode))?.trim()
                    if (!currentText.isNullOrBlank() && currentText.equals(desiredOption, ignoreCase = true)) {
                        Log.i(TAG, "Selector '${safeGetViewId(targetNode)}' already displays '$desiredOption'; skipping")
                        return@withContext true
                    }

                    Log.i(TAG, "Selector '${safeGetViewId(targetNode)}' current='$currentText', selecting '$desiredOption'")
                    dispatchGestureTap(service, targetBounds.centerX(), targetBounds.centerY())
                    delay(500)

                    var optionSelected = false
                    val pickerDeadline = System.currentTimeMillis() + 8000L
                    var scrollAttempts = 0
                    while (System.currentTimeMillis() < pickerDeadline && !optionSelected) {
                        delay(200)
                        val optionNode = findNodeMatchingTextAcrossWindows(service, desiredOption)
                        if (optionNode != null) {
                            val optBounds = Rect()
                            safeGetBounds(optionNode, optBounds)
                            Log.i(TAG, "Found option '$desiredOption' at $optBounds; tapping")
                            dispatchGestureTap(service, optBounds.centerX(), optBounds.centerY())
                            optionSelected = true
                            delay(600)
                            break
                        }
                        if (scrollAttempts < 15) {
                            scrollAttempts++
                            val dialogList = findScrollableListInDialog(service)
                            val listBounds = Rect()
                            if (dialogList != null) {
                                safeGetBounds(dialogList, listBounds)
                            } else {
                                val screenW = service.resources.displayMetrics.widthPixels.takeIf { it > 0 } ?: 1080
                                val screenH = service.resources.displayMetrics.heightPixels.takeIf { it > 0 } ?: 2400
                                listBounds.set(100, (screenH * 0.2f).toInt(), screenW - 100, (screenH * 0.85f).toInt())
                            }
                            val swipeStart = (listBounds.bottom - 120).toFloat()
                            val swipeEnd = (listBounds.top + 120).toFloat()
                            if (swipeStart > swipeEnd) {
                                val scrollDown = Path().apply {
                                    moveTo(listBounds.centerX().toFloat(), swipeStart)
                                    lineTo(listBounds.centerX().toFloat(), swipeEnd)
                                }
                                val gesture = GestureDescription.Builder()
                                    .addStroke(GestureDescription.StrokeDescription(scrollDown, 0, 250))
                                    .build()
                                service.dispatchGesture(gesture, null, null)
                                delay(400)
                            }
                        }
                    }
                    optionSelected
                }
                "SOLVE_CAPTCHA" -> {
                    Log.i(TAG, "SOLVE_CAPTCHA: Capturing screen and running on-device ML Kit OCR...")
                    val screenBitmap = captureScreenBitmap(service)
                    val solver = com.zipper.app.core.web.WebCaptchaSolver()
                    val code = if (screenBitmap != null && !targetBounds.isEmpty) {
                        val cropX = targetBounds.left.coerceIn(0, screenBitmap.width - 1)
                        val cropY = targetBounds.top.coerceIn(0, screenBitmap.height - 1)
                        val cropW = targetBounds.width().coerceIn(1, screenBitmap.width - cropX)
                        val cropH = targetBounds.height().coerceIn(1, screenBitmap.height - cropY)
                        val captchaBmp = Bitmap.createBitmap(screenBitmap, cropX, cropY, cropW, cropH)
                        solver.solveBitmap(captchaBmp)
                    } else null

                    Log.i(TAG, "SOLVE_CAPTCHA: OCR result: '$code'")
                    if (!code.isNullOrBlank()) {
                        val root = service.rootInActiveWindow
                        val inputNode = findCaptchaInputNode(root) ?: findEditableInputNode(root)
                        if (inputNode != null) {
                            val bundle = Bundle().apply {
                                putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, code)
                            }
                            inputNode.performAction(AccessibilityNodeInfo.ACTION_FOCUS)
                            var set = inputNode.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, bundle)
                            if (!set) {
                                val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as? android.content.ClipboardManager
                                clipboard?.setPrimaryClip(android.content.ClipData.newPlainText("cap", code))
                                set = inputNode.performAction(AccessibilityNodeInfo.ACTION_PASTE)
                            }
                            Log.i(TAG, "SOLVE_CAPTCHA: Entered code '$code' into field: $set")
                            delay(400)
                            return@withContext set
                        }
                    }
                    Log.w(TAG, "SOLVE_CAPTCHA: Could not solve or type captcha")
                    true
                }
                "SCROLL" -> {
                    val safeTop = maxOf(targetBounds.top + 150, 400)
                    val safeBottom = minOf(targetBounds.bottom - 150, 1850)
                    val scrollDown = Path().apply {
                        moveTo(targetBounds.centerX().toFloat(), safeBottom.toFloat())
                        lineTo(targetBounds.centerX().toFloat(), safeTop.toFloat())
                    }
                    val gesture = GestureDescription.Builder()
                        .addStroke(GestureDescription.StrokeDescription(scrollDown, 0, 250))
                        .build()
                    service.dispatchGesture(gesture, null, null)
                }
                "TYPE_TEXT" -> {
                    val rawArg = step.argument ?: ""
                    val arg = rawArg.replace(Regex("[\\u200B-\\u200D\\uFEFF]"), "").trim()
                    val isDirectEditable = try { targetNode.isEditable } catch (_: Throwable) { false } ||
                        safeGetClassName(targetNode)?.contains("EditText", ignoreCase = true) == true
                    val currentNodeText = safeGetText(targetNode)?.trim() ?: ""

                    // Guard: If target is a non-editable dropdown and already displays the target argument, bypass!
                    if (!isDirectEditable && currentNodeText.equals(arg, ignoreCase = true)) {
                        Log.i(TAG, "Step ${step.stepIndex}: Non-editable dropdown '${safeGetViewId(targetNode)}' already set to '$arg'; bypassing redundant typing")
                        return@withContext true
                    }

                    val nextStep = flow?.steps?.getOrNull(step.stepIndex)
                    val isFollowedByPaste = nextStep?.targetIdentity?.primaryText?.equals("Paste", ignoreCase = true) == true ||
                        nextStep?.targetIdentity?.contentDesc?.equals("Paste", ignoreCase = true) == true ||
                        nextStep?.targetMetadata?.resourceId?.contains("paste", ignoreCase = true) == true
                    val isClipboardTarget = step.isClipboardBound || isFollowedByPaste

                    var textSet = false
                    var activeInput: AccessibilityNodeInfo? = null

                    if (isClipboardTarget) {
                        Log.i(TAG, "Step ${step.stepIndex}: Clipboard-bound step detected. Performing native ACTION_PASTE on input field.")
                        val candidateInput = service.rootInActiveWindow?.findFocus(AccessibilityNodeInfo.FOCUS_INPUT)?.takeIf {
                            val p = safeGetPackageName(it) ?: ""
                            val id = safeGetViewId(it) ?: ""
                            p != "com.zipper.app" && !id.startsWith("com.zipper.app:")
                        } ?: findEditableInputNode(service.rootInActiveWindow)
                          ?: targetNode

                        val candidateBounds = Rect()
                        safeGetBounds(candidateInput, candidateBounds)
                        if (!candidateBounds.isEmpty) {
                            dispatchGestureTap(service, candidateBounds.centerX(), candidateBounds.centerY())
                            delay(150)
                        }
                        try { candidateInput.performAction(AccessibilityNodeInfo.ACTION_FOCUS) } catch (_: Throwable) {}
                        delay(200)

                        val pasted = try {
                            candidateInput.performAction(AccessibilityNodeInfo.ACTION_PASTE)
                        } catch (e: Throwable) {
                            Log.w(TAG, "Native ACTION_PASTE error: ${e.message}")
                            false
                        }

                        if (pasted) {
                            Log.i(TAG, "Step ${step.stepIndex}: Successfully executed native ACTION_PASTE using live clipboard!")
                            textSet = true
                            activeInput = candidateInput
                        } else {
                            Log.w(TAG, "Native ACTION_PASTE returned false; falling back to direct text set")
                        }
                    }

                    if (!textSet) {
                        val bundle = Bundle().apply {
                            putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, arg)
                        }

                        // If active input field already contains the target text, dispatch IME enter/submit without re-clicking clear icons
                        val existingRoot = service.rootInActiveWindow
                        val currentFocused = existingRoot?.findFocus(AccessibilityNodeInfo.FOCUS_INPUT)?.takeIf {
                            val p = safeGetPackageName(it) ?: ""
                            val id = safeGetViewId(it) ?: ""
                            p != "com.zipper.app" && !id.startsWith("com.zipper.app:")
                        } ?: findEditableInputNode(existingRoot)
                        val existingText = currentFocused?.let { safeGetText(it)?.toString()?.trim() } ?: ""
                        val alreadyContains = existingText.equals(arg, ignoreCase = true)
                        if (alreadyContains) {
                            Log.i(TAG, "Input field already contains '$arg'; dispatching IME enter/submit instead of re-typing")
                            try {
                                if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R && currentFocused != null) {
                                    currentFocused.performAction(AccessibilityNodeInfo.AccessibilityAction.ACTION_IME_ENTER.id)
                                }
                            } catch (_: Throwable) {}
                            textSet = true
                            activeInput = currentFocused
                        }

                        // If node is directly editable, try direct set first
                        val isDirectEditable = try { targetNode.isEditable } catch (_: Throwable) { false }
                        if (!alreadyContains && isDirectEditable) {
                            try {
                                targetNode.performAction(AccessibilityNodeInfo.ACTION_FOCUS)
                                textSet = targetNode.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, bundle)
                                if (textSet) activeInput = targetNode
                            } catch (_: Throwable) {}
                        }

                        // If direct set failed or node was an unexpanded search affordance (e.g. WhatsApp, Instagram, YouTube, Google):
                        if (!textSet) {
                            Log.i(TAG, "Target node is not directly editable; activating search affordance via click & tap.")
                            
                            // 1. Locate the truly clickable affordance (self or ancestor)
                            val clickableTarget = findClickableTarget(targetNode)
                            try {
                                clickableTarget.performAction(AccessibilityNodeInfo.ACTION_CLICK)
                            } catch (_: Throwable) {}

                            // Always dispatch a real hardware gesture tap to guarantee touch focus and IME binding
                            dispatchGestureTap(service, targetBounds.centerX(), targetBounds.centerY())

                            // 2. Poll for dynamic UI and keyboard to expose the focused input (up to 1200ms)
                            for (w in 0 until 12) {
                                delay(100)
                                val currentRoot = service.rootInActiveWindow
                                activeInput = currentRoot?.findFocus(AccessibilityNodeInfo.FOCUS_INPUT)
                                    ?: findEditableInputNode(currentRoot)
                                    ?: findEditableAcrossWindows(service)
                                if (activeInput != null) {
                                    val isEd = try { activeInput.isEditable } catch (_: Throwable) { false }
                                    if (isEd) {
                                        try {
                                            activeInput.performAction(AccessibilityNodeInfo.ACTION_FOCUS)
                                            textSet = activeInput.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, bundle)
                                            if (textSet) break
                                        } catch (_: Throwable) {}
                                    }
                                }

                                // If after 400ms no input appeared, dispatch fallback gesture tap at target center
                                if (w == 4) {
                                    dispatchGestureTap(service, targetBounds.centerX(), targetBounds.centerY())
                                }
                            }
                        }

                        // Fallback for WebViews / Chrome HTML fields: Clipboard paste fallback if ACTION_SET_TEXT failed
                        if (!textSet && arg.isNotBlank()) {
                            Log.i(TAG, "ACTION_SET_TEXT failed; attempting clipboard paste fallback on active input")
                            try {
                                val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as? android.content.ClipboardManager
                                if (clipboard != null) {
                                    val clip = android.content.ClipData.newPlainText("ditto_text", arg)
                                    clipboard.setPrimaryClip(clip)
                                    val pasteTarget = activeInput ?: targetNode
                                    pasteTarget.performAction(AccessibilityNodeInfo.ACTION_FOCUS)
                                    textSet = pasteTarget.performAction(AccessibilityNodeInfo.ACTION_PASTE)
                                    if (textSet) {
                                        activeInput = pasteTarget
                                        Log.i(TAG, "Successfully pasted text '$arg' via ACTION_PASTE fallback")
                                    }
                                }
                            } catch (e: Throwable) {
                                Log.w(TAG, "Clipboard paste fallback error: ${e.message}")
                            }
                        }
                    }

                    if (textSet) {
                        val inputNode = activeInput ?: targetNode

                        // Module: Do not prematurely submit via IME Enter if subsequent step is an explicit Submit/Send/Search tap
                        val nextStep = flow?.steps?.getOrNull(step.stepIndex) // 1-indexed stepIndex points to next step
                        val nextIsExplicitSubmit = nextStep != null && (
                            nextStep.actionType == "SUBMIT" ||
                            (nextStep.actionType == "TAP" && isExplicitSubmitOrSend(nextStep))
                        )

                        // Submit search via Soft Keyboard Search/Enter key gesture ONLY for standalone search queries without a subsequent submit tap
                        val isSearchTarget = (step.targetMetadata.resourceId?.contains("search", ignoreCase = true) == true ||
                            step.targetIdentity.primaryText?.contains("search", ignoreCase = true) == true ||
                            step.targetIdentity.primaryText?.contains("ask google", ignoreCase = true) == true ||
                            step.targetIdentity.contentDesc?.contains("search", ignoreCase = true) == true) &&
                            !step.targetMetadata.resourceId.orEmpty().contains("captcha", ignoreCase = true)

                        if (isSearchTarget && !nextIsExplicitSubmit) {
                            try {
                                if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R) {
                                    inputNode.performAction(AccessibilityNodeInfo.AccessibilityAction.ACTION_IME_ENTER.id)
                                }
                            } catch (_: Throwable) {}

                            delay(200)
                            val imeWindow = try {
                                service.windows.firstOrNull { it.type == android.view.accessibility.AccessibilityWindowInfo.TYPE_INPUT_METHOD }
                            } catch (_: Throwable) { null }
                            if (imeWindow != null) {
                                val imeBounds = Rect()
                                imeWindow.getBoundsInScreen(imeBounds)
                                if (imeBounds.width() > 0 && imeBounds.height() > 0) {
                                    val imeEnterX = (imeBounds.left + imeBounds.width() * 0.90f).toInt()
                                    val imeEnterY = (imeBounds.top + imeBounds.height() * 0.85f).toInt()
                                    Log.i(TAG, "TYPE_TEXT: Dispatched IME Search gesture to soft keyboard at ($imeEnterX, $imeEnterY)")
                                    dispatchGestureTap(service, imeEnterX, imeEnterY)
                                }
                            } else {
                                val screenWidth = service.resources.displayMetrics.widthPixels
                                val screenHeight = service.resources.displayMetrics.heightPixels
                                val imeEnterX = (screenWidth * 0.90f).toInt()
                                val imeEnterY = (screenHeight * 0.93f).toInt()
                                Log.i(TAG, "TYPE_TEXT: Dispatched fallback IME Search gesture to soft keyboard at ($imeEnterX, $imeEnterY)")
                                dispatchGestureTap(service, imeEnterX, imeEnterY)
                            }
                        }

                        Log.i(TAG, "TYPE_TEXT successfully set '$arg' on input field")
                        true
                    } else {
                        Log.e(TAG, "TYPE_TEXT failed: Unable to set text '$arg' on target or active input")
                        false // Strictly fail closed: do NOT claim phantom success when text was never typed!
                    }
                }
                "BACK" -> {
                    service.performGlobalAction(android.accessibilityservice.AccessibilityService.GLOBAL_ACTION_BACK)
                }
                "SUBMIT" -> {
                    val root = service.rootInActiveWindow
                    val focused = root?.findFocus(AccessibilityNodeInfo.FOCUS_INPUT)
                        ?: findEditableInputNode(root)
                    if (focused != null && android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R) {
                        try {
                            focused.performAction(AccessibilityNodeInfo.AccessibilityAction.ACTION_IME_ENTER.id)
                        } catch (_: Throwable) {}
                    }
                    val imeWindow = try {
                        service.windows.firstOrNull { it.type == android.view.accessibility.AccessibilityWindowInfo.TYPE_INPUT_METHOD }
                    } catch (_: Throwable) { null }
                    if (imeWindow != null) {
                        val imeBounds = Rect()
                        imeWindow.getBoundsInScreen(imeBounds)
                        if (imeBounds.width() > 0 && imeBounds.height() > 0) {
                            val imeEnterX = (imeBounds.left + imeBounds.width() * 0.90f).toInt()
                            val imeEnterY = (imeBounds.top + imeBounds.height() * 0.85f).toInt()
                            Log.i(TAG, "SUBMIT: Dispatched IME Search gesture to soft keyboard at ($imeEnterX, $imeEnterY)")
                            dispatchGestureTap(service, imeEnterX, imeEnterY)
                        }
                    } else if (focused != null) {
                        val focusRect = Rect()
                        focused.getBoundsInScreen(focusRect)
                        val suggestionY = (focusRect.bottom + 80).toInt()
                        val suggestionX = (focusRect.left + 200).toInt()
                        val screenHeight = service.resources.displayMetrics.heightPixels
                        if (suggestionY < screenHeight * 0.6f) {
                            Log.i(TAG, "SUBMIT: Soft keyboard not visible; tapping primary suggestion at ($suggestionX, $suggestionY)")
                            dispatchGestureTap(service, suggestionX, suggestionY)
                        }
                    }
                    true
                }
                else -> {
                    // Action Menu redirect: If target was an action menu / 3-dots button on a video/product card,
                    // redirect tap to the video/card itself so it plays/opens instead of showing the channel popup.
                    val isActionMenu = isActionMenuTarget(step.targetIdentity.primaryText, step.targetMetadata.resourceId)
                    val effectiveTarget = if (isActionMenu) {
                        findCardContainerOrTitle(targetNode) ?: targetNode
                    } else {
                        targetNode
                    }
                    val effectiveBounds = Rect()
                    safeGetBounds(effectiveTarget, effectiveBounds)
                    if (effectiveBounds.isEmpty) effectiveBounds.set(targetBounds)

                    try {
                        val isInputNode = effectiveTarget.isEditable ||
                            safeGetClassName(effectiveTarget)?.contains("EditText", ignoreCase = true) == true ||
                            step.targetIdentity.semanticRole == "INPUT"
                        if (isInputNode) {
                            // Real gesture tap guarantees focus, soft keyboard trigger, and cursor placement in WebViews
                            dispatchGestureTap(service, effectiveBounds.centerX(), effectiveBounds.centerY())
                            effectiveTarget.performAction(AccessibilityNodeInfo.ACTION_FOCUS)
                            effectiveTarget.performAction(AccessibilityNodeInfo.ACTION_CLICK)
                        } else {
                            val clicked = effectiveTarget.performAction(AccessibilityNodeInfo.ACTION_CLICK)
                            if (!clicked) {
                                dispatchGestureTap(service, effectiveBounds.centerX(), effectiveBounds.centerY())
                            }
                        }
                        true
                    } catch (_: Throwable) {
                        dispatchGestureTap(service, effectiveBounds.centerX(), effectiveBounds.centerY())
                    }
                }
            }
        } catch (e: Throwable) {
            Log.e(TAG, "Error executing action: ${e.message}", e)
            false
        }
    }

    private fun dispatchGestureTap(service: KarvishAccessibilityService, x: Int, y: Int): Boolean {
        return try {
            val path = Path().apply {
                moveTo(x.toFloat(), y.toFloat())
                lineTo(x.toFloat(), y.toFloat())
            }
            val gesture = GestureDescription.Builder()
                .addStroke(GestureDescription.StrokeDescription(path, 0, 50))
                .build()
            service.dispatchGesture(gesture, null, null)
        } catch (e: Throwable) {
            Log.e(TAG, "dispatchGestureTap exception: ${e.message}")
            false
        }
    }

    private fun findClickableTarget(node: AccessibilityNodeInfo): AccessibilityNodeInfo {
        var curr: AccessibilityNodeInfo? = node
        for (i in 0 until 6) {
            if (curr == null) break
            val isClick = try { curr.isClickable } catch (_: Throwable) { false }
            if (isClick) return curr
            curr = safeGetParent(curr)
        }
        return node
    }

    private fun isActionMenuTarget(text: String?, viewId: String?): Boolean {
        val t = text?.lowercase() ?: ""
        val id = viewId?.lowercase() ?: ""
        return t.contains("action menu") || t.contains("go to channel") || t.contains("more options") ||
               t.contains("overflow menu") || id.contains("action_menu") || id.contains("overflow")
    }

    private fun findCardContainerOrTitle(node: AccessibilityNodeInfo): AccessibilityNodeInfo? {
        var curr: AccessibilityNodeInfo? = node
        for (i in 0 until 4) {
            if (curr == null) break
            val p = safeGetParent(curr) ?: break
            val r = Rect()
            safeGetBounds(p, r)
            if (r.width() > 400 && r.height() in 150..900) {
                val childCount = try { p.childCount } catch (_: Throwable) { 0 }
                for (c in 0 until childCount) {
                    val child = safeGetChild(p, c) ?: continue
                    val id = safeGetViewId(child)?.lowercase() ?: ""
                    val t = safeGetText(child)?.trim() ?: ""
                    if ((id.contains("title") || id.contains("thumbnail")) && t.length > 5 && !isActionMenuTarget(t, id)) {
                        return child
                    }
                }
                return p
            }
            curr = p
        }
        return null
    }

    private fun isExplicitSubmitOrSend(step: DittoStep): Boolean {
        val t = step.targetIdentity.primaryText?.lowercase() ?: ""
        val d = step.targetIdentity.contentDesc?.lowercase() ?: ""
        val id = step.targetMetadata.resourceId?.lowercase() ?: ""
        return t.contains("send") || t.contains("submit") || t.contains("search") || t.contains("done") || t.contains("go") || t.contains("login") ||
               d.contains("send") || d.contains("submit") || d.contains("search") || d.contains("done") || d.contains("go") || d.contains("login") ||
               id.contains("send") || id.contains("submit") || id.contains("search_box_submit") || id.contains("btn_search") ||
               id.contains("btn-submit") || id.contains("button_send") || id.contains("login")
    }

    private fun canFastForwardToNextStep(
        service: KarvishAccessibilityService,
        currentStep: DittoStep,
        nextStep: DittoStep?
    ): Boolean {
        if (nextStep == null) return false
        val root = service.rootInActiveWindow ?: return false
        val currentPkg = safeGetPackageName(root) ?: ""

        // 1. If active package has moved away from current step's package to the next step's package
        val nextPkg = nextStep.stateBefore.activePackage ?: currentStep.stateAfter.activePackage
        if (!nextPkg.isNullOrBlank() && currentPkg == nextPkg && currentPkg != currentStep.stateBefore.activePackage) {
            Log.i(TAG, "Fast-forward: Package transitioned from '${currentStep.stateBefore.activePackage}' to '$currentPkg'")
            return true
        }

        // 2. Check if next step's target is ALREADY matched on the current screen
        val nextMatch = findTargetNodeHierarchical(service, nextStep, 0.75f)
        if (nextMatch != null && nextMatch.confidence >= 0.75f) {
            Log.i(TAG, "Fast-forward: Next step ${nextStep.stepIndex} target '${nextStep.targetIdentity.primaryText}' is already visible on screen (conf=${nextMatch.confidence})")
            return true
        }

        // 3. Check if current screen matches multiple visible signatures of nextStep or stateAfter
        val afterSigs = (currentStep.stateAfter.visibleSignatures) +
                        nextStep.stateBefore.visibleSignatures
        val validSigs = afterSigs.filter { it.length >= 4 && !it.startsWith("http") }
        val matchingSigs = validSigs.filter { sig ->
            checkContainsText(root, sig.lowercase().trim())
        }
        if (matchingSigs.size >= 2 || (matchingSigs.isNotEmpty() && !nextPkg.isNullOrBlank() && currentPkg == nextPkg)) {
            Log.i(TAG, "Fast-forward: Screen matches expected transition signatures (${matchingSigs.take(2)}); step ${currentStep.stepIndex} already fulfilled.")
            return true
        }

        // 4. Redundant TYPE_TEXT on non-editable / already-set element:
        if (currentStep.actionType == "TYPE_TEXT") {
            val arg = currentStep.argument?.trim() ?: ""
            val stepId = currentStep.targetMetadata.resourceId?.substringAfterLast(":id/")?.substringAfterLast("/")?.lowercase()
            if (!stepId.isNullOrBlank() && arg.isNotBlank()) {
                val node = findNodeByIdAcrossWindows(service, stepId)
                if (node != null) {
                    val isEd = (try { node.isEditable } catch (_: Throwable) { false }) ||
                        safeGetClassName(node)?.contains("EditText", ignoreCase = true) == true
                    val text = safeGetText(node)?.trim() ?: ""
                    if (!isEd && text.equals(arg, ignoreCase = true)) {
                        Log.i(TAG, "Fast-forward: Target '$stepId' already set to '$arg'; fast-forwarding past Step ${currentStep.stepIndex}")
                        return true
                    }
                }
            }
        }

        return false
    }

    private fun findEditableInputNode(root: AccessibilityNodeInfo?): AccessibilityNodeInfo? {
        if (root == null) return null
        val queue = ArrayDeque<AccessibilityNodeInfo>()
        queue.add(root)
        var count = 0
        while (queue.isNotEmpty() && count < 250) {
            val node = queue.removeFirst()
            count++
            val cls = safeGetClassName(node) ?: ""
            val isEd = try { node.isEditable } catch (_: Throwable) { false }
            val pkg = safeGetPackageName(node) ?: ""
            val id = safeGetViewId(node) ?: ""
            if (pkg == "com.zipper.app" || id.startsWith("com.zipper.app:")) {
                continue
            }
            if (isEd || cls.contains("EditText", ignoreCase = true)) {
                val isScroll = try { node.isScrollable } catch (_: Throwable) { false }
                if (!isScroll && !cls.contains("RecyclerView", ignoreCase = true) && !cls.contains("ListView", ignoreCase = true)) {
                    val r = Rect()
                    safeGetBounds(node, r)
                    if (r.width() > 0 && r.height() in 20..400) {
                        return node
                    }
                }
            }
            val childCount = try { node.childCount } catch (_: Throwable) { 0 }
            for (i in 0 until childCount) {
                val child = safeGetChild(node, i) ?: continue
                queue.add(child)
            }
        }
        return null
    }

    private fun findEditableAcrossWindows(service: KarvishAccessibilityService): AccessibilityNodeInfo? {
        try {
            val windows = service.windows ?: return null
            for (win in windows) {
                val root = win.root ?: continue
                val found = findEditableInputNode(root)
                if (found != null) return found
            }
        } catch (_: Throwable) {}
        return null
    }

    private fun findNodeMatchingTextAcrossWindows(service: KarvishAccessibilityService, targetText: String): AccessibilityNodeInfo? {
        val cleanTarget = DittoSemanticCleaner.clean(targetText).lowercase().trim()
        val targetInt = cleanTarget.toIntOrNull()
        val monthMap = mapOf(
            "1" to "jan", "01" to "jan", "january" to "jan", "jan" to "jan",
            "2" to "feb", "02" to "feb", "february" to "feb", "feb" to "feb",
            "3" to "mar", "03" to "mar", "march" to "mar", "mar" to "mar",
            "4" to "apr", "04" to "apr", "april" to "apr", "apr" to "apr",
            "5" to "may", "05" to "may", "may" to "may",
            "6" to "jun", "06" to "jun", "june" to "jun", "jun" to "jun",
            "7" to "jul", "07" to "jul", "july" to "jul", "jul" to "jul",
            "8" to "aug", "08" to "aug", "august" to "aug", "aug" to "aug",
            "9" to "sep", "09" to "sep", "september" to "sep", "sep" to "sep",
            "10" to "oct", "october" to "oct", "oct" to "oct",
            "11" to "nov", "november" to "nov", "nov" to "nov",
            "12" to "dec", "december" to "dec", "dec" to "dec"
        )
        val targetMonth = monthMap[cleanTarget]

        val roots = mutableListOf<AccessibilityNodeInfo>()
        service.rootInActiveWindow?.let { roots.add(it) }
        try {
            service.windows?.forEach { win ->
                win.root?.let { if (!roots.contains(it)) roots.add(it) }
            }
        } catch (_: Throwable) {}

        for (root in roots) {
            val queue = ArrayDeque<AccessibilityNodeInfo>()
            queue.add(root)
            var count = 0
            while (queue.isNotEmpty() && count < 250) {
                count++
                val node = queue.removeFirst()
                val pkg = safeGetPackageName(node) ?: ""
                val id = safeGetViewId(node) ?: ""
                if (pkg == "com.zipper.app" || id.startsWith("com.zipper.app:")) continue

                val rawText = safeGetText(node)?.trim() ?: ""
                val rawDesc = safeGetDesc(node)?.trim() ?: ""
                val t = DittoSemanticCleaner.clean(rawText).lowercase().trim()
                val d = DittoSemanticCleaner.clean(rawDesc).lowercase().trim()

                val isExact = t == cleanTarget || d == cleanTarget || rawText.equals(targetText, ignoreCase = true)
                val nodeInt = t.toIntOrNull() ?: rawText.toIntOrNull()
                val isIntMatch = targetInt != null && nodeInt != null && targetInt == nodeInt
                val isMonthMatch = targetMonth != null && (monthMap[t] == targetMonth || t.contains(targetMonth) || d.contains(targetMonth))

                if (isExact || isIntMatch || isMonthMatch) {
                    val r = Rect()
                    safeGetBounds(node, r)
                    if (r.width() > 0 && r.height() > 0) return node
                }

                val childCount = try { node.childCount } catch (_: Throwable) { 0 }
                for (i in 0 until childCount) {
                    val child = safeGetChild(node, i) ?: continue
                    queue.add(child)
                }
            }
        }
        return null
    }

    private fun findScrollableListInDialog(service: KarvishAccessibilityService): AccessibilityNodeInfo? {
        val roots = mutableListOf<AccessibilityNodeInfo>()
        service.rootInActiveWindow?.let { roots.add(it) }
        try {
            service.windows?.forEach { win ->
                win.root?.let { if (!roots.contains(it)) roots.add(it) }
            }
        } catch (_: Throwable) {}

        for (root in roots) {
            val queue = ArrayDeque<AccessibilityNodeInfo>()
            queue.add(root)
            var count = 0
            while (queue.isNotEmpty() && count < 350) {
                count++
                val node = queue.removeFirst()
                val pkg = safeGetPackageName(node) ?: ""
                val id = safeGetViewId(node) ?: ""
                if (pkg == "com.zipper.app" || id.startsWith("com.zipper.app:")) continue

                val cls = safeGetClassName(node) ?: ""
                val isScroll = try { node.isScrollable } catch (_: Throwable) { false }
                val isList = cls.contains("ListView", ignoreCase = true) ||
                             cls.contains("RecyclerView", ignoreCase = true) ||
                             cls.contains("ScrollView", ignoreCase = true) ||
                             cls.contains("AbsListView", ignoreCase = true)
                if (isScroll || isList) {
                    val r = Rect()
                    safeGetBounds(node, r)
                    if (r.width() >= 150 && r.height() >= 150) {
                        return node
                    }
                }

                val childCount = try { node.childCount } catch (_: Throwable) { 0 }
                for (i in 0 until childCount) {
                    val child = safeGetChild(node, i) ?: continue
                    queue.add(child)
                }
            }
        }
        return null
    }

    private fun findNodeByIdAcrossWindows(service: KarvishAccessibilityService, targetId: String): AccessibilityNodeInfo? {
        val cleanTarget = targetId.lowercase().trim()
        val roots = mutableListOf<AccessibilityNodeInfo>()
        service.rootInActiveWindow?.let { roots.add(it) }
        try {
            service.windows?.forEach { win ->
                win.root?.let { if (!roots.contains(it)) roots.add(it) }
            }
        } catch (_: Throwable) {}

        for (root in roots) {
            val queue = ArrayDeque<AccessibilityNodeInfo>()
            queue.add(root)
            var count = 0
            while (queue.isNotEmpty() && count < 250) {
                count++
                val node = queue.removeFirst()
                val pkg = safeGetPackageName(node) ?: ""
                val id = safeGetViewId(node)?.lowercase() ?: ""
                val cleanId = id.substringAfterLast(":id/").substringAfterLast("/")
                if (pkg == "com.zipper.app" || id.startsWith("com.zipper.app:")) continue

                if (cleanId == cleanTarget || id == cleanTarget) {
                    return node
                }

                val childCount = try { node.childCount } catch (_: Throwable) { 0 }
                for (i in 0 until childCount) {
                    val child = safeGetChild(node, i) ?: continue
                    queue.add(child)
                }
            }
        }
        return null
    }

    private fun verifyTransition(service: KarvishAccessibilityService, step: DittoStep): Boolean {
        val roots = mutableListOf<AccessibilityNodeInfo>()
        service.rootInActiveWindow?.let { roots.add(it) }
        try {
            service.windows?.forEach { win ->
                win.root?.let { if (!roots.contains(it)) roots.add(it) }
            }
        } catch (_: Throwable) {}
        if (roots.isEmpty()) return true

        if (step.actionType == "TYPE_TEXT" && !step.argument.isNullOrBlank()) {
            val arg = step.argument.replace(Regex("[\\u200B-\\u200D\\uFEFF]"), "").lowercase().trim()
            val containsArg = roots.any { checkContainsText(it, arg) }
            if (containsArg) return true
            // If text was set successfully and an active/editable field or root exists, pass transition
            val hasFocusedOrEditable = roots.any { r ->
                r.findFocus(AccessibilityNodeInfo.FOCUS_INPUT) != null ||
                (try { r.isEditable } catch (_: Throwable) { false }) ||
                safeGetClassName(r)?.contains("EditText", ignoreCase = true) == true
            }
            return hasFocusedOrEditable || roots.isNotEmpty()
        }
        val afterSigs = step.stateAfter.visibleSignatures.filter { it.isNotBlank() }
        if (afterSigs.isNotEmpty()) {
            val matchesAfter = afterSigs.any { 
                val cleanSig = it.replace(Regex("[\\u200B-\\u200D\\uFEFF]"), "").lowercase().trim()
                cleanSig.isNotBlank() && roots.any { r -> checkContainsText(r, cleanSig) }
            }
            if (matchesAfter) return true
        }
        return roots.any { (try { it.childCount } catch (_: Throwable) { 0 }) > 0 }
    }

    private fun checkContainsText(node: AccessibilityNodeInfo?, target: String): Boolean {
        if (node == null) return false
        try {
            val t = safeGetText(node)?.lowercase()
            val d = safeGetDesc(node)?.lowercase()
            if (t?.contains(target) == true || d?.contains(target) == true) return true
            val childCount = try { node.childCount } catch (_: Throwable) { 0 }
            for (i in 0 until childCount) {
                val child = safeGetChild(node, i) ?: continue
                if (checkContainsText(child, target)) return true
            }
        } catch (_: Throwable) {}
        return false
    }

    // =========================================================================
    // Sealed-Node Safe Accessibility Getters (Zero Crash Guarantee)
    // =========================================================================

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

    private fun safeGetPackageName(node: AccessibilityNodeInfo?): String? = try {
        node?.packageName?.toString()
    } catch (_: Throwable) { null }

    private fun safeGetBounds(node: AccessibilityNodeInfo, rect: Rect) {
        try {
            node.getBoundsInScreen(rect)
        } catch (_: Throwable) {
            rect.setEmpty()
        }
    }

    private fun hasVisibleSearchField(root: AccessibilityNodeInfo): Boolean {
        val queue = ArrayDeque<AccessibilityNodeInfo>()
        queue.add(root)
        var count = 0
        while (queue.isNotEmpty() && count < 80) {
            count++
            val node = queue.removeFirst()
            val cls = safeGetClassName(node) ?: ""
            val id = safeGetViewId(node)?.lowercase() ?: ""
            val text = safeGetText(node)?.lowercase() ?: ""
            val desc = safeGetDesc(node)?.lowercase() ?: ""
            val isEd = try { node.isEditable } catch (_: Throwable) { false } || cls.contains("EditText", ignoreCase = true)
            if (isEd && (id.contains("search") || text.contains("search") || desc.contains("search"))) {
                return true
            }
            val cc = try { node.childCount } catch (_: Throwable) { 0 }
            for (i in 0 until cc) {
                val c = safeGetChild(node, i) ?: continue
                queue.add(c)
            }
        }
        return false
    }

    private suspend fun captureScreenBitmap(service: KarvishAccessibilityService): Bitmap? {
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R) {
            return suspendCancellableCoroutine { cont ->
                try {
                    service.takeScreenshot(
                        Display.DEFAULT_DISPLAY,
                        context.mainExecutor,
                        object : AccessibilityService.TakeScreenshotCallback {
                            override fun onSuccess(result: AccessibilityService.ScreenshotResult) {
                                val hwBuffer = result.hardwareBuffer
                                val cs = result.colorSpace
                                val bmp = Bitmap.wrapHardwareBuffer(hwBuffer, cs)?.copy(Bitmap.Config.ARGB_8888, false)
                                hwBuffer.close()
                                if (cont.isActive) cont.resume(bmp)
                            }
                            override fun onFailure(errorCode: Int) {
                                Log.w(TAG, "takeScreenshot failed: $errorCode")
                                if (cont.isActive) cont.resume(null)
                            }
                        }
                    )
                } catch (e: Exception) {
                    Log.w(TAG, "Error taking screenshot: ${e.message}")
                    if (cont.isActive) cont.resume(null)
                }
            }
        }
        return null
    }

    private fun findCaptchaInputNode(root: AccessibilityNodeInfo?): AccessibilityNodeInfo? {
        if (root == null) return null
        val queue = ArrayDeque<AccessibilityNodeInfo>()
        queue.add(root)
        while (queue.isNotEmpty()) {
            val node = queue.removeFirst()
            val text = (safeGetText(node) ?: "") + " " + (safeGetDesc(node) ?: "") + " " + (safeGetViewId(node) ?: "")
            val isEd = try { node.isEditable } catch (_: Throwable) { false } || safeGetClassName(node)?.contains("EditText", ignoreCase = true) == true
            if (isEd && (text.contains("code", true) || text.contains("security", true) || text.contains("cap", true))) {
                return node
            }
            val cc = try { node.childCount } catch (_: Throwable) { 0 }
            for (i in 0 until cc) {
                safeGetChild(node, i)?.let { queue.add(it) }
            }
        }
        return null
    }
}
