package com.zipper.app.core.agent

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import com.zipper.app.R
import com.zipper.app.accessibility.KarvishAccessibilityService
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

class AgentRuntimeService : Service() {
    companion object {
        const val EXTRA_GOAL = "agent_goal"
        const val EXTRA_RECOVERY = "agent_recovery"
        const val EXTRA_USE_REMOTE_RANKER = "use_remote_ranker"
        const val ACTION_GOAL_FINISHED = "com.zipper.app.ACTION_AGENT_GOAL_FINISHED"
        const val ACTION_STOP_AGENT = "com.zipper.app.ACTION_STOP_AGENT"
        const val EXTRA_SUCCESS = "agent_success"
        const val EXTRA_SUMMARY = "agent_summary"
        private const val CHANNEL_ID = "karvish_agent_runtime"
        private const val NOTIFICATION_ID = 4101
        private const val TAG = "KarvishAgentService"

        @Volatile
        var isRunning: Boolean = false
            private set

        @Volatile
        var activeRuntime: AgentRuntime? = null
            internal set
    }

    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var activeJob: kotlinx.coroutines.Job? = null

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        startForeground(NOTIFICATION_ID, buildNotification("Preparing autonomous runtime"))
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP_AGENT) {
            Log.w(TAG, "[HARD STOP] Emergency abort triggered by user.")
            isRunning = false
            activeRuntime = null
            activeJob?.cancel()
            com.zipper.app.core.graphics.LiveOverlayManager.dismissOverlay()
            updateNotification("⛔ Agent Aborted (Hard Stop)")
            sendBroadcast(Intent(ACTION_GOAL_FINISHED).apply {
                setPackage(packageName)
                putExtra(EXTRA_SUCCESS, false)
                putExtra(EXTRA_SUMMARY, "Emergency Hard Stop triggered by user")
            })
            stopSelf()
            return START_NOT_STICKY
        }

        val goal = intent?.getStringExtra(EXTRA_GOAL)?.trim().orEmpty()
        if (goal.isBlank()) {
            if (!isRunning) stopSelf(startId)
            return START_NOT_STICKY
        }

        // 0. Check for 0ms Instant Local Action (Notes, WhatsApp, Calls, SMS, Alarms, Maps, App Launch)
        val localResult = com.zipper.app.core.local.LocalActionDispatcher.tryDispatch(this, goal)
        if (localResult.handled) {
            Log.i(TAG, "⚡ Goal '$goal' handled instantly via LocalActionDispatcher: ${localResult.feedback}")
            com.zipper.app.core.graphics.LiveOverlayManager.dismissOverlay()
            updateNotification("✓ ${localResult.feedback}")

            if (localResult.disambiguationNeeded && localResult.contactCandidates.isNotEmpty()) {
                // If multiple contacts found, launch WebmindActivity so user can select between candidates!
                val disambigIntent = Intent(this, com.zipper.app.webmind.WebmindActivity::class.java).apply {
                    putExtra("initial_query", goal)
                    addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP)
                }
                startActivity(disambigIntent)
            } else {
                android.os.Handler(android.os.Looper.getMainLooper()).post {
                    android.widget.Toast.makeText(applicationContext, "⚡ ${localResult.feedback}", android.widget.Toast.LENGTH_SHORT).show()
                }
            }

            sendBroadcast(Intent(ACTION_GOAL_FINISHED).apply {
                setPackage(packageName)
                putExtra(EXTRA_SUCCESS, true)
                putExtra(EXTRA_SUMMARY, localResult.feedback)
            })
            stopSelf()
            return START_NOT_STICKY
        }

        // 0b. Check for Shopping / Quick-Commerce (e.g. "order 100g paneer", "buy milk and bread")
        val qLower = goal.lowercase()
        val isShoppingQuery = qLower.startsWith("order ") || qLower.startsWith("buy ") || qLower.contains("grocery")
        val parsedList = com.zipper.app.core.web.ShoppingListParser.parseShoppingList(goal)
        if (isShoppingQuery && parsedList.isNotEmpty()) {
            Log.i(TAG, "🛒 Goal '$goal' routed to Webmind Quick-Commerce Cart Optimizer")
            com.zipper.app.core.graphics.LiveOverlayManager.dismissOverlay()
            val webmindIntent = Intent(this, com.zipper.app.webmind.WebmindActivity::class.java).apply {
                putExtra("initial_query", goal)
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP)
            }
            startActivity(webmindIntent)
            sendBroadcast(Intent(ACTION_GOAL_FINISHED).apply {
                setPackage(packageName)
                putExtra(EXTRA_SUCCESS, true)
                putExtra(EXTRA_SUMMARY, "Opened Webmind Quick-Commerce for '$goal'")
            })
            stopSelf()
            return START_NOT_STICKY
        }

        if (isRunning) {
            val prevGoal = activeRuntime?.activeGoal ?: ""
            val isSameApp = prevGoal.isNotBlank() &&
                goal.contains(prevGoal, ignoreCase = true) ||
                (prevGoal.contains("search", ignoreCase = true) && goal.contains("search", ignoreCase = true))
            if (isSameApp && activeRuntime != null) {
                Log.i(TAG, "[MID-FLIGHT UPDATE] Active agent receives parameter refinement: '$goal'")
                activeRuntime?.updateCurrentGoal(goal)
                updateNotification("Running: $goal")
                return START_STICKY
            } else {
                Log.i(TAG, "New standalone goal '$goal' supersedes previous job ('$prevGoal'). Resetting cleanly.")
                activeJob?.cancel()
                activeRuntime = null
                isRunning = false
            }
        }

        isRunning = true
        activeJob?.cancel()
        activeJob = serviceScope.launch {
            var success = false
            var summary = "Agent stopped before producing a result"
            try {
                val useRemote = intent?.getBooleanExtra(EXTRA_USE_REMOTE_RANKER, false) ?: false
                val result = executeGoal(
                    goal = goal,
                    enableRecovery = intent?.getBooleanExtra(EXTRA_RECOVERY, true) ?: true,
                    useRemoteRanker = useRemote
                )
                success = result.first
                summary = result.second
            } catch (ce: kotlinx.coroutines.CancellationException) {
                Log.w(TAG, "Agent execution cancelled: ${ce.message}")
                summary = "Agent cancelled: ${ce.message ?: "Task aborted"}"
            } catch (t: Throwable) {
                Log.e(TAG, "Agent execution failed with unhandled exception", t)
                summary = "Agent error: ${t.localizedMessage ?: t.javaClass.simpleName}"
            } finally {
                activeRuntime = null
                isRunning = false
                sendBroadcast(Intent(ACTION_GOAL_FINISHED).apply {
                    setPackage(packageName)
                    putExtra(EXTRA_SUCCESS, success)
                    putExtra(EXTRA_SUMMARY, summary)
                })
                stopSelf(startId)
            }
        }
        return START_NOT_STICKY
    }

    private suspend fun executeGoal(
        goal: String,
        enableRecovery: Boolean,
        useRemoteRanker: Boolean = false
    ): Pair<Boolean, String> {
        var a11yService = KarvishAccessibilityService.instance
        for (attempt in 0 until 120) {
            if (a11yService != null) break
            delay(500)
            a11yService = KarvishAccessibilityService.instance
        }
        if (a11yService == null) {
            Log.e(TAG, "Cannot start goal; accessibility service is disconnected")
            updateNotification("Accessibility service disconnected")
            return false to "Accessibility service disconnected"
        }

        updateNotification("Running: $goal")

        val modelEngine: AgentModelEngine = if (useRemoteRanker) {
            val liveEngine = LiveRankerAgentModelEngine()
            val isGpuReady = kotlinx.coroutines.withContext(Dispatchers.IO) {
                liveEngine.checkBridgeHealth()
            }
            if (!isGpuReady) {
                val err = "GPU bridge on port 8765 is not reachable or /health check failed."
                Log.e(TAG, "[ROUTING ERROR] $err")
                updateNotification("Error: GPU bridge disconnected")
                return false to err
            }
            Log.i(TAG, "[MODEL] Host GPU V54 Bridge on localhost:8765")
            liveEngine
        } else {
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
            val modelVersion = if (modelPath.contains("v62")) "V62" else if (modelPath.contains("v61")) "V61" else if (modelPath.contains("v60")) "V60" else if (modelPath.contains("v59")) "V59" else if (modelPath.contains("v58")) "V58" else if (modelPath.contains("v57")) "V57" else if (modelPath.contains("v56")) "V56" else if (modelPath.contains("v55")) "V55" else if (modelPath.contains("v54")) "V54" else "V52"
            Log.i(TAG, "[MODEL] Pure On-Device $modelVersion Universal ONNX Engine ($modelPath offline)")
            OnDeviceRankerEngine.getOrCreate(this, modelPath)
        }

        val runtime = AgentRuntime(
            a11yService = a11yService,
            needleEngine = modelEngine,
            maxTurns = 30,
            confidenceThreshold = 0.001f,
            enableActiveRecovery = enableRecovery
        )
        activeRuntime = runtime

        // Attach Antigravity Live Agent Overlay with 3D master core
        val overlayContext = a11yService ?: this
        com.zipper.app.accessibility.KarvishBubbleOverlay.startAutonomousSession(overlayContext, goal)
        com.zipper.app.accessibility.KarvishBubbleOverlay.setAgentControls(
            onPause = { paused -> runtime.isPaused = paused },
            onCancel = { runtime.isCancelled = true }
        )
        com.zipper.app.core.graphics.LiveOverlayManager.startExecutionOverlay(overlayContext, goal)
        com.zipper.app.accessibility.KarvishControlOverlay.show(
            context = overlayContext,
            goal = goal,
            onPause = { paused -> runtime.isPaused = paused },
            onCancel = { runtime.isCancelled = true }
        )

        val result = runtime.executeGoal(goal) { step ->
            Log.i(TAG, "Turn ${step.turn}: ${step.actionDescription} -> ${step.outcome}: ${step.reflection}")
            updateNotification("Turn ${step.turn}: ${step.actionDescription}")
            com.zipper.app.core.graphics.LiveOverlayManager.onStepUpdated(
                turn = step.turn,
                actionText = step.actionDescription,
                coreState = com.zipper.app.core.graphics.ZipperCoreState.WORKING
            )
            com.zipper.app.accessibility.KarvishBubbleOverlay.updateAgentTurn(
                turn = step.turn,
                maxTurns = 30,
                actionName = step.actionDescription,
                targetLabel = "",
                statusText = step.actionDescription,
                subtext = step.reflection,
                packageName = step.packageName
            )
        }
        Log.i(TAG, "Goal finished: success=${result.success}, reason=${result.terminationReason}, summary=${result.finalSummary}")
        updateNotification(if (result.success) "Completed: $goal" else "Stopped: ${result.finalSummary}")

        if (result.success) {
            val successfulMilestones = result.history
                .filter { it.outcome == StepOutcome.SUCCESS }
                .map { "Turn ${it.turn}: ${it.actionDescription}" }
            com.zipper.app.accessibility.KarvishBubbleOverlay.onGoalCompleted(goal, result.finalSummary)
            com.zipper.app.core.graphics.LiveOverlayManager.onGoalCompleted(result.finalSummary)
            com.zipper.app.accessibility.KarvishControlOverlay.showCompleted(
                milestones = successfulMilestones
            ) {
                com.zipper.app.accessibility.KarvishControlOverlay.hide()
            }
        } else {
            com.zipper.app.accessibility.KarvishBubbleOverlay.onGoalStopped(result.finalSummary)
            com.zipper.app.core.graphics.LiveOverlayManager.dismissOverlay()
            com.zipper.app.accessibility.KarvishControlOverlay.hide()
        }

        return result.success to result.finalSummary
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Karvish agent runtime",
                NotificationManager.IMPORTANCE_LOW
            )
            getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
        }
    }

    private fun buildNotification(text: String): Notification {
        val stopIntent = Intent(this, AgentRuntimeService::class.java).apply {
            action = ACTION_STOP_AGENT
        }
        val stopPendingIntent = android.app.PendingIntent.getService(
            this,
            4102,
            stopIntent,
            android.app.PendingIntent.FLAG_UPDATE_CURRENT or (if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) android.app.PendingIntent.FLAG_IMMUTABLE else 0)
        )

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.mipmap.ic_launcher)
            .setContentTitle("Karvish Agent Runtime")
            .setContentText(text)
            .setOngoing(true)
            .addAction(R.drawable.ic_stop, "⛔ Hard Stop", stopPendingIntent)
            .build()
    }

    private fun updateNotification(text: String) {
        getSystemService(NotificationManager::class.java).notify(NOTIFICATION_ID, buildNotification(text))
    }

    override fun onDestroy() {
        activeJob?.cancel()
        serviceScope.cancel()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null
}