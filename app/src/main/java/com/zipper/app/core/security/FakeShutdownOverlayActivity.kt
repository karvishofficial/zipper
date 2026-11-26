package com.zipper.app.core.security

import android.app.Activity
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Color
import android.media.AudioManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.KeyEvent
import android.view.View
import android.view.WindowInsets
import android.view.WindowInsetsController
import android.view.WindowManager
import android.widget.FrameLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import kotlinx.coroutines.*

/**
 * The "Mimic" (Fake Shutdown) stealth activity.
 * Mimics a realistic shutdown sequence, then turns the display pitch black and silences all
 * audio streams while keeping the CPU awake and anti-theft services fully active.
 *
 * Disarm Mechanism:
 * - Press Volume UP three times within 2.5 seconds, OR
 * - Plug in charging cable, OR
 * - Remote command via WhatsApp/SMS ("Zipper: disarm").
 */
class FakeShutdownOverlayActivity : Activity() {

    private var audioManager: AudioManager? = null
    private val handler = Handler(Looper.getMainLooper())
    private var volumeUpPressCount = 0
    private var lastVolumeUpPressTime = 0L

    private val disarmReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            when (intent.action) {
                ACTION_DISARM -> {
                    disarmAndExit("Remote disarm command received")
                }
                Intent.ACTION_SCREEN_ON -> {
                    if (SecurityConfig.isStealthModeActive(context)) {
                        // When screen turns on (e.g. thief presses power button), immediately pin black screen over lockscreen
                        FakeShutdownOverlayActivity.start(context)
                        val lp = window.attributes
                        lp.screenBrightness = 0.001f
                        window.attributes = lp
                        hideSystemUI()
                        collapseStatusBar()
                    }
                }
            }
        }
    }

    companion object {
        const val ACTION_DISARM = "com.zipper.app.core.security.ACTION_DISARM_STEALTH"

        fun start(context: Context) {
            val intent = Intent(context, FakeShutdownOverlayActivity::class.java).apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP)
            }
            context.startActivity(intent)
        }

        fun disarm(context: Context) {
            val intent = Intent(ACTION_DISARM).apply {
                setPackage(context.packageName)
            }
            context.sendBroadcast(intent)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Ensure activity renders over the lockscreen
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
            setShowWhenLocked(true)
            setTurnScreenOn(true)
        } else {
            @Suppress("DEPRECATION")
            window.addFlags(
                WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED or
                WindowManager.LayoutParams.FLAG_DISMISS_KEYGUARD or
                WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON
            )
        }

        @Suppress("DEPRECATION")
        window.addFlags(
            WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON or
            WindowManager.LayoutParams.FLAG_FULLSCREEN
        )

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            val lp = window.attributes
            lp.layoutInDisplayCutoutMode = WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES
            window.attributes = lp
        }
        window.statusBarColor = Color.BLACK
        window.navigationBarColor = Color.BLACK

        audioManager = getSystemService(Context.AUDIO_SERVICE) as? AudioManager
        SecurityConfig.setStealthModeActive(this, true)

        // Forcibly lock in high accuracy GPS, Wi-Fi, Mobile Data, Bluetooth ON and Airplane Mode OFF
        HardwareEnforcer.enforceStealthConnectivity(this)

        val rootLayout = FrameLayout(this).apply {
            setBackgroundColor(Color.BLACK)
            isClickable = true
            isFocusable = true
            setOnTouchListener { _, _ -> true } // Consume all touch events so black screen ignores taps
        }

        val fakeShutdownContainer = FrameLayout(this).apply {
            setBackgroundColor(Color.BLACK)
        }

        val spinner = ProgressBar(this).apply {
            isIndeterminate = true
        }
        val layoutParams = FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.WRAP_CONTENT,
            FrameLayout.LayoutParams.WRAP_CONTENT
        ).apply {
            gravity = android.view.Gravity.CENTER
        }

        val shuttingDownText = TextView(this).apply {
            text = "Shutting down..."
            setTextColor(Color.WHITE)
            textSize = 16f
        }
        val textParams = FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.WRAP_CONTENT,
            FrameLayout.LayoutParams.WRAP_CONTENT
        ).apply {
            gravity = android.view.Gravity.CENTER
            topMargin = 120
        }

        fakeShutdownContainer.addView(spinner, layoutParams)
        fakeShutdownContainer.addView(shuttingDownText, textParams)
        rootLayout.addView(fakeShutdownContainer)
        setContentView(rootLayout)

        hideSystemUI()

        // After 2.5 seconds of realistic "Shutting down..." animation, turn screen pitch-black
        handler.postDelayed({
            runCatching {
                fakeShutdownContainer.visibility = View.GONE
                // Set screen brightness to minimum
                val lp = window.attributes
                lp.screenBrightness = 0.01f
                window.attributes = lp

                // Mute all audio streams so phone appears completely silent
                silenceAudio()

                // Trigger an initial intruder capture of the thief holding the "dead" phone
                IntruderCaptureService.start(this@FakeShutdownOverlayActivity, "MIMIC_SHUTDOWN_TRIGGERED")
            }.onFailure { e ->
                android.util.Log.e("FakeShutdown", "Failed during blackout transition: ${e.message}", e)
            }
        }, 2500L)

        // Register disarm listener
        val filter = IntentFilter().apply {
            addAction(ACTION_DISARM)
            addAction(Intent.ACTION_SCREEN_ON)
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(disarmReceiver, filter, Context.RECEIVER_NOT_EXPORTED)
        } else {
            registerReceiver(disarmReceiver, filter)
        }
    }

    override fun onResume() {
        super.onResume()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
            setShowWhenLocked(true)
            setTurnScreenOn(true)
        }
        hideSystemUI()
        collapseStatusBar()
        val lp = window.attributes
        lp.screenBrightness = 0.001f
        window.attributes = lp
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (!hasFocus && SecurityConfig.isStealthModeActive(this)) {
            // Instantly slam shut any pulled-down status bar, notification shade, or quick settings
            collapseStatusBar()
            hideSystemUI()
            // Keep fake shutdown blackout screen firmly pinned on top
            FakeShutdownOverlayActivity.start(this)
        }
    }

    private fun collapseStatusBar() {
        runCatching {
            val statusBarService = getSystemService("statusbar")
            val statusBarManager = Class.forName("android.app.StatusBarManager")
            val collapse = statusBarManager.getMethod("collapsePanels")
            collapse.isAccessible = true
            collapse.invoke(statusBarService)
        }
    }

    private fun hideSystemUI() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            window.insetsController?.let { controller ->
                controller.hide(WindowInsets.Type.statusBars() or WindowInsets.Type.navigationBars())
                controller.systemBarsBehavior = WindowInsetsController.BEHAVIOR_DEFAULT
            }
        } else {
            @Suppress("DEPRECATION")
            window.decorView.systemUiVisibility = (
                View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY or
                View.SYSTEM_UI_FLAG_FULLSCREEN or
                View.SYSTEM_UI_FLAG_HIDE_NAVIGATION or
                View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN or
                View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
            )
        }
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        // Disarm Combo: 3 quick presses of Volume UP
        if (keyCode == KeyEvent.KEYCODE_VOLUME_UP) {
            val now = System.currentTimeMillis()
            if (now - lastVolumeUpPressTime < 1000L) {
                volumeUpPressCount++
            } else {
                volumeUpPressCount = 1
            }
            lastVolumeUpPressTime = now

            if (volumeUpPressCount >= 3) {
                disarmAndExit("Secret 3x Volume-Up Combo Pressed")
                return true
            }
            return true // Consume key event
        }

        // Consume all other keys (Volume Down, etc.)
        if (keyCode == KeyEvent.KEYCODE_VOLUME_DOWN || keyCode == KeyEvent.KEYCODE_BACK) {
            return true
        }

        return super.onKeyDown(keyCode, event)
    }

    private fun silenceAudio() {
        audioManager?.let { am ->
            runCatching {
                val nm = getSystemService(Context.NOTIFICATION_SERVICE) as? android.app.NotificationManager
                val canChangeDnd = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                    nm?.isNotificationPolicyAccessGranted == true
                } else true

                if (canChangeDnd) {
                    am.setStreamVolume(AudioManager.STREAM_RING, 0, 0)
                    am.setStreamVolume(AudioManager.STREAM_NOTIFICATION, 0, 0)
                }
                am.setStreamVolume(AudioManager.STREAM_MUSIC, 0, 0)
                am.setStreamVolume(AudioManager.STREAM_SYSTEM, 0, 0)
            }.onFailure { e ->
                android.util.Log.w("FakeShutdown", "silenceAudio suppressed: ${e.message}")
            }
        }
    }

    private fun restoreAudio() {
        audioManager?.let { am ->
            runCatching {
                val nm = getSystemService(Context.NOTIFICATION_SERVICE) as? android.app.NotificationManager
                val canChangeDnd = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                    nm?.isNotificationPolicyAccessGranted == true
                } else true

                if (canChangeDnd) {
                    val maxRing = am.getStreamMaxVolume(AudioManager.STREAM_RING)
                    am.setStreamVolume(AudioManager.STREAM_RING, maxRing / 2, 0)
                }
            }.onFailure { e ->
                android.util.Log.w("FakeShutdown", "restoreAudio suppressed: ${e.message}")
            }
        }
    }

    private fun disarmAndExit(reason: String) {
        SecurityConfig.setStealthModeActive(this, false)
        SecurityConfig.setLiveTrackingActive(this, false)
        IntruderCaptureService.stopTracking(this)

        kotlinx.coroutines.CoroutineScope(kotlinx.coroutines.Dispatchers.IO).launch {
            EmailAlertDispatcher.sendDisarmNotice(applicationContext, reason)
        }

        restoreAudio()
        HardwareEnforcer.restoreNormalHardwareSettings(applicationContext)

        val lp = window.attributes
        lp.screenBrightness = WindowManager.LayoutParams.BRIGHTNESS_OVERRIDE_NONE
        window.attributes = lp

        Toast.makeText(this, "Zipper: Stealth Mode Disarmed ($reason)", Toast.LENGTH_SHORT).show()
        finishAndRemoveTask()
    }

    override fun onDestroy() {
        super.onDestroy()
        runCatching { unregisterReceiver(disarmReceiver) }
        handler.removeCallbacksAndMessages(null)
        SecurityConfig.setStealthModeActive(this, false)
        restoreAudio()
    }
}
