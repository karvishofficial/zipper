package com.zipper.app.core.plugins.sentry

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.util.Log
import com.zipper.app.core.remote.EmergencyRinger
import kotlin.math.sqrt

enum class SentryState {
    DISARMED,
    ARMING,
    ARMED,
    ALARMING
}

/**
 * Library & Train Sentry Anti-Theft Plugin.
 *
 * Runs 100% in the background with the screen locked/off.
 * Uses SensorManager (Accelerometer) to detect unauthorized movement or charger disconnection.
 *
 * Zero foreground screen interruption.
 */
class SentryGuardService(private val context: Context) : SensorEventListener {

    companion object {
        private const val TAG = "SentryGuardService"
        private const val MOTION_THRESHOLD = 3.5f // Acceleration delta threshold
    }

    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)

    var state: SentryState = SentryState.DISARMED
        private set

    private var baselineX = 0f
    private var baselineY = 0f
    private var baselineZ = 0f
    private var hasBaseline = false

    var onAlarmTriggered: ((reason: String) -> Unit)? = null

    private val powerReceiver = object : BroadcastReceiver() {
        override fun onReceive(ctx: Context?, intent: Intent?) {
            if (intent?.action == Intent.ACTION_POWER_DISCONNECTED && state == SentryState.ARMED) {
                Log.w(TAG, "Charger unplugged while Sentry was armed!")
                triggerAlarm("Charger was unplugged")
            }
        }
    }

    fun arm(gracePeriodMs: Long = 5000L) {
        if (state == SentryState.ARMED || state == SentryState.ARMING) return
        state = SentryState.ARMING
        hasBaseline = false

        context.registerReceiver(powerReceiver, IntentFilter(Intent.ACTION_POWER_DISCONNECTED))

        if (accelerometer != null) {
            sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_NORMAL)
        }

        Log.i(TAG, "Sentry arming started ($gracePeriodMs ms grace period)...")

        // Allow user grace period to set the phone down
        android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
            if (state == SentryState.ARMING) {
                state = SentryState.ARMED
                Log.i(TAG, "Sentry is now ARMED and actively guarding.")
            }
        }, gracePeriodMs)
    }

    fun disarm() {
        state = SentryState.DISARMED
        hasBaseline = false
        try {
            sensorManager.unregisterListener(this)
            context.unregisterReceiver(powerReceiver)
        } catch (_: Exception) {}

        EmergencyRinger.stopRinging(context)
        Log.i(TAG, "Sentry disarmed. Alarm silenced.")
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (event == null || state != SentryState.ARMED) return

        val x = event.values[0]
        val y = event.values[1]
        val z = event.values[2]

        if (!hasBaseline) {
            baselineX = x
            baselineY = y
            baselineZ = z
            hasBaseline = true
            return
        }

        val dx = x - baselineX
        val dy = y - baselineY
        val dz = z - baselineZ
        val delta = sqrt((dx * dx + dy * dy + dz * dz).toDouble()).toFloat()

        if (delta > MOTION_THRESHOLD) {
            Log.w(TAG, "Unauthorized motion detected! Delta: $delta > Threshold: $MOTION_THRESHOLD")
            triggerAlarm("Phone was moved (Motion delta: ${"%.1f".format(delta)})")
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    private fun triggerAlarm(reason: String) {
        if (state == SentryState.ALARMING) return
        state = SentryState.ALARMING

        // 1. Blast emergency siren at max volume
        EmergencyRinger.startRinging(context)

        // 2. Fire alert callback
        onAlarmTriggered?.invoke(reason)
        Log.e(TAG, "🚨 SENTRY ALARM TRIGGERED: $reason")
    }
}
