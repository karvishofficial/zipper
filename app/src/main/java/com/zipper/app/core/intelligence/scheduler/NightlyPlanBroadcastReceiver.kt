package com.zipper.app.core.intelligence.scheduler

import android.app.AlarmManager
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.os.Build
import android.util.Log
import com.zipper.app.core.intelligence.planner.NightlyPlannerEngine
import com.zipper.app.core.intelligence.store.LifeRhythmPrefs
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.util.Calendar

/**
 * BroadcastReceiver for triggering the Nightly Planning cycle.
 * Fires at the configured time (e.g. 10:00 PM) via AlarmManager,
 * generates tomorrow's daily plan, arms 5-minute pre-call alarms,
 * and re-schedules itself for the next day.
 */
class NightlyPlanBroadcastReceiver : BroadcastReceiver() {

    companion object {
        private const val TAG = "NightlyPlanReceiver"
        const val ACTION_TRIGGER_NIGHTLY_PLAN = "com.zipper.app.ACTION_TRIGGER_NIGHTLY_PLAN"
        private const val REQUEST_CODE = 92201

        /**
         * Arms the exact AlarmManager alarm for the upcoming nightly planning cycle.
         */
        fun armNightlyAlarm(context: Context) {
            val prefs = LifeRhythmPrefs(context)
            val now = Calendar.getInstance()

            val target = Calendar.getInstance().apply {
                set(Calendar.HOUR_OF_DAY, prefs.nightPlanHour)
                set(Calendar.MINUTE, prefs.nightPlanMinute)
                set(Calendar.SECOND, 0)
                set(Calendar.MILLISECOND, 0)
            }

            // If time already passed today, arm for tomorrow
            if (target.before(now) || target.timeInMillis <= now.timeInMillis) {
                target.add(Calendar.DAY_OF_YEAR, 1)
            }

            val alarmManager = context.getSystemService(Context.ALARM_SERVICE) as? AlarmManager ?: return
            val intent = Intent(context, NightlyPlanBroadcastReceiver::class.java).apply {
                action = ACTION_TRIGGER_NIGHTLY_PLAN
            }
            val flags = PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
            val pendingIntent = PendingIntent.getBroadcast(context, REQUEST_CODE, intent, flags)

            try {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                    alarmManager.setExactAndAllowWhileIdle(
                        AlarmManager.RTC_WAKEUP,
                        target.timeInMillis,
                        pendingIntent
                    )
                } else {
                    alarmManager.setExact(
                        AlarmManager.RTC_WAKEUP,
                        target.timeInMillis,
                        pendingIntent
                    )
                }
                Log.i(TAG, "Armed next Nightly Plan alarm for: ${target.time}")
            } catch (e: Exception) {
                Log.w(TAG, "Failed to arm exact nightly alarm: ${e.message}")
                alarmManager.set(
                    AlarmManager.RTC_WAKEUP,
                    target.timeInMillis,
                    pendingIntent
                )
            }
        }
    }

    override fun onReceive(context: Context, intent: Intent) {
        val action = intent.action
        Log.i(TAG, "NightlyPlanBroadcastReceiver received action: $action")

        if (action == Intent.ACTION_BOOT_COMPLETED || action == Intent.ACTION_MY_PACKAGE_REPLACED) {
            armNightlyAlarm(context)
            LifePlanCallScheduler.scheduleUpcomingAlarms(context)
            return
        }

        if (action == ACTION_TRIGGER_NIGHTLY_PLAN) {
            val pendingResult = goAsync()
            CoroutineScope(Dispatchers.IO).launch {
                try {
                    val engine = NightlyPlannerEngine(context.applicationContext)
                    engine.executeNightlyPlan()
                } catch (e: Exception) {
                    Log.e(TAG, "Error executing nightly plan: ${e.message}", e)
                } finally {
                    armNightlyAlarm(context)
                    pendingResult.finish()
                }
            }
        }
    }
}
