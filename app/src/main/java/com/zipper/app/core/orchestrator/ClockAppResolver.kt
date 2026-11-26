package com.zipper.app.core.orchestrator

import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager

/**
 * Resolves the installed Clock application package across different Android OEMs
 * (Samsung OneUI, Google Pixel, Xiaomi HyperOS, OnePlus OxygenOS).
 */
object ClockAppResolver {

    fun resolveClockPackage(context: Context): String {
        val pm = context.packageManager

        // 1. Standard Android Alarm Clock intent query
        try {
            val alarmIntent = Intent(android.provider.AlarmClock.ACTION_SET_ALARM)
            val alarmResolve = pm.resolveActivity(alarmIntent, PackageManager.MATCH_DEFAULT_ONLY)
            val pkg = alarmResolve?.activityInfo?.packageName
            if (!pkg.isNullOrBlank() && pkg != "android") {
                return pkg
            }
        } catch (_: Exception) {}

        // 2. Generic Launcher query for application named or matching "Clock"
        try {
            val mainIntent = Intent(Intent.ACTION_MAIN).apply {
                addCategory(Intent.CATEGORY_LAUNCHER)
            }
            val resolveInfos = pm.queryIntentActivities(mainIntent, 0)
            val match = resolveInfos.firstOrNull {
                pm.getApplicationLabel(it.activityInfo.applicationInfo).toString().trim().equals("clock", ignoreCase = true)
            } ?: resolveInfos.firstOrNull {
                it.activityInfo.packageName.contains("clock", ignoreCase = true)
            }
            if (match != null) {
                return match.activityInfo.packageName
            }
        } catch (_: Exception) {}

        return "com.android.deskclock"
    }
}
