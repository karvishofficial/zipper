package com.zipper.app.core.history

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

data class KarvishHistoryTask(
    val id: String,
    val goal: String,
    val appName: String,
    val timestampMs: Long,
    val isSuccess: Boolean
) {
    fun getFormattedTime(): String {
        val diffSec = (System.currentTimeMillis() - timestampMs) / 1000L
        return when {
            diffSec < 60 -> "Just now"
            diffSec < 3600 -> "${diffSec / 60} mins ago"
            diffSec < 86400 -> "${diffSec / 3600} hours ago"
            else -> "${diffSec / 86400} days ago"
        }
    }
}

/**
 * Real persistent task history store.
 * Strictly stores and restores actual goals executed by Zipper on this physical device.
 * NO fake or mocked data.
 */
object KarvishTaskStore {
    private const val PREFS_NAME = "karvish_task_history"
    private const val KEY_TASKS = "history_tasks_json"

    fun getTasks(context: Context): List<KarvishHistoryTask> {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val jsonStr = prefs.getString(KEY_TASKS, null)
        if (jsonStr.isNullOrBlank()) {
            return emptyList()
        }
        return try {
            val array = JSONArray(jsonStr)
            val list = mutableListOf<KarvishHistoryTask>()
            for (i in 0 until array.length()) {
                val obj = array.getJSONObject(i)
                val goal = obj.optString("goal", "").trim()
                if (goal.isNotEmpty()) {
                    list.add(
                        KarvishHistoryTask(
                            id = obj.optString("id", System.currentTimeMillis().toString()),
                            goal = goal,
                            appName = obj.optString("appName", "Zipper"),
                            timestampMs = obj.optLong("timestampMs", System.currentTimeMillis()),
                            isSuccess = obj.optBoolean("isSuccess", true)
                        )
                    )
                }
            }
            list
        } catch (_: Exception) {
            emptyList()
        }
    }

    fun addTask(context: Context, goal: String, appName: String, isSuccess: Boolean) {
        if (goal.isBlank()) return
        val current = getTasks(context).toMutableList()
        current.removeAll { it.goal.equals(goal, ignoreCase = true) }
        val newTask = KarvishHistoryTask(
            id = System.currentTimeMillis().toString(),
            goal = goal.trim(),
            appName = appName.trim(),
            timestampMs = System.currentTimeMillis(),
            isSuccess = isSuccess
        )
        current.add(0, newTask)
        val trimmed = current.take(20)

        val array = JSONArray()
        for (task in trimmed) {
            val obj = JSONObject().apply {
                put("id", task.id)
                put("goal", task.goal)
                put("appName", task.appName)
                put("timestampMs", task.timestampMs)
                put("isSuccess", task.isSuccess)
            }
            array.put(obj)
        }

        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        prefs.edit().putString(KEY_TASKS, array.toString()).apply()
    }

    fun clearAll(context: Context) {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        prefs.edit().remove(KEY_TASKS).apply()
    }
}
