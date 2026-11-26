package com.zipper.app.core.web

import android.content.Context
import android.content.SharedPreferences
import org.json.JSONArray
import org.json.JSONObject

data class WebSavedFlow(
    val id: String,
    val title: String,
    val targetUrl: String,
    val originalQuery: String,
    val actionsCount: Int,
    val timestamp: Long = System.currentTimeMillis()
)

class WebReplayStore(context: Context) {
    private val prefs: SharedPreferences =
        context.applicationContext.getSharedPreferences("karvish_web_replay_flows", Context.MODE_PRIVATE)

    companion object {
        private const val KEY_FLOWS = "saved_replay_flows"
    }

    fun saveFlow(title: String, targetUrl: String, originalQuery: String, actionsCount: Int): WebSavedFlow {
        val flow = WebSavedFlow(
            id = "flow_${System.currentTimeMillis()}",
            title = title,
            targetUrl = targetUrl,
            originalQuery = originalQuery,
            actionsCount = actionsCount
        )
        val existing = getSavedFlows().toMutableList()
        existing.removeAll { it.targetUrl == targetUrl || it.title == title }
        existing.add(0, flow)
        saveList(existing)
        return flow
    }

    fun getSavedFlows(): List<WebSavedFlow> {
        val raw = prefs.getString(KEY_FLOWS, null) ?: return emptyList()
        val list = mutableListOf<WebSavedFlow>()
        try {
            val arr = JSONArray(raw)
            for (i in 0 until arr.length()) {
                val obj = arr.getJSONObject(i)
                list.add(
                    WebSavedFlow(
                        id = obj.getString("id"),
                        title = obj.getString("title"),
                        targetUrl = obj.getString("targetUrl"),
                        originalQuery = obj.getString("originalQuery"),
                        actionsCount = obj.optInt("actionsCount", 0),
                        timestamp = obj.optLong("timestamp", 0L)
                    )
                )
            }
        } catch (_: Exception) {}
        return list
    }

    fun getLatestFlow(): WebSavedFlow? = getSavedFlows().firstOrNull()

    fun clearFlows() {
        prefs.edit().remove(KEY_FLOWS).apply()
    }

    private fun saveList(list: List<WebSavedFlow>) {
        val arr = JSONArray()
        for (f in list) {
            val obj = JSONObject().apply {
                put("id", f.id)
                put("title", f.title)
                put("targetUrl", f.targetUrl)
                put("originalQuery", f.originalQuery)
                put("actionsCount", f.actionsCount)
                put("timestamp", f.timestamp)
            }
            arr.put(obj)
        }
        prefs.edit().putString(KEY_FLOWS, arr.toString()).apply()
    }
}
