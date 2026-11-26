package com.zipper.app.core.web

import android.app.AlarmManager
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import android.os.Build
import android.util.Log
import androidx.core.app.NotificationCompat
import com.zipper.app.R
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.json.JSONArray
import org.json.JSONObject

enum class WatchType {
    PRICE_DROP,
    STATUS_CHANGE
}

data class WebWatchItem(
    val id: String,
    val title: String,
    val url: String,
    val watchType: WatchType,
    val targetThreshold: Double? = null,
    val lastKnownValue: String? = null,
    val isTriggered: Boolean = false,
    val createdAt: Long = System.currentTimeMillis()
)

/**
 * Background Passive Watcher for Web Deals, Price Drops, and Status Transitions.
 *
 * Runs scheduled checks via Android AlarmManager without waking or hijacking the screen.
 * Triggers rich notifications when:
 * - A watched product's price drops below target
 * - A train PNR seat is confirmed
 * - Exam results or portal notices are updated
 */
class WebWatcherManager(private val context: Context) {

    companion object {
        private const val TAG = "WebWatcherManager"
        private const val PREFS_NAME = "karvish_web_watcher_prefs"
        private const val KEY_WATCH_ITEMS = "watch_items_json"
        const val ACTION_CHECK_WATCHLIST = "com.zipper.app.core.web.ACTION_CHECK_WATCHLIST"
        private const val CHECK_INTERVAL_MS = 2 * 60 * 60 * 1000L // 2 Hours
    }

    private val prefs: SharedPreferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun addWatchItem(item: WebWatchItem) {
        val list = getAllWatchItems().filter { it.id != item.id }.toMutableList()
        list.add(0, item)
        saveWatchItems(list)
        schedulePeriodicAlarm()
        Log.i(TAG, "Added watch item: ${item.title} (${item.watchType})")
    }

    fun getAllWatchItems(): List<WebWatchItem> {
        val raw = prefs.getString(KEY_WATCH_ITEMS, null) ?: return emptyList()
        return try {
            val arr = JSONArray(raw)
            val list = mutableListOf<WebWatchItem>()
            for (i in 0 until arr.length()) {
                val obj = arr.getJSONObject(i)
                list.add(
                    WebWatchItem(
                        id = obj.getString("id"),
                        title = obj.getString("title"),
                        url = obj.getString("url"),
                        watchType = WatchType.valueOf(obj.getString("watchType")),
                        targetThreshold = if (obj.has("targetThreshold")) obj.getDouble("targetThreshold") else null,
                        lastKnownValue = obj.optString("lastKnownValue").takeIf { it.isNotBlank() },
                        isTriggered = obj.optBoolean("isTriggered", false),
                        createdAt = obj.optLong("createdAt", System.currentTimeMillis())
                    )
                )
            }
            list
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse watch items", e)
            emptyList()
        }
    }

    fun removeWatchItem(id: String) {
        val list = getAllWatchItems().filter { it.id != id }
        saveWatchItems(list)
    }

    fun updateWatchItem(item: WebWatchItem) {
        val list = getAllWatchItems().toMutableList()
        val idx = list.indexOfFirst { it.id == item.id }
        if (idx != -1) {
            list[idx] = item
            saveWatchItems(list)
        }
    }

    private fun saveWatchItems(items: List<WebWatchItem>) {
        val arr = JSONArray()
        for (item in items) {
            val obj = JSONObject()
            obj.put("id", item.id)
            obj.put("title", item.title)
            obj.put("url", item.url)
            obj.put("watchType", item.watchType.name)
            item.targetThreshold?.let { obj.put("targetThreshold", it) }
            item.lastKnownValue?.let { obj.put("lastKnownValue", it) }
            obj.put("isTriggered", item.isTriggered)
            obj.put("createdAt", item.createdAt)
            arr.put(obj)
        }
        prefs.edit().putString(KEY_WATCH_ITEMS, arr.toString()).apply()
    }

    fun schedulePeriodicAlarm() {
        val alarmManager = context.getSystemService(Context.ALARM_SERVICE) as? AlarmManager ?: return
        val intent = Intent(context, WebWatcherReceiver::class.java).apply {
            action = ACTION_CHECK_WATCHLIST
        }
        val flags = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        } else {
            PendingIntent.FLAG_UPDATE_CURRENT
        }
        val pendingIntent = PendingIntent.getBroadcast(context, 1001, intent, flags)

        val triggerAt = System.currentTimeMillis() + CHECK_INTERVAL_MS
        alarmManager.setInexactRepeating(
            AlarmManager.RTC_WAKEUP,
            triggerAt,
            CHECK_INTERVAL_MS,
            pendingIntent
        )
        Log.i(TAG, "Scheduled periodic web watcher alarm (Every 2h)")
    }
}

/**
 * BroadcastReceiver triggered by AlarmManager to check watchlist items silently in the background.
 */
class WebWatcherReceiver : BroadcastReceiver() {

    companion object {
        private const val TAG = "WebWatcherReceiver"
        private const val CHANNEL_ID = "karvish_web_alerts"
    }

    override fun onReceive(context: Context, intent: Intent?) {
        Log.i(TAG, "WebWatcher wake-up triggered: scanning active watchlist")
        val watcher = WebWatcherManager(context)
        val activeItems = watcher.getAllWatchItems().filter { !it.isTriggered }
        if (activeItems.isEmpty()) return

        val scope = CoroutineScope(Dispatchers.Main)
        scope.launch {
            val controller = HiddenWebController(context)
            try {
                for (item in activeItems) {
                    val loaded = controller.navigate(item.url, timeoutMs = 10000L)
                    if (!loaded) continue

                    val state = controller.distillWebState()

                    if (item.watchType == WatchType.PRICE_DROP) {
                        val text = state.nodes.mapNotNull { it.text }.joinToString(" ")
                        val priceMatch = Regex("""₹\s*(\d+(?:,\d+)*(?:\.\d+)?)""").find(text)
                        if (priceMatch != null) {
                            val parsedPrice = priceMatch.groupValues[1].replace(",", "").toDoubleOrNull()
                            if (parsedPrice != null && item.targetThreshold != null && parsedPrice <= item.targetThreshold) {
                                postAlertNotification(
                                    context,
                                    "🎉 Price Drop Alert: ${item.title}",
                                    "Price dropped to ₹$parsedPrice! Target was ₹${item.targetThreshold}. Tap to view."
                                )
                                watcher.updateWatchItem(item.copy(isTriggered = true, lastKnownValue = "₹$parsedPrice"))
                            }
                        }
                    } else if (item.watchType == WatchType.STATUS_CHANGE) {
                        val text = state.nodes.mapNotNull { it.text }.joinToString(" ")
                        if (text.contains("confirmed", ignoreCase = true) || text.contains("published", ignoreCase = true)) {
                            postAlertNotification(
                                context,
                                "📢 Status Update: ${item.title}",
                                "Status change detected for watched link! Tap to open."
                            )
                            watcher.updateWatchItem(item.copy(isTriggered = true, lastKnownValue = "Confirmed/Published"))
                        }
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error checking watchlist in background", e)
            } finally {
                controller.destroy()
            }
        }
    }

    private fun postAlertNotification(context: Context, title: String, body: String) {
        val notificationManager = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Karvish Web Alerts",
                NotificationManager.IMPORTANCE_HIGH
            ).apply {
                description = "Price drop and status notifications from Karvish-Web"
            }
            notificationManager.createNotificationChannel(channel)
        }

        val notification = NotificationCompat.Builder(context, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setContentTitle(title)
            .setContentText(body)
            .setStyle(NotificationCompat.BigTextStyle().bigText(body))
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setAutoCancel(true)
            .build()

        notificationManager.notify(System.currentTimeMillis().toInt(), notification)
    }
}
