package com.zipper.app.core.intelligence.reply

import android.service.notification.StatusBarNotification
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.util.concurrent.ConcurrentHashMap

data class QueuedPing(
    val sender: String,
    val text: String,
    val sbn: StatusBarNotification? = null,
    val timestampMs: Long = System.currentTimeMillis()
)

/**
 * Multi-Ping Debouncer.
 *
 * Solves the rapid-fire texting reality in WhatsApp/SMS.
 * Holds incoming messages from the same sender within a 15-second window,
 * concatenates them into one unified thought, and dispatches only when the sender finishes typing.
 */
object MultiPingDebouncer {

    private const val TAG = "MultiPingDebouncer"
    const val DEFAULT_DEBOUNCE_DELAY_MS = 7000L // 7 seconds of silence before replying (fast yet merges rapid bursts)
    const val DEFAULT_MAX_ACCUMULATION_MS = 12000L // Maximum 12 seconds before forced dispatch

    var debounceDelayMs = DEFAULT_DEBOUNCE_DELAY_MS
    var maxAccumulationMs = DEFAULT_MAX_ACCUMULATION_MS

    private val scope = CoroutineScope(Dispatchers.Default)

    private val activeBatches = ConcurrentHashMap<String, MutableList<QueuedPing>>()
    private val debounceJobs = ConcurrentHashMap<String, Job>()
    private val batchStartTimes = ConcurrentHashMap<String, Long>()

    /**
     * Enqueues a message. Returns true if queued for debouncing, false if handled immediately.
     */
    fun enqueueMessage(
        senderKey: String,
        senderName: String,
        text: String,
        sbn: StatusBarNotification? = null,
        onBatchReady: (senderKey: String, senderName: String, unifiedText: String, latestSbn: StatusBarNotification?) -> Unit
    ) {
        val now = System.currentTimeMillis()
        val pings = activeBatches.computeIfAbsent(senderKey) { mutableListOf() }

        synchronized(pings) {
            pings.add(QueuedPing(sender = senderName, text = text.trim(), sbn = sbn, timestampMs = now))
            if (!batchStartTimes.containsKey(senderKey)) {
                batchStartTimes[senderKey] = now
            }
        }

        val startTime = batchStartTimes[senderKey] ?: now
        val elapsed = now - startTime

        // Cancel previous pending dispatch timer
        debounceJobs[senderKey]?.cancel()

        val waitMs = if (elapsed >= maxAccumulationMs) {
            50L // Max accumulation reached, flush immediately
        } else {
            debounceDelayMs
        }

        Log.d(TAG, "Queued ping from '$senderName' (${pings.size} in batch). Debouncing for ${waitMs}ms...")

        val job = scope.launch {
            delay(waitMs)
            flushBatch(senderKey, onBatchReady)
        }
        debounceJobs[senderKey] = job
    }

    private fun flushBatch(
        senderKey: String,
        onBatchReady: (senderKey: String, senderName: String, unifiedText: String, latestSbn: StatusBarNotification?) -> Unit
    ) {
        val pings = activeBatches.remove(senderKey) ?: return
        debounceJobs.remove(senderKey)
        batchStartTimes.remove(senderKey)

        if (pings.isEmpty()) return

        val latestSbn = pings.last().sbn
        val senderName = pings.last().sender
        val unifiedText = pings.map { it.text }.filter { it.isNotBlank() }.joinToString("\n")

        Log.i(TAG, "⚡ Multi-ping debounce finished for '$senderName'. Merged ${pings.size} pings into: '$unifiedText'")
        onBatchReady(senderKey, senderName, unifiedText, latestSbn)
    }

    fun clear(senderKey: String) {
        debounceJobs[senderKey]?.cancel()
        debounceJobs.remove(senderKey)
        activeBatches.remove(senderKey)
        batchStartTimes.remove(senderKey)
    }

    fun restoreDefaults() {
        debounceDelayMs = DEFAULT_DEBOUNCE_DELAY_MS
        maxAccumulationMs = DEFAULT_MAX_ACCUMULATION_MS
    }

    fun resetForTesting(debounceMs: Long = DEFAULT_DEBOUNCE_DELAY_MS, maxAccMs: Long = DEFAULT_MAX_ACCUMULATION_MS) {
        debounceDelayMs = debounceMs
        maxAccumulationMs = maxAccMs
        debounceJobs.values.forEach { it.cancel() }
        debounceJobs.clear()
        activeBatches.clear()
        batchStartTimes.clear()
    }
}
