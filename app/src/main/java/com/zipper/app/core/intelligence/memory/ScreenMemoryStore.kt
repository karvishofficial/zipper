package com.zipper.app.core.intelligence.memory

import android.content.Context
import android.util.Log
import java.util.Locale
import java.util.concurrent.ConcurrentLinkedDeque

data class ScreenMemoryItem(
    val id: String = java.util.UUID.randomUUID().toString(),
    val packageName: String,
    val appName: String,
    val screenTitle: String,
    val keySnippets: List<String>,
    val extractedEntities: List<String>,
    val timestampMs: Long = System.currentTimeMillis()
)

/**
 * Universal Contextual Screen Memory (Universal Mobile Rewind).
 *
 * Keeps a privacy-first, 100% on-device rolling memory of recent UI text,
 * product names, addresses, and screen entities seen while navigating apps.
 */
object ScreenMemoryStore {

    private const val TAG = "ScreenMemoryStore"
    private const val MAX_MEMORY_ITEMS = 500

    private val memoryBuffer = ConcurrentLinkedDeque<ScreenMemoryItem>()
    private var lastRecordedHash: Int = 0
    private var lastRecordedTimeMs: Long = 0L

    private val PRICE_REGEX = Regex("""(?i)(?:₹|rs\.?|inr|\$)\s*[\d,]+(?:\.\d{2})?""")
    private val PHONE_REGEX = Regex("""\b(?:\+91[-\s]?)?[6-9]\d{9}\b""")

    fun recordScreen(
        packageName: String,
        appName: String,
        screenTitle: String,
        textNodes: List<String>
    ) {
        // Skip system packages and Zipper itself
        if (packageName == "com.zipper.app" || packageName == "com.android.systemui" || packageName == "android") {
            return
        }

        val cleanedTexts = textNodes
            .map { it.trim() }
            .filter { it.length in 3..250 }
            .distinct()
            .take(25)

        if (cleanedTexts.isEmpty() && screenTitle.isBlank()) return

        // Deduplicate rapid consecutive identical screens
        val currentHash = (packageName + screenTitle + cleanedTexts.take(3).joinToString()).hashCode()
        val now = System.currentTimeMillis()
        if (currentHash == lastRecordedHash && now - lastRecordedTimeMs < 4000L) {
            return
        }
        lastRecordedHash = currentHash
        lastRecordedTimeMs = now

        // Extract entities (prices, phone numbers, codes)
        val entities = mutableListOf<String>()
        val combinedText = cleanedTexts.joinToString(" ")
        PRICE_REGEX.findAll(combinedText).forEach { entities.add(it.value) }
        PHONE_REGEX.findAll(combinedText).forEach { entities.add(it.value) }

        val item = ScreenMemoryItem(
            packageName = packageName,
            appName = appName.ifBlank { packageName.substringAfterLast('.') },
            screenTitle = screenTitle.ifBlank { cleanedTexts.firstOrNull() ?: appName },
            keySnippets = cleanedTexts,
            extractedEntities = entities.distinct(),
            timestampMs = now
        )

        memoryBuffer.addFirst(item)
        while (memoryBuffer.size > MAX_MEMORY_ITEMS) {
            memoryBuffer.removeLast()
        }
        Log.d(TAG, "Recorded screen memory for ${item.appName}: '${item.screenTitle}' (${cleanedTexts.size} snippets)")
    }

    fun search(query: String, maxResults: Int = 5): List<ScreenMemoryItem> {
        val queryTokens = query.lowercase(Locale.ROOT)
            .split(" ", ",", ":", "-")
            .filter { it.length >= 2 }

        if (queryTokens.isEmpty()) return memoryBuffer.take(maxResults)

        return memoryBuffer.asSequence()
            .map { item ->
                var score = 0
                val searchable = "${item.appName} ${item.screenTitle} ${item.keySnippets.joinToString(" ")} ${item.extractedEntities.joinToString(" ")}".lowercase(Locale.ROOT)
                for (token in queryTokens) {
                    if (item.appName.lowercase(Locale.ROOT).contains(token)) score += 5
                    if (item.screenTitle.lowercase(Locale.ROOT).contains(token)) score += 4
                    if (searchable.contains(token)) score += 2
                }
                Pair(item, score)
            }
            .filter { it.second > 0 }
            .sortedByDescending { it.second }
            .map { it.first }
            .take(maxResults)
            .toList()
    }

    fun formatMemorySummary(query: String): String {
        val results = search(query)
        if (results.isEmpty()) {
            return "No recent screen memory found matching '$query'."
        }

        val sb = StringBuilder("🧠 Screen Memory Recall for '$query':\n")
        val timeFormat = java.text.SimpleDateFormat("hh:mm a", Locale.getDefault())
        for ((idx, item) in results.withIndex()) {
            val timeStr = timeFormat.format(java.util.Date(item.timestampMs))
            sb.append("\n${idx + 1}. [${item.appName}] at $timeStr: \"${item.screenTitle}\"\n")
            if (item.extractedEntities.isNotEmpty()) {
                sb.append("   • Detected: ${item.extractedEntities.joinToString(", ")}\n")
            }
            val snippet = item.keySnippets.take(3).joinToString(" • ")
            if (snippet.isNotBlank()) {
                sb.append("   • Preview: $snippet\n")
            }
        }
        return sb.toString().trim()
    }

    fun clear() {
        memoryBuffer.clear()
        lastRecordedHash = 0
    }
}
