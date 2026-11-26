package com.zipper.app.ditto

import org.json.JSONArray
import org.json.JSONObject
import java.util.UUID

/**
 * KARVISH DITTO DATA MODELS
 * Version 2.0 (Senior-Calibrated Architecture)
 *
 * Implements strict separation:
 * 1. Semantic Target Identity (Invariant under UI shuffling & layout changes)
 * 2. Diagnostic App Metadata (Launch package, screen density, device)
 * 3. Physical Execution Metadata (Tap coordinates, duration, gesture)
 * 4. Explicit State Before & After Snapshots with Transition Contracts
 */

data class DittoAppMetadata(
    val launchPackage: String? = null,
    val deviceModel: String = "Samsung Galaxy A26",
    val screenDensity: Float = 2.625f,
    val recordedTimestamp: Long = System.currentTimeMillis()
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("launchPackage", launchPackage ?: JSONObject.NULL)
        put("deviceModel", deviceModel)
        put("screenDensity", screenDensity.toDouble())
        put("recordedTimestamp", recordedTimestamp)
    }

    companion object {
        fun fromJson(json: JSONObject): DittoAppMetadata = DittoAppMetadata(
            launchPackage = if (json.isNull("launchPackage")) null else json.optString("launchPackage"),
            deviceModel = json.optString("deviceModel", "Android Device"),
            screenDensity = json.optDouble("screenDensity", 2.625).toFloat(),
            recordedTimestamp = json.optLong("recordedTimestamp", System.currentTimeMillis())
        )
    }
}

data class DittoTargetIdentity(
    val primaryText: String? = null,
    val contentDesc: String? = null,
    val semanticRole: String = "BUTTON", // BUTTON, INPUT, CARD, TAB, LIST_ITEM
    val structuralContainer: String? = null
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("primaryText", primaryText ?: JSONObject.NULL)
        put("contentDesc", contentDesc ?: JSONObject.NULL)
        put("semanticRole", semanticRole)
        put("structuralContainer", structuralContainer ?: JSONObject.NULL)
    }

    companion object {
        fun fromJson(json: JSONObject): DittoTargetIdentity = DittoTargetIdentity(
            primaryText = if (json.isNull("primaryText")) null else json.optString("primaryText"),
            contentDesc = if (json.isNull("contentDesc")) null else json.optString("contentDesc"),
            semanticRole = json.optString("semanticRole", "BUTTON"),
            structuralContainer = if (json.isNull("structuralContainer")) null else json.optString("structuralContainer")
        )
    }
}

data class DittoTargetMetadata(
    val resourceId: String? = null,
    val className: String? = null,
    val recordedBounds: List<Int> = emptyList() // [left, top, right, bottom]
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("resourceId", resourceId ?: JSONObject.NULL)
        put("className", className ?: JSONObject.NULL)
        put("recordedBounds", JSONArray(recordedBounds))
    }

    companion object {
        fun fromJson(json: JSONObject): DittoTargetMetadata {
            val boundsArray = json.optJSONArray("recordedBounds") ?: JSONArray()
            val boundsList = mutableListOf<Int>()
            for (i in 0 until boundsArray.length()) {
                boundsList.add(boundsArray.getInt(i))
            }
            return DittoTargetMetadata(
                resourceId = if (json.isNull("resourceId")) null else json.optString("resourceId"),
                className = if (json.isNull("className")) null else json.optString("className"),
                recordedBounds = boundsList
            )
        }
    }
}

data class DittoPhysicalExecution(
    val tapX: Int,
    val tapY: Int,
    val durationMs: Long = 50L
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("tapX", tapX)
        put("tapY", tapY)
        put("durationMs", durationMs)
    }

    companion object {
        fun fromJson(json: JSONObject): DittoPhysicalExecution = DittoPhysicalExecution(
            tapX = json.optInt("tapX", 0),
            tapY = json.optInt("tapY", 0),
            durationMs = json.optLong("durationMs", 50L)
        )
    }
}

data class DittoStateSnapshot(
    val keyAnchorText: String? = null,
    val visibleSignatures: List<String> = emptyList(),
    val nodeCount: Int = 0,
    val activePackage: String? = null
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("keyAnchorText", keyAnchorText ?: JSONObject.NULL)
        put("visibleSignatures", JSONArray(visibleSignatures))
        put("nodeCount", nodeCount)
        put("activePackage", activePackage ?: JSONObject.NULL)
    }

    companion object {
        fun fromJson(json: JSONObject): DittoStateSnapshot {
            val sigsArray = json.optJSONArray("visibleSignatures") ?: JSONArray()
            val sigsList = mutableListOf<String>()
            for (i in 0 until sigsArray.length()) {
                sigsList.add(sigsArray.getString(i))
            }
            return DittoStateSnapshot(
                keyAnchorText = if (json.isNull("keyAnchorText")) null else json.optString("keyAnchorText"),
                visibleSignatures = sigsList,
                nodeCount = json.optInt("nodeCount", 0),
                activePackage = if (json.isNull("activePackage")) null else json.optString("activePackage")
            )
        }
    }
}

data class DittoTransitionContract(
    val expectedMutation: String = "SCREEN_MUTATED",
    val requiredElement: String? = null
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("expectedMutation", expectedMutation)
        put("requiredElement", requiredElement ?: JSONObject.NULL)
    }

    companion object {
        fun fromJson(json: JSONObject): DittoTransitionContract = DittoTransitionContract(
            expectedMutation = json.optString("expectedMutation", "SCREEN_MUTATED"),
            requiredElement = if (json.isNull("requiredElement")) null else json.optString("requiredElement")
        )
    }
}

data class DittoStep(
    val stepIndex: Int,
    val actionType: String, // "TAP", "TYPE_TEXT", "SCROLL", "BACK", "SUBMIT"
    val argument: String? = null,
    val targetIdentity: DittoTargetIdentity,
    val targetMetadata: DittoTargetMetadata,
    val physicalExecution: DittoPhysicalExecution,
    val stateBefore: DittoStateSnapshot,
    var stateAfter: DittoStateSnapshot,
    val transitionContract: DittoTransitionContract,
    val isClipboardBound: Boolean = false
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("stepIndex", stepIndex)
        put("actionType", actionType)
        put("argument", argument ?: JSONObject.NULL)
        put("targetIdentity", targetIdentity.toJson())
        put("targetMetadata", targetMetadata.toJson())
        put("physicalExecution", physicalExecution.toJson())
        put("stateBefore", stateBefore.toJson())
        put("stateAfter", stateAfter.toJson())
        put("transitionContract", transitionContract.toJson())
        put("isClipboardBound", isClipboardBound)
    }

    companion object {
        fun fromJson(json: JSONObject): DittoStep = DittoStep(
            stepIndex = json.getInt("stepIndex"),
            actionType = json.getString("actionType"),
            argument = if (json.isNull("argument")) null else json.optString("argument"),
            targetIdentity = DittoTargetIdentity.fromJson(json.getJSONObject("targetIdentity")),
            targetMetadata = DittoTargetMetadata.fromJson(json.getJSONObject("targetMetadata")),
            physicalExecution = DittoPhysicalExecution.fromJson(json.getJSONObject("physicalExecution")),
            stateBefore = DittoStateSnapshot.fromJson(json.getJSONObject("stateBefore")),
            stateAfter = DittoStateSnapshot.fromJson(json.getJSONObject("stateAfter")),
            transitionContract = DittoTransitionContract.fromJson(json.getJSONObject("transitionContract")),
            isClipboardBound = json.optBoolean("isClipboardBound", false)
        )
    }
}

data class DittoFlow(
    val flowId: String = "ditto_" + UUID.randomUUID().toString().take(8),
    val title: String,
    val provenance: String = "verified_human_demonstration",
    val createdAt: Long = System.currentTimeMillis(),
    val appMetadata: DittoAppMetadata,
    val steps: List<DittoStep>
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("flowId", flowId)
        put("title", title)
        put("provenance", provenance)
        put("createdAt", createdAt)
        put("appMetadata", appMetadata.toJson())
        val stepsArray = JSONArray()
        steps.forEach { stepsArray.put(it.toJson()) }
        put("steps", stepsArray)
    }

    companion object {
        fun fromJson(json: JSONObject): DittoFlow {
            val stepsArray = json.getJSONArray("steps")
            val stepsList = mutableListOf<DittoStep>()
            for (i in 0 until stepsArray.length()) {
                stepsList.add(DittoStep.fromJson(stepsArray.getJSONObject(i)))
            }
            return DittoFlow(
                flowId = json.getString("flowId"),
                title = json.getString("title"),
                provenance = json.optString("provenance", "verified_human_demonstration"),
                createdAt = json.optLong("createdAt", System.currentTimeMillis()),
                appMetadata = DittoAppMetadata.fromJson(json.getJSONObject("appMetadata")),
                steps = stepsList
            )
        }
    }
}

/**
 * Universal Semantic Cleaner for UI Targets:
 * Normalizes Unicode spaces, removes dynamic message previews, timestamps,
 * badges, and image labels so target matching is stable across app updates and dynamic feeds.
 */
object DittoSemanticCleaner {
    fun clean(raw: String?): String {
        if (raw.isNullOrBlank()) return ""
        // 1. Normalize unicode spaces (non-breaking space, narrow no-break space \u202F, zero-width spaces)
        var s = raw.replace(Regex("[\\u00A0\\u2000-\\u200F\\u202F\\uFEFF]"), " ").trim()

        // 1b. Strip Private Use Area (PUA) icon symbols (FontAwesome, Material Icons, web icon fonts) and unprintable glyphs
        s = s.replace(Regex("[\\uE000-\\uF8FF\\u2630\\u22EE\\u22EF\\uFFF0-\\uFFFF]"), "").trim()

        // 2. Strip multi-line previews
        if (s.contains("\n")) {
            s = s.substringBefore("\n").trim()
        }

        // 3. Strip WhatsApp and messaging dynamic message snippets: " ~ God bless all..." or "~ God..."
        if (s.contains("~")) {
            s = s.substringBefore("~").trim()
        }

        // 4. Strip trailing timestamps / date indicators: "Yesterday", "Today", days of week, time
        s = s.replace(Regex("(?i)\\b(yesterday|today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun)\\b.*"), "").trim()
        s = s.replace(Regex("(?i)\\s*\\b\\d{1,2}/\\d{1,2}/\\d{2,4}\\b.*"), "").trim()
        s = s.replace(Regex("(?i)\\s*\\b\\d{1,2}:\\d{2}\\s*(am|pm)?\\b.*"), "").trim()

        // 5. Strip dynamic message durations, counts, media indicators: "4s", "4 photos", "2 videos", "3 messages"
        s = s.replace(Regex("(?i)\\s*\\b\\d+\\s*(s|m|h|d|w|mo|yr|sec|secs|min|mins|hr|hrs|photos?|videos?|messages?|voice\\s*messages?|audio)\\b.*"), "").trim()

        // 6. Strip transient presence & action verbs: "typing...", "online", "active now", "seen", "delivered", "read", "forwarded...", "sent...", "liked..."
        s = s.replace(Regex("(?i)\\b(typing\\.\\.\\.|online|active\\s*(now|\\d+\\w*\\s*ago)?|seen|delivered|read|opened|forwarded|sent|liked|reacted|replying\\s*to|pinned\\s*chat)\\b.*"), "").trim()

        // 7. Strip WhatsApp group suffix / duplicate names: e.g. "31st July PD, group 31st July PD" -> "31st July PD"
        if (s.contains(", group", ignoreCase = true)) {
            s = s.substringBefore(", group").trim()
        }

        // 8. Strip unread badges / call status: "4 unread messages", "(2)", "Incoming voice call", "Outgoing video call"
        s = s.replace(Regex("(?i)\\b\\d+\\s+unread\\s+messages?\\b.*"), "").trim()
        s = s.replace(Regex("(?i)\\b(incoming|outgoing)\\s+(voice|video|audio)?\\s*calls?\\b.*"), "").trim()
        s = s.replace(Regex("(?i)\\s*\\(\\d+\\)\\s*$"), "").trim()

        // 9. Strip generic image/avatar descriptors: "picture", "photo", "avatar", "icon", "thumbnail"
        s = s.replace(Regex("(?i)\\s*(picture|photo|avatar|profile|icon|thumbnail|image)"), "").trim()

        // 10. Deduplicate consecutive duplicate words / repeated halves (e.g. "Appa Appa" -> "Appa", "John Doe John Doe" -> "John Doe")
        val words = s.split(Regex("\\s+")).filter { it.isNotBlank() }
        if (words.isNotEmpty()) {
            val deduplicated = mutableListOf<String>()
            for (w in words) {
                if (deduplicated.isEmpty() || !deduplicated.last().equals(w, ignoreCase = true)) {
                    deduplicated.add(w)
                }
            }
            val half = deduplicated.size / 2
            if (half > 0 && deduplicated.size % 2 == 0) {
                val firstHalf = deduplicated.subList(0, half)
                val secondHalf = deduplicated.subList(half, deduplicated.size)
                if (firstHalf.map { it.lowercase() } == secondHalf.map { it.lowercase() }) {
                    s = firstHalf.joinToString(" ")
                } else {
                    s = deduplicated.joinToString(" ")
                }
            } else {
                s = deduplicated.joinToString(" ")
            }
        }

        return s.trim().ifEmpty { raw.trim() }
    }
}
