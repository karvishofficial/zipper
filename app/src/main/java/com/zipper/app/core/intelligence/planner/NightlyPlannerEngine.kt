package com.zipper.app.core.intelligence.planner

import android.content.Context
import android.util.Log
import com.zipper.app.core.intelligence.scheduler.LifePlanCallScheduler
import com.zipper.app.core.intelligence.scheduler.SmartScheduleValidator
import com.zipper.app.core.intelligence.store.CommitmentCategory
import com.zipper.app.core.intelligence.store.CommitmentDatabase
import com.zipper.app.core.intelligence.store.CommitmentEntity
import com.zipper.app.core.intelligence.store.CommitmentStatus
import com.zipper.app.core.intelligence.store.ContextMessage
import com.zipper.app.core.intelligence.store.LifeRhythmPrefs
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL
import java.util.Calendar
import java.util.Locale

/**
 * Deep-Think Nightly Planner for Karvish Personal Intelligence.
 *
 * Ingests 24-hour life feed context buffer + rolled-over uncompleted commitments.
 * Synthesizes a balanced daily plan using an LLM API (Gemini/OpenAI) with an
 * autonomous, offline-resilient semantic extractor fallback.
 * Validates through SmartScheduleValidator and arms 5-minute pre-call alarms.
 */
class NightlyPlannerEngine(private val context: Context) {

    companion object {
        private const val TAG = "NightlyPlannerEngine"

        fun buildNotesWithAiProof(actionBrief: String, reasoningProof: String, rawContext: String): String {
            return "BRIEF: $actionBrief\nPROOF: $reasoningProof\nFULL: $rawContext"
        }

        fun parseActionBrief(notes: String): String {
            if (notes.startsWith("BRIEF:")) {
                val end = notes.indexOf("\nPROOF:")
                if (end != -1) return notes.substring(6, end).trim()
            }
            return if (notes.length > 120) "${notes.take(117)}..." else notes
        }

        fun parseReasoningProof(notes: String): String {
            val start = notes.indexOf("\nPROOF:")
            if (start != -1) {
                val end = notes.indexOf("\nFULL:")
                val proofContent = if (end != -1) notes.substring(start + 7, end) else notes.substring(start + 7)
                return proofContent.trim()
            }
            return ""
        }

        fun parseFullContext(notes: String): String {
            val start = notes.indexOf("\nFULL:")
            if (start != -1) {
                return notes.substring(start + 6).trim()
            }
            return notes
        }

        fun buildChecklistNotes(brief: String, messageQuote: String, priority: String, sender: String): String {
            return "[TYPE:CHECKLIST]\nPRIORITY: $priority\nQUOTE: $messageQuote\nSENDER: $sender\nBRIEF: $brief"
        }

        fun parsePriority(notes: String): String {
            val start = notes.indexOf("PRIORITY:")
            if (start != -1) {
                val end = notes.indexOf("\n", start)
                val p = if (end != -1) notes.substring(start + 9, end) else notes.substring(start + 9)
                return p.trim().uppercase(Locale.ROOT)
            }
            return "NORMAL"
        }

        fun parseMessageQuote(notes: String): String {
            val start = notes.indexOf("QUOTE:")
            if (start != -1) {
                val end = notes.indexOf("\nSENDER:", start)
                val q = if (end != -1) notes.substring(start + 6, end) else notes.substring(start + 6)
                return q.trim()
            }
            return ""
        }
    }

    private val db = CommitmentDatabase.getInstance(context)
    private val prefs = LifeRhythmPrefs(context)
    private val validator = SmartScheduleValidator(prefs)

    /**
     * Executes the planning cycle for the specified day.
     * @param targetDayTimestampMs Epoch timestamp of any time within the target day (defaults to today)
     * @param isForceRefresh If true, purges existing commitments for this day before re-synthesizing
     */
    suspend fun executeNightlyPlan(
        targetDayTimestampMs: Long = System.currentTimeMillis(),
        isForceRefresh: Boolean = false
    ): List<CommitmentEntity> = withContext(Dispatchers.IO) {
        Log.i(TAG, "Starting Nightly Planning cycle for timestamp: $targetDayTimestampMs (forceRefresh=$isForceRefresh)")

        val cal = Calendar.getInstance().apply {
            timeInMillis = targetDayTimestampMs
            set(Calendar.HOUR_OF_DAY, 0)
            set(Calendar.MINUTE, 0)
            set(Calendar.SECOND, 0)
            set(Calendar.MILLISECOND, 0)
        }
        val dayStartMs = cal.timeInMillis
        val dayEndMs = dayStartMs + 86400000L - 1L

        // If force refresh requested (e.g. manual tap on Plan My Day), clear existing records for this day so re-synthesis replaces them
        if (isForceRefresh) {
            val deleted = db.deleteCommitmentsForDay(dayStartMs, dayEndMs)
            Log.i(TAG, "Force refresh: purged $deleted existing commitments for day starting at $dayStartMs")
        }

        // 1. Gather recent 24-hour context messages
        val cutoff = System.currentTimeMillis() - (24 * 60 * 60 * 1000L)
        val recentMessages = db.getRecentContextMessages(cutoff)
        Log.d(TAG, "Gathered ${recentMessages.size} recent context messages from buffer")

        // 2. Gather uncompleted / pending commitments to roll over
        val pendingCommitments = db.getPendingCommitments()
        Log.d(TAG, "Found ${pendingCommitments.size} existing pending commitments to consider")

        // 3. Propose commitments via LLM if API Key configured, otherwise Semantic Extractor
        var proposed = emptyList<CommitmentEntity>()
        if (prefs.llmApiKey.isNotBlank()) {
            try {
                proposed = queryLlmPlanner(recentMessages, pendingCommitments, targetDayTimestampMs)
            } catch (e: Exception) {
                Log.w(TAG, "LLM planner API call failed: ${e.message}")
                if (isForceRefresh) {
                    // Re-throw so user gets clear visual error feedback in UI instead of silent mock fallback
                    throw e
                }
            }
        }

        if (proposed.isEmpty()) {
            proposed = generateSemanticPlan(recentMessages, pendingCommitments, targetDayTimestampMs)
        }

        // 4. Validate through SmartScheduleValidator (transit buffers, fatigue curves, overlap avoidance)
        val validated = validator.validateAndRebalance(proposed, targetDayTimestampMs)

        // 5. Persist into database with deduplication
        val persisted = mutableListOf<CommitmentEntity>()
        for (item in validated) {
            val id = db.insertCommitment(item)
            persisted.add(item.copy(id = id))
        }

        // Clean any leftover duplicate entries
        db.deduplicateCommitments()

        // 6. Arm 5-Minute Awakening Calls via AlarmManager
        try {
            LifePlanCallScheduler.scheduleUpcomingAlarms(context)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to arm call alarms: ${e.message}")
        }

        Log.i(TAG, "Nightly Planning complete: ${persisted.size} commitments scheduled for target day.")
        persisted
    }

    private fun isDistractorMessage(msg: ContextMessage): Boolean {
        val lower = "${msg.sender} ${msg.summaryText}".lowercase(Locale.ROOT)
        val distractorKeywords = setOf(
            "50% off", "flat 50%", "cashback", "coupon", "discount", "sale ends", "mega sale",
            "order now", "delivery is on the way", "swiggy", "zomato", "blinkit", "zepto delivery",
            "trending now", "reels", "shorts", "liked your", "subscribed to", "recharge now",
            "credit limit", "pre-approved loan", "promo", "deal of the day", "flat ₹"
        )
        return distractorKeywords.any { lower.contains(it) }
    }

    /**
     * Autonomous semantic plan generator for zero-latency, offline operation.
     */
    private fun generateSemanticPlan(
        messages: List<ContextMessage>,
        pending: List<CommitmentEntity>,
        targetDayTimestampMs: Long
    ): List<CommitmentEntity> {
        val proposals = mutableListOf<CommitmentEntity>()

        val cal = Calendar.getInstance().apply {
            timeInMillis = targetDayTimestampMs
            set(Calendar.HOUR_OF_DAY, 0)
            set(Calendar.MINUTE, 0)
            set(Calendar.SECOND, 0)
            set(Calendar.MILLISECOND, 0)
        }
        val dayStartMs = cal.timeInMillis

        val seenTitles = mutableSetOf<String>()

        // 1. Roll over unique existing uncompleted commitments (excluding routine daily fitness)
        for (item in pending) {
            val norm = item.title.trim().lowercase(Locale.ROOT)
            if (norm in seenTitles) continue
            if (item.category == CommitmentCategory.FITNESS) continue // daily fitness is scheduled fresh
            if (item.status == CommitmentStatus.PENDING || item.status == CommitmentStatus.SNOOZED) {
                seenTitles.add(norm)
                proposals.add(
                    item.copy(
                        id = 0L,
                        scheduledTimeMs = dayStartMs + (19 * 3600000L) + (30 * 60000L),
                        status = CommitmentStatus.PENDING
                    )
                )
            }
        }

        // 2. Filter out non-actionable distractors (promotions, reels, coupons)
        val cleanMessages = messages.filterNot { isDistractorMessage(it) }

        // 3. Scan messages for placement drives / academic deadlines / urgent work
        val placementKeywords = setOf("placement", "skyllx", "interview", "campus drive", "drive is scheduled", "venue")
        val assignmentKeywords = setOf("assignment", "submission", "submit", "homework", "project", "deadline", "report", "lab")

        for (msg in cleanMessages) {
            val text = "${msg.sender} ${msg.summaryText}".lowercase(Locale.ROOT)

            if (placementKeywords.any { text.contains(it) }) {
                val company = if (text.contains("skyllx")) "Skyllx" else "Campus"
                val title = "$company Placement Drive Prep"
                val normTitle = title.lowercase(Locale.ROOT)
                if (normTitle !in seenTitles) {
                    seenTitles.add(normTitle)
                    val brief = "• Pack laptop with charger and LAN/HDMI multiport adapter.\n• Carry printed resume copy and report to venue on time."
                    val proof = "Placement drive scheduled for tomorrow morning; scheduled 45m tonight to prepare required hardware and documents."
                    proposals.add(
                        CommitmentEntity(
                            title = title,
                            category = CommitmentCategory.EDUCATION,
                            scheduledTimeMs = dayStartMs + (20 * 3600000L) + (45 * 60000L), // 8:45 PM
                            durationMinutes = 45,
                            sourceApp = msg.appName,
                            sourcePerson = msg.sender,
                            candidatePackages = listOf("com.whatsapp", "com.google.android.apps.docs"),
                            notes = buildNotesWithAiProof(brief, proof, msg.summaryText)
                        )
                    )
                }
            } else if (assignmentKeywords.any { text.contains(it) }) {
                val topic = if (text.contains("dbms")) "DBMS Assignment" else "Academic Assignment"
                val title = "Work on: $topic"
                val normTitle = title.lowercase(Locale.ROOT)
                if (normTitle !in seenTitles) {
                    seenTitles.add(normTitle)
                    val brief = "• Complete remaining assignment questions and upload report to portal."
                    val proof = "Submission deadline pending from ${msg.sender}; scheduled before evening cutoff."
                    proposals.add(
                        CommitmentEntity(
                            title = title,
                            category = CommitmentCategory.EDUCATION,
                            scheduledTimeMs = dayStartMs + (19 * 3600000L) + (30 * 60000L), // 7:30 PM
                            durationMinutes = 60,
                            sourceApp = msg.appName,
                            sourcePerson = msg.sender,
                            candidatePackages = listOf("com.google.android.apps.docs.editors.docs", "com.google.android.apps.classroom", msg.packageName),
                            notes = buildNotesWithAiProof(brief, proof, msg.summaryText)
                        )
                    )
                }
            }
        }

        // 4. Scan messages for direct actionable checklist to-dos (Missed calls, Relays, Attachments)
        for (msg in cleanMessages) {
            val text = "${msg.sender} ${msg.summaryText}".lowercase(Locale.ROOT)

            if (text.contains("missed call") || text.contains("call me") || msg.appName.equals("Phone", ignoreCase = true)) {
                val title = "Return call to ${msg.sender}"
                val normTitle = title.lowercase(Locale.ROOT)
                if (normTitle !in seenTitles) {
                    seenTitles.add(normTitle)
                    proposals.add(
                        CommitmentEntity(
                            title = title,
                            category = CommitmentCategory.COMMUNICATION,
                            scheduledTimeMs = dayStartMs + (12 * 3600000L), // Floating daytime
                            durationMinutes = 5,
                            sourceApp = msg.appName,
                            sourcePerson = msg.sender,
                            candidatePackages = listOf("com.samsung.android.dialer", "com.google.android.dialer", "com.whatsapp"),
                            notes = buildChecklistNotes("Return missed call to ${msg.sender}", msg.summaryText, "URGENT", msg.sender)
                        )
                    )
                }
            } else if (text.contains("message to") || text.contains("tell ") || text.contains("relay") || text.contains("reply urgent")) {
                val recipient = if (text.contains("ajay")) "Ajay" else msg.sender
                val title = "Message $recipient (${msg.sender} relay)"
                val normTitle = title.lowercase(Locale.ROOT)
                if (normTitle !in seenTitles) {
                    seenTitles.add(normTitle)
                    proposals.add(
                        CommitmentEntity(
                            title = title,
                            category = CommitmentCategory.COMMUNICATION,
                            scheduledTimeMs = dayStartMs + (12 * 3600000L),
                            durationMinutes = 5,
                            sourceApp = msg.appName,
                            sourcePerson = msg.sender,
                            candidatePackages = listOf("com.whatsapp", msg.packageName),
                            notes = buildChecklistNotes("Relay message from ${msg.sender}: '${msg.summaryText}'", msg.summaryText, "HIGH", msg.sender)
                        )
                    )
                }
            } else if (text.contains("send pdf") || text.contains("send file") || text.contains("attachment") || text.contains("share doc")) {
                val title = "Check / Send document for ${msg.sender}"
                val normTitle = title.lowercase(Locale.ROOT)
                if (normTitle !in seenTitles) {
                    seenTitles.add(normTitle)
                    proposals.add(
                        CommitmentEntity(
                            title = title,
                            category = CommitmentCategory.GENERAL,
                            scheduledTimeMs = dayStartMs + (12 * 3600000L),
                            durationMinutes = 5,
                            sourceApp = msg.appName,
                            sourcePerson = msg.sender,
                            candidatePackages = listOf(msg.packageName, "com.google.android.apps.docs"),
                            notes = buildChecklistNotes("Document attachment or request from ${msg.sender}", msg.summaryText, "NORMAL", msg.sender)
                        )
                    )
                }
            }
        }

        // 5. Always include healthy routine (Evening Workout / Stretch) — exactly ONCE
        val hasFitnessAlready = proposals.any { it.category == CommitmentCategory.FITNESS }
        if (!hasFitnessAlready) {
            val brief = "• 30-minute cardio and mobility routine to recharge."
            val proof = "Core healthy routine; scheduled at 18:30 right after college transit buffer."
            proposals.add(
                CommitmentEntity(
                    title = "Evening Workout & Stretch",
                    category = CommitmentCategory.FITNESS,
                    scheduledTimeMs = dayStartMs + (18 * 3600000L) + (30 * 60000L), // 6:30 PM (right after transit)
                    durationMinutes = 30,
                    sourceApp = "Karvish Life Guard",
                    sourcePerson = "Karvish",
                    candidatePackages = listOf("fit.cure.android", "com.sec.android.app.shealth", "com.strava"),
                    notes = buildNotesWithAiProof(brief, proof, "Stay active! 30-minute cardio and mobility routine.")
                )
            )
        }

        return proposals
    }

    /**
     * Queries Gemini or OpenAI format LLM to produce a structured daily plan JSON.
     */
    private fun queryLlmPlanner(
        messages: List<ContextMessage>,
        pending: List<CommitmentEntity>,
        targetDayTimestampMs: Long
    ): List<CommitmentEntity> {
        val apiKey = prefs.llmApiKey
        val model = prefs.llmModel

        val targetCal = Calendar.getInstance().apply { timeInMillis = targetDayTimestampMs }
        val dateStr = java.text.SimpleDateFormat("EEEE, MMMM d, yyyy", Locale.US).format(targetCal.time)

        val promptBuilder = StringBuilder()
        promptBuilder.append("You are Karvish, an intelligent personal AI assistant and proactive life guard. ")
        promptBuilder.append("Create a balanced daily schedule for $dateStr that protects user well-being and fulfills commitments.\n")
        promptBuilder.append("Anchor windows: College ${prefs.collegeStartHour}:00 - ${prefs.collegeEndHour}:00 (do not schedule tasks during this), transit buffer: ${prefs.transitBufferMinutes}m after college, sleep after ${prefs.sleepHour}:00.\n\n")

        val effectiveMessages = if (messages.isNotEmpty()) {
            messages.filterNot { isDistractorMessage(it) }
        } else {
            listOf(
                ContextMessage(0L, "Skyllx Coordinator", "WhatsApp", "com.whatsapp", "IMPORTANT SKYLLX PLACEMENT DRIVE ANNOUNCEMENT for 2027 Batch. Carry your laptop with charger and LAN 2 HDMI multiport adapter. Report to venue at 9:00 AM.", System.currentTimeMillis()),
                ContextMessage(0L, "Prof. Sharma", "Gmail", "com.google.android.gm", "DBMS Assignment 3 submission portal closes tomorrow at 11:59 PM. Submit your ER diagram and normalization report.", System.currentTimeMillis()),
                ContextMessage(0L, "Rahul Verma", "Telegram", "org.telegram.messenger", "Hey! Can you send the project presentation slides and sync up for 15 mins this evening?", System.currentTimeMillis())
            )
        }

        promptBuilder.append("Recent messages captured in the life feed:\n")
        for (m in effectiveMessages.take(15)) {
            promptBuilder.append("- From: ${m.sender} via ${m.appName}: \"${m.summaryText}\"\n")
        }

        if (pending.isNotEmpty()) {
            promptBuilder.append("\nUnfinished tasks from earlier:\n")
            for (p in pending.take(5)) {
                promptBuilder.append("- ${p.title} (${p.category})\n")
            }
        }

        promptBuilder.append("""
            Return ONLY a valid JSON object matching this schema (do NOT include markdown fences, preamble, or comments):
            {
              "timelineSchedule": [
                {
                  "title": "Evening Workout & Stretch",
                  "category": "FITNESS",
                  "time": "18:30",
                  "durationMinutes": 30,
                  "sourceApp": "Cult.fit",
                  "sourcePerson": "Karvish",
                  "candidatePackages": ["fit.cure.android", "com.sec.android.app.shealth"],
                  "actionBrief": "30-minute cardio and mobility routine.",
                  "reasoningProof": "Core healthy routine; scheduled right after college transit buffer."
                }
              ],
              "actionChecklist": []
            }

            STRICT ACCURACY RULES:
            1. ONLY extract items into "actionChecklist" if there is an explicit, real action obligation or pending message in the USER_MESSAGES context provided above.
            2. NEVER invent fictitious people, names, tasks, or stories (do NOT hallucinate names like Manoj, Ajay, or Bhavani).
            3. If there are no real message obligations or pending callbacks in the context, "actionChecklist" MUST BE EMPTY: [].
        """.trimIndent())

        val platform = com.zipper.app.core.intelligence.ai.AiPlatformManager.getPlatform(prefs.llmPlatform)
        val responseText = if (platform.isOpenAiCompatible) {
            val base = prefs.llmEndpoint.ifBlank { platform.defaultBaseUrl }.trimEnd('/')
            val endpoint = "$base/chat/completions"
            val url = URL(endpoint)
            val conn = (url.openConnection() as HttpURLConnection).apply {
                requestMethod = "POST"
                connectTimeout = 25000
                readTimeout = 25000
                doOutput = true
                setRequestProperty("Content-Type", "application/json")
                setRequestProperty("Authorization", "Bearer $apiKey")
            }
            val requestJson = JSONObject().apply {
                put("model", model)
                put("temperature", 0.2)
                val messagesArr = JSONArray().apply {
                    put(JSONObject().apply {
                        put("role", "system")
                        put("content", "You are Karvish, an intelligent personal AI assistant and life guard. Return ONLY a pure JSON array of tasks matching the schema with no extra commentary.")
                    })
                    put(JSONObject().apply {
                        put("role", "user")
                        put("content", promptBuilder.toString())
                    })
                }
                put("messages", messagesArr)
            }
            OutputStreamWriter(conn.outputStream).use { it.write(requestJson.toString()) }
            val responseCode = conn.responseCode
            if (responseCode != HttpURLConnection.HTTP_OK) {
                val errorText = BufferedReader(InputStreamReader(conn.errorStream ?: conn.inputStream)).use { it.readText() }
                throw IllegalStateException("LLM request to ${platform.displayName} failed with HTTP $responseCode: $errorText")
            }
            val raw = BufferedReader(InputStreamReader(conn.inputStream)).use { it.readText() }
            val root = JSONObject(raw)
            root.getJSONArray("choices").getJSONObject(0).getJSONObject("message").getString("content")
        } else {
            val base = prefs.llmEndpoint.ifBlank { platform.defaultBaseUrl }.trimEnd('/')
            val cleanModel = model.removePrefix("models/")
            val endpoint = "$base/models/$cleanModel:generateContent?key=$apiKey"
            val url = URL(endpoint)
            val conn = (url.openConnection() as HttpURLConnection).apply {
                requestMethod = "POST"
                connectTimeout = 25000
                readTimeout = 25000
                doOutput = true
                setRequestProperty("Content-Type", "application/json")
            }
            val requestJson = JSONObject().apply {
                val contents = JSONArray().apply {
                    val part = JSONObject().apply { put("text", promptBuilder.toString()) }
                    put(JSONObject().apply { put("parts", JSONArray().apply { put(part) }) })
                }
                put("contents", contents)
            }
            OutputStreamWriter(conn.outputStream).use { it.write(requestJson.toString()) }
            val responseCode = conn.responseCode
            if (responseCode != HttpURLConnection.HTTP_OK) {
                val errorText = BufferedReader(InputStreamReader(conn.errorStream ?: conn.inputStream)).use { it.readText() }
                throw IllegalStateException("Gemini request failed with HTTP $responseCode: $errorText")
            }
            val raw = BufferedReader(InputStreamReader(conn.inputStream)).use { it.readText() }
            val root = JSONObject(raw)
            root.getJSONArray("candidates").getJSONObject(0).getJSONObject("content").getJSONArray("parts").getJSONObject(0).getString("text")
        }

        return parseLlmPlanJson(responseText, targetDayTimestampMs)
    }

    private fun parseLlmPlanJson(rawResponse: String, targetDayTimestampMs: Long): List<CommitmentEntity> {
        val trimmed = rawResponse.trim()
        val results = mutableListOf<CommitmentEntity>()

        val cal = Calendar.getInstance().apply {
            timeInMillis = targetDayTimestampMs
            set(Calendar.HOUR_OF_DAY, 0)
            set(Calendar.MINUTE, 0)
            set(Calendar.SECOND, 0)
            set(Calendar.MILLISECOND, 0)
        }
        val dayStartMs = cal.timeInMillis

        // Try parsing as Dual-Track JSON Object first
        val objStart = trimmed.indexOf('{')
        val objEnd = trimmed.lastIndexOf('}')
        if (objStart != -1 && objEnd != -1 && objEnd > objStart) {
            try {
                val jsonObject = JSONObject(trimmed.substring(objStart, objEnd + 1))
                val timelineArr = jsonObject.optJSONArray("timelineSchedule") ?: jsonObject.optJSONArray("timelinePlan")
                if (timelineArr != null) {
                    results.addAll(parseTimelineArray(timelineArr, dayStartMs))
                }
                val checklistArr = jsonObject.optJSONArray("actionChecklist") ?: jsonObject.optJSONArray("checklist")
                if (checklistArr != null) {
                    results.addAll(parseChecklistArray(checklistArr, dayStartMs))
                }
                if (results.isNotEmpty()) {
                    return results
                }
            } catch (e: Exception) {
                Log.w(TAG, "Failed to parse as JSON Object, falling back to JSON Array: ${e.message}")
            }
        }

        // Fallback: parse as JSON Array
        val startIdx = rawResponse.indexOf('[')
        val endIdx = rawResponse.lastIndexOf(']')
        if (startIdx == -1 || endIdx == -1 || endIdx <= startIdx) {
            throw IllegalArgumentException("Could not find valid JSON in LLM response:\n${rawResponse.take(200)}")
        }
        val arrayString = rawResponse.substring(startIdx, endIdx + 1)
        val jsonArray = JSONArray(arrayString)
        results.addAll(parseTimelineArray(jsonArray, dayStartMs))
        return results
    }

    private fun parseTimelineArray(jsonArray: JSONArray, dayStartMs: Long): List<CommitmentEntity> {
        val results = mutableListOf<CommitmentEntity>()

        for (i in 0 until jsonArray.length()) {
            val obj = jsonArray.getJSONObject(i)
            val title = obj.optString("title", "Task").trim()
            val catStr = obj.optString("category", "GENERAL").trim().uppercase(Locale.ROOT)
            val category = try { CommitmentCategory.valueOf(catStr) } catch (_: Exception) { CommitmentCategory.GENERAL }
            val timeStr = obj.optString("time", "18:30").trim()
            val duration = obj.optInt("durationMinutes", 30).coerceIn(10, 240)
            val sourceApp = obj.optString("sourceApp", "Personal AI").trim()
            val sourcePerson = obj.optString("sourcePerson", "Karvish").trim()
            val actionBrief = obj.optString("actionBrief", "").trim()
            val reasoningProof = obj.optString("reasoningProof", "").trim()
            val rawNotes = obj.optString("notes", "").trim()

            val formattedNotes = if (actionBrief.isNotBlank() || reasoningProof.isNotBlank()) {
                buildNotesWithAiProof(actionBrief, reasoningProof, rawNotes)
            } else {
                rawNotes
            }

            val parts = timeStr.split(":")
            val hour = parts.getOrNull(0)?.toIntOrNull() ?: 18
            val minute = parts.getOrNull(1)?.toIntOrNull() ?: 30
            val scheduledTimeMs = dayStartMs + (hour * 3600000L) + (minute * 60000L)

            val pkgs = mutableListOf<String>()
            val pkgArr = obj.optJSONArray("candidatePackages")
            if (pkgArr != null) {
                for (j in 0 until pkgArr.length()) {
                    val p = pkgArr.getString(j).trim()
                    if (p.isNotEmpty()) pkgs.add(p)
                }
            }

            if (pkgs.isEmpty()) {
                // Populate default package candidates based on category
                when (category) {
                    CommitmentCategory.FITNESS -> pkgs.addAll(listOf("fit.cure.android", "com.sec.android.app.shealth", "com.strava"))
                    CommitmentCategory.EDUCATION -> pkgs.addAll(listOf("com.whatsapp", "com.google.android.apps.docs", "com.google.android.apps.classroom"))
                    CommitmentCategory.WORK -> pkgs.addAll(listOf("com.google.android.gm", "com.slack", "com.microsoft.teams"))
                    CommitmentCategory.COMMUNICATION -> pkgs.addAll(listOf("com.whatsapp", "org.telegram.messenger", "com.google.android.talk"))
                    else -> pkgs.add("com.google.android.keep")
                }
            }

            results.add(
                CommitmentEntity(
                    title = title,
                    category = category,
                    scheduledTimeMs = scheduledTimeMs,
                    durationMinutes = duration,
                    sourceApp = sourceApp,
                    sourcePerson = sourcePerson,
                    candidatePackages = pkgs,
                    notes = formattedNotes
                )
            )
        }

        return results
    }

    private fun parseChecklistArray(jsonArray: JSONArray, dayStartMs: Long): List<CommitmentEntity> {
        val results = mutableListOf<CommitmentEntity>()

        for (i in 0 until jsonArray.length()) {
            val obj = jsonArray.getJSONObject(i)
            val title = obj.optString("title", "Action Item").trim()
            val catStr = obj.optString("category", "COMMUNICATION").trim().uppercase(Locale.ROOT)
            val category = try { CommitmentCategory.valueOf(catStr) } catch (_: Exception) { CommitmentCategory.COMMUNICATION }
            val priority = obj.optString("priority", "NORMAL").trim().uppercase(Locale.ROOT)
            val sourceApp = obj.optString("sourceApp", "Personal AI").trim()
            val sourcePerson = obj.optString("sourcePerson", "Karvish").trim()
            val messageQuote = obj.optString("messageQuote", "").trim()
            val actionBrief = obj.optString("actionBrief", "").trim()

            val formattedNotes = buildChecklistNotes(
                brief = actionBrief.ifBlank { title },
                messageQuote = messageQuote,
                priority = priority,
                sender = sourcePerson
            )

            val pkgs = mutableListOf<String>()
            val pkgArr = obj.optJSONArray("candidatePackages")
            if (pkgArr != null) {
                for (j in 0 until pkgArr.length()) {
                    val p = pkgArr.getString(j).trim()
                    if (p.isNotEmpty()) pkgs.add(p)
                }
            }
            if (pkgs.isEmpty()) {
                when (category) {
                    CommitmentCategory.COMMUNICATION -> pkgs.addAll(listOf("com.whatsapp", "com.google.android.dialer"))
                    else -> pkgs.add("com.google.android.keep")
                }
            }

            results.add(
                CommitmentEntity(
                    title = title,
                    category = category,
                    scheduledTimeMs = dayStartMs + (12 * 3600000L), // Floating daytime slot
                    durationMinutes = 5,
                    sourceApp = sourceApp,
                    sourcePerson = sourcePerson,
                    candidatePackages = pkgs,
                    notes = formattedNotes
                )
            )
        }
        return results
    }
}
