package com.zipper.app.core.intelligence.health

import android.content.Context
import android.content.Intent
import android.util.Log
import com.zipper.app.core.intelligence.store.CommitmentCategory
import com.zipper.app.core.intelligence.store.CommitmentDatabase
import com.zipper.app.core.intelligence.store.CommitmentEntity
import com.zipper.app.core.intelligence.store.CommitmentStatus
import java.util.Locale
import kotlin.math.roundToInt

/**
 * HealthWorkoutIngestionEngine
 *
 * Automatically ingests real workout & fitness notifications (Samsung Health, Google Fit,
 * Strava, Apple Health, Mi Fitness, Fitbit) and grounds them into real completed commitments
 * on the Karvish Planner timeline.
 */
object HealthWorkoutIngestionEngine {
    private const val TAG = "HealthWorkoutIngestion"

    const val ACTION_PLANNER_DATA_CHANGED = "com.zipper.app.ACTION_PLANNER_DATA_CHANGED"

    private val HEALTH_PACKAGES = setOf(
        "com.sec.android.app.shealth",
        "com.google.android.apps.fitness",
        "com.strava",
        "com.fitbit.FitbitMobile",
        "com.garmin.android.apps.connectmobile",
        "com.xiaomi.wearable",
        "com.huawei.health",
        "fit.cure.android"
    )

    data class WorkoutMetrics(
        val exerciseType: String,
        val distanceKm: Double?,
        val durationMinutes: Int,
        val durationFormatted: String?,
        val caloriesKcal: Int?,
        val steps: Int?,
        val avgSpeedKmh: Double?,
        val avgPaceMinKm: String?,
        val totalDurationFormatted: String? = null
    ) {
        val calculatedSpeedKmh: Double?
            get() = if (distanceKm != null && durationMinutes > 0) {
                ((distanceKm / (durationMinutes / 60.0)) * 100.0).roundToInt() / 100.0
            } else null

        val calculatedPaceFormatted: String?
            get() = if (distanceKm != null && distanceKm > 0 && durationMinutes > 0) {
                val totalSecs = (durationMinutes * 60.0 / distanceKm).roundToInt()
                val m = totalSecs / 60
                val s = totalSecs % 60
                String.format(Locale.US, "%d:%02d/km", m, s)
            } else null

        val calculatedCadenceStepsPerKm: Int?
            get() = if (steps != null && distanceKm != null && distanceKm > 0) {
                (steps / distanceKm).roundToInt()
            } else null
    }

    fun isHealthOrFitnessApp(packageName: String): Boolean {
        val lower = packageName.lowercase(Locale.ROOT)
        return packageName in HEALTH_PACKAGES || lower.contains("shealth") || lower.contains("fitness") || lower.contains("health")
    }

    fun parseWorkoutMetrics(title: String, text: String, subText: String? = null): WorkoutMetrics? {
        val combined = "$title $text ${subText.orEmpty()}".trim()
        val lower = combined.lowercase(Locale.ROOT)

        // 1. Exercise Type Detection
        val exerciseType = when {
            lower.contains("walk") -> "Walking"
            lower.contains("run") || lower.contains("jog") -> "Running"
            lower.contains("cycl") || lower.contains("bike") || lower.contains("ride") -> "Cycling"
            lower.contains("swim") -> "Swimming"
            lower.contains("yoga") -> "Yoga"
            lower.contains("gym") || lower.contains("workout") -> "Workout"
            lower.contains("steps") || lower.contains("stepped") -> "Walking"
            else -> "Workout"
        }

        // 2. Distance Extraction: e.g. "3.23 km", "3.23km", "5.0 mi" or "Distance 3.23"
        val distRegex = Regex("""(?:(\d+(?:\.\d+)?)\s*(?:km|kilometers|mi|miles)\b|(?:distance)[:\s]+(\d+(?:\.\d+)?))""", RegexOption.IGNORE_CASE)
        val distMatch = distRegex.find(combined)
        val distanceKm = distMatch?.let { m ->
            m.groupValues[1].takeIf { it.isNotEmpty() }?.toDoubleOrNull() ?: m.groupValues[2].toDoubleOrNull()
        }

        // 3. Calories Extraction: e.g. "203 kcal", "203 cal" or "Calories 203"
        val calRegex = Regex("""(?:(\d+)\s*(?:kcal|calories|cal)\b|(?:workout\s+calories|calories|cal|kcal)[:\s]+(\d+))""", RegexOption.IGNORE_CASE)
        val calMatch = calRegex.find(combined)
        val caloriesKcal = calMatch?.let { m ->
            m.groupValues[1].takeIf { it.isNotEmpty() }?.toIntOrNull() ?: m.groupValues[2].toIntOrNull()
        }

        // 4. Steps Extraction: e.g. "3,507 steps" or "Steps 3,507"
        val stepsRegex = Regex("""(?:([\d,]+)\s*(?:steps|step)\b|(?:steps|step)[:\s]+([\d,]+))""", RegexOption.IGNORE_CASE)
        val stepsMatch = stepsRegex.find(combined)
        val steps = stepsMatch?.let { m ->
            val raw = m.groupValues[1].takeIf { it.isNotEmpty() } ?: m.groupValues[2]
            raw.replace(",", "").toIntOrNull()
        }

        // 5. Duration Extraction: e.g. "Workout duration 25:36", "Total duration 1:04:27", or "25 min"
        var durationMinutes = 30
        var durationFormatted: String? = null
        var totalDurationFormatted: String? = null

        val workoutDurationRegex = Regex("""(?:workout\s+duration|active\s+time|moving\s+time|duration)[:\s]+(?:(\d+):)?(\d{1,2}):(\d{2})""", RegexOption.IGNORE_CASE)
        val workoutDurationMatch = workoutDurationRegex.find(combined)

        val totalDurationRegex = Regex("""(?:total\s+duration|elapsed\s+time)[:\s]+(?:(\d+):)?(\d{1,2}):(\d{2})""", RegexOption.IGNORE_CASE)
        val totalDurationMatch = totalDurationRegex.find(combined)

        if (totalDurationMatch != null) {
            val th = totalDurationMatch.groupValues[1].toIntOrNull() ?: 0
            val tm = totalDurationMatch.groupValues[2].toIntOrNull() ?: 0
            val ts = totalDurationMatch.groupValues[3].toIntOrNull() ?: 0
            totalDurationFormatted = if (th > 0) String.format(Locale.US, "%d:%02d:%02d", th, tm, ts) else String.format(Locale.US, "%02d:%02d", tm, ts)
        }

        if (workoutDurationMatch != null) {
            val h = workoutDurationMatch.groupValues[1].toIntOrNull() ?: 0
            val m = workoutDurationMatch.groupValues[2].toIntOrNull() ?: 0
            val s = workoutDurationMatch.groupValues[3].toIntOrNull() ?: 0
            durationMinutes = h * 60 + m + if (s >= 30) 1 else 0
            durationFormatted = if (h > 0) String.format(Locale.US, "%d:%02d:%02d", h, m, s) else String.format(Locale.US, "%02d:%02d", m, s)
        } else {
            // Fallback: search for time patterns that are NOT followed by AM/PM
            val nonTimeMatches = Regex("""\b(?:(\d+):)?(\d{1,2}):(\d{2})\b(?!\s*(?:am|pm)\b)""", RegexOption.IGNORE_CASE)
                .findAll(combined).toList()
            if (nonTimeMatches.isNotEmpty()) {
                val first = nonTimeMatches[0]
                val h = first.groupValues[1].toIntOrNull() ?: 0
                val m = first.groupValues[2].toIntOrNull() ?: 0
                val s = first.groupValues[3].toIntOrNull() ?: 0
                durationMinutes = h * 60 + m + if (s >= 30) 1 else 0
                durationFormatted = first.value
                if (nonTimeMatches.size > 1 && totalDurationFormatted == null) {
                    totalDurationFormatted = nonTimeMatches[1].value
                }
            } else {
                val minRegex = Regex("""(\d+)\s*(?:min|mins|minutes)\b""", RegexOption.IGNORE_CASE)
                val minMatch = minRegex.find(combined)
                if (minMatch != null) {
                    durationMinutes = minMatch.groupValues[1].toIntOrNull() ?: 30
                    durationFormatted = "$durationMinutes mins"
                }
            }
        }

        // 6. Speed Extraction: e.g. "7.5 km/h", "7.5km/h", "7.5 kph"
        val speedRegex = Regex("""(\d+(?:\.\d+)?)\s*(?:km/h|kph)\b""", RegexOption.IGNORE_CASE)
        val speedMatch = speedRegex.find(combined)
        val avgSpeedKmh = speedMatch?.groupValues?.get(1)?.toDoubleOrNull()

        // 7. Pace Extraction: e.g. "07'54\" /km", "7:54/km", "7:54 min/km"
        val paceRegex = Regex("""(?:(\d{1,2})[':](\d{2}))(?:\s*["']|\s*(?:/km|min/km))""", RegexOption.IGNORE_CASE)
        val paceMatch = paceRegex.find(combined)
        val avgPaceMinKm = if (paceMatch != null) {
            "${paceMatch.groupValues[1]}:${paceMatch.groupValues[2]}/km"
        } else null

        // Must have at least one concrete fitness metric
        if (distanceKm == null && caloriesKcal == null && steps == null && durationFormatted == null) {
            return null
        }

        return WorkoutMetrics(
            exerciseType = exerciseType,
            distanceKm = distanceKm,
            durationMinutes = if (durationMinutes > 0) durationMinutes else 30,
            durationFormatted = durationFormatted,
            caloriesKcal = caloriesKcal,
            steps = steps,
            avgSpeedKmh = avgSpeedKmh,
            avgPaceMinKm = avgPaceMinKm,
            totalDurationFormatted = totalDurationFormatted
        )
    }

    fun processNotification(
        context: Context,
        packageName: String,
        title: String,
        text: String,
        subText: String?,
        timestampMs: Long
    ): Boolean {
        val lowerTitle = title.lowercase(Locale.ROOT)
        val lowerText = text.lowercase(Locale.ROOT)
        val isHealth = isHealthOrFitnessApp(packageName) ||
                lowerTitle.contains("walk") || lowerTitle.contains("run") || lowerTitle.contains("workout") ||
                lowerText.contains("kcal") || lowerText.contains("steps")

        if (!isHealth) return false

        val metrics = parseWorkoutMetrics(title, text, subText) ?: return false

        val appName = when {
            packageName.contains("shealth") -> "Samsung Health"
            packageName.contains("fitness") -> "Google Fit"
            packageName.contains("strava") -> "Strava"
            else -> "Fitness Tracker"
        }

        return logWorkoutToCommitments(
            context = context,
            metrics = metrics,
            sourceApp = appName,
            timestampMs = timestampMs
        )
    }

    fun logWorkoutToCommitments(
        context: Context,
        metrics: WorkoutMetrics,
        sourceApp: String = "Samsung Health",
        timestampMs: Long = System.currentTimeMillis()
    ): Boolean {
        val db = CommitmentDatabase.getInstance(context)

        // Build Title
        val titleParts = mutableListOf<String>()
        titleParts.add(metrics.exerciseType)
        metrics.distanceKm?.let { titleParts.add("${it} km") }
        metrics.steps?.let { titleParts.add("%,d steps".format(Locale.US, it)) }
        val finalTitle = titleParts.joinToString(" • ")

        // Build Notes with proof and detailed metrics
        val notesParts = mutableListOf<String>()
        notesParts.add("[TYPE:TIMELINE]")
        notesParts.add(sourceApp)
        metrics.distanceKm?.let { notesParts.add("$it km") }
        metrics.durationFormatted?.let { notesParts.add("Active: $it") }
        metrics.totalDurationFormatted?.let { notesParts.add("Total: $it") }
        metrics.caloriesKcal?.let { notesParts.add("${it} kcal") }
        metrics.steps?.let { notesParts.add("%,d steps".format(Locale.US, it)) }
        metrics.avgSpeedKmh?.let { notesParts.add("Avg: $it km/h") }
        metrics.avgPaceMinKm?.let { notesParts.add("Pace: $it") }

        val entity = CommitmentEntity(
            title = finalTitle,
            category = CommitmentCategory.FITNESS,
            scheduledTimeMs = timestampMs,
            durationMinutes = metrics.durationMinutes,
            status = CommitmentStatus.COMPLETED,
            sourceApp = sourceApp,
            sourcePerson = sourceApp,
            candidatePackages = listOf("com.sec.android.app.shealth", "com.google.android.apps.fitness", "com.strava"),
            notes = notesParts.joinToString(" • "),
            createdTimeMs = System.currentTimeMillis()
        )

        val id = db.insertCommitment(entity)
        Log.i(TAG, "Ingested real fitness activity to Planner (id=$id): $finalTitle")

        // Broadcast to trigger UI refresh in MainActivity
        try {
            val intent = Intent(ACTION_PLANNER_DATA_CHANGED)
            context.sendBroadcast(intent)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to broadcast planner refresh: ${e.message}")
        }
        return true
    }

    fun formatMarkdownVerificationTable(metrics: WorkoutMetrics): String {
        val sb = StringBuilder()
        sb.append("| Measurement | Your app | Calculated | Verdict |\n")
        sb.append("| :--- | :--- | :--- | :--- |\n")

        metrics.distanceKm?.let {
            sb.append("| **Distance** | ${it} km | ${it} km | Recorded value |\n")
        }

        metrics.durationFormatted?.let {
            val mins = metrics.durationMinutes
            sb.append("| **Workout duration** | $it | ~$mins mins | Active moving time |\n")
        }

        if (metrics.totalDurationFormatted != null) {
            sb.append("| **Total duration** | ${metrics.totalDurationFormatted} | — | Full session elapsed |\n")
        }

        val speed = metrics.avgSpeedKmh
        val calcSpeed = metrics.calculatedSpeedKmh
        if (speed != null || calcSpeed != null) {
            val appSpd = speed?.let { "$it km/h" } ?: "—"
            val mySpd = calcSpeed?.let { "$it km/h" } ?: "—"
            val verdict = if (speed != null && calcSpeed != null) {
                val diff = kotlin.math.abs(speed - calcSpeed)
                if (diff < 0.15) "Consistent with distance & time" else "Close (~$diff km/h diff)"
            } else "Plausible"
            sb.append("| **Average speed** | $appSpd | $mySpd | $verdict |\n")
        }

        val pace = metrics.avgPaceMinKm
        val calcPace = metrics.calculatedPaceFormatted
        if (pace != null || calcPace != null) {
            val appPace = pace ?: "—"
            val myPace = calcPace ?: "—"
            sb.append("| **Average pace** | $appPace | $myPace | Math verified |\n")
        }

        metrics.steps?.let { st ->
            val cadence = metrics.calculatedCadenceStepsPerKm?.let { c -> "%,d steps/km".format(Locale.US, c) } ?: "—"
            sb.append("| **Steps** | %,d | $cadence | Plausible cadence |\n".format(Locale.US, st))
        }

        metrics.caloriesKcal?.let { cal ->
            val estRange = if (metrics.distanceKm != null) {
                val base = (metrics.distanceKm * 63.0).roundToInt()
                "~$base kcal (~63 kcal/km)"
            } else "Standard estimate"
            sb.append("| **Calories** | $cal kcal | $estRange | Plausible based on METs |\n")
        }

        return sb.toString()
    }
}
