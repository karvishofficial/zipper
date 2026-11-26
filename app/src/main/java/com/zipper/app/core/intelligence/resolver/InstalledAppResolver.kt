package com.zipper.app.core.intelligence.resolver

import android.content.Context
import android.content.Intent
import android.content.pm.ApplicationInfo
import android.content.pm.PackageManager
import android.graphics.drawable.Drawable
import android.net.Uri
import android.util.Log
import com.zipper.app.core.intelligence.store.CommitmentCategory

/**
 * Represents an installed Android application resolved as a candidate
 * for fulfilling or tracking a user commitment.
 */
data class ResolvedAppCandidate(
    val packageName: String,
    val appName: String,
    val icon: Drawable?,
    val launchIntent: Intent?,
    val isPrimary: Boolean = false,
    val deepLinkUri: String? = null
)

/**
 * Scans the user's device for real installed apps matching a commitment's
 * intent category or candidate package suggestions.
 * Extracts high-resolution application icons and launch intents.
 */
class InstalledAppResolver(private val context: Context) {

    companion object {
        private const val TAG = "InstalledAppResolver"

        private val KNOWN_FITNESS_PACKAGES = listOf(
            "com.sec.android.app.shealth",          // Samsung Health
            "fit.cure.android",                     // Cult.fit
            "com.google.android.apps.fitness",      // Google Fit
            "com.strava",                           // Strava
            "com.nike.plusgps",                     // Nike Run Club
            "com.nike.omega",                       // Nike Training Club
            "com.myfitnesspal.android",             // MyFitnessPal
            "com.fitbit.FitbitMobile",              // Fitbit
            "com.popularapp.sevenmins",             // 7 Minute Workout
            "homeworkout.homeworkouts.noequipment", // Home Workout
            "com.xiaomi.wearable",                  // Mi Fitness
            "com.huawei.health",                    // Huawei Health
            "com.oneplus.health"                    // OnePlus Health
        )

        private val KNOWN_EDUCATION_PACKAGES = listOf(
            "com.google.android.apps.classroom",        // Google Classroom
            "com.google.android.apps.docs.editors.docs",// Google Docs
            "com.google.android.apps.docs",             // Google Drive
            "com.google.android.keep",                  // Google Keep Notes
            "com.microsoft.office.word",                // Microsoft Word
            "com.microsoft.office.officehubrow",        // Microsoft 365 / Office
            "com.microsoft.office.onenote",             // Microsoft OneNote
            "com.notion.id",                            // Notion
            "com.canvas.student",                       // Canvas Student
            "com.adobe.reader",                         // Adobe Acrobat
            "com.google.android.apps.docs.editors.sheets", // Sheets
            "com.google.android.apps.docs.editors.slides"  // Slides
        )

        private val KNOWN_COMMUNICATION_PACKAGES = listOf(
            "com.whatsapp",
            "com.whatsapp.w4b",
            "org.telegram.messenger",
            "com.google.android.gm",
            "com.google.android.apps.messaging",
            "com.slack",
            "com.discord",
            "com.instagram.android"
        )
    }

    private val packageManager: PackageManager = context.packageManager

    /**
     * Resolves the best installed apps for a given commitment.
     * Returns a list of [ResolvedAppCandidate] with real icons and launch intents.
     */
    fun resolveCandidates(
        category: CommitmentCategory,
        suggestedPackages: List<String> = emptyList(),
        sourcePerson: String = "",
        maxResults: Int = 4
    ): List<ResolvedAppCandidate> {
        val candidates = mutableListOf<ResolvedAppCandidate>()
        val seenPackages = mutableSetOf<String>()

        // 1. Check suggested packages first
        for (pkg in suggestedPackages) {
            if (pkg.isBlank() || seenPackages.contains(pkg)) continue
            val resolved = resolvePackage(pkg, isPrimary = candidates.isEmpty())
            if (resolved != null) {
                candidates.add(resolved)
                seenPackages.add(pkg)
            }
        }

        // 2. Query category-matched apps
        val categoryPackages = when (category) {
            CommitmentCategory.FITNESS, CommitmentCategory.HEALTH -> KNOWN_FITNESS_PACKAGES
            CommitmentCategory.EDUCATION, CommitmentCategory.WORK -> KNOWN_EDUCATION_PACKAGES
            CommitmentCategory.COMMUNICATION -> KNOWN_COMMUNICATION_PACKAGES
            CommitmentCategory.GENERAL -> emptyList()
        }

        for (pkg in categoryPackages) {
            if (candidates.size >= maxResults) break
            if (seenPackages.contains(pkg)) continue
            val resolved = resolvePackage(pkg, isPrimary = candidates.isEmpty())
            if (resolved != null) {
                candidates.add(resolved)
                seenPackages.add(pkg)
            }
        }

        // 3. If still below target, search all installed apps by label keyword
        if (candidates.size < maxResults) {
            val keywords = when (category) {
                CommitmentCategory.FITNESS, CommitmentCategory.HEALTH -> listOf("fit", "health", "workout", "gym", "exercise", "run")
                CommitmentCategory.EDUCATION, CommitmentCategory.WORK -> listOf("doc", "classroom", "notes", "word", "office", "study", "pdf")
                CommitmentCategory.COMMUNICATION -> listOf("chat", "message", "mail", "telegram")
                CommitmentCategory.GENERAL -> listOf("task", "to-do", "reminder", "calendar")
            }
            val dynamicMatches = findInstalledAppsByKeywords(keywords, seenPackages)
            for (match in dynamicMatches) {
                if (candidates.size >= maxResults) break
                candidates.add(match)
                seenPackages.add(match.packageName)
            }
        }

        // 4. Fallback if none found: provide WhatsApp or standard browser if available
        if (candidates.isEmpty()) {
            val fallbacks = listOf("com.whatsapp", "com.google.android.keep", "com.android.chrome")
            for (pkg in fallbacks) {
                val resolved = resolvePackage(pkg, isPrimary = candidates.isEmpty())
                if (resolved != null && !seenPackages.contains(pkg)) {
                    candidates.add(resolved)
                    seenPackages.add(pkg)
                    break
                }
            }
        }

        return candidates
    }

    private fun resolvePackage(packageName: String, isPrimary: Boolean): ResolvedAppCandidate? {
        return try {
            val appInfo = packageManager.getApplicationInfo(packageName, 0)
            if (!appInfo.enabled) return null

            val launchIntent = packageManager.getLaunchIntentForPackage(packageName)?.apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            } ?: return null

            val label = packageManager.getApplicationLabel(appInfo).toString()
            val icon = packageManager.getApplicationIcon(appInfo)

            ResolvedAppCandidate(
                packageName = packageName,
                appName = label,
                icon = icon,
                launchIntent = launchIntent,
                isPrimary = isPrimary
            )
        } catch (_: PackageManager.NameNotFoundException) {
            null
        } catch (e: Exception) {
            Log.w(TAG, "Error resolving package $packageName: ${e.message}")
            null
        }
    }

    private fun findInstalledAppsByKeywords(
        keywords: List<String>,
        excludePackages: Set<String>
    ): List<ResolvedAppCandidate> {
        val matches = mutableListOf<ResolvedAppCandidate>()
        try {
            val installedApps = packageManager.getInstalledApplications(PackageManager.GET_META_DATA)
            for (appInfo in installedApps) {
                if (excludePackages.contains(appInfo.packageName)) continue
                // Exclude system non-launchable apps
                val launchIntent = packageManager.getLaunchIntentForPackage(appInfo.packageName) ?: continue

                val label = packageManager.getApplicationLabel(appInfo).toString().lowercase()
                val pkg = appInfo.packageName.lowercase()

                val isMatch = keywords.any { kw -> label.contains(kw) || pkg.contains(kw) }
                if (isMatch) {
                    val icon = packageManager.getApplicationIcon(appInfo)
                    matches.add(
                        ResolvedAppCandidate(
                            packageName = appInfo.packageName,
                            appName = packageManager.getApplicationLabel(appInfo).toString(),
                            icon = icon,
                            launchIntent = launchIntent.apply { addFlags(Intent.FLAG_ACTIVITY_NEW_TASK) },
                            isPrimary = false
                        )
                    )
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Error finding apps by keywords: ${e.message}")
        }
        return matches
    }
}
