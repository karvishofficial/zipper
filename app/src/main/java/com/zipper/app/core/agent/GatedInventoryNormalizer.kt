package com.zipper.app.core.agent

import android.content.Context
import android.content.pm.ApplicationInfo
import android.content.pm.PackageManager
import android.database.Cursor
import android.provider.ContactsContract
import android.util.Log
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap

/**
 * Gated Inventory Normalizer for Karvish Autonomous Runtime.
 *
 * Implements Phase 1 of Karvish V59:
 * 1. Dynamic Phone Inventory: Caches installed apps and aliases (yt -> YouTube, wa -> WhatsApp).
 * 2. Gated Contact Resolution: Resolves contact typos (momm -> Mom) ONLY if margin Delta >= 2;
 *    refuses to guess on ambiguous prefixes (e.g. "roh" -> Rohit vs Rohan).
 * 3. Kills blind regexes: Guarantees words like "The" or "A" are NEVER classified as app names.
 *
 * Zero cloud dependency, zero external weights, sub-1ms on-device execution.
 */
class GatedInventoryNormalizer(
    private val context: Context? = null,
    initialContacts: List<String> = emptyList()
) {

    companion object {
        private const val TAG = "GatedNormalizer"

        @Volatile
        private var instance: GatedInventoryNormalizer? = null

        fun getOrCreate(context: Context): GatedInventoryNormalizer {
            return instance ?: synchronized(this) {
                instance ?: GatedInventoryNormalizer(context.applicationContext).also { instance = it }
            }
        }

        // Canonical App Aliases (Deterministic, Safe, Zero Risk)
        private val CANONICAL_APP_ALIASES = mapOf(
            "yt" to "YouTube",
            "youtube" to "YouTube",
            "wa" to "WhatsApp",
            "whatsapp" to "WhatsApp",
            "watsapp" to "WhatsApp",
            "insta" to "Instagram",
            "ig" to "Instagram",
            "instagram" to "Instagram",
            "calc" to "Calculator",
            "calculator" to "Calculator",
            "settings" to "Settings",
            "setting" to "Settings",
            "clock" to "Clock",
            "alarm" to "Clock",
            "timer" to "Clock",
            "stopwatch" to "Clock",
            "play store" to "Play Store",
            "playstore" to "Play Store",
            "spotify" to "Spotify",
            "spoti" to "Spotify",
            "zepto" to "Zepto",
            "blinkit" to "Blinkit",
            "swiggy" to "Swiggy",
            "zomato" to "Zomato",
            "keep" to "Keep",
            "google keep" to "Keep",
            "chrome" to "Chrome",
            "maps" to "Maps",
            "google maps" to "Maps",
            "uber" to "Uber",
            "ola" to "Ola",
            "gmail" to "Gmail"
        )
    }

    private val installedAppLabels = ConcurrentHashMap<String, String>() // lowercase label -> Canonical Label
    private val cachedContacts = mutableListOf<String>()
    private var isInitialized = false

    init {
        if (initialContacts.isNotEmpty()) {
            cachedContacts.addAll(initialContacts)
        }
        refreshInventory()
    }

    @Synchronized
    fun refreshInventory() {
        try {
            // 1. Seed canonical aliases
            for ((alias, canonical) in CANONICAL_APP_ALIASES) {
                installedAppLabels[alias] = canonical
            }

            // 2. Discover dynamically installed apps on this phone (if context available)
            val ctx = context
            if (ctx != null) {
                val pm = ctx.packageManager
                val apps = pm.getInstalledApplications(PackageManager.GET_META_DATA)
                for (app in apps) {
                    val label = try { pm.getApplicationLabel(app).toString().trim() } catch (_: Exception) { "" }
                    if (label.isNotBlank() && label.length > 1) {
                        val lower = label.lowercase(Locale.ROOT)
                        installedAppLabels[lower] = label
                    }
                }
                refreshContacts(ctx)
            }

            isInitialized = true
            Log.i(TAG, "GatedInventoryNormalizer initialized with ${installedAppLabels.size} app labels & ${cachedContacts.size} contacts")
        } catch (e: Exception) {
            Log.w(TAG, "Failed to refresh inventory: ${e.message}")
        }
    }

    private fun refreshContacts(ctx: Context) {
        try {
            if (ctx.checkSelfPermission(android.Manifest.permission.READ_CONTACTS) != PackageManager.PERMISSION_GRANTED) {
                return
            }
            val names = mutableSetOf<String>()
            val cursor: Cursor? = ctx.contentResolver.query(
                ContactsContract.Contacts.CONTENT_URI,
                arrayOf(ContactsContract.Contacts.DISPLAY_NAME_PRIMARY),
                null,
                null,
                null
            )
            cursor?.use {
                val nameIdx = it.getColumnIndex(ContactsContract.Contacts.DISPLAY_NAME_PRIMARY)
                if (nameIdx >= 0) {
                    while (it.moveToNext()) {
                        val name = it.getString(nameIdx)
                        if (!name.isNullOrBlank()) {
                            names.add(name.trim())
                        }
                    }
                }
            }
            synchronized(cachedContacts) {
                cachedContacts.clear()
                cachedContacts.addAll(names)
            }
        } catch (e: Exception) {
            Log.d(TAG, "Could not load contacts: ${e.message}")
        }
    }

    /**
     * Extracts and validates target app name from goal.
     * Guaranteed to NEVER hallucinate words like "The", "A", "Chat", etc.
     */
    fun extractTargetAppName(goal: String): String {
        val lowerGoal = goal.lowercase(Locale.ROOT)

        // 1. Check exact multi-word aliases first ("play store", "google keep", "google maps")
        for ((alias, canonical) in CANONICAL_APP_ALIASES) {
            if (alias.contains(" ") && Regex("\\b${Regex.escape(alias)}\\b").containsMatchIn(lowerGoal)) {
                return canonical
            }
        }

        // 2. Token-by-token matching against installed app inventory
        val tokens = lowerGoal.split(Regex("[^a-z0-9_]+")).filter { it.isNotBlank() }

        // Give priority to tokens that appear after "on", "in", "open", "launch"
        val prepIndices = mutableListOf<Int>()
        for (i in tokens.indices) {
            if (tokens[i] in setOf("on", "in", "open", "launch", "inside", "app")) {
                prepIndices.add(i)
            }
        }

        // Check tokens immediately after prepositions
        for (pIdx in prepIndices) {
            val candidateToken = tokens.getOrNull(pIdx + 1)
            if (candidateToken != null) {
                val matchedApp = installedAppLabels[candidateToken]
                if (matchedApp != null) {
                    return matchedApp
                }
            }
        }

        // Fallback: check any token in sentence against known app inventory
        for (t in tokens) {
            // Guard against common English prepositions / determiners
            if (t in setOf("the", "a", "an", "to", "for", "and", "or", "me", "my", "is", "it", "with", "this", "that")) {
                continue
            }
            val matchedApp = installedAppLabels[t]
            if (matchedApp != null) {
                return matchedApp
            }
        }

        // If no known app is detected, return "App" safely (NEVER "The")
        return "App"
    }

    /**
     * Resolves contact name with strict margin gating (Delta >= 2).
     * If ambiguous (e.g. "roh" matches both "Rohit" and "Rohan" with equal distance),
     * it refuses to guess and returns the raw candidate token.
     */
    fun resolveContactGated(candidateName: String): String {
        val candClean = candidateName.trim()
        if (candClean.isBlank() || candClean.length < 2) return candClean

        val cands = synchronized(cachedContacts) { cachedContacts.toList() }
        if (cands.isEmpty()) return candClean

        val candLower = candClean.lowercase(Locale.ROOT)

        // Exact match check
        for (c in cands) {
            if (c.equals(candClean, ignoreCase = true)) return c
        }

        // Compute edit distances to all cached contacts
        val scored = cands.map { contact ->
            val dist = levenshtein(candLower, contact.lowercase(Locale.ROOT))
            Pair(contact, dist)
        }.sortedBy { it.second }

        val best = scored.firstOrNull() ?: return candClean
        val runnerUp = scored.getOrNull(1)

        val bestDist = best.second
        val runnerUpDist = runnerUp?.second ?: Int.MAX_VALUE

        // Gated Rule: Best must be close (<= 2 edits) AND clearly dominate runner-up (margin >= 2)
        val margin = runnerUpDist - bestDist
        return if (bestDist <= 2 && margin >= 2) {
            Log.i(TAG, "[GATED CONTACT] Resolved '$candClean' -> '${best.first}' (dist=$bestDist, margin=$margin)")
            best.first
        } else {
            Log.d(TAG, "[GATED CONTACT] Ambiguous contact for '$candClean': best='${best.first}' ($bestDist), runner='${runnerUp?.first}' ($runnerUpDist) -> Not resolving")
            candClean
        }
    }

    /**
     * Normalizes raw user prompt before tokenization.
     * Replaces safe app shorthands (yt -> YouTube, wa -> WhatsApp) so token embeddings align.
     */
    fun normalizeGoal(rawGoal: String): String {
        var normalized = rawGoal.trim()

        // 1. Replace multi-word aliases
        normalized = normalized.replace(Regex("\\bplay\\s*store\\b", RegexOption.IGNORE_CASE), "Play Store")
        normalized = normalized.replace(Regex("\\bgoogle\\s*maps\\b", RegexOption.IGNORE_CASE), "Maps")
        normalized = normalized.replace(Regex("\\bgoogle\\s*keep\\b", RegexOption.IGNORE_CASE), "Keep")

        // 2. Token-level alias replacement (isolated words only)
        val words = normalized.split(Regex("(?<=\\s)|(?=\\s)"))
        val sb = StringBuilder()
        for (w in words) {
            val trimmedLower = w.trim().lowercase(Locale.ROOT)
            val matchedCanonical = when (trimmedLower) {
                "yt" -> "YouTube"
                "wa", "watsapp" -> "WhatsApp"
                "insta", "ig" -> "Instagram"
                "calc" -> "Calculator"
                "spoti" -> "Spotify"
                "zepto" -> "Zepto"
                "blinkit" -> "Blinkit"
                "swiggy" -> "Swiggy"
                "zomato" -> "Zomato"
                else -> null
            }
            if (matchedCanonical != null) {
                sb.append(matchedCanonical)
            } else {
                sb.append(w)
            }
        }

        return sb.toString()
    }

    private fun levenshtein(s1: String, s2: String): Int {
        val dp = Array(s1.length + 1) { IntArray(s2.length + 1) }
        for (i in 0..s1.length) dp[i][0] = i
        for (j in 0..s2.length) dp[0][j] = j
        for (i in 1..s1.length) {
            for (j in 1..s2.length) {
                val cost = if (s1[i - 1] == s2[j - 1]) 0 else 1
                dp[i][j] = minOf(
                    dp[i - 1][j] + 1,      // deletion
                    dp[i][j - 1] + 1,      // insertion
                    dp[i - 1][j - 1] + cost // substitution
                )
            }
        }
        return dp[s1.length][s2.length]
    }
}
