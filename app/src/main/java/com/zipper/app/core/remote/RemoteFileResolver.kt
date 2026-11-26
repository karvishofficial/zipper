package com.zipper.app.core.remote

import android.content.Context
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.webkit.MimeTypeMap
import java.io.File
import java.util.Locale

/**
 * Searches local Android device storage (Downloads, Documents, Internal/External Storage, MediaStore)
 * for files matching remote natural-language file requests (e.g. "12th marks card pdf", "karthikresume.pdf").
 */
object RemoteFileResolver {

    private const val TAG = "RemoteFileResolver"
    private const val MAX_CANDIDATES = 10
    private const val MAX_SEARCH_DEPTH = 3

    // Extensions that indicate file query
    private val KNOWN_EXTENSIONS = setOf(
        "pdf", "doc", "docx", "txt", "jpg", "jpeg", "png", "webp", "gif",
        "zip", "xlsx", "xls", "csv", "mp3", "mp4", "apk"
    )

    // Stop words to remove from search query
    private val STOP_WORDS = setOf(
        "send", "my", "the", "a", "an", "please", "file",
        "doc", "document", "to", "on", "whatsapp", "me", "in"
    )

    // Common compound sub-stems for intelligent decomposition (e.g. "karthikresume" -> "karthik", "resume")
    private val COMPOUND_STEMS = listOf(
        "resume", "marks", "card", "markscard", "hallticket", "admit", "ticket",
        "certificate", "invoice", "receipt", "screen", "wa", "karthik"
    )

    private val SYNONYMS = mapOf(
        "marks" to listOf("markscard", "marksheet", "scorecard", "result"),
        "admit" to listOf("hallticket", "hall_ticket", "admitcard"),
        "aadhaar" to listOf("aadhar", "uidai", "id_card"),
        "resume" to listOf("cv", "curriculum_vitae", "biodata"),
        "screenshot" to listOf("screen", "screencap", "screenshot_wa", "snap")
    )

    /**
     * Resolves matching files on the local filesystem and MediaStore based on a raw user query.
     */
    fun findMatchingFiles(context: Context, rawQuery: String): List<ResolvedFile> {
        val cleanQuery = rawQuery.trim().lowercase(Locale.ROOT)
        if (cleanQuery.isBlank()) return emptyList()

        // 1. Detect target extension if specified
        var targetExt: String? = null
        for (ext in KNOWN_EXTENSIONS) {
            if (cleanQuery.contains(".$ext") ||
                cleanQuery.contains(". $ext") ||
                cleanQuery.split(Regex("""\s+""")).contains(ext)
            ) {
                targetExt = ext
                break
            }
        }

        // 2. Extract significant keywords and decompose compound words
        val rawTokens = cleanQuery
            .replace(Regex("""\.\s*[a-zA-Z0-9]{2,5}\b"""), " ")
            .replace(Regex("""[.,\/#!$%\^&\*;:{}=\-_`~()?"']"""), " ")
            .split(Regex("""\s+"""))
            .filter { it.isNotBlank() }

        val searchKeywords = mutableListOf<String>()
        for (token in rawTokens) {
            if (token in STOP_WORDS) continue
            if (targetExt != null && token == targetExt) continue

            searchKeywords.add(token)

            // Synonym expansion
            SYNONYMS[token]?.let { syns -> searchKeywords.addAll(syns) }

            // Decompose compound tokens (e.g. "karthikresume" -> adds "karthik" and "resume")
            for (stem in COMPOUND_STEMS) {
                if (token.length > stem.length && token.contains(stem)) {
                    searchKeywords.add(stem)
                    val remainder = token.replace(stem, "").trim()
                    if (remainder.length >= 3 && remainder !in STOP_WORDS) {
                        searchKeywords.add(remainder)
                    }
                }
            }
        }

        val effectiveKeywords = (if (searchKeywords.isEmpty()) rawTokens else searchKeywords).distinct()
        Log.i(TAG, "Searching files with keywords: $effectiveKeywords, target extension: $targetExt")

        // 3. Search MediaStore first (super fast and indexed)
        val candidateFiles = mutableListOf<File>()
        candidateFiles.addAll(searchMediaStore(context, targetExt, effectiveKeywords))

        // 4. Search directory trees directly
        val directoriesToScan = getSearchDirectories(context)
        for (dir in directoriesToScan) {
            if (dir.exists() && dir.isDirectory) {
                scanDirectory(dir, candidateFiles, 0)
            }
        }

        val uniqueCandidates = candidateFiles.distinctBy { it.absolutePath }
        if (uniqueCandidates.isEmpty()) {
            Log.w(TAG, "No files found on storage during scan.")
            return emptyList()
        }

        // 5. Score and filter candidates
        val scoredList = uniqueCandidates.mapNotNull { file ->
            val fileNameLower = file.name.lowercase(Locale.ROOT)
            val fileExt = file.extension.lowercase(Locale.ROOT)

            // Extension filtering
            if (targetExt != null && fileExt != targetExt) {
                return@mapNotNull null
            }

            var score = 0
            var matchedKeywords = 0

            val cleanFileName = fileNameLower.substringBeforeLast(".").replace(Regex("""[^a-z0-9]"""), "")

            for (kw in effectiveKeywords) {
                val cleanKw = kw.replace(Regex("""[^a-z0-9]"""), "")
                if (cleanKw.isBlank()) continue

                if (fileNameLower.contains(kw) || cleanFileName.contains(cleanKw)) {
                    score += 15
                    matchedKeywords++
                } else if (cleanKw.length >= 4 && cleanKw.contains(cleanFileName)) {
                    score += 10
                    matchedKeywords++
                }
            }

            // Must match at least one keyword
            if (matchedKeywords == 0) {
                return@mapNotNull null
            }

            // Bonus if file name contains exact joined query
            val joined = effectiveKeywords.joinToString("")
            if (cleanFileName.contains(joined) || joined.contains(cleanFileName)) {
                score += 25
            }

            // Recency weighting
            val ageDays = (System.currentTimeMillis() - file.lastModified()) / (1000 * 60 * 60 * 24)
            if (ageDays < 30) score += 5
            if (ageDays < 7) score += 5

            Pair(file, score)
        }

        // 6. Sort by score descending, then by last modified descending
        val sorted = scoredList
            .sortedWith(compareByDescending<Pair<File, Int>> { it.second }.thenByDescending { it.first.lastModified() })
            .map { it.first }
            .distinctBy { it.absolutePath }
            .take(MAX_CANDIDATES)

        return sorted.map { file ->
            val mime = getMimeType(file)
            ResolvedFile(
                name = file.name,
                path = file.absolutePath,
                sizeBytes = file.length(),
                mimeType = mime,
                lastModified = file.lastModified()
            )
        }
    }

    private fun searchMediaStore(context: Context, targetExt: String?, effectiveKeywords: List<String>): List<File> {
        val found = mutableListOf<File>()
        try {
            val uri = MediaStore.Files.getContentUri("external")
            val projection = arrayOf(
                MediaStore.Files.FileColumns.DATA,
                MediaStore.Files.FileColumns.DISPLAY_NAME,
                MediaStore.Files.FileColumns.SIZE
            )

            val selectionSb = StringBuilder()
            val selectionArgs = mutableListOf<String>()

            if (targetExt != null) {
                selectionSb.append("(${MediaStore.Files.FileColumns.DATA} LIKE ?)")
                selectionArgs.add("%.${targetExt}")
            }

            val keywordClause = StringBuilder()
            for (kw in effectiveKeywords) {
                if (kw.length >= 3) {
                    if (keywordClause.isNotEmpty()) keywordClause.append(" OR ")
                    keywordClause.append("${MediaStore.Files.FileColumns.DISPLAY_NAME} LIKE ?")
                    selectionArgs.add("%$kw%")
                }
            }

            if (keywordClause.isNotEmpty()) {
                if (selectionSb.isNotEmpty()) selectionSb.append(" AND ")
                selectionSb.append("($keywordClause)")
            }

            val selection = if (selectionSb.isNotEmpty()) selectionSb.toString() else null
            val args = if (selectionArgs.isNotEmpty()) selectionArgs.toTypedArray() else null
            val sortOrder = "${MediaStore.Files.FileColumns.DATE_MODIFIED} DESC LIMIT 50"

            context.contentResolver.query(uri, projection, selection, args, sortOrder)?.use { cursor ->
                val dataIdx = cursor.getColumnIndex(MediaStore.Files.FileColumns.DATA)
                while (cursor.moveToNext()) {
                    if (dataIdx >= 0) {
                        val path = cursor.getString(dataIdx)
                        if (!path.isNullOrBlank()) {
                            val f = File(path)
                            if (f.exists() && f.isFile && f.length() > 0) {
                                found.add(f)
                            }
                        }
                    }
                }
            }
            Log.d(TAG, "MediaStore search located ${found.size} candidate files")
        } catch (e: Exception) {
            Log.w(TAG, "MediaStore search error: ${e.message}")
        }
        return found
    }

    private fun getSearchDirectories(context: Context): List<File> {
        val dirs = mutableListOf<File>()

        // 1. Standard Public Downloads
        try {
            val downloads = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
            if (downloads != null) dirs.add(downloads)
        } catch (_: Exception) {}

        // 2. Standard Public Documents
        try {
            val docs = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS)
            if (docs != null) dirs.add(docs)
        } catch (_: Exception) {}

        // 3. Primary External Storage Root and common directories
        try {
            val extRoot = Environment.getExternalStorageDirectory()
            if (extRoot != null) {
                dirs.add(extRoot) // includes root files like /storage/emulated/0/screen_wa.png!
                dirs.add(File(extRoot, "Download"))
                dirs.add(File(extRoot, "Documents"))
                dirs.add(File(extRoot, "Pictures/Screenshots"))
                dirs.add(File(extRoot, "DCIM/Screenshots"))
                dirs.add(File(extRoot, "Telegram/Telegram Documents"))
                dirs.add(File(extRoot, "CamScanner"))
                dirs.add(File(extRoot, "WhatsApp/Media/WhatsApp Documents"))
                dirs.add(File(extRoot, "Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Documents"))
                dirs.add(File(extRoot, "Android/media/org.telegram.messenger/Telegram/Telegram Documents"))
            }
        } catch (_: Exception) {}

        // 4. App External Files
        try {
            val appExt = context.getExternalFilesDir(null)
            if (appExt != null) dirs.add(appExt)
        } catch (_: Exception) {}

        return dirs.distinctBy { it.absolutePath }
    }

    private fun scanDirectory(dir: File, collected: MutableList<File>, currentDepth: Int) {
        if (currentDepth > MAX_SEARCH_DEPTH) return
        val files = dir.listFiles() ?: return

        for (f in files) {
            // Ignore hidden files
            if (f.name.startsWith(".")) continue
            if (f.isDirectory) {
                // Skip Android/data except for WhatsApp and Telegram media
                if (f.name.equals("Android", ignoreCase = true)) {
                    val waDocs = File(f, "media/com.whatsapp/WhatsApp/Media/WhatsApp Documents")
                    if (waDocs.exists()) scanDirectory(waDocs, collected, currentDepth + 1)
                    val tgDocs = File(f, "media/org.telegram.messenger/Telegram/Telegram Documents")
                    if (tgDocs.exists()) scanDirectory(tgDocs, collected, currentDepth + 1)
                    continue
                }
                scanDirectory(f, collected, currentDepth + 1)
            } else if (f.isFile && f.length() > 0) {
                collected.add(f)
            }
        }
    }

    fun formatFileSize(bytes: Long): String {
        return when {
            bytes < 1024 -> "$bytes B"
            bytes < 1024 * 1024 -> String.format(Locale.ROOT, "%.1f KB", bytes / 1024.0)
            else -> String.format(Locale.ROOT, "%.1f MB", bytes / (1024.0 * 1024.0))
        }
    }

    fun getMimeType(file: File): String {
        val ext = file.extension.lowercase(Locale.ROOT)
        val fromMap = MimeTypeMap.getSingleton().getMimeTypeFromExtension(ext)
        if (!fromMap.isNullOrBlank()) return fromMap

        return when (ext) {
            "pdf" -> "application/pdf"
            "doc", "docx" -> "application/msword"
            "xls", "xlsx" -> "application/vnd.ms-excel"
            "jpg", "jpeg" -> "image/jpeg"
            "png" -> "image/png"
            "webp" -> "image/webp"
            "gif" -> "image/gif"
            "zip" -> "application/zip"
            "txt" -> "text/plain"
            "apk" -> "application/vnd.android.package-archive"
            "mp4" -> "video/mp4"
            "mp3" -> "audio/mpeg"
            else -> "*/*"
        }
    }
}
