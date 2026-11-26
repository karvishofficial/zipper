package com.zipper.app.core.intelligence.resolver

import android.content.ContentUris
import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import androidx.core.content.FileProvider
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

data class DisambiguatedFile(
    val file: File?,
    val uri: Uri,
    val name: String,
    val sizeFormatted: String,
    val sizeBytes: Long,
    val lastModifiedFormatted: String,
    val lastModifiedMs: Long,
    val isMostRecent: Boolean = false,
    val mimeType: String = "application/pdf"
)

object DeviceAssetResolver {

    /**
     * Scans device storage and MediaStore for files matching a keyword (e.g. "resume", "cv", "pdf").
     * Returns structured, disambiguated candidate files sorted with the most recent first.
     */
    fun resolveMatchingFiles(
        context: Context,
        keyword: String = "",
        mimeType: String = "application/pdf"
    ): List<DisambiguatedFile> {
        val results = mutableListOf<DisambiguatedFile>()
        val seenNames = mutableSetOf<String>()
        val cleanKeyword = keyword.trim().lowercase(Locale.ROOT)

        // 1. Direct scan of standard public directories: Download & Documents
        val searchDirs = listOf(
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS),
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS),
            File(Environment.getExternalStorageDirectory(), "Download"),
            File(Environment.getExternalStorageDirectory(), "Documents")
        )

        for (dir in searchDirs) {
            if (dir.exists() && dir.isDirectory) {
                val files = dir.listFiles() ?: continue
                for (f in files) {
                    if (f.isFile && f.name.endsWith(".pdf", ignoreCase = true)) {
                        val nameLower = f.name.lowercase(Locale.ROOT)
                        val matchesKeyword = cleanKeyword.isBlank() ||
                            nameLower.contains(cleanKeyword) ||
                            (cleanKeyword.contains("resume") && (nameLower.contains("resume") || nameLower.contains("cv") || nameLower.contains("profile"))) ||
                            (cleanKeyword.contains("cv") && (nameLower.contains("cv") || nameLower.contains("resume")))

                        if (matchesKeyword && seenNames.add(f.name)) {
                            try {
                                val contentUri = FileProvider.getUriForFile(
                                    context,
                                    "${context.packageName}.fileprovider",
                                    f
                                )
                                results.add(
                                    DisambiguatedFile(
                                        file = f,
                                        uri = contentUri,
                                        name = f.name,
                                        sizeFormatted = formatFileSize(f.length()),
                                        sizeBytes = f.length(),
                                        lastModifiedFormatted = formatLastModified(f.lastModified()),
                                        lastModifiedMs = f.lastModified(),
                                        mimeType = "application/pdf"
                                    )
                                )
                            } catch (_: Exception) {
                                // Fallback Uri if FileProvider throws
                                val fileUri = Uri.fromFile(f)
                                results.add(
                                    DisambiguatedFile(
                                        file = f,
                                        uri = fileUri,
                                        name = f.name,
                                        sizeFormatted = formatFileSize(f.length()),
                                        sizeBytes = f.length(),
                                        lastModifiedFormatted = formatLastModified(f.lastModified()),
                                        lastModifiedMs = f.lastModified(),
                                        mimeType = "application/pdf"
                                    )
                                )
                            }
                        }
                    }
                }
            }
        }

        // 2. MediaStore query for Android 10+ scoped storage coverage
        try {
            val projection = arrayOf(
                MediaStore.Files.FileColumns._ID,
                MediaStore.Files.FileColumns.DISPLAY_NAME,
                MediaStore.Files.FileColumns.SIZE,
                MediaStore.Files.FileColumns.DATE_MODIFIED
            )

            val selection = "${MediaStore.Files.FileColumns.MIME_TYPE} = ?"
            val selectionArgs = arrayOf("application/pdf")
            val sortOrder = "${MediaStore.Files.FileColumns.DATE_MODIFIED} DESC"

            val queryUri = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                MediaStore.Files.getContentUri(MediaStore.VOLUME_EXTERNAL)
            } else {
                MediaStore.Files.getContentUri("external")
            }

            context.contentResolver.query(
                queryUri,
                projection,
                selection,
                selectionArgs,
                sortOrder
            )?.use { cursor ->
                val idCol = cursor.getColumnIndexOrThrow(MediaStore.Files.FileColumns._ID)
                val nameCol = cursor.getColumnIndexOrThrow(MediaStore.Files.FileColumns.DISPLAY_NAME)
                val sizeCol = cursor.getColumnIndexOrThrow(MediaStore.Files.FileColumns.SIZE)
                val dateCol = cursor.getColumnIndexOrThrow(MediaStore.Files.FileColumns.DATE_MODIFIED)

                while (cursor.moveToNext()) {
                    val id = cursor.getLong(idCol)
                    val name = cursor.getString(nameCol) ?: "document.pdf"
                    val size = cursor.getLong(sizeCol)
                    val dateSec = cursor.getLong(dateCol)
                    val dateMs = dateSec * 1000L

                    val nameLower = name.lowercase(Locale.ROOT)
                    val matchesKeyword = cleanKeyword.isBlank() ||
                        nameLower.contains(cleanKeyword) ||
                        (cleanKeyword.contains("resume") && (nameLower.contains("resume") || nameLower.contains("cv") || nameLower.contains("profile"))) ||
                        (cleanKeyword.contains("cv") && (nameLower.contains("cv") || nameLower.contains("resume")))

                    if (matchesKeyword && seenNames.add(name)) {
                        val contentUri = ContentUris.withAppendedId(queryUri, id)
                        results.add(
                            DisambiguatedFile(
                                file = null,
                                uri = contentUri,
                                name = name,
                                sizeFormatted = formatFileSize(size),
                                sizeBytes = size,
                                lastModifiedFormatted = formatLastModified(dateMs),
                                lastModifiedMs = dateMs,
                                mimeType = "application/pdf"
                            )
                        )
                    }
                }
            }
        } catch (_: Exception) {}

        // Sort descending by recency
        val sorted = results.sortedByDescending { it.lastModifiedMs }
        return sorted.mapIndexed { index, item ->
            item.copy(isMostRecent = (index == 0))
        }
    }

    fun formatFileSize(bytes: Long): String {
        if (bytes <= 0) return "0 KB"
        val kb = bytes / 1024.0
        val mb = kb / 1024.0
        return when {
            mb >= 1.0 -> String.format(Locale.US, "%.1f MB", mb)
            else -> String.format(Locale.US, "%.0f KB", kb)
        }
    }

    fun formatLastModified(ms: Long): String {
        if (ms <= 0L) return "Unknown date"
        val now = System.currentTimeMillis()
        val diffHours = (now - ms) / (1000 * 60 * 60)
        return when {
            diffHours < 24 -> {
                val timeStr = SimpleDateFormat("h:mm a", Locale.getDefault()).format(Date(ms))
                "Today, $timeStr"
            }
            diffHours < 48 -> "Yesterday"
            else -> SimpleDateFormat("MMM d, yyyy", Locale.getDefault()).format(Date(ms))
        }
    }
}
