package com.zipper.app.core.web.features

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.Locale
import java.util.UUID

data class VaultSnapshot(
    val id: String = UUID.randomUUID().toString(),
    val title: String,
    val sourceUrl: String,
    val timestamp: Long = System.currentTimeMillis(),
    val executiveSummary: String,
    val keyPoints: List<String>,
    val fullMarkdown: String
)

/**
 * Offline Web Snapshot & Knowledge Vault.
 * Inlines web content into durable offline markdown documents with
 * key-takeaways summaries, stored locally in internal storage.
 */
object WebSnapshotVault {

    private const val VAULT_DIR = "web_knowledge_vault"

    fun convertHtmlToCleanMarkdown(htmlText: String, title: String, sourceUrl: String): String {
        // Strip scripts, styles, svg, and unwanted elements
        var clean = htmlText
            .replace(Regex("""<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>""", RegexOption.IGNORE_CASE), "")
            .replace(Regex("""<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>""", RegexOption.IGNORE_CASE), "")
            .replace(Regex("""<svg\b[^<]*(?:(?!<\/svg>)<[^<]*)*<\/svg>""", RegexOption.IGNORE_CASE), "")

        // Convert common markdown elements
        clean = clean.replace(Regex("""<h1[^>]*>(.*?)<\/h1>""", RegexOption.IGNORE_CASE), "\n# $1\n")
        clean = clean.replace(Regex("""<h2[^>]*>(.*?)<\/h2>""", RegexOption.IGNORE_CASE), "\n## $1\n")
        clean = clean.replace(Regex("""<h3[^>]*>(.*?)<\/h3>""", RegexOption.IGNORE_CASE), "\n### $1\n")
        clean = clean.replace(Regex("""<p[^>]*>(.*?)<\/p>""", RegexOption.IGNORE_CASE), "\n$1\n")
        clean = clean.replace(Regex("""<li[^>]*>(.*?)<\/li>""", RegexOption.IGNORE_CASE), "\n* $1")
        clean = clean.replace(Regex("""<br\s*\/?>""", RegexOption.IGNORE_CASE), "\n")
        clean = clean.replace(Regex("""<strong[^>]*>(.*?)<\/strong>""", RegexOption.IGNORE_CASE), "**$1**")
        clean = clean.replace(Regex("""<b[^>]*>(.*?)<\/b>""", RegexOption.IGNORE_CASE), "**$1**")

        // Strip remaining HTML tags
        clean = clean.replace(Regex("""<[^>]+>"""), " ")
        clean = clean.replace("&nbsp;", " ")
            .replace("&amp;", "&")
            .replace("&quot;", "\"")
            .replace("&lt;", "<")
            .replace("&gt;", ">")

        // Normalize blank lines
        clean = clean.lines()
            .map { it.trim() }
            .filter { it.isNotBlank() }
            .joinToString("\n\n")

        val sb = StringBuilder()
        sb.appendLine("# $title")
        sb.appendLine("**Source:** $sourceUrl")
        sb.appendLine("---")
        sb.appendLine(clean)
        return sb.toString()
    }

    fun generateExecutiveSummary(markdown: String): Pair<String, List<String>> {
        val lines = markdown.lines()
            .map { it.trim() }
            .filter { it.isNotBlank() && !it.startsWith("#") && !it.startsWith("**Source:**") && !it.startsWith("---") }

        val firstFew = lines.take(5).joinToString(" ")
        val execSummary = if (firstFew.length > 250) firstFew.take(247) + "..." else firstFew.ifBlank { "Offline web document snapshot." }

        val keyPoints = lines.drop(2)
            .filter { it.length in 25..150 }
            .take(3)
            .map { it.trimStart('*', '-', ' ') }

        return Pair(execSummary, keyPoints.ifEmpty { listOf("Document saved for offline access.") })
    }

    fun saveSnapshot(
        context: Context?,
        title: String,
        sourceUrl: String,
        rawHtmlOrMarkdown: String
    ): VaultSnapshot {
        val isHtml = rawHtmlOrMarkdown.contains("<html") || rawHtmlOrMarkdown.contains("<body") || rawHtmlOrMarkdown.contains("<p>")
        val markdown = if (isHtml) convertHtmlToCleanMarkdown(rawHtmlOrMarkdown, title, sourceUrl) else rawHtmlOrMarkdown
        val (execSummary, keyPoints) = generateExecutiveSummary(markdown)

        val snapshot = VaultSnapshot(
            title = title,
            sourceUrl = sourceUrl,
            executiveSummary = execSummary,
            keyPoints = keyPoints,
            fullMarkdown = markdown
        )

        if (context != null) {
            runCatching {
                val dir = File(context.filesDir, VAULT_DIR)
                if (!dir.exists()) dir.mkdirs()

                val file = File(dir, "${snapshot.id}.md")
                file.writeText(markdown)

                val metaFile = File(dir, "${snapshot.id}.meta.json")
                val json = JSONObject().apply {
                    put("id", snapshot.id)
                    put("title", snapshot.title)
                    put("sourceUrl", snapshot.sourceUrl)
                    put("timestamp", snapshot.timestamp)
                    put("executiveSummary", snapshot.executiveSummary)
                    put("keyPoints", JSONArray(snapshot.keyPoints))
                }
                metaFile.writeText(json.toString())
            }
        }

        return snapshot
    }

    fun listSnapshots(context: Context?): List<VaultSnapshot> {
        if (context == null) return emptyList()
        val dir = File(context.filesDir, VAULT_DIR)
        if (!dir.exists()) return emptyList()

        val results = mutableListOf<VaultSnapshot>()
        val metaFiles = dir.listFiles { f -> f.name.endsWith(".meta.json") } ?: return emptyList()

        for (mf in metaFiles) {
            runCatching {
                val json = JSONObject(mf.readText())
                val id = json.getString("id")
                val mdFile = File(dir, "$id.md")
                val md = if (mdFile.exists()) mdFile.readText() else ""
                val kpList = mutableListOf<String>()
                val kpArr = json.optJSONArray("keyPoints")
                if (kpArr != null) {
                    for (i in 0 until kpArr.length()) kpList.add(kpArr.getString(i))
                }

                results.add(
                    VaultSnapshot(
                        id = id,
                        title = json.getString("title"),
                        sourceUrl = json.getString("sourceUrl"),
                        timestamp = json.getLong("timestamp"),
                        executiveSummary = json.getString("executiveSummary"),
                        keyPoints = kpList,
                        fullMarkdown = md
                    )
                )
            }
        }
        return results.sortedByDescending { it.timestamp }
    }
}
