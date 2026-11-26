package com.zipper.app.core.web.features

import com.zipper.app.core.model.DistilledUiState
import com.zipper.app.core.web.HiddenWebController
import kotlinx.coroutines.delay
import java.util.Locale

data class ArticleSummaryResult(
    val title: String,
    val sourceUrl: String,
    val executiveSummary: String,
    val keyTakeaways: List<String>,
    val fullMarkdown: String
)

/**
 * Ad-Free Article Reader and TL;DR Summarizer Engine.
 * Strips out cookie consent banners, paywall modals, and ad clutter,
 * extracting the core journalistic or technical content.
 */
object ArticleReaderEngine {

    fun isArticleQuery(query: String): Boolean {
        val lower = query.lowercase(Locale.ROOT)
        val hasUrl = query.contains("http://", ignoreCase = true) || query.contains("https://", ignoreCase = true)
        return (hasUrl && (lower.contains("summarize") || lower.contains("read") || lower.contains("article") || lower.contains("tldr") || lower.contains("explain"))) ||
            lower.startsWith("summarize http") || lower.startsWith("read http")
    }

    fun extractUrlFromQuery(query: String): String? {
        val urlRegex = Regex("""https?://[^\s"'>]+""")
        return urlRegex.find(query)?.value
    }

    suspend fun readAndSummarize(
        url: String,
        webController: HiddenWebController
    ): ArticleSummaryResult {
        val loaded = webController.navigate(url, timeoutMs = 12000L)
        if (!loaded) {
            return ArticleSummaryResult(
                title = "Error Loading Page",
                sourceUrl = url,
                executiveSummary = "Could not load webpage within timeout.",
                keyTakeaways = emptyList(),
                fullMarkdown = "⚠️ Failed to load article from $url"
            )
        }

        delay(1000L) // Settle DOM
        val state = webController.distillWebState()
        return summarizeDistilledState(state, url)
    }

    fun summarizeDistilledState(state: DistilledUiState, sourceUrl: String): ArticleSummaryResult {
        val articleTitle = state.activityName?.takeIf { !it.startsWith("http") && it.isNotBlank() } ?: "Web Article"

        // Filter out non-content boilerplate (buttons, cookie notices, banners)
        val boilerplateKeywords = setOf("cookie", "privacy policy", "terms of service", "subscribe to newsletter", "accept all cookies")
        val contentNodes = state.nodes.filter { node ->
            val text = (node.text ?: "").trim()
            val lower = text.lowercase()
            val isButton = node.role == "BUTTON" || (node.isClickable && text.length < 50)
            val isBoilerplate = boilerplateKeywords.any { lower.contains(it) }
            text.length >= 35 && !isButton && !isBoilerplate
        }

        val paragraphs = contentNodes.mapNotNull { it.text?.trim() }.distinct()
        val keyTakeaways = paragraphs.take(4).map { p ->
            val firstSentence = p.split(Regex("""(?<=[.!?])\s+""")).firstOrNull()?.trim() ?: p
            firstSentence.take(150)
        }

        val executiveSummary = if (paragraphs.isNotEmpty()) {
            paragraphs.first().take(300)
        } else {
            "Content extracted from $sourceUrl."
        }

        val markdown = buildString {
            appendLine("📰 **$articleTitle**")
            appendLine("🔗 Source: `$sourceUrl`")
            appendLine()
            appendLine("📝 **Executive Summary:**")
            appendLine(executiveSummary)
            appendLine()
            if (keyTakeaways.isNotEmpty()) {
                appendLine("💡 **Key Takeaways:**")
                for (t in keyTakeaways) {
                    appendLine("• $t")
                }
            }
        }

        return ArticleSummaryResult(
            title = articleTitle,
            sourceUrl = sourceUrl,
            executiveSummary = executiveSummary,
            keyTakeaways = keyTakeaways,
            fullMarkdown = markdown
        )
    }
}
