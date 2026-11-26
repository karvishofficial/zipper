package com.zipper.app.core.web.features

import java.net.URI
import java.util.Locale

enum class MediaType {
    VIDEO,
    AUDIO,
    PDF,
    DOCUMENT,
    STREAM_PLAYLIST
}

data class ExtractedMedia(
    val type: MediaType,
    val url: String,
    val title: String,
    val mimeType: String,
    val directDownloadable: Boolean = true
)

/**
 * Direct Media & Clean Document Extractor.
 * Detects underlying PDFs, streaming video (.m3u8), audio, and documents,
 * bypassing artificial countdown timers and ad redirect mazes.
 */
object WebMediaExtractor {

    private val VIDEO_EXTENSIONS = listOf(".mp4", ".webm", ".mkv", ".mov")
    private val AUDIO_EXTENSIONS = listOf(".mp3", ".wav", ".aac", ".ogg", ".m4a")
    private val STREAM_EXTENSIONS = listOf(".m3u8", ".mpd")
    private val PDF_EXTENSIONS = listOf(".pdf")
    private val DOC_EXTENSIONS = listOf(".epub", ".docx", ".xlsx", ".pptx", ".zip")

    fun resolveAbsoluteUrl(basePageUrl: String, relativeOrAbsolute: String): String {
        return runCatching {
            val baseUri = URI(basePageUrl)
            baseUri.resolve(relativeOrAbsolute).toString()
        }.getOrDefault(relativeOrAbsolute)
    }

    fun extractMediaFromPage(
        pageHtml: String,
        pageUrl: String,
        interceptedNetworkUrls: List<String> = emptyList()
    ): List<ExtractedMedia> {
        val results = mutableListOf<ExtractedMedia>()
        val seenUrls = mutableSetOf<String>()

        // 1. Check intercepted network requests first (highest fidelity)
        for (req in interceptedNetworkUrls) {
            val lower = req.lowercase(Locale.ROOT)
            when {
                STREAM_EXTENSIONS.any { lower.contains(it) } -> {
                    if (seenUrls.add(req)) {
                        results.add(
                            ExtractedMedia(
                                type = MediaType.STREAM_PLAYLIST,
                                url = req,
                                title = extractFileNameFromUrl(req),
                                mimeType = "application/x-mpegURL"
                            )
                        )
                    }
                }
                VIDEO_EXTENSIONS.any { lower.contains(it) } -> {
                    if (seenUrls.add(req)) {
                        results.add(
                            ExtractedMedia(
                                type = MediaType.VIDEO,
                                url = req,
                                title = extractFileNameFromUrl(req),
                                mimeType = "video/mp4"
                            )
                        )
                    }
                }
                AUDIO_EXTENSIONS.any { lower.contains(it) } -> {
                    if (seenUrls.add(req)) {
                        results.add(
                            ExtractedMedia(
                                type = MediaType.AUDIO,
                                url = req,
                                title = extractFileNameFromUrl(req),
                                mimeType = "audio/mpeg"
                            )
                        )
                    }
                }
                PDF_EXTENSIONS.any { lower.contains(it) } -> {
                    if (seenUrls.add(req)) {
                        results.add(
                            ExtractedMedia(
                                type = MediaType.PDF,
                                url = req,
                                title = extractFileNameFromUrl(req),
                                mimeType = "application/pdf"
                            )
                        )
                    }
                }
            }
        }

        // 2. Scan HTML anchor and source tags
        val linkRegex = Regex("""(?:href|src)=["']([^"']+)["']""", RegexOption.IGNORE_CASE)
        val matches = linkRegex.findAll(pageHtml)
        for (m in matches) {
            val raw = m.groupValues[1].trim()
            val lower = raw.lowercase(Locale.ROOT)
            val fullUrl = resolveAbsoluteUrl(pageUrl, raw)

            if (!seenUrls.contains(fullUrl) && fullUrl.startsWith("http", ignoreCase = true)) {
                when {
                    PDF_EXTENSIONS.any { lower.contains(it) } -> {
                        seenUrls.add(fullUrl)
                        results.add(
                            ExtractedMedia(
                                type = MediaType.PDF,
                                url = fullUrl,
                                title = extractFileNameFromUrl(fullUrl),
                                mimeType = "application/pdf"
                            )
                        )
                    }
                    VIDEO_EXTENSIONS.any { lower.contains(it) } -> {
                        seenUrls.add(fullUrl)
                        results.add(
                            ExtractedMedia(
                                type = MediaType.VIDEO,
                                url = fullUrl,
                                title = extractFileNameFromUrl(fullUrl),
                                mimeType = "video/mp4"
                            )
                        )
                    }
                    AUDIO_EXTENSIONS.any { lower.contains(it) } -> {
                        seenUrls.add(fullUrl)
                        results.add(
                            ExtractedMedia(
                                type = MediaType.AUDIO,
                                url = fullUrl,
                                title = extractFileNameFromUrl(fullUrl),
                                mimeType = "audio/mpeg"
                            )
                        )
                    }
                    DOC_EXTENSIONS.any { lower.contains(it) } -> {
                        seenUrls.add(fullUrl)
                        results.add(
                            ExtractedMedia(
                                type = MediaType.DOCUMENT,
                                url = fullUrl,
                                title = extractFileNameFromUrl(fullUrl),
                                mimeType = "application/octet-stream"
                            )
                        )
                    }
                }
            }
        }

        return results
    }

    fun extractFileNameFromUrl(url: String): String {
        return runCatching {
            val clean = url.substringBefore('?').substringBefore('#')
            val name = clean.substringAfterLast('/')
            if (name.isNotBlank()) name else "downloaded_media"
        }.getOrDefault("media_file")
    }
}
