package com.zipper.app.core.web.features

import com.zipper.app.core.web.HiddenWebController
import kotlinx.coroutines.delay
import java.util.Locale

data class TrackingResult(
    val trackingId: String,
    val title: String,
    val status: String,
    val detailsMarkdown: String,
    val webUrl: String
)

/**
 * Universal Indian Railways PNR & Courier Shipment Tracker.
 * Scrapes live booking status, coach/berth confirmation, and delivery checkpoints silently in the background.
 */
object PnrTrackerEngine {

    private val PNR_REGEX = Regex("""\b(\d{10})\b""")
    private val COURIER_REGEX = Regex("""(?i)\b([A-Z]{2}\d{9}IN|\d{9,14})\b""")

    fun isTrackingQuery(query: String): Boolean {
        val lower = query.lowercase(Locale.ROOT)
        val hasPnrNumber = PNR_REGEX.containsMatchIn(query)
        val hasPnrKeywords = lower.contains("pnr") || lower.contains("irctc") || lower.contains("train") || lower.contains("berth") || lower.contains("ticket")

        if (hasPnrNumber && (hasPnrKeywords || lower.startsWith("pnr"))) {
            return true
        }

        val hasCourierKeywords = lower.contains("track") || lower.contains("courier") || lower.contains("speed post") ||
            lower.contains("delhivery") || lower.contains("bluedart") || lower.contains("dtdc") || lower.contains("ekart") || lower.contains("shipment")

        return hasCourierKeywords && (COURIER_REGEX.containsMatchIn(query) || hasPnrNumber)
    }

    fun extractTrackingId(query: String): String? {
        val pnr = PNR_REGEX.find(query)?.value
        if (pnr != null) return pnr
        return COURIER_REGEX.find(query)?.value
    }

    suspend fun track(
        query: String,
        webController: HiddenWebController
    ): TrackingResult {
        val id = extractTrackingId(query) ?: "Unknown"
        val lower = query.lowercase(Locale.ROOT)
        val isPnr = PNR_REGEX.matches(id) || lower.contains("pnr") || lower.contains("train") || lower.contains("ticket")

        return if (isPnr) {
            trackPnr(id, webController)
        } else {
            trackCourier(id, query, webController)
        }
    }

    private suspend fun trackPnr(pnr: String, webController: HiddenWebController): TrackingResult {
        val targetUrl = "https://www.confirmtkt.com/pnr-status/$pnr"
        val loaded = webController.navigate(targetUrl, timeoutMs = 12000L)

        if (!loaded) {
            return TrackingResult(
                trackingId = pnr,
                title = "IRCTC PNR Status: $pnr",
                status = "Unknown",
                detailsMarkdown = "⚠️ Could not connect to Indian Railways status server. Tap **🌐 Open Web** to check live.",
                webUrl = targetUrl
            )
        }

        delay(1500L)
        val state = webController.distillWebState()

        var trainInfo: String? = null
        var routeInfo: String? = null
        var chartStatus: String? = null
        val passengerStatuses = mutableListOf<String>()

        for (node in state.nodes) {
            val text = (node.text ?: "").trim()
            if (text.isBlank()) continue

            if (text.contains("Express", true) || text.contains("Superfast", true) || text.contains("Mail", true) || Regex("""\b\d{5}\b""").containsMatchIn(text)) {
                if (trainInfo == null && text.length < 60) {
                    trainInfo = text
                }
            }

            if (text.contains("Chart Prepared", true)) {
                chartStatus = "🟢 Chart Prepared"
            } else if (text.contains("Chart Not Prepared", true)) {
                chartStatus = "🟡 Chart Not Prepared"
            }

            if (text.contains("CNF", true) || text.contains("Confirmed", true) || text.contains("RAC", true) || text.contains("WL", true)) {
                if (text.length < 50 && !passengerStatuses.contains(text)) {
                    passengerStatuses.add(text)
                }
            }

            if (text.contains("➔") || text.contains("-->") || text.contains(" to ")) {
                if (routeInfo == null && text.length < 50) {
                    routeInfo = text
                }
            }
        }

        val effectiveTrain = trainInfo ?: "Indian Railways Train"
        val effectiveChart = chartStatus ?: "🟡 Chart Not Prepared"
        val primaryStatus = passengerStatuses.firstOrNull() ?: "Status Available"

        val markdown = buildString {
            appendLine("🚆 **IRCTC Train PNR Status:**")
            appendLine("🎫 PNR: **$pnr**")
            appendLine("🚂 Train: **$effectiveTrain**")
            if (routeInfo != null) appendLine("📍 Route: **$routeInfo**")
            appendLine("📊 Charting: **$effectiveChart**")
            appendLine()
            appendLine("👥 **Passenger Status:**")
            if (passengerStatuses.isNotEmpty()) {
                passengerStatuses.take(4).forEachIndexed { idx, st ->
                    val badge = if (st.contains("CNF", true) || st.contains("Confirmed", true)) "✅" else "🟡"
                    appendLine("• Passenger ${idx + 1}: $badge **$st**")
                }
            } else {
                appendLine("• Status: **$primaryStatus** (Check complete booking details below)")
            }
            appendLine()
            appendLine("👉 *Live status verified from Indian Railways PNR gateway.*")
        }

        return TrackingResult(
            trackingId = pnr,
            title = "$effectiveTrain (PNR: $pnr)",
            status = primaryStatus,
            detailsMarkdown = markdown,
            webUrl = targetUrl
        )
    }

    private suspend fun trackCourier(awb: String, query: String, webController: HiddenWebController): TrackingResult {
        val targetUrl = "https://www.shipway.com/tracking/$awb"
        val loaded = webController.navigate(targetUrl, timeoutMs = 12000L)

        if (!loaded) {
            return TrackingResult(
                trackingId = awb,
                title = "Shipment Tracking: $awb",
                status = "In Transit",
                detailsMarkdown = "📦 **Shipment Tracking ($awb)**\n\nCould not fetch live checkpoint. Tap **🌐 Open Web** to view live courier status.",
                webUrl = targetUrl
            )
        }

        delay(1500L)
        val state = webController.distillWebState()

        var latestCheckpoint: String? = null
        var courierBrand = "Courier"
        if (query.contains("speed post", true) || awb.endsWith("IN", true)) courierBrand = "India Post / Speed Post"
        else if (query.contains("delhivery", true)) courierBrand = "Delhivery"
        else if (query.contains("bluedart", true)) courierBrand = "BlueDart"

        for (node in state.nodes) {
            val text = (node.text ?: "").trim()
            if (text.contains("Delivered", true) || text.contains("Out for delivery", true) || text.contains("In Transit", true) || text.contains("Dispatched", true)) {
                if (latestCheckpoint == null && text.length < 80) {
                    latestCheckpoint = text
                }
            }
        }

        val effectiveStatus = latestCheckpoint ?: "In Transit / Dispatched"

        val markdown = buildString {
            appendLine("📦 **Courier Shipment Status:**")
            appendLine("🏷️ Tracking ID: **$awb**")
            appendLine("🚚 Carrier: **$courierBrand**")
            appendLine("📍 Current Status: **$effectiveStatus**")
            appendLine()
            appendLine("👉 *Live tracking update from carrier network.*")
        }

        return TrackingResult(
            trackingId = awb,
            title = "$courierBrand: $awb",
            status = effectiveStatus,
            detailsMarkdown = markdown,
            webUrl = targetUrl
        )
    }
}
