package com.zipper.app.core.web.features

import org.json.JSONArray
import org.json.JSONObject
import java.util.Locale

data class FareQuery(
    val origin: String,
    val destination: String,
    val date: String,
    val travelClass: String = "Economy"
)

data class FareItem(
    val provider: String,
    val operatorOrAirline: String,
    val flightOrTrainNumber: String,
    val departureTime: String,
    val arrivalTime: String,
    val price: Double,
    val bookingUrl: String,
    val isCheapest: Boolean = false
)

data class FareMatrixResult(
    val query: FareQuery,
    val fares: List<FareItem>,
    val lowestPrice: Double,
    val highestPrice: Double,
    val totalSavings: Double
)

/**
 * Multi-Store Fare & Price Matrix Scanner.
 * Collects, aligns, and ranks travel and ticket options across multiple portals
 * to instantly surface the absolute cheapest departure.
 */
object WebFareMatrixScanner {

    fun parseFaresFromJson(jsonArrayStr: String, defaultProvider: String): List<FareItem> {
        val result = mutableListOf<FareItem>()
        runCatching {
            val arr = JSONArray(jsonArrayStr)
            for (i in 0 until arr.length()) {
                val obj = arr.getJSONObject(i)
                val price = obj.optDouble("price", 0.0)
                if (price > 0) {
                    result.add(
                        FareItem(
                            provider = obj.optString("provider", defaultProvider),
                            operatorOrAirline = obj.optString("airline", obj.optString("operator", "Carrier")),
                            flightOrTrainNumber = obj.optString("number", "Direct"),
                            departureTime = obj.optString("dept", "--:--"),
                            arrivalTime = obj.optString("arr", "--:--"),
                            price = price,
                            bookingUrl = obj.optString("url", "")
                        )
                    )
                }
            }
        }
        return result
    }

    fun parseFaresFromHtml(html: String, provider: String): List<FareItem> {
        val items = mutableListOf<FareItem>()
        val priceRegex = Regex("""(?:₹|Rs\.?|INR)\s*([0-9,]+(?:\.[0-9]+)?)""")
        val matches = priceRegex.findAll(html)
        var count = 1

        for (match in matches) {
            val rawPrice = match.groupValues[1].replace(",", "")
            val priceVal = rawPrice.toDoubleOrNull() ?: continue
            if (priceVal in 500.0..50000.0) { // realistic domestic ticket range
                items.add(
                    FareItem(
                        provider = provider,
                        operatorOrAirline = "Option #$count",
                        flightOrTrainNumber = "FL-$count",
                        departureTime = "Morning",
                        arrivalTime = "Afternoon",
                        price = priceVal,
                        bookingUrl = ""
                    )
                )
                count++
                if (count > 8) break
            }
        }
        return items
    }

    fun buildMatrix(rawFares: List<FareItem>, query: FareQuery): FareMatrixResult {
        if (rawFares.isEmpty()) {
            return FareMatrixResult(
                query = query,
                fares = emptyList(),
                lowestPrice = 0.0,
                highestPrice = 0.0,
                totalSavings = 0.0
            )
        }

        val sorted = rawFares.sortedBy { it.price }
        val lowest = sorted.first().price
        val highest = sorted.last().price
        val savings = (highest - lowest).coerceAtLeast(0.0)

        val annotated = sorted.mapIndexed { idx, item ->
            item.copy(isCheapest = idx == 0)
        }

        return FareMatrixResult(
            query = query,
            fares = annotated,
            lowestPrice = lowest,
            highestPrice = highest,
            totalSavings = savings
        )
    }

    fun formatMatrixMarkdown(result: FareMatrixResult): String {
        if (result.fares.isEmpty()) {
            return "No fare options found for ${result.query.origin} ➔ ${result.query.destination} on ${result.query.date}."
        }

        val sb = StringBuilder()
        sb.appendLine("### ✈️ Travel Fare Matrix: ${result.query.origin} ➔ ${result.query.destination}")
        sb.appendLine("**Date:** ${result.query.date} | **Lowest Fare:** ₹${String.format(Locale.ROOT, "%.2f", result.lowestPrice)} | **Max Savings:** ₹${String.format(Locale.ROOT, "%.2f", result.totalSavings)}")
        sb.appendLine()
        sb.appendLine("| Operator | Time | Provider | Price | Best Value |")
        sb.appendLine("|---|---|---|---|---|")
        for (f in result.fares) {
            val star = if (f.isCheapest) "🏆 **CHEAPEST**" else ""
            sb.appendLine("| ${f.operatorOrAirline} (${f.flightOrTrainNumber}) | ${f.departureTime} - ${f.arrivalTime} | ${f.provider} | ₹${String.format(Locale.ROOT, "%.2f", f.price)} | $star |")
        }
        return sb.toString().trim()
    }
}
