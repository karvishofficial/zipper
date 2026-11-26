package com.zipper.app.webmind.data

import java.util.Locale

/**
 * Standardized Unit Price Normalizer.
 * Accurately extracts weights, volumes, and storage capacities from product titles,
 * converting varied package sizes to unified base metrics (e.g. ₹/kg, ₹/liter, ₹/GB).
 */
object UnitPriceNormalizer {

    data class NormalizedQuantity(
        val amount: Double,
        val unit: String,
        val standardUnit: String,
        val standardMultiplier: Double
    )

    data class UnitComparisonResult(
        val unitPrice: Double,
        val unitDisplay: String,
        val standardMetric: String
    )

    private val WEIGHT_REGEX = Regex("""(?i)\b(\d+(?:\.\d+)?)\s*(kg|kilo|kilograms?|gms?|grams?|g)\b""")
    private val VOLUME_REGEX = Regex("""(?i)\b(\d+(?:\.\d+)?)\s*(l|ltr|litres?|liters?|ml|millilitres?)\b""")
    private val STORAGE_REGEX = Regex("""(?i)\b(\d+)\s*(gb|tb)\b""")
    private val COUNT_REGEX = Regex("""(?i)\b(\d+)\s*(pcs?|pieces?|pack|units?|count)\b""")

    fun parseQuantity(title: String): NormalizedQuantity? {
        val t = title.lowercase(Locale.ROOT)

        // 1. Check Weight
        val weightMatch = WEIGHT_REGEX.find(t)
        if (weightMatch != null) {
            val amt = weightMatch.groupValues[1].toDoubleOrNull() ?: 1.0
            val unit = weightMatch.groupValues[2]
            return if (unit.startsWith("k")) {
                NormalizedQuantity(amt, "kg", "kg", 1.0)
            } else {
                NormalizedQuantity(amt, "g", "kg", 0.001)
            }
        }

        // 2. Check Volume
        val volumeMatch = VOLUME_REGEX.find(t)
        if (volumeMatch != null) {
            val amt = volumeMatch.groupValues[1].toDoubleOrNull() ?: 1.0
            val unit = volumeMatch.groupValues[2]
            return if (unit.startsWith("m")) {
                NormalizedQuantity(amt, "ml", "L", 0.001)
            } else {
                NormalizedQuantity(amt, "L", "L", 1.0)
            }
        }

        // 3. Check Storage
        val storageMatch = STORAGE_REGEX.find(t)
        if (storageMatch != null) {
            val amt = storageMatch.groupValues[1].toDoubleOrNull() ?: 128.0
            val unit = storageMatch.groupValues[2]
            return if (unit == "tb") {
                NormalizedQuantity(amt, "TB", "GB", 1024.0)
            } else {
                NormalizedQuantity(amt, "GB", "GB", 1.0)
            }
        }

        // 4. Check Count
        val countMatch = COUNT_REGEX.find(t)
        if (countMatch != null) {
            val amt = countMatch.groupValues[1].toDoubleOrNull() ?: 1.0
            return NormalizedQuantity(amt, "pcs", "piece", 1.0)
        }

        return null
    }

    fun computeUnitPrice(price: Double, title: String): UnitComparisonResult? {
        if (price <= 0.0) return null
        val qty = parseQuantity(title) ?: return null

        val totalStandardUnits = qty.amount * qty.standardMultiplier
        if (totalStandardUnits <= 0.0) return null

        val costPerStandard = price / totalStandardUnits
        val formatted = "₹${"%,.0f".format(costPerStandard)}/${qty.standardUnit}"

        return UnitComparisonResult(
            unitPrice = costPerStandard,
            unitDisplay = formatted,
            standardMetric = qty.standardUnit
        )
    }

    fun computeSavingsDelta(
        cheapestPrice: Double,
        cheapestTitle: String,
        competitorPrice: Double,
        competitorTitle: String
    ): String {
        val cheapUnit = computeUnitPrice(cheapestPrice, cheapestTitle)
        val compUnit = computeUnitPrice(competitorPrice, competitorTitle)

        if (cheapUnit != null && compUnit != null && cheapUnit.standardMetric == compUnit.standardMetric) {
            if (compUnit.unitPrice > cheapUnit.unitPrice) {
                val diff = compUnit.unitPrice - cheapUnit.unitPrice
                val pct = ((diff / compUnit.unitPrice) * 100).toInt()
                return "${cheapUnit.unitDisplay} vs ${compUnit.unitDisplay} → Saves ₹${"%,.0f".format(diff)}/${cheapUnit.standardMetric} ($pct% cheaper)"
            }
        }

        if (competitorPrice > cheapestPrice) {
            val diff = competitorPrice - cheapestPrice
            val pct = ((diff / competitorPrice) * 100).toInt()
            return "₹${"%,.0f".format(cheapestPrice)} vs ₹${"%,.0f".format(competitorPrice)} → Saves ₹${"%,.0f".format(diff)} ($pct% cheaper)"
        }

        return "Best matching price across verified stores"
    }
}
