package com.zipper.app.core.web

import android.content.Context
import android.content.SharedPreferences
import java.util.Locale

/**
 * UserPreferenceVault (Muse-Style Persona Store for Zipper / Karvish).
 * Retains user shopping preferences, brand affinities, and default delivery constraints
 * locally on device with zero cloud tokens.
 */
class UserPreferenceVault(context: Context) {

    private val prefs: SharedPreferences = context.getSharedPreferences("karvish_user_preferences", Context.MODE_PRIVATE)

    companion object {
        private const val KEY_PREFERRED_STORE = "pref_store"
        private const val KEY_DELIVERY_ADDRESS = "pref_address"
        private const val KEY_DELIVERY_PINCODE = "pref_pincode"
        private const val PREFIX_BRAND = "pref_brand_"

        // Default verified brand affinities
        val DEFAULT_BRAND_AFFINITIES = mapOf(
            "milk" to "Amul",
            "curd" to "Amul",
            "paneer" to "Amul",
            "butter" to "Amul",
            "atta" to "Aashirvaad",
            "dal" to "Tata Sampann",
            "salt" to "Tata Salt",
            "sugar" to "Madhur",
            "bread" to "Modern",
            "eggs" to "Farm Fresh"
        )

        /**
         * Pure functional enrichment for testing and zero-overhead resolution.
         */
        fun enrichWithAffinities(rawQuery: String, brandOverrides: Map<String, String>? = null): String {
            var result = rawQuery
            val affinities = brandOverrides ?: DEFAULT_BRAND_AFFINITIES
            for ((itemType, preferredBrand) in affinities) {
                val lower = result.lowercase(Locale.ROOT)
                if (lower.contains(itemType) && !lower.contains(preferredBrand.lowercase(Locale.ROOT))) {
                    val hasOtherBrand = listOf("nandini", "mother dairy", "country delight", "heritage", "fortune", "nature fresh")
                        .any { lower.contains(it) }
                    if (!hasOtherBrand) {
                        result = result.replace(Regex("(?i)\\b$itemType\\b"), "$preferredBrand $itemType")
                    }
                }
            }
            return result
        }
    }

    fun getPreferredStore(): String {
        return prefs.getString(KEY_PREFERRED_STORE, "Zepto") ?: "Zepto"
    }

    fun setPreferredStore(store: String) {
        prefs.edit().putString(KEY_PREFERRED_STORE, store).apply()
    }

    fun getDeliveryAddress(): String? {
        return prefs.getString(KEY_DELIVERY_ADDRESS, null)
    }

    fun setDeliveryAddress(address: String) {
        prefs.edit().putString(KEY_DELIVERY_ADDRESS, address).apply()
    }

    fun getDeliveryPincode(): String? {
        return prefs.getString(KEY_DELIVERY_PINCODE, null)
    }

    fun setDeliveryPincode(pincode: String) {
        prefs.edit().putString(KEY_DELIVERY_PINCODE, pincode).apply()
    }

    fun getBrandPreference(itemCategoryOrName: String): String? {
        val key = PREFIX_BRAND + itemCategoryOrName.lowercase(Locale.ROOT).trim()
        val custom = prefs.getString(key, null)
        if (!custom.isNullOrBlank()) return custom
        return DEFAULT_BRAND_AFFINITIES[itemCategoryOrName.lowercase(Locale.ROOT).trim()]
    }

    fun setBrandPreference(itemCategoryOrName: String, brand: String) {
        val key = PREFIX_BRAND + itemCategoryOrName.lowercase(Locale.ROOT).trim()
        prefs.edit().putString(key, brand.trim()).apply()
    }

    /**
     * Enriches an item query with user's preferred brand if the user didn't explicitly specify one.
     * Example: "1L milk" -> "Amul milk 1L"
     */
    fun enrichItemQuery(rawQuery: String): String {
        var result = rawQuery
        for ((itemType, defaultBrand) in DEFAULT_BRAND_AFFINITIES) {
            val custom = getBrandPreference(itemType) ?: defaultBrand
            val lower = result.lowercase(Locale.ROOT)
            if (lower.contains(itemType) && !lower.contains(custom.lowercase(Locale.ROOT))) {
                val hasOtherBrand = listOf("nandini", "mother dairy", "country delight", "heritage", "fortune", "nature fresh")
                    .any { lower.contains(it) }
                if (!hasOtherBrand) {
                    result = result.replace(Regex("(?i)\\b$itemType\\b"), "$custom $itemType")
                }
            }
        }
        return result
    }
}
