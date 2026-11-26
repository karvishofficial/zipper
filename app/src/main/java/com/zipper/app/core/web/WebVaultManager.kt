package com.zipper.app.core.web

import android.content.Context
import android.content.SharedPreferences
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import android.util.Log
import com.zipper.app.core.model.DistilledNode
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

/**
 * Encrypted Hardware-Backed Form & Credential Vault for Karvish-Web.
 *
 * Stores user profiles, college portal credentials, and shipping addresses
 * using Android KeyStore AES-256 GCM encryption.
 *
 * Enables 1-Tap Form Filling in:
 * - AI: OFF (deterministic semantic regex/attribute matching in <50ms)
 * - AI: ON (contextual resolution for non-standard form fields)
 */
class WebVaultManager(private val context: Context) {

    companion object {
        private const val TAG = "WebVaultManager"
        private const val ANDROID_KEYSTORE = "AndroidKeyStore"
        private const val KEY_ALIAS = "karvish_web_vault_key"
        private const val PREFS_NAME = "karvish_web_vault_secure_store"
        private const val GCM_IV_LENGTH = 12
        private const val GCM_TAG_LENGTH = 128

        // Standard Vault Keys
        const val KEY_USN = "usn"
        const val KEY_DOB_DAY = "dob_day"
        const val KEY_DOB_MONTH = "dob_month"
        const val KEY_DOB_YEAR = "dob_year"
        const val KEY_USERNAME = "username"
        const val KEY_PASSWORD = "password"
        const val KEY_FULL_NAME = "full_name"
        const val KEY_PHONE = "phone"
        const val KEY_PARENT_PHONE = "parent_phone"
        const val KEY_PINCODE = "pincode"
        const val KEY_ADDRESS = "address"
        const val KEY_CITY = "city"
    }

    private val prefs: SharedPreferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    init {
        ensureKeyGenerated()
    }

    private fun ensureKeyGenerated() {
        try {
            val keyStore = KeyStore.getInstance(ANDROID_KEYSTORE).apply { load(null) }
            if (!keyStore.containsAlias(KEY_ALIAS)) {
                val keyGen = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, ANDROID_KEYSTORE)
                val spec = KeyGenParameterSpec.Builder(
                    KEY_ALIAS,
                    KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT
                )
                    .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                    .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                    .setKeySize(256)
                    .build()
                keyGen.init(spec)
                keyGen.generateKey()
                Log.i(TAG, "Hardware AES-256 GCM key generated in AndroidKeyStore")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize KeyStore key", e)
        }
    }

    private fun getSecretKey(): SecretKey {
        val keyStore = KeyStore.getInstance(ANDROID_KEYSTORE).apply { load(null) }
        return (keyStore.getEntry(KEY_ALIAS, null) as KeyStore.SecretKeyEntry).secretKey
    }

    fun saveField(key: String, plainText: String) {
        if (plainText.isBlank()) {
            prefs.edit().remove(key).apply()
            return
        }
        try {
            val cipher = Cipher.getInstance("AES/GCM/NoPadding")
            cipher.init(Cipher.ENCRYPT_MODE, getSecretKey())
            val iv = cipher.iv
            val cipherText = cipher.doFinal(plainText.toByteArray(Charsets.UTF_8))
            val combined = ByteArray(iv.size + cipherText.size)
            System.arraycopy(iv, 0, combined, 0, iv.size)
            System.arraycopy(cipherText, 0, combined, iv.size, cipherText.size)
            val encoded = Base64.encodeToString(combined, Base64.NO_WRAP)
            prefs.edit().putString(key, encoded).apply()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to encrypt vault field: $key", e)
        }
    }

    fun getField(key: String): String? {
        val encoded = prefs.getString(key, null) ?: return null
        return try {
            val combined = Base64.decode(encoded, Base64.NO_WRAP)
            if (combined.size < GCM_IV_LENGTH) return null
            val iv = ByteArray(GCM_IV_LENGTH)
            val cipherText = ByteArray(combined.size - GCM_IV_LENGTH)
            System.arraycopy(combined, 0, iv, 0, GCM_IV_LENGTH)
            System.arraycopy(combined, GCM_IV_LENGTH, cipherText, 0, cipherText.size)

            val cipher = Cipher.getInstance("AES/GCM/NoPadding")
            val spec = GCMParameterSpec(GCM_TAG_LENGTH, iv)
            cipher.init(Cipher.DECRYPT_MODE, getSecretKey(), spec)
            String(cipher.doFinal(cipherText), Charsets.UTF_8)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to decrypt vault field: $key", e)
            null
        }
    }

    fun getAllConfiguredKeys(): List<String> {
        return prefs.all.keys.toList()
    }

    /**
     * Matches a DOM input field node against stored vault values using smart semantic heuristics.
     * Returns Pair(vaultKey, valueToFill) if a match is found.
     */
    fun matchVaultFieldForNode(node: DistilledNode): Pair<String, String>? {
        val label = node.semanticLabel.lowercase()
        val hint = (node.hintText ?: "").lowercase()
        val res = (node.resourceId ?: "").lowercase()
        val combined = "$label $hint $res"

        return when {
            combined.contains("usn") || combined.contains("university seat") || combined.contains("roll no") -> {
                getField(KEY_USN)?.let { KEY_USN to it }
            }
            combined.contains("day") || combined.contains("dd") || combined.contains("birth day") -> {
                getField(KEY_DOB_DAY)?.let { KEY_DOB_DAY to it }
            }
            combined.contains("month") || combined.contains("mm") || combined.contains("birth month") -> {
                getField(KEY_DOB_MONTH)?.let { KEY_DOB_MONTH to it }
            }
            combined.contains("year") || combined.contains("yyyy") || combined.contains("birth year") -> {
                getField(KEY_DOB_YEAR)?.let { KEY_DOB_YEAR to it }
            }
            combined.contains("password") || combined.contains("passcode") || combined.contains("pin") -> {
                getField(KEY_PASSWORD)?.let { KEY_PASSWORD to it }
            }
            combined.contains("father") || combined.contains("mother") || combined.contains("parent") -> {
                getField(KEY_PARENT_PHONE)?.let { KEY_PARENT_PHONE to it }
            }
            combined.contains("pincode") || combined.contains("postal") || combined.contains("zip") -> {
                getField(KEY_PINCODE)?.let { KEY_PINCODE to it }
            }
            combined.contains("phone") || combined.contains("mobile") -> {
                getField(KEY_PHONE)?.let { KEY_PHONE to it }
            }
            combined.contains("username") || combined.contains("login id") || combined.contains("user id") -> {
                getField(KEY_USERNAME)?.let { KEY_USERNAME to it } ?: getField(KEY_USN)?.let { KEY_USN to it }
            }
            combined.contains("address") || combined.contains("street") -> {
                getField(KEY_ADDRESS)?.let { KEY_ADDRESS to it }
            }
            combined.contains("city") -> {
                getField(KEY_CITY)?.let { KEY_CITY to it }
            }
            else -> null
        }
    }
}
