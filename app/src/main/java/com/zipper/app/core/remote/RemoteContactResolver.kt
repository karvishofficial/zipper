package com.zipper.app.core.remote

import android.content.Context
import android.content.pm.PackageManager
import android.database.Cursor
import android.provider.ContactsContract
import android.util.Log

/**
 * Represents a contact matching a remote messaging query.
 */
data class ResolvedContact(
    val name: String,
    val phoneNumber: String,
    val normalizedPhone: String
)

/**
 * Resolves contacts in the background via ContactsContract without launching
 * or interrupting any foreground application.
 */
object RemoteContactResolver {

    private const val TAG = "RemoteContactResolver"

    /**
     * Finds contacts whose display name matches or contains [query].
     * Returns a deduplicated list of candidates.
     */
    fun findMatchingContacts(context: Context, query: String): List<ResolvedContact> {
        val cleanQuery = query.trim()
        if (cleanQuery.isBlank()) return emptyList()

        if (context.checkSelfPermission(android.Manifest.permission.READ_CONTACTS) != PackageManager.PERMISSION_GRANTED) {
            Log.w(TAG, "READ_CONTACTS permission not granted; returning empty list.")
            return emptyList()
        }

        val results = mutableListOf<ResolvedContact>()
        var cursor: Cursor? = null

        try {
            val uri = ContactsContract.CommonDataKinds.Phone.CONTENT_URI
            val projection = arrayOf(
                ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME,
                ContactsContract.CommonDataKinds.Phone.NUMBER,
                ContactsContract.CommonDataKinds.Phone.NORMALIZED_NUMBER
            )
            val selection = "${ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME} LIKE ?"
            val selectionArgs = arrayOf("%$cleanQuery%")
            val sortOrder = "${ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME} ASC LIMIT 10"

            cursor = context.contentResolver.query(uri, projection, selection, selectionArgs, sortOrder)
            if (cursor != null && cursor.moveToFirst()) {
                val nameIdx = cursor.getColumnIndex(ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME)
                val numberIdx = cursor.getColumnIndex(ContactsContract.CommonDataKinds.Phone.NUMBER)
                val normIdx = cursor.getColumnIndex(ContactsContract.CommonDataKinds.Phone.NORMALIZED_NUMBER)

                val seenNumbers = mutableSetOf<String>()

                do {
                    val name = if (nameIdx >= 0) cursor.getString(nameIdx).orEmpty() else ""
                    val rawNumber = if (numberIdx >= 0) cursor.getString(numberIdx).orEmpty() else ""
                    val normNumber = if (normIdx >= 0) cursor.getString(normIdx).orEmpty() else ""

                    val effectivePhone = if (normNumber.isNotBlank()) normNumber else normalizePhoneNumber(rawNumber)
                    val phoneKey = effectivePhone.takeLast(10)

                    if (phoneKey.isNotBlank() && seenNumbers.add(phoneKey)) {
                        results.add(
                            ResolvedContact(
                                name = name,
                                phoneNumber = rawNumber,
                                normalizedPhone = effectivePhone
                            )
                        )
                    }
                } while (cursor.moveToNext())
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error querying contacts: ${e.message}")
        } finally {
            cursor?.close()
        }

        return results
    }

    /**
     * Cleans up phone numbers by stripping whitespace, hyphens, and brackets.
     */
    fun normalizePhoneNumber(rawNumber: String): String {
        return rawNumber.replace(Regex("[^0-9+]"), "").trim()
    }
}
