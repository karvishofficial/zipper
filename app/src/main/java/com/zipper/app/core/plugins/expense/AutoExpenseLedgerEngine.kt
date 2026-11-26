package com.zipper.app.core.plugins.expense

import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.UUID

enum class ExpenseCategory {
    FOOD_AND_DINING,
    GROCERIES,
    SHOPPING,
    TRAVEL,
    BILLS_AND_UTILITIES,
    TRANSFER,
    OTHER
}

data class ExpenseTransaction(
    val id: String = UUID.randomUUID().toString().take(8),
    val amount: Double,
    val merchant: String,
    val category: ExpenseCategory,
    val mode: String, // "UPI", "CARD", "NETBANKING", "WALLET"
    val timestamp: Long = System.currentTimeMillis(),
    val rawText: String
)

data class DailyExpenseSummary(
    val dateString: String,
    val totalAmount: Double,
    val transactionCount: Int,
    val categoryTotals: Map<ExpenseCategory, Double>,
    val topMerchant: String?
) {
    fun formatReport(): String {
        val catLines = categoryTotals.entries.joinToString("\n") { "• ${it.key.name.replace('_', ' ')}: ₹${it.value.toInt()}" }
        return """
            Date: $dateString
            Total Spent: ₹${totalAmount.toInt()}
            Transactions: $transactionCount
            Top Merchant: ${topMerchant ?: "N/A"}
            
            Category Breakdown:
            ${catLines.ifBlank { "• No transactions yet" }}
        """.trimIndent()
    }
}

/**
 * Silent UPI & Auto-Expense Ledger Plugin.
 *
 * Runs 100% in the background, intercepting payment notifications and bank SMS in RAM.
 * Automatically parses Amount, Merchant, and Category without manual data entry.
 *
 * Zero foreground screen interruption.
 */
class AutoExpenseLedgerEngine(private val context: Context? = null) {

    companion object {
        private const val TAG = "AutoExpenseLedger"
        private const val PREFS_NAME = "karvish_auto_expenses_db"
        private const val KEY_TRANSACTIONS = "transactions_json"
    }

    private val prefs: SharedPreferences? by lazy {
        context?.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    }

    fun processIncomingMessage(text: String, sourceAppOrSender: String): ExpenseTransaction? {
        val txn = parseTransaction(text, sourceAppOrSender) ?: return null
        recordTransaction(txn)
        Log.i(TAG, "Recorded background expense: ₹${txn.amount} at ${txn.merchant} [${txn.category.name}] via ${txn.mode}")
        return txn
    }

    /**
     * Inspects incoming text and parses the transaction if it's a debit/payment alert.
     */
    fun parseTransaction(text: String, sourceAppOrSender: String): ExpenseTransaction? {
        val lower = text.lowercase(Locale.ROOT)

        // Only process debit / paid alerts, ignore credits / OTPs / marketing
        val isDebit = lower.contains("debited") || lower.contains("paid") ||
            lower.contains("sent rs") || lower.contains("txn of rs") ||
            lower.contains("spent") || lower.contains("payment of rs")

        if (!isDebit || lower.contains("otp") || lower.contains("credited")) {
            return null
        }

        // 1. Extract Amount
        val amountRegex = Regex("""(?:rs\.?|inr)\s*([0-9,]+(?:\.[0-9]{1,2})?)""", RegexOption.IGNORE_CASE)
        val amountMatch = amountRegex.find(text) ?: return null
        val amountStr = amountMatch.groupValues[1].replace(",", "")
        val amount = amountStr.toDoubleOrNull() ?: return null

        // 2. Extract Merchant / Beneficiary
        var merchant = "Merchant"
        val toMatch = Regex("""(?:to|at|vpa)\s+([a-zA-Z0-9\s&'-]{3,25}?)(?=\s+(?:on|ref|upi|via|bal|avl|\.|$))""", RegexOption.IGNORE_CASE).find(text)
        if (toMatch != null) {
            merchant = toMatch.groupValues[1].trim()
        } else {
            // Check known app sources
            if (sourceAppOrSender.contains("swiggy", true)) merchant = "Swiggy"
            else if (sourceAppOrSender.contains("zomato", true)) merchant = "Zomato"
            else if (sourceAppOrSender.contains("zepto", true)) merchant = "Zepto"
            else if (sourceAppOrSender.contains("blinkit", true)) merchant = "Blinkit"
            else if (sourceAppOrSender.contains("uber", true)) merchant = "Uber"
            else if (sourceAppOrSender.contains("ola", true)) merchant = "Ola"
            else if (sourceAppOrSender.contains("amazon", true)) merchant = "Amazon"
        }

        // 3. Categorize
        val category = categorizeMerchant(merchant, lower)

        // 4. Payment Mode
        val mode = when {
            lower.contains("upi") -> "UPI"
            lower.contains("card") || lower.contains("debit card") || lower.contains("credit card") -> "CARD"
            lower.contains("wallet") -> "WALLET"
            else -> "BANK"
        }

        val txn = ExpenseTransaction(
            amount = amount,
            merchant = merchant,
            category = category,
            mode = mode,
            rawText = text
        )

        return txn
    }

    private fun categorizeMerchant(merchant: String, rawText: String): ExpenseCategory {
        val combined = "$merchant $rawText".lowercase(Locale.ROOT)
        return when {
            combined.contains("swiggy") || combined.contains("zomato") ||
                combined.contains("mcdonald") || combined.contains("starbucks") ||
                combined.contains("kfc") || combined.contains("burger") ||
                combined.contains("restaurant") || combined.contains("cafe") ||
                combined.contains("hotel") || combined.contains("food") -> ExpenseCategory.FOOD_AND_DINING

            combined.contains("zepto") || combined.contains("blinkit") ||
                combined.contains("instamart") || combined.contains("bigbasket") ||
                combined.contains("grocery") || combined.contains("supermarket") ||
                combined.contains("vegetable") -> ExpenseCategory.GROCERIES

            combined.contains("uber") || combined.contains("ola") ||
                combined.contains("rapido") || combined.contains("metro") ||
                combined.contains("petrol") || combined.contains("fuel") ||
                combined.contains("hpcl") || combined.contains("bpcl") ||
                combined.contains("ioc") || combined.contains("toll") -> ExpenseCategory.TRAVEL

            combined.contains("amazon") || combined.contains("flipkart") ||
                combined.contains("myntra") || combined.contains("zara") ||
                combined.contains("ajio") || combined.contains("meesho") -> ExpenseCategory.SHOPPING

            combined.contains("bescom") || combined.contains("airtel") ||
                combined.contains("jio") || combined.contains("electricity") ||
                combined.contains("bill") || combined.contains("rent") ||
                combined.contains("broadband") || combined.contains("wifi") -> ExpenseCategory.BILLS_AND_UTILITIES

            else -> ExpenseCategory.OTHER
        }
    }

    fun recordTransaction(txn: ExpenseTransaction) {
        val all = getAllTransactions().toMutableList()
        all.add(0, txn)
        persistTransactions(all)
    }

    fun getAllTransactions(): List<ExpenseTransaction> {
        val raw = prefs?.getString(KEY_TRANSACTIONS, null) ?: return emptyList()
        return try {
            val arr = JSONArray(raw)
            val list = mutableListOf<ExpenseTransaction>()
            for (i in 0 until arr.length()) {
                val obj = arr.getJSONObject(i)
                list.add(
                    ExpenseTransaction(
                        id = obj.getString("id"),
                        amount = obj.getDouble("amount"),
                        merchant = obj.getString("merchant"),
                        category = ExpenseCategory.valueOf(obj.getString("category")),
                        mode = obj.getString("mode"),
                        timestamp = obj.getLong("timestamp"),
                        rawText = obj.getString("rawText")
                    )
                )
            }
            list
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse transactions", e)
            emptyList()
        }
    }

    fun getTodaySummary(): DailyExpenseSummary {
        val sdf = SimpleDateFormat("yyyy-MM-dd", Locale.getDefault())
        val todayStr = sdf.format(Date())

        val todayTxns = getAllTransactions().filter {
            sdf.format(Date(it.timestamp)) == todayStr
        }

        val total = todayTxns.sumOf { it.amount }
        val categoryTotals = todayTxns.groupBy { it.category }
            .mapValues { entry -> entry.value.sumOf { it.amount } }

        val topMerchant = todayTxns.groupBy { it.merchant }
            .maxByOrNull { entry -> entry.value.sumOf { it.amount } }?.key

        return DailyExpenseSummary(
            dateString = todayStr,
            totalAmount = total,
            transactionCount = todayTxns.size,
            categoryTotals = categoryTotals,
            topMerchant = topMerchant
        )
    }

    private fun persistTransactions(list: List<ExpenseTransaction>) {
        val arr = JSONArray()
        // Keep the latest 500 transactions
        for (txn in list.take(500)) {
            val obj = JSONObject()
            obj.put("id", txn.id)
            obj.put("amount", txn.amount)
            obj.put("merchant", txn.merchant)
            obj.put("category", txn.category.name)
            obj.put("mode", txn.mode)
            obj.put("timestamp", txn.timestamp)
            obj.put("rawText", txn.rawText)
            arr.put(obj)
        }
        prefs?.edit()?.putString(KEY_TRANSACTIONS, arr.toString())?.apply()
    }
}
