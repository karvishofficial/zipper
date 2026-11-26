package com.zipper.app

import com.zipper.app.core.plugins.clipboard.ClipboardActionEngine
import com.zipper.app.core.plugins.clipboard.ClipboardContentType
import com.zipper.app.core.plugins.expense.AutoExpenseLedgerEngine
import com.zipper.app.core.plugins.expense.ExpenseCategory
import com.zipper.app.core.remote.RemoteActionType
import com.zipper.app.core.remote.RemoteCommandParser
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class BackgroundPluginTests {

    @Test
    fun testClipboardCurrencyConversion() {
        val engine = ClipboardActionEngine()

        val usdAction = engine.analyzeText("$50")
        assertNotNull(usdAction)
        assertEquals(ClipboardContentType.CURRENCY_CONVERSION, usdAction?.type)
        assertTrue(usdAction?.badge?.contains("₹") == true)

        val eurAction = engine.analyzeText("€20.50")
        assertNotNull(eurAction)
        assertEquals(ClipboardContentType.CURRENCY_CONVERSION, eurAction?.type)
        assertTrue(eurAction?.badge?.contains("₹") == true)
    }

    @Test
    fun testClipboardWhatsAppPhoneDirectChat() {
        val engine = ClipboardActionEngine()

        val phoneAction = engine.analyzeText("+91 9876543210")
        assertNotNull(phoneAction)
        assertEquals(ClipboardContentType.PHONE_NUMBER_WHATSAPP, phoneAction?.type)
        assertEquals("https://wa.me/919876543210", phoneAction?.directLink)
    }

    @Test
    fun testClipboardIndiaPostConsignment() {
        val engine = ClipboardActionEngine()

        val trackingAction = engine.analyzeText("Your consignment EM123456789IN is out for delivery")
        assertNotNull(trackingAction)
        assertEquals(ClipboardContentType.COURIER_TRACKING, trackingAction?.type)
        assertEquals("EM123456789IN", trackingAction?.badge)
    }

    @Test
    fun testExpenseLedgerDebitParsing() {
        val ledger = AutoExpenseLedgerEngine()

        val text = "Your a/c no. XX1234 is debited for Rs 350.00 on 10-Oct to Swiggy UPI ref 12345"
        val txn = ledger.parseTransaction(text, "HDFC Bank")

        assertNotNull(txn)
        assertEquals(350.0, txn?.amount ?: 0.0, 0.01)
        assertEquals("Swiggy", txn?.merchant)
        assertEquals(ExpenseCategory.FOOD_AND_DINING, txn?.category)
        assertEquals("UPI", txn?.mode)
    }

    @Test
    fun testExpenseLedgerRejectsCreditAndOtp() {
        val ledger = AutoExpenseLedgerEngine()

        val creditText = "Your a/c XX1234 is credited with Rs 5000.00 on 10-Oct"
        val creditTxn = ledger.parseTransaction(creditText, "HDFC Bank")
        assertNull(creditTxn)

        val otpText = "Your OTP for Rs 500 txn is 849201. Do not share with anyone."
        val otpTxn = ledger.parseTransaction(otpText, "SBI")
        assertNull(otpTxn)
    }

    @Test
    fun testRemoteCommandParserBackgroundPluginCommands() {
        val cmdExpense = RemoteCommandParser.parse("Zipper: expenses today", "Boss", "WhatsApp")
        assertNotNull(cmdExpense)
        assertEquals(RemoteActionType.GET_EXPENSE_SUMMARY, cmdExpense?.actionType)

        val cmdArmSentry = RemoteCommandParser.parse("Zipper: arm sentry", "Boss", "WhatsApp")
        assertNotNull(cmdArmSentry)
        assertEquals(RemoteActionType.ARM_SENTRY, cmdArmSentry?.actionType)

        val cmdDisarmSentry = RemoteCommandParser.parse("Zipper: disarm sentry", "Boss", "WhatsApp")
        assertNotNull(cmdDisarmSentry)
        assertEquals(RemoteActionType.DISARM_SENTRY, cmdDisarmSentry?.actionType)
    }
}
