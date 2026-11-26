package com.zipper.app.telephony

/**
 * Represents the live physical lifecycle of an autonomous cellular call.
 */
sealed class KarvishCallState {
    object Idle : KarvishCallState()
    data class Dialing(val targetContact: String, val phoneNumber: String) : KarvishCallState()
    data class Ringing(val targetContact: String, val phoneNumber: String) : KarvishCallState()
    data class Connected(val targetContact: String, val message: String) : KarvishCallState()
    data class Speaking(val targetContact: String, val text: String, val progressPct: Int = 0) : KarvishCallState()
    data class Completed(val targetContact: String, val summary: String) : KarvishCallState()
    data class Failed(val targetContact: String, val reason: String) : KarvishCallState()
}
