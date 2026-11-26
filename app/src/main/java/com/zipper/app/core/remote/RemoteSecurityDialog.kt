package com.zipper.app.core.remote

import android.app.Dialog
import android.content.Context
import android.graphics.Color
import android.graphics.drawable.ColorDrawable
import android.text.Editable
import android.text.TextWatcher
import android.view.LayoutInflater
import android.view.Window
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.widget.SwitchCompat
import com.zipper.app.R

/**
 * Controller for Remote Commands Security & Password configuration dialog.
 * Allows users to easily view, configure, or change their secret command password (e.g. #1234).
 */
class RemoteSecurityDialog(
    private val context: Context,
    private val onSaved: (() -> Unit)? = null
) {

    fun show() {
        val dialog = Dialog(context)
        dialog.requestWindowFeature(Window.FEATURE_NO_TITLE)

        val view = LayoutInflater.from(context).inflate(R.layout.dialog_remote_security, null)
        dialog.setContentView(view)
        dialog.window?.setBackgroundDrawable(ColorDrawable(Color.TRANSPARENT))
        dialog.window?.setLayout(
            (context.resources.displayMetrics.widthPixels * 0.94).toInt(),
            (context.resources.displayMetrics.heightPixels * 0.88).toInt()
        )

        val switchRemoteEnabled = view.findViewById<SwitchCompat>(R.id.switchRemoteEnabled)
        val etSecurityPin = view.findViewById<EditText>(R.id.etSecurityPin)
        val tvPinPreviewBadge = view.findViewById<TextView>(R.id.tvPinPreviewBadge)
        val cbRequirePinForUnknown = view.findViewById<CheckBox>(R.id.cbRequirePinForUnknown)
        val tvTestModeStatus = view.findViewById<TextView>(R.id.tvTestModeStatus)
        val btnToggleTestMode = view.findViewById<Button>(R.id.btnToggleTestMode)

        val tvCheatSheetExampleRing = view.findViewById<TextView>(R.id.tvCheatSheetExampleRing)
        val tvCheatSheetExampleBattery = view.findViewById<TextView>(R.id.tvCheatSheetExampleBattery)
        val tvCheatSheetExampleSchedule = view.findViewById<TextView>(R.id.tvCheatSheetExampleSchedule)
        val tvCheatSheetExampleFlash = view.findViewById<TextView>(R.id.tvCheatSheetExampleFlash)

        val btnCancel = view.findViewById<Button>(R.id.btnCancelSecurity)
        val btnSave = view.findViewById<Button>(R.id.btnSaveSecurity)

        // 1. Load initial state
        val initialPin = RemoteSecurityGate.getSecurityPin(context)
        switchRemoteEnabled.isChecked = RemoteSecurityGate.isEnabled(context)
        cbRequirePinForUnknown.isChecked = RemoteSecurityGate.isRequirePinForUnknown(context)
        etSecurityPin.setText(initialPin)
        etSecurityPin.setSelection(initialPin.length)

        fun updateExamples(pin: String) {
            val displayPin = if (pin.isBlank()) "1234" else pin.removePrefix("#")
            tvPinPreviewBadge.text = "#$displayPin"
            tvCheatSheetExampleRing.text = "#$displayPin: ring"
            tvCheatSheetExampleBattery.text = "#$displayPin: battery"
            tvCheatSheetExampleSchedule.text = "#$displayPin: at 2:30 send hello bhavani to bhavani"
            tvCheatSheetExampleFlash.text = "#$displayPin: flash on  (or flash off)"
        }

        updateExamples(initialPin)

        // 2. Real-time preview updating
        etSecurityPin.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
            override fun afterTextChanged(s: Editable?) {
                updateExamples(s?.toString()?.trim().orEmpty())
            }
        })

        // 3. Test mode UI updates
        fun updateTestModeUi() {
            val isActive = RemoteSecurityGate.isTestModeActive(context)
            if (isActive) {
                tvTestModeStatus.text = "⚡ ACTIVE! Remote commands bypass password for next 60 min."
                tvTestModeStatus.setTextColor(Color.parseColor("#34D399"))
                btnToggleTestMode.text = "Disable"
                btnToggleTestMode.setTextColor(Color.parseColor("#F87171"))
            } else {
                tvTestModeStatus.text = "Inactive. Secret password is required from senders."
                tvTestModeStatus.setTextColor(Color.parseColor("#94A3B8"))
                btnToggleTestMode.text = "Enable (1h)"
                btnToggleTestMode.setTextColor(Color.parseColor("#38BDF8"))
            }
        }
        updateTestModeUi()

        btnToggleTestMode.setOnClickListener {
            if (RemoteSecurityGate.isTestModeActive(context)) {
                RemoteSecurityGate.disableTestMode(context)
                Toast.makeText(context, "🔒 Test mode disabled (Password required)", Toast.LENGTH_SHORT).show()
            } else {
                RemoteSecurityGate.enableTestModeForDuration(context, 60 * 60 * 1000L)
                Toast.makeText(context, "🔓 Test mode active for 1 hour! PIN bypassed.", Toast.LENGTH_LONG).show()
            }
            updateTestModeUi()
        }

        btnCancel.setOnClickListener {
            dialog.dismiss()
        }

        btnSave.setOnClickListener {
            val rawPin = etSecurityPin.text.toString().trim()
            val cleanPin = rawPin.removePrefix("#").trim()

            if (cleanPin.length < 2) {
                Toast.makeText(context, "⚠️ Please enter a password of at least 2 characters", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            RemoteSecurityGate.setSecurityPin(context, cleanPin)
            RemoteSecurityGate.setEnabled(context, switchRemoteEnabled.isChecked)
            RemoteSecurityGate.setRequirePinForUnknown(context, cbRequirePinForUnknown.isChecked)

            Toast.makeText(context, "✅ Remote security password set to #$cleanPin!", Toast.LENGTH_LONG).show()
            onSaved?.invoke()
            dialog.dismiss()
        }

        dialog.show()
    }
}
