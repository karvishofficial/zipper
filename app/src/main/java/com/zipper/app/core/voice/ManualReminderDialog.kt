package com.zipper.app.core.voice

import android.app.Dialog
import android.content.Context
import android.graphics.Color
import android.graphics.drawable.ColorDrawable
import android.view.LayoutInflater
import android.view.View
import android.view.Window
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import com.zipper.app.R
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Locale

/**
 * ManualReminderDialog
 *
 * Provides a dedicated, premium manual UI for setting Call-Kind Voice Reminders
 * with fast presets, custom titles, voice prompts, and optional Karvish agent handoff.
 */
class ManualReminderDialog(
    private val context: Context,
    private val onReminderScheduled: ((CallReminder) -> Unit)? = null
) {

    fun show() {
        val dialog = Dialog(context)
        dialog.requestWindowFeature(Window.FEATURE_NO_TITLE)

        val view = LayoutInflater.from(context).inflate(R.layout.dialog_manual_reminder, null)
        dialog.setContentView(view)
        dialog.window?.setBackgroundDrawable(ColorDrawable(Color.TRANSPARENT))
        dialog.window?.setLayout(
            (context.resources.displayMetrics.widthPixels * 0.92).toInt(),
            android.view.ViewGroup.LayoutParams.WRAP_CONTENT
        )

        val etReminderTitle = view.findViewById<EditText>(R.id.etReminderTitle)
        val etSpokenPrompt = view.findViewById<EditText>(R.id.etSpokenPrompt)
        val tvSelectedTimeSummary = view.findViewById<TextView>(R.id.tvSelectedTimeSummary)
        val cbAgentHandoff = view.findViewById<CheckBox>(R.id.cbAgentHandoff)
        val etTargetAgentGoal = view.findViewById<EditText>(R.id.etTargetAgentGoal)
        val btnCancel = view.findViewById<Button>(R.id.btnCancelReminder)
        val btnSchedule = view.findViewById<Button>(R.id.btnScheduleCallReminder)

        val chip15m = view.findViewById<Button>(R.id.chip15m)
        val chip30m = view.findViewById<Button>(R.id.chip30m)
        val chip1h = view.findViewById<Button>(R.id.chip1h)
        val chipTonight = view.findViewById<Button>(R.id.chipTonight)
        val chipTomorrow = view.findViewById<Button>(R.id.chipTomorrow)

        var selectedTriggerEpochMs = System.currentTimeMillis() + (15 * 60 * 1000L)
        val timeFormat = SimpleDateFormat("h:mm a, d MMM", Locale.getDefault())

        fun updateTimeDisplay(ms: Long, label: String) {
            selectedTriggerEpochMs = ms
            tvSelectedTimeSummary.text = "Scheduled for: $label (${timeFormat.format(ms)})"
        }

        chip15m.setOnClickListener {
            updateTimeDisplay(System.currentTimeMillis() + (15 * 60 * 1000L), "+15 minutes")
        }

        chip30m.setOnClickListener {
            updateTimeDisplay(System.currentTimeMillis() + (30 * 60 * 1000L), "+30 minutes")
        }

        chip1h.setOnClickListener {
            updateTimeDisplay(System.currentTimeMillis() + (60 * 60 * 1000L), "+1 hour")
        }

        chipTonight.setOnClickListener {
            val cal = Calendar.getInstance().apply {
                set(Calendar.HOUR_OF_DAY, 20)
                set(Calendar.MINUTE, 0)
                set(Calendar.SECOND, 0)
            }
            if (cal.timeInMillis <= System.currentTimeMillis()) {
                cal.add(Calendar.DAY_OF_YEAR, 1)
            }
            updateTimeDisplay(cal.timeInMillis, "Tonight 8:00 PM")
        }

        chipTomorrow.setOnClickListener {
            val cal = Calendar.getInstance().apply {
                add(Calendar.DAY_OF_YEAR, 1)
                set(Calendar.HOUR_OF_DAY, 9)
                set(Calendar.MINUTE, 0)
                set(Calendar.SECOND, 0)
            }
            updateTimeDisplay(cal.timeInMillis, "Tomorrow 9:00 AM")
        }

        cbAgentHandoff.setOnCheckedChangeListener { _, isChecked ->
            etTargetAgentGoal.visibility = if (isChecked) View.VISIBLE else View.GONE
            if (isChecked && etTargetAgentGoal.text.isNullOrBlank()) {
                val title = etReminderTitle.text.toString().trim()
                if (title.isNotEmpty()) {
                    etTargetAgentGoal.setText("Perform task: $title")
                }
            }
        }

        btnCancel.setOnClickListener {
            dialog.dismiss()
        }

        btnSchedule.setOnClickListener {
            val title = etReminderTitle.text.toString().trim().ifBlank { "Task Reminder" }
            val promptInput = etSpokenPrompt.text.toString().trim()
            val spokenPrompt = if (promptInput.isNotBlank()) {
                promptInput
            } else {
                "Hey Karthik, it's time to $title."
            }

            val isAgentHandoff = cbAgentHandoff.isChecked
            val targetGoal = if (isAgentHandoff) {
                etTargetAgentGoal.text.toString().trim().ifBlank { title }
            } else null

            val reminder = CallReminder(
                title = title,
                spokenPrompt = spokenPrompt,
                triggerTimeEpochMs = selectedTriggerEpochMs,
                actionType = if (isAgentHandoff) CallReminderActionType.AGENT_HANDOFF else CallReminderActionType.VOICE_ONLY,
                targetGoal = targetGoal
            )

            CallReminderScheduler.schedule(context, reminder)
            Toast.makeText(context, "📞 Call Reminder set for ${timeFormat.format(selectedTriggerEpochMs)}", Toast.LENGTH_LONG).show()
            onReminderScheduled?.invoke(reminder)
            dialog.dismiss()
        }

        dialog.show()
    }
}
