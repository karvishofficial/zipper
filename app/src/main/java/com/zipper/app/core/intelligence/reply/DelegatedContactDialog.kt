package com.zipper.app.core.intelligence.reply

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
import androidx.appcompat.widget.SwitchCompat
import android.text.Editable
import android.text.TextWatcher
import com.zipper.app.R
import com.zipper.app.core.web.WebAiConfig
import com.zipper.app.core.web.WebConfigManager

/**
 * Controller for the Delegated Contact configuration dialog.
 * Enables users to designate, configure, or revoke AI auto-reply access for specific contacts.
 */
class DelegatedContactDialog(
    private val context: Context,
    private val existingContactId: String? = null,
    private val initialContactName: String? = null,
    private val onContactSaved: ((DelegatedContact) -> Unit)? = null,
    private val onContactDeleted: ((String) -> Unit)? = null
) {

    fun show() {
        val dialog = Dialog(context)
        dialog.requestWindowFeature(Window.FEATURE_NO_TITLE)

        val view = LayoutInflater.from(context).inflate(R.layout.dialog_delegated_contact, null)
        dialog.setContentView(view)
        dialog.window?.setBackgroundDrawable(ColorDrawable(Color.TRANSPARENT))
        dialog.window?.setLayout(
            (context.resources.displayMetrics.widthPixels * 0.94).toInt(),
            (context.resources.displayMetrics.heightPixels * 0.88).toInt()
        )

        val manager = DelegatedContactManager.getInstance(context)
        val existing = existingContactId?.let { manager.getContact(it) }

        val tvDialogTitle = view.findViewById<TextView>(R.id.tvDialogTitle)
        val switchAutoReplyEnabled = view.findViewById<SwitchCompat>(R.id.switchAutoReplyEnabled)
        val etContactName = view.findViewById<EditText>(R.id.etContactName)
        val etContactPhone = view.findViewById<EditText>(R.id.etContactPhone)
        val etCustomContext = view.findViewById<EditText>(R.id.etCustomContext)
        val cbRequireApproval = view.findViewById<CheckBox>(R.id.cbRequireApproval)
        val cbAppendAiBadge = view.findViewById<CheckBox>(R.id.cbAppendAiBadge)

        val chipPresetMeeting = view.findViewById<Button>(R.id.chipPresetMeeting)
        val chipPresetDriving = view.findViewById<Button>(R.id.chipPresetDriving)
        val chipPresetGym = view.findViewById<Button>(R.id.chipPresetGym)
        val chipPresetTravel = view.findViewById<Button>(R.id.chipPresetTravel)

        val chipToneCasual = view.findViewById<Button>(R.id.chipToneCasual)
        val chipTonePro = view.findViewById<Button>(R.id.chipTonePro)
        val chipToneBrief = view.findViewById<Button>(R.id.chipToneBrief)

        val tvMaxRepliesLabel = view.findViewById<TextView>(R.id.tvMaxRepliesLabel)
        val tvAiKeyStatusBadge = view.findViewById<TextView>(R.id.tvAiKeyStatusBadge)
        val etAiApiKey = view.findViewById<EditText>(R.id.etAiApiKey)

        val btnDelete = view.findViewById<Button>(R.id.btnDeleteContact)
        val btnCancel = view.findViewById<Button>(R.id.btnCancelContact)
        val btnSave = view.findViewById<Button>(R.id.btnSaveContact)

        var selectedTone = existing?.personaTone ?: "Casual & Friendly"

        fun updateToneUi(tone: String) {
            selectedTone = tone
            val cyan = Color.parseColor("#38BDF8")
            val gray = Color.parseColor("#94A3B8")

            chipToneCasual.setTextColor(if (tone.contains("Casual", true)) cyan else gray)
            chipTonePro.setTextColor(if (tone.contains("Pro", true)) cyan else gray)
            chipToneBrief.setTextColor(if (tone.contains("Short", true) || tone.contains("Brief", true)) cyan else gray)
        }

        updateToneUi(selectedTone)

        // Session Budget Cycling: Continuous (0, default) -> 5 -> 15 -> 30
        val budgetLevels = listOf(0, 5, 15, 30)
        var currentBudget = existing?.maxRepliesPerSession ?: 0
        if (currentBudget !in budgetLevels && currentBudget < 100) currentBudget = 0

        fun updateBudgetUi(limit: Int) {
            currentBudget = limit
            tvMaxRepliesLabel.text = if (limit <= 0 || limit >= 100) "Continuous (Unlimited)" else "Max $limit replies / session"
        }
        updateBudgetUi(currentBudget)

        tvMaxRepliesLabel.setOnClickListener {
            val nextIdx = (budgetLevels.indexOf(currentBudget) + 1) % budgetLevels.size
            updateBudgetUi(budgetLevels[nextIdx])
        }

        // Connected LLM Brain API Key Setup
        val webConfigMgr = WebConfigManager(context)
        val currentAiConfig = webConfigMgr.getConfig()

        val tvAiFallbackStatusBadge = view.findViewById<TextView>(R.id.tvAiFallbackStatusBadge)
        val etAiFallbackApiKey = view.findViewById<EditText>(R.id.etAiFallbackApiKey)

        if (currentAiConfig.isConfigured && currentAiConfig.apiKey.isNotBlank()) {
            tvAiKeyStatusBadge.text = "Connected (${currentAiConfig.provider})"
            tvAiKeyStatusBadge.setTextColor(Color.parseColor("#34D399"))
            etAiApiKey.setText(currentAiConfig.apiKey)
        } else {
            tvAiKeyStatusBadge.text = "API Key Needed"
            tvAiKeyStatusBadge.setTextColor(Color.parseColor("#F59E0B"))
        }

        if (currentAiConfig.fallbackApiKey.isNotBlank()) {
            tvAiFallbackStatusBadge.text = "Active (${currentAiConfig.fallbackProvider})"
            tvAiFallbackStatusBadge.setTextColor(Color.parseColor("#34D399"))
            etAiFallbackApiKey.setText(currentAiConfig.fallbackApiKey)
        }

        etAiApiKey.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
            override fun afterTextChanged(s: Editable?) {
                val key = s?.toString()?.trim().orEmpty()
                if (key.length >= 8) {
                    val provider = when {
                        key.startsWith("gsk_") -> "Groq"
                        key.startsWith("sk-") -> "OpenAI"
                        key.startsWith("AIza") -> "Gemini"
                        else -> "Custom LLM"
                    }
                    tvAiKeyStatusBadge.text = "Ready ($provider)"
                    tvAiKeyStatusBadge.setTextColor(Color.parseColor("#34D399"))
                } else {
                    tvAiKeyStatusBadge.text = "API Key Needed"
                    tvAiKeyStatusBadge.setTextColor(Color.parseColor("#F59E0B"))
                }
            }
        })

        etAiFallbackApiKey.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
            override fun afterTextChanged(s: Editable?) {
                val key = s?.toString()?.trim().orEmpty()
                if (key.length >= 8) {
                    val provider = when {
                        key.startsWith("gsk_") -> "Groq"
                        key.startsWith("sk-") -> "OpenAI"
                        else -> "Backup LLM"
                    }
                    tvAiFallbackStatusBadge.text = "Ready ($provider)"
                    tvAiFallbackStatusBadge.setTextColor(Color.parseColor("#34D399"))
                } else {
                    tvAiFallbackStatusBadge.text = "Optional (Groq/OpenAI)"
                    tvAiFallbackStatusBadge.setTextColor(Color.parseColor("#64748B"))
                }
            }
        })

        // Remote Security PIN Setup
        val etRemoteSecurityPin = view.findViewById<EditText>(R.id.etRemoteSecurityPin)
        val tvRemotePinStatusBadge = view.findViewById<TextView>(R.id.tvRemotePinStatusBadge)
        val currentRemotePin = com.zipper.app.core.remote.RemoteSecurityGate.getSecurityPin(context)
        etRemoteSecurityPin?.setText(currentRemotePin)
        tvRemotePinStatusBadge?.text = "#$currentRemotePin"

        etRemoteSecurityPin?.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
            override fun afterTextChanged(s: Editable?) {
                val pin = s?.toString()?.trim().orEmpty().removePrefix("#")
                tvRemotePinStatusBadge?.text = if (pin.isBlank()) "#1234" else "#$pin"
            }
        })

        val llClearHistorySection = view.findViewById<View>(R.id.llClearHistorySection)
        val tvChatHistoryTurnsCount = view.findViewById<TextView>(R.id.tvChatHistoryTurnsCount)
        val btnClearChatHistory = view.findViewById<Button>(R.id.btnClearChatHistory)
        val llConfiguredContactsContainer = view.findViewById<View>(R.id.llConfiguredContactsContainer)
        val llConfiguredContactChips = view.findViewById<android.widget.LinearLayout>(R.id.llConfiguredContactChips)

        var activeContactId: String? = existing?.id ?: existingContactId
        val memory = DelegatedConversationMemory.getInstance(context)

        fun updateHistoryUi(contactName: String, contactId: String?) {
            val cleanName = contactName.trim()
            if (cleanName.isBlank()) {
                llClearHistorySection?.visibility = View.GONE
                return
            }
            val turns = memory.getHistory(cleanName, maxTurns = 100)
            llClearHistorySection?.visibility = View.VISIBLE
            if (turns.isNotEmpty()) {
                tvChatHistoryTurnsCount?.text = "${turns.size} conversation turn(s) stored in 2-way memory"
                btnClearChatHistory?.isEnabled = true
                btnClearChatHistory?.alpha = 1.0f
            } else {
                tvChatHistoryTurnsCount?.text = "0 turns stored (Fresh clean slate)"
                btnClearChatHistory?.isEnabled = false
                btnClearChatHistory?.alpha = 0.5f
            }

            btnClearChatHistory?.setOnClickListener {
                memory.clearHistory(cleanName)
                contactId?.let { cid -> manager.resetSession(cid) }
                tvChatHistoryTurnsCount?.text = "0 turns stored (Fresh clean slate)"
                btnClearChatHistory?.isEnabled = false
                btnClearChatHistory?.alpha = 0.5f
                Toast.makeText(context, "🧹 Chat history cleared for $cleanName! Next message starts completely fresh.", Toast.LENGTH_SHORT).show()
            }
        }

        // Contact Switcher Chips
        val allContacts = manager.getContacts()
        fun populateContactChips() {
            if (allContacts.isEmpty()) {
                llConfiguredContactsContainer?.visibility = View.GONE
                return
            }
            llConfiguredContactsContainer?.visibility = View.VISIBLE
            llConfiguredContactChips?.removeAllViews()

            for (c in allContacts) {
                val chipBtn = Button(context, null, 0, com.google.android.material.R.style.Widget_MaterialComponents_Button_OutlinedButton).apply {
                    val isSelected = c.id == activeContactId || (activeContactId == null && c.contactName.equals(etContactName.text.toString(), ignoreCase = true))
                    text = if (c.isEnabled) "🤖 ${c.contactName}" else "⏸️ ${c.contactName}"
                    textSize = 11f
                    setTextColor(if (isSelected) Color.parseColor("#38BDF8") else Color.parseColor("#94A3B8"))
                    setOnClickListener {
                        activeContactId = c.id
                        tvDialogTitle.text = "Configure AI Reply: ${c.contactName}"
                        switchAutoReplyEnabled.isChecked = c.isEnabled
                        etContactName.setText(c.contactName)
                        etContactPhone.setText(c.phoneNumber)
                        etCustomContext.setText(c.customContext)
                        cbRequireApproval.isChecked = c.requireApproval
                        cbAppendAiBadge.isChecked = c.appendAiBadge
                        updateToneUi(c.personaTone)
                        updateBudgetUi(c.maxRepliesPerSession)
                        btnDelete.visibility = View.VISIBLE
                        updateHistoryUi(c.contactName, c.id)
                        populateContactChips()
                    }
                }
                val params = android.widget.LinearLayout.LayoutParams(
                    android.widget.LinearLayout.LayoutParams.WRAP_CONTENT,
                    (36 * context.resources.displayMetrics.density).toInt()
                ).apply {
                    marginEnd = (6 * context.resources.displayMetrics.density).toInt()
                }
                llConfiguredContactChips?.addView(chipBtn, params)
            }

            // "+ Add New" Chip
            val newChip = Button(context, null, 0, com.google.android.material.R.style.Widget_MaterialComponents_Button_OutlinedButton).apply {
                text = "➕ Add New"
                textSize = 11f
                setTextColor(Color.parseColor("#34D399"))
                setOnClickListener {
                    activeContactId = null
                    tvDialogTitle.text = "Delegate Contact to AI"
                    switchAutoReplyEnabled.isChecked = true
                    etContactName.setText("")
                    etContactPhone.setText("")
                    etCustomContext.setText("")
                    cbRequireApproval.isChecked = false
                    cbAppendAiBadge.isChecked = true
                    updateToneUi("Casual & Friendly")
                    updateBudgetUi(0)
                    btnDelete.visibility = View.GONE
                    llClearHistorySection?.visibility = View.GONE
                    populateContactChips()
                }
            }
            val newParams = android.widget.LinearLayout.LayoutParams(
                android.widget.LinearLayout.LayoutParams.WRAP_CONTENT,
                (36 * context.resources.displayMetrics.density).toInt()
            )
            llConfiguredContactChips?.addView(newChip, newParams)
        }

        populateContactChips()

        // Pre-fill existing data
        if (existing != null) {
            tvDialogTitle.text = "Configure AI Reply: ${existing.contactName}"
            switchAutoReplyEnabled.isChecked = existing.isEnabled
            etContactName.setText(existing.contactName)
            etContactPhone.setText(existing.phoneNumber)
            etCustomContext.setText(existing.customContext)
            cbRequireApproval.isChecked = existing.requireApproval
            cbAppendAiBadge.isChecked = existing.appendAiBadge
            btnDelete.visibility = View.VISIBLE
            updateHistoryUi(existing.contactName, existing.id)
        } else if (!initialContactName.isNullOrBlank()) {
            etContactName.setText(initialContactName)
            updateHistoryUi(initialContactName, null)
        }

        etContactName.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
            override fun afterTextChanged(s: Editable?) {
                val name = s?.toString()?.trim().orEmpty()
                updateHistoryUi(name, activeContactId)
            }
        })

        // Preset Chips
        chipPresetMeeting.setOnClickListener {
            etCustomContext.setText("In a client meeting until 5 PM. Will review and send APK right after.")
        }
        chipPresetDriving.setOnClickListener {
            etCustomContext.setText("Driving right now, hands-free. Will call once parked!")
        }
        chipPresetGym.setOnClickListener {
            etCustomContext.setText("At the gym working out. Catch you in an hour!")
        }
        chipPresetTravel.setOnClickListener {
            etCustomContext.setText("Traveling with intermittent network. Will check messages tonight.")
        }

        chipToneCasual.setOnClickListener { updateToneUi("Casual & Friendly, Bangalore slang & Kanglish") }
        chipTonePro.setOnClickListener { updateToneUi("Professional & Courteous, concise") }
        chipToneBrief.setOnClickListener { updateToneUi("Short, direct, maximum 10 words") }

        btnDelete.setOnClickListener {
            activeContactId?.let { id ->
                manager.deleteContact(id)
                onContactDeleted?.invoke(id)
                Toast.makeText(context, "🗑️ Auto-reply revoked for contact", Toast.LENGTH_SHORT).show()
            }
            dialog.dismiss()
        }

        btnCancel.setOnClickListener {
            dialog.dismiss()
        }

        btnSave.setOnClickListener {
            val name = etContactName.text.toString().trim()
            if (name.isBlank()) {
                etContactName.error = "Contact name required"
                return@setOnClickListener
            }

            // Save Primary and Fallback LLM API keys
            val enteredApiKey = etAiApiKey.text.toString().trim()
            val enteredFallbackKey = etAiFallbackApiKey.text.toString().trim()

            val primaryProvider = when {
                enteredApiKey.startsWith("gsk_") -> "Groq"
                enteredApiKey.startsWith("sk-") -> "OpenAI"
                enteredApiKey.startsWith("AIza") -> "Google Gemini"
                else -> currentAiConfig.provider.ifBlank { "Google Gemini" }
            }

            val fallbackProvider = when {
                enteredFallbackKey.startsWith("gsk_") -> "Groq"
                enteredFallbackKey.startsWith("sk-") -> "OpenAI"
                else -> currentAiConfig.fallbackProvider.ifBlank { "Groq" }
            }

            webConfigMgr.saveConfig(
                currentAiConfig.copy(
                    provider = primaryProvider,
                    apiKey = enteredApiKey.ifBlank { currentAiConfig.apiKey },
                    model = if (primaryProvider.contains("Gemini", true)) "gemini-3-flash-preview" else WebConfigManager.getDefaultModelForProvider(primaryProvider),
                    isVerified = true,
                    lastVerified = System.currentTimeMillis(),
                    fallbackProvider = fallbackProvider,
                    fallbackApiKey = enteredFallbackKey.ifBlank { currentAiConfig.fallbackApiKey },
                    fallbackModel = WebConfigManager.getDefaultModelForProvider(fallbackProvider)
                )
            )

            val enteredPin = etRemoteSecurityPin?.text?.toString()?.trim().orEmpty().removePrefix("#")
            if (enteredPin.isNotBlank()) {
                com.zipper.app.core.remote.RemoteSecurityGate.setSecurityPin(context, enteredPin)
            }

            val contact = DelegatedContact(
                id = activeContactId ?: existing?.id ?: java.util.UUID.randomUUID().toString(),
                contactName = name,
                phoneNumber = etContactPhone.text.toString().trim(),
                customContext = etCustomContext.text.toString().trim(),
                personaTone = selectedTone,
                isEnabled = switchAutoReplyEnabled.isChecked,
                maxRepliesPerSession = currentBudget,
                requireApproval = cbRequireApproval.isChecked,
                appendAiBadge = cbAppendAiBadge.isChecked
            )

            manager.saveContact(contact)
            onContactSaved?.invoke(contact)

            val statusMsg = if (contact.isEnabled) "🤖 Auto-Reply active for $name!" else "⏸️ Auto-Reply saved (disabled) for $name"
            Toast.makeText(context, statusMsg, Toast.LENGTH_LONG).show()
            dialog.dismiss()
        }

        dialog.show()
    }
}
