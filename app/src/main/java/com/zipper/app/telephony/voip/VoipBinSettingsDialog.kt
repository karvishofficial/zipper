package com.zipper.app.telephony.voip

import android.app.Dialog
import android.content.Context
import android.content.Intent
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.ColorDrawable
import android.graphics.drawable.GradientDrawable
import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.Window
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.Switch
import android.widget.TextView
import android.widget.Toast
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Premium OneUI 7 / Dark Slate aesthetic configuration dialog for VoIPBin Cloud Telephony.
 * Fully programmatic: zero missing XML layout risks, responsive, and tactile.
 */
class VoipBinSettingsDialog(
    private val context: Context,
    private val onConfigSaved: ((VoipConfig) -> Unit)? = null
) {

    private val scope = CoroutineScope(Dispatchers.Main)
    private val configManager = VoipConfigManager(context)

    fun show() {
        val currentConfig = configManager.getConfig()
        val dialog = Dialog(context)
        dialog.requestWindowFeature(Window.FEATURE_NO_TITLE)

        val density = context.resources.displayMetrics.density
        fun dp(v: Int): Int = (v * density).toInt()

        val rootScroll = ScrollView(context).apply {
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            )
            isFillViewport = true
            setBackgroundColor(Color.parseColor("#0F172A"))
        }

        val container = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(20), dp(20), dp(20))
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
        }
        rootScroll.addView(container)

        // --- 1. HEADER ---
        val headerRow = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
        }

        val tvHeader = TextView(context).apply {
            text = "📞 Zipper AI Telephony"
            textSize = 20f
            setTypeface(null, Typeface.BOLD)
            setTextColor(Color.WHITE)
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }

        val btnClose = Button(context).apply {
            text = "✕"
            textSize = 16f
            setTextColor(Color.parseColor("#94A3B8"))
            setBackgroundColor(Color.TRANSPARENT)
            setOnClickListener { dialog.dismiss() }
            layoutParams = LinearLayout.LayoutParams(dp(44), dp(44))
        }

        headerRow.addView(tvHeader)
        headerRow.addView(btnClose)
        container.addView(headerRow)

        val tvSub = TextView(context).apply {
            text = "Live Conversational AI Phone Calls powered by VoIPBin & Groq Llama 3.3"
            textSize = 12f
            setTextColor(Color.parseColor("#94A3B8"))
            setPadding(0, dp(4), 0, dp(16))
        }
        container.addView(tvSub)

        // --- 2. STATUS CARD ---
        val statusCard = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(14), dp(10), dp(14), dp(10))
            background = GradientDrawable().apply {
                setColor(Color.parseColor("#1E293B"))
                cornerRadius = dp(12).toFloat()
                setStroke(dp(1), Color.parseColor("#334155"))
            }
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { bottomMargin = dp(16) }
        }

        val tvStatusBadge = TextView(context).apply {
            text = if (currentConfig.isVerified) "🟢 VERIFIED & READY" else "🟠 SETUP REQUIRED"
            textSize = 12f
            setTypeface(null, Typeface.BOLD)
            setTextColor(if (currentConfig.isVerified) Color.parseColor("#34D399") else Color.parseColor("#F59E0B"))
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }

        val switchEnabled = Switch(context).apply {
            isChecked = currentConfig.isEnabled
        }

        statusCard.addView(tvStatusBadge)
        statusCard.addView(switchEnabled)
        container.addView(statusCard)

        // --- 3. INPUT FIELDS ---
        fun createLabel(title: String): TextView {
            return TextView(context).apply {
                text = title
                textSize = 12f
                setTypeface(null, Typeface.BOLD)
                setTextColor(Color.parseColor("#38BDF8"))
                setPadding(0, dp(8), 0, dp(4))
            }
        }

        fun createEditText(initialValue: String, hintText: String): EditText {
            return EditText(context).apply {
                setText(initialValue)
                hint = hintText
                setHintTextColor(Color.parseColor("#64748B"))
                setTextColor(Color.WHITE)
                textSize = 13f
                setPadding(dp(12), dp(10), dp(12), dp(10))
                background = GradientDrawable().apply {
                    setColor(Color.parseColor("#1E293B"))
                    cornerRadius = dp(8).toFloat()
                    setStroke(dp(1), Color.parseColor("#475569"))
                }
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply { bottomMargin = dp(8) }
            }
        }

        // --- 3A. QUICK LOGIN CARD ---
        val loginCard = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(14), dp(12), dp(14), dp(12))
            background = GradientDrawable().apply {
                setColor(Color.parseColor("#1E293B"))
                cornerRadius = dp(12).toFloat()
                setStroke(dp(1), Color.parseColor("#334155"))
            }
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { bottomMargin = dp(14) }
        }

        val tvLoginTitle = TextView(context).apply {
            text = "⚡ Auto-Fetch Token via Account Login"
            textSize = 13f
            setTypeface(null, Typeface.BOLD)
            setTextColor(Color.parseColor("#38BDF8"))
        }
        val tvLoginDesc = TextView(context).apply {
            text = "VoIPBin uses JWT tokens. Enter your username & password to fetch it automatically:"
            textSize = 11f
            setTextColor(Color.parseColor("#94A3B8"))
            setPadding(0, dp(2), 0, dp(8))
        }
        loginCard.addView(tvLoginTitle)
        loginCard.addView(tvLoginDesc)

        val etUsername = createEditText("", "VoIPBin Username")
        val etPassword = createEditText("", "VoIPBin Password").apply {
            inputType = android.text.InputType.TYPE_CLASS_TEXT or android.text.InputType.TYPE_TEXT_VARIATION_PASSWORD
        }
        loginCard.addView(etUsername)
        loginCard.addView(etPassword)

        val btnLogin = Button(context).apply {
            text = "🔑 Login & Fetch Token"
            textSize = 12f
            setTextColor(Color.WHITE)
            background = GradientDrawable().apply {
                setColor(Color.parseColor("#059669"))
                cornerRadius = dp(8).toFloat()
            }
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                dp(40)
            )
        }
        loginCard.addView(btnLogin)
        container.addView(loginCard)

        container.addView(createLabel("OR PASTE ACCESS TOKEN / API KEY"))
        val etApiKey = createEditText(currentConfig.apiKey, "Bearer Token or API Key...")
        container.addView(etApiKey)

        container.addView(createLabel("AI ASSISTANT ID"))
        val etAssistantId = createEditText(
            currentConfig.assistantId.ifBlank { VoipConfigManager.DEFAULT_ASSISTANT_ID },
            "VoIPBin Assistant UUID"
        )
        container.addView(etAssistantId)

        container.addView(createLabel("DIRECT SIP URI"))
        val etSipUri = createEditText(
            currentConfig.sipUri.ifBlank { VoipConfigManager.DEFAULT_SIP_URI },
            "sip:direct.c95b295237fe@sip.voipbin.net"
        )
        container.addView(etSipUri)

        // --- 4. CALL FORWARDING GUIDE CARD ---
        val guideCard = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(14), dp(12), dp(14), dp(12))
            background = GradientDrawable().apply {
                setColor(Color.parseColor("#1E1B4B"))
                cornerRadius = dp(12).toFloat()
                setStroke(dp(1), Color.parseColor("#4338CA"))
            }
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                topMargin = dp(8)
                bottomMargin = dp(16)
            }
        }

        val tvGuideTitle = TextView(context).apply {
            text = "💡 How to Receive Calls with Zipper:"
            textSize = 13f
            setTypeface(null, Typeface.BOLD)
            setTextColor(Color.parseColor("#A5B4FC"))
        }

        val tvGuideBody = TextView(context).apply {
            text = "1. Dial *67*<VoIPBin-Number># on your phone (Forward when busy).\n" +
                    "2. When someone calls your normal SIM and you press Decline, telecom forwards to Zipper.\n" +
                    "3. Zipper talks to them using Groq Llama 3.3 & alerts you on-screen!"
            textSize = 11.5f
            setTextColor(Color.parseColor("#CBD5E1"))
            setPadding(0, dp(4), 0, 0)
        }

        guideCard.addView(tvGuideTitle)
        guideCard.addView(tvGuideBody)
        container.addView(guideCard)

        // --- 5. ACTION BUTTONS ---
        val btnRow = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
        }

        val btnTest = Button(context).apply {
            text = "Test Connection"
            setTextColor(Color.parseColor("#38BDF8"))
            background = GradientDrawable().apply {
                setColor(Color.parseColor("#0369A1"))
                cornerRadius = dp(8).toFloat()
            }
            layoutParams = LinearLayout.LayoutParams(0, dp(44), 1f).apply {
                marginEnd = dp(8)
            }
        }

        val btnSave = Button(context).apply {
            text = "Save & Activate"
            setTextColor(Color.WHITE)
            background = GradientDrawable().apply {
                setColor(Color.parseColor("#2563EB"))
                cornerRadius = dp(8).toFloat()
            }
            layoutParams = LinearLayout.LayoutParams(0, dp(44), 1f)
        }

        btnRow.addView(btnTest)
        btnRow.addView(btnSave)
        container.addView(btnRow)

        // --- HANDLERS ---
        btnLogin.setOnClickListener {
            val u = etUsername.text.toString().trim()
            val p = etPassword.text.toString().trim()
            if (u.isBlank() || p.isBlank()) {
                Toast.makeText(context, "Please enter your VoIPBin username and password", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            btnLogin.text = "Logging in..."
            btnLogin.isEnabled = false

            val client = VoipBinClient(currentConfig)
            scope.launch {
                val loginResult = withContext(Dispatchers.IO) { client.login(u, p) }
                btnLogin.isEnabled = true
                btnLogin.text = "🔑 Login & Fetch Token"

                loginResult.onSuccess { token ->
                    etApiKey.setText(token)
                    tvStatusBadge.text = "🟢 TOKEN ACQUIRED & READY"
                    tvStatusBadge.setTextColor(Color.parseColor("#34D399"))
                    switchEnabled.isChecked = true
                    Toast.makeText(context, "✅ Successfully logged in! Token acquired.", Toast.LENGTH_LONG).show()
                }.onFailure { err ->
                    Toast.makeText(context, "❌ Login failed: ${err.message}", Toast.LENGTH_LONG).show()
                }
            }
        }

        btnTest.setOnClickListener {
            val key = etApiKey.text.toString().trim()
            val astId = etAssistantId.text.toString().trim()

            if (key.isBlank()) {
                Toast.makeText(context, "Please enter your VoIPBin API key first.", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            btnTest.text = "Testing..."
            btnTest.isEnabled = false

            val testConfig = currentConfig.copy(
                apiKey = key,
                assistantId = astId.ifBlank { VoipConfigManager.DEFAULT_ASSISTANT_ID }
            )
            val client = VoipBinClient(testConfig)

            scope.launch {
                val result = withContext(Dispatchers.IO) { client.verifyConnection() }
                btnTest.isEnabled = true
                btnTest.text = "Test Connection"

                result.onSuccess { msg ->
                    tvStatusBadge.text = "🟢 VERIFIED & READY"
                    tvStatusBadge.setTextColor(Color.parseColor("#34D399"))
                    Toast.makeText(context, "✅ $msg", Toast.LENGTH_LONG).show()
                }.onFailure { err ->
                    tvStatusBadge.text = "❌ CONNECTION FAILED"
                    tvStatusBadge.setTextColor(Color.parseColor("#EF4444"))
                    Toast.makeText(context, "❌ ${err.message}", Toast.LENGTH_LONG).show()
                }
            }
        }

        btnSave.setOnClickListener {
            val updatedConfig = currentConfig.copy(
                apiKey = etApiKey.text.toString().trim(),
                assistantId = etAssistantId.text.toString().trim().ifBlank { VoipConfigManager.DEFAULT_ASSISTANT_ID },
                sipUri = etSipUri.text.toString().trim().ifBlank { VoipConfigManager.DEFAULT_SIP_URI },
                isEnabled = switchEnabled.isChecked,
                isVerified = tvStatusBadge.text.toString().contains("VERIFIED"),
                lastVerified = System.currentTimeMillis()
            )

            configManager.saveConfig(updatedConfig)
            onConfigSaved?.invoke(updatedConfig)

            val modeMsg = if (updatedConfig.isEnabled) "📞 Zipper Cloud AI Telephony Active!" else "⏸️ VoIP Telephony Saved (Disabled)"
            Toast.makeText(context, modeMsg, Toast.LENGTH_LONG).show()
            dialog.dismiss()
        }

        dialog.setContentView(rootScroll)
        dialog.window?.setBackgroundDrawable(ColorDrawable(Color.TRANSPARENT))
        dialog.window?.setLayout(
            (context.resources.displayMetrics.widthPixels * 0.94).toInt(),
            ViewGroup.LayoutParams.WRAP_CONTENT
        )
        dialog.show()
    }
}
