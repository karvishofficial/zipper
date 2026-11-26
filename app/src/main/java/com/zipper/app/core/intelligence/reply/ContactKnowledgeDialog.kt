package com.zipper.app.core.intelligence.reply

import android.app.Dialog
import android.content.Context
import android.graphics.Color
import android.graphics.drawable.ColorDrawable
import android.view.LayoutInflater
import android.view.View
import android.view.Window
import android.widget.Button
import android.widget.EditText
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import com.zipper.app.R

/**
 * Controller for the Contact Knowledge Book dialog.
 * Enables users to view, manage, and edit personal dossiers, relationship metadata,
 * language adaptation styles, and memory points for any contact.
 */
class ContactKnowledgeDialog(
    private val context: Context,
    private val initialContactName: String? = null,
    private val onDismiss: (() -> Unit)? = null
) {

    private val knowledgeBook = ContactKnowledgeBook.getInstance(context)
    private var currentDossiers = knowledgeBook.getAllDossiers().toMutableList()
    private var selectedDossier: ContactDossier? = null
    private var activeMemoryPoints = mutableListOf<String>()
    private var activeLanguageStyle = LanguageStyle.CASUAL_SLANG

    fun show() {
        val dialog = Dialog(context)
        dialog.requestWindowFeature(Window.FEATURE_NO_TITLE)

        val view = LayoutInflater.from(context).inflate(R.layout.dialog_contact_knowledge, null)
        dialog.setContentView(view)
        dialog.window?.setBackgroundDrawable(ColorDrawable(Color.TRANSPARENT))
        dialog.window?.setLayout(
            (context.resources.displayMetrics.widthPixels * 0.94).toInt(),
            (context.resources.displayMetrics.heightPixels * 0.88).toInt()
        )

        val btnClose = view.findViewById<ImageButton>(R.id.btnCloseKnowledgeDialog)
        val llContactChips = view.findViewById<LinearLayout>(R.id.llDossierContactChips)
        val btnAddContactChip = view.findViewById<TextView>(R.id.btnAddNewDossierChip)

        val etContactName = view.findViewById<EditText>(R.id.etDossierContactName)
        val etRelationship = view.findViewById<EditText>(R.id.etDossierRelationship)
        val tvLanguageStyle = view.findViewById<TextView>(R.id.tvDossierLanguageStyle)
        val btnCycleLanguageStyle = view.findViewById<TextView>(R.id.btnCycleLanguageStyle)

        val llMemoryPointsList = view.findViewById<LinearLayout>(R.id.llMemoryPointsList)
        val tvEmptyMemoryPoints = view.findViewById<TextView>(R.id.tvEmptyMemoryPoints)
        val etNewMemoryFact = view.findViewById<EditText>(R.id.etNewMemoryFact)
        val btnAddMemoryFact = view.findViewById<TextView>(R.id.btnAddMemoryFact)

        val btnDeleteDossier = view.findViewById<Button>(R.id.btnDeleteDossier)
        val btnSaveDossier = view.findViewById<Button>(R.id.btnSaveDossier)

        fun getStyleDisplayName(style: LanguageStyle): String {
            return when (style) {
                LanguageStyle.CASUAL_SLANG -> "⚡ Bangalore Casual Slang & Kanglish"
                LanguageStyle.BROKEN_BASIC_ENGLISH -> "💬 Basic / Broken English"
                LanguageStyle.CASUAL_SHORT -> "⏱️ Ultra-Short (2-4 words)"
                LanguageStyle.FORMAL_ENGLISH -> "👔 Formal English (Respectful)"
                LanguageStyle.KANNADA_REGIONAL -> "🌾 Kannada / Regional (Mother / Elders)"
            }
        }

        fun renderMemoryPoints() {
            llMemoryPointsList.removeAllViews()
            if (activeMemoryPoints.isEmpty()) {
                tvEmptyMemoryPoints.visibility = View.VISIBLE
            } else {
                tvEmptyMemoryPoints.visibility = View.GONE
                for (point in activeMemoryPoints) {
                    val itemView = LayoutInflater.from(context).inflate(R.layout.item_knowledge_memory_point, llMemoryPointsList, false)
                    val tvText = itemView.findViewById<TextView>(R.id.tvMemoryPointText)
                    val btnDelete = itemView.findViewById<ImageButton>(R.id.btnDeleteMemoryPoint)

                    tvText.text = point
                    btnDelete.setOnClickListener {
                        activeMemoryPoints.remove(point)
                        renderMemoryPoints()
                    }
                    llMemoryPointsList.addView(itemView)
                }
            }
        }

        fun populateDossierUi(dossier: ContactDossier?) {
            selectedDossier = dossier
            if (dossier != null) {
                etContactName.setText(dossier.contactName)
                etRelationship.setText(dossier.relationship)
                activeLanguageStyle = dossier.languageStyle
                tvLanguageStyle.text = getStyleDisplayName(dossier.languageStyle)
                activeMemoryPoints = dossier.memoryPoints.toMutableList()
                btnDeleteDossier.visibility = View.VISIBLE
            } else {
                etContactName.setText("")
                etRelationship.setText("Friend")
                activeLanguageStyle = LanguageStyle.CASUAL_SLANG
                tvLanguageStyle.text = getStyleDisplayName(LanguageStyle.CASUAL_SLANG)
                activeMemoryPoints = mutableListOf()
                btnDeleteDossier.visibility = View.GONE
            }
            renderMemoryPoints()
        }

        fun refreshContactChips() {
            while (llContactChips.childCount > 1) {
                llContactChips.removeViewAt(1)
            }
            currentDossiers = knowledgeBook.getAllDossiers().toMutableList()

            for (d in currentDossiers) {
                val chip = TextView(context).apply {
                    val isSelected = selectedDossier?.contactName.equals(d.contactName, ignoreCase = true)
                    text = "${d.contactName} (${d.memoryPoints.size})"
                    textSize = 12f
                    setPadding(32, 16, 32, 16)
                    val layoutParams = LinearLayout.LayoutParams(
                        LinearLayout.LayoutParams.WRAP_CONTENT,
                        LinearLayout.LayoutParams.WRAP_CONTENT
                    ).apply {
                        marginEnd = 16
                    }
                    this.layoutParams = layoutParams
                    setBackgroundResource(R.drawable.bg_pill_clean_blue)
                    backgroundTintList = android.content.res.ColorStateList.valueOf(
                        if (isSelected) Color.parseColor("#0284C7") else Color.parseColor("#1E293B")
                    )
                    setTextColor(Color.parseColor("#F8FAFC"))
                    setOnClickListener {
                        populateDossierUi(d)
                        refreshContactChips()
                    }
                }
                llContactChips.addView(chip)
            }
        }

        btnAddContactChip.setOnClickListener {
            populateDossierUi(null)
            refreshContactChips()
        }

        btnCycleLanguageStyle.setOnClickListener {
            val allStyles = LanguageStyle.values()
            val nextIdx = (activeLanguageStyle.ordinal + 1) % allStyles.size
            activeLanguageStyle = allStyles[nextIdx]
            tvLanguageStyle.text = getStyleDisplayName(activeLanguageStyle)
        }

        btnAddMemoryFact.setOnClickListener {
            val fact = etNewMemoryFact.text.toString().trim()
            if (fact.isNotBlank()) {
                if (!activeMemoryPoints.contains(fact)) {
                    activeMemoryPoints.add(fact)
                    renderMemoryPoints()
                }
                etNewMemoryFact.setText("")
            } else {
                Toast.makeText(context, "Please enter a fact to remember", Toast.LENGTH_SHORT).show()
            }
        }

        btnSaveDossier.setOnClickListener {
            val name = etContactName.text.toString().trim()
            if (name.isBlank()) {
                Toast.makeText(context, "Contact name cannot be empty", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            val rel = etRelationship.text.toString().trim().ifBlank { "Friend" }

            val updatedDossier = ContactDossier(
                contactName = name,
                relationship = rel,
                memoryPoints = activeMemoryPoints.toList(),
                languageStyle = activeLanguageStyle,
                recentHistory = selectedDossier?.recentHistory ?: emptyList()
            )
            knowledgeBook.saveDossier(updatedDossier)
            Toast.makeText(context, "✅ Dossier saved for $name (${activeMemoryPoints.size} facts)", Toast.LENGTH_SHORT).show()
            populateDossierUi(updatedDossier)
            refreshContactChips()
        }

        btnDeleteDossier.setOnClickListener {
            val d = selectedDossier ?: return@setOnClickListener
            // Remove from list and persist
            val all = knowledgeBook.getAllDossiers().filterNot { it.contactName.equals(d.contactName, ignoreCase = true) }
            val prefs = context.getSharedPreferences("zipper_contact_knowledge_prefs", Context.MODE_PRIVATE)
            val arr = org.json.JSONArray()
            all.forEach { arr.put(it.toJson()) }
            prefs.edit().putString("contact_dossiers_json", arr.toString()).apply()

            Toast.makeText(context, "Deleted dossier for ${d.contactName}", Toast.LENGTH_SHORT).show()
            populateDossierUi(null)
            refreshContactChips()
        }

        btnClose.setOnClickListener {
            dialog.dismiss()
        }

        dialog.setOnDismissListener {
            onDismiss?.invoke()
        }

        // Initialize state
        val targetDossier = if (!initialContactName.isNullOrBlank()) {
            knowledgeBook.getDossier(initialContactName) ?: ContactDossier(contactName = initialContactName)
        } else {
            currentDossiers.firstOrNull()
        }

        populateDossierUi(targetDossier)
        refreshContactChips()

        dialog.show()
    }
}
