package com.zipper.app.core.teach

import android.content.Context
import android.util.Log
import com.zipper.app.core.model.NormalizedBounds
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

class TeachCompiler(private val context: Context) {

    companion object {
        private const val TAG = "KarvishTeachCompiler"
        private const val FILE_NAME = "learned_alarm_skill.json"
    }

    private val storageFile: File get() = File(context.filesDir, FILE_NAME)

    fun isSkillTaught(): Boolean {
        return storageFile.exists() && loadSkill()?.isTaught == true
    }

    fun compile(interactions: List<RecordedInteraction>): LearnedAlarmSkill {
        Log.i(TAG, "Compiling ${interactions.size} recorded demonstration interactions...")

        var addBtnBounds: NormalizedBounds? = null
        var saveBtnBounds: NormalizedBounds? = null
        var hourBounds: NormalizedBounds? = null
        var minuteBounds: NormalizedBounds? = null
        var amBounds: NormalizedBounds? = null
        var pmBounds: NormalizedBounds? = null

        for (i in interactions) {
            // ONLY consider CLICK events (ignore scroll containers, window state changes, etc.)
            val isClick = i.eventType == 1 || i.eventTypeName.equals("CLICK", ignoreCase = true)
            if (!isClick) continue

            val text = i.text?.lowercase() ?: ""
            val desc = i.contentDescription?.lowercase() ?: ""
            val id = i.viewIdResourceName?.lowercase() ?: ""
            val cls = i.className?.lowercase() ?: ""

            // Skip large containers, lists, RecyclerViews, or whole-page layouts
            if (cls.contains("recyclerview") || cls.contains("scrollview") || cls.contains("listview") ||
                cls.contains("viewpager") || i.boundsNorm.width > 450 || i.boundsNorm.height > 350
            ) {
                continue
            }

            // Identify Add Alarm: Must be an explicit add alarm action, NOT "additional actions"
            val isAddAlarm = id.contains("menu_alarm_add") || id.contains("add_alarm") ||
                    desc.contains("add alarm") || desc == "add" || desc == "+" ||
                    text == "+" || (desc.split("\\s+".toRegex()).contains("add") && !desc.contains("additional"))

            if (addBtnBounds == null && isAddAlarm) {
                addBtnBounds = i.boundsNorm
                Log.i(TAG, "Identified Add Alarm target: bounds=${i.boundsNorm}")
                continue
            }

            // Identify Save
            if (text.contains("save") || desc.contains("save") || id.contains("save") || text.contains("done") || desc.contains("done")) {
                saveBtnBounds = i.boundsNorm
                Log.i(TAG, "Identified Save target: bounds=${i.boundsNorm}")
                continue
            }

            // Identify AM
            if (text == "am" || desc == "am") {
                amBounds = i.boundsNorm
                Log.i(TAG, "Identified AM target: bounds=${i.boundsNorm}")
                continue
            }

            // Identify PM
            if (text == "pm" || desc == "pm") {
                pmBounds = i.boundsNorm
                Log.i(TAG, "Identified PM target: bounds=${i.boundsNorm}")
                continue
            }

            // Identify Hour or Minute picker wheel/input
            // Usually in screen top half (Y < 550)
            if (i.boundsCenterNormY < 550) {
                // If X is on the left third of screen (X < 420) -> Hour
                if (i.boundsCenterNormX < 420) {
                    hourBounds = i.boundsNorm
                    Log.i(TAG, "Identified Hour wheel/target: bounds=${i.boundsNorm}")
                }
                // If X is in middle third (420 <= X <= 720) -> Minute
                else if (i.boundsCenterNormX in 420..720) {
                    minuteBounds = i.boundsNorm
                    Log.i(TAG, "Identified Minute wheel/target: bounds=${i.boundsNorm}")
                }
                // If X > 720 and not yet marked -> AM/PM area
                else if (i.boundsCenterNormX > 720) {
                    if (amBounds == null) amBounds = i.boundsNorm
                }
            }
        }

        // If addBtnBounds wasn't explicitly matched, look through interactions for any button with Add
        if (addBtnBounds == null) {
            val candidateAdd = interactions.firstOrNull {
                val isClick = it.eventType == 1 || it.eventTypeName.equals("CLICK", ignoreCase = true)
                val id = it.viewIdResourceName?.lowercase() ?: ""
                val desc = it.contentDescription?.lowercase() ?: ""
                val text = it.text?.lowercase() ?: ""
                isClick && it.boundsNorm.width < 400 && it.boundsNorm.height < 300 &&
                        (id.contains("add") || desc.contains("add alarm") || text == "+")
            }
            if (candidateAdd != null) {
                addBtnBounds = candidateAdd.boundsNorm
                Log.i(TAG, "Recovered Add Alarm bounds from interactions: $addBtnBounds")
            }
        }

        // If saveBtnBounds wasn't explicitly named, the last small clickable element is Save
        if (saveBtnBounds == null) {
            val candidateSave = interactions.lastOrNull {
                val isClick = it.eventType == 1 || it.eventTypeName.equals("CLICK", ignoreCase = true)
                isClick && it.boundsNorm.width < 450 && it.boundsNorm.height < 250 && it.boundsCenterNormY > 700
            }
            if (candidateSave != null) {
                saveBtnBounds = candidateSave.boundsNorm
            }
        }

        val skill = LearnedAlarmSkill(
            isTaught = true,
            taughtTimestamp = System.currentTimeMillis(),
            hourPickerBounds = hourBounds,
            minutePickerBounds = minuteBounds,
            amBounds = amBounds,
            pmBounds = pmBounds,
            addAlarmButtonBounds = addBtnBounds,
            saveButtonBounds = saveBtnBounds,
            recordedInteractions = interactions
        )

        saveSkill(skill)
        return skill
    }

    fun saveSkill(skill: LearnedAlarmSkill) {
        try {
            val json = JSONObject().apply {
                put("isTaught", skill.isTaught)
                put("taughtTimestamp", skill.taughtTimestamp)
                put("deviceModel", skill.deviceModel)
                put("clockPackage", skill.clockPackage)

                skill.hourPickerBounds?.let { put("hourBounds", boundsToJson(it)) }
                skill.minutePickerBounds?.let { put("minuteBounds", boundsToJson(it)) }
                skill.amBounds?.let { put("amBounds", boundsToJson(it)) }
                skill.pmBounds?.let { put("pmBounds", boundsToJson(it)) }
                skill.addAlarmButtonBounds?.let { put("addBounds", boundsToJson(it)) }
                skill.saveButtonBounds?.let { put("saveBounds", boundsToJson(it)) }

                val interactionsArr = JSONArray()
                for (it in skill.recordedInteractions) {
                    interactionsArr.put(JSONObject().apply {
                        put("eventType", it.eventType)
                        put("eventTypeName", it.eventTypeName)
                        put("className", it.className)
                        put("viewIdResourceName", it.viewIdResourceName)
                        put("text", it.text)
                        put("contentDescription", it.contentDescription)
                        put("centerX", it.boundsCenterNormX)
                        put("centerY", it.boundsCenterNormY)
                        put("bounds", boundsToJson(it.boundsNorm))
                    })
                }
                put("interactions", interactionsArr)
            }
            storageFile.writeText(json.toString(2))
            Log.i(TAG, "Successfully saved learned skill to ${storageFile.absolutePath}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save learned skill", e)
        }
    }

    fun loadSkill(): LearnedAlarmSkill? {
        if (!storageFile.exists()) return null
        return try {
            val json = JSONObject(storageFile.readText())
            var addBounds = json.optJSONObject("addBounds")?.let { jsonToBounds(it) }
            val saveBounds = json.optJSONObject("saveBounds")?.let { jsonToBounds(it) }

            // Validate addBounds - if it was stored as a whole-screen/list container (> 40% width or height), reject it
            if (addBounds != null && (addBounds.width > 400 || addBounds.height > 350)) {
                Log.w(TAG, "Stored addBounds was invalid container ($addBounds). Re-extracting from recorded interactions...")
                addBounds = null
            }

            // Search recorded interactions for the real Add button click (e.g. menu_alarm_add)
            if (addBounds == null && json.has("interactions")) {
                val arr = json.getJSONArray("interactions")
                for (idx in 0 until arr.length()) {
                    val obj = arr.getJSONObject(idx)
                    val evType = obj.optInt("eventType", 0)
                    val evTypeName = obj.optString("eventTypeName", "")
                    val id = obj.optString("viewIdResourceName", "").lowercase()
                    val desc = obj.optString("contentDescription", "").lowercase()
                    val text = obj.optString("text", "").lowercase()
                    val cls = obj.optString("className", "").lowercase()

                    val isClick = evType == 1 || evTypeName.equals("CLICK", ignoreCase = true)
                    val isNotContainer = !cls.contains("recyclerview") && !cls.contains("scrollview")
                    val isAdd = id.contains("menu_alarm_add") || id.contains("add_alarm") || desc.contains("add alarm") || text == "+"

                    if (isClick && isNotContainer && isAdd) {
                        val b = obj.getJSONObject("bounds")
                        val candBounds = jsonToBounds(b)
                        if (candBounds.width < 400 && candBounds.height < 300) {
                            addBounds = candBounds
                            Log.i(TAG, "Successfully repaired addBounds from recorded interactions: $addBounds")
                            break
                        }
                    }
                }
            }

            // Samsung One UI top-right Add button default if still null
            if (addBounds == null && json.optString("clockPackage", "").contains("clockpackage")) {
                addBounds = NormalizedBounds(left = 750, top = 50, right = 880, bottom = 130)
                Log.i(TAG, "Applied Samsung One UI standard Add button bounds: $addBounds")
            }

            LearnedAlarmSkill(
                isTaught = json.optBoolean("isTaught", false),
                taughtTimestamp = json.optLong("taughtTimestamp", 0L),
                deviceModel = json.optString("deviceModel", ""),
                clockPackage = json.optString("clockPackage", "com.sec.android.app.clockpackage"),
                hourPickerBounds = json.optJSONObject("hourBounds")?.let { jsonToBounds(it) },
                minutePickerBounds = json.optJSONObject("minuteBounds")?.let { jsonToBounds(it) },
                amBounds = json.optJSONObject("amBounds")?.let { jsonToBounds(it) },
                pmBounds = json.optJSONObject("pmBounds")?.let { jsonToBounds(it) },
                addAlarmButtonBounds = addBounds,
                saveButtonBounds = saveBounds
            )
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load learned skill", e)
            null
        }
    }

    fun clearSkill() {
        if (storageFile.exists()) {
            storageFile.delete()
        }
    }

    private fun boundsToJson(b: NormalizedBounds): JSONObject {
        return JSONObject().apply {
            put("left", b.left)
            put("top", b.top)
            put("right", b.right)
            put("bottom", b.bottom)
        }
    }

    private fun jsonToBounds(j: JSONObject): NormalizedBounds {
        return NormalizedBounds(
            left = j.getInt("left"),
            top = j.getInt("top"),
            right = j.getInt("right"),
            bottom = j.getInt("bottom")
        )
    }
}
