package com.zipper.app.core.web

import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import kotlinx.coroutines.delay
import org.json.JSONArray
import org.json.JSONObject

data class WebMacroStep(
    val actionType: String, // "TYPE", "CLICK", "SELECT", "SUBMIT"
    val selector: String,   // CSS selector, e.g. "input#usn", "button.submit"
    val value: String? = null,
    val expectedLabel: String? = null
)

data class WebMacro(
    val id: String,
    val title: String,
    val targetUrlPattern: String,
    val steps: List<WebMacroStep>,
    val createdAt: Long = System.currentTimeMillis()
)

data class WebMacroExecutionResult(
    val success: Boolean,
    val stepsCompleted: Int,
    val failedStepIndex: Int? = null,
    val failedSelector: String? = null,
    val summary: String
)

/**
 * Self-Healing Web Macro Engine (Ditto Web 2.0).
 *
 * Enables 1-Tap deterministic execution of routine multi-step web flows:
 * - AI: OFF: Replays pure CSS/DOM selectors via JavaScript in <300ms (0 tokens).
 * - AI: ON: If a selector breaks due to website redesign, acts as an auto-repair
 *           mechanic to re-ground the missing element and update the saved recipe.
 */
class WebMacroEngine(private val context: Context) {

    companion object {
        private const val TAG = "WebMacroEngine"
        private const val PREFS_NAME = "karvish_web_macros"
        private const val KEY_MACRO_LIST = "macro_recipes_json"
    }

    private val prefs: SharedPreferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun saveMacro(macro: WebMacro) {
        val existing = getAllMacros().filter { it.id != macro.id }.toMutableList()
        existing.add(0, macro)
        persistMacros(existing)
        Log.i(TAG, "Saved web macro '${macro.title}' with ${macro.steps.size} steps")
    }

    fun getAllMacros(): List<WebMacro> {
        val raw = prefs.getString(KEY_MACRO_LIST, null) ?: return emptyList()
        return try {
            val arr = JSONArray(raw)
            val list = mutableListOf<WebMacro>()
            for (i in 0 until arr.length()) {
                val obj = arr.getJSONObject(i)
                val stepsArr = obj.getJSONArray("steps")
                val steps = mutableListOf<WebMacroStep>()
                for (j in 0 until stepsArr.length()) {
                    val sObj = stepsArr.getJSONObject(j)
                    steps.add(
                        WebMacroStep(
                            actionType = sObj.getString("actionType"),
                            selector = sObj.getString("selector"),
                            value = sObj.optString("value").takeIf { it.isNotBlank() },
                            expectedLabel = sObj.optString("expectedLabel").takeIf { it.isNotBlank() }
                        )
                    )
                }
                list.add(
                    WebMacro(
                        id = obj.getString("id"),
                        title = obj.getString("title"),
                        targetUrlPattern = obj.getString("targetUrlPattern"),
                        steps = steps,
                        createdAt = obj.optLong("createdAt", System.currentTimeMillis())
                    )
                )
            }
            list
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse stored macros", e)
            emptyList()
        }
    }

    fun findMacroForUrl(url: String): WebMacro? {
        val lowerUrl = url.lowercase()
        return getAllMacros().firstOrNull { macro ->
            val pattern = macro.targetUrlPattern.lowercase()
            lowerUrl.contains(pattern) || pattern.contains(lowerUrl)
        }
    }

    private fun persistMacros(macros: List<WebMacro>) {
        val arr = JSONArray()
        for (m in macros) {
            val obj = JSONObject()
            obj.put("id", m.id)
            obj.put("title", m.title)
            obj.put("targetUrlPattern", m.targetUrlPattern)
            obj.put("createdAt", m.createdAt)
            val stepsArr = JSONArray()
            for (s in m.steps) {
                val sObj = JSONObject()
                sObj.put("actionType", s.actionType)
                sObj.put("selector", s.selector)
                s.value?.let { sObj.put("value", it) }
                s.expectedLabel?.let { sObj.put("expectedLabel", it) }
                stepsArr.put(sObj)
            }
            obj.put("steps", stepsArr)
            arr.put(obj)
        }
        prefs.edit().putString(KEY_MACRO_LIST, arr.toString()).apply()
    }

    /**
     * Executes a macro deterministically on the provided web controller.
     * Evaluates selectors directly in JavaScript for maximum speed (<300ms).
     */
    suspend fun executeMacro(controller: HiddenWebController, macro: WebMacro): WebMacroExecutionResult {
        Log.i(TAG, "Executing macro '${macro.title}' (${macro.steps.size} steps)")
        var completed = 0

        for ((idx, step) in macro.steps.withIndex()) {
            val safeSelector = step.selector.replace("'", "\\'")
            val safeVal = (step.value ?: "").replace("'", "\\'")

            val jsCheck = "!!document.querySelector('$safeSelector')"
            val exists = controller.evaluateJs(jsCheck).replace("\"", "").toBoolean()

            if (!exists) {
                Log.w(TAG, "Macro step ${idx + 1} failed: Selector '$safeSelector' not found in DOM")
                return WebMacroExecutionResult(
                    success = false,
                    stepsCompleted = completed,
                    failedStepIndex = idx,
                    failedSelector = step.selector,
                    summary = "Step ${idx + 1} failed: Element not found (${step.expectedLabel ?: step.selector})"
                )
            }

            when (step.actionType) {
                "TYPE" -> {
                    val jsType = """
                        (function() {
                            const el = document.querySelector('$safeSelector');
                            if (el) {
                                el.value = '$safeVal';
                                el.dispatchEvent(new Event('input', { bubbles: true }));
                                el.dispatchEvent(new Event('change', { bubbles: true }));
                                return 'true';
                            }
                            return 'false';
                        })();
                    """.trimIndent()
                    controller.evaluateJs(jsType)
                }
                "CLICK" -> {
                    val jsClick = """
                        (function() {
                            const el = document.querySelector('$safeSelector');
                            if (el) { el.click(); return 'true'; }
                            return 'false';
                        })();
                    """.trimIndent()
                    controller.evaluateJs(jsClick)
                }
                "SELECT" -> {
                    val jsSelect = """
                        (function() {
                            const el = document.querySelector('$safeSelector');
                            if (el) {
                                el.value = '$safeVal';
                                el.dispatchEvent(new Event('change', { bubbles: true }));
                                return 'true';
                            }
                            return 'false';
                        })();
                    """.trimIndent()
                    controller.evaluateJs(jsSelect)
                }
                "SUBMIT" -> {
                    controller.submit()
                }
            }
            completed++
            delay(150L)
        }

        return WebMacroExecutionResult(
            success = true,
            stepsCompleted = completed,
            summary = "Successfully executed all $completed macro steps for '${macro.title}'"
        )
    }

    /**
     * Self-Healing Helper: Updates a broken step selector with the newly grounded selector.
     */
    fun healStep(macroId: String, stepIndex: Int, newSelector: String) {
        val all = getAllMacros().toMutableList()
        val targetIdx = all.indexOfFirst { it.id == macroId }
        if (targetIdx != -1) {
            val macro = all[targetIdx]
            if (stepIndex in macro.steps.indices) {
                val updatedSteps = macro.steps.toMutableList()
                val oldStep = updatedSteps[stepIndex]
                updatedSteps[stepIndex] = oldStep.copy(selector = newSelector)
                all[targetIdx] = macro.copy(steps = updatedSteps)
                persistMacros(all)
                Log.i(TAG, "Self-healed macro '${macro.title}' step $stepIndex with new selector: $newSelector")
            }
        }
    }
}
