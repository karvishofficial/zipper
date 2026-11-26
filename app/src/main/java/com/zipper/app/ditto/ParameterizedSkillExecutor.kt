package com.zipper.app.ditto

import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.io.File

/**
 * Parameterized Universal Skill Engine.
 *
 * Implements 1-Shot Apprenticeship:
 * Allows user-demonstrated DittoFlows to be compiled into reusable, parameterized skills.
 *
 * Example:
 * 1. User records a flow: "Pay Bescom electricity bill"
 * 2. ParameterizedSkillExecutor extracts variable slots (e.g., 'consumer_id', 'amount')
 * 3. Next time, calling `executeSkill("pay_electricity", mapOf("consumer_id" to "12345678"))`
 *    replays the flow while dynamically substituting the parameters into input fields,
 *    with neural target re-grounding handling any UI layout shifts.
 *
 * 100% on-device, zero API cost.
 */
class ParameterizedSkillExecutor(private val context: Context) {

    companion object {
        private const val TAG = "ParamSkillExecutor"
        private const val SKILLS_DIR_NAME = "ditto_skills"
    }

    private val player = DittoPlayer(context)
    private val skillsDir = File(context.filesDir, SKILLS_DIR_NAME).apply { mkdirs() }

    data class SkillParameter(
        val name: String,
        val defaultValue: String,
        val stepIndex: Int
    )

    data class ParameterizedSkill(
        val skillId: String,
        val name: String,
        val description: String,
        val flow: DittoFlow,
        val parameters: List<SkillParameter>
    )

    /**
     * Compiles a demonstrated DittoFlow into a ParameterizedSkill.
     * Identifies all TYPE_TEXT steps and exposes them as parameterized slots.
     */
    fun compileFlowToSkill(
        flow: DittoFlow,
        skillName: String,
        description: String = ""
    ): ParameterizedSkill {
        val params = mutableListOf<SkillParameter>()

        flow.steps.forEach { step ->
            if (step.actionType == "TYPE_TEXT" && !step.argument.isNullOrBlank()) {
                val paramName = deriveParameterName(step)
                params.add(
                    SkillParameter(
                        name = paramName,
                        defaultValue = step.argument,
                        stepIndex = step.stepIndex
                    )
                )
            }
        }

        val skill = ParameterizedSkill(
            skillId = flow.flowId,
            name = skillName,
            description = description.ifBlank { "Parameterized skill for ${flow.title}" },
            flow = flow,
            parameters = params
        )

        saveSkill(skill)
        return skill
    }

    private fun deriveParameterName(step: DittoStep): String {
        val targetText = step.targetIdentity.primaryText?.lowercase() ?: ""
        val targetDesc = step.targetIdentity.contentDesc?.lowercase() ?: ""
        val resId = step.targetMetadata.resourceId?.lowercase() ?: ""

        return when {
            targetText.contains("search") || targetDesc.contains("search") || resId.contains("search") -> "query"
            targetText.contains("amount") || targetDesc.contains("amount") || resId.contains("amount") -> "amount"
            targetText.contains("recipient") || targetText.contains("contact") || resId.contains("recipient") -> "recipient"
            targetText.contains("message") || targetDesc.contains("message") -> "message"
            targetText.contains("number") || resId.contains("phone") || resId.contains("mobile") -> "phone_number"
            targetText.contains("id") || resId.contains("account") || resId.contains("consumer") -> "account_id"
            else -> "input_step_${step.stepIndex}"
        }
    }

    fun saveSkill(skill: ParameterizedSkill) {
        try {
            val file = File(skillsDir, "${skill.skillId}.json")
            val json = JSONObject().apply {
                put("skillId", skill.skillId)
                put("name", skill.name)
                put("description", skill.description)
                put("flow", skill.flow.toJson())
            }
            file.writeText(json.toString(2), Charsets.UTF_8)
            Log.i(TAG, "Saved parameterized skill '${skill.name}' (${skill.skillId})")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save skill", e)
        }
    }

    fun listSkills(): List<ParameterizedSkill> {
        val result = mutableListOf<ParameterizedSkill>()
        val files = skillsDir.listFiles { f -> f.extension == "json" } ?: return result

        for (f in files) {
            try {
                val json = JSONObject(f.readText(Charsets.UTF_8))
                val flow = DittoFlow.fromJson(json.getJSONObject("flow"))
                val name = json.getString("name")
                val desc = json.optString("description", "")
                result.add(compileFlowToSkill(flow, name, desc))
            } catch (e: Exception) {
                Log.w(TAG, "Error parsing skill file ${f.name}", e)
            }
        }
        return result
    }

    /**
     * Executes a skill with runtime parameters substituted into target slots.
     */
    fun executeSkill(
        skill: ParameterizedSkill,
        runtimeParameters: Map<String, String> = emptyMap(),
        onComplete: (Boolean, String) -> Unit
    ) {
        val adaptedSteps = skill.flow.steps.map { step ->
            if (step.actionType == "TYPE_TEXT") {
                val paramSlot = skill.parameters.find { it.stepIndex == step.stepIndex }
                val slotName = paramSlot?.name
                val overrideValue = if (slotName != null && runtimeParameters.containsKey(slotName)) {
                    runtimeParameters[slotName]
                } else if (runtimeParameters.size == 1 && skill.parameters.size == 1) {
                    runtimeParameters.values.first()
                } else null

                if (overrideValue != null) {
                    Log.i(TAG, "Substituting parameter for Step ${step.stepIndex} ($slotName): '$overrideValue'")
                    step.copy(argument = overrideValue)
                } else {
                    step
                }
            } else {
                step
            }
        }

        val adaptedFlow = skill.flow.copy(steps = adaptedSteps)
        player.replayFlow(
            flow = adaptedFlow,
            confidenceThreshold = 0.80f,
            recoveryScrollBudget = 4,
            onComplete = onComplete
        )
    }
}
