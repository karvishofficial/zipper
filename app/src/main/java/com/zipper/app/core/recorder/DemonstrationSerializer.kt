package com.zipper.app.core.recorder

import com.zipper.app.core.agent.AgentAction
import com.zipper.app.core.agent.ScrollDirection
import com.zipper.app.core.model.DistilledNode
import com.zipper.app.core.model.DistilledUiState
import com.zipper.app.core.model.NormalizedBounds
import org.json.JSONArray
import org.json.JSONObject

/**
 * Serializes and deserializes Demonstration trajectories and Training examples to/from JSON and JSONL.
 */
class DemonstrationSerializer {

    // =========================================================================
    // Trajectory Serialization
    // =========================================================================

    fun serializeTrajectoryToJson(trajectory: DemonstrationTrajectory): JSONObject {
        val root = JSONObject()

        // Metadata
        val meta = JSONObject().apply {
            put("demonstrationId", trajectory.metadata.demonstrationId)
            put("schemaVersion", trajectory.metadata.schemaVersion)
            put("createdAt", trajectory.metadata.createdAt)
            put("deviceModel", trajectory.metadata.deviceModel)
            put("androidApiLevel", trajectory.metadata.androidApiLevel)
            put("targetAppPackage", trajectory.metadata.targetAppPackage ?: JSONObject.NULL)
        }
        root.put("metadata", meta)

        root.put("goal", trajectory.goal)
        root.put("totalSteps", trajectory.totalSteps)
        root.put("isSuccess", trajectory.isSuccess)
        root.put("finalOutcome", trajectory.finalOutcome)

        // Steps array
        val stepsArray = JSONArray()
        trajectory.steps.forEach { step ->
            stepsArray.put(serializeStepToJson(step))
        }
        root.put("steps", stepsArray)

        return root
    }

    fun serializeStepToJson(step: RecordedStep): JSONObject {
        val obj = JSONObject()
        obj.put("stepIndex", step.stepIndex)
        obj.put("timestamp", step.timestamp)
        obj.put("goal", step.goal)
        obj.put("packageName", step.packageName ?: JSONObject.NULL)
        obj.put("activityName", step.activityName ?: JSONObject.NULL)
        obj.put("preObservation", serializeUiStateToJson(step.preObservation))

        val toolsArray = JSONArray()
        step.availableTools.forEach { tool ->
            toolsArray.put(JSONObject().apply {
                put("name", tool.name)
                put("description", tool.description)
                put("parametersJson", tool.parametersJson)
            })
        }
        obj.put("availableTools", toolsArray)

        obj.put("humanAction", serializeActionToJson(step.humanAction))

        val argsObj = JSONObject()
        step.actionArguments.forEach { (k, v) -> argsObj.put(k, v) }
        obj.put("actionArguments", argsObj)

        obj.put("postObservation", serializeUiStateToJson(step.postObservation))
        obj.put("isStateChanged", step.isStateChanged)
        obj.put("isSuccessful", step.isSuccessful)
        obj.put("deltaSummary", step.deltaSummary)
        obj.put("failureRecoveryInfo", step.failureRecoveryInfo ?: JSONObject.NULL)

        return obj
    }

    fun deserializeTrajectoryFromJson(json: JSONObject): DemonstrationTrajectory {
        val metaObj = json.getJSONObject("metadata")
        val metadata = DemonstrationMetadata(
            demonstrationId = metaObj.getString("demonstrationId"),
            schemaVersion = metaObj.getString("schemaVersion"),
            createdAt = metaObj.getLong("createdAt"),
            deviceModel = metaObj.optString("deviceModel", "Unknown"),
            androidApiLevel = metaObj.optInt("androidApiLevel", 34),
            targetAppPackage = if (metaObj.isNull("targetAppPackage")) null else metaObj.optString("targetAppPackage")
        )

        val stepsArray = json.getJSONArray("steps")
        val steps = mutableListOf<RecordedStep>()
        for (i in 0 until stepsArray.length()) {
            steps.add(deserializeStepFromJson(stepsArray.getJSONObject(i)))
        }

        return DemonstrationTrajectory(
            metadata = metadata,
            goal = json.getString("goal"),
            totalSteps = json.getInt("totalSteps"),
            isSuccess = json.getBoolean("isSuccess"),
            finalOutcome = json.getString("finalOutcome"),
            steps = steps
        )
    }

    fun deserializeStepFromJson(json: JSONObject): RecordedStep {
        val toolsArray = json.optJSONArray("availableTools") ?: JSONArray()
        val tools = mutableListOf<ActionToolDescriptor>()
        for (i in 0 until toolsArray.length()) {
            val t = toolsArray.getJSONObject(i)
            tools.add(
                ActionToolDescriptor(
                    name = t.getString("name"),
                    description = t.getString("description"),
                    parametersJson = t.optString("parametersJson", "{}")
                )
            )
        }

        val argsObj = json.optJSONObject("actionArguments") ?: JSONObject()
        val args = mutableMapOf<String, String>()
        val keys = argsObj.keys()
        while (keys.hasNext()) {
            val k = keys.next()
            args[k] = argsObj.getString(k)
        }

        return RecordedStep(
            stepIndex = json.getInt("stepIndex"),
            timestamp = json.getLong("timestamp"),
            goal = json.getString("goal"),
            packageName = if (json.isNull("packageName")) null else json.optString("packageName"),
            activityName = if (json.isNull("activityName")) null else json.optString("activityName"),
            preObservation = deserializeUiStateFromJson(json.getJSONObject("preObservation")),
            availableTools = tools,
            humanAction = deserializeActionFromJson(json.getJSONObject("humanAction")),
            actionArguments = args,
            postObservation = deserializeUiStateFromJson(json.getJSONObject("postObservation")),
            isStateChanged = json.getBoolean("isStateChanged"),
            isSuccessful = json.getBoolean("isSuccessful"),
            deltaSummary = json.getString("deltaSummary"),
            failureRecoveryInfo = if (json.isNull("failureRecoveryInfo")) null else json.optString("failureRecoveryInfo")
        )
    }

    // =========================================================================
    // Training Example Serialization
    // =========================================================================

    fun serializeTrainingExampleToJson(example: TrainingExample): JSONObject {
        val root = JSONObject()
        root.put("exampleId", example.exampleId)
        root.put("goal", example.goal)
        root.put("packageName", example.packageName ?: JSONObject.NULL)
        root.put("currentObservation", serializeUiStateToJson(example.currentObservation))

        val actionsArray = JSONArray()
        example.availableActions.forEach { a ->
            actionsArray.put(JSONObject().apply {
                put("name", a.name)
                put("description", a.description)
                put("parametersJson", a.parametersJson)
            })
        }
        root.put("availableActions", actionsArray)

        val historyArray = JSONArray()
        example.shortHistory.forEach { h ->
            historyArray.put(JSONObject().apply {
                put("stepIndex", h.stepIndex)
                put("actionSummary", h.actionSummary)
                put("outcomeSummary", h.outcomeSummary)
                put("isStateChanged", h.isStateChanged)
            })
        }
        root.put("shortHistory", historyArray)

        root.put("previousOutcome", example.previousOutcome ?: JSONObject.NULL)
        root.put("targetAction", serializeActionToJson(example.targetAction))
        root.put("targetActionName", example.targetActionName)

        val argsObj = JSONObject()
        example.targetActionArguments.forEach { (k, v) -> argsObj.put(k, v) }
        root.put("targetActionArguments", argsObj)

        return root
    }

    fun deserializeTrainingExampleFromJson(json: JSONObject): TrainingExample {
        val actionsArray = json.optJSONArray("availableActions") ?: JSONArray()
        val actions = mutableListOf<ActionToolDescriptor>()
        for (i in 0 until actionsArray.length()) {
            val a = actionsArray.getJSONObject(i)
            actions.add(
                ActionToolDescriptor(
                    name = a.getString("name"),
                    description = a.getString("description"),
                    parametersJson = a.optString("parametersJson", "{}")
                )
            )
        }

        val historyArray = json.optJSONArray("shortHistory") ?: JSONArray()
        val history = mutableListOf<HistoryStep>()
        for (i in 0 until historyArray.length()) {
            val h = historyArray.getJSONObject(i)
            history.add(
                HistoryStep(
                    stepIndex = h.getInt("stepIndex"),
                    actionSummary = h.getString("actionSummary"),
                    outcomeSummary = h.getString("outcomeSummary"),
                    isStateChanged = h.getBoolean("isStateChanged")
                )
            )
        }

        val argsObj = json.optJSONObject("targetActionArguments") ?: JSONObject()
        val args = mutableMapOf<String, String>()
        val keys = argsObj.keys()
        while (keys.hasNext()) {
            val k = keys.next()
            args[k] = argsObj.getString(k)
        }

        return TrainingExample(
            exampleId = json.getString("exampleId"),
            goal = json.getString("goal"),
            packageName = if (json.isNull("packageName")) null else json.optString("packageName"),
            currentObservation = deserializeUiStateFromJson(json.getJSONObject("currentObservation")),
            availableActions = actions,
            shortHistory = history,
            previousOutcome = if (json.isNull("previousOutcome")) null else json.optString("previousOutcome"),
            targetAction = deserializeActionFromJson(json.getJSONObject("targetAction")),
            targetActionName = json.getString("targetActionName"),
            targetActionArguments = args
        )
    }

    // =========================================================================
    // UI State & Primitives Serialization
    // =========================================================================

    private fun serializeUiStateToJson(state: DistilledUiState): JSONObject {
        val obj = JSONObject()
        obj.put("packageName", state.packageName ?: JSONObject.NULL)
        obj.put("activityName", state.activityName ?: JSONObject.NULL)
        obj.put("rawNodeCount", state.rawNodeCount)
        obj.put("timestamp", state.timestamp)

        val nodesArray = JSONArray()
        state.nodes.forEach { node ->
            nodesArray.put(JSONObject().apply {
                put("index", node.index)
                put("role", node.role)
                put("text", node.text ?: JSONObject.NULL)
                put("contentDescription", node.contentDescription ?: JSONObject.NULL)
                put("hintText", node.hintText ?: JSONObject.NULL)
                put("resourceId", node.resourceId ?: JSONObject.NULL)
                put("bounds", JSONObject().apply {
                    put("left", node.bounds.left)
                    put("top", node.bounds.top)
                    put("right", node.bounds.right)
                    put("bottom", node.bounds.bottom)
                })
                put("isClickable", node.isClickable)
                put("isEditable", node.isEditable)
                put("isScrollable", node.isScrollable)
                put("isEnabled", node.isEnabled)
                put("isFocused", node.isFocused)
                put("isCheckable", node.isCheckable)
                put("isChecked", node.isChecked)
                put("isSelected", node.isSelected)
                put("depth", node.depth)
            })
        }
        obj.put("nodes", nodesArray)
        return obj
    }

    private fun deserializeUiStateFromJson(json: JSONObject): DistilledUiState {
        val nodesArray = json.optJSONArray("nodes") ?: JSONArray()
        val nodes = mutableListOf<DistilledNode>()
        for (i in 0 until nodesArray.length()) {
            val n = nodesArray.getJSONObject(i)
            val b = n.getJSONObject("bounds")
            nodes.add(
                DistilledNode(
                    index = n.getInt("index"),
                    role = n.getString("role"),
                    text = if (n.isNull("text")) null else n.optString("text"),
                    contentDescription = if (n.isNull("contentDescription")) null else n.optString("contentDescription"),
                    hintText = if (n.isNull("hintText")) null else n.optString("hintText"),
                    resourceId = if (n.isNull("resourceId")) null else n.optString("resourceId"),
                    bounds = NormalizedBounds(
                        left = b.getInt("left"),
                        top = b.getInt("top"),
                        right = b.getInt("right"),
                        bottom = b.getInt("bottom")
                    ),
                    isClickable = n.optBoolean("isClickable", false),
                    isEditable = n.optBoolean("isEditable", false),
                    isScrollable = n.optBoolean("isScrollable", false),
                    isEnabled = n.optBoolean("isEnabled", true),
                    isFocused = n.optBoolean("isFocused", false),
                    isCheckable = n.optBoolean("isCheckable", false),
                    isChecked = n.optBoolean("isChecked", false),
                    isSelected = n.optBoolean("isSelected", false),
                    depth = n.optInt("depth", 0)
                )
            )
        }

        return DistilledUiState(
            packageName = if (json.isNull("packageName")) null else json.optString("packageName"),
            activityName = if (json.isNull("activityName")) null else json.optString("activityName"),
            rawNodeCount = json.optInt("rawNodeCount", nodes.size),
            timestamp = json.optLong("timestamp", System.currentTimeMillis()),
            nodes = nodes
        )
    }

    private fun serializeActionToJson(action: AgentAction): JSONObject {
        val obj = JSONObject()
        when (action) {
            is AgentAction.OpenApp -> {
                obj.put("type", "OpenApp")
                obj.put("appName", action.appName)
            }
            is AgentAction.Tap -> {
                obj.put("type", "Tap")
                obj.put("targetNodeIndex", action.targetNodeIndex)
            }
            is AgentAction.TypeText -> {
                obj.put("type", "TypeText")
                obj.put("targetNodeIndex", action.targetNodeIndex)
                obj.put("text", action.text)
            }
            is AgentAction.Scroll -> {
                obj.put("type", "Scroll")
                obj.put("targetNodeIndex", action.targetNodeIndex)
                obj.put("direction", action.direction.name)
            }
            is AgentAction.PressBack -> {
                obj.put("type", "PressBack")
            }
            AgentAction.Submit -> {
                obj.put("type", "Submit")
            }
            is AgentAction.PressHome -> {
                obj.put("type", "PressHome")
            }
            is AgentAction.Wait -> {
                obj.put("type", "Wait")
                obj.put("durationMs", action.durationMs)
            }
            is AgentAction.CompleteGoal -> {
                obj.put("type", "CompleteGoal")
                obj.put("summary", action.summary)
            }
            is AgentAction.FailGoal -> {
                obj.put("type", "FailGoal")
                obj.put("reason", action.reason)
            }
        }
        return obj
    }

    private fun deserializeActionFromJson(json: JSONObject): AgentAction {
        return when (val type = json.getString("type")) {
            "OpenApp" -> AgentAction.OpenApp(json.getString("appName"))
            "Tap" -> AgentAction.Tap(json.getInt("targetNodeIndex"))
            "TypeText" -> AgentAction.TypeText(json.getInt("targetNodeIndex"), json.getString("text"))
            "Scroll" -> AgentAction.Scroll(
                json.getInt("targetNodeIndex"),
                ScrollDirection.valueOf(json.optString("direction", "DOWN"))
            )
            "PressBack" -> AgentAction.PressBack
            "Submit" -> AgentAction.Submit
            "PressHome" -> AgentAction.PressHome
            "Wait" -> AgentAction.Wait(json.optLong("durationMs", 500L))
            "CompleteGoal" -> AgentAction.CompleteGoal(json.optString("summary", "Goal Complete"))
            "FailGoal" -> AgentAction.FailGoal(json.optString("reason", "Goal Failed"))
            else -> throw IllegalArgumentException("Unknown action type: $type")
        }
    }

    // =========================================================================
    // JSONL Line Helpers
    // =========================================================================

    fun trajectoryToJsonlLine(trajectory: DemonstrationTrajectory): String {
        return serializeTrajectoryToJson(trajectory).toString()
    }

    fun trainingExampleToJsonlLine(example: TrainingExample): String {
        return serializeTrainingExampleToJson(example).toString()
    }

    fun parseTrajectoryFromJsonlLine(line: String): DemonstrationTrajectory {
        return deserializeTrajectoryFromJson(JSONObject(line.trim()))
    }

    fun parseTrainingExampleFromJsonlLine(line: String): TrainingExample {
        return deserializeTrainingExampleFromJson(JSONObject(line.trim()))
    }
}
