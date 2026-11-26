package com.zipper.app.core.web

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import com.zipper.app.core.agent.KarvishWordPieceTokenizer
import com.zipper.app.core.model.DistilledNode
import com.zipper.app.core.model.DistilledUiState
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.exp
import kotlin.math.max

/**
 * Result of on-device Web Agent prediction.
 */
data class WebOnDeviceDecision(
    val action: String, // CLICK, TYPE_TEXT, SUBMIT, SCROLL, SELECT, EXTRACT, WAIT, NAVIGATE
    val targetNode: DistilledNode?,
    val argument: String? = null,
    val confidence: Float = 0f,
    val inferenceTimeMs: Long = 0L,
    val allScores: List<Pair<DistilledNode, Float>> = emptyList()
)

/**
 * Dedicated On-Device Neural Engine for Web Automation (Karvish-Web).
 *
 * Runs `karvish_web.onnx` locally on CPU using ONNX Runtime with zero cloud tokens.
 * Completely isolated from the Android OS physical control baseline (V52-LOCKED-GOLD).
 *
 * Capabilities:
 * - 2D Spatial Geometry & Region Context (Header, Body, Modal, Footer)
 * - Semantic Affordance Matching via MiniLM-L6 WordPiece representations
 * - Real-time candidate ranking (~20-30ms)
 * - Learned Span Pointer Head for extracting argument text directly from user goals
 */
class WebOnDeviceEngine(
    private val context: Context,
    private val modelAssetPath: String = "models/karvish_web.onnx"
) {

    companion object {
        private const val TAG = "WebOnDeviceEngine"
        private const val MAX_GOAL_LEN = 32
        private const val MAX_CAND_LEN = 16
        private const val MAX_CANDIDATES = 32

        val CANONICAL_ACTIONS = arrayOf(
            "CLICK",
            "TYPE_TEXT",
            "SUBMIT",
            "SCROLL",
            "SELECT",
            "EXTRACT",
            "WAIT",
            "NAVIGATE"
        )

        private val WEB_ROLES_MAP = mapOf(
            "BUTTON" to 0L,
            "SEARCH_FIELD" to 1L,
            "TEXT_FIELD" to 2L,
            "INPUT" to 2L,
            "EDITTEXT" to 2L,
            "LINK" to 3L,
            "SELECT" to 4L,
            "CHECKBOX" to 5L,
            "RADIO" to 6L,
            "HEADER" to 7L,
            "LIST_ITEM" to 8L,
            "TAB" to 9L,
            "MENU_ITEM" to 10L,
            "LABEL" to 11L,
            "TEXT" to 11L,
            "CONTAINER" to 12L
        )

        private val WEB_REGIONS_MAP = mapOf(
            "HEADER" to 0L,
            "BODY" to 1L,
            "MODAL" to 2L,
            "FOOTER" to 3L
        )
    }

    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private val tokenizer: KarvishWordPieceTokenizer by lazy {
        KarvishWordPieceTokenizer(context)
    }
    private var isInitialized = false

    init {
        initializeEngine()
    }

    private fun resolveAvailableModelPath(): String {
        return try {
            val list = context.assets.list("models")?.toList() ?: emptyList()
            when {
                list.contains("karvish_web.onnx") -> "models/karvish_web.onnx"
                list.contains("karvish_v62.onnx") -> "models/karvish_v62.onnx"
                list.contains("karvish_v61.onnx") -> "models/karvish_v61.onnx"
                else -> modelAssetPath
            }
        } catch (_: Exception) {
            modelAssetPath
        }
    }

    @Synchronized
    fun initializeEngine(): Boolean {
        if (isInitialized && ortSession != null) return true
        val t0 = System.currentTimeMillis()
        val path = resolveAvailableModelPath()
        try {
            ortEnv = OrtEnvironment.getEnvironment()
            val sessionOptions = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(4)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }
            val modelBytes = context.assets.open(path).use { it.readBytes() }
            ortSession = ortEnv?.createSession(modelBytes, sessionOptions)
            isInitialized = true
            Log.i(TAG, "Initialized Karvish-Web ONNX Engine from $path in ${System.currentTimeMillis() - t0}ms (${modelBytes.size / (1024 * 1024)} MB)")
            return true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize WebOnDeviceEngine from $path", e)
            return false
        }
    }

    fun isAvailable(): Boolean {
        if (isInitialized && ortSession != null) return true
        val path = resolveAvailableModelPath()
        return try {
            context.assets.open(path).close()
            true
        } catch (_: Exception) {
            false
        }
    }

    /**
     * Predicts the next web action and target element in < 30ms locally on device.
     */
    suspend fun predictNextAction(
        goal: String,
        uiState: DistilledUiState
    ): WebOnDeviceDecision = withContext(Dispatchers.Default) {
        val t0 = System.currentTimeMillis()

        val session = ortSession ?: run {
            if (!initializeEngine()) {
                return@withContext fallbackDecision("WAIT", null, "Engine not initialized", 0L)
            }
            ortSession!!
        }
        val env = ortEnv ?: return@withContext fallbackDecision("WAIT", null, "Environment not initialized", 0L)

        // 1. Filter candidates to salient web elements
        val candidates = uiState.nodes.filter {
            it.isClickable || it.isEditable || it.isScrollable ||
                    it.role in setOf("BUTTON", "LINK", "SELECT", "SEARCH_FIELD", "TEXT_FIELD", "TAB", "MENU_ITEM") ||
                    !it.semanticLabel.isBlank()
        }.take(MAX_CANDIDATES).ifEmpty {
            uiState.nodes.take(MAX_CANDIDATES)
        }

        if (candidates.isEmpty()) {
            return@withContext fallbackDecision("WAIT", null, "No actionable candidates found", System.currentTimeMillis() - t0)
        }

        val numCands = candidates.size

        // 2. Tokenize user goal
        val (goalIdsArray, goalMaskArray, _) = tokenizer.encodeGoal(goal, MAX_GOAL_LEN)
        val goalMaskLong = LongArray(MAX_GOAL_LEN) { goalMaskArray[it].toLong() }

        // 3. Prepare candidate tensors
        val candIds = Array(1) { Array(numCands) { LongArray(MAX_CAND_LEN) { 0L } } }
        val candAttn = Array(1) { Array(numCands) { LongArray(MAX_CAND_LEN) { 0L } } }
        val candSpatial = Array(1) { Array(numCands) { FloatArray(10) { 0f } } }
        val candRoles = Array(1) { LongArray(numCands) { 0L } }
        val candRegions = Array(1) { LongArray(numCands) { 1L } }
        val candFlags = Array(1) { Array(numCands) { FloatArray(8) { 0f } } }
        val candMask = Array(1) { LongArray(numCands) { 0L } }

        for (i in 0 until numCands) {
            val node = candidates[i]
            candMask[0][i] = 1L

            // Label tokenization
            val (lblIds, lblMask) = tokenizer.encodeLabel(node.semanticLabel, MAX_CAND_LEN)
            candIds[0][i] = lblIds
            lblMask.forEachIndexed { idx, v -> candAttn[0][i][idx] = v.toLong() }

            // Continuous 2D Spatial Geometry (10 features)
            val b = node.bounds
            val left = b.left / 1000f
            val top = b.top / 1000f
            val right = b.right / 1000f
            val bottom = b.bottom / 1000f
            val width = max(0f, (b.right - b.left) / 1000f)
            val height = max(0f, (b.bottom - b.top) / 1000f)
            val cx = (b.left + b.right) / 2000f
            val cy = (b.top + b.bottom) / 2000f
            val area = width * height
            val aspect = if (height > 1e-4f) width / height else 1.0f

            candSpatial[0][i][0] = left
            candSpatial[0][i][1] = top
            candSpatial[0][i][2] = right
            candSpatial[0][i][3] = bottom
            candSpatial[0][i][4] = cx
            candSpatial[0][i][5] = cy
            candSpatial[0][i][6] = width
            candSpatial[0][i][7] = height
            candSpatial[0][i][8] = area
            candSpatial[0][i][9] = aspect

            // Roles & 2D Spatial Regions
            candRoles[0][i] = WEB_ROLES_MAP[node.role.uppercase()] ?: 11L
            candRegions[0][i] = WEB_REGIONS_MAP[node.region?.uppercase()] ?: 1L

            // Interaction & Accessibility Flags
            candFlags[0][i][0] = if (node.isClickable) 1f else 0f
            candFlags[0][i][1] = if (node.isEditable) 1f else 0f
            candFlags[0][i][2] = if (node.isScrollable) 1f else 0f
            candFlags[0][i][3] = if (node.isEnabled) 1f else 0f
            candFlags[0][i][4] = if (node.isFocused) 1f else 0f
            candFlags[0][i][5] = if (node.isCheckable) 1f else 0f
            candFlags[0][i][6] = if (node.isChecked) 1f else 0f
            candFlags[0][i][7] = if (node.isSelected) 1f else 0f
        }

        // 4. Construct ONNX input tensors
        val tensorsToClose = ArrayList<OnnxTensor>(9)
        try {
            val tGoalIds = OnnxTensor.createTensor(env, Array(1) { goalIdsArray }).also { tensorsToClose.add(it) }
            val tGoalMask = OnnxTensor.createTensor(env, Array(1) { goalMaskLong }).also { tensorsToClose.add(it) }
            val tCandIds = OnnxTensor.createTensor(env, candIds).also { tensorsToClose.add(it) }
            val tCandAttn = OnnxTensor.createTensor(env, candAttn).also { tensorsToClose.add(it) }
            val tCandSpatial = OnnxTensor.createTensor(env, candSpatial).also { tensorsToClose.add(it) }
            val tCandRoles = OnnxTensor.createTensor(env, candRoles).also { tensorsToClose.add(it) }
            val tCandRegions = OnnxTensor.createTensor(env, candRegions).also { tensorsToClose.add(it) }
            val tCandFlags = OnnxTensor.createTensor(env, candFlags).also { tensorsToClose.add(it) }
            val tCandMask = OnnxTensor.createTensor(env, candMask).also { tensorsToClose.add(it) }

            val inputsMap = mapOf(
                "goal_input_ids" to tGoalIds,
                "goal_attention_mask" to tGoalMask,
                "cand_input_ids" to tCandIds,
                "cand_attention_mask" to tCandAttn,
                "cand_spatial" to tCandSpatial,
                "cand_roles" to tCandRoles,
                "cand_regions" to tCandRegions,
                "cand_flags" to tCandFlags,
                "cand_mask" to tCandMask
            )

            val outputs = session.run(inputsMap)

            // 5. Parse output logits
            var rankLogits: FloatArray? = null
            var actionLogits: FloatArray? = null
            var spanStartLogits: FloatArray? = null
            var spanEndLogits: FloatArray? = null
            var bioLogits: Array<FloatArray>? = null

            for (entry in outputs) {
                when (entry.key) {
                    "rank_logits" -> {
                        @Suppress("UNCHECKED_CAST")
                        val v = entry.value.value as? Array<FloatArray>
                        rankLogits = v?.getOrNull(0)
                    }
                    "action_logits" -> {
                        @Suppress("UNCHECKED_CAST")
                        val v = entry.value.value as? Array<FloatArray>
                        actionLogits = v?.getOrNull(0)
                    }
                    "span_start_logits" -> {
                        @Suppress("UNCHECKED_CAST")
                        val v = entry.value.value as? Array<FloatArray>
                        spanStartLogits = v?.getOrNull(0)
                    }
                    "span_end_logits" -> {
                        @Suppress("UNCHECKED_CAST")
                        val v = entry.value.value as? Array<FloatArray>
                        spanEndLogits = v?.getOrNull(0)
                    }
                    "bio_logits" -> {
                        @Suppress("UNCHECKED_CAST")
                        val v = entry.value.value as? Array<Array<FloatArray>>
                        bioLogits = v?.getOrNull(0)
                    }
                }
            }

            outputs.close()

            val rScores = rankLogits ?: FloatArray(numCands) { 0f }
            val aScores = actionLogits ?: FloatArray(8) { 0f }
            val sStart = spanStartLogits ?: FloatArray(MAX_GOAL_LEN) { 0f }
            val sEnd = spanEndLogits ?: FloatArray(MAX_GOAL_LEN) { 0f }

            // 6. Find Best Target Candidate
            var bestCandIdx = 0
            var bestScore = Float.NEGATIVE_INFINITY
            val scoredList = ArrayList<Pair<DistilledNode, Float>>(numCands)

            for (i in 0 until numCands) {
                val score = rScores[i]
                scoredList.add(candidates[i] to score)
                if (score > bestScore) {
                    bestScore = score
                    bestCandIdx = i
                }
            }

            // Softmax confidence calculation
            val maxScore = rScores.take(numCands).maxOrNull() ?: 0f
            val expSum = rScores.take(numCands).sumOf { exp((it - maxScore).toDouble()) }.toFloat()
            val confidence = if (expSum > 0f) exp((bestScore - maxScore).toDouble()).toFloat() / expSum else 0.5f

            val targetNode = candidates.getOrNull(bestCandIdx)

            // 7. Find Predicted Action
            var bestActionIdx = 0
            var bestActionScore = Float.NEGATIVE_INFINITY
            for (i in 0 until minOf(8, aScores.size)) {
                if (aScores[i] > bestActionScore) {
                    bestActionScore = aScores[i]
                    bestActionIdx = i
                }
            }
            val predictedAction = CANONICAL_ACTIONS.getOrElse(bestActionIdx) { "CLICK" }

            // 8. Extract Argument Text via Learned BIO Head (with pointer head fallback)
            val extractedArgument = if (predictedAction == "TYPE_TEXT" || predictedAction == "SELECT") {
                var bioExtracted: String? = null
                val bLogits = bioLogits
                if (bLogits != null) {
                    val activeTokens = ArrayList<Long>()
                    for (i in 0 until minOf(MAX_GOAL_LEN, bLogits.size)) {
                        if (goalMaskLong.getOrNull(i) == 1L) {
                            val probs = bLogits[i]
                            val label = if (probs.size >= 3) {
                                var bestLabel = 0
                                var maxL = probs[0]
                                for (l in 1 until 3) {
                                    if (probs[l] > maxL) {
                                        maxL = probs[l]
                                        bestLabel = l
                                    }
                                }
                                bestLabel
                            } else 0
                            // 1 = B-QUERY, 2 = I-QUERY
                            if (label == 1 || label == 2) {
                                activeTokens.add(goalIdsArray[i])
                            }
                        }
                    }
                    if (activeTokens.isNotEmpty()) {
                        val text = tokenizer.decodeTokens(activeTokens)
                        if (text.isNotBlank()) bioExtracted = text
                    }
                }

                if (bioExtracted != null) {
                    bioExtracted
                } else {
                    // Fallback to learned span pointer
                    var bestStartToken = 1
                    var bestStartScore = Float.NEGATIVE_INFINITY
                    for (i in 1 until MAX_GOAL_LEN) {
                        if (goalMaskLong[i] == 1L && sStart[i] > bestStartScore) {
                            bestStartScore = sStart[i]
                            bestStartToken = i
                        }
                    }

                    var bestEndToken = bestStartToken
                    var bestEndScore = Float.NEGATIVE_INFINITY
                    for (i in bestStartToken until MAX_GOAL_LEN) {
                        if (goalMaskLong[i] == 1L && sEnd[i] > bestEndScore) {
                            bestEndScore = sEnd[i]
                            bestEndToken = i
                        }
                    }

                    val text = tokenizer.decode(goalIdsArray, bestStartToken, bestEndToken)
                    text.ifBlank { null }
                }
            } else {
                null
            }

            val inferenceTime = System.currentTimeMillis() - t0
            Log.d(
                TAG,
                "Inference finished in ${inferenceTime}ms: action=$predictedAction, target='${targetNode?.semanticLabel}', arg='$extractedArgument', conf=$confidence"
            )

            return@withContext WebOnDeviceDecision(
                action = predictedAction,
                targetNode = targetNode,
                argument = extractedArgument,
                confidence = confidence,
                inferenceTimeMs = inferenceTime,
                allScores = scoredList.sortedByDescending { it.second }
            )
        } catch (e: Exception) {
            Log.e(TAG, "Error running on-device web model inference", e)
            return@withContext fallbackDecision("WAIT", null, "Inference error: ${e.message}", System.currentTimeMillis() - t0)
        } finally {
            tensorsToClose.forEach {
                try { it.close() } catch (ignored: Exception) {}
            }
        }
    }

    /**
     * Pure neural query entity extraction from a natural language goal.
     * Extracts the target product/search query argument using the trained MiniLM + BIO head,
     * without any regex stripping or keyword hardcoding.
     */
    suspend fun extractQueryEntity(goal: String): String? = withContext(Dispatchers.Default) {
        val session = ortSession ?: run {
            if (!initializeEngine()) return@withContext null
            ortSession!!
        }
        val env = ortEnv ?: return@withContext null

        val (goalIdsArray, goalMaskArray, _) = tokenizer.encodeGoal(goal, MAX_GOAL_LEN)
        val goalMaskLong = LongArray(MAX_GOAL_LEN) { goalMaskArray[it].toLong() }

        // Dummy candidate to satisfy ONNX fixed graph structure
        val candIds = Array(1) { Array(1) { LongArray(MAX_CAND_LEN) { 0L } } }
        val candAttn = Array(1) { Array(1) { LongArray(MAX_CAND_LEN) { 0L } } }
        val candSpatial = Array(1) { Array(1) { FloatArray(10) { 0f } } }
        val candRoles = Array(1) { LongArray(1) { 0L } }
        val candRegions = Array(1) { LongArray(1) { 1L } }
        val candFlags = Array(1) { Array(1) { FloatArray(8) { 0f } } }
        val candMask = Array(1) { LongArray(1) { 1L } }

        val tensorsToClose = ArrayList<OnnxTensor>(9)
        try {
            val tGoalIds = OnnxTensor.createTensor(env, Array(1) { goalIdsArray }).also { tensorsToClose.add(it) }
            val tGoalMask = OnnxTensor.createTensor(env, Array(1) { goalMaskLong }).also { tensorsToClose.add(it) }
            val tCandIds = OnnxTensor.createTensor(env, candIds).also { tensorsToClose.add(it) }
            val tCandAttn = OnnxTensor.createTensor(env, candAttn).also { tensorsToClose.add(it) }
            val tCandSpatial = OnnxTensor.createTensor(env, candSpatial).also { tensorsToClose.add(it) }
            val tCandRoles = OnnxTensor.createTensor(env, candRoles).also { tensorsToClose.add(it) }
            val tCandRegions = OnnxTensor.createTensor(env, candRegions).also { tensorsToClose.add(it) }
            val tCandFlags = OnnxTensor.createTensor(env, candFlags).also { tensorsToClose.add(it) }
            val tCandMask = OnnxTensor.createTensor(env, candMask).also { tensorsToClose.add(it) }

            val inputsMap = mapOf(
                "goal_input_ids" to tGoalIds,
                "goal_attention_mask" to tGoalMask,
                "cand_input_ids" to tCandIds,
                "cand_attention_mask" to tCandAttn,
                "cand_spatial" to tCandSpatial,
                "cand_roles" to tCandRoles,
                "cand_regions" to tCandRegions,
                "cand_flags" to tCandFlags,
                "cand_mask" to tCandMask
            )
            val outputs = session.run(inputsMap)
            var bioLogits: Array<FloatArray>? = null
            var sStart: FloatArray? = null
            var sEnd: FloatArray? = null

            for (entry in outputs) {
                when (entry.key) {
                    "bio_logits" -> {
                        @Suppress("UNCHECKED_CAST")
                        val v = entry.value.value as? Array<Array<FloatArray>>
                        bioLogits = v?.getOrNull(0)
                    }
                    "span_start_logits" -> {
                        @Suppress("UNCHECKED_CAST")
                        val v = entry.value.value as? Array<FloatArray>
                        sStart = v?.getOrNull(0)
                    }
                    "span_end_logits" -> {
                        @Suppress("UNCHECKED_CAST")
                        val v = entry.value.value as? Array<FloatArray>
                        sEnd = v?.getOrNull(0)
                    }
                }
            }
            outputs.close()

            // 1. Try BIO Decoding
            if (bioLogits != null) {
                val activeTokens = ArrayList<Long>()
                for (i in 0 until minOf(MAX_GOAL_LEN, bioLogits.size)) {
                    if (goalMaskLong.getOrNull(i) == 1L) {
                        val probs = bioLogits[i]
                        val label = if (probs.size >= 3) {
                            var best = 0
                            var maxL = probs[0]
                            for (l in 1 until 3) {
                                if (probs[l] > maxL) {
                                    maxL = probs[l]
                                    best = l
                                }
                            }
                            best
                        } else 0
                        if (label == 1 || label == 2) {
                            activeTokens.add(goalIdsArray[i])
                        }
                    }
                }
                if (activeTokens.isNotEmpty()) {
                    val text = tokenizer.decodeTokens(activeTokens)
                    if (text.isNotBlank()) return@withContext text
                }
            }

            // 2. Fallback to learned span pointer
            if (sStart != null && sEnd != null) {
                var bestStart = 1
                var bestStartScore = Float.NEGATIVE_INFINITY
                for (i in 1 until MAX_GOAL_LEN) {
                    if (goalMaskLong[i] == 1L && sStart[i] > bestStartScore) {
                        bestStartScore = sStart[i]
                        bestStart = i
                    }
                }
                var bestEnd = bestStart
                var bestEndScore = Float.NEGATIVE_INFINITY
                for (i in bestStart until MAX_GOAL_LEN) {
                    if (goalMaskLong[i] == 1L && sEnd[i] > bestEndScore) {
                        bestEndScore = sEnd[i]
                        bestEnd = i
                    }
                }
                val text = tokenizer.decode(goalIdsArray, bestStart, bestEnd)
                if (text.isNotBlank()) return@withContext text
            }
            null
        } catch (e: Exception) {
            Log.e(TAG, "extractQueryEntity failed: ${e.message}")
            null
        } finally {
            tensorsToClose.forEach { try { it.close() } catch (_: Exception) {} }
        }
    }

    private fun fallbackDecision(action: String, node: DistilledNode?, reason: String, timeMs: Long): WebOnDeviceDecision {
        return WebOnDeviceDecision(
            action = action,
            targetNode = node,
            argument = null,
            confidence = 0f,
            inferenceTimeMs = timeMs
        )
    }
}
