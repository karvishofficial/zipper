package com.zipper.app.core.agent

import com.zipper.app.core.model.DistilledNode
import org.json.JSONObject

/**
 * First-class candidate representation carrying direct node identity and semantic snapshot.
 *
 * Core Invariant:
 * A candidate ID identifies one candidate within one observation.
 * The executor must execute only the node bound to that candidate and must never infer
 * the target from the human-readable tool name, label, coordinate, or ambiguous numeric substring.
 * If that binding becomes invalid, re-observe instead of guessing.
 */
data class ActionCandidate(
    val candidateId: String,
    val actionType: String,
    val nodeIndex: Int? = null,
    val nodeIdentity: String? = null,
    val semanticLabel: String,
    val role: String,
    val node: DistilledNode? = null,
    val parameters: JSONObject? = null
)
