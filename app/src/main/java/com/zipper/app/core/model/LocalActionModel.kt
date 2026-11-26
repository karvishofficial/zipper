package com.zipper.app.core.model

/**
 * Pluggable interface for the on-device Action Model.
 *
 * This abstraction allows swapping between:
 * 1. Cactus Needle 2 (V0 baseline tool-calling model)
 * 2. Heuristic Semantic Matcher (deterministic test / fallback engine)
 * 3. Future Karvish-PAT (custom-trained Action Transformer)
 */
interface LocalActionModel {
    val modelName: String

    suspend fun predictAction(
        step: FlowStep,
        uiState: DistilledUiState
    ): ActionPrediction
}
