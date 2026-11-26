package com.zipper.app.core.model

/**
 * Normalized rectangular bounding box mapped to a density-independent [0, 1000] integer grid.
 */
data class NormalizedBounds(
    val left: Int,
    val top: Int,
    val right: Int,
    val bottom: Int
) {
    val width: Int get() = right - left
    val height: Int get() = bottom - top
    val centerX: Int get() = left + width / 2
    val centerY: Int get() = top + height / 2

    fun contains(x: Int, y: Int): Boolean {
        return x in left..right && y in top..bottom
    }
}

/**
 * A distilled, actionable UI node extracted from the raw Android Accessibility Tree.
 * Redundant layout containers are pruned, and inline labels are merged.
 */
data class DistilledNode(
    val index: Int,
    val role: String,
    val text: String? = null,
    val contentDescription: String? = null,
    val resourceId: String? = null,
    val bounds: NormalizedBounds,
    val rawBounds: android.graphics.Rect? = null,
    val isClickable: Boolean = false,
    val isEditable: Boolean = false,
    val isScrollable: Boolean = false,
    val isEnabled: Boolean = true,
    val isFocused: Boolean = false,
    val isCheckable: Boolean = false,
    val isChecked: Boolean = false,
    val isSelected: Boolean = false,
    val depth: Int = 0,
    val parentIndex: Int? = null,
    val className: String? = null,
    val hintText: String? = null,
    val href: String? = null,
    val imageUrl: String? = null,
    val region: String? = "BODY"
) {
    /**
     * Primary human-readable identifier for semantic matching (text, content description, or input placeholder hint).
     */
    val semanticLabel: String
        get() = when {
            !text.isNullOrBlank() -> text
            !contentDescription.isNullOrBlank() -> contentDescription
            !hintText.isNullOrBlank() -> hintText
            !resourceId.isNullOrBlank() -> resourceId.substringAfterLast(":id/").substringAfterLast("/")
            else -> role
        }

    val currentValue: String?
        get() = text?.takeIf { it.isNotBlank() } ?: contentDescription?.takeIf { it.isNotBlank() } ?: hintText?.takeIf { it.isNotBlank() }

    val capabilities: Set<String>
        get() = buildSet {
            if (isClickable) add("click")
            if (isEditable) add("type")
            if (isScrollable) {
                add("scroll_forward")
                add("scroll_backward")
            }
            if (isCheckable) {
                add("toggle")
                add("select")
            }
            if (role.contains("RADIO", ignoreCase = true)) add("select")
        }
}

/**
 * The distilled state of the current visible screen.
 * Contains only actionable or semantically informative candidate elements (typically 15-35 nodes).
 */
data class DistilledUiState(
    val packageName: String? = null,
    val activityName: String? = null,
    val nodes: List<DistilledNode> = emptyList(),
    val rawNodeCount: Int = 0,
    val timestamp: Long = System.currentTimeMillis(),
    val images: List<String> = emptyList()
)
