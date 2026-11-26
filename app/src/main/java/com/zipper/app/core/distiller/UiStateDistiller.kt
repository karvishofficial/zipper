package com.zipper.app.core.distiller

import android.graphics.Rect
import android.view.accessibility.AccessibilityNodeInfo
import com.zipper.app.core.model.DistilledNode
import com.zipper.app.core.model.DistilledUiState
import com.zipper.app.core.model.NormalizedBounds

/**
 * Interface abstraction over accessibility nodes to permit both real Android
 * AccessibilityNodeInfo traversal and lightweight in-memory unit testing.
 */
interface AccessibleNode {
    val className: CharSequence?
    val text: CharSequence?
    val contentDescription: CharSequence?
    val hintText: CharSequence? get() = null
    val roleDescription: CharSequence? get() = null
    val viewIdResourceName: String?
    val isClickable: Boolean
    val isEditable: Boolean
    val isScrollable: Boolean
    val isEnabled: Boolean
    val isFocused: Boolean
    val isCheckable: Boolean
    val isChecked: Boolean
    val isSelected: Boolean
    val childCount: Int
    fun getChild(index: Int): AccessibleNode?
    fun getBoundsInScreen(outBounds: Rect)
}

/**
 * Standard implementation wrapping Android's AccessibilityNodeInfo.
 */
class AndroidAccessibilityNodeWrapper(val node: AccessibilityNodeInfo) : AccessibleNode {
    override val className: CharSequence? get() = node.className
    override val text: CharSequence? get() = node.text
    override val contentDescription: CharSequence? get() = node.contentDescription
    override val hintText: CharSequence?
        get() = if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
            node.hintText
        } else {
            null
        }
    override val roleDescription: CharSequence?
        get() = try {
            androidx.core.view.accessibility.AccessibilityNodeInfoCompat.wrap(node).roleDescription
                ?: node.extras?.getCharSequence("AccessibilityNodeInfo.roleDescription")
        } catch (_: Throwable) {
            null
        }
    override val viewIdResourceName: String? get() = node.viewIdResourceName
    override val isClickable: Boolean get() = node.isClickable
    override val isEditable: Boolean get() = node.isEditable
    override val isScrollable: Boolean get() = node.isScrollable
    override val isEnabled: Boolean get() = node.isEnabled
    override val isFocused: Boolean get() = node.isFocused
    override val isCheckable: Boolean get() = node.isCheckable
    override val isChecked: Boolean get() = node.isChecked
    override val isSelected: Boolean get() = node.isSelected
    override val childCount: Int get() = node.childCount

    override fun getChild(index: Int): AccessibleNode? {
        val child = node.getChild(index) ?: return null
        return AndroidAccessibilityNodeWrapper(child)
    }

    override fun getBoundsInScreen(outBounds: Rect) {
        node.getBoundsInScreen(outBounds)
    }
}

/**
 * UI State Distiller: Prunes redundant layout wrappers (e.g. FrameLayout, LinearLayout)
 * and normalizes pixel coordinates to [0, 1000] density-independent grid space.
 * Reduces 1000+ raw node trees down to 15-35 actionable candidate elements.
 */
class UiStateDistiller {

    fun distill(
        root: AccessibilityNodeInfo?,
        displayWidth: Int = 1080,
        displayHeight: Int = 2400,
        packageName: String? = null,
        activityName: String? = null,
        onNodeMapped: ((Int, AccessibilityNodeInfo) -> Unit)? = null
    ): DistilledUiState {
        if (root == null) {
            return DistilledUiState(packageName, activityName, emptyList(), 0)
        }
        return distill(AndroidAccessibilityNodeWrapper(root), displayWidth, displayHeight, packageName, activityName, onNodeMapped)
    }

    fun distill(
        root: AccessibleNode?,
        displayWidth: Int = 1080,
        displayHeight: Int = 2400,
        packageName: String? = null,
        activityName: String? = null,
        onNodeMapped: ((Int, AccessibilityNodeInfo) -> Unit)? = null
    ): DistilledUiState {
        if (root == null) {
            return DistilledUiState(packageName, activityName, emptyList(), 0)
        }

        val candidates = mutableListOf<DistilledNode>()
        var rawCount = 0
        val boundsRect = Rect()

        fun descendantLabels(node: AccessibleNode): List<String> {
            val labels = linkedSetOf<String>()
            for (i in 0 until node.childCount) {
                val child = node.getChild(i) ?: continue
                val t = child.text?.toString()?.trim()?.takeIf { it.isNotBlank() }
                val d = child.contentDescription?.toString()?.trim()?.takeIf { it.isNotBlank() }
                val h = child.hintText?.toString()?.trim()?.takeIf { it.isNotBlank() }
                if (t != null && !t.contains("edit suggestion", ignoreCase = true)) labels.add(t)
                if (d != null && !d.contains("edit suggestion", ignoreCase = true)) labels.add(d)
                if (h != null && !h.contains("edit suggestion", ignoreCase = true)) labels.add(h)
                labels.addAll(descendantLabels(child))
            }
            return labels.toList()
        }

        fun traverse(node: AccessibleNode, depth: Int, parentCandidateIndex: Int?) {
            rawCount++
            node.getBoundsInScreen(boundsRect)

            val role = simplifyRole(node.className?.toString(), node.roleDescription)
            val isClickableEffective = node.isClickable || role in setOf("BUTTON", "LINK", "TAB", "MENU_ITEM")
            val text = node.text?.toString()?.trim()?.takeIf { it.isNotBlank() }
            val desc = node.contentDescription?.toString()?.trim()?.takeIf { it.isNotBlank() }
            val hint = node.hintText?.toString()?.trim()?.takeIf { it.isNotBlank() }
            val inheritedText = if (isClickableEffective && text == null && desc == null && hint == null) {
                descendantLabels(node).joinToString(" ").takeIf { it.isNotBlank() }
            } else {
                null
            }
            val id = node.viewIdResourceName
            val isActionable = isClickableEffective || node.isEditable || node.isScrollable || node.isCheckable
            val hasContent = !text.isNullOrBlank() || !desc.isNullOrBlank() || !hint.isNullOrBlank() || !inheritedText.isNullOrBlank()

            // Normalize bounds to [0, 1000]
            val safeWidth = if (displayWidth > 0) displayWidth else 1080
            val safeHeight = if (displayHeight > 0) displayHeight else 2400

            val normLeft = ((boundsRect.left.coerceAtLeast(0).toFloat() / safeWidth) * 1000).toInt().coerceIn(0, 1000)
            val normTop = ((boundsRect.top.coerceAtLeast(0).toFloat() / safeHeight) * 1000).toInt().coerceIn(0, 1000)
            val normRight = ((boundsRect.right.coerceAtMost(safeWidth).toFloat() / safeWidth) * 1000).toInt().coerceIn(0, 1000)
            val normBottom = ((boundsRect.bottom.coerceAtMost(safeHeight).toFloat() / safeHeight) * 1000).toInt().coerceIn(0, 1000)

            val normalizedBounds = NormalizedBounds(normLeft, normTop, normRight, normBottom)

            // Keep if actionable or contains significant semantic text
            val isVisible = (boundsRect.right - boundsRect.left) > 0 && (boundsRect.bottom - boundsRect.top) > 0
            var candidateIndex: Int? = null
            if (isVisible && (isActionable || hasContent)) {
                val newCandidateIndex = candidates.size
                candidateIndex = newCandidateIndex
                candidates.add(
                    DistilledNode(
                        index = newCandidateIndex,
                        role = role,
                        text = text ?: inheritedText,
                        contentDescription = desc,
                        resourceId = id,
                        bounds = normalizedBounds,
                        rawBounds = Rect(boundsRect),
                        isClickable = isClickableEffective,
                        isEditable = node.isEditable,
                        isScrollable = node.isScrollable,
                        isEnabled = node.isEnabled,
                        isFocused = node.isFocused,
                        isCheckable = node.isCheckable,
                        isChecked = node.isChecked,
                        isSelected = node.isSelected,
                        depth = depth,
                        parentIndex = parentCandidateIndex,
                        className = node.className?.toString(),
                        hintText = hint
                    )
                )
                if (node is AndroidAccessibilityNodeWrapper) {
                    onNodeMapped?.invoke(newCandidateIndex, node.node)
                }
            }

            for (i in 0 until node.childCount) {
                val child = node.getChild(i) ?: continue
                traverse(child, depth + 1, candidateIndex ?: parentCandidateIndex)
            }
        }

        traverse(root, 0, null)
        return DistilledUiState(
            packageName = packageName,
            activityName = activityName,
            nodes = candidates,
            rawNodeCount = rawCount
        )
    }

    private fun simplifyRole(className: String?, roleDescription: CharSequence? = null): String {
        val roleDescStr = roleDescription?.toString()?.trim()?.lowercase()
        if (!roleDescStr.isNullOrBlank()) {
            when {
                roleDescStr.contains("button") -> return "BUTTON"
                roleDescStr.contains("link") -> return "LINK"
                roleDescStr.contains("heading") || roleDescStr.contains("header") -> return "HEADER"
                roleDescStr.contains("menu") -> return "MENU_ITEM"
                roleDescStr.contains("tab") -> return "TAB"
                roleDescStr.contains("text box") || roleDescStr.contains("search") || roleDescStr.contains("input") || roleDescStr.contains("edit") -> return "INPUT"
                roleDescStr.contains("checkbox") -> return "CHECKBOX"
                roleDescStr.contains("radio") -> return "RADIO_BUTTON"
                roleDescStr.contains("switch") || roleDescStr.contains("toggle") -> return "SWITCH"
                roleDescStr.contains("navigation") || roleDescStr.contains("nav") -> return "NAVIGATION"
                roleDescStr.contains("list item") -> return "LIST_ITEM"
                roleDescStr.contains("image") -> return "IMAGE"
            }
        }
        if (className == null) return "VIEW"
        val lower = className.lowercase()
        return when {
            lower.contains("button") -> "BUTTON"
            lower.contains("edittext") || lower.contains("input") -> "INPUT"
            lower.contains("textview") -> "TEXT"
            lower.contains("imageview") || lower.contains("image") -> "IMAGE"
            lower.contains("checkbox") -> "CHECKBOX"
            lower.contains("switch") -> "SWITCH"
            lower.contains("radiobutton") -> "RADIO_BUTTON"
            lower.contains("recyclerview") || lower.contains("listview") || lower.contains("scrollview") -> "SCROLL_VIEW"
            lower.contains("tab") -> "TAB"
            else -> className.substringAfterLast(".").uppercase()
        }
    }
}
