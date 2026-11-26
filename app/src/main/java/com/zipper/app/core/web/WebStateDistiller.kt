package com.zipper.app.core.web

import android.graphics.Rect
import com.zipper.app.core.model.DistilledNode
import com.zipper.app.core.model.DistilledUiState
import com.zipper.app.core.model.NormalizedBounds
import org.json.JSONObject

/**
 * Parses raw DOM extraction JSON from web_distiller.js into Karvish's standard DistilledUiState.
 * Completely aligns web candidates with the UniversalSemanticRole and NormalizedBounds schema.
 */
class WebStateDistiller {

    fun distill(jsonString: String?): DistilledUiState {
        if (jsonString.isNullOrBlank()) {
            return DistilledUiState(packageName = "com.zipper.web", activityName = "about:blank")
        }
        return try {
            val root = JSONObject(jsonString)
            val url = root.optString("url", "about:blank")
            val title = root.optString("title", "")
            val candidatesArray = root.optJSONArray("candidates") ?: return DistilledUiState(
                packageName = "com.zipper.web",
                activityName = url
            )

            val nodes = mutableListOf<DistilledNode>()
            for (i in 0 until candidatesArray.length()) {
                val obj = candidatesArray.getJSONObject(i)
                val boundsArr = obj.optJSONArray("bounds")
                val left = boundsArr?.optInt(0, 0) ?: 0
                val top = boundsArr?.optInt(1, 0) ?: 0
                val right = boundsArr?.optInt(2, 1000) ?: 1000
                val bottom = boundsArr?.optInt(3, 100) ?: 100

                val node = DistilledNode(
                    index = obj.optInt("index", i),
                    role = obj.optString("role", "BUTTON"),
                    text = obj.optString("text").takeIf { it.isNotBlank() },
                    contentDescription = obj.optString("contentDescription").takeIf { it.isNotBlank() },
                    resourceId = obj.optString("resourceId").takeIf { it.isNotBlank() } ?: obj.optString("karvishId"),
                    bounds = NormalizedBounds(left, top, right, bottom),
                    rawBounds = Rect(left, top, right, bottom),
                    isClickable = obj.optBoolean("isClickable", false),
                    isEditable = obj.optBoolean("isEditable", false),
                    isScrollable = obj.optBoolean("isScrollable", false),
                    isEnabled = !obj.optBoolean("isDisabled", false),
                    isFocused = obj.optBoolean("isFocused", false),
                    isCheckable = obj.optBoolean("isCheckable", false),
                    isChecked = obj.optBoolean("isChecked", false),
                    isSelected = obj.optBoolean("isSelected", false),
                    className = obj.optString("className", "HTML"),
                    hintText = obj.optString("hintText").takeIf { it.isNotBlank() },
                    href = obj.optString("href").takeIf { it.isNotBlank() },
                    imageUrl = obj.optString("imageUrl").takeIf { it.isNotBlank() },
                    region = obj.optString("region", "BODY")
                )
                nodes.add(node)
            }

            val imagesList = mutableListOf<String>()
            val imagesArr = root.optJSONArray("images")
            if (imagesArr != null) {
                for (j in 0 until imagesArr.length()) {
                    val imgStr = imagesArr.optString(j)
                    if (imgStr.isNotBlank() && (imgStr.startsWith("http://") || imgStr.startsWith("https://") || imgStr.startsWith("data:image/"))) {
                        imagesList.add(imgStr)
                    }
                }
            }
            // Also collect from node imageUrls if not already present
            nodes.mapNotNull { it.imageUrl }.filter { it.startsWith("http://") || it.startsWith("https://") || it.startsWith("data:image/") }.forEach {
                if (!imagesList.contains(it)) imagesList.add(it)
            }

            DistilledUiState(
                packageName = "com.zipper.web",
                activityName = if (title.isNotBlank()) title else url,
                nodes = nodes,
                rawNodeCount = nodes.size,
                images = imagesList
            )
        } catch (e: Exception) {
            DistilledUiState(packageName = "com.zipper.web", activityName = "error: ${e.message}")
        }
    }
}
