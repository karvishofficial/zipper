package com.zipper.app.core.teach

import com.zipper.app.core.model.NormalizedBounds

data class RecordedInteraction(
    val eventType: Int,
    val eventTypeName: String,
    val className: String?,
    val viewIdResourceName: String?,
    val text: String?,
    val contentDescription: String?,
    val boundsCenterNormX: Int,
    val boundsCenterNormY: Int,
    val boundsNorm: NormalizedBounds,
    val packageName: String?,
    val timestamp: Long = System.currentTimeMillis()
)

data class LearnedAlarmSkill(
    val isTaught: Boolean = false,
    val taughtTimestamp: Long = 0L,
    val deviceModel: String = android.os.Build.MODEL,
    val clockPackage: String = "com.sec.android.app.clockpackage",
    val hourPickerBounds: NormalizedBounds? = null,
    val minutePickerBounds: NormalizedBounds? = null,
    val amBounds: NormalizedBounds? = null,
    val pmBounds: NormalizedBounds? = null,
    val addAlarmButtonBounds: NormalizedBounds? = null,
    val saveButtonBounds: NormalizedBounds? = null,
    val recordedInteractions: List<RecordedInteraction> = emptyList()
)
