package com.zipper.app.core.recorder

import android.content.Context
import java.io.File

/** Persists validated human trajectories for later transition export. */
class DemonstrationStore(context: Context) {
    private val serializer = DemonstrationSerializer()
    private val outputFile = File(context.filesDir, "demonstrations.jsonl")

    fun append(trajectory: DemonstrationTrajectory): File {
        outputFile.parentFile?.mkdirs()
        outputFile.appendText(serializer.trajectoryToJsonlLine(trajectory) + "\n", Charsets.UTF_8)
        return outputFile
    }

    fun file(): File = outputFile
}