package com.zipper.app.ditto

import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.io.File

/**
 * DITTO TRAJECTORY STORE
 * Handles local persistence, retrieval, and management of human-demonstrated Ditto flows.
 * Storage location: context.filesDir / "ditto_flows" / "{flowId}.json"
 */
class DittoStore(private val context: Context) {

    companion object {
        private const val TAG = "DittoStore"
        private const val DITTO_DIR = "ditto_flows"
    }

    private val storageDir: File by lazy {
        File(context.filesDir, DITTO_DIR).apply {
            if (!exists()) mkdirs()
        }
    }

    /**
     * Saves a recorded DittoFlow to disk.
     */
    fun saveFlow(flow: DittoFlow): Boolean {
        return try {
            val file = File(storageDir, "${flow.flowId}.json")
            val jsonString = flow.toJson().toString(2)
            file.writeText(jsonString, Charsets.UTF_8)
            Log.i(TAG, "Saved DittoFlow '${flow.title}' (${flow.steps.size} steps) to ${file.name}")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save DittoFlow: ${e.message}", e)
            false
        }
    }

    /**
     * Retrieves all saved DittoFlows ordered by creation time descending.
     */
    fun listFlows(): List<DittoFlow> {
        val list = mutableListOf<DittoFlow>()
        try {
            val files = storageDir.listFiles { f -> f.extension == "json" } ?: emptyArray()
            for (f in files) {
                try {
                    val text = f.readText(Charsets.UTF_8)
                    val json = JSONObject(text)
                    list.add(DittoFlow.fromJson(json))
                } catch (e: Exception) {
                    Log.w(TAG, "Skipping corrupt flow file ${f.name}: ${e.message}")
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error listing flows: ${e.message}", e)
        }
        return list.sortedByDescending { it.createdAt }
    }

    /**
     * Loads a specific DittoFlow by its flowId.
     */
    fun loadFlow(flowId: String): DittoFlow? {
        val file = File(storageDir, "$flowId.json")
        if (!file.exists()) return null
        return try {
            val text = file.readText(Charsets.UTF_8)
            DittoFlow.fromJson(JSONObject(text))
        } catch (e: Exception) {
            Log.e(TAG, "Error loading flow $flowId: ${e.message}", e)
            null
        }
    }

    /**
     * Deletes a DittoFlow by flowId.
     */
    fun deleteFlow(flowId: String): Boolean {
        val file = File(storageDir, "$flowId.json")
        return if (file.exists()) {
            val deleted = file.delete()
            Log.i(TAG, "Deleted flow $flowId: $deleted")
            deleted
        } else false
    }

    /**
     * Exports all flows to training JSONL format for future Karvish V57+ distillation.
     */
    fun exportTrainingJsonl(outputFile: File): Int {
        val flows = listFlows()
        var lineCount = 0
        outputFile.bufferedWriter(Charsets.UTF_8).use { writer ->
            for (flow in flows) {
                writer.write(flow.toJson().toString())
                writer.newLine()
                lineCount++
            }
        }
        Log.i(TAG, "Exported $lineCount Ditto human demonstrations to ${outputFile.absolutePath}")
        return lineCount
    }
}
