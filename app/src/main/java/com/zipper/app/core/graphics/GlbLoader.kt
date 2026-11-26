package com.zipper.app.core.graphics

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.IntBuffer
import org.json.JSONObject

/**
 * High-performance, zero-dependency binary glTF 2.0 (.glb) parser.
 * Specifically tuned for Zipper's 3D core mesh with embedded PBR textures.
 */
data class GlbMesh(
    val vertexBuffer: FloatBuffer,
    val texCoordBuffer: FloatBuffer,
    val indexBuffer: IntBuffer,
    val vertexCount: Int,
    val indexCount: Int,
    val textureBitmap: Bitmap?
)

object GlbLoader {
    private const val TAG = "GlbLoader"

    fun loadFromAssets(context: Context, assetPath: String): GlbMesh? {
        return try {
            context.assets.open(assetPath).use { inputStream ->
                parseGlb(inputStream)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load GLB from assets: $assetPath", e)
            null
        }
    }

    private fun parseGlb(inputStream: InputStream): GlbMesh? {
        val bytes = inputStream.readBytes()
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)

        // 1. Read 12-byte GLB Header
        val magic = buffer.int
        if (magic != 0x46546C67) { // "glTF"
            Log.e(TAG, "Invalid GLB magic header: 0x${Integer.toHexString(magic)}")
            return null
        }
        val version = buffer.int
        val totalLength = buffer.int

        // 2. Read JSON Chunk Header
        val jsonChunkLength = buffer.int
        val jsonChunkType = buffer.int
        if (jsonChunkType != 0x4E4F534A) { // "JSON"
            Log.e(TAG, "First chunk is not JSON")
            return null
        }
        val jsonBytes = ByteArray(jsonChunkLength)
        buffer.get(jsonBytes)
        val jsonString = String(jsonBytes, Charsets.UTF_8)
        val gltf = JSONObject(jsonString)

        // 3. Read Binary Chunk Header
        val binChunkLength = buffer.int
        val binChunkType = buffer.int
        if (binChunkType != 0x004E4942) { // "BIN\0"
            Log.e(TAG, "Second chunk is not BIN")
            return null
        }
        val binOffset = buffer.position()

        // 4. Parse Accessors & BufferViews
        val accessors = gltf.getJSONArray("accessors")
        val bufferViews = gltf.getJSONArray("bufferViews")
        val meshes = gltf.getJSONArray("meshes")
        val mesh = meshes.getJSONObject(0)
        val primitives = mesh.getJSONArray("primitives")
        val primitive = primitives.getJSONObject(0)
        val attributes = primitive.getJSONObject("attributes")

        val posAccIdx = attributes.getInt("POSITION")
        val uvAccIdx = attributes.getInt("TEXCOORD_0")
        val idxAccIdx = primitive.getInt("indices")

        // 5. Extract Vertex Positions (vec3 float32)
        val posAcc = accessors.getJSONObject(posAccIdx)
        val posBv = bufferViews.getJSONObject(posAcc.getInt("bufferView"))
        val posCount = posAcc.getInt("count")
        val posByteOffset = binOffset + posBv.optInt("byteOffset", 0) + posAcc.optInt("byteOffset", 0)

        val vertexBuffer = ByteBuffer.allocateDirect(posCount * 3 * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        val posSrcBuffer = ByteBuffer.wrap(bytes, posByteOffset, posCount * 3 * 4).order(ByteOrder.LITTLE_ENDIAN)
        for (i in 0 until posCount * 3) {
            vertexBuffer.put(posSrcBuffer.float)
        }
        vertexBuffer.position(0)

        // 6. Extract UV Coordinates (vec2 float32)
        val uvAcc = accessors.getJSONObject(uvAccIdx)
        val uvBv = bufferViews.getJSONObject(uvAcc.getInt("bufferView"))
        val uvCount = uvAcc.getInt("count")
        val uvByteOffset = binOffset + uvBv.optInt("byteOffset", 0) + uvAcc.optInt("byteOffset", 0)

        val texCoordBuffer = ByteBuffer.allocateDirect(uvCount * 2 * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        val uvSrcBuffer = ByteBuffer.wrap(bytes, uvByteOffset, uvCount * 2 * 4).order(ByteOrder.LITTLE_ENDIAN)
        for (i in 0 until uvCount * 2) {
            texCoordBuffer.put(uvSrcBuffer.float)
        }
        texCoordBuffer.position(0)

        // 7. Extract Indices (uint32 or uint16)
        val idxAcc = accessors.getJSONObject(idxAccIdx)
        val idxBv = bufferViews.getJSONObject(idxAcc.getInt("bufferView"))
        val indexCount = idxAcc.getInt("count")
        val idxComponentType = idxAcc.getInt("componentType") // 5125 = UNSIGNED_INT, 5123 = UNSIGNED_SHORT
        val idxByteOffset = binOffset + idxBv.optInt("byteOffset", 0) + idxAcc.optInt("byteOffset", 0)

        val indexBuffer = ByteBuffer.allocateDirect(indexCount * 4)
            .order(ByteOrder.nativeOrder())
            .asIntBuffer()

        if (idxComponentType == 5125) { // UNSIGNED_INT (32-bit)
            val idxSrcBuffer = ByteBuffer.wrap(bytes, idxByteOffset, indexCount * 4).order(ByteOrder.LITTLE_ENDIAN)
            for (i in 0 until indexCount) {
                indexBuffer.put(idxSrcBuffer.int)
            }
        } else { // UNSIGNED_SHORT (16-bit)
            val idxSrcBuffer = ByteBuffer.wrap(bytes, idxByteOffset, indexCount * 2).order(ByteOrder.LITTLE_ENDIAN)
            for (i in 0 until indexCount) {
                indexBuffer.put(idxSrcBuffer.short.toInt() and 0xFFFF)
            }
        }
        indexBuffer.position(0)

        // 8. Extract Embedded PNG Texture
        var textureBitmap: Bitmap? = null
        if (gltf.has("images")) {
            val images = gltf.getJSONArray("images")
            if (images.length() > 0) {
                val img = images.getJSONObject(0)
                val imgBvIdx = img.getInt("bufferView")
                val imgBv = bufferViews.getJSONObject(imgBvIdx)
                val imgByteOffset = binOffset + imgBv.optInt("byteOffset", 0)
                val imgByteLength = imgBv.getInt("byteLength")
                textureBitmap = BitmapFactory.decodeByteArray(bytes, imgByteOffset, imgByteLength)
            }
        }

        Log.i(TAG, "Successfully loaded 3D GLB mesh: $posCount vertices, ${indexCount / 3} triangles, texture=${textureBitmap?.width}x${textureBitmap?.height}")
        return GlbMesh(
            vertexBuffer = vertexBuffer,
            texCoordBuffer = texCoordBuffer,
            indexBuffer = indexBuffer,
            vertexCount = posCount,
            indexCount = indexCount,
            textureBitmap = textureBitmap
        )
    }
}
