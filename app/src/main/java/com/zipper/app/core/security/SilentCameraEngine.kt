package com.zipper.app.core.security

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.ImageFormat
import android.hardware.camera2.*
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.util.Size
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withTimeoutOrNull
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.coroutines.resume

data class DualPhotoEvidence(
    val frontPhoto: ByteArray?,
    val backPhoto: ByteArray?
)

/**
 * Headless Camera2 engine for taking silent photos from front and back cameras without showing
 * any UI, activity, or viewfinder surface.
 */
@Suppress("DEPRECATION")
object SilentCameraEngine {

    private const val TAG = "SilentCameraEngine"

    suspend fun captureFrontPhoto(context: Context, timeoutMs: Long = 4000L): ByteArray? {
        return capturePhotoByFacing(context, CameraCharacteristics.LENS_FACING_FRONT, timeoutMs)
    }

    suspend fun captureBackPhoto(context: Context, timeoutMs: Long = 4000L): ByteArray? {
        return capturePhotoByFacing(context, CameraCharacteristics.LENS_FACING_BACK, timeoutMs)
    }

    suspend fun captureBothPhotos(context: Context): DualPhotoEvidence {
        val front = captureFrontPhoto(context)
        // Camera HAL needs ~500ms to cleanly release sensor pipeline before switching facing
        kotlinx.coroutines.delay(600L)
        val back = captureBackPhoto(context)
        return DualPhotoEvidence(frontPhoto = front, backPhoto = back)
    }

    @SuppressLint("MissingPermission")
    suspend fun capturePhotoByFacing(context: Context, lensFacing: Int, timeoutMs: Long = 4000L): ByteArray? {
        val firstTry = executeSingleCapture(context, lensFacing, timeoutMs)
        if (firstTry != null) return firstTry
        // If first attempt failed (e.g. HAL busy releasing previous camera), wait 500ms and retry once
        Log.w(TAG, "First capture attempt for facing $lensFacing returned null; retrying after 500ms HAL release...")
        kotlinx.coroutines.delay(500L)
        return executeSingleCapture(context, lensFacing, timeoutMs)
    }

    @SuppressLint("MissingPermission")
    private suspend fun executeSingleCapture(context: Context, lensFacing: Int, timeoutMs: Long): ByteArray? {
        val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as? CameraManager
        if (cameraManager == null) {
            Log.e(TAG, "CameraManager not available")
            return null
        }

        val hasCamera = androidx.core.content.ContextCompat.checkSelfPermission(
            context, android.Manifest.permission.CAMERA
        ) == android.content.pm.PackageManager.PERMISSION_GRANTED
        if (!hasCamera) {
            Log.w(TAG, "Camera permission not granted, skipping silent photo capture")
            return null
        }

        val cameraId = findCameraId(cameraManager, lensFacing)
        if (cameraId == null) {
            Log.e(TAG, "No camera found with facing: $lensFacing")
            return null
        }

        return withTimeoutOrNull(timeoutMs) {
            suspendCancellableCoroutine { continuation ->
                val thread = HandlerThread("SilentCameraThread_${lensFacing}_${System.currentTimeMillis()}").apply { start() }
                val handler = Handler(thread.looper)

                var cameraDevice: CameraDevice? = null
                var captureSession: CameraCaptureSession? = null
                var imageReader: ImageReader? = null
                val resumed = AtomicBoolean(false)

                fun cleanUp() {
                    runCatching { captureSession?.close() }
                    runCatching { cameraDevice?.close() }
                    runCatching { imageReader?.close() }
                    runCatching { thread.quitSafely() }
                }

                continuation.invokeOnCancellation {
                    cleanUp()
                }

                try {
                    val characteristics = cameraManager.getCameraCharacteristics(cameraId)
                    val map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
                    val targetSize = selectOptimalSize(map?.getOutputSizes(ImageFormat.JPEG))

                    Log.d(TAG, "Using silent camera size: ${targetSize.width}x${targetSize.height} for camera $cameraId")

                    imageReader = ImageReader.newInstance(targetSize.width, targetSize.height, ImageFormat.JPEG, 2)
                    imageReader!!.setOnImageAvailableListener({ reader ->
                        val image = runCatching { reader.acquireLatestImage() }.getOrNull()
                        if (image != null) {
                            try {
                                val plane = image.planes[0]
                                val buffer = plane.buffer
                                val bytes = ByteArray(buffer.remaining())
                                buffer.get(bytes)

                                Log.d(TAG, "Captured photo from camera $cameraId: ${bytes.size} bytes")
                                if (resumed.compareAndSet(false, true)) {
                                    continuation.resume(bytes)
                                }
                            } finally {
                                image.close()
                                cleanUp()
                            }
                        }
                    }, handler)

                    cameraManager.openCamera(cameraId, object : CameraDevice.StateCallback() {
                        override fun onOpened(camera: CameraDevice) {
                            cameraDevice = camera
                            try {
                                val targetSurface = imageReader!!.surface
                                camera.createCaptureSession(
                                    listOf(targetSurface),
                                    object : CameraCaptureSession.StateCallback() {
                                        override fun onConfigured(session: CameraCaptureSession) {
                                            captureSession = session
                                            try {
                                                val captureBuilder = camera.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE).apply {
                                                    addTarget(targetSurface)
                                                    set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
                                                    set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
                                                }
                                                session.capture(captureBuilder.build(), null, handler)
                                                Log.d(TAG, "Still capture request dispatched for camera $cameraId")
                                            } catch (e: Exception) {
                                                Log.e(TAG, "Failed to dispatch capture request: ${e.message}")
                                                if (resumed.compareAndSet(false, true)) {
                                                    continuation.resume(null)
                                                }
                                                cleanUp()
                                            }
                                        }

                                        override fun onConfigureFailed(session: CameraCaptureSession) {
                                            Log.e(TAG, "Camera capture session configuration failed")
                                            if (resumed.compareAndSet(false, true)) {
                                                continuation.resume(null)
                                            }
                                            cleanUp()
                                        }
                                    },
                                    handler
                                )
                            } catch (e: Exception) {
                                Log.e(TAG, "Failed to create capture session: ${e.message}")
                                if (resumed.compareAndSet(false, true)) {
                                    continuation.resume(null)
                                }
                                cleanUp()
                            }
                        }

                        override fun onDisconnected(camera: CameraDevice) {
                            Log.w(TAG, "Camera disconnected")
                            if (resumed.compareAndSet(false, true)) {
                                continuation.resume(null)
                            }
                            cleanUp()
                        }

                        override fun onError(camera: CameraDevice, error: Int) {
                            Log.e(TAG, "Camera open error: $error")
                            if (resumed.compareAndSet(false, true)) {
                                continuation.resume(null)
                            }
                            cleanUp()
                        }
                    }, handler)

                } catch (e: Exception) {
                    Log.e(TAG, "Exception opening camera: ${e.message}")
                    if (resumed.compareAndSet(false, true)) {
                        continuation.resume(null)
                    }
                    cleanUp()
                }
            }
        }
    }

    private fun findCameraId(cameraManager: CameraManager, targetFacing: Int): String? {
        for (id in cameraManager.cameraIdList) {
            val characteristics = cameraManager.getCameraCharacteristics(id)
            val facing = characteristics.get(CameraCharacteristics.LENS_FACING)
            if (facing == targetFacing) {
                return id
            }
        }
        return if (targetFacing == CameraCharacteristics.LENS_FACING_FRONT) {
            cameraManager.cameraIdList.firstOrNull()
        } else {
            cameraManager.cameraIdList.lastOrNull()
        }
    }

    private fun selectOptimalSize(sizes: Array<Size>?): Size {
        if (sizes.isNullOrEmpty()) return Size(1280, 720)
        val preferred = sizes.filter { it.width in 1280..1920 && it.height in 720..1080 }
        return preferred.minByOrNull { it.width * it.height } ?: sizes[0]
    }
}
