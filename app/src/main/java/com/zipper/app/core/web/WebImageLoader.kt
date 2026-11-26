package com.zipper.app.core.web

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.util.LruCache
import android.widget.ImageView
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.Executors

/**
 * Ultra-lightweight, zero-dependency async image loader for Karvish Web cards.
 * Uses an in-memory LruCache, async worker threads, downsampling, and tag validation
 * to prevent flickering and OutOfMemory errors on low-memory Android devices.
 */
object WebImageLoader {

    private const val TAG = "WebImageLoader"
    private val mainHandler = Handler(Looper.getMainLooper())
    private val executor = Executors.newFixedThreadPool(2)

    private val maxMemory = (Runtime.getRuntime().maxMemory() / 1024).toInt()
    private val cacheSize = maxMemory / 8 // 1/8th of available memory for cache
    private val memoryCache = object : LruCache<String, Bitmap>(if (cacheSize > 0) cacheSize else 1024 * 8) {
        override fun sizeOf(key: String, bitmap: Bitmap): Int {
            return bitmap.byteCount / 1024
        }
    }

    fun load(context: Context, url: String?, target: ImageView) {
        if (url.isNullOrBlank()) {
            target.setImageDrawable(null)
            return
        }

        // Tag view with URL to guard against fast RecyclerView recycling
        target.tag = url

        // Check memory cache first (instant 0ms render)
        val cached = memoryCache.get(url)
        if (cached != null && !cached.isRecycled) {
            target.setImageBitmap(cached)
            return
        }

        // Placeholder or clear while loading
        target.setImageDrawable(null)

        // Handle Base64 inline images (e.g. Google Search thumbnails)
        if (url.startsWith("data:image", ignoreCase = true) && url.contains(",")) {
            executor.execute {
                try {
                    val base64Data = url.substringAfter(",")
                    val bytes = android.util.Base64.decode(base64Data, android.util.Base64.DEFAULT)
                    val bitmap = decodeSampledBitmap(bytes, reqWidth = 320, reqHeight = 320)
                    if (bitmap != null) {
                        memoryCache.put(url, bitmap)
                        mainHandler.post {
                            if (target.tag == url) {
                                target.setImageBitmap(bitmap)
                            }
                        }
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to decode Base64 image: ${e.message}")
                }
            }
            return
        }

        executor.execute {
            var bitmap: Bitmap? = null
            var conn: HttpURLConnection? = null
            var stream: InputStream? = null
            try {
                val u = URL(url)
                conn = u.openConnection() as HttpURLConnection
                conn.connectTimeout = 4000
                conn.readTimeout = 4000
                conn.instanceFollowRedirects = true
                conn.setRequestProperty("User-Agent", "Mozilla/5.0 (Linux; Android 14; Mobile) AppleWebKit/537.36")
                conn.connect()

                if (conn.responseCode == HttpURLConnection.HTTP_OK) {
                    val bytes = conn.inputStream.readBytes()
                    bitmap = decodeSampledBitmap(bytes, reqWidth = 320, reqHeight = 320)
                }
            } catch (e: Exception) {
                Log.w(TAG, "Failed to load image from $url: ${e.message}")
            } finally {
                try { stream?.close() } catch (_: Exception) {}
                try { conn?.disconnect() } catch (_: Exception) {}
            }

            if (bitmap != null) {
                memoryCache.put(url, bitmap)
                mainHandler.post {
                    if (target.tag == url) {
                        target.setImageBitmap(bitmap)
                    }
                }
            }
        }
    }

    private fun decodeSampledBitmap(data: ByteArray, reqWidth: Int, reqHeight: Int): Bitmap? {
        val options = BitmapFactory.Options().apply {
            inJustDecodeBounds = true
        }
        BitmapFactory.decodeByteArray(data, 0, data.size, options)

        var inSampleSize = 1
        if (options.outHeight > reqHeight || options.outWidth > reqWidth) {
            val halfHeight: Int = options.outHeight / 2
            val halfWidth: Int = options.outWidth / 2
            while ((halfHeight / inSampleSize) >= reqHeight && (halfWidth / inSampleSize) >= reqWidth) {
                inSampleSize *= 2
            }
        }

        options.inJustDecodeBounds = false
        options.inSampleSize = inSampleSize
        return BitmapFactory.decodeByteArray(data, 0, data.size, options)
    }
}
