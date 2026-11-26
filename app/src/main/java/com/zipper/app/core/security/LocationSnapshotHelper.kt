package com.zipper.app.core.security

import android.annotation.SuppressLint
import android.content.Context
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Bundle
import android.os.Looper
import android.util.Log
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withTimeoutOrNull
import kotlin.coroutines.resume

data class LocationSnapshot(
    val latitude: Double,
    val longitude: Double,
    val accuracyMeters: Float,
    val timestamp: Long,
    val mapsUrl: String,
    val altitudeMeters: Double = 0.0,
    val speedMps: Float = 0f,
    val bearingDegrees: Float = 0f
)

/**
 * Fast, lightweight location snapshot helper using Android's native LocationManager.
 * Zero external Play Services dependency required, works immediately in background.
 */
@Suppress("DEPRECATION")
object LocationSnapshotHelper {

    private const val TAG = "LocationSnapshotHelper"

    @SuppressLint("MissingPermission")
    suspend fun getBestLocation(context: Context, timeoutMs: Long = 6000L): LocationSnapshot? {
        val lm = context.getSystemService(Context.LOCATION_SERVICE) as? LocationManager ?: return null

        // 1. First check last known location (instantaneous)
        val lastGps = runCatching { lm.getLastKnownLocation(LocationManager.GPS_PROVIDER) }.getOrNull()
        val lastNetwork = runCatching { lm.getLastKnownLocation(LocationManager.NETWORK_PROVIDER) }.getOrNull()
        val lastPassive = runCatching { lm.getLastKnownLocation(LocationManager.PASSIVE_PROVIDER) }.getOrNull()

        val bestLastKnown = listOfNotNull(lastGps, lastNetwork, lastPassive)
            .filter { (System.currentTimeMillis() - it.time) < 180_000 } // within 3 minutes
            .minByOrNull { it.accuracy }

        if (bestLastKnown != null && bestLastKnown.accuracy <= 50f) {
            Log.d(TAG, "Using fresh last-known location (${bestLastKnown.accuracy}m accuracy)")
            return toSnapshot(bestLastKnown)
        }

        var candidate: Location? = bestLastKnown

        // 2. Request live GPS and Network updates with high-accuracy satellite threshold
        val snapshot = withTimeoutOrNull(timeoutMs) {
            suspendCancellableCoroutine<LocationSnapshot?> { continuation ->
                val resumed = java.util.concurrent.atomic.AtomicBoolean(false)

                val listener = object : LocationListener {
                    override fun onLocationChanged(loc: Location) {
                        Log.d(TAG, "Location fix acquired: ${loc.provider} -> ${loc.latitude},${loc.longitude} (~${loc.accuracy}m)")

                        // Keep best candidate seen so far
                        if (candidate == null || loc.accuracy < candidate!!.accuracy) {
                            candidate = loc
                        }

                        // If high-accuracy fix achieved (<= 25m, real GPS satellite lock), finish immediately
                        if (loc.accuracy <= 25f) {
                            Log.i(TAG, "High-accuracy GPS fix achieved (${loc.accuracy}m)")
                            runCatching { lm.removeUpdates(this) }
                            if (resumed.compareAndSet(false, true)) {
                                continuation.resume(toSnapshot(loc))
                            }
                        }
                    }

                    @Deprecated("Deprecated in Java")
                    override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) {}
                    override fun onProviderEnabled(provider: String) {}
                    override fun onProviderDisabled(provider: String) {}
                }

                continuation.invokeOnCancellation {
                    runCatching { lm.removeUpdates(listener) }
                }

                var requested = false
                if (lm.isProviderEnabled(LocationManager.GPS_PROVIDER)) {
                    runCatching {
                        lm.requestLocationUpdates(LocationManager.GPS_PROVIDER, 500L, 0f, listener, Looper.getMainLooper())
                        requested = true
                    }
                }
                if (lm.isProviderEnabled(LocationManager.NETWORK_PROVIDER)) {
                    runCatching {
                        lm.requestLocationUpdates(LocationManager.NETWORK_PROVIDER, 500L, 0f, listener, Looper.getMainLooper())
                        requested = true
                    }
                }

                if (!requested) {
                    val fallback = candidate ?: lastGps ?: lastNetwork ?: lastPassive
                    if (resumed.compareAndSet(false, true)) {
                        continuation.resume(fallback?.let { toSnapshot(it) })
                    }
                }
            }
        }

        // Return real satellite GPS fix if achieved, or best candidate acquired
        return snapshot ?: candidate?.let { toSnapshot(it) } ?: listOfNotNull(lastGps, lastNetwork, lastPassive).minByOrNull { it.accuracy }?.let { toSnapshot(it) }
    }

    private fun toSnapshot(loc: Location): LocationSnapshot {
        val url = "https://maps.google.com/?q=${loc.latitude},${loc.longitude}"
        return LocationSnapshot(
            latitude = loc.latitude,
            longitude = loc.longitude,
            accuracyMeters = loc.accuracy,
            timestamp = loc.time,
            mapsUrl = url,
            altitudeMeters = if (loc.hasAltitude()) loc.altitude else 0.0,
            speedMps = if (loc.hasSpeed()) loc.speed else 0f,
            bearingDegrees = if (loc.hasBearing()) loc.bearing else 0f
        )
    }
}
