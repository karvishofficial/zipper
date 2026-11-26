package com.zipper.app.core.remote

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.location.Location
import android.location.LocationManager
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.net.wifi.WifiManager
import android.os.BatteryManager
import android.util.Log
import com.zipper.app.core.notification.NotificationCategory
import com.zipper.app.core.notification.NotificationStorageManager

/**
 * Provides silent background device diagnostics (Battery, Wi-Fi, Location, OTP)
 * for Karvish Remote Co-Pilot queries.
 */
object RemoteSystemDiagnostics {

    private const val TAG = "RemoteDiagnostics"

    /**
     * Reads battery percentage, charging state, and active network without waking the screen.
     */
    fun getBatteryAndNetworkStatus(context: Context): String {
        var batteryPct = -1
        var isCharging = false

        // 1. Direct BatteryManager query (Requires 0 permissions, never throws, works on modern Android)
        try {
            val bm = context.getSystemService(Context.BATTERY_SERVICE) as? BatteryManager
            if (bm != null) {
                val capacity = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
                if (capacity in 0..100) {
                    batteryPct = capacity
                }
                val status = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_STATUS)
                isCharging = status == BatteryManager.BATTERY_STATUS_CHARGING ||
                        status == BatteryManager.BATTERY_STATUS_FULL
            }
        } catch (e: Exception) {
            Log.w(TAG, "BatteryManager service query warning: ${e.message}")
        }

        // Fallback: sticky battery intent if BatteryManager capacity was unavailable
        if (batteryPct < 0) {
            try {
                val batteryFilter = IntentFilter(Intent.ACTION_BATTERY_CHANGED)
                val batteryIntent = context.registerReceiver(null, batteryFilter)
                val level = batteryIntent?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
                val scale = batteryIntent?.getIntExtra(BatteryManager.EXTRA_SCALE, -1) ?: -1
                val status = batteryIntent?.getIntExtra(BatteryManager.EXTRA_STATUS, -1) ?: -1
                isCharging = status == BatteryManager.BATTERY_STATUS_CHARGING ||
                        status == BatteryManager.BATTERY_STATUS_FULL

                if (level >= 0 && scale > 0) {
                    batteryPct = (level * 100 / scale.toFloat()).toInt()
                }
            } catch (e: Exception) {
                Log.w(TAG, "Sticky battery intent query warning: ${e.message}")
            }
        }

        val batteryDesc = if (batteryPct in 0..100) "$batteryPct%" else "Available"
        val chargingDesc = if (isCharging) "Yes ⚡" else "No"

        // 2. Wi-Fi and Connectivity (Fully isolated - any Wi-Fi permission/exception will never fail battery output)
        var netDesc = "📶 Network Active"
        try {
            val connectivityManager = context.getSystemService(Context.CONNECTIVITY_SERVICE) as? ConnectivityManager
            val activeNetwork = connectivityManager?.activeNetwork
            val capabilities = connectivityManager?.getNetworkCapabilities(activeNetwork)

            val isWifi = capabilities?.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) == true
            val isCellular = capabilities?.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR) == true

            netDesc = when {
                isWifi -> {
                    var wifiSsid = "Connected"
                    try {
                        val wifiManager = context.applicationContext.getSystemService(Context.WIFI_SERVICE) as? WifiManager
                        val ssid = wifiManager?.connectionInfo?.ssid?.replace("\"", "").orEmpty()
                        if (ssid.isNotBlank() && ssid != "<unknown ssid>") {
                            wifiSsid = ssid
                        }
                    } catch (e: Exception) {
                        Log.d(TAG, "SSID read skipped or restricted: ${e.message}")
                    }
                    "📶 Wi-Fi ($wifiSsid)"
                }
                isCellular -> "📱 Mobile Data"
                else -> "📴 Offline"
            }
        } catch (e: Exception) {
            Log.w(TAG, "Network inspection warning: ${e.message}")
        }

        return "🔋 Battery: $batteryDesc (Charging: $chargingDesc) | $netDesc"
    }

    /**
     * Retrieves the last known GPS or Network location and generates a Google Maps URL.
     */
    fun getLocationUrl(context: Context): String {
        return try {
            val locationManager = context.getSystemService(Context.LOCATION_SERVICE) as? LocationManager
                ?: return "📍 Location service unavailable on device."

            val hasFine = context.checkSelfPermission(android.Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED
            val hasCoarse = context.checkSelfPermission(android.Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED

            if (!hasFine && !hasCoarse) {
                return "📍 Location permission not yet granted in Zipper App Settings."
            }

            var bestLoc: Location? = null
            if (locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)) {
                try {
                    bestLoc = locationManager.getLastKnownLocation(LocationManager.GPS_PROVIDER)
                } catch (e: SecurityException) {
                    Log.w(TAG, "GPS location security exception: ${e.message}")
                }
            }

            if (bestLoc == null && locationManager.isProviderEnabled(LocationManager.NETWORK_PROVIDER)) {
                try {
                    bestLoc = locationManager.getLastKnownLocation(LocationManager.NETWORK_PROVIDER)
                } catch (e: SecurityException) {
                    Log.w(TAG, "Network location security exception: ${e.message}")
                }
            }

            if (bestLoc != null) {
                val lat = String.format(java.util.Locale.US, "%.5f", bestLoc.latitude)
                val lon = String.format(java.util.Locale.US, "%.5f", bestLoc.longitude)
                val accuracy = bestLoc.accuracy.toInt()
                "📍 Phone Location: https://maps.google.com/?q=$lat,$lon (Accuracy: ~${accuracy}m)"
            } else {
                "📍 Location unavailable (No recent GPS fix recorded)."
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error fetching location: ${e.message}")
            "📍 Location unavailable: ${e.message}"
        }
    }

    /**
     * Retrieves the latest OTP received in the last 15 minutes.
     */
    fun getLastOtp(context: Context): String {
        return try {
            val otpItem = NotificationStorageManager.getInstance(context).getLatestOtp()
            val now = System.currentTimeMillis()
            val fifteenMinsAgo = now - (15 * 60 * 1000L)

            if (otpItem != null && !otpItem.otpCode.isNullOrBlank() && otpItem.timestamp >= fifteenMinsAgo) {
                val minutesAgo = ((now - otpItem.timestamp) / 60000L).coerceAtLeast(0L)
                val timeDesc = if (minutesAgo == 0L) "just now" else "$minutesAgo mins ago"
                "🔐 Last OTP: ${otpItem.otpCode} (${otpItem.appName} - $timeDesc)"
            } else {
                "🔐 No OTP received in the last 15 minutes."
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error fetching last OTP: ${e.message}")
            "🔐 Unable to read recent OTP."
        }
    }
}
