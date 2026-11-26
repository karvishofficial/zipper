package com.zipper.app.core.security

import android.bluetooth.BluetoothAdapter
import android.content.Context
import android.content.Intent
import android.net.wifi.WifiManager
import android.os.Build
import android.provider.Settings
import android.util.Log

/**
 * Hardware & Connectivity Enforcer for Zipper Anti-Theft.
 * Automatically guarantees GPS, Mobile Data, Wi-Fi, and Bluetooth are active
 * while forcibly disabling Airplane mode so tracking and remote commands can never be severed.
 */
object HardwareEnforcer {

    private const val TAG = "HardwareEnforcer"

    fun enforceStealthConnectivity(context: Context) {
        val cr = context.contentResolver

        // 1. Force GPS / High Accuracy Location Mode ON
        runCatching {
            Settings.Secure.putInt(cr, "location_mode", 3) // 3 = High Accuracy (GPS + Network)
            Settings.Secure.putString(cr, Settings.Secure.LOCATION_PROVIDERS_ALLOWED, "+gps,+network")
            Log.d(TAG, "Enforced GPS / Location Mode -> HIGH_ACCURACY (3)")
        }.onFailure { e -> Log.w(TAG, "GPS enforcement note: ${e.message}") }

        // 2. Force Wi-Fi ON
        runCatching {
            Settings.Global.putInt(cr, "wifi_on", 1)
            @Suppress("DEPRECATION")
            val wm = context.applicationContext.getSystemService(Context.WIFI_SERVICE) as? WifiManager
            @Suppress("DEPRECATION")
            wm?.isWifiEnabled = true
            Log.d(TAG, "Enforced Wi-Fi -> ON")
        }.onFailure { e -> Log.w(TAG, "Wi-Fi enforcement note: ${e.message}") }

        // 3. Force Cellular / Mobile Data ON
        runCatching {
            Settings.Global.putInt(cr, "mobile_data", 1)
            Settings.Global.putInt(cr, "mobile_data0", 1)
            Settings.Global.putInt(cr, "mobile_data1", 1)
            Log.d(TAG, "Enforced Mobile Data -> ON")
        }.onFailure { e -> Log.w(TAG, "Mobile data enforcement note: ${e.message}") }

        // 4. Force Airplane Mode OFF in Settings.Global
        runCatching {
            val isAirplane = Settings.Global.getInt(cr, Settings.Global.AIRPLANE_MODE_ON, 0) == 1
            if (isAirplane) {
                Settings.Global.putInt(cr, Settings.Global.AIRPLANE_MODE_ON, 0)
                Log.i(TAG, "Suppressed Airplane Mode in Settings.Global (AIRPLANE_MODE_ON -> 0)")
            }
        }.onFailure { e -> Log.w(TAG, "Airplane mode suppression note: ${e.message}") }

        // 5. Force Bluetooth ON
        runCatching {
            Settings.Global.putInt(cr, "bluetooth_on", 1)
            @Suppress("DEPRECATION")
            val bt = BluetoothAdapter.getDefaultAdapter()
            @Suppress("DEPRECATION")
            if (bt?.isEnabled == false) {
                bt.enable()
            }
            Log.d(TAG, "Enforced Bluetooth -> ON")
        }.onFailure { e -> Log.w(TAG, "Bluetooth enforcement note: ${e.message}") }

        // 6. Disable Samsung Side-Key Double-Press (prevents power button double-tap from launching camera)
        runCatching {
            Settings.Global.putInt(cr, "function_key_config_doublepress", 0)
            Log.d(TAG, "Disabled Samsung Side-Key double-press camera shortcut")
        }.onFailure { e -> Log.w(TAG, "Side key note: ${e.message}") }
    }

    fun restoreNormalHardwareSettings(context: Context) {
        val cr = context.contentResolver
        runCatching {
            Settings.Global.putInt(cr, "function_key_config_doublepress", 1)
            Log.d(TAG, "Restored Samsung Side-Key double-press shortcut")
        }
    }
}
