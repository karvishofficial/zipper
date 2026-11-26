package com.zipper.app.core.remote

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioManager
import android.media.MediaPlayer
import android.media.RingtoneManager
import android.util.Log

/**
 * Handles Find My Phone / Emergency Audio Alarm playback.
 * Plays loud looping alarm audio on STREAM_ALARM (bypasses Silent / DND mode).
 */
object EmergencyRinger {

    private const val TAG = "EmergencyRinger"
    private var mediaPlayer: MediaPlayer? = null
    private var previousVolume: Int? = null

    @Synchronized
    fun startRinging(context: Context): Boolean {
        try {
            stopRinging(context)

            val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as? AudioManager
            if (audioManager != null) {
                // Save previous volume and boost to 100% max volume
                val stream = AudioManager.STREAM_ALARM
                previousVolume = audioManager.getStreamVolume(stream)
                val maxVol = audioManager.getStreamMaxVolume(stream)
                audioManager.setStreamVolume(stream, maxVol, 0)
            }

            val alarmUri = RingtoneManager.getDefaultUri(RingtoneManager.TYPE_ALARM)
                ?: RingtoneManager.getDefaultUri(RingtoneManager.TYPE_RINGTONE)

            mediaPlayer = MediaPlayer().apply {
                setDataSource(context, alarmUri)
                setAudioAttributes(
                    AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_ALARM)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                        .build()
                )
                isLooping = true
                prepare()
                start()
            }
            Log.i(TAG, "Emergency Siren started at maximum volume.")
            return true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start emergency ringing: ${e.message}", e)
            return false
        }
    }

    @Synchronized
    fun stopRinging(context: Context): Boolean {
        return try {
            mediaPlayer?.let {
                if (it.isPlaying) {
                    it.stop()
                }
                it.release()
            }
            mediaPlayer = null

            // Restore previous volume if saved
            previousVolume?.let { savedVol ->
                val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as? AudioManager
                audioManager?.setStreamVolume(AudioManager.STREAM_ALARM, savedVol, 0)
                previousVolume = null
            }
            Log.i(TAG, "Emergency Siren stopped.")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Error stopping emergency ringing: ${e.message}")
            false
        }
    }

    @Synchronized
    fun isRinging(): Boolean {
        return try {
            mediaPlayer?.isPlaying == true
        } catch (e: Exception) {
            false
        }
    }
}
