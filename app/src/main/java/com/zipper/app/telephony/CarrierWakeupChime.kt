package com.zipper.app.telephony

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

/**
 * Pure programmatic synthesizer that generates a soft 200ms dual-tone chime
 * to open the cellular carrier Voice Activity Detection (VAD) gate and phone AGC
 * before speech starts, preventing the first words from being cut off.
 *
 * 100% offline, zero audio asset files required.
 */
object CarrierWakeupChime {
    private const val TAG = "CarrierWakeupChime"
    private const val SAMPLE_RATE = 44100
    private const val DURATION_MS = 220

    suspend fun play() = withContext(Dispatchers.IO) {
        var track: AudioTrack? = null
        try {
            val numSamples = (SAMPLE_RATE * DURATION_MS) / 1000
            val pcmData = ShortArray(numSamples)

            // Pleasant dual-harmony chime: C5 (523.25 Hz) + G5 (783.99 Hz)
            val freq1 = 523.25
            val freq2 = 783.99

            for (i in 0 until numSamples) {
                // Raised cosine envelope for gentle attack and smooth decay (zero clicks)
                val t = i.toDouble() / numSamples
                val envelope = 0.5 * (1.0 - cos(2.0 * PI * t))

                val sample1 = sin(2.0 * PI * freq1 * (i.toDouble() / SAMPLE_RATE))
                val sample2 = sin(2.0 * PI * freq2 * (i.toDouble() / SAMPLE_RATE))
                val combined = (sample1 * 0.6 + sample2 * 0.4) * envelope

                pcmData[i] = (combined * 31500.0).toInt().coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt()).toShort()
            }

            val bufferSize = numSamples * 2
            val audioAttributes = AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_ALARM)
                .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                .build()

            val audioFormat = AudioFormat.Builder()
                .setSampleRate(SAMPLE_RATE)
                .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                .build()

            track = AudioTrack.Builder()
                .setAudioAttributes(audioAttributes)
                .setAudioFormat(audioFormat)
                .setBufferSizeInBytes(bufferSize)
                .setTransferMode(AudioTrack.MODE_STATIC)
                .build()

            track.write(pcmData, 0, pcmData.size)
            track.play()
            // Wait for duration of chime
            kotlinx.coroutines.delay(DURATION_MS.toLong() + 30L)
            Log.d(TAG, "Carrier VAD wakeup chime played successfully.")
        } catch (e: Exception) {
            Log.w(TAG, "Failed to play carrier wakeup chime: ${e.message}")
        } finally {
            try {
                track?.stop()
                track?.release()
            } catch (_: Exception) {}
        }
    }
}
