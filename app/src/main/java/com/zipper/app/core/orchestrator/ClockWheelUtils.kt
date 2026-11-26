package com.zipper.app.core.orchestrator

/**
 * Utility functions for circular loop delta calculation and dynamic wheel navigation
 * across diverse Android OEM clock pickers (zero hardcoding).
 */
object ClockWheelUtils {

    /**
     * Calculates the shortest signed distance on a circular loop (12-hour or 60-minute).
     *
     * @param current The currently resting value on the wheel.
     * @param target The desired target value.
     * @param modulus The cycle modulus (e.g. 12 for 12-hour clock, 60 for minutes).
     * @return Positive steps for forward (increment), negative steps for backward (decrement).
     */
    fun shortestDistance(current: Int, target: Int, modulus: Int): Int {
        val fwd = ((target - current) % modulus + modulus) % modulus
        val bwd = ((current - target) % modulus + modulus) % modulus
        return if (fwd <= bwd) fwd else -bwd
    }
}
