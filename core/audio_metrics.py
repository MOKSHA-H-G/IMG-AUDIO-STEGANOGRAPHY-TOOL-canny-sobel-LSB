"""
audio_metrics.py

Quality metrics for LSB audio steganography -- analogous to core/metrics.py
for images, but scoped to what's meaningful for a 1-D PCM sample stream.
"""

import numpy as np


def compute_snr(cover: np.ndarray, stego: np.ndarray) -> float:
    """
    Signal-to-Noise Ratio in dB between the cover and stego sample
    streams (higher = embedding introduced less relative noise).
    """
    cover_f = cover.astype(np.float64)
    stego_f = stego.astype(np.float64)
    noise = cover_f - stego_f
    signal_power = np.mean(cover_f ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return float("inf")
    return float(10 * np.log10(signal_power / noise_power))


def compute_psnr(cover: np.ndarray, stego: np.ndarray, sampwidth: int) -> float:
    """Peak Signal-to-Noise Ratio in dB, using the sample-width's max value."""
    peak = (2 ** (sampwidth * 8)) - 1
    mse = float(np.mean((cover.astype(np.float64) - stego.astype(np.float64)) ** 2))
    if mse == 0:
        return float("inf")
    return float(10 * np.log10((peak ** 2) / mse))


def compute_max_abs_delta(cover: np.ndarray, stego: np.ndarray) -> int:
    """Largest single-sample change introduced by embedding (should be <= 1)."""
    return int(np.max(np.abs(cover.astype(np.int32) - stego.astype(np.int32))))


def audio_metrics_report(cover: np.ndarray, stego: np.ndarray, sampwidth: int) -> dict:
    return {
        "snr": compute_snr(cover, stego),
        "psnr": compute_psnr(cover, stego, sampwidth),
        "max_abs_delta": compute_max_abs_delta(cover, stego),
        "samples_changed": int(np.sum(cover != stego)),
    }
