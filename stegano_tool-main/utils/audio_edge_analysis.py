
from pathlib import Path

import cv2
import numpy as np
from scipy import signal


def analyze_audio_edges(
    samples,
    sample_rate=44100,
    channels=1,
    output_dir="audio_edge_results",
):
    """Analyze audio using a spectrogram, Sobel, and Canny.

    The input samples are never modified.
    """
    audio = np.asarray(samples).reshape(-1)

    if audio.size < 8:
        raise ValueError("Audio is too short for analysis.")

    if sample_rate <= 0 or channels < 1:
        raise ValueError("Invalid sample rate or channel count.")

    if audio.size % channels != 0:
        raise ValueError("Audio samples do not match the channel count.")

    # Separate interleaved channels for analysis only.
    frames = audio.reshape(-1, channels).astype(np.float64)
    mono = frames.mean(axis=1)

    if not np.all(np.isfinite(mono)):
        raise ValueError("Audio contains invalid numeric values.")

    # Create a time-frequency spectrogram.
    nperseg = min(1024, mono.size)
    noverlap = min(512, nperseg - 1)

    _, _, spectrum = signal.stft(
        mono,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
    )

    magnitude_db = 20 * np.log10(np.abs(spectrum) + 1e-10)

    low, high = np.percentile(magnitude_db, [1, 99.5])
    if high <= low:
        high = low + 1.0

    spectrogram = np.clip(
        (magnitude_db - low) / (high - low) * 255,
        0, 255,
    ).astype(np.uint8)

    # Sobel edge map.
    gx = cv2.Sobel(spectrogram, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(spectrogram, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)

    max_value = float(magnitude.max())
    if max_value > 0:
        sobel = np.clip(
            magnitude / max_value * 255, 0, 255
        ).astype(np.uint8)
    else:
        sobel = np.zeros_like(spectrogram)

    # Canny edge map.
    canny = cv2.Canny(spectrogram, 50, 150)

    # Save analysis images in a separate output folder.
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)

    results = {
        "spectrogram": spectrogram,
        "sobel_edges": sobel,
        "canny_edges": canny,
    }

    filenames = {
        "spectrogram": "spectrogram.png",
        "sobel_edges": "sobel_edges.png",
        "canny_edges": "canny_edges.png",
    }

    saved_files = {}
    for key, image in results.items():
        path = folder / filenames[key]
        if not cv2.imwrite(str(path), image):
            raise OSError(f"Could not save image: {path}")
        saved_files[key] = str(path)

    results["files"] = saved_files
    return results
