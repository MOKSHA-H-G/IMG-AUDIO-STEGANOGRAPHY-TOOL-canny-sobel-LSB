
"""
Prepare spectrogram regions for the separate audio CNN.

This script does not modify audio files or the existing LSB algorithm.
It saves regions for later labeling and training.
"""

from pathlib import Path

import numpy as np
from scipy.signal import stft
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_PATH = PROJECT_ROOT / "test_audio.wav"
OUTPUT_DIR = PROJECT_ROOT / "data" / "audio_cnn"
REGION_SIZE = 32


def main():
    if not AUDIO_PATH.exists():
        raise FileNotFoundError(
            f"Audio file not found: {AUDIO_PATH}"
        )

    # Read the WAV file using the Python standard library.
    import wave

    with wave.open(str(AUDIO_PATH), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise ValueError("Expected 16-bit PCM WAV audio.")

        if wav.getnchannels() != 1:
            raise ValueError("Expected mono WAV audio.")

        sample_rate = wav.getframerate()
        samples = np.frombuffer(
            wav.readframes(wav.getnframes()),
            dtype="<i2"
        ).astype(np.float32)

    if len(samples) < 64:
        raise ValueError("Audio is too short to analyze.")

    # Create a spectrogram.
    _, _, spectrum = stft(
        samples,
        fs=sample_rate,
        nperseg=256,
        noverlap=128,
        boundary=None
    )

    magnitude = np.abs(spectrum)
    log_spectrum = 20 * np.log10(magnitude + 1e-8)

    # Normalize the spectrogram to grayscale values from 0 to 255.
    low, high = np.percentile(log_spectrum, [1, 99])

    if high <= low:
        high = low + 1.0

    grayscale = np.clip(
        (log_spectrum - low) / (high - low),
        0,
        1
    )
    grayscale = (grayscale * 255).astype(np.uint8)

    # Split into non-overlapping 32x32 regions.
    height, width = grayscale.shape
    regions = []

    for top in range(0, height - REGION_SIZE + 1, REGION_SIZE):
        for left in range(0, width - REGION_SIZE + 1, REGION_SIZE):
            region = grayscale[
                top:top + REGION_SIZE,
                left:left + REGION_SIZE
            ]
            regions.append(region)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save the spectrogram for visual inspection.
    Image.fromarray(grayscale).save(
        OUTPUT_DIR / "spectrogram_preview.png"
    )

    if regions:
        region_array = np.stack(regions)
    else:
        region_array = np.empty(
            (0, REGION_SIZE, REGION_SIZE),
            dtype=np.uint8
        )

    # Save regions only. Labels will be added after we define
    # a defensible suitability criterion.
    np.savez_compressed(
        OUTPUT_DIR / "unlabeled_regions.npz",
        regions=region_array
    )

    print("Audio analysis completed.")
    print(f"Audio: {AUDIO_PATH}")
    print(f"Spectrogram dimensions: {grayscale.shape}")
    print(f"Regions extracted: {len(region_array)}")
    print(f"Preview: {OUTPUT_DIR / 'spectrogram_preview.png'}")
    print(f"Regions: {OUTPUT_DIR / 'unlabeled_regions.npz'}")
    print("No training labels were created.")


if __name__ == "__main__":
    main()
