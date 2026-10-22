
"""
Create transparent, heuristic suitability labels for audio
spectrogram regions.

These labels are a starting point for experimentation, not
ground-truth proof of perceptual quality.
"""

from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "audio_cnn" / "unlabeled_regions.npz"
OUTPUT_PATH = PROJECT_ROOT / "data" / "audio_cnn" / "regions.npz"


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_PATH}")

    data = np.load(INPUT_PATH)
    regions = data["regions"]

    if regions.ndim != 3 or regions.shape[1:] != (32, 32):
        raise ValueError("Expected regions with shape (N, 32, 32).")

    if len(regions) < 2:
        raise ValueError("At least two regions are needed.")

    # Normalize each region to 0–1.
    x = regions.astype(np.float32) / 255.0

    # Calculate simple, explainable region features.
    contrast = x.std(axis=(1, 2))
    edge_strength = (
        np.abs(np.diff(x, axis=1)).mean(axis=(1))
        .mean(axis=1)
    )

    # Normalize features across this dataset.
    def normalize(values):
        lo, hi = values.min(), values.max()
        if hi <= lo:
            return np.zeros_like(values)
        return (values - lo) / (hi - lo)

    contrast_n = normalize(contrast)
    edge_n = normalize(edge_strength)

    # Experimental heuristic: prefer regions with greater
    # local contrast and spectral variation.
    # These are proxy labels, not verified perceptual-quality labels.
    scores = 0.5 * contrast_n + 0.5 * edge_n

    # Keep scores continuous in [0, 1] for the CNN regressor.
    labels = scores.astype(np.float32)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_PATH,
        regions=regions,
        labels=labels
    )

    print("Heuristic labels created.")
    print(f"Regions: {len(regions)}")
    print(f"Label range: {labels.min():.3f} to {labels.max():.3f}")
    print(f"Saved dataset: {OUTPUT_PATH}")
    print(
        "Note: labels are experimental proxies, not ground-truth "
        "evidence of inaudibility or optimal hiding locations."
    )


if __name__ == "__main__":
    main()
