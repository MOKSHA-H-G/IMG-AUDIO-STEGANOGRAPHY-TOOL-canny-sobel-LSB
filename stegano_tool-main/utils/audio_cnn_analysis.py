
from pathlib import Path

import cv2
import numpy as np
import torch

from models.audio_cnn_classifier import AudioSuitabilityCNN
from utils.audio_edge_analysis import analyze_audio_edges


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = (
    PROJECT_ROOT
    / "models"
    / "weights"
    / "audio_suitability_cnn.pth"
)


def rank_audio_regions(samples, sample_rate, channels, top_n=5):
    """Rank spectrogram regions using the trained audio CNN."""

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Audio CNN weights not found: {WEIGHTS_PATH}"
        )

    results = analyze_audio_edges(
        samples,
        sample_rate=sample_rate,
        channels=channels,
    )

    spectrogram = np.asarray(results["spectrogram"])

    if spectrogram.ndim != 2:
        raise ValueError("Expected a 2D spectrogram.")

    height, width = spectrogram.shape
    region_size = 32

    # Match the grayscale input format used during training.
    if spectrogram.max() <= 1.0:
        gray = np.clip(spectrogram * 255, 0, 255).astype(np.uint8)
    else:
        gray = np.clip(spectrogram, 0, 255).astype(np.uint8)

    model = AudioSuitabilityCNN()
    state = torch.load(
        WEIGHTS_PATH,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()

    regions = []

    # Use non-overlapping 32 x 32 regions, as in dataset preparation.
    for y in range(0, height - region_size + 1, region_size):
        for x in range(0, width - region_size + 1, region_size):
            region = gray[y:y + region_size, x:x + region_size]

            # Resize only if needed to maintain the model input shape.
            if region.shape != (32, 32):
                region = cv2.resize(region, (32, 32))

            tensor = torch.from_numpy(
                region.astype(np.float32) / 255.0
            ).unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                score = model(tensor).item()

            regions.append({
                "x": x,
                "y": y,
                "score": float(score),
            })

    if not regions:
        raise ValueError(
            "The spectrogram is too small to contain a 32 x 32 region."
        )

    regions.sort(key=lambda item: item["score"], reverse=True)

    return {
        "edge_results": results,
        "regions_tested": len(regions),
        "top_regions": regions[:top_n],
    }
