"""
Train the separate audio CNN using labeled spectrogram regions.

Expected dataset:
data/audio_cnn/regions.npz

Required arrays:
- regions: (N, 32, 32), grayscale regions with values from 0 to 255
- labels:  (N,), suitability labels from 0 to 1

This script does not modify the existing image CNN.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from audio_cnn_classifier import AudioSuitabilityCNN


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "audio_cnn" / "regions.npz"
WEIGHTS_PATH = PROJECT_ROOT / "models" / "weights" / "audio_suitability_cnn.pth"


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            "Create a labeled dataset before training the audio CNN."
        )

    data = np.load(DATASET_PATH)
    regions = data["regions"]
    labels = data["labels"]

    if regions.ndim != 3 or regions.shape[1:] != (32, 32):
        raise ValueError("regions must have shape (N, 32, 32).")

    if labels.ndim != 1 or len(labels) != len(regions):
        raise ValueError("labels must have shape (N,) and match regions.")

    if len(regions) < 2:
        raise ValueError("At least two labeled regions are required.")

    if not np.isfinite(regions).all() or not np.isfinite(labels).all():
        raise ValueError("Dataset contains invalid numeric values.")

    if labels.min() < 0 or labels.max() > 1:
        raise ValueError("Labels must be between 0 and 1.")

    # Normalize grayscale values to 0–1.
    x = torch.from_numpy(regions.astype(np.float32) / 255.0)
    x = x.unsqueeze(1)

    y = torch.from_numpy(labels.astype(np.float32)).unsqueeze(1)

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = AudioSuitabilityCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    model.train()
    epochs = 20

    for epoch in range(epochs):
        total_loss = 0.0

        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)

        average_loss = total_loss / len(dataset)
        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Training loss: {average_loss:.4f}"
        )

    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"\nModel weights saved to: {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()