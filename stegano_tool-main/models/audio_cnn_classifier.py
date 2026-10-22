"""
audio_cnn_classifier.py

A separate CNN for scoring audio spectrogram regions.
This does not modify audio LSB embedding or extraction.
"""

import torch
import torch.nn as nn


class AudioSuitabilityCNN(nn.Module):
    """Predict a suitability score for a single-channel spectrogram region."""

    def __init__(self, input_size=32):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Input shape: (batch, 1, height, width)
        Output shape: (batch, 1), scores between 0 and 1.
        """
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(
                "Input must have shape (batch, 1, height, width)."
            )

        x = self.features(x)
        return self.regressor(x)

    def score_region(self, region):
        """
        Score one grayscale spectrogram region.

        region: a 2D NumPy array or a PyTorch tensor.
        Values should be normalized to the range 0–1.
        """
        if not isinstance(region, torch.Tensor):
            raise TypeError("region must be a PyTorch tensor.")

        if region.ndim != 2:
            raise ValueError("region must be a 2D spectrogram region.")

        device = next(self.parameters()).device
        tensor = region.float().unsqueeze(0).unsqueeze(0).to(device)

        was_training = self.training
        self.eval()

        try:
            with torch.no_grad():
                score = self(tensor).item()
        finally:
            self.train(was_training)

        return float(score)