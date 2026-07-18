"""
cnn_classifier.py

CNN that predicts a continuous embedding-suitability SCORE (0-1) for a
16x16 image patch, instead of a binary edge/non-edge classification.
Higher score = more texture / less perceptible if modified = better
place to embed data.

Architecture: two conv blocks (with BatchNorm) + one residual block,
then a small fully-connected head ending in a sigmoid.

IMPORTANT: the caller is responsible for masking the low bits of the
image before extracting patches to score (see core/intelligent_pixel_selector.py
and utils/image_utils.mask_low_bits). This keeps the CNN's inference
input identical between the cover image and any stego image derived
from it, which is what keeps encode/decode selection synchronized.
"""

import numpy as np
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Simple residual block: conv-bn-relu-conv-bn + skip connection."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = (nn.Identity() if in_channels == out_channels
                          else nn.Conv2d(in_channels, out_channels, 1))

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return self.relu(out)


class EdgePixelCNN(nn.Module):
    """
    Regression CNN: predicts a single suitability score in [0, 1] for a
    patch_size x patch_size x 3 (BGR) input patch.
    """

    def __init__(self, patch_size: int = 16, in_channels: int = 3):
        super().__init__()
        self.patch_size = patch_size

        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.res_block = ResidualBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2, 2)

        flattened_size = 128 * max(patch_size // 8, 1) * max(patch_size // 8, 1)
        self.fc1 = nn.Linear(flattened_size, 128)
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.relu_fc2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.res_block(x))
        x = x.reshape(x.size(0), -1)
        x = self.relu_fc1(self.fc1(x))
        x = self.dropout(x)
        x = self.relu_fc2(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)

    def score_patch(self, patch) -> float:
        """
        Score a single patch, already the correct patch_size x patch_size x 3.
        Accepts a numpy array (H, W, 3) BGR, uint8-like.
        """
        if isinstance(patch, np.ndarray):
            tensor = torch.from_numpy(np.ascontiguousarray(patch)).permute(2, 0, 1).unsqueeze(0)
        else:
            tensor = patch.unsqueeze(0) if patch.dim() == 3 else patch

        tensor = tensor.float() / 255.0
        device = next(self.parameters()).device
        tensor = tensor.to(device)

        with torch.no_grad():
            score = self(tensor).squeeze().item()
        return float(score)

    def score_patches_batch(self, patches: np.ndarray) -> np.ndarray:
        """
        Score a batch of patches at once (much faster than one-by-one).
        Args:
            patches: array of shape (N, patch_size, patch_size, 3)
        Returns:
            array of shape (N,) with scores in [0, 1]
        """
        tensor = torch.from_numpy(np.ascontiguousarray(patches)).permute(0, 3, 1, 2).float() / 255.0
        device = next(self.parameters()).device
        tensor = tensor.to(device)

        with torch.no_grad():
            scores = self(tensor).squeeze(-1).cpu().numpy()
        return scores


def load_pretrained(weights_path: str = "models/weights/edge_pixel_cnn.pth"):
    """
    Load a trained EdgePixelCNN from disk. Returns None (graceful
    fallback) if weights are missing, rather than raising, so the
    pipeline can still run using pure edge/texture scoring.
    """
    model = EdgePixelCNN()
    try:
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model
    except FileNotFoundError:
        return None
    except Exception:
        return None
