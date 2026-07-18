"""
train.py

Trains EdgePixelCNN as a REGRESSION model: given a 16x16 patch, predict
a continuous embedding-suitability score in [0, 1].

Label strategy (self-supervised, no manual annotation needed):
  label = normalized local variance of the patch's grayscale values.
  High variance (busy texture) -> label close to 1 (safe to embed).
  Low variance (smooth region)  -> label close to 0 (avoid).

Training data:
  - If real images exist in `image_dir`, uses those (recommended --
    real photos generalize far better than synthetic noise).
  - If the directory is empty/missing, falls back to generated
    multi-octave synthetic images so the script never hard-fails.

Both training-time patches and inference-time patches (see
core/intelligent_pixel_selector.py) are extracted from a LOW-BIT-MASKED
copy of the image, so the CNN always sees the same kind of input it
will see at inference time on a stego image.
"""

import os
import sys
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.nn import MSELoss

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_classifier import EdgePixelCNN
from utils.image_utils import mask_low_bits

VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp')


class PatchDataset(Dataset):
    def __init__(self, image_dir: str, patch_size: int = 16,
                 variance_threshold: float = 500.0, max_patches: int = 20000):
        self.patch_size = patch_size
        self.variance_threshold = variance_threshold
        self.samples = []

        real_images_found = (
            os.path.exists(image_dir)
            and any(f.lower().endswith(VALID_EXTENSIONS) for f in os.listdir(image_dir))
        )

        if real_images_found:
            print(f"Loading real images from '{image_dir}'...")
            self._load_from_directory(image_dir)
        else:
            print(f"No real images found in '{image_dir}'. "
                  f"Generating synthetic training data instead.\n"
                  f"(Tip: add real photos to this folder and rerun for better generalization.)")
            self._create_synthetic_dataset()

        if len(self.samples) > max_patches:
            idx = np.random.choice(len(self.samples), max_patches, replace=False)
            self.samples = [self.samples[i] for i in idx]

        print(f"Final dataset size: {len(self.samples)} patches")

    def _extract_patches(self, image: np.ndarray):
        masked = mask_low_bits(image, num_bits=2)
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape

        stride = max(self.patch_size // 2, 1)
        for y in range(0, h - self.patch_size, stride):
            for x in range(0, w - self.patch_size, stride):
                patch = masked[y:y + self.patch_size, x:x + self.patch_size]
                gray_patch = gray[y:y + self.patch_size, x:x + self.patch_size]

                variance = float(np.var(gray_patch))
                label = min(variance / self.variance_threshold, 1.0)

                self.samples.append((patch.copy(), label))

    def _load_from_directory(self, image_dir: str):
        files = [f for f in os.listdir(image_dir) if f.lower().endswith(VALID_EXTENSIONS)]
        for i, filename in enumerate(files, 1):
            filepath = os.path.join(image_dir, filename)
            image = cv2.imread(filepath)
            if image is None:
                print(f"  [{i}/{len(files)}] Skipping unreadable file: {filename}")
                continue
            print(f"  [{i}/{len(files)}] {filename} ({image.shape[1]}x{image.shape[0]})")
            self._extract_patches(image)

    def _create_synthetic_dataset(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            h, w = 256, 256
            base = np.zeros((h, w), dtype=np.float32)
            for octave, amp in [(4, 80), (16, 40), (64, 20)]:
                small = rng.integers(0, 255, (octave, octave)).astype(np.float32)
                upsampled = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
                base += upsampled * (amp / 255.0)
            base = np.clip(base, 0, 255).astype(np.uint8)
            image = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
            self._extract_patches(image)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patch, label = self.samples[idx]
        patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
        label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        return patch_tensor, label_tensor


def train_model(image_dir="data/cover_images", epochs=30, batch_size=32,
                 lr=1e-3, save_path="models/weights/edge_pixel_cnn.pth"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    dataset = PatchDataset(image_dir)
    if len(dataset) == 0:
        raise RuntimeError("No training patches could be generated.")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Batches per epoch: {len(loader)}")

    model = EdgePixelCNN().to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = MSELoss()

    best_loss = float("inf")
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for patches, labels in loader:
            patches, labels = patches.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(patches)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)

        if (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch + 1:3d}/{epochs} - MSE Loss: {avg_loss:.5f}")

    print(f"\nTraining complete. Best loss: {best_loss:.5f}")
    print(f"Weights saved to: {save_path}")


if __name__ == "__main__":
    img_dir = sys.argv[1] if len(sys.argv) > 1 else "data/cover_images"
    train_model(image_dir=img_dir)
