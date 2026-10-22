
from pathlib import Path

import numpy as np
import torch

from audio_cnn_classifier import AudioSuitabilityCNN

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "audio_cnn" / "regions.npz"
WEIGHTS_PATH = ROOT / "models" / "weights" / "audio_suitability_cnn.pth"

data = np.load(DATA_PATH)
regions = data["regions"]

model = AudioSuitabilityCNN()
model.load_state_dict(
    torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
)
model.eval()

# Convert the 128 regions to normalized CNN input.
x = torch.tensor(regions, dtype=torch.float32) / 255.0
x = x.unsqueeze(1)

with torch.no_grad():
    scores = model(x).squeeze(1).numpy()

order = np.argsort(scores)[::-1]

print(f"Regions tested: {len(scores)}")
print("\nTop 5 regions by predicted suitability score:")

for index in order[:5]:
    print(f"Region {index}: score = {scores[index]:.4f}")
