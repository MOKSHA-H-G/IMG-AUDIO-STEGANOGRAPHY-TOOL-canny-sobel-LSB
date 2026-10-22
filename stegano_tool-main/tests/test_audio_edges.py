import wave
import numpy as np

from utils.audio_utils import load_audio
from utils.audio_edge_analysis import analyze_audio_edges


# Step 1: Generate a sample WAV file
sample_rate = 44100
duration = 3
time = np.arange(sample_rate * duration) / sample_rate

# Create a tone whose frequency changes over time
audio = (
    10000 * np.sin(2 * np.pi * (300 + 500 * time) * time)
).astype(np.int16)

wav_path = "test_audio.wav"

with wave.open(wav_path, "wb") as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(sample_rate)
    wav.writeframes(audio.tobytes())

print(f"Created test audio: {wav_path}")

# Step 2: Load the WAV using your existing project utility
samples, params = load_audio(wav_path)

# Step 3: Analyze the audio without modifying its samples
results = analyze_audio_edges(
    samples,
    sample_rate=params.framerate,
    channels=params.nchannels,
)

print("\nAudio analysis completed successfully!")
for name, path in results["files"].items():
    print(f"{name}: {path}")