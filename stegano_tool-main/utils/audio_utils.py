"""
audio_utils.py

WAV audio I/O helpers for the audio-steganography pipeline. Uses only the
Python standard library (`wave`) plus numpy, so no extra dependency is
required beyond what the project already ships with.

Only uncompressed PCM WAV (8-bit unsigned or 16-bit signed, mono or
multi-channel) is supported. LSB steganography requires a lossless,
uncompressed sample stream -- MP3/AAC/etc. would destroy embedded bits
the same way JPEG does for images.
"""

import wave
import numpy as np


# dtype used internally per sample width (bytes)
_DTYPE_BY_SAMPWIDTH = {
    1: np.uint8,   # 8-bit PCM is unsigned in the WAV spec
    2: np.int16,   # 16-bit PCM is signed
}


class AudioParams:
    """Lightweight container mirroring wave.Wave_read.getparams()."""

    def __init__(self, nchannels, sampwidth, framerate, nframes,
                 comptype="NONE", compname="not compressed"):
        self.nchannels = nchannels
        self.sampwidth = sampwidth
        self.framerate = framerate
        self.nframes = nframes
        self.comptype = comptype
        self.compname = compname

    def duration_seconds(self) -> float:
        return self.nframes / float(self.framerate) if self.framerate else 0.0


def load_audio(path: str):
    """
    Load a PCM WAV file.

    Returns:
        (samples, params) where `samples` is a 1-D numpy array (all
        channels interleaved, in on-disk order) and `params` is an
        AudioParams describing the stream needed to write it back out.

    Raises:
        ValueError if the file is not a supported PCM format
        (e.g. compressed WAV, or a sample width other than 8/16-bit).
    """
    try:
        with wave.open(path, "rb") as wf:
            p = wf.getparams()
            raw = wf.readframes(p.nframes)
    except wave.Error as e:
        raise ValueError(f"Could not read WAV file '{path}': {e}")

    if p.comptype != "NONE":
        raise ValueError(
            f"'{path}' is a compressed WAV ({p.comptype}); only "
            "uncompressed PCM WAV is supported for LSB embedding."
        )
    if p.sampwidth not in _DTYPE_BY_SAMPWIDTH:
        raise ValueError(
            f"Unsupported sample width ({p.sampwidth * 8}-bit) in "
            f"'{path}'. Only 8-bit or 16-bit PCM WAV is supported."
        )

    dtype = _DTYPE_BY_SAMPWIDTH[p.sampwidth]
    samples = np.frombuffer(raw, dtype=dtype).copy()

    params = AudioParams(p.nchannels, p.sampwidth, p.framerate,
                          p.nframes, p.comptype, p.compname)
    return samples, params


def save_audio(samples: np.ndarray, params: AudioParams, path: str) -> None:
    """
    Write a sample array back out as a PCM WAV file using `params`.

    Must stay in an uncompressed format (WAV/PCM) to preserve the
    embedded LSBs -- never re-encode to a lossy codec.
    """
    if not path.lower().endswith(".wav"):
        raise ValueError(
            "Stego audio must be saved as WAV (uncompressed PCM) to "
            f"preserve embedded data. Got: {path}"
        )
    dtype = _DTYPE_BY_SAMPWIDTH.get(params.sampwidth)
    if dtype is None:
        raise ValueError(f"Unsupported sample width: {params.sampwidth * 8}-bit")

    out = samples.astype(dtype)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(params.nchannels)
        wf.setsampwidth(params.sampwidth)
        wf.setframerate(params.framerate)
        wf.writeframes(out.tobytes())


def capacity_bits(samples: np.ndarray) -> int:
    """Max payload bits embeddable at 1 bit per sample (1 LSB each)."""
    return int(samples.size)
