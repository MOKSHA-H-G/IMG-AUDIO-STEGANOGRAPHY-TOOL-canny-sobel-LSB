"""
test_audio.py

Tests for the audio steganography pipeline:
  - utils/audio_utils.py  (WAV load/save)
  - core/audio_embedder.py (AudioEmbedder / AudioExtractor)
"""

import numpy as np
import pytest

from utils.audio_utils import load_audio, save_audio, capacity_bits, AudioParams
from core.audio_embedder import AudioEmbedder, AudioExtractor


def _make_tone(n_samples=16000, dtype=np.int16, nchannels=1):
    t = np.linspace(0, 1, n_samples, endpoint=False)
    if dtype == np.int16:
        tone = (0.2 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    else:  # uint8
        tone = ((0.2 * np.sin(2 * np.pi * 440 * t) + 0.5) * 255).astype(np.uint8)
    if nchannels > 1:
        tone = np.column_stack([tone] * nchannels).flatten()
    return tone


def _write_wav(tmp_path, samples, nchannels, sampwidth, framerate=8000):
    path = str(tmp_path / "audio.wav")
    nframes = samples.size // nchannels
    params = AudioParams(nchannels=nchannels, sampwidth=sampwidth,
                          framerate=framerate, nframes=nframes)
    save_audio(samples, params, path)
    return path


class TestAudioUtils:
    def test_roundtrip_16bit_stereo(self, tmp_path):
        samples = _make_tone(16000, np.int16, nchannels=2)
        path = _write_wav(tmp_path, samples, nchannels=2, sampwidth=2)
        loaded, params = load_audio(path)
        assert params.nchannels == 2
        assert params.sampwidth == 2
        np.testing.assert_array_equal(loaded, samples)

    def test_roundtrip_8bit_mono(self, tmp_path):
        samples = _make_tone(8000, np.uint8, nchannels=1)
        path = _write_wav(tmp_path, samples, nchannels=1, sampwidth=1)
        loaded, params = load_audio(path)
        assert params.sampwidth == 1
        np.testing.assert_array_equal(loaded, samples)

    def test_save_rejects_non_wav_extension(self, tmp_path):
        samples = _make_tone(100)
        params = AudioParams(nchannels=1, sampwidth=2, framerate=8000, nframes=100)
        with pytest.raises(ValueError):
            save_audio(samples, params, str(tmp_path / "audio.mp3"))

    def test_load_missing_file_raises(self):
        with pytest.raises(Exception):
            load_audio("/nonexistent/path/to/audio.wav")

    def test_capacity_bits_equals_sample_count(self):
        samples = _make_tone(1234)
        assert capacity_bits(samples) == 1234


class TestAudioEmbedderExtractor:
    def test_embed_extract_roundtrip(self, tmp_path):
        samples = _make_tone(20000, np.int16, nchannels=2)
        path = _write_wav(tmp_path, samples, nchannels=2, sampwidth=2)
        cover, params = load_audio(path)

        message = "The quick brown fox jumps over the lazy dog. 12345!"
        emb = AudioEmbedder()
        stego = emb.embed(cover, message)

        out_path = str(tmp_path / "stego.wav")
        save_audio(stego, params, out_path)

        loaded_stego, _ = load_audio(out_path)
        ext = AudioExtractor()
        assert ext.extract(loaded_stego) == message

    def test_embed_only_modifies_lsb(self):
        samples = _make_tone(5000, np.int16)
        emb = AudioEmbedder()
        stego = emb.embed(samples, "short message")
        diff = np.abs(samples.astype(np.int32) - stego.astype(np.int32))
        assert diff.max() <= 1

    def test_embed_raises_on_insufficient_capacity(self):
        samples = _make_tone(20)  # far too small
        emb = AudioEmbedder()
        with pytest.raises(ValueError):
            emb.embed(samples, "this message will not fit in 20 samples at all")

    def test_extract_raises_on_cover_with_no_message(self):
        samples = _make_tone(5000, np.int16)
        ext = AudioExtractor()
        with pytest.raises(ValueError):
            ext.extract(samples)

    def test_empty_message_roundtrip(self):
        samples = _make_tone(5000, np.int16)
        emb = AudioEmbedder()
        ext = AudioExtractor()
        stego = emb.embed(samples, "")
        assert ext.extract(stego) == ""

    def test_embed_from_file_and_extract_from_file(self, tmp_path):
        samples = _make_tone(16000, np.int16, nchannels=1)
        cover_path = _write_wav(tmp_path, samples, nchannels=1, sampwidth=2)
        stego_path = str(tmp_path / "stego_out.wav")

        emb = AudioEmbedder()
        emb.embed_from_file(cover_path, "file-based roundtrip", stego_path)

        ext = AudioExtractor()
        assert ext.extract_from_file(stego_path) == "file-based roundtrip"
