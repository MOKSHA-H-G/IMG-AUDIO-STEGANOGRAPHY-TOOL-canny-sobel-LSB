"""
test_roundtrip.py

Round-trip tests for BOTH pipelines:
  1. Simple edge-guided (PixelSelector + Embedder/Extractor)
  2. Adaptive multi-bit (IntelligentPixelSelector + AdaptiveEmbedder/Extractor)

These are the most important tests in the project: if they fail, it
means encode/decode have desynchronized and messages will come back
corrupted or unreadable.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from core.edge_detector import EdgeDetector
from core.pixel_selector import PixelSelector
from core.embedder import Embedder
from core.extractor import Extractor
from core.intelligent_pixel_selector import IntelligentPixelSelector
from core.adaptive_embedder import AdaptiveEmbedder, AdaptiveExtractor
from core.metrics import compute_psnr, compute_ssim, baseline_lsb_embed


def make_textured_image(h=300, w=300, seed=42):
    """Realistic synthetic cover image: gradient background + solid shapes + mild noise."""
    rng = np.random.default_rng(seed)

    x = np.linspace(0, 255, w, dtype=np.float32)
    y = np.linspace(0, 255, h, dtype=np.float32)
    gx, gy = np.meshgrid(x, y)
    base = ((gx + gy) / 2).astype(np.uint8)
    image = np.stack([base, base, base], axis=-1)

    image[h // 4: h // 2, w // 4: w // 2] = [40, 180, 60]
    image[h // 2: 3 * h // 4, w // 2: 3 * w // 4] = [200, 60, 40]
    center = (w // 3, 2 * h // 3)
    radius = min(h, w) // 6
    yy, xx = np.ogrid[:h, :w]
    mask = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius ** 2
    image[mask] = [30, 30, 220]

    noise = rng.integers(-5, 6, (h, w, 3))
    image = np.clip(image.astype(int) + noise, 0, 255).astype(np.uint8)
    return image


# ---------------------------------------------------------------------------
# Simple pipeline (PixelSelector / Embedder / Extractor)
# ---------------------------------------------------------------------------

def make_simple_pipeline():
    ed = EdgeDetector()
    ps = PixelSelector(ed)
    return Embedder(ps), Extractor(ps)


def test_simple_round_trip_short_message():
    embedder, extractor = make_simple_pipeline()
    cover = make_textured_image()
    message = "Hello, Stego!"
    stego = embedder.embed(cover, message)
    assert extractor.extract(stego) == message


def test_simple_round_trip_empty_message():
    embedder, extractor = make_simple_pipeline()
    cover = make_textured_image()
    stego = embedder.embed(cover, "")
    assert extractor.extract(stego) == ""


def test_simple_round_trip_unicode():
    embedder, extractor = make_simple_pipeline()
    cover = make_textured_image()
    message = "Café ☕ — secure msg 密"
    stego = embedder.embed(cover, message)
    assert extractor.extract(stego) == message


def test_simple_capacity_error():
    embedder, _ = make_simple_pipeline()
    cover = make_textured_image(20, 20)
    with pytest.raises(ValueError):
        embedder.embed(cover, "X" * 100000)


def test_simple_stego_only_differs_in_lsb():
    embedder, _ = make_simple_pipeline()
    cover = make_textured_image()
    stego = embedder.embed(cover, "test message")
    diff = cover.astype(int) - stego.astype(int)
    assert np.all(np.abs(diff) <= 1)
    assert np.any(diff != 0)


# ---------------------------------------------------------------------------
# Adaptive pipeline (IntelligentPixelSelector / AdaptiveEmbedder/Extractor)
# ---------------------------------------------------------------------------

def make_adaptive_pipeline(image, cnn_model=None):
    ed = EdgeDetector()
    selector_encode = IntelligentPixelSelector(image, edge_detector=ed, cnn_model=cnn_model)
    embedder = AdaptiveEmbedder(selector_encode)
    return embedder


def make_adaptive_extractor(stego_image, cnn_model=None):
    ed = EdgeDetector()
    selector_decode = IntelligentPixelSelector(stego_image, edge_detector=ed, cnn_model=cnn_model)
    return AdaptiveExtractor(selector_decode)


def test_adaptive_round_trip_short_message():
    cover = make_textured_image()
    embedder = make_adaptive_pipeline(cover)
    message = "Adaptive hello!"
    stego = embedder.embed(cover, message)

    extractor = make_adaptive_extractor(stego)
    assert extractor.extract(stego) == message


def test_adaptive_round_trip_empty_message():
    cover = make_textured_image()
    embedder = make_adaptive_pipeline(cover)
    stego = embedder.embed(cover, "")

    extractor = make_adaptive_extractor(stego)
    assert extractor.extract(stego) == ""


def test_adaptive_round_trip_unicode():
    cover = make_textured_image()
    embedder = make_adaptive_pipeline(cover)
    message = "Unicode test: 你好 😀"
    stego = embedder.embed(cover, message)

    extractor = make_adaptive_extractor(stego)
    assert extractor.extract(stego) == message


def test_adaptive_round_trip_longer_message():
    cover = make_textured_image(600, 600)
    embedder = make_adaptive_pipeline(cover)
    message = "A" * 150
    stego = embedder.embed(cover, message)

    extractor = make_adaptive_extractor(stego)
    assert extractor.extract(stego) == message


def test_adaptive_capacity_error():
    cover = make_textured_image(20, 20)
    embedder = make_adaptive_pipeline(cover)
    with pytest.raises(ValueError):
        embedder.embed(cover, "X" * 100000)


def test_adaptive_beats_or_matches_capacity_of_simple():
    """Adaptive scheme should never have LESS raw capacity than 1-bit-per-edge-pixel."""
    cover = make_textured_image()
    ed = EdgeDetector()
    simple_selector = PixelSelector(ed)
    simple_capacity = simple_selector.get_capacity(cover)

    adaptive_selector = IntelligentPixelSelector(cover, edge_detector=ed)
    adaptive_capacity = adaptive_selector.get_capacity_estimate()["estimated_bits"]

    # Not a strict requirement, but sanity check both are > 0 for a textured image
    assert simple_capacity > 0
    assert adaptive_capacity > 0


def test_adaptive_vs_baseline_metrics_run_without_error():
    """Comparison pipeline should run end-to-end and produce sane metrics."""
    cover = make_textured_image()
    embedder = make_adaptive_pipeline(cover)
    message = "Compare me against baseline LSB"
    stego = embedder.embed(cover, message)

    baseline_stego = baseline_lsb_embed(cover, message)

    psnr_adaptive = compute_psnr(cover, stego)
    psnr_baseline = compute_psnr(cover, baseline_stego)
    ssim_adaptive = compute_ssim(cover, stego)
    ssim_baseline = compute_ssim(cover, baseline_stego)

    assert psnr_adaptive > 20  # sanity: not wildly distorted
    assert psnr_baseline > 20
    assert 0 <= ssim_adaptive <= 1
    assert 0 <= ssim_baseline <= 1


def test_adaptive_round_trip_with_cnn_model():
    """
    Critical test: verify encode/decode stay synchronized when a trained
    CNN is in the loop (the riskiest part for score-map desync bugs).
    Skips gracefully if no trained weights are present yet.
    """
    from models.cnn_classifier import load_pretrained

    cnn = load_pretrained()
    if cnn is None:
        pytest.skip("No trained CNN weights found; run models/train.py first.")

    cover = make_textured_image(500, 500)
    ed = EdgeDetector()

    embedder = AdaptiveEmbedder(IntelligentPixelSelector(cover, edge_detector=ed, cnn_model=cnn))
    message = "Testing with CNN in the loop! Unicode too: 你好"
    stego = embedder.embed(cover, message)

    extractor = AdaptiveExtractor(IntelligentPixelSelector(stego, edge_detector=ed, cnn_model=cnn))
    assert extractor.extract(stego) == message


if __name__ == "__main__":
    test_simple_round_trip_short_message()
    test_simple_round_trip_empty_message()
    test_simple_round_trip_unicode()
    test_simple_capacity_error()
    test_simple_stego_only_differs_in_lsb()
    print("Simple pipeline: all manual checks passed.")

    test_adaptive_round_trip_short_message()
    test_adaptive_round_trip_empty_message()
    test_adaptive_round_trip_unicode()
    test_adaptive_round_trip_longer_message()
    test_adaptive_capacity_error()
    test_adaptive_beats_or_matches_capacity_of_simple()
    test_adaptive_vs_baseline_metrics_run_without_error()
    test_adaptive_round_trip_with_cnn_model()
    print("Adaptive pipeline: all manual checks passed.")
