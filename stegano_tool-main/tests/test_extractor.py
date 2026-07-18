"""
test_extractor.py

Round-trip tests: embed a message, extract it back, verify it matches.
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


def make_textured_image(h=300, w=300):
    """
    Synthetic image resembling a real photo: smooth gradient background with
    a few solid shapes (giving well-defined edges) plus mild sensor-like
    noise. Pure random noise is deliberately avoided as a test image --
    it makes nearly every pixel register as an 'edge', which is an
    unrealistic worst case no real cover photo resembles.
    """
    rng = np.random.default_rng(42)

    # smooth gradient background
    x = np.linspace(0, 255, w, dtype=np.float32)
    y = np.linspace(0, 255, h, dtype=np.float32)
    gx, gy = np.meshgrid(x, y)
    base = ((gx + gy) / 2).astype(np.uint8)
    image = np.stack([base, base, base], axis=-1)

    # a few solid rectangles/circles for real edges
    image[h // 4: h // 2, w // 4: w // 2] = [40, 180, 60]
    image[h // 2: 3 * h // 4, w // 2: 3 * w // 4] = [200, 60, 40]
    center = (w // 3, 2 * h // 3)
    radius = min(h, w) // 6
    yy, xx = np.ogrid[:h, :w]
    mask = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius ** 2
    image[mask] = [30, 30, 220]

    # mild noise for realism
    noise = rng.integers(-5, 6, (h, w, 3))
    image = np.clip(image.astype(int) + noise, 0, 255).astype(np.uint8)

    return image


def make_pipeline():
    edge_detector = EdgeDetector()
    pixel_selector = PixelSelector(edge_detector)
    embedder = Embedder(pixel_selector)
    extractor = Extractor(pixel_selector)
    return embedder, extractor


def test_round_trip_short_message():
    embedder, extractor = make_pipeline()
    cover = make_textured_image()
    message = "Hello, Stego!"

    stego = embedder.embed(cover, message)
    recovered = extractor.extract(stego)

    assert recovered == message


def test_round_trip_empty_message():
    embedder, extractor = make_pipeline()
    cover = make_textured_image()

    stego = embedder.embed(cover, "")
    recovered = extractor.extract(stego)

    assert recovered == ""


def test_round_trip_unicode_message():
    embedder, extractor = make_pipeline()
    cover = make_textured_image()
    message = "Café ☕ — secure msg 密"

    stego = embedder.embed(cover, message)
    recovered = extractor.extract(stego)

    assert recovered == message


def test_round_trip_longer_message_larger_image():
    embedder, extractor = make_pipeline()
    cover = make_textured_image(400, 400)
    message = "A" * 500  # 500 chars, needs decent capacity

    stego = embedder.embed(cover, message)
    recovered = extractor.extract(stego)

    assert recovered == message


def test_embed_raises_on_capacity_exceeded():
    embedder, _ = make_pipeline()
    cover = make_textured_image(20, 20)  # tiny image, low capacity
    message = "X" * 100000  # way too long

    with pytest.raises(ValueError):
        embedder.embed(cover, message)


def test_extract_raises_on_image_with_no_embedded_data():
    _, extractor = make_pipeline()
    # Fresh random image, never embedded into -- header bits are garbage,
    # should fail either on marker mismatch or produce nonsense that
    # raises during bits_to_text/marker check.
    cover = make_textured_image()
    with pytest.raises(ValueError):
        extractor.extract(cover)


def test_stego_image_only_differs_in_lsb():
    embedder, _ = make_pipeline()
    cover = make_textured_image()
    message = "test"

    stego = embedder.embed(cover, message)
    diff = cover.astype(int) - stego.astype(int)

    # Any changed pixel value must differ by at most 1 (LSB flip only)
    assert np.all(np.abs(diff) <= 1)
    # At least some pixels should have changed
    assert np.any(diff != 0)


if __name__ == "__main__":
    # Allow running this file directly for a quick manual check
    test_round_trip_short_message()
    test_round_trip_empty_message()
    test_round_trip_unicode_message()
    test_round_trip_longer_message_larger_image()
    test_embed_raises_on_capacity_exceeded()
    test_extract_raises_on_image_with_no_embedded_data()
    test_stego_image_only_differs_in_lsb()
    print("All manual round-trip checks passed!")