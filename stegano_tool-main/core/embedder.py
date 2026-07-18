"""
embedder.py

Simple embedder: 1 bit per channel, into pixels chosen by PixelSelector
(binary edge mask, raster-scan order). This is the "basic edge-guided"
mode used for the clean baseline-vs-edge-guided comparison.

See adaptive_embedder.py for the more advanced multi-bit version.
"""

import numpy as np
from core.pixel_selector import PixelSelector
from utils.binary_utils import build_payload
from utils.image_utils import load_image, save_image


class Embedder:
    def __init__(self, pixel_selector: PixelSelector):
        self.pixel_selector = pixel_selector

    def embed(self, cover_image: np.ndarray, message: str) -> np.ndarray:
        """
        Embed `message` into `cover_image` and return the stego image.

        Raises:
            ValueError if image capacity is insufficient for the message.
        """
        payload_bits = build_payload(message)
        coords = self.pixel_selector.select_pixels(cover_image, len(payload_bits))

        stego_image = cover_image.copy()
        for bit, (r, c, ch) in zip(payload_bits, coords):
            stego_image[r, c, ch] = (stego_image[r, c, ch] & 0xFE) | int(bit)

        return stego_image

    def embed_from_file(self, cover_image_path: str, message: str, output_path: str) -> None:
        cover_image = load_image(cover_image_path)
        stego_image = self.embed(cover_image, message)
        save_image(stego_image, output_path)
