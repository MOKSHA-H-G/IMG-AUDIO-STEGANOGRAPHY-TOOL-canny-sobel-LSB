"""
extractor.py

Simple extractor, paired with Embedder / PixelSelector.
"""

import numpy as np
from core.pixel_selector import PixelSelector
from utils.binary_utils import get_message_bit_length, bits_to_text, HEADER_BITS, END_MARKER
from utils.image_utils import load_image


class Extractor:
    def __init__(self, pixel_selector: PixelSelector):
        self.pixel_selector = pixel_selector

    def _read_bits(self, stego_image: np.ndarray, coords: list) -> str:
        return ''.join(str(stego_image[r, c, ch] & 1) for r, c, ch in coords)

    def extract(self, stego_image: np.ndarray) -> str:
        """
        Extract and return the hidden message from a stego image.

        Raises:
            ValueError if no valid payload/header is found, or the end
            marker doesn't match (corruption or wrong image).
        """
        header_coords = self.pixel_selector.select_pixels(stego_image, HEADER_BITS)
        header_bits = self._read_bits(stego_image, header_coords)
        msg_bit_length = get_message_bit_length(header_bits)

        total_bits_needed = HEADER_BITS + msg_bit_length + len(END_MARKER)
        all_coords = self.pixel_selector.select_pixels(stego_image, total_bits_needed)
        all_bits = self._read_bits(stego_image, all_coords)

        message_bits = all_bits[HEADER_BITS:HEADER_BITS + msg_bit_length]
        marker_bits = all_bits[HEADER_BITS + msg_bit_length:]

        if marker_bits != END_MARKER:
            raise ValueError(
                "End marker mismatch -- this image may not contain a hidden "
                "message, or the data has been corrupted/altered."
            )

        return bits_to_text(message_bits)

    def extract_from_file(self, stego_image_path: str) -> str:
        stego_image = load_image(stego_image_path)
        return self.extract(stego_image)
