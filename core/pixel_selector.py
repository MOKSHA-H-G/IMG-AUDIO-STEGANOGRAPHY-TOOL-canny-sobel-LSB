"""
pixel_selector.py

Simple, deterministic edge-guided pixel selector: 1 bit per channel,
selected from the binary edge mask in raster-scan (row-major) order.

This is the "basic" mode of the pipeline, paired with Embedder /
Extractor in core/embedder.py and core/extractor.py. It exists
alongside the more advanced IntelligentPixelSelector (adaptive,
multi-bit, CNN + multi-detector fused scoring) so the project can
show a clean baseline-vs-basic-edge-guided-vs-adaptive comparison.

DETERMINISM NOTE: pixels are ordered by position only (not by any
score), because the edge map itself is already guaranteed stable
across encode/decode (see edge_detector.py's masking). Position-only
ordering is the simplest possible deterministic tie-break.
"""

import numpy as np
from core.edge_detector import EdgeDetector


class PixelSelector:
    def __init__(self, edge_detector: EdgeDetector):
        self.edge_detector = edge_detector

    def _get_ordered_coords(self, image: np.ndarray) -> list:
        edge_map = self.edge_detector.get_merged_edge_map(image)
        candidates = np.argwhere(edge_map == 1)  # (N, 2) -> row, col
        if len(candidates) == 0:
            return []
        order = sorted(range(len(candidates)),
                        key=lambda i: (int(candidates[i][0]), int(candidates[i][1])))
        return [(int(candidates[i][0]), int(candidates[i][1])) for i in order]

    def select_pixels(self, image: np.ndarray, num_bits_needed: int) -> list:
        """
        Select an ordered list of (row, col, channel) tuples, one bit
        slot each, sufficient for num_bits_needed bits.
        """
        if num_bits_needed <= 0:
            return []

        ordered_coords = self._get_ordered_coords(image)

        full_coords = []
        for row, col in ordered_coords:
            for channel in range(3):
                full_coords.append((row, col, channel))
                if len(full_coords) >= num_bits_needed:
                    return full_coords

        raise ValueError(
            f"Insufficient capacity: image provides {len(full_coords)} bits, "
            f"but {num_bits_needed} bits are needed. Use a larger cover image "
            f"or a shorter message."
        )

    def get_capacity(self, image: np.ndarray) -> int:
        """Total available embedding capacity (bits) without truncation."""
        return len(self._get_ordered_coords(image)) * 3
