"""
edge_detector.py

Edge / texture detection used to guide pixel selection for embedding.

Provides TWO outputs:
  1. get_merged_edge_map()  -> binary {0,1} mask (Canny OR Sobel).
     Used by the simple/baseline "edge-guided" pipeline (PixelSelector).
  2. get_fused_score_map()  -> continuous [0,1] score combining Canny,
     Sobel, Laplacian, and local variance. Used by the more advanced
     IntelligentPixelSelector for adaptive multi-bit embedding.

CRITICAL STABILITY RULE (read before modifying):
Both outputs are computed from a copy of the image with its low bits
masked off (see utils.image_utils.mask_low_bits). Embedding only ever
changes those low bits. Masking them before any edge/score computation
guarantees the encoder (looking at the cover image) and the decoder
(looking at the stego image) compute BYTE-IDENTICAL scores, so pixel
selection never desynchronizes -- this was verified empirically in
earlier iterations of this project: without masking, even a handful of
+/-1 LSB flips can shift gradient magnitudes enough to change which
pixels are classified as edges, silently corrupting extraction.
"""

import cv2
import numpy as np
from utils.image_utils import mask_low_bits


class EdgeDetector:
    def __init__(self, canny_low=50, canny_high=150, sobel_ksize=3,
                 sobel_threshold=40.0, mask_bits=2):
        """
        Args:
            canny_low, canny_high: Canny hysteresis thresholds (fixed,
                not adaptive -- adaptive/Otsu-style thresholds recompute
                from the image's own histogram, which shifts slightly
                once LSBs are embedded, causing the same desync problem
                that masking solves for gradient-based methods).
            sobel_ksize: Sobel kernel size (odd).
            sobel_threshold: fixed absolute gradient-magnitude threshold
                for the binary edge map.
            mask_bits: number of low bits to clear before any computation.
                Must be >= the number of bits ever embedded per channel
                (2 supports the adaptive 1-2 bit embedder).
        """
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.sobel_ksize = sobel_ksize
        self.sobel_threshold = sobel_threshold
        self.mask_bits = mask_bits

    def _prepare_gray(self, image: np.ndarray) -> np.ndarray:
        """Mask low bits, then convert to grayscale."""
        masked = mask_low_bits(image, self.mask_bits)
        if len(masked.shape) == 3:
            gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        else:
            gray = masked
        return gray

    # ---- Individual detectors (all operate on already-masked grayscale) ----

    def get_canny_edges(self, gray_image: np.ndarray) -> np.ndarray:
        """Canny edge detection -> binary map {0,255}."""
        return cv2.Canny(gray_image, self.canny_low, self.canny_high)

    def get_sobel_edges(self, gray_image: np.ndarray) -> np.ndarray:
        """Sobel gradient magnitude, fixed-thresholded -> binary map {0,255}."""
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=self.sobel_ksize)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=self.sobel_ksize)
        magnitude = np.hypot(sobel_x, sobel_y)
        return np.where(magnitude >= self.sobel_threshold, 255, 0).astype(np.uint8)

    def _sobel_magnitude_normalized(self, gray_image: np.ndarray) -> np.ndarray:
        """Sobel gradient magnitude normalized to [0,1] (continuous, not thresholded)."""
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=self.sobel_ksize)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=self.sobel_ksize)
        magnitude = np.hypot(sobel_x, sobel_y)
        max_mag = magnitude.max()
        return (magnitude / max_mag).astype(np.float32) if max_mag > 0 else np.zeros_like(magnitude, dtype=np.float32)

    def _laplacian_normalized(self, gray_image: np.ndarray) -> np.ndarray:
        """Laplacian response, normalized to [0,1]."""
        laplacian = np.abs(cv2.Laplacian(gray_image, cv2.CV_64F))
        max_lap = laplacian.max()
        return (laplacian / max_lap).astype(np.float32) if max_lap > 0 else np.zeros_like(laplacian, dtype=np.float32)

    def _variance_normalized(self, gray_image: np.ndarray, window_size=5) -> np.ndarray:
        """Local variance (box-filter based), normalized to [0,1]."""
        g = gray_image.astype(np.float32)
        mean = cv2.blur(g, (window_size, window_size))
        mean_sq = cv2.blur(g * g, (window_size, window_size))
        variance = np.maximum(mean_sq - mean * mean, 0)
        max_var = variance.max()
        return (variance / max_var).astype(np.float32) if max_var > 0 else np.zeros_like(variance, dtype=np.float32)

    # ---- Combined outputs ----

    def get_merged_edge_map(self, image: np.ndarray) -> np.ndarray:
        """
        Simple binary edge mask: Canny OR Sobel (fixed thresholds).
        Returns {0,1} array, shape (H, W).
        """
        if image is None:
            raise ValueError("Input image is None.")
        gray = self._prepare_gray(image)
        canny = self.get_canny_edges(gray)
        sobel = self.get_sobel_edges(gray)
        merged = cv2.bitwise_or(canny, sobel)
        return (merged > 0).astype(np.uint8)

    def get_fused_score_map(self, image: np.ndarray,
                             weights=(0.35, 0.25, 0.20, 0.20)) -> np.ndarray:
        """
        Continuous [0,1] embedding-suitability score combining:
          Canny (binary, weight[0]), Sobel magnitude (weight[1]),
          Laplacian (weight[2]), local variance (weight[3]).

        Returns:
            float32 array, shape (H, W), values in [0, 1].
        """
        if image is None:
            raise ValueError("Input image is None.")
        gray = self._prepare_gray(image)

        canny_score = (self.get_canny_edges(gray) / 255.0).astype(np.float32)
        sobel_score = self._sobel_magnitude_normalized(gray)
        laplacian_score = self._laplacian_normalized(gray)
        variance_score = self._variance_normalized(gray)

        fused = (
            weights[0] * canny_score +
            weights[1] * sobel_score +
            weights[2] * laplacian_score +
            weights[3] * variance_score
        )
        max_fused = fused.max()
        if max_fused > 0:
            fused = fused / max_fused
        return np.clip(fused, 0, 1).astype(np.float32)

    def visualize(self, image: np.ndarray, edge_map: np.ndarray) -> np.ndarray:
        """Overlay a binary edge map on the original image (green highlight)."""
        if len(image.shape) == 2:
            overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            overlay = image.copy()
        overlay[edge_map == 1] = [0, 255, 0]
        return overlay

    def visualize_heatmap(self, score_map: np.ndarray) -> np.ndarray:
        """Render a continuous [0,1] score map as a JET colormap BGR image."""
        as_uint8 = (np.clip(score_map, 0, 1) * 255).astype(np.uint8)
        return cv2.applyColorMap(as_uint8, cv2.COLORMAP_JET)
