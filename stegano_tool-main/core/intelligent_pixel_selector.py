"""
intelligent_pixel_selector.py

Advanced pixel scoring combining:
  1. Fused edge/texture score (Canny + Sobel + Laplacian + variance) -- 40%
  2. CNN suitability score (if a trained model is provided)          -- 30%
  3. Local variance (again, emphasized)                              -- 20%
  4. Sobel gradient magnitude (again, emphasized)                    -- 10%

into ONE continuous per-pixel score map in [0, 1], used by
AdaptiveEmbedder / AdaptiveExtractor for variable-bit-depth embedding.

STABILITY: every input to every scoring stage (edge detector AND the
CNN) is derived from a low-bit-masked copy of the image (see
edge_detector.py and utils.image_utils.mask_low_bits). Since embedding
never touches anything above the lowest 2 bits, the masked image --
and therefore this entire score map -- is BYTE-IDENTICAL whether
computed from the cover image or the resulting stego image. This is
what allows the decoder to reconstruct the exact same score map (and
therefore the exact same pixel order) without ever seeing the original
cover image.
"""

import numpy as np
import cv2
from core.edge_detector import EdgeDetector
from utils.image_utils import mask_low_bits


class IntelligentPixelSelector:
    def __init__(self, image: np.ndarray, edge_detector: EdgeDetector = None,
                 cnn_model=None, cnn_stride: int = 8, patch_size: int = 16):
        """
        Args:
            image: the image to score (cover OR stego -- must produce
                   identical results for both, see module docstring).
            edge_detector: an EdgeDetector instance (created if not given).
            cnn_model: optional trained EdgePixelCNN (models.cnn_classifier).
                       If None, falls back to fused edge/texture score only.
            cnn_stride: grid spacing (pixels) at which the CNN is actually
                        evaluated; the result is upsampled to full
                        resolution. Keeps inference fast on large images.
            patch_size: patch size fed to the CNN.
        """
        self.image = image
        self.edge_detector = edge_detector or EdgeDetector()
        self.cnn_model = cnn_model
        self.cnn_stride = cnn_stride
        self.patch_size = patch_size
        self.h, self.w = image.shape[:2]
        self._score_map = None  # lazy-computed, cached

    def _compute_cnn_score_map(self) -> np.ndarray:
        """Evaluate the CNN on a strided grid over the MASKED image, upsample to full res."""
        masked = mask_low_bits(self.image, num_bits=2)
        half = self.patch_size // 2
        padded = cv2.copyMakeBorder(masked, half, half, half, half, cv2.BORDER_REFLECT)

        grid_rows = list(range(0, self.h, self.cnn_stride))
        grid_cols = list(range(0, self.w, self.cnn_stride))

        patches = np.empty((len(grid_rows) * len(grid_cols), self.patch_size, self.patch_size, 3),
                            dtype=np.uint8)
        idx = 0
        for r in grid_rows:
            for c in grid_cols:
                patches[idx] = padded[r:r + self.patch_size, c:c + self.patch_size]
                idx += 1

        scores_flat = self.cnn_model.score_patches_batch(patches)
        small_map = scores_flat.reshape(len(grid_rows), len(grid_cols)).astype(np.float32)

        full_map = cv2.resize(small_map, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        return np.clip(full_map, 0, 1)

    def compute_combined_score(self) -> np.ndarray:
        """Combine fused edge score, CNN score (if available), variance, gradient."""
        fused_edge = self.edge_detector.get_fused_score_map(self.image)

        if self.cnn_model is not None:
            cnn_score = self._compute_cnn_score_map()
        else:
            cnn_score = fused_edge  # graceful fallback, no double-counting bias

        gray = self.edge_detector._prepare_gray(self.image)
        variance_score = self.edge_detector._variance_normalized(gray)
        gradient_score = self.edge_detector._sobel_magnitude_normalized(gray)

        combined = (
            0.40 * fused_edge +
            0.30 * cnn_score +
            0.20 * variance_score +
            0.10 * gradient_score
        )
        return np.clip(combined, 0, 1).astype(np.float32)

    def get_score_map(self) -> np.ndarray:
        """Cached combined score map (computed once per instance)."""
        if self._score_map is None:
            self._score_map = self.compute_combined_score()
        return self._score_map

    def get_ordered_positions(self) -> np.ndarray:
        """
        Flat pixel indices (row-major) ordered by score descending, with
        position as a deterministic tie-break (np.argsort is stable, and
        flat index already encodes row-major position, so ties resolve
        to ascending position automatically).
        """
        score_map = self.get_score_map()
        flat_scores = score_map.flatten()
        # Stable sort descending by score; ties keep ascending flat-index order
        order = np.argsort(-flat_scores, kind="stable")
        return order

    def get_capacity_estimate(self, high_thresh=0.85, mid_thresh=0.55) -> dict:
        """Estimate total embeddable bits under the adaptive scheme, per channel."""
        score_map = self.get_score_map()
        high = int(np.sum(score_map > high_thresh))
        mid = int(np.sum((score_map > mid_thresh) & (score_map <= high_thresh)))
        total_bits = (high * 2 + mid * 1) * 3  # x3 channels
        return {
            "high_score_pixels": high,
            "mid_score_pixels": mid,
            "estimated_bits": total_bits,
            "estimated_characters": total_bits // 8,
        }

    def iter_bit_slots(self, high_thresh=0.85, mid_thresh=0.55):
        """
        Deterministic generator of (row, col, channel, depth) slots, in
        score-descending / position-tie-broken order. `depth` is how many
        low bits (1 or 2) may be used at that channel. Pixels scoring
        below mid_thresh are skipped entirely (smooth / unsafe regions).

        Both AdaptiveEmbedder and AdaptiveExtractor iterate this SAME
        generator (called on cover vs. stego image respectively) and are
        guaranteed to see the identical sequence of slots, because the
        underlying score map is mask-safe (see module docstring).
        """
        score_map = self.get_score_map()
        flat_scores = score_map.flatten()
        order = self.get_ordered_positions()

        for idx in order:
            row = idx // self.w
            col = idx % self.w
            score = flat_scores[idx]

            if score > high_thresh:
                depth = 2
            elif score > mid_thresh:
                depth = 1
            else:
                continue  # smooth region, skip entirely

            for ch in range(3):
                yield (int(row), int(col), ch, depth)
