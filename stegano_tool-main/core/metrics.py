"""
metrics.py

Comprehensive evaluation metrics for the steganography pipeline:
  - PSNR, SSIM, MSE (imperceptibility / distortion)
  - Entropy (information content)
  - Histogram Chi-square (statistical similarity to cover)
  - RS Analysis (a classic LSB-steganalysis detection test)
  - Combined "detectability" score
  - Baseline sequential-LSB embedding, for apples-to-apples comparison
"""

import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from utils.binary_utils import text_to_bits


def compute_psnr(cover: np.ndarray, stego: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio (higher = less distortion)."""
    return float(peak_signal_noise_ratio(cover, stego, data_range=255))


def compute_ssim(cover: np.ndarray, stego: np.ndarray) -> float:
    """Structural Similarity Index (closer to 1.0 = more similar)."""
    if len(cover.shape) == 3:
        return float(structural_similarity(cover, stego, channel_axis=2, data_range=255))
    return float(structural_similarity(cover, stego, data_range=255))


def compute_mse(cover: np.ndarray, stego: np.ndarray) -> float:
    """Mean Squared Error (lower = better)."""
    return float(np.mean((cover.astype(np.float32) - stego.astype(np.float32)) ** 2))


def compute_capacity_bpp(image_shape: tuple, bits_embedded: int) -> float:
    """Bits-per-pixel capacity metric."""
    h, w = image_shape[:2]
    total_pixels = h * w
    return bits_embedded / total_pixels if total_pixels > 0 else 0.0


def compute_entropy(image: np.ndarray) -> float:
    """Shannon entropy of the image (0-8 bits for an 8-bit image)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-10)
    return float(-np.sum(hist * np.log2(hist + 1e-10)))


def histogram_chi_square(cover: np.ndarray, stego: np.ndarray) -> float:
    """Chi-square distance between grayscale histograms (lower = more similar)."""
    cover_gray = cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY) if len(cover.shape) == 3 else cover
    stego_gray = cv2.cvtColor(stego, cv2.COLOR_BGR2GRAY) if len(stego.shape) == 3 else stego

    hist_c = cv2.calcHist([cover_gray], [0], None, [256], [0, 256]).flatten()
    hist_s = cv2.calcHist([stego_gray], [0], None, [256], [0, 256]).flatten()

    chi2 = 0.5 * np.sum(((hist_c - hist_s) ** 2) / (hist_c + hist_s + 1e-10))
    return float(chi2)


def rs_analysis(image: np.ndarray) -> float:
    """
    Simplified RS steganalysis: estimates how much LSB embedding has
    disturbed pixel-pair regularity. Lower = more resistant to
    detection by this class of attack.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gray = gray.astype(np.int32)

    def count_regular_groups(img):
        count = 0
        h, w = img.shape
        for i in range(0, h - 1, 2):
            for j in range(0, w - 1, 2):
                block = img[i:i + 2, j:j + 2]
                grad = abs(int(block[0, 0]) - int(block[0, 1])) + abs(int(block[1, 0]) - int(block[1, 1]))
                if grad < 2:
                    count += 1
        return count

    r_minus = count_regular_groups(gray)
    flipped = (gray & ~1) | (~gray & 1)  # flip LSB of every pixel
    r_plus = count_regular_groups(flipped)

    denom = r_minus + r_plus
    if denom == 0:
        return 0.0
    return float(abs(r_minus - r_plus) / denom)


def detectability_score(cover: np.ndarray, stego: np.ndarray) -> float:
    """
    Combined detectability estimate in [0, 1]. Lower = harder to detect
    statistically (better for steganography).
    """
    chi2 = histogram_chi_square(cover, stego)
    rs = rs_analysis(stego)
    chi2_norm = min(chi2 / 100.0, 1.0)
    return float(np.clip((chi2_norm + rs) / 2, 0, 1))


def baseline_lsb_embed(cover_image: np.ndarray, message: str) -> np.ndarray:
    """
    Plain sequential LSB embedding with NO edge/CNN guidance -- the
    classic baseline every steganography paper compares against.
    """
    from utils.binary_utils import build_payload
    payload_bits = build_payload(message)

    if len(payload_bits) > cover_image.size:
        raise ValueError(f"Message too long for baseline: {len(payload_bits)} bits > "
                          f"{cover_image.size} available pixels*channels.")

    stego = cover_image.copy().flatten()
    for i, bit in enumerate(payload_bits):
        stego[i] = (stego[i] & 0xFE) | int(bit)

    return stego.reshape(cover_image.shape).astype(np.uint8)


def full_metrics_report(cover: np.ndarray, stego: np.ndarray) -> dict:
    """Compute every metric at once for a single cover/stego pair."""
    return {
        "psnr": compute_psnr(cover, stego),
        "ssim": compute_ssim(cover, stego),
        "mse": compute_mse(cover, stego),
        "entropy": compute_entropy(stego),
        "chi2": histogram_chi_square(cover, stego),
        "rs": rs_analysis(stego),
        "detectability": detectability_score(cover, stego),
    }


def compare_embedding_methods(cover: np.ndarray, baseline_stego: np.ndarray,
                               proposed_stego: np.ndarray) -> dict:
    """Side-by-side metrics for baseline LSB vs. the proposed method."""
    baseline = full_metrics_report(cover, baseline_stego)
    proposed = full_metrics_report(cover, proposed_stego)

    improvement = {
        "psnr": proposed["psnr"] - baseline["psnr"],
        "ssim": proposed["ssim"] - baseline["ssim"],
        "chi2_reduction": baseline["chi2"] - proposed["chi2"],
        "rs_reduction": baseline["rs"] - proposed["rs"],
        "detectability_reduction": baseline["detectability"] - proposed["detectability"],
    }

    return {"baseline": baseline, "proposed": proposed, "improvement": improvement}
