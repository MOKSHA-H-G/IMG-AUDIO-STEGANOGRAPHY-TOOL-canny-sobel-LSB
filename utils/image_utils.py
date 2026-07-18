"""
image_utils.py

Common image I/O and preprocessing helpers used across the pipeline.
"""

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """Load an image from disk as a BGR numpy array."""
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load image at: {path}")
    return image


def save_image(image: np.ndarray, path: str) -> None:
    """
    Save a BGR numpy array to disk. Must use a lossless format (PNG)
    to preserve LSB data -- JPEG will destroy embedded bits.
    """
    if not path.lower().endswith('.png'):
        raise ValueError("Stego images must be saved as PNG (lossless format) "
                          f"to preserve embedded data. Got: {path}")
    success = cv2.imwrite(path, image)
    if not success:
        raise IOError(f"Failed to write image to: {path}")


def preprocess_image(image: np.ndarray, target_size: tuple = None) -> np.ndarray:
    """Resize/normalize cover image before processing, if needed."""
    result = image.copy()
    if target_size is not None:
        result = cv2.resize(result, target_size, interpolation=cv2.INTER_AREA)
    return result


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to single-channel grayscale."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def extract_patch(image: np.ndarray, row: int, col: int, patch_size: int) -> np.ndarray:
    """
    Extract a patch_size x patch_size neighborhood centered on (row, col),
    with edge padding for pixels near image borders. Used to feed the CNN.
    """
    half = patch_size // 2
    padded = cv2.copyMakeBorder(image, half, half, half, half, cv2.BORDER_REFLECT)
    patch = padded[row:row + patch_size, col:col + patch_size]
    return patch


def mask_low_bits(image: np.ndarray, num_bits: int = 2) -> np.ndarray:
    """
    Zero out the lowest `num_bits` bits of every channel/pixel.

    This is the key trick that keeps encode and decode synchronized:
    embedding NEVER changes any bit except the lowest `num_bits` bits
    (s = (c & mask) | payload_bits). Therefore (s & mask) == (c & mask)
    ALWAYS, exactly, regardless of what was embedded. Running edge/score
    computation on this masked image guarantees byte-for-byte identical
    input on the encode side (cover image) and the decode side (stego
    image), so pixel selection stays perfectly in sync.

    Args:
        image: input image (any shape/dtype convertible to uint8)
        num_bits: number of low bits to clear (2 supports up to 2-bit
                  adaptive embedding; use 1 if only ever embedding 1 bit)
    """
    mask = (0xFF << num_bits) & 0xFF
    return (image.astype(np.uint8) & mask)
