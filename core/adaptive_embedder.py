"""
adaptive_embedder.py

Adaptive embedding: pixels scoring highly (fused edge + CNN + variance
+ gradient) get up to 2 bits per channel; medium-scoring pixels get 1
bit per channel; low-scoring (smooth) pixels are skipped entirely.

Both AdaptiveEmbedder and AdaptiveExtractor consume the SAME
deterministic slot sequence from IntelligentPixelSelector.iter_bit_slots(),
computed on the cover image (embed side) or the stego image (extract
side) respectively. Because that score map is mask-safe (see
intelligent_pixel_selector.py), both sides see byte-identical slots
without ever needing to share side-channel information.
"""

import numpy as np
from core.intelligent_pixel_selector import IntelligentPixelSelector
from utils.binary_utils import build_payload, get_message_bit_length, bits_to_text, HEADER_BITS, END_MARKER
from utils.image_utils import load_image, save_image


class AdaptiveEmbedder:
    def __init__(self, pixel_selector: IntelligentPixelSelector):
        self.pixel_selector = pixel_selector

    def embed(self, cover_image: np.ndarray, message: str) -> np.ndarray:
        payload = build_payload(message)
        stego = cover_image.copy().astype(np.int32)

        bit_idx = 0
        payload_len = len(payload)

        for row, col, ch, depth in self.pixel_selector.iter_bit_slots():
            if bit_idx >= payload_len:
                break

            remaining = payload_len - bit_idx
            take = min(depth, remaining)
            bits_chunk = payload[bit_idx:bit_idx + take]

            if take == 2:
                value = int(bits_chunk, 2)
                stego[row, col, ch] = (stego[row, col, ch] & ~0b11) | value
            elif take == 1:
                value = int(bits_chunk, 2)
                stego[row, col, ch] = (stego[row, col, ch] & ~0b1) | value

            bit_idx += take

        if bit_idx < payload_len:
            raise ValueError(
                f"Insufficient capacity: needed {payload_len} bits, image "
                f"could only hold {bit_idx} bits under the adaptive scheme. "
                f"Use a larger/more textured cover image or a shorter message."
            )

        return np.clip(stego, 0, 255).astype(np.uint8)

    def embed_from_file(self, cover_path: str, message: str, output_path: str) -> None:
        cover = load_image(cover_path)
        stego = self.embed(cover, message)
        save_image(stego, output_path)


class AdaptiveExtractor:
    def __init__(self, pixel_selector: IntelligentPixelSelector):
        self.pixel_selector = pixel_selector

    def extract(self, stego_image: np.ndarray) -> str:
        # Phase 1: read exactly HEADER_BITS bits to learn the message length.
        bits = []
        for row, col, ch, depth in self.pixel_selector.iter_bit_slots():
            if len(bits) >= HEADER_BITS:
                break
            remaining = HEADER_BITS - len(bits)
            take = min(depth, remaining)
            value = int(stego_image[row, col, ch]) & ((1 << take) - 1)
            bits.append(format(value, f'0{take}b'))

        header_bits = ''.join(bits)[:HEADER_BITS]
        if len(header_bits) < HEADER_BITS:
            raise ValueError("Not enough data to extract a header -- image too small "
                              "or has no embedded data.")
        msg_bit_length = get_message_bit_length(header_bits)

        # Phase 2: read header + message + end marker, same deterministic order.
        total_bits_needed = HEADER_BITS + msg_bit_length + len(END_MARKER)
        bits = []
        collected = 0
        for row, col, ch, depth in self.pixel_selector.iter_bit_slots():
            if collected >= total_bits_needed:
                break
            remaining = total_bits_needed - collected
            take = min(depth, remaining)
            value = int(stego_image[row, col, ch]) & ((1 << take) - 1)
            bits.append(format(value, f'0{take}b'))
            collected += take

        all_bits = ''.join(bits)
        if len(all_bits) < total_bits_needed:
            raise ValueError("Not enough data to extract -- image may be corrupted "
                              "or does not contain a hidden message.")

        message_bits = all_bits[HEADER_BITS:HEADER_BITS + msg_bit_length]
        marker_bits = all_bits[HEADER_BITS + msg_bit_length:HEADER_BITS + msg_bit_length + len(END_MARKER)]

        if marker_bits != END_MARKER:
            raise ValueError(
                "End marker mismatch -- this image may not contain a hidden "
                "message, or the data has been corrupted/altered."
            )

        return bits_to_text(message_bits)

    def extract_from_file(self, stego_path: str) -> str:
        stego = load_image(stego_path)
        return self.extract(stego)
