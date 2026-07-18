"""
audio_embedder.py

LSB audio steganography: 1 bit per PCM sample, embedded sequentially
across the entire sample stream (all channels interleaved, in on-disk
order). Uses the same [32-bit length header][message bits][end marker]
framing as the image embedder (utils/binary_utils.py), so encode/decode
stay perfectly synchronized and messages self-describe their length.

Only the LSB of each sample is touched, which keeps the change
inaudible for normal 16-bit PCM audio (worst case a change of 1 part
in 65536, well below the noise floor of most recordings).
"""

import numpy as np
from utils.binary_utils import (
    build_payload, get_message_bit_length, bits_to_text,
    HEADER_BITS, END_MARKER,
)
from utils.audio_utils import load_audio, save_audio, capacity_bits


class AudioEmbedder:
    def embed(self, samples: np.ndarray, message: str) -> np.ndarray:
        """
        Embed `message` into `samples` (1 bit/sample, sequential) and
        return the stego sample array. `samples` is left untouched;
        a modified copy is returned.

        Raises:
            ValueError if capacity is insufficient for the message.
        """
        payload_bits = build_payload(message)
        if len(payload_bits) > capacity_bits(samples):
            raise ValueError(
                f"Message too large: needs {len(payload_bits)} bits but "
                f"this audio only has capacity for {capacity_bits(samples)} bits "
                f"(≈{capacity_bits(samples) // 8} bytes)."
            )

        stego = samples.copy()
        dtype = stego.dtype
        for i, bit in enumerate(payload_bits):
            # Clear the LSB, then set it -- works for both uint8 and int16
            # since bitwise ops on numpy ints operate on the two's-complement
            # bit pattern regardless of signedness.
            stego[i] = np.array((int(stego[i]) & ~1) | int(bit), dtype=dtype)

        return stego

    def embed_from_file(self, cover_audio_path: str, message: str,
                         output_path: str) -> None:
        samples, params = load_audio(cover_audio_path)
        stego = self.embed(samples, message)
        save_audio(stego, params, output_path)


class AudioExtractor:
    def _read_bits(self, samples: np.ndarray, count: int) -> str:
        return ''.join(str(int(samples[i]) & 1) for i in range(count))

    def extract(self, samples: np.ndarray) -> str:
        """
        Extract and return the hidden message from stego audio samples.

        Raises:
            ValueError if no valid payload/header is found, or the end
            marker doesn't match (corruption or wrong file).
        """
        if samples.size < HEADER_BITS:
            raise ValueError("Audio too short to contain a valid header.")

        header_bits = self._read_bits(samples, HEADER_BITS)
        msg_bit_length = get_message_bit_length(header_bits)

        total_bits_needed = HEADER_BITS + msg_bit_length + len(END_MARKER)
        if total_bits_needed > samples.size:
            raise ValueError(
                "Declared message length exceeds available samples -- "
                "this file may not contain a hidden message, or it has "
                "been corrupted/re-encoded."
            )

        all_bits = self._read_bits(samples, total_bits_needed)
        message_bits = all_bits[HEADER_BITS:HEADER_BITS + msg_bit_length]
        marker_bits = all_bits[HEADER_BITS + msg_bit_length:]

        if marker_bits != END_MARKER:
            raise ValueError(
                "End marker mismatch -- this audio file may not contain a "
                "hidden message, or the data has been corrupted/altered."
            )

        return bits_to_text(message_bits)

    def extract_from_file(self, stego_audio_path: str) -> str:
        samples, _params = load_audio(stego_audio_path)
        return self.extract(samples)
