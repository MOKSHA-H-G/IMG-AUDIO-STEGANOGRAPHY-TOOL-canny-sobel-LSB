"""
binary_utils.py

Handles conversion between text payloads and bitstreams, plus the
length-header and end-marker framing needed so the extractor knows
exactly how many bits to read.

Payload framing format:
    [32-bit length header (number of message bits)] + [message bits] + [END_MARKER]

END_MARKER acts as a redundant sanity check in case the header is corrupted.
"""

END_MARKER = "1111111111111110"  # 16-bit sentinel, unlikely to occur naturally
HEADER_BITS = 32  # bits reserved for encoding message length


def text_to_bits(message: str) -> str:
    """Convert a UTF-8 string to a string of '0'/'1' characters."""
    if message == "":
        return ""
    return ''.join(format(byte, '08b') for byte in message.encode('utf-8'))


def bits_to_text(bits: str) -> str:
    """Convert a string of '0'/'1' characters back to a UTF-8 string."""
    if bits == "":
        return ""
    if len(bits) % 8 != 0:
        raise ValueError(f"Bit string length ({len(bits)}) is not a multiple of 8; "
                          "data may be corrupted.")
    byte_chunks = [bits[i:i + 8] for i in range(0, len(bits), 8)]
    byte_array = bytearray(int(chunk, 2) for chunk in byte_chunks)
    return byte_array.decode('utf-8')


def build_payload(message: str) -> str:
    """Build the full bitstream to embed: header (length) + message bits + end marker."""
    message_bits = text_to_bits(message)
    if len(message_bits) >= 2 ** HEADER_BITS:
        raise ValueError("Message too long to encode in header field.")
    header = format(len(message_bits), f'0{HEADER_BITS}b')
    return header + message_bits + END_MARKER


def get_message_bit_length(header_bits: str) -> int:
    """Decode the 32-bit header into an integer bit-length for the message."""
    if len(header_bits) != HEADER_BITS:
        raise ValueError(f"Header must be exactly {HEADER_BITS} bits, got {len(header_bits)}.")
    return int(header_bits, 2)
