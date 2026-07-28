"""
LZSS codec for compressed terrain chunks.

Only the terrain chunks of the circuit tracks (LEV1-LEV7) are compressed.
Everything else, including the LEVEL.DAT container itself and the arena tracks'
terrain, is stored plainly.

Stream format:

    u32                total decompressed size
    then repeatedly:
    u8   control       eight flag bits, least significant first
    per bit:
      bit set          one literal byte follows
      bit clear        two bytes follow, a back-reference:
                           offset = (b1 | ((b2 & 0xF0) << 4)) - 0x1000
                           length = (b2 & 0x0F) + 3

The offset is always negative: the raw 12-bit value is biased by -0x1000,
making it a displacement back from the current write position rather than an
index into a ring buffer. Matches are copied one byte at a time so overlapping
copies work.
"""

from __future__ import annotations

import struct

from .binio import FormatError

# Back-references reach at most this far behind the write position.
WINDOW = 0x1000
MIN_MATCH = 3
HEADER_SIZE = 4


def decompress(src: bytes, limit: int | None = None) -> bytes:
    """
    Decompress an LZSS chunk.

    `limit` optionally caps the output, matching the original's ability to
    decode a chunk in slices. Raises FormatError on a malformed stream rather
    than returning a short or padded result.
    """
    if len(src) < HEADER_SIZE:
        raise FormatError(
            f"LZSS: {len(src)} bytes is too short to hold a size header")

    total = struct.unpack_from("<I", src, 0)[0]
    if limit is not None:
        total = min(total, limit)

    out = bytearray()
    pos = HEADER_SIZE

    while len(out) < total:
        if pos >= len(src):
            raise FormatError(
                f"LZSS: input exhausted at 0x{pos:X} with {len(out)} of "
                f"{total} bytes produced")

        # OR with 0xFF00 so the loop below runs exactly eight times.
        flags = src[pos] | 0xFF00
        pos += 1

        while flags & 0x100:
            if len(out) >= total:
                break

            if flags & 1:
                # Literal.
                if pos >= len(src):
                    raise FormatError(
                        f"LZSS: literal at 0x{pos:X} past end of input")
                out.append(src[pos])
                pos += 1
            else:
                # Back-reference.
                if pos + 1 >= len(src):
                    raise FormatError(
                        f"LZSS: back-reference at 0x{pos:X} past end of input")
                b1, b2 = src[pos], src[pos + 1]
                pos += 2
                offset = (b1 | ((b2 & 0xF0) << 4)) - WINDOW
                length = (b2 & 0x0F) + MIN_MATCH

                start = len(out) + offset
                if start < 0:
                    raise FormatError(
                        f"LZSS: back-reference at output 0x{len(out):X} "
                        f"reaches {-start} bytes before the start of the "
                        f"window; the stream or our decoding is wrong")
                # Byte at a time: matches may overlap the write position.
                for _ in range(length):
                    out.append(out[len(out) + offset])

            flags >>= 1

    if len(out) != total:
        raise FormatError(
            f"LZSS: produced {len(out)} bytes, header declared {total}")

    return bytes(out)


def declared_size(src: bytes) -> int:
    """The decompressed size a chunk claims, without decoding it."""
    if len(src) < HEADER_SIZE:
        raise FormatError("LZSS: too short to hold a size header")
    return struct.unpack_from("<I", src, 0)[0]
