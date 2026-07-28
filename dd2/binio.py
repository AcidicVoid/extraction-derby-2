"""
binio.py — bounds-checked little-endian binary reading primitives.

The PS1 data files contain no magic numbers and no length fields we can trust
blindly, so every read goes through here. A read past the end of the buffer is
a bug in our format assumptions, not a recoverable condition — hence the hard
exception rather than a silent zero.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass


class FormatError(Exception):
    """Raised when parsed data violates a documented structural invariant."""


class OutOfBounds(FormatError):
    """Raised when a read would leave the bounds of the source buffer."""


# ---------------------------------------------------------------------------
# Free functions — for random access into a buffer we already hold
# ---------------------------------------------------------------------------

def _check(data: bytes, offset: int, length: int, what: str) -> None:
    """Verify that `length` bytes can be read at `offset`."""
    if offset < 0:
        raise OutOfBounds(f"{what}: negative offset {offset}")
    if offset + length > len(data):
        raise OutOfBounds(
            f"{what}: read of {length} bytes at 0x{offset:X} "
            f"exceeds buffer of 0x{len(data):X} bytes"
        )


def u8(data: bytes, offset: int) -> int:
    _check(data, offset, 1, "u8")
    return data[offset]


def i8(data: bytes, offset: int) -> int:
    _check(data, offset, 1, "i8")
    return struct.unpack_from("<b", data, offset)[0]


def u16(data: bytes, offset: int) -> int:
    _check(data, offset, 2, "u16")
    return struct.unpack_from("<H", data, offset)[0]


def i16(data: bytes, offset: int) -> int:
    _check(data, offset, 2, "i16")
    return struct.unpack_from("<h", data, offset)[0]


def u32(data: bytes, offset: int) -> int:
    _check(data, offset, 4, "u32")
    return struct.unpack_from("<I", data, offset)[0]


def i32(data: bytes, offset: int) -> int:
    _check(data, offset, 4, "i32")
    return struct.unpack_from("<i", data, offset)[0]


def u16_array(data: bytes, offset: int, count: int) -> tuple[int, ...]:
    _check(data, offset, count * 2, f"u16[{count}]")
    return struct.unpack_from(f"<{count}H", data, offset)


def u32_array(data: bytes, offset: int, count: int) -> tuple[int, ...]:
    _check(data, offset, count * 4, f"u32[{count}]")
    return struct.unpack_from(f"<{count}I", data, offset)


def i32_array(data: bytes, offset: int, count: int) -> tuple[int, ...]:
    _check(data, offset, count * 4, f"i32[{count}]")
    return struct.unpack_from(f"<{count}i", data, offset)


def cstring(data: bytes, offset: int, max_length: int,
            encoding: str = "ascii") -> str:
    """
    Read a NUL-padded fixed-width string field.

    DD2 pads its name fields with NULs to a fixed width rather than storing a
    length, so we take everything up to the first NUL within the field.
    """
    _check(data, offset, max_length, f"cstring[{max_length}]")
    raw = data[offset:offset + max_length]
    return raw.split(b"\0", 1)[0].decode(encoding, errors="replace")


# ---------------------------------------------------------------------------
# Cursor — for sequential parsing of a record stream
# ---------------------------------------------------------------------------

@dataclass
class Cursor:
    """
    A sequential read head over a buffer.

    Used where the format is a stream of variable-length records (e.g. the
    polygon command stream) and tracking the offset by hand would be noisy.
    """

    data: bytes
    offset: int = 0
    # Optional hard limit, so a sub-region can be parsed without slicing.
    limit: int | None = None

    @property
    def end(self) -> int:
        return len(self.data) if self.limit is None else self.limit

    @property
    def remaining(self) -> int:
        return self.end - self.offset

    def eof(self) -> bool:
        return self.offset >= self.end

    def _take(self, length: int, what: str) -> int:
        if self.offset + length > self.end:
            raise OutOfBounds(
                f"{what}: read of {length} bytes at 0x{self.offset:X} "
                f"exceeds region ending at 0x{self.end:X}"
            )
        start = self.offset
        self.offset += length
        return start

    def u8(self) -> int:
        return self.data[self._take(1, "u8")]

    def u16(self) -> int:
        return struct.unpack_from("<H", self.data, self._take(2, "u16"))[0]

    def i16(self) -> int:
        return struct.unpack_from("<h", self.data, self._take(2, "i16"))[0]

    def u32(self) -> int:
        return struct.unpack_from("<I", self.data, self._take(4, "u32"))[0]

    def i32(self) -> int:
        return struct.unpack_from("<i", self.data, self._take(4, "i32"))[0]

    def bytes(self, length: int) -> bytes:
        start = self._take(length, f"bytes[{length}]")
        return self.data[start:start + length]

    def skip(self, length: int) -> None:
        self._take(length, f"skip[{length}]")

    def seek(self, offset: int) -> None:
        if offset < 0 or offset > self.end:
            raise OutOfBounds(f"seek to 0x{offset:X} outside region")
        self.offset = offset
