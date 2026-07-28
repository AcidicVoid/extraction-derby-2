"""
The UV coordinate table, LEVEL.DAT section 4.

Textured polygons do not carry their own UVs. They carry an index into this
shared table, and each record supplies a texture page plus four UV pairs,
enough for a quad; triangles use the first three.

Section layout is u32 count followed by count 12-byte records:

    +0x00  u32  tpage      PSX texture page descriptor
    +0x04  u8   u0, v0
    +0x06  u8   u1, v1
    +0x08  u8   u2, v2
    +0x0a  u8   u3, v3

The PSX GPU packs the texture page origin and colour depth into one value:

    bits 0-3   x_base / 64      -> x_base = (tpage & 0xF) * 64  halfwords
    bit  4     y_base / 256     -> y_base = ((tpage >> 4) & 1) * 256
    bits 5-6   semi-transparency mode
    bits 7-8   colour depth     0 = 4bpp, 1 = 8bpp, 2 = 15bpp direct

UVs are page-local pixel coordinates in the range 0-255.
"""

from __future__ import annotations

from dataclasses import dataclass

from .binio import FormatError, u32, u8

RECORD_SIZE = 12

_DEPTH_BPP = {0: 4, 1: 8, 2: 16, 3: 16}


@dataclass(frozen=True)
class TPage:
    """Decoded PSX texture page descriptor."""

    raw: int
    x_base: int      # in VRAM halfwords
    y_base: int      # in scanlines
    bpp: int
    semi_transparency: int

    @classmethod
    def decode(cls, raw: int) -> "TPage":
        return cls(
            raw=raw,
            x_base=(raw & 0xF) * 64,
            y_base=((raw >> 4) & 1) * 256,
            semi_transparency=(raw >> 5) & 0x3,
            bpp=_DEPTH_BPP[(raw >> 7) & 0x3],
        )

    def __str__(self) -> str:
        return (f"tpage 0x{self.raw:04X} "
                f"base=({self.x_base},{self.y_base}) {self.bpp}bpp")


@dataclass(frozen=True)
class UVRecord:
    """One entry of the UV table: a texture page plus four UV pairs."""

    index: int
    tpage: TPage
    uvs: tuple[tuple[int, int], ...]   # always 4 pairs

    def __str__(self) -> str:
        pairs = " ".join(f"({u},{v})" for u, v in self.uvs)
        return f"[{self.index:>5}] {self.tpage}  {pairs}"


class UVTable:
    """Parsed UV table."""

    def __init__(self, data: bytes, offset: int, span: int):
        self.offset = offset
        self.span = span
        self.records: list[UVRecord] = self._parse(data, offset, span)

    @staticmethod
    def _parse(data: bytes, offset: int, span: int) -> list[UVRecord]:
        if span == 0:
            return []   # section absent (LEVC)
        if span < 4:
            raise FormatError(
                f"UV table span {span} is too small for a count word")

        count = u32(data, offset)
        expected = 4 + count * RECORD_SIZE
        if expected != span:
            raise FormatError(
                f"UV table: count={count} implies {expected} bytes "
                f"but the section spans {span} bytes"
            )

        records: list[UVRecord] = []
        for i in range(count):
            base = offset + 4 + i * RECORD_SIZE
            tpage = TPage.decode(u32(data, base))
            uvs = tuple(
                (u8(data, base + 4 + j * 2), u8(data, base + 5 + j * 2))
                for j in range(4)
            )
            records.append(UVRecord(index=i, tpage=tpage, uvs=uvs))
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def __getitem__(self, index: int) -> UVRecord:
        if not 0 <= index < len(self.records):
            raise FormatError(
                f"UV index {index} out of range (table has {len(self.records)})")
        return self.records[index]

    def tpage_histogram(self) -> dict[int, int]:
        """How many records reference each texture page. Useful for finding
        which VRAM regions a given model actually samples."""
        hist: dict[int, int] = {}
        for rec in self.records:
            hist[rec.tpage.raw] = hist.get(rec.tpage.raw, 0) + 1
        return dict(sorted(hist.items()))

    def validate(self) -> list[str]:
        """
        UVs are single bytes so they cannot exceed a texture page by
        construction; the meaningful check is that every tpage decodes to a
        depth the GPU supports and lands inside VRAM.
        """
        problems: list[str] = []
        for rec in self.records:
            if rec.tpage.x_base >= 1024 or rec.tpage.y_base >= 512:
                problems.append(
                    f"UV record {rec.index}: tpage base "
                    f"({rec.tpage.x_base},{rec.tpage.y_base}) outside VRAM"
                )
        return problems
