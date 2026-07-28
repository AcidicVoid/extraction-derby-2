"""
Track terrain: LEVEL.DAT section 0.

Section 0 opens with a nested table of chunk offsets, and each chunk holds a
list of model blocks placed at absolute world positions. Assembled, they are
the complete track including the drivable road.

Section 0 layout:

    u32           table size in bytes (table_size / 4 = chunk count)
    u32 x N       chunk offsets, relative to the start of section 0
    ...           chunk payloads

Chunk payload, after decompression where applicable:

    u32           record count, never more than 32
    per record, 16 bytes:
        +0x00  u32  offset of a model block, relative to the chunk start
        +0x04  i32  world X
        +0x08  i32  world Y
        +0x0C  i32  world Z

Every record offset resolves to a valid 0x2C model block, so terrain geometry
is read by dd2.model unchanged.

Circuit tracks (levels 0-7) store their chunks LZSS compressed; arena tracks
(levels 8-B) store them plainly. Both use the same record format. The variant
is detected from the data: an uncompressed chunk begins with a record count of
32 or less, a compressed one with its decompressed size, which is thousands.

The game streams chunks in and out as the player drives, keeping at most 14
resident. For extraction the union of all chunks is taken.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .binio import FormatError, i32, u32, u32_array
from .lzss import declared_size
from .lzss import decompress as lzss_decompress
from .model import HEADER_SIZE as MODEL_HEADER_SIZE
from .model import ModelBlock, parse_model_block

RECORD_SIZE = 16

# A chunk never holds more than this many records; the loader uses the count
# reaching 32 as its "there may be more chunks" signal.
MAX_RECORDS = 32

# Slack after the last model in a chunk. The decompressed size is not aligned,
# so a handful of bytes trail the final polygon terminator; observed 1-6.
MAX_CHUNK_PADDING = 16


def coarse_translation(value: int) -> int:
    """
    The translation actually applied to a placed model, from its record field.

    `FUN_80025454` splits each of the three position words in two:

        coarse (int)   (value & 0xFFFF8000) + 0x4000     -> object translation
        fine   (short) (value & 0x7FFF)     + 0xC000     -> a separate array

    The two sum back to `value`, but only the coarse half is the object's
    translation. The fine half goes to a separate array of shorts and is not a
    vertex offset, so adding the raw word double-counts the low 15 bits and
    drops the object 0x4000 below where it belongs.
    """
    signed = value & 0xFFFF8000
    if signed & 0x80000000:
        signed -= 1 << 32
    return signed + 0x4000


@dataclass(frozen=True)
class TerrainInstance:
    """One model block placed at a world position."""

    chunk: int
    index: int
    model_offset: int
    # The record's three position words, exactly as stored.
    position: tuple[int, int, int]
    model: ModelBlock

    @property
    def origin(self) -> tuple[int, int, int]:
        """
        Where to actually put the model: the coarse half of each position word.
        Add this to the model's vertices; see `coarse_translation`.
        """
        return (coarse_translation(self.position[0]),
                coarse_translation(self.position[1]),
                coarse_translation(self.position[2]))

    def __str__(self) -> str:
        x, y, z = self.origin
        return (f"chunk{self.chunk:>3}[{self.index:>2}] @0x{self.model_offset:06X} "
                f"origin=({x},{y},{z}) {len(self.model.vertices)}v "
                f"{len(self.model.polygons)}p")


@dataclass
class TerrainChunk:
    """One decoded terrain chunk."""

    index: int
    offset: int              # within section 0
    stored_size: int         # bytes as stored on disc
    compressed: bool
    data: bytes              # payload, decompressed if it needed to be
    instances: list[TerrainInstance] = field(default_factory=list)

class Terrain:
    """Section 0 of a track LEVEL.DAT, fully decoded."""

    def __init__(self, data: bytes, offset: int, size: int,
                 name: str = "terrain"):
        self.name = name
        self.offset = offset
        self.size = size
        self.chunks: list[TerrainChunk] = []
        self.compressed: bool | None = None
        if size > 0:
            self._parse(data, offset, size)

    # -- parsing ------------------------------------------------------------

    def _parse(self, data: bytes, offset: int, size: int) -> None:
        table_bytes = u32(data, offset)
        if table_bytes % 4 or table_bytes < 4 or table_bytes > size:
            raise FormatError(
                f"{self.name}: chunk table size {table_bytes} is not a "
                f"sensible multiple of 4 within a {size}-byte section")
        count = table_bytes // 4

        offsets = list(u32_array(data, offset, count))
        if offsets[0] != table_bytes:
            raise FormatError(
                f"{self.name}: first chunk offset is 0x{offsets[0]:X}, "
                f"expected the table size 0x{table_bytes:X}")
        for i in range(count - 1):
            if offsets[i] >= offsets[i + 1]:
                raise FormatError(
                    f"{self.name}: chunk offsets are not increasing at "
                    f"index {i} (0x{offsets[i]:X} >= 0x{offsets[i + 1]:X})")
        if offsets[-1] > size:
            raise FormatError(
                f"{self.name}: last chunk offset 0x{offsets[-1]:X} is past "
                f"the end of the section")

        bounds = offsets + [size]
        for i in range(count):
            start = offset + bounds[i]
            stored = bounds[i + 1] - bounds[i]
            self.chunks.append(
                self._parse_chunk(i, data[start:start + stored],
                                  bounds[i], stored))

    def _parse_chunk(self, index: int, raw: bytes, offset: int,
                     stored: int) -> TerrainChunk:
        # Discriminate by the leading word: a record count is at most 32, a
        # decompressed size is in the thousands.
        leading = u32(raw, 0)
        compressed = leading > MAX_RECORDS

        if compressed:
            payload = lzss_decompress(raw)
            expected = declared_size(raw)
            if len(payload) != expected:
                raise FormatError(
                    f"{self.name}: chunk {index} decompressed to "
                    f"{len(payload)} bytes, header said {expected}")
        else:
            payload = raw

        if self.compressed is None:
            self.compressed = compressed
        elif self.compressed != compressed:
            raise FormatError(
                f"{self.name}: chunk {index} is "
                f"{'compressed' if compressed else 'uncompressed'} but "
                f"earlier chunks were not; mixed storage is not expected")

        chunk = TerrainChunk(index=index, offset=offset, stored_size=stored,
                             compressed=compressed, data=payload)

        count = u32(payload, 0)
        if count > MAX_RECORDS:
            raise FormatError(
                f"{self.name}: chunk {index} declares {count} records, "
                f"more than the {MAX_RECORDS} the loader allows")
        needed = 4 + count * RECORD_SIZE
        if needed > len(payload):
            raise FormatError(
                f"{self.name}: chunk {index} needs {needed} bytes for "
                f"{count} records but holds only {len(payload)}")

        records = [
            (u32(payload, 4 + i * RECORD_SIZE),
             (i32(payload, 4 + i * RECORD_SIZE + 4),
              i32(payload, 4 + i * RECORD_SIZE + 8),
              i32(payload, 4 + i * RECORD_SIZE + 12)))
            for i in range(count)
        ]

        # A model block carries no length, so its extent is "up to the next
        # model in the chunk". Records may share an offset - the same mesh
        # placed more than once - so work from the sorted distinct offsets.
        # The last block runs to the end of the chunk. The parser insists the
        # polygon stream lands exactly on the end it is given, which turns this
        # into a strong check rather than an assumption.
        starts = sorted({offset for offset, _ in records})
        extent = {
            start: (starts[i + 1] if i + 1 < len(starts) else len(payload))
            for i, start in enumerate(starts)
        }

        parsed: dict[int, ModelBlock] = {}
        for i, (model_offset, position) in enumerate(records):
            if model_offset + MODEL_HEADER_SIZE > len(payload):
                raise FormatError(
                    f"{self.name}: chunk {index} record {i} points at "
                    f"0x{model_offset:X}, past the end of the chunk")

            model = parsed.get(model_offset)
            if model is None:
                # Only the final block's extent is uncertain: every earlier one
                # is bounded by the next block's offset and must land exactly.
                is_last = model_offset == starts[-1]
                model = parse_model_block(
                    payload, model_offset, extent[model_offset] - model_offset,
                    name=f"{self.name}:c{index}@0x{model_offset:X}",
                    allow_trailing=is_last)
                if model.trailing_bytes > MAX_CHUNK_PADDING:
                    raise FormatError(
                        f"{self.name}: chunk {index} has "
                        f"{model.trailing_bytes} bytes left after its last "
                        f"model, more than the {MAX_CHUNK_PADDING} expected "
                        f"as padding")
                parsed[model_offset] = model

            chunk.instances.append(
                TerrainInstance(chunk=index, index=i,
                                model_offset=model_offset, position=position,
                                model=model))

        return chunk

    # -- access -------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.chunks)

    def __iter__(self):
        return iter(self.chunks)

    @property
    def instances(self) -> list[TerrainInstance]:
        """Every placed model across all chunks."""
        return [inst for chunk in self.chunks for inst in chunk.instances]

    def bounds(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """World-space bounds of the assembled terrain."""
        lo = [1 << 30] * 3
        hi = [-(1 << 30)] * 3
        for inst in self.instances:
            mlo, mhi = inst.model.bounds()
            origin = inst.origin
            for k in range(3):
                lo[k] = min(lo[k], origin[k] + mlo[k])
                hi[k] = max(hi[k], origin[k] + mhi[k])
        if lo[0] > hi[0]:
            return ((0, 0, 0), (0, 0, 0))
        return (tuple(lo), tuple(hi))

    def summary(self) -> dict:
        insts = self.instances
        lo, hi = self.bounds()
        return {
            "chunks": len(self.chunks),
            "compressed": bool(self.compressed),
            "instances": len(insts),
            "vertices": sum(len(i.model.vertices) for i in insts),
            "polygons": sum(len(i.model.polygons) for i in insts),
            "stored_bytes": self.size,
            "decoded_bytes": sum(len(c.data) for c in self.chunks),
            "bounds_min": list(lo),
            "bounds_max": list(hi),
        }
