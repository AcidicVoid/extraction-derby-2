"""
The LEVEL.TX0-TX3 tile archive and the LEVEL.TXC palette archive.

LEVEL.TX0-TX3 is one logical archive split across four files because the loader
streams them off the CD in sequence. TX0 carries the directory; the pixel
payload starts in TX0 after the directory and continues byte for byte through
TX1, TX2 and TX3 concatenated.

    TX0:  u32 count
          count x 16-byte tile descriptors
          payload (first part)
    TX1..TX3:  payload continued

Tile descriptor (16 bytes, eight u16 fields):

    +0x00  u16  bpp        4 or 8
    +0x02  u16  width      pixels
    +0x04  u16  height     scanlines
    +0x06  u16  vram_x     destination X, in halfwords
    +0x08  u16  vram_y     destination Y
    +0x0A  u16  clut_x     palette destination X, in halfwords
    +0x0C  u16  clut_y     palette destination Y, or 0xFFFE for "no palette"
    +0x0E  u16  pad        always 0

Payload block per tile, in descriptor order:

    "CLUT"                 4-byte ASCII tag
    palette                32 bytes (4bpp) or 512 bytes (8bpp)
    "TEXT"                 4-byte ASCII tag
    pixels                 width*height/2 (4bpp) or width*height (8bpp)

A descriptor with clut_y == 0xFFFE supplies no palette of its own. Its palette
bytes are still present in the stream but must not be uploaded; the tile
borrows a palette uploaded by another tile or by the TXC.

LEVEL.TXC holds extra palettes uploaded on top of what TX0-TX3 established,
which is the mechanism behind per-car liveries:

    u32 count
    count x 8-byte entries:  u16 bpp, u16 vram_x, u16 vram_y, u16 pad
    palette stream: 32 bytes per 4bpp entry, 512 per 8bpp, in entry order

Three levels carry unreferenced filler beyond what their entry count describes.
Only what the table describes is consumed; the rest is reported as slack.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .binio import FormatError, u16, u16_array, u32
from .vram import CLUT_ENTRIES, NO_CLUT, VRAM, pixels_per_halfword

TILE_DESCRIPTOR_SIZE = 16
TXC_ENTRY_SIZE = 8

TAG_CLUT = b"CLUT"
TAG_TEXT = b"TEXT"
TAG_SIZE = 4

TX_PART_NAMES = ("LEVEL.TX0", "LEVEL.TX1", "LEVEL.TX2", "LEVEL.TX3")
TXC_NAME = "LEVEL.TXC"


def palette_bytes(bpp: int) -> int:
    return CLUT_ENTRIES[bpp] * 2


@dataclass(frozen=True)
class Tile:
    """One tile descriptor plus the location of its data in the payload."""

    index: int
    bpp: int
    width: int          # pixels
    height: int
    vram_x: int         # halfwords
    vram_y: int
    clut_x: int
    clut_y: int
    palette_offset: int   # into the concatenated payload
    pixel_offset: int

    @property
    def has_clut(self) -> bool:
        return self.clut_y != NO_CLUT

    @property
    def width_halfwords(self) -> int:
        return self.width // pixels_per_halfword(self.bpp)

    @property
    def pixel_bytes(self) -> int:
        return self.width_halfwords * self.height * 2

    def __str__(self) -> str:
        clut = f"clut({self.clut_x},{self.clut_y})" if self.has_clut \
            else "no clut"
        return (f"[{self.index:>4}] {self.width:>4}x{self.height:<4} "
                f"{self.bpp}bpp  vram({self.vram_x:>4},{self.vram_y:>3})  {clut}")


class TileArchive:
    """LEVEL.TX0-TX3 parsed as one archive."""

    def __init__(self, table_data: bytes, payload: bytes,
                 source: str = "LEVEL.TX*"):
        self.source = source
        self.payload = payload
        self.tiles: list[Tile] = []
        self.payload_slack = 0
        self._parse(table_data, payload)

    # -- construction -------------------------------------------------------

    @classmethod
    def from_dir(cls, lev_dir: str | Path) -> "TileArchive":
        """
        Load from a directory holding LEVEL.TX0 and, optionally, TX1-TX3.

        Missing continuation parts are tolerated at load time; a short payload
        will be caught by the block walk with a precise error.
        """
        lev_dir = Path(lev_dir)
        head = lev_dir / TX_PART_NAMES[0]
        if not head.is_file():
            raise FileNotFoundError(f"{head} not found")

        first = head.read_bytes()
        count = u32(first, 0)
        table_end = 4 + count * TILE_DESCRIPTOR_SIZE
        if table_end > len(first):
            raise FormatError(
                f"{head.name}: descriptor count {count} needs {table_end} "
                f"bytes but the file is only {len(first)}")

        parts = [first[table_end:]]
        for name in TX_PART_NAMES[1:]:
            part = lev_dir / name
            if part.is_file():
                parts.append(part.read_bytes())

        return cls(first[:table_end], b"".join(parts),
                   source=f"{lev_dir.name}/LEVEL.TX*")

    def _parse(self, table_data: bytes, payload: bytes) -> None:
        count = u32(table_data, 0)
        cursor = 0     # into payload

        for i in range(count):
            base = 4 + i * TILE_DESCRIPTOR_SIZE
            bpp, width, height, vx, vy, cx, cy, pad = u16_array(table_data,
                                                                base, 8)
            if bpp not in (4, 8):
                raise FormatError(
                    f"{self.source}: tile {i} has bit depth {bpp}, "
                    f"expected 4 or 8")
            if pad != 0:
                raise FormatError(
                    f"{self.source}: tile {i} pad field is {pad}, expected 0")

            pal_size = palette_bytes(bpp)

            if payload[cursor:cursor + TAG_SIZE] != TAG_CLUT:
                raise FormatError(
                    f"{self.source}: tile {i} at payload 0x{cursor:X}: "
                    f"expected {TAG_CLUT!r} tag, found "
                    f"{payload[cursor:cursor + TAG_SIZE]!r}")
            palette_offset = cursor + TAG_SIZE

            text_at = palette_offset + pal_size
            if payload[text_at:text_at + TAG_SIZE] != TAG_TEXT:
                raise FormatError(
                    f"{self.source}: tile {i} at payload 0x{text_at:X}: "
                    f"expected {TAG_TEXT!r} tag, found "
                    f"{payload[text_at:text_at + TAG_SIZE]!r}")
            pixel_offset = text_at + TAG_SIZE

            tile = Tile(index=i, bpp=bpp, width=width, height=height,
                        vram_x=vx, vram_y=vy, clut_x=cx, clut_y=cy,
                        palette_offset=palette_offset,
                        pixel_offset=pixel_offset)

            end = pixel_offset + tile.pixel_bytes
            if end > len(payload):
                raise FormatError(
                    f"{self.source}: tile {i} pixel data ends at 0x{end:X}, "
                    f"past the end of the payload (0x{len(payload):X}). "
                    f"Are LEVEL.TX1-TX3 all present?")

            self.tiles.append(tile)
            cursor = end

        self.payload_slack = len(payload) - cursor

    # -- access -------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.tiles)

    def __iter__(self):
        return iter(self.tiles)

    def palette_of(self, tile: Tile) -> bytes:
        return self.payload[tile.palette_offset:
                            tile.palette_offset + palette_bytes(tile.bpp)]

    def pixels_of(self, tile: Tile) -> bytes:
        return self.payload[tile.pixel_offset:
                            tile.pixel_offset + tile.pixel_bytes]

    # -- VRAM ---------------------------------------------------------------

    def upload_to(self, vram: VRAM) -> None:
        """
        Replay every tile upload into `vram`, in archive order.

        Order matters: later uploads legitimately overwrite earlier ones, so
        replaying faithfully is what reproduces the state the game renders
        from.
        """
        for tile in self.tiles:
            vram.upload(tile.vram_x, tile.vram_y, tile.width_halfwords,
                        tile.height, self.pixels_of(tile),
                        label=f"tile{tile.index}")
            if tile.has_clut:
                vram.upload_clut(tile.clut_x, tile.clut_y, tile.bpp,
                                 self.palette_of(tile),
                                 label=f"tile{tile.index}.clut")

    # -- validation ---------------------------------------------------------

    def validate(self) -> list[str]:
        problems: list[str] = []
        occupied: dict[tuple[int, int], int] = {}

        for tile in self.tiles:
            if tile.vram_x + tile.width_halfwords > 1024 or \
                    tile.vram_y + tile.height > 512:
                problems.append(
                    f"tile {tile.index}: ({tile.vram_x},{tile.vram_y}) "
                    f"{tile.width}x{tile.height}@{tile.bpp}bpp leaves VRAM")
                continue
            for y in range(tile.vram_y, tile.vram_y + tile.height):
                for x in range(tile.vram_x,
                               tile.vram_x + tile.width_halfwords):
                    prev = occupied.get((x, y))
                    if prev is not None:
                        problems.append(
                            f"tile {tile.index} overwrites tile {prev} "
                            f"at VRAM ({x},{y})")
                        break
                    occupied[(x, y)] = tile.index
                else:
                    continue
                break

        if self.payload_slack:
            problems.append(
                f"{self.payload_slack} bytes of payload left unconsumed")

        return problems


# ---------------------------------------------------------------------------
# LEVEL.TXC
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClutUpload:
    """One palette upload described by LEVEL.TXC."""

    index: int
    bpp: int
    vram_x: int
    vram_y: int
    offset: int         # into the palette stream

    @property
    def entries(self) -> int:
        return CLUT_ENTRIES[self.bpp]

    def __str__(self) -> str:
        return (f"[{self.index:>3}] {self.bpp}bpp {self.entries:>3} colours "
                f"-> vram({self.vram_x},{self.vram_y})")


class ClutArchive:
    """LEVEL.TXC - additional CLUT uploads."""

    def __init__(self, data: bytes, source: str = TXC_NAME):
        self.source = source
        self.data = data
        self.uploads: list[ClutUpload] = []
        self.stream_slack = 0
        self._parse(data)

    @classmethod
    def from_dir(cls, lev_dir: str | Path) -> "ClutArchive":
        lev_dir = Path(lev_dir)
        path = lev_dir / TXC_NAME
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found")
        return cls(path.read_bytes(), source=f"{lev_dir.name}/{TXC_NAME}")

    def _parse(self, data: bytes) -> None:
        if len(data) < 4:
            raise FormatError(f"{self.source}: too small for a count word")
        count = u32(data, 0)
        table_end = 4 + count * TXC_ENTRY_SIZE
        if table_end > len(data):
            raise FormatError(
                f"{self.source}: count {count} needs {table_end} bytes "
                f"but the file is {len(data)}")

        cursor = table_end
        for i in range(count):
            base = 4 + i * TXC_ENTRY_SIZE
            bpp = u16(data, base)
            vx = u16(data, base + 2)
            vy = u16(data, base + 4)
            pad = u16(data, base + 6)
            if bpp not in (4, 8):
                raise FormatError(
                    f"{self.source}: entry {i} bit depth {bpp}, expected 4 or 8")
            if pad != 0:
                raise FormatError(
                    f"{self.source}: entry {i} pad is {pad}, expected 0")

            size = palette_bytes(bpp)
            if cursor + size > len(data):
                raise FormatError(
                    f"{self.source}: entry {i} palette ends at "
                    f"0x{cursor + size:X}, past end of file")

            self.uploads.append(ClutUpload(index=i, bpp=bpp, vram_x=vx,
                                           vram_y=vy, offset=cursor))
            cursor += size

        # LEV2/LEV7/LEV8 carry unreferenced filler here; see module docstring.
        self.stream_slack = len(data) - cursor

    def __len__(self) -> int:
        return len(self.uploads)

    def __iter__(self):
        return iter(self.uploads)

    def palette_of(self, upload: ClutUpload) -> bytes:
        return self.data[upload.offset:
                         upload.offset + palette_bytes(upload.bpp)]

    def upload_to(self, vram: VRAM) -> None:
        for up in self.uploads:
            vram.upload_clut(up.vram_x, up.vram_y, up.bpp,
                             self.palette_of(up), label=f"txc{up.index}")

    def validate(self) -> list[str]:
        """
        Trailing slack is reported for the record but is not an error: three
        retail levels ship unreferenced filler palettes.
        """
        problems: list[str] = []
        for up in self.uploads:
            if up.vram_x + up.entries > 1024 or not 0 <= up.vram_y < 512:
                problems.append(
                    f"TXC entry {up.index}: CLUT at ({up.vram_x},{up.vram_y}) "
                    f"for {up.bpp}bpp does not fit in VRAM")
        return problems
