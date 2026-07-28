"""
texnames.py — the texture name table (LEVEL.DAT section 3).

This table is the master index for every named graphical asset in a level. It
ties a name to a rectangle of PSX VRAM and to the CLUT (palette) that
rectangle is drawn with.

Record layout (24 bytes), verified on all 14 level files
--------------------------------------------------------
    +0x00  u16  vram_x    X position in VRAM, in 16-bit halfwords
    +0x02  u16  vram_y    Y position in VRAM, in scanlines
    +0x04  u16  width     tile width in PIXELS  (not halfwords)
    +0x06  u16  height    tile height in scanlines
    +0x08  u16  clut_x    CLUT X position in VRAM, in halfwords
    +0x0a  u16  clut_y    CLUT Y position in VRAM, in scanlines
    +0x0c  u16  bpp       literal bit depth: only ever 4 or 8
    +0x0e  char[10]       name, NUL-padded

Section layout: u32 count, then count * 24 bytes. The computed size matches
the section span exactly for every level file, which is what lets us trust
the record size.

Why we know the units are right
-------------------------------
Converting width from pixels to halfwords (16/bpp pixels per halfword) and
rasterising every record into a 1024x512 VRAM grid produces **zero
overlapping halfwords across all 14 levels**, and no record leaves the VRAM
bounds. Any other interpretation of the unit fields produces mass collisions.

Two sentinel forms mark records that are not resident in VRAM
-------------------------------------------------------------
    (vram_x, vram_y) == (320, 0)   staging slot: paged in on demand, or a
                                   palette-only record such as CLUT00A
    vram_y == 0xFFFF               palette-only record (seen on SMCL* in
                                   LEV1 and LEV3)

For those records only clut_x/clut_y carry meaning. They are the mechanism
behind the per-car livery system: one set of body tiles in VRAM, many CLUT
records recolouring it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .binio import FormatError, cstring, u16, u16_array, u32
from .vram import NO_CLUT

RECORD_SIZE = 24
NAME_OFFSET = 0x0E
NAME_SIZE = 10

VRAM_WIDTH_HALFWORDS = 1024
VRAM_HEIGHT = 512

# Coordinates that mean "not permanently resident in VRAM".
STAGING_SLOT = (320, 0)
NO_VRAM_Y = 0xFFFF

# --------------------------------------------------------------------------
# Car asset naming
# --------------------------------------------------------------------------
# Per-car assets are named <PREFIX><2-digit car number><variant>, for example
# BUMP88A, DR01B, CLUT99E. A bare regex over that shape produces false
# positives (LEV0's track thumbnails TRACK01L..TRACK11L match it), so we match
# only against prefixes we have actually identified.
#
# Surveyed across all 14 level files, the prefixes fall into three groups:
#
#   BODY_TILE_PREFIXES   real pixel data. Every one of these exists for car
#                        88 ONLY — that is the shared body tile set the whole
#                        grid is drawn from.
#                          BUMP  front/rear bumper     BKWN  back wing
#                          FRNT  front panel           FRWN  front wing
#                          BON   bonnet                ROOF  roof
#                          BOOT  boot                  DEBR  debris
#
#   PALETTE_PREFIXES     CLUT-only records that recolour the shared tiles.
#                          CLUT  19 cars (every car except 88, which uses the
#                                palette embedded in its own tile data)
#                          SMCL  20 cars, low-detail/distant palettes
#                          CLT   player alternate liveries only: slots 01, 02
#                                with variants A2/A3, B2/B3 ...
#
#   NUMBER_PANEL_PREFIX  DR — the door number panels, 3 damage states each
#                        (A undamaged, B, C). 21 numbers: the 20 cars plus a
#                        DR02 belonging to the player alternate livery.
#
BODY_TILE_PREFIXES = frozenset(
    {"BUMP", "BKWN", "FRNT", "FRWN", "BON", "ROOF", "BOOT", "DEBR"})
PALETTE_PREFIXES = frozenset({"CLUT", "SMCL", "CLT"})
NUMBER_PANEL_PREFIX = "DR"

# Prefixes that identify a genuine car number. CLT is excluded: its "01"/"02"
# are player livery slots, not cars.
_CAR_NUMBER_PREFIXES = frozenset({"CLUT", "SMCL", "DR"}) | BODY_TILE_PREFIXES

CAR_ASSET_PREFIXES = (BODY_TILE_PREFIXES | PALETTE_PREFIXES
                      | {NUMBER_PANEL_PREFIX})

# The 20 cars on the grid. Identical in every track file, derived from the
# SMCL* palette records, which are the one prefix that covers all 20 with no
# extras. Car 88 is the odd one out: it owns the shared body tiles and so has
# no CLUT* record of its own.
CANONICAL_CAR_NUMBERS = (
    "00", "01", "07", "13", "17", "35", "37", "40", "42", "47",
    "50", "52", "53", "64", "66", "69", "77", "82", "88", "99",
)

# Car whose pixel data is the shared body tile set.
BASE_CAR_NUMBER = "88"

_CAR_ASSET = re.compile(r"^(?P<part>[A-Z]+?)(?P<car>\d{2})(?P<variant>[A-Z]\d?)$")


@dataclass(frozen=True)
class TexName:
    """One record of the texture name table."""

    index: int
    name: str
    vram_x: int
    vram_y: int
    width: int
    height: int
    clut_x: int
    clut_y: int
    bpp: int

    # -- derived ------------------------------------------------------------

    @property
    def width_halfwords(self) -> int:
        """Tile width expressed in VRAM halfwords."""
        return self.width // (16 // self.bpp)

    @property
    def is_resident(self) -> bool:
        """True if this record occupies a fixed rectangle of VRAM."""
        return (self.vram_y != NO_VRAM_Y
                and (self.vram_x, self.vram_y) != STAGING_SLOT)

    @property
    def clut_entries(self) -> int:
        """Number of colours in this record's palette."""
        return 16 if self.bpp == 4 else 256

    @property
    def has_clut(self) -> bool:
        """
        False when clut_y is the 0xFFFE sentinel: the record has real pixels
        but names no palette of its own, so it cannot be decoded until we
        pair it with one. The same sentinel appears in the LEVEL.TX0 tile
        descriptors; see vram.NO_CLUT.
        """
        return self.clut_y != NO_CLUT

    @property
    def is_decodable(self) -> bool:
        """Resident pixels and a palette of its own — safe to turn into an image."""
        return self.is_resident and self.has_clut

    @property
    def _car_match(self) -> re.Match | None:
        m = _CAR_ASSET.match(self.name)
        if m is None or m.group("part") not in CAR_ASSET_PREFIXES:
            return None
        return m

    @property
    def part(self) -> str | None:
        """
        The asset prefix if this is a recognised car asset,
        e.g. 'BUMP88A' -> 'BUMP'. None for anything else, including
        lookalikes such as 'TRACK01L'.
        """
        m = self._car_match
        return m.group("part") if m else None

    @property
    def variant(self) -> str | None:
        """
        The variant suffix, e.g. 'BUMP88A' -> 'A', 'CLT01A2' -> 'A2'.
        For body tiles and DR panels this is the damage state.
        """
        m = self._car_match
        return m.group("variant") if m else None

    @property
    def car_number(self) -> str | None:
        """
        The two-digit car number, e.g. 'BUMP88A' -> '88', 'CLUT01E' -> '01'.

        Returns None for CLT* records: their '01'/'02' identify player
        alternate livery slots, not cars.
        """
        m = self._car_match
        if m is None or m.group("part") not in _CAR_NUMBER_PREFIXES:
            return None
        return m.group("car")

    @property
    def is_body_tile(self) -> bool:
        """A shared car body tile (pixel data, car 88 only)."""
        return self.part in BODY_TILE_PREFIXES

    @property
    def is_palette_record(self) -> bool:
        """A CLUT-only record that recolours the shared body tiles."""
        return self.part in PALETTE_PREFIXES

    @property
    def is_number_panel(self) -> bool:
        """A door number panel (DRnnA/B/C)."""
        return self.part == NUMBER_PANEL_PREFIX

    def __str__(self) -> str:
        where = (f"vram({self.vram_x},{self.vram_y})" if self.is_resident
                 else "non-resident")
        return (f"{self.name:<10} {self.width:>4}x{self.height:<4} "
                f"{self.bpp}bpp  {where:<22} clut({self.clut_x},{self.clut_y})")


class TextureNameTable:
    """Parsed texture name table with lookup and validation."""

    def __init__(self, data: bytes, offset: int, span: int):
        self.offset = offset
        self.span = span
        self.records: list[TexName] = self._parse(data, offset, span)
        self._by_name = {r.name: r for r in self.records}
        # Built on first tile_at() call; see the note there.
        self._occupancy: dict[tuple[int, int], TexName] | None = None

    @staticmethod
    def _parse(data: bytes, offset: int, span: int) -> list[TexName]:
        if span == 0:
            return []
        if span < 4:
            raise FormatError(
                f"texture name table span {span} is too small for a count word")

        count = u32(data, offset)
        expected = 4 + count * RECORD_SIZE
        if expected != span:
            raise FormatError(
                f"texture name table: count={count} implies {expected} bytes "
                f"but the section spans {span} bytes"
            )

        records: list[TexName] = []
        for i in range(count):
            base = offset + 4 + i * RECORD_SIZE
            vx, vy, w, h, cx, cy, bpp = u16_array(data, base, 7)
            name = cstring(data, base + NAME_OFFSET, NAME_SIZE)
            if bpp not in (4, 8):
                raise FormatError(
                    f"texture name table entry {i} ('{name}'): "
                    f"bpp={bpp}, expected 4 or 8"
                )
            records.append(TexName(index=i, name=name, vram_x=vx, vram_y=vy,
                                   width=w, height=h, clut_x=cx, clut_y=cy,
                                   bpp=bpp))
        return records

    # -- access -------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def get(self, name: str) -> TexName | None:
        return self._by_name.get(name)

    def resident(self) -> list[TexName]:
        return [r for r in self.records if r.is_resident]

    def non_resident(self) -> list[TexName]:
        return [r for r in self.records if not r.is_resident]

    def car_numbers(self) -> list[str]:
        """Sorted list of every car number referenced by this level."""
        return sorted({r.car_number for r in self.records
                       if r.car_number is not None})

    def for_car(self, car_number: str) -> list[TexName]:
        return [r for r in self.records if r.car_number == car_number]

    def body_tiles(self) -> list[TexName]:
        """The shared car body tiles (car 88's pixel data)."""
        return [r for r in self.records if r.is_body_tile]

    def number_panels(self) -> list[TexName]:
        """Door number panels, all cars and damage states."""
        return [r for r in self.records if r.is_number_panel]

    def player_alt_liveries(self) -> list[TexName]:
        """CLT* records: the player's alternate colour schemes."""
        return [r for r in self.records if r.part == "CLT"]

    # -- palette inheritance ------------------------------------------------

    def clut_source(self, rec: TexName) -> TexName | None:
        """
        Find the record that supplies `rec`'s palette.

        Returns `rec` itself when it has its own CLUT.

        Otherwise we apply exactly one rule, the damage-variant rule: a name
        ending in a letter other than "A" is a later damage state of the "A"
        variant of the same part, so `DR40B` and `DR40C` take `DR40A`'s
        palette. The candidate must be a recognised car asset, must share
        `rec`'s part and car number, and must have the same bit depth.

        Deliberately narrow. An earlier, looser version matched on any name
        sharing a prefix, and promptly paired LEV0's `CARD` menu sprite with an
        unrelated `CAR*` entry — a plausible-looking result that was simply
        wrong. Only the damage-variant pattern has actually been verified
        against the data, so only that is applied.

        Returns None where the palette is genuinely unknown: LEV0's menu
        sprites (`CONFIG`/`CONFIGP`, the `I*` icons) and LEV6's `FLASH1-6`
        and `GOOSE1-8` animation frames, whose whole families are palette-less
        and whose CLUT is chosen by code we have not traced. Callers should
        fall back to dumping raw indices rather than inventing colour.
        """
        if rec.has_clut:
            return rec

        # Must be a recognised car asset with a variant suffix to reason about.
        part, car, variant = rec.part, rec.car_number, rec.variant
        if part is None or car is None or not variant or variant == "A":
            return None

        candidate = self._by_name.get(f"{part}{car}A")
        if (candidate is not None and candidate.has_clut
                and candidate.bpp == rec.bpp):
            return candidate
        return None

    # -- validation ---------------------------------------------------------

    def validate(self) -> list[str]:
        """
        Check the VRAM residency model. Every resident tile must fit inside
        1024x512 and no two may claim the same halfword. This holds for all
        14 retail level files, so a violation means our field interpretation
        has broken.
        """
        problems: list[str] = []
        occupied: dict[tuple[int, int], str] = {}

        # Duplicate names make get() ambiguous and silently break palette
        # inheritance, so surface them.
        counts: dict[str, int] = {}
        for rec in self.records:
            counts[rec.name] = counts.get(rec.name, 0) + 1
        for name, n in sorted(counts.items()):
            if n > 1:
                problems.append(f"name {name!r} appears {n} times")

        for rec in self.resident():
            right = rec.vram_x + rec.width_halfwords
            bottom = rec.vram_y + rec.height
            if right > VRAM_WIDTH_HALFWORDS or bottom > VRAM_HEIGHT:
                problems.append(
                    f"{rec.name}: extends to ({right},{bottom}), "
                    f"outside {VRAM_WIDTH_HALFWORDS}x{VRAM_HEIGHT} VRAM"
                )
                continue

            for y in range(rec.vram_y, bottom):
                for x in range(rec.vram_x, right):
                    other = occupied.get((x, y))
                    if other is not None and other != rec.name:
                        problems.append(
                            f"{rec.name}: VRAM halfword ({x},{y}) "
                            f"already claimed by {other}"
                        )
                        break
                    occupied[(x, y)] = rec.name
                else:
                    continue
                break

        return problems

    def vram_halfwords_used(self) -> int:
        return sum(r.width_halfwords * r.height for r in self.resident())

    # -- VRAM lookup --------------------------------------------------------

    def tile_at(self, x: int, y: int) -> TexName | None:
        """
        Which named tile, if any, occupies VRAM halfword (x, y).

        Returns None for most of VRAM: the name table only indexes *named*
        assets — car body parts, UI elements, specific props. In LEV1 that is
        194 tiles covering 134,776 of VRAM's 524,288 halfwords. The remainder
        is filled by TX0-TX3 uploads that carry no name and are addressed
        purely through the UV table. Track surface textures live there, so
        this lookup identifies models but cannot substitute for building
        real VRAM.
        """
        if self._occupancy is None:
            self._occupancy = {}
            for rec in self.resident():
                for yy in range(rec.vram_y, rec.vram_y + rec.height):
                    for xx in range(rec.vram_x,
                                    rec.vram_x + rec.width_halfwords):
                        self._occupancy[(xx, yy)] = rec
        return self._occupancy.get((x, y))

    def tiles_for_uv(self, tpage, uvs) -> set[str]:
        """
        Names of the tiles a UV record samples.

        `tpage` is a uvtable.TPage and `uvs` a sequence of page-local pixel
        pairs. UV u is in pixels, so it is converted to halfwords using the
        page's colour depth before the lookup.
        """
        pixels_per_halfword = 16 // tpage.bpp if tpage.bpp in (4, 8) else 1
        found: set[str] = set()
        for u, v in uvs:
            rec = self.tile_at(tpage.x_base + u // pixels_per_halfword,
                               tpage.y_base + v)
            if rec is not None:
                found.add(rec.name)
        return found
