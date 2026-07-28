"""
Assemble a level's VRAM and read images out of it.

Combines the three sources:

    LEVEL.TX0-TX3   tile pixels and their palettes
    LEVEL.TXC       extra palette uploads
    LEVEL.DAT s.3   names for a subset of the tiles

Upload order matters and reproduces the game's: tiles first, then TXC on top.
TXC exists to overwrite palettes that TX0-TX3 just wrote, so reversing the
order produces the wrong colours for every car.

The name table only covers named assets such as car parts and UI elements.
Most of VRAM is unnamed track surface texture, reachable only through the UV
table, so tiles can also be read directly from a TX descriptor.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .binio import FormatError
from .texnames import TexName, TextureNameTable
from .txfiles import ClutArchive, Tile, TileArchive
from .vram import VRAM


class LevelTextures:
    """A level's fully populated VRAM, plus the indexes needed to read it."""

    def __init__(self, tiles: TileArchive, cluts: ClutArchive | None,
                 names: TextureNameTable | None = None):
        self.tiles = tiles
        self.cluts = cluts
        self.names = names

        self.vram = VRAM()
        # Order is load-bearing: TXC palettes must land on top of the
        # palettes TX0-TX3 uploaded.
        tiles.upload_to(self.vram)
        if cluts is not None:
            cluts.upload_to(self.vram)

    @classmethod
    def from_dir(cls, lev_dir: str | Path,
                 names: TextureNameTable | None = None) -> "LevelTextures":
        lev_dir = Path(lev_dir)
        tiles = TileArchive.from_dir(lev_dir)
        try:
            cluts = ClutArchive.from_dir(lev_dir)
        except FileNotFoundError:
            cluts = None
        return cls(tiles, cluts, names=names)

    # -- image extraction ---------------------------------------------------

    def tile_image(self, tile: Tile) -> Image.Image:
        """
        Decode a TX tile from VRAM.

        Read back from VRAM rather than straight from the payload, so the
        result reflects any palette that a later upload replaced - which is
        the whole point of the TXC pass.
        """
        if not tile.has_clut:
            raise FormatError(
                f"tile {tile.index} declares no CLUT of its own; its palette "
                f"is supplied elsewhere. Decode it through a named record, or "
                f"pass an explicit CLUT position.")
        return self.vram.tile_image(tile.vram_x, tile.vram_y, tile.width,
                                    tile.height, tile.bpp,
                                    tile.clut_x, tile.clut_y)

    def tile_image_with_clut(self, tile: Tile, clut_x: int,
                             clut_y: int) -> Image.Image:
        """Decode a tile against an explicitly chosen palette."""
        return self.vram.tile_image(tile.vram_x, tile.vram_y, tile.width,
                                    tile.height, tile.bpp, clut_x, clut_y)

    def resolve_clut(self, tile: Tile, rec: TexName | None
                     ) -> tuple[tuple[int, int] | None, str | None]:
        """
        Decide which palette to decode `tile` with.

        Returns ((clut_x, clut_y), borrowed_from_name). `borrowed_from_name` is
        None when the palette belongs to the tile or its own name record, and
        the source record's name when it was inherited from a sibling.
        Returns (None, None) when the palette cannot be determined.

        Resolution order, in decreasing authority:

        1. The name-table record's own CLUT. Preferred because the name table
           is the more complete of the two sources: in LEV0, 42 tiles have no
           CLUT in their TX descriptor but do have one here. Everywhere else
           the two agree exactly, so preferring one costs nothing.
        2. The TX descriptor's CLUT.
        3. A sibling record's CLUT via the damage-variant rule
           (DR40B -> DR40A). See TextureNameTable.clut_source.
        4. Give up.
        """
        if rec is not None and rec.has_clut:
            return (rec.clut_x, rec.clut_y), None

        if tile.has_clut:
            return (tile.clut_x, tile.clut_y), None

        if rec is not None and self.names is not None:
            source = self.names.clut_source(rec)
            if source is not None and source.name != rec.name:
                return (source.clut_x, source.clut_y), source.name

        return None, None

    # -- diagnostics --------------------------------------------------------

    def validate(self) -> list[str]:
        problems = list(self.tiles.validate())
        if self.cluts is not None:
            problems += self.cluts.validate()

        # Every resident name-table record should correspond to a real tile.
        if self.names is not None:
            descriptors = {(t.vram_x, t.vram_y, t.width, t.height, t.bpp)
                           for t in self.tiles}
            for rec in self.names.resident():
                key = (rec.vram_x, rec.vram_y, rec.width, rec.height, rec.bpp)
                if key not in descriptors:
                    problems.append(
                        f"named record {rec.name} at vram"
                        f"({rec.vram_x},{rec.vram_y}) {rec.width}x{rec.height}"
                        f"@{rec.bpp}bpp has no matching TX tile descriptor")

        return problems

    def advisories(self) -> list[str]:
        """
        Inconsistencies in the source data that we handle deliberately.

        Kept apart from validate() so a known, benign quirk in the retail files
        does not mask a real parsing failure. One case exists: LEV0's MEMLOAD is
        assigned a different CLUT by its TX descriptor than by the name table.
        The name table wins, per resolve_clut()'s ordering.
        """
        notes: list[str] = []
        if self.names is None:
            return notes

        by_pos = {(r.vram_x, r.vram_y): r for r in self.names.resident()}
        for tile in self.tiles:
            rec = by_pos.get((tile.vram_x, tile.vram_y))
            if rec is None or not tile.has_clut or not rec.has_clut:
                continue
            if (tile.clut_x, tile.clut_y) != (rec.clut_x, rec.clut_y):
                notes.append(
                    f"CLUT disagreement for {rec.name}: TX descriptor says "
                    f"({tile.clut_x},{tile.clut_y}), name table says "
                    f"({rec.clut_x},{rec.clut_y}) - using the name table")
        return notes

    def summary(self) -> dict:
        written, total = self.vram.coverage()
        return {
            "tiles": len(self.tiles),
            "tiles_without_clut": sum(1 for t in self.tiles if not t.has_clut),
            "txc_uploads": len(self.cluts) if self.cluts else 0,
            "txc_slack": self.cluts.stream_slack if self.cluts else 0,
            "vram_halfwords_written": written,
            "vram_halfwords_total": total,
            "named_records": len(self.names) if self.names else 0,
        }


# ---------------------------------------------------------------------------
# PNG export
# ---------------------------------------------------------------------------

def _safe(name: str) -> str:
    """Make a texture name safe for a filename."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


@dataclass
class ExportStats:
    """Outcome of a texture export pass."""

    named: int = 0          # named tile, own palette
    inherited: int = 0      # named tile, palette borrowed from a sibling
    unnamed: int = 0        # tile with no name-table entry (track surface, props)
    unpaletted: int = 0     # palette unknown; written as a greyscale index map

    @property
    def total(self) -> int:
        return self.named + self.inherited + self.unnamed + self.unpaletted

    def __str__(self) -> str:
        return (f"{self.total} tiles: {self.named} named, "
                f"{self.inherited} named+inherited palette, "
                f"{self.unnamed} unnamed, "
                f"{self.unpaletted} unpaletted (index maps)")


def export_tiles(textures: LevelTextures, dest: Path,
                 named_only: bool = False) -> ExportStats:
    """
    Write every tile in the level as a PNG, sorted into subdirectories.

        named/        tiles the name table names, in full colour
        unnamed/      everything else - road surface, props, scenery
        unpaletted/   greyscale index maps for tiles whose palette is unknown

    Every tile in the archive is accounted for in exactly one directory, so
    nothing is silently dropped. `named_only` restricts output to named/ for
    when the unnamed bulk is just noise.

    Filenames carry the VRAM position and depth so a PNG can always be traced
    back to the bytes it came from.
    """
    stats = ExportStats()

    # Name-table records keyed by VRAM position, so a TX tile can be matched
    # to its name. Position is the join key both formats agree on.
    by_pos: dict[tuple[int, int], TexName] = {}
    if textures.names is not None:
        by_pos = {(r.vram_x, r.vram_y): r
                  for r in textures.names.resident()}

    named_dir = dest / "named"
    unnamed_dir = dest / "unnamed"
    unpal_dir = dest / "unpaletted"

    for tile in textures.tiles:
        rec = by_pos.get((tile.vram_x, tile.vram_y))
        geometry = (f"{tile.width}x{tile.height}_{tile.bpp}bpp"
                    f"_{tile.vram_x}-{tile.vram_y}")
        clut, borrowed_from = textures.resolve_clut(tile, rec)

        # Palette could not be determined. Preserve the pixels as an index map
        # rather than inventing colour or dropping the tile.
        if clut is None:
            if not named_only:
                unpal_dir.mkdir(parents=True, exist_ok=True)
                label = _safe(rec.name) if rec is not None \
                    else f"tile{tile.index:04d}"
                textures.vram.index_image(
                    tile.vram_x, tile.vram_y, tile.width, tile.height,
                    tile.bpp).save(
                    unpal_dir / f"{label}_{geometry}_indices.png")
                stats.unpaletted += 1
            continue

        img = textures.tile_image_with_clut(tile, *clut)

        if rec is None:
            if not named_only:
                unnamed_dir.mkdir(parents=True, exist_ok=True)
                img.save(unnamed_dir / f"tile{tile.index:04d}_{geometry}.png")
                stats.unnamed += 1
            continue

        named_dir.mkdir(parents=True, exist_ok=True)
        if borrowed_from is not None:
            img.save(named_dir / f"{_safe(rec.name)}_{geometry}"
                                 f"_clut-{_safe(borrowed_from)}.png")
            stats.inherited += 1
        else:
            img.save(named_dir / f"{_safe(rec.name)}_{geometry}.png")
            stats.named += 1

    return stats
