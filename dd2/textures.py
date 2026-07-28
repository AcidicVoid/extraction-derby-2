"""
textures.py — assemble a level's VRAM and pull images out of it.

Ties together the three pieces:

    LEVEL.TX0-TX3   tile pixels + their palettes   -> txfiles.TileArchive
    LEVEL.TXC       extra palette uploads          -> txfiles.ClutArchive
    LEVEL.DAT s.3   names for a subset of tiles    -> texnames.TextureNameTable

Upload order reproduces the game's: tiles first, then TXC on top. TXC exists
precisely to overwrite palettes that TX0-TX3 just wrote, so doing it in the
other order would silently produce the wrong colours for every car.

Two ways to get an image out:

    named_tile(name)        via the name table — for car parts and UI, where
                            we know what we are looking at
    tile_image(tile)        via a TX descriptor — works for all tiles,
                            including the unnamed track surface texture that
                            the name table does not cover
"""

from __future__ import annotations

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
        result reflects any palette that a later upload replaced — which is
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

    def named_tile(self, name: str) -> Image.Image:
        """Decode a tile by its name-table entry."""
        if self.names is None:
            raise FormatError("no texture name table was supplied")
        rec = self.names.get(name)
        if rec is None:
            raise KeyError(f"no texture named {name!r}")
        return self.record_image(rec)

    def record_image(self, rec: TexName) -> Image.Image:
        """Decode the tile a name-table record points at."""
        if not rec.is_resident:
            raise FormatError(
                f"{rec.name} is not resident in VRAM (it is a palette-only or "
                f"staged record); there are no pixels to read")
        if not rec.has_clut:
            raise FormatError(
                f"{rec.name} names no palette of its own (clut_y is the "
                f"0xFFFE sentinel); pair it with a CLUT record and use "
                f"record_image_for_car() or tile_image_with_clut()")
        return self.vram.tile_image(rec.vram_x, rec.vram_y, rec.width,
                                    rec.height, rec.bpp,
                                    rec.clut_x, rec.clut_y)

    def record_image_for_car(self, rec: TexName, car_clut: TexName
                             ) -> Image.Image:
        """
        Decode a shared body tile using another record's palette.

        This is the livery mechanism in one call: `rec` is a car-88 body tile
        such as BUMP88A, and `car_clut` is the CLUTnnA record whose palette
        recolours it.
        """
        return self.vram.tile_image(rec.vram_x, rec.vram_y, rec.width,
                                   rec.height, rec.bpp,
                                   car_clut.clut_x, car_clut.clut_y)

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


def export_named_tiles(textures: LevelTextures, dest: Path) -> tuple[int, int]:
    """
    Write every named, resident, decodable tile as an RGBA PNG.

    Returns (written, skipped). Skipped records are resident but name no
    palette of their own; they need pairing with a CLUT record first.

    Filenames carry the VRAM position and depth so a PNG can be traced back
    to the byte it came from without consulting the report.
    """
    if textures.names is None:
        return 0, 0
    dest.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    for rec in textures.names.resident():
        if not rec.has_clut:
            skipped += 1
            continue
        img = textures.record_image(rec)
        img.save(dest / f"{_safe(rec.name)}_{rec.width}x{rec.height}"
                        f"_{rec.bpp}bpp_{rec.vram_x}-{rec.vram_y}.png")
        written += 1
    return written, skipped


def export_all_tiles(textures: LevelTextures, dest: Path) -> tuple[int, int]:
    """
    Write every TX tile as an RGBA PNG, named or not.

    Returns (written, skipped). Tiles with no CLUT of their own are skipped —
    without a palette there is nothing meaningful to write.
    """
    dest.mkdir(parents=True, exist_ok=True)
    # Map VRAM position back to a name where we have one, so the unnamed
    # majority is easy to tell apart from the named minority.
    by_pos = {}
    if textures.names is not None:
        by_pos = {(r.vram_x, r.vram_y): r.name
                  for r in textures.names.resident()}

    written = skipped = 0
    for tile in textures.tiles:
        if not tile.has_clut:
            skipped += 1
            continue
        label = by_pos.get((tile.vram_x, tile.vram_y), "unnamed")
        img = textures.tile_image(tile)
        img.save(dest / f"tile{tile.index:04d}_{_safe(label)}"
                        f"_{tile.width}x{tile.height}_{tile.bpp}bpp"
                        f"_{tile.vram_x}-{tile.vram_y}.png")
        written += 1
    return written, skipped
