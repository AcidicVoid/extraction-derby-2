"""
vram.py — a model of PlayStation video RAM.

The PS1 GPU has one flat 1024 x 512 framebuffer of 16-bit halfwords, and
*everything* lives in it: the visible framebuffer, texture pixels, and colour
lookup tables. Textures are addressed by halfword position but their pixels
are packed 4 or 8 bits wide, so a 4bpp tile 64 pixels across occupies only 16
halfwords. Keeping that distinction straight is the whole job of this module.

Colour format is BGR555 packed into a halfword:

    bit  0-4    red
    bit  5-9    green
    bit 10-14   blue
    bit 15      STP — "semi-transparent" flag, used with blending modes

Transparency: for a paletted texture the GPU treats a palette entry of
0x0000 (all channels zero, STP clear) as fully transparent. That is the rule
applied here. Note this means genuine black must be stored as 0x8000, and the
game does exactly that. In LEV1, 109 of 340 palettes use 0x0000 at index 0.

A tempting false lead: 63 of LEV1's TXC palettes hold 0x83E0 — pure green
with STP set — at index 0, and WINFRN88 holds 0xF360 (bright cyan) there.
Both look exactly like a chroma key. They are not. PS1 hardware keys only on
0x0000, and these are opaque colours. Checked directly: the 8bpp car body
tiles that reference those TXC palettes (BUMP88A, FRNT88A, FRWN88A, BKWN88A)
never use index 0 at all, so the slot is simply unused. Treating green as
transparent would have been wrong and would have punched holes in the cars.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .binio import FormatError

WIDTH = 1024        # halfwords
HEIGHT = 512        # scanlines

# CLUT sizes in halfwords, by texture bit depth.
CLUT_ENTRIES = {4: 16, 8: 256}

# Sentinel used in the clut_y field of BOTH the LEVEL.DAT texture name table
# and the LEVEL.TX0 tile descriptors: "this record supplies no palette of its
# own". The tile's pixels are real; its colours come from an upload made
# elsewhere, so it cannot be decoded until we know which palette to pair it
# with. Shared convention across the two formats, hence defined here.
NO_CLUT = 0xFFFE


def bgr555_to_rgba(word: int) -> tuple[int, int, int, int]:
    """
    Convert one BGR555 halfword to RGBA8888.

    The 5-bit channels are expanded to 8 bits by replicating the top 3 bits
    into the low bits, so 31 maps to 255 rather than 248.
    """
    r = word & 0x1F
    g = (word >> 5) & 0x1F
    b = (word >> 10) & 0x1F
    alpha = 0 if word == 0 else 255
    return (
        (r << 3) | (r >> 2),
        (g << 3) | (g >> 2),
        (b << 3) | (b >> 2),
        alpha,
    )


def bgr555_array_to_rgba(words: np.ndarray) -> np.ndarray:
    """Vectorised bgr555_to_rgba over an array of halfwords -> (..., 4) uint8."""
    words = words.astype(np.uint16)
    r = (words & 0x1F).astype(np.uint8)
    g = ((words >> 5) & 0x1F).astype(np.uint8)
    b = ((words >> 10) & 0x1F).astype(np.uint8)
    out = np.empty(words.shape + (4,), dtype=np.uint8)
    out[..., 0] = (r << 3) | (r >> 2)
    out[..., 1] = (g << 3) | (g >> 2)
    out[..., 2] = (b << 3) | (b >> 2)
    out[..., 3] = np.where(words == 0, 0, 255).astype(np.uint8)
    return out


def pixels_per_halfword(bpp: int) -> int:
    if bpp not in (4, 8):
        raise FormatError(f"unsupported texture bit depth {bpp}")
    return 16 // bpp


class VRAM:
    """
    A 1024 x 512 halfword framebuffer.

    Uploads are tracked so we can tell which halfwords actually received data
    and report on the rest — unwritten VRAM means a tile source we have not
    accounted for.
    """

    def __init__(self):
        self.data = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)
        self.written = np.zeros((HEIGHT, WIDTH), dtype=bool)
        # (x, y, w, h, label) for every upload, for diagnostics.
        self.uploads: list[tuple[int, int, int, int, str]] = []

    # -- writing ------------------------------------------------------------

    def upload(self, x: int, y: int, width_halfwords: int, height: int,
               raw: bytes, label: str = "") -> None:
        """
        Copy `raw` into VRAM as a rectangle of halfwords.

        `raw` must be exactly width_halfwords * height * 2 bytes; the PS1's
        VRAM DMA has no notion of stride padding.
        """
        expected = width_halfwords * height * 2
        if len(raw) != expected:
            raise FormatError(
                f"upload '{label}': got {len(raw)} bytes, expected {expected} "
                f"for {width_halfwords}x{height} halfwords")
        if x < 0 or y < 0 or x + width_halfwords > WIDTH or y + height > HEIGHT:
            raise FormatError(
                f"upload '{label}': rectangle ({x},{y}) "
                f"{width_halfwords}x{height} does not fit in "
                f"{WIDTH}x{HEIGHT} VRAM")

        block = np.frombuffer(raw, dtype="<u2").reshape(height, width_halfwords)
        self.data[y:y + height, x:x + width_halfwords] = block
        self.written[y:y + height, x:x + width_halfwords] = True
        self.uploads.append((x, y, width_halfwords, height, label))

    def upload_clut(self, clut_x: int, clut_y: int, bpp: int,
                    raw: bytes, label: str = "") -> None:
        """Upload a colour lookup table: one row of 16 or 256 halfwords."""
        entries = CLUT_ENTRIES.get(bpp)
        if entries is None:
            raise FormatError(f"cannot upload CLUT for {bpp}bpp")
        self.upload(clut_x, clut_y, entries, 1, raw, label=label or "clut")

    # -- reading ------------------------------------------------------------

    def clut(self, clut_x: int, clut_y: int, bpp: int) -> np.ndarray:
        """Read a palette out of VRAM as an array of BGR555 halfwords."""
        entries = CLUT_ENTRIES[bpp]
        if clut_x + entries > WIDTH or not 0 <= clut_y < HEIGHT:
            raise FormatError(
                f"CLUT at ({clut_x},{clut_y}) for {bpp}bpp does not fit in VRAM")
        return self.data[clut_y, clut_x:clut_x + entries]

    def indices(self, x: int, y: int, width_px: int, height: int,
                bpp: int) -> np.ndarray:
        """
        Read a tile's palette indices as a (height, width_px) uint8 array.

        4bpp packs two pixels per byte, low nibble first — that ordering is
        what makes tiles come out un-mirrored.
        """
        per_hw = pixels_per_halfword(bpp)
        width_hw = width_px // per_hw
        if width_px % per_hw:
            raise FormatError(
                f"tile width {width_px} is not a whole number of halfwords "
                f"at {bpp}bpp")
        if x + width_hw > WIDTH or y + height > HEIGHT:
            raise FormatError(
                f"tile at ({x},{y}) {width_px}x{height} @{bpp}bpp "
                f"leaves VRAM")

        block = self.data[y:y + height, x:x + width_hw]
        raw = block.astype("<u2").tobytes()
        byte_rows = np.frombuffer(raw, dtype=np.uint8).reshape(height, width_hw * 2)

        if bpp == 8:
            return byte_rows
        # 4bpp: expand each byte into (low nibble, high nibble).
        low = byte_rows & 0x0F
        high = (byte_rows >> 4) & 0x0F
        out = np.empty((height, width_hw * 4), dtype=np.uint8)
        out[:, 0::2] = low
        out[:, 1::2] = high
        return out

    def indices_at_pixels(self, px: int, py: int, width_px: int, height: int,
                          bpp: int) -> np.ndarray:
        """
        Like indices(), but the X origin is given in *pixels* of this bit
        depth rather than halfwords.

        Convenient when working from UV coordinates, which are pixel-based.
        `px` must land on a halfword boundary, since VRAM cannot be addressed
        more finely than that.
        """
        per_hw = pixels_per_halfword(bpp)
        if px % per_hw:
            raise FormatError(
                f"pixel X {px} is not on a halfword boundary at {bpp}bpp "
                f"({per_hw} pixels per halfword)")
        return self.indices(px // per_hw, py, width_px, height, bpp)

    def region_image(self, px: int, py: int, width_px: int, height: int,
                     bpp: int, clut_x: int, clut_y: int) -> Image.Image:
        """Decode an arbitrary pixel-addressed region against a palette."""
        idx = self.indices_at_pixels(px, py, width_px, height, bpp)
        palette = bgr555_array_to_rgba(self.clut(clut_x, clut_y, bpp))
        return Image.fromarray(palette[idx], mode="RGBA")

    def tile_image(self, x: int, y: int, width_px: int, height: int, bpp: int,
                   clut_x: int, clut_y: int) -> Image.Image:
        """Decode a tile plus its palette into an RGBA image."""
        idx = self.indices(x, y, width_px, height, bpp)
        palette = bgr555_array_to_rgba(self.clut(clut_x, clut_y, bpp))
        return Image.fromarray(palette[idx], mode="RGBA")

    def index_image(self, x: int, y: int, width_px: int, height: int,
                    bpp: int) -> Image.Image:
        """
        A tile's palette indices as a greyscale image, no palette applied.

        For tiles whose palette we cannot determine. Indices are stretched to
        the full 0-255 range so the artwork is actually legible; the result is
        a shape reference, not real colour, and is written to a separate
        directory so it can never be mistaken for one.
        """
        idx = self.indices(x, y, width_px, height, bpp)
        scale = 17 if bpp == 4 else 1     # 4bpp: 0-15 -> 0-255
        return Image.fromarray((idx * scale).astype(np.uint8), mode="L")

    # -- diagnostics --------------------------------------------------------

    def coverage(self) -> tuple[int, int]:
        """(halfwords written, total halfwords)."""
        return int(self.written.sum()), WIDTH * HEIGHT
