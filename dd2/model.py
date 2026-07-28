"""
model.py — model blocks and the polygon command stream.

A model block is a self-contained mesh: header, vertex array, normal array,
and a stream of polygon batches. Sections 5 and up of every LEVEL.DAT hold
one block each — props, cars, wheels and their LOD variants.

Everything below was derived from the data and then checked against all 197
model blocks in the retail files. Where a field's meaning is not established,
it is named by its offset and carried through unchanged rather than guessed
at; see UNKNOWN FIELDS at the end of this docstring.


Block header (0x2C bytes)
-------------------------
    +0x00  u32  unknown_00
    +0x04  u16  unknown_04     always 0
    +0x06  u16  unknown_06     equals unknown_00 in 170 of 197 blocks
    +0x08  u16  unknown_08
    +0x0A  u16  vertex_count
    +0x0C  u16  normal_count
    +0x0E  u16  polygon_count
    +0x10  u16  triangle_count
    +0x12  u16  quad_count
    +0x14  u32  unknown_14     always 0
    +0x18  u16  unknown_18
    +0x1A  u16  unknown_1a
    +0x1C  u16  unknown_1c
    +0x1E  u16  unknown_1e     always 0 on disc; the game sets it after it
                               relocates the three offsets below to pointers
    +0x20  u32  vertex_offset  always 0x2C
    +0x24  u32  normal_offset  always vertex_offset + 8 * vertex_count
    +0x28  u32  polygon_offset always normal_offset + 8 * normal_count

The three offsets are redundant with the counts, which is exactly what makes
them useful: they cross-check our reading of the counts. All 197 blocks agree.

Vertices and normals are both 8 bytes: i16 x, i16 y, i16 z, i16 pad.


Polygon command stream
----------------------
A sequence of batches. Each batch is a 4-byte header followed by `count`
fixed-size entries. A batch header with entry_size == 0 terminates the stream.

    u16 count
    u8  type
    u8  entry_size

Summing `count` over all batches reproduces the header's polygon_count for
every block, and the stream always ends exactly at the end of the block.


Polygon entry layout
--------------------
The type byte splits cleanly into two parts:

    base  = type & 0x1C      the primitive kind
    flags = type & 0x03      how shading data is supplied

`base` maps one-to-one onto the PSX GPU primitive codes, and the entry stores
that code at +0x07 — a partially pre-built GP0 packet. Verified on every
entry:

    base  primitive                       GPU code
    0x00  flat triangle                     0x20
    0x04  flat quad                         0x28
    0x08  textured triangle                 0x24
    0x0C  textured quad                     0x2C
    0x10  gouraud triangle                  0x30
    0x14  gouraud quad                      0x38
    0x18  gouraud textured triangle         0x34
    0x1C  gouraud textured quad             0x3C

    base & 0x04  ->  quad (4 corners) rather than triangle (3)
    base & 0x08  ->  textured
    base & 0x10  ->  gouraud shaded

`flags` bit 0 means the entry carries a face normal index at +0x02; when
clear that slot holds 0xFFFF. For gouraud primitives bit 0 additionally
selects how per-corner shading arrives: set means per-vertex NORMAL indices
(shading computed at runtime), clear means literal per-corner COLOURS baked
into the entry. Bit 1's meaning is not yet established.

That yields one layout rule covering all 24 observed type values:

    +0x00  u8   base type
    +0x01  u8   attribute byte (semantics unknown, preserved verbatim)
    +0x02  u16  face normal index, or 0xFFFF
    +0x04  u32  GPU packet word: 0xCCBBGGRR (colour + primitive code)
           u32  x (n - 1) more colour words, if gouraud with literal colours
    ...    u16  uv_index    | only if textured
           u16  clut_id     |
    ...    u16 x n          vertex indices
    ...    u16 x n          per-vertex normal indices, if gouraud with normals
    ...    padding to a multiple of 4

The predicted entry size from this rule matches the batch header's
entry_size for all 24 type values, and every index it locates is in range:
6532 of 6532 polygons pass. 210 entries carry junk in the 2-byte tail pad
(ASCII fragments left over from the build tool), so padding is not asserted
to be zero.

`clut_id` is the PSX CLUT descriptor: clut_x = (id & 0x3F) * 16,
clut_y = id >> 6.


UNKNOWN FIELDS
--------------
Not yet established, carried through unchanged so nothing is silently lost:
  - header unknown_00 / _06 / _08 / _18 / _1a / _1c
  - polygon entry attribute byte at +0x01
  - meaning of type flag bit 1 (0x02)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .binio import FormatError, i16, u8, u16, u32

HEADER_SIZE = 0x2C
VERTEX_SIZE = 8
NORMAL_SIZE = 8
BATCH_HEADER_SIZE = 4

NO_FACE_NORMAL = 0xFFFF

# Primitive base -> the GPU code the entry must carry at +0x07.
GPU_CODE = {
    0x00: 0x20, 0x04: 0x28, 0x08: 0x24, 0x0C: 0x2C,
    0x10: 0x30, 0x14: 0x38, 0x18: 0x34, 0x1C: 0x3C,
}

BASE_NAME = {
    0x00: "flat_tri", 0x04: "flat_quad",
    0x08: "tex_tri", 0x0C: "tex_quad",
    0x10: "gouraud_tri", 0x14: "gouraud_quad",
    0x18: "gouraud_tex_tri", 0x1C: "gouraud_tex_quad",
}


# ---------------------------------------------------------------------------
# Type decoding
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PolyLayout:
    """Where each field sits inside a polygon entry of a given type."""

    type_id: int
    base: int
    flags: int
    corners: int              # 3 or 4
    textured: bool
    gouraud: bool
    per_vertex_normals: bool
    colour_count: int
    colour_offset: int
    uv_offset: int | None     # u16 uv index; clut id follows at +2
    vertex_offset: int
    normal_offset: int | None
    entry_size: int

    @property
    def name(self) -> str:
        return f"{BASE_NAME.get(self.base, f'base{self.base:#04x}')}" \
               f"/{self.flags}"


_LAYOUT_CACHE: dict[int, PolyLayout] = {}


def layout_for(type_id: int) -> PolyLayout:
    """
    Derive the entry layout for a polygon type byte.

    This is the single rule the whole decoder rests on; see the module
    docstring for how it was established and verified.
    """
    cached = _LAYOUT_CACHE.get(type_id)
    if cached is not None:
        return cached

    base = type_id & 0x1C
    flags = type_id & 0x03
    if base not in GPU_CODE:
        raise FormatError(f"unknown polygon primitive base 0x{base:02X} "
                          f"(from type 0x{type_id:02X})")

    corners = 4 if (base & 0x04) else 3
    textured = bool(base & 0x08)
    gouraud = bool(base & 0x10)
    # Flag bit 0 selects normal-driven shading over baked per-corner colours.
    per_vertex_normals = gouraud and bool(flags & 0x01)
    colour_count = corners if (gouraud and not (flags & 0x01)) else 1

    cursor = 4                       # past type, attribute and face normal
    colour_offset = cursor
    cursor += 4 * colour_count

    uv_offset = None
    if textured:
        uv_offset = cursor
        cursor += 4                  # u16 uv index + u16 clut id

    vertex_offset = cursor
    cursor += 2 * corners

    normal_offset = None
    if per_vertex_normals:
        normal_offset = cursor
        cursor += 2 * corners

    entry_size = (cursor + 3) & ~3   # entries are 4-byte aligned

    layout = PolyLayout(
        type_id=type_id, base=base, flags=flags, corners=corners,
        textured=textured, gouraud=gouraud,
        per_vertex_normals=per_vertex_normals,
        colour_count=colour_count, colour_offset=colour_offset,
        uv_offset=uv_offset, vertex_offset=vertex_offset,
        normal_offset=normal_offset, entry_size=entry_size,
    )
    _LAYOUT_CACHE[type_id] = layout
    return layout


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Polygon:
    """One decoded polygon."""

    layout: PolyLayout
    attribute: int                     # entry byte +0x01, meaning unknown
    face_normal: int | None            # index into the block's normal array
    colours: tuple[tuple[int, int, int], ...]   # one, or one per corner
    uv_index: int | None               # index into the level's UV table
    clut_id: int | None                # PSX CLUT descriptor
    vertices: tuple[int, ...]          # indices into the block's vertex array
    normals: tuple[int, ...]           # per-corner normal indices, may be empty

    @property
    def corners(self) -> int:
        return self.layout.corners

    @property
    def textured(self) -> bool:
        return self.layout.textured

    @property
    def clut_xy(self) -> tuple[int, int] | None:
        """Decode clut_id into VRAM (x in halfwords, y in scanlines)."""
        if self.clut_id is None:
            return None
        return ((self.clut_id & 0x3F) * 16, self.clut_id >> 6)

    def triangles(self) -> tuple[tuple[int, int, int], ...]:
        """
        Triangulate into vertex-index triples.

        PSX quads are two triangles over corners (0,1,2) and (1,3,2) — the
        corner order is a zig-zag strip, not a fan, so this winding is what
        keeps quads planar and consistently oriented.
        """
        v = self.vertices
        if len(v) == 3:
            return ((v[0], v[1], v[2]),)
        return ((v[0], v[1], v[2]), (v[1], v[3], v[2]))


@dataclass(frozen=True)
class Vec3:
    """A vertex or normal. `pad` is retained so we can assert on it."""

    x: int
    y: int
    z: int
    pad: int

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.x, self.y, self.z)


@dataclass
class ModelBlock:
    """A parsed model block."""

    name: str
    size: int

    # Header fields whose meaning is established.
    vertex_count: int
    normal_count: int
    polygon_count: int
    triangle_count: int
    quad_count: int
    vertex_offset: int
    normal_offset: int
    polygon_offset: int

    # Header fields whose meaning is not.
    unknown: dict[str, int] = field(default_factory=dict)

    vertices: list[Vec3] = field(default_factory=list)
    normals: list[Vec3] = field(default_factory=list)
    polygons: list[Polygon] = field(default_factory=list)

    # Which (type, entry_size) batches were seen, for reporting.
    batch_types: dict[int, int] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.vertices or not self.polygons

    @property
    def textured_polygon_count(self) -> int:
        return sum(1 for p in self.polygons if p.textured)

    def bounds(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Axis-aligned bounds in raw model units."""
        if not self.vertices:
            return ((0, 0, 0), (0, 0, 0))
        xs = [v.x for v in self.vertices]
        ys = [v.y for v in self.vertices]
        zs = [v.z for v in self.vertices]
        return ((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))

    def uv_indices(self) -> set[int]:
        return {p.uv_index for p in self.polygons if p.uv_index is not None}

    def clut_ids(self) -> set[int]:
        return {p.clut_id for p in self.polygons if p.clut_id is not None}

    # -- validation ---------------------------------------------------------

    def validate(self, uv_table_size: int | None = None) -> list[str]:
        """
        Cross-check the block against its own header and, if given, against
        the level's UV table size.
        """
        problems: list[str] = []

        if self.vertex_offset != HEADER_SIZE:
            problems.append(
                f"{self.name}: vertex_offset is 0x{self.vertex_offset:X}, "
                f"expected 0x{HEADER_SIZE:X}")

        expected_normal = self.vertex_offset + VERTEX_SIZE * self.vertex_count
        if self.normal_offset != expected_normal:
            problems.append(
                f"{self.name}: normal_offset 0x{self.normal_offset:X} != "
                f"vertex_offset + 8*vertex_count (0x{expected_normal:X})")

        expected_poly = self.normal_offset + NORMAL_SIZE * self.normal_count
        if self.polygon_offset != expected_poly:
            problems.append(
                f"{self.name}: polygon_offset 0x{self.polygon_offset:X} != "
                f"normal_offset + 8*normal_count (0x{expected_poly:X})")

        if len(self.polygons) != self.polygon_count:
            problems.append(
                f"{self.name}: decoded {len(self.polygons)} polygons but "
                f"header says {self.polygon_count}")

        tris = sum(1 for p in self.polygons if p.corners == 3)
        quads = sum(1 for p in self.polygons if p.corners == 4)
        if tris != self.triangle_count:
            problems.append(
                f"{self.name}: {tris} triangles decoded but header says "
                f"{self.triangle_count}")
        if quads != self.quad_count:
            problems.append(
                f"{self.name}: {quads} quads decoded but header says "
                f"{self.quad_count}")

        for i, poly in enumerate(self.polygons):
            for vi in poly.vertices:
                if vi >= self.vertex_count:
                    problems.append(
                        f"{self.name}: polygon {i} references vertex {vi} "
                        f"of {self.vertex_count}")
            for ni in poly.normals:
                if ni >= self.normal_count:
                    problems.append(
                        f"{self.name}: polygon {i} references normal {ni} "
                        f"of {self.normal_count}")
            if poly.face_normal is not None and \
                    poly.face_normal >= self.normal_count:
                problems.append(
                    f"{self.name}: polygon {i} face normal {poly.face_normal} "
                    f"of {self.normal_count}")
            if uv_table_size is not None and poly.uv_index is not None \
                    and poly.uv_index >= uv_table_size:
                problems.append(
                    f"{self.name}: polygon {i} uv index {poly.uv_index} "
                    f"of {uv_table_size}")

        return problems


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_vec3_array(data: bytes, base: int, count: int,
                      size: int) -> list[Vec3]:
    out: list[Vec3] = []
    for i in range(count):
        o = base + i * size
        out.append(Vec3(i16(data, o), i16(data, o + 2),
                        i16(data, o + 4), i16(data, o + 6)))
    return out


def _parse_polygon(data: bytes, offset: int, lay: PolyLayout) -> Polygon:
    base_byte = u8(data, offset)
    if base_byte != lay.base:
        raise FormatError(
            f"polygon entry at 0x{offset:X}: leading byte 0x{base_byte:02X} "
            f"does not match batch primitive base 0x{lay.base:02X}")

    gpu_code = u8(data, offset + 7)
    if gpu_code != GPU_CODE[lay.base]:
        raise FormatError(
            f"polygon entry at 0x{offset:X}: GPU code 0x{gpu_code:02X}, "
            f"expected 0x{GPU_CODE[lay.base]:02X} for {lay.name}")

    attribute = u8(data, offset + 1)

    raw_face_normal = u16(data, offset + 2)
    if lay.flags & 0x01:
        face_normal: int | None = raw_face_normal
    else:
        if raw_face_normal != NO_FACE_NORMAL:
            raise FormatError(
                f"polygon entry at 0x{offset:X}: type 0x{lay.type_id:02X} "
                f"carries no face normal, but the slot holds "
                f"0x{raw_face_normal:04X} instead of 0xFFFF")
        face_normal = None

    colours = tuple(
        (u8(data, offset + lay.colour_offset + 4 * i),        # R
         u8(data, offset + lay.colour_offset + 4 * i + 1),    # G
         u8(data, offset + lay.colour_offset + 4 * i + 2))    # B
        for i in range(lay.colour_count)
    )

    uv_index = clut_id = None
    if lay.uv_offset is not None:
        uv_index = u16(data, offset + lay.uv_offset)
        clut_id = u16(data, offset + lay.uv_offset + 2)

    vertices = tuple(u16(data, offset + lay.vertex_offset + 2 * i)
                     for i in range(lay.corners))

    normals: tuple[int, ...] = ()
    if lay.normal_offset is not None:
        normals = tuple(u16(data, offset + lay.normal_offset + 2 * i)
                        for i in range(lay.corners))

    return Polygon(layout=lay, attribute=attribute, face_normal=face_normal,
                   colours=colours, uv_index=uv_index, clut_id=clut_id,
                   vertices=vertices, normals=normals)


def parse_model_block(data: bytes, offset: int, size: int,
                      name: str = "model") -> ModelBlock:
    """
    Parse one model block out of `data` at `offset`.

    Raises FormatError on anything that contradicts the documented layout;
    a silent partial parse would be far more expensive than a crash here.
    """
    if size < HEADER_SIZE:
        raise FormatError(
            f"{name}: block is {size} bytes, smaller than the "
            f"{HEADER_SIZE}-byte header")

    block = data[offset:offset + size]

    header_unknown = {
        "unknown_00": u32(block, 0x00),
        "unknown_04": u16(block, 0x04),
        "unknown_06": u16(block, 0x06),
        "unknown_08": u16(block, 0x08),
        "unknown_14": u32(block, 0x14),
        "unknown_18": u16(block, 0x18),
        "unknown_1a": u16(block, 0x1A),
        "unknown_1c": u16(block, 0x1C),
        "unknown_1e": u16(block, 0x1E),
    }

    model = ModelBlock(
        name=name,
        size=size,
        vertex_count=u16(block, 0x0A),
        normal_count=u16(block, 0x0C),
        polygon_count=u16(block, 0x0E),
        triangle_count=u16(block, 0x10),
        quad_count=u16(block, 0x12),
        vertex_offset=u32(block, 0x20),
        normal_offset=u32(block, 0x24),
        polygon_offset=u32(block, 0x28),
        unknown=header_unknown,
    )

    # The offsets are redundant with the counts; trust but verify (validate()
    # reports the mismatch). Read using the stored offsets so a disagreement
    # shows up as bad geometry rather than being papered over.
    model.vertices = _parse_vec3_array(block, model.vertex_offset,
                                       model.vertex_count, VERTEX_SIZE)
    model.normals = _parse_vec3_array(block, model.normal_offset,
                                      model.normal_count, NORMAL_SIZE)

    # -- polygon command stream --
    cursor = model.polygon_offset
    while True:
        if cursor + BATCH_HEADER_SIZE > size:
            raise FormatError(
                f"{name}: polygon stream ran past the end of the block "
                f"looking for a batch header at 0x{cursor:X}")

        count = u16(block, cursor)
        type_id = u8(block, cursor + 2)
        entry_size = u8(block, cursor + 3)
        cursor += BATCH_HEADER_SIZE

        if entry_size == 0:
            break                      # terminator

        lay = layout_for(type_id)
        if entry_size != lay.entry_size:
            raise FormatError(
                f"{name}: batch of type 0x{type_id:02X} declares entry size "
                f"{entry_size}, but the layout rule predicts {lay.entry_size}")

        if cursor + count * entry_size > size:
            raise FormatError(
                f"{name}: batch of {count} x {entry_size} bytes at "
                f"0x{cursor:X} runs past the end of the block")

        model.batch_types[type_id] = model.batch_types.get(type_id, 0) + count
        for i in range(count):
            model.polygons.append(
                _parse_polygon(block, cursor + i * entry_size, lay))
        cursor += count * entry_size

    if cursor != size:
        raise FormatError(
            f"{name}: polygon stream ended at 0x{cursor:X} but the block "
            f"ends at 0x{size:X} (delta {size - cursor})")

    return model
