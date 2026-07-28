"""
Model blocks and the polygon command stream.

A model block is a self-contained mesh: header, vertex array, normal array and
a stream of polygon batches. Sections 5 and up of every LEVEL.DAT hold one
block each. Fields whose meaning is not established are named by their offset
and carried through unchanged.

Block header (0x2C bytes):

    +0x00  u32  unknown_00
    +0x04  u16  unknown_04     always 0
    +0x06  u16  unknown_06
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
    +0x1E  u16  unknown_1e     always 0 on disc; set by the game once it
                               relocates the three offsets below to pointers
    +0x20  u32  vertex_offset  always 0x2C
    +0x24  u32  normal_offset  always vertex_offset + 8 * vertex_count
    +0x28  u32  polygon_offset always normal_offset + 8 * normal_count

The three offsets are redundant with the counts, which makes them useful as a
cross-check on the counts.

Vertices and normals are both 8 bytes: i16 x, i16 y, i16 z, i16 pad. Normals
are 1.12 fixed point, so 4096 is unit length.

The polygon stream is a sequence of batches, each a 4-byte header followed by
`count` fixed-size entries. A header with entry_size == 0 terminates it.

    u16 count
    u8  type
    u8  entry_size

Entry layout is derived from the type byte:

    base  = type & 0x1C      the primitive kind
    flags = type & 0x03      how shading data is supplied

    base & 0x04  ->  quad (4 corners) rather than triangle (3)
    base & 0x08  ->  textured
    base & 0x10  ->  gouraud shaded

Flag bit 0 means the entry carries a face normal index at +0x02; when clear
that slot holds 0xFFFF. On gouraud primitives it also selects per-vertex normal
indices over literal per-corner colours.

    +0x00  u8   base type
    +0x01  u8   attribute byte, meaning unknown, preserved verbatim
    +0x02  u16  face normal index, or 0xFFFF
    +0x04  u32  GPU packet word: 0xCCBBGGRR, colour plus primitive code
           u32  x (n - 1) more colour words, if gouraud with literal colours
    ...    u16  uv_index and u16 clut_id, only if textured
    ...    u16 x n          vertex indices
    ...    u16 x n          per-vertex normal indices, where applicable
    ...    padding to a multiple of 4

clut_id is the PSX CLUT descriptor: clut_x = (id & 0x3F) * 16, clut_y = id >> 6.

Terrain batches set bit 0x20 in the type byte, in which case the low five bits
do not describe the primitive at all; see TYPE_TERRAIN_QUAD.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .binio import FormatError, i16, u8, u16, u32

HEADER_SIZE = 0x2C
VERTEX_SIZE = 8
NORMAL_SIZE = 8
BATCH_HEADER_SIZE = 4

NO_FACE_NORMAL = 0xFFFF

# PSX GP0 polygon command encoding. The command byte the entry carries at
# +0x07 is 0x20 plus these flags, and it is the authoritative description of the
# primitive - more reliable than the batch type byte, as the terrain types below
# demonstrate.
GPU_POLYGON = 0x20
GPU_RAW_TEXTURE = 0x01      # do not modulate the texture by the colour
GPU_SEMI_TRANSPARENT = 0x02
GPU_TEXTURED = 0x04
GPU_QUAD = 0x08
GPU_GOURAUD = 0x10
# Bits that affect blending but not the entry's field layout.
GPU_LAYOUT_IRRELEVANT = GPU_RAW_TEXTURE | GPU_SEMI_TRANSPARENT

# Terrain batches set bit 0x20 in the type byte. When it is set the low five
# bits do NOT describe the primitive: every such entry is a 20-byte textured
# quad with no face normal, whatever the low bits say. The observed terrain
# types 0x22, 0x25, 0x27 and 0x29 all carry GPU code 0x2C or 0x2E, a textured
# quad with the 0x02 being semi-transparency.
TYPE_TERRAIN_QUAD = 0x20


def _primitive_name(corners: int, textured: bool, gouraud: bool) -> str:
    kind = "quad" if corners == 4 else "tri"
    parts = []
    if gouraud:
        parts.append("gouraud")
    parts.append("tex" if textured else "flat")
    parts.append(kind)
    return "_".join(parts)


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

    # Whether +0x02 holds a face normal index rather than the 0xFFFF sentinel.
    # Not derivable from `flags` alone: terrain types put other data there.
    has_face_normal: bool = False
    # Whether +0x02 is required to be 0xFFFF when it is not a face normal.
    # True for ordinary types; false for terrain quads, which store something
    # else in that slot entirely.
    sentinel_enforced: bool = True
    # The GPU command byte this entry must carry, ignoring blend-only bits.
    gpu_code: int = 0

    @property
    def name(self) -> str:
        return (f"{_primitive_name(self.corners, self.textured, self.gouraud)}"
                f"/{self.flags:02X}")


_LAYOUT_CACHE: dict[int, PolyLayout] = {}


def layout_for(type_id: int) -> PolyLayout:
    """
    Derive the entry layout for a polygon type byte.

    This is the single rule the whole decoder rests on; see the module
    docstring for the field ordering.
    """
    cached = _LAYOUT_CACHE.get(type_id)
    if cached is not None:
        return cached

    if type_id & TYPE_TERRAIN_QUAD:
        # Terrain quad: fixed layout, low bits carry something else.
        base = 0x0C
        flags = type_id & 0x1F
        corners = 4
        textured = True
        gouraud = False
        has_face_normal = False
    else:
        base = type_id & 0x1C
        flags = type_id & 0x03
        corners = 4 if (base & 0x04) else 3
        textured = bool(base & 0x08)
        gouraud = bool(base & 0x10)
        # Flag bit 0 adds a face normal index, and on gouraud primitives also
        # selects normal-driven shading over baked per-corner colours.
        has_face_normal = bool(flags & 0x01)

    per_vertex_normals = gouraud and has_face_normal
    colour_count = corners if (gouraud and not has_face_normal) else 1
    gpu_code = (GPU_POLYGON
                | (GPU_QUAD if corners == 4 else 0)
                | (GPU_TEXTURED if textured else 0)
                | (GPU_GOURAUD if gouraud else 0))

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
        has_face_normal=has_face_normal, gpu_code=gpu_code,
        sentinel_enforced=not (type_id & TYPE_TERRAIN_QUAD),
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
    # Raw +0x02 when it is not a face normal. 0xFFFF for ordinary primitives;
    # terrain quads store some other unidentified value here.
    slot_02: int | None = None

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

        PSX quads are two triangles over corners (0,1,2) and (1,3,2) - the
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

    # Bytes left unread after the polygon terminator, when the caller supplied
    # only an upper bound for `size`. Zero for exactly-sized blocks.
    trailing_bytes: int = 0

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
    # The GPU command byte is the authoritative description of the primitive,
    # so it is what we check. The leading byte is only loosely related to the
    # batch type and is not verified: terrain entries carry values that do not
    # match their batch's type byte at all.
    gpu_code = u8(data, offset + 7)
    if gpu_code & ~GPU_LAYOUT_IRRELEVANT != lay.gpu_code:
        raise FormatError(
            f"polygon entry at 0x{offset:X}: GPU code 0x{gpu_code:02X} "
            f"describes a different primitive than 0x{lay.gpu_code:02X} "
            f"({lay.name}) implied by batch type 0x{lay.type_id:02X}")

    attribute = u8(data, offset + 1)

    raw_face_normal = u16(data, offset + 2)
    if lay.has_face_normal:
        face_normal: int | None = raw_face_normal
    else:
        if lay.sentinel_enforced and raw_face_normal != NO_FACE_NORMAL:
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
                   vertices=vertices, normals=normals,
                   slot_02=raw_face_normal if face_normal is None else None)


def parse_model_block(data: bytes, offset: int, size: int,
                      name: str = "model",
                      allow_trailing: bool = False) -> ModelBlock:
    """
    Parse one model block out of `data` at `offset`.

    Raises FormatError on anything that contradicts the documented layout;
    a silent partial parse would be far more expensive than a crash here.

    `size` is normally the block's exact length, and the polygon stream is
    required to end precisely there - a strong check that everything before it
    was read correctly. Set `allow_trailing` when the caller only knows an
    upper bound, as with the last model in a terrain chunk, which is followed
    by a few bytes of padding. The unread remainder is recorded in
    `ModelBlock.trailing_bytes` instead of raising.
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
        if not allow_trailing:
            raise FormatError(
                f"{name}: polygon stream ended at 0x{cursor:X} but the block "
                f"ends at 0x{size:X} (delta {size - cursor})")
        model.trailing_bytes = size - cursor

    return model
