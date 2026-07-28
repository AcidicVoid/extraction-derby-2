"""
carmodel.py — assemble a textured car mesh, in any of the 20 liveries.

Where the car lives
-------------------
The in-race car is a model block in the *track* files, not LEV0. In LEV1 two
blocks reference the shared car body tiles and nothing else does: section 17
(107 vertices, 147 polygons) and section 16 (102 / 108). They share **zero
vertices**, so they are independently authored meshes rather than one
decimated from the other.

Section 17 is the one we export. Two reasons: it is near-symmetric about X
(spanning -184..183) whereas section 16 is offset (-172..195, with parts
centred on x=11), and it carries the detail — 8 roof polygons and 16 door
polygons against section 16's 1 and 4.

Section 17 is a complete car including wheels. Grouping its polygons by the
texture they sample lays the whole thing out:

    FRWN88A   z +190..417   front wheel arches
    FRNT88A   z +441..464   front panel, frontmost
    BON88A    z +245..402   bonnet
    WINFRN88  z +123..135   windscreen
    DR88A     z  -73..122   doors, at x +-184
    WINSID88  y  +65..95    side windows
    ROOF88A   y +118..124   roof, highest
    WINBCK88  z    -248     rear window
    BKWN88A   z -407..-133  rear wheel arches
    BOOT88A   z -424..-343  boot
    BUMP88A   z -441..-427  rear bumper, rearmost
    unnamed   y  -97..-86   16 polygons at the lowest point: the wheels
    (no texture)            32 flat-shaded polygons: underside and interior

That resolves the open question about the wheels — they are part of the car
mesh, textured from a region of VRAM the name table does not name.

How liveries work
-----------------
Every polygon references exactly its own tile's CLUT; there is no sharing to
untangle. So car 88 renders with no substitution at all — it *is* the shared
tile set.

For any other car, the pixels stay the same and only the palette changes. The
part-to-palette mapping below was taken from the CLUT naming (CLUTnnA..E) and
confirmed by rendering: bit depths line up (CLUTnnA is 8bpp like the body
tiles, B..E are 4bpp like the rest) and the resulting cars are coherent.

    body       BUMP FRNT FRWN BKWN   8bpp  ->  CLUTnnA
    bonnet     BON ROOF               4bpp  ->  CLUTnnB
    boot       BOOT                   4bpp  ->  CLUTnnC
    windows    WINFRN WINBCK          4bpp  ->  CLUTnnD
    side glass WINSID                 4bpp  ->  CLUTnnE

The door number panel is the exception: it is a different *tile* per car, not
just a different palette, because the number itself is painted into the
pixels. DRnnA has identical dimensions to DR88A, so the polygon's UVs are
translated by the difference between the two tiles' VRAM positions and decoded
with DRnnA's own palette.

The wheels keep their original palette for every car — they are not bodywork.

Wheels
------
The body mesh has no wheels; they are a separate model attached four times.
Section 18 is the wheel: two textured hubcap faces at x = +-33 plus eight flat
quads forming an octagonal tyre tread, all inside a 66 x 134 x 134 box.

Sections 5 and 6 are simplified alternatives — six black quads with a single
visible side face, at x = +33 and x = -33 respectively, so a right/left pair.
`FUN_8002b874` binds exactly those two models to four wheel slots
(ptr[5], ptr[6], ptr[5], ptr[6]), but they carry no hubcap texture and only one
side face each, so section 18 is what we export.

Section 18 contains **coincident duplicate faces**: polygons 0 and 1 are the
same two hubcap quads as polygons 10 and 11, with the same UV index, but
coloured (0,0,0) instead of neutral (128,128,128). The game selects between
them at draw time; exported together they z-fight and the black pair wins,
turning every wheel into a black disc. `_deduplicate` drops the darker of any
two polygons sharing a vertex set and UV index.

Wheel placement is not stored anywhere we can read — the game computes it from
live suspension state (`DAT_80091028 + car*0x288 + wheel*0x84`). The positions
below are therefore *derived from the wheel arch openings*: grouping body
polygons by sampled tile gives FRWN88A spanning x 134..184, z 174..439 and
BKWN88A spanning x 141..186, z -442..-118, symmetric about X. Centre those and
you get the mounting points. Height is the one free parameter, chosen so the
tyre sits in the arch without clipping the sill; it is a constant here and easy
to adjust, and each wheel is a separate named node so it can be moved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image

from .binio import FormatError
from .glb import (COLOUR_NEUTRAL, Object3D, Primitive3D, to_gltf_direction,
                  to_gltf_position)
from .level import LevelFile
from .model import ModelBlock, Polygon
from .texnames import TexName, TextureNameTable
from .textures import LevelTextures
from .uvtable import UVRecord
from .vram import pixels_per_halfword

# Body part prefix -> the CLUT letter that recolours it.
#
# Derived, not assumed. Six 4bpp tiles have to share four per-car palettes
# (B..E), so the grouping matters and guessing it produces cars that look
# almost right. The test: entries in a palette that are car-*independent*
# (glass, the driver, chrome) must be byte-identical to the corresponding
# entries of car 88's own tile palette, since car 88 is itself one of the 20.
# Counting, for every tile and letter, how many of the 16 entries agree with
# car 88 across all 19 other cars gives an unambiguous answer:
#
#     tile          B    C    D    E
#     BON88A        4    0    1    0
#     ROOF88A       4    0    1    0
#     BOOT88A       4    0    1    0
#     WINFRN88      1    0   13    0
#     WINBCK88      0   13    0    0
#     WINSID88      0    0    0    8
#
# So B is the shared painted-panel ramp for all three opening panels, and C, D
# and E are one window type each. Off-diagonal agreement is 0 or 1 throughout.
#
# This corrects two earlier assumptions carried over from prior sessions,
# which had BOOT on C and both WINFRN and WINBCK on D.
PART_CLUT_LETTER = {
    "BUMP": "A", "FRNT": "A", "FRWN": "A", "BKWN": "A",
    "BON": "B", "ROOF": "B", "BOOT": "B",
    "WINBCK": "C",
    "WINFRN": "D",
    "WINSID": "E",
}

# The car whose pixels every livery shares.
BASE_CAR = "88"

# Normals are stored as 1.12 fixed point: magnitude 4096 == unit length.
NORMAL_SCALE = 1.0 / 4096.0

# Default model-unit to metre conversion. The car is 905 units long, so 1/256
# yields about 3.5 m. A power of two matching PSX fixed-point practice; the
# true figure is not established, hence the CLI override.
DEFAULT_SCALE = 1.0 / 256.0

# Model block holding the car body, and the one holding a single wheel.
BODY_SECTION = 17
WHEEL_SECTION = 18

# Wheel mounting points in model space, derived from the arch openings; see the
# module docstring. Height is the free parameter.
WHEEL_CENTRE_Y = -34
WHEEL_POSITIONS = {
    "wheel_front_left": (-159, WHEEL_CENTRE_Y, 306),
    "wheel_front_right": (159, WHEEL_CENTRE_Y, 306),
    "wheel_rear_left": (-163, WHEEL_CENTRE_Y, -280),
    "wheel_rear_right": (163, WHEEL_CENTRE_Y, -280),
}


@dataclass(frozen=True)
class Livery:
    """Everything that distinguishes one car's appearance from another's."""

    car_number: str
    # Part prefix -> the CLUT record recolouring it. Empty for the base car.
    part_clut: dict[str, TexName] = field(default_factory=dict)
    # Replacement door tile, and its offset from DR88A in VRAM pixels.
    door_tile: TexName | None = None
    door_offset: tuple[int, int] = (0, 0)

    @property
    def is_base(self) -> bool:
        return self.car_number == BASE_CAR


def build_livery(names: TextureNameTable, car_number: str,
                 base_door: str = f"DR{BASE_CAR}A") -> Livery:
    """
    Gather the palette substitutions for one car.

    Raises FormatError if a car is missing palettes we expect, rather than
    quietly emitting a car that looks like #88.
    """
    if car_number == BASE_CAR:
        return Livery(car_number=car_number)

    part_clut: dict[str, TexName] = {}
    missing: list[str] = []
    for part, letter in PART_CLUT_LETTER.items():
        rec = names.get(f"CLUT{car_number}{letter}")
        if rec is None:
            missing.append(f"CLUT{car_number}{letter}")
            continue
        part_clut[part] = rec
    if missing:
        raise FormatError(
            f"car {car_number}: missing palette record(s) "
            f"{', '.join(sorted(set(missing)))}")

    door = names.get(f"DR{car_number}A")
    base = names.get(base_door)
    if door is None or base is None:
        raise FormatError(
            f"car {car_number}: need both DR{car_number}A and {base_door} "
            f"to place the door number panel")
    if (door.width, door.height, door.bpp) != (base.width, base.height,
                                               base.bpp):
        raise FormatError(
            f"car {car_number}: DR{car_number}A is "
            f"{door.width}x{door.height}@{door.bpp}bpp but {base_door} is "
            f"{base.width}x{base.height}@{base.bpp}bpp; cannot substitute by "
            f"translation")

    per_hw = pixels_per_halfword(door.bpp)
    offset = ((door.vram_x - base.vram_x) * per_hw,
              door.vram_y - base.vram_y)
    return Livery(car_number=car_number, part_clut=part_clut,
                  door_tile=door, door_offset=offset)


# ---------------------------------------------------------------------------
# Polygon -> texture source resolution
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TextureSource:
    """Where a polygon's texels come from, after livery substitution."""

    bpp: int
    clut_x: int
    clut_y: int
    offset: tuple[int, int]     # added to absolute pixel coords
    label: str                  # for the material name


def _tile_at_uv_centre(textures: LevelTextures, uv: UVRecord,
                       corners: int) -> TexName | None:
    """
    The named tile a polygon samples, identified from the centre of its UV
    footprint.

    Sampling the centre rather than the corners matters: a polygon that covers
    a whole tile has its corner UVs sitting exactly on the tile's edges, where
    they can land in a neighbouring tile and produce an ambiguous answer.
    """
    if textures.names is None:
        return None
    pts = uv.uvs[:corners]
    mid_u = sum(p[0] for p in pts) // len(pts)
    mid_v = sum(p[1] for p in pts) // len(pts)
    per_hw = pixels_per_halfword(uv.tpage.bpp)
    return textures.names.tile_at(uv.tpage.x_base + mid_u // per_hw,
                                  uv.tpage.y_base + mid_v)


def resolve_source(textures: LevelTextures, poly: Polygon, uv: UVRecord,
                   livery: Livery) -> TextureSource:
    """Apply the livery to one polygon's texture reference."""
    clut = poly.clut_xy
    if clut is None:
        raise FormatError("resolve_source called on an untextured polygon")

    rec = _tile_at_uv_centre(textures, uv, poly.corners)
    if rec is None or livery.is_base:
        # Unnamed VRAM (the wheels) or the base car: nothing to substitute.
        label = rec.name if rec is not None else "unnamed"
        return TextureSource(uv.tpage.bpp, clut[0], clut[1], (0, 0), label)

    part = rec.part

    # Door number panel: a different tile, not merely a different palette.
    if part == "DR" and livery.door_tile is not None:
        door = livery.door_tile
        return TextureSource(uv.tpage.bpp, door.clut_x, door.clut_y,
                             livery.door_offset, door.name)

    replacement = livery.part_clut.get(part) if part else None
    if replacement is None:
        # Not bodywork (wheels, decals): keep the original palette.
        return TextureSource(uv.tpage.bpp, clut[0], clut[1], (0, 0), rec.name)

    return TextureSource(uv.tpage.bpp, replacement.clut_x, replacement.clut_y,
                         (0, 0), f"{rec.name}+{replacement.name}")


# ---------------------------------------------------------------------------
# Mesh assembly
# ---------------------------------------------------------------------------

@dataclass
class _Group:
    """Polygons that will share one material, plus their absolute UV extent."""

    source: TextureSource
    polygons: list[tuple[Polygon, UVRecord]] = field(default_factory=list)
    min_x: int = 1 << 30
    min_y: int = 1 << 30
    max_x: int = -(1 << 30)
    max_y: int = -(1 << 30)

    def note(self, x: int, y: int) -> None:
        self.min_x = min(self.min_x, x)
        self.max_x = max(self.max_x, x)
        self.min_y = min(self.min_y, y)
        self.max_y = max(self.max_y, y)


def _absolute_uv(uv: UVRecord, u: int, v: int,
                 offset: tuple[int, int]) -> tuple[int, int]:
    """
    Page-local UV -> absolute VRAM pixel coordinates.

    tpage.x_base is in halfwords while u is in pixels, so the base is scaled
    by the page's pixels-per-halfword before adding.
    """
    per_hw = pixels_per_halfword(uv.tpage.bpp)
    return (uv.tpage.x_base * per_hw + u + offset[0],
            uv.tpage.y_base + v + offset[1])


def _colour(rgb: tuple[int, int, int]) -> tuple[float, float, float, float]:
    """PSX primitive colour -> glTF COLOR_0, with 0x80 mapping to 1.0."""
    return (min(rgb[0] / COLOUR_NEUTRAL, 1.0),
            min(rgb[1] / COLOUR_NEUTRAL, 1.0),
            min(rgb[2] / COLOUR_NEUTRAL, 1.0),
            1.0)


def _deduplicate(polygons: list[Polygon]) -> list[Polygon]:
    """
    Remove coincident duplicate faces, keeping the brighter one.

    The wheel (section 18) stores its two hubcap quads twice: once coloured
    (0,0,0) and once neutral (128,128,128), with identical vertices and UVs.
    The game picks one at draw time. Exported together they occupy the same
    plane and the black copy wins the depth test, turning the wheel into a
    featureless black disc.

    Two polygons are considered the same face when they use the same set of
    vertex indices and the same UV record. Among duplicates we keep the
    brightest, which is the one that leaves the texture visible.
    """
    best: dict[tuple, Polygon] = {}
    order: list[tuple] = []

    for poly in polygons:
        key = (frozenset(poly.vertices), poly.uv_index)
        incumbent = best.get(key)
        if incumbent is None:
            best[key] = poly
            order.append(key)
            continue
        if sum(poly.colours[0]) > sum(incumbent.colours[0]):
            best[key] = poly

    return [best[key] for key in order]


def build_car(level: LevelFile, textures: LevelTextures, livery: Livery,
              scale: float = DEFAULT_SCALE,
              body_section: int = BODY_SECTION,
              wheel_section: int = WHEEL_SECTION,
              with_wheels: bool = True) -> list[Object3D]:
    """
    Build a complete car: body plus four positioned wheels.

    Returns a list of objects so each ends up as its own named node in the
    GLB — the wheels stay separately movable, which matters because their
    height is derived rather than read from the game.
    """
    body = build_car_object(level, textures, body_section, livery,
                            scale=scale, name="body")
    objects = [body]
    if not with_wheels:
        return objects

    for node_name, (px, py, pz) in WHEEL_POSITIONS.items():
        wheel = build_car_object(level, textures, wheel_section, livery,
                                 scale=scale, name=node_name)
        # Offset in glTF space, matching to_gltf_position's 180-degree turn
        # about Y: x and z are negated, y is not.
        dx, dy, dz = -px * scale, py * scale, -pz * scale
        for prim in wheel.primitives:
            prim.positions = [(x + dx, y + dy, z + dz)
                              for x, y, z in prim.positions]
        objects.append(wheel)

    return objects


def build_car_object(level: LevelFile, textures: LevelTextures,
                     section: int, livery: Livery,
                     scale: float = DEFAULT_SCALE,
                     name: str | None = None) -> Object3D:
    """
    Turn one model block into a textured Object3D in the given livery.

    Polygons are grouped by resolved texture source; each group becomes one
    primitive with its own cropped, palette-applied image. Cropping to the
    group's actual UV extent keeps the embedded textures small and avoids
    baking in unrelated parts of VRAM.
    """
    model: ModelBlock = level.model(section)
    obj = Object3D(name=name or f"car_{livery.car_number}")

    groups: dict[tuple, _Group] = {}
    untextured: list[Polygon] = []

    for poly in _deduplicate(model.polygons):
        if poly.uv_index is None:
            untextured.append(poly)
            continue
        uv = level.uv_table[poly.uv_index]
        source = resolve_source(textures, poly, uv, livery)
        key = (source.bpp, source.clut_x, source.clut_y, source.offset)
        group = groups.get(key)
        if group is None:
            group = groups[key] = _Group(source=source)
        group.polygons.append((poly, uv))
        for u, v in uv.uvs[:poly.corners]:
            group.note(*_absolute_uv(uv, u, v, source.offset))

    for group in groups.values():
        obj.primitives.append(_build_textured_primitive(model, textures,
                                                        group, scale))
    if untextured:
        obj.primitives.append(_build_untextured_primitive(model, untextured,
                                                          scale))
    return obj


def _emit_corner(prim: Primitive3D, model: ModelBlock, poly: Polygon,
                 corner: int, scale: float,
                 uv_value: tuple[float, float] | None,
                 cache: dict) -> int:
    """
    Append one polygon corner as a vertex, reusing an identical earlier one.

    Vertices cannot simply be shared by index: UV and colour vary per corner
    per polygon, so the same position may need several distinct vertices. The
    cache key covers everything that can differ.
    """
    vertex_index = poly.vertices[corner]
    vertex = model.vertices[vertex_index]

    if poly.normals:
        normal_index = poly.normals[corner]
    elif poly.face_normal is not None:
        normal_index = poly.face_normal
    else:
        normal_index = None

    colour_index = corner if len(poly.colours) > 1 else 0
    colour = _colour(poly.colours[colour_index])

    key = (vertex_index, normal_index, colour, uv_value)
    cached = cache.get(key)
    if cached is not None:
        return cached

    prim.positions.append(to_gltf_position(vertex.x, vertex.y, vertex.z,
                                           scale))
    if normal_index is not None and normal_index < len(model.normals):
        n = model.normals[normal_index]
        prim.normals.append(to_gltf_direction(n.x * NORMAL_SCALE,
                                              n.y * NORMAL_SCALE,
                                              n.z * NORMAL_SCALE))
    prim.colours.append(colour)
    if uv_value is not None:
        prim.uvs.append(uv_value)

    index = len(prim.positions) - 1
    cache[key] = index
    return index


def _finish(prim: Primitive3D) -> Primitive3D:
    """
    Drop the NORMAL attribute unless every vertex got one.

    glTF requires attributes to be complete; a partially filled normal array
    would be silently misaligned with the positions.
    """
    if len(prim.normals) != len(prim.positions):
        prim.normals = []
    return prim


def _build_textured_primitive(model: ModelBlock, textures: LevelTextures,
                              group: _Group, scale: float) -> Primitive3D:
    source = group.source
    per_hw = pixels_per_halfword(source.bpp)

    # Crop must start on a halfword boundary and cover whole halfwords.
    crop_x = (group.min_x // per_hw) * per_hw
    crop_y = group.min_y
    crop_w = group.max_x - crop_x + 1
    crop_w += (-crop_w) % per_hw
    crop_h = group.max_y - crop_y + 1

    image = textures.vram.region_image(crop_x, crop_y, crop_w, crop_h,
                                       source.bpp, source.clut_x,
                                       source.clut_y)

    prim = Primitive3D(name=source.label, image=image)
    cache: dict = {}

    for poly, uv in group.polygons:
        corner_indices = []
        for corner in range(poly.corners):
            u, v = uv.uvs[corner]
            abs_x, abs_y = _absolute_uv(uv, u, v, source.offset)
            # +0.5 samples the texel centre, which is what NEAREST filtering
            # needs to reproduce the original pixels exactly.
            uv_value = ((abs_x - crop_x + 0.5) / crop_w,
                        (abs_y - crop_y + 0.5) / crop_h)
            corner_indices.append(
                _emit_corner(prim, model, poly, corner, scale, uv_value,
                             cache))
        _add_triangles(prim, corner_indices)

    return _finish(prim)


def _build_untextured_primitive(model: ModelBlock, polygons: list[Polygon],
                                scale: float) -> Primitive3D:
    prim = Primitive3D(name="flat", image=None)
    cache: dict = {}
    for poly in polygons:
        corner_indices = [
            _emit_corner(prim, model, poly, corner, scale, None, cache)
            for corner in range(poly.corners)
        ]
        _add_triangles(prim, corner_indices)
    return _finish(prim)


def _add_triangles(prim: Primitive3D, corners: list[int]) -> None:
    """
    Triangulate a polygon's corner list.

    PSX quads are a zig-zag strip rather than a fan: corners run
    top-left, top-right, bottom-left, bottom-right. Splitting as (0,1,2) and
    (1,3,2) keeps both triangles' winding consistent with the authored order,
    which the model's own normals confirm is outward-facing.
    """
    if len(corners) == 3:
        prim.indices.extend(corners)
    else:
        prim.indices.extend([corners[0], corners[1], corners[2],
                             corners[1], corners[3], corners[2]])
