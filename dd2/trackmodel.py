"""
Assemble a track into textured geometry.

Section 0 gives a list of model blocks each placed at a world position. This
module groups their polygons by the texture they sample, so each group becomes
one primitive with its own cropped, palette-applied image.

Section 0 contains the complete track: the road ribbon with its lane markings
and banked walls, the surrounding landscape, grandstands, buildings, hoardings
and barriers.

Placement uses TerrainInstance.origin, the coarse half of each position word,
never the raw value. See dd2.terrain.coarse_translation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .binio import FormatError
from .carmodel import DEFAULT_SCALE, _absolute_uv, _colour
from .glb import (PSX_ANGLE_UNITS, Object3D, Primitive3D,
                  psx_euler_to_quaternion, to_gltf_direction, to_gltf_position)
from .level import LevelFile
from .model import ModelBlock, Polygon
from .terrain import TerrainInstance
from .textures import LevelTextures
from .uvtable import UVRecord
from .vram import pixels_per_halfword

NORMAL_SCALE = 1.0 / 4096.0


@dataclass
class _Group:
    """Polygons sharing one palette, and the VRAM extent they sample."""

    bpp: int
    clut: tuple[int, int]
    members: list[tuple[ModelBlock, Polygon, UVRecord, tuple[int, int, int]]] = \
        field(default_factory=list)
    min_x: int = 1 << 30
    min_y: int = 1 << 30
    max_x: int = -(1 << 30)
    max_y: int = -(1 << 30)

    def note(self, x: int, y: int) -> None:
        self.min_x = min(self.min_x, x)
        self.max_x = max(self.max_x, x)
        self.min_y = min(self.min_y, y)
        self.max_y = max(self.max_y, y)


def build_placed_object(level: LevelFile, textures: LevelTextures,
                        placements: list[tuple[ModelBlock, tuple[int, int, int]]],
                        scale: float = DEFAULT_SCALE,
                        name: str = "object") -> Object3D:
    """
    Merge (model, origin) pairs into one object of texture-grouped primitives.

    Takes explicit origins rather than TerrainInstance, because only terrain
    placement goes through the coarse/fine split. Props are positioned by other
    means and must not have coarse_translation applied, which maps 0 to 0x4000
    and would displace a prop built at the origin.

    Grouping is by (bit depth, CLUT position) rather than per model: the same
    palette is shared by thousands of polygons across many models, so grouping
    this way keeps the primitive and texture count low instead of emitting one
    material per mesh.
    """
    obj = Object3D(name=name)
    groups: dict[tuple, _Group] = {}
    untextured: list[tuple[ModelBlock, Polygon, tuple[int, int, int]]] = []

    for model, origin in placements:
        for poly in model.polygons:
            if poly.uv_index is None or poly.clut_id is None:
                untextured.append((model, poly, origin))
                continue
            uv = level.uv_table[poly.uv_index]
            clut = poly.clut_xy
            key = (uv.tpage.bpp, clut)
            group = groups.get(key)
            if group is None:
                group = groups[key] = _Group(bpp=uv.tpage.bpp, clut=clut)
            group.members.append((model, poly, uv, origin))
            for u, v in uv.uvs[:poly.corners]:
                group.note(*_absolute_uv(uv, u, v, (0, 0)))

    for group in groups.values():
        prim = _textured_primitive(textures, group, scale)
        if prim.indices:
            obj.primitives.append(prim)

    if untextured:
        obj.primitives.append(_untextured_primitive(untextured, scale))

    return obj


def build_scenery_object(level: LevelFile, textures: LevelTextures,
                         instances: list[TerrainInstance],
                         scale: float = DEFAULT_SCALE,
                         name: str = "scenery") -> Object3D:
    """Terrain instances -> one object, using each instance's coarse origin."""
    return build_placed_object(
        level, textures, [(i.model, i.origin) for i in instances],
        scale=scale, name=name)


def _emit(prim: Primitive3D, model: ModelBlock, poly: Polygon, corner: int,
          origin: tuple[int, int, int], scale: float,
          uv_value: tuple[float, float] | None, cache: dict) -> int:
    """Append one polygon corner, reusing an identical earlier vertex."""
    vertex_index = poly.vertices[corner]
    vertex = model.vertices[vertex_index]

    if poly.normals:
        normal_index = poly.normals[corner]
    elif poly.face_normal is not None:
        normal_index = poly.face_normal
    else:
        normal_index = None

    colour = _colour(poly.colours[corner if len(poly.colours) > 1 else 0])
    key = (id(model), vertex_index, normal_index, colour, uv_value, origin)
    cached = cache.get(key)
    if cached is not None:
        return cached

    prim.positions.append(to_gltf_position(vertex.x + origin[0],
                                           vertex.y + origin[1],
                                           vertex.z + origin[2], scale))
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


def _add_triangles(prim: Primitive3D, corners: list[int]) -> None:
    if len(corners) == 3:
        prim.indices.extend(corners)
    else:
        prim.indices.extend([corners[0], corners[1], corners[2],
                             corners[1], corners[3], corners[2]])


def _textured_primitive(textures: LevelTextures, group: _Group,
                        scale: float) -> Primitive3D:
    per_hw = pixels_per_halfword(group.bpp)
    crop_x = (group.min_x // per_hw) * per_hw
    crop_y = group.min_y
    crop_w = group.max_x - crop_x + 1
    crop_w += (-crop_w) % per_hw
    crop_h = group.max_y - crop_y + 1

    image = textures.vram.region_image(crop_x, crop_y, crop_w, crop_h,
                                       group.bpp, group.clut[0], group.clut[1])
    prim = Primitive3D(
        name=f"clut_{group.clut[0]}_{group.clut[1]}_{group.bpp}bpp",
        image=image)
    cache: dict = {}

    for model, poly, uv, origin in group.members:
        corners = []
        for c in range(poly.corners):
            u, v = uv.uvs[c]
            ax, ay = _absolute_uv(uv, u, v, (0, 0))
            corners.append(_emit(
                prim, model, poly, c, origin, scale,
                ((ax - crop_x + 0.5) / crop_w, (ay - crop_y + 0.5) / crop_h),
                cache))
        _add_triangles(prim, corners)

    if len(prim.normals) != len(prim.positions):
        prim.normals = []
    return prim


def _untextured_primitive(members, scale: float) -> Primitive3D:
    prim = Primitive3D(name="flat", image=None)
    cache: dict = {}
    for model, poly, origin in members:
        corners = [_emit(prim, model, poly, c, origin, scale, None, cache)
                   for c in range(poly.corners)]
        _add_triangles(prim, corners)
    if len(prim.normals) != len(prim.positions):
        prim.normals = []
    return prim


# Sections 5..20 are the shared car set - wheels, dust decals, car LODs, the
# detachable bonnet and boot - and are exported as part of the cars. Anything
# from 21 up is a track-specific prop.
FIRST_PROP_SECTION = 21


def build_prop_object(level: LevelFile, textures: LevelTextures, section: int,
                      scale: float = DEFAULT_SCALE,
                      name: str | None = None) -> Object3D:
    """
    Build one standalone prop model at the origin.

    These are the moving trackside objects: LEV1's head and signpost, LEV4's
    mine carts and LEV8's banner. They are not instanced in section 0, so they
    have no stored placement. Each is exported untransformed at the origin.
    """
    model = level.model(section)
    return build_placed_object(level, textures, [(model, (0, 0, 0))],
                               scale=scale,
                               name=name or f"section_{section:02d}")


def prop_sections(level: LevelFile) -> list[int]:
    """Section indices holding track-specific props, if any."""
    return [s.index for s in level.model_sections
            if s.index >= FIRST_PROP_SECTION]


# --------------------------------------------------------------------------
# LEV1's animated props
# --------------------------------------------------------------------------
# FUN_8001e278 has a switch on the level number whose case 1 draws two extra
# objects every frame:
#
#     FUN_8001f8c8(&DAT_800848dc, rot, &DAT_8006ec78)   ptr[21], the head
#     FUN_8001f8c8(&DAT_8008c464, rot, &DAT_8006ec68)   ptr[22], the signpost
#
# (`FUN_8001d944` is what binds those two handles to ptr[21] and ptr[22].)
#
# Both positions read from the executable as the same VECTOR, (55504, 3240,
# 9956), which lands inside LEV1's terrain bounds - the sign is the post and the
# head sits on it. Both rotation seeds are the SVECTOR (512, 0, 0): a fixed
# 45-degree tilt about X, since 512 of 4096 units is an eighth of a turn.
#
# The animation is procedural, so it has to be sampled rather than read:
#
#     phase += 0x40            every frame, wrapping at 4096
#     rot.z  = rsin(phase) >> 4
#
# `rsin` returns a 1.12 fixed-point sine, so `>> 4` gives +-256 units - the two
# objects rock +-22.5 degrees. 4096 / 0x40 = 64 frames per cycle, which at
# the NTSC 30 Hz this build runs at is about 2.13 seconds.
PROP_POSITION = (55504, 3240, 9956)
PROP_ROTATION_SEED = (512, 0, 0)
PROP_PHASE_STEP = 0x40
PROP_SINE_SHIFT = 4
FRAMES_PER_SECOND = 30.0

# Which section is which, and the node name to give it.
LEV1_ANIMATED_PROPS = {21: "head", 22: "signpost"}


def _prop_rotation_track() -> tuple[list[float], list[tuple[float, ...]]]:
    """Bake the procedural rock into one keyframe per source frame."""
    frames = PSX_ANGLE_UNITS // PROP_PHASE_STEP
    times: list[float] = []
    quats: list[tuple[float, ...]] = []
    rx, ry, _ = PROP_ROTATION_SEED

    for frame in range(frames + 1):          # +1 so the loop closes cleanly
        phase = (frame * PROP_PHASE_STEP) % PSX_ANGLE_UNITS
        # rsin: 1.12 fixed point sine of a 4096-unit angle.
        rz = int(math.sin(2.0 * math.pi * phase / PSX_ANGLE_UNITS)
                 * PSX_ANGLE_UNITS) >> PROP_SINE_SHIFT
        times.append(frame / FRAMES_PER_SECOND)
        quats.append(psx_euler_to_quaternion(rx, ry, rz))

    return times, quats


# The switch is on the level number, so the placement above applies to LEV1 and
# nothing else. LEV4 happens to have sections 21 and 22 as well (its mine carts),
# but they are driven by a different case and have their own positions, so
# keying off "does this level have sections 21 and 22" would place them wrongly.
ANIMATED_PROP_LEVEL = 1


def build_animated_props(level: LevelFile, textures: LevelTextures,
                         scale: float = DEFAULT_SCALE) -> list[Object3D]:
    """
    Build LEV1's head and signpost, placed and animated.

    The caller is responsible for only invoking this for LEV1; see
    ANIMATED_PROP_LEVEL.
    """
    objects: list[Object3D] = []
    times, quats = _prop_rotation_track()
    px, py, pz = PROP_POSITION

    for section, label in LEV1_ANIMATED_PROPS.items():
        if section not in level.models:
            continue
        obj = build_prop_object(level, textures, section, scale=scale,
                                name=label)
        obj.translation = to_gltf_position(px, py, pz, scale)
        obj.rotation = quats[0]
        obj.rotation_track = (times, quats)
        objects.append(obj)

    return objects


def build_track(level: LevelFile, textures: LevelTextures,
                scale: float = DEFAULT_SCALE,
                level_id: int | None = None) -> list[Object3D]:
    """
    Build the objects making up one track.

    Section 0 carries the whole track - road, landscape and structures - so
    this is the complete visual model. Splitting the road out as its own object
    would need a way to tell road polygons from the rest; section 2's path can
    serve as that classifier if it is ever wanted.
    """
    instances = level.terrain.instances
    if not instances:
        raise FormatError(f"{level.name}: no terrain instances to export")
    objects = [build_scenery_object(level, textures, instances, scale=scale,
                                    name="track")]
    # LEV1's head and signpost are drawn by dedicated per-level code rather
    # than instanced in section 0, so they are added here as their own
    # animated nodes. Gated on the level number, not on which sections exist.
    if level_id == ANIMATED_PROP_LEVEL:
        objects.extend(build_animated_props(level, textures, scale=scale))
    return objects
