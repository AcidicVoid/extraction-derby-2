"""
trackmodel.py — assemble a track's scenery into textured geometry.

Section 0 gives a list of model blocks each placed at a world position (see
dd2/terrain.py). This module turns that into renderable objects, grouping
polygons by the texture they sample so each group becomes one primitive with
its own cropped, palette-applied image — the same approach `dd2.carmodel` uses
for cars, minus the livery substitution, and extended to walk many models with
per-instance offsets.

What section 0 actually contains
--------------------------------
The **complete track**, drivable road included:

  * the road ribbon itself, tarmac with lane markings and banked walls
  * large ground panels, median extent about 23 700 units, painted with dirt
    and scrub-grass — the land around the circuit
  * grandstands, buildings, advertising hoardings, barriers

Confirmed by projecting section 2's path points straight down onto this
geometry: **all 2105 of them land on a triangle, with median and mean vertical
offset of exactly 0**. Section 2 is therefore a path sampled on the road
surface — a racing line or collision reference — and not a separate mesh that
needs building. Nothing is missing from section 0.

Beware flat-shaded previews. Drawn in a single colour with a painter's
algorithm the road blends into the surrounding ground and the scene reads as
disconnected fragments with holes in it. Both impressions are artefacts of the
preview; a z-buffered render shows a continuous circuit.

Placement uses `TerrainInstance.origin`, the **coarse** half of each position
word, not the raw value — see `dd2.terrain.coarse_translation`. Getting this
wrong is subtle: the raw value still lays the scenery out in roughly the right
ring around the circuit, so a top-down view looks broadly plausible, but every
piece is displaced by up to 0x8000 and the whole scene sits 0x4000 below the
road. The symptom is scenery that reads as scattered fragments rather than a
contiguous landscape.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .binio import FormatError
from .carmodel import DEFAULT_SCALE, _absolute_uv, _colour
from .glb import Object3D, Primitive3D, to_gltf_direction, to_gltf_position
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


def build_scenery_object(level: LevelFile, textures: LevelTextures,
                         instances: list[TerrainInstance],
                         scale: float = DEFAULT_SCALE,
                         name: str = "scenery") -> Object3D:
    """
    Merge placed model blocks into one object of texture-grouped primitives.

    Grouping is by (bit depth, CLUT position) rather than per instance: the
    same palette is shared by thousands of polygons across many models, so
    grouping this way keeps the primitive and texture count low instead of
    emitting one material per mesh.
    """
    obj = Object3D(name=name)
    groups: dict[tuple, _Group] = {}
    untextured: list[tuple[ModelBlock, Polygon, tuple[int, int, int]]] = []

    for inst in instances:
        model = inst.model
        for poly in model.polygons:
            if poly.uv_index is None or poly.clut_id is None:
                untextured.append((model, poly, inst.origin))
                continue
            uv = level.uv_table[poly.uv_index]
            clut = poly.clut_xy
            key = (uv.tpage.bpp, clut)
            group = groups.get(key)
            if group is None:
                group = groups[key] = _Group(bpp=uv.tpage.bpp, clut=clut)
            group.members.append((model, poly, uv, inst.origin))
            for u, v in uv.uvs[:poly.corners]:
                group.note(*_absolute_uv(uv, u, v, (0, 0)))

    for group in groups.values():
        prim = _textured_primitive(textures, group, scale)
        if prim.indices:
            obj.primitives.append(prim)

    if untextured:
        obj.primitives.append(_untextured_primitive(untextured, scale))

    return obj


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


def build_track(level: LevelFile, textures: LevelTextures,
                scale: float = DEFAULT_SCALE) -> list[Object3D]:
    """
    Build the objects making up one track.

    Section 0 carries the whole track — road, landscape and structures — so
    this is the complete visual model. Splitting the road out as its own object
    would need a way to tell road polygons from the rest; section 2's path can
    serve as that classifier if it is ever wanted.
    """
    instances = level.terrain.instances
    if not instances:
        raise FormatError(f"{level.name}: no terrain instances to export")
    return [build_scenery_object(level, textures, instances, scale=scale,
                                 name="track")]
