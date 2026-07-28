"""
glb.py — write meshes to self-contained binary glTF (.glb).

Textures are embedded as PNG inside the GLB buffer, so a single file carries
the whole model with nothing to resolve at load time.

Coordinate conversion from DD2 to glTF
--------------------------------------
DD2 model space, established from the car's own texture layout: ROOF88A sits
at y +124 and the wheels at y -97, so **+Y is up**; FRNT88A sits at z +464 and
the rear bumper at z -441, so **+Z is the front**.

glTF is also Y-up but conventionally faces -Z, so we rotate 180 degrees about
Y: (x, y, z) -> (-x, y, -z). That is a proper rotation, determinant +1, so
triangle winding is preserved and normals stay valid. A mirror such as
(x, y, -z) would have required flipping every winding as well, which is an
easy thing to get subtly wrong.

Winding is left as authored. Checked against the model's own normal data on
LEV1 section 17: `cross(v1-v0, v2-v0)` agrees in sign with the stored face
normal for 115 polygons and disagrees for 8, and the stored normals point away
from the model centroid 114 to 9. So the authored order is counter-clockwise /
outward, which is what glTF expects.

Vertex colours
--------------
PSX modulates texture colour by the primitive colour, with 0x80 meaning
"unchanged" — so the neutral value is 128, not 255. We emit COLOR_0 as
colour/128, which makes the common 0x808080 exactly 1.0 and leaves the glTF
product `baseColorTexture * COLOR_0` faithful to the hardware. Values above
128 would brighten past 1.0, so they are clamped; that affects a small number
of polygons and is preferable to darkening everything by using /255.
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from pygltflib import (
    ARRAY_BUFFER, CLAMP_TO_EDGE, ELEMENT_ARRAY_BUFFER, FLOAT, GLTF2, MASK,
    NEAREST, OPAQUE, TRIANGLES, UNSIGNED_INT, UNSIGNED_SHORT, Accessor, Asset,
    Attributes, Buffer, BufferView, Image as GLTFImage, Material, Mesh, Node,
    Primitive, PbrMetallicRoughness, Sampler, Scene, Texture, TextureInfo,
)

# PSX primitive colour 0x80 means "leave the texture alone".
COLOUR_NEUTRAL = 128.0


def _has_transparency(image: Image.Image) -> bool:
    """True if any pixel is fully transparent (PSX palette entry 0x0000)."""
    if image.mode != "RGBA":
        return False
    return int(np.asarray(image)[:, :, 3].min()) == 0


def to_gltf_position(x: float, y: float, z: float,
                     scale: float) -> tuple[float, float, float]:
    """DD2 model space -> glTF space (180 degree turn about Y, then scale)."""
    return (-x * scale, y * scale, -z * scale)


def to_gltf_direction(x: float, y: float,
                      z: float) -> tuple[float, float, float]:
    """Same rotation for normals; no scale, no translation."""
    return (-x, y, -z)


@dataclass
class Primitive3D:
    """One drawable group: geometry sharing a single material."""

    name: str
    positions: list[tuple[float, float, float]] = field(default_factory=list)
    normals: list[tuple[float, float, float]] = field(default_factory=list)
    colours: list[tuple[float, float, float, float]] = field(default_factory=list)
    uvs: list[tuple[float, float]] = field(default_factory=list)
    indices: list[int] = field(default_factory=list)
    image: Image.Image | None = None      # None -> untextured, colours only

    @property
    def textured(self) -> bool:
        return self.image is not None

    @property
    def vertex_count(self) -> int:
        return len(self.positions)


@dataclass
class Object3D:
    """A named node holding one or more primitives."""

    name: str
    primitives: list[Primitive3D] = field(default_factory=list)

    @property
    def triangle_count(self) -> int:
        return sum(len(p.indices) // 3 for p in self.primitives)

    @property
    def vertex_count(self) -> int:
        return sum(p.vertex_count for p in self.primitives)


class GLBBuilder:
    """
    Accumulates binary data and glTF structures, then writes one .glb.

    Buffer views are 4-byte aligned as the spec requires; PNG blobs are padded
    the same way so accessors that follow stay aligned.
    """

    def __init__(self):
        self.gltf = GLTF2(asset=Asset(generator="extraction-derby-2"))
        self.blob = bytearray()
        self._image_cache: dict[bytes, int] = {}
        self._sampler_index: int | None = None

    # -- low level ----------------------------------------------------------

    def _align(self) -> None:
        while len(self.blob) % 4:
            self.blob.append(0)

    def _add_view(self, data: bytes, target: int | None = None) -> int:
        self._align()
        offset = len(self.blob)
        self.blob.extend(data)
        self.gltf.bufferViews.append(
            BufferView(buffer=0, byteOffset=offset, byteLength=len(data),
                       target=target))
        return len(self.gltf.bufferViews) - 1

    def _add_accessor(self, array: np.ndarray, component_type: int,
                      accessor_type: str, target: int | None,
                      with_bounds: bool = False) -> int:
        view = self._add_view(array.tobytes(), target=target)
        accessor = Accessor(
            bufferView=view, componentType=component_type,
            count=int(array.shape[0]), type=accessor_type,
        )
        if with_bounds:
            accessor.min = array.min(axis=0).tolist()
            accessor.max = array.max(axis=0).tolist()
        self.gltf.accessors.append(accessor)
        return len(self.gltf.accessors) - 1

    def _sampler(self) -> int:
        """One shared sampler: nearest filtering, clamped — PSX look."""
        if self._sampler_index is None:
            self.gltf.samplers.append(
                Sampler(magFilter=NEAREST, minFilter=NEAREST,
                        wrapS=CLAMP_TO_EDGE, wrapT=CLAMP_TO_EDGE))
            self._sampler_index = len(self.gltf.samplers) - 1
        return self._sampler_index

    def _add_texture(self, image: Image.Image) -> int:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        png = buf.getvalue()

        # Identical images are common across cars and parts; store once.
        cached = self._image_cache.get(png)
        if cached is not None:
            return cached

        view = self._add_view(png)
        self.gltf.images.append(GLTFImage(bufferView=view,
                                          mimeType="image/png"))
        self.gltf.textures.append(
            Texture(sampler=self._sampler(),
                    source=len(self.gltf.images) - 1))
        index = len(self.gltf.textures) - 1
        self._image_cache[png] = index
        return index

    def _add_material(self, prim: Primitive3D) -> int:
        pbr = PbrMetallicRoughness(metallicFactor=0.0, roughnessFactor=0.9,
                                   baseColorFactor=[1.0, 1.0, 1.0, 1.0])
        alpha_mode = OPAQUE

        if prim.image is not None:
            pbr.baseColorTexture = TextureInfo(
                index=self._add_texture(prim.image))
            # PSX paletted textures are binary-transparent: palette entry
            # 0x0000 means "draw nothing". MASK reproduces that exactly and,
            # unlike BLEND, needs no draw-order sorting. Without this the
            # wheel's hubcap quad renders as an opaque black square instead of
            # a disc, because 314 of its 1024 texels are the transparent key.
            if _has_transparency(prim.image):
                alpha_mode = MASK

        material = Material(name=prim.name, pbrMetallicRoughness=pbr,
                            alphaMode=alpha_mode, doubleSided=True)
        if alpha_mode == MASK:
            material.alphaCutoff = 0.5
        self.gltf.materials.append(material)
        return len(self.gltf.materials) - 1

    # -- high level ---------------------------------------------------------

    def add_objects(self, objects: list[Object3D]) -> None:
        root_children: list[int] = []

        for obj in objects:
            gltf_prims: list[Primitive] = []

            for prim in obj.primitives:
                if not prim.indices:
                    continue

                positions = np.asarray(prim.positions, dtype=np.float32)
                attributes = Attributes(
                    POSITION=self._add_accessor(
                        positions, FLOAT, "VEC3", ARRAY_BUFFER,
                        with_bounds=True))

                if prim.normals:
                    attributes.NORMAL = self._add_accessor(
                        np.asarray(prim.normals, dtype=np.float32),
                        FLOAT, "VEC3", ARRAY_BUFFER)
                if prim.colours:
                    attributes.COLOR_0 = self._add_accessor(
                        np.asarray(prim.colours, dtype=np.float32),
                        FLOAT, "VEC4", ARRAY_BUFFER)
                if prim.uvs:
                    attributes.TEXCOORD_0 = self._add_accessor(
                        np.asarray(prim.uvs, dtype=np.float32),
                        FLOAT, "VEC2", ARRAY_BUFFER)

                # uint16 indices where possible; these meshes are tiny.
                if prim.vertex_count <= 0xFFFF:
                    idx = np.asarray(prim.indices, dtype=np.uint16)
                    idx_type = UNSIGNED_SHORT
                else:
                    idx = np.asarray(prim.indices, dtype=np.uint32)
                    idx_type = UNSIGNED_INT

                gltf_prims.append(Primitive(
                    attributes=attributes,
                    indices=self._add_accessor(idx, idx_type, "SCALAR",
                                               ELEMENT_ARRAY_BUFFER),
                    material=self._add_material(prim),
                    mode=TRIANGLES,
                ))

            if not gltf_prims:
                continue

            self.gltf.meshes.append(Mesh(name=obj.name,
                                         primitives=gltf_prims))
            self.gltf.nodes.append(Node(name=obj.name,
                                        mesh=len(self.gltf.meshes) - 1))
            root_children.append(len(self.gltf.nodes) - 1)

        self.gltf.scenes.append(Scene(nodes=root_children))
        self.gltf.scene = 0

    def save(self, path: str | Path) -> None:
        self._align()
        self.gltf.buffers.append(Buffer(byteLength=len(self.blob)))
        self.gltf.set_binary_blob(bytes(self.blob))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.gltf.save_binary(str(path))


def write_glb(objects: list[Object3D], path: str | Path) -> None:
    """Write one or more objects into a single self-contained .glb."""
    builder = GLBBuilder()
    builder.add_objects(objects)
    builder.save(path)
