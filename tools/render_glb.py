"""
render_glb.py — minimal software renderer for verifying exported GLBs.

A development aid, not part of the extraction pipeline. It exists so model
output can be checked without a GPU or a DCC application: if the geometry,
winding, UVs or palettes are wrong, that shows up here immediately.

    python tools/render_glb.py output/cars/car_88.glb out.png
    python tools/render_glb.py output/cars --sheet sheet.png

Nearest-neighbour texture sampling and a z-buffer, which is enough to judge
correctness. No lighting model beyond a single directional term.
"""

from __future__ import annotations

import argparse
import io
import math
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pygltflib import GLTF2

COMPONENT_DTYPE = {
    5120: np.int8, 5121: np.uint8, 5122: np.int16,
    5123: np.uint16, 5125: np.uint32, 5126: np.float32,
}
TYPE_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def read_accessor(gltf: GLTF2, blob: bytes, index: int) -> np.ndarray:
    acc = gltf.accessors[index]
    view = gltf.bufferViews[acc.bufferView]
    dtype = COMPONENT_DTYPE[acc.componentType]
    count = TYPE_COUNT[acc.type]
    start = (view.byteOffset or 0) + (acc.byteOffset or 0)
    data = np.frombuffer(blob, dtype=dtype,
                         count=acc.count * count, offset=start)
    return data.reshape(acc.count, count) if count > 1 else data


def read_image(gltf: GLTF2, blob: bytes, texture_index: int) -> Image.Image:
    src = gltf.textures[texture_index].source
    view = gltf.bufferViews[gltf.images[src].bufferView]
    start = view.byteOffset or 0
    raw = blob[start:start + view.byteLength]
    return Image.open(io.BytesIO(raw)).convert("RGBA")


def _node_matrix(node) -> np.ndarray:
    """Local TRS of a node as a 4x4 matrix."""
    m = np.eye(4)
    if node.rotation:
        x, y, z, w = node.rotation
        m[:3, :3] = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
    if node.scale:
        m[:3, :3] = m[:3, :3] @ np.diag(node.scale)
    if node.translation:
        m[:3, 3] = node.translation
    return m


def load_primitives(path: Path) -> list[dict]:
    """
    Read every primitive, with each node's transform baked into its positions.

    Node transforms matter: the animated props are stored at the origin and
    positioned entirely by their node's translation and rotation, so ignoring
    it draws them in the wrong place.
    """
    gltf = GLTF2().load_binary(str(path))
    blob = gltf.binary_blob()
    out = []
    mesh_transform: dict[int, np.ndarray] = {}
    for node in gltf.nodes:
        if node.mesh is not None:
            mesh_transform[node.mesh] = _node_matrix(node)

    for mesh_index, mesh in enumerate(gltf.meshes):
        xform = mesh_transform.get(mesh_index, np.eye(4))
        for prim in mesh.primitives:
            pos = read_accessor(gltf, blob, prim.attributes.POSITION)
            pos = (xform @ np.hstack(
                [pos, np.ones((len(pos), 1))]).T).T[:, :3]
            entry = {
                "pos": pos.astype(np.float32),
                "idx": read_accessor(gltf, blob, prim.indices).astype(np.int64),
                "uv": None, "col": None, "tex": None,
            }
            if prim.attributes.TEXCOORD_0 is not None:
                entry["uv"] = read_accessor(gltf, blob,
                                            prim.attributes.TEXCOORD_0)
            if prim.attributes.COLOR_0 is not None:
                entry["col"] = read_accessor(gltf, blob,
                                             prim.attributes.COLOR_0)
            mat = gltf.materials[prim.material] if prim.material is not None \
                else None
            if mat is not None and mat.pbrMetallicRoughness is not None \
                    and mat.pbrMetallicRoughness.baseColorTexture is not None:
                entry["tex"] = read_image(
                    gltf, blob,
                    mat.pbrMetallicRoughness.baseColorTexture.index)
            out.append(entry)
    return out


def look_at(eye, target, up):
    f = np.array(target, dtype=np.float64) - np.array(eye, dtype=np.float64)
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ np.array(eye, dtype=np.float64)
    return m


def render(prims: list[dict], size=(640, 480), yaw=35.0, pitch=22.0,
           background=(28, 28, 34)) -> Image.Image:
    width, height = size

    allpos = np.concatenate([p["pos"] for p in prims], axis=0)
    centre = (allpos.min(axis=0) + allpos.max(axis=0)) / 2.0
    radius = float(np.linalg.norm(allpos.max(axis=0) - centre)) or 1.0

    ay, ap = math.radians(yaw), math.radians(pitch)
    dist = radius * 2.9
    eye = centre + np.array([
        dist * math.cos(ap) * math.sin(ay),
        dist * math.sin(ap),
        dist * math.cos(ap) * math.cos(ay),
    ])
    view = look_at(eye, centre, (0, 1, 0))

    fov, near, far = math.radians(35.0), radius * 0.05, dist + radius * 4
    t = 1.0 / math.tan(fov / 2)
    proj = np.zeros((4, 4))
    proj[0, 0] = t / (width / height)
    proj[1, 1] = t
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = 2 * far * near / (near - far)
    proj[3, 2] = -1.0

    frame = np.zeros((height, width, 3), dtype=np.float64)
    frame[:, :] = np.array(background) / 255.0
    depth = np.full((height, width), np.inf)
    light = np.array([0.4, 0.8, 0.5])
    light /= np.linalg.norm(light)

    for prim in prims:
        pos = np.asarray(prim["pos"], dtype=np.float64)
        clip = (proj @ view @ np.hstack([pos, np.ones((len(pos), 1))]).T).T
        w = clip[:, 3].copy()
        w[np.abs(w) < 1e-9] = 1e-9
        ndc = clip[:, :3] / w[:, None]
        screen = np.empty((len(pos), 3))
        screen[:, 0] = (ndc[:, 0] * 0.5 + 0.5) * width
        screen[:, 1] = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * height
        screen[:, 2] = w

        tex = prim["tex"]
        texels = np.asarray(tex, dtype=np.float64) / 255.0 if tex else None
        idx = prim["idx"].reshape(-1, 3)

        for tri in idx:
            p = screen[tri]
            if np.any(p[:, 2] <= 0):
                continue
            x0 = max(int(np.floor(p[:, 0].min())), 0)
            x1 = min(int(np.ceil(p[:, 0].max())) + 1, width)
            y0 = max(int(np.floor(p[:, 1].min())), 0)
            y1 = min(int(np.ceil(p[:, 1].max())) + 1, height)
            if x0 >= x1 or y0 >= y1:
                continue

            ax, ay_, bx, by, cx, cy = (p[0, 0], p[0, 1], p[1, 0], p[1, 1],
                                       p[2, 0], p[2, 1])
            area = (bx - ax) * (cy - ay_) - (cx - ax) * (by - ay_)
            if abs(area) < 1e-9:
                continue

            xs = np.arange(x0, x1) + 0.5
            ys = np.arange(y0, y1) + 0.5
            gx, gy = np.meshgrid(xs, ys)
            w0 = ((bx - ax) * (gy - ay_) - (gx - ax) * (by - ay_)) / area
            w1 = ((gx - ax) * (cy - ay_) - (cx - ax) * (gy - ay_)) / area
            l2, l0, l1 = w0, 1.0 - w0 - w1, w1
            inside = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
            if not inside.any():
                continue

            iw = (l0 / screen[tri[0], 2] + l1 / screen[tri[1], 2]
                  + l2 / screen[tri[2], 2])
            zview = 1.0 / np.where(np.abs(iw) < 1e-12, 1e-12, iw)
            sub = depth[y0:y1, x0:x1]
            visible = inside & (zview < sub)
            if not visible.any():
                continue

            # Perspective-correct barycentrics for attribute interpolation.
            pl0 = (l0 / screen[tri[0], 2]) * zview
            pl1 = (l1 / screen[tri[1], 2]) * zview
            pl2 = (l2 / screen[tri[2], 2]) * zview

            rgb = np.ones(gx.shape + (3,))
            if prim["col"] is not None:
                c = np.asarray(prim["col"], dtype=np.float64)[tri][:, :3]
                rgb = (pl0[..., None] * c[0] + pl1[..., None] * c[1]
                       + pl2[..., None] * c[2])
            if texels is not None and prim["uv"] is not None:
                uv = np.asarray(prim["uv"], dtype=np.float64)[tri]
                u = pl0 * uv[0, 0] + pl1 * uv[1, 0] + pl2 * uv[2, 0]
                v = pl0 * uv[0, 1] + pl1 * uv[1, 1] + pl2 * uv[2, 1]
                th, tw = texels.shape[:2]
                tu = np.clip((u * tw).astype(int), 0, tw - 1)
                tv = np.clip((v * th).astype(int), 0, th - 1)
                rgb = rgb * texels[tv, tu, :3]
                # Honour binary transparency, or the wheel hubcap's square
                # quad hides the disc behind it.
                visible &= texels[tv, tu, 3] >= 0.5
                if not visible.any():
                    continue

            # Flat directional shade from the triangle's geometric normal.
            e1 = pos[tri[1]] - pos[tri[0]]
            e2 = pos[tri[2]] - pos[tri[0]]
            n = np.cross(e1, e2)
            ln = np.linalg.norm(n)
            shade = 0.55 + 0.45 * abs(float(n @ light) / ln) if ln > 0 else 1.0

            target = frame[y0:y1, x0:x1]
            target[visible] = np.clip(rgb[visible] * shade, 0, 1)
            sub[visible] = zview[visible]

    return Image.fromarray((frame * 255).astype(np.uint8), mode="RGB")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path, help=".glb file or a directory of them")
    ap.add_argument("out", type=Path, nargs="?", help="output PNG")
    ap.add_argument("--sheet", type=Path, help="contact sheet for a directory")
    ap.add_argument("--yaw", type=float, default=35.0)
    ap.add_argument("--pitch", type=float, default=22.0)
    ap.add_argument("--size", type=int, nargs=2, default=[640, 480])
    args = ap.parse_args()

    if args.path.is_dir():
        files = sorted(args.path.glob("*.glb"))
        if not files:
            print("no .glb files found", file=sys.stderr)
            return 1
        images = []
        for f in files:
            img = render(load_primitives(f), tuple(args.size), args.yaw,
                         args.pitch)
            images.append((f.stem, img))
            print(f"rendered {f.name}")
        cols = min(5, len(images))
        rows = (len(images) + cols - 1) // cols
        cw, ch = images[0][1].size
        sheet = Image.new("RGB", (cw * cols, ch * rows), (20, 20, 24))
        for i, (_, img) in enumerate(images):
            sheet.paste(img, ((i % cols) * cw, (i // cols) * ch))
        dest = args.sheet or (args.out or Path("sheet.png"))
        sheet.save(dest)
        print(f"wrote {dest}")
        return 0

    img = render(load_primitives(args.path), tuple(args.size), args.yaw,
                 args.pitch)
    dest = args.out or args.path.with_suffix(".png")
    img.save(dest)
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
