# Extraction Derby 2

Asset extractor for **Destruction Derby 2** (PlayStation, 1996).

Reads the game's `DIRINFO` archive straight off the disc and converts its
contents into modern formats:

- **Cars** as textured GLB models, one per livery, including the player car's
  three class variants. Body plus four wheels as separate named nodes.
- **Tracks** as textured GLB models, one per track, covering the road surface,
  landscape, grandstands and other scenery. Pine Hills' animated head and
  signpost are included with their rotation animation baked to keyframes.
- **Textures** as PNG, every tile in every level.

Textures are embedded inside the GLB files, so each model is self-contained and
opens directly in Blender or any other glTF viewer.

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- A `DIRINFO` file from an original Destruction Derby 2 disc

## Usage

```
uv run main.py <path_to_DIRINFO>
```

Everything is written to `output/`:

```
output/
    gamedata/    the unpacked DIRINFO archive
    cars/        car models (.glb)
    tracks/      track models (.glb), with props/ for standalone prop models
    textures/    texture tiles (.png), per level
    logs/        run log and detailed per-level reports
```

The output directory is wiped at the start of every run. It refuses to delete a
directory it did not create.

### Options

| Option | Effect |
|---|---|
| `-o`, `--output` | Output directory (default `output`) |
| `--scale` | Model units per output unit (default 1/256) |
| `--named-only` | Export only named textures, skipping road and scenery tiles |
| `--no-textures` | Skip texture extraction |
| `--no-cars` | Skip car models |
| `--no-tracks` | Skip track models |
| `-v`, `--verbose` | Mirror the log to the console |

## Notes

The extractor validates as it goes and fails loudly rather than writing
plausible but wrong output. Structural checks cover the archive, the VRAM
layout, polygon index ranges and the triangle count of every exported mesh.
Detailed findings for each level land in `output/logs/`.

`tools/render_glb.py` is a small software renderer used to check exported
models without a GPU:

```
uv run tools/render_glb.py output/cars/car_88.glb preview.png
uv run tools/render_glb.py output/tracks --sheet tracks.png
```

## License

Provided for research and preservation. Destruction Derby 2 and its assets
remain the property of their respective rights holders.
