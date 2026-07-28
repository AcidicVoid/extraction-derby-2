# Extraction Derby 2 — Evaluation & Implementation Plan

Headless asset extractor for **Destruction Derby 2** (PS1, 1996 — Reflections / Psygnosis).
Target output: textured GLB models (cars + tracks) with baked-in textures.

Status: **planning complete, implementation not started.**

---

## 1. Evaluation of current state

### 1.1 What exists on disk

| Location | Contents | Verdict |
|---|---|---|
| `Game Data/__DIRINFO/` | Fully extracted disc archive (LEV0–LEVF, VAGS, BIN/RAW/TIM) | **Primary input.** Complete. |
| `Decompiles/SCUS_943.50.txt` | Ghidra pseudocode, ~56k lines | **Reference.** Used to settle format questions. |
| `Decompiles/ASM.txt` | MIPS disassembly | Reference of last resort. |
| `Hexdumps/` | Clean hexdumps of the same files | Convenience only. |
| `Scripts/dd2_extractor.py` | DIRINFO unpacker | Already run; output is `__DIRINFO/`. Not needed again. |
| `Scripts/dd2_extract_textures.py` + `out/`, `out2/` | Standalone tile dumper | Hint source only. Not a dependency. |
| `Panda3D/`, `Panda3D.bak000/` | Viewer + earlier parser/exporter | **Obsolete. Will not be used or ported.** |
| `extraction-derby-2/` | Empty `uv` project (py3.12, no deps, no commits) | **Implementation target.** |

### 1.2 Findings carried over from prior sessions (all re-verified)

- **LEVEL.DAT is never compressed.** Re-confirmed two ways: (a) the 29-entry `uint32`
  pointer table at offset `0x00` is monotonically increasing and fully in-bounds for
  LEV0–LEVF; (b) the loader `FUN_80042a48` does nothing but add the base address to
  each of the 29 pointers in place — there is no decompression call anywhere in the
  load path. The old `LZSS` code path and the stale `LEVEL_DECOMP.BIN` artifacts are
  wrong and are discarded.
- **TXC format**: `u32 count`, then `count × 8` bytes of `(i16 bpp, i16 vram_x, i16 vram_y, i16 pad)`,
  then a raw palette stream (32 B per 4bpp entry, 512 B per 8bpp entry) with no
  further headers. Total size is an exact checksum.
- **Car colouring is CLUT-driven.** One shared set of body tiles (car #88, light blue)
  lives in TX0–TX3; per-car colour comes from TXC palette uploads. Roles:
  `CLUTnnA` (8bpp) → body (BUMP/BKWN/FRWN/FRNT), `B` → bonnet+roof, `C` → boot,
  `D` → windows, `E` → side windows.

### 1.3 New findings from this session (the ones that unblock the project)

**A. All 29 pointers have the same meaning in every level file.** LEV0 simply leaves
the track slots empty (pointing at `0x74`). This gives one parser for everything:

| Ptr | Meaning | LEV0 | LEV1–LEVB |
|---|---|---|---|
| 0 | Track terrain: nested table of geometry chunks | empty | 28 chunks |
| 1 | Track spline / section table (`u32 count` = 340 in LEV1) | empty | present |
| 2 | Point grid — `i32 x, y, z` triples, X-major, ~1000 units apart | empty | present |
| 3 | **Texture name table** | 258 entries | 351 entries |
| 4 | **UV table** | present | 1384 entries |
| 5 … 28 | Model blocks (props, cars, wheels, LODs) | 24 slots | 17 used |

**B. Texture name table (ptr[3]) format is fully decoded** — `u32 count`, then
24-byte entries:

```
+0x00 u16 vram_x      (X in VRAM halfwords)
+0x02 u16 vram_y      (scanline)
+0x04 u16 width       (PIXELS, not halfwords)
+0x06 u16 height      (scanlines)
+0x08 u16 clut_x      (halfwords)
+0x0a u16 clut_y
+0x0c u16 bpp         (literal bit depth; only ever 4 or 8)
+0x0e char name[10]   (NUL-padded)
```

Size check passes exactly on **all 14** level files. The unit interpretation is
confirmed by rasterising every record into a 1024×512 VRAM grid: **zero
overlapping halfwords and zero out-of-bounds tiles across all 14 files**. Any
other reading of the unit fields produces mass collisions.

Two sentinel forms mark records that are *not* resident in VRAM — for these only
`clut_x`/`clut_y` carry meaning:

- `(vram_x, vram_y) == (320, 0)` — staging slot, or a palette-only record
- `vram_y == 0xFFFF` — palette-only record (seen on `SMCL*` in LEV1/LEV3)

**C. UV table (ptr[4]) format is fully decoded** — `u32 count`, then 12-byte entries:
`u32 tpage` + `4 × (u8 u, u8 v)`. Size matches the section span exactly on 13 of 14
files; LEVC has a zero-length section 4, i.e. absent rather than mismatched.
TPage decodes as `x_base = (tp & 0xF) × 64`, `y_base = ((tp>>4)&1) × 256`,
depth from bits 7–8.

**B2. The car asset naming scheme is fully mapped** (surveyed across all levels):

- **Body tiles** — `BUMP`, `BKWN`, `FRNT`, `FRWN`, `BON`, `ROOF`, `BOOT`, `DEBR`.
  Every one exists for **car 88 only** (31 tiles in LEV1). This is the single shared
  body tile set, confirming the CLUT-swap theory from the binary rather than by
  inference.
- **Palette records** — `CLUT` (19 cars: all but 88, which uses its own embedded
  palette), `SMCL` (20 cars, low-detail palettes), `CLT` (player alternate liveries,
  slots 01/02 only — *not* car numbers).
- **Door number panels** — `DRnn` × 3 damage states = 60 records per track.
- Canonical roster is 20 cars: `00 01 07 13 17 35 37 40 42 47 50 52 53 64 66 69 77 82 88 99`,
  identical in every track. LEV2 alone carries an extra `DR02A/B/C` set, which the
  reports flag rather than silently absorb.
- A naive `PREFIX + 2 digits + letter` regex is **not** safe: LEV0's track thumbnails
  `TRACK01L`–`TRACK11L` match it. Classification uses an explicit prefix allowlist.

**F. The polygon command stream is fully decoded — one rule, all 24 types.**
The type byte splits into `base = type & 0x1C` (primitive kind) and
`flags = type & 0x03` (how shading data arrives). `base` maps one-to-one onto the
PSX GPU primitive codes, and each entry stores that code at +0x07 — these are
partially pre-built GP0 packets. `base & 0x04` = quad, `& 0x08` = textured,
`& 0x10` = gouraud. Flag bit 0 means "entry carries a face normal index at +0x02"
(otherwise that slot is `0xFFFF`), and for gouraud primitives it also selects
per-vertex *normal indices* over baked per-corner *colours*.

Entry layout: `u8 base`, `u8 attribute`, `u16 face_normal|0xFFFF`,
`u32 colour × (1 or n)`, `[u16 uv_index, u16 clut_id]` if textured,
`u16 × n` vertex indices, `u16 × n` normal indices if applicable, pad to 4.

Verification: the predicted entry size matches the batch header's declared size for
all 24 types, and **all 6532 polygons in all 197 model blocks pass** every index
range check. 210 entries carry ASCII junk in the 2-byte tail pad, so padding is not
asserted zero. The old type table was both incomplete (missing `0x02 0x03 0x06 0x07
0x09 0x0B 0x0E 0x0F 0x19 0x1B 0x1D 0x1F`) and wrong about several entry sizes.

`clut_id` decodes as `clut_x = (id & 0x3F) × 16`, `clut_y = id >> 6`.

**G. The car is LEV1 sections 16 and 17** (U3 resolved). Mapping every textured
polygon through the UV table into VRAM and back to named tiles shows these two
blocks — and only these — sample the shared body-part set: `FRWN88A`, `BKWN88A`,
`DR88A`, `BON88A`, `WINSID88`, `ROOF88A`, `FRNT88A`, `BOOT88A`, `WINFRN88`,
`BUMP88A`, `WINBCK88`. Section 17 (107 v / 147 p) is the more detailed;
section 16 (102 v / 108 p) is a second variant. Sections 19 and 20 are the
detachable bonnet and boot panels (`BON88A/C`, `BOOT88A/C`); 5, 6, 18 are debris;
7–14 are dust/skid decals; 21 and 22 are trackside props.

**I. LEVEL.TX0–TX3 is one archive split across four files.** TX0 holds `u32 count`
then 16-byte tile descriptors `(bpp, width_px, height, vram_x, vram_y, clut_x,
clut_y, pad)`; the pixel payload starts after that directory and continues byte-for-byte
through TX1, TX2 and TX3 concatenated. Each payload block is
`"CLUT"` + palette (32 B or 512 B) + `"TEXT"` + pixels — **ASCII magic tags**, which
make excellent parse anchors.

Verification, all 14 levels: both tags present on every block, payload consumed to the
exact byte, zero VRAM overlaps, nothing out of bounds. Decisively: **every resident
name-table record matches a TX descriptor exactly** on
`(vram_x, vram_y, width, height, bpp)` — 200/200 LEV0, 194/194 LEV1, and so on for all
14. Two independently parsed formats agreeing on every field is the strongest evidence
available. The old script's per-tile layout (4-byte header + palette + 4-byte pad)
happened to have the right *total size* but missed that the 8 bytes are two tags.

**J. LEVEL.TXC entries are 8 bytes, not 16** — `u16 bpp, u16 vram_x, u16 vram_y,
u16 pad`, then a palette stream of 32/512 bytes per entry. Sizes match exactly in 11
of 14 levels; LEV2/LEV7/LEV8 carry 2400/1360/1360 bytes of *unreferenced* filler
(0xFC1F magenta, 0xFFFF) which we report as slack rather than guessing at extra
entries. The old script's 16-byte entry size was wrong.

**K. `clut_y == 0xFFFE` is a shared "no palette of my own" sentinel** in both the TX
descriptors *and* the name table. In LEV1 the 38 affected named records are exactly
`DRnnB` and `DRnnC` — the damaged door panels, which reuse their `DRnnA` palette.

**L. Only `0x0000` is transparent.** 63 TXC palettes hold `0x83E0` (pure green) at
index 0 and `WINFRN88` holds `0xF360` (cyan) — both look like chroma keys and are not.
The 8bpp body tiles referencing those palettes never use index 0, so the slot is dead.
Keying on green would have punched holes in every car.

**N. The car is LEV1 section 17 for the body and section 18 for the wheel.**
Section 17 is complete except wheels. Section 18 is one wheel: two textured hubcap
faces at x = ±33 plus eight quads forming an octagonal tread. Sections 5 and 6 are
simplified black single-sided alternatives; `FUN_8002b874` binds *those* to the four
wheel slots (`ptr[5], ptr[6], ptr[5], ptr[6]`), but they have no hubcap texture, so
section 18 is what we export.

**O. Wheel placement is not stored** — the game derives it from live suspension state
(`DAT_80091028 + car*0x288 + wheel*0x84`), so it has to be reconstructed.

The arches are **painted on flat side panels, not cut into the geometry**, so the
panel's bounding box says nothing about where the wheel belongs. A first attempt used
it and the wheels came out visibly off. The arch outline lives in the FRWN88A/BKWN88A
*textures* as a near-black region, so the fix is to locate that region in texture space
and map it back through the polygons' UVs. Fitting an affine UV→XYZ map by least
squares over every arch-panel corner (planar panels, mean error 1–11 units) and
evaluating at the dark region's centre gives:

| arch | x | y | z | painted radius |
|---|---|---|---|---|
| front | ±172 | −36 | +291 | ~79 |
| rear | ±179 | −95 *(clipped)* | −231 | ~82 |

Painted radius exceeding the wheel's 67 is expected — the arch is drawn larger than the
tyre. The rear `y` is unusable: its dark region runs into the tile's bottom edge, so the
widest row reports the tile edge rather than the axle. Axles are level on a car, so the
front's −36 is used for both. `WHEEL_INSET = 16` sets how far inboard of the body
surface the wheel's outer face sits — 0 protrudes past the flanks, 33 (full half-width)
hides the wheels entirely, 16 sits flush and just visible from behind. Each wheel
remains a separate named node.

**P. Section 18 ships coincident duplicate faces.** Polygons 0/1 duplicate 10/11 with
identical vertices and UVs but colour (0,0,0) instead of neutral (128,128,128). The
game picks one at draw time; exported together they z-fight and the black pair wins,
turning every wheel into a black disc. `_deduplicate` keeps the brighter of any two
polygons sharing a vertex set and UV index.

**Q. The part→CLUT-letter mapping was derived, and two prior assumptions were wrong.**
Six 4bpp tiles share four per-car palettes, so the grouping matters. Test: palette
entries that are car-*independent* (glass, driver, chrome) must be byte-identical to
car 88's own tile palette, since car 88 is one of the 20. Agreement counts out of 16,
across all 19 other cars:

| tile | B | C | D | E |
|---|---|---|---|---|
| BON88A / ROOF88A / BOOT88A | **4** | 0 | 1 | 0 |
| WINFRN88 | 1 | 0 | **13** | 0 |
| WINBCK88 | 0 | **13** | 0 | 0 |
| WINSID88 | 0 | 0 | 0 | **8** |

So `A` = 8bpp body, `B` = bonnet + roof + **boot**, `C` = **rear window**,
`D` = windscreen, `E` = side glass. Off-diagonal agreement is 0 or 1 throughout.
This corrects the prior session's notes, which had boot on C and both windows on D.

**R. Two bugs worth remembering, both of which produced plausible-looking output.**
- The car-asset regex required a variant suffix (`<PART><nn><VARIANT>`), but the three
  window tiles are `<PART><nn>` — `WINSID88`, `WINFRN88`, `WINBCK88` have no damage
  states. They failed to match, were treated as non-car assets, and kept car 88's
  light-blue palette: **every** livery came out with cyan window frames, because the
  frame colour is index 1, covering 1019 of WINSID88's 2304 pixels.
- Materials were written `alphaMode=OPAQUE`, discarding the PSX transparency key.
  The hubcap is a square quad whose corners are palette entry `0x0000`; opaque, it
  rendered as a black square instead of a disc. Now `MASK` with cutoff 0.5 wherever
  a texture contains fully transparent texels — which is only the wheel.

**S. The player's car has three class variants, and they need *two* palette sets.**
Car 01 carries `CLUT01A–E` plus `CLT01A2–E2` and `CLT01A3–E3`, exported as
`car_01_rookie`, `car_01_amateur`, `car_01_pro`. Evidence that these are variants of
car 01 rather than separate cars: `CLUT01B/C/D/E` are **byte-identical** to
`CLT01B2/C2/D2/E2`, so variant 2 changes only the 8bpp body colour and inherits every
panel and glass palette from the base. Variant 3 differs across all five letters.

The `CLT01x` sets cover the bodywork but **not the door number panel**. `DR01A` is a
separate 8bpp tile with a single CLUT of its own, and decoding it against a `CLUTnnA`
body palette produces garbage — so at first all three variants shared the rookie's
yellow-on-white/red panel, which clashed badly with the pro's yellow/black body.

The missing palettes are `P1D1T2` and `P1D1T3` — "player 1, door 1, type 2/3" — two
8bpp palette-only records (tile position is the `(320,0)` staging slot, so only
`clut_x`/`clut_y` mean anything) present in all 11 track files. I had passed over them
earlier as unremarkable staging-slot entries. Decoding `DR01A` against them reproduces
each variant's colours exactly:

| variant | door palette | result |
|---|---|---|
| rookie | `DR01A`'s own | yellow 01 on white/red |
| amateur | `P1D1T2` | blue 01 on white/purple |
| pro | `P1D1T3` | blue 01 on yellow/black |

Each matches its body colours, which is the confirmation. There is no `P2D1Tx`, further
evidence that `CLT02x` is not a fourth variant of the player's car.

**T. A second CLT set exists and is not exported.** `CLT02A2–E2` / `CLT02A3–E3` appear
in LEV1, LEV4, LEV5, LEVA and LEVB (5 of 11 tracks) but have **no** `CLUT02x` base and
no `DR02A` door tile in those levels — `DR02A/B/C` exist only in LEV2. Most likely the
second car in two-player mode. Exporting it would need a decision about which door
panel it uses, so it is left out pending confirmation.

**M. Palette sourcing has three tiers, and one guess had to be thrown out.**
- The name-table record is the more authoritative CLUT source: in LEV0, 42 tiles carry
  no CLUT in their TX descriptor but do in the name table. In all other 13 levels the
  two sources agree on every tile. One genuine disagreement exists game-wide — LEV0's
  `MEMLOAD`, TX says (320,490), name table says (320,488) — reported as an advisory.
- `DRnnB`/`DRnnC` inherit `DRnnA`'s palette (38–40 per track level).
- **Rejected heuristic:** a first attempt matched palette-less tiles to any name-table
  sibling sharing a prefix. It resolved LEV0's 42 UI sprites and looked like a success,
  but was pairing `CARD` with an unrelated `CAR*` entry. Only the damage-variant rule is
  actually evidenced, so only that is applied; LEV6's `FLASH1-6` and `GOOSE1-8` (14
  tiles, the sole remaining gap) are admitted as unknown and dumped as index maps.

**H. The name table does not cover most of VRAM.** In LEV1 it names 194 resident
tiles occupying 134,776 of VRAM's 524,288 halfwords. Track surface textures are
unnamed and reachable only through UV/tpage. Consequence for M1: track texturing
must build real VRAM from TX0–TX3; the name table alone is not enough.

**D. The car/livery incompatibility is resolved — build cars from a TRACK level, not LEV0.**
LEV0's texture table contains only flat selection-screen sprites (`DRIVER0`…`DRIVER19`,
`SMLDRIV`, `VIEWDRIV`). The actual 3D car assets are exclusively in the track levels:
LEV1 alone carries `BUMP88A–E`, `BON88A–C`, `BOOT88A–C`, `FRNT88A`, `FRWN88A`,
`BKWN88A–E`, `DR00A`–`DR99C` (60 number panels) and `CLUT00A`–`CLUT99E`.
LEV1 `ptr[16]` (102 v, 44 tri + 64 quad) and `ptr[17]` (107 v, 96 tri + 51 quad) are the
only car-sized textured models. **This means no cross-system palette mapping is needed
at all** — geometry, tiles, CLUTs and number panels all come from one consistent source.
The dead end from last session was caused by starting from LEV0.

**E. Track terrain is a nested chunk table.** `ptr[0]` begins with `u32 0x70` (= table
size, 28 entries), followed by 28 `u32` offsets relative to `ptr[0]`. Chunks are
4.2–7.5 KB each. They are **not** `0x2C`-header model blocks — the chunk format is the
single remaining unknown.

### 1.4 Known-good format specs (carried forward, to be re-validated by asserts)

Model block header (`0x2C` bytes), used by ptr[5…28] and LEV0 car LODs:

```
+0x00 u32 total            +0x10 u16 tri_count      +0x20 u32 vertex_offset
+0x0a u16 vertex_count     +0x12 u16 quad_count     +0x24 u32 normal_offset
+0x0c u16 normal_count     +0x18 u16 tpage_a        +0x28 u32 polygon_offset
+0x0e u16 poly_total       +0x1a u16 tpage_b
```
Vertices and normals are `i16 x, y, z, pad` (8 B). Polygons are batched as
`u16 count, u8 type, u8 entry_size`, terminated by `entry_size == 0`.
Observed types in LEV1: `0x00, 0x01, 0x04, 0x05, 0x08, 0x09, 0x0c, 0x0d, 0x19, 0x1d`
— note `0x09`, `0x19` and `0x1d` are **not** in the old type table and must be decoded.

---

## 2. Open unknowns (ranked by risk)

| # | Unknown | Risk | Approach |
|---|---|---|---|
| U1 | ptr[0] terrain chunk format | **High** — blocks all track output | Ghidra: find the consumer of `DAT_80079a50[0]`; cross-check against ptr[2] point grid, which is almost certainly the shared vertex pool the chunks index into |
| ~~U2~~ | ~~Polygon types `0x09 / 0x19 / 0x1d`~~ | — | **Resolved** — one layout rule covers all 24 observed types; see §1.3 F |
| ~~U3~~ | ~~Which LEV1 section is the car~~ | — | **Resolved** — sections 16 and 17; see §1.3 G |
| U4 | Prop world placement (ptr[1] / ptr[2]) | Medium | Needed to place props in the track GLB; ptr[2] is confirmed to hold world-space `i32` XYZ |
| U5 | Wheel model | Low | **Not** among LEV1's 18 model blocks. No `WHEEL`/`TYRE` tile is referenced by any track model. Check LEV0 section 11, or the wheel may be a sprite |
| U6 | ptr[2] exact stride (25260 B ÷ 12 = 2105, not a clean grid) | Low | Check for a leading header word |
| U7 | Model header fields `unknown_00/_06/_08/_18/_1a/_1c`, and the polygon attribute byte at +0x01 | Low | Dumped per model in the reports; look for a pattern as more models are understood |

---

## 3. Target architecture

Pure-Python `uv` project. **No Panda3D.** Headless CLI, deterministic output.

```
extraction-derby-2/
├── pyproject.toml            # deps: pillow, numpy, pygltflib
├── main.py                   # thin CLI entry
├── src/dd2/
│   ├── io.py                 # binary readers, struct helpers, bounds-checked cursor
│   ├── level.py              # LEVEL.DAT: 29-ptr table, section dispatch
│   ├── model.py              # 0x2C model block → vertices / normals / polygons
│   ├── polygons.py           # polygon command-stream decoder (type table)
│   ├── texnames.py           # ptr[3] texture name table
│   ├── uvtable.py            # ptr[4] UV table + TPage decode
│   ├── vram.py               # PSX VRAM model, TX0-TX3 tile upload, BGR555→RGBA
│   ├── txc.py                # TXC CLUT parsing + palette upload
│   ├── liveries.py           # car number → CLUT set + DRnn number panels
│   ├── terrain.py            # ptr[0] chunk decoder  (U1)
│   ├── atlas.py              # per-model texture atlas builder + UV remap
│   ├── glb.py                # glTF 2.0 writer, textures embedded in the GLB
│   └── cli/
│       ├── extract_cars.py
│       ├── extract_tracks.py
│       └── dump_info.py      # diagnostics: pointer maps, tile lists, name tables
└── output/
    ├── cars/      car_00.glb … car_99.glb          (20 files)
    ├── tracks/    lev1.glb … levb.glb              (11 files)
    └── textures/  *.png                            (side product of atlas building)
```

**Design rules**

- Every parser asserts its size invariants (e.g. UV table size, TXC stream length) and
  fails loudly rather than producing silent garbage.
- No hard-coded absolute paths; game data root is a CLI argument.
- `dump_info` is written first and used as the debugging instrument throughout.
- Texture atlas per exported model, nearest-neighbour sampling, `CLAMP_TO_EDGE`,
  alpha from BGR555 index-0 transparency.

---

## 4. Milestones

**M0 — Project scaffold** — ✅ **done**
`dd2/binio.py` (bounds-checked reads), `dd2/dirinfo.py` (archive, byte-identical to
the known-good extraction on all 110 files), `dd2/level.py` (29-pointer container),
`dd2/texnames.py`, `dd2/uvtable.py`, `dd2/workspace.py` (guarded output wipe),
`dd2/logs.py`, `dd2/report.py`. All 14 level files parse and validate clean.
Reports land in `output/logs/`.

**M1 — VRAM + texture pipeline** — ✅ **done**
`dd2/vram.py` (1024×512 halfword framebuffer, BGR555→RGBA), `dd2/txfiles.py`
(TX0–TX3 tile archive + TXC palette archive), `dd2/textures.py` (assembly and PNG
export). **4862 PNGs — every tile in every level**, plus a full-VRAM dump per level.
Livery CLUT swapping verified visually across all 20 cars. See §1.3 I–M.

Output layout per level, with every archive tile landing in exactly one directory
so nothing can be silently dropped:

```
output/textures/LEVn/
    _vram.png       whole framebuffer, for spotting gaps at a glance
    named/          tiles the name table names (car parts, UI)
    unnamed/        road surface, hoardings, grandstands, crowd, sky, props
    unpaletted/     greyscale index maps where the palette is unknown
```

Palette resolution order (`LevelTextures.resolve_clut`): name-table record →
TX descriptor → damage-variant sibling → give up.

**M2 — Model decoder** — ✅ **done** (ahead of M1)
`dd2/model.py`: `0x2C` header, vertex/normal arrays, polygon command stream, and the
unified entry layout rule. All 197 blocks / 6532 polygons parse and validate.
Reports now list per-model bounds, polygon type usage, referenced CLUTs and the
named tiles each model samples. Remaining: emit untextured GLB to confirm winding
and scale visually.

**M3 — Cars (primary deliverable #1)** — ✅ **done**
`dd2/glb.py` (glTF writer, textures embedded), `dd2/carmodel.py` (livery logic),
`tools/render_glb.py` (software renderer for verification).
Deliverable: **22 textured GLBs in `output/cars/`** — 19 opponents plus the
player car's three class variants — each 5 nodes (body + 4 wheels),
465 vertices / 294 triangles. See §1.3 N–T.

**M4 — Terrain format**
Crack U1 via Ghidra + ptr[2] correlation. Deliverable: a documented chunk spec in
`docs/FORMAT.md` and a raw point cloud / wireframe dump proving the decode.

**M5 — Tracks (primary deliverable #2)**
Merge the 28 chunks into one `terrain` mesh, texture it, and emit each prop as its own
named object placed via U4. Deliverable: **11 textured GLBs in `output/tracks/`.**

**M6 — Verification**
Automated checks: manifold-ish sanity (no degenerate tris, UVs in `[0,1]`, no NaNs),
GLB validator pass, per-car palette diff so two cars never come out identical,
and a rendered turntable PNG per model for eyeball review.

---

## 5. Immediate next steps

1. M0 scaffold + `dump_info` across LEV0–LEVF.
2. Resolve U3 (which LEV1 pointer is the car) — cheap, and it de-risks M3.
3. Resolve U2 (polygon types `0x09/0x19/0x1d`) — blocks M2.
4. Start the Ghidra dig for U1 in parallel, since it is the long pole for tracks.
