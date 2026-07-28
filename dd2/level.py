"""
level.py — the LEVEL.DAT container.

Every LEVEL.DAT in the game — the frontend (LEV0), the eleven race tracks
(LEV1-LEVB) and the two small auxiliary files (LEVC, LEVF) — uses the same
container: a fixed table of 29 uint32 offsets at the very start of the file,
followed by the sections those offsets point at.

    +0x00  u32[29]  section offsets, relative to file start
    +0x74           section data

The file is NOT compressed. Two independent confirmations:

  1. In every one of the 14 retail files the 29 offsets are monotonically
     non-decreasing and all land inside the file.
  2. The game's loader FUN_80042a48 does nothing but add the load address to
     each of the 29 words in place. There is no decompression call anywhere
     in the LEVEL.DAT load path.

The prior project assumed LZSS compression here. It was wrong, and the
LEVEL_DECOMP.BIN files it produced are meaningless.

Section roles
-------------
A section is "absent" when its offset equals the next one, i.e. it spans zero
bytes. LEV0 leaves the three track sections absent; the track files use all of
them. This means one parser covers every file.

    0   terrain      nested table of terrain geometry chunks (tracks only)
    1   track_data   track section/spline records (tracks only)
    2   point_grid   world-space i32 XYZ points on a coarse grid (tracks only)
    3   tex_names    texture name table            -> texnames.py
    4   uv_table     UV coordinate table           -> uvtable.py
    5+  model_NN     model blocks: props, cars, wheels, LODs
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .binio import FormatError, u32_array
from .model import ModelBlock, parse_model_block
from .texnames import TextureNameTable
from .uvtable import UVTable

POINTER_COUNT = 29
HEADER_SIZE = POINTER_COUNT * 4      # 0x74

# Fixed roles for the first five sections; everything from 5 up is a model.
SECTION_TERRAIN = 0
SECTION_TRACK_DATA = 1
SECTION_POINT_GRID = 2
SECTION_TEX_NAMES = 3
SECTION_UV_TABLE = 4
FIRST_MODEL_SECTION = 5

_SECTION_NAMES = {
    SECTION_TERRAIN: "terrain",
    SECTION_TRACK_DATA: "track_data",
    SECTION_POINT_GRID: "point_grid",
    SECTION_TEX_NAMES: "tex_names",
    SECTION_UV_TABLE: "uv_table",
}


def section_name(index: int) -> str:
    return _SECTION_NAMES.get(index, f"model_{index - FIRST_MODEL_SECTION:02d}")


@dataclass(frozen=True)
class Section:
    """One slice of a LEVEL.DAT file."""

    index: int
    name: str
    offset: int
    size: int

    @property
    def end(self) -> int:
        return self.offset + self.size

    @property
    def present(self) -> bool:
        return self.size > 0

    def __str__(self) -> str:
        state = f"0x{self.offset:06X} +{self.size:<8d}" if self.present \
            else f"0x{self.offset:06X} (absent)"
        return f"[{self.index:>2}] {self.name:<12} {state}"


class LevelFile:
    """A parsed LEVEL.DAT container."""

    def __init__(self, data: bytes, name: str = "LEVEL.DAT"):
        self.data = data
        self.name = name

        if len(data) < HEADER_SIZE:
            raise FormatError(
                f"{name}: file is {len(data)} bytes, too small for a "
                f"{HEADER_SIZE}-byte pointer table"
            )

        self.pointers: tuple[int, ...] = u32_array(data, 0, POINTER_COUNT)
        self.sections: list[Section] = self._build_sections()

        # Lazily parsed sub-tables; see the properties below.
        self._tex_names: TextureNameTable | None = None
        self._uv_table: UVTable | None = None
        self._models: dict[int, ModelBlock] | None = None
        self._terrain = None

    # -- construction -------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path) -> "LevelFile":
        path = Path(path)
        return cls(path.read_bytes(), name=path.name)

    def _build_sections(self) -> list[Section]:
        """
        Turn the 29 offsets into explicit (offset, size) slices. A section
        runs until the next pointer, and the last runs to end of file.
        """
        bounds = list(self.pointers) + [len(self.data)]
        sections: list[Section] = []
        for i in range(POINTER_COUNT):
            size = bounds[i + 1] - bounds[i]
            sections.append(Section(index=i, name=section_name(i),
                                    offset=bounds[i], size=size))
        return sections

    # -- access -------------------------------------------------------------

    def section(self, index: int) -> Section:
        return self.sections[index]

    def section_bytes(self, index: int) -> bytes:
        s = self.sections[index]
        return self.data[s.offset:s.end]

    @property
    def model_sections(self) -> list[Section]:
        """Sections that hold model blocks, present ones only."""
        return [s for s in self.sections[FIRST_MODEL_SECTION:] if s.present]

    @property
    def is_track(self) -> bool:
        """Tracks are the files that carry terrain geometry."""
        return self.sections[SECTION_TERRAIN].present

    @property
    def tex_names(self) -> TextureNameTable:
        if self._tex_names is None:
            s = self.sections[SECTION_TEX_NAMES]
            self._tex_names = TextureNameTable(self.data, s.offset, s.size)
        return self._tex_names

    @property
    def uv_table(self) -> UVTable:
        if self._uv_table is None:
            s = self.sections[SECTION_UV_TABLE]
            self._uv_table = UVTable(self.data, s.offset, s.size)
        return self._uv_table

    @property
    def models(self) -> dict[int, ModelBlock]:
        """
        Parsed model blocks, keyed by section index. Only present sections
        appear. Parsed once and cached.
        """
        if self._models is None:
            self._models = {
                s.index: parse_model_block(self.data, s.offset, s.size,
                                           name=f"{self.name}:{s.name}")
                for s in self.model_sections
            }
        return self._models

    @property
    def terrain(self) -> "Terrain":
        """
        Section 0 decoded: the track's positioned terrain meshes.
        Empty for non-track files, where section 0 spans zero bytes.
        """
        if self._terrain is None:
            from .terrain import Terrain
            s = self.sections[SECTION_TERRAIN]
            self._terrain = Terrain(self.data, s.offset, s.size,
                                    name=f"{self.name}:terrain")
        return self._terrain

    def model(self, section_index: int) -> ModelBlock:
        try:
            return self.models[section_index]
        except KeyError:
            raise FormatError(
                f"{self.name}: section {section_index} holds no model block"
            ) from None

    # -- validation ---------------------------------------------------------

    def validate(self) -> list[str]:
        """
        Container-level checks. These are the invariants that prove the file
        is an uncompressed LEVEL.DAT and that our slicing is sound.
        """
        problems: list[str] = []

        if self.pointers[0] != HEADER_SIZE:
            problems.append(
                f"pointer[0] is 0x{self.pointers[0]:X}, expected "
                f"0x{HEADER_SIZE:X} (section data must start after the table)"
            )

        for i in range(POINTER_COUNT - 1):
            if self.pointers[i] > self.pointers[i + 1]:
                problems.append(
                    f"pointer[{i}]=0x{self.pointers[i]:X} > "
                    f"pointer[{i + 1}]=0x{self.pointers[i + 1]:X} "
                    f"(table must be non-decreasing)"
                )

        for i, p in enumerate(self.pointers):
            if p > len(self.data):
                problems.append(
                    f"pointer[{i}]=0x{p:X} points past end of file "
                    f"(0x{len(self.data):X})"
                )

        return problems

    def summary(self) -> dict:
        """Compact machine-readable description, handy for regression diffs."""
        return {
            "name": self.name,
            "size": len(self.data),
            "is_track": self.is_track,
            "sections": [
                {"index": s.index, "name": s.name,
                 "offset": s.offset, "size": s.size}
                for s in self.sections
            ],
            "tex_name_count": len(self.tex_names),
            "uv_record_count": len(self.uv_table),
            "model_section_count": len(self.model_sections),
        }
