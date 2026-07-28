"""
dirinfo.py — DIRINFO archive container.

DIRINFO is the single monolithic archive shipped on the DD2 disc. It is
self-describing: a directory table sits at the very front of the file and the
payload of every entry lives further inside the *same* file, addressed by CD
sector number.

Layout
------
    offset 0x0000   directory table: N x 24-byte entries, terminated by an
                    entry whose 18-byte name field is all NUL
    sector 0..      payload data, each entry starting on a 2048-byte boundary

Directory entry (24 bytes)
--------------------------
    +0x00  char[18]  name, NUL-padded, backslash-separated sub-paths
    +0x12  u16       start sector (byte offset = sector * 2048)
    +0x14  u32       size in bytes

Verified against the retail disc image: 110 entries, no overlapping payloads,
no entry reaching past the end of the file, and the entries tile the file with
only 2028 bytes of trailing slack. The u16 sector field is sufficient because
the highest sector used is 7906.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .binio import FormatError, cstring, u16, u32

SECTOR_SIZE = 2048
ENTRY_SIZE = 24
NAME_FIELD_SIZE = 18


@dataclass(frozen=True)
class DirEntry:
    """One file inside the DIRINFO archive."""

    index: int
    name: str      # original name, e.g. "VAGS\\BANK1.SBK"
    sector: int
    size: int

    @property
    def offset(self) -> int:
        """Byte offset of this entry's payload within the DIRINFO file."""
        return self.sector * SECTOR_SIZE

    @property
    def path_parts(self) -> tuple[str, ...]:
        """
        Name split into portable path components.

        The archive stores Windows-style separators; normalise so extraction
        behaves identically on every host OS.
        """
        return tuple(p for p in self.name.replace("\\", "/").split("/") if p)

    @property
    def relative_path(self) -> Path:
        return Path(*self.path_parts)

    def __str__(self) -> str:
        return f"{self.name} (sector {self.sector}, {self.size} bytes)"


class DirInfo:
    """Parsed DIRINFO archive, held in memory."""

    def __init__(self, data: bytes, source: Path | None = None):
        self.data = data
        self.source = source
        self.entries: list[DirEntry] = self._parse_directory(data)
        self._by_name: dict[str, DirEntry] = {
            "/".join(e.path_parts): e for e in self.entries
        }

    # -- construction -------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path) -> "DirInfo":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"DIRINFO not found: {path}")
        return cls(path.read_bytes(), source=path)

    @staticmethod
    def _parse_directory(data: bytes) -> list[DirEntry]:
        """
        Walk the directory table until the NUL terminator entry.

        The table has no explicit count. We stop at the first entry whose name
        field is entirely NUL, which is how the game's own loader finds the end.
        """
        entries: list[DirEntry] = []
        offset = 0
        index = 0

        while offset + ENTRY_SIZE <= len(data):
            name_field = data[offset:offset + NAME_FIELD_SIZE]
            if name_field == b"\0" * NAME_FIELD_SIZE:
                break  # terminator reached

            name = cstring(data, offset, NAME_FIELD_SIZE)
            sector = u16(data, offset + 0x12)
            size = u32(data, offset + 0x14)

            entries.append(DirEntry(index=index, name=name,
                                    sector=sector, size=size))
            offset += ENTRY_SIZE
            index += 1
        else:
            # Loop ran off the end of the buffer without hitting a terminator.
            raise FormatError(
                "DIRINFO directory table has no terminator entry — "
                "this file is probably not a DIRINFO archive"
            )

        if not entries:
            raise FormatError("DIRINFO directory table is empty")

        return entries

    # -- access -------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def read(self, entry: DirEntry) -> bytes:
        """Return the raw payload bytes for one entry."""
        start = entry.offset
        end = start + entry.size
        if end > len(self.data):
            raise FormatError(
                f"entry '{entry.name}' payload ends at 0x{end:X}, "
                f"past end of archive (0x{len(self.data):X})"
            )
        return self.data[start:end]

    def get(self, name: str) -> bytes:
        """
        Read an entry by name. Accepts either separator style,
        e.g. get("LEV1/LEVEL.DAT") or get("LEV1\\LEVEL.DAT").
        """
        key = name.replace("\\", "/").strip("/")
        entry = self._by_name.get(key)
        if entry is None:
            raise KeyError(f"no such entry in DIRINFO: {name}")
        return self.read(entry)

    def has(self, name: str) -> bool:
        return name.replace("\\", "/").strip("/") in self._by_name

    def names(self) -> list[str]:
        return list(self._by_name.keys())

    # -- validation ---------------------------------------------------------

    def validate(self) -> list[str]:
        """
        Structural self-check. Returns a list of human-readable problems;
        an empty list means the archive is internally consistent.

        These checks are what let us trust the container before we start
        making claims about the files inside it.
        """
        problems: list[str] = []
        table_end = len(self.entries) * ENTRY_SIZE + ENTRY_SIZE  # incl. terminator

        for entry in self.entries:
            if entry.size == 0:
                problems.append(f"{entry.name}: zero-length entry")
            if entry.offset + entry.size > len(self.data):
                problems.append(
                    f"{entry.name}: payload 0x{entry.offset:X}+0x{entry.size:X} "
                    f"exceeds archive size 0x{len(self.data):X}"
                )
            if entry.offset < table_end:
                problems.append(
                    f"{entry.name}: payload at 0x{entry.offset:X} overlaps "
                    f"the directory table (ends 0x{table_end:X})"
                )

        # Payloads must not overlap each other. Sort by offset and walk.
        ordered = sorted(self.entries, key=lambda e: e.offset)
        for prev, cur in zip(ordered, ordered[1:]):
            prev_end = prev.offset + prev.size
            if cur.offset < prev_end:
                problems.append(
                    f"{cur.name} at 0x{cur.offset:X} overlaps "
                    f"{prev.name} ending at 0x{prev_end:X}"
                )

        # Duplicate names would make get() ambiguous.
        seen: set[str] = set()
        for entry in self.entries:
            key = "/".join(entry.path_parts)
            if key in seen:
                problems.append(f"{entry.name}: duplicate entry name")
            seen.add(key)

        return problems

    def coverage(self) -> tuple[int, int]:
        """
        Return (bytes accounted for by entries + table, archive size).

        A large unaccounted remainder would mean we are missing entries.
        """
        ordered = sorted(self.entries, key=lambda e: e.offset)
        used = len(self.entries) * ENTRY_SIZE + ENTRY_SIZE
        if ordered:
            last = ordered[-1]
            used = max(used, last.offset + last.size)
        return used, len(self.data)

    # -- extraction ---------------------------------------------------------

    def extract_all(self, dest_dir: str | Path,
                    overwrite: bool = True) -> list[Path]:
        """
        Write every entry to `dest_dir`, recreating the archive's directory
        structure. Returns the list of written paths.
        """
        dest_dir = Path(dest_dir)
        written: list[Path] = []

        for entry in self.entries:
            target = dest_dir / entry.relative_path
            if target.exists() and not overwrite:
                written.append(target)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self.read(entry))
            written.append(target)

        return written
