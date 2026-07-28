"""
Extraction Derby 2 — command line entry point.

    uv run main.py <path_to_DIRINFO>

Unpacks the DIRINFO archive from the PS1 disc, parses every LEVEL.DAT it
contains, and validates the results.

Console output stays minimal. Detailed reports go to <output>/logs:

    logs/extract.log      run log
    logs/levels.txt       one-line overview of every level file
    logs/levels.json      the same, machine readable
    logs/levels/LEVn.txt  full per-level dump: sections, texture names,
                          UV table, validation findings

The output directory is wiped at the start of every run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dd2 import logs, report, workspace
from dd2.binio import FormatError
from dd2.dirinfo import DirInfo
from dd2.level import LevelFile
from dd2.textures import LevelTextures, export_all_tiles, export_named_tiles


def human(n: float) -> str:
    """Format a byte count compactly."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n} B"


def resolve_dirinfo(path: Path) -> Path:
    """Accept either the DIRINFO file itself or the folder containing it."""
    if path.is_dir():
        candidate = path / "DIRINFO"
        if not candidate.is_file():
            raise FileNotFoundError(
                f"{path} is a directory and contains no DIRINFO file")
        return candidate
    if not path.is_file():
        raise FileNotFoundError(f"DIRINFO not found: {path}")
    return path


def unpack_archive(dirinfo_path: Path, dest: Path) -> DirInfo:
    """Validate and unpack the archive. Raises on an inconsistent archive."""
    log = logs.get("dirinfo")
    archive = DirInfo.from_file(dirinfo_path)

    used, total = archive.coverage()
    log.info("archive %s: %d entries, %d bytes, %d unaccounted",
             dirinfo_path.name, len(archive), total, total - used)
    for entry in archive:
        log.debug("entry %3d  %-20s sector %5d  %9d bytes",
                  entry.index, entry.name, entry.sector, entry.size)

    problems = archive.validate()
    if problems:
        for p in problems:
            log.error("archive: %s", p)
        raise FormatError(
            f"DIRINFO failed validation with {len(problems)} problem(s); "
            f"see the log for detail")

    archive.extract_all(dest)
    log.info("unpacked %d files to %s", len(archive), dest)
    return archive


def parse_levels(gamedata_dir: Path, log_dir: Path) -> tuple[int, int]:
    """
    Parse every LEVEL.DAT under the unpacked game data and write reports.
    Returns (levels parsed, total validation problems).
    """
    log = logs.get("level")
    level_paths = sorted(gamedata_dir.glob("LEV*/LEVEL.DAT"))

    levels: dict[str, LevelFile] = {}
    problem_total = 0

    for path in level_paths:
        key = path.parent.name
        try:
            level = LevelFile.from_file(path)
        except FormatError as exc:
            log.error("%s: could not parse: %s", key, exc)
            problem_total += 1
            continue

        # Touch the lazy sub-tables now so parse errors surface here, where we
        # can attribute them to a specific file, rather than deep in a report.
        try:
            _ = level.tex_names, level.uv_table, level.models
        except FormatError as exc:
            log.error("%s: section parse failed: %s", key, exc)
            problem_total += 1
            continue

        levels[key] = level
        log.info("%s: %s, %d sections used, %d models "
                 "(%d verts, %d polys), %d texnames, %d uv recs",
                 key, "track" if level.is_track else "other",
                 sum(1 for s in level.sections if s.present),
                 len(level.models),
                 sum(len(m.vertices) for m in level.models.values()),
                 sum(len(m.polygons) for m in level.models.values()),
                 len(level.tex_names), len(level.uv_table))

        found = report.write_level_report(
            level, log_dir / "levels" / f"{key}.txt", title=key)
        if found:
            problem_total += found
            log.warning("%s: %d validation problem(s)", key, found)

    if levels:
        report.write_index(levels, log_dir / "levels.txt")

    return levels, problem_total


def extract_textures(gamedata_dir: Path, levels: dict[str, LevelFile],
                     out_dir: Path, log_dir: Path,
                     all_tiles: bool) -> tuple[int, int]:
    """
    Build each level's VRAM and export tiles as PNG.
    Returns (images written, validation problems).
    """
    log = logs.get("textures")
    written_total = 0
    problem_total = 0
    summaries: dict[str, dict] = {}

    for key, level in sorted(levels.items()):
        lev_dir = gamedata_dir / key
        try:
            textures = LevelTextures.from_dir(lev_dir, names=level.tex_names)
        except (FileNotFoundError, FormatError) as exc:
            log.error("%s: texture load failed: %s", key, exc)
            problem_total += 1
            continue

        problems = textures.validate()
        if problems:
            problem_total += len(problems)
            log.warning("%s: %d texture validation problem(s)",
                        key, len(problems))
            for p in problems:
                log.warning("%s: %s", key, p)

        info = textures.summary()
        summaries[key] = info
        log.info("%s: %d tiles, %d txc cluts, %d/%d vram halfwords written",
                 key, info["tiles"], info["txc_uploads"],
                 info["vram_halfwords_written"], info["vram_halfwords_total"])

        dest = out_dir / "textures" / key
        if all_tiles:
            n, skipped = export_all_tiles(textures, dest)
            log.info("%s: exported %d tile PNGs (%d skipped, no CLUT)",
                     key, n, skipped)
        else:
            n, skipped = export_named_tiles(textures, dest)
            log.info("%s: exported %d named tile PNGs (%d skipped, no CLUT)",
                     key, n, skipped)
        written_total += n

        # A full-VRAM dump makes gaps and stray uploads obvious at a glance.
        textures.vram.raw_image().save(dest / "_vram.png")

    report.write_texture_index(summaries, log_dir / "textures.txt")
    return written_total, problem_total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="extraction-derby-2",
        description="Extract assets from Destruction Derby 2 (PS1, 1996).",
    )
    parser.add_argument("dirinfo", type=Path,
                        help="path to the DIRINFO file from the game disc")
    parser.add_argument("-o", "--output", type=Path, default=Path("output"),
                        help="output directory, wiped on each run "
                             "(default: ./output)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="mirror the log to the console")
    parser.add_argument("--all-tiles", action="store_true",
                        help="export every texture tile, not just the named "
                             "ones (includes unnamed track surface texture)")
    parser.add_argument("--no-textures", action="store_true",
                        help="skip texture extraction")
    args = parser.parse_args(argv)

    try:
        dirinfo_path = resolve_dirinfo(args.dirinfo)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        output_dir = workspace.prepare(args.output)
    except workspace.UnsafeOutputDir as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    log_dir = output_dir / "logs"
    logs.setup(log_dir, verbose=args.verbose)

    gamedata_dir = output_dir / "gamedata"
    try:
        archive = unpack_archive(dirinfo_path, gamedata_dir)
    except (FileNotFoundError, FormatError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"unpacked  {len(archive)} files "
          f"({human(sum(e.size for e in archive))})")

    levels, problems = parse_levels(gamedata_dir, log_dir)
    print(f"parsed    {len(levels)} LEVEL.DAT files")

    if not args.no_textures and levels:
        images, tex_problems = extract_textures(
            gamedata_dir, levels, output_dir, log_dir, args.all_tiles)
        problems += tex_problems
        print(f"textures  {images} PNGs -> {output_dir / 'textures'}")

    if problems:
        print(f"WARNING   {problems} validation problem(s) — see {log_dir}")
        return 1

    print("validated all sections OK")
    print(f"logs      {log_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
