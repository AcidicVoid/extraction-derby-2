"""
report.py — human-readable dumps of what we parsed.

These files are written to <output>/logs. They are the primary instrument for
checking format assumptions: run the extractor, read the report, and diff it
against the previous run when something changes.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .level import LevelFile
from .model import layout_for
from .texnames import CANONICAL_CAR_NUMBERS, TexName

RULE = "=" * 78
THIN = "-" * 78


def _fmt_section_table(level: LevelFile) -> list[str]:
    lines = ["Section table (29 uint32 offsets at file start)", THIN,
             f"{'idx':>3}  {'name':<12} {'offset':>10} {'size':>10}  state"]
    for s in level.sections:
        state = "" if s.present else "absent"
        lines.append(f"{s.index:>3}  {s.name:<12} 0x{s.offset:08X} "
                     f"{s.size:>10}  {state}")
    return lines


def _fmt_texnames(level: LevelFile) -> list[str]:
    table = level.tex_names
    resident = table.resident()
    non_resident = table.non_resident()

    lines = ["", RULE, f"Texture name table — {len(table)} records", THIN,
             f"resident in VRAM     : {len(resident)}",
             f"non-resident         : {len(non_resident)} "
             f"(staging slot or palette-only)",
             f"VRAM halfwords used  : {table.vram_halfwords_used()} "
             f"of {1024 * 512}"]

    bpp = Counter(r.bpp for r in table)
    lines.append(f"bit depths           : "
                 + ", ".join(f"{k}bpp x{v}" for k, v in sorted(bpp.items())))

    cars = table.car_numbers()
    if cars:
        parts = sorted({r.part for r in table if r.part})
        lines += [
            f"car numbers present  : {len(cars)} -> {', '.join(cars)}",
            f"car asset prefixes   : {', '.join(parts)}",
            f"shared body tiles    : {len(table.body_tiles())} "
            f"({', '.join(sorted(r.name for r in table.body_tiles()))})",
            f"door number panels   : {len(table.number_panels())}",
            f"player alt liveries  : {len(table.player_alt_liveries())}",
        ]
        # Flag any deviation from the 20-car grid roster. LEV2 legitimately
        # carries an extra DR02A/B/C door panel set; anything else is a
        # signal that our naming model is incomplete.
        extra = sorted(set(cars) - set(CANONICAL_CAR_NUMBERS))
        missing = sorted(set(CANONICAL_CAR_NUMBERS) - set(cars))
        if extra:
            lines.append(f"  NOTE non-roster numbers: {', '.join(extra)} "
                         f"-> " + ", ".join(
                             sorted(r.name for r in table
                                    if r.car_number in extra)))
        if missing:
            lines.append(f"  NOTE roster cars absent: {', '.join(missing)}")

    def block(title: str, records: list[TexName]) -> list[str]:
        out = ["", f"-- {title} ({len(records)}) " + "-" * max(0, 60 - len(title))]
        out += [f"  [{r.index:>3}] {r}" for r in records]
        return out

    lines += block("resident tiles", resident)
    lines += block("non-resident records", non_resident)
    return lines


def _fmt_uvtable(level: LevelFile) -> list[str]:
    table = level.uv_table
    lines = ["", RULE, f"UV table — {len(table)} records", THIN]
    if not table:
        lines.append("  (section absent)")
        return lines

    hist = table.tpage_histogram()
    lines.append(f"distinct texture pages: {len(hist)}")
    for raw, count in hist.items():
        tp = table.records[0].tpage.__class__.decode(raw)
        lines.append(f"  tpage 0x{raw:04X}  base=({tp.x_base:>4},{tp.y_base:>3}) "
                     f"{tp.bpp:>2}bpp   used by {count} record(s)")

    lines += ["", "-- first 64 records " + "-" * 58]
    lines += [f"  {r}" for r in table.records[:64]]
    if len(table) > 64:
        lines.append(f"  ... {len(table) - 64} more (see level JSON for full data)")
    return lines


def _fmt_models(level: LevelFile) -> list[str]:
    models = level.models
    total_polys = sum(len(m.polygons) for m in models.values())
    total_verts = sum(len(m.vertices) for m in models.values())

    lines = ["", RULE,
             f"Model blocks — {len(models)} present, "
             f"{total_verts} vertices, {total_polys} polygons", THIN,
             f"{'sec':>3}  {'offset':>9} {'bytes':>7} {'verts':>6} "
             f"{'norms':>6} {'polys':>6} {'tri':>5} {'quad':>5} {'texd':>5}  "
             f"bounds (x,y,z)"]

    for index in sorted(models):
        m = models[index]
        lo, hi = m.bounds()
        lines.append(
            f"{index:>3}  0x{level.section(index).offset:07X} {m.size:>7} "
            f"{len(m.vertices):>6} {len(m.normals):>6} {len(m.polygons):>6} "
            f"{m.triangle_count:>5} {m.quad_count:>5} "
            f"{m.textured_polygon_count:>5}  "
            f"[{lo[0]},{lo[1]},{lo[2]}]..[{hi[0]},{hi[1]},{hi[2]}]"
        )

    # Polygon type usage across the whole level. A type we have never seen
    # before would show up here first.
    type_hist: Counter = Counter()
    for m in models.values():
        type_hist.update(m.batch_types)
    if type_hist:
        lines += ["", "-- polygon types used " + "-" * 56,
                  f"{'type':>6} {'layout':<22} {'size':>5} {'count':>7}"]
        for type_id, count in sorted(type_hist.items()):
            lay = layout_for(type_id)
            lines.append(f"  0x{type_id:02X} {lay.name:<22} "
                         f"{lay.entry_size:>5} {count:>7}")

    # Which named tiles each model samples. This is how a model is
    # identified: the car is whichever block references the shared body-part
    # tiles. Most track geometry samples unnamed VRAM and shows up as
    # "(unnamed VRAM)".
    if level.uv_table:
        lines += ["", "-- named tiles sampled per model " + "-" * 45]
        for index in sorted(models):
            m = models[index]
            hits: Counter = Counter()
            unnamed = 0
            for poly in m.polygons:
                if poly.uv_index is None:
                    continue
                rec = level.uv_table[poly.uv_index]
                names = level.tex_names.tiles_for_uv(
                    rec.tpage, rec.uvs[:poly.corners])
                if names:
                    hits.update(names)
                else:
                    unnamed += 1
            if not hits and not unnamed:
                summary = "(untextured)"
            elif not hits:
                summary = f"(unnamed VRAM only, {unnamed} polys)"
            else:
                summary = ", ".join(f"{n} x{c}" for n, c in hits.most_common())
                if unnamed:
                    summary += f", (unnamed VRAM) x{unnamed}"
            lines.append(f"  sec {index:>2}: {summary}")

    # Distinct CLUT descriptors referenced by textured polygons — this is the
    # bridge from geometry to the per-car livery palettes.
    cluts = set()
    for m in models.values():
        cluts |= m.clut_ids()
    if cluts:
        lines += ["", f"-- distinct CLUTs referenced by polygons "
                      f"({len(cluts)}) " + "-" * 30]
        for cid in sorted(cluts):
            lines.append(f"  0x{cid:04X} -> vram({(cid & 0x3F) * 16},"
                         f"{cid >> 6})")

    # Header fields we have not identified. Listed so a pattern can be spotted
    # later rather than being quietly dropped.
    lines += ["", "-- unidentified header fields " + "-" * 48,
              f"{'sec':>3}  " + "  ".join(f"{k:>10}" for k in
                                          _UNKNOWN_KEYS)]
    for index in sorted(models):
        u = models[index].unknown
        lines.append(f"{index:>3}  " + "  ".join(
            f"{u[k]:>10}" for k in _UNKNOWN_KEYS))

    return lines


_UNKNOWN_KEYS = ("unknown_00", "unknown_06", "unknown_08",
                 "unknown_18", "unknown_1a", "unknown_1c")


def _fmt_validation(level: LevelFile) -> tuple[list[str], int]:
    lines = ["", RULE, "Validation", THIN]
    total = 0
    model_problems: list[str] = []
    uv_size = len(level.uv_table) if level.uv_table else None
    for m in level.models.values():
        model_problems += m.validate(uv_table_size=uv_size)

    for label, problems in (
        ("container", level.validate()),
        ("texture name table", level.tex_names.validate()),
        ("uv table", level.uv_table.validate()),
        ("model blocks", model_problems),
    ):
        if problems:
            total += len(problems)
            lines.append(f"  {label}: {len(problems)} PROBLEM(S)")
            lines += [f"      ! {p}" for p in problems]
        else:
            lines.append(f"  {label}: OK")
    return lines, total


def write_level_report(level: LevelFile, dest: Path, title: str) -> int:
    """
    Write a full text report for one level file.
    Returns the number of validation problems found.
    """
    lines = [RULE, f"{title}  —  {level.name}",
             f"size {len(level.data)} bytes  |  "
             f"{'TRACK' if level.is_track else 'non-track'}", RULE, ""]
    lines += _fmt_section_table(level)
    lines += _fmt_models(level)
    lines += _fmt_texnames(level)
    lines += _fmt_uvtable(level)
    v_lines, problem_count = _fmt_validation(level)
    lines += v_lines
    lines.append("")

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines), encoding="utf-8")
    return problem_count


def write_texture_index(summaries: dict[str, dict], dest: Path) -> None:
    """Overview of each level's assembled VRAM."""
    lines = [RULE, "Texture / VRAM overview", RULE, "",
             f"{'level':<6} {'tiles':>6} {'noclut':>7} {'txc':>5} "
             f"{'txcslack':>9} {'vram written':>13} {'coverage':>9} "
             f"{'named':>6}"]
    for key in sorted(summaries):
        s = summaries[key]
        pct = 100.0 * s["vram_halfwords_written"] / s["vram_halfwords_total"]
        lines.append(
            f"{key:<6} {s['tiles']:>6} {s['tiles_without_clut']:>7} "
            f"{s['txc_uploads']:>5} {s['txc_slack']:>9} "
            f"{s['vram_halfwords_written']:>13} {pct:>8.1f}% "
            f"{s['named_records']:>6}"
        )
    lines += ["", THIN,
              "noclut   = tiles whose descriptor sets clut_y to 0xFFFE; their",
              "           palette is supplied by another upload.",
              "txcslack = unreferenced trailing bytes in LEVEL.TXC. Three",
              "           retail levels ship filler palettes here.",
              "coverage = fraction of the 1024x512 halfword framebuffer that",
              "           received an upload."]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_index(levels: dict[str, LevelFile], dest: Path) -> None:
    """Write a one-line-per-level overview plus a machine-readable JSON."""
    lines = [RULE, "LEVEL.DAT overview", RULE, "",
             f"{'level':<6} {'size':>9} {'kind':<7} {'models':>7} "
             f"{'verts':>7} {'polys':>7} {'texd':>6} {'texnames':>9} "
             f"{'uvrecs':>7}"]
    for key in sorted(levels):
        lv = levels[key]
        models = lv.models.values()
        lines.append(
            f"{key:<6} {len(lv.data):>9} "
            f"{'track' if lv.is_track else 'other':<7} {len(lv.models):>7} "
            f"{sum(len(m.vertices) for m in models):>7} "
            f"{sum(len(m.polygons) for m in models):>7} "
            f"{sum(m.textured_polygon_count for m in models):>6} "
            f"{len(lv.tex_names):>9} {len(lv.uv_table):>7}"
        )

    # Which car numbers appear in which levels — needed for the car milestone.
    lines += ["", THIN, "Car numbers referenced by each level", THIN]
    for key in sorted(levels):
        cars = levels[key].tex_names.car_numbers()
        if cars:
            lines.append(f"{key:<6} {len(cars):>2} cars: {', '.join(cars)}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    json_dest = dest.with_suffix(".json")
    payload = {key: lv.summary() for key, lv in sorted(levels.items())}
    json_dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
