"""
workspace.py — output directory management.

The output directory is wiped at the start of every run so results can never
be a mix of old and new. Because wiping a directory is destructive, we refuse
to touch anything we did not create: a marker file identifies directories that
belong to us.
"""

from __future__ import annotations

import shutil
from pathlib import Path

MARKER_NAME = ".extraction-derby-2"
MARKER_TEXT = (
    "This directory is managed by extraction-derby-2.\n"
    "It is deleted and recreated on every run. Do not store anything here.\n"
)


class UnsafeOutputDir(Exception):
    """Raised when the output path exists but was not created by us."""


def prepare(output_dir: Path) -> Path:
    """
    Clear and recreate `output_dir`.

    Safety rule: if the directory already exists, is non-empty, and has no
    marker file, we refuse rather than delete a directory that might be the
    user's own data.
    """
    output_dir = Path(output_dir)

    if output_dir.exists():
        if not output_dir.is_dir():
            raise UnsafeOutputDir(
                f"{output_dir} exists and is not a directory")

        has_marker = (output_dir / MARKER_NAME).is_file()
        is_empty = not any(output_dir.iterdir())

        if not has_marker and not is_empty:
            raise UnsafeOutputDir(
                f"{output_dir} is not empty and carries no "
                f"'{MARKER_NAME}' marker. Refusing to delete it. "
                f"Point --output somewhere else, or remove the directory "
                f"yourself if it is safe to do so."
            )

        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True)
    (output_dir / MARKER_NAME).write_text(MARKER_TEXT, encoding="utf-8")
    return output_dir
