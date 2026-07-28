"""
logs.py — logging setup.

Console output is deliberately sparse: only high-level progress and real
problems. Everything detailed (pointer maps, per-tile dumps, validation
findings) goes to files under <output>/logs so it can be diffed between runs.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

LOGGER_NAME = "dd2"


def setup(log_dir: Path, verbose: bool = False) -> logging.Logger:
    """
    Configure the package logger.

    Console gets WARNING and above (plus whatever the CLI prints directly),
    or INFO when --verbose is given. The log file always gets DEBUG.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    file_handler = logging.FileHandler(log_dir / "extract.log",
                                       mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                          datefmt="%H:%M:%S")
    )
    logger.addHandler(file_handler)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.INFO if verbose else logging.WARNING)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(console)

    return logger


def get(name: str) -> logging.Logger:
    """Get a child logger, e.g. get('level') -> 'dd2.level'."""
    return logging.getLogger(f"{LOGGER_NAME}.{name}")
