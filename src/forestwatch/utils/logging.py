"""Logger helper — format konsisten antar modul."""

from __future__ import annotations

import logging
import os
import sys

_LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str = "forestwatch", level: str | int | None = None) -> logging.Logger:
    """Return logger yang sudah dikonfigurasi (idempoten)."""
    logger = logging.getLogger(name)
    if logger.handlers:
        # Sudah dikonfigurasi sebelumnya
        return logger
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level or _LOG_LEVEL)
    logger.propagate = False
    return logger
