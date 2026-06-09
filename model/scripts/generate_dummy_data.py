"""Thin wrapper untuk ``forestwatch.cli.dummy``.

Penggunaan:
    python scripts/generate_dummy_data.py --out outputs/dummy --n-polygons 60 --seed 42
"""

from __future__ import annotations

import sys

from forestwatch.cli.dummy import main

if __name__ == "__main__":
    sys.exit(main())
