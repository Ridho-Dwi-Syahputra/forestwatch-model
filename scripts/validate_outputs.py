"""Thin wrapper untuk ``forestwatch.cli.validate``.

Penggunaan:
    python scripts/validate_outputs.py --dir outputs/dummy
"""

from __future__ import annotations

import sys

from forestwatch.cli.validate import main

if __name__ == "__main__":
    sys.exit(main())
