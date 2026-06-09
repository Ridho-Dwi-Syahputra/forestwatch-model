"""Pytest bootstrap — pastikan folder backend ada di sys.path agar `import app` jalan."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
