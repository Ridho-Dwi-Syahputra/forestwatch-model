"""ForestWatch Papua — segmentasi tutupan lahan & deteksi deforestasi Papua.

Package ini mengimplementasikan pipeline penuh dari PRD v2.0:
  1. Komposit Sentinel-2 bebas awan + label fusion 6 aturan (modul ``gee``)
  2. Patch extraction + PyTorch dataset (modul ``data``)
  3. ResNet50-U-Net + combined loss (modul ``model``)
  4. Training loop dengan AMP + early stopping (modul ``training``)
  5. Inferensi tile + deteksi 4 transisi (modul ``inference``)
  6. Generate 7 file kontrak untuk WebGIS (modul ``outputs``)
  7. Validasi schema (modul ``validation``)

Import paling umum:

    >>> from forestwatch.constants import CLASS_NAMES, PALETTE_RGB
    >>> from forestwatch.config import load_config
    >>> from forestwatch.outputs.orchestrator import generate_all_outputs
"""

from forestwatch.constants import (
    BANDS,
    CLASS_COLORS,
    CLASS_NAMES,
    N_CLASSES,
    PALETTE_RGB,
    PROVINCES,
    TRANSITION_MAP,
    TRANSITION_COLORS,
)

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "BANDS",
    "CLASS_COLORS",
    "CLASS_NAMES",
    "N_CLASSES",
    "PALETTE_RGB",
    "PROVINCES",
    "TRANSITION_MAP",
    "TRANSITION_COLORS",
]
