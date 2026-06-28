"""Load model segmentasi sekali saat startup, dipakai ulang tiap request ``/api/analyze``.

Memuat model sekali (bukan per-request) penting karena ``build_unet`` + load weight ResNet50
butuh waktu nyata -- kalau diulang tiap request, tiap analisis jadi lebih lambat tanpa perlu.
"""

from __future__ import annotations

from app.core.config import MODEL_ARCHITECTURE, MODEL_CHECKPOINT_PATH, MODEL_ENCODER_NAME
from app.core.forestwatch_bridge import build_unet

_model = None
_load_error: str | None = None


def load_model() -> bool:
    """Build + load checkpoint sekali (idempotent). Return ``True`` bila siap dipakai."""
    global _model, _load_error
    if _model is not None:
        return True
    if not MODEL_CHECKPOINT_PATH.exists():
        _load_error = f"Checkpoint tidak ditemukan: {MODEL_CHECKPOINT_PATH}"
        return False

    try:
        import torch
    except ImportError as exc:
        _load_error = f"torch belum terinstal: {exc}"
        return False

    try:
        # encoder_weights=None: bobot asli datang dari load_state_dict di bawah, jadi tak perlu
        # unduh ImageNet saat startup (butuh internet & sia-sia karena langsung ditimpa).
        model = build_unet(
            architecture=MODEL_ARCHITECTURE,
            encoder_name=MODEL_ENCODER_NAME,
            encoder_weights=None,
        )
        state = torch.load(MODEL_CHECKPOINT_PATH, map_location="cpu")
        state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as exc:  # noqa: BLE001
        _load_error = f"Gagal load checkpoint: {exc}"
        return False

    _model = model
    return True


def get_model():
    return _model


def get_load_error() -> str | None:
    return _load_error


def is_ready() -> bool:
    return _model is not None
