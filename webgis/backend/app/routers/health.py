from fastapi import APIRouter

from app.core import gee_client, model_singleton
from app.core.forestwatch_bridge import FORESTWATCH_AVAILABLE

router = APIRouter()


@router.get("/health")
def health_check():
    """Status service + kesiapan fitur analisis live (GEE & model).

    Endpoint pre-computed (legend/landcover/deforestation/statistics) selalu jalan.
    ``analyze_ready`` = True hanya kalau forestwatch + GEE + model checkpoint semuanya siap;
    kalau False, ``POST /api/analyze`` akan respond 503 (lihat ``model_error`` utk sebabnya).
    """
    model_ready = model_singleton.is_ready()
    gee_ready = gee_client.is_ready()
    return {
        "status": "ok",
        "service": "ForestWatch Papua API",
        "forestwatch_available": FORESTWATCH_AVAILABLE,
        "gee_ready": gee_ready,
        "model_ready": model_ready,
        "analyze_ready": FORESTWATCH_AVAILABLE and gee_ready and model_ready,
        "gee_error": None if gee_ready else gee_client.get_init_error(),
        "model_error": None if model_ready else model_singleton.get_load_error(),
    }
