"""Pydantic schemas untuk POST /api/analyze (analisis wilayah custom, live)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class AnalyzeRequest(BaseModel):
    aoi: list[float] = Field(
        ..., min_length=4, max_length=4,
        description="[lon_min, lat_min, lon_max, lat_max] -- kotak yang digambar user di peta",
    )
    year_t1: int = Field(..., ge=2015, le=2030)
    year_t2: int = Field(..., ge=2015, le=2030)
    min_area_ha: float = Field(default=0.5, ge=0)

    @field_validator("aoi")
    @classmethod
    def _validate_bbox(cls, v: list[float]) -> list[float]:
        lon_min, lat_min, lon_max, lat_max = v
        if not (lon_min < lon_max and lat_min < lat_max):
            raise ValueError("aoi harus [lon_min, lat_min, lon_max, lat_max] dengan min < max")
        return v


class AnalyzeResponse(BaseModel):
    deforestation: dict[str, Any]
    statistics: dict[str, Any]
    bounds: list[list[float]]
    year_t1: int
    year_t2: int
    # Preview klasifikasi tutupan lahan model (data URL PNG RGBA) utk peta "sebelum vs sesudah".
    # Bukan foto satelit tahun tsb -- ini hasil klasifikasi model di AOI yang diminta.
    landcover_t1_png: str | None = None
    landcover_t2_png: str | None = None
