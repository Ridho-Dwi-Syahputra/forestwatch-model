"""Pydantic schemas untuk GeoJSON responses."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class LandcoverResponse(BaseModel):
    image_url: str
    bounds: list[list[float]]
    crs: str = "EPSG:4326"
    year: int


class DeforestationProperties(BaseModel):
    id: str
    transition_type: str
    area_ha: float
    period_from: int
    period_to: int
    province: str | None = None
    kawasan_status: str | None = None


class GeoJSONGeometry(BaseModel):
    type: str
    coordinates: Any


class GeoJSONFeature(BaseModel):
    type: str = "Feature"
    geometry: GeoJSONGeometry
    properties: DeforestationProperties


class FeatureCollection(BaseModel):
    type: str = "FeatureCollection"
    name: str | None = None
    crs: dict[str, Any] | None = None
    features: list[dict[str, Any]]
    total: int


class LegendItem(BaseModel):
    id: int
    name: str
    color: str
