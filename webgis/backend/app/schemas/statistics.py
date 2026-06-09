"""Pydantic schemas untuk statistics responses."""

from __future__ import annotations

from pydantic import BaseModel


class PerClassMetric(BaseModel):
    class_name: str
    iou: float
    f1: float


class ModelMetrics(BaseModel):
    overall_accuracy: float
    mean_iou: float
    per_class: list[PerClassMetric]


class ProvinceStats(BaseModel):
    province: str
    deforestation_ha: float


class StatisticsResponse(BaseModel):
    period_from: int
    period_to: int
    total_deforestation_ha: float
    n_hotspots: int
    per_transition_ha: dict[str, float]
    per_province: list[ProvinceStats]
    per_class_area_ha: dict[str, float]
    model_metrics: dict


class PerProvinceResponse(BaseModel):
    data: list[ProvinceStats]


class PerTransitionResponse(BaseModel):
    data: dict[str, float]
