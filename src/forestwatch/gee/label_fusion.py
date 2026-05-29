"""Label fusion 6 aturan — inti metodologi PRD §A.4.

Aturan diterapkan **berurutan** karena aturan berikutnya menimpa yang sebelumnya:

    1. Init semua piksel = Kelas 4 (Pertanian Lain default)
    2. Perairan (ESA water/wetland/mangrove ∩ DW water/flooded) → 0
    3. Hutan (ESA tree_cover ∩ DW trees) → 1
    4. Lahan Terbuka (ESA bare ∪ DW bare) → 2
    5a. Cropland ∩ BIOPAMA oil palm → 3 (Sawit)
    5b. Cropland ∩ NOT oil palm → 4 (Pertanian Lain eksplisit)
    6. Hansen lossyear >= 19 (setelah erosion 1 piksel) → 2 (Deforestasi, MENIMPA)
    7. dNBR >= threshold → 5 (Lahan Terbakar, MENIMPA)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from forestwatch.constants import GEE_ASSETS
from forestwatch.gee.composite import compute_dnbr

if TYPE_CHECKING:
    import ee


def build_label(
    region: "ee.Geometry",
    label_year: int,
    *,
    hansen_loss_year_min: int = 19,
    hansen_erosion_pixels: int = 1,
    dnbr_threshold: float = 0.27,
    default_class_id: int = 4,
    dw_filter_year: int | None = None,
) -> "ee.Image":
    """Bangun ``ee.Image`` label 6 kelas dengan 6 aturan fusion (PRD §A.4).

    Args:
        region: ``ee.Geometry`` area of interest (mis. Papua).
        label_year: Tahun untuk komputasi dNBR & filter Dynamic World.
        hansen_loss_year_min: Threshold ``lossyear`` (19 = sejak 2019).
        hansen_erosion_pixels: Radius erosi morfologi untuk Hansen.
        dnbr_threshold: Ambang dNBR untuk kelas Lahan Terbakar.
        default_class_id: ID kelas awal sebelum aturan diterapkan.
        dw_filter_year: Tahun untuk mode Dynamic World. Default = ``label_year``.

    Returns:
        ``ee.Image`` 1 band ``label`` (uint8) dengan nilai 0..5.
    """
    import ee  # noqa: PLC0415

    if dw_filter_year is None:
        dw_filter_year = label_year

    # ---- Ambil 4 sumber label ----
    esa = ee.Image(GEE_ASSETS["esa_worldcover"]).select("Map")

    dw = (
        ee.ImageCollection(GEE_ASSETS["dynamic_world"])
        .filterBounds(region)
        .filterDate(f"{dw_filter_year}-01-01", f"{dw_filter_year}-12-31")
        .select("label")
        .mode()
    )

    hansen = ee.Image(GEE_ASSETS["hansen_gfc"])
    lossyear = hansen.select("lossyear")
    defo_mask = lossyear.gte(hansen_loss_year_min)
    defo_eroded = defo_mask.focal_min(radius=hansen_erosion_pixels, units="pixels")

    oilpalm = ee.ImageCollection(GEE_ASSETS["biopama_oilpalm"]).first().select("classification")
    is_oilpalm = oilpalm.eq(1).Or(oilpalm.eq(2))  # industrial OR smallholder

    dnbr = compute_dnbr(label_year, region)
    is_burned = dnbr.gte(dnbr_threshold)

    # ---- 6 aturan fusion (urutan penting) ----
    label = ee.Image(default_class_id).rename("label")

    # Aturan 2: Perairan
    is_water_esa = esa.eq(80).Or(esa.eq(90)).Or(esa.eq(95))
    is_water_dw = dw.eq(0).Or(dw.eq(3))
    label = label.where(is_water_esa.And(is_water_dw), 0)

    # Aturan 3: Hutan
    is_forest_esa = esa.eq(10)
    is_forest_dw = dw.eq(1)
    label = label.where(is_forest_esa.And(is_forest_dw), 1)

    # Aturan 4: Lahan Terbuka
    is_bare = esa.eq(60).Or(dw.eq(7))
    label = label.where(is_bare, 2)

    # Aturan 5a & 5b: Cropland → Sawit / Pertanian Lain
    is_cropland = esa.eq(40)
    label = label.where(is_cropland.And(is_oilpalm), 3)
    label = label.where(is_cropland.And(is_oilpalm.Not()), 4)

    # Aturan 6: Deforestasi (Hansen, MENIMPA)
    label = label.where(defo_eroded, 2)

    # Aturan 7: Lahan Terbakar (dNBR, MENIMPA)
    label = label.where(is_burned, 5)

    return label.toByte().clip(region)


# ============================================================================
# Versi NUMPY (untuk testing tanpa GEE)
# ============================================================================


def apply_label_fusion_numpy(
    esa: "object",
    dw: "object",
    hansen_loss_eroded: "object",
    is_oilpalm: "object",
    is_burned: "object",
    *,
    default_class_id: int = 4,
):
    """Versi numpy dari ``build_label`` — untuk unit test.

    Semua input adalah numpy ``ndarray`` dengan shape yang sama. Return label
    ``uint8`` dengan nilai 0..5. Lazy import numpy.

    CATATAN: input boolean-masks (``hansen_loss_eroded``, ``is_oilpalm``,
    ``is_burned``) di-cast ke bool eksplisit untuk menghindari advanced
    indexing yang tidak sengaja saat user mengirim ``uint8``.
    """
    import numpy as np  # noqa: PLC0415

    esa = np.asarray(esa)
    dw = np.asarray(dw)
    is_oilpalm_b = np.asarray(is_oilpalm).astype(bool)
    hansen_b = np.asarray(hansen_loss_eroded).astype(bool)
    burned_b = np.asarray(is_burned).astype(bool)

    label = np.full(esa.shape, default_class_id, dtype=np.uint8)

    is_water_esa = np.isin(esa, [80, 90, 95])
    is_water_dw = np.isin(dw, [0, 3])
    label[is_water_esa & is_water_dw] = 0

    is_forest_esa = esa == 10
    is_forest_dw = dw == 1
    label[is_forest_esa & is_forest_dw] = 1

    is_bare = (esa == 60) | (dw == 7)
    label[is_bare] = 2

    is_cropland = esa == 40
    label[is_cropland & is_oilpalm_b] = 3
    label[is_cropland & ~is_oilpalm_b] = 4

    label[hansen_b] = 2  # MENIMPA

    label[burned_b] = 5  # MENIMPA

    return label
