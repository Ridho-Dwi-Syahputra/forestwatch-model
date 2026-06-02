"""Label fusion 6 aturan — inti metodologi PRD §A.4 (revisi: kelas 5 = Tambang).

Aturan diterapkan **berurutan** karena aturan berikutnya menimpa yang sebelumnya:

    1. Init semua piksel = Kelas 4 (Pertanian Lain default)
    2. Perairan (ESA water/wetland/mangrove ∩ DW water/flooded) → 0
    3. Hutan (ESA tree_cover ∩ DW trees) → 1
    4. Lahan Terbuka (ESA bare ∪ DW bare) → 2
    5b. Cropland → 4 (Pertanian Lain eksplisit)
    6. Hansen lossyear >= 19 (setelah erosion 1 piksel) → 2 (Deforestasi, MENIMPA)
    7. Poligon Tambang (Tang & Werner 2023) → 5 (Tambang, MENIMPA)
    5a. BIOPAMA oil palm → 3 (Sawit, DITERAPKAN TERAKHIR agar pasti terwakili)

Catatan revisi: kelas 5 yang dulu "Lahan Terbakar" (dNBR) DIGANTI menjadi
"Tambang" memakai Global Mining Footprint (poligon, didelineasi dari Sentinel-2),
karena lebih relevan untuk Papua (mis. Grasberg) & berbasis dataset tervalidasi.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from forestwatch.constants import GEE_ASSETS

if TYPE_CHECKING:
    import ee


def build_label(
    region: "ee.Geometry",
    label_year: int,
    *,
    hansen_loss_year_min: int = 19,
    hansen_erosion_pixels: int = 1,
    default_class_id: int = 4,
    dw_filter_year: int | None = None,
) -> "ee.Image":
    """Bangun ``ee.Image`` label 6 kelas dengan aturan fusion (PRD §A.4 revisi).

    Args:
        region: ``ee.Geometry`` area of interest (mis. Papua).
        label_year: Tahun untuk filter Dynamic World.
        hansen_loss_year_min: Threshold ``lossyear`` (19 = sejak 2019).
        hansen_erosion_pixels: Radius erosi morfologi untuk Hansen.
        default_class_id: ID kelas awal sebelum aturan diterapkan.
        dw_filter_year: Tahun untuk mode Dynamic World. Default = ``label_year``.

    Returns:
        ``ee.Image`` 1 band ``label`` (uint8) dengan nilai 0..5
        (5 = Tambang).
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

    # BIOPAMA adalah ImageCollection ubin regional — WAJIB mosaic se-region,
    # JANGAN .first() (itu hanya 1 ubin, kemungkinan di luar Papua → Sawit=0).
    oilpalm = (
        ee.ImageCollection(GEE_ASSETS["biopama_oilpalm"])
        .filterBounds(region)
        .mosaic()
        .select("classification")
    )
    is_oilpalm = oilpalm.eq(1).Or(oilpalm.eq(2)).unmask(0)  # industrial OR smallholder

    # Tambang: rasterisasi poligon Global Mining Footprint (Tang & Werner 2023).
    # FeatureCollection → mask 1 di area tambang, 0 di luar.
    mining_fc = ee.FeatureCollection(GEE_ASSETS["mining"]).filterBounds(region)
    is_mining = ee.Image().byte().paint(mining_fc, 1).unmask(0)

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

    # Aturan 5b: Cropland (non-sawit) → Pertanian Lain (eksplisit)
    is_cropland = esa.eq(40)
    label = label.where(is_cropland, 4)

    # Aturan 6: Deforestasi (Hansen, MENIMPA)
    label = label.where(defo_eroded, 2)

    # Aturan 7: Tambang (poligon Global Mining Footprint, MENIMPA)
    label = label.where(is_mining, 5)

    # Aturan 5a (DITERAPKAN TERAKHIR): Sawit langsung dari BIOPAMA.
    # BIOPAMA adalah peta sawit terdedikasi & tervalidasi (Descals dkk. 2021);
    # diterapkan paling akhir agar kelas Sawit pasti terwakili — menimpa Hutan/
    # Cropland yang sebenarnya perkebunan sawit (ESA sering melabeli sawit dewasa
    # sebagai 'tree cover', sehingga penggerbangan cropland membuat Sawit hilang).
    label = label.where(is_oilpalm, 3)

    return label.toByte().clip(region)


# ============================================================================
# Versi NUMPY (untuk testing tanpa GEE)
# ============================================================================


def apply_label_fusion_numpy(
    esa: "object",
    dw: "object",
    hansen_loss_eroded: "object",
    is_oilpalm: "object",
    is_mining: "object",
    *,
    default_class_id: int = 4,
):
    """Versi numpy dari ``build_label`` — untuk unit test.

    Semua input adalah numpy ``ndarray`` dengan shape yang sama. Return label
    ``uint8`` dengan nilai 0..5 (5 = Tambang). Lazy import numpy.

    CATATAN: input boolean-masks (``hansen_loss_eroded``, ``is_oilpalm``,
    ``is_mining``) di-cast ke bool eksplisit untuk menghindari advanced
    indexing yang tidak sengaja saat user mengirim ``uint8``.
    """
    import numpy as np  # noqa: PLC0415

    esa = np.asarray(esa)
    dw = np.asarray(dw)
    is_oilpalm_b = np.asarray(is_oilpalm).astype(bool)
    hansen_b = np.asarray(hansen_loss_eroded).astype(bool)
    mining_b = np.asarray(is_mining).astype(bool)

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
    label[is_cropland] = 4               # cropland → Pertanian Lain (default sawit ditangani terakhir)

    label[hansen_b] = 2                  # MENIMPA: Deforestasi

    label[mining_b] = 5                  # MENIMPA: Tambang (poligon mining footprint)

    label[is_oilpalm_b] = 3              # TERAKHIR: Sawit dari BIOPAMA (pasti terwakili)

    return label
