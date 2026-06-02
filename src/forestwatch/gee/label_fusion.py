"""Label fusion berbasis Dynamic World — inti metodologi (revisi besar).

**Mengapa berubah?** Versi lama meng-inisialisasi *semua* piksel ke kelas 4
(Pertanian Lain) lalu hanya menandai Hutan bila ``ESA AND DW`` setuju (irisan
ketat). Akibatnya hutan yang tak disepakati dua sumber **jatuh ke Pertanian
Lain** → EDA menunjukkan Hutan hanya 24% & Pertanian Lain 49% (mustahil untuk
Papua yang ~75–85% hutan). Lihat catatan EDA di notebook.

**Pendekatan baru — Dynamic World sebagai PETA DASAR (consensus product):**

    1. Peta dasar  = Dynamic World (mode tahunan), di-remap 9→6 kelas
       (lihat ``constants.DW_TO_CLASS``). Lengkap, tanpa "default Pertanian Lain".
    2. (opsional) Perkuat Hutan dgn ESA WorldCover via UNION — hanya MENAMBAH
       hutan, tidak pernah menjatuhkannya ke kelas lain.
    3. Overlay tematik (urutan menimpa):
         a. Hansen lossyear ≥ threshold (setelah erosi 1 piksel) → 2 Deforestasi
         b. Poligon Tambang (Tang & Werner 2023)               → 5 Tambang
         c. Sawit (FDP Palm Probability ≥ ambang)              → 3 Sawit (TERAKHIR)

Dynamic World adalah produk LULC konsensus near-real-time (Brown dkk. 2022) yang
dirancang persis untuk pemetaan tutupan lahan per-piksel, sehingga jadi dasar
yang jauh lebih sahih daripada irisan dua peta tahunan beda metode/tahun. ESA
WorldCover (v200/2021) dipakai sebagai cross-check di EDA.

Sumber Sawit di-upgrade dari BIOPAMA v1 (2019) ke FDP Palm Probability model
2025a (10 m, selaras dengan tahun label T2=2025).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from forestwatch.constants import DW_TO_CLASS, GEE_ASSETS

if TYPE_CHECKING:
    import ee


def build_label(
    region: "ee.Geometry",
    label_year: int,
    *,
    hansen_loss_year_min: int = 19,
    hansen_erosion_pixels: int = 1,
    palm_prob_threshold: float = 0.5,
    dw_filter_year: int | None = None,
    esa_forest_union: bool = False,
) -> "ee.Image":
    """Bangun ``ee.Image`` label 6 kelas — Dynamic World base + overlay tematik.

    Args:
        region: ``ee.Geometry`` area of interest (mis. Papua).
        label_year: Tahun untuk filter Dynamic World & FDP Palm.
        hansen_loss_year_min: Threshold ``lossyear`` (19 = sejak 2019).
        hansen_erosion_pixels: Radius erosi morfologi untuk Hansen.
        palm_prob_threshold: Ambang probabilitas FDP Palm → Sawit (default 0.5).
        dw_filter_year: Tahun mode Dynamic World. Default = ``label_year``.
        esa_forest_union: Bila True, perkuat Hutan dgn ESA tree_cover (UNION;
            hanya menambah hutan, tidak menghapus). Default False (murni DW).

    Returns:
        ``ee.Image`` 1 band ``label`` (uint8) bernilai 0..5 (5 = Tambang).
    """
    import ee  # noqa: PLC0415

    if dw_filter_year is None:
        dw_filter_year = label_year

    # ---- 1. PETA DASAR: Dynamic World (mode tahunan), remap 9→6 ----
    dw = (
        ee.ImageCollection(GEE_ASSETS["dynamic_world"])
        .filterBounds(region)
        .filterDate(f"{dw_filter_year}-01-01", f"{dw_filter_year}-12-31")
        .select("label")
        .mode()
    )
    dw_from = list(DW_TO_CLASS.keys())
    dw_to = [DW_TO_CLASS[k] for k in dw_from]
    label = dw.remap(dw_from, dw_to).rename("label")

    # ---- 2. (opsional) Perkuat Hutan dgn ESA WorldCover (UNION) ----
    if esa_forest_union:
        esa = ee.Image(GEE_ASSETS["esa_worldcover"]).select("Map")
        # ESA tree_cover (10) di mana DW men-undercall sbg trees/grass/shrub
        # (1/2/5) → promosikan ke Hutan. Hanya MENAMBAH hutan; tak menyentuh
        # air/bare/crops, jadi tak mengulang bug "forest jatuh ke kelas lain".
        is_extra_forest = esa.eq(10).And(dw.eq(1).Or(dw.eq(2)).Or(dw.eq(5)))
        label = label.where(is_extra_forest, 1)

    # ---- 3a. Hansen deforestasi (MENIMPA) ----
    hansen = ee.Image(GEE_ASSETS["hansen_gfc"])
    defo_mask = hansen.select("lossyear").gte(hansen_loss_year_min)
    defo_eroded = defo_mask.focal_min(radius=hansen_erosion_pixels, units="pixels")
    label = label.where(defo_eroded, 2)

    # ---- 3b. Tambang: rasterisasi poligon Global Mining Footprint (MENIMPA) ----
    mining_fc = ee.FeatureCollection(GEE_ASSETS["mining"]).filterBounds(region)
    is_mining = ee.Image().byte().paint(mining_fc, 1).unmask(0)
    label = label.where(is_mining, 5)

    # ---- 3c. Sawit: FDP Palm Probability ≥ ambang (TERAKHIR, MENIMPA) ----
    # Pakai mosaic tahun target; jika tahun kosong, fallback ke seluruh koleksi.
    palm_ic = ee.ImageCollection(GEE_ASSETS["fdp_palm"]).filterBounds(region)
    palm_year = palm_ic.filterDate(f"{label_year}-01-01", f"{label_year}-12-31")
    palm = ee.Image(
        ee.Algorithms.If(
            palm_year.size().gt(0),
            palm_year.select("probability").mosaic(),
            palm_ic.select("probability").mosaic(),
        )
    )
    is_palm = palm.gte(palm_prob_threshold).unmask(0)
    label = label.where(is_palm, 3)

    return label.toByte().clip(region)


# ============================================================================
# Versi NUMPY (untuk testing tanpa GEE)
# ============================================================================


def apply_label_fusion_numpy(
    dw: "object",
    hansen_loss_eroded: "object",
    is_mining: "object",
    is_palm: "object",
    *,
    esa: "object | None" = None,
):
    """Versi numpy dari ``build_label`` (DW-base) — untuk unit test.

    Args:
        dw: ndarray label Dynamic World (nilai 0..8).
        hansen_loss_eroded: ndarray boolean — piksel Deforestasi (Hansen).
        is_mining: ndarray boolean — piksel Tambang (poligon mining).
        is_palm: ndarray boolean — piksel Sawit (FDP palm ≥ ambang).
        esa: (opsional) ndarray ESA WorldCover untuk UNION Hutan.

    Returns:
        Label ``uint8`` 0..5 (5 = Tambang). Lazy import numpy.
    """
    import numpy as np  # noqa: PLC0415

    dw = np.asarray(dw)
    # Peta dasar: remap DW 9→6.
    label = np.zeros(dw.shape, dtype=np.uint8)
    for src, dst in DW_TO_CLASS.items():
        label[dw == src] = dst

    # (opsional) UNION Hutan dgn ESA: ESA tree_cover (10) di mana DW trees/
    # grass/shrub (1/2/5) → Hutan. Hanya menambah, tak menyentuh air/bare/crops.
    if esa is not None:
        esa = np.asarray(esa)
        label[(esa == 10) & np.isin(dw, [1, 2, 5])] = 1

    label[np.asarray(hansen_loss_eroded).astype(bool)] = 2  # MENIMPA: Deforestasi
    label[np.asarray(is_mining).astype(bool)] = 5           # MENIMPA: Tambang
    label[np.asarray(is_palm).astype(bool)] = 3             # TERAKHIR: Sawit

    return label
