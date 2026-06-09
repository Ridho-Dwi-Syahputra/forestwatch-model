"""Komposit Sentinel-2 bebas awan + indeks bakar (dNBR).

Adopsi langsung dari PRD §A.5 Cell 3 dengan parameterisasi.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from forestwatch.constants import BANDS, REFLECTANCE_DIVISOR

if TYPE_CHECKING:
    import ee


def mask_scl(image: "ee.Image", mask_classes: list[int] | None = None) -> "ee.Image":
    """Mask awan/bayangan via Scene Classification Layer (SCL).

    Args:
        image: ``ee.Image`` Sentinel-2 SR Harmonized.
        mask_classes: Daftar SCL class id yang akan di-mask. Default:
            ``[3, 8, 9, 10, 11]`` (bayangan, awan medium/tinggi, cirrus, salju).

    Returns:
        Image yang sudah ter-masking.
    """
    import ee  # noqa: PLC0415

    if mask_classes is None:
        mask_classes = [3, 8, 9, 10, 11]

    scl = image.select("SCL")
    keep = ee.Image(1)
    for cls in mask_classes:
        keep = keep.And(scl.neq(cls))
    return image.updateMask(keep)


def s2_composite(
    year: int,
    region: "ee.Geometry",
    *,
    bands: tuple[str, ...] = BANDS,
    cloud_threshold: float = 40.0,
    scl_mask_classes: list[int] | None = None,
    reflectance_divisor: int = REFLECTANCE_DIVISOR,
) -> "ee.Image":
    """Komposit median Sentinel-2 bebas awan untuk satu tahun.

    Sumber: PRD §A.5 Cell 3 ``s2_composite(year)`` — diparameterisasi region & band.

    Args:
        year: Tahun (e.g. 2021).
        region: ``ee.Geometry`` area of interest.
        bands: Band yang akan dipilih (default: 6 band PRD §A.2.1).
        cloud_threshold: Filter scene dengan CLOUDY_PIXEL_PERCENTAGE > nilai ini.
        scl_mask_classes: SCL class id yang di-mask (default: lihat ``mask_scl``).
        reflectance_divisor: Pembagi reflektansi ke skala [0, 1].

    Returns:
        ``ee.Image`` 6 band float, reflektansi [0, 1], clipped ke region.
    """
    import ee  # noqa: PLC0415

    start = f"{year}-01-01"
    end = f"{year}-12-31"

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold))
    )

    masked = collection.map(lambda img: mask_scl(img, scl_mask_classes))
    composite = (
        masked.select(list(bands))
        .median()
        .divide(reflectance_divisor)
        .clip(region)
        .toFloat()
    )
    return composite


def compute_dnbr(
    year: int,
    region: "ee.Geometry",
    *,
    nir_band: str = "B8",
    swir_band: str = "B12",
) -> "ee.Image":
    """Delta NBR = NBR(year-1) - NBR(year) — indikator area terbakar.

    Sumber: PRD §A.5 Cell 3 ``compute_dnbr``. Jika ``year == 2017`` (awal S2 SR),
    fallback memakai NBR tahun yang sama (delta = 0 untuk piksel tanpa awan).
    """
    pre_year = year - 1 if year > 2017 else year
    pre = s2_composite(pre_year, region)
    post = s2_composite(year, region)
    nbr_pre = pre.normalizedDifference([nir_band, swir_band]).rename("nbr")
    nbr_post = post.normalizedDifference([nir_band, swir_band]).rename("nbr")
    return nbr_pre.subtract(nbr_post).rename("dnbr")
