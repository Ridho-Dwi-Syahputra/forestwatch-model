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


def s2_composite_range(
    start: str,
    end: str,
    region: "ee.Geometry",
    *,
    bands: tuple[str, ...] = BANDS,
    cloud_threshold: float = 40.0,
    scl_mask_classes: list[int] | None = None,
    reflectance_divisor: int = REFLECTANCE_DIVISOR,
) -> "ee.Image":
    """Komposit median Sentinel-2 bebas awan untuk rentang tanggal custom.

    Logika inti dipakai bersama oleh ``s2_composite(year, ...)`` (rentang 1 Jan - 31 Des)
    dan untuk komposit "isi celah" -- mis. tahun yang datanya belum penuh (lihat
    ``fill_composite_gaps``), di mana butuh komposit dari periode LAIN (bukan tahun kalender
    penuh) sebagai fallback.

    Args:
        start, end: Tanggal ISO ``"YYYY-MM-DD"`` (dipakai langsung di ``ee.Filter.date``).
        region: ``ee.Geometry`` area of interest.
        bands: Band yang akan dipilih (default: 6 band PRD §A.2.1).
        cloud_threshold: Filter scene dengan CLOUDY_PIXEL_PERCENTAGE > nilai ini.
        scl_mask_classes: SCL class id yang di-mask (default: lihat ``mask_scl``).
        reflectance_divisor: Pembagi reflektansi ke skala [0, 1].

    Returns:
        ``ee.Image`` 6 band float, reflektansi [0, 1], clipped ke region. Piksel yang TAK
        punya observasi valid sama sekali di rentang ini akan masked (no-data) -- bukan 0.
    """
    import ee  # noqa: PLC0415

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


def s2_composite(
    year: int,
    region: "ee.Geometry",
    *,
    bands: tuple[str, ...] = BANDS,
    cloud_threshold: float = 40.0,
    scl_mask_classes: list[int] | None = None,
    reflectance_divisor: int = REFLECTANCE_DIVISOR,
) -> "ee.Image":
    """Komposit median Sentinel-2 bebas awan untuk satu tahun (1 Jan - 31 Des).

    Sumber: PRD §A.5 Cell 3 ``s2_composite(year)`` — diparameterisasi region & band.
    Wrapper tipis di atas ``s2_composite_range`` dgn rentang tahun kalender penuh.

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
    return s2_composite_range(
        f"{year}-01-01",
        f"{year}-12-31",
        region,
        bands=bands,
        cloud_threshold=cloud_threshold,
        scl_mask_classes=scl_mask_classes,
        reflectance_divisor=reflectance_divisor,
    )


def fill_composite_gaps(primary: "ee.Image", fallback: "ee.Image") -> "ee.Image":
    """Isi piksel kosong (tak ada observasi valid) di ``primary`` pakai nilai dari ``fallback``.

    Dipakai utk tahun yang datanya belum penuh (mis. tahun berjalan baru terisi separuh) --
    piksel yang ``median()``-nya kosong krn semua observasi di rentang ``primary`` ke-mask
    awan/tak ada data sama sekali, diisi dari komposit periode lain (mis. semester
    sebelumnya) drpd dibiarkan no-data total di hasil akhir.

    ``ee.Image.unmask`` bekerja per-piksel-per-band: piksel ``primary`` yang valid TETAP
    dipakai (fallback tak menimpa apa pun yang sudah ada) -- ini cuma isi celah, bukan blend.
    """
    return primary.unmask(fallback)


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
