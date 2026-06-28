// Peta Leaflet -- DIBATASI ke Papua (zoom & pan boleh, tapi tak bisa keluar bounds / zoom-out
// melewati overview). Zoom-in menampilkan detail wilayah dari citra satelit. Gaya ala SiPongi:
// basemap satelit, deforestasi sbg TITIK (bukan overlay hijau penuh). Overlay tutupan lahan
// opsional (muncul saat opacity > 0).
import {
  CircleMarker,
  ImageOverlay,
  MapContainer,
  Popup,
  TileLayer,
} from "react-leaflet";

import { AVAILABLE_LANDCOVER_YEARS, BASEMAPS, PAPUA, TRANSITION_META } from "../lib/constants";
import { formatNumber, transitionLabel } from "../lib/format";
import { useApp } from "../context/AppContext";

// Centroid sederhana (rata-rata vertex ring luar) -> [lat, lng] utk Leaflet.
function polygonCentroid(geometry) {
  const ring = geometry?.coordinates?.[0];
  if (!ring || !ring.length) return null;
  const first = ring[0];
  const last = ring[ring.length - 1];
  const pts =
    ring.length > 1 && first[0] === last[0] && first[1] === last[1] ? ring.slice(0, -1) : ring;
  let sx = 0;
  let sy = 0;
  for (const [lng, lat] of pts) {
    sx += lng;
    sy += lat;
  }
  return [sy / pts.length, sx / pts.length];
}

// Radius titik: sedikit mengikuti luas, tapi DIBATASI kecil (gaya titik, bukan blob).
function dotRadius(areaHa) {
  const a = Number(areaHa) || 0;
  return Math.max(4, Math.min(11, 4 + Math.sqrt(a) / 7));
}

export default function MapView() {
  const { data, activeYear, opacity, basemap, filteredFeatures } = useApp();

  const landcoverYear = AVAILABLE_LANDCOVER_YEARS.includes(activeYear)
    ? activeYear
    : activeYear < 2024
      ? 2021
      : 2025;
  const landcover = data.landcover?.[landcoverYear];
  const basemapConfig = BASEMAPS[basemap];
  const showOverlay = opacity > 0 && landcover; // overlay tutupan lahan OFF saat opacity 0

  return (
    <section className="map-panel" aria-label="Peta Kasuari AI Papua">
      <MapContainer
        center={PAPUA.center}
        zoom={PAPUA.zoom}
        minZoom={PAPUA.minZoom}
        maxZoom={PAPUA.maxZoom}
        maxBounds={PAPUA.bounds}
        maxBoundsViscosity={1.0}
        scrollWheelZoom
        zoomControl
        className="leaflet-map"
      >
        <TileLayer key={basemap} attribution={basemapConfig.attribution} url={basemapConfig.url} />

        {showOverlay && (
          <ImageOverlay
            key={`${landcoverYear}-${opacity}`}
            url={landcover.image_url}
            bounds={landcover.bounds}
            opacity={opacity}
          />
        )}

        {filteredFeatures.map((feature) => {
          const center = polygonCentroid(feature.geometry);
          if (!center) return null;
          const p = feature.properties || {};
          const color = TRANSITION_META[p.transition_type]?.color || "#B45309";
          return (
            <CircleMarker
              key={p.id}
              center={center}
              radius={dotRadius(p.area_ha)}
              pathOptions={{ color: "#ffffff", weight: 1, fillColor: color, fillOpacity: 0.9 }}
            >
              <Popup>
                <div className="popup-card">
                  <strong>{p.id || "-"}</strong>
                  <span>{transitionLabel(p.transition_type)}</span>
                  <dl>
                    <dt>Luas</dt>
                    <dd>{formatNumber(p.area_ha, 1)} ha</dd>
                    <dt>Periode</dt>
                    <dd>
                      {p.period_from || "-"} - {p.period_to || "-"}
                    </dd>
                    <dt>Provinsi</dt>
                    <dd>{p.province || "-"}</dd>
                    <dt>Kawasan</dt>
                    <dd>{p.kawasan_status || "-"}</dd>
                  </dl>
                </div>
              </Popup>
            </CircleMarker>
          );
        })}
      </MapContainer>

      <div className="legend-box">
        <strong>Legenda Transisi</strong>
        {Object.entries(TRANSITION_META).map(([key, meta]) => (
          <span key={key}>
            <i style={{ backgroundColor: meta.color }} />
            {meta.label}
          </span>
        ))}
      </div>
    </section>
  );
}
