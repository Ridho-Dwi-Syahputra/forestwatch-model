// Peta mini untuk preview AOI hasil Analisis Custom: basemap satelit + overlay klasifikasi
// tutupan lahan model (PNG dari backend) yang dipas ke `bounds`. Opsional: titik perubahan
// (untuk peta "Sesudah"). Dipakai berdampingan T1 vs T2 di CustomAnalysisPage.
import { CircleMarker, ImageOverlay, MapContainer, TileLayer } from "react-leaflet";

import { BASEMAPS, TRANSITION_META } from "../lib/constants";

// Centroid sederhana (rata-rata vertex ring luar) -> [lat, lng].
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

export default function AoiPreviewMap({ title, pngUrl, bounds, features = [], caption }) {
  if (!bounds) return null;
  const satellite = BASEMAPS.satelit;

  return (
    <div className="aoi-preview">
      <div className="aoi-preview-head">
        <strong>{title}</strong>
        {caption && <small>{caption}</small>}
      </div>
      <MapContainer
        bounds={bounds}
        maxBounds={bounds}
        maxBoundsViscosity={1.0}
        scrollWheelZoom
        zoomControl
        className="aoi-preview-map"
      >
        <TileLayer attribution={satellite.attribution} url={satellite.url} />
        {pngUrl && <ImageOverlay url={pngUrl} bounds={bounds} opacity={0.75} />}
        {features.map((feature, idx) => {
          const center = polygonCentroid(feature.geometry);
          if (!center) return null;
          const p = feature.properties || {};
          const color = TRANSITION_META[p.transition_type]?.color || "#B45309";
          return (
            <CircleMarker
              key={p.id || idx}
              center={center}
              radius={5}
              pathOptions={{ color: "#ffffff", weight: 1, fillColor: color, fillOpacity: 0.95 }}
            />
          );
        })}
      </MapContainer>
    </div>
  );
}
