// Peta Leaflet -- TERKUNCI ke Papua (tak bisa di-pan/zoom), tapi layer & popup tetap interaktif.
import {
  GeoJSON,
  ImageOverlay,
  MapContainer,
  TileLayer,
  Tooltip as LeafletTooltip,
} from "react-leaflet";

import { AVAILABLE_LANDCOVER_YEARS, BASEMAPS, PAPUA, TRANSITION_META } from "../lib/constants";
import { formatNumber, transitionLabel } from "../lib/format";
import { useApp } from "../context/AppContext";

export default function MapView() {
  const {
    data,
    activeYear,
    opacity,
    basemap,
    filteredGeojson,
    filteredFeatures,
    selectedTransitions,
    selectedProvince,
  } = useApp();

  // Tahun layer: pakai tahun yg tersedia (2021/2025); selain itu pakai visual referensi terdekat.
  const landcoverYear = AVAILABLE_LANDCOVER_YEARS.includes(activeYear)
    ? activeYear
    : activeYear < 2024
      ? 2021
      : 2025;
  const landcover = data.landcover?.[landcoverYear];
  const basemapConfig = BASEMAPS[basemap];
  const mapKey = `${activeYear}-${landcoverYear}-${opacity}-${basemap}`;
  const geoKey = `${filteredFeatures.length}-${Array.from(selectedTransitions).join("-")}-${selectedProvince}`;

  return (
    <section className="map-panel" aria-label="Peta Kasuari AI Papua">
      <MapContainer
        center={PAPUA.center}
        zoom={PAPUA.zoom}
        minZoom={PAPUA.zoom}
        maxZoom={PAPUA.zoom}
        maxBounds={PAPUA.bounds}
        maxBoundsViscosity={1.0}
        dragging={false}
        scrollWheelZoom={false}
        doubleClickZoom={false}
        touchZoom={false}
        boxZoom={false}
        keyboard={false}
        zoomControl={false}
        className="leaflet-map"
      >
        <TileLayer key={basemap} attribution={basemapConfig.attribution} url={basemapConfig.url} />

        {landcover && (
          <ImageOverlay key={mapKey} url={landcover.image_url} bounds={landcover.bounds} opacity={opacity} />
        )}

        <GeoJSON
          key={geoKey}
          data={filteredGeojson}
          style={(feature) => {
            const type = feature?.properties?.transition_type;
            const color = TRANSITION_META[type]?.color || "#123D2F";
            return { color, weight: 1.6, fillColor: color, fillOpacity: 0.58 };
          }}
          onEachFeature={(feature, layer) => {
            const p = feature.properties || {};
            layer.bindPopup(`
              <div class="popup-card">
                <strong>${p.id || "-"}</strong>
                <span>${transitionLabel(p.transition_type)}</span>
                <dl>
                  <dt>Luas</dt><dd>${formatNumber(p.area_ha, 1)} ha</dd>
                  <dt>Periode</dt><dd>${p.period_from || "-"} - ${p.period_to || "-"}</dd>
                  <dt>Provinsi</dt><dd>${p.province || "-"}</dd>
                  <dt>Kawasan</dt><dd>${p.kawasan_status || "-"}</dd>
                </dl>
              </div>
            `);
          }}
        >
          <LeafletTooltip sticky>Area perubahan hutan</LeafletTooltip>
        </GeoJSON>
      </MapContainer>

      <div className="legend-box">
        <strong>Legenda</strong>
        {data.legend.map((item) => (
          <span key={item.id}>
            <i style={{ backgroundColor: item.color }} />
            {item.name}
          </span>
        ))}
      </div>
    </section>
  );
}
