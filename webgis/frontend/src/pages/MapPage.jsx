// Halaman 2 -- Peta Interaktif: peta Papua (terkunci) + time-series, opacity, basemap, filter.
import { Layers, SlidersHorizontal } from "lucide-react";

import MapView from "../components/MapView";
import { AVAILABLE_LANDCOVER_YEARS, BASEMAPS, YEARS } from "../lib/constants";
import { formatNumber, formatPercent } from "../lib/format";
import { useApp } from "../context/AppContext";

export default function MapPage() {
  const {
    activeYear,
    setActiveYear,
    opacity,
    setOpacity,
    basemap,
    setBasemap,
    selectedProvince,
    setSelectedProvince,
    selectedTransitions,
    toggleTransition,
    selectAllTransitions,
    allTransitionsSelected,
    resetFilters,
    transitionRows,
    provinceRows,
    filteredFeatures,
    selectedArea,
  } = useApp();

  const isEstimatedLayer = !AVAILABLE_LANDCOVER_YEARS.includes(activeYear);

  return (
    <section className="page webgis-page">
      <MapView />

      <aside className="webgis-tools">
        <section className="summary-panel">
          <p className="eyebrow">Area terfilter</p>
          <strong>{formatNumber(selectedArea, 1)} ha</strong>
          <div className="summary-grid">
            <span>
              <b>{formatNumber(filteredFeatures.length, 0)}</b>
              Polygon
            </span>
            <span>
              <b>{activeYear}</b>
              Layer aktif
            </span>
          </div>
        </section>

        <section className="tool-panel">
          <div className="control-heading">
            <Layers size={16} />
            Time Series Peta
          </div>
          <label className="year-slider">
            <span>{activeYear}</span>
            <input
              type="range"
              min={YEARS[0]}
              max={YEARS[YEARS.length - 1]}
              step="1"
              value={activeYear}
              onChange={(event) => setActiveYear(Number(event.target.value))}
            />
          </label>
          <div className="year-ticks" aria-label="Tahun tersedia">
            {YEARS.map((year) => (
              <button
                key={year}
                type="button"
                className={activeYear === year ? "active" : ""}
                onClick={() => setActiveYear(year)}
              >
                {year}
              </button>
            ))}
          </div>
          <p className="layer-note">
            {isEstimatedLayer
              ? `Layer ${activeYear} memakai visual referensi sampai komposit tahunan tersedia.`
              : `Layer ${activeYear} tersedia dari data.`}
          </p>
          <label className="range-row">
            <span>Opacity</span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={opacity}
              onChange={(event) => setOpacity(Number(event.target.value))}
            />
            <b>{formatPercent(opacity)}</b>
          </label>
          <label className="select-row">
            <span>Basemap</span>
            <select value={basemap} onChange={(event) => setBasemap(event.target.value)}>
              {Object.entries(BASEMAPS).map(([key, value]) => (
                <option key={key} value={key}>
                  {value.label}
                </option>
              ))}
            </select>
          </label>
        </section>

        <section className="tool-panel">
          <div className="control-heading">
            <SlidersHorizontal size={16} />
            Filter
            <button type="button" className="mini-button" onClick={resetFilters}>
              Reset
            </button>
          </div>
          <label className="select-row">
            <span>Provinsi</span>
            <select
              value={selectedProvince}
              onChange={(event) => setSelectedProvince(event.target.value)}
            >
              <option>Semua Provinsi</option>
              {provinceRows.map((row) => (
                <option key={row.province}>{row.province}</option>
              ))}
            </select>
          </label>
          <div className="filter-grid vertical">
            {transitionRows.map((item) => (
              <label key={item.key} className="check-row">
                <input
                  type="checkbox"
                  checked={selectedTransitions.has(item.key)}
                  onChange={() => toggleTransition(item.key)}
                />
                <span className="swatch" style={{ backgroundColor: item.color }} />
                <span>{item.label}</span>
              </label>
            ))}
          </div>
          <button
            className="secondary-button wide"
            type="button"
            onClick={selectAllTransitions}
            disabled={allTransitionsSelected}
          >
            {allTransitionsSelected ? "Semua Transisi Aktif" : "Tampilkan Semua Transisi"}
          </button>
        </section>
      </aside>
    </section>
  );
}
