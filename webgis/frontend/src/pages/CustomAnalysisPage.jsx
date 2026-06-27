// Halaman 7 -- Analisis Wilayah Custom: kirim AOI + rentang tahun ke backend (POST /api/analyze).
import { Search } from "lucide-react";

import { CUSTOM_AOI_PRESETS } from "../lib/constants";
import { formatNumber, transitionLabel } from "../lib/format";
import { useApp } from "../context/AppContext";

export default function CustomAnalysisPage() {
  const { customAnalysis, runCustomAnalysis, updateCustomAnalysis } = useApp();
  const isLoading = customAnalysis.status === "loading";

  return (
    <section className="page custom-page">
      <section className="method-hero">
        <p className="eyebrow">Live Capability</p>
        <h2>Analisis AOI kecil dari frontend ke backend.</h2>
        <p>
          Form ini memanggil <b>POST /api/analyze</b> ketika backend API aktif. Tanpa API URL,
          Kasuari AI menampilkan simulasi lokal dari dummy GeoJSON.
        </p>
      </section>

      <div className="custom-layout">
        <section className="panel-section">
          <div className="section-title">
            <Search size={18} />
            Parameter Analisis
          </div>
          <form className="custom-analysis-form large" onSubmit={runCustomAnalysis}>
            <label className="select-row custom-select-row">
              <span>AOI</span>
              <select
                value={customAnalysis.aoi}
                onChange={(event) => updateCustomAnalysis("aoi", event.target.value)}
              >
                {Object.entries(CUSTOM_AOI_PRESETS).map(([key, preset]) => (
                  <option key={key} value={key}>
                    {preset.label}
                  </option>
                ))}
              </select>
            </label>

            <div className="custom-year-grid">
              <label>
                <span>T1</span>
                <input
                  type="number"
                  min="2017"
                  max="2026"
                  value={customAnalysis.yearT1}
                  onChange={(event) => updateCustomAnalysis("yearT1", event.target.value)}
                />
              </label>
              <label>
                <span>T2</span>
                <input
                  type="number"
                  min="2017"
                  max="2026"
                  value={customAnalysis.yearT2}
                  onChange={(event) => updateCustomAnalysis("yearT2", event.target.value)}
                />
              </label>
              <label>
                <span>Min ha</span>
                <input
                  type="number"
                  min="0.5"
                  max="25"
                  step="0.5"
                  value={customAnalysis.minAreaHa}
                  onChange={(event) => updateCustomAnalysis("minAreaHa", event.target.value)}
                />
              </label>
            </div>

            <button className="primary-button wide" type="submit" disabled={isLoading}>
              {isLoading ? "Memproses..." : "Jalankan Analisis"}
            </button>
          </form>
        </section>

        <section className="summary-panel custom-result-panel">
          <p className="eyebrow">Hasil Analisis</p>
          <strong>
            {customAnalysis.result ? `${formatNumber(customAnalysis.result.areaHa, 1)} ha` : "-"}
          </strong>
          <small>{customAnalysis.message}</small>
          {customAnalysis.result && (
            <div className="summary-grid">
              <span>
                <b>{formatNumber(customAnalysis.result.featureCount, 0)}</b>
                Polygon
              </span>
              <span>
                <b>{customAnalysis.result.period}</b>
                Periode
              </span>
              <span>
                <b>{transitionLabel(customAnalysis.result.topTransition)}</b>
                Transisi dominan
              </span>
              <span>
                <b>{customAnalysis.result.source}</b>
                Sumber
              </span>
            </div>
          )}
        </section>
      </div>
    </section>
  );
}
