// Halaman 7 -- Analisis Wilayah Custom: jalankan inferensi NYATA on-demand untuk AOI kecil.
// Tiap klik: backend tarik komposit Sentinel-2 (T1 & T2) dari GEE -> model klasifikasi 7 kelas
// -> deteksi transisi Hutan -> non-hutan. Hasil + preview "sebelum vs sesudah" (klasifikasi
// model, bukan foto satelit) ditampilkan.
import { Info, Search } from "lucide-react";

import AoiPreviewMap from "../components/AoiPreviewMap";
import { CUSTOM_AOI_PRESETS } from "../lib/constants";
import { formatNumber, transitionLabel } from "../lib/format";
import { useApp } from "../context/AppContext";

export default function CustomAnalysisPage() {
  const { customAnalysis, runCustomAnalysis, updateCustomAnalysis } = useApp();
  const isLoading = customAnalysis.status === "loading";
  const isError = customAnalysis.status === "error";
  const result = customAnalysis.result;
  const hasPreview = result?.bounds && (result.landcoverT1Png || result.landcoverT2Png);

  return (
    <section className="page custom-page">
      <section className="method-hero">
        <p className="eyebrow">Analisis On-Demand</p>
        <h2>Analisis deforestasi langsung untuk satu wilayah kecil.</h2>
        <p>
          Pilih sebuah titik rawan, lalu Kasuari AI menarik komposit Sentinel-2 dua tahun (T1 &amp;
          T2) dari Google Earth Engine, menjalankan model klasifikasi tutupan lahan, dan menghitung
          area yang berubah dari <b>Hutan</b> ke kelas lain (sawit, tambang, lahan terbuka, dst).
          Berbeda dari halaman lain yang menampilkan hasil siap pakai, di sini inferensi dijalankan
          saat itu juga.
        </p>
      </section>

      <section className="panel-section custom-note">
        <div className="section-title">
          <Info size={18} />
          Cara kerja &amp; batasan
        </div>
        <ul className="custom-note-list">
          <li>Wilayah dibatasi maksimum <b>~12 km per sisi</b> (batas ukuran unduh Earth Engine).</li>
          <li>Butuh <b>backend aktif</b> + kredensial GEE (lihat <code>backend/SETUP_GEE.md</code>).</li>
          <li>Komposit memakai median <b>satu tahun penuh</b>; tahun berjalan bisa kurang bebas awan.</li>
          <li>
            Preview <b>Sebelum/Sesudah</b> adalah <b>klasifikasi model</b> tahun tersebut, bukan
            foto satelit tahun itu (citra satelit historis per-tahun tak tersedia gratis).
          </li>
          <li>Proses bisa memakan <b>beberapa menit</b> (unduh citra + inferensi 2 tahun).</li>
        </ul>
      </section>

      <div className="custom-layout">
        <section className="panel-section">
          <div className="section-title">
            <Search size={18} />
            Parameter Analisis
          </div>
          <form className="custom-analysis-form large" onSubmit={runCustomAnalysis}>
            <label className="select-row custom-select-row">
              <span>Wilayah</span>
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

        <section className={`summary-panel custom-result-panel ${isError ? "is-error" : ""}`}>
          <p className="eyebrow">Hasil Analisis</p>
          <strong>{result ? `${formatNumber(result.areaHa, 1)} ha` : "-"}</strong>
          <small>{customAnalysis.message}</small>
          {result && (
            <div className="summary-grid">
              <span>
                <b>{formatNumber(result.featureCount, 0)}</b>
                Polygon perubahan
              </span>
              <span>
                <b>{result.period}</b>
                Periode
              </span>
              <span>
                <b>{transitionLabel(result.topTransition)}</b>
                Transisi dominan
              </span>
              <span>
                <b>{result.aoiLabel}</b>
                Wilayah
              </span>
            </div>
          )}
        </section>
      </div>

      {hasPreview && (
        <section className="panel-section custom-preview-section">
          <div className="section-title">Peta Sebelum &amp; Sesudah (klasifikasi model)</div>
          <div className="aoi-preview-grid">
            <AoiPreviewMap
              title={`Sebelum (${customAnalysis.yearT1})`}
              pngUrl={result.landcoverT1Png}
              bounds={result.bounds}
              caption="Tutupan lahan hasil klasifikasi model"
            />
            <AoiPreviewMap
              title={`Sesudah (${customAnalysis.yearT2})`}
              pngUrl={result.landcoverT2Png}
              bounds={result.bounds}
              features={result.deforestation?.features || []}
              caption="Tutupan lahan + titik perubahan terdeteksi"
            />
          </div>
        </section>
      )}
    </section>
  );
}
