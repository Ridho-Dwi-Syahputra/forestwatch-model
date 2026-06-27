// Halaman 1 -- Dashboard: ringkasan angka besar sebelum eksplorasi peta.
import { Activity, BarChart3, Database, MapPinned, Search, Sparkles } from "lucide-react";
import { Bar, Doughnut } from "react-chartjs-2";

import KpiCard from "../components/KpiCard";
import { KPI_ACCENTS } from "../lib/constants";
import { barOptions, buildDoughnutData, buildProvinceBarData, chartOptions } from "../lib/charts";
import { formatNumber, formatPercent } from "../lib/format";
import { useApp } from "../context/AppContext";

export default function DashboardPage() {
  const {
    data,
    transitionRows,
    provinceRows,
    selectedArea,
    topProvince,
    topTransition,
    focusTransition,
    setActiveView,
  } = useApp();

  const modelMetrics = data.statistics.model_metrics || {};
  const doughnutData = buildDoughnutData(transitionRows);
  const barData = buildProvinceBarData(provinceRows);

  return (
    <section className="page dashboard-page">
      <div className="hero-band">
        <div>
          <p className="eyebrow">Overview 2021 - 2025</p>
          <h2>Pantau perubahan hutan Papua dari angka besar ke bukti spasial.</h2>
          <p>
            Dashboard ini memadatkan statistik utama, lalu peta interaktif dipakai untuk
            menelusuri lokasi dan jenis perubahan secara interaktif.
          </p>
          <div className="hero-actions">
            <button className="primary-button" type="button" onClick={() => setActiveView("webgis")}>
              <MapPinned size={18} />
              Buka Peta
            </button>
            <button
              className="secondary-button"
              type="button"
              onClick={() => setActiveView("statistics")}
            >
              <BarChart3 size={18} />
              Lihat Statistik
            </button>
          </div>
        </div>
        <div className="hero-summary">
          <span>Total perubahan</span>
          <strong>{formatNumber(data.statistics.total_deforestation_ha, 1)} ha</strong>
          <small>{formatNumber(data.statistics.n_hotspots, 0)} hotspot terdeteksi</small>
        </div>
      </div>

      <div className="kpi-grid">
        <KpiCard
          icon={Activity}
          label="Total perubahan hutan"
          value={`${formatNumber(data.statistics.total_deforestation_ha, 1)} ha`}
          accent={KPI_ACCENTS.forest}
        />
        <KpiCard
          icon={Sparkles}
          label="Transisi dominan"
          value={topTransition?.label || "-"}
          note={`${formatNumber(topTransition?.value, 1)} ha`}
          accent={KPI_ACCENTS.teal}
        />
        <KpiCard
          icon={MapPinned}
          label="Provinsi tertinggi"
          value={topProvince?.province || "-"}
          note={`${formatNumber(topProvince?.deforestation_ha, 1)} ha`}
          accent={KPI_ACCENTS.sage}
        />
        <KpiCard
          icon={Database}
          label="Mean IoU model"
          value={formatNumber(modelMetrics.mean_iou, 2)}
          note={`OA ${formatPercent(modelMetrics.overall_accuracy)}`}
          accent={KPI_ACCENTS.gold}
        />
      </div>

      <div className="dashboard-grid">
        <section className="panel-section large-panel">
          <div className="section-title">
            <Activity size={18} />
            Komposisi Transisi
          </div>
          <div className="chart-wrap dashboard-doughnut">
            <Doughnut
              data={doughnutData}
              options={{
                ...chartOptions,
                onClick: (_, elements) => {
                  const first = elements[0];
                  if (first) focusTransition(transitionRows[first.index].key);
                },
              }}
            />
          </div>
          <div className="transition-list compact-list">
            {transitionRows.map((row) => (
              <button key={row.key} type="button" onClick={() => focusTransition(row.key)}>
                <span>
                  <i style={{ backgroundColor: row.color }} />
                  {row.label}
                </span>
                <b>{formatNumber(row.value, 1)} ha</b>
              </button>
            ))}
          </div>
        </section>

        <section className="panel-section large-panel">
          <div className="section-title">
            <MapPinned size={18} />
            Peringkat Provinsi
          </div>
          <div className="chart-wrap dashboard-bar">
            <Bar data={barData} options={barOptions} />
          </div>
          <div className="insight-strip">
            <span>Area terfilter saat ini</span>
            <strong>{formatNumber(selectedArea, 1)} ha</strong>
          </div>
        </section>
      </div>

      <section className="panel-section insight-panel">
        <div className="section-title">
          <Search size={18} />
          Insight Cepat
        </div>
        <div className="insight-grid">
          <span>
            <b>{topTransition?.label}</b>
            adalah kategori terluas pada data saat ini.
          </span>
          <span>
            <b>{topProvince?.province}</b>
            menjadi provinsi dengan area perubahan terbesar.
          </span>
          <span>
            Gunakan halaman <b>Peta Interaktif</b> untuk cek lokasi, layer, dan popup tiap polygon.
          </span>
        </div>
      </section>
    </section>
  );
}
