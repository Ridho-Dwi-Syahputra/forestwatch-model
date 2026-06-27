// Halaman 3 -- Statistik & Analitik: pecahan per transisi/provinsi/kelas + tren tahunan.
import { Activity, Layers, MapPinned } from "lucide-react";
import { Bar, Doughnut } from "react-chartjs-2";

import { YEAR_TREND } from "../lib/constants";
import { barOptions, buildDoughnutData, buildProvinceBarData, chartOptions } from "../lib/charts";
import { formatNumber } from "../lib/format";
import { useApp } from "../context/AppContext";

function YearTrendChart() {
  const max = Math.max(...YEAR_TREND.map((item) => item.value));
  return (
    <div className="trend-chart">
      {YEAR_TREND.map((item) => (
        <div key={item.year} className="trend-row">
          <span>{item.year}</span>
          <div>
            <i style={{ width: `${Math.max(4, (item.value / max) * 100)}%` }} />
          </div>
          <strong>{formatNumber(item.value, 1)} ha</strong>
          <em>{item.label}</em>
        </div>
      ))}
    </div>
  );
}

export default function StatisticsPage() {
  const { data, transitionRows, provinceRows, focusProvince, focusTransition } = useApp();

  const doughnutData = buildDoughnutData(transitionRows);
  const barData = buildProvinceBarData(provinceRows);
  const classAreas = Object.entries(data.statistics.per_class_area_ha || {}).sort(
    (a, b) => b[1] - a[1]
  );

  return (
    <section className="page analysis-page">
      <div className="analysis-grid">
        <section className="panel-section analysis-chart-panel">
          <div className="section-title">
            <Activity size={18} />
            Transisi Deforestasi
          </div>
          <div className="chart-wrap analysis-doughnut">
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
          <div className="ranking-list">
            {transitionRows
              .slice()
              .sort((a, b) => b.value - a.value)
              .map((row, index) => (
                <button key={row.key} type="button" onClick={() => focusTransition(row.key)}>
                  <span>
                    <b>{index + 1}</b>
                    <i style={{ backgroundColor: row.color }} />
                    {row.longLabel}
                  </span>
                  <strong>{formatNumber(row.value, 1)} ha</strong>
                </button>
              ))}
          </div>
        </section>

        <section className="panel-section analysis-chart-panel">
          <div className="section-title">
            <MapPinned size={18} />
            Per Provinsi
          </div>
          <div className="chart-wrap analysis-bar">
            <Bar data={barData} options={barOptions} />
          </div>
          <div className="ranking-list">
            {provinceRows.map((row, index) => (
              <button key={row.province} type="button" onClick={() => focusProvince(row.province)}>
                <span>
                  <b>{index + 1}</b>
                  {row.province}
                </span>
                <strong>{formatNumber(row.deforestation_ha, 1)} ha</strong>
              </button>
            ))}
          </div>
        </section>
      </div>

      <div className="analysis-grid lower">
        <section className="panel-section">
          <div className="section-title">
            <Layers size={18} />
            Luas Kelas Tutupan Lahan
          </div>
          <div className="data-table">
            {classAreas.map(([name, value]) => (
              <div key={name}>
                <span>{name}</span>
                <strong>{formatNumber(value, 1)} ha</strong>
              </div>
            ))}
          </div>
        </section>

        <section className="panel-section">
          <div className="section-title">
            <Activity size={18} />
            Tren Perubahan Tahunan
          </div>
          <YearTrendChart />
        </section>
      </div>
    </section>
  );
}
