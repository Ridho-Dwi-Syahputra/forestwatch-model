// Halaman 4 -- Akurasi Model: confusion matrix heatmap, F1 per kelas, KPI OA/Kappa/F1.
import { Activity, BarChart3, Database, Sparkles, Target } from "lucide-react";

import KpiCard from "../components/KpiCard";
import { KPI_ACCENTS } from "../lib/constants";
import { formatNumber, formatPercent } from "../lib/format";
import { useApp } from "../context/AppContext";

function MatrixRow({ label, maxCell, row }) {
  return (
    <>
      <strong className="matrix-row-label">{label}</strong>
      {row.map((value, index) => {
        const intensity = Number(value || 0) / maxCell;
        return (
          <span
            key={`${label}-${index}`}
            className="matrix-cell"
            style={{ backgroundColor: `rgba(47, 107, 87, ${0.12 + intensity * 0.72})` }}
            title={`${label}: ${value}`}
          >
            {formatNumber(value, 0)}
          </span>
        );
      })}
    </>
  );
}

export default function AccuracyPage() {
  const { data } = useApp();
  const metrics = data.metrics || data.statistics.model_metrics || {};
  const rows = metrics.per_class || [];
  const matrix = metrics.confusion_matrix || [];
  const labels = rows.map((row) => row.class);
  const maxCell = Math.max(...matrix.flat().map((value) => Number(value || 0)), 1);
  const f1Macro = rows.length
    ? rows.reduce((sum, row) => sum + (Number(row.f1) || 0), 0) / rows.length
    : 0;

  return (
    <section className="page accuracy-page">
      <div className="kpi-grid compact">
        <KpiCard
          icon={Target}
          label="Overall Accuracy"
          value={formatPercent(metrics.overall_accuracy)}
          accent={KPI_ACCENTS.forest}
        />
        <KpiCard
          icon={Activity}
          label="F1 Makro"
          value={formatPercent(f1Macro)}
          accent={KPI_ACCENTS.teal}
        />
        <KpiCard
          icon={Sparkles}
          label="Kappa"
          value={formatNumber(metrics.kappa, 3)}
          accent={KPI_ACCENTS.gold}
        />
        <KpiCard
          icon={Database}
          label="Kelas Model"
          value={`${rows.length} kelas`}
          note="Semantic segmentation"
          accent={KPI_ACCENTS.sage}
        />
      </div>

      <div className="analysis-grid accuracy-grid">
        <section className="panel-section">
          <div className="section-title">
            <Target size={18} />
            Confusion Matrix
          </div>
          <div className="matrix-wrap">
            <div className="matrix-grid" style={{ "--matrix-size": labels.length + 1 }}>
              <span className="matrix-corner">Pred</span>
              {labels.map((label) => (
                <b key={`head-${label}`}>{label}</b>
              ))}
              {matrix.map((row, rowIndex) => (
                <MatrixRow
                  key={labels[rowIndex] || rowIndex}
                  label={labels[rowIndex] || `Kelas ${rowIndex + 1}`}
                  maxCell={maxCell}
                  row={row}
                />
              ))}
            </div>
          </div>
        </section>

        <section className="panel-section">
          <div className="section-title">
            <BarChart3 size={18} />
            F1 per Kelas
          </div>
          <div className="data-table metric-table">
            {rows.map((row) => (
              <div key={row.class}>
                <span>{row.class}</span>
                <strong>F1 {formatPercent(row.f1)}</strong>
              </div>
            ))}
          </div>
        </section>
      </div>
    </section>
  );
}
