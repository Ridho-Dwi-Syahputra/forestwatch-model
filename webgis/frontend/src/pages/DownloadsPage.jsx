// Halaman 6 -- Unduh Data: tombol unduh artefak (mengikuti mode data aktif).
import { BarChart3, Download, FileText, Layers, MapPinned, Target } from "lucide-react";

import { API_URL, localPath } from "../lib/api";
import { useApp } from "../context/AppContext";

export default function DownloadsPage() {
  const { data } = useApp();

  const files = [
    {
      title: "Deforestation GeoJSON",
      description: "Polygon perubahan hutan untuk layer peta dan analisis spasial.",
      href: API_URL ? `${API_URL}/api/download/geojson` : localPath("deforestation.geojson"),
      icon: MapPinned,
    },
    {
      title: "Deforestation CSV",
      description: "Tabel atribut polygon deforestasi untuk spreadsheet.",
      href: API_URL ? `${API_URL}/api/download/deforestation/csv` : localPath("deforestation.geojson"),
      icon: FileText,
    },
    {
      title: "Statistics JSON",
      description: "Ringkasan KPI, provinsi, transisi, dan luas kelas.",
      href: API_URL ? `${API_URL}/api/statistics` : localPath("statistics.json"),
      icon: BarChart3,
    },
    {
      title: "Legend JSON",
      description: "Kode kelas, nama label, dan warna legenda.",
      href: API_URL ? `${API_URL}/api/download/legend` : localPath("legend.json"),
      icon: Layers,
    },
    {
      title: "Metrics JSON",
      description: "Overall accuracy, Kappa, F1 per kelas, dan confusion matrix.",
      href: API_URL ? `${API_URL}/api/download/metrics` : localPath("metrics.json"),
      icon: Target,
    },
  ];

  return (
    <section className="page downloads-page">
      <section className="method-hero">
        <p className="eyebrow">Keterbukaan Data</p>
        <h2>Unduh artefak hasil model Kasuari AI untuk validasi.</h2>
        <p>
          Link mengikuti mode data aktif: file hasil model lokal saat frontend berdiri sendiri,
          atau endpoint backend ketika <b>{data.source}</b> digunakan.
        </p>
      </section>

      <div className="download-grid">
        {files.map((file) => {
          const Icon = file.icon;
          return (
            <a key={file.title} className="download-card" href={file.href} download>
              <span>
                <Icon size={20} />
              </span>
              <div>
                <strong>{file.title}</strong>
                <p>{file.description}</p>
              </div>
              <Download size={18} />
            </a>
          );
        })}
      </div>
    </section>
  );
}
