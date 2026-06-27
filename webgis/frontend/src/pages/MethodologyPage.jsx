// Halaman 5 -- Tentang & Metodologi: sumber data, pipeline, batasan interpretasi.
import { Info } from "lucide-react";

import MethodCard from "../components/MethodCard";
import { useApp } from "../context/AppContext";

export default function MethodologyPage() {
  const { data } = useApp();

  return (
    <section className="page methodology-page">
      <section className="method-hero">
        <p className="eyebrow">Pipeline</p>
        <h2>Data ditampilkan di Kasuari AI, inferensi berat tetap offline.</h2>
        <p>
          Aplikasi ini dibuat untuk presentasi dan eksplorasi hasil. Backend atau dummy-data
          menyajikan produk pre-computed dari model, sehingga antarmuka tetap ringan dan stabil.
        </p>
      </section>

      <div className="method-grid-cards">
        <MethodCard
          title="Citra dan Label"
          items={[
            "Sentinel-2 SR Harmonized",
            "Dynamic World base",
            "Hansen GFC 2025",
            "FDP Palm 2025a",
            "Global Mining Footprint",
          ]}
        />
        <MethodCard
          title="Model"
          items={[
            "ResNet50-U-Net",
            "Input 6 band Sentinel-2",
            "Output 7 kelas",
            "Focal-Tversky loss",
            "Weighted sampler",
          ]}
        />
        <MethodCard
          title="Output Kasuari AI"
          items={[
            "PNG landcover 2021 dan 2025",
            "GeoJSON deforestasi",
            "statistics.json",
            "legend.json",
            "metrics.json",
          ]}
        />
      </div>

      <section className="panel-section methodology-note">
        <div className="section-title">
          <Info size={18} />
          Catatan Interpretasi
        </div>
        <p>
          Data saat ini bersumber dari <b>{data.source}</b>. Jika masih dummy, angka dan polygon
          dipakai untuk pengembangan UI terlebih dahulu. Saat output model asli tersedia, file dapat
          diganti tanpa mengubah struktur frontend.
        </p>
      </section>
    </section>
  );
}
