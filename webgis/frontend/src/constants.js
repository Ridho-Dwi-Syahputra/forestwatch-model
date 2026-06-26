import {
  BarChart3,
  Crosshair,
  Download,
  Home,
  MapPinned,
  Microscope,
  ScanSearch,
  BookOpen,
} from "lucide-react";

export const API_URL = import.meta.env.VITE_API_URL?.replace(/\/$/, "") || "";
export const YEARS = [2021, 2025];

// Skema 5 kelas (Perairan, Hutan, Lahan Terbuka, Pertanian, Permukiman) -- Lahan Terbuka
// mencakup bekas tambang, Pertanian mencakup sawit & pertanian lain (lihat REMAP_7_TO_5 /
// TRANSITION_MAP_5 di model/src/forestwatch/constants.py).
export const TRANSITION_META = {
  hutan_ke_lahan_terbuka: {
    label: "Lahan Terbuka",
    longLabel: "Hutan ke Lahan Terbuka",
    color: "#7F1D1D",
  },
  hutan_ke_pertanian: {
    label: "Pertanian",
    longLabel: "Hutan ke Pertanian",
    color: "#F97316",
  },
  hutan_ke_permukiman: {
    label: "Permukiman",
    longLabel: "Hutan ke Permukiman",
    color: "#757575",
  },
};

export function transitionLabel(type) {
  return TRANSITION_META[type]?.longLabel || type?.replaceAll("_", " ") || "-";
}

export const BASEMAPS = {
  terang: {
    label: "Terang",
    url: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    attribution: "&copy; OpenStreetMap contributors",
  },
  satelit: {
    label: "Satelit",
    url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attribution: "Tiles &copy; Esri",
  },
};

// 8 halaman sidebar -- urutan sengaja: pre-computed dulu (instan, aman demo), baru fitur live
// (Analisis Custom) di paling bawah supaya jelas terpisah dari data yang sudah jadi.
export const NAV_ITEMS = [
  { id: "dashboard", label: "Dashboard", icon: Home },
  { id: "webgis", label: "Peta Interaktif", icon: MapPinned },
  { id: "analysis", label: "Statistik & Analitik", icon: BarChart3 },
  { id: "merauke", label: "Studi Kasus Merauke", icon: Crosshair },
  { id: "accuracy", label: "Akurasi Model", icon: Microscope },
  { id: "methodology", label: "Tentang & Metodologi", icon: BookOpen },
  { id: "download", label: "Unduh Data", icon: Download },
  { id: "custom", label: "Analisis Wilayah Custom", icon: ScanSearch },
];

export const PAGE_META = {
  dashboard: {
    title: "Dashboard",
    subtitle: "Ringkasan temuan utama sebelum eksplorasi peta.",
  },
  webgis: {
    title: "Peta Interaktif",
    subtitle: "Eksplorasi layer tutupan lahan, transisi, dan area perubahan -- seluruh Papua.",
  },
  analysis: {
    title: "Statistik & Analitik",
    subtitle: "Bandingkan provinsi, transisi, dan kelas tutupan lahan.",
  },
  merauke: {
    title: "Studi Kasus Merauke",
    subtitle: "Zoom ke food estate Papua Selatan -- bukti spasial untuk paragraf esai.",
  },
  accuracy: {
    title: "Akurasi & Metodologi Model",
    subtitle: "Confusion matrix, IoU per kelas, dan metrik evaluasi model.",
  },
  methodology: {
    title: "Tentang & Metodologi",
    subtitle: "Sumber data, pipeline model, dan batasan interpretasi.",
  },
  download: {
    title: "Unduh Data",
    subtitle: "Data terbuka -- sejalan dengan prinsip Satu Data Indonesia.",
  },
  custom: {
    title: "Analisis Wilayah Custom",
    subtitle: "Gambar kotak wilayah kecil, jalankan model secara live (butuh backend aktif).",
  },
};
