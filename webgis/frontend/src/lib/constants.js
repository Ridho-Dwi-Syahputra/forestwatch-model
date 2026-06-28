// Konstanta global aplikasi Kasuari AI (dipakai lintas halaman & komponen).
import {
  BarChart3,
  BookOpen,
  Download,
  Home,
  MapPinned,
  Search,
  Target,
} from "lucide-react";

// Tahun yang muncul di slider time-series peta.
export const YEARS = [2021, 2022, 2023, 2024, 2025, 2026];
// Tahun yang BENAR-BENAR punya layer tutupan lahan (sisanya pakai visual referensi terdekat).
export const AVAILABLE_LANDCOVER_YEARS = [2021, 2025];

// Skema 7 kelas final -> 5 jenis transisi deforestasi dari Hutan.
// color = warna polygon peta; chartColor = warna lembut utk chart.
export const TRANSITION_META = {
  hutan_ke_lahan_terbuka: {
    label: "Lahan Terbuka",
    longLabel: "Hutan ke Lahan Terbuka",
    color: "#7F1D1D",
    chartColor: "#B86A63",
  },
  hutan_ke_sawit: {
    label: "Sawit",
    longLabel: "Hutan ke Sawit",
    color: "#F97316",
    chartColor: "#DFA05E",
  },
  hutan_ke_pertanian_lain: {
    label: "Pertanian Lain",
    longLabel: "Hutan ke Pertanian Lain",
    color: "#EAB308",
    chartColor: "#D7BF63",
  },
  hutan_ke_tambang: {
    label: "Tambang",
    longLabel: "Hutan ke Tambang",
    color: "#8E24AA",
    chartColor: "#8F79A8",
  },
  hutan_ke_permukiman: {
    label: "Permukiman",
    longLabel: "Hutan ke Permukiman",
    color: "#757575",
    chartColor: "#8D9690",
  },
};

export const PROVINCE_CHART_COLORS = [
  "#2F6B57",
  "#4F8B75",
  "#78A894",
  "#A9C5B5",
  "#D7BF63",
  "#B8876F",
];

export const KPI_ACCENTS = {
  forest: "#2F6B57",
  teal: "#4F8B75",
  sage: "#7C9885",
  gold: "#B79A52",
};

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

// Preset AOI utk halaman Analisis Wilayah Custom (POST /api/analyze).
// Semua dibuat KECIL (~10 km/sisi) karena backend membatasi AOI <=12 km (batas unduh GEE).
// bbox = [lon_min, lat_min, lon_max, lat_max]. Titik = hotspot deforestasi yang relevan.
export const CUSTOM_AOI_PRESETS = {
  merauke: {
    label: "Merauke (Food Estate)",
    province: "Papua Selatan",
    bbox: [140.36, -8.54, 140.45, -8.45],
  },
  boven_digoel: {
    label: "Boven Digoel (Sawit)",
    province: "Papua Selatan",
    bbox: [140.22, -6.13, 140.32, -6.03],
  },
  timika: {
    label: "Timika (Tambang)",
    province: "Papua Tengah",
    bbox: [136.83, -4.55, 136.93, -4.45],
  },
  jayapura: {
    label: "Jayapura (Urban)",
    province: "Papua",
    bbox: [140.62, -2.63, 140.72, -2.53],
  },
};

// Item navigasi sidebar -- 7 halaman. Icon = komponen lucide-react (BUKAN emoji).
export const NAV_ITEMS = [
  { id: "dashboard", label: "Dashboard", icon: Home },
  { id: "webgis", label: "Peta Interaktif", icon: MapPinned },
  { id: "statistics", label: "Statistik", icon: BarChart3 },
  { id: "accuracy", label: "Akurasi Model", icon: Target },
  { id: "methodology", label: "Metodologi", icon: BookOpen },
  { id: "downloads", label: "Unduh Data", icon: Download },
  { id: "custom", label: "Analisis Custom", icon: Search },
];

// Tren perubahan tahunan (estimasi/placeholder utk chart tren di halaman Statistik).
export const YEAR_TREND = [
  { year: 2021, value: 0, label: "Baseline" },
  { year: 2022, value: 32150.4, label: "Estimasi" },
  { year: 2023, value: 68782.9, label: "Estimasi" },
  { year: 2024, value: 123440.2, label: "Estimasi" },
  { year: 2025, value: 181578.8, label: "Model" },
  { year: 2026, value: 181578.8, label: "Proyeksi datar" },
];

// Konfigurasi peta -- TERKUNCI ke Papua (tak bisa pan/zoom). bounds = [[S,W],[N,E]].
export const PAPUA = {
  center: [-4.5, 138],
  zoom: 6,
  minZoom: 5, // batas zoom-out: tetap fokus Papua (tak bisa keluar ke benua lain)
  maxZoom: 15, // batas zoom-in: lihat detail wilayah dari citra satelit
  bounds: [
    [-9.8, 129.5],
    [0.8, 141.5],
  ],
};

// Judul + subjudul header per halaman.
export const PAGE_TITLES = {
  dashboard: "Dashboard",
  webgis: "Peta Interaktif",
  statistics: "Statistik & Analitik",
  accuracy: "Akurasi Model",
  methodology: "Metodologi",
  downloads: "Unduh Data",
  custom: "Analisis Wilayah Custom",
};

export const PAGE_SUBTITLES = {
  dashboard: "Ringkasan temuan utama sebelum eksplorasi peta.",
  webgis: "Eksplorasi layer tutupan lahan, transisi, dan area perubahan.",
  statistics: "Bandingkan provinsi, transisi, kelas, dan tren perubahan.",
  accuracy: "Bukti performa model segmentasi tujuh kelas tutupan lahan.",
  methodology: "Sumber data, pipeline model, dan batasan interpretasi.",
  downloads: "Unduh artefak hasil model & endpoint API untuk transparansi data.",
  custom: "Analisis deforestasi on-demand untuk wilayah kecil yang dipilih.",
};
