import { useEffect, useMemo, useState } from "react";
import {
  ArcElement,
  BarElement,
  CategoryScale,
  Chart as ChartJS,
  Legend as ChartLegend,
  LinearScale,
  Tooltip,
} from "chart.js";
import { Bar, Doughnut } from "react-chartjs-2";
import {
  Activity,
  BarChart3,
  BookOpen,
  ChevronDown,
  Database,
  Download,
  FileText,
  Home,
  Info,
  Layers,
  MapPinned,
  Menu,
  PanelLeftClose,
  PanelLeftOpen,
  RotateCcw,
  Search,
  SlidersHorizontal,
  Sparkles,
  Target,
} from "lucide-react";
import {
  GeoJSON,
  ImageOverlay,
  MapContainer,
  TileLayer,
  Tooltip as LeafletTooltip,
  useMap,
} from "react-leaflet";

ChartJS.register(ArcElement, BarElement, CategoryScale, LinearScale, Tooltip, ChartLegend);

const API_URL = import.meta.env.VITE_API_URL?.replace(/\/$/, "") || "";
const YEARS = [2021, 2022, 2023, 2024, 2025, 2026];
const AVAILABLE_LANDCOVER_YEARS = [2021, 2025];

const TRANSITION_META = {
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

const PROVINCE_CHART_COLORS = ["#2F6B57", "#4F8B75", "#78A894", "#A9C5B5", "#D7BF63", "#B8876F"];
const KPI_ACCENTS = {
  forest: "#2F6B57",
  teal: "#4F8B75",
  sage: "#7C9885",
  gold: "#B79A52",
};

const BASEMAPS = {
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

const CUSTOM_AOI_PRESETS = {
  papua: {
    label: "Papua Overview",
    province: "Semua Provinsi",
    bbox: [130.0, -9.5, 141.2, 0.5],
  },
  papua_selatan: {
    label: "Papua Selatan",
    province: "Papua Selatan",
    bbox: [137.0, -9.5, 141.2, -5.0],
  },
  mimika: {
    label: "Mimika",
    province: "Papua Tengah",
    bbox: [135.4, -5.8, 137.5, -3.6],
  },
  jayapura: {
    label: "Jayapura",
    province: "Papua",
    bbox: [139.0, -3.2, 140.4, -2.0],
  },
};

const NAV_ITEMS = [
  { id: "dashboard", label: "Dashboard", icon: Home },
  { id: "webgis", label: "Peta Interaktif", icon: MapPinned },
  { id: "statistics", label: "Statistik", icon: BarChart3 },
  { id: "accuracy", label: "Akurasi Model", icon: Target },
  { id: "methodology", label: "Metodologi", icon: BookOpen },
  { id: "downloads", label: "Unduh Data", icon: Download },
  { id: "custom", label: "Analisis Custom", icon: Search },
];

const YEAR_TREND = [
  { year: 2021, value: 0, label: "Baseline" },
  { year: 2022, value: 32150.4, label: "Estimasi" },
  { year: 2023, value: 68782.9, label: "Estimasi" },
  { year: 2024, value: 123440.2, label: "Estimasi" },
  { year: 2025, value: 181578.8, label: "Model" },
  { year: 2026, value: 181578.8, label: "Proyeksi datar" },
];

function localPath(path) {
  return `/${path.replace(/^\//, "")}`;
}

async function getJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Gagal memuat ${url}`);
  return res.json();
}

async function loadData() {
  if (API_URL) {
    const [legend, landcover2021, landcover2025, deforestation, statistics] =
      await Promise.all([
        getJson(`${API_URL}/api/legend`),
        getJson(`${API_URL}/api/landcover/2021`),
        getJson(`${API_URL}/api/landcover/2025`),
        getJson(`${API_URL}/api/deforestation`),
        getJson(`${API_URL}/api/statistics`),
      ]);
    return {
      legend,
      landcover: { 2021: landcover2021, 2025: landcover2025 },
      deforestation,
      statistics,
      metrics: statistics.model_metrics || {},
      source: "Backend API",
    };
  }

  const [legend, bounds2021, bounds2025, deforestation, statistics, metrics] = await Promise.all([
    getJson(localPath("legend.json")),
    getJson(localPath("landcover_2021_bounds.json")),
    getJson(localPath("landcover_2025_bounds.json")),
    getJson(localPath("deforestation.geojson")),
    getJson(localPath("statistics.json")),
    getJson(localPath("metrics.json")),
  ]);

  return {
    legend,
    landcover: {
      2021: {
        year: 2021,
        image_url: localPath("landcover_2021.png"),
        bounds: bounds2021.bounds,
        crs: bounds2021.crs,
      },
      2025: {
        year: 2025,
        image_url: localPath("landcover_2025.png"),
        bounds: bounds2025.bounds,
        crs: bounds2025.crs,
      },
    },
    deforestation,
    statistics,
    metrics,
    source: "Dummy Data",
  };
}

function formatNumber(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return new Intl.NumberFormat("id-ID", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(Number(value));
}

function formatPercent(value) {
  if (value === null || value === undefined) return "-";
  return `${formatNumber(Number(value) * 100, 0)}%`;
}

function transitionLabel(type) {
  return TRANSITION_META[type]?.longLabel || type?.replaceAll("_", " ") || "-";
}

function FitPapuaButton() {
  const map = useMap();
  return (
    <button
      className="icon-button map-reset-button"
      type="button"
      onClick={() => map.flyTo([-4.5, 138], 6, { duration: 1 })}
      title="Kembali ke Papua"
    >
      <RotateCcw size={16} />
    </button>
  );
}

function App() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [activeView, setActiveView] = useState("dashboard");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [activeYear, setActiveYear] = useState(2025);
  const [opacity, setOpacity] = useState(0.72);
  const [basemap, setBasemap] = useState("terang");
  const [selectedProvince, setSelectedProvince] = useState("Semua Provinsi");
  const [selectedTransitions, setSelectedTransitions] = useState(
    () => new Set(Object.keys(TRANSITION_META))
  );
  const [customAnalysis, setCustomAnalysis] = useState({
    aoi: "papua_selatan",
    yearT1: 2021,
    yearT2: 2025,
    minAreaHa: 1,
    status: "idle",
    message: "Siap menjalankan analisis wilayah.",
    result: null,
  });

  useEffect(() => {
    let mounted = true;
    loadData()
      .then((payload) => mounted && setData(payload))
      .catch((err) => mounted && setError(err.message));
    return () => {
      mounted = false;
    };
  }, []);

  const transitionRows = useMemo(() => {
    const perTransition = data?.statistics?.per_transition_ha || {};
    return Object.entries(TRANSITION_META).map(([key, meta]) => ({
      key,
      ...meta,
      value: Number(perTransition[key] || 0),
    }));
  }, [data]);

  const provinceRows = useMemo(() => {
    return [...(data?.statistics?.per_province || [])].sort(
      (a, b) => Number(b.deforestation_ha) - Number(a.deforestation_ha)
    );
  }, [data]);

  const filteredFeatures = useMemo(() => {
    const features = data?.deforestation?.features || [];
    return features.filter((feature) => {
      const p = feature.properties || {};
      const transitionOk = selectedTransitions.has(p.transition_type);
      const provinceOk =
        selectedProvince === "Semua Provinsi" || p.province === selectedProvince;
      return transitionOk && provinceOk;
    });
  }, [data, selectedTransitions, selectedProvince]);

  const filteredGeojson = useMemo(
    () => ({ type: "FeatureCollection", features: filteredFeatures }),
    [filteredFeatures]
  );

  const selectedArea = useMemo(
    () =>
      filteredFeatures.reduce(
        (sum, feature) => sum + Number(feature.properties?.area_ha || 0),
        0
      ),
    [filteredFeatures]
  );

  const topProvince = provinceRows[0];
  const topTransition = useMemo(
    () => [...transitionRows].sort((a, b) => b.value - a.value)[0],
    [transitionRows]
  );
  const allTransitionsSelected = selectedTransitions.size === Object.keys(TRANSITION_META).length;

  const chartOptions = useMemo(
    () => ({
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (context) => `${context.label}: ${formatNumber(context.raw, 1)} ha`,
          },
        },
      },
    }),
    []
  );

  const doughnutData = useMemo(
    () => ({
      labels: transitionRows.map((row) => row.label),
      datasets: [
        {
          data: transitionRows.map((row) => row.value),
          backgroundColor: transitionRows.map((row) => row.chartColor),
          borderColor: "#ffffff",
          borderWidth: 3,
        },
      ],
    }),
    [transitionRows]
  );

  const barData = useMemo(
    () => ({
      labels: provinceRows.map((row) => row.province),
      datasets: [
        {
          label: "ha",
          data: provinceRows.map((row) => row.deforestation_ha),
          backgroundColor: provinceRows.map(
            (_, index) => PROVINCE_CHART_COLORS[index % PROVINCE_CHART_COLORS.length]
          ),
          borderRadius: 6,
          barThickness: 18,
        },
      ],
    }),
    [provinceRows]
  );

  const barOptions = useMemo(
    () => ({
      ...chartOptions,
      indexAxis: "y",
      scales: {
        x: { grid: { color: "#e3ece6" }, ticks: { color: "#53645b" } },
        y: { grid: { display: false }, ticks: { color: "#314239" } },
      },
    }),
    [chartOptions]
  );

  function toggleTransition(key) {
    setSelectedTransitions((current) => {
      const next = new Set(current);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  function focusTransition(key) {
    setSelectedTransitions(new Set([key]));
    setActiveView("webgis");
  }

  function focusProvince(province) {
    setSelectedProvince(province);
    setActiveView("webgis");
  }

  function selectAllTransitions() {
    setSelectedTransitions(new Set(Object.keys(TRANSITION_META)));
  }

  function resetFilters() {
    selectAllTransitions();
    setSelectedProvince("Semua Provinsi");
  }

  function updateCustomAnalysis(field, value) {
    setCustomAnalysis((current) => ({
      ...current,
      [field]: value,
      status: current.status === "success" || current.status === "error" ? "idle" : current.status,
      message:
        current.status === "success" || current.status === "error"
          ? "Siap menjalankan analisis wilayah."
          : current.message,
    }));
  }

  async function runCustomAnalysis(event) {
    event.preventDefault();

    const yearT1 = Number(customAnalysis.yearT1);
    const yearT2 = Number(customAnalysis.yearT2);
    const minAreaHa = Number(customAnalysis.minAreaHa);
    const preset = CUSTOM_AOI_PRESETS[customAnalysis.aoi] || CUSTOM_AOI_PRESETS.papua;

    if (!Number.isFinite(yearT1) || !Number.isFinite(yearT2) || yearT2 <= yearT1) {
      setCustomAnalysis((current) => ({
        ...current,
        status: "error",
        message: "T2 harus lebih besar dari T1.",
        result: null,
      }));
      return;
    }

    setCustomAnalysis((current) => ({
      ...current,
      yearT1,
      yearT2,
      minAreaHa,
      status: "loading",
      message: API_URL ? "Mengirim request ke backend /api/analyze..." : "Menjalankan simulasi lokal...",
      result: null,
    }));

    if (API_URL) {
      try {
        const response = await fetch(`${API_URL}/api/analyze`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            aoi: preset.bbox,
            year_t1: yearT1,
            year_t2: yearT2,
            min_area_ha: minAreaHa,
          }),
        });
        if (!response.ok) throw new Error(`Backend mengembalikan status ${response.status}`);
        const payload = await response.json();
        const stats = payload.statistics || {};
        const transitions = Object.entries(stats.per_transition_ha || {}).sort((a, b) => b[1] - a[1]);
        setCustomAnalysis((current) => ({
          ...current,
          status: "success",
          message: payload.message || `${preset.label}: hasil backend siap ditampilkan.`,
          result: {
            aoiLabel: preset.label,
            featureCount: stats.n_hotspots || payload.deforestation?.features?.length || 0,
            areaHa: Number(stats.total_deforestation_ha || 0),
            period: `${yearT1} - ${yearT2}`,
            topTransition: transitions[0]?.[0] || null,
            source: "Backend API",
          },
        }));
        return;
      } catch (err) {
        setCustomAnalysis((current) => ({
          ...current,
          status: "error",
          message: `Gagal memanggil backend: ${err.message}`,
          result: null,
        }));
        return;
      }
    }

    const features = data.deforestation.features.filter((feature) => {
      const props = feature.properties || {};
      const provinceOk =
        preset.province === "Semua Provinsi" || props.province === preset.province;
      return provinceOk && Number(props.area_ha || 0) >= minAreaHa;
    });
    const areaHa = features.reduce(
      (sum, feature) => sum + Number(feature.properties?.area_ha || 0),
      0
    );
    const transitionArea = features.reduce((acc, feature) => {
      const type = feature.properties?.transition_type || "lainnya";
      acc[type] = (acc[type] || 0) + Number(feature.properties?.area_ha || 0);
      return acc;
    }, {});
    const topTransition = Object.entries(transitionArea).sort((a, b) => b[1] - a[1])[0];

    setCustomAnalysis((current) => ({
      ...current,
      yearT1,
      yearT2,
      minAreaHa,
      status: "success",
      message: `${preset.label}: ${features.length} polygon simulasi siap.`,
      result: {
        aoiLabel: preset.label,
        featureCount: features.length,
        areaHa,
        period: `${yearT1} - ${yearT2}`,
        topTransition: topTransition?.[0] || null,
        source: "Simulasi lokal",
      },
    }));
  }

  if (error) {
    return (
      <main className="app-shell app-error">
        <div className="error-panel">
          <Database size={28} />
          <h1>Data belum bisa dimuat</h1>
          <p>{error}</p>
        </div>
      </main>
    );
  }

  if (!data) {
    return (
      <main className="app-shell loading-shell">
        <div className="loading-mark" />
        <p>Memuat Kasuari AI...</p>
      </main>
    );
  }

  const context = {
    activeYear,
    allTransitionsSelected,
    barData,
    barOptions,
    basemap,
    chartOptions,
    customAnalysis,
    data,
    doughnutData,
    filteredFeatures,
    filteredGeojson,
    focusProvince,
    focusTransition,
    opacity,
    provinceRows,
    resetFilters,
    selectAllTransitions,
    selectedArea,
    selectedProvince,
    selectedTransitions,
    setActiveView,
    setActiveYear,
    setBasemap,
    setCustomAnalysis,
    setOpacity,
    setSelectedProvince,
    runCustomAnalysis,
    toggleTransition,
    topProvince,
    topTransition,
    transitionRows,
    updateCustomAnalysis,
  };

  return (
    <main className={`app-shell app-frame ${sidebarCollapsed ? "sidebar-collapsed" : ""}`}>
      <Sidebar
        activeView={activeView}
        collapsed={sidebarCollapsed}
        dataSource={data.source}
        onCollapse={() => setSidebarCollapsed((value) => !value)}
        onNavigate={setActiveView}
      />

      <section className="main-stage">
        <TopHeader
          activeView={activeView}
          dataSource={data.source}
          onMobileMenu={() => setSidebarCollapsed((value) => !value)}
        />

        {activeView === "dashboard" && <DashboardPage {...context} />}
        {activeView === "webgis" && <WebGISPage {...context} />}
        {activeView === "statistics" && <StatisticsPage {...context} />}
        {activeView === "accuracy" && <AccuracyPage data={data} />}
        {activeView === "methodology" && <MethodologyPage data={data} />}
        {activeView === "downloads" && <DownloadPage data={data} />}
        {activeView === "custom" && <CustomAnalysisPage {...context} />}
      </section>
    </main>
  );
}

function Sidebar({ activeView, collapsed, dataSource, onCollapse, onNavigate }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <span className="brand-icon">
          <MapPinned size={21} />
        </span>
        <div className="sidebar-brand-text">
          <strong>Kasuari AI</strong>
          <small>Papua Forest Intelligence</small>
        </div>
      </div>

      <nav className="sidebar-nav" aria-label="Navigasi utama">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.id}
              className={activeView === item.id ? "active" : ""}
              type="button"
              onClick={() => onNavigate(item.id)}
              title={item.label}
            >
              <Icon size={19} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="sidebar-footer">
        <span className="data-pill">
          <Database size={15} />
          <span>{dataSource}</span>
        </span>
        <button className="sidebar-toggle" type="button" onClick={onCollapse}>
          {collapsed ? <PanelLeftOpen size={18} /> : <PanelLeftClose size={18} />}
          <span>{collapsed ? "Buka" : "Collapse"}</span>
        </button>
      </div>
    </aside>
  );
}

function TopHeader({ activeView, dataSource, onMobileMenu }) {
  const title = {
    dashboard: "Dashboard",
    webgis: "Peta Interaktif",
    statistics: "Statistik & Analitik",
    accuracy: "Akurasi Model",
    methodology: "Metodologi",
    downloads: "Unduh Data",
    custom: "Analisis Wilayah Custom",
  }[activeView];

  const subtitle = {
    dashboard: "Ringkasan temuan utama sebelum eksplorasi peta.",
    webgis: "Eksplorasi layer tutupan lahan, transisi, dan area perubahan.",
    statistics: "Bandingkan provinsi, transisi, kelas, dan tren perubahan.",
    accuracy: "Bukti performa model segmentasi tujuh kelas tutupan lahan.",
    methodology: "Sumber data, pipeline model, dan batasan interpretasi.",
    downloads: "Akses file dummy dan endpoint unduhan untuk transparansi data.",
    custom: "Kirim AOI dan rentang tahun ke backend analisis.",
  }[activeView];

  return (
    <header className="topbar">
      <button className="mobile-menu-button" type="button" onClick={onMobileMenu}>
        <Menu size={18} />
      </button>
      <div>
        <p className="eyebrow">Kasuari AI</p>
        <h1>{title}</h1>
        <p>{subtitle}</p>
      </div>
      <span className="data-pill topbar-source">
        <Database size={15} />
        {dataSource}
      </span>
    </header>
  );
}

function DashboardPage({
  barData,
  barOptions,
  chartOptions,
  data,
  doughnutData,
  focusTransition,
  provinceRows,
  selectedArea,
  setActiveView,
  topProvince,
  topTransition,
  transitionRows,
}) {
  const modelMetrics = data.statistics.model_metrics || {};

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
            <button className="secondary-button" type="button" onClick={() => setActiveView("statistics")}>
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

function KpiCard({ accent, icon: Icon, label, note, value }) {
  return (
    <section className="kpi-card" style={{ "--accent": accent }}>
      <span className="kpi-icon">
        <Icon size={20} />
      </span>
      <p>{label}</p>
      <strong>{value}</strong>
      {note && <small>{note}</small>}
    </section>
  );
}

function WebGISPage({
  activeYear,
  allTransitionsSelected,
  basemap,
  data,
  filteredFeatures,
  filteredGeojson,
  opacity,
  provinceRows,
  resetFilters,
  selectAllTransitions,
  selectedArea,
  selectedProvince,
  selectedTransitions,
  setActiveYear,
  setBasemap,
  setOpacity,
  setSelectedProvince,
  toggleTransition,
  transitionRows,
}) {
  const landcoverYear = AVAILABLE_LANDCOVER_YEARS.includes(activeYear)
    ? activeYear
    : activeYear < 2024
      ? 2021
      : 2025;
  const landcover = data.landcover?.[landcoverYear];
  const isEstimatedLayer = landcoverYear !== activeYear;
  const basemapConfig = BASEMAPS[basemap];
  const mapKey = `${activeYear}-${landcoverYear}-${opacity}-${basemap}`;

  return (
    <section className="page webgis-page">
      <section className="map-panel" aria-label="Peta Kasuari AI Papua">
        <MapContainer
          center={[-4.5, 138]}
          zoom={6}
          minZoom={5}
          maxZoom={12}
          scrollWheelZoom
          className="leaflet-map"
        >
          <TileLayer key={basemap} attribution={basemapConfig.attribution} url={basemapConfig.url} />

          {landcover && (
            <ImageOverlay
              key={mapKey}
              url={landcover.image_url}
              bounds={landcover.bounds}
              opacity={opacity}
            />
          )}

          <GeoJSON
            key={`${filteredFeatures.length}-${Array.from(selectedTransitions).join("-")}-${selectedProvince}`}
            data={filteredGeojson}
            style={(feature) => {
              const type = feature?.properties?.transition_type;
              const color = TRANSITION_META[type]?.color || "#123D2F";
              return { color, weight: 1.6, fillColor: color, fillOpacity: 0.58 };
            }}
            onEachFeature={(feature, layer) => {
              const p = feature.properties || {};
              layer.bindPopup(`
                <div class="popup-card">
                  <strong>${p.id || "-"}</strong>
                  <span>${transitionLabel(p.transition_type)}</span>
                  <dl>
                    <dt>Luas</dt><dd>${formatNumber(p.area_ha, 1)} ha</dd>
                    <dt>Periode</dt><dd>${p.period_from || "-"} - ${p.period_to || "-"}</dd>
                    <dt>Provinsi</dt><dd>${p.province || "-"}</dd>
                    <dt>Kawasan</dt><dd>${p.kawasan_status || "-"}</dd>
                  </dl>
                </div>
              `);
            }}
          >
            <LeafletTooltip sticky>Area perubahan hutan</LeafletTooltip>
          </GeoJSON>

          <FitPapuaButton />
        </MapContainer>

        <div className="legend-box">
          <strong>Legenda</strong>
          {data.legend.map((item) => (
            <span key={item.id}>
              <i style={{ backgroundColor: item.color }} />
              {item.name}
            </span>
          ))}
        </div>
      </section>

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
              ? `Layer ${activeYear} memakai visual referensi ${landcoverYear} sampai komposit tahunan tersedia.`
              : `Layer ${activeYear} tersedia dari dummy data.`}
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
            <select value={selectedProvince} onChange={(event) => setSelectedProvince(event.target.value)}>
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

function StatisticsPage({
  barData,
  barOptions,
  chartOptions,
  data,
  doughnutData,
  focusProvince,
  focusTransition,
  provinceRows,
  transitionRows,
}) {
  const classAreas = Object.entries(data.statistics.per_class_area_ha || {}).sort((a, b) => b[1] - a[1]);

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

function AccuracyPage({ data }) {
  const metrics = data.metrics || data.statistics.model_metrics || {};
  const rows = metrics.per_class || [];
  const matrix = metrics.confusion_matrix || [];
  const labels = rows.map((row) => row.class);
  const maxCell = Math.max(...matrix.flat().map((value) => Number(value || 0)), 1);

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
          label="Mean IoU"
          value={formatPercent(metrics.mean_iou)}
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
                <FragmentRow
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
            IoU dan F1 per Kelas
          </div>
          <div className="data-table metric-table">
            {rows.map((row) => (
              <div key={row.class}>
                <span>{row.class}</span>
                <strong>IoU {formatPercent(row.iou)} / F1 {formatPercent(row.f1)}</strong>
              </div>
            ))}
          </div>
        </section>
      </div>
    </section>
  );
}

function FragmentRow({ label, maxCell, row }) {
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

function DownloadPage({ data }) {
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
      description: "Overall accuracy, mIoU, Kappa, IoU/F1, dan confusion matrix.",
      href: API_URL ? `${API_URL}/api/download/metrics` : localPath("metrics.json"),
      icon: Target,
    },
  ];

  return (
    <section className="page downloads-page">
      <section className="method-hero">
        <p className="eyebrow">Keterbukaan Data</p>
        <h2>Unduh artefak data Kasuari AI untuk validasi dan demo.</h2>
        <p>
          Link mengikuti mode data aktif: file dummy lokal saat frontend berdiri sendiri,
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

function CustomAnalysisPage({
  customAnalysis,
  data,
  runCustomAnalysis,
  updateCustomAnalysis,
}) {
  const isLoading = customAnalysis.status === "loading";

  return (
    <section className="page custom-page">
      <section className="method-hero">
        <p className="eyebrow">Live Capability</p>
        <h2>Analisis AOI kecil dari frontend ke backend.</h2>
        <p>
          Form ini sudah siap memanggil <b>POST /api/analyze</b> ketika backend API aktif.
          Tanpa API URL, Kasuari AI menampilkan simulasi lokal dari dummy GeoJSON.
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
          <strong>{customAnalysis.result ? `${formatNumber(customAnalysis.result.areaHa, 1)} ha` : "-"}</strong>
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

function MethodologyPage({ data }) {
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
          items={["Sentinel-2 SR Harmonized", "Dynamic World base", "Hansen GFC 2025", "FDP Palm 2025a", "Global Mining Footprint"]}
        />
        <MethodCard
          title="Model"
          items={["ResNet50-U-Net", "Input 6 band Sentinel-2", "Output 7 kelas", "Focal-Tversky loss", "Weighted sampler"]}
        />
        <MethodCard
          title="Output Kasuari AI"
          items={["PNG landcover 2021 dan 2025", "GeoJSON deforestasi", "statistics.json", "legend.json", "metrics.json"]}
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

function MethodCard({ items, title }) {
  return (
    <section className="method-card">
      <h3>{title}</h3>
      <ul>
        {items.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </section>
  );
}

export default App;
