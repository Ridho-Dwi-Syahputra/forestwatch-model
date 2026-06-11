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
  Home,
  Info,
  Layers,
  LocateFixed,
  MapPinned,
  Menu,
  PanelLeftClose,
  PanelLeftOpen,
  RotateCcw,
  Search,
  SlidersHorizontal,
  Sparkles,
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
const YEARS = [2021, 2025];

const TRANSITION_META = {
  hutan_ke_lahan_terbuka: {
    label: "Lahan Terbuka",
    longLabel: "Hutan ke Lahan Terbuka",
    color: "#7F1D1D",
  },
  hutan_ke_sawit: {
    label: "Sawit",
    longLabel: "Hutan ke Sawit",
    color: "#F97316",
  },
  hutan_ke_pertanian_lain: {
    label: "Pertanian Lain",
    longLabel: "Hutan ke Pertanian Lain",
    color: "#EAB308",
  },
  hutan_ke_tambang: {
    label: "Tambang",
    longLabel: "Hutan ke Tambang",
    color: "#8E24AA",
  },
  hutan_ke_permukiman: {
    label: "Permukiman",
    longLabel: "Hutan ke Permukiman",
    color: "#757575",
  },
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

const NAV_ITEMS = [
  { id: "dashboard", label: "Dashboard", icon: Home },
  { id: "webgis", label: "WebGIS", icon: MapPinned },
  { id: "analysis", label: "Analisis", icon: BarChart3 },
  { id: "methodology", label: "Metodologi", icon: BookOpen },
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
      source: "Backend API",
    };
  }

  const [legend, bounds2021, bounds2025, deforestation, statistics] = await Promise.all([
    getJson(localPath("legend.json")),
    getJson(localPath("landcover_2021_bounds.json")),
    getJson(localPath("landcover_2025_bounds.json")),
    getJson(localPath("deforestation.geojson")),
    getJson(localPath("statistics.json")),
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

function MeraukeControl() {
  const map = useMap();
  return (
    <button
      className="map-action-button"
      type="button"
      onClick={() => map.flyTo([-8.5, 140.4], 9, { duration: 1.2 })}
      title="Pusat ke Merauke"
    >
      <LocateFixed size={16} />
      Merauke
    </button>
  );
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
          backgroundColor: transitionRows.map((row) => row.color),
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
          backgroundColor: ["#0B3D0B", "#2CA6A4", "#F97316", "#8E24AA", "#EAB308", "#607D8B"],
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
        <p>Memuat WebGIS ForestWatch Papua...</p>
      </main>
    );
  }

  const context = {
    activeYear,
    barData,
    barOptions,
    basemap,
    chartOptions,
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
    setOpacity,
    setSelectedProvince,
    toggleTransition,
    topProvince,
    topTransition,
    transitionRows,
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
        {activeView === "analysis" && <AnalysisPage {...context} />}
        {activeView === "methodology" && <MethodologyPage data={data} />}
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
          <strong>ForestWatch</strong>
          <small>Papua WebGIS</small>
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
    webgis: "WebGIS Interaktif",
    analysis: "Analisis Spasial",
    methodology: "Metodologi",
  }[activeView];

  const subtitle = {
    dashboard: "Ringkasan temuan utama sebelum eksplorasi peta.",
    webgis: "Eksplorasi layer tutupan lahan, transisi, dan area perubahan.",
    analysis: "Bandingkan provinsi, transisi, kelas, dan metrik model.",
    methodology: "Sumber data, pipeline model, dan batasan interpretasi.",
  }[activeView];

  return (
    <header className="topbar">
      <button className="mobile-menu-button" type="button" onClick={onMobileMenu}>
        <Menu size={18} />
      </button>
      <div>
        <p className="eyebrow">ForestWatch Papua</p>
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
            Dashboard ini memadatkan statistik utama, lalu WebGIS dipakai untuk
            menelusuri lokasi dan jenis perubahan secara interaktif.
          </p>
          <div className="hero-actions">
            <button className="primary-button" type="button" onClick={() => setActiveView("webgis")}>
              <MapPinned size={18} />
              Buka WebGIS
            </button>
            <button className="secondary-button" type="button" onClick={() => setActiveView("analysis")}>
              <BarChart3 size={18} />
              Lihat Analisis
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
          accent="#0B3D0B"
        />
        <KpiCard
          icon={Sparkles}
          label="Transisi dominan"
          value={topTransition?.label || "-"}
          note={`${formatNumber(topTransition?.value, 1)} ha`}
          accent={topTransition?.color || "#2CA6A4"}
        />
        <KpiCard
          icon={MapPinned}
          label="Provinsi tertinggi"
          value={topProvince?.province || "-"}
          note={`${formatNumber(topProvince?.deforestation_ha, 1)} ha`}
          accent="#F97316"
        />
        <KpiCard
          icon={Database}
          label="Mean IoU model"
          value={formatNumber(modelMetrics.mean_iou, 2)}
          note={`OA ${formatPercent(modelMetrics.overall_accuracy)}`}
          accent="#8E24AA"
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
            Gunakan halaman <b>WebGIS</b> untuk cek lokasi, layer, dan popup tiap polygon.
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
  const landcover = data.landcover?.[activeYear];
  const basemapConfig = BASEMAPS[basemap];
  const mapKey = `${activeYear}-${opacity}-${basemap}`;

  return (
    <section className="page webgis-page">
      <section className="map-panel" aria-label="Peta ForestWatch Papua">
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

          <MeraukeControl />
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
            Layer Peta
          </div>
          <div className="segmented">
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
          <button className="secondary-button wide" type="button" onClick={selectAllTransitions}>
            Tampilkan Semua Transisi
          </button>
        </section>
      </aside>
    </section>
  );
}

function AnalysisPage({
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
  const metrics = data.statistics.model_metrics?.per_class || [];
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
            <Database size={18} />
            Metrik Model per Kelas
          </div>
          <div className="data-table">
            {metrics.map((row) => (
              <div key={row.class}>
                <span>{row.class}</span>
                <strong>IoU {formatNumber(row.iou, 2)} / F1 {formatNumber(row.f1, 2)}</strong>
              </div>
            ))}
          </div>
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
        <h2>Data jadi ditampilkan di WebGIS, inferensi berat tetap offline.</h2>
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
          title="Output WebGIS"
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
