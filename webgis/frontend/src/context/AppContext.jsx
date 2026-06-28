// State, data turunan, dan aksi global aplikasi -- disediakan via Context supaya
// App.jsx tetap tipis (hanya pemanggil) dan halaman tinggal pakai useApp().
import { createContext, useContext, useEffect, useMemo, useState } from "react";

import { API_URL, loadData } from "../lib/api";
import { CUSTOM_AOI_PRESETS, TRANSITION_META } from "../lib/constants";

const AppContext = createContext(null);

export function useApp() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useApp harus dipakai di dalam <AppProvider>");
  return ctx;
}

export function AppProvider({ children }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [activeView, setActiveView] = useState("dashboard");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [activeYear, setActiveYear] = useState(2025);
  const [opacity, setOpacity] = useState(0); // 0 = overlay tutupan lahan OFF (default: peta satelit + titik)
  const [basemap, setBasemap] = useState("satelit");
  const [selectedProvince, setSelectedProvince] = useState("Semua Provinsi");
  const [selectedTransitions, setSelectedTransitions] = useState(
    () => new Set(Object.keys(TRANSITION_META))
  );
  const [customAnalysis, setCustomAnalysis] = useState({
    aoi: "merauke",
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
      filteredFeatures.reduce((sum, feature) => sum + Number(feature.properties?.area_ha || 0), 0),
    [filteredFeatures]
  );

  const topProvince = provinceRows[0];
  const topTransition = useMemo(
    () => [...transitionRows].sort((a, b) => b.value - a.value)[0],
    [transitionRows]
  );
  const allTransitionsSelected = selectedTransitions.size === Object.keys(TRANSITION_META).length;

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

    // Fitur ini menjalankan inferensi NYATA di server -- tak ada hasil tiruan offline.
    if (!API_URL) {
      setCustomAnalysis((current) => ({
        ...current,
        yearT1,
        yearT2,
        minAreaHa,
        status: "error",
        message:
          "Backend analisis belum aktif. Fitur ini butuh server + kredensial GEE " +
          "(set VITE_API_URL & ikuti backend/SETUP_GEE.md).",
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
      message: `Menarik citra & menganalisis ${preset.label} (bisa beberapa menit)...`,
      result: null,
    }));

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
      if (!response.ok) {
        let detail = `status ${response.status}`;
        try {
          const errBody = await response.json();
          if (errBody?.detail) detail = errBody.detail;
        } catch {
          /* abaikan: body bukan JSON */
        }
        throw new Error(detail);
      }
      const payload = await response.json();
      const stats = payload.statistics || {};
      const transitions = Object.entries(stats.per_transition_ha || {}).sort((a, b) => b[1] - a[1]);
      setCustomAnalysis((current) => ({
        ...current,
        status: "success",
        message: `${preset.label}: analisis ${yearT1}-${yearT2} selesai.`,
        result: {
          aoiLabel: preset.label,
          featureCount: stats.n_hotspots || payload.deforestation?.features?.length || 0,
          areaHa: Number(stats.total_deforestation_ha || 0),
          period: `${yearT1} - ${yearT2}`,
          topTransition: transitions[0]?.[0] || null,
          source: "Backend API",
          deforestation: payload.deforestation || null,
          bounds: payload.bounds || null,
          landcoverT1Png: payload.landcover_t1_png || null,
          landcoverT2Png: payload.landcover_t2_png || null,
        },
      }));
    } catch (err) {
      setCustomAnalysis((current) => ({
        ...current,
        status: "error",
        message: `Gagal: ${err.message}`,
        result: null,
      }));
    }
  }

  const value = {
    // data & status
    data,
    error,
    // navigasi & layout
    activeView,
    setActiveView,
    sidebarCollapsed,
    setSidebarCollapsed,
    // kontrol peta
    activeYear,
    setActiveYear,
    opacity,
    setOpacity,
    basemap,
    setBasemap,
    // filter
    selectedProvince,
    setSelectedProvince,
    selectedTransitions,
    toggleTransition,
    selectAllTransitions,
    allTransitionsSelected,
    resetFilters,
    focusTransition,
    focusProvince,
    // turunan
    transitionRows,
    provinceRows,
    filteredFeatures,
    filteredGeojson,
    selectedArea,
    topProvince,
    topTransition,
    // analisis custom
    customAnalysis,
    updateCustomAnalysis,
    runCustomAnalysis,
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}
