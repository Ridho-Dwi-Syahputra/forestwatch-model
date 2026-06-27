// Pemuatan data: dari backend API (kalau VITE_API_URL diset) atau dummy-data lokal.
export const API_URL = import.meta.env.VITE_API_URL?.replace(/\/$/, "") || "";

export function localPath(path) {
  return `/${path.replace(/^\//, "")}`;
}

export async function getJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Gagal memuat ${url}`);
  return res.json();
}

export async function loadData() {
  if (API_URL) {
    const [legend, landcover2021, landcover2025, deforestation, statistics] = await Promise.all([
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
