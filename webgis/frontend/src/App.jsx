// Kasuari AI -- shell tipis: bungkus AppProvider, render Sidebar + TopHeader + halaman aktif.
// Semua state/logika ada di context/AppContext; tiap halaman ada di src/pages/.
import { Database } from "lucide-react";

import Sidebar from "./components/Sidebar";
import TopHeader from "./components/TopHeader";
import { AppProvider, useApp } from "./context/AppContext";
import AccuracyPage from "./pages/AccuracyPage";
import CustomAnalysisPage from "./pages/CustomAnalysisPage";
import DashboardPage from "./pages/DashboardPage";
import DownloadsPage from "./pages/DownloadsPage";
import MapPage from "./pages/MapPage";
import MethodologyPage from "./pages/MethodologyPage";
import StatisticsPage from "./pages/StatisticsPage";

const PAGES = {
  dashboard: DashboardPage,
  webgis: MapPage,
  statistics: StatisticsPage,
  accuracy: AccuracyPage,
  methodology: MethodologyPage,
  downloads: DownloadsPage,
  custom: CustomAnalysisPage,
};

function Shell() {
  const { data, error, activeView, sidebarCollapsed } = useApp();

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

  const ActivePage = PAGES[activeView] || DashboardPage;

  return (
    <main className={`app-shell app-frame ${sidebarCollapsed ? "sidebar-collapsed" : ""}`}>
      <Sidebar />
      <section className="main-stage">
        <TopHeader />
        <ActivePage />
      </section>
    </main>
  );
}

export default function App() {
  return (
    <AppProvider>
      <Shell />
    </AppProvider>
  );
}
