// Sidebar navigasi -- 7 halaman, icon lucide-react (bukan emoji).
import { Database, MapPinned, PanelLeftClose, PanelLeftOpen } from "lucide-react";

import { NAV_ITEMS } from "../lib/constants";
import { useApp } from "../context/AppContext";

export default function Sidebar() {
  const { activeView, setActiveView, sidebarCollapsed, setSidebarCollapsed, data } = useApp();

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
              onClick={() => setActiveView(item.id)}
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
          <span>{data?.source}</span>
        </span>
        <button
          className="sidebar-toggle"
          type="button"
          onClick={() => setSidebarCollapsed((value) => !value)}
        >
          {sidebarCollapsed ? <PanelLeftOpen size={18} /> : <PanelLeftClose size={18} />}
          <span>{sidebarCollapsed ? "Buka" : "Collapse"}</span>
        </button>
      </div>
    </aside>
  );
}
