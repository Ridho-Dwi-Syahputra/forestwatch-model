// Header atas: judul halaman aktif + sumber data + tombol menu (mobile).
import { Database, Menu } from "lucide-react";

import { PAGE_SUBTITLES, PAGE_TITLES } from "../lib/constants";
import { useApp } from "../context/AppContext";

export default function TopHeader() {
  const { activeView, data, setSidebarCollapsed } = useApp();

  return (
    <header className="topbar">
      <button
        className="mobile-menu-button"
        type="button"
        onClick={() => setSidebarCollapsed((value) => !value)}
      >
        <Menu size={18} />
      </button>
      <div>
        <p className="eyebrow">Kasuari AI</p>
        <h1>{PAGE_TITLES[activeView]}</h1>
        <p>{PAGE_SUBTITLES[activeView]}</p>
      </div>
      <span className="data-pill topbar-source">
        <Database size={15} />
        {data?.source}
      </span>
    </header>
  );
}
