// Kartu KPI reusable (icon lucide + label + nilai + catatan opsional).
export default function KpiCard({ accent, icon: Icon, label, note, value }) {
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
