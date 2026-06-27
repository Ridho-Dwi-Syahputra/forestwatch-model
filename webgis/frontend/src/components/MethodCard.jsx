// Kartu daftar item (dipakai halaman Metodologi).
export default function MethodCard({ items, title }) {
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
