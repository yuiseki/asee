const focusAreas = [
  {
    title: 'Python backend',
    body: 'Camera capture, biometric status, and HTTP contracts stay in Python.',
  },
  {
    title: 'Electron viewer',
    body: 'The desktop surface will render camera grids and HUD layers with Chromium.',
  },
  {
    title: 'Adapter boundary',
    body: 'A narrow bridge will connect Python runtime outputs to this renderer.',
  },
];

export function App() {
  return (
    <main className="shell">
      <section className="hero">
        <p className="eyebrow">Agentic Seeing</p>
        <h1>ASEE Viewer</h1>
        <p className="lede">
          Electron surface scaffold for the GOD MODE migration. The Python backend
          remains the source of truth while the renderer contract is stabilized here.
        </p>
      </section>

      <section className="grid" aria-label="Migration focus areas">
        {focusAreas.map((area) => (
          <article className="card" key={area.title}>
            <h2>{area.title}</h2>
            <p>{area.body}</p>
          </article>
        ))}
      </section>
    </main>
  );
}
