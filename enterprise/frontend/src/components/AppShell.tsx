import { useState, type ReactNode } from "react";

const pages = [
  ["home", "Overview"],
  ["benchmark", "Benchmark"],
  ["runs", "Runs"],
  ["compare", "Compare"],
  ["targets", "Targets"],
  ["methodology", "Methodology"],
  ["research", "Research"],
];

export function AppShell({
  page,
  message,
  children,
}: {
  page: string;
  message: string;
  children: ReactNode;
}) {
  const [menu, setMenu] = useState(false);
  return (
    <>
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <header className="topbar">
        <a
          className="brand"
          href="#home"
          onClick={() => setMenu(false)}
          aria-label="VAIS Enterprise home"
        >
          <span className="brand-mark" aria-hidden="true">
            ▤
          </span>
          <span>
            VAIS <b>Enterprise</b>
          </span>
        </a>
        <nav
          id="primary-navigation"
          className={menu ? "nav-open" : ""}
          aria-label="Primary navigation"
        >
          {pages.map(([id, label]) => (
            <a
              className={page === id ? "active" : ""}
              aria-current={page === id ? "page" : undefined}
              href={`#${id}`}
              key={id}
              onClick={() => setMenu(false)}
            >
              {label}
            </a>
          ))}
        </nav>
        <div className="top-actions">
          <a className="portfolio-link" href="https://vaislabs.com/">
            VAIS Labs ↗
          </a>
          <button
            className="nav-toggle"
            onClick={() => setMenu(!menu)}
            aria-expanded={menu}
            aria-controls="primary-navigation"
            aria-label={menu ? "Close navigation" : "Open navigation"}
          >
            {menu ? "✕" : "☰"}
          </button>
        </div>
      </header>
      <main id="main">
        <div className="status" role="status">
          <span className="dot" aria-hidden="true" />
          {message}
        </div>
        {children}
      </main>
      <footer className="site-footer">
        <div>
          <a className="brand" href="#home">
            <span className="brand-mark" aria-hidden="true">
              ▤
            </span>
            VAIS <b>Enterprise</b>
          </a>
          <p>KZ/RU enterprise AI evaluation with traceable evidence.</p>
          <small>v0.1 · Synthetic demo benchmark</small>
        </div>
        <nav aria-label="Project links">
          <a href="https://vaislabs.com/">VAIS Labs · Research portfolio ↗</a>
          <a href="https://vaislabs.com/enterprise">About VAIS Enterprise ↗</a>
          <a href="https://vaislabs.com/projects/voice">VAIS Voice ↗</a>
          <a href="https://vaislabs.com/projects/geoai">GeoAI ↗</a>
        </nav>
      </footer>
    </>
  );
}
