type Dataset = {
  version: string;
  case_count: number;
  languages: string[];
  published_at: string;
  dataset_hash: string;
};
type Target = { id: string; name: string; adapter_type: string };

export function HomePage({
  benchmark,
  targets,
  runDemo,
  go,
}: {
  benchmark?: Dataset;
  targets: Target[];
  runDemo: (id: string) => void;
  go: (id: string) => void;
}) {
  return (
    <>
      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">KZ/RU ENTERPRISE AI BENCHMARK</p>
          <h1>Evaluate enterprise AI before it reaches production.</h1>
          <p>
            Reproducible benchmarks for KZ/RU RAG, copilots and AI agents — with
            citation checks and versioned evidence.
          </p>
          <div className="actions">
            <button onClick={() => go("benchmark")}>Explore benchmark →</button>
            <button
              className="secondary"
              disabled={!benchmark}
              onClick={() => runDemo("mock-a")}
            >
              Run demo benchmark
            </button>
          </div>
          <div className="pilot-line">
            <span className="pill">Research MVP</span> Public synthetic dataset
            · v0.1
          </div>
        </div>
        <aside className="hero-report" aria-label="Current benchmark dataset">
          <div className="report-head">
            <div>
              <p className="eyebrow">BENCHMARK DATASET</p>
              <h2>KZ/RU Enterprise</h2>
            </div>
            <span className="pill">{benchmark?.version ?? "Unavailable"}</span>
          </div>
          <dl className="report-facts">
            <div>
              <dt>Cases</dt>
              <dd>{benchmark?.case_count ?? "Unavailable"}</dd>
            </div>
            <div>
              <dt>Languages</dt>
              <dd>
                {benchmark?.languages
                  .map((l) => (l === "kk" ? "KZ" : l.toUpperCase()))
                  .join(" / ") ?? "Unavailable"}
              </dd>
            </div>
            <div>
              <dt>Evidence</dt>
              <dd>Self-written policies</dd>
            </div>
            <div>
              <dt>Protocol</dt>
              <dd>Deterministic checks</dd>
            </div>
          </dl>
          <p className="muted">
            Inspect citation checks, version conflicts and insufficient-evidence
            cases. Demo targets illustrate the protocol.
          </p>
          <button className="text-button" onClick={() => go("methodology")}>
            Read the methodology →
          </button>
        </aside>
      </section>
      <section className="workflow">
        <div className="section-heading">
          <div>
            <p className="eyebrow">EVALUATION WORKFLOW</p>
            <h2>Evidence you can trace.</h2>
          </div>
          <span className="pill">Offline demo</span>
        </div>
        <ol>
          {[
            "Select a target",
            "Run the benchmark",
            "Inspect case evidence",
            "Compare snapshots",
          ].map((item, i) => (
            <li key={item}>
              <b>0{i + 1}</b>
              <span>{item}</span>
            </li>
          ))}
        </ol>
      </section>
      <section className="section">
        <div className="section-heading">
          <div>
            <p className="eyebrow">WHAT IS EVALUATED</p>
            <h2>From answers to their sources.</h2>
          </div>
        </div>
        <div className="grid three">
          <article className="card">
            <span className="card-number">01</span>
            <h2>Answer checks</h2>
            <p>
              Normalised answer matching and selected unsupported-claim checks
              on the published demo cases.
            </p>
          </article>
          <article className="card">
            <span className="card-number">02</span>
            <h2>Citations & versions</h2>
            <p>
              Expected source coverage and document versions, with per-case
              citations available for inspection.
            </p>
          </article>
          <article className="card">
            <span className="card-number">03</span>
            <h2>KZ/RU reasoning</h2>
            <p>
              Cross-language questions over Kazakh and Russian policy documents.
              Scores remain tied to this dataset.
            </p>
          </article>
        </div>
      </section>
      <section className="callout">
        <div>
          <p className="eyebrow">EXPLORE THE PROTOCOL</p>
          <h2>Two targets. Inspectable differences.</h2>
          <p>
            Mock A follows the expected evidence. Mock B includes intentional
            citation and refusal failures. Run both and inspect how their
            results differ.
          </p>
        </div>
        <div className="target-buttons">
          {targets
            .filter((t) => t.adapter_type === "mock")
            .map((t) => (
              <button
                key={t.id}
                className="secondary"
                onClick={() => runDemo(t.id)}
              >
                Run {t.name} →
              </button>
            ))}
        </div>
      </section>
    </>
  );
}
