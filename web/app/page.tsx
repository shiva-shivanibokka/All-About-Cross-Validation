import FoldExplorer from "./components/FoldExplorer";
import Headlines from "./components/Headlines";

const REPO = "https://github.com/shiva-shivanibokka/All-About-Cross-Validation";

export default function Home() {
  return (
    <main>
      <header className="hero">
        <div className="wrap">
          <h1>
            See how <span className="grad">cross-validation</span> splits your data
          </h1>
          <p>
            Cross-validation is not a model — it&apos;s a discipline for measuring one honestly.
            This visualizer shows the real fold layouts behind five splitters, and the leakage
            traps that quietly manufacture fake scores.
          </p>
          <div className="pills">
            <a className="pill" href="#explorer">▶ Explore the folds</a>
            <a className="pill" href={REPO}>★ Notebooks &amp; code on GitHub</a>
          </div>
        </div>
      </header>

      <section id="explorer">
        <div className="wrap">
          <p className="eyebrow">Interactive</p>
          <h2>The Fold Explorer</h2>
          <p className="lede">
            Every splitter answers the same question — &ldquo;which rows train, which rows
            test?&rdquo; — differently. Plain <span className="kbd">KFold</span> ignores structure;{" "}
            <span className="kbd">StratifiedKFold</span> preserves class balance;{" "}
            <span className="kbd">GroupKFold</span> keeps each entity whole;{" "}
            <span className="kbd">TimeSeriesSplit</span> only ever trains on the past; and{" "}
            <span className="kbd">Purged</span> carves a gap to stop labels leaking across time.
          </p>
          <FoldExplorer />
        </div>
      </section>

      <section>
        <div className="wrap">
          <p className="eyebrow">Why it matters</p>
          <h2>The same model, scored honestly vs. not</h2>
          <p className="lede">
            Pick the wrong split and your metrics don&apos;t just get noisier — they lie. Each pair
            below is one model evaluated two ways; only the cross-validation strategy changed. The
            numbers come straight from the notebooks.
          </p>
          <Headlines />
        </div>
      </section>

      <footer>
        <div className="wrap">
          <p>
            Built from the <strong>All About Cross-Validation</strong> notebooks — four
            deeply-explained notebooks on real datasets (German Credit, Bike Sharing, Parkinsons
            Telemonitoring).
          </p>
          <p>
            <a href={REPO}>github.com/shiva-shivanibokka/All-About-Cross-Validation</a>
          </p>
        </div>
      </footer>
    </main>
  );
}
