const REPO = "https://github.com/shiva-shivanibokka/All-About-Cross-Validation";

export default function AboutTab() {
  return (
    <div className="about">
      <p>
        Cross-validation is not a model — it&apos;s the discipline for measuring one <span className="k">honestly</span>.
        This visualizer is the companion to four notebooks that go from &ldquo;why training accuracy lies&rdquo; all the way
        to nested cross-validation, each written in plain English on a real dataset. Every number shown here comes straight
        from a notebook that executes end-to-end with zero errors.
      </p>

      <h3><span className="num">01</span>Foundations &amp; the Leakage Trap</h3>
      <p>
        Training error, the single-split lottery, and the trap that dominates the whole series:{" "}
        <span className="k">leakage</span>. The <strong>Leakage</strong> tab shows the headline — 0.82 accuracy conjured
        from pure noise — and how a <span className="kbd">Pipeline</span> re-fit per fold makes it vanish.{" "}
        <em>German Credit.</em>
      </p>

      <h3><span className="num">02</span>The K-Fold Family</h3>
      <p>
        <span className="k">KFold</span>, <span className="k">StratifiedKFold</span>, <span className="k">RepeatedStratifiedKFold</span>,{" "}
        <span className="k">ShuffleSplit</span>, LOOCV / Leave-P-Out, and out-of-fold prediction. The{" "}
        <strong>Fold Explorer</strong> tab draws the first three (plus group and time splitters) on a live demo strip.{" "}
        <em>German Credit + Bike Sharing.</em>
      </p>

      <h3><span className="num">03</span>Grouped &amp; Time-Aware CV</h3>
      <p>
        When one entity recurs, split on the entity; when time flows, never shuffle it. See both failures live in the{" "}
        <strong>Group Leakage</strong> and <strong>Time Leakage</strong> tabs, plus purged &amp; embargoed CV for
        overlapping-label data. <em>Parkinsons Telemonitoring + Bike Sharing.</em>
      </p>

      <h3><span className="num">04</span>Model Selection with CV</h3>
      <p>
        Grid, Random, Successive Halving, and Bayesian search (GP + TPE), then the punchline in the{" "}
        <strong>Nested CV</strong> tab: <span className="k">best_score_</span> is a selection score, not a generalization
        estimate. <em>German Credit.</em>
      </p>

      <h3><span className="num">05</span>Resampling Leakage &amp; Diagnostic Curves</h3>
      <p>
        Oversampling to fix class imbalance is a modeling step — do it before the split and it leaks
        (<strong>Resampling Leakage</strong> tab). Then <span className="k">learning</span> and{" "}
        <span className="k">validation</span> curves turn CV into a bias/variance diagnostic in the{" "}
        <strong>Learning Curves</strong> tab. <em>German Credit.</em>
      </p>

      <h3>The golden rules underneath all of it</h3>
      <ul>
        <li><strong>Never score on training rows,</strong> and never let one split&apos;s luck decide your result.</li>
        <li><strong>Every data-dependent step goes inside a Pipeline</strong> so CV re-fits it per fold — or you leak.</li>
        <li><strong>StratifiedKFold is the classification default;</strong> repeat it to pin down the mean.</li>
        <li><strong>If an entity recurs, split on the entity;</strong> if time flows, never shuffle it.</li>
        <li><strong>Tune with CV, but report with nested CV.</strong></li>
      </ul>

      <p style={{ marginTop: "1.4rem" }}>
        The five notebooks, the dataset loaders, and this app all live on{" "}
        <a href={REPO} target="_blank" rel="noreferrer">GitHub</a>.
      </p>
    </div>
  );
}
