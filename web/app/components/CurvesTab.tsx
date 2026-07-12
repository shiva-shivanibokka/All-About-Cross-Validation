"use client";

import { useEffect, useState } from "react";
import { Charts, getJSON } from "../lib/data";
import { TwoLineChart } from "./charts";
import { HowToRead, Sw } from "./HowToRead";

const TRAIN = "var(--train)", CV = "var(--purge)";

export default function CurvesTab() {
  const [c, setC] = useState<Charts | null>(null);
  useEffect(() => {
    getJSON<Charts>("/charts.json").then(setC).catch(() => setC(null));
  }, []);
  if (!c) return <p className="note">Loading curves…</p>;

  const lc = c.curves.learning;
  const vc = c.curves.validation;
  const bestIdx = vc.depths.indexOf(vc.best_depth);

  return (
    <div>
      <div className="chart-box">
        <p className="chart-cap">Learning curve — does more data help?</p>
        <TwoLineChart xs={lc.sizes} train={lc.train} cv={lc.cv} xLabel="training-set size" />
      </div>

      <HowToRead
        points={[
          <>Both lines are ROC AUC for a depth-5 tree; the x-axis is how many rows it trained on.</>,
          <><Sw c={TRAIN} /> <b>training</b> AUC starts near-perfect on tiny data (the tree memorizes 80 rows) and <em>falls</em> as more rows make memorization harder.</>,
          <><Sw c={CV} /> <b>cross-validated</b> AUC <em>rises</em> as more data improves real generalization.</>,
          <>A persistent gap that&apos;s still closing, with the CV line still climbing at full data, is the signature of a <b>high-variance model that would benefit from more data and/or regularization</b>.</>,
        ]}
        takeaway={<>A learning curve answers &ldquo;would more data help?&rdquo; A flat CV line with a small gap says no — fix the model or the features instead.</>}
      />

      <div className="chart-box" style={{ marginTop: "1.6rem" }}>
        <p className="chart-cap">Validation curve — how complex should the model be?</p>
        <TwoLineChart
          xs={vc.depths}
          train={vc.train}
          cv={vc.cv}
          xLabel="tree max_depth (complexity →)"
          markerIdx={bestIdx >= 0 ? bestIdx : undefined}
          markerLabel={`CV-best = ${vc.best_depth}`}
        />
      </div>

      <HowToRead
        points={[
          <>Read left→right as <b>model complexity increasing</b> (a stump on the left, a fully-grown tree on the right).</>,
          <><b>Left:</b> both lines low and together — too simple to fit even the training data. That&apos;s <b>underfitting (high bias)</b>.</>,
          <><b>Middle:</b> the <Sw c={CV} /> CV line peaks (here at depth <b>{vc.best_depth}</b>) — the complexity that generalizes best.</>,
          <><b>Right:</b> the <Sw c={TRAIN} /> training line marches to a perfect 1.0 while the CV line <em>falls</em> — the widening gap is <b>overfitting (high variance)</b>.</>,
        ]}
        takeaway={<>The training score always improves with complexity, so it can never tell you when to stop — only the cross-validated score reveals the sweet spot. This is the picture behind a <span className="kbd">GridSearchCV</span> <span className="kbd">best_params_</span>.</>}
      />
    </div>
  );
}
