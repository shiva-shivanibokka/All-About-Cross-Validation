"use client";

import { useEffect, useState } from "react";
import { Charts, getJSON } from "../lib/data";
import { RocCurve, ConfusionMatrix } from "./charts";
import { HowToRead } from "./HowToRead";

export default function OofTab() {
  const [c, setC] = useState<Charts | null>(null);
  useEffect(() => {
    getJSON<Charts>("/charts.json").then(setC).catch(() => setC(null));
  }, []);
  if (!c) return <p className="note">Loading out-of-fold results…</p>;

  return (
    <div>
      <div className="viz-grid">
        <div className="chart-box">
          <p className="chart-cap">Out-of-fold ROC curve</p>
          <RocCurve d={c.oof.roc} />
        </div>
        <div className="chart-box">
          <p className="chart-cap">Out-of-fold confusion matrix @ 0.5</p>
          <ConfusionMatrix d={c.oof.confusion} />
        </div>
      </div>

      <p className="callout note">
        <span className="kbd">cross_val_predict</span> gives every row exactly one prediction — the one made while that row
        sat in the held-out fold. Concatenating those out-of-fold predictions lets you evaluate the whole dataset at once,
        honestly, instead of averaging a metric fold by fold.
      </p>

      <HowToRead
        points={[
          <>Both charts use one out-of-fold prediction per row on German Credit (5-fold stratified), from <span className="kbd">cross_val_predict</span>.</>,
          <>The <b>ROC curve</b> sweeps the decision threshold: each point is a (false-positive, true-positive) rate. Up-and-left is better; the dashed diagonal is random guessing.</>,
          <><b>AUC {c.oof.roc.auc.toFixed(3)}</b> is the area under it — the chance the model ranks a random bad risk above a random good one.</>,
          <>The <b>confusion matrix</b> fixes the threshold at 0.5: green cells are correct, red are the two error types. Here it catches {c.oof.confusion.tp} of {c.oof.confusion.pos} bad risks and false-alarms on {c.oof.confusion.fp}.</>,
        ]}
        takeaway={
          <>Out-of-fold prediction turns cross-validation into a single, leak-free test set — the right way to build an ROC curve, a
          confusion matrix, or any threshold-dependent view without a separate holdout.</>
        }
      />
    </div>
  );
}
