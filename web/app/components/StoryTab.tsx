"use client";

import { useEffect, useState } from "react";
import { Charts, Headline, getJSON } from "../lib/data";
import ScoreBars, { Bar } from "./ScoreBars";
import { LeakageCurve, PatientErrors, PredVsActual } from "./charts";
import { HowToRead, Sw } from "./HowToRead";

export type StoryId = "leakage" | "resampling" | "groups" | "time" | "nested";

const RED = "var(--test)", GREEN = "var(--good)";

type Story = {
  bars: Bar[];
  min: number;
  max: number;
  baseline?: { value: number; label: string };
  extra?: React.ReactNode;
  callout: React.ReactNode;
  how: { points: React.ReactNode[]; takeaway: React.ReactNode };
};

function build(h: Headline, c: Charts | null, id: StoryId): Story {
  switch (id) {
    case "leakage":
      return {
        min: 0, max: 1, baseline: { value: 0.5, label: "truth = 0.50" },
        bars: [
          { name: "Leaky", sub: "select on all data", value: h.leakage_noise.leaky, color: RED, display: h.leakage_noise.leaky.toFixed(2) },
          { name: "Correct", sub: "select inside each fold", value: h.leakage_noise.correct, color: GREEN, display: h.leakage_noise.correct.toFixed(2) },
        ],
        extra: c && (
          <div className="chart-box" style={{ marginTop: "1.4rem" }}>
            <p className="chart-cap">The lie grows with opportunity — accuracy vs. number of noise features</p>
            <LeakageCurve d={c.leakage_curve} />
          </div>
        ),
        callout: (
          <>
            <strong>{h.leakage_noise.n_rows} rows × {h.leakage_noise.n_features.toLocaleString()} pure-noise features</strong> — the target
            is independent of every column. {h.leakage_noise.note} The fix: put feature selection inside a{" "}
            <span className="kbd">Pipeline</span> so cross-validation re-runs it on each fold&apos;s training rows only.
          </>
        ),
        how: {
          points: [
            <>The two bars are the same model scored two ways; only the feature-selection step moved.</>,
            <><Sw c={RED} /> <b>leaky</b> selects the &ldquo;best&rdquo; features on all rows first, so the test rows&apos; labels leak into the choice.</>,
            <><Sw c={GREEN} /> <b>honest</b> selects inside each fold; the dashed amber line is the real truth — pure noise can&apos;t beat a coin flip.</>,
            <>In the line chart, the more noise features you offer, the more spuriously-&ldquo;predictive&rdquo; ones leaky selection finds — the red line climbs while the honest green line stays pinned at chance.</>,
          ],
          takeaway: <>Leakage doesn&apos;t add a little noise — it manufactures skill that isn&apos;t there, and the more features you have, the bigger the illusion.</>,
        },
      };
    case "resampling":
      return {
        min: 0.5, max: 1,
        bars: [
          { name: "Leaky", sub: "oversample before split", value: h.resample_leak.leaky, color: RED, display: h.resample_leak.leaky.toFixed(3) },
          { name: "Correct", sub: "oversample inside fold", value: h.resample_leak.correct, color: GREEN, display: h.resample_leak.correct.toFixed(3) },
        ],
        callout: (
          <>
            German Credit, 30% &ldquo;bad&rdquo; risks. {h.resample_leak.note} A RandomForest reads{" "}
            <strong>AUC {h.resample_leak.leaky.toFixed(2)}</strong> when the minority class is balanced before the
            split, but <strong>{h.resample_leak.correct.toFixed(2)}</strong> once oversampling happens inside each
            fold — a fake <strong>+{(h.resample_leak.leaky - h.resample_leak.correct).toFixed(2)} AUC</strong>.
          </>
        ),
        how: {
          points: [
            <>Both bars are the same RandomForest and the same 5-fold CV; only <em>when</em> the oversampling happens differs.</>,
            <><Sw c={RED} /> <b>Leaky</b> balances the whole dataset first, so a minority row and its duplicate copy can land on opposite sides of the split — the model is tested on rows it memorized.</>,
            <><Sw c={GREEN} /> <b>Correct</b> oversamples only each fold&apos;s training rows and scores on the untouched, still-imbalanced test rows.</>,
            <>The axis starts at 0.5 because <b>AUC 0.5 is chance</b> for a binary classifier.</>,
          ],
          takeaway: <>Oversampling (SMOTE, random duplication) is a modeling step — it belongs <b>inside</b> the CV loop, via an <span className="kbd">imblearn.pipeline.Pipeline</span>. Balancing before the split is the same leakage sin as feature selection before the split.</>,
        },
      };
    case "groups":
      return {
        min: -0.7, max: 1, baseline: { value: 0, label: "R² = 0 · predict-the-mean" },
        bars: [
          { name: "Random KFold", sub: "patient rows leak", value: h.group_leak.random_r2, color: RED, display: h.group_leak.random_r2.toFixed(2) },
          { name: "GroupKFold", sub: "each patient whole", value: h.group_leak.group_r2, color: GREEN, display: h.group_leak.group_r2.toFixed(2) },
        ],
        extra: c && (
          <>
            <div className="chart-box" style={{ marginTop: "1.4rem" }}>
              <p className="chart-cap">Every patient&apos;s error — one dot per person, under each split</p>
              <PatientErrors d={c.patient_errors} />
            </div>
            <div className="viz-grid" style={{ marginTop: "1.1rem" }}>
              <div className="chart-box">
                <p className="chart-cap">One held-out patient: predicted vs. actual severity</p>
                <PredVsActual d={c.patient_scatter} />
              </div>
              <div className="how" style={{ marginTop: 0 }}>
                <h4>◧ What the scatter shows</h4>
                <ul>
                  <li><Sw c={RED} /> under a <b>random split</b> the model has already seen this patient in training, so its predictions hug the diagonal — memorized, not learned.</li>
                  <li><Sw c={GREEN} /> under <b>GroupKFold</b> the patient is entirely unseen, and the predictions scatter far off the &ldquo;perfect&rdquo; y = x line.</li>
                  <li>Same model, same patient — the only difference is whether the split let it cheat.</li>
                </ul>
              </div>
            </div>
          </>
        ),
        callout: (
          <>
            Parkinsons voice data — <strong>42 patients, ~139 recordings each</strong>. {h.group_leak.note} A negative R² means the
            &ldquo;great&rdquo; model does <strong>worse than predicting the mean</strong> on a patient it has never heard.
          </>
        ),
        how: {
          points: [
            <>The bars are R² (fraction of variance explained); the dashed line is <b>R² = 0</b>, what you&apos;d get by always guessing the average.</>,
            <>In the dot plot each dot is one of the 42 patients; the x-axis is that patient&apos;s average error in UPDRS points.</>,
            <><Sw c={RED} /> under a <b>random split</b> every patient sits near-perfect (all left of the amber baseline) — the flattering lie.</>,
            <><Sw c={GREEN} /> under <b>GroupKFold</b> a long tail of patients lands far past the baseline — the honest picture: the model is badly wrong about people it never trained on.</>,
          ],
          takeaway: <>When one entity contributes many rows, a random split lets the model recognize the entity instead of learning the pattern. Split on the entity and the real, much worse, performance appears.</>,
        },
      };
    case "time":
      return {
        min: 0, max: 1,
        bars: [
          { name: "Shuffled KFold", sub: "trains on the future", value: h.time_leak.shuffled_r2, color: RED, display: h.time_leak.shuffled_r2.toFixed(3) },
          { name: "TimeSeriesSplit", sub: "only trains on the past", value: h.time_leak.timeseries_r2, color: GREEN, display: h.time_leak.timeseries_r2.toFixed(3) },
        ],
        callout: (
          <>
            Bike-sharing hourly demand. {h.time_leak.note} The gap is a <strong>fake +{(h.time_leak.shuffled_r2 - h.time_leak.timeseries_r2).toFixed(2)} R²</strong> —
            skill you&apos;d never see in production, where the future hasn&apos;t happened yet.
          </>
        ),
        how: {
          points: [
            <>Both bars are R² on the same bike-demand model; only the splitter changed.</>,
            <><Sw c={RED} /> <b>Shuffled KFold</b> scatters hours at random, so training rows include hours <em>after</em> the ones being tested — the model peeks at the future.</>,
            <><Sw c={GREEN} /> <b>TimeSeriesSplit</b> only ever trains on earlier hours and tests on later ones, the way any real forecast must work.</>,
            <>See the <span className="kbd">Fold Explorer</span> tab&apos;s TimeSeriesSplit view for the fold shape this produces.</>,
          ],
          takeaway: <>Shuffling time-ordered data leaks the future into training. The honest score is lower because forecasting the unknown future is genuinely harder than interpolating a shuffled past.</>,
        },
      };
    case "nested":
      return {
        min: 0.74, max: 0.82,
        bars: [
          { name: "Non-nested", sub: "best_score_", value: h.nested.non_nested, color: RED, display: h.nested.non_nested.toFixed(3) },
          { name: "Nested CV", sub: "honest estimate", value: h.nested.nested, color: GREEN, display: `${h.nested.nested.toFixed(3)} ± ${h.nested.nested_std.toFixed(2)}` },
        ],
        callout: (
          <>
            {h.nested.note} The <span className="kbd">best_score_</span> you tuned to is the maximum over many tries — it&apos;s
            biased upward. Nested CV re-tunes inside each outer fold, so the number it reports never saw the choice that produced it.
          </>
        ),
        how: {
          points: [
            <>Both bars use the same tuning procedure on German Credit; the axis is zoomed to 0.74–0.82 so the gap is visible.</>,
            <><Sw c={RED} /> <b>Non-nested</b> reports <span className="kbd">best_score_</span> — the single highest CV score across all the configurations you tried. You picked the luckiest one.</>,
            <><Sw c={GREEN} /> <b>Nested CV</b> re-runs the whole tuning inside each outer fold, so the reported score is measured on data the selection never touched. The <b>± {h.nested.nested_std.toFixed(2)}</b> is its fold-to-fold spread.</>,
            <>The gap (about +{(h.nested.non_nested - h.nested.nested).toFixed(3)}) is the <b>optimism</b> — the &ldquo;winner&apos;s curse&rdquo; of reporting the best of many tries.</>,
          ],
          takeaway: <>Tune with CV, but never report <span className="kbd">best_score_</span> as your generalization estimate — it is a selection score, biased upward. Nested CV is the honest number.</>,
        },
      };
  }
}

export default function StoryTab({ id }: { id: StoryId }) {
  const [h, setH] = useState<Headline | null>(null);
  const [c, setC] = useState<Charts | null>(null);
  useEffect(() => {
    getJSON<Headline>("/headline.json").then(setH).catch(() => setH(null));
    getJSON<Charts>("/charts.json").then(setC).catch(() => setC(null));
  }, []);
  if (!h) return <p className="note">Loading results…</p>;

  const s = build(h, c, id);
  return (
    <div>
      <ScoreBars bars={s.bars} min={s.min} max={s.max} baseline={s.baseline} />
      <p className="callout note">{s.callout}</p>
      {s.extra}
      <HowToRead points={s.how.points} takeaway={s.how.takeaway} />
    </div>
  );
}
