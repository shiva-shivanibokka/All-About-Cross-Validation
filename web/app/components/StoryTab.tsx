"use client";

import { useEffect, useState } from "react";
import { Headline, getJSON } from "../lib/data";
import ScoreBars, { Bar } from "./ScoreBars";

export type StoryId = "leakage" | "groups" | "time" | "nested";

type Story = {
  bars: Bar[];
  min: number;
  max: number;
  baseline?: { value: number; label: string };
  callout: React.ReactNode;
};

function build(h: Headline, id: StoryId): Story {
  switch (id) {
    case "leakage":
      return {
        min: 0, max: 1, baseline: { value: 0.5, label: "truth = 0.50" },
        bars: [
          { name: "Leaky", sub: "select on all data", value: h.leakage_noise.leaky, color: "var(--test)", display: h.leakage_noise.leaky.toFixed(2) },
          { name: "Correct", sub: "select inside each fold", value: h.leakage_noise.correct, color: "var(--good)", display: h.leakage_noise.correct.toFixed(2) },
        ],
        callout: (
          <>
            <strong>{h.leakage_noise.n_rows} rows × {h.leakage_noise.n_features.toLocaleString()} pure-noise features</strong> — the target
            is independent of every column. {h.leakage_noise.note} The fix: put feature selection inside a{" "}
            <span className="kbd">Pipeline</span> so cross-validation re-runs it on each fold&apos;s training rows only.
          </>
        ),
      };
    case "groups":
      return {
        min: -0.7, max: 1, baseline: { value: 0, label: "R² = 0 · predict-the-mean" },
        bars: [
          { name: "Random KFold", sub: "patient rows leak", value: h.group_leak.random_r2, color: "var(--test)", display: h.group_leak.random_r2.toFixed(2) },
          { name: "GroupKFold", sub: "each patient whole", value: h.group_leak.group_r2, color: "var(--good)", display: h.group_leak.group_r2.toFixed(2) },
        ],
        callout: (
          <>
            Parkinsons voice data — <strong>42 patients, ~139 recordings each</strong>. {h.group_leak.note} A negative R² means the
            &ldquo;great&rdquo; model does <strong>worse than predicting the mean</strong> on a patient it has never heard.
          </>
        ),
      };
    case "time":
      return {
        min: 0, max: 1,
        bars: [
          { name: "Shuffled KFold", sub: "trains on the future", value: h.time_leak.shuffled_r2, color: "var(--test)", display: h.time_leak.shuffled_r2.toFixed(3) },
          { name: "TimeSeriesSplit", sub: "only trains on the past", value: h.time_leak.timeseries_r2, color: "var(--good)", display: h.time_leak.timeseries_r2.toFixed(3) },
        ],
        callout: (
          <>
            Bike-sharing hourly demand. {h.time_leak.note} The gap is a <strong>fake +{(h.time_leak.shuffled_r2 - h.time_leak.timeseries_r2).toFixed(2)} R²</strong> —
            skill you&apos;d never see in production, where the future hasn&apos;t happened yet.
          </>
        ),
      };
    case "nested":
      return {
        min: 0.74, max: 0.82,
        bars: [
          { name: "Non-nested", sub: "best_score_", value: h.nested.non_nested, color: "var(--test)", display: h.nested.non_nested.toFixed(3) },
          { name: "Nested CV", sub: "honest estimate", value: h.nested.nested, color: "var(--good)", display: `${h.nested.nested.toFixed(3)} ± ${h.nested.nested_std.toFixed(2)}` },
        ],
        callout: (
          <>
            {h.nested.note} The <span className="kbd">best_score_</span> you tuned to is the maximum over many tries — it&apos;s
            biased upward. Nested CV re-tunes inside each outer fold, so the number it reports never saw the choice that produced it.
          </>
        ),
      };
  }
}

export default function StoryTab({ id }: { id: StoryId }) {
  const [h, setH] = useState<Headline | null>(null);
  useEffect(() => {
    getJSON<Headline>("/headline.json").then(setH).catch(() => setH(null));
  }, []);
  if (!h) return <p className="note">Loading results…</p>;

  const s = build(h, id);
  return (
    <div>
      <ScoreBars bars={s.bars} min={s.min} max={s.max} baseline={s.baseline} />
      <p className="callout note">{s.callout}</p>
    </div>
  );
}
