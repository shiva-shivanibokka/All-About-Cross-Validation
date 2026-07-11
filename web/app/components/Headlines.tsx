"use client";

import { useEffect, useState } from "react";
import { Headline, getJSON } from "../lib/data";
import ScoreBars from "./ScoreBars";

export default function Headlines() {
  const [h, setH] = useState<Headline | null>(null);
  useEffect(() => {
    getJSON<Headline>("/headline.json").then(setH).catch(() => setH(null));
  }, []);
  if (!h) return <div className="card">Loading results…</div>;

  return (
    <div className="panel-grid">
      {/* Leakage: pure noise */}
      <div className="card">
        <h3>Leakage invents skill from noise</h3>
        <ScoreBars
          min={0}
          max={1}
          baseline={{ value: 0.5, label: "truth = 0.50" }}
          bars={[
            { name: "Leaky\n(select on all data)", value: h.leakage_noise.leaky, color: "var(--test)", display: h.leakage_noise.leaky.toFixed(2) },
            { name: "Correct\n(select in-fold)", value: h.leakage_noise.correct, color: "var(--good)", display: h.leakage_noise.correct.toFixed(2) },
          ]}
        />
        <p className="note">{h.leakage_noise.n_rows} rows × {h.leakage_noise.n_features.toLocaleString()} pure-noise features. {h.leakage_noise.note}</p>
      </div>

      {/* Nested CV */}
      <div className="card">
        <h3>Tune with CV, report with nested CV</h3>
        <ScoreBars
          min={0.74}
          max={0.82}
          bars={[
            { name: "Non-nested\n(best_score_)", value: h.nested.non_nested, color: "var(--test)", display: h.nested.non_nested.toFixed(3) },
            { name: "Nested CV\n(honest)", value: h.nested.nested, color: "var(--good)", display: `${h.nested.nested.toFixed(3)}±${h.nested.nested_std.toFixed(2)}` },
          ]}
        />
        <p className="note">{h.nested.note}</p>
      </div>

      {/* Group leakage */}
      <div className="card">
        <h3>Groups: R² 0.91 → −0.57</h3>
        <ScoreBars
          min={-0.7}
          max={1}
          baseline={{ value: 0, label: "R² = 0" }}
          bars={[
            { name: "Random KFold\n(patient leaks)", value: h.group_leak.random_r2, color: "var(--test)", display: h.group_leak.random_r2.toFixed(2) },
            { name: "GroupKFold\n(honest)", value: h.group_leak.group_r2, color: "var(--good)", display: h.group_leak.group_r2.toFixed(2) },
          ]}
        />
        <p className="note">Parkinsons voice data. {h.group_leak.note}</p>
      </div>

      {/* Time leakage */}
      <div className="card">
        <h3>Time: a fake +0.15 R²</h3>
        <ScoreBars
          min={0}
          max={1}
          bars={[
            { name: "Shuffled KFold\n(sees future)", value: h.time_leak.shuffled_r2, color: "var(--test)", display: h.time_leak.shuffled_r2.toFixed(3) },
            { name: "TimeSeriesSplit\n(honest)", value: h.time_leak.timeseries_r2, color: "var(--good)", display: h.time_leak.timeseries_r2.toFixed(3) },
          ]}
        />
        <p className="note">Bike Sharing demand. {h.time_leak.note}</p>
      </div>
    </div>
  );
}
