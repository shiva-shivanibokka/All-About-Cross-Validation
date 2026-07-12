"use client";

// Small, dependency-free SVG charts. Each takes plain arrays from charts.json and draws
// with a fixed viewBox; the parent .chart-box scales it to full width.
import { Charts } from "../lib/data";

const AXIS = "var(--muted)";
const GRID = "var(--border)";
const RED = "var(--test)";
const GREEN = "var(--good)";
const BLUE = "var(--train)";
const CYAN = "var(--cyan)";

const lin = (v: number, d0: number, d1: number, r0: number, r1: number) =>
  r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);

function txt(x: number, y: number, s: string, opts: { anchor?: string; color?: string; size?: number; mono?: boolean } = {}) {
  return (
    <text
      x={x}
      y={y}
      textAnchor={opts.anchor ?? "middle"}
      style={{ fill: opts.color ?? AXIS, fontSize: opts.size ?? 11, fontFamily: opts.mono ? "var(--font-mono)" : "var(--font-body)" }}
    >
      {s}
    </text>
  );
}

/* ---- Leakage grows with opportunity (log-x line chart) ---- */
export function LeakageCurve({ d }: { d: Charts["leakage_curve"] }) {
  const W = 560, H = 330, L = 52, R = 20, T = 18, B = 52;
  const xs = d.feature_counts.map((c) => Math.log10(c));
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = 0.4, y1 = 0.9;
  const X = (c: number) => lin(Math.log10(c), x0, x1, L, W - R);
  const Y = (v: number) => lin(v, y0, y1, H - B, T);
  const path = (ys: number[]) => d.feature_counts.map((c, i) => `${i ? "L" : "M"}${X(c).toFixed(1)},${Y(ys[i]).toFixed(1)}`).join(" ");
  const fmt = (c: number) => (c >= 1000 ? `${c / 1000}k` : `${c}`);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Leaky accuracy rises as noise features increase">
      {[0.4, 0.5, 0.6, 0.7, 0.8, 0.9].map((g) => (
        <g key={g}>
          <line x1={L} x2={W - R} y1={Y(g)} y2={Y(g)} stroke={GRID} strokeWidth={1} opacity={0.5} />
          {txt(L - 8, Y(g) + 4, g.toFixed(1), { anchor: "end", mono: true })}
        </g>
      ))}
      {/* chance baseline */}
      <line x1={L} x2={W - R} y1={Y(d.truth)} y2={Y(d.truth)} stroke="var(--amber)" strokeWidth={1.5} strokeDasharray="5 4" />
      {txt(W - R, Y(d.truth) - 7, "chance = 0.50", { anchor: "end", color: "var(--amber)", mono: true })}
      {/* series */}
      <path d={path(d.correct)} fill="none" stroke={GREEN} strokeWidth={2.5} />
      <path d={path(d.leaky)} fill="none" stroke={RED} strokeWidth={2.5} />
      {d.feature_counts.map((c, i) => (
        <g key={c}>
          <circle cx={X(c)} cy={Y(d.correct[i])} r={3.5} fill={GREEN} />
          <circle cx={X(c)} cy={Y(d.leaky[i])} r={3.5} fill={RED} />
          {txt(X(c), H - B + 18, fmt(c), { mono: true })}
        </g>
      ))}
      {txt((L + W - R) / 2, H - 8, "number of pure-noise features offered", {})}
      {/* end labels */}
      {txt(X(d.feature_counts[d.feature_counts.length - 1]), Y(d.leaky[d.leaky.length - 1]) - 10, "leaky", { anchor: "end", color: RED, mono: true, size: 12 })}
      {txt(L + 4, Y(d.correct[0]) - 10, "honest", { anchor: "start", color: GREEN, mono: true, size: 12 })}
    </svg>
  );
}

/* ---- Per-patient error strip plot (two lanes) ---- */
export function PatientErrors({ d }: { d: Charts["patient_errors"] }) {
  const W = 560, H = 210, L = 96, R = 24, T = 22, B = 44;
  const max = Math.max(...d.group_mae) * 1.05;
  const X = (v: number) => lin(v, 0, max, L, W - R);
  const laneR = T + 30, laneG = H - B - 30;
  const jitter = (i: number) => ((i * 47) % 23) - 11; // deterministic vertical spread
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Per-patient error under random vs grouped split">
      {/* baseline (predict the mean) */}
      <line x1={X(d.baseline_mae)} x2={X(d.baseline_mae)} y1={T - 6} y2={H - B + 6} stroke="var(--amber)" strokeWidth={1.5} strokeDasharray="5 4" />
      {txt(X(d.baseline_mae), T - 10, `predict-the-mean = ${d.baseline_mae}`, { color: "var(--amber)", mono: true })}
      {/* lanes */}
      {txt(L - 10, laneR + 4, "Random", { anchor: "end", size: 12 })}
      {txt(L - 10, laneR + 18, "split", { anchor: "end", size: 11 })}
      {txt(L - 10, laneG + 4, "GroupKFold", { anchor: "end", size: 12 })}
      {d.random_mae.map((v, i) => (
        <circle key={`r${i}`} cx={X(v)} cy={laneR + jitter(i)} r={4} fill={BLUE} opacity={0.7} />
      ))}
      {d.group_mae.map((v, i) => (
        <circle key={`g${i}`} cx={X(v)} cy={laneG + jitter(i)} r={4} fill={v > d.baseline_mae ? RED : GREEN} opacity={0.75} />
      ))}
      {/* x axis */}
      <line x1={L} x2={W - R} y1={H - B + 14} y2={H - B + 14} stroke={GRID} />
      {[0, 5, 10, 15, 20, 25, 30].filter((t) => t <= max).map((t) => (
        <g key={t}>{txt(X(t), H - B + 30, `${t}`, { mono: true })}</g>
      ))}
      {txt((L + W - R) / 2, H - 4, "mean absolute error in UPDRS points (lower is better)", {})}
    </svg>
  );
}

/* ---- Predicted vs actual scatter for one held-out patient ---- */
export function PredVsActual({ d }: { d: Charts["patient_scatter"] }) {
  const S = 360, L = 46, R = 14, T = 14, B = 44;
  const all = [...d.actual, ...d.pred_random, ...d.pred_group];
  const lo = Math.min(...all) - 1, hi = Math.max(...all) + 1;
  const X = (v: number) => lin(v, lo, hi, L, S - R);
  const Y = (v: number) => lin(v, lo, hi, S - B, T);
  const ticks = [Math.ceil(lo / 5) * 5, Math.round((lo + hi) / 2 / 5) * 5, Math.floor(hi / 5) * 5];
  return (
    <svg viewBox={`0 0 ${S} ${S}`} role="img" aria-label="Predicted vs actual for one patient">
      {/* perfect-prediction diagonal */}
      <line x1={X(lo)} y1={Y(lo)} x2={X(hi)} y2={Y(hi)} stroke={AXIS} strokeWidth={1.5} strokeDasharray="5 4" opacity={0.7} />
      {txt(X(hi) - 4, Y(hi) + 14, "perfect = y = x", { anchor: "end", mono: true })}
      {ticks.map((t) => (
        <g key={t}>
          {txt(X(t), S - B + 18, `${t}`, { mono: true })}
          {txt(L - 8, Y(t) + 4, `${t}`, { anchor: "end", mono: true })}
        </g>
      ))}
      {d.pred_group.map((p, i) => <circle key={`g${i}`} cx={X(d.actual[i])} cy={Y(p)} r={3.2} fill={RED} opacity={0.7} />)}
      {d.pred_random.map((p, i) => <circle key={`r${i}`} cx={X(d.actual[i])} cy={Y(p)} r={3.2} fill={GREEN} opacity={0.7} />)}
      {txt(S / 2, S - 4, "actual UPDRS", {})}
      <text x={14} y={S / 2} transform={`rotate(-90 14 ${S / 2})`} textAnchor="middle" style={{ fill: AXIS, fontSize: 11 }}>predicted UPDRS</text>
    </svg>
  );
}

/* ---- Out-of-fold ROC curve ---- */
export function RocCurve({ d }: { d: Charts["oof"]["roc"] }) {
  const S = 340, L = 44, R = 16, T = 16, B = 42;
  const X = (v: number) => lin(v, 0, 1, L, S - R);
  const Y = (v: number) => lin(v, 0, 1, S - B, T);
  const path = d.fpr.map((f, i) => `${i ? "L" : "M"}${X(f).toFixed(1)},${Y(d.tpr[i]).toFixed(1)}`).join(" ");
  const area = `M${X(0)},${Y(0)} ${path} L${X(1)},${Y(0)} Z`;
  return (
    <svg viewBox={`0 0 ${S} ${S}`} role="img" aria-label={`ROC curve, AUC ${d.auc}`}>
      {[0, 0.5, 1].map((t) => (
        <g key={t}>
          <line x1={X(t)} x2={X(t)} y1={T} y2={S - B} stroke={GRID} opacity={0.4} />
          <line x1={L} x2={S - R} y1={Y(t)} y2={Y(t)} stroke={GRID} opacity={0.4} />
          {txt(X(t), S - B + 18, t.toFixed(1), { mono: true })}
          {txt(L - 8, Y(t) + 4, t.toFixed(1), { anchor: "end", mono: true })}
        </g>
      ))}
      <line x1={X(0)} y1={Y(0)} x2={X(1)} y2={Y(1)} stroke={AXIS} strokeDasharray="5 4" opacity={0.6} />
      {txt(X(0.72), Y(0.62), "chance", { color: AXIS, mono: true })}
      <path d={area} fill={CYAN} opacity={0.12} />
      <path d={path} fill="none" stroke={CYAN} strokeWidth={2.5} />
      {txt(X(0.62), Y(0.28), `AUC ${d.auc.toFixed(3)}`, { color: CYAN, mono: true, size: 16 })}
      {txt(S / 2, S - 4, "false-positive rate", {})}
      <text x={13} y={S / 2} transform={`rotate(-90 13 ${S / 2})`} textAnchor="middle" style={{ fill: AXIS, fontSize: 11 }}>true-positive rate</text>
    </svg>
  );
}

/* ---- Out-of-fold confusion matrix (CSS grid) ---- */
export function ConfusionMatrix({ d }: { d: Charts["oof"]["confusion"] }) {
  const cell = (n: number, label: string, ok: boolean) => (
    <div className={`cm-cell ${ok ? "ok" : "err"}`}>
      <div className="n">{n}</div>
      <div className="t">{label}</div>
    </div>
  );
  return (
    <div className="cm-wrap">
      <div className="cm-axis-top">predicted →</div>
      <div className="cm-grid">
        <div className="cm-col-head">good</div>
        <div className="cm-col-head">bad</div>
        {cell(d.tn, "true good", true)}
        {cell(d.fp, "false bad", false)}
        {cell(d.fn, "missed bad", false)}
        {cell(d.tp, "caught bad", true)}
      </div>
      <div className="cm-note">
        {d.n} rows, {d.pos} truly bad · caught {d.tp} of {d.pos} bad risks ({Math.round((100 * d.tp) / d.pos)}%)
      </div>
    </div>
  );
}
