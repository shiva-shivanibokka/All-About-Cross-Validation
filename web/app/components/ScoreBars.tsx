"use client";

export type Bar = { name: string; sub?: string; value: number; color: string; display: string };

export default function ScoreBars({
  bars,
  min,
  max,
  baseline,
}: {
  bars: Bar[];
  min: number;
  max: number;
  baseline?: { value: number; label: string };
}) {
  const frac = (v: number) => Math.max(0, Math.min(1, (v - min) / (max - min)));
  return (
    <div className="bars">
      {bars.map((b, i) => (
        <div className="bar-col" key={b.name}>
          <div className="bar-val" style={{ color: b.color }}>{b.display}</div>
          <div className="bar-track">
            {baseline && (
              <div className="baseline" style={{ bottom: `${frac(baseline.value) * 100}%` }}>
                {/* label once, on the rightmost bar, so it never collides mid-chart */}
                {i === bars.length - 1 && <span>{baseline.label}</span>}
              </div>
            )}
            <div className="bar-fill" style={{ height: `${frac(b.value) * 100}%`, background: b.color }} />
          </div>
          <div className="bar-name">
            <b>{b.name}</b>
            {b.sub && <span>{b.sub}</span>}
          </div>
        </div>
      ))}
    </div>
  );
}
