"use client";

export type Bar = { name: string; value: number; color: string; display: string; err?: number };

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
      {bars.map((b) => (
        <div className="bar-col" key={b.name}>
          <div className="bar-val" style={{ color: b.color }}>{b.display}</div>
          <div className="bar-track">
            {baseline && (
              <div className="baseline" style={{ bottom: `${frac(baseline.value) * 100}%` }}>
                <span>{baseline.label}</span>
              </div>
            )}
            <div
              className="bar-fill"
              style={{ height: `${frac(b.value) * 100}%`, background: b.color }}
            />
          </div>
          <div className="bar-name">{b.name}</div>
        </div>
      ))}
    </div>
  );
}
