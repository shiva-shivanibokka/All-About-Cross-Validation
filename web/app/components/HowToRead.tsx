import React from "react";

/** A colored legend swatch for inline use inside explanation bullets. */
export function Sw({ c }: { c: string }) {
  return <span className="swatch" style={{ background: c }} />;
}

/** The "How to read this" explainer that sits under every chart. */
export function HowToRead({ points, takeaway }: { points: React.ReactNode[]; takeaway: React.ReactNode }) {
  return (
    <div className="how">
      <h4>◧ How to read this</h4>
      <ul>
        {points.map((p, i) => (
          <li key={i}>{p}</li>
        ))}
      </ul>
      <p className="how-take">
        <b>Takeaway:</b> {takeaway}
      </p>
    </div>
  );
}
