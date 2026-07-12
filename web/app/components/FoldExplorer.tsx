"use client";

import { useEffect, useState } from "react";
import { FoldsData, Method, getJSON } from "../lib/data";
import { HowToRead } from "./HowToRead";

const GROUP_COLORS = [
  "#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860",
  "#da8bc3", "#8c8c8c", "#ccb974", "#64b5cd", "#4c72b0", "#dd8452",
];

function stateClass(s: number) {
  return s === 1 ? "cell train" : s === 2 ? "cell test" : s === 3 ? "cell purge" : "cell";
}

export default function FoldExplorer() {
  const [data, setData] = useState<FoldsData | null>(null);
  const [active, setActive] = useState("KFold");

  useEffect(() => {
    getJSON<FoldsData>("/folds.json").then(setData).catch(() => setData(null));
  }, []);

  if (!data) return <p className="note">Loading fold layouts…</p>;

  const order = ["KFold", "StratifiedKFold", "GroupKFold", "TimeSeriesSplit", "Purged"];
  const method: Method = data.methods[active];
  const n = data.n;
  const cols = { gridTemplateColumns: `repeat(${n}, 1fr)` };

  return (
    <div className="fold-explorer">
      <div className="seg">
        {order.map((m) => (
          <button
            key={m}
            aria-pressed={active === m}
            onClick={() => setActive(m)}
          >
            {m}
          </button>
        ))}
      </div>

      <p className="blurb">{method.blurb}</p>

      <div className="grid-rows">
        {/* Context row: class labels (stratified) or group ids (group) */}
        {method.labels && (
          <div className="fold-row">
            <div className="fold-label">class</div>
            <div className="cells" style={cols}>
              {method.labels.map((c, i) => (
                <div
                  key={i}
                  className="cell g"
                  title={c === 1 ? "positive" : "negative"}
                  style={{ background: c === 1 ? "var(--test)" : "var(--train)", opacity: 0.55 }}
                />
              ))}
            </div>
          </div>
        )}
        {method.groups && (
          <div className="fold-row">
            <div className="fold-label">group</div>
            <div className="cells" style={cols}>
              {method.groups.map((g, i) => (
                <div
                  key={i}
                  className="cell g"
                  title={`group ${g}`}
                  style={{ background: GROUP_COLORS[g % GROUP_COLORS.length], opacity: 0.55 }}
                />
              ))}
            </div>
          </div>
        )}

        {/* One row per fold */}
        {method.folds.map((fold, fi) => (
          <div className="fold-row" key={fi}>
            <div className="fold-label">fold {fi + 1}</div>
            <div className="cells" style={cols}>
              {fold.map((s, i) => (
                <div key={i} className={stateClass(s)} />
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="legend">
        <span><i className="swatch" style={{ background: "var(--train)" }} /> train</span>
        <span><i className="swatch" style={{ background: "var(--test)" }} /> test</span>
        {active === "Purged" && (
          <span><i className="swatch" style={{ background: "var(--purge)" }} /> purged + embargo</span>
        )}
        {(active === "TimeSeriesSplit" || active === "Purged") && (
          <span><i className="swatch" style={{ background: "var(--unused)" }} /> not yet available (future)</span>
        )}
      </div>

      <p className="note">
        These are real scikit-learn splits on a {n}-sample demo strip (each column is one sample,
        left→right is row order). Switch tabs to watch how the same data gets partitioned five
        different ways.
      </p>

      <HowToRead
        points={[
          <>Each <b>column</b> is one sample; each <b>row</b> is one fold — the model trains and tests once per row.</>,
          <><i className="swatch" style={{ background: "var(--train)" }} /> <b>train</b> rows fit the model; <i className="swatch" style={{ background: "var(--test)" }} /> <b>test</b> rows are held out and scored. Every sample is tested exactly once, across the folds.</>,
          <>The top context strip shows what the splitter respects: the <b>class</b> row (Stratified keeps its ~30% balance in every fold) or the colored <b>group</b> row (GroupKFold keeps each color entirely in one fold).</>,
          <>For <span className="kbd">TimeSeriesSplit</span> and <span className="kbd">Purged</span>, test always sits to the <em>right</em> of train (later in time), and grey cells are the future the model can&apos;t use yet.</>,
        ]}
        takeaway={<>Every splitter answers &ldquo;which rows train, which rows test?&rdquo; — the right one is the one whose fold shape matches how your data is actually structured.</>}
      />
    </div>
  );
}
