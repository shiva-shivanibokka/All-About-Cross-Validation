"use client";

import { useState } from "react";
import { CV_TABS } from "./tabs";
import { Tip } from "./lib/tip";
import FoldExplorer from "./components/FoldExplorer";
import StoryTab, { StoryId } from "./components/StoryTab";
import OofTab from "./components/OofTab";
import CurvesTab from "./components/CurvesTab";
import AboutTab from "./components/AboutTab";

const REPO = "https://github.com/shiva-shivanibokka/All-About-Cross-Validation";
const STORY_IDS = ["leakage", "resampling", "groups", "time", "nested"];

export default function Home() {
  const [active, setActive] = useState(CV_TABS[0].id);
  const tab = CV_TABS.find((t) => t.id === active)!;

  // Arrow / Home / End move between tabs and focus follows, per the ARIA tabs pattern.
  function onTabKey(e: React.KeyboardEvent<HTMLElement>) {
    const keys: Record<string, number> = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 };
    const i = CV_TABS.findIndex((t) => t.id === active);
    let next = -1;
    if (e.key in keys) next = (i + keys[e.key] + CV_TABS.length) % CV_TABS.length;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = CV_TABS.length - 1;
    if (next < 0) return;
    e.preventDefault();
    setActive(CV_TABS[next].id);
    document.getElementById(`tab-${CV_TABS[next].id}`)?.focus();
  }

  return (
    <main className="wrap">
      <header className="hero">
        <h1>All About Cross-Validation</h1>
        <p>
          Cross-validation isn&apos;t a model — it&apos;s the discipline for measuring one <strong>honestly</strong>.
          This visualizer draws the real scikit-learn fold layouts behind five splitters and the leakage traps that
          quietly manufacture fake scores. Every number here was computed by scikit-learn in the companion
          notebooks and exported as JSON — the page itself is <strong>static, with no backend</strong>.
        </p>
        <span className="live">
          <b>●</b> real scikit-learn splits · notebook-computed · nothing leaves your machine
        </span>
      </header>

      <nav className="tabs" role="tablist" aria-label="Cross-validation topics" onKeyDown={onTabKey}>
        {CV_TABS.map((t) => (
          <button
            key={t.id}
            id={`tab-${t.id}`}
            className="tab"
            role="tab"
            aria-selected={t.id === active}
            aria-controls={`panel-${t.id}`}
            // roving tabindex: one Tab stop for the whole strip, arrows move within it
            tabIndex={t.id === active ? 0 : -1}
            onClick={() => setActive(t.id)}
          >
            {t.title}
          </button>
        ))}
      </nav>

      <section className="panel" role="tabpanel" id={`panel-${tab.id}`} aria-labelledby={`tab-${tab.id}`}>
        <div className="panel-head">
          <div className="htitle">
            <h2>{tab.title}</h2>
            <Tip text={tab.help} />
          </div>
          <span className="chip">{tab.badge}</span>
        </div>
        <p className="panel-tagline">{tab.tagline}</p>

        {tab.id === "explorer" && <FoldExplorer />}
        {STORY_IDS.includes(tab.id) && <StoryTab id={tab.id as StoryId} />}
        {tab.id === "oof" && <OofTab />}
        {tab.id === "curves" && <CurvesTab />}
        {tab.id === "about" && <AboutTab />}
      </section>

      <p className="footer">
        Built by Shivani Bokka · scikit-learn · static site on Vercel ·{" "}
        <a href={REPO} target="_blank" rel="noreferrer">source</a>
      </p>
    </main>
  );
}
