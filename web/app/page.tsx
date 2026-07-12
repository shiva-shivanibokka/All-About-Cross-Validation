"use client";

import { useState } from "react";
import { CV_TABS } from "./tabs";
import { Tip } from "./lib/tip";
import FoldExplorer from "./components/FoldExplorer";
import StoryTab, { StoryId } from "./components/StoryTab";
import AboutTab from "./components/AboutTab";

const REPO = "https://github.com/shiva-shivanibokka/All-About-Cross-Validation";
const STORY_IDS = ["leakage", "groups", "time", "nested"];

export default function Home() {
  const [active, setActive] = useState(CV_TABS[0].id);
  const tab = CV_TABS.find((t) => t.id === active)!;

  return (
    <main className="wrap">
      <header className="hero">
        <h1>All About Cross-Validation</h1>
        <p>
          Cross-validation isn&apos;t a model — it&apos;s the discipline for measuring one <strong>honestly</strong>.
          This visualizer draws the real scikit-learn fold layouts behind five splitters and the leakage traps that
          quietly manufacture fake scores. Every number comes straight from the companion notebooks, running{" "}
          <strong>entirely in your browser</strong>.
        </p>
        <span className="live">
          <b>●</b> live · real scikit-learn splits · nothing leaves your machine
        </span>
      </header>

      <nav className="tabs" role="tablist" aria-label="Cross-validation topics">
        {CV_TABS.map((t) => (
          <button
            key={t.id}
            className="tab"
            role="tab"
            aria-selected={t.id === active}
            onClick={() => setActive(t.id)}
          >
            {t.title}
          </button>
        ))}
      </nav>

      <section className="panel" role="tabpanel">
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
        {tab.id === "about" && <AboutTab />}
      </section>

      <p className="footer">
        Built by Shivani Bokka · scikit-learn · served client-side on Vercel ·{" "}
        <a href={REPO} target="_blank" rel="noreferrer">source</a>
      </p>
    </main>
  );
}
