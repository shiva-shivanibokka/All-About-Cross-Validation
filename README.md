# All About Cross-Validation

> A from-first-principles tour of how to **honestly measure** a machine-learning model — four
> deeply-explained notebooks on real datasets, plus an interactive browser visualizer of the
> fold layouts and leakage traps.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white)](requirements.txt)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-F7931E?logo=scikitlearn&logoColor=white)](requirements.txt)
[![Next.js](https://img.shields.io/badge/Next.js-15-black?logo=nextdotjs)](web/package.json)
[![Notebooks](https://img.shields.io/badge/notebooks-4%20%C2%B7%200%20errors-brightgreen)](.)

---

## 🧭 Recruiter TL;DR

- **What it is:** four notebooks that go from "why training accuracy lies" to **nested
  cross-validation**, covering every CV family a working ML engineer needs — the K-fold family,
  group-aware CV, time-aware CV (including purged/embargoed), and model selection — each written
  in plain English with a **"How to Read This Chart"** section for every figure.
- **The thesis, proven not asserted:** the wrong validation doesn't just add noise, it *lies*.
  The notebooks demonstrate it with hard numbers on real data:
  - **0.82 accuracy manufactured from 100% random noise** via one leaky feature-selection step.
  - **R² 0.91 → −0.57** on Parkinsons voice data by switching a random split to `GroupKFold` —
    the "great" model is actually *worse than predicting the mean* on unseen patients.
  - **A fake +0.15 R²** on bike-demand forecasting, purely from shuffling time.
- **Shipped, not just written:** every notebook executes end-to-end with **0 errors**; a
  companion **Next.js visualizer** renders the real scikit-learn fold layouts and the
  leakage/nested-CV results in the browser, deployed on Vercel.

---

## The Notebooks

| # | Notebook | What it covers | Data |
|---|----------|----------------|------|
| 01 | [Foundations & the Leakage Trap](01_foundations_and_leakage.ipynb) | training-error vs holdout, the single-split lottery, **leakage** (Pipeline-in-fold, 0.82 AUC from noise), choosing K | German Credit |
| 02 | [The K-Fold Family](02_the_kfold_family.ipynb) | `KFold`, `StratifiedKFold`, `RepeatedStratifiedKFold`, `ShuffleSplit`, **LOOCV / Leave-P-Out**, `cross_val_predict` (out-of-fold), **regression CV** | Credit + Bike |
| 03 | [Grouped & Time-Aware CV](03_grouped_and_time_aware.ipynb) | `GroupKFold`, `StratifiedGroupKFold`, `LeaveOneGroupOut`, `TimeSeriesSplit` (expanding vs sliding), **purged & embargoed CV** | Parkinsons + Bike |
| 04 | [Model Selection with CV](04_model_selection.ipynb) | Grid, Random, **Successive Halving**, Bayesian (**GP** via skopt + **TPE** via Optuna), and **nested cross-validation** | German Credit |

Each notebook opens plain-English, comments every line, and explains every chart. Beginner-
followable; practitioner-correct.

---

## Datasets — all real, none toy

Chosen so each one's *structure forces a CV family to exist*. Loaders live in
[`cv_datasets.py`](cv_datasets.py); everything is fetched at runtime (OpenML / UCI) and cached
locally — nothing is committed.

| Dataset | Size | Why it's here |
|---------|------|---------------|
| **German Credit** | 1,000 × 20 | Small + imbalanced (30% "bad") → CV matters most when data is scarce. |
| **Bike Sharing** | 17,379 hourly | A true time order + regression target → time-aware CV, regression CV. |
| **Parkinsons Telemonitoring** | 5,875 × 18, **42 patients** | ~139 recordings per patient → the dramatic group-leakage story. |

Run `python cv_datasets.py` to fetch all three and print a self-check.

---

## Interactive visualizer (`web/`)

A **Next.js** app (100% client-side, no backend) that reads small JSON artifacts exported from
the notebooks by [`scripts/export_web_artifacts.py`](scripts/export_web_artifacts.py):

- **Fold Explorer** — real scikit-learn fold membership for KFold / Stratified / Group /
  TimeSeries / Purged, on a demo strip you can eyeball column by column.
- **Honest-vs-leaky panels** — the leakage, group, time, and nested-CV results as side-by-side bars.

```bash
python scripts/export_web_artifacts.py   # writes web/public/{folds,headline}.json
cd web && npm install && npm run dev      # http://localhost:3000
```

---

## Setup

```bash
pip install -r requirements.txt
jupyter notebook 01_foundations_and_leakage.ipynb
```

The notebooks run top-to-bottom in a few minutes each on a laptop (the Bayesian-search section in
notebook 04 is the slowest at ~2 minutes).

---

## Key ideas, one line each

1. **Never score on training rows**, and never let a single split's luck decide your result.
2. **Every data-dependent step goes inside a `Pipeline`** so CV re-fits it per fold — or you leak.
3. **`StratifiedKFold` is the classification default;** repeat it to pin down the mean.
4. **If an entity recurs, split on the entity;** if time flows, never shuffle it.
5. **Tune with CV, but *report* with nested CV** — `best_score_` is a selection score, not a
   generalization estimate.

> **The golden rule beneath all of it:** the data you use to judge a model must be data the model —
> and your choices about it — never touched.

---

## Repo layout

```
├── 01_foundations_and_leakage.ipynb
├── 02_the_kfold_family.ipynb
├── 03_grouped_and_time_aware.ipynb
├── 04_model_selection.ipynb
├── cv_datasets.py            # shared real-dataset loaders (+ self-check)
├── scripts/
│   └── export_web_artifacts.py
├── web/                      # Next.js CV visualizer (Vercel)
├── PLAN.md                   # living build plan / design decisions
└── requirements.txt
```

## License

MIT — see [LICENSE](LICENSE).
