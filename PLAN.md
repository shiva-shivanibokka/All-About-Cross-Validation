# Build Plan — All About Cross-Validation

Living plan for turning this from one tuning-heavy notebook into a portfolio-grade,
validation-first reference with a live web demo. Mirrors the structure of the
`Dive-Deeper-into-Supervised-Learning` repo.

## Decisions locked
- **Framing:** *validation-first*, not tuning-first. The old single notebook was really
  a hyperparameter-tuning notebook. The rebuild leads with how to honestly *measure* a
  model (leakage, fold families, group/time structure) and treats tuning as one
  application of CV — the correct one being **nested CV**.
- **Datasets:** three real OpenML datasets, each chosen because its *structure forces a
  CV family to exist*. No Seaborn/toy data. None duplicate sibling repos
  (Adult/Ames/Telco/Diamonds already used elsewhere).
- **Model policy:** keep the estimator boring and constant within a notebook so the CV
  method is the only moving part. RandomForest for classification, and a
  HistGradientBoosting/Ridge for regression where speed matters.
- **Frontend:** Next.js, 100% client-side, deployed on Vercel — same stack as the
  supervised repo. But it visualizes the *validation process* (fold layouts, leakage
  gap, nested loops, purged bands), NOT a model runner — CV isn't a model.
- **Scope:** full — repo hygiene + four notebooks + artifact export script + web app.

## Datasets — verified loading (2026-07-11)
| Key | Loader | Shape | Role |
|-----|--------|-------|------|
| German Credit | `fetch_openml('credit-g', 1)` | (1000, 20) | Through-line: leakage, K-fold family, stratified, LOOCV/repeated, model selection + nested CV. Small → CV matters. Positive (`bad`) rate 0.30. |
| Bike Sharing | `fetch_openml('Bike_Sharing_Demand', 2)` | (17379, 12) | Real hourly order + regression target `count`. Time-aware CV + regression CV. |
| Diabetes 130-US | `fetch_openml('diabetes130US', 1)` | (101766, 50) | `patient_nbr` groups (patients recur across encounters). Group-aware CV / leakage. Subsampled to ~20k encounters (whole patients) for runtime. |

All three load and pass a shape/leakage self-check via `python cv_datasets.py`.

## Notebook roadmap
| # | Notebook | Datasets | Key methods | Status |
|---|----------|----------|-------------|--------|
| 01 | Foundations & the Leakage Trap | credit-g | train-error vs holdout, **Pipeline-in-fold vs leaky preprocessing** (measured gap), choosing K (bias/variance/runtime) | TODO |
| 02 | The K-Fold Family | credit-g + bike | KFold, StratifiedKFold, RepeatedStratifiedKFold, ShuffleSplit, LOOCV, Leave-P-Out, `cross_val_predict` / OOF, **regression CV (R²/RMSE)** | TODO |
| 03 | Grouped & Time-Aware CV | diabetes-130 + bike | GroupKFold, StratifiedGroupKFold, LeaveOneGroupOut, TimeSeriesSplit (expanding vs sliding), **purged & embargoed CV** | TODO |
| 04 | Model Selection with CV | credit-g | Grid, Random, **Successive Halving**, Bayesian (skopt GP + Optuna TPE), **nested CV** | TODO (upgrades the old notebook) |

**Style rule (all notebooks):** match/exceed the supervised repo — every section opens in
plain English, heavy inline comments, and every chart gets a dedicated
"How to Read This Chart" markdown cell. Beginner-followable, practitioner-correct.

## What the rebuild fixes in the old notebook
- **Leakage:** old notebook `LabelEncode`s the whole frame before splitting → encoder
  sees test rows. Fixed by encoding inside a Pipeline per fold (nb01).
- **Optimistic bias:** old notebook tunes with CV then reports that same inner CV score
  as the model's performance. Fixed with nested CV (nb04).
- **Contrived target:** old notebook bins continuous `price` into classes to fake a
  classification task. Replaced by datasets with genuine targets (incl. real regression).
- **Missing families:** no group CV, no LOOCV/repeated, no purged/time-aware beyond the
  basic TimeSeriesSplit, no `cross_val_predict`. All added.

## Web app (phase 2)
- `scripts/export_web_artifacts.py` → `web/public/*.json`: fold index arrays + precomputed
  fold scores for each method (small, deterministic; no model shipped).
- Panels: (1) Fold explorer (K slider; KFold/Stratified/Group/TimeSeries), (2) Leakage
  gap animation, (3) Nested-CV outer/inner diagram, (4) Purged/embargoed bands.
- Next.js + client-side only; Vercel Git-connected deploy.

## Explicitly out of scope (mention as "further reading", add later if wanted)
- Bootstrap / .632+ estimator
- Combinatorial Purged CV (CPCV)
- Backend/ONNX inference (nothing to infer — no model is the product here)
