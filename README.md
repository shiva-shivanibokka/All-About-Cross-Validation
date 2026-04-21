# All About Cross-Validation in Machine Learning

A comprehensive, single-notebook reference covering every major cross-validation technique used by ML engineers in industry — from basic train/test splits to Bayesian hyperparameter optimization with two different libraries.

Built as a teaching resource for students and a portfolio reference for practitioners. Every section is written in plain language with diagrams, annotated code, and chart interpretation guides so that a complete beginner can follow along while still being technically rigorous enough for an experienced engineer.

---

## What's Inside

The notebook is organized in two tiers:

### Tier 1 — Validation Fundamentals

These techniques answer the question: *"How do I honestly measure how good my model is?"*

| Technique | Key Idea | When to Use |
|-----------|----------|-------------|
| **Train/Test Split** | Split data once — train on one part, evaluate on the other | Quick baseline, large datasets, starting point for any project |
| **K-Fold CV** | Split data into K folds, rotate the test fold K times, average all K scores | General-purpose evaluation when data is shuffleable |
| **Stratified K-Fold** | Same as K-Fold but each fold preserves the original class proportions | Classification problems — especially with imbalanced classes |
| **Time Series Split** | Training always precedes testing in time — no shuffling allowed | Any data with a temporal order: stock prices, sensor readings, forecasting |

### Tier 2 — Hyperparameter Tuning

These techniques answer the question: *"How do I find the best configuration for my model?"* Each one uses cross-validation internally to evaluate every candidate set of hyperparameters.

| Technique | Surrogate Model | Key Idea | When to Use |
|-----------|----------------|----------|-------------|
| **Grid Search CV** | None — exhaustive | Try every combination in a manually defined grid | Small param spaces (≤ 3 params), when exhaustive coverage is needed |
| **Random Search CV** | None — random | Randomly sample combinations from distributions | Medium-large param spaces, limited compute budget |
| **Bayesian Opt. — scikit-optimize** | Gaussian Process (GP) | Build a GP model of past results and exploit/explore intelligently | Familiar sklearn-style API, moderate search spaces |
| **Bayesian Opt. — Optuna** | Tree-structured Parzen Estimator (TPE) | Model "good" and "bad" trial distributions separately and propose the next trial intelligently | Large search spaces, expensive models, production tuning |

---

## Dataset

**Diamonds** — loaded directly from Seaborn with `sns.load_dataset('diamonds')`. No file downloads or external data sources required.

| Property | Value |
|----------|-------|
| Rows | 53,940 |
| Features | `carat`, `cut`, `color`, `clarity`, `depth`, `table`, `x`, `y`, `z` |
| Target | `price_category` — binned into **Low / Mid / High / Premium** |

### Why the Diamonds Dataset?

- **Large enough** — 54k rows means K-Fold folds have ~10k samples each, which gives reliable fold-level scores
- **Well-known** — employers and students recognise it immediately; no time wasted explaining domain context
- **Imbalanced enough** — the four price categories are not perfectly balanced (~13.5k / 17k / 16k / 7.6k), which makes Stratified K-Fold demonstrably better than plain K-Fold
- **Clean** — no missing values, minimal preprocessing needed, so the focus stays on cross-validation rather than data wrangling

### How the Target is Created

The continuous `price` column is binned into 4 categories using `pd.cut()`:

| Category | Price Range | Sample Count |
|----------|------------|--------------|
| `Low` | < $950 | ~13,490 |
| `Mid` | $950 – $3,000 | ~16,846 |
| `High` | $3,000 – $8,000 | ~15,999 |
| `Premium` | > $8,000 | ~7,605 |

---

## Model

**Random Forest Classifier** — used consistently across all 8 techniques so that results are directly comparable and the effect of each CV method is isolated from model differences.

Random Forest was chosen because:
- It has **multiple meaningful hyperparameters** to tune, making it ideal for demonstrating Grid Search vs Random Search vs Bayesian Optimization
- It is **fast enough** on 54k rows (and a 30% subsample for tuning) to complete in reasonable time on a standard laptop
- It is **widely used in industry** — students will immediately recognize it and employers will see a realistic workflow

### Hyperparameters Tuned in Tier 2

All four methods tune the same Random Forest, but with different search strategies and parameter spaces:

**Grid Search CV** — exhaustive grid over 4 parameters (54 combinations):

| Parameter | What It Controls | Grid Values |
|-----------|-----------------|-------------|
| `n_estimators` | Number of trees in the forest | 50, 100, 200 |
| `max_depth` | Maximum depth each tree can grow | 10, 20, None |
| `min_samples_split` | Minimum samples required to split a node | 2, 5, 10 |
| `max_features` | Number of features considered at each split | `sqrt`, `log2` |

**Random Search CV, scikit-optimize, and Optuna** — broader search over 5 parameters:

| Parameter | What It Controls | Search Range |
|-----------|-----------------|--------------|
| `n_estimators` | Number of trees in the forest | 50 – 300 |
| `max_depth` | Maximum depth each tree can grow | 5 – 30 |
| `min_samples_split` | Minimum samples required to split a node | 2 – 20 |
| `min_samples_leaf` | Minimum samples required at a leaf node | 1 – 10 |
| `max_features` | Number of features considered at each split | `sqrt`, `log2` |

---

## Setup

### Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scikit-optimize optuna
```

> `scipy` is used for `randint` distributions in Random Search and is a standard dependency of most scientific Python environments (it ships with Anaconda and is a transitive dependency of scikit-learn). It does **not** need to be installed separately in most setups.

The notebook also includes a **Cell 0** that installs `scikit-optimize` and `optuna` directly from inside the notebook using `pip`, so you can run it without pre-installing those two packages manually.

### Run the Notebook

```bash
jupyter notebook cross_validation.ipynb
```

Or open it directly in JupyterLab, VS Code, or Google Colab.

### Runtime Note

> The hyperparameter tuning sections (Grid Search, Random Search, Bayesian) run on a **30% stratified sample** of the full dataset to keep runtimes manageable on a standard machine (~1–2 minutes per tuning section). All four tuning methods use the exact same sample, so comparisons are fair. In production you would run on the full dataset.

---

## Notebook Structure

```
cross_validation.ipynb
│
├── Intro   — What is cross-validation, technique overview table, dataset & model summary
├── Cell 0  — Install Required Libraries (scikit-optimize, optuna)
├── Cell 1  — Imports and Setup
├── Cell 2  — Load and Prepare Dataset
│             (pd.cut explanation, LabelEncoder vs One-Hot Encoding comparison)
│
├── TIER 1: Validation Fundamentals
│   ├── Section 1 — Train/Test Split
│   ├── Section 2 — K-Fold Cross-Validation
│   ├── Section 3 — Stratified K-Fold Cross-Validation
│   └── Section 4 — Time Series Split
│                    (includes manual loop walkthrough and code annotation)
│
├── TIER 2: Hyperparameter Tuning
│   ├── Section 5 — Grid Search CV
│   ├── Section 6 — Random Search CV
│   ├── Section 7  — Bayesian Optimization (overview + library comparison table)
│   ├── Section 7a — Bayesian Optimization (scikit-optimize / BayesSearchCV / GP)
│   └── Section 7b — Bayesian Optimization (Optuna / TPE)
│                     (includes full TPE step-by-step explanation with worked example)
│
└── Final Summary — Comparison table + charts for all 8 techniques + Golden Rules
```

---

## Key Concepts Covered

### Cross-Validation Fundamentals
- Why evaluating on training data produces falsely optimistic scores (overfitting)
- Why a single train/test split can mislead you and how K-Fold fixes it
- How the mean and standard deviation across folds together tell you more than accuracy alone
- Why class proportions matter and how stratification prevents silent evaluation bias
- The critical rule of temporal ordering in time series data and what "data leakage" means in practice

### Hyperparameter Tuning
- The difference between model parameters (learned from data) and hyperparameters (set by you)
- Why Grid Search scales exponentially and when it is still the right choice
- How Random Search explores the hyperparameter space more broadly with the same budget (Bergstra & Bengio, 2012)
- How Bayesian optimization builds a surrogate model to guide the search — balancing exploitation and exploration
- The difference between Gaussian Process (skopt) and TPE (Optuna) as surrogate models, including when each one is preferable
- How the Optuna `objective` function pattern works and why it is more flexible than `BayesSearchCV`
- How TPE (Tree-structured Parzen Estimator) works step by step with a concrete worked example

### Python & sklearn Specifics
- What `pd.cut()` does, how bin boundaries work, and why `max() + 1` is needed
- What `LabelEncoder` does, with a side-by-side comparison table vs One-Hot Encoding — and why the choice depends on your model type
- How to manually loop through `TimeSeriesSplit` without breaking the time ordering (and why we loop manually instead of using `cross_val_score` — to capture per-fold train sizes, test sizes, and scores separately)
- How to read sklearn's `cv_results_` dictionary and extract useful information from it
- How to use a `callback` with `BayesSearchCV` to get clean per-trial progress output

### Chart Interpretation
- How to read a convergence curve and what a "good" convergence looks like
- How to interpret error bars on a bar chart and why a small standard deviation matters
- How to read a stacked proportion chart to confirm stratification is working
- How to compare multiple methods on the same axes and draw conclusions about efficiency vs accuracy

---

## Quick Reference

### Choosing a Validation Technique

```
Starting a new project?                        → Train/Test Split first (fast sanity check)
General classification or regression?          → Stratified K-Fold (default choice)
Balanced classes, tabular data?                → K-Fold CV
Time-ordered or sequential data?               → Time Series Split (never shuffle)
Imbalanced classes?                            → Stratified K-Fold (preserves proportions)
```

### Choosing a Hyperparameter Tuning Method

```
Very few hyperparameters (1–3)?                → Grid Search CV (exhaustive, guaranteed)
Many hyperparameters, limited compute?         → Random Search CV (efficient, easy)
Want a familiar sklearn-style API?             → scikit-optimize BayesSearchCV (GP-based)
Large search space or expensive model?         → Optuna (TPE-based, industry standard)
Kaggle competition or production pipeline?     → Optuna (pruning, parallelism, callbacks)
```

---

## License

MIT — free to use, adapt, and share with attribution.
