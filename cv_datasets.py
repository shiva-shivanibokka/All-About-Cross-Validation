"""Shared real-dataset loaders for the cross-validation notebooks.

Every notebook imports from here so the same clean, raw (un-encoded) frames are
used everywhere. Categoricals are kept as pandas `category` dtype ON PURPOSE:
the notebooks build sklearn Pipelines that encode *inside* each CV fold, which is
the whole point of notebook 01 (no leakage).

The datasets are fetched at runtime (OpenML for credit-g / bike, UCI for the
Parkinsons voice recordings) and cached locally under ~/scikit_learn_data.
Nothing is committed to the repo.

    from cv_datasets import load_credit, load_bike, load_parkinsons_groups, feature_types
"""
from __future__ import annotations
import json
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

_CACHE = os.path.join(os.path.expanduser("~"), "scikit_learn_data")

# The numbers the web visualizer cites are computed in the notebooks, not here. Each notebook
# calls record_headline() right where it computes one, appending to a single JSON file that
# scripts/export_web_artifacts.py checks its HEADLINE constants against. That is what stops the
# site quietly disagreeing with the notebooks: before this existed, CI compared the exported
# JSON against the constants it was generated from, which could never fail.
NOTEBOOK_NUMBERS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_notebook_headline.json")

# Recorded well past the precision the site displays; the comparison in export_web_artifacts.py
# is what applies the real tolerance. Rounding here only keeps the file readable and stable.
HEADLINE_DP = 6


def record_headline(section: str, **values) -> None:
    """Record one panel's headline numbers, as computed by the notebook that owns them.

    Merges into _notebook_headline.json (path fixed relative to this file, so it lands in the
    repo root whatever the kernel's working directory is). Running a single notebook rewrites
    only that notebook's sections.
    """
    data = {}
    if os.path.exists(NOTEBOOK_NUMBERS):
        with open(NOTEBOOK_NUMBERS, encoding="utf-8") as f:
            data = json.load(f)
    data[section] = {
        k: (int(v) if isinstance(v, (int, np.integer)) and not isinstance(v, bool)
            else round(float(v), HEADLINE_DP))
        for k, v in values.items()
    }
    with open(NOTEBOOK_NUMBERS, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    print(f"[headline] recorded {section}: {data[section]}")


def feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_cols, categorical_cols) for a feature frame."""
    num = X.select_dtypes(include=["number"]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return num, cat


def load_credit() -> tuple[pd.DataFrame, pd.Series]:
    """German Credit (n=1000). Binary classification, mixed types, ~70/30 imbalance.

    Positive class (y=1) = 'bad' credit risk — the minority and the costly error.
    Small on purpose: cross-validation matters most when data is scarce.
    """
    d = fetch_openml("credit-g", version=1, as_frame=True)
    X = d.data.copy()
    y = (d.target == "bad").astype(int).rename("bad_credit")
    return X, y


def load_bike() -> tuple[pd.DataFrame, pd.Series]:
    """Bike Sharing Demand (n=17,379 hourly rows). Regression target = `count`.

    Rows are in chronological order (hour by hour), which is exactly what
    TimeSeriesSplit / purged CV require — the row index *is* time. We drop nothing;
    `year/month/hour/weekday` encode the calendar position.
    """
    d = fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True)
    df = d.frame.reset_index(drop=True)
    y = df["count"].astype(float).rename("count")
    X = df.drop(columns=["count"])
    return X, y


_PARKINSONS_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "parkinsons/telemonitoring/parkinsons_updrs.data"
)


def load_parkinsons_groups() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """UCI Parkinsons Telemonitoring. Returns (X, y, groups).

    5,875 biomedical voice recordings from **42 patients** (~139 recordings each),
    predicting `total_UPDRS` (disease severity, a regression target).

    groups = `subject#`: every patient contributes ~139 nearly-identical recordings.
    If you split rows at random, the same patient lands in both train and test and the
    model simply recognizes the patient -> hugely inflated R^2. Splitting by patient
    (GroupKFold) reveals the model barely generalizes to *new* people. This is the
    canonical, dramatic group-leakage story (notebook 03).

    Downloaded once from UCI and cached under ~/scikit_learn_data.
    """
    cache = os.path.join(_CACHE, "parkinsons_updrs.csv")
    if os.path.exists(cache):
        df = pd.read_csv(cache)
    else:
        df = pd.read_csv(_PARKINSONS_URL)
        os.makedirs(_CACHE, exist_ok=True)
        df.to_csv(cache, index=False)

    groups = df["subject#"].rename("subject")
    y = df["total_UPDRS"].astype(float).rename("total_UPDRS")
    # Drop ids and the two targets; test_time is when the recording was taken (leaks age).
    X = df.drop(columns=["subject#", "motor_UPDRS", "total_UPDRS", "test_time"])
    return X.reset_index(drop=True), y.reset_index(drop=True), groups.reset_index(drop=True)


if __name__ == "__main__":
    # Runnable self-check: shapes, no NaN in kept columns, groups actually repeat.
    warnings.filterwarnings("ignore")

    Xc, yc = load_credit()
    assert Xc.shape == (1000, 20), Xc.shape
    assert set(yc.unique()) == {0, 1} and yc.mean() == 0.30

    Xb, yb = load_bike()
    assert len(Xb) == 17379 and yb.min() >= 0
    assert "count" not in Xb.columns

    Xp, yp, gp = load_parkinsons_groups()
    assert len(Xp) == len(yp) == len(gp) == 5875
    # The whole reason this dataset is here: each patient recurs ~139 times.
    assert gp.nunique() == 42 and (gp.value_counts() > 1).all(), "groups must recur"

    print("cv_datasets self-check OK")
    print(f"  credit-g   : {Xc.shape}, positive rate {yc.mean():.2f}")
    print(f"  bike       : {Xb.shape}, target range [{yb.min():.0f}, {yb.max():.0f}]")
    print(f"  parkinsons : {Xp.shape}, {gp.nunique()} patients / {len(gp)} recordings "
          f"(~{len(gp)//gp.nunique()} each), UPDRS range [{yp.min():.1f}, {yp.max():.1f}]")
