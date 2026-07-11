"""Shared real-dataset loaders for the cross-validation notebooks.

Every notebook imports from here so the same clean, raw (un-encoded) frames are
used everywhere. Categoricals are kept as pandas `category` dtype ON PURPOSE:
the notebooks build sklearn Pipelines that encode *inside* each CV fold, which is
the whole point of notebook 01 (no leakage).

All three datasets are fetched from OpenML at runtime and cached locally by
scikit-learn (~/scikit_learn_data). Nothing is committed to the repo.

    from cv_datasets import load_credit, load_bike, load_diabetes_groups, feature_types
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml


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


def load_diabetes_groups(
    n_subsample: int | None = 20000, random_state: int = 42
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Diabetes 130-US Hospitals. Returns (X, y, groups).

    groups = `patient_nbr`: one patient can have MANY hospital encounters, so the
    same patient can land in both train and test unless you split by group. That is
    the canonical group-leakage story (notebook 03).

    y = early readmission (readmitted '<30' days = 1, else 0).

    Subsampled to `n_subsample` encounters by default for notebook-friendly runtimes;
    subsampling keeps whole patients together so groups stay intact.
    """
    d = fetch_openml("diabetes130US", version=1, as_frame=True)
    df = d.frame.copy()

    groups_full = df["patient_nbr"]
    y_full = (df["readmitted"] == "<30").astype(int).rename("readmit_lt30")

    # Drop ids, the target, and columns that are mostly missing or extreme-cardinality.
    drop = [
        "encounter_id", "patient_nbr", "readmitted",
        "weight", "payer_code", "medical_specialty",  # ~50-97% missing
        "diag_1", "diag_2", "diag_3",                  # hundreds of ICD codes each
    ]
    X_full = df.drop(columns=drop)

    if n_subsample is not None and n_subsample < len(df):
        # Sample whole patients, not rows, so no group is split by the subsample.
        rng = np.random.RandomState(random_state)
        order = rng.permutation(groups_full.unique())
        keep, seen = [], 0
        gsizes = groups_full.value_counts()
        for g in order:
            keep.append(g)
            seen += int(gsizes[g])
            if seen >= n_subsample:
                break
        mask = groups_full.isin(keep).to_numpy()
        X_full, y_full, groups_full = X_full[mask], y_full[mask], groups_full[mask]

    X = X_full.reset_index(drop=True)
    y = y_full.reset_index(drop=True)
    groups = groups_full.reset_index(drop=True).rename("patient_nbr")
    return X, y, groups


if __name__ == "__main__":
    # Runnable self-check: shapes, no NaN in kept columns, groups actually repeat.
    warnings.filterwarnings("ignore")

    Xc, yc = load_credit()
    assert Xc.shape == (1000, 20), Xc.shape
    assert set(yc.unique()) == {0, 1} and yc.mean() == 0.30

    Xb, yb = load_bike()
    assert len(Xb) == 17379 and yb.min() >= 0
    assert "count" not in Xb.columns

    Xd, yd, gd = load_diabetes_groups(n_subsample=20000)
    assert len(Xd) == len(yd) == len(gd)
    # The whole reason this dataset is here: patients recur across rows.
    assert gd.nunique() < len(gd), "groups must repeat or GroupKFold is pointless"
    n_num, n_cat = feature_types(Xd)
    assert len(n_num) + len(n_cat) == Xd.shape[1]

    print("cv_datasets self-check OK")
    print(f"  credit-g : {Xc.shape}, positive rate {yc.mean():.2f}")
    print(f"  bike     : {Xb.shape}, target range [{yb.min():.0f}, {yb.max():.0f}]")
    print(f"  diabetes : {Xd.shape}, {gd.nunique()} patients / {len(gd)} encounters, "
          f"positive rate {yd.mean():.3f}")
