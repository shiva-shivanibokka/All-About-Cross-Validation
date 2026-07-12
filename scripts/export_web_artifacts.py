"""Export small JSON artifacts that drive the web CV-Visualizer (web/public/).

The visualizer teaches the *process*, not a model, so the artifacts are:
  1. folds.json   - real scikit-learn fold layouts on a tiny 48-sample demo set, so the
                    animation shows genuine KFold / Stratified / Group / TimeSeries / Purged
                    membership (state per sample: 0 unused, 1 train, 2 test, 3 purged/embargo).
  2. headline.json - the key numbers each panel cites, sourced from the notebooks (constants,
                    documented below, so this export stays fast and network-free).

Run:  python scripts/export_web_artifacts.py
"""
from __future__ import annotations
import json
import os
import sys
import numpy as np
from sklearn.model_selection import (
    KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit, cross_val_score, cross_val_predict,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "web", "public")
sys.path.insert(0, os.path.join(HERE, ".."))   # import cv_datasets from repo root

N = 48          # samples in the demo strip
K = 6           # folds for the K-family
UNUSED, TRAIN, TEST, PURGE = 0, 1, 2, 3


def _states(n, splits, purge_rows=None):
    """Turn (train_idx, test_idx) splits into per-sample state arrays, one per fold."""
    out = []
    for i, (tr, te) in enumerate(splits):
        s = np.full(n, UNUSED, dtype=int)
        s[tr] = TRAIN
        s[te] = TEST
        if purge_rows is not None:
            s[purge_rows[i]] = PURGE
        out.append(s.tolist())
    return out


def build_folds():
    rng = np.random.RandomState(0)
    # A ~30% positive imbalanced label -> shows why stratification matters.
    labels = (rng.rand(N) < 0.30).astype(int)
    # 12 groups of 4 consecutive samples -> shows GroupKFold keeping a group whole.
    groups = np.repeat(np.arange(N // 4), 4)

    methods = {}
    methods["KFold"] = {
        "k": K, "labels": labels.tolist(), "groups": None,
        "folds": _states(N, list(KFold(K, shuffle=True, random_state=0).split(np.arange(N)))),
        "blurb": "Plain rotation of K equal chunks. Ignores class balance and groups.",
    }
    methods["StratifiedKFold"] = {
        "k": K, "labels": labels.tolist(), "groups": None,
        "folds": _states(N, list(StratifiedKFold(K, shuffle=True, random_state=0).split(np.arange(N), labels))),
        "blurb": "Every fold keeps the ~30% positive rate. The classification default.",
    }
    methods["GroupKFold"] = {
        "k": K, "labels": None, "groups": groups.tolist(),
        "folds": _states(N, list(GroupKFold(K).split(np.arange(N), groups=groups))),
        "blurb": "Each colored group stays entirely in one fold. No entity leaks across the split.",
    }
    methods["TimeSeriesSplit"] = {
        "k": 5, "labels": None, "groups": None,
        "folds": _states(N, list(TimeSeriesSplit(n_splits=5).split(np.arange(N)))),
        "blurb": "Train only on the past; test on the future that follows. Never shuffled.",
    }

    # Purged + embargoed walk-forward: carve a gap between train and test.
    purge_folds, purge_rows = [], []
    fold = N // 6
    for i in range(1, 6):
        ts, teend = i * fold, (i + 1) * fold
        gap = 3                      # purge+embargo width (rows)
        tr = np.arange(0, max(0, ts - gap))
        te = np.arange(ts, min(teend, N))
        purge_folds.append((tr, te))
        purge_rows.append(np.arange(max(0, ts - gap), ts))
    methods["Purged"] = {
        "k": 5, "labels": None, "groups": None,
        "folds": _states(N, purge_folds, purge_rows),
        "blurb": "Purged + embargoed: drop training rows whose label window overlaps the test block.",
    }
    return {"n": N, "legend": {"unused": UNUSED, "train": TRAIN, "test": TEST, "purge": PURGE},
            "methods": methods}


# Numbers each panel cites. Sourced from the executed notebooks (kept as constants so this
# export needs no network and no multi-minute model runs).
HEADLINE = {
    "leakage_noise": {  # NB01, pure-noise feature-selection demo
        "leaky": 0.82, "correct": 0.49, "n_features": 10000, "n_rows": 200,
        "note": "Feature selection on all rows before CV. Data is 100% random; truth = 0.50.",
    },
    "group_leak": {     # NB03, Parkinsons Telemonitoring
        "random_r2": 0.907, "group_r2": -0.565,
        "random_mae": 2.43, "group_mae": 10.75, "baseline_mae": 8.66,
        "note": "Same model; only the splitter changed. Negative R2 = worse than predicting the mean.",
    },
    "time_leak": {      # NB03, Bike Sharing
        "shuffled_r2": 0.947, "timeseries_r2": 0.802,
        "note": "Shuffling hourly data lets the model train on the future.",
    },
    "nested": {         # NB04, German Credit
        "non_nested": 0.795, "nested": 0.790, "nested_std": 0.019,
        "note": "best_score_ is a selection score; nested CV is the honest generalization estimate.",
    },
}


# ----------------------------------------------------------------------------------------
# Richer charts (charts.json). Everything below is recomputed with real scikit-learn so the
# web visuals are genuine, not drawn by hand. Needs the datasets (cached under
# ~/scikit_learn_data after the first run).
# ----------------------------------------------------------------------------------------

def _round(a, n=4):
    return [round(float(x), n) for x in a]


def leakage_curve():
    """The lie grows with opportunity: offer more pure-noise features and leaky feature
    selection finds ever-more spuriously-'predictive' ones, while the honest (in-fold)
    pipeline stays pinned at chance. Pure noise -> truth is 0.50."""
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    n_rows, k_keep, seeds = 200, 20, 8
    y = np.array([0, 1] * (n_rows // 2))                 # perfectly balanced -> chance = 0.50
    counts = [50, 200, 1000, 4000, 10000]
    leaky, correct = [], []
    for p in counts:
        lk, ok = [], []
        for s in range(seeds):                            # average over seeds -> the expected effect
            X = np.random.RandomState(s).randn(n_rows, p) # 100% noise, target independent
            cv = StratifiedKFold(5, shuffle=True, random_state=0)
            # LEAKY: pick the k "best" features using ALL rows (peeks at the test rows' labels).
            Xsel = SelectKBest(f_classif, k=k_keep).fit_transform(X, y)
            lk.append(cross_val_score(LogisticRegression(max_iter=1000), Xsel, y, cv=cv).mean())
            # CORRECT: selection lives inside the pipeline, re-fit on each fold's train rows only.
            pipe = make_pipeline(SelectKBest(f_classif, k=k_keep), LogisticRegression(max_iter=1000))
            ok.append(cross_val_score(pipe, X, y, cv=cv).mean())
        leaky.append(np.mean(lk)); correct.append(np.mean(ok))
    return {"feature_counts": counts, "leaky": _round(leaky), "correct": _round(correct), "truth": 0.5}


def parkinsons_charts():
    """Per-patient error (strip plot) and one patient's predicted-vs-actual (scatter),
    from out-of-fold predictions under a random split vs a patient-grouped split."""
    from cv_datasets import load_parkinsons_groups
    from sklearn.ensemble import HistGradientBoostingRegressor

    X, y, groups = load_parkinsons_groups()
    y = y.to_numpy(); g = groups.to_numpy()
    model = HistGradientBoostingRegressor(random_state=0)

    pred_rand = cross_val_predict(model, X, y, cv=KFold(5, shuffle=True, random_state=0))
    pred_grp = cross_val_predict(model, X, y, cv=GroupKFold(5), groups=g)

    pids = np.unique(g)
    rand_mae, grp_mae = [], []
    for p in pids:
        m = g == p
        rand_mae.append(np.abs(pred_rand[m] - y[m]).mean())
        grp_mae.append(np.abs(pred_grp[m] - y[m]).mean())
    errors = {"patients": [int(p) for p in pids], "random_mae": _round(rand_mae, 2),
              "group_mae": _round(grp_mae, 2), "baseline_mae": round(float(np.abs(y - y.mean()).mean()), 2)}

    # Scatter: the patient whose group-MAE is the median — a typical, not cherry-picked, case.
    focus = pids[int(np.argsort(grp_mae)[len(grp_mae) // 2])]
    m = g == focus
    idx = np.argsort(y[m])                                # order by true UPDRS for a clean plot
    scatter = {"patient": int(focus), "actual": _round(y[m][idx], 2),
               "pred_random": _round(pred_rand[m][idx], 2), "pred_group": _round(pred_grp[m][idx], 2)}
    return errors, scatter


def oof_classification():
    """Out-of-fold ROC curve + confusion matrix from cross_val_predict on German Credit —
    the honest way to get one prediction per row and evaluate it as a whole."""
    from cv_datasets import load_credit, feature_types
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

    X, y = load_credit()
    num, cat = feature_types(X)
    pre = ColumnTransformer([("num", StandardScaler(), num),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    pipe = make_pipeline(pre, LogisticRegression(max_iter=1000))
    proba = cross_val_predict(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                              method="predict_proba")[:, 1]
    yv = y.to_numpy()
    fpr, tpr, _ = roc_curve(yv, proba)
    keep = np.linspace(0, len(fpr) - 1, 40).astype(int)   # thin to ~40 points for the SVG
    tn, fp, fn, tp = confusion_matrix(yv, (proba >= 0.5).astype(int)).ravel()
    return {"roc": {"fpr": _round(fpr[keep]), "tpr": _round(tpr[keep]),
                    "auc": round(float(roc_auc_score(yv, proba)), 3)},
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                          "n": int(len(yv)), "pos": int(yv.sum())}}


def build_charts():
    print("  leakage curve …");         leak = leakage_curve()
    print("  parkinsons OOF …");        errors, scatter = parkinsons_charts()
    print("  credit OOF ROC/CM …");     oof = oof_classification()
    return {"leakage_curve": leak, "patient_errors": errors, "patient_scatter": scatter, "oof": oof}


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "folds.json"), "w") as f:
        json.dump(build_folds(), f, separators=(",", ":"))
    with open(os.path.join(OUT, "headline.json"), "w") as f:
        json.dump(HEADLINE, f, indent=2)
    print("Computing charts.json (real scikit-learn; first run downloads datasets):")
    with open(os.path.join(OUT, "charts.json"), "w") as f:
        json.dump(build_charts(), f, separators=(",", ":"))
    print("Wrote folds.json, headline.json, charts.json to", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
