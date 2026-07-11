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
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "web", "public")

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
        "random_r2": 0.908, "group_r2": -0.565,
        "random_mae": 2.42, "group_mae": 10.75, "baseline_mae": 8.70,
        "note": "Same model; only the splitter changed. Negative R2 = worse than predicting the mean.",
    },
    "time_leak": {      # NB03, Bike Sharing
        "shuffled_r2": 0.947, "timeseries_r2": 0.801,
        "note": "Shuffling hourly data lets the model train on the future.",
    },
    "nested": {         # NB04, German Credit
        "non_nested": 0.799, "nested": 0.791, "nested_std": 0.022,
        "note": "best_score_ is a selection score; nested CV is the honest generalization estimate.",
    },
}


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "folds.json"), "w") as f:
        json.dump(build_folds(), f, separators=(",", ":"))
    with open(os.path.join(OUT, "headline.json"), "w") as f:
        json.dump(HEADLINE, f, indent=2)
    print("Wrote folds.json and headline.json to", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
