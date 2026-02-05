#!/usr/bin/env python3
"""Train Isolation Forest on synthetic dataset and export JSON for dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/synthetic/synthetic_3w_like.csv")
OUT_PATH = Path("data/model_results.json")

FEATURES = ["p_pdg", "p_tpt", "t_tpt", "p_pck_up", "t_pck_down"]


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    # labels: anomaly if event_code != 0
    y = (df["event_code"] != 0).astype(int).values
    X = df[FEATURES].copy()

    # impute missing with median
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.25, random_state=42, stratify=y
    )

    # Fit only on normal samples
    normal_mask = y_train == 0
    contamination = max(0.001, float(y_train.mean()))

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
    )
    model.fit(X_train[normal_mask])

    # Scores: higher = more normal. Convert to anomaly score.
    score_test = -model.decision_function(X_test)
    pred_test = (model.predict(X_test) == -1).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, pred_test, average="binary", zero_division=0
    )

    roc_auc = roc_auc_score(y_test, score_test)
    pr_auc = average_precision_score(y_test, score_test)

    # Histogram for scores
    bins = 30
    normal_scores = score_test[y_test == 0]
    anomaly_scores = score_test[y_test == 1]

    hist_n, bin_edges = np.histogram(normal_scores, bins=bins)
    hist_a, _ = np.histogram(anomaly_scores, bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Timeline sample
    sample_len = 1000
    score_all = -model.decision_function(X_imputed[:sample_len])
    label_all = y[:sample_len]

    results = {
        "metrics": {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
        },
        "counts": {
            "total": int(len(y)),
            "anomalies": int(y.sum()),
            "normal": int((y == 0).sum()),
        },
        "contamination": contamination,
        "score_hist": {
            "bins": bin_centers.tolist(),
            "normal": hist_n.tolist(),
            "anomaly": hist_a.tolist(),
        },
        "timeline": {
            "score": score_all.tolist(),
            "label": label_all.tolist(),
        },
    }

    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
