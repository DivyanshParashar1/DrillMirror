#!/usr/bin/env python3
"""Train Isolation Forest using instance-level features from full real dataset."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

ROOT = Path("data/Real")
OUT_PATH = Path("data/model_results.json")


def extract_instance_features(path: Path) -> pd.Series:
    df = pq.read_table(path).to_pandas()
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])
    # timestamp is index
    df = df.drop(columns=[c for c in ("class", "state") if c in df.columns], errors="ignore")
    # numeric summary per variable
    feats = {}
    for col in df.columns:
        series = df[col]
        feats[f"{col}_mean"] = series.mean(skipna=True)
        feats[f"{col}_std"] = series.std(skipna=True)
        feats[f"{col}_min"] = series.min(skipna=True)
        feats[f"{col}_max"] = series.max(skipna=True)
    return pd.Series(feats)


def main() -> None:
    X_rows = []
    y = []
    inst_ids = []

    for cls in sorted([d for d in os.listdir(ROOT) if (ROOT / d).is_dir()]):
        files = [f for f in os.listdir(ROOT / cls) if f.endswith(".parquet")]
        for f in files:
            path = ROOT / cls / f
            feats = extract_instance_features(path)
            X_rows.append(feats)
            y.append(0 if cls == "0" else 1)
            inst_ids.append(f"{cls}/{f}")

    X = pd.DataFrame(X_rows)
    # Drop columns that are entirely missing across all instances
    X = X.loc[:, X.notna().any(axis=0)]
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    y = np.array(y)

    # Feature stats from normal instances for dashboard scoring
    normal_X = pd.DataFrame(X_imputed, columns=X.columns)[y == 0]
    feature_stats = {
        "mean": normal_X.mean().to_dict(),
        "std": normal_X.std().replace(0, np.nan).fillna(1.0).to_dict(),
    }

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X_imputed, y, inst_ids, test_size=0.25, random_state=42, stratify=y
    )

    normal_mask = y_train == 0
    contamination = min(0.5, max(0.001, float(y_train.mean())))

    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=42,
    )
    model.fit(X_train[normal_mask])

    score_test = -model.decision_function(X_test)
    pred_test = (model.predict(X_test) == -1).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, pred_test, average="binary", zero_division=0
    )

    roc_auc = roc_auc_score(y_test, score_test)
    pr_auc = average_precision_score(y_test, score_test)

    # Histogram
    bins = 30
    normal_scores = score_test[y_test == 0]
    anomaly_scores = score_test[y_test == 1]
    hist_n, bin_edges = np.histogram(normal_scores, bins=bins)
    hist_a, _ = np.histogram(anomaly_scores, bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Timeline: sample first 200 instance scores
    timeline_scores = score_test[:200].tolist()
    timeline_labels = y_test[:200].tolist()

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
            "score": timeline_scores,
            "label": timeline_labels,
        },
        "features": list(X.columns),
        "instance_ids": ids_test[:200],
    }

    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")

    stats_path = Path("data/feature_stats.json")
    stats_path.write_text(json.dumps(feature_stats, indent=2), encoding="utf-8")
    print(f"Wrote {stats_path}")


if __name__ == "__main__":
    main()
