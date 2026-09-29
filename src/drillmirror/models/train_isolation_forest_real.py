#!/usr/bin/env python3
"""Train Isolation Forest on real 3W-like parquet dataset.

Assumes directory structure: data/Real/<class_code>/*.parquet
Labeling: class_code == 0 -> normal, else anomaly.
"""

from __future__ import annotations

import argparse
import json
import os
import random
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


def list_files(max_files_per_class: int) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for cls in sorted([d for d in os.listdir(ROOT) if (ROOT / d).is_dir()]):
        files = [f for f in os.listdir(ROOT / cls) if f.endswith(".parquet")]
        random.shuffle(files)
        for f in files[:max_files_per_class]:
            items.append((cls, str(ROOT / cls / f)))
    random.shuffle(items)
    return items


def load_sample(paths: list[tuple[str, str]], rows_per_file: int) -> tuple[pd.DataFrame, np.ndarray]:
    frames = []
    labels = []
    columns = None

    for cls, path in paths:
        table = pq.read_table(path)
        df = table.to_pandas()
        if columns is None:
            columns = [c for c in df.columns if c not in ("class", "state", "timestamp")]

        df = df[columns]
        if len(df) > rows_per_file:
            df = df.sample(rows_per_file, random_state=42)

        frames.append(df)
        labels.append(0 if cls == "0" else 1)

    X = pd.concat(frames, ignore_index=True)
    # Drop columns that are entirely missing in the sample
    X = X.loc[:, X.notna().any(axis=0)]
    y = np.repeat(labels, [len(f) for f in frames])
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-files-per-class", type=int, default=60)
    parser.add_argument("--rows-per-file", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    paths = list_files(args.max_files_per_class)
    X, y = load_sample(paths, args.rows_per_file)
    feature_cols = list(X.columns)

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.25, random_state=args.seed, stratify=y
    )

    normal_mask = y_train == 0
    contamination = min(0.5, max(0.001, float(y_train.mean())))

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=args.seed,
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

    # Timeline from first file
    cls0, path0 = paths[0]
    df0 = pq.read_table(path0).to_pandas()
    df0 = df0[feature_cols].head(1000)
    score_all = -model.decision_function(imputer.transform(df0))
    label_all = np.array([0 if cls0 == "0" else 1] * len(df0))

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
        "features": feature_cols,
    }

    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
