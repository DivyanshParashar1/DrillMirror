#!/usr/bin/env python3
"""Train Isolation Forest on synthetic dataset and save model for pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import joblib

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/synthetic/synthetic_3w_like.csv")
MODEL_PATH = Path("data/isolation_forest_model.joblib")
IMPUTER_PATH = Path("data/imputer.joblib")

FEATURES = ["p_pdg", "p_tpt", "t_tpt", "p_pck_up", "t_pck_down"]


def main() -> None:
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    # labels: anomaly if event_code != 0
    y = (df["event_code"] != 0).astype(int).values
    X = df[FEATURES].copy()

    # impute missing with median
    print("Fitting imputer...")
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.25, random_state=42, stratify=y
    )

    # Fit only on normal samples
    normal_mask = y_train == 0
    contamination = max(0.001, float(y_train.mean()))

    print(f"Training Isolation Forest (contamination={contamination:.4f})...")
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
    )
    model.fit(X_train[normal_mask])

    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)

    print(f"Saving imputer to {IMPUTER_PATH}...")
    joblib.dump(imputer, IMPUTER_PATH)

    print("Done.")

if __name__ == "__main__":
    main()
