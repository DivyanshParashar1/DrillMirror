#!/usr/bin/env python3
"""Extract instance-level features from a parquet file for dashboard input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--output", default="data/instance_features.json")
    args = parser.parse_args()

    df = pq.read_table(args.parquet).to_pandas()
    df = df.drop(columns=[c for c in ("class", "state") if c in df.columns], errors="ignore")

    feats = {}
    for col in df.columns:
        series = df[col]
        feats[f"{col}_mean"] = float(series.mean(skipna=True))
        feats[f"{col}_std"] = float(series.std(skipna=True))
        feats[f"{col}_min"] = float(series.min(skipna=True))
        feats[f"{col}_max"] = float(series.max(skipna=True))

    Path(args.output).write_text(json.dumps(feats, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
