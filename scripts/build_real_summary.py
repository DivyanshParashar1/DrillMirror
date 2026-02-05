#!/usr/bin/env python3
"""Summarize real dataset (class counts, state counts, tag coverage)."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path("data/Real")
OUT = Path("data/real_summary.json")


def main() -> None:
    class_counts = {}
    state_counts = Counter()
    tag_coverage = Counter()

    for cls in sorted([d for d in os.listdir(ROOT) if (ROOT / d).is_dir()]):
        files = [f for f in os.listdir(ROOT / cls) if f.endswith(".parquet")]
        class_counts[cls] = len(files)

        for f in files:
            path = ROOT / cls / f
            table = pq.read_table(path, columns=["state"])  # state is a column
            series = table.to_pandas()["state"].dropna()
            state_counts.update(series.astype(int).tolist())

            # tag coverage from schema (lightweight)
            schema = pq.read_schema(path)
            for name in schema.names:
                if name not in ("class", "state", "timestamp"):
                    tag_coverage[name] += 1

    summary = {
        "class_counts": class_counts,
        "state_counts": dict(sorted(state_counts.items())),
        "tag_coverage": dict(sorted(tag_coverage.items())),
    }

    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
