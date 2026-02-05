#!/usr/bin/env python3
"""Infer event instances from data using simple rules and emit Turtle."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def infer_event(window: pd.DataFrame) -> tuple[str, str, float]:
    # Use deltas across the window to infer a coarse event type and state.
    first = window.iloc[0]
    last = window.iloc[-1]

    dpck = (last["p_pck_up"] - first["p_pck_up"]) if not pd.isna(last["p_pck_up"]) else 0.0
    dtpt = (last["t_tpt"] - first["t_tpt"]) if not pd.isna(last["t_tpt"]) else 0.0
    dpdg = (last["p_pdg"] - first["p_pdg"]) if not pd.isna(last["p_pdg"]) else 0.0

    if dpck > 1.5e6 and dtpt < -3.0:
        return ("QuickRestrictionPCK", "FaultyTransient", 0.7)
    if dpdg > 1.0e6 and dtpt < -2.0:
        return ("SpuriousClosureDHSV", "FaultyTransient", 0.6)
    if dpck < -1.0e6 and dtpt < -4.0:
        return ("AbruptIncreaseBSW", "FaultyTransient", 0.55)

    return ("NormalOperation", "NormalState", 0.4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--window", type=int, default=300)
    parser.add_argument("--step", type=int, default=300)
    parser.add_argument("--output", default="ontology/inferred_events.ttl")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    lines = [
        "@prefix dt: <http://example.org/oilwell/dt#> .",
        "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
        "",
    ]

    event_id = 0
    for start in range(0, len(df) - args.window + 1, args.step):
        window = df.iloc[start : start + args.window]
        event_type, state, score = infer_event(window)

        lines.append(f"dt:Event_{event_id} a dt:EventInstance ;")
        lines.append(f"  dt:hasEventType dt:{event_type} ;")
        lines.append(f"  dt:hasState dt:{state} ;")
        lines.append(f"  dt:hasValue {score:.2f} .")
        lines.append("")
        event_id += 1

    Path(args.output).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
