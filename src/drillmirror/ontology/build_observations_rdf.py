#!/usr/bin/env python3
"""Create observation instances in Turtle from synthetic CSV."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

MAP_VAR = {
    "p_pdg": "Pressure_PDG",
    "p_tpt": "Pressure_TPT",
    "t_tpt": "Temperature_TPT",
    "p_pck_up": "Pressure_PCK_Upstream",
    "t_pck_down": "Temperature_PCK_Downstream",
}

SENSOR_FOR_VAR = {
    "Pressure_PDG": "PDG_PressureSensor",
    "Pressure_TPT": "TPT_PressureSensor",
    "Temperature_TPT": "TPT_TemperatureSensor",
    "Pressure_PCK_Upstream": "PCK_UpstreamPressureSensor",
    "Temperature_PCK_Downstream": "PCK_DownstreamTemperatureSensor",
}

COMP_FOR_VAR = {
    "Pressure_PDG": "ProductionTubingA",
    "Pressure_TPT": "SubseaTreeA",
    "Temperature_TPT": "SubseaTreeA",
    "Pressure_PCK_Upstream": "ProductionLineA",
    "Temperature_PCK_Downstream": "ProductionLineA",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--output", default="ontology/observations.ttl")
    args = parser.parse_args()

    df = pd.read_csv(args.csv).head(args.limit)

    start = datetime(2026, 2, 5, 12, 0, 0, tzinfo=timezone.utc)

    lines = [
        "@prefix dt: <http://example.org/oilwell/dt#> .",
        "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
        "",
    ]

    obs_id = 0
    for _, row in df.iterrows():
        ts = start + timedelta(seconds=int(row["timestamp_s"]))
        ts_literal = ts.isoformat().replace("+00:00", "Z")

        for col, var in MAP_VAR.items():
            value = row.get(col)
            if pd.isna(value):
                continue

            obs = f"Obs_{obs_id}"
            sensor = SENSOR_FOR_VAR[var]
            comp = COMP_FOR_VAR[var]

            lines.append(f"dt:{obs} a dt:Observation ;")
            lines.append(f"  dt:hasValue {float(value):.4f} ;")
            lines.append(f"  dt:hasTimestamp \"{ts_literal}\"^^xsd:dateTime ;")
            lines.append(f"  dt:generatedBy dt:{sensor} ;")
            lines.append(f"  dt:observedAt dt:{comp} .")
            lines.append("")

            lines.append(f"dt:{var} dt:hasObservation dt:{obs} .")
            lines.append("")

            obs_id += 1

    Path(args.output).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
