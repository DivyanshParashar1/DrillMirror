#!/usr/bin/env python3
"""Ontology-aligned digital twin for an offshore naturally flowing oil well.

This is a lightweight twin that maps live observations to ontology entities
and infers coarse event types with simple rules. It does not require external
RDF libraries; it uses ontology URIs as identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


ONTOLOGY = {
    "Pressure_PDG": "http://example.org/oilwell/dt#Pressure_PDG",
    "Pressure_TPT": "http://example.org/oilwell/dt#Pressure_TPT",
    "Temperature_TPT": "http://example.org/oilwell/dt#Temperature_TPT",
    "Pressure_PCK_Upstream": "http://example.org/oilwell/dt#Pressure_PCK_Upstream",
    "Temperature_PCK_Downstream": "http://example.org/oilwell/dt#Temperature_PCK_Downstream",
}


@dataclass
class VariableState:
    uri: str
    latest_value: Optional[float] = None


@dataclass
class EventHypothesis:
    event_label: str
    score: float


@dataclass
class DigitalTwin:
    variables: Dict[str, VariableState] = field(default_factory=dict)
    history: List[Dict[str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        for name, uri in ONTOLOGY.items():
            self.variables[name] = VariableState(uri=uri)

    def update_from_row(self, row: pd.Series) -> None:
        snapshot: Dict[str, float] = {}
        for key, state in self.variables.items():
            value = row.get(self._df_key(key))
            if pd.notna(value):
                state.latest_value = float(value)
                snapshot[key] = float(value)
        if snapshot:
            self.history.append(snapshot)

    def infer_event(self) -> EventHypothesis:
        # Very simple heuristics using recent deltas.
        if len(self.history) < 5:
            return EventHypothesis("insufficient_data", 0.0)

        cur = self.history[-1]
        prev = self.history[-5]

        dpck = cur.get("Pressure_PCK_Upstream", 0.0) - prev.get("Pressure_PCK_Upstream", 0.0)
        dtpt = cur.get("Temperature_TPT", 0.0) - prev.get("Temperature_TPT", 0.0)
        dpdg = cur.get("Pressure_PDG", 0.0) - prev.get("Pressure_PDG", 0.0)

        if dpck > 1.5e6 and dtpt < -3.0:
            return EventHypothesis("quick_restriction_pck", 0.7)
        if dpdg > 1.0e6 and dtpt < -2.0:
            return EventHypothesis("spurious_closure_dhsv", 0.6)
        if dpck < -1.0e6 and dtpt < -4.0:
            return EventHypothesis("abrupt_increase_bsw", 0.5)

        return EventHypothesis("normal", 0.4)

    @staticmethod
    def _df_key(ontology_key: str) -> str:
        mapping = {
            "Pressure_PDG": "p_pdg",
            "Pressure_TPT": "p_tpt",
            "Temperature_TPT": "t_tpt",
            "Pressure_PCK_Upstream": "p_pck_up",
            "Temperature_PCK_Downstream": "t_pck_down",
        }
        return mapping[ontology_key]


def run_demo(csv_path: str, rows: int = 300) -> None:
    df = pd.read_csv(csv_path)
    twin = DigitalTwin()

    for _, row in df.head(rows).iterrows():
        twin.update_from_row(row)

    hypothesis = twin.infer_event()
    print(f"Event hypothesis: {hypothesis.event_label} (score={hypothesis.score:.2f})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--rows", type=int, default=300)
    args = parser.parse_args()

    run_demo(args.csv, args.rows)
