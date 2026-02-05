#!/usr/bin/env python3
"""Build a lightweight ontology summary from Turtle for the dashboard."""

from __future__ import annotations

import json
import re
from pathlib import Path

TTL_PATH = Path("ontology/oil_well_dt.ttl")
OUT_PATH = Path("ontology/ontology_summary.json")

CLASS_RE = re.compile(r"^dt:([A-Za-z0-9_]+)\s+a\s+owl:Class")
OBJPROP_RE = re.compile(r"^dt:([A-Za-z0-9_]+)\s+a\s+owl:ObjectProperty")
DATAPROP_RE = re.compile(r"^dt:([A-Za-z0-9_]+)\s+a\s+owl:DatatypeProperty")
INSTANCE_RE = re.compile(r"^dt:([A-Za-z0-9_]+)\s+a\s+dt:([A-Za-z0-9_]+)")


def main() -> None:
    text = TTL_PATH.read_text(encoding="utf-8")
    classes = []
    obj_props = []
    data_props = []
    instances = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("@prefix"):
            continue

        m = CLASS_RE.match(line)
        if m:
            classes.append(m.group(1))
            continue

        m = OBJPROP_RE.match(line)
        if m:
            obj_props.append(m.group(1))
            continue

        m = DATAPROP_RE.match(line)
        if m:
            data_props.append(m.group(1))
            continue

        m = INSTANCE_RE.match(line)
        if m:
            subject, cls = m.group(1), m.group(2)
            # skip if this is a class definition line (already captured above)
            if subject not in classes:
                instances.append({"id": subject, "class": cls})

    summary = {
        "classes": sorted(set(classes)),
        "object_properties": sorted(set(obj_props)),
        "datatype_properties": sorted(set(data_props)),
        "instances": sorted(instances, key=lambda x: (x["class"], x["id"])),
    }

    OUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
