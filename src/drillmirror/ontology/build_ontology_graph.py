#!/usr/bin/env python3
"""Build a simple graph JSON from Turtle for visualization."""

from __future__ import annotations

import json
import re
from pathlib import Path

TTL_PATH = Path("ontology/oil_well_dt.ttl")
OUT_PATH = Path("ontology/ontology_graph.json")

TRIPLE_RE = re.compile(r"^dt:([A-Za-z0-9_]+)\s+dt:([A-Za-z0-9_]+)\s+dt:([A-Za-z0-9_]+)")
TYPE_RE = re.compile(r"^dt:([A-Za-z0-9_]+)\s+a\s+dt:([A-Za-z0-9_]+)")
TAG_RE = re.compile(r"^dt:([A-Za-z0-9_]+).*dt:tagName\s+\"([^\"]+)\"")


def main() -> None:
    text = TTL_PATH.read_text(encoding="utf-8")

    nodes = {}
    edges = []
    tag_map = {}

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("@prefix"):
            continue

        m = TAG_RE.match(line)
        if m:
            tag_map[m.group(2)] = m.group(1)

        m = TYPE_RE.match(line)
        if m:
            node_id, cls = m.group(1), m.group(2)
            nodes.setdefault(node_id, {"id": node_id, "type": cls})
            continue

        m = TRIPLE_RE.match(line)
        if m:
            src, pred, dst = m.group(1), m.group(2), m.group(3)
            nodes.setdefault(src, {"id": src, "type": "Entity"})
            nodes.setdefault(dst, {"id": dst, "type": "Entity"})
            edges.append({"source": src, "target": dst, "label": pred})

    graph = {"nodes": list(nodes.values()), "edges": edges, "tag_map": tag_map}
    OUT_PATH.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
