# Drill Mirror - Oil Well Digital Twin

Digital twin prototype for offshore naturally flowing oil wells, aligned to the 3W dataset and an ontology-driven event model.

This repo includes:
- A lightweight ontology (OWL/Turtle) for equipment, variables, and undesirable events.
- Synthetic and real multivariate time series (MTS) data aligned to the 3W dataset structure.
- A dashboard to explore the ontology graph and run model outputs.
- A `drillmirror` Python package with feature extraction, anomaly detection, RDF generation, and a chatbot API.

## Repo Layout

```
DrillMirror/
├── src/drillmirror/          # Python package
│   ├── twin/                 # Digital twin runner
│   ├── data_pipeline/        # Synthetic gen, feature extraction, real summary
│   ├── models/               # Isolation Forest training (synthetic + real)
│   ├── ontology/             # RDF/Turtle builders and event inference
│   └── api/                  # Flask + Groq chatbot server
├── dashboard/                # Static UI (index.html, app.js, styles.css)
├── ontology/                 # OWL/Turtle files and JSON exports
├── data/                     # Small tracked artifacts; large data gitignored
├── tests/                    # Test suite
├── pyproject.toml
└── README.md
```

## Dataset Reference (3W)

Key points from the paper:
- The 3W dataset is a public dataset of multivariate time series for offshore naturally flowing oil wells.
- Each instance belongs to one of three sources: real, simulated, or hand-drawn.
- Instances are labeled at two levels: instance-level (single event code) and observation-level (normal vs. event).
- Data is stored as CSV/parquet files grouped by event label, sampled at 1 Hz.
- The paper defines eight undesirable event types (BSW increase, DHSV closure, severe slugging, flow instability, productivity loss, PCK restriction, PCK scaling, hydrate).

This repo models the five variables explicitly listed in Section 2.1 of the paper:
Pressure at PDG, Pressure at TPT, Temperature at TPT, Pressure upstream of PCK, Temperature downstream of PCK.

## Install

```bash
git clone https://github.com/DivyanshParashar1/DrillMirror.git
cd DrillMirror
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

All commands below assume you run them from the repo root.

## Quickstart (Dashboard Only)

```bash
python3 -m http.server 8000
# open http://localhost:8000/dashboard/index.html
```

## Chatbot + Dashboard

1. Create `.env` in the project root:
   ```
   GROQ_API_KEY=your_key_here
   GROQ_MODEL=llama3-8b-8192
   ```

2. Start the chatbot server and dashboard:
   ```bash
   python3 -m drillmirror.api.server
   python3 -m http.server 8000
   ```

Open `http://localhost:8000/dashboard/index.html`.

## Data Pipeline

Generate synthetic data:
```bash
python3 -m drillmirror.data_pipeline.generate_synthetic \
    --instances 120 --length 3600 \
    --output data/synthetic/synthetic_3w_like.csv
```

Extract instance-level features from a real parquet:
```bash
python3 -m drillmirror.data_pipeline.extract_instance_features \
    --parquet data/Real/0/WELL-00001_20170201010207.parquet
```

Build real dataset label summary:
```bash
python3 -m drillmirror.data_pipeline.build_real_summary
```

## Digital Twin Demo

```bash
python3 -m drillmirror.twin.digital_twin \
    --csv data/synthetic/synthetic_3w_like.csv --rows 300
```

## Ontology / RDF

```bash
python3 -m drillmirror.ontology.build_observations_rdf \
    --csv data/synthetic/synthetic_3w_like.csv --limit 300
python3 -m drillmirror.ontology.infer_events \
    --csv data/synthetic/synthetic_3w_like.csv --window 300 --step 300
python3 -m drillmirror.ontology.build_ontology_graph
python3 -m drillmirror.ontology.build_ontology_summary
```

## Isolation Forest

Synthetic:
```bash
python3 -m drillmirror.models.train_isolation_forest
```

Real (sampled):
```bash
python3 -m drillmirror.models.train_isolation_forest_real \
    --max-files-per-class 10 --rows-per-file 120
```

Real (full instance-level features):
```bash
python3 -m drillmirror.models.train_isolation_forest_real_full
```

## Data Directory

Small artifacts (JSON summaries, feature stats, model results) are tracked in `data/`.
Large binary datasets are gitignored and expected to live locally:

- `data/Real/`          — raw 3W parquet dataset (~1.7 GB)
- `data/synthetic/`     — generated CSVs (~50 MB)
- `data/*.joblib`       — trained scikit-learn artifacts
