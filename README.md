# Drill Mirror - Oil Well Digital Twin

Digital twin prototype for offshore naturally flowing oil wells, aligned to the 3W dataset and an ontology-driven event model.

This repo includes:
- A lightweight ontology (OWL/Turtle) for equipment, variables, and undesirable events.
- Synthetic and real multivariate time series (MTS) data aligned to the 3W dataset structure.
- A dashboard to explore the ontology graph and run model outputs.
- Scripts for feature extraction, anomaly detection, and RDF graph generation.

## Dataset Reference (3W)

Key points from the paper:
- The 3W dataset is a public dataset of multivariate time series for offshore naturally flowing oil wells.
- Each instance belongs to one of three sources: real, simulated, or hand-drawn.
- Instances are labeled at two levels: instance-level (single event code) and observation-level (normal vs. event).
- Data is stored as CSV files grouped by event label, sampled at 1 Hz.
- Units include Pascal (Pa), standard cubic meters per second (sm3/s), and degrees Celsius (C).
- The paper defines eight undesirable event types with typical confirmation windows:
  - Abrupt Increase of BSW (12 h)
  - Spurious Closure of DHSV (5 min to 20 min)
  - Severe Slugging (5 h)
  - Flow Instability (15 min)
  - Rapid Productivity Loss (12 h)
  - Quick Restriction in PCK (15 min)
  - Scaling in PCK (72 h)
  - Hydrate in Production Line (30 min to 5 h)
- The paper states the dataset includes eight process variables. The variables explicitly listed in Section 2.1 are:
  - Pressure at PDG
  - Pressure at TPT
  - Temperature at TPT
  - Pressure upstream of PCK
  - Temperature downstream of PCK

This repo models those five explicitly listed variables in synthetic generation and the ontology. If you want the other three variables included, open an issue or tell me which ones to add.

## Repo Layout
- `data/` real and synthetic datasets plus exported model artifacts
- `ontology/` OWL/Turtle ontology and graph exports
- `dashboard/` static UI for graph and model exploration
- `scripts/` data prep, feature extraction, modeling, and RDF generation
- `src/` digital twin demo runner
- `server_groq.py` optional LLM chatbot backend

## Clone
```bash
git clone https://github.com/DivyanshParashar1/DrillMirror.git
cd DrillMirror
```

## Quickstart (Dashboard Only)
Serve the static dashboard:
```bash
python3 -m http.server 8000
```

Open:
`http://localhost:8000/dashboard/index.html`

## Chatbot + Dashboard (Optional)
1. Create `.env` in the project root:
```bash
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama3-8b-8192
```

2. Start the Groq chatbot server:
```bash
python3 server_groq.py
```

3. Start the dashboard:
```bash
python3 -m http.server 8000
```

Open:
`http://localhost:8000/dashboard/index.html`

## Generate Synthetic Data
```bash
python3 scripts/generate_synthetic.py --instances 120 --length 3600 \
  --output data/synthetic/synthetic_3w_like.csv
```

Output columns:
- `timestamp_s`, `instance_id`, `event_code`, `event_label`, `state`
- `p_pdg`, `p_tpt`, `t_tpt`, `p_pck_up`, `t_pck_down`

## Digital Twin Demo
```bash
python3 src/digital_twin.py --csv data/synthetic/synthetic_3w_like.csv --rows 300
```

## RDF Observations + Inferred Events
Generate observation instances (limited for size):
```bash
python3 scripts/build_observations_rdf.py --csv data/synthetic/synthetic_3w_like.csv --limit 300
```

Infer event instances from data:
```bash
python3 scripts/infer_events.py --csv data/synthetic/synthetic_3w_like.csv --window 300 --step 300
```

Rebuild the graph data used by the dashboard:
```bash
python3 scripts/build_ontology_graph.py
```

## Isolation Forest Model
Train the model and export dashboard JSON:
```bash
python3 scripts/train_isolation_forest.py
```

Train on the real parquet dataset (sampled):
```bash
python3 scripts/train_isolation_forest_real.py --max-files-per-class 10 --rows-per-file 120
```

Train on the full real dataset using instance-level features:
```bash
python3 scripts/train_isolation_forest_real_full.py
```

Build real dataset label summaries:
```bash
python3 scripts/build_real_summary.py
```

## Test Input + Chatbot Flow
Extract instance-level features from a parquet file for input:
```bash
python3 scripts/extract_instance_features.py --parquet data/Real/0/WELL-00001_20170201010207.parquet
```

Paste the resulting JSON into the dashboard and click Evaluate.

Dashboard extras:
- Toggle chatbot mode (Engineer/Manager).
- Generate PDF reports for Manager and Engineers (server-side).
