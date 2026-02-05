# Oil Well Digital Twin (Ontology + Synthetic Data)

This workspace contains:
- Synthetic multivariate time series data based on the feature list in the PDF.
- An ontology (OWL/Turtle) that models the well, equipment, variables, and events.
- A lightweight digital twin that aligns observations to ontology entities and infers coarse events.

**Features used from the PDF (section 2.1):**
- Pressure at PDG
- Pressure at TPT
- Temperature at TPT
- Pressure upstream of PCK
- Temperature downstream of PCK

Note: The paper also states the dataset has eight process variables. The list above is the explicitly listed set in the PDF text. If you want the other three variables included, tell me which ones and I’ll extend the generator and ontology.

## Generate Synthetic Data

```bash
python3 scripts/generate_synthetic.py --instances 120 --length 3600 \
  --output data/synthetic/synthetic_3w_like.csv
```

Output columns:
- `timestamp_s`, `instance_id`, `event_code`, `event_label`, `state`
- `p_pdg`, `p_tpt`, `t_tpt`, `p_pck_up`, `t_pck_down`

## Run Digital Twin Demo

```bash
python3 src/digital_twin.py --csv data/synthetic/synthetic_3w_like.csv --rows 300
```

## Ontology

The ontology is stored at `ontology/oil_well_dt.ttl` and models:
- Equipment (PDG, TPT, DHSV, PCK)
- Sensors and process variables
- Event types and states (normal, faulty transient, faulty steady)
- Units and locations

Use it with RDF tools if desired (e.g., rdflib or Protégé).

## Ontology Dashboard

Open the dashboard in a local web server:

```bash
python3 -m http.server
```

Then visit `dashboard/index.html` in the browser.

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

Train on the real parquet dataset:

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

## Test Input + Chatbot

Extract instance-level features from a parquet file for input:

```bash
python3 scripts/extract_instance_features.py --parquet data/Real/0/WELL-00001_20170201010207.parquet
```

Paste the resulting JSON into the dashboard and click Evaluate.

### Groq LLM Chatbot (Agentic)

1. Create a `.env` file in the project root:

```
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

Then open `http://localhost:8000/dashboard/index.html` and use the chatbot.

The dashboard can now:
- Load a real parquet file via the dropdown.
- Re-train the model with the button (runs `scripts/train_isolation_forest_real_full.py`).
