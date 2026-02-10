#!/usr/bin/env python3
"""Groq-backed agentic chatbot with ontology access.

Run:
  python3 server_groq.py

Requires:
  GROQ_API_KEY in .env
Optional:
  GROQ_MODEL (default: llama3-8b-8192)
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List

import requests
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
import subprocess

# Import Pipeline
from src.pipeline import IsolationForestPipeline

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

ONTO_GRAPH = Path("ontology/ontology_graph.json")
REAL_ROOT = Path("data/Real").resolve()
RETRAIN_SCRIPT = Path("scripts/train_isolation_forest_real_full.py").resolve()

app = Flask(__name__)

# Initialize Pipeline
PIPELINE = None
try:
    PIPELINE = IsolationForestPipeline("data/isolation_forest_model.joblib", "data/imputer.joblib")
    print("Pipeline loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load pipeline: {e}")


def _load_graph() -> Dict[str, Any]:
    return json.loads(ONTO_GRAPH.read_text(encoding="utf-8"))


def _lookup_ontology(terms: List[str]) -> Dict[str, Any]:
    graph = _load_graph()
    tag_map = graph.get("tag_map", {})
    edges = graph.get("edges", [])
    nodes = {n["id"]: n for n in graph.get("nodes", [])}

    results = []
    for term in terms:
        var = tag_map.get(term, term)
        related_equipment = [e["target"] for e in edges if e["label"] == "relatedToEquipment" and e["source"] == var]
        connected = [e for e in edges if e["source"] == var or e["target"] == var]
        results.append({
            "term": term,
            "variable": var,
            "variable_type": nodes.get(var, {}).get("type"),
            "equipment": related_equipment,
            "edges": connected[:10],
        })
    return {"results": results}


def _call_groq(messages: List[Dict[str, str]]) -> str:
    if not GROQ_API_KEY:
        return "GROQ_API_KEY is missing. Add it to .env and restart the server."

    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0.2,
    }
    try:
        resp = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        if not resp.ok:
            return f"Groq request failed ({resp.status_code}): {resp.text}"
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except requests.RequestException as exc:
        return f"Groq request failed: {exc}"
    except Exception as exc:
        return f"Groq response error: {exc}"


def _maybe_tool_request(text: str) -> Dict[str, Any] | None:
    text = text.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None

    tool_name = obj.get("tool")
    if tool_name in ["ontology_lookup", "select_user_role"]:
        return obj
    return None


def _is_safe_path(path: Path) -> bool:
    try:
        path.resolve().relative_to(REAL_ROOT)
        return True
    except ValueError:
        return False


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS, GET"
    return response


@app.route("/api/list-files", methods=["GET"])
def list_files():
    files: List[str] = []
    if REAL_ROOT.exists():
        for root, _, filenames in os.walk(REAL_ROOT):
            for name in filenames:
                if name.endswith(".parquet"):
                    rel = str(Path(root).resolve().relative_to(REAL_ROOT) / name)
                    files.append(rel)
                    if len(files) >= 300:
                        break
            if len(files) >= 300:
                break
    return jsonify({"files": files})


@app.route("/api/extract-features", methods=["POST", "OPTIONS"])
def extract_features():
    if request.method == "OPTIONS":
        return ("", 204)

    body = request.get_json(force=True)
    rel_path = body.get("path", "")
    if not rel_path:
        return jsonify({"error": "Missing path"}), 400

    full_path = (REAL_ROOT / rel_path).resolve()
    if not _is_safe_path(full_path) or not full_path.exists():
        return jsonify({"error": "Invalid path"}), 400

    # Use the same logic as scripts/extract_instance_features.py
    import pyarrow.parquet as pq

    df = pq.read_table(full_path).to_pandas()
    df = df.drop(columns=[c for c in ("class", "state") if c in df.columns], errors="ignore")

    feats = {}
    for col in df.columns:
        series = df[col]
        mean = float(series.mean(skipna=True))
        std = float(series.std(skipna=True))
        min_v = float(series.min(skipna=True))
        max_v = float(series.max(skipna=True))

        # Drop NaN or non-finite values to avoid JSON issues on the client
        if not (mean == mean and std == std and min_v == min_v and max_v == max_v):
            continue
        if not all(map(lambda v: abs(v) != float("inf"), [mean, std, min_v, max_v])):
            continue

        feats[f"{col}_mean"] = mean
        feats[f"{col}_std"] = std
        feats[f"{col}_min"] = min_v
        feats[f"{col}_max"] = max_v

    return jsonify({"features": feats})


@app.route("/api/retrain", methods=["POST", "OPTIONS"])
def retrain():
    if request.method == "OPTIONS":
        return ("", 204)

    if not RETRAIN_SCRIPT.exists():
        return jsonify({"error": "Retrain script not found"}), 500

    try:
        subprocess.run(["python3", str(RETRAIN_SCRIPT)], check=True, timeout=3600)
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Retraining timed out"}), 500
    except subprocess.CalledProcessError as exc:
        return jsonify({"error": f"Retraining failed: {exc}"}), 500

    model_path = Path("data/model_results.json")
    stats_path = Path("data/feature_stats.json")
    if model_path.exists() and stats_path.exists():
        return jsonify({
            "model": json.loads(model_path.read_text(encoding="utf-8")),
            "feature_stats": json.loads(stats_path.read_text(encoding="utf-8")),
        })
    return jsonify({"error": "Model outputs missing"}), 500


@app.route("/api/generate-report", methods=["POST", "OPTIONS"])
def generate_report():
    if request.method == "OPTIONS":
        return ("", 204)

    if not PIPELINE:
         return jsonify({"error": "Pipeline not loaded. Ensure models are trained and saved in data/."}), 500

    try:
        body = request.get_json(force=True)
        data = body.get("data")
        user_role = body.get("user_role", "Manager") # Default to Manager

        if not data:
             return jsonify({"error": "Missing data"}), 400

        prediction = PIPELINE.predict(data)

        # Ensure download directory exists
        download_dir = Path("dashboard/downloads")
        download_dir.mkdir(parents=True, exist_ok=True)
        # Use UUID for unique filenames
        filename = f"report_{uuid.uuid4()}.pdf"
        output_path = download_dir / filename

        PIPELINE.generate_infographic_pdf(prediction, user_role, str(output_path))

        return jsonify({
            "pdf_url": f"/download/{filename}",
            "prediction": prediction
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/download/<path:filename>")
def download_file(filename):
    return send_from_directory("dashboard/downloads", filename)


@app.route("/api/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return ("", 204)

    try:
        body = request.get_json(force=True)
        question = body.get("question", "").strip()
        top_contrib = body.get("top_contrib", [])
        verdict = body.get("verdict", "unknown")
        score = body.get("score", 0)
        user_role = body.get("user_role", "unknown")
    except Exception as exc:
        return jsonify({"answer": f"Invalid request: {exc}"}), 400

    if not question:
        return jsonify({"answer": "Please ask a question."})

    terms = [c.get("tag", "") for c in top_contrib if c.get("tag")]
    context = _lookup_ontology(terms) if terms else {"results": []}

    system = (
        "You are an assistant that explains anomaly model outputs using ontology context. "
        f"The user is a {user_role}. "
        "If the user is an Engineer, provide technical details and focus on maintenance quality. "
        "If the user is a Manager, provide cost/profit analysis and high-level summaries. "
        "Remember there is a cooldown period of 1 month every 6 months for maintenance. "
        "If you need more ontology details, respond ONLY with JSON: "
        '{"tool":"ontology_lookup","args":{"terms":["P-PDG","ABER-CKP"]}}. '
        "If you need to switch user role or clarify it, respond ONLY with JSON: "
        '{"tool":"select_user_role","args":{}}. '
        "Otherwise, answer concisely and cite tags and equipment involved."
    )

    user = {
        "verdict": verdict,
        "score": score,
        "top_contributors": top_contrib,
        "ontology_context": context,
        "question": question,
        "user_role": user_role
    }

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user)},
    ]

    first = _call_groq(messages)
    tool_req = _maybe_tool_request(first)

    if tool_req:
        tool_name = tool_req.get("tool")
        if tool_name == "ontology_lookup":
            tool_terms = tool_req.get("args", {}).get("terms", [])
            tool_result = _lookup_ontology(tool_terms)
            messages.append({"role": "assistant", "content": first})
            messages.append({"role": "tool", "content": json.dumps(tool_result)})
            final = _call_groq(messages)
            return jsonify({"answer": final})
        elif tool_name == "select_user_role":
             # This tells the client to prompt for user role
             return jsonify({"tool_request": "select_user_role", "answer": "I need to know if you are an Engineer or a Manager to provide the best answer."})

    return jsonify({"answer": first})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
