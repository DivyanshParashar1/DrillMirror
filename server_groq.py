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
from pathlib import Path
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_file
import subprocess
import tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

ONTO_GRAPH = Path("ontology/ontology_graph.json")
REAL_ROOT = Path("data/Real").resolve()
RETRAIN_SCRIPT = Path("scripts/train_isolation_forest_real_full.py").resolve()
MODEL_PATH = Path("data/model_results.json")
SUMMARY_PATH = Path("data/real_summary.json")

app = Flask(__name__)


def _load_graph() -> Dict[str, Any]:
    return json.loads(ONTO_GRAPH.read_text(encoding="utf-8"))


def _load_model() -> Dict[str, Any]:
    return json.loads(MODEL_PATH.read_text(encoding="utf-8"))


def _load_summary() -> Dict[str, Any]:
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


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
    if obj.get("tool") == "ontology_lookup":
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
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    return response


@app.route("/api/list-files", methods=["GET"])
def list_files():
    files: List[str] = []
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


def _render_report_pdf(report_type: str, verdict: str, score: float, top_contrib: List[Dict[str, Any]]) -> str:
    model = _load_model()
    summary = _load_summary()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        out_path = tmp.name

    with PdfPages(out_path) as pdf:
        # Page 1: Summary
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
        fig.text(0.08, 0.95, "Manager Summary Report" if report_type == "manager" else "Engineering Incident Report", fontsize=16, weight="bold")
        fig.text(0.08, 0.92, f"Verdict: {verdict} | Score: {score:.2f}", fontsize=10)

        if report_type == "manager":
            notes = (
                "Plain-language impact: an unusual pattern was detected.\n"
                "Typical offshore consequences include hours to days of downtime and measurable production loss.\n"
                "Assumptions: public offshore drilling context; actual impact depends on well state and response time."
            )
        else:
            notes = (
                "Technical context: Isolation Forest trained on instance-level statistics across wells.\n"
                "High anomaly score indicates deviation from baseline distributions.\n"
                "Assumptions: public offshore drilling context; expect sensor noise, missing tags, and drift."
            )
        fig.text(0.08, 0.86, notes, fontsize=9)

        # Top contributors bar
        contrib_names = [c.get("tag", c.get("name", "tag")) for c in top_contrib][:10]
        contrib_vals = [c.get("z", 0) for c in top_contrib][:10]
        ax = fig.add_axes([0.1, 0.55, 0.8, 0.25])
        ax.barh(contrib_names[::-1], contrib_vals[::-1], color="#ff8a65")
        ax.set_title("Top Contributors (Current)", fontsize=10)
        ax.set_xlabel("Deviation (z-score)")

        # Score hist
        ax2 = fig.add_axes([0.1, 0.25, 0.8, 0.22])
        bins = model["score_hist"]["bins"]
        ax2.bar(bins, model["score_hist"]["normal"], width=0.02, alpha=0.6, label="Normal")
        ax2.bar(bins, model["score_hist"]["anomaly"], width=0.02, alpha=0.6, label="Anomaly")
        ax2.set_title("Score Distribution (Historical)", fontsize=10)
        ax2.legend(fontsize=8)

        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Class distribution
        fig2 = plt.figure(figsize=(8.27, 11.69))
        fig2.text(0.08, 0.95, "Historical Instance Distribution", fontsize=14, weight="bold")
        labels = list(summary["class_counts"].keys())
        values = list(summary["class_counts"].values())
        ax3 = fig2.add_axes([0.1, 0.55, 0.8, 0.35])
        ax3.bar(labels, values, color="#7e57c2")
        ax3.set_title("Instances by Class", fontsize=10)
        ax3.set_xlabel("Class")
        ax3.set_ylabel("Count")

        pdf.savefig(fig2)
        plt.close(fig2)

    return out_path


@app.route("/api/report", methods=["POST", "OPTIONS"])
def report():
    if request.method == "OPTIONS":
        return ("", 204)

    body = request.get_json(force=True)
    report_type = body.get("type", "manager")
    verdict = body.get("verdict", "unknown")
    score = float(body.get("score", 0))
    top_contrib = body.get("top_contrib", [])

    out_path = _render_report_pdf(report_type, verdict, score, top_contrib)
    return send_file(out_path, mimetype="application/pdf", as_attachment=True, download_name=f"{report_type}_report.pdf")


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
        mode = body.get("mode", "engineer")
    except Exception as exc:
        return jsonify({"answer": f"Invalid request: {exc}"}), 400

    if not question:
        return jsonify({"answer": "Please ask a question."})

    terms = [c.get("tag", "") for c in top_contrib if c.get("tag")]
    context = _lookup_ontology(terms) if terms else {"results": []}

    if mode == "manager":
        system = (
            "You are a manager-focused assistant. Use plain language and avoid technical jargon. "
            "Give concrete consequences (downtime, production loss, safety impact) and estimated ranges. "
            "If details are missing, make reasonable public-domain assumptions about offshore drilling. "
            "If you need more ontology details, respond ONLY with JSON: "
            '{"tool":"ontology_lookup","args":{"terms":["P-PDG","ABER-CKP"]}}.'
        )
    else:
        system = (
            "You are an engineer-focused assistant. Explain model outputs with technical detail and subsystem reasoning. "
            "Be explicit about which tags and equipment are implicated and how the system behaves. "
            "If details are missing, make reasonable public-domain assumptions about offshore drilling. "
            "If you need more ontology details, respond ONLY with JSON: "
            '{"tool":"ontology_lookup","args":{"terms":["P-PDG","ABER-CKP"]}}.'
        )

    user = {
        "verdict": verdict,
        "score": score,
        "top_contributors": top_contrib,
        "ontology_context": context,
        "question": question,
    }

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user)},
    ]

    first = _call_groq(messages)
    tool_req = _maybe_tool_request(first)

    if tool_req:
        tool_terms = tool_req.get("args", {}).get("terms", [])
        tool_result = _lookup_ontology(tool_terms)
        messages.append({"role": "assistant", "content": first})
        messages.append({"role": "tool", "content": json.dumps(tool_result)})
        final = _call_groq(messages)
        return jsonify({"answer": final})

    return jsonify({"answer": first})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
