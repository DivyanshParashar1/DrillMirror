"""Pipeline for Isolation Forest model and Groq-based infographic generation."""

import os
import json
import joblib
import pandas as pd
import requests
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from weasyprint import HTML
except ImportError:
    HTML = None

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

class IsolationForestPipeline:
    def __init__(self, model_path: str, imputer_path: str):
        self.model = joblib.load(model_path)
        self.imputer = joblib.load(imputer_path)
        # Expected features order
        self.features = ["p_pdg", "p_tpt", "t_tpt", "p_pck_up", "t_pck_down"]

    def predict(self, data: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict anomaly score for a single data point.
        Data should be a dictionary with keys: p_pdg, p_tpt, t_tpt, p_pck_up, t_pck_down.
        """
        # Create DataFrame
        df = pd.DataFrame([data])

        # Ensure columns are in correct order
        df = df[self.features]

        # Impute
        X_imputed = self.imputer.transform(df)

        # Predict
        # Decision function: negative values are anomalies, positive are normal (if not inverted).
        # But sklearn isolation forest decision_function returns:
        # positive for inliers, negative for outliers.
        score = self.model.decision_function(X_imputed)[0]
        is_anomaly = self.model.predict(X_imputed)[0] == -1

        return {
            "score": float(score),
            "is_anomaly": bool(is_anomaly),
            "input_data": data
        }

    def _call_groq(self, messages: list) -> str:
        if not GROQ_API_KEY:
            return "<html><body><h1>Error: GROQ_API_KEY missing</h1></body></html>"

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
                return f"<html><body><h1>Groq Error: {resp.status_code}</h1><p>{resp.text}</p></body></html>"
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as exc:
            return f"<html><body><h1>Groq Exception: {exc}</h1></body></html>"

    def generate_infographic_pdf(self, prediction: Dict[str, Any], user_role: str, output_path: str) -> str:
        """
        Generates an infographic PDF based on prediction and user role.
        """
        if HTML is None:
            raise ImportError("WeasyPrint not installed.")

        # Construct Prompt
        role_desc = "Engineer" if user_role.lower() == "engineer" else "Manager"

        system_prompt = (
            "You are an expert data visualization assistant. "
            "You generate HTML reports with inline CSS and SVG for visualizations. "
            "Do NOT use external JavaScript libraries (like Chart.js) because the PDF renderer cannot execute them. "
            "Use pure HTML/CSS or SVG for graphs."
        )

        user_content = f"""
        Generate a single-page HTML infographic report for a {role_desc}.

        Data Analysis:
        - Anomaly Score: {prediction['score']:.4f} (Negative = Anomaly, Positive = Normal)
        - Is Anomaly: {prediction['is_anomaly']}
        - Input Data: {json.dumps(prediction['input_data'], indent=2)}

        Context & Constraints:
        - There is a mandatory cooldown period of 1 month every 6 months. Mention this in the context of operations planning.

        Role-Specific Focus:
        - If Engineer: Focus on technical details, sensor readings (Pressure/Temperature), and quality of maintenance. Use technical terminology.
        - If Manager: Focus on monetary impact, potential profit/loss due to downtime or efficiency, and high-level operational status. Use business terminology.

        Visualizations:
        - Create a visual representation of the anomaly score (e.g., a gauge or bar chart) using SVG or CSS.
        - Create a simple bar chart comparing the input values (normalized if possible or just raw) using SVG/CSS.

        Output:
        - Return ONLY the full HTML code. Start with <!DOCTYPE html>.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        html_content = self._call_groq(messages)

        # Clean up markdown code blocks if present
        if "```html" in html_content:
            html_content = html_content.split("```html")[1].split("```")[0]
        elif "```" in html_content:
            html_content = html_content.split("```")[1].split("```")[0]

        html_content = html_content.strip()

        # Generate PDF
        try:
            HTML(string=html_content).write_pdf(output_path)
        except Exception as e:
            # Fallback if HTML is malformed or WeasyPrint fails
            error_html = f"<html><body><h1>Error generating PDF</h1><p>{e}</p><pre>{html_content}</pre></body></html>"
            HTML(string=error_html).write_pdf(output_path)

        return output_path
