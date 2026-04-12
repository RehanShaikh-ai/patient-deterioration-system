import json
import os
import pandas as pd
from flask import Flask, jsonify, render_template_string

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PREDICTIONS_FILE = os.path.join(BASE_DIR, "outputs", "predictions", "predictions.csv")
SUMMARY_FILE = os.path.join(BASE_DIR, "outputs", "evaluation", "summary.json")

HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Patient Deterioration Dashboard</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 30px; background: #f4f6f9; }
    h1 { color: #2c3e50; }
    .card { background: white; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 6px rgba(0,0,0,0.1); }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; }
    th { background: #2c3e50; color: white; }
    tr:nth-child(even) { background: #f9f9f9; }
    .high { color: #e74c3c; font-weight: bold; }
    .medium { color: #e67e22; font-weight: bold; }
    .low { color: #27ae60; font-weight: bold; }
    input { padding: 6px; margin-right: 8px; border: 1px solid #ccc; border-radius: 4px; }
    button { padding: 6px 14px; background: #2c3e50; color: white; border: none; border-radius: 4px; cursor: pointer; }
  </style>
</head>
<body>
  <h1>🏥 Patient Deterioration Risk Dashboard</h1>

  <div class="card">
    <h2>Model Performance</h2>
    <table id="metrics-table">
      <thead><tr><th>Model</th><th>Precision</th><th>Recall</th><th>ROC-AUC</th></tr></thead>
      <tbody id="metrics-body"></tbody>
    </table>
  </div>

  <div class="card">
    <h2>Risk Distribution</h2>
    <div id="risk-dist"></div>
  </div>

  <div class="card">
    <h2>Patient Lookup</h2>
    <input type="text" id="patient-id" placeholder="Enter RecordID" />
    <button onclick="lookupPatient()">Search</button>
    <div id="patient-result" style="margin-top:12px;"></div>
  </div>

  <div class="card">
    <h2>High Risk Patients</h2>
    <table>
      <thead><tr><th>RecordID</th><th>Risk Score</th><th>Risk Category</th><th>Actual</th></tr></thead>
      <tbody id="high-risk-body"></tbody>
    </table>
  </div>

  <script>
    async function loadMetrics() {
      const res = await fetch('/api/metrics');
      const data = await res.json();
      const tbody = document.getElementById('metrics-body');
      for (const [model, m] of Object.entries(data)) {
        tbody.innerHTML += `<tr><td>${model}</td><td>${m.precision}</td><td>${m.recall}</td><td>${m.roc_auc}</td></tr>`;
      }
    }

    async function loadRiskDist() {
      const res = await fetch('/api/risk_distribution');
      const data = await res.json();
      const div = document.getElementById('risk-dist');
      div.innerHTML = Object.entries(data).map(([k,v]) =>
        `<span class="${k}" style="margin-right:20px;">${k.toUpperCase()}: ${v}</span>`
      ).join('');
    }

    async function loadHighRisk() {
      const res = await fetch('/api/high_risk');
      const data = await res.json();
      const tbody = document.getElementById('high-risk-body');
      data.forEach(p => {
        tbody.innerHTML += `<tr><td>${p.RecordID}</td><td>${p.risk_score}</td>
          <td class="${p.risk_category}">${p.risk_category}</td><td>${p.actual}</td></tr>`;
      });
    }

    async function lookupPatient() {
      const id = document.getElementById('patient-id').value;
      const res = await fetch(`/api/patient/${id}`);
      const data = await res.json();
      const div = document.getElementById('patient-result');
      if (data.error) {
        div.innerHTML = `<span style="color:red;">${data.error}</span>`;
      } else {
        div.innerHTML = `<b>RecordID:</b> ${data.RecordID} &nbsp;
          <b>Risk Score:</b> ${data.risk_score} &nbsp;
          <b>Category:</b> <span class="${data.risk_category}">${data.risk_category}</span> &nbsp;
          <b>Actual:</b> ${data.actual}`;
      }
    }

    loadMetrics();
    loadRiskDist();
    loadHighRisk();
  </script>
</body>
</html>
"""


def _load_predictions():
    return pd.read_csv(PREDICTIONS_FILE)


def _load_summary():
    with open(SUMMARY_FILE) as f:
        return json.load(f)


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/metrics")
def metrics():
    return jsonify(_load_summary())


@app.route("/api/risk_distribution")
def risk_distribution():
    df = _load_predictions()
    return jsonify(df["risk_category"].value_counts().to_dict())


@app.route("/api/high_risk")
def high_risk():
    df = _load_predictions()
    top = df[df["risk_category"] == "high"].sort_values("risk_score", ascending=False).head(50)
    return jsonify(top.to_dict(orient="records"))


@app.route("/api/patient/<int:record_id>")
def patient(record_id):
    df = _load_predictions()
    row = df[df["RecordID"] == record_id]
    if row.empty:
        return jsonify({"error": f"Patient {record_id} not found"})
    return jsonify(row.iloc[0].to_dict())


if __name__ == "__main__":
    app.run(debug=True, port=5000)
