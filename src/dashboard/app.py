import os
import io
import json
import random
import pandas as pd
import numpy as np
import yaml
import sys
from flask import Flask, jsonify, render_template, request

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PREDS_FILE = os.path.join(BASE_DIR, "outputs", "predictions", "predictions.csv")
FEATS_FILE = os.path.join(BASE_DIR, "outputs", "features",    "features.parquet")
SUMMARY_FILE = os.path.join(BASE_DIR, "outputs", "evaluation", "summary.json")

app = Flask(__name__)

FIRST_NAMES = ["James","Maria","Robert","Linda","Michael","Barbara","William","Patricia",
               "David","Jennifer","Richard","Susan","Joseph","Jessica","Thomas","Sarah",
               "Charles","Karen","Christopher","Lisa","Daniel","Nancy","Matthew","Betty",
               "Anthony","Margaret","Mark","Sandra","Donald","Ashley"]
LAST_NAMES  = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis",
               "Wilson","Taylor","Anderson","Thomas","Jackson","White","Harris","Martin",
               "Thompson","Robinson","Clark","Lewis","Walker","Hall","Allen","Young","King"]

random.seed(42)

def _build_patients():
    preds = pd.read_csv(PREDS_FILE)
    feats = pd.read_parquet(FEATS_FILE)
    df = preds.merge(feats[["RecordID","Age","Gender",
                             "HR_mean","HR_last","HR_trend",
                             "NIMAP_mean","RespRate_mean","Temp_mean",
                             "SpO2_mean","Lactate_mean"]], on="RecordID", how="left")

    rows = []
    for _, r in df.iterrows():
        rid   = int(r["RecordID"])
        rng   = random.Random(rid)
        name  = rng.choice(FIRST_NAMES) + " " + rng.choice(LAST_NAMES)
        age   = int(r["Age"]) if pd.notna(r["Age"]) else rng.randint(40, 85)
        gender = "Female" if r.get("Gender", 0) == 1 else "Male"
        xgb_prob = float(r["xgb_prob"]) if pd.notna(r.get("xgb_prob")) else 0.0
        lr_prob  = float(r["lr_prob"]) if pd.notna(r.get("lr_prob")) else 0.0

        if lr_prob >= 0.30 or xgb_prob >= 0.20:
            risk = "high"
        elif lr_prob >= 0.15 or xgb_prob >= 0.10:
            risk = "medium"
        else:
            risk = "low"
            
        score = float(max(xgb_prob, lr_prob))

        rows.append({
            "id":       rid,
            "name":     name,
            "age":      age,
            "gender":   gender,
            "risk":     risk,
            "score":    score,
            "actual":   int(r["actual"]) if pd.notna(r.get("actual")) else 0,
            "hr":       round(r["HR_mean"],  1) if pd.notna(r.get("HR_mean"))       else None,
            "hr_last":  round(r["HR_last"],  1) if pd.notna(r.get("HR_last"))       else None,
            "hr_trend": round(r["HR_trend"], 3) if pd.notna(r.get("HR_trend"))      else None,
            "map":      round(r["NIMAP_mean"],   1) if pd.notna(r.get("NIMAP_mean"))    else None,
            "rr":       round(r["RespRate_mean"],1) if pd.notna(r.get("RespRate_mean")) else None,
            "temp":     round(r["Temp_mean"],    1) if pd.notna(r.get("Temp_mean"))     else None,
            "spo2":     round(r["SpO2_mean"],    1) if pd.notna(r.get("SpO2_mean"))     else None,
            "lactate":  round(r["Lactate_mean"], 2) if pd.notna(r.get("Lactate_mean"))  else None,
        })

    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("hospital.html")

@app.route("/patient")
@app.route("/patient/<int:pid>")
def patient_view(pid=None):
    return render_template("dashboard.html", initial_pid=pid)

@app.route("/api/patients")
def api_patients():
    return jsonify(_build_patients())

@app.route("/api/patient/<int:pid>")
def api_patient(pid):
    patients = _build_patients()
    p = next((x for x in patients if x["id"] == pid), None)
    if not p:
        return jsonify({"error": "Not found"}), 404
    return jsonify(p)

@app.route("/api/patient/<int:pid>/timeseries")
def api_patient_ts(pid):
    parent_dir = os.path.dirname(BASE_DIR)
    for subdir in ["set-a", "set-b"]:
        path = os.path.join(parent_dir, subdir, f"{pid}.txt")
        if os.path.exists(path):
            df = pd.read_csv(path)
            out = {}
            for param, grp in df.groupby("Parameter"):
                try:
                    # Filter out static demographic rows at 00:00
                    vals = [{"x": row["Time"], "y": float(row["Value"])} for _, row in grp.iterrows() if row["Time"] != "00:00"]
                    if vals:
                        out[param] = vals
                except ValueError:
                    continue
            return jsonify(out)
    return jsonify({"error": "Time series not found"}), 404

@app.route("/api/summary")
def api_summary():
    patients = _build_patients()
    high   = sum(1 for p in patients if p["risk"] == "high")
    medium = sum(1 for p in patients if p["risk"] == "medium")
    low    = sum(1 for p in patients if p["risk"] == "low")
    survived     = sum(1 for p in patients if p["actual"] == 0)
    deteriorated = sum(1 for p in patients if p["actual"] == 1)
    with open(SUMMARY_FILE) as f:
        eval_data = json.load(f)
    return jsonify({
        "total": len(patients), "high": high, "medium": medium, "low": low,
        "survived": survived, "deteriorated": deteriorated,
        "eval": eval_data
    })

@app.route("/avatar/<filename>")
def api_avatar(filename):
    import os
    from flask import send_file
    path = os.path.join(r"C:\Users\rehan\.gemini\antigravity\brain\80657cb1-a69b-4a8b-a180-e600ec0b8a06", filename)
    return send_file(path)

@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "GET":
        return render_template("upload.html")

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        df = pd.read_csv(io.StringIO(file.stream.read().decode("utf-8")))

        if not all(col in df.columns for col in ["Time", "Parameter", "Value"]):
            return jsonify({"error": "CSV must contain columns: Time, Parameter, Value"}), 400

        def parse_hours(t):
            try:
                hh, mm = str(t).split(":")
                return int(hh) + int(mm)/60.0
            except:
                return 0.0

        df["hours"] = df["Time"].apply(parse_hours)
        
        import re
        match = re.search(r'\d+', file.filename)
        record_id = int(match.group()) if match else random.randint(100000, 999999)
        df["RecordID"] = record_id

        with open(os.path.join(BASE_DIR, "config.yaml")) as f:
            cfg = yaml.safe_load(f)

        static_params = cfg["features"]["static_params"]
        dynamic_params = cfg["features"]["vital_params"] + cfg["features"]["lab_params"]
        recent_hours = cfg["features"]["recent_hours"]
        early_hours = cfg["features"]["early_hours"]

        sys.path.append(BASE_DIR)
        from src.pipeline.features import _extract_static, _extract_dynamic, _extract_interactions

        row = {"RecordID": record_id}
        row.update(_extract_static(df, static_params))
        row.update(_extract_dynamic(df, dynamic_params, recent_hours, early_hours))
        row.update(_extract_interactions(row))

        model_dir = os.path.join(BASE_DIR, "outputs", "models")
        with open(os.path.join(model_dir, "xgboost_meta.json")) as meta_f:
            f_cols = json.load(meta_f)["feature_cols"]

        X = np.array([[row.get(c, np.nan) for c in f_cols]])

        import joblib
        xgb_model = joblib.load(os.path.join(model_dir, "xgboost.joblib"))
        lr_model  = joblib.load(os.path.join(model_dir, "logistic_regression.joblib"))

        xgb_prob = float(xgb_model.predict_proba(X)[:, 1][0])
        lr_prob  = float(lr_model.predict_proba(X)[:, 1][0])
        score = max(xgb_prob, lr_prob)

        xgb_t = cfg["evaluation"]["ensemble"]["xgb_threshold"]
        lr_t  = cfg["evaluation"]["ensemble"]["lr_threshold"]

        if score >= 0.95:
            display_score = "95%+"
        else:
            display_score = f"{int(score * 100)}%"

        risk_level = "HIGH" if (xgb_prob >= xgb_t or lr_prob >= lr_t) else "LOW"
        if not risk_level == "HIGH" and (xgb_prob >= xgb_t-0.1 or lr_prob >= lr_t-0.15):
             risk_level = "MEDIUM"
        
        return jsonify({
            "status": "success",
            "patient_id": record_id,
            "risk": risk_level,
            "score": display_score,
            "raw_score": score,
            "records": len(df),
            "hours": round(df["hours"].max(), 1) if not df.empty else 0,
            "hr_mean": round(row.get("HR_mean", 0) if pd.notna(row.get("HR_mean")) else 0, 1),
            "map_trend": round(row.get("NIMAP_trend", 0) if pd.notna(row.get("NIMAP_trend")) else 0, 3),
            "temp_max": round(row.get("Temp_max", 0) if pd.notna(row.get("Temp_max")) else 0, 1),
            "rr_mean": round(row.get("RespRate_mean", 0) if pd.notna(row.get("RespRate_mean")) else 0, 1),
            "age": row.get("Age", "Unknown") if pd.notna(row.get("Age")) else "Unknown"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
