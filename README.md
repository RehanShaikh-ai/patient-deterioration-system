# Patient Deterioration Prediction System

An end-to-end, production-style machine learning pipeline for predicting in-hospital patient deterioration using the [PhysioNet Challenge 2012](https://physionet.org/content/challenge-2012/1.0.0/) dataset.

---

## Overview

This system ingests raw ICU time-series data, engineers meaningful patient-level features, trains and evaluates multiple ML models, generates risk predictions, and exposes results through a Flask dashboard — all orchestrated through a single entry point: `build.py`.

---

## Project Structure

```
amazon-q/
├── build.py                        # Single entry point — runs the full pipeline
├── config.yaml                     # All pipeline settings (no hardcoded values)
├── requirements.txt                # Python dependencies
├── .gitignore
│
├── src/
│   ├── logger.py                   # Shared structured logger
│   ├── pipeline/
│   │   ├── ingestion.py            # Load raw patient files and outcome labels
│   │   ├── transformation.py       # Parse time, clean data, remove sentinels
│   │   ├── features.py             # Build patient-level feature matrix
│   │   ├── modeling.py             # Train LR, RandomForest, XGBoost
│   │   ├── evaluation.py           # Evaluate models, save reports
│   │   └── prediction.py          # Generate risk scores and categories
│   └── dashboard/
│       └── app.py                  # Flask dashboard (reads from outputs only)
│
├── outputs/
│   ├── features/                   # features.parquet
│   ├── models/                     # Serialized .joblib model files
│   ├── evaluation/                 # Per-model classification reports + summary.json
│   └── predictions/                # predictions.csv with risk scores and categories
│
└── logs/
    └── pipeline.log                # Full pipeline execution log
```

---

## Dataset

**PhysioNet Challenge 2012 — Set A**

- 4,000 ICU patient records
- Each patient file contains time-stamped measurements in `(Time, Parameter, Value)` format
- Observations span up to 48 hours of ICU stay
- Outcome label: `In-hospital_death` (binary: 0 = survived, 1 = died)
- Class distribution: ~3,446 survived / 554 died (~13.9% positive rate)

**Raw data layout (not included in repo):**
```
set-a/
    132539.txt
    132540.txt
    ...
Outcomes-a.txt
```

Place these in the parent directory of `amazon-q/` before running.

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Or using conda (recommended — matches development environment):

```bash
conda activate torch
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python build.py
```

This executes all stages in sequence:
1. Ingestion
2. Transformation
3. Feature Engineering
4. Model Training
5. Evaluation
6. Prediction

### 3. Launch the dashboard

```bash
python src/dashboard/app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Pipeline Stages

### Stage 1 — Ingestion (`ingestion.py`)

- Reads all 4,000 `.txt` patient files from `set-a/`
- Parses the `(Time, Parameter, Value)` event structure
- Loads `Outcomes-a.txt` for labels
- Skips malformed files with a warning

### Stage 2 — Transformation (`transformation.py`)

- Converts `HH:MM` time strings to fractional hours
- Casts all values to numeric, drops unparseable rows
- Removes sentinel values (`-1`) used in the dataset to indicate missing data

### Stage 3 — Feature Engineering (`features.py`)

Builds a flat, patient-level feature matrix with two types of features:

**Static features** (recorded at admission, `t=0`):
- `Age`, `Gender`, `Height`, `Weight`, `ICUType`

**Dynamic features** (computed over the last `N` hours, configurable):

For each vital sign and lab parameter:
- `_mean` — average value
- `_max` — peak value
- `_min` — lowest value
- `_std` — variability
- `_trend` — linear slope (direction of change)

Vital signs: `HR`, `NIDiasABP`, `NIMAP`, `NISysABP`, `RespRate`, `Temp`, `SpO2`

Lab values: `BUN`, `Creatinine`, `Glucose`, `HCO3`, `HCT`, `K`, `Lactate`, `Mg`, `Na`, `Platelets`, `WBC`

Final feature matrix: **4,000 patients × 97 features**

### Stage 4 — Modeling (`modeling.py`)

Three models are trained, each wrapped in a `sklearn.Pipeline` with median imputation:

| Model | Imbalance Handling |
|---|---|
| Logistic Regression | `class_weight="balanced"` |
| Random Forest | `class_weight="balanced"` |
| XGBoost | `scale_pos_weight=6` |

- 80/20 stratified train/test split
- Models serialized to `outputs/models/` as `.joblib` files

### Stage 5 — Evaluation (`evaluation.py`)

Each model is evaluated on the held-out test set:

| Model | Precision | Recall | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.28 | **0.64** | 0.77 |
| Random Forest | 0.67 | 0.05 | 0.80 |
| XGBoost | 0.48 | 0.27 | **0.81** |

- Best model selected by **recall** (critical for high-risk patient identification)
- Per-model classification reports saved to `outputs/evaluation/`
- Aggregated metrics saved to `outputs/evaluation/summary.json`

### Stage 6 — Prediction (`prediction.py`)

Using the best model:
- Generates a `risk_score` (probability of in-hospital death) for every patient
- Assigns a `risk_category`:
  - `low` — score < 0.30
  - `medium` — score 0.30–0.60
  - `high` — score > 0.60
- Output saved to `outputs/predictions/predictions.csv`

---

## Configuration

All pipeline behavior is controlled via `config.yaml`:

```yaml
data:
  raw_dir: "../set-a"
  outcomes_file: "../Outcomes-a.txt"
  features_output: "outputs/features/features.parquet"

features:
  recent_hours: 24          # Use last 24h of ICU stay for dynamic features
  static_params: [...]
  vital_params: [...]
  lab_params: [...]

model:
  target: "In-hospital_death"
  test_size: 0.2
  random_state: 42
  models_to_train: ["logistic_regression", "random_forest", "xgboost"]
  best_model_metric: "recall"

prediction:
  risk_thresholds:
    low: 0.3
    medium: 0.6
```

---

## Dashboard

The Flask dashboard reads exclusively from `outputs/` — it does not touch raw data.

**Features:**
- Model performance table (precision, recall, ROC-AUC for all models)
- Risk distribution summary (low / medium / high counts)
- Top 50 high-risk patients table
- Patient lookup by `RecordID`

```bash
python src/dashboard/app.py
# → http://localhost:5000
```

---

## Logging

All pipeline stages log to both console and `logs/pipeline.log`:

```
2026-04-12 16:30:04 | INFO | ingestion    | Loaded 4000 outcome records
2026-04-12 16:30:23 | INFO | ingestion    | Loaded 4000 patient files, 1757980 total rows
2026-04-12 16:32:13 | INFO | features     | Feature matrix shape: (4000, 97)
2026-04-12 16:32:17 | INFO | evaluation   | Best model by recall: logistic_regression (0.6396)
2026-04-12 16:32:17 | INFO | prediction   | Risk distribution: {'medium': 1638, 'low': 1545, 'high': 817}
```

---

## Reproducibility

- `random_state` is set globally in `config.yaml`
- Pipeline is idempotent — safe to re-run, outputs are overwritten cleanly
- All paths are resolved relative to `build.py` location, not the working directory

---

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn, xgboost, flask, pyyaml, pyarrow, joblib

See `requirements.txt` for pinned versions.

---

## Git History

```
7edda11  fix: resolve path resolution and suppress polyfit warnings
3e449d6  feat(build): add build.py pipeline orchestrator as single entry point
9a45997  feat(dashboard): add Flask dashboard with metrics, risk distribution, and patient lookup
ce4c4df  feat(evaluation): add model evaluation with precision/recall/AUC and best model selection
5b5d686  feat(modeling): add LR, RandomForest, XGBoost training with class imbalance handling
4072d9b  feat(features): add patient-level feature engineering with static/dynamic/trend features
38c1f17  feat(transformation): add time parsing and data cleaning module
959d3eb  feat(ingestion): add logger utility and raw data ingestion module
d1d633c  chore: initial project structure with config and dependencies
```
