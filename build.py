"""
build.py — Single entry point for the full patient deterioration ML pipeline.
Run: python build.py

Caching:
  - Raw data cached to outputs/cache/raw_data.parquet  (skip ingestion+transform on re-run)
  - Features cached to outputs/features/features.parquet (skip feature engineering on re-run)
  - Models cached to outputs/models/*.joblib             (skip training on re-run)
  Use --force to ignore all caches and rerun from scratch.
"""
import os
import sys
import yaml
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.logger import get_logger
from src.pipeline.ingestion import load_outcomes, load_all_patients
from src.pipeline.transformation import clean_raw_data
from src.pipeline.features import build_features
from src.pipeline.modeling import train_models
from src.pipeline.evaluation import evaluate_models
from src.pipeline.prediction import generate_predictions

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def load_config(path: str = "config.yaml") -> dict:
    with open(os.path.join(PROJECT_ROOT, path)) as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["raw_dir"]          = os.path.normpath(os.path.join(PROJECT_ROOT, cfg["data"]["raw_dir"]))
    cfg["data"]["outcomes_file"]    = os.path.normpath(os.path.join(PROJECT_ROOT, cfg["data"]["outcomes_file"]))
    cfg["data"]["raw_cache"]        = os.path.join(PROJECT_ROOT, cfg["data"]["raw_cache"])
    cfg["data"]["features_output"]  = os.path.join(PROJECT_ROOT, cfg["data"]["features_output"])
    cfg["model"]["output_dir"]      = os.path.join(PROJECT_ROOT, cfg["model"]["output_dir"])
    cfg["evaluation"]["output_dir"] = os.path.join(PROJECT_ROOT, cfg["evaluation"]["output_dir"])
    cfg["prediction"]["output_file"]= os.path.join(PROJECT_ROOT, cfg["prediction"]["output_file"])
    cfg["logging"]["log_dir"]       = os.path.join(PROJECT_ROOT, cfg["logging"]["log_dir"])
    return cfg


def _models_exist(model_dir: str, model_names: list) -> bool:
    return all(os.path.exists(os.path.join(model_dir, f"{n}.joblib")) for n in model_names)


def run_pipeline(force: bool = False):
    cfg = load_config()
    log_cfg = {k: cfg["logging"][k] for k in ("log_dir", "log_file", "level")}
    logger = get_logger("build", **log_cfg)

    logger.info("=" * 60)
    logger.info("Starting patient deterioration pipeline")
    logger.info(f"Force rerun: {force}")
    logger.info("=" * 60)

    outcomes_df = load_outcomes(cfg["data"]["outcomes_file"], log_cfg)

    # ── Stage 1+2: Ingestion + Transformation (cached) ───────────────────────
    raw_cache = cfg["data"]["raw_cache"]
    if not force and os.path.exists(raw_cache):
        logger.info(f"[CACHE HIT] Loading raw data from {raw_cache}")
        clean_df = pd.read_parquet(raw_cache)
    else:
        logger.info("Stage 1: Ingestion")
        raw_df = load_all_patients(cfg["data"]["raw_dir"], log_cfg)
        logger.info("Stage 2: Transformation")
        clean_df = clean_raw_data(raw_df, log_cfg)
        os.makedirs(os.path.dirname(raw_cache), exist_ok=True)
        clean_df.to_parquet(raw_cache, index=False)
        logger.info(f"Raw data cached → {raw_cache}")

    # ── Stage 3: Feature Engineering (cached) ────────────────────────────────
    features_path = cfg["data"]["features_output"]
    if not force and os.path.exists(features_path):
        logger.info(f"[CACHE HIT] Loading features from {features_path}")
        features_df = pd.read_parquet(features_path)
    else:
        logger.info("Stage 3: Feature Engineering")
        features_df = build_features(clean_df, outcomes_df, cfg["features"], log_cfg)
        os.makedirs(os.path.dirname(features_path), exist_ok=True)
        features_df.to_parquet(features_path, index=False)
        logger.info(f"Features cached → {features_path}")

    # ── Stage 4: Modeling (cached) ────────────────────────────────────────────
    model_names = cfg["model"]["models_to_train"]
    if not force and _models_exist(cfg["model"]["output_dir"], model_names):
        logger.info("[CACHE HIT] Models already exist, loading from disk")
        import joblib, json
        trained = {}
        model_dir = cfg["model"]["output_dir"]
        target = cfg["model"]["target"]
        from sklearn.model_selection import train_test_split
        import numpy as np
        feature_cols = [c for c in features_df.columns if c not in ["RecordID", target]]
        X = features_df[feature_cols].values
        y = features_df[target].values
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=cfg["model"]["test_size"],
            random_state=cfg["model"]["random_state"], stratify=y
        )
        for name in model_names:
            model = joblib.load(os.path.join(model_dir, f"{name}.joblib"))
            trained[name] = {"model": model, "X_test": X_test, "y_test": y_test, "feature_cols": feature_cols}
        logger.info(f"Loaded {len(model_names)} cached models")
    else:
        logger.info("Stage 4: Modeling")
        trained = train_models(features_df, cfg["model"], log_cfg)

    # ── Stage 5: Evaluation ───────────────────────────────────────────────────
    logger.info("Stage 5: Evaluation")
    results, best_model, best_thresholds = evaluate_models(trained, cfg["evaluation"], log_cfg)

    # ── Stage 6: Prediction ───────────────────────────────────────────────────
    logger.info("Stage 6: Prediction")
    generate_predictions(features_df, best_model, best_thresholds, cfg["prediction"], cfg["model"], log_cfg)

    logger.info("=" * 60)
    logger.info(f"Pipeline complete. Best model: {best_model}")
    logger.info("To launch dashboard: python src/dashboard/app.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Ignore all caches and rerun from scratch")
    args = parser.parse_args()
    run_pipeline(force=args.force)
