"""
build.py — Single entry point for the full patient deterioration ML pipeline.
Run: python build.py
"""
import os
import sys
import yaml

# Ensure src is importable from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.logger import get_logger
from src.pipeline.ingestion import load_outcomes, load_all_patients
from src.pipeline.transformation import clean_raw_data
from src.pipeline.features import build_features
from src.pipeline.modeling import train_models
from src.pipeline.evaluation import evaluate_models
from src.pipeline.prediction import generate_predictions


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def load_config(path: str = "config.yaml") -> dict:
    with open(os.path.join(PROJECT_ROOT, path)) as f:
        cfg = yaml.safe_load(f)
    # Resolve all paths relative to project root
    cfg["data"]["raw_dir"] = os.path.normpath(os.path.join(PROJECT_ROOT, cfg["data"]["raw_dir"]))
    cfg["data"]["outcomes_file"] = os.path.normpath(os.path.join(PROJECT_ROOT, cfg["data"]["outcomes_file"]))
    cfg["data"]["features_output"] = os.path.join(PROJECT_ROOT, cfg["data"]["features_output"])
    cfg["model"]["output_dir"] = os.path.join(PROJECT_ROOT, cfg["model"]["output_dir"])
    cfg["evaluation"]["output_dir"] = os.path.join(PROJECT_ROOT, cfg["evaluation"]["output_dir"])
    cfg["prediction"]["output_file"] = os.path.join(PROJECT_ROOT, cfg["prediction"]["output_file"])
    cfg["logging"]["log_dir"] = os.path.join(PROJECT_ROOT, cfg["logging"]["log_dir"])
    return cfg


def run_pipeline():
    cfg = load_config()
    log_cfg = {
        "log_dir": cfg["logging"]["log_dir"],
        "log_file": cfg["logging"]["log_file"],
        "level": cfg["logging"]["level"],
    }
    logger = get_logger("build", **log_cfg)
    logger.info("=" * 60)
    logger.info("Starting patient deterioration pipeline")
    logger.info("=" * 60)

    # --- Stage 1: Ingestion ---
    logger.info("Stage 1: Ingestion")
    outcomes_df = load_outcomes(cfg["data"]["outcomes_file"], log_cfg)
    raw_df = load_all_patients(cfg["data"]["raw_dir"], log_cfg)

    # --- Stage 2: Transformation ---
    logger.info("Stage 2: Transformation")
    clean_df = clean_raw_data(raw_df, log_cfg)

    # --- Stage 3: Feature Engineering ---
    logger.info("Stage 3: Feature Engineering")
    features_df = build_features(clean_df, outcomes_df, cfg["features"], log_cfg)
    os.makedirs(os.path.dirname(cfg["data"]["features_output"]), exist_ok=True)
    features_df.to_parquet(cfg["data"]["features_output"], index=False)
    logger.info(f"Features saved to {cfg['data']['features_output']}")

    # --- Stage 4: Modeling ---
    logger.info("Stage 4: Modeling")
    trained = train_models(features_df, cfg["model"], cfg["tuning"], log_cfg)

    # --- Stage 5: Evaluation ---
    logger.info("Stage 5: Evaluation")
    results, best_model, best_thresholds = evaluate_models(trained, cfg["evaluation"], log_cfg)

    # --- Stage 6: Prediction ---
    logger.info("Stage 6: Prediction")
    generate_predictions(features_df, best_model, best_thresholds, cfg["prediction"], cfg["model"], log_cfg)

    logger.info("=" * 60)
    logger.info(f"Pipeline complete. Best model: {best_model}")
    logger.info("To launch dashboard: python src/dashboard/app.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()
