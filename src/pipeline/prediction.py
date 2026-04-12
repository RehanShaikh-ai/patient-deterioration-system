import os
import joblib
import numpy as np
import pandas as pd
from src.logger import get_logger


def _load_model(model_dir: str, name: str):
    return joblib.load(os.path.join(model_dir, f"{name}.joblib"))


def _ensemble_risk(lr_prob: float, xgb_prob: float, lr_t: float, xgb_t: float) -> str:
    """
    Three-level decision rule:
      HIGH   — LR fires (safety override, high recall)
      MEDIUM — LR silent but XGBoost fires (precision-guided alert)
      LOW    — both below threshold
    """
    if lr_prob >= lr_t:
        return "high"
    if xgb_prob >= xgb_t:
        return "medium"
    return "low"


def generate_predictions(
    features_df: pd.DataFrame,
    cfg: dict,
    model_cfg: dict,
    log_cfg: dict,
) -> pd.DataFrame:
    logger = get_logger("prediction", **log_cfg)
    target      = model_cfg["target"]
    model_dir   = model_cfg["output_dir"]
    ensemble    = cfg["ensemble"]
    lr_t        = ensemble["logistic_regression_threshold"]
    xgb_t       = ensemble["xgboost_threshold"]

    logger.info(f"Ensemble prediction — LR threshold: {lr_t}, XGBoost threshold: {xgb_t}")

    lr_model  = _load_model(model_dir, "logistic_regression")
    xgb_model = _load_model(model_dir, "xgboost")

    feature_cols = [c for c in features_df.columns if c not in ["RecordID", target]]
    X = features_df[feature_cols].values

    lr_probs  = lr_model.predict_proba(X)[:, 1]
    xgb_probs = xgb_model.predict_proba(X)[:, 1]

    risk_categories = [
        _ensemble_risk(lr, xgb, lr_t, xgb_t)
        for lr, xgb in zip(lr_probs, xgb_probs)
    ]

    preds = pd.DataFrame({
        "RecordID":        features_df["RecordID"].values,
        "lr_prob":         np.round(lr_probs, 4),
        "xgb_prob":        np.round(xgb_probs, 4),
        "risk_category":   risk_categories,
        "actual":          features_df[target].values if target in features_df.columns else np.nan,
    })

    dist = preds["risk_category"].value_counts().to_dict()
    total = len(preds)
    logger.info(f"Risk distribution: {dist}")
    logger.info(
        f"  HIGH   (LR≥{lr_t}):              {dist.get('high', 0):4d} / {total}  "
        f"({100 * dist.get('high', 0) / total:.1f}%)"
    )
    logger.info(
        f"  MEDIUM (XGB≥{xgb_t}, LR<{lr_t}): {dist.get('medium', 0):4d} / {total}  "
        f"({100 * dist.get('medium', 0) / total:.1f}%)"
    )
    logger.info(
        f"  LOW    (both below threshold):   {dist.get('low', 0):4d} / {total}  "
        f"({100 * dist.get('low', 0) / total:.1f}%)"
    )

    # Evaluate ensemble against ground truth if available
    if target in features_df.columns:
        actual = features_df[target].values
        predicted_positive = (preds["risk_category"] != "low").astype(int)
        tp = int(((predicted_positive == 1) & (actual == 1)).sum())
        fp = int(((predicted_positive == 1) & (actual == 0)).sum())
        fn = int(((predicted_positive == 0) & (actual == 1)).sum())
        tn = int(((predicted_positive == 0) & (actual == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        alert_rate = (tp + fp) / total
        logger.info("Ensemble evaluation (high+medium = positive):")
        logger.info(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
        logger.info(f"  Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}  AlertRate={alert_rate:.4f}")

    os.makedirs(os.path.dirname(cfg["output_file"]), exist_ok=True)
    preds.to_csv(cfg["output_file"], index=False)
    logger.info(f"Predictions saved → {cfg['output_file']}")
    return preds
