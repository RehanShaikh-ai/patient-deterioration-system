import os
import joblib
import numpy as np
import pandas as pd
from src.logger import get_logger


def generate_predictions(
    features_df: pd.DataFrame,
    best_model: str,
    best_thresholds: dict,
    cfg: dict,
    model_cfg: dict,
    log_cfg: dict,
) -> pd.DataFrame:
    logger = get_logger("prediction", **log_cfg)
    target    = model_cfg["target"]
    model_dir = model_cfg["output_dir"]
    xgb_t     = cfg["ensemble"]["xgb_threshold"]
    lr_t      = cfg["ensemble"]["lr_threshold"]

    logger.info(f"Final prediction — XGB≥{xgb_t} OR LR≥{lr_t} → HIGH RISK")

    xgb_model = joblib.load(os.path.join(model_dir, "xgboost.joblib"))
    lr_model  = joblib.load(os.path.join(model_dir, "logistic_regression.joblib"))

    feature_cols = [c for c in features_df.columns if c not in ["RecordID", target]]
    X = features_df[feature_cols].values

    xgb_probs = xgb_model.predict_proba(X)[:, 1]
    lr_probs  = lr_model.predict_proba(X)[:, 1]

    risk = np.where((lr_probs >= lr_t) | (xgb_probs >= xgb_t), "high", "low")

    preds = pd.DataFrame({
        "RecordID":      features_df["RecordID"].values,
        "xgb_prob":      np.round(xgb_probs, 4),
        "lr_prob":       np.round(lr_probs, 4),
        "risk_category": risk,
        "actual":        features_df[target].values if target in features_df.columns else np.nan,
    })

    dist = preds["risk_category"].value_counts().to_dict()
    total = len(preds)
    logger.info(f"Risk distribution: {dist}")
    logger.info(f"  HIGH : {dist.get('high', 0):4d} / {total}  ({100 * dist.get('high', 0) / total:.1f}%)")
    logger.info(f"  LOW  : {dist.get('low', 0):4d} / {total}  ({100 * dist.get('low', 0) / total:.1f}%)")

    os.makedirs(os.path.dirname(cfg["output_file"]), exist_ok=True)
    preds.to_csv(cfg["output_file"], index=False)
    logger.info(f"Predictions saved → {cfg['output_file']}")
    return preds
