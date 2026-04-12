import os
import joblib
import numpy as np
import pandas as pd
from src.logger import get_logger


def generate_predictions(features_df: pd.DataFrame, best_model: str, cfg: dict, model_cfg: dict, log_cfg: dict) -> pd.DataFrame:
    logger = get_logger("prediction", **log_cfg)
    target = model_cfg["target"]
    thresholds = cfg["risk_thresholds"]
    model_path = os.path.join(model_cfg["output_dir"], f"{best_model}.joblib")

    logger.info(f"Loading best model: {best_model} from {model_path}")
    model = joblib.load(model_path)

    feature_cols = [c for c in features_df.columns if c not in ["RecordID", target]]
    X = features_df[feature_cols].values
    probs = model.predict_proba(X)[:, 1]

    def categorize(p):
        if p < thresholds["low"]:
            return "low"
        elif p < thresholds["medium"]:
            return "medium"
        return "high"

    preds = pd.DataFrame({
        "RecordID": features_df["RecordID"].values,
        "risk_score": np.round(probs, 4),
        "risk_category": [categorize(p) for p in probs],
        "actual": features_df[target].values if target in features_df.columns else np.nan,
    })

    os.makedirs(os.path.dirname(cfg["output_file"]), exist_ok=True)
    preds.to_csv(cfg["output_file"], index=False)
    logger.info(f"Predictions saved to {cfg['output_file']}")
    logger.info(f"Risk distribution:\n{preds['risk_category'].value_counts().to_dict()}")
    return preds
