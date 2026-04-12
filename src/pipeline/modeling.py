import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from src.logger import get_logger


def _build_model(name: str, random_state: int, scale_pos_weight: float):
    if name == "logistic_regression":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, C=0.1, random_state=random_state))
        ])
    elif name == "random_forest":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=300, class_weight="balanced",
                max_depth=8, min_samples_leaf=5,
                random_state=random_state, n_jobs=-1
            ))
        ])
    elif name == "xgboost":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", XGBClassifier(
                n_estimators=300, scale_pos_weight=scale_pos_weight,
                max_depth=5, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, min_child_weight=3,
                random_state=random_state, eval_metric="aucpr", verbosity=0
            ))
        ])
    else:
        raise ValueError(f"Unknown model: {name}")


def train_models(features_df: pd.DataFrame, cfg: dict, log_cfg: dict) -> dict:
    logger = get_logger("modeling", **log_cfg)
    target        = cfg["target"]
    test_size     = cfg["test_size"]
    random_state  = cfg["random_state"]
    models_to_train = cfg["models_to_train"]
    output_dir    = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    feature_cols = [c for c in features_df.columns if c not in ["RecordID", target]]
    X = features_df[feature_cols].values
    y = features_df[target].values

    neg, pos = np.bincount(y)
    scale_pos_weight = round(neg / pos, 2)
    logger.info(f"Class counts — negative: {neg}, positive: {pos}, scale_pos_weight: {scale_pos_weight}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    trained = {}
    for name in models_to_train:
        logger.info(f"Training {name}")
        model = _build_model(name, random_state, scale_pos_weight)
        model.fit(X_train, y_train)

        model_path = os.path.join(output_dir, f"{name}.joblib")
        meta_path  = os.path.join(output_dir, f"{name}_meta.json")
        joblib.dump(model, model_path)
        with open(meta_path, "w") as f:
            json.dump({"feature_cols": feature_cols}, f)

        logger.info(f"Saved {name} → {model_path}")
        trained[name] = {
            "model": model, "X_test": X_test, "y_test": y_test,
            "feature_cols": feature_cols
        }

    return trained
