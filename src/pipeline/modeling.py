import os
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


def _build_model(name: str, random_state: int):
    if name == "logistic_regression":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state))
        ])
    elif name == "random_forest":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=random_state, n_jobs=-1))
        ])
    elif name == "xgboost":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", XGBClassifier(n_estimators=200, scale_pos_weight=6, random_state=random_state,
                                   eval_metric="logloss", verbosity=0))
        ])
    else:
        raise ValueError(f"Unknown model: {name}")


def train_models(features_df: pd.DataFrame, cfg: dict, log_cfg: dict) -> dict:
    logger = get_logger("modeling", **log_cfg)
    target = cfg["target"]
    test_size = cfg["test_size"]
    random_state = cfg["random_state"]
    models_to_train = cfg["models_to_train"]
    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    feature_cols = [c for c in features_df.columns if c not in ["RecordID", target]]
    X = features_df[feature_cols].values
    y = features_df[target].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    trained = {}
    for name in models_to_train:
        logger.info(f"Training {name}")
        model = _build_model(name, random_state)
        model.fit(X_train, y_train)
        path = os.path.join(output_dir, f"{name}.joblib")
        joblib.dump(model, path)
        logger.info(f"Saved {name} to {path}")
        trained[name] = {"model": model, "X_test": X_test, "y_test": y_test, "feature_cols": feature_cols}

    return trained
