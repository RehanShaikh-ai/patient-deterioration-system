import pandas as pd
import numpy as np
from src.logger import get_logger


def _extract_static(group: pd.DataFrame, static_params: list) -> dict:
    static = {}
    t0 = group[group["hours"] == 0]
    for param in static_params:
        vals = t0.loc[t0["Parameter"] == param, "Value"]
        static[param] = vals.iloc[0] if len(vals) > 0 else np.nan
    return static


def _extract_dynamic(group: pd.DataFrame, params: list, recent_hours: int) -> dict:
    max_time = group["hours"].max()
    cutoff = max(0, max_time - recent_hours)
    recent = group[group["hours"] >= cutoff]
    features = {}
    for param in params:
        vals = recent.loc[recent["Parameter"] == param, "Value"].dropna()
        if len(vals) == 0:
            features[f"{param}_mean"] = np.nan
            features[f"{param}_max"] = np.nan
            features[f"{param}_min"] = np.nan
            features[f"{param}_std"] = np.nan
            features[f"{param}_trend"] = np.nan
        else:
            features[f"{param}_mean"] = vals.mean()
            features[f"{param}_max"] = vals.max()
            features[f"{param}_min"] = vals.min()
            features[f"{param}_std"] = vals.std() if len(vals) > 1 else 0.0
            if len(vals) > 1:
                times = recent.loc[recent["Parameter"] == param, "hours"].values
                features[f"{param}_trend"] = float(np.polyfit(times, vals.values, 1)[0])
            else:
                features[f"{param}_trend"] = 0.0
    return features


def build_features(clean_df: pd.DataFrame, outcomes_df: pd.DataFrame, cfg: dict, log_cfg: dict) -> pd.DataFrame:
    logger = get_logger("features", **log_cfg)
    logger.info("Building patient-level features")

    static_params = cfg["static_params"]
    vital_params = cfg["vital_params"]
    lab_params = cfg["lab_params"]
    recent_hours = cfg["recent_hours"]
    target = "In-hospital_death"

    outcomes = outcomes_df[["RecordID", target]].set_index("RecordID")
    records = []

    grouped = clean_df.groupby("RecordID")
    logger.info(f"Processing {len(grouped)} patients")

    for record_id, group in grouped:
        row = {"RecordID": record_id}
        row.update(_extract_static(group, static_params))
        row.update(_extract_dynamic(group, vital_params + lab_params, recent_hours))
        if record_id in outcomes.index:
            row[target] = outcomes.loc[record_id, target]
        else:
            row[target] = np.nan
        records.append(row)

    features_df = pd.DataFrame(records)
    features_df = features_df.dropna(subset=[target])
    features_df[target] = features_df[target].astype(int)

    logger.info(f"Feature matrix shape: {features_df.shape}")
    logger.info(f"Class distribution:\n{features_df[target].value_counts().to_dict()}")
    return features_df
