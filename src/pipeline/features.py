import warnings
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


def _safe_trend(times, values) -> float:
    if len(values) < 2:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(np.polyfit(times, values, 1)[0])


def _extract_dynamic(group: pd.DataFrame, params: list, recent_hours: int, early_hours: int) -> dict:
    max_time = group["hours"].max()
    recent_cutoff = max(0, max_time - recent_hours)
    early_cutoff = min(early_hours, max_time)

    recent = group[group["hours"] >= recent_cutoff]
    early = group[group["hours"] <= early_cutoff]

    features = {}
    for param in params:
        # --- recent window ---
        r_vals = recent.loc[recent["Parameter"] == param, "Value"].dropna()
        r_times = recent.loc[recent["Parameter"] == param, "hours"].values

        # --- full history ---
        all_rows = group[group["Parameter"] == param]
        all_vals = all_rows["Value"].dropna()
        all_times = all_rows["hours"].values

        # --- early window ---
        e_vals = early.loc[early["Parameter"] == param, "Value"].dropna()

        if len(r_vals) == 0:
            features[f"{param}_mean"]       = np.nan
            features[f"{param}_max"]        = np.nan
            features[f"{param}_min"]        = np.nan
            features[f"{param}_std"]        = np.nan
            features[f"{param}_last"]       = np.nan
            features[f"{param}_trend"]      = np.nan
            features[f"{param}_count"]      = 0
        else:
            features[f"{param}_mean"]  = r_vals.mean()
            features[f"{param}_max"]   = r_vals.max()
            features[f"{param}_min"]   = r_vals.min()
            features[f"{param}_std"]   = r_vals.std() if len(r_vals) > 1 else 0.0
            features[f"{param}_last"]  = r_vals.iloc[-1]
            features[f"{param}_trend"] = _safe_trend(r_times, r_vals.values)
            features[f"{param}_count"] = len(r_vals)

        # delta: last recent value minus first early value (deterioration signal)
        if len(r_vals) > 0 and len(e_vals) > 0:
            features[f"{param}_delta"] = float(r_vals.iloc[-1] - e_vals.iloc[0])
        elif len(all_vals) > 1:
            features[f"{param}_delta"] = float(all_vals.iloc[-1] - all_vals.iloc[0])
        else:
            features[f"{param}_delta"] = np.nan

        # range over full history
        features[f"{param}_range"] = float(all_vals.max() - all_vals.min()) if len(all_vals) > 1 else np.nan

    return features


def _extract_interactions(row: dict) -> dict:
    """Clinically motivated interaction features."""
    interactions = {}
    # Shock index: HR / SBP — elevated = poor perfusion
    hr = row.get("HR_mean", np.nan)
    sbp = row.get("NISysABP_mean", np.nan)
    interactions["shock_index"] = hr / sbp if sbp and sbp > 0 else np.nan

    # HR trend + RespRate trend combined deterioration signal
    hr_trend = row.get("HR_trend", np.nan)
    rr_trend = row.get("RespRate_trend", np.nan)
    interactions["hr_rr_trend_sum"] = (
        (hr_trend if not np.isnan(hr_trend) else 0) +
        (rr_trend if not np.isnan(rr_trend) else 0)
    )

    # BUN/Creatinine ratio — renal stress
    bun = row.get("BUN_mean", np.nan)
    cr = row.get("Creatinine_mean", np.nan)
    interactions["bun_cr_ratio"] = bun / cr if cr and cr > 0 else np.nan

    # Lactate last value — direct deterioration marker
    interactions["lactate_last"] = row.get("Lactate_last", np.nan)

    return interactions


def build_features(clean_df: pd.DataFrame, outcomes_df: pd.DataFrame, cfg: dict, log_cfg: dict) -> pd.DataFrame:
    logger = get_logger("features", **log_cfg)
    logger.info("Building patient-level features")

    static_params = cfg["static_params"]
    vital_params  = cfg["vital_params"]
    lab_params    = cfg["lab_params"]
    recent_hours  = cfg["recent_hours"]
    early_hours   = cfg.get("early_hours", 12)
    target        = "In-hospital_death"

    outcomes = outcomes_df[["RecordID", target]].set_index("RecordID")
    records = []

    grouped = clean_df.groupby("RecordID")
    logger.info(f"Processing {len(grouped)} patients")

    for record_id, group in grouped:
        row = {"RecordID": record_id}
        row.update(_extract_static(group, static_params))
        row.update(_extract_dynamic(group, vital_params + lab_params, recent_hours, early_hours))
        row.update(_extract_interactions(row))
        row[target] = outcomes.loc[record_id, target] if record_id in outcomes.index else np.nan
        records.append(row)

    features_df = pd.DataFrame(records)
    features_df = features_df.dropna(subset=[target])
    features_df[target] = features_df[target].astype(int)

    logger.info(f"Feature matrix shape: {features_df.shape}")
    logger.info(f"Class distribution: {features_df[target].value_counts().to_dict()}")
    return features_df
