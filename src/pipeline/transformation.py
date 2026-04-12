import pandas as pd
import numpy as np
from src.logger import get_logger


def parse_time_to_hours(time_str: str) -> float:
    try:
        h, m = str(time_str).split(":")
        return int(h) + int(m) / 60.0
    except Exception:
        return np.nan


def clean_raw_data(raw_df: pd.DataFrame, log_cfg: dict) -> pd.DataFrame:
    logger = get_logger("transformation", **log_cfg)
    logger.info("Parsing time column and cleaning raw data")

    df = raw_df.copy()
    df["hours"] = df["Time"].apply(parse_time_to_hours)

    # Drop descriptor rows (RecordID, Age, etc. at time 00:00 are handled in features)
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    # Remove rows with missing values or invalid time
    before = len(df)
    df = df.dropna(subset=["hours", "Value"])
    logger.info(f"Dropped {before - len(df)} rows with missing time/value")

    # Remove physiologically impossible sentinel values (-1 used as missing in dataset)
    df = df[df["Value"] >= 0]
    logger.info(f"Remaining rows after cleaning: {len(df)}")
    return df
