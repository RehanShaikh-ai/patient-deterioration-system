import pandas as pd
import numpy as np
from src.logger import get_logger


def clean_raw_data(raw_df: pd.DataFrame, log_cfg: dict) -> pd.DataFrame:
    logger = get_logger("transformation", **log_cfg)
    logger.info("Parsing time column and cleaning raw data")

    df = raw_df.copy()

    # Vectorised HH:MM → fractional hours (100x faster than .apply)
    time_parts = df["Time"].str.split(":", expand=True)
    df["hours"] = pd.to_numeric(time_parts[0], errors="coerce") + \
                  pd.to_numeric(time_parts[1], errors="coerce") / 60.0

    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["hours", "Value"])
    logger.info(f"Dropped {before - len(df)} rows with missing time/value")

    df = df[df["Value"] >= 0]
    logger.info(f"Remaining rows after cleaning: {len(df)}")
    return df
