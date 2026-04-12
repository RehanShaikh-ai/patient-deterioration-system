import os
import pandas as pd
from src.logger import get_logger


def load_outcomes(outcomes_path: str, log_cfg: dict) -> pd.DataFrame:
    logger = get_logger("ingestion", **log_cfg)
    logger.info(f"Loading outcomes from {outcomes_path}")
    df = pd.read_csv(outcomes_path)
    logger.info(f"Loaded {len(df)} outcome records")
    return df


def load_patient_file(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    record_id = int(df.loc[df["Parameter"] == "RecordID", "Value"].iloc[0])
    df["RecordID"] = record_id
    return df


def load_all_patients(raw_dir: str, log_cfg: dict) -> pd.DataFrame:
    logger = get_logger("ingestion", **log_cfg)
    logger.info(f"Loading patient files from {raw_dir}")
    files = [f for f in os.listdir(raw_dir) if f.endswith(".txt")]
    frames = []
    for fname in files:
        try:
            frames.append(load_patient_file(os.path.join(raw_dir, fname)))
        except Exception as e:
            logger.warning(f"Skipping {fname}: {e}")
    df = pd.concat(frames, ignore_index=True)
    logger.info(f"Loaded {len(files)} patient files, {len(df)} total rows")
    return df
