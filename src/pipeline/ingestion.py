import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.logger import get_logger


def load_outcomes(outcomes_path: str, log_cfg: dict) -> pd.DataFrame:
    logger = get_logger("ingestion", **log_cfg)
    logger.info(f"Loading outcomes from {outcomes_path}")
    df = pd.read_csv(outcomes_path)
    logger.info(f"Loaded {len(df)} outcome records")
    return df


def _load_file(filepath: str):
    try:
        df = pd.read_csv(filepath)
        record_id = int(df.loc[df["Parameter"] == "RecordID", "Value"].iloc[0])
        df["RecordID"] = record_id
        return df
    except Exception:
        return None


def load_all_patients(raw_dir: str, log_cfg: dict, n_workers: int = 8) -> pd.DataFrame:
    logger = get_logger("ingestion", **log_cfg)
    logger.info(f"Loading patient files from {raw_dir} (parallel, workers={n_workers})")
    files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".txt")]

    frames = []
    skipped = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_load_file, fp): fp for fp in files}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                frames.append(result)
            else:
                skipped += 1

    if skipped:
        logger.warning(f"Skipped {skipped} files due to errors")
    df = pd.concat(frames, ignore_index=True)
    logger.info(f"Loaded {len(files) - skipped} patient files, {len(df)} total rows")
    return df
