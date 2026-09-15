"""load and clean polymarket trader microstructure parquet data."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ID_COLS: tuple[str, ...] = ("trader", "trader_label")
PERF_COLS: tuple[str, ...] = ("trader_pnl", "trader_volume", "trader_ppv")
COORD_COLS: tuple[str, ...] = (
    "price_levels_consumed_vw",
    "std_time_vw",
    "mean_tx_value",
)
REQUIRED_COLS: tuple[str, ...] = ID_COLS + PERF_COLS + COORD_COLS
TOPIC_PREFIX = "topic_"


def get_topic_columns(df: pd.DataFrame) -> list[str]:
    """return category-share columns (names starting with topic_)."""
    return [col for col in df.columns if col.startswith(TOPIC_PREFIX)]


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """load a trader-level parquet file and return a clustering-ready frame.

    reads parquet with pyarrow, fills unlabeled wallets, drops rows that cannot
    sit in the 3d coordinate space or that lack performance metrics, then adds
    log10 transforms of heavy-tailed size features.

    Args:
        file_path: path to the parquet file (e.g. data/data.parquet).

    Returns:
        cleaned dataframe with original columns plus log_mean_tx_value and
        log_volume. row index is reset.

    Raises:
        FileNotFoundError: if file_path does not exist.
        ValueError: if required columns are missing from the file.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"parquet file not found: {path}")
    if path.suffix.lower() != ".parquet":
        raise ValueError(f"expected a .parquet file, got: {path.suffix}")

    df = pd.read_parquet(path, engine="pyarrow")
    n_raw = len(df)
    logger.info("loaded %s rows x %s cols from %s", n_raw, df.shape[1], path)

    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    topic_cols = get_topic_columns(df)
    if not topic_cols:
        raise ValueError("no topic_* category-share columns found")

    df = df.copy()
    df["trader_label"] = (
        df["trader_label"].astype("string").fillna("Unlabeled")
    )

    # inf is unusable as a 3d coordinate; treat it as missing
    invalid_cols = list(COORD_COLS + PERF_COLS)
    df[invalid_cols] = df[invalid_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=invalid_cols)

    df["log_mean_tx_value"] = np.log10(df["mean_tx_value"].to_numpy() + 1.0)
    df["log_volume"] = np.log10(df["trader_volume"].to_numpy() + 1.0)

    df = df.reset_index(drop=True)
    n_dropped = n_raw - len(df)
    logger.info(
        "dropped %s rows with invalid coords/performance; %s remain",
        n_dropped,
        len(df),
    )
    return df
