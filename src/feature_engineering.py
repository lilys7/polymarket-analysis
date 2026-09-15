"""build clustering features from the cleaned trader-level frame."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data_loader import get_topic_columns

# 3d axes: book-walking, time-of-day dispersion, log trade size
SPATIAL_FEATURE_COLS: tuple[str, ...] = (
    "price_levels_consumed_vw",
    "std_time_vw",
    "log_mean_tx_value",
)
SCALED_FEATURE_COLS: tuple[str, ...] = (
    "z_price_levels_consumed_vw",
    "z_std_time_vw",
    "z_log_mean_tx_value",
)


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """add hhi and a z-scored 3d feature matrix for density clustering.

    expects the output of load_and_clean_data (must include log_mean_tx_value
    and topic_* share columns).

    Args:
        df: cleaned trader-level dataframe.

    Returns:
        out: copy of df with hhi plus z-scored spatial columns.
        X_scaled: ndarray of shape (n_traders, 3), column order matches
            SPATIAL_FEATURE_COLS, each column mean 0 and std 1.
    """
    missing = [col for col in SPATIAL_FEATURE_COLS if col not in df.columns]
    if missing:
        raise ValueError(
            f"missing spatial features {missing}; run load_and_clean_data first"
        )

    topic_cols = get_topic_columns(df)
    if not topic_cols:
        raise ValueError("no topic_* columns found; cannot compute hhi")

    out = df.copy()

    # hhi_i = sum_c s_{i,c}^2  (s = topic share). square so mass on one
    # topic -> 1 (specialist) and an even split -> 1/c (generalist).
    shares = out[topic_cols].to_numpy(dtype=float)
    out["hhi"] = np.square(shares).sum(axis=1)

    # raw 3d matrix: mixed units (levels, ms, log-dollars)
    X = out.loc[:, list(SPATIAL_FEATURE_COLS)].to_numpy(dtype=float)

    # z-score: x' = (x - mu) / sigma so euclidean distance is unit-free
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for name, col in zip(SCALED_FEATURE_COLS, X_scaled.T):
        out[name] = col

    return out, X_scaled
