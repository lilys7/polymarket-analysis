"""export a compact point cloud for the three.js web viewer."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_engineering import SCALED_FEATURE_COLS, SPATIAL_FEATURE_COLS
from src.spatial_model import NOISE_LABEL
from src.visualize import TOP_CLUSTER_K

logger = logging.getLogger(__name__)

LABEL_TO_CODE = {"awful": 0, "bad": 1, "good": 2, "sharp": 3, "unlabeled": 4}
CODE_TO_LABEL = {v: k for k, v in LABEL_TO_CODE.items()}

# tableau-like colors for the 12 largest manifolds
TOP_COLORS = [
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ac",
    "#86bcb6",
    "#d37295",
]


def export_web_pointcloud(
    df: pd.DataFrame,
    out_dir: str | Path,
    top_k: int = TOP_CLUSTER_K,
) -> Path:
    """write web/data/meta.json + points.bin for the three.js viewer."""
    required = (
        *SPATIAL_FEATURE_COLS,
        *SCALED_FEATURE_COLS,
        "spatial_cluster",
        "trader_label",
        "trader_pnl",
        "trader_ppv",
        "hhi",
        "outlier_score",
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns {missing}; run steps 1-3 first")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    n = len(df)
    dense = df.loc[df["spatial_cluster"] != NOISE_LABEL, "spatial_cluster"]
    top_ids = [int(c) for c in dense.value_counts().head(top_k).index.tolist()]

    codes = (
        df["trader_label"]
        .astype("string")
        .str.lower()
        .map(LABEL_TO_CODE)
        .fillna(4)
        .to_numpy(dtype=np.uint8)
    )

    def f32(col: str) -> bytes:
        return np.asarray(df[col], dtype="<f4").tobytes()

    blob = b"".join(
        [
            f32(SCALED_FEATURE_COLS[0]),
            f32(SCALED_FEATURE_COLS[1]),
            f32(SCALED_FEATURE_COLS[2]),
            f32(SPATIAL_FEATURE_COLS[0]),
            f32(SPATIAL_FEATURE_COLS[1]),
            f32(SPATIAL_FEATURE_COLS[2]),
            f32("trader_pnl"),
            f32("trader_ppv"),
            f32("hhi"),
            f32("outlier_score"),
            np.asarray(df["spatial_cluster"], dtype="<i2").tobytes(),
            codes.tobytes(),
        ]
    )
    bin_path = out / "points.bin"
    bin_path.write_bytes(blob)

    meta = {
        "n": n,
        "endian": "little",
        "layout": [
            "zx",
            "zy",
            "zz",
            "x_raw",
            "y_raw",
            "z_raw",
            "pnl",
            "ppv",
            "hhi",
            "outlier",
            "cluster",
            "label",
        ],
        "axes": {
            "x": "Price book impact (z-scored)",
            "y": "Execution timing variance (z-scored)",
            "z": "Order size magnitude (z-scored)",
        },
        "raw_axes": {
            "x": SPATIAL_FEATURE_COLS[0],
            "y": SPATIAL_FEATURE_COLS[1],
            "z": SPATIAL_FEATURE_COLS[2],
        },
        "noise_label": NOISE_LABEL,
        "top_cluster_ids": top_ids,
        "top_colors": TOP_COLORS[: len(top_ids)],
        "noise_color": "#6b7280",
        "other_dense_color": "#8b5cf6",
        "label_colors": {
            "awful": "#e15759",
            "bad": "#f28e2b",
            "good": "#4e79a7",
            "sharp": "#59a14f",
            "unlabeled": "#9ca3af",
        },
        "label_names": CODE_TO_LABEL,
        "bytes": bin_path.stat().st_size,
    }
    meta_path = out / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("wrote %s (%s bytes) and %s", bin_path, meta["bytes"], meta_path)
    return bin_path
