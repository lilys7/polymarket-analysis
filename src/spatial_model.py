"""hdbscan density clustering on the z-scored 3d trader manifold."""

from __future__ import annotations

import logging

import hdbscan
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NOISE_LABEL = -1


class SpatialPointModel:
    """fit hdbscan on a scaled 3d array and attach labels plus glosh scores.

    min_cluster_size is the smallest dense region we will call a manifold.
    min_samples is how many neighbors a point needs to be a core point.
    """

    def __init__(
        self,
        min_cluster_size: int = 50,
        min_samples: int = 15,
    ) -> None:
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        # eom = excess of mass; default splitter for stable dense components
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
            core_dist_n_jobs=1,
        )
        self.labels_: np.ndarray | None = None
        self.outlier_score_: np.ndarray | None = None

    def fit(self, X_scaled: np.ndarray) -> SpatialPointModel:
        """run hdbscan on x_scaled of shape (n_traders, 3)."""
        X = np.asarray(X_scaled, dtype=float)
        if X.ndim != 2 or X.shape[1] != 3:
            raise ValueError(f"expected X_scaled with shape (n, 3), got {X.shape}")
        if not np.isfinite(X).all():
            raise ValueError("X_scaled contains nan/inf; rerun steps 1-2")

        logger.info(
            "fitting hdbscan on %s points (min_cluster_size=%s, min_samples=%s)",
            X.shape[0],
            self.min_cluster_size,
            self.min_samples,
        )
        self.clusterer.fit(X)
        self.labels_ = self.clusterer.labels_.astype(np.int32)
        # glosh: 0 = core of a dense region, 1 = locally sparse / outlier
        glosh = np.asarray(self.clusterer.outlier_scores_, dtype=float)
        glosh = np.nan_to_num(glosh, nan=1.0, posinf=1.0, neginf=0.0)
        self.outlier_score_ = glosh

        n_noise = int((self.labels_ == NOISE_LABEL).sum())
        n_clusters = int(self.labels_.max() + 1) if self.labels_.size else 0
        logger.info(
            "found %s manifolds; %s noise points (%.1f%%)",
            n_clusters,
            n_noise,
            100.0 * n_noise / max(len(self.labels_), 1),
        )
        return self

    def attach(self, df: pd.DataFrame) -> pd.DataFrame:
        """add spatial_cluster and outlier_score, aligned by row order."""
        if self.labels_ is None or self.outlier_score_ is None:
            raise RuntimeError("call fit(X_scaled) before attach(df)")
        if len(df) != len(self.labels_):
            raise ValueError(
                f"row count mismatch: df has {len(df)}, model has {len(self.labels_)}"
            )

        out = df.copy()
        # -1 = unstructured spatial noise; >= 0 = dense execution manifold
        out["spatial_cluster"] = self.labels_
        out["outlier_score"] = self.outlier_score_
        return out

    def fit_attach(self, df: pd.DataFrame, X_scaled: np.ndarray) -> pd.DataFrame:
        """fit on x_scaled then return df with cluster columns."""
        self.fit(X_scaled)
        return self.attach(df)


def summarize_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """per-cluster counts, profit stats, and sharp/awful mix vs noise.

    cluster -1 is the unstructured noise pool. pct_sharp_vs_noise and
    pct_awful_vs_noise are percentage-point gaps vs that pool.
    """
    if "spatial_cluster" not in df.columns:
        raise ValueError("df has no spatial_cluster; fit SpatialPointModel first")

    labels = df["trader_label"].astype("string").str.lower()
    work = df.assign(
        is_sharp=(labels == "sharp").astype(float),
        is_awful=(labels == "awful").astype(float),
    )

    grouped = work.groupby("spatial_cluster", sort=True)
    summary = grouped.agg(
        n_traders=("spatial_cluster", "size"),
        mean_trader_ppv=("trader_ppv", "mean"),
        median_trader_pnl=("trader_pnl", "median"),
        pct_sharp=("is_sharp", "mean"),
        pct_awful=("is_awful", "mean"),
    )
    summary["pct_sharp"] *= 100.0
    summary["pct_awful"] *= 100.0

    if NOISE_LABEL not in summary.index:
        raise ValueError("no noise cluster (-1); cannot compare to the noise pool")

    noise_sharp = summary.loc[NOISE_LABEL, "pct_sharp"]
    noise_awful = summary.loc[NOISE_LABEL, "pct_awful"]
    # positive vs_noise = more of that label than the unstructured pool
    summary["pct_sharp_vs_noise"] = summary["pct_sharp"] - noise_sharp
    summary["pct_awful_vs_noise"] = summary["pct_awful"] - noise_awful
    summary["is_noise"] = summary.index == NOISE_LABEL
    return summary.reset_index()
