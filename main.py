"""run the polymarket 3d spatial clustering pipeline end-to-end."""

from __future__ import annotations

import argparse
import gc
import logging
import resource
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.data_loader import load_and_clean_data
from src.feature_engineering import engineer_features
from src.spatial_model import NOISE_LABEL, SpatialPointModel, summarize_clusters
from src.web_export import export_web_pointcloud

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "data.parquet"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("pipeline")


def _peak_rss_mb() -> float:
    """peak resident set size in mb (mac reports bytes; linux reports kb)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def _sharp_share(df: pd.DataFrame) -> pd.DataFrame:
    label = df["trader_label"].astype("string").str.lower()
    is_sharp = label == "sharp"
    is_noise = df["spatial_cluster"] == NOISE_LABEL
    return pd.DataFrame(
        [
            {
                "pool": "dense manifolds (cluster >= 0)",
                "n_traders": int((~is_noise).sum()),
                "pct_sharp": float(is_sharp[~is_noise].mean() * 100.0),
            },
            {
                "pool": "spatial noise (cluster = -1)",
                "n_traders": int(is_noise.sum()),
                "pct_sharp": float(is_sharp[is_noise].mean() * 100.0),
            },
        ]
    )


def _print_results(
    df: pd.DataFrame,
    n_raw: int,
    timings: dict[str, float],
) -> None:
    n = len(df)
    n_noise = int((df["spatial_cluster"] == NOISE_LABEL).sum())
    n_dense = n - n_noise
    n_manifolds = int(df["spatial_cluster"].max() + 1) if n_dense else 0
    summary = summarize_clusters(df)
    sharp = _sharp_share(df)
    top = summary.sort_values("n_traders", ascending=False).head(8)

    print("\n" + "=" * 72)
    print("POLYMARKET 3D SPATIAL CLUSTERING — KEY RESULTS")
    print("=" * 72)
    print(f"raw rows:              {n_raw:,}")
    print(f"cleaned rows:          {n:,}  (dropped {n_raw - n:,})")
    print(f"dense manifolds:       {n_manifolds:,}")
    print(f"dense traders:         {n_dense:,}  ({100.0 * n_dense / n:.1f}%)")
    print(f"noise traders (-1):    {n_noise:,}  ({100.0 * n_noise / n:.1f}%)")
    print(f"median HHI:            {df['hhi'].median():.3f}")
    print(f"HHI == 1 (specialist): {100.0 * (df['hhi'] >= 0.999).mean():.1f}%")
    print()
    print("sharp share (full sample)")
    for _, row in sharp.iterrows():
        print(
            f"  {row['pool']}: {row['pct_sharp']:.1f}% "
            f"(n={int(row['n_traders']):,})"
        )
    print()
    print("largest clusters (including noise)")
    cols = [
        "spatial_cluster",
        "n_traders",
        "mean_trader_ppv",
        "median_trader_pnl",
        "pct_sharp",
        "pct_awful",
        "pct_sharp_vs_noise",
    ]
    print(top[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print("timings")
    for name, sec in timings.items():
        print(f"  {name:<22} {sec:6.2f}s")
    print(f"  {'total':<22} {sum(timings.values()):6.2f}s")
    print(f"  peak rss              {_peak_rss_mb():.0f} mb")
    print("=" * 72)


def run(data_path: Path = DEFAULT_DATA, export_web: bool = False) -> pd.DataFrame:
    """execute steps 1-3 and print empirical results."""
    timings: dict[str, float] = {}
    n_raw = int(pq.ParquetFile(data_path).metadata.num_rows)

    t0 = time.perf_counter()
    df = load_and_clean_data(str(data_path))
    timings["step 1 load/clean"] = time.perf_counter() - t0
    logger.info(
        "step 1 load/clean finished in %.2fs (peak rss %.0f mb)",
        timings["step 1 load/clean"],
        _peak_rss_mb(),
    )
    gc.collect()

    t0 = time.perf_counter()
    df, X_scaled = engineer_features(df)
    timings["step 2 features"] = time.perf_counter() - t0
    logger.info(
        "step 2 features finished in %.2fs (peak rss %.0f mb)",
        timings["step 2 features"],
        _peak_rss_mb(),
    )
    gc.collect()

    t0 = time.perf_counter()
    model = SpatialPointModel(min_cluster_size=50, min_samples=15)
    df = model.fit_attach(df, X_scaled)
    timings["step 3 hdbscan"] = time.perf_counter() - t0
    logger.info(
        "step 3 hdbscan finished in %.2fs (peak rss %.0f mb)",
        timings["step 3 hdbscan"],
        _peak_rss_mb(),
    )
    # clustering array is now on df; drop the extra (n, 3) copy
    del X_scaled, model
    gc.collect()

    if export_web:
        t0 = time.perf_counter()
        bin_path = export_web_pointcloud(df, ROOT / "web" / "data")
        timings["step 4 web export"] = time.perf_counter() - t0
        logger.info(
            "step 4 web export finished in %.2fs -> %s",
            timings["step 4 web export"],
            bin_path,
        )
        print(
            "\nweb viewer: cd web && python3 -m http.server 8000\n"
            "then open http://localhost:8000"
        )

    _print_results(df, n_raw=n_raw, timings=timings)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="polymarket 3d spatial clustering pipeline")
    parser.add_argument(
        "data_path",
        nargs="?",
        default=str(DEFAULT_DATA),
        help="path to trader parquet",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="export web/data point cloud for the three.js viewer",
    )
    args = parser.parse_args()
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"parquet not found: {data_path}")
    run(data_path, export_web=args.web)


if __name__ == "__main__":
    main()
