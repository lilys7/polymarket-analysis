"""publication figures for the 3d spatial clustering pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib import pyplot as plt
from plotly.graph_objects import Figure

from src.spatial_model import NOISE_LABEL

TOP_CLUSTER_K = 12


def _require_cols(df: pd.DataFrame, cols: tuple[str, ...]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns {missing}; run steps 1-3 first")


def sample_for_3d_plot(
    df: pd.DataFrame,
    top_k: int = TOP_CLUSTER_K,
    per_cluster: int = 350,
    n_other: int = 600,
    n_noise: int = 2000,
    random_state: int = 42,
) -> pd.DataFrame:
    """stratified subsample so the 3d figure stays interactive.

    600k points would freeze a browser and overplot. keep the largest
    manifolds, a slice of remaining dense points, and a capped noise set.
    """
    _require_cols(df, ("spatial_cluster", "trader_label"))
    noise = df[df["spatial_cluster"] == NOISE_LABEL]
    dense = df[df["spatial_cluster"] != NOISE_LABEL]
    top_ids = dense["spatial_cluster"].value_counts().head(top_k).index

    parts: list[pd.DataFrame] = []
    for cid in top_ids:
        g = dense[dense["spatial_cluster"] == cid]
        parts.append(g.sample(n=min(per_cluster, len(g)), random_state=random_state))

    other = dense[~dense["spatial_cluster"].isin(top_ids)]
    if len(other):
        parts.append(
            other.sample(n=min(n_other, len(other)), random_state=random_state)
        )
    if len(noise):
        parts.append(
            noise.sample(n=min(n_noise, len(noise)), random_state=random_state)
        )
    return pd.concat(parts, ignore_index=True)


def _cluster_color_label(cluster_id: int, top_ids: set[int]) -> str:
    if cluster_id == NOISE_LABEL:
        return "noise"
    if cluster_id in top_ids:
        return str(int(cluster_id))
    return "other dense"


def build_3d_scatter(df: pd.DataFrame, top_k: int = TOP_CLUSTER_K) -> Figure:
    """interactive 3d scatter of execution style, colored by cluster."""
    _require_cols(
        df,
        (
            "price_levels_consumed_vw",
            "std_time_vw",
            "log_mean_tx_value",
            "spatial_cluster",
            "trader_label",
            "trader_pnl",
            "trader_ppv",
            "hhi",
        ),
    )
    plot_df = df.copy()
    top_ids = set(
        plot_df.loc[plot_df["spatial_cluster"] != NOISE_LABEL, "spatial_cluster"]
        .value_counts()
        .head(top_k)
        .index.tolist()
    )
    plot_df["cluster_id"] = plot_df["spatial_cluster"].astype(int)
    plot_df["spatial_cluster"] = [
        _cluster_color_label(int(cid), top_ids)
        for cid in plot_df["cluster_id"]
    ]
    # alias for the paper's hover name
    plot_df["hhi_specialization"] = plot_df["hhi"]
    plot_df["trader_label"] = (
        plot_df["trader_label"].astype("string").str.capitalize()
    )

    color_order = ["noise", "other dense"] + [
        str(int(c))
        for c in df.loc[df["spatial_cluster"] != NOISE_LABEL, "spatial_cluster"]
        .value_counts()
        .head(top_k)
        .index
    ]

    fig = px.scatter_3d(
        plot_df,
        x="price_levels_consumed_vw",
        y="std_time_vw",
        z="log_mean_tx_value",
        color="spatial_cluster",
        symbol="trader_label",
        hover_data={
            "cluster_id": True,
            "trader_pnl": ":.2f",
            "trader_ppv": ":.4f",
            "hhi_specialization": ":.3f",
        },
        category_orders={
            "spatial_cluster": color_order,
            "trader_label": ["Sharp", "Good", "Bad", "Awful"],
        },
        labels={
            "price_levels_consumed_vw": "Price Book Impact",
            "std_time_vw": "Execution Timing Variance",
            "log_mean_tx_value": "Order Size Magnitude",
            "spatial_cluster": "Spatial cluster",
            "trader_label": "Trader label",
        },
        title="3D spatial execution manifolds (HDBSCAN)",
    )
    fig.update_traces(marker={"size": 3, "opacity": 0.75})
    fig.update_layout(
        width=980,
        height=720,
        legend_title_text="Cluster / label",
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
        scene={
            "xaxis_title": "Price Book Impact",
            "yaxis_title": "Execution Timing Variance",
            "zaxis_title": "Order Size Magnitude",
        },
    )
    return fig


def export_plotly_png(fig: Figure, path: str | Path, scale: int = 3) -> Path:
    """write a high-resolution png via kaleido for the pdf report."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out), format="png", scale=scale, width=1400, height=1000)
    return out


def sharp_share_dense_vs_noise(df: pd.DataFrame) -> pd.DataFrame:
    """percent of sharp traders in all dense manifolds vs the noise pool."""
    _require_cols(df, ("spatial_cluster", "trader_label"))
    label = df["trader_label"].astype("string").str.lower()
    is_sharp = label == "sharp"
    is_noise = df["spatial_cluster"] == NOISE_LABEL
    rows = [
        {
            "pool": "Dense manifolds (cluster ≥ 0)",
            "n_traders": int((~is_noise).sum()),
            "pct_sharp": float(is_sharp[~is_noise].mean() * 100.0),
        },
        {
            "pool": "Spatial noise (cluster = −1)",
            "n_traders": int(is_noise.sum()),
            "pct_sharp": float(is_sharp[is_noise].mean() * 100.0),
        },
    ]
    return pd.DataFrame(rows)


def plot_sharp_dense_vs_noise(
    df: pd.DataFrame,
    png_path: str | Path | None = None,
) -> pd.DataFrame:
    """seaborn bar chart of sharp share: dense clusters vs noise."""
    stats = sharp_share_dense_vs_noise(df)
    sns.set_theme(style="whitegrid", font_scale=1.05)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    sns.barplot(
        data=stats,
        x="pool",
        y="pct_sharp",
        color="#4a4a4a",
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Share of traders labeled Sharp (%)")
    ax.set_title("Sharp concentration: dense manifolds vs spatial noise")
    ax.set_ylim(0, max(stats["pct_sharp"].max() * 1.25, 1))
    for i, row in stats.iterrows():
        ax.text(
            i,
            row["pct_sharp"] + 0.6,
            f"{row['pct_sharp']:.1f}%\n(n={row['n_traders']:,})",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    if png_path is not None:
        out = Path(png_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
    return stats
