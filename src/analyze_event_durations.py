"""Analyze event duration distributions from detection manifests.

This CLI computes descriptive statistics (mean, median, percentiles)
and produces plots (histograms and boxplots) to help choose duration
ranges for generating "Nothing" fragments.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from extract_fragments import DEFAULT_CSV_PATH

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("docs/figures")


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze duration distributions from a detection CSV."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Path to the detection CSV (expects onset_s/offset_s/label columns).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store plots and the summary CSV.",
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=None,
        help=(
            "Optional path to write the summary statistics CSV. "
            "Defaults to <output-dir>/duration_stats.csv"
        ),
    )
    parser.add_argument(
        "--include-labels",
        nargs="+",
        default=None,
        help="If provided, only analyze these labels.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=None,
        help="Discard events shorter than this duration (seconds).",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Discard events longer than this duration (seconds).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Number of bins to use in histograms.",
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=0.9,
        help=(
            "Fraction of events to cover when computing the central duration interval "
            "(e.g., 0.9 => central 90% range)."
        ),
    )
    return parser.parse_args(args=args)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def validate_columns(df: pd.DataFrame, csv_path: Path) -> None:
    required = {"onset_s", "offset_s", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV {csv_path} is missing required columns: {sorted(missing)}")


def compute_durations(df: pd.DataFrame) -> pd.DataFrame:
    durations = df.copy()
    durations["duration_s"] = durations["offset_s"] - durations["onset_s"]
    return durations


def apply_filters(
    df: pd.DataFrame,
    include_labels: Optional[Iterable[str]],
    min_duration: Optional[float],
    max_duration: Optional[float],
) -> pd.DataFrame:
    filtered = df
    if include_labels:
        filtered = filtered[filtered["label"].isin(include_labels)]
    if min_duration is not None:
        filtered = filtered[filtered["duration_s"] >= min_duration]
    if max_duration is not None:
        filtered = filtered[filtered["duration_s"] <= max_duration]
    return filtered.reset_index(drop=True)


def describe_durations(df: pd.DataFrame, coverage: float) -> pd.DataFrame:
    lower_q = (1 - coverage) / 2 * 100
    upper_q = 100 - lower_q

    def _summary(series: pd.Series) -> dict:
        values = series.to_numpy()
        return {
            "count": int(len(values)),
            "mean": float(np.mean(values)) if len(values) else np.nan,
            "median": float(np.median(values)) if len(values) else np.nan,
            "p05": float(np.percentile(values, 5)) if len(values) else np.nan,
            "p95": float(np.percentile(values, 95)) if len(values) else np.nan,
            "min": float(np.min(values)) if len(values) else np.nan,
            "max": float(np.max(values)) if len(values) else np.nan,
            "central_lower_s": float(np.percentile(values, lower_q))
            if len(values)
            else np.nan,
            "central_upper_s": float(np.percentile(values, upper_q))
            if len(values)
            else np.nan,
        }

    records = []
    overall = _summary(df["duration_s"])
    overall["label"] = "ALL"
    records.append(overall)

    for label, group in df.groupby("label"):
        summary = _summary(group["duration_s"])
        summary["label"] = label
        records.append(summary)

    return pd.DataFrame(records)


def plot_histograms(
    df: pd.DataFrame, output_dir: Path, bins: int, stats: pd.DataFrame
) -> Path:
    labels = ["ALL"] + sorted(df["label"].unique())
    n_plots = len(labels)
    fig, axes = plt.subplots(
        n_plots,
        1,
        figsize=(8, 3 * n_plots),
        constrained_layout=True,
        sharex=True,
    )
    if n_plots == 1:
        axes = [axes]

    all_data = df["duration_s"].to_numpy()
    overall_interval = stats.loc[stats["label"] == "ALL", [
        "central_lower_s",
        "central_upper_s",
    ]].iloc[0]
    axes[0].hist(all_data, bins=bins, color="steelblue", edgecolor="black", alpha=0.8)
    axes[0].set_title("Distribuição de durações (todas as labels)")
    axes[0].set_ylabel("Contagem")
    axes[0].axvline(overall_interval["central_lower_s"], color="red", linestyle="--")
    axes[0].axvline(overall_interval["central_upper_s"], color="red", linestyle="--")

    label_intervals = {
        row.label: (row.central_lower_s, row.central_upper_s)
        for row in stats.itertuples()
    }
    for ax, label in zip(axes[1:], labels[1:]):
        label_data = df.loc[df["label"] == label, "duration_s"].to_numpy()
        ax.hist(label_data, bins=bins, color="orange", edgecolor="black", alpha=0.8)
        ax.set_title(f"Distribuição de durações — {label}")
        ax.set_ylabel("Contagem")
        lower, upper = label_intervals.get(label, (np.nan, np.nan))
        if not np.isnan(lower):
            ax.axvline(lower, color="red", linestyle="--")
        if not np.isnan(upper):
            ax.axvline(upper, color="red", linestyle="--")

    axes[-1].set_xlabel("Duração (s)")
    path = output_dir / "duration_histograms.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_boxplot(df: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = sorted(df["label"].unique())
    data = [df.loc[df["label"] == label, "duration_s"].to_numpy() for label in labels]
    ax.boxplot(data, labels=labels, vert=False, patch_artist=True)
    ax.set_xlabel("Duração (s)")
    ax.set_title("Boxplot de durações por label")
    path = output_dir / "duration_boxplots.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.bins <= 0:
        raise ValueError("--bins must be positive")
    if not 0 < args.coverage < 1:
        raise ValueError("--coverage must be in the interval (0, 1)")

    ensure_output_dir(args.output_dir)
    stats_path = args.stats_path or args.output_dir / "duration_stats.csv"

    df = pd.read_csv(args.csv_path)
    validate_columns(df, args.csv_path)

    df = compute_durations(df)
    df = apply_filters(df, args.include_labels, args.min_duration, args.max_duration)
    if df.empty:
        raise ValueError("No events remain after applying filters.")

    stats = describe_durations(df, args.coverage)
    stats.to_csv(stats_path, index=False)
    logger.info("Saved summary statistics to %s", stats_path)

    hist_path = plot_histograms(df, args.output_dir, args.bins, stats)
    box_path = plot_boxplot(df, args.output_dir)
    logger.info("Saved histogram to %s", hist_path)
    logger.info("Saved boxplot to %s", box_path)

    overall = stats.loc[stats["label"] == "ALL"].iloc[0]
    logger.info(
        "Overall durations — count=%d, mean=%.3f, median=%.3f, p05=%.3f, p95=%.3f",
        overall["count"],
        overall["mean"],
        overall["median"],
        overall["p05"],
        overall["p95"],
    )
    logger.info(
        "Central %.0f%% interval: [%.3f, %.3f] s",
        args.coverage * 100,
        overall["central_lower_s"],
        overall["central_upper_s"],
    )


if __name__ == "__main__":
    main()
