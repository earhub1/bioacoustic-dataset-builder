"""Exploratory analysis for sequence manifests.

This CLI computes frame-level distributions per label and split from
`manifest_sequences.csv` and optionally uses `manifest_sequences_summary.csv`
for split-level totals. It saves CSV summaries and bar plots to help spot
imbalances across train/val/test.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("docs/figures")
DEFAULT_MANIFEST = Path("data/results/sequences/manifest_sequences.csv")
DEFAULT_SUMMARY = Path("data/results/sequences/manifest_sequences_summary.csv")


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate frame distribution plots from manifest_sequences.csv. "
            "Optionally uses manifest_sequences_summary.csv for split totals."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to manifest_sequences.csv (segment-level manifest).",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help=(
            "Optional path to manifest_sequences_summary.csv (per-sequence summary). "
            "If missing, split totals are derived from the segment manifest."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write CSVs and plots.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively in addition to saving them.",
    )
    return parser.parse_args(args=args)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"manifest_sequences.csv not found at {path}. Provide --manifest explicitly."
        )
    df = pd.read_csv(path)
    required = {"split", "label", "duration_frames"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"manifest {path} is missing required columns: {sorted(missing)}"
        )
    return df


def load_summary(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        logger.warning("Summary manifest not found at %s; using segment totals only.", path)
        return None

    df = pd.read_csv(path)
    required = {"split", "total_frames"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"summary manifest {path} is missing required columns: {sorted(missing)}"
        )
    return df


def compute_split_totals(
    manifest: pd.DataFrame, summary: Optional[pd.DataFrame]
) -> Dict[str, int]:
    if summary is not None:
        return summary.groupby("split")["total_frames"].sum().to_dict()

    logger.info("Deriving split totals from segment manifest (no summary provided)")
    totals = manifest.groupby("split")["duration_frames"].sum()
    return totals.to_dict()


def summarize_frames_by_label(
    manifest: pd.DataFrame, split_totals: Dict[str, int]
) -> pd.DataFrame:
    grouped = (
        manifest.groupby(["split", "label"])["duration_frames"]
        .sum()
        .reset_index(name="frames")
    )
    counts = manifest.groupby(["split", "label"]).size().reset_index(name="segments")
    grouped = grouped.merge(counts, on=["split", "label"], how="left")

    grouped["split_total_frames"] = grouped["split"].map(split_totals)
    grouped["frame_fraction"] = grouped["frames"] / grouped["split_total_frames"]
    grouped = grouped.sort_values(["split", "frames"], ascending=[True, False])
    return grouped


def summarize_overall_frames(grouped: pd.DataFrame) -> pd.DataFrame:
    overall = (
        grouped.groupby("label")["frames"]
        .sum()
        .reset_index()
        .sort_values("frames", ascending=False)
    )
    total_frames = overall["frames"].sum()
    overall["frame_fraction"] = overall["frames"] / total_frames
    return overall


def plot_split_bars(grouped: pd.DataFrame, output_dir: Path, show: bool) -> List[Path]:
    paths: List[Path] = []
    for split, split_df in grouped.groupby("split"):
        fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(split_df))))
        ax.barh(split_df["label"], split_df["frames"], color="#4C72B0")
        ax.invert_yaxis()
        ax.set_xlabel("Frames")
        ax.set_title(f"Frames by label — {split}")

        for i, (frames, frac) in enumerate(
            zip(split_df["frames"], split_df["frame_fraction"])
        ):
            ax.text(frames, i, f" {frames:,} ({frac:.1%})", va="center", ha="left")

        plt.tight_layout()
        path = output_dir / f"frames_by_label_{split}.png"
        fig.savefig(path, dpi=200)
        paths.append(path)
        if show:
            plt.show()
        plt.close(fig)
    return paths


def plot_stacked_by_label(grouped: pd.DataFrame, output_dir: Path, show: bool) -> Path:
    pivot = grouped.pivot(index="label", columns="split", values="frames").fillna(0)
    pivot = pivot.sort_values(pivot.columns.tolist(), ascending=False)

    ax = pivot.plot(kind="bar", stacked=True, figsize=(9, max(3, 0.5 * len(pivot))), colormap="tab20")
    ax.set_ylabel("Frames")
    ax.set_title("Frame distribution by label and split")
    ax.legend(title="Split")
    plt.tight_layout()

    path = output_dir / "frames_by_label_split_stacked.png"
    plt.savefig(path, dpi=200)
    if show:
        plt.show()
    plt.close()
    return path


def main(args: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parsed = parse_args(args)
    ensure_output_dir(parsed.output_dir)

    manifest = load_manifest(parsed.manifest)
    summary = load_summary(parsed.summary)
    split_totals = compute_split_totals(manifest, summary)

    grouped = summarize_frames_by_label(manifest, split_totals)
    overall = summarize_overall_frames(grouped)

    frames_csv = parsed.output_dir / "frames_by_label_split.csv"
    overall_csv = parsed.output_dir / "frames_by_label_overall.csv"
    grouped.to_csv(frames_csv, index=False)
    overall.to_csv(overall_csv, index=False)
    logger.info("Wrote split summary to %s", frames_csv)
    logger.info("Wrote overall summary to %s", overall_csv)

    split_plots = plot_split_bars(grouped, parsed.output_dir, parsed.show)
    stacked_plot = plot_stacked_by_label(grouped, parsed.output_dir, parsed.show)
    logger.info("Saved split plots: %s", ", ".join(str(p) for p in split_plots))
    logger.info("Saved stacked plot: %s", stacked_plot)


if __name__ == "__main__":
    main()
