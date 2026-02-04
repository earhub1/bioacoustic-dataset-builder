"""Preview split balance from a fragment manifest without building sequences."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from build_dataset import assign_splits_by_ratio, normalize_split_probs

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST = Path("data/results/fragments/manifest.csv")
DEFAULT_OUTPUT_DIR = Path("data/results/split_preview")


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview split balance from a fragment manifest without building sequences."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to a fragment manifest (must include label and n_frames or onset/offset).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write the manifest_split_preview.csv summary.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Proportion of fragments to route to the train split.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proportion of fragments to route to the validation split.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Proportion of fragments to route to the test split.",
    )
    parser.add_argument(
        "--split-by-fragment",
        action="store_true",
        help="Split fragments into train/val/test without replacement.",
    )
    parser.add_argument(
        "--split-by-event-fragments",
        action="store_true",
        help="Split each label independently into train/val/test without replacement.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splitting.",
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=64000,
        help="Sampling rate used to derive n_frames if missing.",
    )
    parser.add_argument(
        "--frame-length",
        type=int,
        default=6400,
        help="Frame length in samples (used to approximate n_frames).",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=6400,
        help="Hop length in samples (used to approximate n_frames).",
    )
    return parser.parse_args(args=args)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def estimate_n_frames(duration_s: float, target_sr: int, frame_length: int, hop_length: int) -> int:
    total_samples = max(int(round(duration_s * target_sr)), 0)
    if total_samples <= 0:
        return 0
    if total_samples <= frame_length:
        return 1
    return int(np.floor((total_samples - frame_length) / hop_length) + 1)


def ensure_frames_column(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if "n_frames" in df.columns:
        return df
    if {"onset_s", "offset_s"} <= set(df.columns):
        durations = (df["offset_s"] - df["onset_s"]).clip(lower=0)
        df = df.copy()
        df["n_frames"] = durations.apply(
            lambda duration: estimate_n_frames(duration, args.target_sr, args.frame_length, args.hop_length)
        )
        return df
    raise ValueError("Manifest must include n_frames or onset_s/offset_s to estimate frames.")


def assign_splits(
    df: pd.DataFrame,
    split_labels: list[str],
    split_probs: np.ndarray,
    rng: np.random.Generator,
    split_by_fragment: bool,
    split_by_event_fragments: bool,
) -> pd.DataFrame:
    if "split" in df.columns and not (split_by_fragment or split_by_event_fragments):
        return df

    if split_by_fragment and split_by_event_fragments:
        raise ValueError("--split-by-fragment cannot be combined with --split-by-event-fragments.")

    df = df.copy()
    if split_by_event_fragments:
        split_assignments = pd.Series(index=df.index, dtype=str)
        for label in df["label"].unique():
            label_df = df[df["label"] == label]
            label_splits = assign_splits_by_ratio(label_df, split_labels, split_probs, rng)
            split_assignments.loc[label_splits.index] = label_splits.values
        if split_assignments.isnull().any():
            raise ValueError("Unable to assign splits for all fragments.")
        df["split"] = split_assignments.values
        return df

    if split_by_fragment:
        total_fragments = len(df)
        counts = [int(total_fragments * p) for p in split_probs]
        counts[-1] += total_fragments - sum(counts)
        indices = rng.permutation(df.index.to_numpy())
        split_assignments: List[str] = []
        for split, count in zip(split_labels, counts):
            split_assignments.extend([split] * count)
        df = df.loc[indices].copy()
        df["split"] = split_assignments
        return df

    raise ValueError("Manifest has no split column; choose --split-by-fragment or --split-by-event-fragments.")


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["split", "label"])["n_frames"]
        .sum()
        .reset_index(name="frames_sum")
    )
    counts = df.groupby(["split", "label"]).size().reset_index(name="segments_count")
    grouped = grouped.merge(counts, on=["split", "label"], how="left")
    split_totals = grouped.groupby("split")["frames_sum"].sum().to_dict()
    grouped["frame_fraction"] = grouped.apply(
        lambda row: (row["frames_sum"] / split_totals.get(row["split"], 1)),
        axis=1,
    )
    grouped = grouped.sort_values(["split", "frames_sum"], ascending=[True, False])
    return grouped


def log_summary(grouped: pd.DataFrame) -> None:
    for split in grouped["split"].unique():
        rows = grouped[grouped["split"] == split]
        parts = [f"{row['label']}={row['frame_fraction']:.1%}" for _, row in rows.iterrows()]
        logger.info("%s: %s", split, ", ".join(parts))


def main(cli_args: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(cli_args)
    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    df = pd.read_csv(args.manifest)
    if "label" not in df.columns:
        raise ValueError("Manifest must include a label column.")

    df = ensure_frames_column(df, args)
    split_labels, split_probs = normalize_split_probs(
        train=args.train_ratio, val=args.val_ratio, test=args.test_ratio
    )
    rng = np.random.default_rng(args.seed)
    df = assign_splits(
        df,
        split_labels=split_labels,
        split_probs=split_probs,
        rng=rng,
        split_by_fragment=args.split_by_fragment,
        split_by_event_fragments=args.split_by_event_fragments,
    )

    grouped = summarize(df)
    ensure_output_dir(args.output_dir)
    output_path = args.output_dir / "manifest_split_preview.csv"
    grouped.to_csv(output_path, index=False)
    logger.info("Saved split preview to %s", output_path)
    log_summary(grouped)


if __name__ == "__main__":
    main()
