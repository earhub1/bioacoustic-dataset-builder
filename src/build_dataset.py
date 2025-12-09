"""CLI to assemble synthetic sequences from extracted fragment features.

This tool reads one or more fragment manifests produced by
``extract_fragments.py`` and concatenates the stored feature matrices into
longer sequences. It supports class inclusion/exclusion (e.g., ignorar "NI"),
balancing the share of "Nothing" against eventos anotados, and reproducible
sampling.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_FRAGMENTS_DIR = Path("data/results/fragments")
DEFAULT_OUTPUT_DIR = Path("data/results/sequences")


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concatenate fragment features into synthetic sequences."
    )
    parser.add_argument(
        "--fragments-dir",
        action="append",
        type=Path,
        default=None,
        help=(
            "Directory containing fragment subfolders and manifest.csv. Can be passed multiple times; "
            "defaults to data/results/fragments."
        ),
    )
    parser.add_argument(
        "--include-labels",
        nargs="+",
        default=None,
        help="Optional list of labels to include. If omitted, all labels are considered.",
    )
    parser.add_argument(
        "--exclude-labels",
        nargs="+",
        default=["NI"],
        help="Labels to exclude (default: NI).",
    )
    parser.add_argument(
        "--sequence-duration",
        type=float,
        default=5.0,
        help="Target duration in seconds for each synthetic sequence.",
    )
    parser.add_argument(
        "--pack-all-fragments",
        action="store_true",
        help=(
            "Disable sampling with replacement and consume every fragment exactly once, "
            "allocating them to splits by frame budget."
        ),
    )
    parser.add_argument(
        "--max-sequence-duration",
        type=float,
        default=None,
        help=(
            "Optional maximum duration (s) for each sequence when --pack-all-fragments is enabled. "
            "If omitted, a single sequence is produced por split using all assigned frames."
        ),
    )
    parser.add_argument(
        "--max-fragments-per-sequence",
        type=int,
        default=None,
        help=(
            "Optional cap on how many fragments can be concatenated per sequence. "
            "If set, sampling stops when this limit is reached even if the duration target was not met."
        ),
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=10,
        help="Number of sequences to generate.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Proportion of sequences to route to the train split.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proportion of sequences to route to the validation split.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Proportion of sequences to route to the test split.",
    )
    parser.add_argument(
        "--nothing-ratio",
        type=float,
        default=1.0,
        help=(
            "Ratio of selecting 'Nothing' fragments relative to other labels (e.g., 1.0 keeps a 1:1 balance when both pools exist)."
        ),
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=64000,
        help="Sampling rate used to interpret frame/hop durations (match the extractor).",
    )
    parser.add_argument(
        "--allow-partial-fragments",
        action="store_true",
        help=(
            "Permit including fragments longer than the remaining budget; the tail will be trimmed to the sequence limit. "
            "By default, fragments longer than the remaining frames are skipped and resampled."
        ),
    )
    parser.add_argument(
        "--frame-length",
        type=int,
        default=6400,
        help="Frame length in samples (match the extractor).",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=6400,
        help="Hop length in samples (match the extractor).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write the assembled sequences and manifest_sequences.csv.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    return parser.parse_args(args=args)


def frames_for_duration(duration_s: float, sr: int, frame_length: int, hop_length: int) -> int:
    total_samples = max(duration_s * sr, 0)
    if total_samples <= 0:
        return 0
    if total_samples <= frame_length:
        return 1
    return int(np.ceil((total_samples - frame_length) / hop_length + 1))


def frames_to_seconds(n_frames: int, sr: int, frame_length: int, hop_length: int) -> float:
    if n_frames <= 0:
        return 0.0
    return ((n_frames - 1) * hop_length + frame_length) / float(sr)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_snippet_path(snippet: str, manifest_dir: Path) -> Path:
    path = Path(snippet)
    if path.is_absolute() or ":" in snippet:
        return path
    return manifest_dir / path


def load_manifests(
    fragment_dirs: List[Path], include_labels: Optional[List[str]], exclude_labels: List[str]
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for frag_dir in fragment_dirs:
        manifest_path = frag_dir / "manifest.csv"
        if not manifest_path.exists():
            logger.warning("Skipping %s because manifest.csv is missing", frag_dir)
            continue
        df = pd.read_csv(manifest_path)
        df["_manifest_dir"] = manifest_path.parent
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No manifest.csv files found in provided fragments directories.")

    data = pd.concat(frames, ignore_index=True)
    data["n_frames"] = pd.to_numeric(data["n_frames"], errors="coerce")
    if data["n_frames"].isna().any():
        raise ValueError("All fragments must provide n_frames to support packing by frame budget.")
    if include_labels is not None:
        data = data[data["label"].isin(include_labels)]
    if exclude_labels:
        data = data[~data["label"].isin(exclude_labels)]

    if data.empty:
        raise ValueError("No fragments available after applying include/exclude label filters.")
    return data


def normalize_split_probs(train: float, val: float, test: float) -> tuple[list[str], np.ndarray]:
    if min(train, val, test) < 0:
        raise ValueError("Split ratios must be non-negative.")
    split_total = train + val + test
    if not np.isclose(split_total, 1.0):
        raise ValueError("train-ratio + val-ratio + test-ratio must sum to 1.0.")
    split_labels = ["train", "val", "test"]
    split_probs = np.array([train, val, test], dtype=float)
    if split_probs.sum() <= 0:
        raise ValueError("At least one split ratio must be greater than zero.")
    split_probs = split_probs / split_probs.sum()
    return split_labels, split_probs


def pick_label(label_pools: Dict[str, List[int]], rng: np.random.Generator, nothing_ratio: float) -> Optional[str]:
    has_nothing = bool(label_pools.get("Nothing"))
    non_nothing_labels = [lab for lab in label_pools.keys() if lab != "Nothing" and label_pools[lab]]

    if not has_nothing and not non_nothing_labels:
        return None

    if not has_nothing:
        return rng.choice(non_nothing_labels)
    if not non_nothing_labels:
        return "Nothing"

    nothing_weight = max(nothing_ratio, 0.0)
    event_weight = 1.0
    total = nothing_weight + event_weight
    if total <= 0:
        return rng.choice(non_nothing_labels)

    if rng.random() < (nothing_weight / total):
        return "Nothing"
    return rng.choice(non_nothing_labels)


def build_sequence(
    df: pd.DataFrame,
    target_frames: int,
    sr: int,
    frame_length: int,
    hop_length: int,
    nothing_ratio: float,
    rng: np.random.Generator,
    max_fragments: Optional[int] = None,
    allow_partial_fragments: bool = False,
    ) -> tuple[np.ndarray, List[dict], dict]:
    label_pools: Dict[str, List[int]] = {}
    for idx, row in df.iterrows():
        label_pools.setdefault(row["label"], []).append(idx)

    current_frames = 0
    segments: List[dict] = []
    feature_chunks: List[np.ndarray] = []
    max_attempts = max(target_frames * 5, 100)
    attempts = 0
    skipped_too_long = 0
    fragment_limit_reached = False

    while current_frames < target_frames and attempts < max_attempts:
        if max_fragments is not None and len(segments) >= max_fragments:
            fragment_limit_reached = True
            break

        attempts += 1
        label = pick_label(label_pools, rng, nothing_ratio)
        if label is None:
            break

        pool_indices = label_pools.get(label, [])
        if not pool_indices:
            continue

        row_idx = pool_indices[rng.integers(0, len(pool_indices))]
        row = df.loc[row_idx]
        manifest_dir = Path(row["_manifest_dir"])
        snippet_path = resolve_snippet_path(str(row["snippet_path"]), manifest_dir)
        if not snippet_path.exists():
            logger.warning("Skipping missing snippet %s", snippet_path)
            continue

        features = np.load(snippet_path)
        n_frames = features.shape[1]
        if n_frames <= 0:
            continue

        remaining_frames = target_frames - current_frames
        if n_frames > remaining_frames and not allow_partial_fragments:
            skipped_too_long += 1
            continue

        start_frame = current_frames
        end_frame = current_frames + n_frames

        feature_chunks.append(features)
        segments.append(
            {
                "label": label,
                "snippet_path": str(snippet_path),
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
            }
        )

        current_frames = end_frame

    if not feature_chunks:
        raise RuntimeError("Unable to assemble sequence: no valid fragments were sampled.")

    combined = np.concatenate(feature_chunks, axis=1)
    truncated_segments = 0
    if combined.shape[1] > target_frames:
        combined = combined[:, :target_frames]

    trimmed_segments: List[dict] = []
    for seg in segments:
        if seg["start_frame"] >= target_frames:
            continue
        end_frame = min(seg["end_frame"], target_frames)
        start_frame = seg["start_frame"]
        truncated_flag = end_frame < seg["end_frame"]
        if truncated_flag:
            truncated_segments += 1
        trimmed_segments.append(
            {
                **seg,
                "end_frame": int(end_frame),
                "start_s": frames_to_seconds(start_frame, sr, frame_length, hop_length),
                "end_s": frames_to_seconds(end_frame, sr, frame_length, hop_length),
                "truncated": truncated_flag,
            }
        )

    return combined, trimmed_segments, {
        "skipped_too_long": skipped_too_long,
        "fragment_limit_reached": fragment_limit_reached,
        "truncated_segments": truncated_segments,
        "pack_all_mode": False,
    }


def save_sequence(
    output_dir: Path,
    sequence_idx: int,
    features: np.ndarray,
    segments: List[dict],
    sr: int,
    frame_length: int,
    hop_length: int,
    split: str,
    ) -> dict:
    ensure_output_dir(output_dir)
    seq_path = output_dir / f"sequence_{sequence_idx}.npy"
    np.save(seq_path, features)

    total_frames = int(features.shape[1])
    total_duration_s = frames_to_seconds(total_frames, sr, frame_length, hop_length)
    return {
        "sequence_path": str(seq_path),
        "total_frames": total_frames,
        "total_duration_s": total_duration_s,
        "n_segments": len(segments),
        "segments": json.dumps(segments),
        "split": split,
    }


def allocate_fragments_by_split(
    df: pd.DataFrame, split_labels: list[str], split_probs: np.ndarray, rng: np.random.Generator
) -> dict:
    total_frames = int(df["n_frames"].sum())
    if total_frames <= 0:
        raise ValueError("No frames available to pack.")

    budgets = [int(total_frames * p) for p in split_probs]
    # Ensure full coverage by assigning any residual to the last split
    residual = total_frames - sum(budgets)
    budgets[-1] += residual

    assignments: dict[str, list[pd.Series]] = {lbl: [] for lbl in split_labels}
    remaining = budgets[0]
    split_idx = 0

    for row_idx in rng.permutation(df.index):
        row = df.loc[row_idx]
        while split_idx < len(split_labels) - 1 and remaining <= 0:
            split_idx += 1
            remaining = budgets[split_idx]

        assignments[split_labels[split_idx]].append(row)
        remaining -= int(row["n_frames"])

    return assignments, budgets


def finalize_sequence_chunks(
    chunks: List[np.ndarray],
    segments: List[dict],
    sr: int,
    frame_length: int,
    hop_length: int,
) -> tuple[np.ndarray, List[dict], int]:
    if not chunks:
        raise RuntimeError("Cannot finalize an empty sequence.")

    combined = np.concatenate(chunks, axis=1)
    truncated_segments = 0
    enriched_segments: List[dict] = []
    for seg in segments:
        enriched_segments.append(
            {
                **seg,
                "start_s": frames_to_seconds(seg["start_frame"], sr, frame_length, hop_length),
                "end_s": frames_to_seconds(seg["end_frame"], sr, frame_length, hop_length),
                "truncated": False,
            }
        )
    return combined, enriched_segments, truncated_segments


def build_sequences_pack_all(
    args: argparse.Namespace,
    df: pd.DataFrame,
    split_labels: list[str],
    split_probs: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    max_seq_frames = None
    if args.max_sequence_duration is not None:
        max_seq_frames = frames_for_duration(
            duration_s=args.max_sequence_duration,
            sr=args.target_sr,
            frame_length=args.frame_length,
            hop_length=args.hop_length,
        )
        if max_seq_frames <= 0:
            raise ValueError("max-sequence-duration must be positive when provided.")

    assignments, budgets = allocate_fragments_by_split(df, split_labels, split_probs, rng)

    logger.info(
        "Pack-all mode: total_frames=%d -> budgets per split %s", int(df["n_frames"].sum()), budgets
    )

    ensure_output_dir(args.output_dir)
    records: List[dict] = []
    sequence_idx = 0

    for split in split_labels:
        rows = assignments.get(split, [])
        if not rows:
            continue

        chunks: List[np.ndarray] = []
        segments: List[dict] = []
        current_frames = 0
        split_dir = args.output_dir / split

        def flush_sequence() -> None:
            nonlocal chunks, segments, current_frames, sequence_idx
            if not chunks:
                return
            features, seq_segments, truncated_segments = finalize_sequence_chunks(
                chunks, segments, args.target_sr, args.frame_length, args.hop_length
            )
            record = save_sequence(
                output_dir=split_dir,
                sequence_idx=sequence_idx,
                features=features,
                segments=seq_segments,
                sr=args.target_sr,
                frame_length=args.frame_length,
                hop_length=args.hop_length,
                split=split,
            )
            record.update(
                {
                    "seed": args.seed,
                    "skipped_too_long": 0,
                    "fragment_limit_reached": False,
                    "truncated_segments": truncated_segments,
                    "pack_all_mode": True,
                }
            )
            records.append(record)
            sequence_idx += 1
            chunks = []
            segments = []
            current_frames = 0

        for row in rows:
            manifest_dir = Path(row["_manifest_dir"])
            snippet_path = resolve_snippet_path(str(row["snippet_path"]), manifest_dir)
            if not snippet_path.exists():
                logger.warning("Skipping missing snippet %s", snippet_path)
                continue

            features = np.load(snippet_path)
            n_frames = features.shape[1]
            if n_frames <= 0:
                continue

            if max_seq_frames is not None and current_frames > 0:
                if current_frames + n_frames > max_seq_frames:
                    flush_sequence()

            start_frame = current_frames
            end_frame = current_frames + n_frames
            segments.append(
                {
                    "label": row["label"],
                    "snippet_path": str(snippet_path),
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame),
                }
            )
            chunks.append(features)
            current_frames = end_frame

            if max_seq_frames is not None and current_frames >= max_seq_frames:
                flush_sequence()

        flush_sequence()

    manifest = pd.DataFrame(records)
    manifest_path = args.output_dir / "manifest_sequences.csv"
    manifest.to_csv(manifest_path, index=False)
    logger.info("Saved %d sequences to %s", len(manifest), manifest_path)

    for split in split_labels:
        split_df = manifest[manifest["split"] == split]
        if split_df.empty:
            continue
        split_manifest_path = args.output_dir / split / "manifest_sequences.csv"
        ensure_output_dir(split_manifest_path.parent)
        split_df.to_csv(split_manifest_path, index=False)
        logger.info("Saved %d %s sequences to %s", len(split_df), split, split_manifest_path)

    return manifest


def build_sequences(args: argparse.Namespace) -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    fragments_dirs = args.fragments_dir or [DEFAULT_FRAGMENTS_DIR]

    df = load_manifests(
        fragment_dirs=fragments_dirs,
        include_labels=args.include_labels,
        exclude_labels=args.exclude_labels,
    )

    split_labels, split_probs = normalize_split_probs(
        train=args.train_ratio, val=args.val_ratio, test=args.test_ratio
    )

    rng = np.random.default_rng(args.seed)

    if args.pack_all_fragments:
        return build_sequences_pack_all(args, df, split_labels, split_probs, rng)

    target_frames = frames_for_duration(
        duration_s=args.sequence_duration,
        sr=args.target_sr,
        frame_length=args.frame_length,
        hop_length=args.hop_length,
    )
    if target_frames <= 0:
        raise ValueError("sequence-duration must be positive.")

    records: List[dict] = []

    for seq_idx in range(args.num_sequences):
        features, segments, meta = build_sequence(
            df=df,
            target_frames=target_frames,
            sr=args.target_sr,
            frame_length=args.frame_length,
            hop_length=args.hop_length,
            nothing_ratio=args.nothing_ratio,
            rng=rng,
            max_fragments=args.max_fragments_per_sequence,
            allow_partial_fragments=args.allow_partial_fragments,
        )

        split = rng.choice(split_labels, p=split_probs)
        split_dir = args.output_dir / split

        record = save_sequence(
            output_dir=split_dir,
            sequence_idx=seq_idx,
            features=features,
            segments=segments,
            sr=args.target_sr,
            frame_length=args.frame_length,
            hop_length=args.hop_length,
            split=split,
        )
        record.update(
            {
                "seed": args.seed,
                "skipped_too_long": meta["skipped_too_long"],
                "fragment_limit_reached": meta["fragment_limit_reached"],
                "truncated_segments": meta["truncated_segments"],
                "pack_all_mode": False,
            }
        )
        records.append(record)

    manifest = pd.DataFrame(records)
    ensure_output_dir(args.output_dir)
    manifest_path = args.output_dir / "manifest_sequences.csv"
    manifest.to_csv(manifest_path, index=False)
    logger.info("Saved %d sequences to %s", len(manifest), manifest_path)

    for split in split_labels:
        split_df = manifest[manifest["split"] == split]
        if split_df.empty:
            continue
        split_manifest_path = args.output_dir / split / "manifest_sequences.csv"
        ensure_output_dir(split_manifest_path.parent)
        split_df.to_csv(split_manifest_path, index=False)
        logger.info("Saved %d %s sequences to %s", len(split_df), split, split_manifest_path)
    return manifest


def main(cli_args: Optional[Sequence[str]] = None) -> None:
    args = parse_args(cli_args)
    build_sequences(args)


if __name__ == "__main__":
    main()
