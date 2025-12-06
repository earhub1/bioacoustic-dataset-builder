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
        "--num-sequences",
        type=int,
        default=10,
        help="Number of sequences to generate.",
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
    if include_labels is not None:
        data = data[data["label"].isin(include_labels)]
    if exclude_labels:
        data = data[~data["label"].isin(exclude_labels)]

    if data.empty:
        raise ValueError("No fragments available after applying include/exclude label filters.")
    return data


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
) -> tuple[np.ndarray, List[dict]]:
    label_pools: Dict[str, List[int]] = {}
    for idx, row in df.iterrows():
        label_pools.setdefault(row["label"], []).append(idx)

    current_frames = 0
    segments: List[dict] = []
    feature_chunks: List[np.ndarray] = []
    max_attempts = max(target_frames * 5, 100)
    attempts = 0

    while current_frames < target_frames and attempts < max_attempts:
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
    if combined.shape[1] > target_frames:
        combined = combined[:, :target_frames]

    trimmed_segments: List[dict] = []
    for seg in segments:
        if seg["start_frame"] >= target_frames:
            continue
        end_frame = min(seg["end_frame"], target_frames)
        start_frame = seg["start_frame"]
        trimmed_segments.append(
            {
                **seg,
                "end_frame": int(end_frame),
                "start_s": frames_to_seconds(start_frame, sr, frame_length, hop_length),
                "end_s": frames_to_seconds(end_frame, sr, frame_length, hop_length),
            }
        )

    return combined, trimmed_segments


def save_sequence(
    output_dir: Path,
    sequence_idx: int,
    features: np.ndarray,
    segments: List[dict],
    sr: int,
    frame_length: int,
    hop_length: int,
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
    }


def build_sequences(args: argparse.Namespace) -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    fragments_dirs = args.fragments_dir or [DEFAULT_FRAGMENTS_DIR]

    df = load_manifests(
        fragment_dirs=fragments_dirs,
        include_labels=args.include_labels,
        exclude_labels=args.exclude_labels,
    )

    target_frames = frames_for_duration(
        duration_s=args.sequence_duration,
        sr=args.target_sr,
        frame_length=args.frame_length,
        hop_length=args.hop_length,
    )
    if target_frames <= 0:
        raise ValueError("sequence-duration must be positive.")

    rng = np.random.default_rng(args.seed)
    records: List[dict] = []

    for seq_idx in range(args.num_sequences):
        features, segments = build_sequence(
            df=df,
            target_frames=target_frames,
            sr=args.target_sr,
            frame_length=args.frame_length,
            hop_length=args.hop_length,
            nothing_ratio=args.nothing_ratio,
            rng=rng,
        )

        record = save_sequence(
            output_dir=args.output_dir,
            sequence_idx=seq_idx,
            features=features,
            segments=segments,
            sr=args.target_sr,
            frame_length=args.frame_length,
            hop_length=args.hop_length,
        )
        record["seed"] = args.seed
        records.append(record)

    manifest = pd.DataFrame(records)
    manifest_path = args.output_dir / "manifest_sequences.csv"
    manifest.to_csv(manifest_path, index=False)
    logger.info("Saved %d sequences to %s", len(manifest), manifest_path)
    return manifest


def main(cli_args: Optional[Sequence[str]] = None) -> None:
    args = parse_args(cli_args)
    build_sequences(args)


if __name__ == "__main__":
    main()
