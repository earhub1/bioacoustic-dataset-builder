"""CLI to assemble synthetic sequences from extracted fragment features.

This tool reads one or more fragment manifests produced by
``extract_fragments.py`` and concatenates the stored log-mel spectrogram (dB)
matrices into longer sequences. It supports class inclusion/exclusion (e.g.,
ignorar "NI"), balancing the share of "Nothing" against eventos anotados, and
reproducible sampling.
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
FEATURE_TYPE = "logmel_db"
FEATURE_DB_REF = 1.0
FEATURE_TOP_DB = 80


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
        "--split-by-fragment",
        action="store_true",
        help=(
            "Split fragment pools into train/val/test without replacement before building sequences. "
            "When enabled, sequences are sampled only from the assigned split and manifest_split.csv is saved."
        ),
    )
    parser.add_argument(
        "--target-event-fragments",
        type=int,
        default=None,
        help=(
            "When set, compute the sequence frame budget from the total frames of the first N event fragments "
            "and aim for a 50/50 Nothing:event balance (target_frames = 2 * frames_event). Requires --event-label "
            "when multiple event labels exist."
        ),
    )
    parser.add_argument(
        "--target-event-fragments-train",
        type=int,
        default=None,
        help="Number of event fragments to budget for the train split (requires --split-by-fragment).",
    )
    parser.add_argument(
        "--target-event-fragments-val",
        type=int,
        default=None,
        help="Number of event fragments to budget for the validation split (requires --split-by-fragment).",
    )
    parser.add_argument(
        "--target-event-fragments-test",
        type=int,
        default=None,
        help="Number of event fragments to budget for the test split (requires --split-by-fragment).",
    )
    parser.add_argument(
        "--event-label",
        type=str,
        default=None,
        help="Event label to use with --target-event-fragments (e.g., G01).",
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
        "--max-consecutive-event-fragments",
        type=int,
        default=3,
        help=(
            "Maximum number of consecutive event fragments before forcing insertion of Nothing (when available)."
        ),
    )
    parser.add_argument(
        "--max-consecutive-event-frames",
        type=int,
        default=None,
        help=(
            "Optional cap on consecutive event frames before forcing insertion of Nothing (when available)."
        ),
    )
    parser.add_argument(
        "--min-nothing-after-event-frames",
        type=int,
        default=20,
        help=(
            "Minimum frames of Nothing required after an event before allowing another event fragment."
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
        "--expected-n-mels",
        type=int,
        default=64,
        help="Expected number of mel frequency bins (log-mel dB fragments).",
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
        "--validate-composition",
        action="store_true",
        help=(
            "Generate a single sequence for inspection without saving files, logging the timeline and run metrics."
        ),
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
    # Accept absolute paths or explicit drive references (Windows), returning as-is.
    if path.is_absolute() or ":" in snippet:
        return path

    # If the provided relative path already exists from the current working
    # directory (e.g., when manifest entries include the fragments directory
    # prefix), honor it directly to avoid duplicating the fragments dir.
    if path.exists():
        return path

    # Fallback: treat the snippet path as relative to the manifest directory.
    return manifest_dir / path


def normalize_feature_shape(features: np.ndarray, expected_n_mels: int, snippet_path: Path) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError(
            f"Fragment {snippet_path} has shape {features.shape}; expected 2D log-mel dB (n_mels x frames)."
        )

    if features.shape[0] == expected_n_mels:
        return features

    if features.shape[1] == expected_n_mels:
        logger.warning(
            "Transposing fragment %s from shape %s to (n_mels x frames) using expected_n_mels=%d",
            snippet_path,
            features.shape,
            expected_n_mels,
        )
        return features.T

    raise ValueError(
        f"Fragment {snippet_path} has incompatible shape {features.shape}; expected n_mels={expected_n_mels} on the first dimension."
    )


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


def select_row_for_label(
    df: pd.DataFrame, label_pools: Dict[str, List[int]], label: str, rng: np.random.Generator
) -> Optional[pd.Series]:
    pool_indices = label_pools.get(label, [])
    if not pool_indices:
        return None
    row_idx = pool_indices[rng.integers(0, len(pool_indices))]
    return df.loc[row_idx]


def compute_composition_stats(
    segments: List[dict], total_frames: int, sr: int, frame_length: int, hop_length: int
) -> dict:
    frames_by_label: Dict[str, int] = {}
    current_run_frames = 0
    current_run_fragments = 0
    max_event_run_frames = 0
    max_event_run_fragments = 0
    num_event_runs = 0
    in_event_run = False

    for seg in sorted(segments, key=lambda s: s.get("start_frame", 0)):
        label = seg["label"]
        duration_frames = int(seg["end_frame"] - seg["start_frame"])
        frames_by_label[label] = frames_by_label.get(label, 0) + duration_frames
        is_event = label != "Nothing"

        if is_event:
            if not in_event_run:
                num_event_runs += 1
                current_run_frames = 0
                current_run_fragments = 0
            in_event_run = True
            current_run_frames += duration_frames
            current_run_fragments += 1
            max_event_run_frames = max(max_event_run_frames, current_run_frames)
            max_event_run_fragments = max(max_event_run_fragments, current_run_fragments)
        else:
            in_event_run = False
            current_run_frames = 0
            current_run_fragments = 0

    frames_nothing = frames_by_label.get("Nothing", 0)
    frames_events = max(total_frames - frames_nothing, 0)
    pct_nothing = frames_nothing / total_frames if total_frames > 0 else 0.0
    pct_events = frames_events / total_frames if total_frames > 0 else 0.0

    return {
        "frames_by_label": json.dumps(frames_by_label, ensure_ascii=False),
        "frames_nothing": frames_nothing,
        "frames_events": frames_events,
        "pct_nothing": pct_nothing,
        "pct_events": pct_events,
        "max_event_run_frames": max_event_run_frames,
        "max_event_run_seconds": frames_to_seconds(max_event_run_frames, sr, frame_length, hop_length),
        "max_event_run_fragments": max_event_run_fragments,
        "num_event_runs": num_event_runs,
    }


def build_sequence(
    df: pd.DataFrame,
    target_frames: int,
    sr: int,
    frame_length: int,
    hop_length: int,
    expected_n_mels: int,
    nothing_ratio: float,
    rng: np.random.Generator,
    max_fragments: Optional[int] = None,
    allow_partial_fragments: bool = False,
    max_consecutive_event_fragments: Optional[int] = None,
    max_consecutive_event_frames: Optional[int] = None,
    min_nothing_after_event_frames: int = 0,
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

    current_run_frames = 0
    current_run_fragments = 0
    gap_frames_remaining = 0
    in_event_run = False
    max_event_run_frames = 0
    max_event_run_fragments = 0
    num_event_runs = 0

    while current_frames < target_frames and attempts < max_attempts:
        if max_fragments is not None and len(segments) >= max_fragments:
            fragment_limit_reached = True
            break

        attempts += 1
        has_nothing_pool = bool(label_pools.get("Nothing"))
        force_nothing = False
        if gap_frames_remaining > 0:
            force_nothing = True
        if max_consecutive_event_fragments is not None and max_consecutive_event_fragments >= 0:
            if current_run_fragments >= max_consecutive_event_fragments:
                force_nothing = True
        if max_consecutive_event_frames is not None and max_consecutive_event_frames >= 0:
            if current_run_frames >= max_consecutive_event_frames:
                force_nothing = True

        if force_nothing and has_nothing_pool:
            label = "Nothing"
        else:
            label = pick_label(label_pools, rng, nothing_ratio)

        if label is None:
            break

        row = select_row_for_label(df, label_pools, label, rng)
        if row is None:
            continue

        # If event run constraints would be violated, switch to Nothing when possible.
        if label != "Nothing" and has_nothing_pool:
            prospective_frames = current_run_frames + int(row["n_frames"])
            prospective_frags = current_run_fragments + 1
            exceeds_frames = (
                max_consecutive_event_frames is not None
                and max_consecutive_event_frames >= 0
                and prospective_frames > max_consecutive_event_frames
            )
            exceeds_frags = (
                max_consecutive_event_fragments is not None
                and max_consecutive_event_fragments >= 0
                and prospective_frags > max_consecutive_event_fragments
            )
            if exceeds_frames or exceeds_frags or gap_frames_remaining > 0:
                label = "Nothing"
                row = select_row_for_label(df, label_pools, label, rng)
                if row is None:
                    # Fall back to the original event if Nothing pool is empty
                    label = pick_label(label_pools, rng, nothing_ratio)
                    if label is None:
                        break
                    row = select_row_for_label(df, label_pools, label, rng)
                    if row is None:
                        continue

        manifest_dir = Path(row["_manifest_dir"])
        snippet_path = resolve_snippet_path(str(row["snippet_path"]), manifest_dir)
        if not snippet_path.exists():
            logger.warning("Skipping missing snippet %s", snippet_path)
            continue

        features = np.load(snippet_path)
        features = normalize_feature_shape(features, expected_n_mels, snippet_path)
        n_frames = features.shape[1]
        if n_frames <= 0:
            continue

        remaining_frames = target_frames - current_frames
        if remaining_frames <= 0:
            break
        if n_frames > remaining_frames and not allow_partial_fragments:
            skipped_too_long += 1
            continue

        truncated = False
        original_end_frame = current_frames + n_frames
        if n_frames > remaining_frames and allow_partial_fragments:
            features = features[:, :remaining_frames]
            n_frames = features.shape[1]
            truncated = True

        start_frame = current_frames
        end_frame = current_frames + n_frames

        feature_chunks.append(features)
        segments.append(
            {
                "label": label,
                "snippet_path": str(snippet_path),
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "original_end_frame": int(original_end_frame),
                "truncated": truncated,
            }
        )

        if label != "Nothing":
            if not in_event_run:
                num_event_runs += 1
                current_run_frames = 0
                current_run_fragments = 0
                in_event_run = True
            current_run_frames += n_frames
            current_run_fragments += 1
            max_event_run_frames = max(max_event_run_frames, current_run_frames)
            max_event_run_fragments = max(max_event_run_fragments, current_run_fragments)
            gap_frames_remaining = max(min_nothing_after_event_frames, 0)
        else:
            gap_frames_remaining = max(gap_frames_remaining - n_frames, 0)
            in_event_run = False
            current_run_frames = 0
            current_run_fragments = 0

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
        truncated_flag = bool(seg.get("truncated", False)) or end_frame < seg.get(
            "original_end_frame", seg["end_frame"]
        )
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

    composition_stats = compute_composition_stats(
        trimmed_segments, total_frames=combined.shape[1], sr=sr, frame_length=frame_length, hop_length=hop_length
    )

    return combined, trimmed_segments, {
        "skipped_too_long": skipped_too_long,
        "fragment_limit_reached": fragment_limit_reached,
        "truncated_segments": truncated_segments,
        "pack_all_mode": False,
        "composition": composition_stats,
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
    pack_all_mode: bool,
    seed: int,
    expected_n_mels: int,
    feature_type: str,
    db_ref: float,
    top_db: float,
) -> tuple[dict, List[dict]]:
    ensure_output_dir(output_dir)
    seq_path = output_dir / f"sequence_{sequence_idx}.npy"
    np.save(seq_path, features)

    total_frames = int(features.shape[1])
    total_duration_s = frames_to_seconds(total_frames, sr, frame_length, hop_length)

    summary_record = {
        "sequence_path": str(seq_path),
        "sequence_idx": sequence_idx,
        "split": split,
        "total_frames": total_frames,
        "total_duration_s": total_duration_s,
        "n_segments": len(segments),
        "pack_all_mode": pack_all_mode,
        "seed": seed,
        "feature_type": feature_type,
        "mel_bins": expected_n_mels,
        "db_ref": db_ref,
        "top_db": top_db,
    }

    segment_records: List[dict] = []
    for seg_idx, seg in enumerate(segments):
        duration_frames = int(seg["end_frame"] - seg["start_frame"])
        segment_records.append(
            {
                "sequence_path": str(seq_path),
                "sequence_idx": sequence_idx,
                "split": split,
                "segment_idx": seg_idx,
                "label": seg["label"],
                "snippet_path": seg["snippet_path"],
                "start_frame": int(seg["start_frame"]),
                "end_frame": int(seg["end_frame"]),
                "duration_frames": duration_frames,
                "start_s": seg.get("start_s", frames_to_seconds(seg["start_frame"], sr, frame_length, hop_length)),
                "end_s": seg.get("end_s", frames_to_seconds(seg["end_frame"], sr, frame_length, hop_length)),
                "duration_s": frames_to_seconds(duration_frames, sr, frame_length, hop_length),
                "truncated": bool(seg.get("truncated", False)),
                "feature_type": feature_type,
                "mel_bins": expected_n_mels,
            }
        )

    return summary_record, segment_records


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
    summary_records: List[dict] = []
    segment_records: List[dict] = []
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
            composition_stats = compute_composition_stats(
                seq_segments,
                total_frames=features.shape[1],
                sr=args.target_sr,
                frame_length=args.frame_length,
                hop_length=args.hop_length,
            )
            summary_record, segment_list = save_sequence(
                output_dir=split_dir,
                sequence_idx=sequence_idx,
                features=features,
                segments=seq_segments,
                sr=args.target_sr,
                frame_length=args.frame_length,
                hop_length=args.hop_length,
                split=split,
                pack_all_mode=True,
                seed=args.seed,
                expected_n_mels=args.expected_n_mels,
                feature_type=FEATURE_TYPE,
                db_ref=FEATURE_DB_REF,
                top_db=FEATURE_TOP_DB,
            )
            summary_record.update(
                {
                    "skipped_too_long": 0,
                    "fragment_limit_reached": False,
                    "truncated_segments": truncated_segments,
                }
            )
            summary_record.update(composition_stats)
            summary_records.append(summary_record)
            segment_records.extend(segment_list)
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
            features = normalize_feature_shape(features, args.expected_n_mels, snippet_path)
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

    summary_df = pd.DataFrame(summary_records)
    segments_df = pd.DataFrame(segment_records)

    summary_path = args.output_dir / "manifest_sequences_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved %d sequence summaries to %s", len(summary_df), summary_path)

    segment_path = args.output_dir / "manifest_sequences.csv"
    segments_df.to_csv(segment_path, index=False)
    logger.info("Saved %d sequence segments to %s", len(segments_df), segment_path)

    for split in split_labels:
        split_summary = summary_df[summary_df["split"] == split]
        split_segments = segments_df[segments_df["split"] == split]
        if not split_summary.empty:
            split_summary_path = args.output_dir / split / "manifest_sequences_summary.csv"
            ensure_output_dir(split_summary_path.parent)
            split_summary.to_csv(split_summary_path, index=False)
            logger.info("Saved %d %s sequence summaries to %s", len(split_summary), split, split_summary_path)
        if not split_segments.empty:
            split_segment_path = args.output_dir / split / "manifest_sequences.csv"
            ensure_output_dir(split_segment_path.parent)
            split_segments.to_csv(split_segment_path, index=False)
            logger.info("Saved %d %s sequence segments to %s", len(split_segments), split, split_segment_path)

    return summary_df


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
    target_frames = 0

    if args.pack_all_fragments and args.validate_composition:
        raise ValueError("--validate-composition is not supported together with --pack-all-fragments.")

    if args.pack_all_fragments and args.target_event_fragments is not None:
        raise ValueError("--target-event-fragments is not supported together with --pack-all-fragments.")

    if args.pack_all_fragments and args.split_by_fragment:
        raise ValueError("--split-by-fragment is not supported together with --pack-all-fragments.")

    split_target_values = {
        "train": args.target_event_fragments_train,
        "val": args.target_event_fragments_val,
        "test": args.target_event_fragments_test,
    }
    using_split_targets = any(value is not None for value in split_target_values.values())

    if using_split_targets and args.target_event_fragments is not None:
        raise ValueError("Per-split target-event-fragments cannot be combined with --target-event-fragments.")

    if using_split_targets and not args.split_by_fragment:
        raise ValueError("--split-by-fragment is required when using per-split target-event-fragments.")

    if args.target_event_fragments is not None and args.target_event_fragments <= 0:
        raise ValueError("--target-event-fragments must be a positive integer.")

    if using_split_targets:
        for split, value in split_target_values.items():
            if split_probs[split_labels.index(split)] > 0 and (value is None or value <= 0):
                raise ValueError(f"--target-event-fragments-{split} must be a positive integer.")

    event_label = args.event_label
    if args.target_event_fragments is not None or using_split_targets:
        if event_label is None:
            event_candidates = sorted({label for label in df["label"].unique() if label != "Nothing"})
            if len(event_candidates) == 1:
                event_label = event_candidates[0]
                logger.info("Inferred event label '%s' for event fragment budgeting.", event_label)
            else:
                raise ValueError(
                    "--event-label is required when using target-event-fragments and multiple event labels exist."
                )

    if args.target_event_fragments is not None:
        event_df = df[df["label"] == event_label]
        if event_df.empty:
            raise ValueError(f"No fragments found for event label '{event_label}'.")
        if len(event_df) < args.target_event_fragments:
            raise ValueError(
                f"Requested {args.target_event_fragments} event fragments but only {len(event_df)} available for '{event_label}'."
            )
        event_indices = rng.choice(event_df.index.to_numpy(), size=args.target_event_fragments, replace=False)
        event_frames = int(event_df.loc[event_indices, "n_frames"].sum())
        if event_frames <= 0:
            raise ValueError("Selected event fragments have non-positive frame totals.")
        target_frames = int(event_frames * 2)
        logger.info(
            "Using %d event fragments (%s) totaling %d frames -> target_frames=%d for 50/50 balance.",
            args.target_event_fragments,
            event_label,
            event_frames,
            target_frames,
        )
    elif not using_split_targets:
        target_frames = frames_for_duration(
            duration_s=args.sequence_duration,
            sr=args.target_sr,
            frame_length=args.frame_length,
            hop_length=args.hop_length,
        )
        if target_frames <= 0:
            raise ValueError("sequence-duration must be positive.")

    target_frames_by_split: Dict[str, int] = {}

    if args.pack_all_fragments:
        return build_sequences_pack_all(args, df, split_labels, split_probs, rng)

    summary_records: List[dict] = []
    segment_records: List[dict] = []

    if args.split_by_fragment:
        ensure_output_dir(args.output_dir)
        total_fragments = len(df)
        if total_fragments <= 0:
            raise ValueError("No fragments available to split.")
        counts = [int(total_fragments * p) for p in split_probs]
        counts[-1] += total_fragments - sum(counts)
        indices = rng.permutation(df.index.to_numpy())
        split_assignments: List[str] = []
        for split, count in zip(split_labels, counts):
            split_assignments.extend([split] * count)
        df = df.loc[indices].copy()
        df["split"] = split_assignments
        split_manifest_path = args.output_dir / "manifest_split.csv"
        df.to_csv(split_manifest_path, index=False)
        logger.info("Saved split manifest with %d fragments to %s", len(df), split_manifest_path)
        if using_split_targets:
            for split, target_count in split_target_values.items():
                if target_count is None:
                    continue
                split_df = df[df["split"] == split]
                event_df = split_df[split_df["label"] == event_label]
                if event_df.empty:
                    raise ValueError(f"No fragments found for event label '{event_label}' in split '{split}'.")
                if len(event_df) < target_count:
                    raise ValueError(
                        f"Requested {target_count} event fragments for split '{split}' but only {len(event_df)} available."
                    )
                event_indices = rng.choice(event_df.index.to_numpy(), size=target_count, replace=False)
                event_frames = int(event_df.loc[event_indices, "n_frames"].sum())
                if event_frames <= 0:
                    raise ValueError(f"Selected event fragments for split '{split}' have non-positive frame totals.")
                target_frames_by_split[split] = int(event_frames * 2)
                logger.info(
                    "Split %s: using %d event fragments (%s) totaling %d frames -> target_frames=%d.",
                    split,
                    target_count,
                    event_label,
                    event_frames,
                    target_frames_by_split[split],
                )

    if args.validate_composition:
        if args.split_by_fragment:
            df = df[df["split"] == split_labels[0]]
            if df.empty:
                raise ValueError(f"No fragments available for split '{split_labels[0]}' to validate.")
            if target_frames_by_split:
                target_frames = target_frames_by_split[split_labels[0]]
        features, segments, meta = build_sequence(
            df=df,
            target_frames=target_frames,
            sr=args.target_sr,
            frame_length=args.frame_length,
            hop_length=args.hop_length,
            expected_n_mels=args.expected_n_mels,
            nothing_ratio=args.nothing_ratio,
            rng=rng,
            max_fragments=args.max_fragments_per_sequence,
            allow_partial_fragments=args.allow_partial_fragments,
            max_consecutive_event_fragments=args.max_consecutive_event_fragments,
            max_consecutive_event_frames=args.max_consecutive_event_frames,
            min_nothing_after_event_frames=args.min_nothing_after_event_frames,
        )

        logger.info("Validation timeline (label, start_frame -> end_frame, truncated):")
        for seg in segments:
            logger.info(
                "  %s: %d -> %d (truncated=%s)",
                seg["label"],
                int(seg["start_frame"]),
                int(seg["end_frame"]),
                bool(seg.get("truncated", False)),
            )

        composition = meta.get("composition", {})
        logger.info("Composition metrics: %s", composition)
        logger.info("Generated validation sequence with shape %s", features.shape)
        return pd.DataFrame([composition])

    if args.split_by_fragment:
        seq_counts = [int(args.num_sequences * p) for p in split_probs]
        seq_counts[-1] += args.num_sequences - sum(seq_counts)
        seq_idx = 0
        for split, split_count in zip(split_labels, seq_counts):
            if split_count <= 0:
                continue
            split_df = df[df["split"] == split]
            if split_df.empty:
                raise ValueError(f"No fragments available for split '{split}'.")
            split_dir = args.output_dir / split
            for _ in range(split_count):
                split_target_frames = target_frames_by_split.get(split, target_frames)
                features, segments, meta = build_sequence(
                    df=split_df,
                    target_frames=split_target_frames,
                    sr=args.target_sr,
                    frame_length=args.frame_length,
                    hop_length=args.hop_length,
                    expected_n_mels=args.expected_n_mels,
                    nothing_ratio=args.nothing_ratio,
                    rng=rng,
                    max_fragments=args.max_fragments_per_sequence,
                    allow_partial_fragments=args.allow_partial_fragments,
                    max_consecutive_event_fragments=args.max_consecutive_event_fragments,
                    max_consecutive_event_frames=args.max_consecutive_event_frames,
                    min_nothing_after_event_frames=args.min_nothing_after_event_frames,
                )

                summary_record, seq_segments = save_sequence(
                    output_dir=split_dir,
                    sequence_idx=seq_idx,
                    features=features,
                    segments=segments,
                    sr=args.target_sr,
                    frame_length=args.frame_length,
                    hop_length=args.hop_length,
                    split=split,
                    pack_all_mode=False,
                    seed=args.seed,
                    expected_n_mels=args.expected_n_mels,
                    feature_type=FEATURE_TYPE,
                    db_ref=FEATURE_DB_REF,
                    top_db=FEATURE_TOP_DB,
                )
                summary_record.update(
                    {
                        "skipped_too_long": meta["skipped_too_long"],
                        "fragment_limit_reached": meta["fragment_limit_reached"],
                        "truncated_segments": meta["truncated_segments"],
                    }
                )
                summary_record.update(meta.get("composition", {}))
                summary_records.append(summary_record)
                segment_records.extend(seq_segments)
                seq_idx += 1
    else:
        for seq_idx in range(args.num_sequences):
            features, segments, meta = build_sequence(
                df=df,
                target_frames=target_frames,
                sr=args.target_sr,
                frame_length=args.frame_length,
                hop_length=args.hop_length,
                expected_n_mels=args.expected_n_mels,
                nothing_ratio=args.nothing_ratio,
                rng=rng,
                max_fragments=args.max_fragments_per_sequence,
                allow_partial_fragments=args.allow_partial_fragments,
                max_consecutive_event_fragments=args.max_consecutive_event_fragments,
                max_consecutive_event_frames=args.max_consecutive_event_frames,
                min_nothing_after_event_frames=args.min_nothing_after_event_frames,
            )

            split = rng.choice(split_labels, p=split_probs)
            split_dir = args.output_dir / split

            summary_record, seq_segments = save_sequence(
                output_dir=split_dir,
                sequence_idx=seq_idx,
                features=features,
                segments=segments,
                sr=args.target_sr,
                frame_length=args.frame_length,
                hop_length=args.hop_length,
                split=split,
                pack_all_mode=False,
                seed=args.seed,
                expected_n_mels=args.expected_n_mels,
                feature_type=FEATURE_TYPE,
                db_ref=FEATURE_DB_REF,
                top_db=FEATURE_TOP_DB,
            )
            summary_record.update(
                {
                    "skipped_too_long": meta["skipped_too_long"],
                    "fragment_limit_reached": meta["fragment_limit_reached"],
                    "truncated_segments": meta["truncated_segments"],
                }
            )
            summary_record.update(meta.get("composition", {}))
            summary_records.append(summary_record)
            segment_records.extend(seq_segments)

    summary_df = pd.DataFrame(summary_records)
    segments_df = pd.DataFrame(segment_records)
    ensure_output_dir(args.output_dir)

    summary_path = args.output_dir / "manifest_sequences_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved %d sequence summaries to %s", len(summary_df), summary_path)

    segment_path = args.output_dir / "manifest_sequences.csv"
    segments_df.to_csv(segment_path, index=False)
    logger.info("Saved %d sequence segments to %s", len(segments_df), segment_path)

    for split in split_labels:
        split_summary = summary_df[summary_df["split"] == split]
        split_segments = segments_df[segments_df["split"] == split]
        if not split_summary.empty:
            split_summary_path = args.output_dir / split / "manifest_sequences_summary.csv"
            ensure_output_dir(split_summary_path.parent)
            split_summary.to_csv(split_summary_path, index=False)
            logger.info("Saved %d %s sequence summaries to %s", len(split_summary), split, split_summary_path)
        if not split_segments.empty:
            split_segment_path = args.output_dir / split / "manifest_sequences.csv"
            ensure_output_dir(split_segment_path.parent)
            split_segments.to_csv(split_segment_path, index=False)
            logger.info("Saved %d %s sequence segments to %s", len(split_segments), split, split_segment_path)
    return summary_df


def main(cli_args: Optional[Sequence[str]] = None) -> None:
    args = parse_args(cli_args)
    build_sequences(args)


if __name__ == "__main__":
    main()
