# build_dataset_balanced.py
# -*- coding: utf-8 -*-
"""
Build 1 balanced (50/50 by frames) sequence per split from fragment manifests.

Usage example (similar to your previous command):
python src/build_dataset_balanced.py \
  --fragments-dir data/results/fragments_combined/fragments_32khz \
  --event-label G01 \
  --output-dir data/results/sequences_low_freq \
  --seed 42 \
  --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1

Assumptions:
- Each fragments-dir contains a manifest.csv with columns:
  - label
  - n_frames
  - snippet_path
- Each row's snippet_path points to a .npy file with shape (n_mels, frames) or (frames, n_mels)

What it guarantees:
- For each split (train/val/test): 1 sequence with frames(G01) == frames(Nothing) (50/50)
- All event fragments in that split are used (no dropping of G01)
- Excess Nothing is ignored (removed by not being selected)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_FRAGMENTS_DIR = Path("data/results/fragments")
DEFAULT_OUTPUT_DIR = Path("data/results/sequences_balanced")


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build 1 balanced sequence per split (50/50 by frames) from fragments manifests."
    )
    p.add_argument(
        "--fragments-dir",
        action="append",
        type=Path,
        default=None,
        help="Directory containing manifest.csv. Can be passed multiple times.",
    )
    p.add_argument(
        "--event-label",
        type=str,
        default="G01",
        help="Event label to balance against Nothing (default: G01).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for sequences and manifests.",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--train-ratio", type=float, default=0.7, help="Train ratio (default 0.7).")
    p.add_argument("--val-ratio", type=float, default=0.2, help="Val ratio (default 0.2).")
    p.add_argument("--test-ratio", type=float, default=0.1, help="Test ratio (default 0.1).")
    p.add_argument(
        "--expected-n-mels",
        type=int,
        default=64,
        help="Expected number of mel bins in stored fragments (default 64).",
    )
    p.add_argument(
        "--target-sr",
        type=int,
        default=64000,
        help="Sampling rate used when interpreting frames into seconds (default 64000).",
    )
    p.add_argument(
        "--frame-length",
        type=int,
        default=6400,
        help="Frame length in samples (default 6400).",
    )
    p.add_argument(
        "--hop-length",
        type=int,
        default=6400,
        help="Hop length in samples (default 6400).",
    )
    return p.parse_args(args=args)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def frames_to_seconds(n_frames: int, sr: int, frame_length: int, hop_length: int) -> float:
    if n_frames <= 0:
        return 0.0
    return ((n_frames - 1) * hop_length + frame_length) / float(sr)


def resolve_snippet_path(snippet: str, manifest_dir: Path) -> Path:
    normalized = snippet.replace("\\", "/")
    path = Path(normalized)

    if path.is_absolute() or ":" in snippet:
        return path

    # If it already exists as given, keep it
    if path.exists():
        return path

    # Otherwise treat as relative to manifest_dir
    return manifest_dir / path


def normalize_feature_shape(features: np.ndarray, expected_n_mels: int, snippet_path: Path) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError(f"Fragment {snippet_path} has shape {features.shape}; expected 2D matrix.")
    if features.shape[0] == expected_n_mels:
        return features
    if features.shape[1] == expected_n_mels:
        logger.warning("Transposing %s from %s to (n_mels x frames)", snippet_path, features.shape)
        return features.T
    raise ValueError(
        f"Fragment {snippet_path} has incompatible shape {features.shape}; "
        f"expected n_mels={expected_n_mels} on one dimension."
    )


def load_manifests(fragment_dirs: List[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for frag_dir in fragment_dirs:
        manifest_path = frag_dir / "manifest.csv"
        if not manifest_path.exists():
            logger.warning("Skipping %s (missing manifest.csv)", frag_dir)
            continue
        df = pd.read_csv(manifest_path)
        df["_manifest_dir"] = str(manifest_path.parent)
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No manifest.csv files found in provided --fragments-dir paths.")

    data = pd.concat(frames, ignore_index=True)
    if "label" not in data.columns or "n_frames" not in data.columns or "snippet_path" not in data.columns:
        raise ValueError("manifest.csv must contain columns: label, n_frames, snippet_path")

    data["n_frames"] = pd.to_numeric(data["n_frames"], errors="coerce")
    if data["n_frames"].isna().any():
        raise ValueError("All rows must have a valid numeric n_frames.")
    return data


def validate_split_ratios(train: float, val: float, test: float) -> Tuple[List[str], np.ndarray]:
    if min(train, val, test) < 0:
        raise ValueError("Split ratios must be non-negative.")
    s = train + val + test
    if not np.isclose(s, 1.0):
        raise ValueError("train-ratio + val-ratio + test-ratio must sum to 1.0.")
    labels = ["train", "val", "test"]
    probs = np.array([train, val, test], dtype=float)
    probs = probs / probs.sum()
    return labels, probs


def stratified_split(df: pd.DataFrame, split_labels: List[str], split_probs: np.ndarray, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["split"] = None

    for label, g in out.groupby("label"):
        idx = g.index.to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        counts = [int(n * p) for p in split_probs]
        counts[-1] += n - sum(counts)

        cursor = 0
        for split, c in zip(split_labels, counts):
            sel = idx[cursor : cursor + c]
            out.loc[sel, "split"] = split
            cursor += c

    if out["split"].isna().any():
        raise RuntimeError("Failed to assign splits to all rows.")
    return out


def load_fragment(row: pd.Series, expected_n_mels: int) -> Tuple[np.ndarray, Path]:
    manifest_dir = Path(str(row["_manifest_dir"]))
    snippet_path = resolve_snippet_path(str(row["snippet_path"]), manifest_dir)
    if not snippet_path.exists():
        raise FileNotFoundError(f"Missing snippet: {snippet_path}")
    feat = np.load(snippet_path)
    feat = normalize_feature_shape(feat, expected_n_mels, snippet_path)
    if feat.shape[1] <= 0:
        raise ValueError(f"Non-positive frames in {snippet_path}")
    return feat, snippet_path


def build_one_sequence_50_50(
    df_split: pd.DataFrame,
    event_label: str,
    expected_n_mels: int,
    seed: int,
) -> Tuple[np.ndarray, List[dict], Dict[str, int]]:
    """
    Build a single sequence where:
      - All event fragments in df_split are included (shuffled)
      - Nothing fragments are added until frames(Nothing) == frames(event)
      - Nothing can be reused with replacement if needed
      - The last Nothing may be truncated to fit exactly
    """
    rng = np.random.default_rng(seed)

    events = df_split[df_split["label"] == event_label].copy()
    nothings = df_split[df_split["label"] == "Nothing"].copy()

    if events.empty:
        raise ValueError(f"Split has no event fragments for '{event_label}'. Cannot build 50/50.")
    if nothings.empty:
        raise ValueError("Split has no Nothing fragments. Cannot build 50/50.")

    # Shuffle order deterministically
    events = events.sample(frac=1.0, random_state=seed)
    nothings = nothings.sample(frac=1.0, random_state=seed)

    chunks: List[np.ndarray] = []
    segments: List[dict] = []
    cur = 0

    # 1) Include all events (no truncation of events)
    for _, row in events.iterrows():
        try:
            feat, path = load_fragment(row, expected_n_mels)
        except Exception as e:
            # If an event fragment is missing/corrupt, we skip with warning.
            # (If you want strict failure, replace this with raise.)
            logger.warning("Skipping event fragment due to error: %s", e)
            continue
        n_frames = int(feat.shape[1])
        chunks.append(feat)
        segments.append(
            {
                "label": event_label,
                "snippet_path": str(path),
                "start_frame": cur,
                "end_frame": cur + n_frames,
                "truncated": False,
            }
        )
        cur += n_frames

    event_frames_used = cur
    if event_frames_used <= 0:
        raise RuntimeError("No valid event fragments could be loaded; sequence cannot be built.")

    target_nothing_frames = event_frames_used

    # 2) Add Nothing until it matches event frames
    nothing_used = 0
    nothing_rows = [r for _, r in nothings.iterrows()]
    ptr = 0

    while nothing_used < target_nothing_frames:
        if ptr >= len(nothing_rows):
            # Replacement: reshuffle and reuse Nothing
            rng.shuffle(nothing_rows)
            ptr = 0

        row = nothing_rows[ptr]
        ptr += 1

        try:
            feat, path = load_fragment(row, expected_n_mels)
        except Exception as e:
            logger.warning("Skipping nothing fragment due to error: %s", e)
            continue

        n_frames = int(feat.shape[1])
        remaining = target_nothing_frames - nothing_used

        truncated = False
        if n_frames > remaining:
            feat = feat[:, :remaining]
            n_frames = int(feat.shape[1])
            truncated = True

        chunks.append(feat)
        segments.append(
            {
                "label": "Nothing",
                "snippet_path": str(path),
                "start_frame": cur,
                "end_frame": cur + n_frames,
                "truncated": truncated,
            }
        )
        cur += n_frames
        nothing_used += n_frames

    combined = np.concatenate(chunks, axis=1)

    meta = {
        "frames_event": int(event_frames_used),
        "frames_nothing": int(nothing_used),
        "total_frames": int(combined.shape[1]),
    }
    return combined, segments, meta


def save_sequence_and_manifests(
    output_dir: Path,
    split: str,
    features: np.ndarray,
    segments: List[dict],
    meta: Dict[str, int],
    sr: int,
    frame_length: int,
    hop_length: int,
    expected_n_mels: int,
) -> Tuple[dict, List[dict]]:
    split_dir = output_dir / split
    ensure_dir(split_dir)

    seq_path = split_dir / "sequence_0.npy"
    np.save(seq_path, features)

    total_frames = int(features.shape[1])
    total_duration_s = frames_to_seconds(total_frames, sr, frame_length, hop_length)

    summary = {
        "sequence_path": str(seq_path),
        "sequence_idx": 0,
        "split": split,
        "total_frames": total_frames,
        "total_duration_s": total_duration_s,
        "mel_bins": expected_n_mels,
        "frames_event": meta["frames_event"],
        "frames_nothing": meta["frames_nothing"],
        "pct_event": meta["frames_event"] / total_frames if total_frames else 0.0,
        "pct_nothing": meta["frames_nothing"] / total_frames if total_frames else 0.0,
        "n_segments": len(segments),
    }

    seg_rows: List[dict] = []
    for i, seg in enumerate(segments):
        start = int(seg["start_frame"])
        end = int(seg["end_frame"])
        dur_frames = max(end - start, 0)
        seg_rows.append(
            {
                "sequence_path": str(seq_path),
                "sequence_idx": 0,
                "split": split,
                "segment_idx": i,
                "label": seg["label"],
                "snippet_path": seg["snippet_path"],
                "start_frame": start,
                "end_frame": end,
                "duration_frames": dur_frames,
                "start_s": frames_to_seconds(start, sr, frame_length, hop_length),
                "end_s": frames_to_seconds(end, sr, frame_length, hop_length),
                "duration_s": frames_to_seconds(dur_frames, sr, frame_length, hop_length),
                "truncated": bool(seg.get("truncated", False)),
            }
        )

    # per-split manifests
    pd.DataFrame([summary]).to_csv(split_dir / "manifest_sequences_summary.csv", index=False)
    pd.DataFrame(seg_rows).to_csv(split_dir / "manifest_sequences.csv", index=False)

    return summary, seg_rows


def main(cli_args: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args(cli_args)

    fragment_dirs = args.fragments_dir or [DEFAULT_FRAGMENTS_DIR]
    split_labels, split_probs = validate_split_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    df = load_manifests(fragment_dirs)

    # Keep only the 2-class world: event_label + Nothing
    df = df[df["label"].isin([args.event_label, "Nothing"])].copy()
    if df.empty:
        raise ValueError(f"No rows found for labels [{args.event_label}, Nothing]. Check your manifests.")

    # Split stratified by label
    df = stratified_split(df, split_labels, split_probs, seed=args.seed)

    ensure_dir(args.output_dir)
    split_manifest_path = args.output_dir / "manifest_split.csv"
    df.to_csv(split_manifest_path, index=False)
    logger.info("Saved split manifest: %s (rows=%d)", split_manifest_path, len(df))

    all_summaries: List[dict] = []
    all_segments: List[dict] = []

    # One sequence per split
    for split in split_labels:
        df_split = df[df["split"] == split].copy()
        if df_split.empty:
            logger.warning("Split '%s' is empty, skipping.", split)
            continue

        # Build 50/50 by frames (event is anchor)
        features, segments, meta = build_one_sequence_50_50(
            df_split=df_split,
            event_label=args.event_label,
            expected_n_mels=args.expected_n_mels,
            seed=args.seed + (0 if split == "train" else 1 if split == "val" else 2),
        )

        # Save
        summary, seg_rows = save_sequence_and_manifests(
            output_dir=args.output_dir,
            split=split,
            features=features,
            segments=segments,
            meta=meta,
            sr=args.target_sr,
            frame_length=args.frame_length,
            hop_length=args.hop_length,
            expected_n_mels=args.expected_n_mels,
        )

        all_summaries.append(summary)
        all_segments.extend(seg_rows)

        logger.info(
            "Split %s: total_frames=%d | event=%d | nothing=%d | pct_event=%.4f | pct_nothing=%.4f",
            split,
            summary["total_frames"],
            summary["frames_event"],
            summary["frames_nothing"],
            summary["pct_event"],
            summary["pct_nothing"],
        )

    # Global manifests
    summary_path = args.output_dir / "manifest_sequences_summary.csv"
    segments_path = args.output_dir / "manifest_sequences.csv"
    pd.DataFrame(all_summaries).to_csv(summary_path, index=False)
    pd.DataFrame(all_segments).to_csv(segments_path, index=False)
    logger.info("Saved global summary: %s", summary_path)
    logger.info("Saved global segments: %s", segments_path)


if __name__ == "__main__":
    main()
