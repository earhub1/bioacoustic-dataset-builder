# build_dataset_balanced.py
# -*- coding: utf-8 -*-
"""
Build 1 balanced (50/50 by frames) *INTERLEAVED* sequence per split from fragment manifests.

Usage example:
python src/build_dataset_balanced.py \
  --fragments-dir data/results/fragments_combined/fragments_32khz \
  --event-label G01 \
  --expected-n-mels 13 \
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
- The ORDER is INTERLEAVED (reproducible by seed), avoiding the "all G01 then all Nothing" artifact
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
        description="Build 1 balanced, interleaved sequence per split (50/50 by frames) from fragments manifests."
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
    p.add_argument(
        "--max-run-same-label",
        type=int,
        default=3,
        help="Max consecutive chunks of the same label in the assembled sequence (default 3).",
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

    if path.exists():
        return path

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


def build_one_sequence_50_50_interleaved(
    df_split: pd.DataFrame,
    event_label: str,
    expected_n_mels: int,
    seed: int,
    max_run_same_label: int = 3,
) -> Tuple[np.ndarray, List[dict], Dict[str, int]]:
    """
    Build a single sequence 50/50 by frames, but INTERLEAVED.

    Guarantees:
      - Uses ALL event fragments in this split as anchor (no dropping of G01)
      - Adds Nothing fragments until frames(Nothing) == frames(Event)
      - Nothing can be reused (replacement) and may be truncated at the end
      - Interleaves by deficit scheduling + run limit (reproducible via seed)
    """
    rng = np.random.default_rng(seed)

    events_df = df_split[df_split["label"] == event_label].copy()
    nothings_df = df_split[df_split["label"] == "Nothing"].copy()

    if events_df.empty:
        raise ValueError(f"Split has no event fragments for '{event_label}'. Cannot build 50/50.")
    if nothings_df.empty:
        raise ValueError("Split has no Nothing fragments. Cannot build 50/50.")

    # Deterministic shuffle for stable ordering
    events_df = events_df.sample(frac=1.0, random_state=seed)
    nothings_df = nothings_df.sample(frac=1.0, random_state=seed)

    # ---- Load ALL event fragments first (anchor) into memory as chunks ----
    event_chunks: List[Tuple[np.ndarray, str, int]] = []  # (feat, path, n_frames)
    for _, row in events_df.iterrows():
        try:
            feat, path = load_fragment(row, expected_n_mels)
        except Exception as e:
            logger.warning("Skipping event fragment due to error: %s", e)
            continue
        n_frames = int(feat.shape[1])
        if n_frames <= 0:
            continue
        event_chunks.append((feat, str(path), n_frames))

    if not event_chunks:
        raise RuntimeError("No valid event fragments could be loaded; sequence cannot be built.")

    total_event_frames = sum(n for _, _, n in event_chunks)
    target_event = total_event_frames
    target_nothing = total_event_frames  # 50/50

    # ---- Prepare Nothing source rows for sampling (replacement allowed) ----
    nothing_rows = [r for _, r in nothings_df.iterrows()]
    if not nothing_rows:
        raise RuntimeError("No Nothing rows available.")

    rng.shuffle(nothing_rows)
    nothing_ptr = 0

    def next_nothing_chunk(need_frames: int) -> Tuple[np.ndarray, str, int, bool]:
        """Get a Nothing chunk; truncate if needed to exactly fit remaining budget."""
        nonlocal nothing_ptr, nothing_rows

        if nothing_ptr >= len(nothing_rows):
            rng.shuffle(nothing_rows)
            nothing_ptr = 0

        row = nothing_rows[nothing_ptr]
        nothing_ptr += 1

        feat, path = load_fragment(row, expected_n_mels)
        n_frames = int(feat.shape[1])
        truncated = False

        if n_frames > need_frames:
            feat = feat[:, :need_frames]
            n_frames = int(feat.shape[1])
            truncated = True

        return feat, str(path), n_frames, truncated

    # ---- Interleaving scheduler ----
    chunks: List[np.ndarray] = []
    segments: List[dict] = []
    cur = 0

    used_event = 0
    used_nothing = 0
    i_event = 0

    last_label: Optional[str] = None
    run_len = 0

    # Build until both budgets are satisfied
    while used_event < target_event or used_nothing < target_nothing:
        remaining_event = target_event - used_event
        remaining_nothing = target_nothing - used_nothing

        # Prefer the label with the largest remaining budget (deficit scheduling)
        if remaining_event <= 0:
            preferred = "Nothing"
        elif remaining_nothing <= 0:
            preferred = event_label
        else:
            if remaining_event > remaining_nothing:
                preferred = event_label
            elif remaining_nothing > remaining_event:
                preferred = "Nothing"
            else:
                preferred = event_label if rng.random() < 0.5 else "Nothing"

        # Apply run-limit to avoid long streaks
        if max_run_same_label is not None and max_run_same_label > 0:
            if last_label is not None and run_len >= max_run_same_label:
                if last_label == event_label and remaining_nothing > 0:
                    preferred = "Nothing"
                elif last_label == "Nothing" and remaining_event > 0:
                    preferred = event_label

        # If preferred is event but we ran out of event chunks, force Nothing
        if preferred == event_label and i_event >= len(event_chunks):
            preferred = "Nothing"

        # If preferred is Nothing but its budget is done, force event if available
        if preferred == "Nothing" and remaining_nothing <= 0 and i_event < len(event_chunks):
            preferred = event_label

        # Emit next chunk
        if preferred == event_label:
            feat, path, n_frames = event_chunks[i_event]
            i_event += 1

            # Safety guard: don't exceed event budget (normally shouldn't happen)
            truncated = False
            if used_event + n_frames > target_event:
                keep = target_event - used_event
                if keep <= 0:
                    continue
                feat = feat[:, :keep]
                n_frames = int(feat.shape[1])
                truncated = True

            chunks.append(feat)
            segments.append(
                {
                    "label": event_label,
                    "snippet_path": path,
                    "start_frame": cur,
                    "end_frame": cur + n_frames,
                    "truncated": truncated,
                }
            )
            cur += n_frames
            used_event += n_frames

        else:
            if remaining_nothing <= 0:
                break

            feat, path, n_frames, truncated = next_nothing_chunk(remaining_nothing)
            if n_frames <= 0:
                continue

            chunks.append(feat)
            segments.append(
                {
                    "label": "Nothing",
                    "snippet_path": path,
                    "start_frame": cur,
                    "end_frame": cur + n_frames,
                    "truncated": truncated,
                }
            )
            cur += n_frames
            used_nothing += n_frames

        # Run tracking
        if preferred == last_label:
            run_len += 1
        else:
            last_label = preferred
            run_len = 1

    combined = np.concatenate(chunks, axis=1)

    meta = {
        "frames_event": int(used_event),
        "frames_nothing": int(used_nothing),
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

    split_seed_offset = {"train": 0, "val": 1, "test": 2}

    # One sequence per split
    for split in split_labels:
        df_split = df[df["split"] == split].copy()
        if df_split.empty:
            logger.warning("Split '%s' is empty, skipping.", split)
            continue

        # Build 50/50 by frames with INTERLEAVING
        features, segments, meta = build_one_sequence_50_50_interleaved(
            df_split=df_split,
            event_label=args.event_label,
            expected_n_mels=args.expected_n_mels,
            seed=args.seed + split_seed_offset.get(split, 0),
            max_run_same_label=args.max_run_same_label,
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
            "Split %s: total_frames=%d | event=%d | nothing=%d | pct_event=%.4f | pct_nothing=%.4f | segments=%d",
            split,
            summary["total_frames"],
            summary["frames_event"],
            summary["frames_nothing"],
            summary["pct_event"],
            summary["pct_nothing"],
            summary["n_segments"],
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
