"""Lightweight sequence colormesh visualizer.

This tool plots the saved sequence tensor (freq x frames) alongside a
binary mask (Nothing vs. outras classes). It uses the hop length and
target sample rate to derive the time axis in seconds (~10 fps by
default with hop_length=6400 and sr=64000).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot sequence colormesh from manifest_sequences.csv")
    parser.add_argument("--sequence-manifest", required=True, type=Path, help="Path to manifest_sequences.csv")
    parser.add_argument("--fragments-dir", type=Path, nargs="*", default=None, help="Optional fragment dirs (unused but kept for parity)")
    parser.add_argument("--sequence-idx", type=int, action="append", help="Sequence indices to plot (repeatable)")
    parser.add_argument(
        "--sequence-idx-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Inclusive range of sequence indices to plot",
    )
    parser.add_argument(
        "--segment-idx-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Inclusive range of segment_idx to focus on within each sequence",
    )
    parser.add_argument("--splits", nargs="*", help="Optional split filter (e.g., train val test)")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to save PNGs")
    parser.add_argument("--hop-length", type=int, default=6400, help="Hop length used when building sequences")
    parser.add_argument("--target-sr", type=int, default=64000, help="Target sample rate used when building sequences")
    parser.add_argument(
        "--max-plot-duration",
        type=float,
        default=None,
        help="Optional max duration in seconds to plot (crop timeline before rendering)",
    )
    return parser.parse_args()


def resolve_path(path_value: str, manifest_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute() and path.exists():
        return path
    if not path.is_absolute() and path.exists():
        return path
    candidate = manifest_dir / path
    if candidate.exists():
        return candidate
    return path


def build_sequence_index(args: argparse.Namespace) -> List[int]:
    indices: List[int] = []
    if args.sequence_idx:
        indices.extend(args.sequence_idx)
    if args.sequence_idx_range:
        start, end = args.sequence_idx_range
        indices.extend(list(range(start, end + 1)))
    return indices


def filter_segments(df: pd.DataFrame, segment_range: Optional[Tuple[int, int]]) -> pd.DataFrame:
    if segment_range is None:
        return df
    start, end = segment_range
    return df[(df["segment_idx"] >= start) & (df["segment_idx"] <= end)]


def slice_frames_for_segments(df: pd.DataFrame, total_frames: int, max_plot_duration: Optional[float], hop_length: int, sr: int) -> Tuple[int, int]:
    if df.empty:
        return 0, 0
    start_frame = int(df["start_frame"].min())
    end_frame = int(df["end_frame"].max())
    end_frame = min(end_frame, total_frames)
    if max_plot_duration is not None:
        budget_frames = max(1, int(max_plot_duration * sr / hop_length))
        end_frame = min(end_frame, start_frame + budget_frames)
    return start_frame, end_frame


def plot_sequence(seq_path: Path, seq_rows: pd.DataFrame, args: argparse.Namespace) -> Optional[Path]:
    if not seq_path.exists():
        logging.warning("Sequence file not found: %s", seq_path)
        return None

    sequence = np.load(seq_path)
    total_frames = sequence.shape[1]

    filtered_rows = filter_segments(seq_rows, args.segment_idx_range)
    if filtered_rows.empty:
        logging.warning("No segments remain after applying segment filter for %s", seq_path)
        return None

    start_frame, end_frame = slice_frames_for_segments(
        filtered_rows, total_frames, args.max_plot_duration, args.hop_length, args.target_sr
    )
    if end_frame <= start_frame:
        logging.warning("Computed empty frame window for %s (start=%s, end=%s)", seq_path, start_frame, end_frame)
        return None

    window = sequence[:, start_frame:end_frame]
    if window.size == 0:
        logging.warning("No data to plot for %s after slicing", seq_path)
        return None

    mask = np.zeros(window.shape[1], dtype=float)
    for _, seg in filtered_rows.iterrows():
        seg_label = str(seg.get("label", ""))
        seg_start = int(seg.get("start_frame", 0))
        seg_end = int(seg.get("end_frame", 0))
        seg_end = min(seg_end, end_frame)
        seg_start = max(seg_start, start_frame)
        if seg_end <= seg_start:
            continue
        rel_start = seg_start - start_frame
        rel_end = seg_end - start_frame
        if seg_label and seg_label != "Nothing":
            mask[rel_start:rel_end] = 1.0

    time_axis = (np.arange(window.shape[1]) + start_frame) * (args.hop_length / args.target_sr)
    freq_axis = np.arange(window.shape[0])

    split = seq_rows["split"].iloc[0] if "split" in seq_rows else "unknown"
    seq_idx = int(seq_rows["sequence_idx"].iloc[0])

    fig, (ax_feat, ax_mask) = plt.subplots(nrows=2, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [4, 1]})

    mesh = ax_feat.pcolormesh(time_axis, freq_axis, window, shading="auto", cmap="magma")
    fig.colorbar(mesh, ax=ax_feat, label="Feature value")
    ax_feat.set_ylabel("Bin de frequência/coeficiente")
    ax_feat.set_title(f"Sequência {seq_idx} ({split}) | frames {start_frame}-{end_frame}")

    ax_mask.step(time_axis, mask, where="post", color="tab:blue")
    ax_mask.set_xlabel("Tempo (s)")
    ax_mask.set_ylabel("Evento")
    ax_mask.set_yticks([0, 1])
    ax_mask.set_yticklabels(["Nothing", "Outro"])
    ax_mask.set_ylim(-0.1, 1.1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = ""
    if args.segment_idx_range:
        suffix = f"_seg{args.segment_idx_range[0]}-{args.segment_idx_range[1]}"
    out_path = args.output_dir / f"sequence_{seq_idx}_split-{split}{suffix}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logging.info("Saved plot: %s", out_path)
    return out_path


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    manifest_path = args.sequence_manifest
    if not manifest_path.exists():
        raise FileNotFoundError(f"Sequence manifest not found: {manifest_path}")
    manifest_dir = manifest_path.parent

    df = pd.read_csv(manifest_path)
    if df.empty:
        raise ValueError("Sequence manifest is empty")

    if args.splits:
        df = df[df["split"].isin(args.splits)]
        if df.empty:
            raise ValueError("No sequence segments found after applying split filter")

    target_indices = build_sequence_index(args)
    if target_indices:
        df = df[df["sequence_idx"].isin(target_indices)]
        if df.empty:
            raise ValueError("No sequence segments found after applying sequence filters")

    processed = 0
    skipped = 0
    for seq_idx, seq_rows in df.groupby("sequence_idx"):
        if seq_rows.empty:
            skipped += 1
            continue
        resolved_path = resolve_path(seq_rows["sequence_path"].iloc[0], manifest_dir)
        out = plot_sequence(resolved_path, seq_rows, args)
        if out is None:
            skipped += 1
        else:
            processed += 1

    if processed == 0:
        raise ValueError("No plots were generated; check filters and paths")

    logging.info("Finished plotting: processed=%s, skipped=%s", processed, skipped)


if __name__ == "__main__":
    main()
