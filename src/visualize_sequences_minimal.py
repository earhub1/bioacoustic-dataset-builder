"""Minimal visualizer for specific synthetic sequences.

Select sequences by index from manifest_sequences.csv, rebuild the timeline
from the original audio fragments, and render a 4-row panel (waveform,
spectrogram, stored MFCCs, binary mask) for each selected sequence.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_dataset import resolve_snippet_path
from visualize_sequences import load_fragment_metadata, reconstruct_waveform

logger = logging.getLogger(__name__)

DEFAULT_SEQUENCE_MANIFEST = Path("data/results/sequences/manifest_sequences.csv")
DEFAULT_FRAGMENTS_DIR = Path("data/results/fragments")
DEFAULT_OUTPUT_DIR = Path("data/results/sequence_viz")


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize selected synthetic sequences by index with waveform, spectrogram, MFCCs, and binary mask."
    )
    parser.add_argument(
        "--sequence-manifest",
        type=Path,
        default=DEFAULT_SEQUENCE_MANIFEST,
        help="Path to manifest_sequences.csv produced by build_dataset.py (segment-level manifest).",
    )
    parser.add_argument(
        "--fragments-dir",
        action="append",
        type=Path,
        default=None,
        help=(
            "Directory containing fragment outputs and manifest.csv. Can be passed multiple times; "
            "defaults to data/results/fragments."
        ),
    )
    parser.add_argument(
        "--sequence-idx",
        action="append",
        type=int,
        default=None,
        help="Sequence index to render (can repeat the flag to render multiple indices).",
    )
    parser.add_argument(
        "--sequence-idx-range",
        nargs=2,
        type=int,
        default=None,
        metavar=("START", "END"),
        help="Inclusive range of sequence_idx values to render (e.g., 0 4).",
    )
    parser.add_argument(
        "--segment-idx",
        action="append",
        type=int,
        default=None,
        help="Segment indices to render within the selected sequences (can repeat).",
    )
    parser.add_argument(
        "--segment-idx-range",
        nargs=2,
        type=int,
        default=None,
        metavar=("START", "END"),
        help="Inclusive range of segment_idx values to render (e.g., 10 25).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Optional list of splits to visualize (e.g., train val test).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to save the visualization PNGs.",
    )
    parser.add_argument(
        "--frame-length",
        type=int,
        default=6400,
        help="Frame length in samples used when converting frames to seconds/samples.",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=6400,
        help="Hop length in samples used when converting frames to seconds/samples.",
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=64000,
        help="Sampling rate used for waveform reconstruction and duration calculations.",
    )
    parser.add_argument(
        "--n-fft",
        type=int,
        default=1024,
        help="FFT size for the spectrogram plot.",
    )
    parser.add_argument(
        "--spectrogram-hop-length",
        type=int,
        default=512,
        help="Hop length for the spectrogram plot.",
    )
    return parser.parse_args(args=args)


def collect_sequence_indices(args: argparse.Namespace) -> set[int]:
    indices: set[int] = set()
    if args.sequence_idx:
        indices.update(args.sequence_idx)
    if args.sequence_idx_range:
        start, end = args.sequence_idx_range
        if end < start:
            raise ValueError("END must be >= START in --sequence-idx-range")
        indices.update(range(start, end + 1))
    return indices


def collect_segment_indices(args: argparse.Namespace) -> set[int]:
    indices: set[int] = set()
    if args.segment_idx:
        indices.update(args.segment_idx)
    if args.segment_idx_range:
        start, end = args.segment_idx_range
        if end < start:
            raise ValueError("END must be >= START in --segment-idx-range")
        indices.update(range(start, end + 1))
    return indices


def resolve_sequence_segments(seq_df: pd.DataFrame, manifest_dir: Path) -> list[dict]:
    segments = []
    for _, row in seq_df.sort_values("segment_idx").iterrows():
        snippet_path = resolve_snippet_path(str(row["snippet_path"]), manifest_dir)
        if not snippet_path.exists():
            raise FileNotFoundError(f"Snippet not found on disk: {snippet_path}")
        segments.append(
            {
                "label": row["label"],
                "snippet_path": str(snippet_path),
                "start_frame": int(row["start_frame"]),
                "end_frame": int(row["end_frame"]),
            }
        )
    return segments


def resolve_sequence_path(sequence_path: str, manifest_dir: Path) -> Path:
    path = Path(sequence_path)
    if path.is_absolute() or ":" in sequence_path:
        return path
    if path.exists():
        return path
    return manifest_dir / path


def render_sequence(
    sequence_idx: int,
    split: str,
    sequence_path: Path,
    segments: list[dict],
    meta_map: dict,
    output_dir: Path,
    sr: int,
    frame_length: int,
    hop_length: int,
    n_fft: int,
    spectrogram_hop_length: int,
) -> Path:
    mfcc = np.load(sequence_path)
    total_frames = int(mfcc.shape[1])

    waveform = reconstruct_waveform(
        segments=segments,
        total_frames=total_frames,
        meta_map=meta_map,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    spec = np.abs(librosa.stft(waveform, n_fft=n_fft, hop_length=spectrogram_hop_length))
    spec_db = librosa.amplitude_to_db(spec + 1e-9, ref=np.max)

    mask = np.zeros(total_frames, dtype=int)
    for seg in segments:
        start = int(seg["start_frame"])
        end = int(seg["end_frame"])
        mask[start:end] = 0 if seg["label"] == "Nothing" else 1

    time_wave = np.arange(len(waveform)) / float(sr)
    time_mask = np.arange(total_frames) * (hop_length / float(sr))

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=False)

    axes[0].plot(time_wave, waveform)
    axes[0].set_title(f"Waveform timeline (seq {sequence_idx}, split={split})")
    axes[0].set_ylabel("Amplitude")

    img = axes[1].imshow(
        spec_db,
        origin="lower",
        aspect="auto",
        extent=[time_wave[0] if len(time_wave) else 0, time_wave[-1] if len(time_wave) else 0, 0, sr / 2],
        cmap="magma",
    )
    axes[1].set_title("Spectrogram (dB)")
    axes[1].set_ylabel("Frequency (Hz)")
    fig.colorbar(img, ax=axes[1], format="%.0f dB")

    axes[2].imshow(mfcc, origin="lower", aspect="auto", cmap="viridis")
    axes[2].set_title("Stored MFCC sequence")
    axes[2].set_ylabel("Coeff")

    axes[3].step(time_mask, mask, where="post")
    axes[3].set_title("Binary mask (1 = evento, 0 = Nothing)")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Mask")
    axes[3].set_ylim(-0.1, 1.1)

    fig.tight_layout()
    safe_split = split or "unsplit"
    out_path = output_dir / f"sequence_{sequence_idx}_{safe_split}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main(cli_args: Optional[Sequence[str]] = None) -> None:
    args = parse_args(cli_args)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    fragments_dirs = args.fragments_dir or [DEFAULT_FRAGMENTS_DIR]
    meta_map = load_fragment_metadata(fragments_dirs)

    seq_manifest = pd.read_csv(args.sequence_manifest)
    if args.splits:
        seq_manifest = seq_manifest[seq_manifest["split"].isin(args.splits)]

    selected_indices = collect_sequence_indices(args)
    if selected_indices:
        seq_manifest = seq_manifest[seq_manifest["sequence_idx"].isin(selected_indices)]

    segment_indices = collect_segment_indices(args)
    if segment_indices:
        seq_manifest = seq_manifest[seq_manifest["segment_idx"].isin(segment_indices)]

    if seq_manifest.empty:
        raise ValueError("No sequence segments found after applying filters.")

    manifest_dir = args.sequence_manifest.parent
    grouped = seq_manifest.groupby(["sequence_idx", "split"])

    for (seq_idx, split), group in grouped:
        sequence_path = resolve_sequence_path(str(group["sequence_path"].iloc[0]), manifest_dir)
        if not sequence_path.exists():
            raise FileNotFoundError(f"Sequence file not found: {sequence_path}")

        segments = resolve_sequence_segments(group, manifest_dir)
        out_path = render_sequence(
            sequence_idx=int(seq_idx),
            split=str(split),
            sequence_path=sequence_path,
            segments=segments,
            meta_map=meta_map,
            output_dir=args.output_dir,
            sr=args.target_sr,
            frame_length=args.frame_length,
            hop_length=args.hop_length,
            n_fft=args.n_fft,
            spectrogram_hop_length=args.spectrogram_hop_length,
        )
        logger.info("Saved visualization to %s", out_path)


if __name__ == "__main__":
    main()
