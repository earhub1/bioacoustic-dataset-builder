"""CLI to generate a manifest of non-event ("Nothing") fragments.

This tool inspects the annotated detection CSV, derives free intervals per
source audio file, and samples onsets/durations for background snippets.
The resulting manifest can be used with ``extract_fragments.py`` to
materialize MFCCs for the generated "Nothing" segments, or combined with the
original detections for a unified manifest.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import librosa
import numpy as np
import pandas as pd

from extract_fragments import (
    DEFAULT_CSV_PATH,
    build_events_by_file,
    choose_non_event_duration,
    find_free_intervals,
    resolve_source_path,
    validate_non_event_args,
)

logger = logging.getLogger(__name__)

DEFAULT_NOTHING_MANIFEST = Path("data/events/manifest_nothing.csv")
DEFAULT_COMBINED_MANIFEST = Path("data/events/manifest_combined.csv")


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a manifest of non-event (Nothing) fragments."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Path to the CSV file containing detections.",
    )
    parser.add_argument(
        "--non-event-count",
        type=int,
        default=0,
        help="Number of 'Nothing' fragments to sample per audio file.",
    )
    parser.add_argument(
        "--non-event-duration",
        type=float,
        default=None,
        help="Fixed duration (in seconds) for each 'Nothing' fragment.",
    )
    parser.add_argument(
        "--non-event-duration-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Duration range (seconds) to sample 'Nothing' fragments. Ignored if --non-event-duration is set.",
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=64000,
        help="Sampling rate used to derive frame durations.",
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling of intervals.",
    )
    parser.add_argument(
        "--nothing-manifest-path",
        type=Path,
        default=DEFAULT_NOTHING_MANIFEST,
        help="Path to write the Nothing-only manifest CSV.",
    )
    parser.add_argument(
        "--combined-manifest-path",
        type=Path,
        default=None,
        help="Optional path to write a manifest that combines detections and Nothing entries.",
    )
    return parser.parse_args(args=args)


def estimate_n_frames(duration_s: float, target_sr: int, frame_length: int, hop_length: int) -> int:
    total_samples = max(int(round(duration_s * target_sr)), 0)
    if total_samples <= 0:
        return 0
    if total_samples <= frame_length:
        return 1
    return int(np.floor((total_samples - frame_length) / hop_length) + 1)


def ensure_output_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def sample_nothing_records(
    df: pd.DataFrame,
    csv_dir: Path,
    non_event_count: int,
    target_sr: int,
    frame_length: int,
    hop_length: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> List[dict]:
    if non_event_count <= 0:
        return []

    default_duration = frame_length / float(target_sr)
    min_duration = args.non_event_duration or (
        args.non_event_duration_range[0] if args.non_event_duration_range else default_duration
    )

    events_by_file = build_events_by_file(df, csv_dir)
    records: List[dict] = []
    bg_counter = 0

    for raw_path, events in events_by_file.items():
        source_path = resolve_source_path(str(raw_path), csv_dir)
        try:
            audio_duration = librosa.get_duration(filename=str(source_path))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping %s due to duration error: %s", source_path, exc)
            continue

        free_intervals = find_free_intervals(events, audio_duration, min_duration)
        if not free_intervals:
            logger.info("No free intervals long enough for non-event fragments in %s", source_path)
            continue

        for _ in range(non_event_count):
            duration = choose_non_event_duration(args, rng, default_duration)

            attempts = 0
            placed = False
            max_attempts = max(3 * len(free_intervals), 10)
            while attempts < max_attempts and not placed:
                interval = free_intervals[rng.integers(0, len(free_intervals))]
                available = interval[1] - interval[0]
                attempts += 1
                if available < duration or available < min_duration:
                    continue

                onset = float(interval[0] + rng.uniform(0, available - duration)) if available > duration else float(interval[0])
                offset = onset + duration
                n_frames = estimate_n_frames(duration, target_sr, frame_length, hop_length)

                records.append(
                    {
                        "index": -1,
                        "label": "Nothing",
                        "filepath": str(source_path),
                        "source_filepath": str(source_path),
                        "onset_s": onset,
                        "offset_s": offset,
                        "duration_s": float(duration),
                        "n_frames": int(n_frames),
                        "id": f"bg_{bg_counter}",
                    }
                )

                bg_counter += 1
                placed = True

            if not placed:
                logger.warning(
                    "Unable to place a non-event fragment of %.3f s in %s after %d attempts",
                    duration,
                    source_path,
                    max_attempts,
                )

    return records


def write_manifests(
    df: pd.DataFrame,
    nothing_records: List[dict],
    nothing_manifest_path: Path,
    combined_manifest_path: Optional[Path],
) -> None:
    ensure_output_path(nothing_manifest_path)
    nothing_df = pd.DataFrame(nothing_records)
    nothing_df.to_csv(nothing_manifest_path, index=False)
    logger.info("Saved Nothing manifest with %d entries to %s", len(nothing_df), nothing_manifest_path)

    if combined_manifest_path is not None:
        ensure_output_path(combined_manifest_path)
        combined_df = pd.concat([df, nothing_df], ignore_index=True)
        combined_df.to_csv(combined_manifest_path, index=False)
        logger.info("Saved combined manifest with %d entries to %s", len(combined_df), combined_manifest_path)


def generate_nothing_manifest(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    validate_non_event_args(args)
    df = pd.read_csv(args.csv_path)
    csv_dir = args.csv_path.parent
    rng = np.random.default_rng(args.seed)

    records = sample_nothing_records(
        df=df,
        csv_dir=csv_dir,
        non_event_count=args.non_event_count,
        target_sr=args.target_sr,
        frame_length=args.frame_length,
        hop_length=args.hop_length,
        args=args,
        rng=rng,
    )
    write_manifests(
        df=df,
        nothing_records=records,
        nothing_manifest_path=args.nothing_manifest_path,
        combined_manifest_path=args.combined_manifest_path,
    )


def main(cli_args: Optional[List[str]] = None) -> None:
    args = parse_args(cli_args)
    generate_nothing_manifest(args)


if __name__ == "__main__":
    main()
