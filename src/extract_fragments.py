"""Utility for extracting annotated audio fragments and MFCC features.

This script reads a CSV file containing detection metadata and extracts the
corresponding audio fragments, optionally downsampling them before computing
MFCC features. Each fragment is saved as a ``.npy`` file inside a label-named
subdirectory. A companion CSV manifest summarises the saved fragments.

Example usage
-------------
python src/extract_fragments.py \
    --csv-path data/events/labels_0_30kHz_reapath.csv \
    --target-sr 64000 \
    --output-dir data/results/fragments
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import librosa
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


DEFAULT_CSV_PATH = Path("data/events/labels_0_30kHz_reapath.csv")
DEFAULT_OUTPUT_DIR = Path("data/results/fragments")


class FragmentExtractionError(Exception):
    """Raised when a fragment cannot be extracted."""


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract annotated audio fragments and MFCC features.")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Path to the CSV file containing detections.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where fragment .npy files and manifest.csv will be stored.",
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=64000,
        help="Target sampling rate for loaded audio (downsample if needed).",
    )
    parser.add_argument(
        "--n-mels",
        type=int,
        default=9,
        help="Number of MFCC coefficients to compute for each frame.",
    )
    parser.add_argument(
        "--window",
        type=str,
        default="hann",
        help="Window function to apply when computing MFCCs.",
    )
    parser.add_argument(
        "--frame-length",
        type=int,
        default=6400,
        help="Frame length (in samples) at the target sampling rate.",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=6400,
        help="Hop length (in samples) at the target sampling rate.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of fragments to extract (random selection).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when selecting a subset of rows.",
    )
    return parser.parse_args(args=args)


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def resolve_source_path(raw_path: str, csv_dir: Path) -> Path:
    source_path = Path(raw_path)
    if not source_path.is_absolute() and ":" not in str(raw_path):
        return csv_dir / source_path
    return source_path


def load_audio_fragment(filepath: Path, onset_s: float, offset_s: float, target_sr: int) -> np.ndarray:
    if not filepath.exists():
        raise FragmentExtractionError(f"Audio file does not exist: {filepath}")

    duration = max(offset_s - onset_s, 0)
    if duration <= 0:
        raise FragmentExtractionError(f"Invalid duration for onset {onset_s} and offset {offset_s} in {filepath}")

    audio, _ = librosa.load(
        filepath,
        sr=target_sr,
        mono=False,
        offset=onset_s,
        duration=duration,
    )

    if audio.ndim == 1:
        return audio

    if audio.ndim == 2:
        return audio[0]

    raise FragmentExtractionError(f"Unsupported audio shape {audio.shape} in {filepath}")


def compute_mfcc(
    audio: np.ndarray,
    target_sr: int,
    n_mels: int,
    frame_length: int,
    hop_length: int,
    window: str,
) -> np.ndarray:
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=target_sr,
        n_mfcc=n_mels,
        n_fft=frame_length,
        hop_length=hop_length,
        window=window,
    )
    return mfcc


def save_fragment(
    features: np.ndarray,
    output_dir: Path,
    label: str,
    base_filename: str,
    index: int,
) -> Path:
    label_dir = output_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)
    fragment_path = label_dir / f"{Path(base_filename).stem}_idx{index}.npy"
    np.save(fragment_path, features)
    return fragment_path


def validate_non_event_args(args: argparse.Namespace) -> None:
    """Validation retained for backward compatibility with non-event utilities."""


def choose_non_event_duration(args: argparse.Namespace, rng: np.random.Generator, default: float) -> float:
    if args.non_event_duration is not None:
        return float(args.non_event_duration)
    if args.non_event_duration_range is not None:
        low, high = args.non_event_duration_range
        return float(rng.uniform(low, high))
    return default


def find_free_intervals(events: List[tuple], audio_duration: float, min_duration: float) -> List[tuple]:
    if audio_duration <= 0:
        return []

    sorted_events = sorted(events, key=lambda x: x[0])
    free: List[tuple] = []
    cursor = 0.0

    for onset, offset in sorted_events:
        onset = max(onset, 0.0)
        offset = max(offset, onset)
        if onset - cursor >= min_duration:
            free.append((cursor, onset))
        cursor = max(cursor, offset)

    if audio_duration - cursor >= min_duration:
        free.append((cursor, audio_duration))
    return [interval for interval in free if interval[1] - interval[0] >= min_duration]


def process_row(
    row: pd.Series,
    output_dir: Path,
    target_sr: int,
    n_mels: int,
    frame_length: int,
    hop_length: int,
    window: str,
    csv_dir: Path,
) -> Optional[dict]:
    raw_path = row.get("filepath") or row.get("file")
    if not raw_path:
        raise FragmentExtractionError("Row is missing a filepath entry.")
    source_path = resolve_source_path(str(raw_path), csv_dir)

    onset_s = float(row["onset_s"])
    offset_s = float(row["offset_s"])
    label = str(row["label"])

    try:
        audio = load_audio_fragment(source_path, onset_s, offset_s, target_sr)
        mfcc = compute_mfcc(audio, target_sr, n_mels, frame_length, hop_length, window)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Skipping index %s due to error: %s", row.name, exc)
        return None

    fragment_path = save_fragment(mfcc, output_dir, label, row.get("file", "fragment"), row.name)
    n_frames = mfcc.shape[1]
    duration = offset_s - onset_s

    return {
        "index": int(row.name),
        "snippet_path": str(fragment_path),
        "label": label,
        "source_filepath": str(source_path),
        "onset_s": onset_s,
        "offset_s": offset_s,
        "duration_s": duration,
        "n_frames": int(n_frames),
    }


def select_rows(df: pd.DataFrame, limit: Optional[int], seed: int) -> pd.DataFrame:
    if limit is None or limit >= len(df):
        return df
    return df.sample(n=limit, random_state=seed)


def build_events_by_file(df: pd.DataFrame, csv_dir: Path) -> dict:
    events_by_file: dict = {}
    for _, row in df.iterrows():
        raw_path = row.get("filepath") or row.get("file")
        if not raw_path:
            continue
        source_path = resolve_source_path(str(raw_path), csv_dir)
        events_by_file.setdefault(source_path, []).append(
            (float(row["onset_s"]), float(row["offset_s"]))
        )
    return events_by_file


def extract_fragments(args: argparse.Namespace) -> pd.DataFrame:
    ensure_output_dir(args.output_dir)
    df = pd.read_csv(args.csv_path)
    selected = select_rows(df, args.limit, args.seed)
    csv_dir = args.csv_path.parent
    rng = np.random.default_rng(args.seed)

    records: List[dict] = []
    for _, row in selected.iterrows():
        record = process_row(
            row=row,
            output_dir=args.output_dir,
            target_sr=args.target_sr,
            n_mels=args.n_mels,
            frame_length=args.frame_length,
            hop_length=args.hop_length,
            window=args.window,
            csv_dir=csv_dir,
        )
        if record:
            records.append(record)

    manifest = pd.DataFrame(records)
    manifest_path = args.output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    logger.info("Saved manifest with %d entries to %s", len(manifest), manifest_path)
    return manifest


def main(cli_args: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args(cli_args)
    extract_fragments(args)


if __name__ == "__main__":
    main()
