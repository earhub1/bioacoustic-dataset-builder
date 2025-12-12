"""Utility for extracting annotated audio fragments and acoustic features.

This script reads a CSV file containing detection metadata and extracts the
corresponding audio fragments, optionally downsampling them before computing
MFCC or Mel-spectrogram features. Each fragment is saved as a ``.npy`` file inside a label-named
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
    parser = argparse.ArgumentParser(description="Extract annotated audio fragments and acoustic features.")
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
        "--feature-type",
        choices=["mfcc", "melspectrogram"],
        default="mfcc",
        help="Type of acoustic feature to extract for each fragment.",
    )
    parser.add_argument(
        "--n-mels",
        "--n-mfcc",
        dest="n_mfcc",
        type=int,
        default=9,
        help="Number of MFCC coefficients to compute for each frame (alias --n-mfcc).",
    )
    parser.add_argument(
        "--mel-bins",
        type=int,
        default=64,
        help="Number of Mel bins for the mel spectrogram (when --feature-type=melspectrogram).",
    )
    parser.add_argument(
        "--mel-nfft",
        type=int,
        default=1024,
        help="FFT size used when computing the mel spectrogram (when --feature-type=melspectrogram).",
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
        "--max-per-label",
        type=int,
        default=None,
        help=(
            "Optional cap on rows per label (applied to all labels except 'Nothing'); "
            "use with --seed for reproducible sampling."
        ),
    )
    parser.add_argument(
        "--max-nothing",
        type=int,
        default=None,
        help=(
            "Optional cap on rows for the 'Nothing' label; if unset, falls back to --max-per-label "
            "when that flag is provided."
        ),
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=None,
        help="Minimum event duration in seconds; rows shorter than this are skipped.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Maximum event duration in seconds; rows longer than this are skipped.",
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
    n_mfcc: int,
    frame_length: int,
    hop_length: int,
    window: str,
) -> np.ndarray:
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=target_sr,
        n_mfcc=n_mfcc,
        n_fft=frame_length,
        hop_length=hop_length,
        window=window,
    )
    return mfcc


def compute_mel_spectrogram(
    audio: np.ndarray,
    target_sr: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
) -> np.ndarray:
    spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=target_sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    return librosa.power_to_db(spectrogram, ref=np.max)


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


def save_empty_manifest(output_dir: Path) -> pd.DataFrame:
    manifest = pd.DataFrame(
        columns=[
            "index",
            "snippet_path",
            "label",
            "source_filepath",
            "onset_s",
            "offset_s",
            "duration_s",
            "n_frames",
        ]
    )
    manifest_path = output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    logger.info("Saved empty manifest to %s", manifest_path)
    return manifest


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
    feature_type: str,
    n_mfcc: int,
    mel_bins: int,
    mel_nfft: int,
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
        if feature_type == "mfcc":
            features = compute_mfcc(audio, target_sr, n_mfcc, frame_length, hop_length, window)
        else:
            features = compute_mel_spectrogram(audio, target_sr, mel_nfft, hop_length, mel_bins)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Skipping index %s due to error: %s", row.name, exc)
        return None
    base_candidate = row.get("file")
    if pd.isna(base_candidate) or str(base_candidate).strip() == "":
        base_candidate = row.get("filepath")
    base_filename = str(base_candidate) if base_candidate is not None else "fragment"

    fragment_path = save_fragment(features, output_dir, label, base_filename, row.name)
    n_frames = features.shape[1]
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


def apply_label_limits(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if df.empty or (args.max_per_label is None and args.max_nothing is None):
        return df

    if args.max_per_label is not None and args.max_per_label < 0:
        raise ValueError("--max-per-label must be non-negative")
    if args.max_nothing is not None and args.max_nothing < 0:
        raise ValueError("--max-nothing must be non-negative")

    rng = np.random.default_rng(args.seed)
    kept_frames: List[pd.DataFrame] = []
    log_rows = []
    total_before = len(df)

    for label, group in df.groupby("label", sort=False):
        limit = args.max_per_label
        if label == "Nothing" and args.max_nothing is not None:
            limit = args.max_nothing

        if limit is None or len(group) <= limit:
            kept = group
            skipped = 0
        else:
            random_state = int(rng.integers(0, 2**32 - 1))
            kept = group.sample(n=limit, random_state=random_state)
            skipped = len(group) - len(kept)

        kept_frames.append(kept)
        log_rows.append((label, len(group), len(kept), skipped))

    result = pd.concat(kept_frames).sort_index()
    kept_total = len(result)
    skipped_total = total_before - kept_total

    details = ", ".join(
        f"{label}: kept {kept} of {total}, skipped {skipped}"
        for label, total, kept, skipped in log_rows
    )
    logger.info(
        "Label caps applied: kept %d of %d rows (skipped %d). Details: %s",
        kept_total,
        total_before,
        skipped_total,
        details,
    )
    return result


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
    durations = df["offset_s"] - df["onset_s"]

    mask = pd.Series(True, index=df.index)
    below_min = above_max = 0
    if args.min_duration is not None:
        below_min = int((durations < args.min_duration).sum())
        mask &= durations >= args.min_duration
    if args.max_duration is not None:
        above_max = int((durations > args.max_duration).sum())
        mask &= durations <= args.max_duration

    if args.min_duration is not None or args.max_duration is not None:
        filtered = df[mask]
        skipped = len(df) - len(filtered)
        logger.info(
            "Duration filter applied: kept %d rows, skipped %d (below_min=%d, above_max=%d)",
            len(filtered),
            skipped,
            below_min,
            above_max,
        )
        df = filtered
        if df.empty:
            logger.warning("No rows remain after applying duration filters; exiting early.")
            return save_empty_manifest(args.output_dir)

    df = apply_label_limits(df, args)
    if df.empty:
        logger.warning("No rows remain after applying label caps; exiting early.")
        return save_empty_manifest(args.output_dir)
    selected = select_rows(df, args.limit, args.seed)
    csv_dir = args.csv_path.parent

    records: List[dict] = []
    for _, row in selected.iterrows():
        record = process_row(
            row=row,
            output_dir=args.output_dir,
            target_sr=args.target_sr,
            feature_type=args.feature_type,
            n_mfcc=args.n_mfcc,
            mel_bins=args.mel_bins,
            mel_nfft=args.mel_nfft,
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
