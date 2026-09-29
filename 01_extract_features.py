"""Step 1: turn raw MaFaulDa CSV recordings into a feature table.

Each recording is cut into ``N_WINDOWS`` equal, non-overlapping windows.  For
every window and EVERY sensor we compute the SAME 18 features, so that a sensor
subset's performance reflects the sensors themselves and not a differing number
of features.

Standard deviation is deliberately absent: after DC removal it equals RMS
exactly, so including it would add a duplicate feature.  ``tests/test_features.py``
enforces that no such algebraic duplicate exists.

Output: ``features/features.csv``, one row per window, with columns
    recording_id, window, label, condition_path, speed_hz, speed_hz_tach,
    <sensor>__<feature>...

Usage:
    python 01_extract_features.py              # every recording
    python 01_extract_features.py --limit 60   # seeded random sample, for testing
    python 01_extract_features.py --jobs 4     # limit the worker count
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import NDArray
from scipy.stats import kurtosis, skew

import common
from common import (ALL_CHANNELS, BANDS, CLASS_ORDER, FS, N_WINDOWS, SEED,
                    write_run_metadata)

LOG = logging.getLogger(__name__)

# Guard against division by zero in every feature denominator.
EPS: float = 1e-12

# Shape every MaFaulDa recording is expected to have.
EXPECTED_ROWS: int = 250_000
EXPECTED_RECORDINGS: int = 1_951

# The 18 features, in the order of Section 7 of the specification.
FEATURE_NAMES: list[str] = [
    "rms", "peak", "p2p", "crest", "kurtosis", "skewness", "shape", "impulse",
    "clearance", "spec_centroid", "dominant_freq",
] + [f"band_{lo}_{hi}" for lo, hi in BANDS]

METADATA_COLUMNS: list[str] = [
    "recording_id", "window", "label", "condition_path", "speed_hz",
    "speed_hz_tach",
]

# MaFaulDa encodes the rotation frequency in Hz in the file name, e.g.
# "12.288.csv". That is ground truth from the dataset authors; see PROTOCOL.md
# section 3.1 for why it replaced the tachometer estimate as speed_hz.
FILENAME_SPEED_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)$")


def window_features(x: NDArray[np.float64], fs: int = FS) -> dict[str, float]:
    """Compute the 18 features for one sensor channel in one window.

    The mean (DC component) is removed first, so the features describe the
    fluctuation of the signal rather than its operating point.

    Args:
        x: One window of one channel.
        fs: Sampling rate in Hz.

    Returns:
        A mapping from feature name to value, keyed by ``FEATURE_NAMES``.

    Raises:
        ValueError: If the window has fewer than two samples.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size < 2:
        raise ValueError(f"Window has {x.size} samples; at least 2 are required.")
    x = x - x.mean()

    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    abs_mean = float(np.mean(np.abs(x)))
    sqrt_abs_mean = float(np.mean(np.sqrt(np.abs(x))))

    feats: dict[str, float] = {
        "rms": rms,
        "peak": peak,
        "p2p": float(x.max() - x.min()),
        "crest": peak / (rms + EPS),
        "kurtosis": float(kurtosis(x)),          # Fisher definition: 0 = Gaussian
        "skewness": float(skew(x)),
        "shape": rms / (abs_mean + EPS),
        "impulse": peak / (abs_mean + EPS),
        "clearance": peak / (sqrt_abs_mean ** 2 + EPS),
    }

    spectrum = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    total = float(spectrum.sum()) + EPS
    feats["spec_centroid"] = float((freqs * spectrum).sum() / total)
    # Exclude the DC bin: after mean removal it carries only leakage.
    feats["dominant_freq"] = float(freqs[1 + int(np.argmax(spectrum[1:]))])
    for lo, hi in BANDS:
        in_band = (freqs >= lo) & (freqs < hi)
        feats[f"band_{lo}_{hi}"] = float(spectrum[in_band].sum() / total)

    if set(feats) != set(FEATURE_NAMES):
        raise ValueError("Feature set drifted from FEATURE_NAMES; fix one of them.")
    return feats


def speed_from_filename(path: Path) -> float:
    """Read the rotation frequency in Hz from the recording's file name.

    MaFaulDa names each recording after its rotation frequency, for example
    ``12.288.csv``.  This is the dataset authors' own figure, so it is used
    directly rather than estimated.

    Args:
        path: The CSV file.

    Returns:
        Rotation frequency in Hz.

    Raises:
        ValueError: If the file name is not a bare number.  A name that does
            not parse means the assumption behind this column has broken, so
            it fails loudly rather than falling back silently.
    """
    match = FILENAME_SPEED_RE.match(path.stem)
    if match is None:
        raise ValueError(
            f"{path}: file name {path.stem!r} is not a bare rotation frequency "
            "in Hz. speed_hz is parsed from the file name; see PROTOCOL.md "
            "section 3.1."
        )
    return float(match.group(1))


def estimate_speed_hz(tach: NDArray[np.float64], fs: int = FS) -> float:
    """Estimate the rotation frequency from the tachometer channel.

    Retained as ``speed_hz_tach`` for comparison against the file-name value;
    it is NOT the ``speed_hz`` used by the pipeline.  On 67 MaFaulDa
    recordings it overcounts badly; see PROTOCOL.md section 3.1.

    The signal is thresholded halfway between its extremes and below-to-above
    crossings are counted; one crossing is one revolution.

    Args:
        tach: The full tachometer column of a recording.
        fs: Sampling rate in Hz.

    Returns:
        Rotation frequency in Hz.

    Raises:
        ValueError: If the channel has fewer than two samples.
    """
    tach = np.asarray(tach, dtype=np.float64)
    if tach.size < 2:
        raise ValueError(f"Tachometer channel has {tach.size} samples.")
    threshold = (tach.max() + tach.min()) / 2.0
    above = tach > threshold
    rising = int(np.count_nonzero(above[1:] & ~above[:-1]))
    return rising / (tach.size / fs)


def process_file(path: Path, rec_id: int, data_dir: Path) -> list[dict[str, object]]:
    """Extract one row per window from a single recording.

    Args:
        path: The CSV file.
        rec_id: Stable identifier for this recording; the cross-validation
            grouping depends on it.
        data_dir: The dataset root, used to derive ``label`` and
            ``condition_path``.

    Returns:
        ``N_WINDOWS`` row dictionaries.

    Raises:
        ValueError: If the file does not have exactly ``len(ALL_CHANNELS)``
            columns,
            is too short to window, or sits under an unknown class folder.
            Bad files are never skipped silently.
    """
    sig = pd.read_csv(path, header=None).to_numpy(dtype=np.float64)
    if sig.ndim != 2 or sig.shape[1] != len(ALL_CHANNELS):
        raise ValueError(
            f"{path}: expected {len(ALL_CHANNELS)} columns, got shape {sig.shape}."
        )
    if sig.shape[0] != EXPECTED_ROWS:
        LOG.warning("%s: %d rows, expected %d", path, sig.shape[0], EXPECTED_ROWS)
    if sig.shape[0] < N_WINDOWS:
        raise ValueError(
            f"{path}: {sig.shape[0]} rows cannot be split into {N_WINDOWS} windows."
        )

    rel = path.relative_to(data_dir)
    label = rel.parts[0]
    if label not in CLASS_ORDER:
        raise ValueError(
            f"{path}: top-level folder {label!r} is not a known class. "
            f"Expected one of {CLASS_ORDER}."
        )

    speed_hz = speed_from_filename(path)
    speed_hz_tach = estimate_speed_hz(sig[:, ALL_CHANNELS.index("tach")])
    win_len = sig.shape[0] // N_WINDOWS

    rows: list[dict[str, object]] = []
    for w in range(N_WINDOWS):
        segment = sig[w * win_len:(w + 1) * win_len]
        row: dict[str, object] = {
            "recording_id": rec_id,
            "window": w,
            "label": label,
            "condition_path": rel.parent.as_posix(),
            "speed_hz": speed_hz,
            "speed_hz_tach": speed_hz_tach,
        }
        for j, sensor in enumerate(ALL_CHANNELS):
            for name, value in window_features(segment[:, j]).items():
                row[f"{sensor}__{name}"] = value
        rows.append(row)
    return rows


def extract_features(
    data_dir: Path,
    limit: int | None = None,
    jobs: int = -1,
) -> pd.DataFrame:
    """Extract the feature table for every recording under ``data_dir``.

    Files are sorted before ``recording_id`` is assigned, so the identifiers,
    and therefore the cross-validation groups, are identical on every run and
    on every machine.

    Args:
        data_dir: Dataset root holding the class folders.
        limit: If given, process a seeded random sample of this many files.
        jobs: joblib worker count; -1 uses every core.

    Returns:
        One row per window.

    Raises:
        FileNotFoundError: If no CSV files are found.
    """
    files = sorted(Path(data_dir).rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {data_dir}.")
    if limit is not None:
        rng = np.random.default_rng(SEED)
        keep = rng.choice(len(files), size=min(limit, len(files)), replace=False)
        files = [files[i] for i in sorted(keep)]

    LOG.info("Extracting features from %d recordings under %s", len(files), data_dir)
    results = Parallel(n_jobs=jobs, verbose=5)(
        delayed(process_file)(path, rec_id, Path(data_dir))
        for rec_id, path in enumerate(files)
    )
    return pd.DataFrame([row for rows in results for row in rows])


def report(df: pd.DataFrame, limited: bool) -> None:
    """Log the dataset summary required by Sections 3 and 7.

    Args:
        df: The extracted feature table.
        limited: Whether ``--limit`` was used, in which case a short total is
            expected and no warning is raised.
    """
    per_class = df.groupby("label", observed=True)["recording_id"].nunique()
    LOG.info("Recordings per class:")
    for label in CLASS_ORDER:
        LOG.info("  %-24s %4d", label, int(per_class.get(label, 0)))
    total = int(df["recording_id"].nunique())
    LOG.info("  %-24s %4d", "TOTAL", total)
    if not limited and total != EXPECTED_RECORDINGS:
        LOG.warning(
            "Total is %d, but the full MaFaulDa dataset has %d recordings. "
            "Check that the dataset is complete.", total, EXPECTED_RECORDINGS
        )
    LOG.info("Speed range (from file names): %.2f-%.2f Hz",
             df["speed_hz"].min(), df["speed_hz"].max())
    LOG.info("Speed range (tachometer estimate): %.2f-%.2f Hz",
             df["speed_hz_tach"].min(), df["speed_hz_tach"].max())
    LOG.info("Feature table: %d rows x %d columns", df.shape[0], df.shape[1])


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=None,
                    help="process a seeded random sample of N files, for testing")
    ap.add_argument("--jobs", type=int, default=-1,
                    help="number of parallel workers (-1 = all cores)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    n_files = len(sorted(Path(common.DATA_DIR).rglob("*.csv")))
    df = extract_features(common.DATA_DIR, limit=args.limit, jobs=args.jobs)
    common.FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(common.FEATURES_FILE, index=False)
    LOG.info("Saved -> %s", common.FEATURES_FILE)
    report(df, limited=args.limit is not None)

    write_run_metadata(
        common.FEATURES_FILE.parent, "01_extract_features.py",
        data_dir=str(common.DATA_DIR),
        n_csv_files_found=n_files,
        n_recordings_processed=int(df["recording_id"].nunique()),
        n_rows=int(df.shape[0]),
        n_columns=int(df.shape[1]),
        limit=args.limit,
        jobs=args.jobs,
    )


if __name__ == "__main__":
    main()
