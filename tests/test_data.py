"""Level B: real-data validation.

These are the only tests that touch ``data/mafaulda/``.  They are skipped
automatically when the dataset is absent, so the rest of the suite runs on a
clean checkout.

Run them after placing the dataset and before ``01_extract_features.py``:

    python -m pytest -m realdata -s

``-s`` is worth passing: the class-count test prints a table for manual
comparison against the dataset documentation.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from common import CLASS_ORDER, DATA_DIR, FEATURES_FILE, N_WINDOWS, SENSORS

pytestmark = pytest.mark.realdata

EXPECTED_RECORDINGS = 1_951
EXPECTED_ROWS = 250_000
SAMPLE_SIZE = 50
SPEED_RANGE_HZ = (10.0, 65.0)

LOG = logging.getLogger(__name__)


def _require_dataset() -> Path:
    """Skip the test unless the real dataset is present."""
    if not DATA_DIR.exists():
        pytest.skip(f"{DATA_DIR} not present; place MaFaulDa there to run "
                    "the Level B tests")
    return DATA_DIR


def _require_features() -> pd.DataFrame:
    """Skip the test unless the feature table has been extracted."""
    _require_dataset()
    if not FEATURES_FILE.exists():
        pytest.skip(f"{FEATURES_FILE} not present; run 01_extract_features.py "
                    "to run the feature-level Level B tests")
    return pd.read_csv(FEATURES_FILE)


@pytest.fixture(scope="module")
def csv_files() -> list[Path]:
    """Every CSV under the dataset root, sorted."""
    return sorted(_require_dataset().rglob("*.csv"))


def test_recording_count(csv_files: list[Path]) -> None:
    """MaFaulDa has exactly 1,951 recordings."""
    assert len(csv_files) == EXPECTED_RECORDINGS, (
        f"found {len(csv_files)} CSV files under {DATA_DIR}, expected "
        f"{EXPECTED_RECORDINGS}. The download may be incomplete.")


def test_top_level_folders_are_the_six_classes(csv_files: list[Path]) -> None:
    """Every recording sits under one of the six class folders."""
    folders = sorted({p.relative_to(DATA_DIR).parts[0] for p in csv_files})
    assert folders == sorted(CLASS_ORDER), (
        f"top-level folders are {folders}, expected {sorted(CLASS_ORDER)}")


def test_sampled_files_have_the_expected_shape(csv_files: list[Path]) -> None:
    """A seeded sample of at least 50 recordings is (250000, 8)."""
    rng = np.random.default_rng(0)
    picked = rng.choice(len(csv_files), size=min(SAMPLE_SIZE, len(csv_files)),
                        replace=False)
    for index in sorted(picked):
        path = csv_files[index]
        data = pd.read_csv(path, header=None).to_numpy(dtype=np.float64)
        assert data.shape == (EXPECTED_ROWS, len(SENSORS)), (
            f"{path} has shape {data.shape}, expected "
            f"({EXPECTED_ROWS}, {len(SENSORS)})")


def test_class_counts(csv_files: list[Path]) -> None:
    """Print the recordings per class for manual comparison with the docs."""
    counts = {name: 0 for name in CLASS_ORDER}
    for path in csv_files:
        counts[path.relative_to(DATA_DIR).parts[0]] += 1

    print("\nRecordings per class (compare against the dataset documentation):")
    for name in CLASS_ORDER:
        print(f"  {name:<26} {counts[name]:>5}")
    print(f"  {'TOTAL':<26} {sum(counts.values()):>5}")

    assert all(count > 0 for count in counts.values()), (
        f"a class folder is empty: {counts}")
    assert sum(counts.values()) == EXPECTED_RECORDINGS


def test_feature_table_is_finite() -> None:
    """The extracted features contain no NaN and no infinite values."""
    df = _require_features()
    numeric = df.select_dtypes(include=[np.number])
    nan_columns = [c for c in numeric.columns if numeric[c].isna().any()]
    assert not nan_columns, f"NaN in columns: {nan_columns[:10]}"
    bad = [c for c in numeric.columns
           if not np.isfinite(numeric[c].to_numpy()).all()]
    assert not bad, f"infinite values in columns: {bad[:10]}"


def test_every_label_is_known() -> None:
    """Every label in the feature table is one of the six classes."""
    df = _require_features()
    unknown = sorted(set(df["label"].astype(str)) - set(CLASS_ORDER))
    assert not unknown, f"labels outside CLASS_ORDER: {unknown}"


def test_every_recording_has_five_windows() -> None:
    """Windowing produced exactly N_WINDOWS rows per recording."""
    df = _require_features()
    per_recording = df.groupby("recording_id")["window"].nunique()
    bad = per_recording[per_recording != N_WINDOWS]
    assert bad.empty, f"recordings without {N_WINDOWS} windows: {bad.head().to_dict()}"
    assert len(df) == EXPECTED_RECORDINGS * N_WINDOWS, (
        f"{len(df)} rows, expected {EXPECTED_RECORDINGS * N_WINDOWS}")


def test_feature_table_shape() -> None:
    """8 sensors x 18 features plus 5 metadata columns."""
    df = _require_features()
    feature_columns = [c for c in df.columns if "__" in c]
    assert len(feature_columns) == len(SENSORS) * 18
    assert df.shape[1] == len(SENSORS) * 18 + 5


def test_estimated_speeds_are_plausible() -> None:
    """Estimated rotation speeds lie inside the documented range."""
    df = _require_features()
    speeds = df.groupby("recording_id")["speed_hz"].first()
    low, high = SPEED_RANGE_HZ
    outside = speeds[(speeds < low) | (speeds > high)]
    print(f"\nEstimated speed range: {speeds.min():.2f}-{speeds.max():.2f} Hz "
          f"over {len(speeds)} recordings")
    assert outside.empty, (
        f"{len(outside)} recording(s) outside {low}-{high} Hz, for example "
        f"{outside.head().to_dict()}. Check the tachometer channel index.")


def test_speeds_vary_within_every_class() -> None:
    """Each class spans a range of speeds, which speed-mode splitting needs."""
    df = _require_features()
    per_recording = df.groupby("recording_id").agg(
        label=("label", "first"), speed_hz=("speed_hz", "first"))
    spread = per_recording.groupby("label", observed=True)["speed_hz"].agg(
        ["min", "max", "nunique"])
    print("\nSpeed spread per class:")
    print(spread.to_string())
    assert (spread["nunique"] > 1).all(), (
        "a class was recorded at a single speed; speed-mode splitting would "
        "confound speed with condition")


def test_extractor_agrees_with_a_freshly_read_file(csv_files: list[Path]) -> None:
    """Re-extract one real recording and check it matches the saved table."""
    df = _require_features()
    from conftest import load_script

    module = load_script("01_extract_features")
    rows = module.process_file(csv_files[0], 0, DATA_DIR)
    assert len(rows) == N_WINDOWS
    saved = df[df["recording_id"] == 0].sort_values("window")
    assert len(saved) == N_WINDOWS
    for column, value in rows[0].items():
        if isinstance(value, str):
            assert saved.iloc[0][column] == value, column
        else:
            assert float(saved.iloc[0][column]) == pytest.approx(
                float(value), rel=1e-9, abs=1e-12), column
