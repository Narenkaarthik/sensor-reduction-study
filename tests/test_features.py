"""Level A: unit tests for feature extraction.

These run entirely on signals with known analytic properties, plus the
synthetic dataset.  They never touch the real data, so they must pass before
any real-data run.
"""
from __future__ import annotations

from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from common import ALL_CHANNELS, EXCLUDED_CHANNELS, FS, SENSORS, subset_name

# One second at the real sampling rate, used for the analytic cases.
N = FS
T = np.arange(N, dtype=np.float64) / FS


def _sine(freq: float, amplitude: float = 2.0, offset: float = 0.0) -> np.ndarray:
    """Generate a sine wave, optionally shifted by a constant DC offset."""
    return amplitude * np.sin(2.0 * np.pi * freq * T) + offset


def test_sine_analytic_values(extract: ModuleType) -> None:
    """A 100 Hz, amplitude-2 sine has exactly known feature values."""
    f = extract.window_features(_sine(100.0))
    assert f["rms"] == pytest.approx(2 / np.sqrt(2), abs=1e-4)      # 1.4142
    assert f["peak"] == pytest.approx(2.0, abs=1e-4)
    assert f["p2p"] == pytest.approx(4.0, abs=1e-4)
    assert f["crest"] == pytest.approx(np.sqrt(2), abs=1e-4)        # 1.4142
    assert f["shape"] == pytest.approx(np.pi / (2 * np.sqrt(2)), abs=1e-4)  # 1.1107
    assert f["kurtosis"] == pytest.approx(-1.5, abs=1e-3)
    assert f["skewness"] == pytest.approx(0.0, abs=1e-6)
    assert f["dominant_freq"] == pytest.approx(100.0, abs=1e-9)


def test_gaussian_noise_kurtosis(extract: ModuleType) -> None:
    """Gaussian noise has Fisher kurtosis near zero."""
    rng = np.random.default_rng(0)
    f = extract.window_features(rng.normal(0.0, 1.0, N))
    assert f["kurtosis"] == pytest.approx(0.0, abs=0.1)


def test_band_fractions_sum_to_one(extract: ModuleType) -> None:
    """With no content below 5 Hz, the seven band fractions account for all power."""
    rng = np.random.default_rng(1)
    signal = _sine(100.0) + _sine(7_000.0, 1.0) + rng.normal(0.0, 0.5, N)
    f = extract.window_features(signal)
    bands = [v for k, v in f.items() if k.startswith("band_")]
    assert len(bands) == 7
    assert sum(bands) == pytest.approx(1.0, abs=1e-3)


def test_dc_offset_is_removed(extract: ModuleType) -> None:
    """A constant offset changes nothing: the mean is removed first."""
    plain = extract.window_features(_sine(100.0))
    shifted = extract.window_features(_sine(100.0, offset=17.5))
    for name in plain:
        assert shifted[name] == pytest.approx(plain[name], rel=1e-9, abs=1e-9), name


def test_speed_estimate_from_tach_pulses(extract: ModuleType) -> None:
    """A 30 Hz tachometer pulse train is estimated as 30 Hz within 1 percent."""
    from make_synthetic_data import _tach_pulses

    speed = extract.estimate_speed_hz(_tach_pulses(T, 30.0))
    assert speed == pytest.approx(30.0, rel=0.01)


def test_feature_count_per_sensor(extract: ModuleType) -> None:
    """Exactly 18 features are produced per sensor channel."""
    f = extract.window_features(_sine(100.0))
    assert len(f) == 18
    assert sorted(f) == sorted(extract.FEATURE_NAMES)


def test_feature_column_count(synthetic_features: pd.DataFrame,
                              extract: ModuleType) -> None:
    """The feature table has 8 x 18 = 144 feature columns plus 6 metadata columns.

    All EIGHT channels are extracted, including the excluded tachometer: the
    exclusion happens at the sensor-set level, not at extraction. See
    PROTOCOL.md section 2.2.
    """
    feature_cols = [c for c in synthetic_features.columns if "__" in c]
    assert len(feature_cols) == len(ALL_CHANNELS) * 18 == 144
    for sensor in ALL_CHANNELS:
        owned = [c for c in feature_cols if c.startswith(f"{sensor}__")]
        assert len(owned) == 18, sensor
    assert list(synthetic_features.columns[:6]) == extract.METADATA_COLUMNS
    assert synthetic_features.shape[1] == 150


def _varied_signals() -> list[np.ndarray]:
    """Seeded signals of genuinely different SHAPE, not merely different scale.

    Every signal carries broadband noise so that no band fraction collapses to
    zero, and at least one is strongly asymmetric so that peak and p2p are not
    locked to a 2:1 ratio.
    """
    rng = np.random.default_rng(7)
    noise = lambda scale: rng.normal(0.0, scale, N)  # noqa: E731

    impulses = np.zeros(N)
    impulses[::523] = 10.0                      # one-sided: skews the distribution
    harmonics = sum(_sine(f, a) for f, a in [(100, 3.0), (200, 1.5), (400, 0.7)])

    return [
        _sine(200.0) + noise(0.3),
        _sine(3_000.0) + noise(0.3),
        noise(1.0),
        impulses + noise(0.3),
        harmonics + noise(0.3),
        _sine(1_500.0, 5.0) + noise(0.3),
        noise(10.0),
        _sine(80.0, 0.5) + _sine(12_000.0, 2.0) + noise(0.3),
    ]


def test_no_duplicate_features(extract: ModuleType) -> None:
    """No pair of features is an exact algebraic duplicate of another.

    A duplicate such as ``std == rms`` shows up as a ratio that is constant
    across signal shapes.  Raw correlation is deliberately NOT used: rms, peak
    and p2p are all amplitude-scaled and so are legitimately highly correlated.
    """
    names = extract.FEATURE_NAMES
    values = np.array([[extract.window_features(s)[n] for n in names]
                       for s in _varied_signals()])
    assert np.all(np.isfinite(values)), "a feature is not finite on the test signals"

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            denom = values[:, j]
            assert np.all(np.abs(denom) > 1e-12), f"{names[j]} vanishes; widen signals"
            ratio = values[:, i] / denom
            cov = float(np.std(ratio) / (np.abs(np.mean(ratio)) + 1e-12))
            assert cov > 1e-3, (
                f"{names[i]} and {names[j]} have a near-constant ratio "
                f"(CoV={cov:.2e}); they are duplicates"
            )


def test_std_would_be_caught_as_duplicate(extract: ModuleType) -> None:
    """The duplicate detector really works: std vs rms must be flagged.

    This is a negative control for ``test_no_duplicate_features``.  After DC
    removal, std and rms are identical, so their ratio is constant at 1.
    """
    ratios = []
    for signal in _varied_signals():
        centred = signal - signal.mean()
        rms = extract.window_features(signal)["rms"]
        ratios.append(float(np.std(centred)) / rms)
    cov = float(np.std(ratios) / np.abs(np.mean(ratios)))
    assert cov <= 1e-3, "std and rms should be indistinguishable after DC removal"


@pytest.mark.parametrize("permutation", [
    ["ub_ax", "ub_rad", "mic"],
    ["mic", "ub_ax", "ub_rad"],
    ["ub_rad", "mic", "ub_ax"],
    ["mic", "ub_rad", "ub_ax"],
])
def test_subset_name_is_canonical(permutation: list[str]) -> None:
    """Any ordering of the same sensors yields one canonical name."""
    assert subset_name(permutation) == "ub_ax+ub_rad+mic"


def test_subset_name_rejects_bad_input() -> None:
    """Unknown or repeated sensors are errors, not silently accepted."""
    with pytest.raises(ValueError):
        subset_name(["ub_ax", "not_a_sensor"])
    with pytest.raises(ValueError):
        subset_name(["ub_ax", "ub_ax"])


# ---------------------------------------------------------- tach exclusion
def test_tachometer_is_excluded_from_the_sensor_set() -> None:
    """The study evaluates 7 sensors; the tachometer is not one of them.

    It carries a label-correlated recording artefact, so including it would
    make the RQ3 sensor ranking answer the wrong question. See PROTOCOL.md
    section 2.2.
    """
    assert "tach" in ALL_CHANNELS, "the channel is still extracted"
    assert "tach" in EXCLUDED_CHANNELS
    assert "tach" not in SENSORS
    assert len(SENSORS) == 7
    assert len(ALL_CHANNELS) == 8
    assert set(SENSORS) == set(ALL_CHANNELS) - set(EXCLUDED_CHANNELS)


def test_tach_features_are_still_extracted() -> None:
    """The tachometer features remain in the table as evidence."""
    from conftest import load_script

    extract = load_script("01_extract_features")
    assert len(extract.FEATURE_NAMES) == 18


def test_tach_features_are_present_in_the_table(synthetic_features) -> None:
    """All 18 tachometer feature columns exist even though tach is excluded."""
    tach_cols = [c for c in synthetic_features.columns if c.startswith("tach__")]
    assert len(tach_cols) == 18


def test_synthetic_table_is_well_formed(synthetic_features: pd.DataFrame) -> None:
    """The synthetic feature table is complete, finite and correctly windowed."""
    df = synthetic_features
    numeric = df.select_dtypes(include=[np.number])
    assert not numeric.isna().any().any(), "NaN in the feature table"
    assert np.isfinite(numeric.to_numpy()).all(), "infinite value in the feature table"
    assert (df.groupby("recording_id")["window"].nunique() == 5).all()
    assert df["condition_path"].str.startswith(tuple(df["label"].unique())).all()


def test_rejects_wrong_column_count(extract: ModuleType, tmp_path) -> None:
    """A file without exactly 8 columns is an error, never a silent skip."""
    root = tmp_path / "bad"
    (root / "normal").mkdir(parents=True)
    path = root / "normal" / "broken.csv"
    np.savetxt(path, np.zeros((100, 3)), delimiter=",")
    with pytest.raises(ValueError, match="expected 8 columns"):
        extract.process_file(path, 0, root)


def test_rejects_unknown_class_folder(extract: ModuleType, tmp_path) -> None:
    """A top-level folder outside CLASS_ORDER is an error."""
    root = tmp_path / "odd"
    (root / "not-a-class").mkdir(parents=True)
    path = root / "not-a-class" / "rec.csv"
    rng = np.random.default_rng(0)
    np.savetxt(path, rng.normal(size=(100, len(ALL_CHANNELS))), delimiter=",")
    with pytest.raises(ValueError, match="not a known class"):
        extract.process_file(path, 0, root)


def test_limit_sample_is_seeded(extract: ModuleType, synthetic_root) -> None:
    """--limit draws the same files every time, so recording_id stays stable."""
    first = extract.extract_features(synthetic_root, limit=12, jobs=1)
    second = extract.extract_features(synthetic_root, limit=12, jobs=1)
    assert first["recording_id"].nunique() == 12
    pd.testing.assert_frame_equal(first, second)
