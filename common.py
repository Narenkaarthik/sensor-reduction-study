"""Shared configuration and helpers for the sensor-reduction study.

This module is the SINGLE SOURCE OF TRUTH for every constant and every
cross-validation decision in the study.  Each script imports from here; nothing
below may be redefined anywhere else.  Changing a value here invalidates every
result downstream, so do not touch it once the protocol is frozen.

See PROTOCOL.md for the justification behind each choice.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

LOG = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Paths (all relative to the repository root; never absolute)
# ----------------------------------------------------------------------------
ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = ROOT / "data" / "mafaulda"   # the user unzips MaFaulDa here
FEATURES_FILE: Path = ROOT / "features" / "features.csv"
RESULTS_DIR: Path = ROOT / "results"
FIGURES_DIR: Path = ROOT / "figures"
TABLES_DIR: Path = ROOT / "tables"

# ----------------------------------------------------------------------------
# Dataset constants (MaFaulDa)
# ----------------------------------------------------------------------------
FS: int = 50_000                # sampling rate in Hz

# Column order of every MaFaulDa CSV (the files have no header).  This is the
# PHYSICAL channel layout of the file and must never change: it drives reading,
# shape validation and feature extraction.  All 8 channels are extracted.
ALL_CHANNELS: list[str] = [
    "tach",     # tachometer
    "ub_ax",    # underhang bearing accelerometer, axial
    "ub_rad",   # underhang bearing accelerometer, radial
    "ub_tan",   # underhang bearing accelerometer, tangential
    "ob_ax",    # overhang bearing accelerometer, axial
    "ob_rad",   # overhang bearing accelerometer, radial
    "ob_tan",   # overhang bearing accelerometer, tangential
    "mic",      # microphone
]

# Channels extracted but EXCLUDED from the study's sensor set.  The tachometer
# carries a recording artefact that is label-correlated, so a sensor ranking
# that included it would answer RQ3 falsely.  Its features stay in the feature
# table as the evidence for that artefact and for the secondary 8-sensor
# comparison.  See PROTOCOL.md section 2.2.
EXCLUDED_CHANNELS: list[str] = ["tach"]

# The sensor set of the study: 7 sensors, 127 non-empty subsets.  This order is
# canonical for subset naming and for tie-breaking.
SENSORS: list[str] = [c for c in ALL_CHANNELS if c not in EXCLUDED_CHANNELS]

N_WINDOWS: int = 5              # equal, non-overlapping windows per recording

# The class label is the top-level folder name inside DATA_DIR.
CLASS_ORDER: list[str] = [
    "normal",
    "imbalance",
    "horizontal-misalignment",
    "vertical-misalignment",
    "underhang",
    "overhang",
]
HEALTHY_CLASS: str = "normal"

# The integer codes of every class.  Used for FIXED-SIZE outputs such as
# confusion matrices and per-class arrays, so that they line up across folds
# and across k.  This is deliberately NOT the scoring label set: see FoldScorer.
CLASS_LABELS: list[int] = list(range(len(CLASS_ORDER)))

# ----------------------------------------------------------------------------
# Experimental protocol (fixed before any result is seen; never tuned)
# ----------------------------------------------------------------------------
SEED: int = 42
N_SPLITS: int = 5               # outer folds
N_REPEATS: int = 2              # 5 folds x 2 repeats = 10 paired scores
                                # (5 alone cannot reach p < 0.05 in Wilcoxon)
N_TREES: int = 200
TOLERANCE: float = 0.01         # used by H1 and H2
PERM_REPEATS: int = 5           # repeats of the grouped permutation

# Speed-band resolutions tried, in order, by choose_speed_q().  A test fold
# missing a class is a split defect, so the banding is coarsened until every
# fold has full class coverage.
SPEED_QUANTILE_CANDIDATES: tuple[int, ...] = (10, 8, 6)

# Frequency bands (Hz) for relative band energy features.
BANDS: list[tuple[int, int]] = [
    (5, 250), (250, 500), (500, 1000), (1000, 2000),
    (2000, 5000), (5000, 10000), (10000, 25000),
]


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def make_model(n_trees: int = N_TREES) -> RandomForestClassifier:
    """Return THE classifier of the study, identical for every sensor subset.

    There is exactly one model configuration in this work: no hyper-parameter
    tuning, no alternative classifiers, no feature scaling.  Any difference
    between subsets must therefore come from the sensors, not from the model.

    Args:
        n_trees: Number of trees.  Only lowered for quick smoke runs.

    Returns:
        An unfitted ``RandomForestClassifier``.
    """
    return RandomForestClassifier(
        n_estimators=n_trees,
        class_weight="balanced",   # 'normal' is a comparatively small class
        n_jobs=-1,
        random_state=SEED,
    )


# ----------------------------------------------------------------------------
# Metrics
#
# Every score in the study goes through these two functions.  Calling
# ``f1_score`` or ``recall_score`` directly is a bug: the defaults average over
# only the classes present in the current fold, which makes folds, split modes
# and sensor subsets silently incomparable.
# ----------------------------------------------------------------------------
def fold_label_set(y_test: NDArray[np.int64], context: str) -> tuple[int, ...]:
    """Derive the scoring label set for ONE fold from its true test labels.

    Every comparison in this study is paired inside a fold (subset against
    subset, selected against the per-fold median), so every subset in a fold
    must be scored on an identical class set.  Taking that set from ``y_true``
    achieves this without ever penalising a fold for a class it does not
    contain.  A class that IS present but is never predicted still scores 0,
    which is correct.

    Args:
        y_test: True class codes of the fold's test set.
        context: Fold name, used in messages.

    Returns:
        The sorted class codes present in ``y_test``.

    Raises:
        ValueError: If fewer than 5 classes are present, which makes the fold
            too degenerate to report.
    """
    labels = tuple(int(v) for v in np.unique(np.asarray(y_test)))
    if len(labels) < len(CLASS_ORDER):
        absent = [CLASS_ORDER[i] for i in CLASS_LABELS if i not in labels]
        LOG.warning(
            "%s: test fold contains only %d of %d classes; missing %s. "
            "Every subset in this fold is scored on the same reduced label "
            "set, so comparisons stay paired, but the fold is not comparable "
            "with a full one.", context, len(labels), len(CLASS_ORDER), absent
        )
    if len(labels) < 5:
        raise ValueError(
            f"{context}: only {len(labels)} classes in the test fold "
            f"({[CLASS_ORDER[i] for i in labels]}). Refusing to score a fold "
            "this degenerate; fix the split instead."
        )
    return labels


@dataclass(frozen=True)
class FoldScorer:
    """Scores every sensor subset of ONE outer fold on one fixed label set.

    The label set is derived once, from the fold's true test labels, and then
    reused for every subset.  Each scoring call re-derives the set from the
    ``y_true`` it is handed and refuses to score if it differs, so a subset
    cannot silently be measured on a different set of classes from its
    neighbours.

    Attributes:
        labels: The fold's class codes, sorted.
        context: Fold name, used in messages.
    """

    labels: tuple[int, ...]
    context: str

    @classmethod
    def for_fold(cls, y_test: NDArray[np.int64], context: str) -> FoldScorer:
        """Build the scorer for a fold from its true test labels."""
        return cls(labels=fold_label_set(y_test, context), context=context)

    def _check(self, y_true: NDArray[np.int64]) -> None:
        """Refuse to score against a different label set from the fold's own."""
        got = tuple(int(v) for v in np.unique(np.asarray(y_true)))
        if got != self.labels:
            raise ValueError(
                f"{self.context}: this scorer is fixed to classes "
                f"{list(self.labels)} but was given {list(got)}. Every subset "
                "in a fold must be scored on an identical class set."
            )

    def macro_f1(self, y_true: NDArray[np.int64],
                 y_pred: NDArray[np.int64]) -> float:
        """Macro-F1 over exactly this fold's label set."""
        self._check(y_true)
        return float(f1_score(y_true, y_pred, labels=list(self.labels),
                              average="macro", zero_division=0))

    def accuracy(self, y_true: NDArray[np.int64],
                 y_pred: NDArray[np.int64]) -> float:
        """Plain accuracy; reported alongside macro-F1, never used to select."""
        self._check(y_true)
        return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def per_class_recall(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Recall for every class, in ``CLASS_ORDER`` order, always length 6.

    A class absent from ``y_true`` has no recall to report and becomes NaN, so
    that it is excluded from any averaging rather than dragging it down with a
    fictitious zero.  A class that is present but never predicted is a genuine
    0.  The fixed length keeps the per-class figure aligned across k.

    Args:
        y_true: True integer class codes.
        y_pred: Predicted integer class codes.

    Returns:
        Length-``len(CLASS_ORDER)`` array; NaN where the class is absent.
    """
    present = np.unique(np.asarray(y_true))
    out = np.full(len(CLASS_ORDER), np.nan, dtype=np.float64)
    if present.size:
        out[present] = recall_score(y_true, y_pred, labels=present,
                                    average=None, zero_division=0)
    return out


# ----------------------------------------------------------------------------
# Leakage guards (Section 8: enforced in code, not only in comments)
# ----------------------------------------------------------------------------
def assert_no_test_indices(
    used: NDArray[np.int64],
    test_idx: NDArray[np.int64],
    context: str,
) -> None:
    """Raise if data destined for fitting or selection includes a test index.

    Every ranking, selection and model-fitting call in the study is wrapped in
    this check, so that a leak fails the run loudly instead of quietly
    inflating a score.

    Args:
        used: Indices about to be used for fitting, ranking or selection.
        test_idx: Indices of the held-out test fold.
        context: Human-readable name of the caller, used in the message.

    Raises:
        ValueError: If the two index sets overlap at all.
    """
    overlap = np.intersect1d(np.asarray(used), np.asarray(test_idx))
    if overlap.size:
        raise ValueError(
            f"{context}: {overlap.size} test index/indices reached training "
            f"data (for example {overlap[:5].tolist()}). This is a leak."
        )


# ----------------------------------------------------------------------------
# Feature table
# ----------------------------------------------------------------------------
def load_features(path: Path | None = None) -> pd.DataFrame:
    """Load the feature table written by ``01_extract_features.py``.

    The ``label`` column becomes an ordered categorical in ``CLASS_ORDER``, so
    that integer codes are stable across runs and machines.

    Args:
        path: Location of ``features.csv``; defaults to ``FEATURES_FILE``,
            resolved at call time so that the location can be redirected.

    Returns:
        The feature table.

    Raises:
        FileNotFoundError: If the feature file does not exist.
        ValueError: If any label is outside ``CLASS_ORDER``.  Unknown labels are
            never dropped silently; they mean the data root is wrong.
    """
    path = Path(path) if path is not None else FEATURES_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 01_extract_features.py first."
        )
    df = pd.read_csv(path)
    unknown = sorted(set(df["label"].astype(str)) - set(CLASS_ORDER))
    if unknown:
        raise ValueError(
            f"{path} contains labels outside CLASS_ORDER: {unknown}. "
            "Check that DATA_DIR points at the MaFaulDa class folders."
        )
    df["label"] = pd.Categorical(df["label"], categories=CLASS_ORDER)
    return df


def feature_columns(df: pd.DataFrame, subset: Sequence[str]) -> list[str]:
    """Return the feature columns belonging to the given sensors.

    Columns are returned in the order they appear in ``df`` so that the matrix
    handed to the model depends only on which sensors were requested, never on
    the order in which they were listed.

    Args:
        df: A feature table with ``<sensor>__<feature>`` columns.
        subset: Sensor IDs, in any order.

    Returns:
        Matching column names, in table order.

    Raises:
        ValueError: If a sensor is unknown, or has no columns in ``df``.
    """
    unknown = [s for s in subset if s not in ALL_CHANNELS]
    if unknown:
        raise ValueError(
            f"Unknown channel(s): {unknown}. Expected any of {ALL_CHANNELS}.")
    prefixes = tuple(f"{s}__" for s in subset)
    cols = [c for c in df.columns if c.startswith(prefixes)]
    if len(subset) > 0 and not cols:
        raise ValueError(f"No feature columns found for sensors {list(subset)}.")
    return cols


def encode_labels(df: pd.DataFrame) -> NDArray[np.int64]:
    """Encode the ``label`` column as integer codes in ``CLASS_ORDER`` order.

    Args:
        df: A feature table whose ``label`` column is categorical.

    Returns:
        One integer per row; ``CLASS_ORDER.index(label)``.

    Raises:
        ValueError: If any label failed to encode (code -1).
    """
    if not isinstance(df["label"].dtype, pd.CategoricalDtype):
        df = df.assign(label=pd.Categorical(df["label"], categories=CLASS_ORDER))
    y = df["label"].cat.codes.to_numpy()
    if (y < 0).any():
        raise ValueError("Some labels are not in CLASS_ORDER; refusing to encode.")
    return y.astype(np.int64)


def subset_name(subset: Sequence[str]) -> str:
    """Return the canonical name of a sensor subset.

    Sensors are sorted into ``SENSORS`` order and joined with ``'+'``, so every
    permutation of the same sensors yields the same string.  This name is the
    key used in every results file.

    Args:
        subset: Sensor IDs, in any order, without duplicates.

    Returns:
        For example ``"ub_rad+mic"``.

    Raises:
        ValueError: If a sensor is unknown or repeated.
    """
    unknown = [s for s in subset if s not in ALL_CHANNELS]
    if unknown:
        raise ValueError(
            f"Unknown channel(s): {unknown}. Expected any of {ALL_CHANNELS}.")
    if len(set(subset)) != len(subset):
        raise ValueError(f"Duplicate sensors in subset: {list(subset)}.")
    return "+".join(s for s in ALL_CHANNELS if s in set(subset))


# ----------------------------------------------------------------------------
# Cross-validation protocol (see Section 8 of the specification)
# ----------------------------------------------------------------------------
def speed_bands(df: pd.DataFrame, q: int) -> NDArray[np.int64]:
    """Cut ``speed_hz`` into ``q`` quantile bands.

    Args:
        df: The feature table.
        q: Number of quantiles requested; empty bands are dropped, so the
            result can have fewer.

    Returns:
        One band index per row.
    """
    return pd.qcut(df["speed_hz"], q=q, labels=False,
                   duplicates="drop").to_numpy()


def _folds_missing_classes(
    y: NDArray[np.int64],
    groups: NDArray[np.int64],
) -> list[tuple[int, int, int]]:
    """List the outer folds whose test set would not contain every class.

    Args:
        y: Encoded labels.
        groups: Grouping used for the split.

    Returns:
        ``(repeat, fold, n_classes_present)`` for each deficient fold.
    """
    bad: list[tuple[int, int, int]] = []
    for r in range(N_REPEATS):
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                  random_state=SEED + r)
        for f, (_, test_idx) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
            present = len(np.unique(y[test_idx]))
            if present < len(CLASS_ORDER):
                bad.append((r, f, present))
    return bad


def choose_speed_q(
    df: pd.DataFrame,
    y: NDArray[np.int64],
) -> tuple[int, NDArray[np.int64]]:
    """Pick the speed banding that keeps every class in every test fold.

    A test fold missing a class is a split DEFECT, not a result to be scored,
    so the banding is retried at successively coarser resolutions and the first
    one that gives full coverage is used.  On the real dataset every condition
    was recorded across the speed range, so ``q=10`` is expected to succeed
    immediately; this ladder is a guard, not an expected path.

    Args:
        df: The feature table.
        y: Encoded labels.

    Returns:
        ``(q, groups)`` for the banding that was accepted.

    Raises:
        ValueError: If no candidate ``q`` gives full class coverage.  Scoring a
            degenerate split is worse than stopping.
    """
    attempts: list[str] = []
    for q in SPEED_QUANTILE_CANDIDATES:
        groups = speed_bands(df, q)
        n_bands = len(np.unique(groups))
        if n_bands < N_SPLITS:
            attempts.append(f"q={q}: only {n_bands} non-empty bands, need "
                            f"at least {N_SPLITS}")
            LOG.warning("speed split: %s", attempts[-1])
            continue
        bad = _folds_missing_classes(y, groups)
        if not bad:
            LOG.info("speed split: using q=%d (%d bands); every test fold "
                     "contains all %d classes", q, n_bands, len(CLASS_ORDER))
            return q, groups
        attempts.append(
            f"q={q}: {len(bad)} fold(s) missing a class, e.g. "
            f"r{bad[0][0]}_f{bad[0][1]} has {bad[0][2]}/{len(CLASS_ORDER)}")
        LOG.warning("speed split: %s; trying a coarser banding", attempts[-1])

    raise ValueError(
        "speed split: no quantile count in "
        f"{list(SPEED_QUANTILE_CANDIDATES)} puts all {len(CLASS_ORDER)} "
        "classes in every test fold.\n  " + "\n  ".join(attempts) +
        "\nEvery condition must be recorded across the speed range for this "
        "split mode to be valid. Use --split recording, or supply data with "
        "broader speed coverage per class."
    )


def outer_splits(
    df: pd.DataFrame,
    y: NDArray[np.int64],
    split_mode: str = "recording",
) -> Iterator[tuple[int, int, NDArray[np.int64], NDArray[np.int64]]]:
    """Yield the outer cross-validation splits, ``(repeat, fold, train, test)``.

    ``N_REPEATS`` repeats of ``StratifiedGroupKFold(N_SPLITS)``, stratified by
    class and grouped so that no unit of measurement is split across the
    train/test boundary:

    * ``"recording"`` (default): grouped by ``recording_id``, so the five
      windows cut from one recording never straddle the boundary.
    * ``"speed"``: grouped by quantile band of ``speed_hz``, so whole
      rotation-speed bands are held out.  This is the fallback if the recording
      split shows a ceiling effect.  The number of bands is chosen by
      ``choose_speed_q``, which refuses a banding that would leave a test fold
      without every class.

    Args:
        df: The feature table.
        y: Encoded labels, aligned with ``df``.
        split_mode: ``"recording"`` or ``"speed"``.

    Yields:
        ``(repeat, fold, train_idx, test_idx)`` with positional indices into
        ``df``.

    Raises:
        ValueError: If ``split_mode`` is unknown or ``y`` is misaligned.
    """
    if len(y) != len(df):
        raise ValueError(f"y has {len(y)} rows but df has {len(df)}.")
    if split_mode == "recording":
        groups = df["recording_id"].to_numpy()
    elif split_mode == "speed":
        _, groups = choose_speed_q(df, y)
    else:
        raise ValueError(
            f"Unknown split_mode {split_mode!r}; expected 'recording' or 'speed'."
        )

    x_dummy = np.zeros(len(y))
    for r in range(N_REPEATS):
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                  random_state=SEED + r)
        for f, (train_idx, test_idx) in enumerate(cv.split(x_dummy, y, groups)):
            # The two validity claims of the whole study, checked every time.
            assert_no_test_indices(train_idx, test_idx, f"outer split r{r}_f{f}")
            shared = np.intersect1d(groups[train_idx], groups[test_idx])
            if shared.size:
                raise ValueError(
                    f"outer split r{r}_f{f}: {shared.size} group(s) appear in "
                    f"both train and test under split_mode={split_mode!r}."
                )
            yield r, f, train_idx, test_idx


def inner_split(
    df: pd.DataFrame,
    y: NDArray[np.int64],
    train_idx: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Split one OUTER TRAINING fold into inner-train and inner-validation.

    Roughly 75/25, grouped by ``recording_id``.  This is the only data available
    to sensor ranking and forward selection; the outer test fold is never seen.

    Args:
        df: The feature table.
        y: Encoded labels, aligned with ``df``.
        train_idx: Positional indices of the outer training fold.

    Returns:
        ``(inner_train_idx, inner_val_idx)``, both subsets of ``train_idx``.
    """
    train_idx = np.asarray(train_idx)
    groups = df["recording_id"].to_numpy()[train_idx]
    cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)
    itr, iva = next(cv.split(np.zeros(len(train_idx)), y[train_idx], groups))
    inner_train, inner_val = train_idx[itr], train_idx[iva]

    # Both halves must stay inside the outer training fold, and must not share
    # a recording with each other.
    outside = np.setdiff1d(np.union1d(inner_train, inner_val), train_idx)
    if outside.size:
        raise ValueError(
            f"inner_split produced {outside.size} index/indices outside the "
            f"outer training fold (for example {outside[:5].tolist()})."
        )
    shared = np.intersect1d(groups[itr], groups[iva])
    if shared.size:
        raise ValueError(
            f"inner_split: {shared.size} recording(s) appear in both "
            "inner-train and inner-validation."
        )
    return inner_train, inner_val


# ----------------------------------------------------------------------------
# Run metadata (traceability: every output file can be traced to the exact
# code and environment that produced it)
# ----------------------------------------------------------------------------
TRACKED_PACKAGES: list[str] = [
    "numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "joblib",
]

METADATA_FILENAME: str = "run_metadata.json"


def git_state() -> dict[str, object]:
    """Return the repository commit and whether the working tree is dirty.

    A dirty tree means the code that produced a result is not exactly what the
    commit contains, which matters when a reviewer tries to reproduce it.

    Returns:
        ``{"commit": <sha or None>, "dirty": <bool or None>}``.  Both are None
        when this is not a git repository or git is unavailable, which is a
        legitimate state rather than an error.
    """
    def run(args: list[str]) -> str | None:
        result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    commit = run(["git", "rev-parse", "HEAD"])
    if commit is None:
        return {"commit": None, "dirty": None}
    status = run(["git", "status", "--porcelain"])
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def package_versions() -> dict[str, str]:
    """Return the installed version of every dependency the study relies on."""
    versions: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def run_metadata(script: str, **extra: object) -> dict[str, object]:
    """Build the metadata record written alongside every generated output.

    Args:
        script: Name of the script producing the output.
        **extra: Script-specific fields, for example the split mode or the
            number of input files.

    Returns:
        A JSON-serialisable record.
    """
    git = git_state()
    return {
        "script": script,
        "utc_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "package_versions": package_versions(),
        "seed": SEED,
        "command_line": " ".join(sys.argv),
        **extra,
    }


def write_run_metadata(directory: Path, script: str, **extra: object) -> Path:
    """Write ``run_metadata.json`` into ``directory``.

    Args:
        directory: Where the script's outputs are written.
        script: Name of the script.
        **extra: Script-specific fields.

    Returns:
        The path written.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / METADATA_FILENAME
    path.write_text(json.dumps(run_metadata(script, **extra), indent=2))
    LOG.info("Run metadata -> %s", path)
    return path


def read_run_metadata(directory: Path) -> dict[str, object] | None:
    """Read ``run_metadata.json`` from a directory, if it is there.

    Args:
        directory: Directory to look in.

    Returns:
        The record, or None when the directory predates metadata recording.
    """
    path = Path(directory) / METADATA_FILENAME
    if not path.exists():
        return None
    return json.loads(path.read_text())


def peak_memory_mb() -> float | None:
    """Return this process's peak resident memory in MiB.

    Uses only the standard library, so it adds no dependency.  Returns None
    when the platform does not expose the figure, which is reported honestly
    rather than guessed at.

    Returns:
        Peak resident set size in MiB, or None.
    """
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class _Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)

        kernel32 = ctypes.WinDLL("kernel32")
        # GetCurrentProcess returns a pseudo-handle; without an explicit
        # restype ctypes marshals it as a truncated int and the call fails.
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        handle = kernel32.GetCurrentProcess()

        get_info = ctypes.WinDLL("psapi").GetProcessMemoryInfo
        get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Counters),
                             wintypes.DWORD]
        get_info.restype = wintypes.BOOL
        if not get_info(handle, ctypes.byref(counters), counters.cb):
            return None
        return counters.PeakWorkingSetSize / (1024 ** 2)

    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS reports bytes.
    divisor = 1024.0 if sys.platform != "darwin" else 1024.0 ** 2
    return peak / divisor
