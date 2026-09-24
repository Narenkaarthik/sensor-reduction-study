"""Step 2: the go/no-go checkpoint, run BEFORE the expensive exhaustive study.

Four checks, all on repeat 0 only (5 folds), so the answer arrives in minutes:

  1. Leakage check.  The full-sensor model is trained twice: once with a naive
     random split over windows, and once with the correct grouped split.  A
     large gap means windows from one recording were leaking across the
     train/test boundary and inflating the naive score.
  2. Baseline.  Full-sensor macro-F1 under the grouped split.
  3. Single-sensor models.  If one sensor alone already matches all eight, the
     sensor-count curve will be flat and the study has nothing to measure.
  4. Sanity baselines.  Shuffled labels must collapse to chance, and the model
     must beat a most-frequent-class dummy by a wide margin.  If either fails,
     something is wrong with the pipeline, not with the sensors.

The naive split exists ONLY here, as the leakage measurement.  No reported
result anywhere else in the study splits windows at random.

Usage:
    python 02_checkpoint.py                    # grouped by recording
    python 02_checkpoint.py --split speed      # held-out speed bands
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold

import common
from common import (CLASS_ORDER, N_SPLITS, N_TREES, SEED, SENSORS,
                    SPEED_QUANTILE_CANDIDATES, TOLERANCE, FoldScorer,
                    assert_no_test_indices, choose_speed_q, encode_labels,
                    feature_columns, load_features, make_model, outer_splits,
                    speed_bands)

LOG = logging.getLogger(__name__)

Split = tuple[NDArray[np.int64], NDArray[np.int64]]

# A single sensor within this margin of all eight leaves nothing to measure.
CEILING_F1 = 0.99


def cv_score(
    df: pd.DataFrame,
    y: NDArray[np.int64],
    cols: list[str],
    splits: list[Split],
    n_trees: int = N_TREES,
) -> tuple[float, float]:
    """Mean and standard deviation of macro-F1 across the given splits.

    Args:
        df: The feature table.
        y: Encoded labels.
        cols: Feature columns to train on.
        splits: ``(train_idx, test_idx)`` pairs.
        n_trees: Forest size.

    Returns:
        ``(mean, sd)`` of macro-F1.
    """
    scores = []
    for i, (train_idx, test_idx) in enumerate(splits):
        assert_no_test_indices(train_idx, test_idx, "cv_score")
        scorer = FoldScorer.for_fold(y[test_idx], f"checkpoint fold {i}")
        model = make_model(n_trees).fit(df.iloc[train_idx][cols], y[train_idx])
        scores.append(scorer.macro_f1(y[test_idx],
                                      model.predict(df.iloc[test_idx][cols])))
    return float(np.mean(scores)), float(np.std(scores))


def shuffled_label_score(
    df: pd.DataFrame,
    y: NDArray[np.int64],
    cols: list[str],
    splits: list[Split],
    n_trees: int = N_TREES,
) -> tuple[float, float]:
    """Train on permuted labels and score against the TRUE test labels.

    The permutation is applied strictly inside the training fold; the test
    labels are never touched.  Any real signal is therefore destroyed while the
    evaluation stays honest, so the result must fall to chance.  A score well
    above chance means information is reaching the model some other way.

    Args:
        df: The feature table.
        y: Encoded labels.
        cols: Feature columns to train on.
        splits: ``(train_idx, test_idx)`` pairs.
        n_trees: Forest size.

    Returns:
        ``(mean, sd)`` of macro-F1.
    """
    rng = np.random.default_rng(SEED)
    scores = []
    for i, (train_idx, test_idx) in enumerate(splits):
        assert_no_test_indices(train_idx, test_idx, "shuffled_label_score")
        scorer = FoldScorer.for_fold(y[test_idx], f"checkpoint fold {i}")
        y_shuffled = rng.permutation(y[train_idx])
        model = make_model(n_trees).fit(df.iloc[train_idx][cols], y_shuffled)
        scores.append(scorer.macro_f1(y[test_idx],
                                      model.predict(df.iloc[test_idx][cols])))
    return float(np.mean(scores)), float(np.std(scores))


def dummy_score(
    df: pd.DataFrame,
    y: NDArray[np.int64],
    cols: list[str],
    splits: list[Split],
) -> tuple[float, float]:
    """Score a most-frequent-class baseline, the floor any model must clear.

    Args:
        df: The feature table.
        y: Encoded labels.
        cols: Feature columns (unused by the dummy, kept for a uniform call).
        splits: ``(train_idx, test_idx)`` pairs.

    Returns:
        ``(mean, sd)`` of macro-F1.
    """
    scores = []
    for i, (train_idx, test_idx) in enumerate(splits):
        assert_no_test_indices(train_idx, test_idx, "dummy_score")
        scorer = FoldScorer.for_fold(y[test_idx], f"checkpoint fold {i}")
        model = DummyClassifier(strategy="most_frequent").fit(
            df.iloc[train_idx][cols], y[train_idx])
        scores.append(scorer.macro_f1(y[test_idx],
                                      model.predict(df.iloc[test_idx][cols])))
    return float(np.mean(scores)), float(np.std(scores))


def naive_splits(y: NDArray[np.int64]) -> list[Split]:
    """Build the leakage-check split: random over windows, ignoring recordings.

    Uses the same number of folds as the real protocol so that the two scores
    differ only in the grouping, never in the amount of training data.

    Args:
        y: Encoded labels.

    Returns:
        ``N_SPLITS`` ``(train_idx, test_idx)`` pairs.
    """
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    return list(cv.split(np.zeros(len(y)), y))


def decide(full: float, best_single: float, best_sensor: str) -> str:
    """Apply the Section 9 decision rules and log the recommendation.

    Args:
        full: Full-sensor macro-F1 under the grouped split.
        best_single: Best single-sensor macro-F1.
        best_sensor: Which sensor achieved it.

    Returns:
        ``"CEILING EFFECT"``, ``"WARNING"`` or ``"OK"``.
    """
    gap = full - best_single
    LOG.info("full = %.4f, best single sensor = %s (%.4f), gap = %.4f",
             full, best_sensor, best_single, gap)

    if full >= CEILING_F1 and gap <= TOLERANCE:
        verdict = "CEILING EFFECT"
        LOG.warning("CEILING EFFECT: the task is too easy for a sensor-count study.")
        LOG.warning("  -> rerun with --split speed, so whole speed bands are held out.")
    elif gap <= TOLERANCE:
        verdict = "WARNING"
        LOG.warning("One sensor already matches all eight; the curve may be flat.")
        LOG.warning("  -> consider --split speed before the full experiment.")
    else:
        verdict = "OK"
        LOG.info("OK: there is room between one sensor and all eight.")
        LOG.info("  -> proceed to 03_run_experiments.py with this split.")

    if best_sensor == "tach":
        LOG.warning(
            "The tachometer is the single best sensor. That is suspicious: it "
            "measures shaft speed, not machine health, so this may be a "
            "recording artefact (speed correlating with condition) rather than "
            "a diagnostic result. Investigate before trusting the ranking."
        )
    return verdict


def report_class_coverage(
    df: pd.DataFrame,
    y: NDArray[np.int64],
    split_mode: str,
) -> list[dict[str, object]]:
    """Log class coverage per speed band and per test fold.

    A test fold missing a class cannot be compared with a full one, so this is
    the information needed to choose between the two split modes BEFORE
    committing to the expensive run.

    Args:
        df: The feature table.
        y: Encoded labels.
        split_mode: The mode whose folds are reported.

    Returns:
        One row per band and per fold, for the saved results file.
    """
    rows: list[dict[str, object]] = []

    LOG.info("=" * 68)
    LOG.info("0. CLASS COVERAGE")
    LOG.info("   Recordings per class in each speed band "
             "(q=%d, the first candidate):", SPEED_QUANTILE_CANDIDATES[0])
    bands = speed_bands(df, SPEED_QUANTILE_CANDIDATES[0])
    header = "  ".join(f"{c[:5]:>5}" for c in CLASS_ORDER)
    LOG.info("     band   n_cls  %s   speed range", header)
    for band in np.unique(bands):
        mask = bands == band
        counts = [int(df.loc[mask & (y == i), "recording_id"].nunique())
                  for i in range(len(CLASS_ORDER))]
        n_present = sum(1 for c in counts if c > 0)
        lo, hi = df.loc[mask, "speed_hz"].min(), df.loc[mask, "speed_hz"].max()
        flag = "" if n_present == len(CLASS_ORDER) else "   <-- incomplete"
        LOG.info("     %4d   %4d   %s   %.1f-%.1f Hz%s", band, n_present,
                 "  ".join(f"{c:>5d}" for c in counts), lo, hi, flag)
        rows.append({"split_mode": split_mode, "kind": "band_coverage",
                     "config": f"band_{band}", "f1_mean": float(n_present),
                     "f1_std": float("nan")})

    LOG.info("   Classes present in each test fold (%s split, repeat 0):",
             split_mode)
    for r, f, _, test_idx in outer_splits(df, y, split_mode):
        if r != 0:
            continue
        present = np.unique(y[test_idx])
        absent = [CLASS_ORDER[i] for i in range(len(CLASS_ORDER))
                  if i not in present]
        LOG.info("     r%d_f%d: %d/%d classes, %d windows%s", r, f, len(present),
                 len(CLASS_ORDER), len(test_idx),
                 "" if not absent else f"   <-- missing {absent}")
        rows.append({"split_mode": split_mode, "kind": "fold_coverage",
                     "config": f"r{r}_f{f}", "f1_mean": float(len(present)),
                     "f1_std": float("nan")})
    return rows


def run_checkpoint(
    df: pd.DataFrame,
    y: NDArray[np.int64],
    split_mode: str = "recording",
    n_trees: int = N_TREES,
) -> pd.DataFrame:
    """Run all four checks and return the results table.

    Args:
        df: The feature table.
        y: Encoded labels.
        split_mode: ``"recording"`` or ``"speed"``.
        n_trees: Forest size.

    Returns:
        One row per reported number, with a ``kind`` column so that every
        figure in the paper can be traced back to a line in this file.
    """
    grouped = [(tr, te) for r, _, tr, te in outer_splits(df, y, split_mode) if r == 0]
    naive = naive_splits(y)
    all_cols = feature_columns(df, SENSORS)
    rows: list[dict[str, object]] = report_class_coverage(df, y, split_mode)

    def record(kind: str, config: str, mean: float, sd: float) -> None:
        rows.append({"split_mode": split_mode, "kind": kind, "config": config,
                     "f1_mean": mean, "f1_std": sd})

    LOG.info("=" * 68)
    LOG.info("1. LEAKAGE CHECK (all sensors, %d folds)", N_SPLITS)
    naive_mean, naive_sd = cv_score(df, y, all_cols, naive, n_trees)
    full_mean, full_sd = cv_score(df, y, all_cols, grouped, n_trees)
    LOG.info("   naive random window split : %.4f +/- %.4f", naive_mean, naive_sd)
    LOG.info("   grouped (%-9s) split : %.4f +/- %.4f", split_mode, full_mean, full_sd)
    LOG.info("   inflation from leakage    : %+.4f", naive_mean - full_mean)
    record("leakage", "naive_window_split", naive_mean, naive_sd)
    record("leakage", f"grouped_{split_mode}_split", full_mean, full_sd)
    record("leakage_gap", "naive_minus_grouped", naive_mean - full_mean, float("nan"))

    LOG.info("=" * 68)
    LOG.info("2. BASELINE (all sensors, grouped split)")
    LOG.info("   %.4f +/- %.4f", full_mean, full_sd)
    record("baseline", "ALL SENSORS", full_mean, full_sd)

    LOG.info("=" * 68)
    LOG.info("3. SINGLE-SENSOR MODELS (grouped split)")
    singles: dict[str, float] = {}
    for sensor in SENSORS:
        mean, sd = cv_score(df, y, feature_columns(df, [sensor]), grouped, n_trees)
        singles[sensor] = mean
        LOG.info("   %-8s: %.4f +/- %.4f", sensor, mean, sd)
        record("single_sensor", sensor, mean, sd)

    LOG.info("=" * 68)
    LOG.info("4. SANITY BASELINES (grouped split)")
    shuf_mean, shuf_sd = shuffled_label_score(df, y, all_cols, grouped, n_trees)
    dum_mean, dum_sd = dummy_score(df, y, all_cols, grouped)
    LOG.info("   shuffled training labels  : %.4f +/- %.4f  (must be near chance)",
             shuf_mean, shuf_sd)
    LOG.info("   most-frequent dummy       : %.4f +/- %.4f", dum_mean, dum_sd)
    record("sanity", "shuffled_labels", shuf_mean, shuf_sd)
    record("sanity", "dummy_most_frequent", dum_mean, dum_sd)

    LOG.info("=" * 68)
    LOG.info("DECISION GUIDE")
    best_sensor = max(singles, key=lambda s: singles[s])
    verdict = decide(full_mean, singles[best_sensor], best_sensor)
    record("decision", verdict, full_mean - singles[best_sensor], float("nan"))
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="recording", choices=["recording", "speed"],
                    help="outer-split grouping")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    df = load_features()
    y = encode_labels(df)
    results = run_checkpoint(df, y, args.split)

    out_dir: Path = common.RESULTS_DIR / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "checkpoint_results.csv"
    results.to_csv(out_path, index=False)
    LOG.info("Saved -> %s", out_path)


if __name__ == "__main__":
    main()
