"""Step 3: the main experiment.

For every outer split (5 folds x 2 repeats):

  a) Rank the sensors by GROUPED permutation importance, using the training
     fold only, via an inner train/validation split.
  b) Run greedy forward selection, also on the training fold only.
  c) Train and test a model on EVERY non-empty sensor subset: 255 of them for
     8 sensors, which includes the full configuration and all 8 single-sensor
     models.

(a) and (b) never see the outer test fold; this is enforced by assertions, not
comments.  Because the model and the seed are fixed, a "selected" subset's test
score is simply its entry in the exhaustive results, so nothing is refit and no
selected score can disagree with its exhaustive counterpart.

Results are written per split, so an interrupted run resumes where it stopped.

Usage:
    python 03_run_experiments.py                  # full run (hours)
    python 03_run_experiments.py --split speed    # held-out speed bands
    python 03_run_experiments.py --quick          # first split only, 50 trees
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

import common
from common import (N_REPEATS, N_SPLITS, N_TREES, PERM_REPEATS, SEED, SENSORS,
                    FoldScorer, assert_no_test_indices, encode_labels,
                    feature_columns, inner_split, load_features, make_model,
                    outer_splits, peak_memory_mb, subset_name,
                    write_run_metadata)

LOG = logging.getLogger(__name__)

QUICK_TREES = 50


def format_duration(seconds: float) -> str:
    """Render a duration as a compact human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        For example ``"34s"``, ``"6m 12s"`` or ``"2h 41m"``.
    """
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60):02d}m"


def all_subsets() -> list[tuple[str, ...]]:
    """Every non-empty sensor subset, ordered by size then by ``SENSORS`` order.

    Returns:
        The 255 subsets for 8 sensors.
    """
    subsets = [c for k in range(1, len(SENSORS) + 1)
               for c in combinations(SENSORS, k)]
    expected = 2 ** len(SENSORS) - 1
    if len(subsets) != expected:
        raise ValueError(f"Built {len(subsets)} subsets, expected {expected}.")
    return subsets


def draw_permutations(n_rows: int, n_repeats: int = PERM_REPEATS
                      ) -> list[NDArray[np.int64]]:
    """Draw the row permutations used by the grouped permutation importance.

    Drawn ONCE per split and reused for all 8 sensors, so the per-sensor drops
    are paired (common random numbers).  The ranking comes from comparing
    sensors against each other, so holding the randomness fixed removes shuffle
    noise from that comparison.

    Args:
        n_rows: Number of validation rows to permute.
        n_repeats: How many permutations to draw.

    Returns:
        ``n_repeats`` permutation arrays.
    """
    rng = np.random.default_rng(SEED)
    return [rng.permutation(n_rows) for _ in range(n_repeats)]


def grouped_permutation_importance(
    df: pd.DataFrame,
    y: NDArray[np.int64],
    inner_train: NDArray[np.int64],
    inner_val: NDArray[np.int64],
    test_idx: NDArray[np.int64],
    n_trees: int,
    scorer: FoldScorer,
) -> tuple[list[str], dict[str, float]]:
    """Rank sensors by the macro-F1 drop caused by shuffling them.

    All 18 columns of a sensor are shuffled together with ONE shared row
    permutation, so the sensor is destroyed as a unit rather than having its
    features scrambled against each other.

    Args:
        df: The feature table.
        y: Encoded labels.
        inner_train: Indices to fit on.
        inner_val: Indices to measure the drop on.
        test_idx: The outer test fold, passed only so the guard can check it.
        n_trees: Forest size.
        scorer: Scorer fixed to the inner-validation label set.

    Returns:
        ``(ranking, importance)``; the ranking is by mean drop, largest first,
        ties broken by ``SENSORS`` order.
    """
    assert_no_test_indices(inner_train, test_idx, "permutation importance (fit)")
    assert_no_test_indices(inner_val, test_idx, "permutation importance (score)")

    cols = feature_columns(df, SENSORS)
    model = make_model(n_trees).fit(df.iloc[inner_train][cols], y[inner_train])
    x_val = df.iloc[inner_val][cols].reset_index(drop=True)
    y_val = y[inner_val]
    base = scorer.macro_f1(y_val, model.predict(x_val))

    permutations = draw_permutations(len(x_val))
    importance: dict[str, float] = {}
    for sensor in SENSORS:
        sensor_cols = feature_columns(df, [sensor])
        original = x_val[sensor_cols].to_numpy()
        drops = []
        for perm in permutations:
            x_perm = x_val.copy()
            x_perm[sensor_cols] = original[perm]
            drops.append(base - scorer.macro_f1(y_val, model.predict(x_perm)))
        importance[sensor] = float(np.mean(drops))

    ranking = sorted(SENSORS, key=lambda s: (-importance[s], SENSORS.index(s)))
    return ranking, importance


def forward_selection(
    df: pd.DataFrame,
    y: NDArray[np.int64],
    inner_train: NDArray[np.int64],
    inner_val: NDArray[np.int64],
    test_idx: NDArray[np.int64],
    n_trees: int,
    scorer: FoldScorer,
) -> tuple[list[str], list[dict[str, object]]]:
    """Greedily add the sensor that most improves inner-validation macro-F1.

    Ties are broken by ``SENSORS`` order: candidates are tried in that order and
    a later candidate must be strictly better to displace an earlier one.

    Args:
        df: The feature table.
        y: Encoded labels.
        inner_train: Indices to fit on.
        inner_val: Indices to score on.
        test_idx: The outer test fold, passed only so the guard can check it.
        n_trees: Forest size.
        scorer: Scorer fixed to the inner-validation label set.

    Returns:
        ``(order, path)``: the full 8-sensor order, and the inner score after
        each addition.
    """
    assert_no_test_indices(inner_train, test_idx, "forward selection (fit)")
    assert_no_test_indices(inner_val, test_idx, "forward selection (score)")

    chosen: list[str] = []
    remaining = list(SENSORS)
    path: list[dict[str, object]] = []
    y_val = y[inner_val]

    while remaining:
        best_sensor, best_score = None, -np.inf
        for sensor in remaining:            # remaining stays in SENSORS order
            cols = feature_columns(df, chosen + [sensor])
            model = make_model(n_trees).fit(df.iloc[inner_train][cols],
                                            y[inner_train])
            score = scorer.macro_f1(y_val, model.predict(df.iloc[inner_val][cols]))
            if score > best_score:          # strict: first best wins the tie
                best_sensor, best_score = sensor, score
        chosen.append(best_sensor)
        remaining.remove(best_sensor)
        path.append({"k": len(chosen), "added": best_sensor,
                     "inner_f1": float(best_score)})
    return chosen, path


def run_split(
    df: pd.DataFrame,
    y: NDArray[np.int64],
    repeat: int,
    fold: int,
    train_idx: NDArray[np.int64],
    test_idx: NDArray[np.int64],
    out_dir: Path,
    n_trees: int,
) -> float:
    """Run selection and the exhaustive evaluation for one outer split.

    Args:
        df: The feature table.
        y: Encoded labels.
        repeat: Repeat index.
        fold: Fold index.
        train_idx: Outer training indices.
        test_idx: Outer test indices.
        out_dir: Directory for this split mode's results.
        n_trees: Forest size.

    Returns:
        Elapsed seconds, used to estimate the time remaining.
    """
    tag = f"r{repeat}_f{fold}"
    started = time.time()
    LOG.info("[%s] train=%d test=%d", tag, len(train_idx), len(test_idx))

    # One label set for this fold, reused by every subset in it.
    scorer = FoldScorer.for_fold(y[test_idx], f"{tag} test")
    LOG.info("[%s] scoring on %d classes", tag, len(scorer.labels))

    # (a) and (b): selection sees the training fold ONLY.
    inner_train, inner_val = inner_split(df, y, train_idx)
    inner_scorer = FoldScorer.for_fold(y[inner_val], f"{tag} inner-val")
    ranking, importance = grouped_permutation_importance(
        df, y, inner_train, inner_val, test_idx, n_trees, inner_scorer)
    forward_order, forward_path = forward_selection(
        df, y, inner_train, inner_val, test_idx, n_trees, inner_scorer)
    LOG.info("[%s] permutation ranking: %s", tag, ranking)
    LOG.info("[%s] forward order      : %s", tag, forward_order)

    # (c) exhaustive evaluation against the untouched test fold.
    y_test = y[test_idx]
    rows: list[dict[str, object]] = []
    preds: dict[str, NDArray[np.int8]] = {}
    for subset in all_subsets():
        cols = feature_columns(df, subset)
        model = make_model(n_trees).fit(df.iloc[train_idx][cols], y[train_idx])
        predicted = model.predict(df.iloc[test_idx][cols])
        name = subset_name(subset)
        preds[name] = predicted.astype(np.int8)
        rows.append({
            "repeat": repeat, "fold": fold, "k": len(subset), "subset": name,
            "macro_f1": scorer.macro_f1(y_test, predicted),
            "accuracy": scorer.accuracy(y_test, predicted),
        })

    pd.DataFrame(rows).to_csv(out_dir / f"scores_{tag}.csv", index=False)
    np.savez_compressed(out_dir / f"preds_{tag}.npz",
                        test_idx=np.asarray(test_idx),
                        y_true=y_test.astype(np.int8), **preds)
    selection = {
        "repeat": repeat, "fold": fold,
        "label_set": list(scorer.labels),
        "inner_label_set": list(inner_scorer.labels),
        "perm_ranking": ranking,
        "perm_importance": importance,
        "forward_order": forward_order,
        "forward_path": forward_path,
    }
    (out_dir / f"selection_{tag}.json").write_text(json.dumps(selection, indent=2))
    elapsed = time.time() - started
    LOG.info("[%s] %d subsets in %s", tag, len(rows), format_duration(elapsed))
    return elapsed



def dry_run(
    df: pd.DataFrame,
    y: NDArray[np.int64],
    out_dir: Path,
    split_mode: str = "recording",
    n_trees: int = N_TREES,
) -> dict[str, object]:
    """Run ONE split at the full configuration and project the whole study.

    This exists so that the cost of the real run is a measured number rather
    than a guess.  It uses the production forest size, so the timing is
    representative; every split trains on the same proportion of the data, so
    one split scales linearly to all of them.

    Args:
        df: The feature table.
        y: Encoded labels.
        out_dir: Directory for the probe's own results; kept separate from the
            real results so it can never be mistaken for part of a study run.
        split_mode: ``"recording"`` or ``"speed"``.
        n_trees: Forest size; the production value by default.

    Returns:
        The measured and projected figures, also written to disk.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total_splits = N_SPLITS * N_REPEATS

    splits = list(outer_splits(df, y, split_mode))
    repeat, fold, train_idx, test_idx = splits[0]

    LOG.info("=" * 68)
    LOG.info("DRY RUN: one split at the full configuration (%d trees)", n_trees)
    LOG.info("  feature table : %d rows x %d columns", df.shape[0], df.shape[1])
    LOG.info("  subsets/split : %d", len(all_subsets()))
    LOG.info("  splits total  : %d", total_splits)
    LOG.info("=" * 68)

    started = time.time()
    run_split(df, y, repeat, fold, train_idx, test_idx, out_dir, n_trees)
    elapsed = time.time() - started
    peak_mb = peak_memory_mb()

    projected = elapsed * total_splits
    result = {
        "split_mode": split_mode,
        "n_trees": n_trees,
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "n_subsets": len(all_subsets()),
        "splits_total": total_splits,
        "measured_split": f"r{repeat}_f{fold}",
        "measured_seconds": round(elapsed, 1),
        "measured_wall_time": format_duration(elapsed),
        "peak_memory_mb": None if peak_mb is None else round(peak_mb, 1),
        "projected_total_seconds": round(projected, 1),
        "projected_total_wall_time": format_duration(projected),
    }

    LOG.info("=" * 68)
    LOG.info("DRY RUN RESULT")
    LOG.info("  measured split      : %s", result["measured_split"])
    LOG.info("  measured wall time  : %s (%.1f s)", result["measured_wall_time"],
             elapsed)
    if peak_mb is None:
        LOG.info("  peak memory         : not available on this platform")
    else:
        LOG.info("  peak memory         : %.1f MiB", peak_mb)
    LOG.info("  PROJECTED TOTAL     : %s for all %d splits",
             result["projected_total_wall_time"], total_splits)
    LOG.info("  (projection assumes every split costs the same; they train on")
    LOG.info("   the same proportion of the data, so this scales linearly)")
    LOG.info("=" * 68)
    LOG.info("Probe results are in %s and are NOT part of a study run.", out_dir)

    (out_dir / "dry_run.json").write_text(json.dumps(result, indent=2))
    write_run_metadata(out_dir, "03_run_experiments.py --dry-run", **result)
    return result


def run_experiments(
    df: pd.DataFrame,
    y: NDArray[np.int64],
    out_dir: Path,
    split_mode: str = "recording",
    n_trees: int = N_TREES,
    quick: bool = False,
) -> Path:
    """Run every outer split that has not already been written.

    Args:
        df: The feature table.
        y: Encoded labels.
        out_dir: Where to write results.
        split_mode: ``"recording"`` or ``"speed"``.
        n_trees: Forest size.
        quick: Restrict to the FIRST split only.  The restriction is applied
            before the resume check, so a repeated quick run is a no-op rather
            than quietly computing a different split.

    Returns:
        ``out_dir``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = list(outer_splits(df, y, split_mode))
    if quick:
        splits = splits[:1]
        LOG.info("Quick mode: first split only, %d trees", n_trees)

    pending = [s for s in splits
               if not (out_dir / f"scores_r{s[0]}_f{s[1]}.csv").exists()]
    LOG.info("%d split(s) to run, %d already present",
             len(pending), len(splits) - len(pending))

    elapsed_per_split: list[float] = []
    for position, (repeat, fold, train_idx, test_idx) in enumerate(splits):
        tag = f"r{repeat}_f{fold}"
        if (out_dir / f"scores_{tag}.csv").exists():
            LOG.info("[%s] already done, skipping", tag)
            continue

        elapsed_per_split.append(
            run_split(df, y, repeat, fold, train_idx, test_idx, out_dir, n_trees))

        # A running estimate, from the splits actually computed in this session,
        # so an interrupted-and-resumed run does not report a misleading ETA.
        done = len(elapsed_per_split)
        remaining = len(pending) - done
        if remaining:
            mean_elapsed = float(np.mean(elapsed_per_split))
            LOG.info("   progress: %d/%d splits this session, mean %s each, "
                     "about %s remaining", done, len(pending),
                     format_duration(mean_elapsed),
                     format_duration(mean_elapsed * remaining))

    if elapsed_per_split:
        LOG.info("Ran %d split(s) in %s", len(elapsed_per_split),
                 format_duration(float(np.sum(elapsed_per_split))))
    write_run_metadata(
        out_dir, "03_run_experiments.py",
        split_mode=split_mode,
        n_trees=n_trees,
        quick=quick,
        n_splits_total=len(splits),
        n_splits_run_this_session=len(elapsed_per_split),
        n_subsets=len(all_subsets()),
        n_rows=int(df.shape[0]),
        seconds_per_split=[round(e, 1) for e in elapsed_per_split],
    )
    return out_dir


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="recording", choices=["recording", "speed"],
                    help="outer-split grouping")
    ap.add_argument("--quick", action="store_true",
                    help=f"{QUICK_TREES} trees, first split only; written to "
                         "results/quick_<split>/ so it never mixes with real results")
    ap.add_argument("--dry-run", action="store_true",
                    help="run ONE split at the full configuration and report the "
                         "measured wall time, peak memory and the projected total "
                         "for all splits; written to results/dryrun_<split>/")
    args = ap.parse_args(argv)
    if args.quick and args.dry_run:
        ap.error("--quick and --dry-run are mutually exclusive: --quick uses a "
                 "reduced forest, --dry-run must use the production one.")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    df = load_features()
    y = encode_labels(df)

    if args.dry_run:
        dry_run(df, y, common.RESULTS_DIR / f"dryrun_{args.split}", args.split)
        return

    folder = f"quick_{args.split}" if args.quick else args.split
    run_experiments(df, y, common.RESULTS_DIR / folder, args.split,
                    QUICK_TREES if args.quick else N_TREES, args.quick)
    LOG.info("Results -> %s", common.RESULTS_DIR / folder)


if __name__ == "__main__":
    main()
