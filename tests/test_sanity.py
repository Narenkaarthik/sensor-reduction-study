"""Level E: sanity tests.

These check that the pipeline measures real signal rather than an artefact.
If a model can score well on shuffled labels, or barely beats a dummy, then
nothing downstream means anything.

"""
from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest

from common import (CLASS_ORDER, SENSORS, FoldScorer, feature_columns,
                    make_model, outer_splits, per_class_recall)
from conftest import load_script

# Small forests keep the suite fast; the sanity gaps are far too wide for the
# forest size to change any verdict.
TEST_TREES = 50

# Chance level for 6 balanced classes is about 0.167.
CHANCE_CEILING = 0.35

# The calibrated band for the hard fixture, from make_synthetic_data.py.
HARD_F1_BAND = (0.80, 0.90)


@pytest.fixture(scope="module")
def checkpoint() -> ModuleType:
    """The ``02_checkpoint`` module."""
    return load_script("02_checkpoint")


def _repeat_zero(df, y, mode: str = "recording") -> list:
    """The 5 folds of repeat 0, which is what the checkpoint uses."""
    return [(tr, te) for r, _, tr, te in outer_splits(df, y, mode) if r == 0]


def _score(df, y, cols: list[str], train_idx, test_idx,
           n_trees: int | None = None) -> float:
    """Fit one subset and score it on its fold's own label set."""
    scorer = FoldScorer.for_fold(y[test_idx], "sanity fold")
    model = make_model(n_trees) if n_trees else make_model()
    model.fit(df.iloc[train_idx][cols], y[train_idx])
    return scorer.macro_f1(y[test_idx], model.predict(df.iloc[test_idx][cols]))


def test_shuffled_labels_collapse_to_chance(labelled, checkpoint: ModuleType) -> None:
    """Permuting the training labels destroys all performance."""
    df, y = labelled
    mean, _ = checkpoint.shuffled_label_score(
        df, y, feature_columns(df, SENSORS), _repeat_zero(df, y), TEST_TREES)
    assert mean < CHANCE_CEILING, f"shuffled labels scored {mean:.3f}"


def test_shuffled_labels_do_not_alter_test_labels(labelled,
                                                  checkpoint: ModuleType) -> None:
    """The shuffle touches the training fold only; y is left unmodified."""
    df, y = labelled
    before = y.copy()
    checkpoint.shuffled_label_score(
        df, y, feature_columns(df, SENSORS), _repeat_zero(df, y), TEST_TREES)
    np.testing.assert_array_equal(y, before)


def test_full_model_beats_dummy_by_a_wide_margin(labelled,
                                                 checkpoint: ModuleType) -> None:
    """The full-sensor model must be far above a most-frequent-class baseline."""
    df, y = labelled
    cols = feature_columns(df, SENSORS)
    splits = _repeat_zero(df, y)
    full, _ = checkpoint.cv_score(df, y, cols, splits, TEST_TREES)
    dummy, _ = checkpoint.dummy_score(df, y, cols, splits)
    assert dummy < CHANCE_CEILING, f"dummy scored {dummy:.3f}"
    assert full - dummy > 0.5, f"full {full:.3f} vs dummy {dummy:.3f}"


def test_naive_split_is_not_the_reported_protocol(labelled,
                                                  checkpoint: ModuleType) -> None:
    """The leakage-check split is random over windows and uses the same folds."""
    df, y = labelled
    naive = checkpoint.naive_splits(y)
    recordings = df["recording_id"].to_numpy()
    assert len(naive) == 5
    shares = [np.intersect1d(recordings[tr], recordings[te]).size
              for tr, te in naive]
    assert all(s > 0 for s in shares), (
        "the naive split is supposed to split windows at random, so it should "
        "share recordings across the boundary; that is the leak it measures"
    )


def test_fold_scorer_uses_the_folds_own_label_set() -> None:
    """A fold is scored on the classes IT contains, never on all 6 blindly.

    Five classes present and all predicted perfectly must score 1.0, not 5/6:
    the fold is not penalised for a class it never had.
    """
    y_true = np.array([0, 1, 2, 3, 4])
    scorer = FoldScorer.for_fold(y_true, "probe")
    assert scorer.labels == (0, 1, 2, 3, 4)
    assert scorer.macro_f1(y_true, y_true) == pytest.approx(1.0)


def test_fold_scorer_penalises_a_class_it_never_predicts() -> None:
    """A class that IS present but is never predicted still scores 0."""
    y_true = np.array([0, 1, 2, 3, 4, 5])
    y_pred = np.array([0, 1, 2, 3, 4, 4])      # class 5 never predicted
    scorer = FoldScorer.for_fold(y_true, "probe")
    assert scorer.macro_f1(y_true, y_pred) < 1.0


def test_fold_scorer_refuses_a_different_label_set() -> None:
    """Every subset in a fold must be scored on an identical class set."""
    scorer = FoldScorer.for_fold(np.array([0, 1, 2, 3, 4, 5]), "probe")
    with pytest.raises(ValueError, match="identical class set"):
        scorer.macro_f1(np.array([0, 1, 2, 3, 4]), np.array([0, 1, 2, 3, 4]))


def test_fold_scorer_refuses_a_degenerate_fold() -> None:
    """Fewer than 5 classes is a broken split, not a score to report."""
    with pytest.raises(ValueError, match="Refusing to score"):
        FoldScorer.for_fold(np.array([0, 1, 2, 3]), "probe")


def test_per_class_recall_marks_absent_classes_as_nan() -> None:
    """Absent classes are NaN so averaging excludes them; present-but-missed is 0."""
    recall = per_class_recall(np.array([0, 0, 1, 1]), np.array([0, 0, 2, 2]))
    assert len(recall) == len(CLASS_ORDER)
    assert recall[0] == pytest.approx(1.0)
    assert recall[1] == pytest.approx(0.0)         # present, never predicted
    assert np.isnan(recall[2:]).all()              # absent from y_true
    assert np.nanmean(recall) == pytest.approx(0.5)


def test_per_class_recall_length_is_always_six(labelled) -> None:
    """Per-class recall has one entry per class on every real split."""
    df, y = labelled
    cols = feature_columns(df, SENSORS)
    for train_idx, test_idx in _repeat_zero(df, y):
        model = make_model(TEST_TREES).fit(df.iloc[train_idx][cols], y[train_idx])
        recall = per_class_recall(y[test_idx], model.predict(df.iloc[test_idx][cols]))
        assert recall.shape == (len(CLASS_ORDER),)
        finite = recall[~np.isnan(recall)]
        assert np.all((finite >= 0.0) & (finite <= 1.0))


def test_ceiling_fixture_is_at_ceiling(labelled, checkpoint: ModuleType) -> None:
    """The default fixture is separable, which is what makes it the ceiling case."""
    df, y = labelled
    full, _ = checkpoint.cv_score(df, y, feature_columns(df, SENSORS),
                                  _repeat_zero(df, y), TEST_TREES)
    assert full > 0.95, f"ceiling fixture scored {full:.3f}"


@pytest.mark.slow
def test_hard_fixture_lands_in_its_calibrated_band(hard_labelled) -> None:
    """The hard fixture must stay inside 0.80-0.90 with the real forest size.

    HARD_AMPLITUDE_SCALE was calibrated to put the all-sensor score here.  If
    this fails, the fixture drifted and the Phase 4 figures would no longer
    demonstrate a non-degenerate curve.
    """
    df, y = hard_labelled
    cols = feature_columns(df, SENSORS)
    scores = [_score(df, y, cols, tr, te) for tr, te in _repeat_zero(df, y)]
    mean = float(np.mean(scores))
    assert HARD_F1_BAND[0] <= mean <= HARD_F1_BAND[1], (
        f"hard fixture scored {mean:.3f}, outside {HARD_F1_BAND}; "
        "recalibrate HARD_AMPLITUDE_SCALE"
    )


@pytest.mark.slow
def test_hard_fixture_leaves_room_between_one_sensor_and_all(hard_labelled) -> None:
    """The hard fixture has a real gap, so the sensor-count curve is not flat."""
    df, y = hard_labelled
    splits = _repeat_zero(df, y)

    def score(sensors: list[str]) -> float:
        cols = feature_columns(df, sensors)
        return float(np.mean([_score(df, y, cols, tr, te) for tr, te in splits]))

    full = score(SENSORS)
    best_single = max(score([s]) for s in SENSORS)
    assert full - best_single > 0.05, (
        f"full {full:.3f} vs best single {best_single:.3f}: too flat to be "
        "a useful demonstration fixture"
    )


def test_decision_guide_rules(checkpoint: ModuleType) -> None:
    """The Section 9 decision thresholds behave exactly as specified."""
    assert checkpoint.decide(0.995, 0.990, "ub_rad") == "CEILING EFFECT"
    assert checkpoint.decide(0.800, 0.795, "ub_rad") == "WARNING"
    assert checkpoint.decide(0.800, 0.600, "ub_rad") == "OK"
    # A ceiling verdict needs BOTH conditions, not just a small gap.
    assert checkpoint.decide(0.980, 0.975, "ub_rad") == "WARNING"


def test_tach_best_sensor_raises_a_warning(checkpoint: ModuleType, caplog) -> None:
    """A tachometer-led ranking is flagged as a possible recording artefact."""
    with caplog.at_level("WARNING"):
        checkpoint.decide(0.800, 0.600, "tach")
    assert any("recording artefact" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Level E, completed in Phase 3
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_strong_channels_outrank_the_noise_channels(hard_labelled) -> None:
    """Every strongly injected channel outranks ub_ax and ob_rad, which are noise.

    The synthetic generator decides which channels carry the fault signatures,
    so the ranking must recover that.  Measured worst margin across all 10
    splits: +0.0165.

    ``mic`` is deliberately NOT asserted here.  It carries only WEAK copies of
    each signature, and permutation importance measures a sensor's MARGINAL
    contribution: once the strong channels are present, dropping mic costs
    almost nothing, so its importance sits at zero and can go slightly negative
    (measured range -0.0195 to +0.0160 on the hard fixture; exactly 0.0000 on
    the ceiling fixture, where the strong channels already separate the classes
    perfectly).  That is correct behaviour for a redundant sensor, and it is
    the very redundancy this study sets out to quantify, so the fixture is left
    alone rather than tuned to make the assertion convenient.
    """
    from common import inner_split

    runner = load_script("03_run_experiments")
    df, y = hard_labelled
    strong = ["ub_rad", "ub_tan", "ob_tan", "ob_ax"]
    noise = ["ub_ax", "ob_rad"]

    for r, f, train_idx, test_idx in outer_splits(df, y, "recording"):
        if r != 0:
            continue
        inner_train, inner_val = inner_split(df, y, train_idx)
        scorer = FoldScorer.for_fold(y[inner_val], f"r{r}_f{f} inner-val")
        ranking, importance = runner.grouped_permutation_importance(
            df, y, inner_train, inner_val, test_idx, TEST_TREES, scorer)

        worst_noise = max(importance[s] for s in noise)
        for sensor in strong:
            assert importance[sensor] > worst_noise, (
                f"r{r}_f{f}: {sensor} ({importance[sensor]:+.4f}) did not beat "
                f"the noise channels ({worst_noise:+.4f}); ranking={ranking}")
        for sensor in noise:
            assert abs(importance[sensor]) < 0.05, (
                f"r{r}_f{f}: {sensor} is meant to be pure noise but scored "
                f"{importance[sensor]:+.4f}")
        assert ranking[0] in strong, f"r{r}_f{f}: ranking led by {ranking[0]}"


@pytest.mark.slow
def test_confusion_matrix_rows_sum_to_the_class_window_count(
        hard_labelled, tmp_path) -> None:
    """Each confusion-matrix row totals the number of test windows in that class."""
    from sklearn.metrics import confusion_matrix

    from common import CLASS_LABELS

    runner = load_script("03_run_experiments")
    df, y = hard_labelled
    out = runner.run_experiments(df, y, tmp_path / "cm", "recording",
                                 TEST_TREES, quick=True)

    with np.load(out / "preds_r0_f0.npz") as data:
        y_true = data["y_true"]
        for name in [k for k in data.files if k not in ("test_idx", "y_true")]:
            matrix = confusion_matrix(y_true, data[name], labels=CLASS_LABELS)
            expected = np.array([(y_true == c).sum() for c in CLASS_LABELS])
            np.testing.assert_array_equal(matrix.sum(axis=1), expected)
            assert matrix.sum() == len(y_true)
