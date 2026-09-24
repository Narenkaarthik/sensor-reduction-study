"""Level C: leakage tests.

The central validity claim of the study is that no test data influences any
choice.  These tests check that claim directly on the split helpers, in both
split modes, rather than trusting the comments.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from common import (CLASS_ORDER, N_REPEATS, N_SPLITS, SPEED_QUANTILE_CANDIDATES,
                    assert_no_test_indices, choose_speed_q, inner_split,
                    outer_splits, speed_bands)

SPLIT_MODES = ["recording", "speed"]


def _splits(df: pd.DataFrame, y: np.ndarray, mode: str) -> list:
    """Materialise every outer split for one mode."""
    return list(outer_splits(df, y, mode))


@pytest.fixture
def for_mode(request, labelled, speed_labelled):
    """Pick the fixture a split mode needs.

    Speed mode holds out whole speed bands, which the small fixture cannot
    populate with every class, so it uses the larger one.
    """
    return speed_labelled if request.param == "speed" else labelled


@pytest.mark.parametrize("for_mode, mode", [(m, m) for m in SPLIT_MODES], indirect=["for_mode"])
def test_split_count(for_mode, mode: str) -> None:
    """Both modes yield N_REPEATS x N_SPLITS outer splits."""
    df, y = for_mode
    assert len(_splits(df, y, mode)) == N_REPEATS * N_SPLITS


@pytest.mark.parametrize("for_mode, mode", [(m, m) for m in SPLIT_MODES], indirect=["for_mode"])
def test_train_and_test_share_no_recording(for_mode, mode: str) -> None:
    """No recording contributes windows to both sides of an outer split.

    This holds in speed mode too: a recording has one speed, so it lands
    wholly inside one speed band.
    """
    df, y = for_mode
    recordings = df["recording_id"].to_numpy()
    for r, f, train_idx, test_idx in _splits(df, y, mode):
        assert not np.intersect1d(train_idx, test_idx).size, f"r{r}_f{f}"
        shared = np.intersect1d(recordings[train_idx], recordings[test_idx])
        assert not shared.size, f"r{r}_f{f} shares recordings {shared[:5]}"


def test_speed_mode_holds_out_whole_speed_bands(speed_labelled) -> None:
    """In speed mode, no speed band appears in both train and test."""
    df, y = speed_labelled
    _, bands = choose_speed_q(df, y)
    for r, f, train_idx, test_idx in _splits(df, y, "speed"):
        shared = np.intersect1d(bands[train_idx], bands[test_idx])
        assert not shared.size, f"r{r}_f{f} shares speed bands {shared[:5]}"


@pytest.mark.parametrize("for_mode, mode", [(m, m) for m in SPLIT_MODES], indirect=["for_mode"])
def test_inner_split_never_touches_test(for_mode, mode: str) -> None:
    """Inner-train and inner-validation stay inside the outer training fold."""
    df, y = for_mode
    for r, f, train_idx, test_idx in _splits(df, y, mode):
        inner_train, inner_val = inner_split(df, y, train_idx)
        assert np.isin(inner_train, train_idx).all(), f"r{r}_f{f}"
        assert np.isin(inner_val, train_idx).all(), f"r{r}_f{f}"
        assert not np.intersect1d(inner_train, test_idx).size, f"r{r}_f{f}"
        assert not np.intersect1d(inner_val, test_idx).size, f"r{r}_f{f}"


@pytest.mark.parametrize("for_mode, mode", [(m, m) for m in SPLIT_MODES], indirect=["for_mode"])
def test_inner_halves_share_no_recording(for_mode, mode: str) -> None:
    """Inner-train and inner-validation are grouped by recording as well."""
    df, y = for_mode
    recordings = df["recording_id"].to_numpy()
    for r, f, train_idx, _ in _splits(df, y, mode):
        inner_train, inner_val = inner_split(df, y, train_idx)
        assert not np.intersect1d(inner_train, inner_val).size, f"r{r}_f{f}"
        shared = np.intersect1d(recordings[inner_train], recordings[inner_val])
        assert not shared.size, f"r{r}_f{f} shares recordings {shared[:5]}"


@pytest.mark.parametrize("for_mode, mode", [(m, m) for m in SPLIT_MODES], indirect=["for_mode"])
def test_every_window_is_tested_once_per_repeat(for_mode, mode: str) -> None:
    """The folds of one repeat partition the dataset exactly."""
    df, y = for_mode
    for repeat in range(N_REPEATS):
        tested = np.concatenate(
            [te for r, _, _, te in _splits(df, y, mode) if r == repeat])
        assert len(tested) == len(df)
        assert len(np.unique(tested)) == len(df)


@pytest.mark.parametrize("for_mode, mode", [(m, m) for m in SPLIT_MODES], indirect=["for_mode"])
def test_every_test_fold_contains_all_classes(for_mode, mode: str) -> None:
    """Every test fold contains all 6 classes, in BOTH split modes.

    A fold missing a class is a split defect, not a result to be scored.
    Recording mode gets this from stratification.  Speed mode gets it from
    choose_speed_q, which coarsens the banding until coverage is complete and
    refuses to split at all if no banding works.
    """
    df, y = for_mode
    missing = []
    for r, f, _, test_idx in _splits(df, y, mode):
        present = set(np.unique(y[test_idx]).tolist())
        if present != set(range(len(CLASS_ORDER))):
            absent = [CLASS_ORDER[i] for i in
                      sorted(set(range(len(CLASS_ORDER))) - present)]
            missing.append((r, f, absent))
    assert not missing, f"{mode} mode left folds without every class: {missing}"


@pytest.mark.parametrize("for_mode, mode", [(m, m) for m in SPLIT_MODES], indirect=["for_mode"])
def test_selection_data_is_disjoint_from_test(for_mode, mode: str) -> None:
    """The guard passes for every legitimate inner split."""
    df, y = for_mode
    for r, f, train_idx, test_idx in _splits(df, y, mode):
        inner_train, inner_val = inner_split(df, y, train_idx)
        assert_no_test_indices(inner_train, test_idx, f"r{r}_f{f} inner-train")
        assert_no_test_indices(inner_val, test_idx, f"r{r}_f{f} inner-val")


def test_guard_raises_on_a_single_leaked_index(labelled) -> None:
    """One test index among thousands of training indices is still caught."""
    df, y = labelled
    _, _, train_idx, test_idx = next(iter(_splits(df, y, "recording")))
    leaky = np.append(train_idx, test_idx[0])
    with pytest.raises(ValueError, match="This is a leak"):
        assert_no_test_indices(leaky, test_idx, "deliberate leak")


def test_guard_accepts_clean_indices(labelled) -> None:
    """The guard does not fire on disjoint index sets."""
    df, y = labelled
    _, _, train_idx, test_idx = next(iter(_splits(df, y, "recording")))
    assert_no_test_indices(train_idx, test_idx, "clean") is None


def test_outer_splits_rejects_unknown_mode(labelled) -> None:
    """An unrecognised split mode is an error, not a silent default."""
    df, y = labelled
    with pytest.raises(ValueError, match="Unknown split_mode"):
        list(outer_splits(df, y, "whatever"))


# ---------------------------------------------------------------- speed bands
def test_choose_speed_q_prefers_the_finest_banding(speed_labelled) -> None:
    """With enough data, the first candidate q is accepted unchanged."""
    df, y = speed_labelled
    q, groups = choose_speed_q(df, y)
    assert q == SPEED_QUANTILE_CANDIDATES[0]
    assert len(np.unique(groups)) == q


def test_choose_speed_q_coarsens_when_coverage_fails() -> None:
    """A dataset where q=10 fails but a coarser banding succeeds picks the coarser.

    Built directly rather than extracted, so the ladder is exercised without
    paying for a third synthetic dataset.
    """
    rng = np.random.default_rng(0)
    labels, speeds = [], []
    for class_code in range(len(CLASS_ORDER)):
        for _ in range(20):
            labels.append(class_code)
            speeds.append(5 * round(float(rng.uniform(12, 60)) / 5))
    df = pd.DataFrame({
        "recording_id": np.repeat(np.arange(len(labels)), 5),
        "speed_hz": np.repeat(speeds, 5),
    })
    y = np.repeat(np.array(labels), 5)

    q, _ = choose_speed_q(df, y)
    assert q in SPEED_QUANTILE_CANDIDATES
    for _, _, _, test_idx in outer_splits(df, y, "speed"):
        assert len(np.unique(y[test_idx])) == len(CLASS_ORDER)


def test_choose_speed_q_refuses_when_no_banding_works(labelled) -> None:
    """The small fixture genuinely cannot support speed mode, and says so.

    This is the documented failure path: scoring a degenerate split would be
    worse than stopping.
    """
    df, y = labelled
    with pytest.raises(ValueError, match="no quantile count"):
        choose_speed_q(df, y)


def test_speed_mode_is_refused_on_the_small_fixture(labelled) -> None:
    """outer_splits propagates that refusal rather than splitting anyway."""
    df, y = labelled
    with pytest.raises(ValueError, match="no quantile count"):
        list(outer_splits(df, y, "speed"))
