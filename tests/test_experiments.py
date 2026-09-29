"""Level D, part 2: the experiment runner.

Determinism of feature extraction lives in ``test_pipeline.py``.  This module
covers ``03_run_experiments.py``: the selection procedures, the exhaustive
sweep, resume, quick mode, and the claim that a selected subset's score can be
looked up rather than refitted.

Forests here are deliberately tiny.  These tests check plumbing and
reproducibility, never accuracy, so forest size would only cost time.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from common import (METADATA_FILENAME, SENSORS, FoldScorer, feature_columns,
                    inner_split, make_model, outer_splits, subset_name)
from conftest import load_script

RUN_TREES = 8
N_SUBSETS = 127          # 2**7 - 1: the tachometer is excluded


def _md5(path: Path) -> str:
    """Hex MD5 of a file's bytes."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def _scores(out_dir: Path) -> pd.DataFrame:
    """Concatenate every per-split score file in a results directory."""
    return pd.concat([pd.read_csv(p) for p in sorted(out_dir.glob("scores_*.csv"))])


def _result_files(out_dir: Path) -> list[Path]:
    """Every reproducible output of a run.

    ``run_metadata.json`` is excluded on purpose: it records a wall-clock
    timestamp and the measured elapsed time per split, so it cannot be
    byte-identical between runs and should not be. Everything that carries a
    number the paper reports is in the files returned here.
    """
    return sorted(p for p in out_dir.iterdir() if p.name != METADATA_FILENAME)


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    """The ``03_run_experiments`` module."""
    return load_script("03_run_experiments")


@pytest.fixture(scope="module")
def first_split(hard_labelled) -> tuple:
    """The first outer split of the hard fixture, with its inner split."""
    df, y = hard_labelled
    _, _, train_idx, test_idx = next(iter(outer_splits(df, y, "recording")))
    inner_train, inner_val = inner_split(df, y, train_idx)
    scorer = FoldScorer.for_fold(y[inner_val], "probe inner-val")
    return df, y, train_idx, test_idx, inner_train, inner_val, scorer


@pytest.fixture(scope="module")
def full_run(runner: ModuleType, hard_labelled, tmp_path_factory) -> tuple:
    """One complete synthetic run over all 10 outer splits.

    Module-scoped because 10 splits x 255 subsets is the most expensive fixture
    in the suite; the completeness, resume and lookup tests all read it.
    """
    df, y = hard_labelled
    out = tmp_path_factory.mktemp("full_run")
    runner.run_experiments(df, y, out, "recording", RUN_TREES)
    return out, df, y


# --------------------------------------------------------------- enumeration
def test_subset_enumeration_is_complete(runner: ModuleType) -> None:
    """All 127 non-empty subsets of the 7 sensors, each named exactly once."""
    subsets = runner.all_subsets()
    assert len(subsets) == N_SUBSETS
    names = [subset_name(s) for s in subsets]
    assert len(set(names)) == N_SUBSETS
    assert subset_name(SENSORS) in names


# --------------------------------------------------------- permutation design
def test_permutations_are_drawn_once_and_are_reproducible(
        runner: ModuleType) -> None:
    """The permutation list is seeded, so every split draws the same one."""
    first = runner.draw_permutations(50)
    second = runner.draw_permutations(50)
    assert len(first) == runner.PERM_REPEATS
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    assert not np.array_equal(first[0], first[1]), (
        "the repeats within one draw must differ, or averaging is pointless")


@pytest.mark.slow
def test_every_sensor_gets_identical_permutation_arrays(
        runner: ModuleType, first_split, monkeypatch) -> None:
    """Sensor A and sensor B are shuffled with byte-identical permutations.

    Common random numbers: the ranking compares sensors against each other, so
    holding the randomness fixed removes shuffle noise from that comparison.

    Verified from the data actually handed to the model rather than from an
    internal call count: each permuted matrix is inverted back to the
    permutation that produced it, and those are compared across sensors.
    """
    df, y, _, test_idx, inner_train, inner_val, scorer = first_split

    seen: list[np.ndarray] = []
    real_make_model = runner.make_model

    def spy_make_model(n_trees: int):
        model = real_make_model(n_trees)
        original_fit = model.fit

        def fit(x, y_fit):
            fitted = original_fit(x, y_fit)
            real_predict = fitted.predict

            def predict(matrix):
                seen.append(np.asarray(matrix))
                return real_predict(matrix)

            fitted.predict = predict
            return fitted

        model.fit = fit
        return model

    monkeypatch.setattr(runner, "make_model", spy_make_model)
    runner.grouped_permutation_importance(
        df, y, inner_train, inner_val, test_idx, RUN_TREES, scorer)

    cols = feature_columns(df, SENSORS)
    base = df.iloc[inner_val][cols].reset_index(drop=True)
    n_rep = runner.PERM_REPEATS

    def recover(matrix: np.ndarray, sensor: str) -> np.ndarray:
        """Invert a permuted matrix back to the row permutation applied to it."""
        key = cols.index(feature_columns(df, [sensor])[0])
        lookup = {value: row for row, value in enumerate(base.iloc[:, key].to_numpy())}
        return np.array([lookup[value] for value in matrix[:, key]])

    # seen[0] is the unpermuted base prediction, then n_rep calls per sensor.
    assert len(seen) == 1 + len(SENSORS) * n_rep
    per_sensor = {
        sensor: [recover(seen[1 + i * n_rep + r], sensor) for r in range(n_rep)]
        for i, sensor in enumerate(SENSORS)
    }
    reference = per_sensor[SENSORS[0]]
    for sensor in SENSORS[1:]:
        for r, (a, b) in enumerate(zip(reference, per_sensor[sensor])):
            np.testing.assert_array_equal(
                a, b, err_msg=f"{SENSORS[0]} vs {sensor}, repeat {r}")


# ----------------------------------------------------------------- selection
def test_forward_selection_breaks_ties_by_sensor_order(
        runner: ModuleType, first_split, monkeypatch) -> None:
    """When every candidate ties, sensors are added in SENSORS order."""
    df, y, _, test_idx, inner_train, inner_val, scorer = first_split
    monkeypatch.setattr(type(scorer), "macro_f1", lambda self, a, b: 0.5)

    order, path = runner.forward_selection(
        df, y, inner_train, inner_val, test_idx, RUN_TREES, scorer)
    assert order == SENSORS
    assert [step["k"] for step in path] == list(range(1, len(SENSORS) + 1))
    assert [step["added"] for step in path] == SENSORS


def test_permutation_ranking_breaks_ties_by_sensor_order(
        runner: ModuleType, first_split, monkeypatch) -> None:
    """Equal importances leave the ranking in SENSORS order, not an arbitrary one."""
    df, y, _, test_idx, inner_train, inner_val, scorer = first_split
    monkeypatch.setattr(type(scorer), "macro_f1", lambda self, a, b: 0.5)

    ranking, importance = runner.grouped_permutation_importance(
        df, y, inner_train, inner_val, test_idx, RUN_TREES, scorer)
    assert set(importance) == set(SENSORS)
    assert ranking == SENSORS


def test_selection_functions_refuse_test_indices(runner: ModuleType,
                                                 first_split) -> None:
    """Both selection functions raise if handed any test index.

    This is the Section 8 guard: test-fold indices must never reach a ranking,
    selection or fitting call.
    """
    df, y, _, test_idx, inner_train, inner_val, scorer = first_split
    leaky = np.append(inner_train, test_idx[0])

    for fn in (runner.grouped_permutation_importance, runner.forward_selection):
        with pytest.raises(ValueError, match="This is a leak"):
            fn(df, y, leaky, inner_val, test_idx, RUN_TREES, scorer)
        with pytest.raises(ValueError, match="This is a leak"):
            fn(df, y, inner_train, np.append(inner_val, test_idx[0]),
               test_idx, RUN_TREES, scorer)


# ------------------------------------------------------------- the full sweep
@pytest.mark.slow
def test_full_run_is_complete(full_run) -> None:
    """10 splits x 127 subsets, with every subset appearing once per split."""
    out, _, _ = full_run
    scores = _scores(out)
    n_splits = len(list(out.glob("scores_*.csv")))
    assert n_splits == 10
    assert len(scores) == n_splits * N_SUBSETS
    assert (scores.groupby(["repeat", "fold"])["subset"].nunique() == N_SUBSETS).all()
    assert scores.groupby(["repeat", "fold", "subset"]).size().max() == 1


@pytest.mark.slow
def test_metrics_are_in_range(full_run) -> None:
    """Every reported metric lies in [0, 1] and every k in [1, 8]."""
    out, _, _ = full_run
    scores = _scores(out)
    for column in ("macro_f1", "accuracy"):
        assert scores[column].between(0.0, 1.0).all(), column
    assert scores["k"].between(1, len(SENSORS)).all()


@pytest.mark.slow
def test_predictions_are_int8_with_truth_and_indices(full_run) -> None:
    """Each preds file holds test_idx, y_true and one int8 array per subset."""
    out, _, _ = full_run
    for path in sorted(out.glob("preds_*.npz")):
        with np.load(path) as data:
            assert "test_idx" in data.files and "y_true" in data.files
            assert data["y_true"].dtype == np.int8
            subsets = [k for k in data.files if k not in ("test_idx", "y_true")]
            assert len(subsets) == N_SUBSETS
            assert len(data["test_idx"]) == len(data["y_true"])
            for name in subsets:
                assert data[name].dtype == np.int8, name
                assert len(data[name]) == len(data["y_true"]), name


@pytest.mark.slow
def test_selection_json_has_every_required_key(full_run) -> None:
    """Each split records both rankings, the importances and the forward path."""
    out, _, _ = full_run
    for path in sorted(out.glob("selection_*.json")):
        entry = json.loads(path.read_text())
        for key in ("perm_ranking", "perm_importance", "forward_order",
                    "forward_path"):
            assert key in entry, f"{path.name} missing {key}"
        assert sorted(entry["perm_ranking"]) == sorted(SENSORS)
        assert sorted(entry["forward_order"]) == sorted(SENSORS)
        assert len(entry["forward_path"]) == len(SENSORS)


@pytest.mark.slow
def test_selected_score_equals_an_explicit_refit(full_run) -> None:
    """A looked-up selected score equals refitting that subset from scratch.

    This is what licenses never refitting selected subsets: the model and seed
    are fixed, so the exhaustive entry IS the selected subset's score.
    """
    out, df, y = full_run
    selection = json.loads((out / "selection_r0_f0.json").read_text())
    scores = pd.read_csv(out / "scores_r0_f0.csv").set_index("subset")
    _, _, train_idx, test_idx = next(iter(outer_splits(df, y, "recording")))
    scorer = FoldScorer.for_fold(y[test_idx], "refit check")

    for strategy in ("perm_ranking", "forward_order"):
        for k in (1, 3, len(SENSORS)):
            chosen = selection[strategy][:k]
            name = subset_name(chosen)
            cols = feature_columns(df, chosen)
            model = make_model(RUN_TREES).fit(df.iloc[train_idx][cols], y[train_idx])
            refit = scorer.macro_f1(y[test_idx],
                                    model.predict(df.iloc[test_idx][cols]))
            assert refit == pytest.approx(scores.loc[name, "macro_f1"], abs=1e-12), (
                f"{strategy} k={k} ({name})")


@pytest.mark.slow
def test_every_subset_is_scored_on_the_same_label_set(full_run) -> None:
    """One label set per fold, recorded once and shared by all 255 subsets."""
    out, df, y = full_run
    for path in sorted(out.glob("selection_*.json")):
        entry = json.loads(path.read_text())
        tag = f"r{entry['repeat']}_f{entry['fold']}"
        with np.load(out / f"preds_{tag}.npz") as data:
            present = sorted(int(v) for v in np.unique(data["y_true"]))
        assert entry["label_set"] == present, tag


# --------------------------------------------------------- resume and rerun
@pytest.mark.slow
def test_resume_reproduces_an_uninterrupted_run(runner: ModuleType, full_run,
                                                tmp_path: Path) -> None:
    """Deleting one split's files and rerunning restores identical results."""
    out, df, y = full_run
    resumed = tmp_path / "resumed"
    resumed.mkdir()
    for path in out.iterdir():
        (resumed / path.name).write_bytes(path.read_bytes())
    for name in ("scores_r1_f2.csv", "preds_r1_f2.npz", "selection_r1_f2.json"):
        (resumed / name).unlink()

    runner.run_experiments(df, y, resumed, "recording", RUN_TREES)

    for path in _result_files(out):
        assert _md5(path) == _md5(resumed / path.name), path.name


@pytest.mark.slow
def test_quick_mode_runs_one_split_and_is_idempotent(runner: ModuleType,
                                                     hard_labelled,
                                                     tmp_path: Path) -> None:
    """--quick does the FIRST split only, and rerunning changes nothing.

    The draft applied the resume check before the quick restriction, so a
    second quick run skipped r0_f0 and silently computed r0_f1 instead.
    """
    df, y = hard_labelled
    out = tmp_path / "quick"
    runner.run_experiments(df, y, out, "recording", RUN_TREES, quick=True)
    expected = ["preds_r0_f0.npz", "scores_r0_f0.csv", "selection_r0_f0.json"]
    assert [p.name for p in _result_files(out)] == expected
    before = {p.name: _md5(p) for p in _result_files(out)}

    runner.run_experiments(df, y, out, "recording", RUN_TREES, quick=True)
    assert [p.name for p in _result_files(out)] == expected
    assert {p.name: _md5(p) for p in _result_files(out)} == before

    # The rerun recomputed nothing, and the metadata says so.
    meta = json.loads((out / METADATA_FILENAME).read_text())
    assert meta["n_splits_run_this_session"] == 0
    assert meta["quick"] is True


@pytest.mark.slow
def test_two_independent_runs_are_byte_identical(runner: ModuleType,
                                                 hard_labelled,
                                                 tmp_path: Path) -> None:
    """Two runs into separate folders agree byte for byte."""
    df, y = hard_labelled
    first, second = tmp_path / "a", tmp_path / "b"
    runner.run_experiments(df, y, first, "recording", RUN_TREES, quick=True)
    runner.run_experiments(df, y, second, "recording", RUN_TREES, quick=True)
    for path in _result_files(first):
        assert _md5(path) == _md5(second / path.name), path.name


@pytest.mark.slow
def test_run_metadata_records_the_run(runner: ModuleType, hard_labelled,
                                      tmp_path: Path) -> None:
    """03 writes traceable metadata beside its results."""
    df, y = hard_labelled
    out = tmp_path / "meta"
    runner.run_experiments(df, y, out, "recording", RUN_TREES, quick=True)

    meta = json.loads((out / METADATA_FILENAME).read_text())
    for key in ("script", "utc_timestamp", "git_commit", "git_dirty",
                "python_version", "package_versions", "seed", "command_line",
                "split_mode", "n_trees", "n_subsets", "n_splits_total",
                "n_splits_run_this_session", "seconds_per_split"):
        assert key in meta, key
    assert meta["script"] == "03_run_experiments.py"
    assert meta["seed"] == 42
    assert meta["split_mode"] == "recording"
    assert meta["n_subsets"] == N_SUBSETS
    assert meta["n_splits_run_this_session"] == 1
    assert len(meta["seconds_per_split"]) == 1


def test_metadata_differs_only_where_it_must(runner: ModuleType, hard_labelled,
                                             tmp_path: Path) -> None:
    """Two runs agree on everything except the timing fields.

    This is the counterpart to the byte-identity tests: it confirms that the
    only reason run_metadata.json differs between runs is the wall clock, not
    a drifting configuration.
    """
    df, y = hard_labelled
    first, second = tmp_path / "m1", tmp_path / "m2"
    runner.run_experiments(df, y, first, "recording", RUN_TREES, quick=True)
    runner.run_experiments(df, y, second, "recording", RUN_TREES, quick=True)

    a = json.loads((first / METADATA_FILENAME).read_text())
    b = json.loads((second / METADATA_FILENAME).read_text())
    timing = {"utc_timestamp", "seconds_per_split"}
    assert {k: v for k, v in a.items() if k not in timing} == \
           {k: v for k, v in b.items() if k not in timing}
