"""Level F: result validation, plus the analysis outputs.

The Level F checks live inside ``04_analyze.py`` and must abort the run rather
than emit a half-written set of tables and figures, so the tests here corrupt a
real result set in specific ways and assert both that the error is raised and
that nothing was written.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from common import CLASS_ORDER, SENSORS, subset_name
from conftest import load_script

RUN_TREES = 8
N_SUBSETS = 127          # 2**7 - 1: the tachometer is excluded

# Section 11 forbids describing a low-importance sensor as useless anywhere in
# generated output: permutation importance measures MARGINAL contribution, so a
# near-zero score may mean redundancy rather than absence of information.
FORBIDDEN_WORDS = ["uninformative", "useless", "worthless", "no information",
                   "contributes nothing", "irrelevant"]

EXPECTED_FIGURES = [
    "fig1_f1_vs_k", "fig2_sensor_importance", "fig3_per_class_recall",
    "fig4_detection_errors", "fig5_confusion", "fig6_sensor_correlation",
    "fig6b_mean_abs_correlation",
]
EXPECTED_TABLES = [
    "table_main.csv", "table_main.tex", "hypotheses.txt", "table_per_class.csv",
    "table_detection_errors.csv", "sensor_importance_per_split.csv",
    "sensor_rms_correlation.csv", "sensor_mean_abs_correlation.csv",
    "run_metadata.json",
]


@pytest.fixture(scope="module")
def analyzer() -> ModuleType:
    """The ``04_analyze`` module."""
    return load_script("04_analyze")


@pytest.fixture(scope="module")
def analysed(analyzer: ModuleType, hard_labelled, hard_synthetic_features,
             tmp_path_factory) -> tuple:
    """A complete run plus its analysis, in a temporary directory tree."""
    import common

    runner = load_script("03_run_experiments")
    df, y = hard_labelled
    root = tmp_path_factory.mktemp("analysis")
    results = root / "results" / "recording"
    runner.run_experiments(df, y, results, "recording", RUN_TREES)

    features = root / "features" / "features.csv"
    features.parent.mkdir(parents=True, exist_ok=True)
    hard_synthetic_features.to_csv(features, index=False)

    original = (common.FIGURES_DIR, common.TABLES_DIR, common.FEATURES_FILE)
    common.FIGURES_DIR = root / "figures"
    common.TABLES_DIR = root / "tables"
    common.FEATURES_FILE = features
    try:
        summary = analyzer.analyse(results, "recording")
        yield root, results, summary, df, y
    finally:
        common.FIGURES_DIR, common.TABLES_DIR, common.FEATURES_FILE = original


# ---------------------------------------------------------------- Level F
def test_level_f_rejects_a_short_run(analyzer: ModuleType, analysed) -> None:
    """A missing subset row aborts the analysis."""
    _, results, _, _, _ = analysed
    scores, selection, _ = analyzer.load_results(results)
    with pytest.raises(ValueError, match="score rows"):
        analyzer.check_results(scores.iloc[:-1], selection, expect_full_run=False)


def test_level_f_rejects_duplicate_rows(analyzer: ModuleType, analysed) -> None:
    """A repeated (split, subset) row aborts the analysis."""
    _, results, _, _, _ = analysed
    scores, selection, _ = analyzer.load_results(results)
    doubled = pd.concat([scores.iloc[:-1], scores.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="subsets|duplicated"):
        analyzer.check_results(doubled, selection, expect_full_run=False)


def test_level_f_rejects_out_of_range_metrics(analyzer: ModuleType,
                                              analysed) -> None:
    """A macro-F1 above 1 aborts the analysis."""
    _, results, _, _, _ = analysed
    scores, selection, _ = analyzer.load_results(results)
    broken = scores.copy()
    broken.loc[broken.index[0], "macro_f1"] = 1.5
    with pytest.raises(ValueError, match=r"macro_f1 outside \[0, 1\]"):
        analyzer.check_results(broken, selection, expect_full_run=False)


def test_level_f_requires_a_full_run_when_asked(analyzer: ModuleType,
                                                analysed) -> None:
    """--expect-full-run rejects anything short of 10 splits."""
    _, results, _, _, _ = analysed
    scores, selection, _ = analyzer.load_results(results)
    trimmed = {k: v for k, v in list(selection.items())[:4]}
    subset = scores[scores.apply(
        lambda r: (r["repeat"], r["fold"]) in trimmed, axis=1)]
    with pytest.raises(ValueError, match="A full run has"):
        analyzer.check_results(subset, trimmed, expect_full_run=True)


def test_level_f_catches_a_full_configuration_mismatch(analyzer: ModuleType,
                                                       analysed) -> None:
    """At k = 8 every column describes the same model and must agree."""
    _, results, _, _, _ = analysed
    scores, selection, _ = analyzer.load_results(results)
    table, wide, _ = analyzer.build_main_table(scores, selection)
    full_mean = float(wide[subset_name(SENSORS)].mean())
    analyzer.check_full_configuration(table, full_mean)          # passes
    with pytest.raises(ValueError, match="must be identical"):
        analyzer.check_full_configuration(table, full_mean + 0.01)


def test_level_f_catches_bad_p_values(analyzer: ModuleType, analysed) -> None:
    """A p-value outside [0, 1] aborts the analysis."""
    _, results, _, _, _ = analysed
    scores, selection, _ = analyzer.load_results(results)
    table, _, _ = analyzer.build_main_table(scores, selection)
    analyzer.check_p_values(table)                               # passes
    broken = table.copy()
    broken.loc[broken.index[0], "perm_p"] = 1.4
    with pytest.raises(ValueError, match=r"perm_p outside \[0, 1\]"):
        analyzer.check_p_values(broken)


def test_level_f_aborts_before_writing_anything(analyzer: ModuleType, analysed,
                                                tmp_path: Path) -> None:
    """A failed check leaves no partial tables or figures behind."""
    import common

    root, results, _, _, _ = analysed
    broken_results = tmp_path / "broken"
    broken_results.mkdir()
    for path in results.iterdir():
        (broken_results / path.name).write_bytes(path.read_bytes())
    (broken_results / "scores_r0_f0.csv").unlink()
    (broken_results / "selection_r0_f0.json").unlink()
    (broken_results / "preds_r0_f0.npz").unlink()
    # Now 9 splits of selection but a mismatched row count is not the issue;
    # force the mismatch directly by truncating one remaining score file.
    trimmed = pd.read_csv(broken_results / "scores_r0_f1.csv").iloc[:-3]
    trimmed.to_csv(broken_results / "scores_r0_f1.csv", index=False)

    out_figures = tmp_path / "figures_should_not_appear"
    out_tables = tmp_path / "tables_should_not_appear"
    original = (common.FIGURES_DIR, common.TABLES_DIR)
    common.FIGURES_DIR, common.TABLES_DIR = out_figures, out_tables
    try:
        with pytest.raises(ValueError):
            analyzer.analyse(broken_results, "recording")
    finally:
        common.FIGURES_DIR, common.TABLES_DIR = original
    assert not out_figures.exists(), "figures were written despite a failed check"
    assert not out_tables.exists(), "tables were written despite a failed check"


# ------------------------------------------------------------------ Wilcoxon
def test_wilcoxon_is_one_sided_and_handles_zero_differences(
        analyzer: ModuleType) -> None:
    """All-zero differences report as missing, not as a crash or p = 1."""
    assert analyzer.safe_wilcoxon([0.5] * 10, [0.5] * 10) is None

    higher = analyzer.safe_wilcoxon([0.6] * 10, [0.5] * 10)
    lower = analyzer.safe_wilcoxon([0.4] * 10, [0.5] * 10)
    assert higher is not None and lower is not None
    assert 0.0 <= higher <= 1.0 and 0.0 <= lower <= 1.0
    assert higher < 0.05, "a consistently higher selected subset must be significant"
    assert lower > 0.5, "the test must be one-sided: 'greater', not two-sided"


# ------------------------------------------------------------------- outputs
def test_all_expected_outputs_exist(analysed) -> None:
    """Six figures as PDF and PNG, and all seven tables."""
    root, _, _, _, _ = analysed
    for stem in EXPECTED_FIGURES:
        for suffix in (".pdf", ".png"):
            path = root / "figures" / f"{stem}{suffix}"
            assert path.exists() and path.stat().st_size > 0, path.name
    for name in EXPECTED_TABLES:
        path = root / "tables" / name
        assert path.exists() and path.stat().st_size > 0, name


def test_main_table_has_one_row_per_k(analysed) -> None:
    """Seven rows, and the per-k subset counts are the binomial coefficients."""
    root, _, _, _, _ = analysed
    table = pd.read_csv(root / "tables" / "table_main.csv")
    assert list(table["k"]) == list(range(1, len(SENSORS) + 1))
    assert list(table["n_subsets"]) == [7, 21, 35, 35, 21, 7, 1]
    assert table["n_subsets"].sum() == N_SUBSETS


def test_statistics_are_dashed_where_they_do_not_apply(analysed) -> None:
    """Percentile rank and the p-value read ``--`` at k = 8, where one subset exists."""
    root, _, _, _, _ = analysed
    table = pd.read_csv(root / "tables" / "table_main.csv", dtype=str)
    last = table[table["k"] == str(len(SENSORS))].iloc[0]
    for column in ("Perm. pct. rank", "Perm. p (vs median)",
                   "Fwd. pct. rank", "Fwd. p (vs median)"):
        assert last[column] == "--", column


def test_oracle_is_labelled_an_upper_bound_everywhere(analysed) -> None:
    """The oracle must be impossible to misread as an achievable result."""
    root, _, _, _, _ = analysed
    table = pd.read_csv(root / "tables" / "table_main.csv")
    assert any("upper bound" in c.lower() and "test-selected" in c.lower()
               for c in table.columns), table.columns.tolist()

    tex = (root / "tables" / "table_main.tex").read_text().lower()
    assert "upper bound" in tex and "test scores" in tex

    hypotheses = (root / "tables" / "hypotheses.txt").read_text().lower()
    assert "upper bound" in hypotheses
    assert "chosen using test scores" in hypotheses


def test_hypotheses_file_states_its_thresholds_inline(analysed) -> None:
    """The verdicts are readable without the specification at hand."""
    root, _, _, _, _ = analysed
    text = (root / "tables" / "hypotheses.txt").read_text()
    assert "TOLERANCE = 0.01" in text
    assert "alpha     = 0.05" in text
    assert "n         = 10" in text
    for name in ("H1:", "H2:", "H3:"):
        assert name in text, name
    assert text.count("VERDICT") >= 4     # H1, H2, and H3 for both strategies
    assert "not fully" in text and "independent" in text


def test_generated_text_never_calls_a_sensor_uninformative(analysed) -> None:
    """Section 11 wording rule, enforced across every generated text output."""
    root, _, _, _, _ = analysed
    for path in list((root / "tables").iterdir()):
        text = path.read_text(errors="ignore").lower()
        for word in FORBIDDEN_WORDS:
            assert word not in text, f"{path.name} contains {word!r}"
    for stem in EXPECTED_FIGURES:
        for word in FORBIDDEN_WORDS:
            assert word not in stem.lower()


def test_per_class_table_is_aligned_and_nan_safe(analysed) -> None:
    """One column per class at every k, with NaN excluded from the mean."""
    root, _, _, _, _ = analysed
    table = pd.read_csv(root / "tables" / "table_per_class.csv")
    assert list(table["k"]) == list(range(1, len(SENSORS) + 1))
    for class_name in CLASS_ORDER:
        assert class_name in table.columns, class_name
    values = table[CLASS_ORDER].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    assert ((finite >= 0.0) & (finite <= 1.0)).all()
    expected = np.nanmean(values, axis=1)
    np.testing.assert_allclose(table["mean_recall"].to_numpy(), expected,
                               atol=1e-4)


def test_detection_error_rates_are_in_range(analysed) -> None:
    """All three error rates lie in [0, 1] at every k."""
    root, _, _, _, _ = analysed
    table = pd.read_csv(root / "tables" / "table_detection_errors.csv")
    for column in ("missed_detection", "false_alarm", "misdiagnosis"):
        values = table[column].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        assert ((finite >= 0.0) & (finite <= 1.0)).all(), column


def test_importance_table_has_one_row_per_split(analysed) -> None:
    """Ten splits, one importance value per sensor in each."""
    root, _, _, _, _ = analysed
    table = pd.read_csv(root / "tables" / "sensor_importance_per_split.csv")
    assert len(table) == 10
    assert set(SENSORS).issubset(table.columns)
    assert {"repeat", "fold"}.issubset(table.columns)


def test_correlation_matrix_is_square_and_symmetric(analysed) -> None:
    """The RMS correlation matrix covers all 7 studied sensors."""
    root, _, _, _, _ = analysed
    corr = pd.read_csv(root / "tables" / "sensor_rms_correlation.csv", index_col=0)
    assert list(corr.columns) == SENSORS
    assert list(corr.index) == SENSORS
    np.testing.assert_allclose(np.diag(corr.to_numpy()), 1.0, atol=1e-9)
    np.testing.assert_allclose(corr.to_numpy(), corr.to_numpy().T, atol=1e-9)


def test_confusion_figure_uses_repeat_zero_only(analyzer: ModuleType,
                                                analysed) -> None:
    """Every window is counted exactly once, so the matrix totals one repeat.

    Checked by recomputing the matrix the figure draws and comparing its total
    with the number of windows in repeat 0 alone.
    """
    from sklearn.metrics import confusion_matrix

    from common import CLASS_LABELS

    _, results, summary, df, _ = analysed
    _, selection, preds = analyzer.load_results(results)
    repeat_zero = [s for s in sorted(selection) if s[0] == 0]
    assert len(repeat_zero) == 5

    y_true = np.concatenate([preds[s]["y_true"] for s in repeat_zero])
    assert len(y_true) == len(df), (
        "repeat 0 should cover the dataset exactly once")

    y_pred = np.concatenate([preds[s][subset_name(SENSORS)] for s in repeat_zero])
    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    assert matrix.sum() == len(df)
    expected = np.array([(y_true == c).sum() for c in CLASS_LABELS])
    np.testing.assert_array_equal(matrix.sum(axis=1), expected)


def test_selected_scores_match_the_exhaustive_results(analyzer: ModuleType,
                                                      analysed) -> None:
    """The table's selected means are lookups, not refits."""
    _, results, _, _, _ = analysed
    scores, selection, _ = analyzer.load_results(results)
    table, wide, _ = analyzer.build_main_table(scores, selection)
    splits = sorted(selection)

    for _, row in table.iterrows():
        k = int(row["k"])
        for strategy in ("perm", "fwd"):
            picked = [wide.at[s, analyzer.selected_subset(selection[s], strategy, k)]
                      for s in splits]
            assert row[f"{strategy}_mean"] == pytest.approx(float(np.mean(picked)))


def test_analysis_is_reproducible(analyzer: ModuleType, analysed,
                                  tmp_path: Path) -> None:
    """Rerunning the analysis on the same results reproduces the tables."""
    import common

    root, results, _, _, _ = analysed
    first = (root / "tables" / "table_main.csv").read_bytes()

    original = (common.FIGURES_DIR, common.TABLES_DIR)
    common.FIGURES_DIR = tmp_path / "figures2"
    common.TABLES_DIR = tmp_path / "tables2"
    try:
        analyzer.analyse(results, "recording")
        second = (common.TABLES_DIR / "table_main.csv").read_bytes()
    finally:
        common.FIGURES_DIR, common.TABLES_DIR = original
    assert first == second


# --------------------------------------------------------------- Phase 5 additions
def test_main_table_reports_the_candidate_count(analysed) -> None:
    """The C(7,k) column makes the oracle's maximisation bias visible."""
    root, _, _, _, _ = analysed
    table = pd.read_csv(root / "tables" / "table_main.csv")
    assert "Candidates C(7,k)" in table.columns
    assert list(table["Candidates C(7,k)"]) == [7, 21, 35, 35, 21, 7, 1]

    tex = (root / "tables" / "table_main.tex").read_text()
    assert r"\binom{7}{k}" in tex


def test_h1_primary_verdict_uses_the_fair_curves(analysed) -> None:
    """H1 is judged on the training-selected curves, with the oracle secondary."""
    root, _, _, _, _ = analysed
    text = (root / "tables" / "hypotheses.txt").read_text()
    assert "PRIMARY VERDICT is judged on the FAIR curves only" in text
    assert "PRIMARY VERDICT (fair curves):" in text
    assert "SECONDARY, for reference only: the oracle curve" in text
    assert "Permutation-selected (FAIR) by k:" in text
    assert "Forward-selected (FAIR) by k:" in text

    primary = text.index("PRIMARY VERDICT (fair curves):")
    secondary = text.index("SECONDARY, for reference only")
    assert primary < secondary, "the oracle must be reported after the primary"


def test_h1_explains_the_maximisation_bias(analysed) -> None:
    """The oracle caveat names C(7,k) and where the bias peaks."""
    root, _, _, _, _ = analysed
    text = (root / "tables" / "hypotheses.txt").read_text()
    assert "MAXIMUM over C(7,k)" in text
    assert "C(7,3)=35" in text or "C(7,4)=35" in text
    assert "artefact" in text


def test_h1_helper_finds_only_real_violations(analyzer) -> None:
    """h1_violations flags a rise above TOLERANCE, in the k+1 -> k direction."""
    from common import TOLERANCE

    ks = [1, 2, 3]
    rising = np.array([0.50, 0.60, 0.70])          # monotone: no violation
    assert analyzer.h1_violations(ks, rising) == []

    # Dropping from k=3 to k=2 raises the score well past the tolerance.
    dipping = np.array([0.50, 0.70, 0.60])
    found = analyzer.h1_violations(ks, dipping)
    assert len(found) == 1
    from_k, to_k, rise = found[0]
    assert (from_k, to_k) == (3, 2)
    assert rise == pytest.approx(0.10)

    just_under = np.array([0.50, 0.60 + TOLERANCE * 0.5, 0.60])
    assert analyzer.h1_violations(ks, just_under) == []


def test_correlation_heatmaps_mask_the_diagonal_and_scale_to_data(
        analyzer, analysed) -> None:
    """The colour limit comes from the off-diagonal entries, not from the 1.0s."""
    root, _, _, _, _ = analysed
    corr = pd.read_csv(root / "tables" / "sensor_rms_correlation.csv", index_col=0)
    values = corr.to_numpy(dtype=float).copy()
    np.fill_diagonal(values, np.nan)
    largest = float(np.nanmax(np.abs(values)))
    assert largest < 1.0, "the fixture should have no perfectly correlated pair"

    import common
    original = common.FIGURES_DIR
    common.FIGURES_DIR = root / "figures"
    try:
        limit = analyzer.correlation_heatmap(corr, "probe_heatmap", "probe")
    finally:
        common.FIGURES_DIR = original
    assert limit >= largest, "the scale must contain the data"
    assert limit < 1.0, "the scale must not be dominated by the unit diagonal"
    assert limit == pytest.approx(round(limit / 0.05) * 0.05, abs=1e-9)


def test_mean_absolute_correlation_matrix_is_well_formed(
        analyzer, hard_synthetic_features) -> None:
    """fig6b's matrix averages |rho| over all 18 features and stays in [0, 1]."""
    extract = load_script("01_extract_features")
    matrix = analyzer.mean_absolute_feature_correlation(
        hard_synthetic_features, extract.FEATURE_NAMES)
    assert list(matrix.columns) == SENSORS
    assert list(matrix.index) == SENSORS
    values = matrix.to_numpy(dtype=float)
    np.testing.assert_allclose(np.diag(values), 1.0, atol=1e-9)
    np.testing.assert_allclose(values, values.T, atol=1e-9)
    assert ((values >= 0.0) & (values <= 1.0)).all(), "an absolute mean left [0, 1]"


def test_captions_state_the_aggregation(analysed) -> None:
    """Every figure caption says what was averaged over what."""
    root, _, _, _, _ = analysed
    captions = (root / "figures" / "captions.tex").read_text()
    for name in ("FigOneCaption", "FigTwoCaption", "FigThreeCaption",
                 "FigFourCaption", "FigFiveCaption", "FigSixCaption",
                 "FigSixBCaption"):
        assert name in captions, name
    assert captions.count("not") >= 1
    assert "not mutually independent" in captions
    assert "minimum to maximum" in captions          # fig1 whiskers
    assert "Repeat 0 only" in captions               # fig5
    assert "masked" in captions                      # fig6 / fig6b
    assert "upper bound" in captions                 # fig1 oracle
    assert "MARGINAL" in captions                    # fig2 wording rule
    for word in FORBIDDEN_WORDS:
        assert word not in captions.lower(), word


def test_run_metadata_is_written_and_traceable(analysed) -> None:
    """The analysis records its own provenance and that of its inputs."""
    root, _, _, _, _ = analysed
    meta = json.loads((root / "tables" / "run_metadata.json").read_text())
    for key in ("script", "utc_timestamp", "git_commit", "git_dirty",
                "python_version", "package_versions", "seed", "command_line",
                "split_mode", "n_splits", "n_score_rows", "inputs"):
        assert key in meta, key
    assert meta["script"] == "04_analyze.py"
    assert meta["seed"] == 42
    assert meta["n_splits"] == 10
    assert meta["n_score_rows"] == 10 * N_SUBSETS
    assert set(meta["package_versions"]) >= {"numpy", "pandas", "scikit-learn"}
    assert set(meta["inputs"]) == {"features", "results"}


def test_provenance_warns_on_mismatched_commits(analyzer, analysed, tmp_path,
                                                caplog) -> None:
    """Inputs built from different commits produce a loud warning."""
    import common

    root, results, _, _, _ = analysed
    staged_results = tmp_path / "results"
    staged_results.mkdir()
    staged_features = tmp_path / "features"
    staged_features.mkdir()
    (staged_results / "run_metadata.json").write_text(json.dumps(
        {"git_commit": "a" * 40, "git_dirty": False}))
    (staged_features / "run_metadata.json").write_text(json.dumps(
        {"git_commit": "b" * 40, "git_dirty": False}))

    original = common.FEATURES_FILE
    common.FEATURES_FILE = staged_features / "features.csv"
    try:
        with caplog.at_level("WARNING"):
            inputs = analyzer.check_provenance(staged_results)
    finally:
        common.FEATURES_FILE = original

    assert set(inputs) == {"features", "results"}
    assert any("DIFFERENT COMMITS" in r.message for r in caplog.records)


def test_provenance_warns_on_a_dirty_tree(analyzer, tmp_path, caplog) -> None:
    """A dirty working tree is flagged: the commit does not describe the code."""
    import common

    results = tmp_path / "results"
    results.mkdir()
    features = tmp_path / "features"
    features.mkdir()
    for directory in (results, features):
        (directory / "run_metadata.json").write_text(json.dumps(
            {"git_commit": "c" * 40, "git_dirty": True}))

    original = common.FEATURES_FILE
    common.FEATURES_FILE = features / "features.csv"
    try:
        with caplog.at_level("WARNING"):
            analyzer.check_provenance(results)
    finally:
        common.FEATURES_FILE = original
    assert any("DIRTY" in r.message for r in caplog.records)
