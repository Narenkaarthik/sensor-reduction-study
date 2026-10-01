"""Step 4: turn raw results into the paper's tables, figures and hypothesis tests.

Everything the paper reports is produced here, from the files in ``results/``,
with one command.  No number is ever entered by hand.

A note on wording that matters for interpretation: grouped permutation
importance measures a sensor's MARGINAL contribution.  A sensor can sit near
zero either because it carries no information or because it is redundant given
the sensors around it, and permutation importance alone cannot tell those apart.
Nothing generated here calls a low-scoring sensor uninformative; that is why the
study also reports forward selection and the sensor correlation matrix.

Outputs
  tables/table_main.csv / .tex        macro-F1 against sensor count (RQ1, RQ2)
  tables/hypotheses.txt               H1, H2, H3 verdicts with their thresholds
  tables/table_per_class.csv          per-class recall against k
  tables/table_detection_errors.csv   missed detection, false alarm, misdiagnosis
  tables/sensor_importance_per_split.csv
  tables/sensor_rms_correlation.csv
  figures/fig1..fig6                  PDF and 300 dpi PNG

Usage:
    python 04_analyze.py                  # results/recording
    python 04_analyze.py --split speed    # results/speed
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

import common  # noqa: E402
from common import (CLASS_LABELS, CLASS_ORDER, HEALTHY_CLASS, N_REPEATS,  # noqa: E402
                    N_SPLITS, SENSORS, TOLERANCE, feature_columns,
                    load_features, per_class_recall, read_run_metadata,
                    subset_name, write_run_metadata)

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------- figure style
plt.rcParams.update({
    "font.size": 8,
    "font.family": "serif",
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "savefig.bbox": "tight",
})
COL_W = 3.5           # IEEE single column, inches
DOUBLE_W = 7.16       # IEEE double column, inches
DPI = 300

HEALTHY = CLASS_ORDER.index(HEALTHY_CLASS)
ABBR = {"normal": "N", "imbalance": "IMB", "horizontal-misalignment": "HM",
        "vertical-misalignment": "VM", "underhang": "UB", "overhang": "OB"}

ALPHA = 0.05          # H3 significance level
N_SUBSETS = 2 ** len(SENSORS) - 1

# Every mention of the oracle carries this label.  It is selected using test
# scores, so it is an upper bound on what any honest selection could achieve,
# never a result the study claims.
ORACLE_LABEL = "Oracle (upper bound, test-selected)"


# ---------------------------------------------------------------------- loading
def load_results(res_dir: Path) -> tuple[pd.DataFrame, dict, dict]:
    """Load every per-split score file, selection record and prediction file.

    Args:
        res_dir: ``results/<split_mode>/``.

    Returns:
        ``(scores, selection, preds)``, the latter two keyed by
        ``(repeat, fold)``.

    Raises:
        FileNotFoundError: If the directory holds no score files.
    """
    score_files = sorted(res_dir.glob("scores_*.csv"))
    if not score_files:
        raise FileNotFoundError(
            f"No scores_*.csv in {res_dir}. Run 03_run_experiments.py first.")

    scores = pd.concat([pd.read_csv(p) for p in score_files], ignore_index=True)
    selection: dict[tuple[int, int], dict] = {}
    preds: dict[tuple[int, int], dict] = {}
    for path in sorted(res_dir.glob("selection_*.json")):
        tag = path.stem.split("_", 1)[1]
        entry = json.loads(path.read_text())
        key = (int(entry["repeat"]), int(entry["fold"]))
        selection[key] = entry
        with np.load(res_dir / f"preds_{tag}.npz") as data:
            preds[key] = {k: data[k] for k in data.files}
    return scores, selection, preds


def selected_subset(entry: dict, strategy: str, k: int) -> str:
    """Name the subset a selection strategy picks at size k.

    Args:
        entry: One split's selection record.
        strategy: ``"perm"`` or ``"fwd"``.
        k: Subset size.

    Returns:
        The canonical subset name.
    """
    order = entry["perm_ranking"] if strategy == "perm" else entry["forward_order"]
    return subset_name(order[:k])


# ------------------------------------------------------------------ statistics
def safe_wilcoxon(selected: list[float], reference: list[float]) -> float | None:
    """One-sided Wilcoxon signed-rank test: is ``selected`` greater?

    Args:
        selected: The selected subset's score in each split.
        reference: The per-split median of all same-size subsets.

    Returns:
        The p-value, or ``None`` when every paired difference is zero.  That
        case carries no information about direction, so it is reported as
        ``--`` rather than as a p-value of 1.
    """
    diff = np.asarray(selected, dtype=float) - np.asarray(reference, dtype=float)
    if np.allclose(diff, 0.0):
        return None
    return float(wilcoxon(selected, reference, alternative="greater").pvalue)


def error_rates(y_true: NDArray[np.int64],
                y_pred: NDArray[np.int64]) -> dict[str, float]:
    """Detection error rates, as defined in Section 11.

    Args:
        y_true: True class codes.
        y_pred: Predicted class codes.

    Returns:
        ``missed_detection``, ``false_alarm`` and ``misdiagnosis``.
    """
    fault = y_true != HEALTHY
    healthy = ~fault
    return {
        "missed_detection": float(np.mean(y_pred[fault] == HEALTHY))
        if fault.any() else float("nan"),
        "false_alarm": float(np.mean(y_pred[healthy] != HEALTHY))
        if healthy.any() else float("nan"),
        "misdiagnosis": float(np.mean((y_pred[fault] != HEALTHY)
                                      & (y_pred[fault] != y_true[fault])))
        if fault.any() else float("nan"),
    }


# --------------------------------------------------------------------- Level F
def check_results(scores: pd.DataFrame, selection: dict,
                  expect_full_run: bool) -> None:
    """Validate the raw results BEFORE anything is written.

    Runs first so that a failed check aborts with a clear message rather than
    leaving a half-written set of tables and figures behind.

    Args:
        scores: All per-split score rows.
        selection: Selection records, keyed by split.
        expect_full_run: Whether to require the full 10-split row count.

    Raises:
        ValueError: If any check fails.
    """
    n_splits = len(selection)
    expected_rows = n_splits * N_SUBSETS
    if len(scores) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} score rows ({n_splits} splits x "
            f"{N_SUBSETS} subsets) but found {len(scores)}. The run is "
            "incomplete; rerun 03_run_experiments.py.")

    if expect_full_run:
        full_splits = N_REPEATS * N_SPLITS
        if n_splits != full_splits:
            raise ValueError(
                f"A full run has {full_splits} splits ({full_splits * N_SUBSETS} "
                f"rows) but this has {n_splits}.")

    per_split = scores.groupby(["repeat", "fold"])["subset"].nunique()
    if not (per_split == N_SUBSETS).all():
        bad = per_split[per_split != N_SUBSETS].to_dict()
        raise ValueError(f"Some splits do not hold all {N_SUBSETS} subsets: {bad}")

    duplicated = scores.duplicated(subset=["repeat", "fold", "subset"]).sum()
    if duplicated:
        raise ValueError(f"{duplicated} duplicated (split, subset) score rows.")

    for column in ("macro_f1", "accuracy"):
        values = scores[column]
        if not values.between(0.0, 1.0).all():
            out = values[~values.between(0.0, 1.0)]
            raise ValueError(f"{column} outside [0, 1]: {out.head().to_dict()}")

    if not scores["k"].between(1, len(SENSORS)).all():
        raise ValueError("k outside [1, 8].")

    LOG.info("Level F: %d rows over %d splits, all metrics in range",
             len(scores), n_splits)


def check_full_configuration(table: pd.DataFrame, full_mean: float) -> None:
    """The full configuration must agree everywhere it appears.

    At k = 8 there is exactly one subset, so the all-subsets mean, the oracle
    best and both selected columns describe the same model and must be equal.

    Args:
        table: The assembled main table, with numeric helper columns.
        full_mean: The full configuration's mean macro-F1.

    Raises:
        ValueError: If the columns disagree by more than floating-point noise.
    """
    row = table[table["k"] == len(SENSORS)]
    if row.empty:
        raise ValueError("The main table has no k = 8 row.")
    row = row.iloc[0]
    for column in ("all_mean", "oracle_mean", "perm_mean", "fwd_mean"):
        if not np.isclose(row[column], full_mean, atol=1e-12):
            raise ValueError(
                f"At k = 8 the column {column!r} is {row[column]:.6f} but the "
                f"full configuration scores {full_mean:.6f}. These describe the "
                "same model and must be identical.")
    LOG.info("Level F: full configuration agrees across all k = 8 columns "
             "(%.6f)", full_mean)


def check_p_values(table: pd.DataFrame) -> None:
    """Every p-value is in [0, 1] or deliberately missing.

    Args:
        table: The assembled main table.

    Raises:
        ValueError: If a p-value lies outside [0, 1].
    """
    for column in ("perm_p", "fwd_p"):
        values = table[column].dropna()
        if not values.between(0.0, 1.0).all():
            bad = values[~values.between(0.0, 1.0)].to_dict()
            raise ValueError(f"{column} outside [0, 1]: {bad}")
    LOG.info("Level F: all p-values in [0, 1] or reported as missing")


def check_provenance(res_dir: Path) -> dict[str, object]:
    """Check that the inputs were produced by one consistent version of the code.

    A mismatch does not by itself invalidate the numbers, so this warns loudly
    rather than aborting: the person running it is best placed to judge whether
    an intervening commit mattered.

    Args:
        res_dir: The results directory being analysed.

    Returns:
        The metadata of the inputs consumed, recorded in the analysis metadata.
    """
    features_meta = read_run_metadata(common.FEATURES_FILE.parent)
    results_meta = read_run_metadata(res_dir)
    inputs = {"features": features_meta, "results": results_meta}

    missing = [name for name, meta in inputs.items() if meta is None]
    if missing:
        LOG.warning("Level F: no run_metadata.json for %s; those outputs predate "
                    "metadata recording and cannot be traced to a commit.",
                    ", ".join(missing))

    commits = {name: meta.get("git_commit")
               for name, meta in inputs.items() if meta is not None}
    distinct = {c for c in commits.values() if c is not None}
    if len(distinct) > 1:
        LOG.warning(
            "Level F: INPUTS COME FROM DIFFERENT COMMITS (%s). The feature table "
            "and the results were produced by different versions of the code and "
            "may not be mutually consistent. Rerun the pipeline from "
            "01_extract_features.py before trusting these numbers.",
            ", ".join(f"{k}={v[:12]}" for k, v in commits.items() if v))
    elif distinct:
        LOG.info("Level F: inputs share commit %s", next(iter(distinct))[:12])

    dirty = [name for name, meta in inputs.items()
             if meta is not None and meta.get("git_dirty")]
    if dirty:
        LOG.warning("Level F: %s produced with a DIRTY working tree, so the "
                    "commit hash does not fully describe the code that ran.",
                    ", ".join(dirty))
    return inputs



def _bias_peak() -> int:
    """The k at which C(len(SENSORS), k) is largest, where oracle bias peaks."""
    return max(range(1, len(SENSORS) + 1), key=lambda k: comb(len(SENSORS), k))


def holm_adjusted(p_values: dict[int, float]) -> dict[int, float]:
    """Holm-Bonferroni adjustment of a family of p-values.

    POST HOC and SUPPLEMENTARY. The pre-registered H3 verdict is computed on
    the RAW p-values against alpha and is not affected by this.

    Args:
        p_values: ``{k: p}``; entries whose p is None or NaN are skipped.

    Returns:
        ``{k: adjusted_p}`` for the entries that had a p-value.
    """
    usable = {k: float(v) for k, v in p_values.items()
              if v is not None and not pd.isna(v)}
    if not usable:
        return {}
    ordered = sorted(usable.items(), key=lambda kv: kv[1])
    m = len(ordered)
    adjusted: dict[int, float] = {}
    running = 0.0
    for i, (k, raw) in enumerate(ordered):
        value = (m - i) * raw
        running = max(running, value)          # enforce monotonicity
        adjusted[k] = min(1.0, running)
    return adjusted


def h1_violations(ks: list[int], curve: NDArray[np.float64]
                  ) -> list[tuple[int, int, float]]:
    """Find every consecutive step where DROPPING a sensor raises the score.

    Args:
        ks: Sensor counts, ascending.
        curve: Mean macro-F1 at each k, aligned with ``ks``.

    Returns:
        ``(from_k, to_k, rise)`` for each step whose rise exceeds TOLERANCE.
    """
    violations = []
    for i in range(len(ks) - 1):
        rise = float(curve[i] - curve[i + 1])     # going from k+1 down to k
        if rise > TOLERANCE:
            violations.append((ks[i + 1], ks[i], rise))
    return violations


# ----------------------------------------------------------------- main table
def build_main_table(scores: pd.DataFrame, selection: dict
                     ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Assemble the per-k results table.

    Each subset's score is first averaged over the splits, then summarised
    across the subsets of one size, so that every subset counts once.

    Args:
        scores: All per-split score rows.
        selection: Selection records, keyed by split.

    Returns:
        ``(table, wide, curves)``: the table with both display and numeric
        columns, the split-by-subset score matrix, and the per-k curves needed
        by the figures.
    """
    # A (repeat, fold) MultiIndex, not a column of tuples: pandas flattens a
    # list of tuple keys against a plain Index, which silently breaks lookups.
    wide = scores.pivot_table(index=["repeat", "fold"], columns="subset",
                              values="macro_f1")
    splits = sorted(selection)
    wide = wide.loc[splits]

    k_of = scores.drop_duplicates("subset").set_index("subset")["k"]
    subset_mean = wide.mean()
    ks = sorted(k_of.unique())

    rows: list[dict[str, object]] = []
    curves: dict[str, list[float]] = defaultdict(list)

    for k in ks:
        names = list(k_of[k_of == k].index)
        oracle_name = subset_mean[names].idxmax()
        all_mean = float(subset_mean[names].mean())
        oracle_mean = float(subset_mean[oracle_name])

        row: dict[str, object] = {
            "k": k,
            "n_subsets": len(names),
            "n_candidates": comb(len(SENSORS), k),
            "all_mean": all_mean,
            "all_sd": float(subset_mean[names].std(ddof=0)),
            "oracle_mean": oracle_mean,
            "oracle_subset": oracle_name,
        }
        curves["all_mean"].append(all_mean)
        curves["oracle"].append(oracle_mean)

        for strategy in ("perm", "fwd"):
            picked, medians, percentiles = [], [], []
            for split in splits:
                name = selected_subset(selection[split], strategy, k)
                score = float(wide.at[split, name])
                same_size = wide.loc[split, names].to_numpy(dtype=float)
                picked.append(score)
                medians.append(float(np.median(same_size)))
                percentiles.append(100.0 * float(np.mean(same_size < score)))

            row[f"{strategy}_mean"] = float(np.mean(picked))
            row[f"{strategy}_sd"] = float(np.std(picked))
            row[f"{strategy}_pct_rank"] = (float(np.mean(percentiles))
                                           if len(names) > 1 else np.nan)
            row[f"{strategy}_p"] = (safe_wilcoxon(picked, medians)
                                    if len(names) > 1 else None)
            curves[strategy].append(float(np.mean(picked)))
            curves[f"{strategy}_sd"].append(float(np.std(picked)))

        rows.append(row)

    table = pd.DataFrame(rows)
    table["perm_p"] = table["perm_p"].astype(float)
    table["fwd_p"] = table["fwd_p"].astype(float)
    curves["k"] = [float(k) for k in ks]
    return table, wide, dict(curves)


def format_main_table(table: pd.DataFrame) -> pd.DataFrame:
    """Render the numeric table into the display form saved as CSV.

    Args:
        table: The numeric main table.

    Returns:
        A display table with mean +/- sd strings and ``--`` where a statistic
        does not apply.
    """
    def dash(value: float, spec: str = "{:.0f}") -> str:
        return "--" if value is None or pd.isna(value) else spec.format(value)

    out = pd.DataFrame({
        "k": table["k"],
        "n_subsets": table["n_subsets"],
        "Candidates C(7,k)": table["n_candidates"],
        "All subsets (mean +/- sd)": [
            f"{m:.3f} +/- {s:.3f}" for m, s in zip(table["all_mean"], table["all_sd"])],
        f"{ORACLE_LABEL} best": [f"{m:.3f}" for m in table["oracle_mean"]],
        "Oracle subset": table["oracle_subset"],
        "Perm.-selected (mean +/- sd)": [
            f"{m:.3f} +/- {s:.3f}" for m, s in zip(table["perm_mean"], table["perm_sd"])],
        "Perm. pct. rank": [dash(v) for v in table["perm_pct_rank"]],
        "Perm. p (vs median)": [dash(v, "{:.4f}") for v in table["perm_p"]],
        "Forward-selected (mean +/- sd)": [
            f"{m:.3f} +/- {s:.3f}" for m, s in zip(table["fwd_mean"], table["fwd_sd"])],
        "Fwd. pct. rank": [dash(v) for v in table["fwd_pct_rank"]],
        "Fwd. p (vs median)": [dash(v, "{:.4f}") for v in table["fwd_p"]],
    })
    return out


def write_latex_table(table: pd.DataFrame, path: Path) -> None:
    """Write the single-column IEEE version of the main table.

    Args:
        table: The numeric main table.
        path: Destination ``.tex`` file.
    """
    lines = [
        r"% Generated by 04_analyze.py. Do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Macro-F1 versus the number of sensors $k$. Each subset is "
        r"first averaged over the 10 cross-validation splits, then summarised "
        r"across the $\binom{7}{k}$ subsets of that size. The oracle column is "
        r"the maximum over those $\binom{7}{k}$ candidates chosen using TEST "
        r"scores: it is an UPPER BOUND, not an achievable result, and its "
        r"upward bias grows with the number of candidates, peaking at "
        r"$k=3$ and $k=4$. "
        r"The permutation- and forward-selected columns are chosen on training "
        r"data only and are the honest comparison. The 10 splits are not "
        r"mutually independent.}",
        r"\label{tab:main}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{crcccc}",
        r"\toprule",
        r"$k$ & $\binom{7}{k}$ & All subsets & Oracle$^{\dagger}$ & Perm.-sel. "
        r"& Fwd.-sel. \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"{int(row['k'])} & "
            f"{row['all_mean']:.3f} $\\pm$ {row['all_sd']:.3f} & "
            f"{row['oracle_mean']:.3f} & "
            f"{row['perm_mean']:.3f} $\\pm$ {row['perm_sd']:.3f} & "
            f"{row['fwd_mean']:.3f} $\\pm$ {row['fwd_sd']:.3f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2pt}",
        r"\footnotesize $^{\dagger}$Upper bound: selected using test scores.",
        r"\end{table}",
        "",
    ]
    path.write_text("\n".join(lines))


# ----------------------------------------------------------------- hypotheses
def write_hypotheses(table: pd.DataFrame, curves: dict, full_mean: float,
                     full_sd: float, n_splits: int, split_mode: str,
                     path: Path) -> int | None:
    """Write the H1, H2 and H3 verdicts.

    Every threshold is restated inline so the file stands on its own without
    the specification.

    Args:
        table: The numeric main table.
        curves: Per-k curves.
        full_mean: Full-configuration mean macro-F1.
        full_sd: Its standard deviation across splits.
        n_splits: Number of paired scores behind each test.
        split_mode: Which split mode produced these results.
        path: Destination ``.txt`` file.

    Returns:
        The fair H2 k (permutation-selected), or ``None`` if H2 fails.
    """
    ks = [int(k) for k in table["k"]]
    oracle = np.asarray(curves["oracle"], dtype=float)
    perm = np.asarray(curves["perm"], dtype=float)
    out: list[str] = []

    out.append("HYPOTHESIS VERDICTS")
    out.append(f"Split mode: {split_mode}   Paired scores per test: n = {n_splits}")
    out.append(f"Full configuration ({len(SENSORS)} sensors): "
               f"{full_mean:.4f} +/- {full_sd:.4f} macro-F1")
    out.append("")
    out.append("Thresholds applied here, stated inline so this file is "
               "self-contained:")
    out.append(f"  TOLERANCE = {TOLERANCE:<5} (used by H1 and H2)")
    out.append(f"  alpha     = {ALPHA:<5} (used by H3)")
    full_splits = N_SPLITS * N_REPEATS
    provenance = (f"{N_SPLITS} folds x {N_REPEATS} repeats"
                  if n_splits == full_splits
                  else f"PARTIAL RUN: {n_splits} of {full_splits} splits present")
    out.append(f"  n         = {n_splits:<5} (paired split scores per test; "
               f"{provenance})")
    out.append("")

    # ------------------------------------------------------------------- H1
    out.append("-" * 70)
    out.append("H1: removing one sensor never raises mean macro-F1 by more than")
    out.append(f"    TOLERANCE = {TOLERANCE}. Checked for every consecutive pair of k")
    out.append("    values (k+1 -> k).")
    out.append("")
    out.append("    PRIMARY VERDICT is judged on the FAIR curves only. Those subsets")
    out.append("    are chosen on training data, so no test-set maximisation inflates")
    out.append("    them. The oracle curve is reported below as SECONDARY evidence.")
    out.append("")

    primary_violations: dict[str, list[tuple[int, int, float]]] = {}
    for strategy, label in (("perm", "Permutation-selected"),
                            ("fwd", "Forward-selected")):
        curve = np.asarray(curves[strategy], dtype=float)
        violations = h1_violations(ks, curve)
        primary_violations[strategy] = violations
        out.append(f"    {label} (FAIR) by k:")
        out.append("      " + ", ".join(f"k={k}:{v:.4f}" for k, v in zip(ks, curve)))
        out.append(f"      {'SUPPORTED' if not violations else 'NOT SUPPORTED'}"
                   f" ({len(violations)} violation(s))")
        for hi, lo, rise in violations:
            out.append(f"        k={hi} -> k={lo}: rises by {rise:.4f} "
                       f"(exceeds {TOLERANCE} by {rise - TOLERANCE:.4f})")
        out.append("")

    primary_ok = not any(primary_violations.values())
    out.append(f"    PRIMARY VERDICT (fair curves): "
               f"{'SUPPORTED' if primary_ok else 'NOT SUPPORTED'}")
    out.append("")

    oracle_violations = h1_violations(ks, oracle)
    out.append("    SECONDARY, for reference only: the oracle curve.")
    out.append("      " + ", ".join(f"k={k}:{v:.4f}" for k, v in zip(ks, oracle)))
    out.append(f"      {'no' if not oracle_violations else len(oracle_violations)} "
               f"violation(s) on this curve:")
    for hi, lo, rise in oracle_violations:
        out.append(f"        k={hi} -> k={lo}: rises by {rise:.4f} "
                   f"(exceeds {TOLERANCE} by {rise - TOLERANCE:.4f})")
    out.append(f"      CAVEAT: the oracle at each k is a MAXIMUM over "
               f"C({len(SENSORS)},k)")
    out.append("      candidate subsets (" +
               ", ".join(f"k={k}:{comb(len(SENSORS), k)}" for k in ks) + "),")
    out.append("      and a maximum over more candidates carries more upward bias,")
    out.append("      so the oracle curve is inflated most at middle k (peaking at")
    out.append(f"      k={_bias_peak()}, where C({len(SENSORS)},{_bias_peak()})="
               f"{comb(len(SENSORS), _bias_peak())}) purely from")
    out.append("      maximisation and independently of any sensor effect. An")
    out.append("      apparent violation at middle k may therefore be an artefact of")
    out.append("      that bias rather than evidence of a redundant or harmful")
    out.append("      sensor. This is why the primary verdict uses the fair curves.")
    out.append("")

    # ------------------------------------------------------------------- H2
    threshold = full_mean - TOLERANCE
    out.append("-" * 70)
    out.append(f"H2: some k < {len(SENSORS)} reaches mean macro-F1 >= "
               f"(full - TOLERANCE) = {threshold:.4f}.")
    out.append("")
    fair_k = next((k for k, v in zip(ks, perm)
                   if v >= threshold and k < len(SENSORS)), None)
    oracle_k = next((k for k, v in zip(ks, oracle)
                     if v >= threshold and k < len(SENSORS)), None)
    out.append("    Permutation-selected (FAIR: chosen on training data only)")
    out.append(f"      smallest qualifying k = {fair_k if fair_k else 'none'}")
    out.append("    Oracle (UPPER BOUND: chosen using test scores, not "
               "achievable in practice)")
    out.append(f"      smallest qualifying k = {oracle_k if oracle_k else 'none'}")
    out.append("")
    out.append(f"    VERDICT: {'SUPPORTED' if fair_k else 'NOT SUPPORTED'} "
               "(on the fair, permutation-selected result)")
    out.append("    The FAIR number is the answer to the research question. The")
    out.append("    oracle number only shows what would be possible if the best")
    out.append("    subset were known in advance, which it never is.")
    out.append("")

    # ------------------------------------------------------------------- H3
    out.append("-" * 70)
    out.append(f"H3: for each k from 1 to {len(SENSORS) - 1}, the selected subset")
    out.append("    scores higher than the median of all k-sensor subsets.")
    out.append(f"    One-sided Wilcoxon signed-rank test over the {n_splits} paired")
    out.append(f"    split scores, alpha = {ALPHA}.")
    out.append("")
    verdicts = {}
    for strategy, label in (("perm", "Permutation-selected"),
                            ("fwd", "Forward-selected")):
        out.append(f"    {label}:")
        significant, tested = [], []
        for _, row in table.iterrows():
            k = int(row["k"])
            if k >= len(SENSORS):
                continue
            p = row[f"{strategy}_p"]
            rank = row[f"{strategy}_pct_rank"]
            p_text = "--" if pd.isna(p) else f"{p:.4f}"
            rank_text = "--" if pd.isna(rank) else f"{rank:.0f}"
            flag = ""
            if not pd.isna(p):
                tested.append(k)
                if p < ALPHA:
                    significant.append(k)
                    flag = "  significant"
            else:
                flag = "  (all paired differences zero: no evidence either way)"
            out.append(f"      k={k}: p={p_text}, percentile rank={rank_text}{flag}")
        verdict = ("SUPPORTED" if tested and len(significant) == len(tested)
                   else "NOT SUPPORTED")
        verdicts[strategy] = verdict
        out.append(f"      VERDICT: {verdict} "
                   f"({len(significant)}/{len(tested)} tested k values reach "
                   f"p < {ALPHA})")
        if len(tested) < len(SENSORS) - 1:
            skipped = sorted(set(range(1, len(SENSORS))) - set(tested))
            out.append(f"      k values not testable (all differences zero): "
                       f"{skipped}")
        out.append("")

    out.append("-" * 70)
    out.append("SUPPLEMENTARY, POST HOC: HOLM-ADJUSTED p-VALUES FOR H3")
    out.append("  NOT the pre-registered test. The H3 verdicts above stand on the")
    out.append("  RAW p-values against alpha = " + str(ALPHA) + ", regardless of")
    out.append("  anything below. This block adjusts the family of "
               f"{len(SENSORS) - 1} comparisons")
    out.append("  per strategy for multiplicity, and is reported for completeness.")
    out.append("")
    for strategy, label in (("perm", "Permutation-selected"),
                            ("fwd", "Forward-selected")):
        raw = {int(r["k"]): r[f"{strategy}_p"] for _, r in table.iterrows()
               if int(r["k"]) < len(SENSORS)}
        adjusted = holm_adjusted(raw)
        out.append(f"    {label}:")
        for k in sorted(raw):
            if k in adjusted:
                flag = "  significant after Holm" if adjusted[k] < ALPHA else ""
                out.append(f"      k={k}: raw p={raw[k]:.4f}  "
                           f"Holm-adjusted p={adjusted[k]:.4f}{flag}")
            else:
                out.append(f"      k={k}: raw p=--  Holm-adjusted p=--")
        n_sig = sum(1 for v in adjusted.values() if v < ALPHA)
        out.append(f"      {n_sig}/{len(adjusted)} survive Holm at alpha = {ALPHA}")
        out.append("")

    out.append("-" * 70)
    out.append("NOTE ON INDEPENDENCE")
    out.append("  Scores from repeated cross-validation are not fully")
    out.append(f"  independent: the {N_REPEATS} repeats reuse the same recordings,")
    out.append("  so the paired scores overlap. The p-values above are therefore")
    out.append("  supporting evidence alongside the effect sizes and percentile")
    out.append("  ranks, not standalone proof. Say so in the paper.")
    out.append("")
    out.append("NOTE ON SENSOR IMPORTANCE")
    out.append("  Grouped permutation importance measures a sensor's MARGINAL")
    out.append("  contribution. A sensor scoring near zero may carry no")
    out.append("  information, or may be redundant given the other sensors")
    out.append("  present. This method cannot distinguish those two cases; the")
    out.append("  forward-selection order and the sensor correlation matrix are")
    out.append("  reported to help separate them.")

    path.write_text("\n".join(out) + "\n")
    return fair_k


# --------------------------------------------------------------------- figures
def figures_dir(split_mode: str) -> Path:
    """Figure directory for one split mode.

    Outputs are separated by mode so that running the second mode cannot
    overwrite the first.

    Args:
        split_mode: ``"recording"`` or ``"speed"``.

    Returns:
        ``figures/<split_mode>/``.
    """
    return common.FIGURES_DIR / split_mode


def tables_dir(split_mode: str) -> Path:
    """Table directory for one split mode. See ``figures_dir``."""
    return common.TABLES_DIR / split_mode


def save(fig: plt.Figure, stem: str, split_mode: str) -> None:
    """Save a figure as both PDF and 300 dpi PNG, under its mode's directory.

    Args:
        fig: The figure.
        stem: File name without extension.
        split_mode: Which mode produced it.
    """
    out = figures_dir(split_mode)
    fig.savefig(out / f"{stem}.pdf")
    fig.savefig(out / f"{stem}.png", dpi=DPI)
    plt.close(fig)


def fig1_f1_vs_k(wide: pd.DataFrame, table: pd.DataFrame, curves: dict,
                 full_mean: float, split_mode: str) -> None:
    """Box plots of subset-mean macro-F1 per k, with the strategy curves over them."""
    ks = [int(k) for k in table["k"]]
    subset_mean = wide.mean()
    k_lookup = {name: len(name.split("+")) for name in wide.columns}
    data = [subset_mean[[n for n in wide.columns if k_lookup[n] == k]].to_numpy()
            for k in ks]

    fig, ax = plt.subplots(figsize=(COL_W, 2.7))
    # whis=(0, 100): the whiskers span the FULL min-max range. We enumerate
    # every subset, so this is a complete population, not a sample; the
    # 1.5*IQR convention and hidden outliers would conceal the worst-case
    # subsets, which are exactly what a sensor-reduction study cares about.
    ax.boxplot(data, positions=ks, widths=0.5, whis=(0, 100), showfliers=False,
               medianprops={"color": "black"}, boxprops={"color": "0.5"},
               whiskerprops={"color": "0.5"}, capprops={"color": "0.5"})
    ax.plot(ks, curves["oracle"], "k^-", ms=4, label="Oracle (upper bound)")
    ax.plot(ks, curves["perm"], "ko--", ms=4, mfc="white",
            label="Permutation-selected")
    ax.plot(ks, curves["fwd"], "ks:", ms=4, mfc="white", label="Forward-selected")
    ax.axhline(full_mean, color="0.3", lw=0.8, ls="-.",
               label=f"All {len(SENSORS)} sensors")
    ax.set_xlabel("Number of sensors $k$")
    ax.set_ylabel("Macro-F1")
    ax.set_xticks(ks)
    ax.legend(fontsize=6.5, loc="lower right", framealpha=0.95, edgecolor="none")
    ax.grid(axis="y", lw=0.3, color="0.85")
    fig.tight_layout()
    save(fig, "fig1_f1_vs_k", split_mode)


def fig2_sensor_importance(importance: pd.DataFrame,
                           split_mode: str) -> None:
    """Mean and sd of grouped permutation importance across splits."""
    order = importance.mean().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    ax.bar(range(len(order)), importance[order].mean(),
           yerr=importance[order].std(), color="0.6", edgecolor="black",
           lw=0.5, capsize=2)
    ax.axhline(0.0, color="black", lw=0.6)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right")
    # Neutral wording: a low bar means low MARGINAL contribution, which may be
    # redundancy rather than absence of information.
    ax.set_ylabel("Macro-F1 drop when permuted\n(marginal contribution)")
    ax.grid(axis="y", lw=0.3, color="0.85")
    fig.tight_layout()
    save(fig, "fig2_sensor_importance", split_mode)


def fig3_per_class_recall(recall: pd.DataFrame, ks: list[int],
                          split_mode: str) -> None:
    """Per-class recall against k for the permutation-selected subsets."""
    markers = ["o", "s", "^", "v", "D", "x"]
    styles = ["-", "--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 2))]
    fig, ax = plt.subplots(figsize=(COL_W, 2.7))
    for class_name, marker, style in zip(CLASS_ORDER, markers, styles):
        values = recall[class_name].to_numpy(dtype=float)
        finite = np.isfinite(values)
        # Plotting only the finite points keeps the line continuous across a
        # class that is absent from some k, instead of breaking it at NaN.
        # Several classes can sit at recall 1.0, where their lines overlap
        # exactly. They are deliberately NOT offset or jittered: the caption
        # says so instead, because nudging a line would misreport its value.
        ax.plot(np.asarray(ks)[finite], values[finite], marker=marker, ls=style,
                ms=4.0, color="black", lw=0.7, mfc="white",
                label=ABBR.get(class_name, class_name))
    ax.set_xlabel("Number of sensors $k$ (permutation-selected)")
    ax.set_ylabel("Recall")
    ax.set_xticks(ks)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False, fontsize=7, ncol=6, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), handletextpad=0.3, columnspacing=0.8)
    ax.grid(axis="y", lw=0.3, color="0.85")
    fig.tight_layout()
    save(fig, "fig3_per_class_recall", split_mode)


def fig4_detection_errors(errors: pd.DataFrame, ks: list[int],
                          split_mode: str) -> None:
    """Missed detection, false alarm and misdiagnosis against k."""
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    for column, marker, style in [("missed_detection", "o", "-"),
                                  ("false_alarm", "s", "--"),
                                  ("misdiagnosis", "^", ":")]:
        values = errors[column].to_numpy(dtype=float)
        finite = np.isfinite(values)
        ax.plot(np.asarray(ks)[finite], values[finite], marker=marker, ls=style,
                ms=3.5, color="black", mfc="white",
                label=column.replace("_", " "))
    ax.set_xlabel("Number of sensors $k$ (permutation-selected)")
    ax.set_ylabel("Rate")
    ax.set_xticks(ks)
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="y", lw=0.3, color="0.85")
    fig.tight_layout()
    save(fig, "fig4_detection_errors", split_mode)


def fig5_confusion(preds: dict, selection: dict, k_small: int,
                   h2_supported: bool, split_mode: str,
                   stem: str = "fig5_confusion") -> None:
    """Row-normalised confusion matrices, full configuration versus k_small.

    Uses repeat 0 only, so every window is counted exactly once.

    Args:
        preds: Per-split prediction arrays.
        selection: Per-split selection records.
        k_small: The subset size shown in the right-hand panel.
        h2_supported: Whether the fair H2 k exists; only affects the title note.
        split_mode: Which mode produced these results.
        stem: Output file stem, so more than one variant can be produced.
    """
    repeat_zero = [split for split in sorted(selection) if split[0] == 0]
    y_true = np.concatenate([preds[s]["y_true"] for s in repeat_zero])
    short = [ABBR.get(c, c) for c in CLASS_ORDER]
    full_name = subset_name(SENSORS)

    note = "" if h2_supported else " (H2 not supported; default k)"
    panels = [
        (f"All {len(SENSORS)} sensors", lambda s: preds[s][full_name]),
        (f"Permutation-selected, $k$={k_small}{note}",
         lambda s: preds[s][selected_subset(selection[s], "perm", k_small)]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_W, 3.1))
    for ax, (title, getter) in zip(axes, panels):
        y_pred = np.concatenate([getter(s) for s in repeat_zero])
        matrix = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS,
                                  normalize="true")
        ax.imshow(matrix, cmap="Greys", vmin=0, vmax=1)
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        fontsize=6,
                        color="white" if matrix[i, j] > 0.5 else "black")
        ax.set_xticks(range(len(short)))
        ax.set_xticklabels(short, rotation=45, ha="right")
        ax.set_yticks(range(len(short)))
        ax.set_yticklabels(short)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title, fontsize=8)
    fig.tight_layout()
    save(fig, stem, split_mode)


def correlation_heatmap(corr: pd.DataFrame, stem: str, cbar_label: str,
                        split_mode: str, diverging: bool = True) -> float:
    """Draw a sensor-by-sensor correlation matrix with the diagonal masked.

    The diagonal is 1.0 by definition and carries no information, but on a
    fixed -1..+1 scale it dominates the colour range and flattens the real
    off-diagonal structure into indistinguishable grey.  It is therefore masked
    out, and the colour limits are taken from the off-diagonal data.

    Args:
        corr: Square matrix indexed and columned by ``SENSORS``.
        stem: Output file stem.
        cbar_label: Colour-bar label.
        diverging: True for a SIGNED quantity, which gets a symmetric
            ``-limit..+limit`` scale around a meaningful zero.  False for a
            NON-NEGATIVE quantity, which gets a sequential ``0..limit`` scale:
            a symmetric scale there would imply a zero crossing that the
            quantity does not have, and would waste half the colour range.

    Returns:
        The upper colour limit actually used, for the caption.
    """
    values = corr.to_numpy(dtype=float).copy()
    np.fill_diagonal(values, np.nan)

    off_diagonal = values[np.isfinite(values)]
    largest = float(np.max(np.abs(off_diagonal))) if off_diagonal.size else 0.05
    # Round up to the next 0.05 so the colour bar lands on readable ticks.
    limit = max(0.05, float(np.ceil(largest / 0.05) * 0.05))
    low = -limit if diverging else 0.0

    cmap = plt.get_cmap("Greys").with_extremes(bad="white")

    fig, ax = plt.subplots(figsize=(COL_W, 3.1))
    image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap,
                      vmin=low, vmax=limit)
    ax.set_xticks(range(len(SENSORS)))
    ax.set_xticklabels(SENSORS, rotation=45, ha="right")
    ax.set_yticks(range(len(SENSORS)))
    ax.set_yticklabels(SENSORS)
    for i in range(len(SENSORS)):
        for j in range(len(SENSORS)):
            if i == j:
                ax.text(j, i, "--", ha="center", va="center", fontsize=5.5,
                        color="0.6")
                continue
            value = values[i, j]
            dark = value > low + 0.8 * (limit - low)
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5.5,
                    color="white" if dark else "black")
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.03, label=cbar_label)
    fig.tight_layout()
    save(fig, stem, split_mode)
    return limit


def fig6_sensor_correlation(corr: pd.DataFrame, split_mode: str) -> float:
    """Spearman correlation of the RMS feature between sensors."""
    return correlation_heatmap(corr, "fig6_sensor_correlation",
                               r"Spearman $\rho$ of RMS", split_mode)


def fig6b_mean_absolute_correlation(corr: pd.DataFrame,
                                    split_mode: str) -> float:
    """Mean absolute Spearman correlation over all 18 matched features."""
    return correlation_heatmap(corr, "fig6b_mean_abs_correlation",
                               r"Mean $|\rho|$ over 18 features",
                               split_mode, diverging=False)


def mean_absolute_feature_correlation(features: pd.DataFrame,
                                      feature_names: list[str]) -> pd.DataFrame:
    """Mean |Spearman rho| between sensors, averaged over matched features.

    For each of the 18 features, every sensor's column for THAT feature is
    correlated with each other; the absolute correlations are then averaged
    over the features.  This measures redundancy across the whole feature set
    rather than through RMS alone, which is a single amplitude summary.

    Args:
        features: The feature table.
        feature_names: The 18 feature suffixes.

    Returns:
        A symmetric ``len(SENSORS)`` square matrix; the diagonal is 1.0.
    """
    total = np.zeros((len(SENSORS), len(SENSORS)))
    used = 0
    for name in feature_names:
        columns = [f"{sensor}__{name}" for sensor in SENSORS]
        missing = [c for c in columns if c not in features.columns]
        if missing:
            raise ValueError(f"Feature table is missing columns: {missing}")
        block = features[columns]
        # Spearman is Pearson on ranks; ranking once per block is far cheaper
        # than 28 pairwise scipy calls.
        matrix = block.rank().corr(method="pearson").to_numpy(dtype=float)
        total += np.abs(np.nan_to_num(matrix, nan=0.0))
        used += 1
    return pd.DataFrame(total / used, index=SENSORS, columns=SENSORS)


def write_captions(path: Path, n_splits: int, fair_k: int | None,
                   corr_limit: float, corr_limit_b: float,
                   split_mode: str) -> None:
    """Write the figure captions as LaTeX macros for paper/main.tex to include.

    Captions are generated rather than hand-written so that what they claim
    about aggregation always matches what the code actually did.

    Args:
        path: Destination ``.tex`` file.
        n_splits: Number of cross-validation splits behind the results.
        fair_k: The fair H2 k used in figure 5, or None when H2 failed.
        corr_limit: Symmetric colour limit used by figure 6.
        corr_limit_b: Symmetric colour limit used by figure 6b.
        split_mode: Named in EVERY caption so a figure cannot be misfiled
            into the paper under the wrong evaluation protocol.
    """
    binom = rf"\binom{{{len(SENSORS)}}}{{k}}"
    pm = r"\pm"
    times = r"\times"
    not_independent = (
        f"The {n_splits} splits come from {N_REPEATS} repeats of {N_SPLITS}-fold "
        "cross-validation over the same recordings and are therefore not "
        "mutually independent.")
    k_shown = fair_k if fair_k is not None else 2
    mode_name = {"speed": "held-out speed bands (PRIMARY protocol)",
                 "recording": "held-out recordings (SECONDARY robustness "
                              "analysis)"}.get(split_mode, split_mode)
    mode = f"Evaluation protocol: {mode_name}. "
    h2_note = ("" if fair_k is not None else
               " H2 was not supported, so $k=2$ is shown as a default.")

    captions = {
        "FigOneCaption":
            mode +
            "Macro-F1 against the number of sensors $k$. Each subset is first "
            f"averaged over the {n_splits} splits; the box plots then summarise "
            f"those per-subset means across all ${binom}$ subsets of each size. "
            "Whiskers span the full minimum to maximum range rather than "
            f"1.5${times}$IQR: every subset is enumerated, so this is a complete "
            "population rather than a sample, and the extreme subsets are of "
            "direct interest. The oracle line is the maximum over those "
            f"${binom}$ candidates selected using TEST scores and is an upper "
            "bound only. " + not_independent,
        "FigTwoCaption":
            mode +
            "Grouped permutation importance: the drop in inner-validation "
            "macro-F1 when all 18 features of one sensor are permuted together "
            "with a shared row permutation. Bars are the mean over the "
            f"{n_splits} splits and error bars their standard deviation. This "
            "measures each sensor's MARGINAL contribution, so a bar near zero "
            "may indicate redundancy given the other sensors rather than an "
            "absence of information. " + not_independent,
        "FigThreeCaption":
            mode +
            "Per-class recall against $k$ for the permutation-selected subsets. "
            f"Predictions are pooled across all {n_splits} splits before recall "
            "is computed. Several classes may sit at recall 1.0, where their "
            "lines coincide exactly; they are deliberately not offset or "
            "jittered. " + not_independent,
        "FigFourCaption":
            mode +
            "Missed-detection, false-alarm and misdiagnosis rates against $k$ "
            "for the permutation-selected subsets, computed from predictions "
            f"pooled across all {n_splits} splits. " + not_independent,
        "FigFiveCaption":
            mode +
            "Row-normalised confusion matrices for the full 8-sensor "
            "configuration and for the permutation-selected subset at "
            f"$k={k_shown}$." + h2_note + " Repeat 0 only, so every window is "
            "counted exactly once and the two panels are directly comparable.",
        "FigFiveKOneCaption":
            mode +
            "Row-normalised confusion matrices for the full "
            f"{len(SENSORS)}-sensor configuration and for the "
            "permutation-selected SINGLE sensor ($k=1$). This is the variant "
            "that shows where the configurations actually differ; the $k$ "
            f"={k_shown} variant compares two near-identical configurations by "
            "construction. Repeat 0 only, so every window is counted exactly "
            "once and the two panels are directly comparable.",
        "FigSixCaption":
            mode +
            "Spearman correlation of the RMS feature between sensors, over all "
            "windows. The diagonal is 1.0 by definition and is masked. The "
            "quantity is signed, so the colour scale is symmetric about zero "
            "and derived from the off-diagonal data "
            f"(${pm}{corr_limit:.2f}$), so the real structure is visible instead "
            "of being flattened by the diagonal.",
        "FigSixBCaption":
            mode +
            "Mean absolute Spearman correlation between sensors, averaged over "
            "all 18 matched features rather than RMS alone, over all windows. "
            "The diagonal is masked. The quantity is non-negative, so the "
            "colour scale is sequential and runs from 0 to the largest "
            f"off-diagonal value (${corr_limit_b:.2f}$), rather than the "
            "symmetric scale used for the signed matrix. "
            "This is a broader redundancy measure than the RMS-only matrix; it "
            "is descriptive and enters no hypothesis test.",
    }
    lines = ["% Generated by 04_analyze.py. Do not edit by hand."]
    for name, text in captions.items():
        lines.append("\\newcommand{\\" + name + "}{" + text + "}")
    path.write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------------ main
def analyse(res_dir: Path, split_mode: str, expect_full_run: bool = False) -> dict:
    """Run the whole analysis for one split mode.

    Args:
        res_dir: ``results/<split_mode>/``.
        split_mode: Which split mode produced these results.
        expect_full_run: Require the full 10-split row count.

    Returns:
        A small summary dictionary, for tests and logging.
    """
    scores, selection, preds = load_results(res_dir)
    splits = sorted(selection)

    # Level F runs FIRST: a failure must abort before anything is written.
    check_results(scores, selection, expect_full_run)
    inputs = check_provenance(res_dir)

    table, wide, curves = build_main_table(scores, selection)
    full_name = subset_name(SENSORS)
    full_mean = float(wide[full_name].mean())
    full_sd = float(wide[full_name].std(ddof=0))
    check_full_configuration(table, full_mean)
    check_p_values(table)

    out_figures = figures_dir(split_mode)
    out_tables = tables_dir(split_mode)
    out_figures.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)

    format_main_table(table).to_csv(out_tables / "table_main.csv", index=False)
    write_latex_table(table, out_tables / "table_main.tex")

    fair_k = write_hypotheses(table, curves, full_mean, full_sd, len(splits),
                              split_mode, out_tables / "hypotheses.txt")

    ks = [int(k) for k in table["k"]]
    recall_rows, error_rows = [], []
    for k in ks:
        y_true = np.concatenate([preds[s]["y_true"] for s in splits])
        y_pred = np.concatenate(
            [preds[s][selected_subset(selection[s], "perm", k)] for s in splits])
        recall = per_class_recall(y_true, y_pred)
        recall_rows.append({"k": k, **dict(zip(CLASS_ORDER, recall)),
                            "mean_recall": float(np.nanmean(recall))})
        error_rows.append({"k": k, **error_rates(y_true, y_pred)})
    recall_df = pd.DataFrame(recall_rows)
    errors_df = pd.DataFrame(error_rows)
    recall_df.round(4).to_csv(out_tables / "table_per_class.csv", index=False)
    errors_df.round(4).to_csv(out_tables / "table_detection_errors.csv",
                              index=False)

    importance = pd.DataFrame([selection[s]["perm_importance"] for s in splits])
    importance.insert(0, "fold", [s[1] for s in splits])
    importance.insert(0, "repeat", [s[0] for s in splits])
    importance.to_csv(out_tables / "sensor_importance_per_split.csv",
                      index=False)

    features = load_features()
    rms = features[[feature_columns(features, [s])[0] for s in SENSORS]]
    rms.columns = SENSORS
    corr = rms.corr(method="spearman")
    corr.round(3).to_csv(out_tables / "sensor_rms_correlation.csv")

    feature_names = sorted({c.split("__", 1)[1]
                            for c in feature_columns(features, SENSORS)})
    corr_mean_abs = mean_absolute_feature_correlation(features, feature_names)
    corr_mean_abs.round(3).to_csv(
        out_tables / "sensor_mean_abs_correlation.csv")

    fig1_f1_vs_k(wide, table, curves, full_mean, split_mode)
    fig2_sensor_importance(importance[SENSORS], split_mode)
    fig3_per_class_recall(recall_df, ks, split_mode)
    fig4_detection_errors(errors_df, ks, split_mode)
    fig5_confusion(preds, selection, fair_k if fair_k else 2,
                   fair_k is not None, split_mode)
    # Variant at k=1, where the two panels actually differ. The fair-H2-k
    # variant compares two near-identical configurations by construction.
    fig5_confusion(preds, selection, 1, True, split_mode,
                   stem="fig5b_confusion_k1")
    corr_limit = fig6_sensor_correlation(corr, split_mode)
    corr_limit_b = fig6b_mean_absolute_correlation(corr_mean_abs, split_mode)

    write_captions(out_figures / "captions.tex", len(splits), fair_k,
                   corr_limit, corr_limit_b, split_mode)
    write_run_metadata(
        out_tables, "04_analyze.py",
        split_mode=split_mode,
        results_dir=str(res_dir),
        n_splits=len(splits),
        n_score_rows=int(len(scores)),
        expect_full_run=expect_full_run,
        fair_h2_k=fair_k,
        inputs=inputs,
    )

    LOG.info("Tables  -> %s", out_tables)
    LOG.info("Figures -> %s", out_figures)
    return {"n_splits": len(splits), "full_mean": full_mean, "fair_k": fair_k,
            "table": table}


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="recording", choices=["recording", "speed"],
                    help="which results directory to analyse")
    ap.add_argument("--expect-full-run", action="store_true",
                    help="require the full 10-split result set "
                         f"({N_REPEATS * N_SPLITS * N_SUBSETS} rows)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    summary = analyse(common.RESULTS_DIR / args.split, args.split,
                      args.expect_full_run)
    print((tables_dir(args.split) / "hypotheses.txt").read_text())
    LOG.info("Analysed %d splits; full configuration %.4f",
             summary["n_splits"], summary["full_mean"])


if __name__ == "__main__":
    main()
