"""SUPPLEMENTARY: does the degradation curve survive a non-tree model?

The study fixes one Random Forest by design (PROTOCOL.md section 4), so that
any difference between sensor subsets is attributable to the sensors and not to
model selection. That design has a cost a reviewer is right to press on: a
curve measured with one tree ensemble could in principle reflect a tree-based
inductive bias rather than the sensors.

This script answers that directly by re-evaluating ALL 127 subsets on ALL 10
splits with a support vector machine -- a different model family with a
different bias -- and comparing the curve, H1 and H2 against the primary
result.

WHAT THIS IS NOT. It does not change the primary protocol, and no number it
produces replaces one. It is a third analysis alongside the secondary
(held-out recordings) and supplementary (8-sensor) ones, reported separately
and labelled as such.

TWO DEPARTURES FROM PROTOCOL SECTION 4, both forced by the model and both
confined to this script:

  1. Feature scaling. An RBF SVM is not scale invariant, so features are
     standardised. The scaler is fitted INSIDE the pipeline on the training
     fold only, so it never sees the test fold and introduces no leakage.
     The Random Forest is untouched and is still run without scaling.
  2. Hyper-parameters are left at scikit-learn defaults (C=1, gamma='scale').
     No tuning was performed, for the same reason the forest is not tuned: a
     tuned SVM against an untuned forest would confound model family with
     tuning effort.

The subsets evaluated at each k are the ones the PRIMARY analysis selected by
permutation importance, on training data only. Selection is not redone here.
The question asked is therefore "does the reduction the primary analysis found
still hold under a different model?", which is the question that matters for a
sensor-selection result, and it avoids introducing a second selection pipeline
whose differences would be impossible to attribute.

Usage:
    python scripts/secondary_svm.py --split speed
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import common  # noqa: E402
from common import (SEED, SENSORS, FoldScorer, assert_no_test_indices,  # noqa: E402
                    encode_labels, feature_columns, subset_name)

LOG = logging.getLogger("svm")
TOLERANCE = 0.01


def make_svm():
    """The one SVM used everywhere here: scaled, balanced, untuned.

    Returns:
        A fresh pipeline. Built per fit so no state leaks between folds.
    """
    return make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", class_weight="balanced", random_state=SEED,
            cache_size=500),
    )


def all_subsets() -> list[tuple[str, ...]]:
    """Every non-empty sensor subset, smallest first."""
    out: list[tuple[str, ...]] = []
    for k in range(1, len(SENSORS) + 1):
        out.extend(combinations(SENSORS, k))
    return out


def selected_subsets(results_dir: Path) -> dict[tuple[int, int], dict[int, tuple[str, ...]]]:
    """The permutation-selected subset per split per k, from the primary run.

    Args:
        results_dir: The primary ``results/<mode>`` directory.

    Returns:
        ``{(repeat, fold): {k: subset}}``.
    """
    out: dict[tuple[int, int], dict[int, tuple[str, ...]]] = {}
    for path in sorted(results_dir.glob("selection_r*_f*.json")):
        sel = json.loads(path.read_text())
        ranking = sel["perm_ranking"]
        out[(sel["repeat"], sel["fold"])] = {
            k: tuple(ranking[:k]) for k in range(1, len(SENSORS) + 1)}
    return out


def run(df: pd.DataFrame, y, split_mode: str,
        selected: dict) -> pd.DataFrame:
    """Evaluate every subset on every split under the SVM.

    Args:
        df: The feature table.
        y: Encoded labels.
        split_mode: Which grouping to use.
        selected: Permutation-selected subsets from the primary run.

    Returns:
        One row per (split, subset) with macro-F1 and whether it was selected.
    """
    subsets = all_subsets()
    rows = []
    for r, f, train_idx, test_idx in common.outer_splits(df, y, split_mode):
        tag = f"r{r}_f{f}"
        assert_no_test_indices(train_idx, test_idx, f"svm outer split {tag}")
        scorer = FoldScorer.for_fold(y[test_idx], tag)
        picked = selected.get((r, f), {})
        started = time.perf_counter()

        for subset in subsets:
            cols = feature_columns(df, subset)
            x = df[cols].to_numpy(dtype=np.float64)
            model = make_svm()
            model.fit(x[train_idx], y[train_idx])
            pred = model.predict(x[test_idx])
            name = subset_name(subset)
            rows.append({
                "repeat": r, "fold": f, "k": len(subset), "subset": name,
                "macro_f1": scorer.macro_f1(y[test_idx], pred),
                "is_selected": picked.get(len(subset)) is not None
                and subset_name(picked[len(subset)]) == name,
            })
        LOG.info("[%s] %d subsets in %.1f s", tag, len(subsets),
                 time.perf_counter() - started)
    return pd.DataFrame(rows)


def summarise(scores: pd.DataFrame) -> pd.DataFrame:
    """Per-k curve: all subsets, oracle and the primary-selected subset.

    Args:
        scores: Raw per-split per-subset scores.

    Returns:
        One row per k.
    """
    per_subset = (scores.groupby(["k", "subset"])["macro_f1"]
                  .mean().reset_index())
    sel = (scores[scores.is_selected].groupby(["k", "subset"])["macro_f1"]
           .mean().reset_index())
    sel_sd = (scores[scores.is_selected].groupby("k")["macro_f1"].std(ddof=0))

    rows = []
    for k, grp in per_subset.groupby("k"):
        chosen = sel[sel.k == k]
        rows.append({
            "k": int(k),
            "n_subsets": len(grp),
            "all_mean": grp.macro_f1.mean(),
            "all_sd": grp.macro_f1.std(ddof=0),
            "all_median": grp.macro_f1.median(),
            "oracle": grp.macro_f1.max(),
            "selected_mean": chosen.macro_f1.mean() if len(chosen) else np.nan,
            "selected_sd": float(sel_sd.get(k, np.nan)),
        })
    return pd.DataFrame(rows)


def verdicts(curve: pd.DataFrame) -> list[str]:
    """H1 and H2 under the SVM, by the same thresholds as the primary run."""
    out = []
    sel = curve.set_index("k")["selected_mean"]
    full = float(sel.loc[len(SENSORS)])
    out.append(f"Full configuration ({len(SENSORS)} sensors): {full:.4f}")

    violations = [k for k in range(1, len(SENSORS))
                  if sel.loc[k] - sel.loc[k + 1] > TOLERANCE]
    out.append(f"H1 (primary-selected curve): "
               f"{'SUPPORTED' if not violations else 'VIOLATED at k=' + str(violations)}"
               f" ({len(violations)} violation(s), TOLERANCE={TOLERANCE})")

    threshold = full - TOLERANCE
    qualifying = [k for k in range(1, len(SENSORS)) if sel.loc[k] >= threshold]
    out.append(f"H2 threshold = {threshold:.4f}; smallest qualifying k = "
               f"{min(qualifying) if qualifying else 'none'}")
    return out


def main(argv: list[str] | None = None) -> None:
    """Run the supplementary SVM analysis for one split mode."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="speed", choices=["speed", "recording"])
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    df = common.load_features()
    y = encode_labels(df)
    selected = selected_subsets(common.RESULTS_DIR / args.split)
    assert selected, "no selection_*.json found; run the primary study first"

    scores = run(df, y, args.split, selected)
    curve = summarise(scores)

    out_dir = common.TABLES_DIR / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    curve.round(4).to_csv(out_dir / "supp_svm_curve.csv", index=False)

    lines = verdicts(curve)
    print("\n=== SUPPLEMENTARY: SVM, " + args.split + " protocol ===")
    print(curve.round(4).to_string(index=False))
    print()
    print("\n".join(lines))
    (out_dir / "supp_svm_verdicts.txt").write_text(
        "SUPPLEMENTARY SVM ANALYSIS (not the primary protocol)\n"
        f"Split mode: {args.split}\n"
        "Subsets are those the PRIMARY analysis selected by permutation\n"
        "importance on training data; selection is not redone here.\n\n"
        + "\n".join(lines) + "\n")
    LOG.info("\nCurve    -> %s", out_dir / "supp_svm_curve.csv")
    LOG.info("Verdicts -> %s", out_dir / "supp_svm_verdicts.txt")


if __name__ == "__main__":
    main()
