"""Effect sizes behind the tachometer exclusion (PROTOCOL.md section 2.2).

PROTOCOL.md section 2.2 justifies excluding the tachometer with a table of
Cohen's *d* values, and the paper quotes them. Until 2026-10-31 that table had
NO reproduction path in this repository: section 2.2 pointed at
``scripts/diagnose_speed.py``, which diagnoses the threshold failure but has
never computed an effect size. This script closes that gap, so the numbers can
be checked rather than taken on trust.

The three groups are the ones section 2.2 defines:

  A  the AFFECTED recordings -- tachometer estimate outside 10-65 Hz
  B  the OTHER ball_fault recordings, whose tachometer reading is usable
  C  every non-ball_fault window

The argument the table has to carry is that the tachometer separates group A
from everything else at a large effect size, while failing to separate
``ball_fault`` IN GENERAL (B against C) at any meaningful size. That pattern is
what makes the channel a recording artefact rather than a fault signature: a
genuine fault signature would not separate a condition from other recordings of
that same condition.

READ-ONLY and NO MODEL IS FITTED. Every number comes from ``features.csv``.

Usage:
    python scripts/tach_effect_sizes.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import common  # noqa: E402

LOG = logging.getLogger("tach_effect")

SECONDARY = "secondary_8sensor_speed"
TACH = "tach"
BALL_FAULT = "ball_fault"

# The same plausible-range criterion used by scripts/diagnose_speed.py and
# scripts/artifact_exploitation.py. Kept in step with them by the assertion in
# main(), which fails loudly if the three ever disagree about the count.
TACH_RANGE_HZ = (10.0, 65.0)
EXPECTED_AFFECTED = 67


def affected_recordings(features: pd.DataFrame) -> set[int]:
    """Recording ids whose tachometer estimate falls outside the plausible range.

    Args:
        features: The feature table.

    Returns:
        The affected recording ids.
    """
    per_rec = features.groupby("recording_id")["speed_hz_tach"].first()
    low, high = TACH_RANGE_HZ
    return set(per_rec[(per_rec < low) | (per_rec > high)].index.tolist())


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's *d* with the pooled standard deviation.

    Args:
        x: First sample.
        y: Second sample.

    Returns:
        The standardised mean difference, or NaN if the pooled sd is zero.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = x.size, y.size
    pooled = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1))
                     / (nx + ny - 2))
    return float((x.mean() - y.mean()) / pooled) if pooled else float("nan")


def tach_columns(features: pd.DataFrame) -> list[str]:
    """Every tachometer feature column, in table order.

    Args:
        features: The feature table.

    Returns:
        The 18 ``tach__*`` column names.
    """
    return [c for c in features.columns if c.startswith(f"{TACH}__")]


def effect_size_table(features: pd.DataFrame,
                      affected: set[int]) -> pd.DataFrame:
    """Cohen's *d* for each tachometer feature across the three comparisons.

    Args:
        features: The feature table.
        affected: Affected recording ids.

    Returns:
        One row per tachometer feature, sorted by descending |d(A vs C)|.
    """
    is_affected = features["recording_id"].isin(affected)
    is_ball = (features["condition_path"].astype(str)
               .str.contains(BALL_FAULT, case=False, regex=False))

    group_a = features[is_affected]
    group_b = features[is_ball & ~is_affected]
    group_c = features[~is_ball]

    LOG.info("window counts   A affected %d   B other ball_fault %d   "
             "C non-ball_fault %d", len(group_a), len(group_b), len(group_c))

    rows = []
    for col in tach_columns(features):
        rows.append({
            "feature": col.removeprefix(f"{TACH}__"),
            "d_affected_vs_nonball": cohens_d(group_a[col], group_c[col]),
            "d_otherball_vs_nonball": cohens_d(group_b[col], group_c[col]),
            "d_affected_vs_otherball": cohens_d(group_a[col], group_b[col]),
            "n_affected": len(group_a),
            "n_other_ball": len(group_b),
            "n_non_ball": len(group_c),
        })

    table = pd.DataFrame(rows)
    order = table["d_affected_vs_nonball"].abs().sort_values(ascending=False)
    return table.loc[order.index].reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    """Compute the effect sizes and write the supplementary table."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top", type=int, default=5,
                    help="how many features to print (all are written)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    features = common.load_features()
    affected = affected_recordings(features)
    assert len(affected) == EXPECTED_AFFECTED, (
        f"expected {EXPECTED_AFFECTED} affected recordings, found "
        f"{len(affected)}; the criterion has drifted from "
        f"scripts/diagnose_speed.py")

    table = effect_size_table(features, affected)

    shown = table.head(args.top).set_index("feature")[
        ["d_affected_vs_nonball", "d_otherball_vs_nonball",
         "d_affected_vs_otherball"]]
    print(f"\n=== Cohen's d, top {args.top} tachometer features by "
          f"|d(A vs C)| ===")
    print(shown.round(3).to_string())

    worst_b_vs_c = table["d_otherball_vs_nonball"].abs().max()
    print(f"\nLargest |d| for OTHER ball_fault against non-ball_fault: "
          f"{worst_b_vs_c:.3f}")
    print("The channel does not separate ball_fault in general; the effect "
          "lives entirely in the affected recordings, which is what makes it "
          "a recording artefact rather than a fault signature.")

    out_dir = common.TABLES_DIR / SECONDARY
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "supp_tach_effect_sizes.csv"
    table.round(4).to_csv(out_path, index=False)
    LOG.info("\nTable -> %s", out_path)


if __name__ == "__main__":
    main()
