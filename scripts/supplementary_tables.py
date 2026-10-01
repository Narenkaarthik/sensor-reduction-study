"""SUPPLEMENTARY, READ-ONLY descriptive tables. No protocol implications.

Reads the stored results and writes three tables that describe the sensors and
the selected subsets from different angles. Nothing here feeds a hypothesis
test, and nothing here is used to make any decision.

  supp_sensor_profile.csv    per sensor: marginal permutation importance,
                             standalone single-sensor macro-F1, and mean |rho|
                             to the other sensors
  supp_oracle_membership.csv how often each sensor appears in the ORACLE subset
                             at each k, across the splits
  supp_selected_membership.csv the same counts for the permutation-selected and
                             forward-selected subsets

Usage:
    python scripts/supplementary_tables.py --split speed
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import common  # noqa: E402
from common import SENSORS, subset_name  # noqa: E402

LOG = logging.getLogger("supplementary")


def sensor_profile(res_dir: Path, scores: pd.DataFrame,
                   features: pd.DataFrame) -> pd.DataFrame:
    """One row per sensor: importance, standalone score, and redundancy.

    Args:
        res_dir: Results directory for the split mode.
        scores: All per-split score rows.
        features: The feature table, for the correlation column.

    Returns:
        A table indexed by sensor.
    """
    importance = pd.DataFrame([
        json.loads(p.read_text())["perm_importance"]
        for p in sorted(res_dir.glob("selection_*.json"))])

    singles = scores[scores.k == 1].groupby("subset")["macro_f1"]
    single_mean = {name: float(v) for name, v in singles.mean().items()}
    single_sd = {name: float(v) for name, v in singles.std(ddof=0).items()}

    # Mean |Spearman rho| to every OTHER sensor, averaged over the 18 features.
    names = sorted({c.split("__", 1)[1]
                    for c in common.feature_columns(features, SENSORS)})
    total = np.zeros((len(SENSORS), len(SENSORS)))
    for feature in names:
        block = features[[f"{s}__{feature}" for s in SENSORS]]
        total += np.abs(np.nan_to_num(
            block.rank().corr(method="pearson").to_numpy(dtype=float), nan=0.0))
    corr = total / len(names)
    np.fill_diagonal(corr, np.nan)

    rows = []
    for i, sensor in enumerate(SENSORS):
        rows.append({
            "sensor": sensor,
            "perm_importance_mean": float(importance[sensor].mean()),
            "perm_importance_sd": float(importance[sensor].std(ddof=0)),
            "single_sensor_macro_f1": single_mean.get(sensor, np.nan),
            "single_sensor_sd": single_sd.get(sensor, np.nan),
            "mean_abs_rho_to_others": float(np.nanmean(corr[i])),
        })
    out = pd.DataFrame(rows).set_index("sensor")
    return out.sort_values("perm_importance_mean", ascending=False)


def membership_counts(scores: pd.DataFrame, selection: dict,
                      strategy: str) -> pd.DataFrame:
    """Count how often each sensor appears in a chosen subset at each k.

    Args:
        scores: All per-split score rows.
        selection: Selection records keyed by ``(repeat, fold)``.
        strategy: ``"oracle"``, ``"perm"`` or ``"fwd"``.

    Returns:
        Sensors as rows, k as columns, counts out of the number of splits.
    """
    splits = sorted(selection)
    k_of = scores.drop_duplicates("subset").set_index("subset")["k"]
    ks = sorted(k_of.unique())
    counts = pd.DataFrame(0, index=SENSORS, columns=ks, dtype=int)

    if strategy == "oracle":
        wide = scores.pivot_table(index=["repeat", "fold"], columns="subset",
                                  values="macro_f1").loc[splits]
        subset_mean = wide.mean()
        for k in ks:
            names = list(k_of[k_of == k].index)
            best = subset_mean[names].idxmax()
            for sensor in best.split("+"):
                counts.loc[sensor, k] = len(splits)   # one oracle subset per k
        return counts

    key = "perm_ranking" if strategy == "perm" else "forward_order"
    for split in splits:
        order = selection[split][key]
        for k in ks:
            for sensor in order[:k]:
                counts.loc[sensor, k] += 1
    return counts


def main(argv: list[str] | None = None) -> None:
    """Write the supplementary tables for one split mode."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="speed", choices=["recording", "speed"])
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    res_dir = common.RESULTS_DIR / args.split
    out_dir = common.TABLES_DIR / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = pd.concat([pd.read_csv(p) for p in sorted(res_dir.glob("scores_*.csv"))],
                       ignore_index=True)
    selection = {}
    for path in sorted(res_dir.glob("selection_*.json")):
        entry = json.loads(path.read_text())
        selection[(int(entry["repeat"]), int(entry["fold"]))] = entry
    features = common.load_features()

    profile = sensor_profile(res_dir, scores, features)
    profile.round(4).to_csv(out_dir / "supp_sensor_profile.csv")
    print("\n=== (a) sensor profile ===")
    print(profile.round(4).to_string())

    oracle = membership_counts(scores, selection, "oracle")
    oracle.to_csv(out_dir / "supp_oracle_membership.csv")
    print(f"\n=== (b) oracle subset membership, out of {len(selection)} splits ===")
    print(oracle.to_string())

    frames = []
    for strategy, label in (("perm", "permutation"), ("fwd", "forward")):
        counts = membership_counts(scores, selection, strategy)
        counts.insert(0, "strategy", label)
        frames.append(counts)
        print(f"\n=== (c) {label}-selected membership, out of "
              f"{len(selection)} splits ===")
        print(counts.drop(columns="strategy").to_string())
    pd.concat(frames).to_csv(out_dir / "supp_selected_membership.csv")

    LOG.info("\nSupplementary tables -> %s", out_dir)


if __name__ == "__main__":
    main()
