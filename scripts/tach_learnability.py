"""SUPPLEMENTARY: is the tachometer's recording artefact LEARNABLE?

Near-zero marginal permutation importance shows only that the model did not
NEED the tachometer when seven good sensors were present. Whether the artefact
is learnable is a separate question, answered by how the channel performs ALONE
and on which classes.

READ-ONLY and NO REFITTING. Every number here comes from the predictions already
saved by the 8-sensor secondary run; no model is trained.

A mid-ranked sensor is reported alongside as a control.

Usage:
    python scripts/tach_learnability.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import common  # noqa: E402
from common import CLASS_LABELS, CLASS_ORDER, FoldScorer, per_class_recall  # noqa: E402

LOG = logging.getLogger("tach")

TAG = "secondary_8sensor_speed"
ABBR = {"normal": "N", "imbalance": "IMB", "horizontal-misalignment": "HM",
        "vertical-misalignment": "VM", "underhang": "UB", "overhang": "OB"}


def load_predictions(res_dir: Path) -> list[tuple[str, dict]]:
    """Load every split's saved predictions, newest-ordered by split tag."""
    out = []
    for path in sorted(res_dir.glob("preds_*.npz")):
        tag = path.stem.split("_", 1)[1]
        with np.load(path) as data:
            out.append((tag, {k: data[k] for k in data.files}))
    return out


def single_sensor_scores(res_dir: Path) -> pd.DataFrame:
    """Mean and sd of single-sensor macro-F1, read from the stored score rows."""
    scores = pd.concat(
        [pd.read_csv(p) for p in sorted(res_dir.glob("scores_*.csv"))],
        ignore_index=True)
    singles = scores[scores.k == 1]
    out = singles.groupby("subset")["macro_f1"].agg(["mean", "std", "count"])
    out.columns = ["macro_f1_mean", "macro_f1_sd", "n_splits"]
    return out.sort_values("macro_f1_mean", ascending=False)


def pooled_per_class(predictions: list[tuple[str, dict]],
                     sensor: str) -> np.ndarray:
    """Per-class recall for one single-sensor model, pooled over all splits."""
    y_true = np.concatenate([d["y_true"] for _, d in predictions])
    y_pred = np.concatenate([d[sensor] for _, d in predictions])
    return per_class_recall(y_true, y_pred)


def pooled_confusion(predictions: list[tuple[str, dict]], sensor: str,
                     normalize: str | None = "true") -> np.ndarray:
    """Confusion matrix for one single-sensor model, pooled over all splits."""
    y_true = np.concatenate([d["y_true"] for _, d in predictions])
    y_pred = np.concatenate([d[sensor] for _, d in predictions])
    return confusion_matrix(y_true, y_pred, labels=CLASS_LABELS,
                            normalize=normalize)


def show_matrix(matrix: np.ndarray, title: str) -> None:
    """Print a confusion matrix with class abbreviations."""
    short = [ABBR[c] for c in CLASS_ORDER]
    print(f"\n{title}")
    print("        " + "".join(f"{s:>7}" for s in short) + "   (predicted)")
    for i, row in enumerate(matrix):
        print(f"  {short[i]:<5} " + "".join(f"{v:>7.3f}" for v in row))


def main(argv: list[str] | None = None) -> None:
    """Report tachometer learnability from saved predictions."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--control", default="ob_rad",
                    help="mid-ranked sensor reported as a control")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    res_dir = common.RESULTS_DIR / TAG
    out_dir = common.TABLES_DIR / TAG
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = load_predictions(res_dir)
    print(f"Loaded predictions from {len(predictions)} splits. "
          "NO MODEL WAS REFITTED.")

    # (a) single-sensor macro-F1 for every channel, tachometer included
    singles = single_sensor_scores(res_dir)
    singles.round(4).to_csv(out_dir / "supp_single_sensor_scores.csv")
    print("\n=== (a) single-sensor macro-F1, 8-sensor secondary run ===")
    print(singles.round(4).to_string())

    rows = []
    for sensor in (s for s in ["tach", args.control] if s in singles.index):
        recall = pooled_per_class(predictions, sensor)
        rows.append(pd.Series(recall, index=CLASS_ORDER, name=sensor))
    per_class = pd.DataFrame(rows)
    per_class["mean_recall"] = per_class.mean(axis=1)
    per_class.round(4).to_csv(out_dir / "supp_single_sensor_per_class.csv")

    print(f"\n=== (b, d) per-class recall, pooled over {len(predictions)} "
          "splits ===")
    print(per_class.round(4).to_string())

    faults = [c for c in CLASS_ORDER if c != common.HEALTHY_CLASS]
    bearings = ["underhang", "overhang"]
    others = [c for c in faults if c not in bearings]
    print("\n  bearing classes vs the other fault classes:")
    for sensor in per_class.index:
        b = per_class.loc[sensor, bearings].mean()
        o = per_class.loc[sensor, others].mean()
        print(f"    {sensor:<8} underhang+overhang {b:.4f} | "
              f"IMB+HM+VM {o:.4f} | difference {b - o:+.4f}")

    for sensor in per_class.index:
        matrix = pooled_confusion(predictions, sensor)
        pd.DataFrame(matrix, index=CLASS_ORDER, columns=CLASS_ORDER).round(4) \
            .to_csv(out_dir / f"supp_confusion_{sensor}.csv")
        show_matrix(matrix, f"=== (c, d) confusion matrix, {sensor} alone "
                            f"(row-normalised, pooled) ===")

    LOG.info("\nTables -> %s", out_dir)


if __name__ == "__main__":
    main()
