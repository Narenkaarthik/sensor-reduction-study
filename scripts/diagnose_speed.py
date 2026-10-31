"""Diagnose the out-of-range tachometer speed estimates. READ-ONLY.

Changes nothing: it does not touch the estimator, the plausibility test, the
feature table or the protocol. It reports.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ALL_CHANNELS, DATA_DIR, FS  # noqa: E402

TACH = ALL_CHANNELS.index("tach")


def load_tach(path: Path) -> np.ndarray:
    """Read only the tachometer column of one recording."""
    return pd.read_csv(path, header=None, usecols=[TACH]).to_numpy(
        dtype=np.float64).ravel()


def edges_and_intervals(tach: np.ndarray) -> tuple[float, np.ndarray, float]:
    """Reproduce the estimator exactly, and return the edge positions too."""
    threshold = (tach.max() + tach.min()) / 2.0
    above = tach > threshold
    rising = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    return threshold, rising, len(rising) / (tach.size / FS)


def fft_fundamental(tach: np.ndarray, lo: float = 5.0, hi: float = 70.0
                    ) -> tuple[float, float, float]:
    """Strongest spectral peak of the tach channel inside [lo, hi) Hz.

    Returns:
        (peak_hz, peak_amplitude, ratio_to_next_strongest_distinct_peak)
    """
    x = tach - tach.mean()
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    freqs = np.fft.rfftfreq(x.size, d=1.0 / FS)
    band = (freqs >= lo) & (freqs < hi)
    f_band, a_band = freqs[band], spec[band]

    top = int(np.argmax(a_band))
    peak_hz, peak_amp = float(f_band[top]), float(a_band[top])
    # Exclude a +/- 2 Hz neighbourhood so the "next peak" is a distinct one.
    away = np.abs(f_band - peak_hz) > 2.0
    next_amp = float(np.max(a_band[away])) if away.any() else float("nan")
    return peak_hz, peak_amp, peak_amp / next_amp if next_amp else float("nan")


def describe(path: Path, group: str) -> dict[str, object]:
    """Full diagnostic for one recording."""
    tach = load_tach(path)
    threshold, rising, estimate = edges_and_intervals(tach)
    gaps = np.diff(rising) / FS if rising.size > 1 else np.array([])
    median_gap = float(np.median(gaps)) if gaps.size else float("nan")
    short = int(np.sum(gaps < median_gap / 2)) if gaps.size else 0
    peak_hz, peak_amp, ratio = fft_fundamental(tach)
    name = float(re.fullmatch(r"([0-9.]+)\.csv", path.name).group(1))

    return {
        "group": group,
        "file": path.name,
        "condition": str(path.parent.relative_to(DATA_DIR)),
        "name_hz": name,
        "est_hz": estimate,
        "fft_hz": peak_hz,
        "fft_ratio": ratio,
        "est_over_fft": estimate / peak_hz if peak_hz else float("nan"),
        "edges": int(rising.size),
        "tach_min": float(tach.min()),
        "tach_max": float(tach.max()),
        "tach_mean": float(tach.mean()),
        "threshold": threshold,
        "gap_p10_ms": float(np.percentile(gaps, 10) * 1e3) if gaps.size else np.nan,
        "gap_med_ms": median_gap * 1e3 if gaps.size else np.nan,
        "gap_p90_ms": float(np.percentile(gaps, 90) * 1e3) if gaps.size else np.nan,
        "short_gaps": short,
        "short_pct": 100.0 * short / gaps.size if gaps.size else np.nan,
    }


def main() -> None:
    """Run the whole diagnosis and print it."""
    features = pd.read_csv("features/features.csv")
    # speed_hz_tach, NOT speed_hz: PROTOCOL 3.1 moved the tachometer
    # estimate to its own column and speed_hz now comes from the file
    # name. Reading speed_hz here compares the file name with itself,
    # which makes every recording look in-range and empties the
    # affected group.
    per_rec = features.groupby("recording_id").agg(
        label=("label", "first"), est=("speed_hz_tach", "first"))
    files = sorted(DATA_DIR.rglob("*.csv"))
    per_rec["path"] = [str(f) for f in files]
    per_rec["name_hz"] = [
        float(re.fullmatch(r"([0-9.]+)\.csv", f.name).group(1)) for f in files]
    per_rec["affected"] = (per_rec.est < 10) | (per_rec.est > 65)

    # ------------------------------------------------ filename ground truth
    print("=" * 78)
    print("FILE NAME AS GROUND TRUTH  (all 1,951 recordings)")
    print("=" * 78)
    print(f"  file-name value range : {per_rec.name_hz.min():.3f} - "
          f"{per_rec.name_hz.max():.3f} Hz")
    print(f"  estimator range       : {per_rec.est.min():.1f} - "
          f"{per_rec.est.max():.1f} Hz")
    for subset, rows in (("ALL 1951", per_rec),
                         ("UNAFFECTED 1884", per_rec[~per_rec.affected]),
                         ("AFFECTED 67", per_rec[per_rec.affected])):
        pearson = rows.name_hz.corr(rows.est)
        spearman = rows.name_hz.corr(rows.est, method="spearman")
        err = rows.est - rows.name_hz
        ratio = rows.est / rows.name_hz
        print(f"\n  {subset}")
        print(f"    Pearson r  {pearson:+.4f}   Spearman rho {spearman:+.4f}")
        print(f"    est - name : median {err.median():+.2f} Hz, "
              f"max {err.max():+.1f} Hz")
        print(f"    est / name : median {ratio.median():.3f}, "
              f"min {ratio.min():.3f}, max {ratio.max():.3f}")
        near = np.abs(ratio - np.round(ratio)) < 0.02
        print(f"    est/name within 0.02 of an integer: {int(near.sum())}"
              f"/{len(rows)}")
        if len(rows) < 100:
            counts = pd.Series(np.round(ratio).astype(int)).value_counts()
            print(f"    rounded est/name multiples: {counts.to_dict()}")

    # ------------------------------------------------ per-recording detail
    affected = per_rec[per_rec.affected].nlargest(2, "est")
    overhang = per_rec[per_rec.affected & (per_rec.label == "overhang")].nlargest(1, "est")
    controls = (per_rec[~per_rec.affected & (per_rec.label == "normal")].head(1),
                per_rec[~per_rec.affected & (per_rec.label == "imbalance")].head(1),
                per_rec[~per_rec.affected & (per_rec.label == "underhang")].head(1))

    rows = []
    for _, r in pd.concat([affected, overhang]).iterrows():
        rows.append(describe(Path(r.path), "AFFECTED"))
    for frame in controls:
        for _, r in frame.iterrows():
            rows.append(describe(Path(r.path), "control"))
    detail = pd.DataFrame(rows)

    print()
    print("=" * 78)
    print("PER-RECORDING DIAGNOSTIC")
    print("=" * 78)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(detail[["group", "condition", "file", "name_hz", "est_hz", "fft_hz",
                  "est_over_fft", "edges"]].to_string(index=False))
    print()
    print(detail[["group", "file", "tach_min", "tach_max", "tach_mean",
                  "threshold"]].round(4).to_string(index=False))
    print()
    print(detail[["group", "file", "gap_p10_ms", "gap_med_ms", "gap_p90_ms",
                  "short_gaps", "short_pct", "fft_ratio"]].round(3).to_string(index=False))

    # ------------------------------------------------ amplitude / offset
    print()
    print("=" * 78)
    print("TACH AMPLITUDE AND OFFSET: AFFECTED vs UNAFFECTED (40-file sample each)")
    print("=" * 78)
    rng = np.random.default_rng(0)
    stats = []
    for flag, label in ((True, "AFFECTED"), (False, "unaffected")):
        pool = per_rec[per_rec.affected == flag]
        pick = rng.choice(len(pool), size=min(40, len(pool)), replace=False)
        for path in pool.iloc[sorted(pick)].path:
            tach = load_tach(Path(path))
            stats.append({"group": label, "min": tach.min(), "max": tach.max(),
                          "mean": tach.mean(), "p2p": tach.max() - tach.min(),
                          "std": tach.std()})
    summary = pd.DataFrame(stats).groupby("group").agg(["median", "min", "max"])
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
