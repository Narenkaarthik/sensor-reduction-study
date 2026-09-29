"""Build a small synthetic dataset that mirrors the MaFaulDa layout.

The real dataset is far too large to run tests against, so the test suite uses
a miniature stand-in with the same *structure*: class folders at the top level,
deeper subfolders for fault subtype and severity, and header-less CSV files
with 8 columns in the canonical ``ALL_CHANNELS`` order.

Fault signatures are injected into known channels so that correct behaviour is
predictable.  Two channels, ``ub_ax`` and ``ob_rad``, receive noise only and
must therefore rank at the bottom of any sensor-importance measure; the Level E
tests rely on this.

Channel map (the signature amplitude per class):

    channel   normal  imbalance  horiz-mis  vert-mis  underhang  overhang
    tach      pulse train at the rotation frequency; identical for every class
    ub_ax     -       -          -          -         -          -
    ub_rad    -       1x strong  2x weak    2x strong bursts     -
    ub_tan    -       1x weak    2x strong  2x weak   bursts     -
    ob_ax     -       -          -          -         -          bursts
    ob_rad    -       -          -          -         -          -
    ob_tan    -       1x strong  2x weak    -         -          bursts
    mic       -       1x weak    2x weak    2x weak   burst weak burst weak

"1x" and "2x" are sinusoids at one and two times the rotation frequency;
"bursts" are decaying high-frequency impulses at a bearing-fault-like rate.

Everything is seeded, so the same seed always yields byte-identical CSVs.

Usage (normally called from ``conftest.py``, but runnable for inspection):
    python tests/make_synthetic_data.py --out /tmp/synthetic --seed 0
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import ALL_CHANNELS, CLASS_ORDER, FS  # noqa: E402

LOG = logging.getLogger(__name__)

# Keep the synthetic recordings short: 0.2 s at 50 kHz.
SYNTHETIC_DURATION_S: float = 0.2
SYNTHETIC_N_SAMPLES: int = int(FS * SYNTHETIC_DURATION_S)

# Recordings generated per class (drawn per class from this inclusive range).
N_PER_CLASS: tuple[int, int] = (10, 14)

# Speed mode holds out whole speed bands, so each band needs enough
# recordings of every class.  At N_PER_CLASS a band holds about one
# recording per class and no candidate q achieves full class coverage, so
# the speed-mode tests need a larger fixture.  Measured: 26-32 per class is
# the smallest range at which q=10 covers every class in every test fold.
SPEED_N_PER_CLASS: tuple[int, int] = (26, 32)

# Rotation speeds, in Hz, matching the real dataset's approximate range.
SPEED_RANGE_HZ: tuple[float, float] = (12.0, 60.0)

# Signature amplitudes, relative to a noise floor of sigma = 1.
AMP_STRONG: float = 3.0
AMP_WEAK: float = 1.2
AMP_BURST: float = 4.0
AMP_BURST_WEAK: float = 1.5

# Two fixtures are needed.  At full amplitude every class is separable and the
# all-sensor macro-F1 sits at 1.0, which exercises the degenerate code paths
# (arbitrary forward selection, meaningless percentile ranks, all-zero
# Wilcoxon differences).  The "hard" fixture weakens every fault signature
# against the same noise floor so that the normal, non-degenerate path is
# exercised too.  The value below is calibrated, not guessed: at 0.20 the
# all-sensor macro-F1 is 0.863 with the best single sensor at 0.731, so the
# sensor-count curve has real room in it.  tests/test_sanity.py asserts the
# score stays inside 0.80-0.90 so that a change here cannot pass unnoticed.
HARD_AMPLITUDE_SCALE: float = 0.20

# Bearing-fault burst parameters.
BURST_RATE_RATIO: float = 4.7      # burst repetitions per shaft revolution
BURST_CARRIER_HZ: float = 4_000.0
BURST_DECAY_S: float = 0.0015

# Subfolder names per class, mirroring MaFaulDa's subtype/severity nesting.
CONDITION_SUBFOLDERS: dict[str, list[str]] = {
    "normal": [""],
    "imbalance": ["6g", "20g"],
    "horizontal-misalignment": ["0.5mm", "1.5mm"],
    "vertical-misalignment": ["0.51mm", "1.27mm"],
    "underhang": ["ball_fault/0g", "outer_race/20g"],
    "overhang": ["ball_fault/0g", "cage_fault/20g"],
}


def _tach_pulses(t: NDArray[np.float64], speed_hz: float) -> NDArray[np.float64]:
    """Generate a tachometer pulse train at the given rotation frequency.

    The train is phase-shifted by half a cycle so that the signal is LOW at
    t = 0.  Every revolution therefore produces exactly one below-to-above
    crossing inside the recording, which makes the rising-edge speed estimate
    exact rather than off by one.

    Args:
        t: Time vector in seconds.
        speed_hz: Rotation frequency in Hz.

    Returns:
        A 0/5 V pulse train with a 10 percent duty cycle.
    """
    phase = (speed_hz * t + 0.5) % 1.0
    return 5.0 * (phase < 0.1).astype(np.float64)


def _bursts(
    t: NDArray[np.float64],
    speed_hz: float,
    amplitude: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Generate decaying high-frequency impulses, as a bearing fault produces.

    Args:
        t: Time vector in seconds.
        speed_hz: Rotation frequency in Hz; the burst rate is a fixed,
            non-integer multiple of it.
        amplitude: Peak amplitude of each burst.
        rng: Seeded generator, used only to jitter the burst phase.

    Returns:
        The burst signal, same length as ``t``.
    """
    out = np.zeros_like(t)
    period = 1.0 / (BURST_RATE_RATIO * speed_hz)
    phase = rng.uniform(0.0, period)
    for start in np.arange(phase, t[-1], period):
        dt = t - start
        active = dt >= 0.0
        out[active] += amplitude * np.exp(-dt[active] / BURST_DECAY_S) * np.sin(
            2.0 * np.pi * BURST_CARRIER_HZ * dt[active]
        )
    return out


def _signature(
    label: str,
    t: NDArray[np.float64],
    speed_hz: float,
    rng: np.random.Generator,
    amplitude_scale: float = 1.0,
) -> dict[str, NDArray[np.float64]]:
    """Build the per-channel fault signature for one class.

    Args:
        label: One of ``CLASS_ORDER``.
        t: Time vector in seconds.
        speed_hz: Rotation frequency in Hz.
        rng: Seeded generator for phases and burst jitter.
        amplitude_scale: Multiplies every signature amplitude.  Values below 1
            make the classification task harder against the fixed noise floor.

    Returns:
        A mapping from sensor ID to the signal to add on top of the noise
        floor.  Channels absent from the mapping stay pure noise.

    Raises:
        ValueError: If ``label`` is not a known class.
    """
    if label not in CLASS_ORDER:
        raise ValueError(f"Unknown class {label!r}; expected one of {CLASS_ORDER}.")

    one_x = 2.0 * np.pi * speed_hz * t + rng.uniform(0, 2 * np.pi)
    two_x = 2.0 * np.pi * 2.0 * speed_hz * t + rng.uniform(0, 2 * np.pi)
    strong = AMP_STRONG * amplitude_scale
    weak = AMP_WEAK * amplitude_scale
    burst = AMP_BURST * amplitude_scale
    burst_weak = AMP_BURST_WEAK * amplitude_scale

    if label == "normal":
        return {}
    if label == "imbalance":
        return {
            "ub_rad": strong * np.sin(one_x),
            "ub_tan": weak * np.sin(one_x),
            "ob_tan": strong * np.sin(one_x),
            "mic": weak * np.sin(one_x),
        }
    if label == "horizontal-misalignment":
        return {
            "ub_rad": weak * np.sin(two_x),
            "ub_tan": strong * np.sin(two_x),
            "ob_tan": weak * np.sin(two_x),
            "mic": weak * np.sin(two_x),
        }
    if label == "vertical-misalignment":
        return {
            "ub_rad": strong * np.sin(two_x),
            "ub_tan": weak * np.sin(two_x),
            "mic": weak * np.sin(two_x),
        }
    if label == "underhang":
        return {
            "ub_rad": _bursts(t, speed_hz, burst, rng),
            "ub_tan": _bursts(t, speed_hz, burst, rng),
            "mic": _bursts(t, speed_hz, burst_weak, rng),
        }
    # overhang
    return {
        "ob_ax": _bursts(t, speed_hz, burst, rng),
        "ob_tan": _bursts(t, speed_hz, burst, rng),
        "mic": _bursts(t, speed_hz, burst_weak, rng),
    }


def make_recording(
    label: str,
    speed_hz: float,
    rng: np.random.Generator,
    n_samples: int = SYNTHETIC_N_SAMPLES,
    amplitude_scale: float = 1.0,
) -> NDArray[np.float64]:
    """Generate one synthetic recording.

    Args:
        label: Class name, one of ``CLASS_ORDER``.
        speed_hz: Rotation frequency in Hz.
        rng: Seeded generator.
        n_samples: Number of rows.
        amplitude_scale: Multiplies every fault signature amplitude.

    Returns:
        An ``(n_samples, 8)`` array with columns in ``ALL_CHANNELS`` order.
    """
    t = np.arange(n_samples, dtype=np.float64) / FS
    sig = rng.normal(0.0, 1.0, size=(n_samples, len(ALL_CHANNELS)))
    sig[:, ALL_CHANNELS.index("tach")] = _tach_pulses(t, speed_hz) + 0.02 * rng.normal(
        size=n_samples
    )
    signature = _signature(label, t, speed_hz, rng, amplitude_scale)
    for sensor, component in signature.items():
        sig[:, ALL_CHANNELS.index(sensor)] += component
    return sig


def make_synthetic_dataset(
    root: Path,
    seed: int = 0,
    n_per_class: tuple[int, int] = N_PER_CLASS,
    n_samples: int = SYNTHETIC_N_SAMPLES,
    amplitude_scale: float = 1.0,
) -> Path:
    """Write a complete synthetic dataset under ``root``.

    Args:
        root: Directory to fill; created if absent.  Plays the role of
            ``DATA_DIR``.
        seed: Master seed; the same seed reproduces the dataset exactly.
        n_per_class: Inclusive range for the number of recordings per class.
        n_samples: Rows per recording.
        amplitude_scale: Multiplies every fault signature amplitude; pass
            ``HARD_AMPLITUDE_SCALE`` for the non-degenerate fixture.

    Returns:
        ``root``, for convenience.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    total = 0
    for label in CLASS_ORDER:
        n_rec = int(rng.integers(n_per_class[0], n_per_class[1] + 1))
        subfolders = CONDITION_SUBFOLDERS[label]
        for i in range(n_rec):
            speed = float(rng.uniform(*SPEED_RANGE_HZ))
            sig = make_recording(label, speed, rng, n_samples, amplitude_scale)
            folder = root / label / subfolders[i % len(subfolders)]
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{speed:.4f}.csv"
            np.savetxt(path, sig, delimiter=",", fmt="%.6f")
            total += 1
        LOG.info("%-24s %2d recordings", label, n_rec)
    LOG.info("Synthetic dataset: %d recordings of %d samples "
             "(amplitude_scale=%.3g) -> %s", total, n_samples,
             amplitude_scale, root)
    return root


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point, for generating a dataset to inspect by hand."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True,
                    help="directory to write the synthetic dataset into")
    ap.add_argument("--seed", type=int, default=0, help="master seed")
    ap.add_argument("--amplitude-scale", type=float, default=1.0,
                    help=f"fault signature scale; {HARD_AMPLITUDE_SCALE} is the "
                         "calibrated hard fixture")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    make_synthetic_dataset(args.out, seed=args.seed,
                           amplitude_scale=args.amplitude_scale)


if __name__ == "__main__":
    main()
