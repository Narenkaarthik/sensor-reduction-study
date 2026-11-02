"""Three views of each sensor in one figure: the RQ3 argument, made visible.

Marginal permutation importance alone cannot distinguish a sensor that carries
no information from one that is merely redundant (PROTOCOL.md section 8.2).
The repository already reports the three quantities that together DO separate
those cases, in ``supp_sensor_profile.csv``:

  marginal importance   the drop when the sensor is permuted, given the others
  standalone macro-F1   what the sensor achieves on its own
  mean |rho|            how strongly it correlates with the other sensors

Plotted side by side in a shared sensor order, the separation is immediate: a
sensor low in the first panel but high in the second is redundant, not
uninformative, and the third panel says which of its neighbours covers it.

This figure replaces the separate importance-bar and correlation-matrix
figures in the paper, which carried the first and third panels apart from each
other and left the reader to join them.

READ-ONLY: it plots an existing generated table and fits nothing.

Usage:
    python scripts/fig_sensor_profile.py --split speed
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import common  # noqa: E402

LOG = logging.getLogger("fig_profile")

plt.rcParams.update({
    "font.size": 8,
    "font.family": "serif",
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "savefig.bbox": "tight",
})
COL_W = 3.5
DPI = 300
STEM = "fig7_sensor_profile"

PANELS = [
    ("perm_importance_mean", "perm_importance_sd",
     "Marginal\nimportance", "0.55"),
    ("single_sensor_macro_f1", "single_sensor_sd",
     "Standalone\nmacro-F1", "0.30"),
    ("mean_abs_rho_to_others", None,
     r"Mean $|\rho|$ to" "\nother sensors", "0.75"),
]

PROTOCOL_LABEL = {
    "speed": "held-out speed bands (PRIMARY protocol)",
    "recording": "held-out recordings (SECONDARY robustness analysis)",
}


def build(profile: pd.DataFrame, split_mode: str, out_dir: Path) -> Path:
    """Draw and save the three-panel sensor profile.

    Args:
        profile: ``supp_sensor_profile.csv`` as a frame, importance-ordered.
        split_mode: Which split mode produced it.
        out_dir: Figure directory for that mode.

    Returns:
        The PDF path written.
    """
    sensors = profile["sensor"].tolist()
    y = range(len(sensors))

    fig, axes = plt.subplots(1, 3, figsize=(COL_W, 2.5), sharey=True)
    for ax, (col, err, title, shade) in zip(axes, PANELS):
        xerr = profile[err] if err else None
        ax.barh(y, profile[col], xerr=xerr, color=shade,
                edgecolor="black", linewidth=0.5, height=0.7,
                error_kw={"elinewidth": 0.7, "capsize": 1.5})
        ax.set_title(title, fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.grid(axis="x", linewidth=0.4, alpha=0.5)
        ax.set_axisbelow(True)

    axes[0].set_yticks(list(y))
    axes[0].set_yticklabels(sensors, fontsize=7)
    axes[0].invert_yaxis()
    # the standalone panel is a score, so anchoring it at 0 would waste the
    # axis; start just below the weakest sensor instead
    axes[1].set_xlim(left=max(0.0, profile["single_sensor_macro_f1"].min() - 0.1))
    fig.tight_layout(pad=0.4, w_pad=0.6)

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{STEM}.pdf"
    fig.savefig(pdf)
    fig.savefig(out_dir / f"{STEM}.png", dpi=DPI)
    plt.close(fig)
    return pdf


def write_caption(profile: pd.DataFrame, split_mode: str, path: Path) -> None:
    """Generate the caption, so it cannot drift from what was plotted.

    Args:
        profile: The profile frame, importance-ordered.
        split_mode: Which split mode produced it.
        path: Destination ``.tex`` file.
    """
    weakest = profile.iloc[-1]
    rank = (profile["single_sensor_macro_f1"]
            .rank(ascending=False)[profile.index[-1]])
    ordinal = {1: "best", 2: "second best", 3: "third best"}.get(
        int(rank), f"{int(rank)}th best")

    caption = (
        f"Evaluation protocol: {PROTOCOL_LABEL[split_mode]}. "
        "Three views of each sensor, ordered by marginal permutation "
        "importance. LEFT: the drop in inner-validation macro-F1 when all 18 "
        "features of that sensor are permuted together, mean and standard "
        "deviation over the 10 splits; this is a MARGINAL contribution, "
        "measured with every other sensor present. CENTRE: the macro-F1 that "
        "sensor reaches on its own. RIGHT: its mean absolute Spearman "
        "correlation to the other six, over all 18 matched features. "
        "Reading the panels together separates the two cases a marginal "
        "measure cannot: "
        f"\\texttt{{{weakest['sensor'].replace('_', chr(92) + '_')}}} has the "
        "lowest marginal contribution of the set "
        f"({weakest['perm_importance_mean']:.4f}) yet is the {ordinal} single "
        f"sensor ({weakest['single_sensor_macro_f1']:.4f}), so its low score "
        "reflects redundancy rather than an absence of information. The 10 "
        "splits come from 2 repeats of 5-fold cross-validation over the same "
        "recordings and are therefore not mutually independent."
    )
    path.write_text(
        "% Generated by scripts/fig_sensor_profile.py. Do not edit by hand.\n"
        "\\newcommand{\\FigProfileCaption}{" + caption + "}\n")


def main(argv: list[str] | None = None) -> None:
    """Build the figure and its caption for one split mode."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="speed", choices=["speed", "recording"])
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    table = common.TABLES_DIR / args.split / "supp_sensor_profile.csv"
    profile = pd.read_csv(table)
    # supplementary_tables.py already writes it in importance order; sort
    # anyway so this figure does not depend on that staying true
    profile = profile.sort_values("perm_importance_mean", ascending=False)
    profile = profile.reset_index(drop=True)

    out_dir = common.FIGURES_DIR / args.split
    pdf = build(profile, args.split, out_dir)
    write_caption(profile, args.split, out_dir / "caption_sensor_profile.tex")
    LOG.info("Figure  -> %s", pdf)
    LOG.info("Caption -> %s", out_dir / "caption_sensor_profile.tex")


if __name__ == "__main__":
    main()
