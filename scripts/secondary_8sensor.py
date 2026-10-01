"""SECONDARY ANALYSIS: repeat the pipeline with the tachometer INCLUDED.

This is the artefact demonstration described in PROTOCOL.md section 2.2. It is
NOT part of the study's result. The main study excludes the tachometer because
it carries a label-correlated recording artefact; this run reports what happens
when it is left in, so the paper can show concretely that an artefact-driven
channel can dominate a sensor ranking.

8 sensors, 255 subsets, SPEED mode only. Every output is written under
``results/secondary_8sensor_speed/``, ``tables/secondary_8sensor_speed/`` and
``figures/secondary_8sensor_speed/`` so it can never be confused with the
primary results.

The sensor set is widened by overriding ``SENSORS`` to ``ALL_CHANNELS`` in the
pipeline modules. Nothing in common.py is edited: the override is applied here,
at run time, and only for this script.

Usage:
    python scripts/secondary_8sensor.py              # run and analyse
    python scripts/secondary_8sensor.py --analyse-only
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import common  # noqa: E402

LOG = logging.getLogger("secondary")

TAG = "secondary_8sensor_speed"
SPLIT_MODE = "speed"


def load_script(stem: str) -> ModuleType:
    """Import one of the numbered pipeline scripts by path."""
    spec = importlib.util.spec_from_file_location(stem, ROOT / f"{stem}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[stem] = module
    spec.loader.exec_module(module)
    return module


def widen_sensor_set(*modules: ModuleType) -> None:
    """Point SENSORS at all 8 channels, in common and in each pipeline module.

    Each module did ``from common import SENSORS``, so it holds its own binding
    and both must be rebound. ``N_SUBSETS`` in the analyser is computed at
    import time and is recomputed here.

    Args:
        *modules: Pipeline modules to patch.
    """
    common.SENSORS = list(common.ALL_CHANNELS)
    for module in modules:
        if hasattr(module, "SENSORS"):
            module.SENSORS = list(common.ALL_CHANNELS)
        if hasattr(module, "N_SUBSETS"):
            module.N_SUBSETS = 2 ** len(common.ALL_CHANNELS) - 1
    LOG.info("Sensor set widened to %d channels: %s",
             len(common.SENSORS), common.SENSORS)


def main(argv: list[str] | None = None) -> None:
    """Run and analyse the 8-sensor secondary configuration."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--analyse-only", action="store_true",
                    help="skip the experiment and analyse existing results")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    runner = load_script("03_run_experiments")
    analyzer = load_script("04_analyze")
    widen_sensor_set(runner, analyzer)

    expected = 2 ** len(common.SENSORS) - 1
    if len(runner.all_subsets()) != expected:
        raise ValueError(
            f"expected {expected} subsets, got {len(runner.all_subsets())}")
    LOG.info("SECONDARY ANALYSIS: %d sensors, %d subsets, split=%s",
             len(common.SENSORS), expected, SPLIT_MODE)

    results = common.RESULTS_DIR / TAG
    df = common.load_features()
    y = common.encode_labels(df)

    if not args.analyse_only:
        runner.run_experiments(df, y, results, SPLIT_MODE, common.N_TREES)

    # The analyser writes under <FIGURES_DIR>/<mode>; point it at the tagged
    # directory so nothing can land beside the primary results.
    analyzer.analyse(results, TAG, expect_full_run=True)
    LOG.info("Secondary results  -> %s", results)
    LOG.info("Secondary tables   -> %s", common.TABLES_DIR / TAG)
    LOG.info("Secondary figures  -> %s", common.FIGURES_DIR / TAG)


if __name__ == "__main__":
    main()
