"""Shared pytest fixtures.

The pipeline scripts are named ``01_``..``04_``, which are not importable
module names, so they are loaded by path with ``importlib``.

The synthetic dataset and its feature table are built once per test session and
reused, because extraction is the slowest part of the suite.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from common import CLASS_ORDER, encode_labels  # noqa: E402
from make_synthetic_data import (  # noqa: E402
    HARD_AMPLITUDE_SCALE,
    SPEED_N_PER_CLASS,
    make_synthetic_dataset,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))


def pytest_configure(config: pytest.Config) -> None:
    """Register the markers used by the suite."""
    config.addinivalue_line(
        "markers", "realdata: needs data/mafaulda/; skipped when it is absent")
    config.addinivalue_line(
        "markers", "slow: full-size forests or a whole pipeline run")


def load_script(stem: str) -> ModuleType:
    """Import one of the numbered pipeline scripts by file name.

    Args:
        stem: File stem, for example ``"01_extract_features"``.

    Returns:
        The imported module.

    Raises:
        FileNotFoundError: If the script does not exist.
    """
    path = REPO_ROOT / f"{stem}.py"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")
    spec = importlib.util.spec_from_file_location(stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[stem] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """The repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def extract() -> ModuleType:
    """The ``01_extract_features`` module."""
    return load_script("01_extract_features")


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A seeded synthetic dataset laid out like MaFaulDa, in a temp directory."""
    root = tmp_path_factory.mktemp("mafaulda_synth")
    return make_synthetic_dataset(root, seed=0)


@pytest.fixture(scope="session")
def synthetic_features(extract: ModuleType, synthetic_root: Path) -> pd.DataFrame:
    """The feature table extracted from the synthetic dataset.

    Single-process extraction keeps the fixture cheap and avoids shipping a
    dynamically loaded module to worker processes.
    """
    return extract.extract_features(synthetic_root, jobs=1)


@pytest.fixture(scope="session")
def hard_synthetic_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A synthetic dataset with weakened fault signatures.

    The default fixture is separable to macro-F1 = 1.0, which leaves the normal
    analysis path untested: forward selection becomes arbitrary, percentile
    ranks carry no information and every Wilcoxon call takes the all-zero
    differences branch.  This fixture lands around 0.86 so those paths run for
    real.  Same seed, same layout, only the signature amplitude differs.
    """
    root = tmp_path_factory.mktemp("mafaulda_hard")
    return make_synthetic_dataset(root, seed=0,
                                  amplitude_scale=HARD_AMPLITUDE_SCALE)


@pytest.fixture(scope="session")
def hard_synthetic_features(extract: ModuleType,
                            hard_synthetic_root: Path) -> pd.DataFrame:
    """The feature table extracted from the harder synthetic dataset."""
    return extract.extract_features(hard_synthetic_root, jobs=1)


@pytest.fixture(scope="session")
def labelled(synthetic_features: pd.DataFrame) -> tuple[pd.DataFrame, "np.ndarray"]:
    """The ceiling fixture's feature table plus its encoded labels."""
    df = synthetic_features.copy()
    df["label"] = pd.Categorical(df["label"], categories=CLASS_ORDER)
    return df, encode_labels(df)


@pytest.fixture(scope="session")
def hard_labelled(hard_synthetic_features: pd.DataFrame
                  ) -> tuple[pd.DataFrame, "np.ndarray"]:
    """The hard fixture's feature table plus its encoded labels."""
    df = hard_synthetic_features.copy()
    df["label"] = pd.Categorical(df["label"], categories=CLASS_ORDER)
    return df, encode_labels(df)


@pytest.fixture(scope="session")
def speed_synthetic_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A synthetic dataset large enough for the speed split mode to be valid.

    Speed mode holds out whole speed bands, so every band needs enough
    recordings of every class.  At the default 10-14 recordings per class a
    band holds roughly one recording per class and NO candidate q gives full
    class coverage, so choose_speed_q correctly refuses to split.  That is the
    right behaviour but it leaves speed mode untested, hence this larger
    fixture.  See the Phase 3 report for the measured sizes.
    """
    root = tmp_path_factory.mktemp("mafaulda_speed")
    return make_synthetic_dataset(root, seed=0, n_per_class=SPEED_N_PER_CLASS)


@pytest.fixture(scope="session")
def speed_labelled(extract: ModuleType,
                   speed_synthetic_root: Path) -> tuple[pd.DataFrame, "np.ndarray"]:
    """Feature table and labels for the speed-mode fixture."""
    df = extract.extract_features(speed_synthetic_root, jobs=1)
    df["label"] = pd.Categorical(df["label"], categories=CLASS_ORDER)
    return df, encode_labels(df)
