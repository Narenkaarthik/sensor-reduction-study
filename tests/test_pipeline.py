"""Level D: integration tests.

Phase 2 covers the determinism of feature extraction, which every later result
depends on: if the feature table is not reproducible, nothing downstream can
be.  The end-to-end, resume, completeness and lookup tests arrive in Phase 3
with the experiment runner.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from make_synthetic_data import make_synthetic_dataset


def _md5(path: Path) -> str:
    """Hex MD5 of a file's bytes."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def test_extraction_is_deterministic_across_runs(extract: ModuleType,
                                                 synthetic_root: Path,
                                                 tmp_path: Path) -> None:
    """Two runs over the same data produce byte-identical feature files."""
    hashes = []
    for i in range(2):
        out = tmp_path / f"run{i}.csv"
        extract.extract_features(synthetic_root, jobs=1).to_csv(out, index=False)
        hashes.append(_md5(out))
    assert hashes[0] == hashes[1], "feature extraction is not reproducible"


def test_extraction_is_independent_of_worker_count(extract: ModuleType,
                                                   synthetic_root: Path,
                                                   tmp_path: Path) -> None:
    """--jobs must not change a single value.

    Parallelism is a speed decision, not an experimental one.  recording_id is
    assigned from the sorted file list before any work is dispatched, so the
    worker count cannot reorder or renumber anything.
    """
    serial = tmp_path / "serial.csv"
    parallel = tmp_path / "parallel.csv"
    extract.extract_features(synthetic_root, jobs=1).to_csv(serial, index=False)
    extract.extract_features(synthetic_root, jobs=2).to_csv(parallel, index=False)
    assert _md5(serial) == _md5(parallel), "worker count changed the features"


def test_recording_ids_are_stable_across_runs(extract: ModuleType,
                                              synthetic_root: Path) -> None:
    """The same file always receives the same recording_id.

    The cross-validation groups are built from recording_id, so an unstable
    assignment would silently change every split.
    """
    first = extract.extract_features(synthetic_root, jobs=1)
    second = extract.extract_features(synthetic_root, jobs=2)
    key = ["condition_path", "speed_hz", "window"]
    a = first.sort_values(key)["recording_id"].to_numpy()
    b = second.sort_values(key)["recording_id"].to_numpy()
    assert (a == b).all()


def test_synthetic_dataset_generation_is_seeded(tmp_path: Path) -> None:
    """The same seed regenerates byte-identical CSV files."""
    first = make_synthetic_dataset(tmp_path / "a", seed=3)
    second = make_synthetic_dataset(tmp_path / "b", seed=3)
    files_a = sorted(p.relative_to(first) for p in first.rglob("*.csv"))
    files_b = sorted(p.relative_to(second) for p in second.rglob("*.csv"))
    assert files_a == files_b and files_a
    for rel in files_a:
        assert _md5(first / rel) == _md5(second / rel), rel


def test_hard_fixture_differs_from_ceiling_fixture(synthetic_features: pd.DataFrame,
                                                   hard_synthetic_features
                                                   ) -> None:
    """The two fixtures share a layout but carry different signal strengths."""
    assert synthetic_features.shape == hard_synthetic_features.shape
    assert list(synthetic_features.columns) == list(hard_synthetic_features.columns)
    assert not synthetic_features["ub_rad__rms"].equals(
        hard_synthetic_features["ub_rad__rms"])


@pytest.mark.slow
def test_extractor_runs_as_a_script(repo_root: Path, tmp_path: Path) -> None:
    """01_extract_features.py works when invoked the way the README says.

    The in-process tests use jobs=1; this one exercises the real entry point
    with joblib's default parallelism, where each worker re-imports common in a
    fresh process.
    """
    work = tmp_path / "repo"
    (work / "tests").mkdir(parents=True)
    for name in ["common.py", "01_extract_features.py"]:
        (work / name).write_bytes((repo_root / name).read_bytes())
    make_synthetic_dataset(work / "data" / "mafaulda", seed=0)

    result = subprocess.run(
        [sys.executable, "01_extract_features.py"],
        cwd=work, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr[-3000:]

    out = work / "features" / "features.csv"
    assert out.exists()
    df = pd.read_csv(out)
    assert df.shape[1] == 150
    assert "Total is" in result.stderr, "the incomplete-dataset warning did not fire"


# ---------------------------------------------------------------------------
# Provenance: the dirty-tree check must track SOURCE, not generated output
# ---------------------------------------------------------------------------
def test_dirty_check_ignores_generated_output(repo_root: Path) -> None:
    """Writing into results/ must NOT mark the tree dirty.

    results/, tables/ and figures/ are tracked, so a run in progress always has
    uncommitted output of its own. Counting that would make the flag fire on
    every run and carry no information.
    """
    from common import SOURCE_PATHS, git_state

    if git_state().get("dirty"):
        pytest.skip("source tree is already dirty; cannot test the clean case")

    probe = repo_root / "results" / "_dirty_probe.tmp"
    probe.parent.mkdir(parents=True, exist_ok=True)
    probe.write_text("generated output, not source\n")
    try:
        assert git_state()["dirty"] is False, (
            "a file written into results/ marked the tree dirty")
    finally:
        probe.unlink(missing_ok=True)

    assert "results" not in SOURCE_PATHS
    assert "tables" not in SOURCE_PATHS
    assert "figures" not in SOURCE_PATHS


def test_dirty_check_still_catches_source_changes(repo_root: Path) -> None:
    """A change to a source file MUST still mark the tree dirty."""
    from common import git_state

    if git_state().get("dirty"):
        pytest.skip("source tree is already dirty; cannot test the clean case")

    probe = repo_root / "tests" / "_dirty_probe_source.py"
    probe.write_text("# temporary probe\n")
    try:
        assert git_state()["dirty"] is True, (
            "a new file under tests/ did not mark the tree dirty")
    finally:
        probe.unlink(missing_ok=True)

    assert git_state()["dirty"] is False


def test_run_metadata_records_the_dirty_scope() -> None:
    """The metadata says which paths the dirty flag was computed over."""
    from common import SOURCE_PATHS, run_metadata

    meta = run_metadata("probe.py")
    assert meta["git_dirty_scope"] == SOURCE_PATHS
    assert "common.py" in meta["git_dirty_scope"]
    assert "tests" in meta["git_dirty_scope"]

def test_run_metadata_contains_no_local_absolute_paths() -> None:
    """Metadata is committed and published, so it must not name this machine.

    The six tracked ``run_metadata.json`` files carried the operating user's
    account name in ``python_executable``, ``results_dir`` and a nested
    ``data_dir`` until 2026-11-02. Paths inside the repository are kept, but
    rewritten relative to the root, so the record is identical on every
    machine and leaks nothing.
    """
    import json

    from common import ROOT, run_metadata

    meta = run_metadata(
        "probe.py",
        results_dir=str(ROOT / "results" / "speed"),
        inputs={"features": {"data_dir": str(ROOT / "data" / "mafaulda"),
                             "python_executable": str(ROOT / ".venv" / "x")}},
    )
    blob = json.dumps(meta)

    assert str(ROOT) not in blob, "the repository root survived in the record"
    assert ROOT.as_posix() not in blob
    assert str(Path.home()) not in blob, "a home directory survived"
    assert ROOT.name + "/" not in blob, (
        "a path is still anchored at the repository name, not relative to it")

    # relativised, not deleted: the provenance is still readable
    assert meta["results_dir"] == "results/speed"
    assert meta["inputs"]["features"]["data_dir"] == "data/mafaulda"
    assert meta["inputs"]["features"]["python_executable"] == ".venv/x"
    assert not Path(meta["python_executable"]).is_absolute()
