"""Failures of automatic loading steps reach the caller instead of a log line.

The scans are the synthetic two-slice FLASH series of ``tests/conftest.py``;
reconstruction strategies are replaced by stubs that raise, so no numerical
result is involved.
"""

from __future__ import annotations

import logging
import shutil
from types import SimpleNamespace

import pytest

import hypermri.study
from hypermri import Study, load_scan
from hypermri.exceptions import ReconstructionError
from hypermri.loading.experiment import MRIExperiment
from hypermri.vendors.bruker import BrukerHandler


@pytest.fixture(autouse=True)
def _restore_package_logging():
    """Study changes the shared logging threshold; isolate each test."""
    package_logger = logging.getLogger("hypermri")
    previous_level = package_logger.level
    yield
    package_logger.setLevel(previous_level)


class _FailingStrategy:
    """Strategy stub whose reconstruction always raises."""

    accepted_options = frozenset()

    def reconstruct(self, handler, **options):
        raise ValueError("synthetic strategy failure")


def _failing_strategy(vendor, sequence, kind):
    return _FailingStrategy, {}


@pytest.mark.parametrize("resolve_sequence", [True, False])
def test_failing_default_reconstruction_aborts_a_direct_load(
    synthetic_flash_scan, monkeypatch, resolve_sequence
):
    monkeypatch.setattr(
        "hypermri.reconstruction.selection.find_reconstruction", _failing_strategy
    )

    with pytest.raises(ReconstructionError, match="synthetic strategy failure"):
        load_scan(
            synthetic_flash_scan, mode="minimal", resolve_sequence=resolve_sequence
        )


def test_lazy_load_runs_no_reconstruction_and_does_not_fail(
    synthetic_flash_scan, monkeypatch
):
    monkeypatch.setattr(
        "hypermri.reconstruction.selection.find_reconstruction", _failing_strategy
    )

    scan = load_scan(synthetic_flash_scan, mode="lazy")

    assert scan.reconstructions == {}


def test_failing_parameter_read_aborts_experiment_construction(
    synthetic_flash_scan, monkeypatch
):
    def fail(self):
        raise RuntimeError("synthetic parameter failure")

    monkeypatch.setattr(BrukerHandler, "get_parameters", fail)

    with pytest.raises(RuntimeError, match="synthetic parameter failure"):
        MRIExperiment(str(synthetic_flash_scan), load_data=False)


def test_study_records_a_failed_scan_and_loads_the_others(
    synthetic_flash_scan, tmp_path, monkeypatch
):
    study_dir = tmp_path / "study"
    good = shutil.copytree(synthetic_flash_scan, study_dir / "1")
    bad = shutil.copytree(synthetic_flash_scan, study_dir / "2")
    real_load_scan = hypermri.study.load_scan
    failure = ReconstructionError("synthetic strategy failure")

    def load_or_fail(path, **kwargs):
        if str(path) == str(bad):
            raise failure
        return real_load_scan(path, **kwargs)

    monkeypatch.setattr(hypermri.study, "load_scan", load_or_fail)

    study = Study(study_dir, verbose=False, sort=None)

    assert study.load_errors == {str(bad): failure}
    loaded = [str(scan.path) for scan in study.dataframe["ScanObject"]]
    assert loaded == [str(good)]


def test_study_without_failures_has_no_load_errors(synthetic_flash_scan):
    study = Study(synthetic_flash_scan, verbose=False, sort=None)

    assert study.load_errors == {}
    assert len(study.dataframe) == 1
