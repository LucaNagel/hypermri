"""Study outcomes and clock times use cached results and acquisition metadata."""

import logging
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest
import xarray as xr

from hypermri import Study, load_scan
from hypermri.exceptions import ReconstructionError
from hypermri.loading.experiment import MRIExperiment


@pytest.fixture(autouse=True)
def _restore_package_logging():
    """Study intentionally changes the shared threshold; isolate each test."""
    package_logger = logging.getLogger("hypermri")
    previous_level = package_logger.level
    yield
    package_logger.setLevel(previous_level)


def _strategy_returning(produce):
    """Return a strategy class whose reconstruction returns ``produce()``."""

    class FakeStrategy:
        accepted_options = frozenset({"no_reconstruction"})

        def reconstruct(self, handler, **options):
            return produce()

    return FakeStrategy


@pytest.mark.parametrize("include_other_statuses", [False, True])
@pytest.mark.parametrize(
    ("column", "status", "expected_color"),
    [
        ("Custom Reco", "Success", "#8DEACB"),
        ("Custom Reco", "No raw data", "#7f00ff"),
        ("Custom Reco", "Failed", "#ff0000"),
        ("DICOM", "Yes", "#8DEACB"),
        ("DICOM", "No", "#ff0000"),
    ],
)
def test_status_colors_are_fixed_independently_of_present_outcomes(
    column, status, expected_color, include_other_statuses
):
    """Rendered status colors retain their meaning in single- and mixed-status tables."""
    values = [status]
    if include_other_statuses:
        values += (
            ["Success", "No raw data", "Failed", "Not run"]
            if column == "Custom Reco"
            else ["Yes", "No"]
        )
    study = Study.__new__(Study)
    study.dataframe = pd.DataFrame({column: values})

    html = study.colorize_col(column, opacity=0.3).to_html()

    assert f"background-color: {expected_color}4c;" in html


def test_unattempted_custom_reconstruction_has_no_status_background():
    study = Study.__new__(Study)
    study.dataframe = pd.DataFrame({"Custom Reco": ["Not run"]})

    html = study.colorize_col("Custom Reco").to_html()

    assert "background-color" not in html


@pytest.mark.parametrize(
    ("reconstructions", "outcomes", "has_raw", "expected"),
    [
        ({"raw": xr.DataArray([1.0])}, {}, True, "Success"),
        ({"seq2d": xr.DataArray([1.0])}, {}, True, "Not run"),
        ({"seq2d": xr.DataArray([1.0])}, {"raw": "Failed"}, True, "Failed"),
        ({"raw": xr.DataArray([1.0])}, {"raw": "Failed"}, True, "Failed"),
        ({}, {}, False, "No raw data"),
        ({}, {}, True, "Not run"),
        ({}, {"raw": "Not run"}, True, "Not run"),
    ],
)
def test_study_distinguishes_raw_outcomes_from_scanner_images(
    synthetic_flash_scan, monkeypatch, reconstructions, outcomes, has_raw, expected
):
    """A scanner preview or an old raw cache cannot conceal the latest failure."""
    scan = SimpleNamespace(
        id=7,
        type="<Bruker:FLASH>",
        display_name="synthetic",
        recon_names=[],
        reconstructions=reconstructions,
        _reconstruction_status=outcomes,
        handler=SimpleNamespace(has_raw_data=has_raw),
    )
    monkeypatch.setattr("hypermri.study.load_scan", lambda *args, **kwargs: scan)

    study = Study(synthetic_flash_scan, load_data="lazy", verbose=False, sort=None)

    assert study.dataframe.loc[7, "Custom Reco"] == expected
    assert study.dataframe.columns[-1] == "Measurement Time"


@pytest.mark.parametrize(
    ("acquisition_time", "expected"),
    [
        (
            datetime(2000, 1, 1, 17, 18, 19, tzinfo=timezone(timedelta(hours=5.5))),
            "17:18:19",
        ),
        (None, "MISSING"),
    ],
)
def test_clock_time_preserves_acquisition_timezone_and_has_no_edit_time_fallback(
    synthetic_flash_scan, monkeypatch, acquisition_time, expected
):
    scan = load_scan(synthetic_flash_scan, mode="lazy")
    scan.handler.measurement_time = acquisition_time
    scan.handler.timestamp = datetime(2000, 1, 2, 23, 59, 58)
    monkeypatch.setattr("hypermri.study.load_scan", lambda *args, **kwargs: scan)

    study = Study(synthetic_flash_scan, verbose=False, sort=None)

    assert study.dataframe.iloc[0]["Measurement Time"] == expected


def test_lazy_summary_uses_acq_time_without_loading_samples(
    synthetic_flash_scan, write_paravision_parameters, monkeypatch
):
    """The source is ACQ_time, and optional metadata columns precede clock time."""
    write_paravision_parameters(
        synthetic_flash_scan / "acqp",
        {
            "ACQ_scan_name": "<synthetic>",
            "ACQ_time": "<2000-01-01T12:34:56,789+0530>",
        },
    )

    def forbid_samples(*args, **kwargs):
        raise AssertionError("Study summary must not load or reconstruct samples")

    monkeypatch.setattr(MRIExperiment, "reconstruct", forbid_samples)
    monkeypatch.setattr(
        "hypermri.vendors.bruker.BrukerHandler.load_data", forbid_samples
    )
    monkeypatch.setattr(
        "hypermri.vendors.bruker.BrukerHandler.load_fid_file", forbid_samples
    )

    study = Study(
        synthetic_flash_scan,
        load_data="lazy",
        keywords=["Method"],
        verbose=False,
        sort=None,
    )

    assert study.dataframe.iloc[0]["Measurement Time"] == "12:34:56"
    assert study.dataframe.columns[-1] == "Measurement Time"
    assert "Method" in study.dataframe


def test_display_refreshes_a_failed_raw_attempt_after_successful_retry(
    synthetic_flash_scan, monkeypatch
):
    """A failed automatic attempt stays visible until a direct retry succeeds."""
    (synthetic_flash_scan / "fid").write_bytes(b"\0" * 8)
    scan = load_scan(synthetic_flash_scan, mode="lazy")
    image = xr.DataArray([1.0], dims="acquisition_idx")
    fail_raw = True

    def strategy_for(vendor, sequence, reco_type):
        class FakeStrategy:
            accepted_options = frozenset()

            def reconstruct(self, handler, **options):
                if reco_type == "raw" and fail_raw:
                    raise ValueError("synthetic raw layout mismatch")
                return image

        return FakeStrategy, {}

    monkeypatch.setattr(
        "hypermri.reconstruction.selection.find_reconstruction", strategy_for
    )
    with pytest.raises(ReconstructionError, match="synthetic raw layout mismatch"):
        scan.run_default_reconstructions(load_data="full")
    monkeypatch.setattr("hypermri.study.load_scan", lambda *args, **kwargs: scan)
    study = Study(synthetic_flash_scan, verbose=False, sort=None)
    assert study.dataframe.iloc[0]["Custom Reco"] == "Failed"
    assert "seq2d" in scan.reconstructions

    fail_raw = False
    scan.reconstruct("raw")
    study.add_column("Method")
    displayed = []
    monkeypatch.setattr("IPython.display.display", displayed.append)
    study.display(colorize=False)

    assert study.dataframe.iloc[0]["Custom Reco"] == "Success"
    assert study.dataframe.columns[-1] == "Measurement Time"
    html = displayed[0].to_html()
    assert "Success" in html
    assert html.index(">Method</th>") < html.index(">Measurement Time</th>")


@pytest.mark.parametrize(
    ("options", "expected"),
    [({}, "Failed"), ({"no_reconstruction": True}, "Not run")],
)
def test_none_result_distinguishes_failure_from_explicit_skip(
    synthetic_flash_scan, monkeypatch, options, expected
):
    scan = load_scan(synthetic_flash_scan, mode="lazy")
    monkeypatch.setattr(
        "hypermri.reconstruction.selection.find_reconstruction",
        lambda *args: (_strategy_returning(lambda: None), {}),
    )

    assert scan.reconstruct("raw", **options) is None
    monkeypatch.setattr("hypermri.study.load_scan", lambda *args, **kwargs: scan)
    study = Study(synthetic_flash_scan, verbose=False, sort=None)

    assert study.dataframe.iloc[0]["Custom Reco"] == expected


def test_failed_retry_is_reported_even_when_an_earlier_raw_result_remains_cached(
    synthetic_flash_scan, monkeypatch
):
    scan = load_scan(synthetic_flash_scan, mode="lazy")
    image = xr.DataArray([1.0], dims="acquisition_idx")
    scan.reconstructions["raw"] = image

    def fail():
        raise ValueError("synthetic failure after a cached success")

    monkeypatch.setattr(
        "hypermri.reconstruction.selection.find_reconstruction",
        lambda *args: (_strategy_returning(fail), {}),
    )
    with pytest.raises(ReconstructionError, match="synthetic failure"):
        scan.reconstruct("raw")
    monkeypatch.setattr("hypermri.study.load_scan", lambda *args, **kwargs: scan)
    study = Study(synthetic_flash_scan, verbose=False, sort=None)

    assert scan.reconstructions["raw"] is image
    assert study.dataframe.iloc[0]["Custom Reco"] == "Failed"


def test_column_with_unhashable_values_is_not_coloured():
    study = Study.__new__(Study)
    study.dataframe = pd.DataFrame({"Matrix": [[2, 2], [4, 4]]})

    assert study.colorize_col("Matrix") is None
