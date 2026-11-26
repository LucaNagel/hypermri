"""Attribute access of an experiment: one parameter store and no silent delegation.

The scan is the synthetic ParaVision FLASH series of the ``synthetic_flash_scan``
fixture.  Values are compared by identity or exact equality because no
floating-point operation is involved.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hypermri import Study, load_scan
from hypermri.loading.experiment import MRIExperiment
from hypermri.sequences.FLASH import FLASH
from hypermri.utils.coordinates.image_shifts import (
    determine_FOV_shift,
    get_orientation_flip_multipliers,
)
from hypermri.utils.io.dicom_export import _normalise_reference_datasets
from hypermri.vendors import get_vendor_handler
from hypermri.visualization.data_sources import DataSourceCatalog


def _scans(path):
    """Return the scan of ``load_scan`` and of direct handler construction."""
    direct = FLASH(get_vendor_handler(str(path)))
    return {"load_scan": load_scan(path, mode="lazy"), "handler": direct}


def test_experiment_keeps_no_attribute_copies_of_the_parameters(
    synthetic_flash_scan,
):
    for route, scan in _scans(synthetic_flash_scan).items():
        keys = scan.parameter_keys()

        assert not keys & set(vars(scan)), route
        assert scan.params
        # The handler parsers assign and read the attributes, so it keeps them.
        assert keys <= set(vars(scan.handler)), route


def test_parameter_attributes_read_the_one_parameter_store(synthetic_flash_scan):
    scan = load_scan(synthetic_flash_scan, mode="lazy")
    missing = "slice_gap_mm"
    scan.params.pop(missing, None)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for key in scan.parameter_keys() - {missing}:
            value = getattr(scan, key)
            expected = scan.params.get(key)
            if isinstance(expected, np.ndarray):
                np.testing.assert_array_equal(value, expected, err_msg=key)
            else:
                assert value is expected, key
        assert getattr(scan, missing) is None

    scan.params["repetition_time_ms"] = 12.5
    assert scan.repetition_time_ms == 12.5


def test_bruker_read_bandwidth_is_a_canonical_parameter(
    synthetic_flash_scan, write_paravision_parameters
):
    write_paravision_parameters(
        synthetic_flash_scan / "method",
        {"Method": "<Bruker:FLASH>", "PVM_EffSWh": 50_000.0},
    )

    scan = load_scan(synthetic_flash_scan, mode="lazy")

    assert scan.read_bandwidth_hz == 50_000.0
    assert scan.handler.read_bandwidth_hz == 50_000.0
    assert scan.params["read_bandwidth_hz"] == 50_000.0


def test_handler_attribute_through_the_scan_names_the_handler(synthetic_flash_scan):
    scan = load_scan(synthetic_flash_scan, mode="lazy")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(AttributeError, match=r"read it as 'scan\.handler\.method'"):
            scan.method  # noqa: B018
        assert not hasattr(scan, "acqp")

    assert scan.handler.method["Method"]


def test_handler_parameter_through_the_scan_names_the_handler():
    handler = SimpleNamespace(params={"vendor_only_key": 1}, get_parameters=dict)
    scan = MRIExperiment("synthetic", handler=handler)

    with pytest.raises(AttributeError, match=r"'scan\.handler\.vendor_only_key'"):
        scan.vendor_only_key  # noqa: B018


def test_missing_handler_attribute_error_does_not_evaluate_lazy_properties():
    """The error is built from the name alone; a lazy raw-data read stays unread."""

    class LazyHandler:
        reads = 0

        def get_parameters(self):
            return {}

        @property
        def fid(self):
            LazyHandler.reads += 1
            return np.zeros(4, dtype=complex)

    scan = MRIExperiment("synthetic", handler=LazyHandler())

    with pytest.raises(AttributeError, match=r"'scan\.handler\.fid'"):
        scan.fid  # noqa: B018
    assert LazyHandler.reads == 0


def test_misspelled_attribute_raises_without_warning(synthetic_flash_scan):
    scan = load_scan(synthetic_flash_scan, mode="lazy")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(AttributeError, match="repetition_tme_ms"):
            scan.repetition_tme_ms  # noqa: B018
        assert not hasattr(scan, "repetition_tme_ms")


def test_get_reads_the_handler_without_warning(synthetic_flash_scan):
    scan = load_scan(synthetic_flash_scan, mode="lazy")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert scan.get("method") is scan.handler.method
        assert scan.get("n_slices") == scan.handler.n_slices
        assert scan.get("no_such_attribute", 3) == 3


def test_attribute_lookup_before_initialization_raises(synthetic_flash_scan):
    """A bare instance, as during unpickling, has no handler to delegate to."""
    scan = MRIExperiment.__new__(MRIExperiment)

    with pytest.raises(AttributeError, match="method"):
        scan.method  # noqa: B018


def test_library_paths_read_vendor_fields_without_deprecation(synthetic_flash_scan):
    """Study, plot sources and orientation read handler fields from the handler."""
    scan = load_scan(synthetic_flash_scan, mode="lazy")

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        np.testing.assert_array_equal(
            get_orientation_flip_multipliers(scan),
            get_orientation_flip_multipliers(scan.handler),
        )
        assert "dicom" not in DataSourceCatalog(scan).keys
        study = Study(synthetic_flash_scan, verbose=False)

    assert list(study.dataframe["DICOM"]) == ["No"]


def test_fov_shift_of_a_scan_reads_package_fields_from_its_handler():
    """Offsets in mm and voxel sizes in mm/voxel of a two-package 2D scan."""
    scan_fields = {
        "read_offset_mm": 2.0,
        "phase_offset_mm": -3.0,
        "slice_offset_mm": 0.0,
        "resolution_mm": [0.5, 0.25, 1.0],
        "type": "FLASH",
        "slice_orientation": ["axial", "coronal"],
        "read_orientation": "L_R",
    }
    handler_fields = {
        "patient_pos": "Head_Prone",
        "spatial_dimension": "2D",
        "slice_pack_offsets_mm": np.array([[2.0, -3.0, 0.0], [1.0, 1.0, 4.0]]),
    }
    scan = SimpleNamespace(handler=SimpleNamespace(**handler_fields), **scan_fields)
    flat = SimpleNamespace(**scan_fields, **handler_fields)

    for slice_pack in (None, 1):
        actual = determine_FOV_shift(scan, slice_pack=slice_pack)
        expected = determine_FOV_shift(flat, slice_pack=slice_pack)
        for got, want in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(got, want)


def test_dicom_of_a_scan_is_read_from_its_handler():
    dicom = xr.DataArray(np.zeros((2, 2)), attrs={"source": "dicom"})
    handler = SimpleNamespace(
        has_dicom_reco=True, load_dicom_data=lambda: dicom, dicom=["dataset"]
    )
    scan = SimpleNamespace(handler=handler, data=None, scanner_data=None)

    catalog = DataSourceCatalog(scan)

    assert catalog.keys == ("dicom",)
    assert catalog.load("dicom") is dicom
    assert _normalise_reference_datasets(scan) == ["dataset"]


def _legacy_scan(handler_fields, parameters=None):
    handler = SimpleNamespace(**handler_fields)
    handler.get_parameters = lambda: {}
    scan = MRIExperiment("synthetic", handler=handler)
    scan.params.update(parameters or {})
    return scan


def test_trajectory_reads_fid_from_handler_without_warning():
    from hypermri.analysis.anatomical.trajectory import (
        calc_k_space_trajectory_from_measurement,
    )

    fields = dict(fid=np.exp(1j * np.arange(8) / 8))
    parameters = dict(
        matrix=[2, 2], n_slices=2, n_repetitions=1, n_receivers=1, slice_gap_mm=2
    )
    scan = _legacy_scan(fields, parameters)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        actual = calc_k_space_trajectory_from_measurement(scan)
    expected = calc_k_space_trajectory_from_measurement(
        SimpleNamespace(**fields, **parameters)
    )
    np.testing.assert_array_equal(actual, expected)


def test_scan_timing_reads_vendor_dictionaries_without_warning():
    from hypermri.utils.io.loaders import load_scan_timing

    scan = _legacy_scan(
        dict(
            method={"PVM_ScanTimeStr": "<0h0m2s0ms>", "PVM_Nucleus1": "1H"},
            reco={"RECO_time": "<2026-01-01T12:00:00,000+0100>"},
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        result = load_scan_timing(scan)
        batch = load_scan_timing(scans=[scan])
    assert (
        result.measurement_end_time_obj[0] - result.measurement_start_time_obj[0]
    ).total_seconds() == 2
    assert result.equals(batch)


@pytest.mark.parametrize(
    "frequency_hz, expected_rows",
    [(0, [[0, 1], [2, 3], [4, 5], [6, 7]]), (250, [[2, 3], [4, 5], [6, 7], [2, 3]])],
)
def test_ideal_shift_reads_bssfp_method_without_warning(frequency_hz, expected_rows):
    from hypermri.analysis.fitting.ideal import IDEAL_Solver

    method = {"PVM_EffSWh": 1000, "PVM_Matrix": [4, 2], "ReadFreqList": 0}
    scan = _legacy_scan(dict(method=method, method_parameter=method.__getitem__))
    solver = IDEAL_Solver.__new__(IDEAL_Solver)
    solver.bssfp_object = scan
    solver.solution_matrix = np.arange(8).reshape(1, 4, 2, 1, 1) * (1 + 0.25j)
    solver.relative_frequencies = [frequency_hz]
    solver.num_transients = 0
    original = solver.solution_matrix.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        actual = solver.correct_chemical_shift()
    assert isinstance(actual, np.ndarray)
    # Bandwidth is 250 Hz/read pixel: zero offset is identity and 250 Hz moves
    # one row with legacy scipy wrap boundaries. Allow spline roundoff only.
    expected = np.asarray(expected_rows).reshape(1, 4, 2, 1, 1) * (1 + 0.25j)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(solver.solution_matrix, original)
    solver.bssfp_object = scan.handler
    # Exact comparison uses the same spline shift; a zero shift still rounds.
    np.testing.assert_array_equal(actual, solver.correct_chemical_shift())


def test_b1_calibration_reads_handler_inputs_without_warning(monkeypatch):
    from hypermri.analysis.anatomical import b1_mapping
    from hypermri.analysis.fitting import spectroscopy

    method = {"PulseVectorStart": 1, "PulseVectorEnd": 4, "Method": "synthetic"}
    scan = _legacy_scan(
        dict(
            method=method,
            method_parameter=method.__getitem__,
            fid=np.arange(16).astype(complex),
            dwell_times_s=1e-3,
        ),
        dict(n_repetitions=4, n_receivers=1, rfpulse_duration_ms=1, rfpulse_sint=1),
    )
    monkeypatch.setattr(
        spectroscopy,
        "fit_data_pseudo_inv",
        lambda data, fit_params, n_jobs=1: xr.Dataset(
            {
                "fit_amps": (
                    ("repetition_idx", "channel_idx", "metabolite"),
                    np.arange(4.0).reshape(4, 1, 1),
                )
            }
        ),
    )
    monkeypatch.setattr(
        b1_mapping, "fit_single_voxel", lambda **kwargs: (1, 2, None, None, None)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        result = b1_mapping.b1_calibration(scan, plot=False)
    assert np.isfinite(result)
