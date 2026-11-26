import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hypermri.loading.experiment import MRIExperiment
from hypermri.reconstruction.selection import find_reconstruction
from hypermri.reconstruction.strategies.bruker.raw import BrukerMGERawRecoStrategy
from hypermri.sequences.BSSFP import BSSFP
from hypermri.sequences.CSI import CSI
from hypermri.sequences.DTIEPI import DTIEPI
from hypermri.sequences.EPI import EPI
from hypermri.sequences.FLASH import FLASH
from hypermri.sequences.MGE import MGE


class DummyHandler:
    def __init__(self):
        self.params = {}
        self.type = "dummy"
        self.id = "dummy-id"
        self.fid = None
        self.is_spectroscopic = False
        self.method = {}

    def get_parameters(self):
        return {}


def _patch_sequence_dependencies(monkeypatch):
    monkeypatch.setattr(
        MRIExperiment,
        "_get_vendor_handler",
        lambda self, *args, **kwargs: DummyHandler(),
    )


def test_mri_experiment_initializes_empty_masks_dataset(monkeypatch):
    _patch_sequence_dependencies(monkeypatch)

    experiment = MRIExperiment(path=".")

    assert isinstance(experiment.masks, xr.Dataset)
    assert len(experiment.masks.data_vars) == 0


@pytest.mark.parametrize("sequence_class", [FLASH, CSI])
def test_sequence_rejects_an_experiment_instead_of_a_handler(
    monkeypatch, sequence_class
):
    """Adopting an existing experiment was removed; the handler is required."""
    _patch_sequence_dependencies(monkeypatch)
    experiment = MRIExperiment(path=".")

    with pytest.raises(TypeError, match=r"pass scan\.handler instead"):
        sequence_class(experiment)


def test_sequence_init_does_not_trigger_reconstruction(monkeypatch):
    calls = []

    def _reconstruct(self, reco_type="seq2d", **kwargs):
        calls.append((reco_type, kwargs))

    monkeypatch.setattr(MRIExperiment, "reconstruct", _reconstruct)

    FLASH(DummyHandler(), load_data="minimal")

    assert calls == []


@pytest.mark.parametrize("sequence_class", [FLASH, CSI])
def test_sequence_plot_builds_viewer_only_on_request(monkeypatch, sequence_class):
    calls = []
    monkeypatch.setattr(
        "hypermri.visualization.api.plot",
        lambda *a, **kw: calls.append((a, kw)) or "plot-result",
    )
    sequence = sequence_class(DummyHandler())
    assert calls == []
    assert "_plotter" not in vars(sequence)
    assert "plotter" not in type(sequence).__dict__
    data = xr.DataArray(np.zeros((2, 2)), dims=("read_idx", "phase_idx"))
    assert sequence.plot(data, plot_height=400) == "plot-result"
    assert calls[0][0][0] is data
    assert calls[0][0][1].plot_height == 400
    assert calls[0][1] == {"scan": sequence}


def test_sequence_reconstruction_policy_rejects_unknown_load_mode():
    sequence = FLASH(DummyHandler())

    with pytest.raises(ValueError, match="Unsupported loading mode"):
        sequence.run_default_reconstructions(load_data="eager")


def test_coordinate_aware_seq2d_runs_after_handler_preview_load(monkeypatch):
    """The handler preview is not a registered, coordinate-aware result."""
    flash = FLASH(DummyHandler(), load_data="minimal")
    flash._scanner_data = xr.DataArray(np.zeros((1, 1)), dims=("read", "phase"))
    flash._scanner_data_loaded = True
    calls = []

    def _reconstruct(reco_type="seq2d", **kwargs):
        calls.append((reco_type, kwargs))

    monkeypatch.setattr(flash, "reconstruct", _reconstruct)

    flash.run_default_reconstructions(load_data="minimal")

    assert calls == [("seq2d", {})]


def test_default_policy_skips_cached_seq2d_reconstruction(monkeypatch):
    flash = FLASH(DummyHandler(), load_data="minimal")
    seq2d = xr.DataArray(np.zeros((1, 1)), dims=("read", "phase"))
    flash._scanner_data = seq2d
    flash._scanner_data_loaded = True
    flash.reconstructions["seq2d"] = seq2d
    calls = []

    def _reconstruct(reco_type="seq2d", **kwargs):
        calls.append((reco_type, kwargs))

    monkeypatch.setattr(flash, "reconstruct", _reconstruct)

    flash.run_default_reconstructions(load_data="minimal")

    assert calls == []


def test_anatomy_policy_skips_raw_when_vendor_reports_no_fid_source(monkeypatch):
    handler = DummyHandler()
    handler.has_raw_data = False
    flash = FLASH(handler)
    calls = []

    monkeypatch.setattr(
        flash,
        "reconstruct",
        lambda reco_type="seq2d", **kwargs: calls.append((reco_type, kwargs)),
    )

    flash.run_default_reconstructions(load_data=True)

    assert calls == [("seq2d", {})]


@pytest.mark.parametrize("sequence_class", [BSSFP, EPI, DTIEPI])
def test_epi_family_full_load_reconstructs_scanner_and_raw_data(sequence_class):
    assert sequence_class.get_reconstruction_policy()["full"] == ("seq2d", "raw")


def test_mge_primary_data_prefers_data():
    mge = MGE(DummyHandler())
    scanner = xr.DataArray(np.ones((2, 2)), dims=("read_idx", "phase_idx"))
    raw = xr.DataArray(np.zeros((2, 2)), dims=("read_idx", "phase_idx"))
    mge._scanner_data = scanner
    mge._scanner_data_loaded = True
    mge._data = raw
    mge._data_loaded = True

    assert mge.primary_data is raw
    assert mge.primary_data is raw


def test_mge_construction_keeps_the_selected_mge_strategy():
    before = find_reconstruction("bruker", "mge", "raw")

    MGE(DummyHandler())

    assert before == find_reconstruction("bruker", "mge", "raw")
    assert before == (BrukerMGERawRecoStrategy, {})


def test_csi_primary_data_prefers_data():
    csi = CSI(DummyHandler())
    scanner = xr.DataArray(np.ones((2, 2)), dims=("read_idx", "phase_idx"))
    raw = xr.DataArray(np.zeros((2, 2)), dims=("read_idx", "phase_idx"))
    csi._scanner_data = scanner
    csi._scanner_data_loaded = True
    csi._data = raw
    csi._data_loaded = True

    assert csi.primary_data is raw
    assert csi.primary_data is raw


def test_csi_axes_follow_effective_loaded_raw_reconstruction():
    csi = CSI(DummyHandler())
    effective_time_s = np.arange(4, dtype=float) * 1e-3
    frequency_hz = np.array([-500.0, -250.0, 0.0, 250.0])
    frequency_ppm = np.array([190.0, 180.0, 170.0, 160.0])
    raw = xr.DataArray(
        np.ones((4, 1, 1), dtype=np.complex64),
        dims=("acquisition_idx", "read_idx", "phase_idx"),
        coords={
            "acquisition_idx": np.arange(4),
            "acquisition_time_s": ("acquisition_idx", effective_time_s),
        },
        attrs={
            # Preserve a deliberately full/raw handler-style attribute.  The
            # aligned DataArray coordinate must take precedence.
            "sampling_time_axis_s": np.arange(6, dtype=float) * 1e-3,
            "freq_axis_hz": frequency_hz,
            "freq_axis_ppm": frequency_ppm,
        },
    )
    csi._data = raw
    csi._data_loaded = True
    csi.handler.sampling_time_axis_s = np.arange(6, dtype=float) * 1e-3
    csi.cut_off = 2
    csi.n_acquisition = 6

    np.testing.assert_array_equal(csi.sampling_time_axis, effective_time_s)
    np.testing.assert_array_equal(csi.time_axis, effective_time_s)
    np.testing.assert_array_equal(csi.hz, frequency_hz)
    np.testing.assert_array_equal(csi.ppm, frequency_ppm)


def test_csi_find_peaks_builds_axis_for_effective_data_length(monkeypatch):
    csi = CSI(DummyHandler())
    raw = xr.DataArray(
        np.ones((4, 1, 1), dtype=np.complex64),
        dims=("acquisition_idx", "read_idx", "phase_idx"),
    )
    csi._data = raw
    csi._data_loaded = True
    csi.cut_off = 2
    csi.n_acquisition = 6
    csi.peaks = {}
    csi.peakproperties = {}
    csi.searchranges = {}
    csi.searchranges_idx = {}
    csi.apply_to_every_spec = lambda _func: []

    captured = {}

    def fake_get_freq_axis(*, scan, unit, cut_off, npoints):
        captured.update(
            scan=scan,
            unit=unit,
            cut_off=cut_off,
            npoints=npoints,
        )
        return np.arange(npoints, dtype=float)

    monkeypatch.setattr(
        "hypermri.sequences.base_spectroscopy.get_freq_axis",
        fake_get_freq_axis,
    )

    csi.find_peaks(range_unit="Hz")

    assert captured == {
        "scan": csi,
        "unit": "Hz",
        "cut_off": 2,
        "npoints": 4,
    }


def test_csi_default_policy_reconstructs_seq2d_for_minimal_load(monkeypatch):
    csi = CSI(DummyHandler(), load_data="minimal")
    calls = []

    def _reconstruct(reco_type="seq2d", **kwargs):
        calls.append((reco_type, kwargs))

    monkeypatch.setattr(csi, "reconstruct", _reconstruct)

    csi.run_default_reconstructions(load_data="minimal")

    assert calls == [("seq2d", {})]


def test_csi_reports_spectroscopic_even_when_handler_flag_is_false():
    csi = CSI(DummyHandler())

    assert csi.is_spectroscopic is True


def test_csi_scanner_data_getter_does_not_recurse(monkeypatch):
    csi = CSI(DummyHandler())

    calls = []

    def no_scanner_payload(**kwargs):
        calls.append(kwargs)
        return None

    monkeypatch.setattr(csi, "reconstruct", no_scanner_payload)
    assert csi.scanner_data is None
    assert csi.scanner_data is None
    assert calls == [{"reco_type": "seq2d"}]


def test_csi_scanner_data_getter_propagates_scanner_failure(monkeypatch):
    csi = CSI(DummyHandler())
    failure = RuntimeError("synthetic scanner defect")

    def fail(**kwargs):
        raise failure

    monkeypatch.setattr(csi, "reconstruct", fail)
    with pytest.raises(RuntimeError, match="synthetic scanner defect") as caught:
        _ = csi.scanner_data
    assert caught.value is failure


@pytest.mark.parametrize("patient_position", ["Head_Prone", "Foot_Prone"])
def test_csi_get_plotting_grid_uses_standardized_geometry(
    monkeypatch, patient_position
):
    captured = {}

    def fake_define_xyz_grid(mat, fov):
        captured["mat"] = mat
        captured["fov"] = fov
        return np.array([100.0]), np.array([10.0]), np.array([20.0])

    csi_module = importlib.import_module("hypermri.sequences.CSI")
    monkeypatch.setattr(csi_module, "define_xyz_grid", fake_define_xyz_grid)
    csi_like = SimpleNamespace(
        patient_position=patient_position,
        read_orientation="L_R",
        slice_orientation="axial",
        matrix=np.array([2, 3]),
        fov_mm=np.array([20.0, 30.0]),
        spatial_dimension="2D",
        n_slices=1,
        slice_thickness_mm=2.0,
        slice_gap_mm=0.0,
        read_offset_mm=1.5,
        phase_1_offset_mm=2.5,
    )

    plotting_grid = CSI.get_plotting_grid(csi_like)

    np.testing.assert_array_equal(captured["mat"], np.array([1, 2, 3]))
    np.testing.assert_array_equal(captured["fov"], np.array([2.0, 20.0, 30.0]))
    np.testing.assert_array_equal(plotting_grid[0], np.array([11.5]))
    np.testing.assert_array_equal(plotting_grid[1], np.array([22.5]))


@pytest.mark.parametrize("patient_position", ["Head_Prone", "Foot_Prone"])
@pytest.mark.parametrize("spatial_dimension", ["2D", "3D"])
@pytest.mark.parametrize(
    "slice_orientation, read_orientation, expected",
    [
        ("axial", "L_R", ([-2.5, -0.5, 1.5, 3.5], [-3.5, -0.5, 2.5])),
        ("axial", "A_P", ([-2.5, -0.5, 1.5, 3.5], [-3.5, -0.5, 2.5])),
        ("sagittal", "H_F", ([-2.5, -0.5, 1.5, 3.5], [-3.5, -0.5, 2.5])),
        ("sagittal", "A_P", ([-2.5, -0.5, 1.5, 3.5], [-3.5, -0.5, 2.5])),
        ("coronal", "H_F", ([-2.5, -0.5, 1.5, 3.5], [-3.5, -0.5, 2.5])),
        ("coronal", "L_R", ([-2.5, -0.5, 1.5, 3.5], [-3.5, -0.5, 2.5])),
    ],
)
def test_csi_legacy_plotting_grid_orientation_values(
    patient_position, spatial_dimension, slice_orientation, read_orientation, expected
):
    """Characterize all prone pairs using unequal in-plane counts and extents."""
    scan = SimpleNamespace(
        patient_position=patient_position,
        slice_orientation=slice_orientation,
        read_orientation=read_orientation,
        matrix=np.array([4, 3] if spatial_dimension == "2D" else [4, 3, 2]),
        fov_mm=np.array([8.0, 9.0] if spatial_dimension == "2D" else [8.0, 9.0, 6.0]),
        spatial_dimension=spatial_dimension,
        n_slices=2,
        slice_thickness_mm=3.0,
        slice_gap_mm=0.0,
        read_offset_mm=1.5,
        phase_1_offset_mm=1.0,
    )
    first = CSI.get_plotting_grid(scan)
    second = CSI.get_plotting_grid(scan)
    for line, repeated, reference in zip(first, second, expected, strict=True):
        # Synthetic mm positions; 1e-12 tests arithmetic, not scanner accuracy.
        np.testing.assert_allclose(line, reference, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(line, repeated)
        assert line.dtype == np.float64
    np.testing.assert_array_equal(
        scan.matrix, [4, 3] if spatial_dimension == "2D" else [4, 3, 2]
    )
    np.testing.assert_array_equal(
        scan.fov_mm, [8.0, 9.0] if spatial_dimension == "2D" else [8.0, 9.0, 6.0]
    )


@pytest.mark.parametrize(
    "patient_position", ["Head_Prone", "Foot_Prone", "Head_Supine"]
)
@pytest.mark.parametrize(
    "slice_orientation, read_orientation",
    [
        ("axial", "invalid"),
        ("sagittal", "invalid"),
        ("coronal", "invalid"),
        ("invalid", "L_R"),
    ],
)
def test_csi_plotting_table_preserves_unsupported_orientation_errors(
    monkeypatch, patient_position, slice_orientation, read_orientation
):
    """Isolate plotting-table errors from the earlier metadata helper checks."""
    csi_module = importlib.import_module("hypermri.sequences.CSI")
    monkeypatch.setattr(
        csi_module, "define_imageFOV_parameters", lambda scan: (6.0, 8.0, 9.0)
    )
    monkeypatch.setattr(
        csi_module, "define_imagematrix_parameters", lambda scan: (2, 4, 3)
    )
    scan = SimpleNamespace(
        patient_position=patient_position,
        slice_orientation=slice_orientation,
        read_orientation=read_orientation,
    )
    expected = (
        "unknown slice orientation: invalid"
        if slice_orientation == "invalid"
        else f"unknown read orientation: invalidfor slice_orient: {slice_orientation}"
    )
    with pytest.raises(Exception) as caught:
        CSI.get_plotting_grid(scan)
    assert type(caught.value) is Exception
    assert str(caught.value) == expected


@pytest.mark.parametrize(
    "slice_orientation, read_orientation, expected",
    [
        ("axial", "L_R", ([-4.0, -2.0, 0.0, 2.0], [-4.5, -1.5, 1.5])),
        ("sagittal", "H_F", ([-3.0, 0.0], [-4.5, -1.5, 1.5])),
        ("coronal", "H_F", ([-3.0, 0.0], [-4.0, -2.0, 0.0, 2.0])),
    ],
)
def test_csi_nominal_supine_plotting_table_keeps_offsets_ignored(
    monkeypatch, slice_orientation, read_orientation, expected
):
    """Preserve nominal table entries without adding missing metadata support."""
    csi_module = importlib.import_module("hypermri.sequences.CSI")
    monkeypatch.setattr(
        csi_module, "define_imageFOV_parameters", lambda scan: (6.0, 8.0, 9.0)
    )
    monkeypatch.setattr(
        csi_module, "define_imagematrix_parameters", lambda scan: (2, 4, 3)
    )
    scan = SimpleNamespace(
        patient_position="Head_Supine",
        slice_orientation=slice_orientation,
        read_orientation=read_orientation,
        read_offset_mm=100.0,
        phase_1_offset_mm=200.0,
    )
    for actual, reference in zip(CSI.get_plotting_grid(scan), expected, strict=True):
        np.testing.assert_allclose(actual, reference, rtol=0, atol=1e-12)


def test_csi_legacy_supine_axial_ignores_offsets():
    scan = SimpleNamespace(
        patient_position="Head_Supine",
        slice_orientation="axial",
        read_orientation="L_R",
        matrix=np.array([4, 3, 2]),
        fov_mm=np.array([8.0, 9.0, 6.0]),
        spatial_dimension="3D",
        read_offset_mm=100.0,
        phase_1_offset_mm=200.0,
    )
    with pytest.warns(UserWarning, match="Head_Supine is not fully implemented"):
        lines = CSI.get_plotting_grid(scan)
    np.testing.assert_allclose(lines[0], [-4.0, -2.0, 0.0, 2.0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(lines[1], [-4.5, -1.5, 1.5], rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "slice_orientation, read_orientation, error",
    [
        ("axial", "A_P", Exception),
        ("sagittal", "H_F", ValueError),
        ("coronal", "H_F", ValueError),
    ],
)
def test_csi_legacy_supine_pairs_are_unavailable(
    slice_orientation, read_orientation, error
):
    """The nominal sagittal/coronal table cannot supply a complete matrix."""
    scan = SimpleNamespace(
        patient_position="Head_Supine",
        slice_orientation=slice_orientation,
        read_orientation=read_orientation,
        matrix=np.array([4, 3, 2]),
        fov_mm=np.array([8.0, 9.0, 6.0]),
        spatial_dimension="3D",
    )
    with (
        pytest.warns(UserWarning, match="Head_Supine is not fully implemented"),
        pytest.raises(error),
    ):
        CSI.get_plotting_grid(scan)


@pytest.mark.parametrize("slice_orientation", ["axial", "coronal"])
def test_csi_legacy_plotting_grid_interpolation_and_phase_offset_fallback(
    slice_orientation,
):
    scan = SimpleNamespace(
        patient_position="Head_Prone",
        slice_orientation=slice_orientation,
        read_orientation="L_R",
        matrix=np.array([2, 3, 1]),
        fov_mm=np.array([8.0, 9.0, 6.0]),
        spatial_dimension="3D",
        read_offset_mm=1.5,
        phase_1_offset_mm=None,
        phase_offset_mm=1.0,
    )
    factors = np.array([99, 2, 2, 3])
    lines = CSI.get_plotting_grid(scan, factors)
    np.testing.assert_allclose(lines[0], [-2.5, -0.5, 1.5, 3.5], rtol=0, atol=1e-12)
    expected_phase = (
        [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        if slice_orientation == "axial"
        else [-3.5, -2.0, -0.5, 1.0, 2.5, 4.0]
    )
    np.testing.assert_allclose(lines[1], expected_phase, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(factors, [99, 2, 2, 3])


def test_csi_get_plotting_grid_rejects_feet_prone(monkeypatch):
    """Regression test: ParaVision writes Foot_Prone, not Feet_Prone.

    The field of view and matrix helpers are replaced so that the test
    reaches the orientation table of ``get_plotting_grid`` itself.  Before
    the fix the string ``"Foot_Prone"`` of real feet-first scans was rejected.
    """
    csi_module = importlib.import_module("hypermri.sequences.CSI")
    monkeypatch.setattr(
        csi_module, "define_imageFOV_parameters", lambda scan: (2.0, 20.0, 30.0)
    )
    monkeypatch.setattr(
        csi_module, "define_imagematrix_parameters", lambda scan: (1, 2, 3)
    )
    csi_like = SimpleNamespace(
        patient_position="Feet_Prone",
        read_orientation="L_R",
        slice_orientation="axial",
    )

    with pytest.raises(Exception, match="unknown patient_position: Feet_Prone"):
        CSI.get_plotting_grid(csi_like)


@pytest.mark.parametrize(
    "sequence, method, arguments, missing",
    [
        (CSI, "find_peaks", (), "apply_to_every_spec"),
        (BSSFP, "reco_scrambled_phase", (), "bssfp_fft"),
    ],
)
def test_unavailable_legacy_sequence_operations_explain_the_missing_input(
    sequence, method, arguments, missing
):
    instance = sequence.__new__(sequence)
    with pytest.raises(NotImplementedError, match=missing):
        getattr(instance, method)(*arguments)


@pytest.mark.parametrize(
    "method, args, missing",
    [
        ("calculate_T1", (), "get_spectra"),
        ("GUI_calculate_T1", (), "get_spectra"),
        (
            "InversionRecovery_Calculate_T1_multi_peak",
            ([0, 2], ([0] * 4, [1] * 4)),
            "get_ppm",
        ),
    ],
)
def test_inversion_recovery_legacy_operations_fail_before_plotting(
    method, args, missing
):
    from hypermri.sequences.IVRecovery import IVRecovery

    instance = IVRecovery.__new__(IVRecovery)
    with pytest.raises(NotImplementedError, match=missing):
        getattr(instance, method)(*args)


def test_legacy_inversion_recovery_ui_checks_inputs_before_widgets():
    from hypermri.sequences.IVRecovery import IVRecovery

    instance = IVRecovery.__new__(IVRecovery)
    with pytest.raises(NotImplementedError, match="get_ppm"):
        instance.Interactive_Fitting([0, 2], [])


def test_legacy_csi_simulator_explains_missing_iterator():
    from hypermri.visualization.csi_simulator import CSImulator

    simulator = CSImulator.__new__(CSImulator)
    simulator.csi = object()
    with pytest.raises(NotImplementedError, match="apply_to_every_spec"):
        simulator.simulate([0.0, 1e-3])
