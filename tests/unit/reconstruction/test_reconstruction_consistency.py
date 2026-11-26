"""Tests for the geometric check of two reconstructions of one scan.

The fixtures are synthetic: a 4 x 3 x 2 grid of 1 mm voxels whose mm
coordinates are built like those of the Bruker raw and ``2dseq``
reconstructions, with 3D ``read_idx``, ``phase_idx`` and ``slice_idx``
grids and DICOM LPS ``x``, ``y`` and ``z``.  The raw-like array starts with
``acquisition_idx`` and the ``2dseq``-like array ends with ``echo_idx``, as
in the local studies.  The position tolerance of 1e-3 mm is the geometry
tolerance of the geometry reference tests.
"""

import doctest
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hypermri.exceptions import ReconstructionError
from hypermri.loading.experiment import MRIExperiment
from hypermri.reconstruction import consistency
from hypermri.reconstruction.consistency import check_reconstruction_geometry

SPATIAL_DIMS = ("read_idx", "phase_idx", "slice_idx")
SHAPE = (4, 3, 2)
ATOL_MM = 1e-3


def _mm_coordinates(shape=SHAPE):
    read, phase, slice_ = np.meshgrid(
        *(np.arange(size, dtype=float) - size / 2 for size in shape), indexing="ij"
    )
    grids = {
        "read_idx": read,
        "phase_idx": phase,
        "slice_idx": slice_,
        "x": read + 10.0,
        "y": -phase + 20.0,
        "z": slice_ - 30.0,
    }
    return {name: (SPATIAL_DIMS, grid, {"unit": "mm"}) for name, grid in grids.items()}


def _reconstruction(kind, shape=SHAPE, **attrs):
    attrs = {
        "patient_pos": "Head_Prone",
        "slice_orientation": "axial",
        "read_orientation": "L_R",
        **attrs,
    }
    data = xr.DataArray(
        np.zeros(shape), dims=SPATIAL_DIMS, coords=_mm_coordinates(shape), attrs=attrs
    )
    if kind == "raw":
        return data.expand_dims(acquisition_idx=1)
    return data.expand_dims(echo_idx=1, axis=-1)


def _check(candidate, reference):
    check_reconstruction_geometry(
        candidate, reference, candidate_name="raw", reference_name="seq2d"
    )


def test_matching_reconstructions_pass():
    _check(_reconstruction("raw"), _reconstruction("seq2d"))


def test_other_dimensions_between_the_spatial_ones_pass():
    raw = _reconstruction("raw").transpose(
        "read_idx", "phase_idx", "acquisition_idx", "slice_idx"
    )

    _check(raw, _reconstruction("seq2d"))


@pytest.mark.parametrize(
    "order",
    [
        ("read_idx", "slice_idx", "phase_idx"),
        ("phase_idx", "read_idx", "slice_idx"),
        ("slice_idx", "phase_idx", "read_idx"),
    ],
)
def test_another_spatial_order_raises(order):
    raw = _reconstruction("raw").transpose("acquisition_idx", *order)

    with pytest.raises(ValueError, match=r"spatial order \(") as error:
        _check(raw, _reconstruction("seq2d"))
    assert str(order) in str(error.value)


def test_another_size_raises():
    raw = _reconstruction("raw", shape=(4, 3, 3))

    with pytest.raises(ValueError, match="slice_idx has 3 instead of 2 positions"):
        _check(raw, _reconstruction("seq2d"))


def test_a_spatial_dimension_on_one_side_only_is_not_compared():
    single_slice = _reconstruction("seq2d", shape=(4, 3, 1))

    _check(_reconstruction("raw", shape=(4, 3, 1)), single_slice.squeeze("slice_idx"))


def test_cpmg_echoes_along_phase_only_in_raw_pass():
    """Shapes of a local CPMG scan with 12 echoes.

    The raw reconstruction stores the echoes along ``phase_idx`` with index
    coordinates and the ``2dseq`` reconstruction along ``extra_dim_1_idx``
    without spatial dimensions.
    """
    raw = xr.DataArray(
        np.zeros((186, 1, 12, 1)),
        dims=("acquisition_idx", "read_idx", "phase_idx", "slice_idx"),
        coords={
            dim: (dim, np.arange(size), {"unit": "index"})
            for dim, size in (("read_idx", 1), ("phase_idx", 12), ("slice_idx", 1))
        },
    )
    seq2d = xr.DataArray(
        np.zeros((256, 12)),
        dims=("acquisition_idx", "extra_dim_1_idx"),
        attrs={"patient_pos": "Head_Prone"},
    )

    check_reconstruction_geometry(
        seq2d, raw, candidate_name="seq2d", reference_name="raw"
    )


@pytest.mark.parametrize("name", ["read_idx", "phase_idx", "slice_idx", "x", "y", "z"])
def test_a_coordinate_shift_beyond_the_tolerance_raises(name):
    raw = _reconstruction("raw")
    raw = raw.assign_coords(
        {name: (raw[name] + 2 * ATOL_MM).assign_attrs(raw[name].attrs)}
    )

    with pytest.raises(
        ValueError,
        match=rf"coordinate {name} differs by more than 0.001 mm at 24 of 24",
    ):
        _check(raw, _reconstruction("seq2d"))


@pytest.mark.parametrize("name", ["read_idx", "x"])
def test_a_coordinate_shift_within_the_tolerance_passes(name):
    raw = _reconstruction("raw")
    raw = raw.assign_coords(
        {name: (raw[name] + 0.5 * ATOL_MM).assign_attrs(raw[name].attrs)}
    )

    _check(raw, _reconstruction("seq2d"))


def test_index_coordinates_are_not_compared():
    raw = _reconstruction("raw").drop_vars(SPATIAL_DIMS)
    raw = raw.assign_coords(
        {
            dim: (dim, np.arange(size), {"unit": "index"})
            for dim, size in zip(SPATIAL_DIMS, SHAPE, strict=True)
        }
    )

    _check(raw, _reconstruction("seq2d"))


def test_a_coordinate_on_one_side_only_is_not_compared():
    _check(_reconstruction("raw").drop_vars(["x", "y", "z"]), _reconstruction("seq2d"))


def test_squeezed_single_slice_coordinates_match_the_3d_grid():
    """The ``2dseq`` fallback squeezes ``x``, ``y`` and ``z`` of one slice."""
    single_slice = _reconstruction("seq2d", shape=(4, 3, 1))
    squeezed = single_slice.squeeze("slice_idx", drop=False)
    squeezed = squeezed.assign_coords(
        {name: squeezed[name].squeeze() for name in ("x", "y", "z")}
    )
    assert squeezed["x"].dims == ("read_idx", "phase_idx")

    _check(_reconstruction("raw", shape=(4, 3, 1)), squeezed)


def test_coordinates_over_other_dimensions_raise():
    raw = _reconstruction("raw")
    raw = raw.assign_coords(x=raw["x"].isel(slice_idx=0, drop=True))

    with pytest.raises(ValueError, match="coordinate x spans"):
        _check(raw, _reconstruction("seq2d"))


def test_another_patient_position_raises():
    raw = _reconstruction("raw", patient_pos="Head_Supine")

    with pytest.raises(
        ValueError,
        match="attribute patient_pos is 'Head_Supine' instead of 'Head_Prone'",
    ):
        _check(raw, _reconstruction("seq2d"))


@pytest.mark.parametrize("missing", ["absent", None])
def test_a_missing_orientation_attribute_is_not_compared(missing):
    raw = _reconstruction("raw")
    if missing == "absent":
        del raw.attrs["patient_pos"]
    else:
        raw.attrs["patient_pos"] = missing

    _check(raw, _reconstruction("seq2d"))


def test_orientation_lists_of_several_packages_compare_by_value():
    packages = ["axial", "sagittal", "coronal"]
    raw = _reconstruction("raw", slice_orientation=tuple(packages))
    seq2d = _reconstruction("seq2d", slice_orientation=np.array(packages))

    _check(raw, seq2d)
    with pytest.raises(ValueError, match="attribute slice_orientation"):
        _check(_reconstruction("raw", slice_orientation=packages[::-1]), seq2d)


def test_every_problem_is_reported_in_one_error():
    raw = _reconstruction("raw", shape=(4, 3, 3), read_orientation="A_P")
    raw = raw.assign_coords(x=(raw["x"] + 1.0).assign_attrs(raw["x"].attrs)).transpose(
        "acquisition_idx", "phase_idx", "read_idx", "slice_idx"
    )

    with pytest.raises(ValueError) as error:
        _check(raw, _reconstruction("seq2d"))
    message = str(error.value)
    assert message.startswith(
        "The 'raw' reconstruction does not describe the voxels of the stored "
        "'seq2d' reconstruction: spatial order"
    )
    for problem in (
        "slice_idx has 3 instead of 2",
        "coordinate read_idx spans",
        "coordinate x spans",
        "attribute read_orientation",
        "first remove 'seq2d'",
    ):
        assert problem in message


def test_inputs_are_not_modified():
    raw = _reconstruction("raw").transpose(
        "acquisition_idx", "read_idx", "slice_idx", "phase_idx"
    )
    seq2d = _reconstruction("seq2d")
    raw_before, seq2d_before = raw.copy(deep=True), seq2d.copy(deep=True)

    with pytest.raises(ValueError):
        _check(raw, seq2d)
    xr.testing.assert_identical(raw, raw_before)
    xr.testing.assert_identical(seq2d, seq2d_before)


def test_docstring_examples_run():
    result = doctest.testmod(consistency, optionflags=doctest.ELLIPSIS)

    assert result.attempted > 0
    assert result.failed == 0


def _experiment(monkeypatch, results):
    """Return an experiment whose strategies return ``results[reco_type]``."""
    experiment = MRIExperiment.__new__(MRIExperiment)
    experiment.handler = SimpleNamespace(id="scan", type="FLASH", vendor="bruker")
    experiment._forced_vendor = None
    experiment.reconstructions = {}
    experiment.data = None
    experiment.scanner_data = None

    def fake_strategy(vendor, sequence, reco_type):
        class FakeStrategy:
            accepted_options = frozenset()

            def reconstruct(self, handler, **_):
                return results[reco_type]

        return FakeStrategy, {}

    monkeypatch.setattr(
        "hypermri.reconstruction.selection.find_reconstruction", fake_strategy
    )
    return experiment


def test_reconstruct_stores_a_matching_reconstruction(monkeypatch):
    seq2d, raw = _reconstruction("seq2d"), _reconstruction("raw")
    experiment = _experiment(monkeypatch, {"seq2d": seq2d, "raw": raw})

    experiment.reconstruct("seq2d")
    experiment.reconstruct("raw")

    assert experiment.reconstructions.keys() == {"seq2d", "raw"}
    np.testing.assert_array_equal(
        experiment.reconstructions["seq2d"].squeeze(), seq2d.squeeze()
    )
    assert experiment.scanner_data is experiment.reconstructions["seq2d"]
    assert experiment.data is experiment.reconstructions["raw"]
    np.testing.assert_array_equal(
        experiment.data.squeeze().values, raw.values.squeeze()
    )


def test_reconstruct_stores_the_raw_reconstruction_in_the_imaging_layout(
    monkeypatch,
):
    """Previously the raw dimensions were stored in the order the strategy gave."""
    raw = _reconstruction("raw").transpose(
        "acquisition_idx", "read_idx", "slice_idx", "phase_idx"
    )
    experiment = _experiment(monkeypatch, {"raw": raw})

    experiment.reconstruct("raw")

    assert experiment.data.dims == (
        "read_idx",
        "phase_idx",
        "slice_idx",
        "waterfat_idx",
        "inversion_idx",
        "echo_idx",
        "bvalue_idx",
        "channel_idx",
        "acquisition_idx",
        "repetition_idx",
    )
    assert experiment.data.shape[:3] == SHAPE


@pytest.mark.parametrize("first, second", [("seq2d", "raw"), ("raw", "seq2d")])
def test_reconstruct_rejects_a_mismatch_without_storing(monkeypatch, first, second):
    """Previously the mismatching reconstruction was stored next to the other."""
    results = {
        "seq2d": _reconstruction("seq2d", shape=(4, 3, 3)),
        "raw": _reconstruction("raw"),
    }
    experiment = _experiment(monkeypatch, results)
    experiment.reconstruct(first)
    stored = experiment.reconstructions[first]
    data_array, reco_data = experiment.scanner_data, experiment.data

    with pytest.raises(ReconstructionError, match=r"slice_idx has"):
        experiment.reconstruct(second)

    assert experiment.reconstructions.keys() == {first}
    assert experiment.reconstructions[first] is stored
    assert experiment.scanner_data is data_array
    assert experiment.data is reco_data


def test_reconstruct_replaces_the_same_type_without_a_check(monkeypatch):
    first = _reconstruction("seq2d")
    second = _reconstruction("seq2d", shape=(4, 3, 3))
    results = {"seq2d": first}
    experiment = _experiment(monkeypatch, results)
    experiment.reconstruct("seq2d")

    results["seq2d"] = second
    experiment.reconstruct("seq2d")

    assert experiment.reconstructions.keys() == {"seq2d"}
    np.testing.assert_array_equal(
        experiment.reconstructions["seq2d"].squeeze(), second.squeeze()
    )
    assert experiment.scanner_data is experiment.reconstructions["seq2d"]
