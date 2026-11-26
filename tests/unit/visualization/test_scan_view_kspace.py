"""Image/k-space toggle and spatial filter of the browser view (set_kspace).

Uses an invented eight-point CSI fixture (3 x 4 voxels at 2 mm and 3 mm,
2 slices, 2 repetitions, random voxel amplitudes). The shown data are
compared with ``spatial_to_kspace`` and ``apply_spatial_filter`` of the same
data, which own the numerics and their tolerances.
"""

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.spatial_filtering import (
    SpatialFilterOptions,
    apply_spatial_filter,
    spatial_to_kspace,
)
from hypermri.utils.plot.data_preprocessor import DataPreprocessor
from hypermri.visualization.scan_view import plot_scan


def _signal():
    tone = 20 * np.array([1, 1j, -1, -1j] * 2, dtype=complex)
    amplitude = np.random.default_rng(7).uniform(0.5, 1.5, size=(1, 3, 4, 2, 2))
    return xr.DataArray(
        tone[:, None, None, None, None] * amplitude,
        dims=(
            "acquisition_idx",
            "read_idx",
            "phase_idx",
            "slice_idx",
            "repetition_idx",
        ),
        coords={
            "acquisition_idx": xr.DataArray(
                np.arange(8) * 0.001, dims="acquisition_idx", attrs={"units": "s"}
            ),
            "read_idx": xr.DataArray([4, 2, 0], dims="read_idx", attrs={"units": "mm"}),
            "phase_idx": xr.DataArray(
                [-3, 0, 3, 6], dims="phase_idx", attrs={"units": "mm"}
            ),
            "slice_idx": xr.DataArray([0, 5], dims="slice_idx", attrs={"units": "mm"}),
        },
        attrs={"signal_domain": "temporal", "domain": "spatial", "units": "a.u."},
    )


@pytest.fixture
def data():
    return _signal()


@pytest.fixture
def view(data):
    return plot_scan(data)


def _message(view, **content):
    view._handle_custom_message(view, {"type": "kspace", **content}, [])


def test_kspace_shows_the_centred_kspace_copy_and_leaves_the_data(data, view):
    original = data.copy(deep=True)
    view.set_kspace(True)
    assert view.state.spatial.show_image_space is False
    assert view._view["kspace"]["shown"] is True
    assert view._view["x_label"].endswith("[1/mm]")
    xr.testing.assert_identical(view._shown_data(), spatial_to_kspace(data))
    assert sorted(view._plane_curves.shape[:2]) == [3, 4]
    np.testing.assert_allclose(
        sorted(view._view["x"] + view._view["y"]),
        sorted([-1 / 6, 0, 1 / 6] + [-1 / 6, -1 / 12, 0, 1 / 12]),
        atol=1e-12,
    )
    xr.testing.assert_identical(view._data, original)
    xr.testing.assert_identical(data, original)


def test_kspace_switches_off_masks_and_roi_mean_and_returns(view):
    view.add_mask("roi")
    view.show_roi_mean("roi")
    view.set_kspace(True)
    assert view._view["mask_unavailable_reason"].startswith("masks are drawn")
    assert view._roi_mean == {}
    assert view._roi_mask == "roi"
    view.set_kspace(False)
    assert view._view["mask_unavailable_reason"] == ""
    assert view._view["kspace"]["shown"] is False
    assert view._view["x"] == [0.0, 2.0, 4.0]


def test_spatial_filter_preview_matches_apply_spatial_filter(data, view):
    options = SpatialFilterOptions(
        kind="fermi", radius_per_mm=0.1, transition_per_mm=0.02
    )
    view.set_kspace(False, options)
    expected = apply_spatial_filter(data, options)
    xr.testing.assert_identical(view._shown_data(), expected)
    assert view._shown_data() is view._shown_data()
    view.set_kspace(True, options)
    xr.testing.assert_identical(view._shown_data(), spatial_to_kspace(expected))
    view.set_kspace(False)
    assert view._shown_data() is view._data


def test_browser_messages_set_filter_and_report_errors(view):
    _message(view, filter="gaussian", sigma_per_mm=0.05)
    assert view._spatial_filter == SpatialFilterOptions(sigma_per_mm=0.05)
    assert view._view["kspace"]["sigma_per_mm"] == 0.05
    _message(view, shown=True)
    assert view._view["kspace"]["shown"] is True
    assert view._spatial_filter == SpatialFilterOptions(sigma_per_mm=0.05)
    _message(view, filter="fermi", radius_per_mm=-1)
    assert "radius_per_mm must be positive" in view._view["kspace"]["error"]
    assert view._spatial_filter == SpatialFilterOptions(sigma_per_mm=0.05)
    assert view._view["kspace"]["radius_per_mm"] == 0.1
    _message(view, shown=False, filter="none")
    assert view._spatial_filter is None
    assert view._view["kspace"]["error"] == ""


def test_kspace_without_voxel_spacing_fails_and_keeps_the_image():
    data = _signal().drop_vars(["read_idx", "phase_idx"])
    view = plot_scan(data)
    with pytest.raises(ValueError, match="ScanView.set_kspace"):
        view.set_kspace(True)
    assert view.state.spatial.show_image_space is True
    with pytest.raises(TypeError, match="SpatialFilterOptions"):
        view.set_kspace(False, "gaussian")


def test_display_preprocessing_does_not_transform_kspace_twice(data):
    kspace = spatial_to_kspace(data)
    assert DataPreprocessor().apply_kspace_transform(kspace) is kspace


def test_display_kspace_of_an_odd_matrix_returns_the_acquired_kspace():
    """The k-space view inverts the reconstruction, also for 3 x 5 voxels.

    The image comes from invented random k-space through the reconstruction's
    centred transform; the shown k-space must match it to rounding (1e-12
    absolute for values of order one).
    """
    from hypermri.reconstruction.centred_dft import centered_ifft

    rng = np.random.default_rng(11)
    acquired = rng.normal(size=(3, 5)) + 1j * rng.normal(size=(3, 5))
    image = xr.DataArray(
        centered_ifft(acquired, (0, 1)),
        dims=("read_idx", "phase_idx"),
        coords={
            "read_idx": ("read_idx", np.arange(3) * 2.0, {"units": "mm"}),
            "phase_idx": ("phase_idx", np.arange(5) * 3.0, {"units": "mm"}),
        },
        attrs={"domain": "spatial"},
    )
    shown = DataPreprocessor().apply_kspace_transform(image)
    np.testing.assert_allclose(shown.values, acquired, rtol=0, atol=1e-12)
