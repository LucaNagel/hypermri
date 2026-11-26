"""Spatial overlay layer of the shown plane.

The invented scans below have values that name their voxel
(``1 + 10 * read + phase + 100 * slice``) and an overlay on the same grid
whose values are the primary value plus 1000, so a correctly placed overlay
plane equals the shown primary plane plus 1000 at every pixel, whatever the
rotation or coordinate mode. Values are exact small integers stored as float,
so the comparisons are exact.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr

from hypermri.visualization.render_data import PlotDataModel
from hypermri.visualization.spatial_overlay import (
    active_overlay_source_catalog,
    apply_overlay_start_state,
    default_overlay_variable,
    load_overlay_source,
    normalize_overlays,
    overlay_source_catalogs,
    overlay_value_range,
    overlay_variable_names,
    split_overlay_variable,
)
from hypermri.visualization.state import OverlayState, PlotState

DIMS = ("read_idx", "phase_idx", "slice_idx")


def _grid_coords(shape: tuple[int, int, int]) -> dict:
    r, p, s = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
    return {
        "x": (DIMS, 2.0 * r, {"unit": "mm"}),
        "y": (DIMS, 10.0 - 3.0 * p, {"unit": "mm"}),
        "z": (DIMS, 5.0 * s, {"unit": "mm"}),
    }


def _voxel_values(shape: tuple[int, int, int]) -> np.ndarray:
    r, p, s = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
    return (1 + 10 * r + p + 100 * s).astype(float)


def _scan(shape=(4, 3, 2)) -> xr.DataArray:
    return xr.DataArray(_voxel_values(shape), dims=DIMS, coords=_grid_coords(shape))


def _overlay(shape=(4, 3, 2)) -> xr.DataArray:
    return xr.DataArray(
        _voxel_values(shape) + 1000.0, dims=DIMS, coords=_grid_coords(shape)
    )


def _fit_dataset(shape=(4, 3, 2)) -> xr.Dataset:
    values = _voxel_values(shape)
    amps = np.stack([values + 1000.0, values + 2000.0], axis=-1)
    coords = _grid_coords(shape)
    coords["metabolite"] = ["pyruvate", "lactate"]
    return xr.Dataset(
        {
            "fit_amps": ((*DIMS, "metabolite"), amps),
            "fit_freqs": ((*DIMS, "metabolite"), np.zeros_like(amps)),
        },
        coords=coords,
    )


def test_layer_names_follow_registration_and_variable_order():
    overlays = {"fit": _fit_dataset(), "map": _overlay()}

    assert overlay_variable_names(overlays) == ["fit_fit_amps", "fit_fit_freqs", "map"]


def test_single_overlay_is_registered_as_overlay():
    overlays, source_objects = normalize_overlays(_fit_dataset(), "rps")

    assert list(overlays) == ["overlay"]
    assert source_objects == {}
    assert overlay_variable_names(overlays) == ["overlay_fit_amps", "overlay_fit_freqs"]
    assert normalize_overlays(None, "xyz") == ({}, {})


def test_layer_name_split_prefers_the_longest_registered_name():
    # "fit_2_fit_amps" also starts with "fit_"; only the longer name "fit_2"
    # leaves the real Dataset variable.
    overlays = {"fit": _fit_dataset(), "fit_2": _fit_dataset(), "map": _overlay()}

    assert split_overlay_variable(overlays, "fit_2_fit_amps") == ("fit_2", "fit_amps")
    assert split_overlay_variable(overlays, "fit_fit_freqs") == ("fit", "fit_freqs")
    assert split_overlay_variable(overlays, "map") == ("map", None)
    assert split_overlay_variable(overlays, "unknown") == (None, None)
    assert split_overlay_variable(overlays, None) == (None, None)


def test_default_layer_prefers_amplitudes_then_frequencies_then_first():
    frequencies_only = _fit_dataset()[["fit_freqs"]]

    assert default_overlay_variable({"map": _overlay(), "fit": _fit_dataset()}) == (
        "fit_fit_amps"
    )
    assert default_overlay_variable({"fit": frequencies_only}) == "fit_fit_freqs"
    assert default_overlay_variable({"map": _overlay()}) == "map"
    assert default_overlay_variable({}) is None


def test_start_state_selects_default_layer_and_t1_colormap():
    t1_fit = _fit_dataset().assign_attrs(model_name="t1_inversion_recovery")
    state = OverlayState()
    default_cmap = state.cmap

    apply_overlay_start_state(state, {"fit": _fit_dataset()})
    assert (state.active_variable, state.cmap) == ("fit_fit_amps", default_cmap)

    t1_state = OverlayState()
    apply_overlay_start_state(t1_state, {"t1": t1_fit})
    assert (t1_state.active_variable, t1_state.cmap) == ("t1_fit_amps", "lipari")

    # A layer chosen beforehand, for example from saved settings, is kept.
    saved = OverlayState(active_variable="t1_fit_freqs")
    apply_overlay_start_state(saved, {"t1": t1_fit})
    assert (saved.active_variable, saved.cmap) == ("t1_fit_freqs", default_cmap)


def test_value_range_uses_finite_magnitudes():
    complex_values = xr.DataArray(np.array([3 + 4j, -1 + 0j, np.nan]))
    no_finite = xr.DataArray(np.array([np.nan, np.inf]))

    assert overlay_value_range(complex_values) == (1.0, 5.0)
    assert overlay_value_range(_overlay()) == (1001.0, 1133.0)
    assert overlay_value_range(no_finite) is None


class _ReconstructedScan:
    """Invented scan with hypermri, scanner and DICOM reconstructions.

    Values are 0, 100 and 200 everywhere; the DICOM series is read on request.
    """

    is_spectroscopic = False
    has_dicom_reco = True

    def __init__(self, shape=(2, 3, 1)):
        def layer(value, **attrs):
            return xr.DataArray(np.full(shape, value), dims=DIMS, attrs=attrs)

        self.data = layer(0.0)
        self.scanner_data = layer(100.0)
        self._dicom = layer(200.0, source="dicom")
        self.dicom_load_count = 0

    @property
    def primary_data(self):
        return self.data

    def load_dicom_data(self):
        self.dicom_load_count += 1
        return self._dicom


def test_overlay_source_catalogs_load_only_the_requested_source():
    scan = _ReconstructedScan()
    overlays, source_objects = normalize_overlays(scan, "rps")
    shown, catalogs = overlay_source_catalogs(
        overlays, source_objects, requested=None, coordinate_mode="rps"
    )
    assert shown["overlay"] is overlays["overlay"]
    assert catalogs["overlay"].current_key == "data"
    assert catalogs["overlay"].keys == ("data", "scanner_data", "dicom")
    assert scan.dicom_load_count == 0

    shown, catalogs = overlay_source_catalogs(
        overlays, source_objects, requested="dicom", coordinate_mode="rps"
    )
    assert float(shown["overlay"].max()) == 200.0
    assert catalogs["overlay"].current_key == "dicom"
    assert scan.dicom_load_count == 1
    assert float(overlays["overlay"].max()) == 0.0

    # A source the scan does not offer keeps the shown data.
    shown, catalogs = overlay_source_catalogs(
        overlays, source_objects, requested="pdata", coordinate_mode="rps"
    )
    assert catalogs["overlay"].current_key == "data"


def test_active_source_catalog_and_source_loading():
    scan = _ReconstructedScan()
    fit = xr.Dataset({"fit_amps": (DIMS, np.ones((2, 3, 1)))})
    overlays, source_objects = normalize_overlays({"scan": scan, "fit": fit}, "rps")
    _, catalogs = overlay_source_catalogs(
        overlays, source_objects, requested=None, coordinate_mode="rps"
    )
    assert active_overlay_source_catalog(overlays, catalogs, "scan")[0] == "scan"
    # A layer without a scan falls back to the only scan overlay.
    name, catalog = active_overlay_source_catalog(overlays, catalogs, "fit_fit_amps")
    assert name == "scan" and catalog is catalogs["scan"]
    assert active_overlay_source_catalog(overlays, {}, "scan") == (None, None)

    loaded = load_overlay_source(catalog, "scanner_data", "rps")
    assert float(loaded.max()) == 100.0
    assert catalog.current_key == "scanner_data"
    with pytest.raises(KeyError):
        load_overlay_source(catalog, "pdata", "rps")
    assert catalog.current_key == "scanner_data"


def test_primary_slice_for_overlay_slice_in_scanner_coordinates():
    # Primary slices at z = 0, 5, 10 mm; overlay slices at z = 0, 3, ..., 15 mm.
    primary = _scan((2, 3, 3))
    shape = (2, 3, 6)
    r, p, s = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
    overlay = xr.DataArray(
        np.zeros(shape),
        dims=DIMS,
        coords={
            "x": (DIMS, 2.0 * r, {"unit": "mm"}),
            "y": (DIMS, 10.0 - 3.0 * p, {"unit": "mm"}),
            "z": (DIMS, 3.0 * s, {"unit": "mm"}),
        },
    )
    model = PlotDataModel(SimpleNamespace(primary_data=primary))
    state = PlotState()
    state.spatial.coordinate_mode = "xyz"
    state.spatial.slice_indices = {"read_idx": 0, "phase_idx": 0, "slice_idx": 0}

    def match(index):
        return model.primary_slice_for_secondary_slice(
            primary, state, overlay, "slice_idx", index, label="overlay"
        )

    # z = 6 mm is nearest to the primary slice at 5 mm.
    assert match(2) == ("slice_idx", 1, None)
    dim, index, warning = match(5)
    assert (dim, index) == ("slice_idx", 2)
    assert "overlay slice is outside primary FOV" in warning
    assert (
        model.primary_slice_for_secondary_slice(
            primary, state, overlay, "phase_idx_missing", 0, label="overlay"
        )
        is None
    )
    state.spatial.coordinate_mode = "rps"
    assert match(2) is None
