"""Validate refitting using invented, noiseless two-component signals.

The reference amplitudes 3+0.5j/1-0.25j, frequencies 15/-25 Hz and T2-star
0.03/0.05 s use 2e-6 recovery tolerance for complex64 storage and 1e-12 for
explicit float64 settings. The joint refinement moves the fitted frequency
and T2* to the minimum of the rounded complex64 signal, measured up to
1.3e-8 relative from the truth, so complex64 recovery allows 1e-6 relative.
Window tests compare starts -/+ half widths in Hz, exact up to rounding
(1e-12 relative). Timing tests protect metadata priority, not a new
scientific timing rule. No scanner data or quality claim are introduced.
"""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.spectroscopy import (
    SpectroscopyRefitEdit,
    refit_spectroscopy_voxel,
    refitting,
)


def _inputs(dtype=np.complex64, domain="temporal", permuted=False):
    time_s = np.arange(64) * 0.001
    fid = sum(
        amp * np.exp(2j * np.pi * freq * time_s) * np.exp(-time_s / t2)
        for amp, freq, t2 in [(3 + 0.5j, 15, 0.03), (1 - 0.25j, -25, 0.05)]
    )
    values = np.stack([2 * fid, fid], axis=1).astype(dtype)
    if domain == "spectral":
        values = np.fft.fftshift(np.fft.fft(values, axis=0), axes=0)
    data = xr.DataArray(
        values,
        dims=("acquisition_idx", "read_idx"),
        coords={"acquisition_idx": time_s, "read_idx": [7, 2]},
        attrs={"units": "a.u.", "signal_domain": domain, "provenance": "invented FID"},
    )
    data.acquisition_idx.attrs["units"] = "s"
    coords = {"read_idx": [7, 2], "metabolite": ["p", "l"]}

    def variable(values):
        return xr.DataArray(values, dims=("read_idx", "metabolite"), coords=coords)

    params = xr.Dataset(
        {
            "initial_freqs_hz": variable([[15.0, -25.0], [15.0, -25.0]]),
            "metabs_t2_ranges": variable([[0.02, 0.02], [0.02, 0.02]]),
        },
        attrs={
            "signal_domain": domain,
            "rep_fitting": 1,
            "niter": 1,
            "npoints": 3,
            "metabs_freq_ranges": 4.0,
            "min_t2_s": 0.005,
            "max_t2_s": 0.1,
        },
    )
    results = xr.Dataset(
        {
            "fit_amps": variable(np.full((2, 2), 7 + 2j, dtype=dtype)),
            "fit_freqs": variable([[15.0, -25.0], [15.0, -25.0]]),
            "fit_t2s": variable([[0.03, 0.05], [0.03, 0.05]]),
        },
        attrs={"provenance": "invented starting results"},
    )
    for name, units in [("fit_amps", "a.u."), ("fit_freqs", "Hz"), ("fit_t2s", "s")]:
        results[name].attrs["units"] = units
    if permuted:
        data = data.transpose("read_idx", "acquisition_idx")
        params = params.transpose("metabolite", "read_idx")
        results = results.transpose("metabolite", "read_idx")
    return data, params, results


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("domain", ["temporal", "spectral"])
@pytest.mark.parametrize("permuted", [False, True])
def test_direct_refit_recovers_signal_and_keeps_inputs_and_units(
    dtype, domain, permuted
):
    inputs = _inputs(dtype, domain, permuted)
    before = tuple(value.copy(deep=True) for value in inputs)
    fitted = refit_spectroscopy_voxel(*inputs, voxel_indices={"read_idx": 1})
    repeated = refit_spectroscopy_voxel(*inputs, voxel_indices={"read_idx": 1})
    result = fitted["fit_results"]
    recovery_rtol = 1e-6 if dtype == np.complex64 else 1e-12
    np.testing.assert_allclose(
        result.fit_amps, [3 + 0.5j, 1 - 0.25j], rtol=2e-6, atol=2e-6
    )
    np.testing.assert_allclose(result.fit_freqs, [15, -25], rtol=recovery_rtol)
    np.testing.assert_allclose(result.fit_t2s, [0.03, 0.05], rtol=recovery_rtol)
    assert result.fit_amps.dtype == dtype
    assert result.fit_amps.dims == ("metabolite",)
    assert result.read_idx == 2
    assert result.attrs == inputs[2].attrs
    assert result.fit_freqs.attrs["units"] == "Hz"
    assert fitted["status"] == "refitted"
    assert fitted["active_metabolites"] == ["p", "l"]
    assert fitted["amplitude_refinement_error"] is None
    assert fitted["frequency_search_diagnostics"].shape[0] == 2
    assert fitted["t2_search_diagnostics"].shape[-1] == 3
    assert fitted["reconstructed_signal"] is None
    assert fitted["uncertainty"] is None
    xr.testing.assert_identical(repeated["fit_results"], result)
    for value, original in zip(inputs, before, strict=True):
        xr.testing.assert_identical(value, original)


def _capture_fitter(monkeypatch):
    captured = {}

    def fit(**kwargs):
        captured.update(kwargs)
        count = len(kwargs["metabs"])
        return (
            np.full(count, 2 + 3j),
            np.asarray(kwargs["initial_freq_hz"]),
            np.asarray(kwargs["initial_t2_s"]),
        )

    def solve(**kwargs):
        captured["amplitude_options"] = kwargs
        return None, np.full(len(kwargs["freq_hz"]), 4 + 5j)

    monkeypatch.setattr(refitting.fid_model, "_fit_single_spectrum_pseudo_inv", fit)
    monkeypatch.setattr(refitting.fid_model, "solve_for_complex_amplitudes", solve)
    return captured


@pytest.mark.parametrize(
    "width, bounds",
    [
        (4.0, (15.0, 19.0)),
        (0.0, (17.0, 17.0)),
    ],
)
def test_reviewed_rows_own_bounds_and_disabled_results_are_retained(
    monkeypatch, width, bounds
):
    data, params, results = _inputs()
    captured = _capture_fitter(monkeypatch)
    params["freq_range_lower_bound_hz"] = xr.full_like(params.initial_freqs_hz, 100.0)
    params["freq_range_upper_bound_hz"] = xr.full_like(params.initial_freqs_hz, 200.0)
    edits = {
        "p": SpectroscopyRefitEdit(
            initial_freq_hz=17.0,
            freq_search_range_hz=width,
            initial_t2_s=0.04,
            min_t2_s=0.006,
            max_t2_s=0.08,
        ),
        "l": SpectroscopyRefitEdit(enabled=False),
    }
    fitted = refit_spectroscopy_voxel(
        data, params, results, voxel_indices={"read_idx": 1}, edits=edits
    )
    assert captured["metabs"] == ["p"]
    np.testing.assert_allclose(
        captured["freq_search_range_lower_bound_hz"], [bounds[0]], rtol=1e-12
    )
    np.testing.assert_allclose(
        captured["freq_search_range_upper_bound_hz"], [bounds[1]], rtol=1e-12
    )
    assert captured["initial_t2_s"][0] == 0.04
    assert captured["min_t2_s"][0] == 0.006
    assert captured["max_t2_s"][0] == 0.08
    assert fitted["fit_results"].fit_amps.sel(metabolite="l") == 7 + 2j
    assert fitted["effective_options"]["data_indices"] == {"read_idx": 1}


def test_unreviewed_bounds_follow_start_and_width_without_a_range_map(monkeypatch):
    # Without a metabs_freq_ranges map the attrs width (4 Hz) is the width;
    # stored bounds, including 0 Hz and NaN, never replace start -/+ width/2.
    data, params, results = _inputs()
    captured = _capture_fitter(monkeypatch)
    params["freq_range_lower_bound_hz"] = xr.DataArray(
        [0.0, -40.0], dims="metabolite", coords={"metabolite": ["p", "l"]}
    )
    params["freq_range_upper_bound_hz"] = xr.DataArray(
        [np.nan, -10.0], dims="metabolite", coords={"metabolite": ["p", "l"]}
    )
    refit_spectroscopy_voxel(data, params, results, voxel_indices={"read_idx": 1})
    np.testing.assert_allclose(captured["freq_search_range_hz"], [4.0, 4.0])
    np.testing.assert_allclose(
        captured["freq_search_range_lower_bound_hz"], [13.0, -27.0], rtol=1e-12
    )
    np.testing.assert_allclose(
        captured["freq_search_range_upper_bound_hz"], [17.0, -23.0], rtol=1e-12
    )


def test_unreviewed_window_ending_at_zero_hz_keeps_that_bound(monkeypatch):
    # Start -2 Hz with a 4 Hz map width: the window is [-4, 0] Hz. The attrs
    # width of 50 Hz must not widen it.
    data, params, results = _inputs()
    captured = _capture_fitter(monkeypatch)
    params["initial_freqs_hz"].values[1] = [-2.0, -25.0]
    params["metabs_freq_ranges"] = xr.full_like(params.initial_freqs_hz, 4.0)
    params.attrs["metabs_freq_ranges"] = 50.0
    fitted = refit_spectroscopy_voxel(
        data, params, results, voxel_indices={"read_idx": 1}
    )
    np.testing.assert_allclose(captured["freq_search_range_hz"], [4.0, 4.0])
    np.testing.assert_allclose(
        captured["freq_search_range_lower_bound_hz"], [-4.0, -27.0], rtol=1e-12
    )
    np.testing.assert_array_equal(captured["freq_search_range_upper_bound_hz"][0], 0.0)
    assert fitted["effective_options"]["freq_search_range_upper_bound_hz"][0] == 0.0


def test_unreviewed_lines_take_the_width_of_their_voxel(monkeypatch):
    # Voxel 0 has 10 Hz windows, voxel 1 has 6 Hz for p and a frozen l (0 Hz);
    # the attrs width is 4 Hz. The refit of voxel 1 uses voxel 1's widths.
    data, params, results = _inputs()
    captured = _capture_fitter(monkeypatch)
    params["metabs_freq_ranges"] = params.initial_freqs_hz.copy(
        data=[[10.0, 10.0], [6.0, 0.0]]
    )
    refit_spectroscopy_voxel(data, params, results, voxel_indices={"read_idx": 1})
    np.testing.assert_allclose(captured["freq_search_range_hz"], [6.0, 0.0])
    np.testing.assert_allclose(
        captured["freq_search_range_lower_bound_hz"], [12.0, -25.0], rtol=1e-12
    )
    np.testing.assert_allclose(
        captured["freq_search_range_upper_bound_hz"], [18.0, -25.0], rtol=1e-12
    )


@pytest.mark.parametrize("reviewed", [False, True])
def test_lines_take_the_t2_window_of_their_voxel(monkeypatch, reviewed):
    # Voxel 1 has its own T2* bounds and widths, l frozen (0 s); the attrs
    # hold other values. A reviewed row for p brings its own bounds, but its
    # T2* width still comes from the voxel.
    data, params, results = _inputs()
    captured = _capture_fitter(monkeypatch)
    params["min_t2_s"] = params.initial_freqs_hz.copy(
        data=[[0.002, 0.002], [0.004, 0.006]]
    )
    params["max_t2_s"] = params.initial_freqs_hz.copy(data=[[0.2, 0.2], [0.07, 0.09]])
    params["metabs_t2_ranges"].values[1] = [0.015, 0.0]
    params.attrs["metabs_t2_ranges"] = 0.04
    edits = (
        {
            "p": SpectroscopyRefitEdit(
                freq_search_range_hz=4.0,
                initial_t2_s=0.03,
                min_t2_s=0.01,
                max_t2_s=0.05,
            )
        }
        if reviewed
        else None
    )
    refit_spectroscopy_voxel(
        data, params, results, voxel_indices={"read_idx": 1}, edits=edits
    )
    np.testing.assert_array_equal(
        captured["min_t2_s"], [0.01 if reviewed else 0.004, 0.006]
    )
    np.testing.assert_array_equal(
        captured["max_t2_s"], [0.05 if reviewed else 0.07, 0.09]
    )
    np.testing.assert_array_equal(captured["t2_search_range_s"], [0.015, 0.0])


def test_t2_window_attrs_stand_in_only_without_maps(monkeypatch):
    data, params, results = _inputs()
    captured = _capture_fitter(monkeypatch)
    params = params.drop_vars("metabs_t2_ranges")
    params.attrs.update(
        metabs_t2_ranges=[0.04, 0.05], min_t2_s=[0.004, 0.006], max_t2_s=0.08
    )
    refit_spectroscopy_voxel(data, params, results, voxel_indices={"read_idx": 1})
    np.testing.assert_array_equal(captured["t2_search_range_s"], [0.04, 0.05])
    np.testing.assert_array_equal(captured["min_t2_s"], [0.004, 0.006])
    np.testing.assert_array_equal(captured["max_t2_s"], [0.08, 0.08])


@pytest.mark.parametrize("attr_value", ["absent", np.nan])
def test_window_without_map_or_attr_takes_the_full_fit_defaults(
    monkeypatch, attr_value
):
    # Without a voxel map and without a finite attr, every window value is
    # the full fit's default from fid_model, the single owner of these values.
    data, params, results = _inputs()
    captured = _capture_fitter(monkeypatch)
    params = params.drop_vars("metabs_t2_ranges")
    keys = ("metabs_freq_ranges", "metabs_t2_ranges", "min_t2_s", "max_t2_s")
    for key in keys:
        params.attrs.pop(key, None)
        if attr_value != "absent":
            params.attrs[key] = attr_value
    fid_model = refitting.fid_model
    refit_spectroscopy_voxel(data, params, results, voxel_indices={"read_idx": 1})
    np.testing.assert_array_equal(
        captured["freq_search_range_hz"], [fid_model.METABS_FREQ_RANGE_DEFAULT] * 2
    )
    np.testing.assert_array_equal(
        captured["t2_search_range_s"], [fid_model.METABS_T2_RANGE_DEFAULT] * 2
    )
    np.testing.assert_array_equal(
        captured["min_t2_s"], [fid_model.MIN_T2_S_DEFAULT] * 2
    )
    np.testing.assert_array_equal(
        captured["max_t2_s"], [fid_model.MAX_T2_S_DEFAULT] * 2
    )


def test_zero_t2_width_freezes_t2_in_the_real_refit():
    # p starts at 28 ms, 2 ms off the truth; its voxel width of 0 s keeps it
    # there instead of searching the 40 ms attrs width.
    data, params, results = _inputs(np.complex128)
    results["fit_t2s"].values[1] = [0.028, 0.05]
    params["metabs_t2_ranges"].values[1] = [0.0, 0.02]
    params.attrs["metabs_t2_ranges"] = 0.04
    fitted = refit_spectroscopy_voxel(
        data, params, results, voxel_indices={"read_idx": 1}
    )
    assert fitted["fit_results"].fit_t2s.sel(metabolite="p").item() == 0.028


def test_lorentz_refit_recomputes_both_linewidths_from_the_new_t2():
    # The stored widths belong to T2* values of 20 and 40 ms; the refit finds
    # 30 and 50 ms, so both widths must follow the refitted T2*.
    data, params, results = _inputs(np.complex128)
    stale_t2s = np.array([[0.02, 0.04], [0.02, 0.04]])
    results["fit_fwhm_hz"] = results.fit_t2s.copy(data=1 / (np.pi * stale_t2s))
    results["fit_magnitude_fwhm_hz"] = np.sqrt(3) * results.fit_fwhm_hz
    for name, definition in (
        ("fit_fwhm_hz", "absorption"),
        ("fit_magnitude_fwhm_hz", "magnitude"),
    ):
        results[name].attrs = {"unit": "Hz", "definition": definition}
    before = results.copy(deep=True)

    fitted = refit_spectroscopy_voxel(
        data, params, results, voxel_indices={"read_idx": 1}
    )["fit_results"]

    np.testing.assert_allclose(fitted.fit_t2s, [0.03, 0.05], rtol=1e-12)
    np.testing.assert_allclose(
        fitted.fit_fwhm_hz, 1 / (np.pi * fitted.fit_t2s), rtol=1e-12
    )
    np.testing.assert_allclose(
        fitted.fit_magnitude_fwhm_hz,
        np.sqrt(3) / (np.pi * fitted.fit_t2s),
        rtol=1e-12,
    )
    assert fitted.fit_fwhm_hz.attrs == {"unit": "Hz", "definition": "absorption"}
    assert fitted.fit_magnitude_fwhm_hz.attrs["definition"] == "magnitude"
    xr.testing.assert_identical(results, before)


def test_refit_relabels_bound_hits_of_enabled_lines_only():
    # A reviewed window of 10 to 14 Hz stops p (true 15 Hz) at its upper
    # edge; its T2* of 30 ms lies inside 5 to 100 ms. The disabled line l
    # keeps its earlier labels, as it keeps its earlier values.
    data, params, results = _inputs(np.complex128)
    for name in ("fit_freq_status", "fit_t2_status"):
        results[name] = results.fit_freqs.copy(data=np.full((2, 2), "not_fitted"))
        results[name].attrs = {"description": "kept"}
    edits = {
        "p": SpectroscopyRefitEdit(
            initial_freq_hz=12.0,
            freq_search_range_hz=4.0,
            initial_t2_s=0.03,
            min_t2_s=0.005,
            max_t2_s=0.1,
        ),
        "l": SpectroscopyRefitEdit(enabled=False),
    }

    fitted = refit_spectroscopy_voxel(
        data, params, results, voxel_indices={"read_idx": 1}, edits=edits
    )["fit_results"]

    assert fitted.fit_freqs.sel(metabolite="p").item() == pytest.approx(14.0)
    assert fitted.fit_freq_status.values.tolist() == ["at_upper_bound", "not_fitted"]
    assert fitted.fit_t2_status.values.tolist() == ["estimated", "not_fitted"]
    assert fitted.fit_freq_status.attrs == {"description": "kept"}
    assert results.fit_freq_status.values.tolist() == [["not_fitted"] * 2] * 2


def test_negative_t2_width_in_the_voxel_map_is_rejected():
    # The full fit rejects this map value; the refit must not replace it.
    data, params, results = _inputs(np.complex128)
    params["metabs_t2_ranges"].values[1] = [0.02, -0.03]
    params.attrs["metabs_t2_ranges"] = [0.04, 0.05]
    with pytest.raises(ValueError, match="metabs_t2_ranges must be finite"):
        refit_spectroscopy_voxel(data, params, results, voxel_indices={"read_idx": 1})


@pytest.mark.parametrize("reviewed", [False, True])
def test_zero_width_freezes_the_frequency_in_the_real_refit(reviewed):
    # A 0 Hz width freezes the frequency in the full fit. The refit keeps p
    # at its start 1 Hz off the truth, whether the width comes from the map
    # or from a reviewed row, instead of searching the 4 Hz attrs window or
    # rejecting open bounds.
    data, params, results = _inputs(np.complex128)
    params["initial_freqs_hz"].values[1] = [16.0, -25.0]
    params["metabs_freq_ranges"] = params.initial_freqs_hz.copy(
        data=[[4.0, 4.0], [4.0 if reviewed else 0.0, 4.0]]
    )
    edits = (
        {
            "p": SpectroscopyRefitEdit(
                freq_search_range_hz=0.0,
                initial_t2_s=0.03,
                min_t2_s=0.005,
                max_t2_s=0.1,
            )
        }
        if reviewed
        else None
    )
    fitted = refit_spectroscopy_voxel(
        data, params, results, voxel_indices={"read_idx": 1}, edits=edits
    )
    assert fitted["fit_results"].fit_freqs.sel(metabolite="p").item() == 16.0


def test_stale_dataset_bounds_follow_start_and_range_with_a_warning(monkeypatch):
    data, params, results = _inputs()
    captured = _capture_fitter(monkeypatch)
    params["metabs_freq_ranges"] = xr.full_like(params.initial_freqs_hz, 8.0)
    params["freq_range_lower_bound_hz"] = xr.full_like(params.initial_freqs_hz, 100.0)
    params["freq_range_upper_bound_hz"] = xr.full_like(params.initial_freqs_hz, 200.0)
    with pytest.warns(UserWarning, match=r"refit_spectroscopy_voxel.*\['l', 'p'\]"):
        refit_spectroscopy_voxel(data, params, results, voxel_indices={"read_idx": 1})
    np.testing.assert_allclose(
        captured["freq_search_range_lower_bound_hz"], [11.0, -29.0], rtol=1e-12
    )
    np.testing.assert_allclose(
        captured["freq_search_range_upper_bound_hz"], [19.0, -21.0], rtol=1e-12
    )
    assert params.freq_range_lower_bound_hz.values.tolist() == [[100.0, 100.0]] * 2


def test_selected_data_indices_clamp_and_amplitude_settings_match_fitter(monkeypatch):
    data, params, results = _inputs()
    captured = _capture_fitter(monkeypatch)
    params.attrs.update(amp_solver="ridge", amp_ridge_lambda=0.05, pseudo_inv=False)
    refit_spectroscopy_voxel(
        data,
        params,
        results,
        voxel_indices={"read_idx": 1},
        data_indices={"read_idx": 100},
    )
    np.testing.assert_array_equal(captured["data_slice"], data.isel(read_idx=1))
    assert captured["amplitude_options"]["amp_solver"] == "ridge"
    assert captured["amplitude_options"]["amp_ridge_lambda"] == 0.05
    assert captured["amplitude_options"]["pseudo_inv"] is False


@pytest.mark.parametrize(
    ("attrs", "expected"),
    [({}, "amplitude"), ({"search_order": " Metabolite"}, "metabolite")],
)
def test_search_order_reaches_the_fitter(monkeypatch, attrs, expected):
    data, params, results = _inputs()
    captured = _capture_fitter(monkeypatch)
    params.attrs.update(attrs)
    fitted = refit_spectroscopy_voxel(
        data, params, results, voxel_indices={"read_idx": 1}
    )
    assert captured["search_order"] == expected
    assert fitted["effective_options"]["search_order"] == expected


def test_unknown_search_order_fails_before_fitting(monkeypatch):
    data, params, results = _inputs()
    captured = _capture_fitter(monkeypatch)
    params.attrs["search_order"] = "size"
    with pytest.raises(ValueError, match="search_order"):
        refit_spectroscopy_voxel(data, params, results, voxel_indices={"read_idx": 1})
    assert "metabs" not in captured


def test_all_disabled_skips_numerical_fit_and_ui(monkeypatch):
    from ipywidgets import Widget

    inputs = _inputs()
    inputs[1].attrs["niter"] = "unused invalid count"

    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled refit must not fit or construct UI.")

    monkeypatch.setattr(
        refitting.fid_model, "_fit_single_spectrum_pseudo_inv", forbidden
    )
    monkeypatch.setattr(refitting.fid_model, "solve_for_complex_amplitudes", forbidden)
    monkeypatch.setattr(Widget, "__init__", forbidden)
    fitted = refit_spectroscopy_voxel(
        *inputs,
        voxel_indices={"read_idx": 1},
        edits={name: SpectroscopyRefitEdit(enabled=False) for name in ["p", "l"]},
    )
    assert fitted["status"] == "no_enabled_metabolites"
    assert fitted["active_metabolites"] == []
    xr.testing.assert_identical(fitted["fit_results"], inputs[2].isel(read_idx=1))


def test_fitter_failure_propagates_without_input_mutation(monkeypatch):
    inputs = _inputs()
    before = tuple(value.copy(deep=True) for value in inputs)

    def fail(**kwargs):
        raise ValueError("invented fitter failure")

    monkeypatch.setattr(refitting.fid_model, "_fit_single_spectrum_pseudo_inv", fail)
    with pytest.raises(ValueError, match="invented fitter failure"):
        refit_spectroscopy_voxel(*inputs, voxel_indices={"read_idx": 1})
    for value, original in zip(inputs, before, strict=True):
        xr.testing.assert_identical(value, original)


def test_amplitude_refinement_error_is_explicit_and_fitter_result_is_kept(monkeypatch):
    _capture_fitter(monkeypatch)
    error = RuntimeError("invented amplitude failure")

    def fail(**kwargs):
        raise error

    monkeypatch.setattr(refitting.fid_model, "solve_for_complex_amplitudes", fail)
    fitted = refit_spectroscopy_voxel(*_inputs(), voxel_indices={"read_idx": 1})
    np.testing.assert_allclose(
        fitted["fit_results"].fit_amps, [2 + 3j, 2 + 3j], rtol=1e-12
    )
    assert fitted["amplitude_refinement_error"] is error


@pytest.mark.parametrize(
    "source", ["aligned", "legacy", "dwell", "inferred", "positional"]
)
def test_sampling_metadata_priority_and_legacy_cut_are_preserved(monkeypatch, source):
    data, params, results = _inputs()
    _capture_fitter(monkeypatch)
    data_obj = None
    if source == "aligned":
        params.attrs.update(cut_off=2, sampling_time_axis_s=np.arange(66) * 0.005)
        expected = np.arange(64) * 0.001
    else:
        data.acquisition_idx.attrs.clear()
        if source == "legacy":
            params.attrs.update(cut_off=2, sampling_time_axis_s=np.arange(66) * 0.005)
            expected = np.arange(66)[2:] * 0.005
        elif source == "dwell":
            data.attrs["dwell_time_s"] = 0.002
            params.attrs["dwell_time_s"] = 0.003
            data_obj = SimpleNamespace(dwell_time_s=0.004)
            expected = np.arange(64) * 0.002
        elif source == "inferred":
            params.attrs["sampling_time_axis_s"] = np.arange(10) * 0.003
            expected = np.arange(64) * 0.003
        else:
            expected = np.arange(64)
    fitted = refit_spectroscopy_voxel(
        data, params, results, voxel_indices={"read_idx": 1}, scan=data_obj
    )
    np.testing.assert_allclose(
        fitted["sampling_time_axis_s"], expected, rtol=1e-12, atol=1e-12
    )


def test_edit_is_immutable_and_unknown_fields_are_rejected():
    edit = SpectroscopyRefitEdit()
    with pytest.raises(FrozenInstanceError):
        edit.enabled = False
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        SpectroscopyRefitEdit(unknown_field=2)


@pytest.mark.parametrize(
    "edits, error, match",
    [
        ({"p": {}}, TypeError, "SpectroscopyRefitEdit"),
        ({"unknown": SpectroscopyRefitEdit()}, ValueError, "unknown metabolite"),
        (
            {"p": SpectroscopyRefitEdit(freq_search_range_hz=2)},
            ValueError,
            "requires initial_t2_s",
        ),
        *(
            (
                {
                    "p": SpectroscopyRefitEdit(
                        freq_search_range_hz=width,
                        initial_t2_s=0.03,
                        min_t2_s=0.005,
                        max_t2_s=0.1,
                    )
                },
                ValueError,
                r"'p' has freq_search_range_hz=.*finite full width >= 0 Hz",
            )
            for width in (-4.0, np.nan, np.inf)
        ),
    ],
)
def test_invalid_edits_name_the_problem(edits, error, match):
    with pytest.raises(error, match=match):
        refit_spectroscopy_voxel(*_inputs(), voxel_indices={"read_idx": 1}, edits=edits)


def test_mixed_lineshape_and_missing_sampling_dimension_fail_before_fitting():
    data, params, results = _inputs()
    params.attrs["lineshape"] = "voigt"
    with pytest.raises(ValueError, match="lineshape"):
        refit_spectroscopy_voxel(data, params, results, voxel_indices={"read_idx": 1})
    with pytest.raises(ValueError, match="acquisition_idx"):
        refit_spectroscopy_voxel(
            data.rename(acquisition_idx="unknown"),
            params,
            results,
            voxel_indices={"read_idx": 1},
        )


def test_sequence_keeps_its_private_sampling_helper_alias():
    from hypermri.analysis.spectroscopy.schema import (
        _sampling_time_axis_from_data,
    )
    from hypermri.sequences import base_spectroscopy

    assert (
        base_spectroscopy._sampling_time_axis_from_data is _sampling_time_axis_from_data
    )


def test_voigt_refit_recovers_shared_gaussian_width_and_updates_linewidths():
    """A Voigt voxel refit re-estimates G and the derived linewidth outputs.

    Fixture: a three-line synthetic Voigt FID (-300/0/250 Hz, G = 14 Hz, T2* 0.05/0.04/0.06 s, noise sigma 0.02). The refit starts from a
    degraded result (G = 0 Hz, T2* = 0.03 s). Tolerances: 0.5 Hz for G and 1 %
    for amplitude magnitudes, with margin over the observed 0.2 Hz and 0.5 %.
    """
    from hypermri.analysis.fitting.models.spectroscopy import fid_model
    from hypermri.analysis.fitting.spectroscopy import (
        def_fit_params,
        fit_data_pseudo_inv,
    )

    freqs_hz = np.array([-300.0, 0.0, 250.0])
    amplitudes = np.array([10.0, 30.0, 3.0]) * np.exp(1j * np.array([0.3, -0.2, 0.5]))
    rng = np.random.default_rng(0)
    t = np.arange(256) / 4000.0
    lines = amplitudes[:, None] * np.exp(
        2j * np.pi * freqs_hz[:, None] * t - t / np.array([0.05, 0.04, 0.06])[:, None]
    )
    signal = lines.sum(0) * np.exp(-((np.pi * 14.0 * t) ** 2) / (4 * np.log(2)))
    noise = 0.02 * (
        rng.standard_normal((2, t.size)) + 1j * rng.standard_normal((2, t.size))
    )
    data = xr.DataArray(
        signal[None] + noise, dims=["x", "acquisition_idx"], coords={"x": [0, 1]}
    )
    data.attrs["sampling_time_axis_s"] = t
    params = def_fit_params(
        data=data,
        fit_params={
            "metabs": ["a", "b", "c"],
            "metabs_freqs_hz": list(freqs_hz + np.array([2.0, -1.5, 1.0])),
            "initial_t2s_s": 0.03,
            "metabs_freq_ranges": 10.0,
            "lineshape": "voigt",
        },
    )
    degraded = fit_data_pseudo_inv(data, params)
    degraded["fit_gauss_fwhm_hz"][:] = 0.0
    degraded["fit_t2s"][:] = 0.03
    before = degraded.copy(deep=True)

    out = refit_spectroscopy_voxel(data, params, degraded, voxel_indices={"x": 1})
    fitted = out["fit_results"]

    assert degraded.identical(before)
    assert out["effective_options"]["lineshape"] == "voigt"
    assert out["effective_options"]["initial_gauss_fwhm_hz"] == 0.0
    gauss = float(fitted["fit_gauss_fwhm_hz"])
    assert gauss == pytest.approx(14.0, abs=0.5)
    assert str(fitted["fit_gauss_fwhm_status"].values) == "estimated"
    lorentz = 1.0 / (np.pi * fitted["fit_t2s"].values)
    np.testing.assert_allclose(fitted["fit_lorentz_fwhm_hz"].values, lorentz)
    np.testing.assert_allclose(
        fitted["fit_fwhm_hz"].values, fid_model._voigt_fwhm_hz(lorentz, gauss)
    )
    np.testing.assert_allclose(
        fitted["fit_magnitude_fwhm_hz"].values,
        fid_model._magnitude_fwhm_hz(lorentz, gauss),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.abs(fitted["fit_amps"].values), np.abs(amplitudes), rtol=0.01
    )
