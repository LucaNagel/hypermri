"""Order in which the FID grid search visits the lines (``search_order``)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.fitting import spectroscopy
from hypermri.analysis.fitting.models.spectroscopy import fid_model as model
from hypermri.analysis.fitting.models.spectroscopy.diagnostics import HISTORY_EVENTS

TIME_S = np.arange(256) / 2000.0
SOLVER_OPTIONS = model._prepare_amplitude_solver_kwargs(
    {"sampling_time_axis_s": TIME_S}
)


def doublet(*, ratio=5.0, spacing_hz=12.0, noise=0.0, voxels=1):
    """Strong line at 0 Hz and a weaker one ``spacing_hz`` above it.

    T2* 30 and 25 ms, phases -0.2 and 0.5 rad, areas 20 and 20 / ratio.
    Noise is complex white Gaussian with seed 31; voxels share the clean FID.
    """
    clean = 20.0 * np.exp(-0.2j) * np.exp(-TIME_S / 0.030) + (20.0 / ratio) * np.exp(
        0.5j
    ) * np.exp(2j * np.pi * spacing_hz * TIME_S - TIME_S / 0.025)
    rng = np.random.default_rng(31)
    values = clean[None, :] + noise * (
        rng.normal(size=(voxels, TIME_S.size))
        + 1j * rng.normal(size=(voxels, TIME_S.size))
    )
    data = xr.DataArray(
        values, dims=["x", "acquisition_idx"], coords={"x": np.arange(voxels)}
    )
    data.attrs["sampling_time_axis_s"] = TIME_S
    return data


def params(data, **options):
    """Weak line listed first; both starts 2 Hz above the truth."""
    return spectroscopy.def_fit_params(
        data,
        {
            "metabs": ["weak", "strong"],
            "metabs_freqs_hz": [14.0, 2.0],
            "initial_t2s_s": 0.03,
            "metabs_freq_ranges": 8.0,
            "min_t2_s": 0.005,
            "max_t2_s": 0.2,
            "min_peak_separation_hz": 1.0,
            "npoints": 9,
            "niter": 2,
            "rep_fitting": 2,
            **options,
        },
    )


@pytest.mark.parametrize("value", ["size", "", 1, None, ["amplitude"]])
def test_unknown_search_order_is_rejected_by_name(value):
    with pytest.raises(ValueError, match="search_order"):
        model._prepare_global_fit_options({"search_order": value})


def test_search_order_defaults_to_amplitude_order():
    assert model._prepare_global_fit_options({})["search_order"] == "amplitude"
    assert model.SEARCH_ORDER_DEFAULT == "amplitude"


@pytest.mark.parametrize(
    ("search_order", "frequencies_hz"),
    [("metabolite", [0.0, 12.0]), ("amplitude", [0.0])],
    ids=["metabolite-order", "single-line"],
)
def test_visit_order_needs_no_amplitude_solve(
    monkeypatch, search_order, frequencies_hz
):
    def fail(*args, **kwargs):
        raise AssertionError("this visit order must not solve amplitudes")

    monkeypatch.setattr(model, "solve_for_complex_amplitudes", fail)
    frequencies = np.array(frequencies_hz)
    order = model._search_visit_order(
        TIME_S.astype(complex),
        frequencies,
        np.full(frequencies.size, 0.03),
        {},
        {"search_order": search_order},
    )
    assert order is None


def test_explicit_amplitude_order_reproduces_the_default_fit():
    data = doublet(noise=0.3, voxels=2)
    default = spectroscopy.fit_data_pseudo_inv(data, params(data))
    explicit = spectroscopy.fit_data_pseudo_inv(
        data, params(data, search_order="amplitude")
    )
    for name in ("fit_amps", "fit_freqs", "fit_t2s", "fit_rss"):
        np.testing.assert_array_equal(explicit[name], default[name])


def test_amplitude_order_ranks_lines_by_their_fitted_signal():
    data = doublet().isel(x=0).values
    frequencies = np.array([12.0, 0.0])
    t2s = np.array([0.025, 0.030])
    order = model._search_visit_order(
        data, frequencies, t2s, SOLVER_OPTIONS, {"search_order": "amplitude"}
    )
    np.testing.assert_array_equal(order, [1, 0])


def test_amplitude_order_ranks_a_narrow_line_above_a_broad_line_of_larger_area():
    # Areas 10 (T2* 3 ms) and 6 (T2* 60 ms): the narrow line carries about
    # 2.6 times the signal norm of the broad one and is visited first.
    data = 10.0 * np.exp(-TIME_S / 0.003) + 6.0 * np.exp(
        2j * np.pi * 40.0 * TIME_S - TIME_S / 0.060
    )
    order = model._search_visit_order(
        data,
        np.array([0.0, 40.0]),
        np.array([0.003, 0.060]),
        SOLVER_OPTIONS,
        {"search_order": "amplitude"},
    )
    np.testing.assert_array_equal(order, [1, 0])


def test_amplitude_order_keeps_slot_order_for_equal_lines(monkeypatch):
    def equal_components(values, freq_hz, t2_star_s, **options):
        return np.ones((TIME_S.size, len(freq_hz)), dtype=complex), np.ones(3)

    monkeypatch.setattr(model, "solve_for_complex_amplitudes", equal_components)
    order = model._search_visit_order(
        TIME_S.astype(complex),
        np.array([0.0, 20.0, 40.0]),
        np.full(3, 0.03),
        {},
        {"search_order": "amplitude"},
    )
    np.testing.assert_array_equal(order, [0, 1, 2])


def test_search_moves_follow_the_line_order():
    coupling = [(3, 0, 1.0)]
    bands = [(2, 1, 0.5, 2.0)]
    moves = model._t2_search_moves(4, coupling, bands, visit_order=[2, 3, 1, 0])
    assert moves == [(2, True), (1, True), (1, False), (0, True)]
    assert model._t2_search_moves(4, coupling, bands) == [
        (0, True),
        (1, True),
        (1, False),
        (2, True),
    ]


def first_frequency_candidates(result):
    count = result.fit_history_count.item()
    info = result.fit_history_info.isel(history_step=slice(0, count))
    events = info.sel(history_field="event").values
    index = info.sel(history_field="metabolite_index").values
    return index[events == HISTORY_EVENTS["frequency_candidate"]]


@pytest.mark.parametrize(
    ("search_order", "first"), [("metabolite", "weak"), ("amplitude", "strong")]
)
def test_frequency_search_visits_lines_in_the_chosen_order(search_order, first):
    data = doublet()
    result = spectroscopy.fit_data_pseudo_inv(
        data,
        params(data, search_order=search_order, store_optimization_history=True),
    )
    visited = first_frequency_candidates(result.isel(x=0))
    names = list(result.metabolite.values)
    assert names[visited[0]] == first


def test_amplitude_order_is_deterministic_in_serial_and_parallel_fits():
    data = doublet(noise=0.3, voxels=4)
    options = params(data, search_order="amplitude")
    serial = spectroscopy.fit_data_pseudo_inv(data, options, n_jobs=1)
    parallel = spectroscopy.fit_data_pseudo_inv(data, options, n_jobs=2)
    for name in ("fit_amps", "fit_freqs", "fit_t2s", "fit_rss"):
        np.testing.assert_allclose(parallel[name], serial[name], rtol=0, atol=1e-12)


@pytest.mark.parametrize("listing", [["weak", "strong"], ["strong", "weak"]])
def test_amplitude_order_recovers_a_weak_neighbour_in_either_listing(listing):
    """Regression case from the synthetic search-order study (November 2026).

    Noise-free doublet sampled at 4 kHz with 512 points: strong line at 0 Hz,
    a ten times weaker line 5 Hz above it, both starts 4 Hz too high, search
    range 40 Hz and default grid settings. In metabolite order with the weak
    line listed first the grid alone leaves the weak line near 2.8 Hz; in
    amplitude order both listings reach the truth. The joint refinement after
    the grid passes is off for the metabolite-order failure, which it would
    otherwise repair. 0.05 Hz is far below the 5 Hz spacing and above the
    grid's final resolution in this case.
    """
    time_s = np.arange(512) / 4000.0
    fid = 20.0 * np.exp(-0.2j) * np.exp(-time_s / 0.030) + 2.0 * np.exp(0.5j) * np.exp(
        2j * np.pi * 5.0 * time_s - time_s / 0.025
    )
    data = xr.DataArray(fid[None, :], dims=["x", "acquisition_idx"])
    data.attrs["sampling_time_axis_s"] = time_s
    starts_hz = {"strong": 4.0, "weak": 9.0}

    def fitted_frequencies(search_order, metabs, *, joint_refinement=True):
        options = spectroscopy.def_fit_params(
            data,
            {
                "metabs": metabs,
                "metabs_freqs_hz": [starts_hz[name] for name in metabs],
                "initial_t2s_s": 0.03,
                "metabs_freq_ranges": 40.0,
                "min_t2_s": 0.005,
                "max_t2_s": 0.2,
                "min_peak_separation_hz": 1.0,
                "search_order": search_order,
                "joint_refinement": joint_refinement,
            },
        )
        result = spectroscopy.fit_data_pseudo_inv(data, options)
        return result.fit_freqs.sel(metabolite=["strong", "weak"]).values.ravel()

    np.testing.assert_allclose(
        fitted_frequencies("amplitude", listing), [0.0, 5.0], atol=0.05
    )
    if listing[0] == "weak":
        grid_alone = fitted_frequencies("metabolite", listing, joint_refinement=False)
        assert abs(grid_alone[1] - 5.0) > 1.0
        np.testing.assert_allclose(
            fitted_frequencies("metabolite", listing), [0.0, 5.0], atol=0.05
        )
