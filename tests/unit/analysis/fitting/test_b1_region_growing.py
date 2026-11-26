"""Characterization of the B1 region-growing fit.

The data are invented: a noise-free sinusoid ``A sin(pi sqrt(P) / T)`` over
reference powers ``P`` in W, with a period ``T`` in sqrt(W) that changes
smoothly over a small read x phase grid. The tests fix the growth order, the
neighbour start values, the return shapes and the debug maps, so that a split
of the function into named steps can be checked against them.
"""

from collections import deque

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.fitting.models.quantitative import b1 as b1_model

READ, PHASE = 3, 4
REFPOWER_W = np.linspace(0.0, 16.0, 17)


def _true_period(read_idx: int, phase_idx: int) -> float:
    return 6.0 + 0.25 * read_idx + 0.1 * phase_idx


def _invented_b1_data() -> xr.DataArray:
    values = np.empty((READ, PHASE, REFPOWER_W.size), dtype=complex)
    for r in range(READ):
        for p in range(PHASE):
            amplitude = (1.0 + r + 2 * p) * np.exp(0.3j)
            values[r, p] = amplitude * np.sin(
                np.pi * np.sqrt(REFPOWER_W) / _true_period(r, p)
            )
    return xr.DataArray(
        values[:, :, None, :],
        dims=("read_idx", "phase_idx", "slice_idx", "rfpower_idx"),
    )


def _fit_params(data: xr.DataArray, seed=(1, 1), **overrides) -> dict:
    params = {
        "refpower_W": REFPOWER_W,
        "_b1_period_grid": np.linspace(4.0, 9.0, 501),
        "period_min": 4.0,
        "period_max": 9.0,
        "period_samples": 401,
        "seed_voxel": seed,
    }
    params.update(overrides)
    seed_signal = data.isel(read_idx=seed[0], phase_idx=seed[1], slice_idx=0).values
    params["seed_fit_results"] = b1_model.fit_single_voxel_b1_period_sweep_helper(
        signal=seed_signal,
        sqrt_refpower=np.sqrt(REFPOWER_W),
        fit_params=params,
    )
    return params


def _breadth_first_order(seed, offsets) -> np.ndarray:
    """Independent reference of the FIFO growth order on the full grid."""
    order = np.full((READ, PHASE), np.nan)
    order[seed] = 0
    queue = deque([seed])
    n = 1
    while queue:
        r, p = queue.popleft()
        for dr, dp in offsets:
            nr, np_ = r + dr, p + dp
            if 0 <= nr < READ and 0 <= np_ < PHASE and np.isnan(order[nr, np_]):
                order[nr, np_] = n
                n += 1
                queue.append((nr, np_))
    return order


FOUR = [(-1, 0), (1, 0), (0, -1), (0, 1)]
EIGHT = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


@pytest.mark.parametrize("connectivity,offsets", [(4, FOUR), (8, EIGHT)])
def test_fifo_growth_follows_breadth_first_order(connectivity, offsets):
    data = _invented_b1_data()
    params = _fit_params(data, return_debug_info=True)

    results, first, debug = b1_model.region_growing_b1_fitting(
        data, params, connectivity=connectivity
    )

    np.testing.assert_array_equal(
        debug["fit_order_map"], _breadth_first_order((1, 1), offsets)
    )
    assert debug["region_mask"].all()
    assert set(results) == {(r, p) for r in range(READ) for p in range(PHASE)}
    # The seed has no stored first iteration; every grown voxel has one.
    assert set(first) == set(results) - {(1, 1)}
    assert set(debug["init_params_map"]) == set(results) - {(1, 1)}


def test_every_voxel_recovers_its_invented_period():
    data = _invented_b1_data()
    params = _fit_params(data)

    results, _ = b1_model.region_growing_b1_fitting(data, params)

    # The local grid spans +-30 % of the neighbour period in 401 steps, so
    # the grid spacing is below 0.01 sqrt(W); noise-free data hit the
    # nearest grid point.
    for (r, p), fit in results.items():
        assert fit["period"] == pytest.approx(_true_period(r, p), abs=0.01)
        assert abs(fit["amplitude"]) == pytest.approx(1.0 + r + 2 * p, rel=1e-3)


def test_grown_voxels_start_from_the_median_of_fitted_neighbours():
    data = _invented_b1_data()
    params = _fit_params(data, return_debug_info=True)

    results, _, debug = b1_model.region_growing_b1_fitting(data, params)

    order = debug["fit_order_map"]
    for coord, init in debug["init_params_map"].items():
        r, p = coord
        earlier = [
            results[(r + dr, p + dp)]["period"]
            for dr in (-1, 0, 1)
            for dp in (-1, 0, 1)
            if 0 <= r + dr < READ
            and 0 <= p + dp < PHASE
            and order[r + dr, p + dp] < order[r, p]
        ]
        assert init["neighbor_params"]["n_neighbors"] == len(earlier)
        assert init["neighbor_params"]["period"] == pytest.approx(np.median(earlier))
        assert init["merged_fit_params"]["period"] == pytest.approx(np.median(earlier))


def test_stop_index_comes_from_the_first_fitted_period():
    data = _invented_b1_data()
    params = _fit_params(data, return_debug_info=True, stop_index_extra=2)

    results, first, debug = b1_model.region_growing_b1_fitting(data, params)

    sqrt_refpower = np.sqrt(REFPOWER_W)
    for coord, fit in first.items():
        expected = np.searchsorted(sqrt_refpower, fit["period"]) + 2
        expected = max(min(expected, REFPOWER_W.size - 1), 2)
        assert results[coord]["stop_index"] == expected
        assert debug["stop_index_map"][coord] == expected


def test_single_iteration_returns_only_the_results():
    data = _invented_b1_data()
    params = _fit_params(data, use_two_iteration_fit=False)

    results = b1_model.region_growing_b1_fitting(data, params)

    assert isinstance(results, dict)
    assert len(results) == READ * PHASE


def test_max_iterations_stops_the_growth():
    data = _invented_b1_data()
    params = _fit_params(data, return_debug_info=True)

    results, _, debug = b1_model.region_growing_b1_fitting(
        data, params, max_iterations=1
    )

    # One iteration fits the four direct neighbours of the seed.
    assert len(results) == 5
    assert np.nansum(np.isfinite(debug["fit_order_map"])) == 5
    assert debug["region_mask"].sum() == 5


def test_signal_priority_visits_the_strongest_frontier_voxel_first():
    data = _invented_b1_data()
    params = _fit_params(data, return_debug_info=True, region_growing_priority="signal")

    _, _, debug = b1_model.region_growing_b1_fitting(data, params)

    order = debug["fit_order_map"]
    assert sorted(order.ravel().tolist()) == list(range(READ * PHASE))
    # The seed's four neighbours get 1 to 4. Amplitude grows fastest with
    # phase_idx, so growth continues from (1, 2), the strongest of them, and
    # its unfitted neighbours follow in offset order.
    assert order[0, 2] == 5
    assert order[2, 2] == 6
    assert order[1, 3] == 7


@pytest.mark.parametrize(
    "seed,match",
    [(None, "seed_voxel"), ("bad", "seed_voxel must be")],
)
def test_invalid_seed_fails(seed, match):
    data = _invented_b1_data()
    params = _fit_params(data)
    params["seed_voxel"] = seed

    with pytest.raises(ValueError, match=match):
        b1_model.region_growing_b1_fitting(data, params)


def test_missing_seed_result_fails():
    data = _invented_b1_data()
    params = _fit_params(data)
    params["seed_fit_results"] = None

    with pytest.raises(ValueError, match="seed_fit_results"):
        b1_model.region_growing_b1_fitting(data, params)


def test_invalid_connectivity_fails():
    data = _invented_b1_data()

    with pytest.raises(ValueError, match="connectivity must be 4 or 8"):
        b1_model.region_growing_b1_fitting(data, _fit_params(data), connectivity=6)
