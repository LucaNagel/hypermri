import numpy as np
import xarray as xr

from hypermri.analysis.fitting.api import fit


def _attach_scan_metadata(data_array: xr.DataArray) -> xr.DataArray:
    """Attach representative scan and coordinate metadata to a test array."""
    data_array.attrs.update(
        {
            "patient_pos": "Head_Prone",
            "slice_orientation": "axial",
            "domain": "spatial",
            "signal_domain": "temporal",
            "fov_mm": (24.0, 18.0, 6.0),
            "voxel_size_mm": (2.0, 3.0, 6.0),
            "custom_scan_attr": "preserve-me",
        }
    )

    for dim, extent_mm in (
        ("read_idx", 24.0),
        ("phase_idx", 18.0),
        ("slice_idx", 6.0),
    ):
        if dim not in data_array.coords:
            continue
        data_array.coords[dim].attrs["axis_mm"] = np.linspace(
            -extent_mm / 2,
            extent_mm / 2,
            data_array.sizes[dim],
            dtype=np.float64,
        )
        data_array.coords[dim].attrs["resolution_mm"] = (
            extent_mm / data_array.sizes[dim]
        )

    return data_array


def test_t1_fit_propagates_scan_and_coordinate_metadata():
    inversion_times_ms = np.array([50.0, 150.0, 400.0, 900.0], dtype=np.float64)
    t1_true_s = np.array([0.7, 1.1], dtype=np.float64)[:, None, None, None]
    signal = 100.0 * (
        1.0
        - 2.0 * np.exp(-(inversion_times_ms[None, None, None, :] / 1000.0) / t1_true_s)
    )

    input_data = xr.DataArray(
        signal,
        dims=("read_idx", "phase_idx", "slice_idx", "inversion_idx"),
        coords={
            "read_idx": np.arange(signal.shape[0], dtype=int),
            "phase_idx": np.arange(signal.shape[1], dtype=int),
            "slice_idx": np.arange(signal.shape[2], dtype=int),
            "inversion_idx": inversion_times_ms,
        },
    )
    input_data.coords["inversion_idx"].attrs["unit"] = "ms"
    _attach_scan_metadata(input_data)

    fit_result = fit(
        data=input_data,
        model="t1",
        fit_params={
            "t1_values_s": np.linspace(0.2, 2.0, 40, dtype=np.float64),
            "use_multiprocessing": False,
        },
    )

    assert fit_result.attrs["custom_scan_attr"] == input_data.attrs["custom_scan_attr"]
    assert (
        fit_result.attrs["slice_orientation"] == input_data.attrs["slice_orientation"]
    )
    np.testing.assert_allclose(
        fit_result.coords["read_idx"].attrs["axis_mm"],
        input_data.coords["read_idx"].attrs["axis_mm"],
    )
    assert (
        fit_result.coords["read_idx"].attrs["resolution_mm"]
        == input_data.coords["read_idx"].attrs["resolution_mm"]
    )


def test_adc_fit_propagates_scan_and_coordinate_metadata():
    b_values = np.array([0.0, 250.0, 700.0, 1200.0], dtype=np.float64)
    adc_true = np.array([0.0011, 0.0008], dtype=np.float64)[:, None, None, None]
    signal = 150.0 * np.exp(-b_values[None, None, None, :] * adc_true)

    input_data = xr.DataArray(
        signal,
        dims=("read_idx", "phase_idx", "slice_idx", "bvalue_idx"),
        coords={
            "read_idx": np.arange(signal.shape[0], dtype=int),
            "phase_idx": np.arange(signal.shape[1], dtype=int),
            "slice_idx": np.arange(signal.shape[2], dtype=int),
            "bvalue_idx": b_values,
        },
    )
    input_data.coords["bvalue_idx"].attrs["unit"] = "s/mm^2"
    _attach_scan_metadata(input_data)

    fit_result = fit(
        data=input_data,
        model="adc",
        fit_params={
            "b_values": b_values,
            "use_multiprocessing": False,
        },
    )

    assert fit_result.attrs["custom_scan_attr"] == input_data.attrs["custom_scan_attr"]
    np.testing.assert_allclose(
        fit_result.coords["phase_idx"].attrs["axis_mm"],
        input_data.coords["phase_idx"].attrs["axis_mm"],
    )
    assert fit_result.attrs["voxel_size_mm"] == input_data.attrs["voxel_size_mm"]


def test_ideal_fit_uses_shared_metadata_propagation(monkeypatch):
    from hypermri.analysis.fitting.ideal import IDEAL_Solver

    input_data = xr.DataArray(
        np.ones((2, 1, 1, 3), dtype=np.complex128),
        dims=("read_idx", "phase_idx", "slice_idx", "echo_idx"),
        coords={
            "read_idx": np.arange(2, dtype=int),
            "phase_idx": np.array([0], dtype=int),
            "slice_idx": np.array([0], dtype=int),
            "echo_idx": np.arange(3, dtype=int),
        },
    )
    _attach_scan_metadata(input_data)

    def _fake_solve_xarray(
        self,
        data,
        echo_dim="echo_idx",
        metabolite_names=None,
        **solve_kwargs,
    ):
        return xr.Dataset(
            {
                "fit_amps": xr.DataArray(
                    np.zeros(
                        (
                            data.sizes["read_idx"],
                            data.sizes["phase_idx"],
                            data.sizes["slice_idx"],
                            1,
                        ),
                        dtype=np.complex128,
                    ),
                    dims=("read_idx", "phase_idx", "slice_idx", "metabolite"),
                    coords={"metabolite": metabolite_names or ["water"]},
                )
            },
            attrs={"solver_attr": "keep-me"},
        )

    monkeypatch.setattr(IDEAL_Solver, "solve_xarray", _fake_solve_xarray)

    fit_result = fit(
        data=input_data,
        model="ideal",
        fit_params={
            "relative_frequencies_hz": np.array([0.0], dtype=np.float64),
            "echo_spacing_s": 0.001,
            "metabolite_names": ["water"],
            "use_multiprocessing": False,
        },
    )

    assert fit_result.attrs["custom_scan_attr"] == input_data.attrs["custom_scan_attr"]
    assert fit_result.attrs["solver_attr"] == "keep-me"
    np.testing.assert_allclose(
        fit_result.coords["read_idx"].attrs["axis_mm"],
        input_data.coords["read_idx"].attrs["axis_mm"],
    )
