"""Global PCA contracts on deterministic synthetic rank-one MRI signals."""

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.denoising import pca_denoise


@pytest.mark.parametrize("method", ["eig", "svd"])
@pytest.mark.parametrize("permuted", [False, True])
@pytest.mark.parametrize("components", [1, [0], slice(0, 1), None])
def test_global_pca_preserves_rank_one_signal_and_labelled_input(
    method, permuted, components
):
    """A noiseless rank-one complex signal survives either decomposition.

    The fixture has one spatial pattern and one repetition-dependent complex
    amplitude. Its exact rank-one reconstruction is the original signal. The
    1e-6 relative/absolute tolerance allows floating-point eig/SVD roundoff in
    arbitrary signal units, without deriving a reference from the algorithm.
    """
    spatial = np.arange(12, dtype=float).reshape(3, 4) - 5.5
    temporal = np.array([1, 2j, -2, 1 - 1j, 3 + 2j])
    values = temporal[:, None, None] * spatial + (4 + 3j)
    data = xr.DataArray(
        values,
        dims=("repetition_idx", "read_idx", "phase_idx"),
        coords={
            "repetition_idx": np.arange(5) * 2.0,
            "read_idx": [-1.0, 0.0, 1.0],
            "phase_idx": [-3.0, -1.0, 1.0, 3.0],
            "x": (
                ("read_idx", "phase_idx"),
                np.broadcast_to(np.array([-1.0, 0.0, 1.0])[:, None], (3, 4)),
            ),
        },
        attrs={"unit": "a.u.", "slice_thickness_mm": 2.0},
        name="rank_one_signal",
    )
    data.coords["read_idx"].attrs["unit"] = "mm"
    if permuted:
        data = data.transpose("phase_idx", "repetition_idx", "read_idx")
    original = data.copy(deep=True)

    result = pca_denoise(
        data,
        components=components,
        pca_method=method,
        noise_estimation_method="manual",
        noise_threshold=1e-8,
    )

    xr.testing.assert_allclose(result, original, rtol=1e-6, atol=1e-6)
    assert result.dims == original.dims
    assert result.dtype == original.dtype
    estimation = (
        {} if components is not None else {"pca_noise_estimation_method": "manual"}
    )
    assert result.attrs == {**original.attrs, **estimation}
    assert result.name == original.name
    for name in original.coords:
        xr.testing.assert_identical(result.coords[name], original.coords[name])
    xr.testing.assert_identical(data, original)
