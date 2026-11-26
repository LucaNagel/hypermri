"""Small amplitude-only checks for solver selection and numerical compatibility."""

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.fitting.models.spectroscopy import fid_model as fid_pseudo_inv
from hypermri.analysis.fitting.spectroscopy import def_fit_params


@pytest.mark.parametrize(
    ("solver", "options_to_check"),
    [
        (
            "pinv",
            [
                {},
                {"amp_solver": "pinv", "pseudo_inv": False},
                {"amp_solver": " PSEUDO_INV "},
                {"amp_solver": "pseudoinv"},
            ],
        ),
        (
            "normal_eq",
            [
                {"pseudo_inv": False},
                {"amp_solver": "normal_eq", "pseudo_inv": True},
                {"amp_solver": " NORMAL "},
            ],
        ),
        (
            "ridge",
            [
                {"amp_solver": "ridge", "pseudo_inv": False},
                {"amp_solver": " TIKHONOV "},
            ],
        ),
    ],
)
def test_solver_selection_preserves_amplitudes(solver, options_to_check):
    """Check aliases/defaults in preparation and direct FID/spectrum solving.

    Use independent complex least-squares references for pinv/normal_eq and
    augmented ridge. The noisy, overlapping peaks expose the former incorrect
    plain transpose in normal_eq. No frequency/T2* search is needed.
    """
    time_s = np.arange(32) * 0.0004
    frequencies = np.array([-70.0, -65.0])
    t2_s = np.array([0.025, 0.03])
    basis = np.exp(2j * np.pi * time_s[:, None] * frequencies - time_s[:, None] / t2_s)
    rng = np.random.default_rng(20260904)
    fid = basis @ np.array([1.0 + 0.3j, 0.2 - 0.1j])
    fid += 0.02 * (rng.normal(size=time_s.size) + 1j * rng.normal(size=time_s.size))
    original_fid = fid.copy()
    data = xr.DataArray(
        fid,
        dims="acquisition_idx",
        coords={"acquisition_idx": time_s},
        attrs={"signal_domain": "temporal"},
    )
    ridge_lambda = 0.25
    if solver in {"pinv", "normal_eq"}:
        expected = np.linalg.lstsq(basis, fid, rcond=None)[0]
    elif solver == "ridge":
        expected = np.linalg.lstsq(
            np.vstack([basis, np.sqrt(ridge_lambda) * np.eye(2)]),
            np.concatenate([fid, np.zeros(2)]),
            rcond=None,
        )[0]

    for options in options_to_check:
        params = def_fit_params(
            data,
            {
                "metabs": ["peak_a", "peak_b"],
                "metabs_freqs_hz": frequencies,
                "initial_t2s_s": t2_s,
                "amp_ridge_lambda": ridge_lambda,
                **options,
            },
        )
        assert params.attrs["amp_solver"] == solver
        assert params.attrs["amp_ridge_lambda"] == ridge_lambda
        for domain in ("temporal", "spectral"):
            input_signal = (
                fid if domain == "temporal" else np.fft.fftshift(np.fft.fft(fid))
            )
            components, amplitudes = fid_pseudo_inv.solve_for_complex_amplitudes(
                input_signal,
                frequencies,
                t2_s,
                time_s,
                signal_domain=domain,
                amp_ridge_lambda=ridge_lambda,
                **options,
            )
            np.testing.assert_allclose(amplitudes, expected, rtol=1e-11, atol=1e-12)
            np.testing.assert_allclose(
                components, basis * expected, rtol=1e-11, atol=1e-12
            )
    np.testing.assert_array_equal(fid, original_fid)


def test_unregularized_ridge_handles_identical_components():
    """Keep the singular-matrix fallback and direct negative-lambda clamping.

    One sample at t=0 gives two identical columns and an exactly singular matrix
    of ones. This avoids rounding-dependent singularity detection in LAPACK.
    The minimum-norm solution splits the total amplitude equally between them.
    """
    time_s = np.array([0.0])
    basis = np.exp(-time_s / 0.025)
    fid = basis * (1.0 + 0.3j)
    for ridge_lambda in (0.0, -1.0):
        components, amplitudes = fid_pseudo_inv.solve_for_complex_amplitudes(
            fid,
            np.array([0.0, 0.0]),
            np.array([0.025, 0.025]),
            time_s,
            signal_domain="temporal",
            amp_solver="ridge",
            amp_ridge_lambda=ridge_lambda,
        )
        np.testing.assert_allclose(amplitudes, [0.5 + 0.15j] * 2, atol=1e-12)
        np.testing.assert_allclose(components.sum(axis=1), fid, atol=1e-12)


def test_direct_amplitude_solver_rejects_unknown_name():
    """Report unsupported solver names consistently with parameter preparation."""
    with pytest.raises(ValueError, match="Unknown amp_solver 'not_a_solver'"):
        fid_pseudo_inv.solve_for_complex_amplitudes(
            np.ones(8),
            np.array([0.0]),
            np.array([0.025]),
            np.arange(8) * 0.001,
            amp_solver="not_a_solver",
        )
