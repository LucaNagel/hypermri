"""Noisy parameter-recovery tests for the DCE models on uniform and non-uniform grids.

The noise-free recovery tests in ``test_renal_dce_workflow.py`` generate their
data with the fitted model itself, so a discretisation error of the model
cancels between data and fit.  These tests generate the tissue curves with an
independent reference instead:

* the arterial input function (AIF) is known only at the acquisition times and
  is linearly interpolated between them, which is the information a fit has;
* the convolution with the model kernel is evaluated by trapezoidal quadrature
  on a dense grid with step ``DENSE_STEP_S`` (100 times finer than the shortest
  acquisition interval of 2 s) and sampled at the acquisition times.

Fixture provenance: everything is synthetic.  The AIF is a gamma-variate bolus
(peak 1 mM at 16 s) with a slowly decaying recirculation tail; the parameter
values are typical renal or tumour values chosen so that every parameter is
identifiable at the stated noise level.  No scanner data is used.

Noise: independent Gaussian noise in concentration space with a standard
deviation of 2 % of the noise-free curve peak (peak SNR 50), seed
``NOISE_SEED``, ``N_REALISATIONS`` voxels per case.  Noise in signal space and
its conversion to concentration are outside the scope of these tests.

Tolerance: the median estimate must lie within ``quadrature_allowance`` (a
fraction of the true value) plus three standard errors of the median of the
true value.  The standard error is ``sqrt(pi / 2) * s / sqrt(n)`` with the
sample standard deviation ``s`` of the ``n`` estimates, so the noise part of
the tolerance is estimated from the same realisations.  The quadrature
allowance bounds the difference between a trapezoidal sample-grid convolution
and the dense reference.  Measured noise-free with a trapezoidal fitter on
these grids, it is at most 0.5 % for the exponential kernels and 1.9 % for the
kinked 2C-IRF kernel; the allowances are 1 % and 2.5 %.  Patlak and the
empirical model have no convolution and recover the noise-free parameters to
better than 0.1 %.

These tests validate the single trapezoidal convolution rule.  With the earlier rectangle rule, standard and extended Tofts failed on
both grids and Sourbron on the uniform grid, for example with a ``Ktrans``
bias of -52 % for standard Tofts on the non-uniform grid.

The finite-Tp Sourbron case also validates that both convolutions must be
integrated before sampling tissue, with a linearly interpolated AIF and a
maximum internal step of 0.1 s. Noise-free recovery on the two grids is checked
within 0.5% for vp, Ft, Tt and Tp=5 s. This is an integration regression, not
a guarantee of individual noisy estimates or parameter identifiability.

The separate two-compartment limitation case deliberately does not assert parameter
recovery. It uses a noise-free dense reference with GFR 0.018 1/s, RPF
0.032 1/s, vascular delay/dispersion 4 s / 0.16 1/s and tubular
delay/dispersion 8 s / 0.08 1/s. Both 0.01 s and 0.0025 s quadrature steps
exhibit successful default fits with R² above 0.9999 but GFR error above
20 %, far beyond the nominal quadrature allowance. Starting at the known
synthetic parameters gives GFR error below 10 % but a larger residual. These
thresholds characterize this retained limitation; they are not clinical
acceptance criteria or a guarantee for other sampling or parameter values.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest
import xarray as xr
from scipy.integrate import cumulative_trapezoid
from scipy.signal import fftconvolve

from hypermri.analysis.dce.fitting import (
    compute_fit_quality_metrics,
    fit_empirical_log_logistic_model,
    fit_extended_tofts_model,
    fit_patlak_model,
    fit_sourbron_model,
    fit_standard_tofts_model,
    fit_two_c_irf_model,
)
from hypermri.analysis.dce.fitting_support import (
    empirical_log_logistic_model,
    two_c_irf_retention_function,
)

NOISE_SEED = 20261002
N_REALISATIONS = 24
NOISE_FRACTION_OF_PEAK = 0.02
DENSE_STEP_S = 0.01
STANDARD_ERRORS = 3.0

TIME_GRIDS_S = {
    "uniform": np.arange(0.0, 180.0, 4.0),
    # Dense sampling during the bolus, sparser during washout.
    "nonuniform": np.concatenate(
        [
            [0.0, 4.0, 8.0],
            np.arange(10.0, 42.0, 2.0),
            np.arange(44.0, 84.0, 4.0),
            np.arange(88.0, 180.0, 8.0),
        ]
    ),
}


def _aif_concentration_mM(time_s: np.ndarray) -> np.ndarray:
    """Return the synthetic plasma AIF in mM at ``time_s`` (bolus arrival 10 s)."""
    tau_s = np.clip(time_s - 10.0, 0.0, None)
    bolus = (tau_s / 6.0) ** 2 * np.exp(2.0 * (1.0 - tau_s / 6.0))
    tail = 0.25 * (1.0 - np.exp(-tau_s / 8.0)) * np.exp(-tau_s / 300.0)
    return bolus + tail


def _dense_aif(time_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate the sampled AIF linearly onto the dense reference grid."""
    dense_time_s = np.arange(0.0, time_s[-1] + DENSE_STEP_S / 2.0, DENSE_STEP_S)
    return dense_time_s, np.interp(dense_time_s, time_s, _aif_concentration_mM(time_s))


def _dense_convolution(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve on the dense grid with the trapezoidal rule."""
    convolved = fftconvolve(signal, kernel)[: len(signal)] * DENSE_STEP_S
    convolved -= 0.5 * DENSE_STEP_S * (signal[0] * kernel + signal * kernel[0])
    convolved[0] = 0.0
    return convolved


def _sample(dense: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    """Return the dense-grid values at the acquisition times."""
    return dense[np.rint(time_s / DENSE_STEP_S).astype(int)]


def _patlak_reference(time_s, ktrans_s, vp):
    dense_time_s, aif = _dense_aif(time_s)
    integral = cumulative_trapezoid(aif, dense_time_s, initial=0.0)
    return _sample(ktrans_s * integral + vp * aif, time_s)


def _tofts_reference(time_s, ktrans_s, kep_s, vp=0.0):
    dense_time_s, aif = _dense_aif(time_s)
    tissue = ktrans_s * _dense_convolution(aif, np.exp(-kep_s * dense_time_s))
    return _sample(tissue + vp * aif, time_s)


def _sourbron_reference(time_s, vp, ft_s, tt_s):
    dense_time_s, aif = _dense_aif(time_s)
    tubular = ft_s * _dense_convolution(aif, np.exp(-dense_time_s / tt_s))
    return _sample(vp * aif + tubular, time_s)


def _sourbron_finite_transit_reference(time_s, vp, ft_s, tt_s, tp_s):
    """Integrate both finite-Tp stages before sampling the tissue curve once."""
    dense_time_s, aif = _dense_aif(time_s)
    plasma = _dense_convolution(aif, np.exp(-dense_time_s / tp_s) / tp_s)
    tubular = ft_s * _dense_convolution(plasma, np.exp(-dense_time_s / tt_s))
    return _sample(vp * plasma + tubular, time_s)


def _two_c_irf_reference(time_s, **kernel_parameters):
    dense_time_s, aif = _dense_aif(time_s)
    kernel = two_c_irf_retention_function(dense_time_s, **kernel_parameters)
    return _sample(_dense_convolution(aif, kernel), time_s)


@dataclass(frozen=True)
class _RecoveryCase:
    """One model with its reference curve, fit call and checked parameters.

    ``reference`` maps acquisition times in s to the noise-free tissue
    concentration in mM.  ``fit`` receives the concentration map, the times
    and the sampled AIF.  ``expected`` maps result variables to true values;
    ``derived`` adds checked quantities computed from the result dataset.
    """

    reference: Callable[[np.ndarray], np.ndarray]
    fit: Callable[[xr.DataArray, np.ndarray, np.ndarray], xr.Dataset]
    expected: dict[str, float]
    quadrature_allowance: float = 0.01
    derived: dict[str, Callable[[xr.Dataset], xr.DataArray]] | None = None


# The whole curve is fitted; fit-window selection is tested elsewhere.
_WHOLE_CURVE = {"fit_start_index": 0}

CASES = {
    "patlak": _RecoveryCase(
        reference=lambda t: _patlak_reference(t, ktrans_s=0.003, vp=0.08),
        fit=lambda c, t, a: fit_patlak_model(
            c,
            aif_concentration=a,
            aif_time_s=t,
            fit_params=dict(_WHOLE_CURVE),
            use_multiprocessing=False,
        ),
        expected={"K_trans": 0.003, "Vp": 0.08},
    ),
    "standard_tofts": _RecoveryCase(
        reference=lambda t: _tofts_reference(t, ktrans_s=0.005, kep_s=0.02),
        fit=lambda c, t, a: fit_standard_tofts_model(
            c,
            aif_concentration=a,
            aif_time_s=t,
            fit_params=dict(_WHOLE_CURVE),
            use_multiprocessing=False,
        ),
        expected={"Ktrans_standard": 0.005, "kep_standard": 0.02},
    ),
    "extended_tofts": _RecoveryCase(
        reference=lambda t: _tofts_reference(t, ktrans_s=0.005, kep_s=0.02, vp=0.05),
        fit=lambda c, t, a: fit_extended_tofts_model(
            c,
            aif_concentration=a,
            aif_time_s=t,
            fit_params=dict(_WHOLE_CURVE),
            use_multiprocessing=False,
        ),
        expected={"Ktrans_extended": 0.005, "kep_extended": 0.02, "vp_plasma": 0.05},
    ),
    "sourbron": _RecoveryCase(
        reference=lambda t: _sourbron_reference(t, vp=0.25, ft_s=0.03, tt_s=60.0),
        fit=lambda c, t, a: fit_sourbron_model(
            c,
            aif_concentration=a,
            aif_time_s=t,
            fit_params=dict(_WHOLE_CURVE),
            use_multiprocessing=False,
        ),
        expected={"vp": 0.25, "Ft": 0.03, "Tt": 60.0},
    ),
    "sourbron_finite_transit": _RecoveryCase(
        reference=lambda t: _sourbron_finite_transit_reference(
            t, vp=0.25, ft_s=0.03, tt_s=60.0, tp_s=5.0
        ),
        fit=lambda c, t, a: fit_sourbron_model(
            c,
            aif_concentration=a,
            aif_time_s=t,
            fit_params={**_WHOLE_CURVE, "estimate_tp": True},
            use_multiprocessing=False,
        ),
        expected={"vp": 0.25, "Ft": 0.03, "Tt": 60.0, "Tp": 5.0},
    ),
    "two_c_irf": _RecoveryCase(
        reference=lambda t: _two_c_irf_reference(
            t,
            gfr_s=0.010,
            rpf_s=0.035,
            vascular_delay_s=3.0,
            vascular_dispersion_rate_s=0.2,
            tubular_delay_s=20.0,
            tubular_dispersion_rate_s=0.05,
        ),
        fit=lambda c, t, a: fit_two_c_irf_model(
            c,
            aif_concentration=a,
            aif_time_s=t,
            fit_params=dict(_WHOLE_CURVE),
            use_multiprocessing=False,
        ),
        expected={"GFR": 0.010, "RPF": 0.035},
        quadrature_allowance=0.025,
    ),
    # Bolus arrival and rise time trade off against each other at this noise
    # level (spread about 40 %); their sum, the half-rise time, is well defined.
    "empirical_log_logistic": _RecoveryCase(
        reference=lambda t: empirical_log_logistic_model(
            t, 12.0, 10.0, 4.0, 0.01, 0.5, 0.0
        ),
        fit=lambda c, t, a: fit_empirical_log_logistic_model(
            c,
            fit_params={"time_points_s": t, **_WHOLE_CURVE},
            use_multiprocessing=False,
        ),
        expected={
            "half_rise_time_s": 22.0,
            "washout_rate_s": 0.01,
            "peak_amplitude": 0.5,
        },
    ),
}


def _noisy_concentration_map(curve: np.ndarray, time_s: np.ndarray) -> xr.DataArray:
    """Return ``N_REALISATIONS`` noisy copies of ``curve`` as a 4D map in mM."""
    rng = np.random.default_rng(NOISE_SEED)
    noise_sd = NOISE_FRACTION_OF_PEAK * np.max(np.abs(curve))
    curves = curve[None, :] + noise_sd * rng.standard_normal(
        (N_REALISATIONS, curve.size)
    )
    return xr.DataArray(
        curves.reshape(N_REALISATIONS, 1, 1, curve.size),
        dims=("read_idx", "phase_idx", "slice_idx", "repetition_idx"),
        coords={
            "read_idx": np.arange(N_REALISATIONS),
            "phase_idx": [0],
            "slice_idx": [0],
            "repetition_idx": np.arange(curve.size),
            "time_s": ("repetition_idx", time_s),
        },
    )


@pytest.mark.parametrize("grid", list(TIME_GRIDS_S))
@pytest.mark.parametrize("model", list(CASES))
def test_dce_model_recovers_parameters_from_noisy_reference(model, grid):
    case = CASES[model]
    time_s = TIME_GRIDS_S[grid]
    concentration = _noisy_concentration_map(case.reference(time_s), time_s)

    result = case.fit(concentration, time_s, _aif_concentration_mM(time_s))

    failures = []
    for name, true_value in case.expected.items():
        derived = case.derived or {}
        values = derived[name](result) if name in derived else result[name]
        estimates = np.asarray(values, dtype=np.float64).ravel()
        assert estimates.size == N_REALISATIONS
        assert np.all(np.isfinite(estimates)), f"{name}: non-finite estimates"
        median = float(np.median(estimates))
        standard_error = np.sqrt(np.pi / 2.0) * np.std(estimates, ddof=1)
        standard_error /= np.sqrt(estimates.size)
        tolerance = (
            case.quadrature_allowance * abs(true_value)
            + STANDARD_ERRORS * standard_error
        )
        if abs(median - true_value) > tolerance:
            failures.append(
                f"{name}: median {median:.5g}, true {true_value:.5g}, "
                f"bias {100.0 * (median / true_value - 1.0):+.1f} %, "
                f"tolerance {100.0 * tolerance / abs(true_value):.1f} %"
            )
    assert not failures, "; ".join(failures)


@pytest.mark.parametrize("grid", list(TIME_GRIDS_S))
@pytest.mark.parametrize("dense_step_s", [0.01, 0.0025])
def test_two_c_irf_success_does_not_establish_flow_recovery(
    grid, dense_step_s, monkeypatch
):
    """Retain the two-compartment identifiability limit against dense quadrature."""
    monkeypatch.setitem(globals(), "DENSE_STEP_S", dense_step_s)
    time_s = TIME_GRIDS_S[grid]
    plasma = _aif_concentration_mM(time_s)
    tissue = _two_c_irf_reference(
        time_s,
        gfr_s=0.018,
        rpf_s=0.032,
        vascular_delay_s=4.0,
        vascular_dispersion_rate_s=0.16,
        tubular_delay_s=8.0,
        tubular_dispersion_rate_s=0.08,
    )
    concentration = xr.DataArray(
        tissue,
        dims="repetition_idx",
        coords={
            "repetition_idx": np.arange(tissue.size),
            "time_s": ("repetition_idx", time_s),
        },
        attrs={"units": "mM"},
    )
    before = concentration.copy(deep=True)
    plasma_before = plasma.copy()

    def fit(options):
        return fit_two_c_irf_model(
            concentration,
            aif_concentration=plasma,
            aif_time_s=time_s,
            fit_params={"fit_start_index": 0, **options},
            use_multiprocessing=False,
        )

    default = fit({})
    known_start = fit({"x0": [0.018, 0.014, 4.0, 0.16, 8.0, 0.08]})
    for result in (default, known_start):
        assert result["fit_success"].item()
        assert result["n_points_fit"].item() == tissue.size
        assert result["GFR"].dtype == np.float64
        xr.testing.assert_identical(result.coords["time_s"], concentration.time_s)
        np.testing.assert_array_equal(result["fit_window_mask"], True)

    quality = compute_fit_quality_metrics(
        default,
        concentration,
        fit_type="2c_irf",
        aif_concentration=plasma,
        aif_time_s=time_s,
        fit_params={"fit_start_index": 0},
    )
    assert quality["r_squared"].item() > 0.9999
    assert abs(default["GFR"].item() / 0.018 - 1) > 0.20
    assert abs(known_start["GFR"].item() / 0.018 - 1) < 0.10
    assert default["rss"].item() < known_start["rss"].item()
    xr.testing.assert_identical(concentration, before)
    np.testing.assert_array_equal(plasma, plasma_before)


@pytest.mark.parametrize("grid", list(TIME_GRIDS_S))
@pytest.mark.parametrize("dense_step_s", [0.01, 0.0025])
def test_sourbron_finite_transit_recovers_independent_noise_free_reference(
    grid, dense_step_s, monkeypatch
):
    """The finite-Tp integration requires <0.5% noise-free parameter error, not a clinical noise limit.

    Reference quadrature refinement changes the Tp=5 s curve by <6e-7 of its
    peak. The chosen tolerance allows finite integration/optimizer error while
    excluding the former 32--64% vp/Tp biases on these two synthetic grids.
    """
    monkeypatch.setitem(globals(), "DENSE_STEP_S", dense_step_s)
    time_s = TIME_GRIDS_S[grid]
    plasma = _aif_concentration_mM(time_s)
    tissue = _sourbron_finite_transit_reference(time_s, 0.25, 0.03, 60.0, 5.0)
    concentration = xr.DataArray(
        tissue,
        dims="repetition_idx",
        coords={
            "repetition_idx": np.arange(tissue.size),
            "time_s": ("repetition_idx", time_s),
        },
        attrs={"units": "mM"},
    )
    before = concentration.copy(deep=True)
    plasma_before = plasma.copy()
    times_before = time_s.copy()
    options = {"fit_start_index": 0, "estimate_tp": True}
    result = fit_sourbron_model(
        concentration,
        aif_concentration=plasma,
        aif_time_s=time_s,
        fit_params=options,
        use_multiprocessing=False,
    )
    for name, truth in {"vp": 0.25, "Ft": 0.03, "Tt": 60.0, "Tp": 5.0}.items():
        np.testing.assert_allclose(result[name], truth, rtol=0.005, atol=0)
        assert result[name].dtype == np.float64
    assert result.fit_success.item()
    assert result.fit_success.dtype == bool
    assert result.n_points_fit.item() == np.count_nonzero(plasma > 0)
    assert result.n_points_fit.dtype == np.int32
    assert result.attrs["sourbron_convolution_method"] == "oversampled_trapezoidal"
    assert result.attrs["sourbron_integration_max_step_s"] == 0.1
    assert result.attrs["sourbron_aif_interpolation"] == "linear"
    prediction = result.hp.reconstruct()
    assert prediction.dims == concentration.dims
    assert prediction.dtype == np.float64
    np.testing.assert_allclose(prediction, tissue, rtol=0, atol=np.max(tissue) * 1e-4)
    quality = compute_fit_quality_metrics(
        result,
        concentration,
        fit_type="sourbron",
        aif_concentration=plasma,
        aif_time_s=time_s,
        fit_params=options,
    )
    assert quality.r_squared.item() > 0.999999
    np.testing.assert_allclose(result.Fp, result.vp / result.Tp, rtol=1e-12)
    np.testing.assert_allclose(
        result.extraction_fraction, result.Ft / result.Fp, rtol=1e-12
    )
    np.testing.assert_array_equal(result.time_points_s, time_s)
    np.testing.assert_array_equal(result.repetition_idx, concentration.repetition_idx)
    xr.testing.assert_identical(quality.time_s, concentration.time_s)
    xr.testing.assert_identical(concentration, before)
    np.testing.assert_array_equal(plasma, plasma_before)
    np.testing.assert_array_equal(time_s, times_before)
