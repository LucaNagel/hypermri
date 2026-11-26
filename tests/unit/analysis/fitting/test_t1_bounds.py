import numpy as np

from hypermri.analysis.fitting.models.quantitative.t1 import (
    fit_t1_inversion_recovery_helper,
    inversion_recovery_model,
)


def test_t1_fitting_with_bounds():
    # Generate some data: B=0.05, A=1.0, T1=1000ms
    ti = np.array([50, 200, 500, 1000, 2000, 5000])
    t1_true = 1000.0
    A_true = 1.0
    B_true = 0.05

    # S(t) = B + A*(1 - 2*exp(-t/T1))
    signal = B_true + A_true * (1 - 2 * np.exp(-ti / t1_true))

    t1_values = np.linspace(500, 1500, 50)

    # Test 1: Standard fit (no bounds)
    res_std = fit_t1_inversion_recovery_helper(
        signal=signal, inversion_times_ms=ti, t1_values_ms=t1_values, fit_baseline=True
    )

    assert np.allclose(res_std["t1"], t1_true, atol=100)
    assert np.allclose(res_std["A"], A_true, atol=0.1)
    assert np.allclose(res_std["B"], B_true, atol=0.1)

    # Test 2: Fit with hard bounds on B
    # Force B to be near 0
    res_bounded = fit_t1_inversion_recovery_helper(
        signal=signal,
        inversion_times_ms=ti,
        t1_values_ms=t1_values,
        fit_baseline=True,
        fit_params={"bounds_B": (-0.001, 0.001)},
    )

    assert -0.0011 <= res_bounded["B"] <= 0.0011
    # B is forced to 0, so A and T1 might shift slightly to compensate
    assert abs(res_bounded["B"]) < abs(res_std["B"])

    # Test 3: Complex signal
    signal_complex = signal + 1j * 0.05
    res_complex = fit_t1_inversion_recovery_helper(
        signal=signal_complex,
        inversion_times_ms=ti,
        t1_values_ms=t1_values,
        fit_baseline=True,
    )
    assert np.iscomplexobj(res_complex["A"])
    assert np.allclose(res_complex["t1"], t1_true, atol=100)

    # Test 4: fit_baseline=False
    res_no_baseline = fit_t1_inversion_recovery_helper(
        signal=signal - B_true,  # Signal without baseline
        inversion_times_ms=ti,
        t1_values_ms=t1_values,
        fit_baseline=False,
    )
    assert np.allclose(res_no_baseline["t1"], t1_true, atol=100)
    assert res_no_baseline["B"] == 0.0

    # Test 5: Relative bounds
    # S_max is approx 1.0. Set B to within +/- 1%
    res_rel = fit_t1_inversion_recovery_helper(
        signal=signal,
        inversion_times_ms=ti,
        t1_values_ms=t1_values,
        fit_baseline=True,
        fit_params={"rel_bounds_B": (-0.01, 0.01)},
    )
    s_max = np.max(np.abs(signal))
    assert -0.011 * s_max <= res_rel["B"] <= 0.011 * s_max


def test_inversion_recovery_model_accepts_legacy_keyword_aliases():
    ti = np.array([100.0, 300.0, 900.0, 1500.0], dtype=float)
    amp = 1000.0 * np.exp(1j * np.deg2rad(30.0))
    baseline = 0.0
    t1_ms = 1200.0

    modern = inversion_recovery_model(ti, a=amp, b=baseline, t1_ms=t1_ms)
    legacy = inversion_recovery_model(ti, A=amp, B=baseline, t1=t1_ms)
    historical_misnamed = inversion_recovery_model(ti, a=amp, b=baseline, t1_s=t1_ms)

    np.testing.assert_allclose(modern, legacy, atol=1e-12)
    np.testing.assert_allclose(modern, historical_misnamed, atol=1e-12)


if __name__ == "__main__":
    test_t1_fitting_with_bounds()
