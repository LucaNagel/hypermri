"""Invented spectra and pulse trains characterize retained simulator contracts."""

from types import SimpleNamespace

import numpy as np
import pytest

from hypermri.utils.physics import (
    hyperpolarized_Mz_flipangle,
    hyperpolarized_Mz_flipangle_T1,
)
from hypermri.visualization.csi_simulator import CSImulator
from hypermri.visualization.spectrum_simulator import SpeckSimulator


def test_csi_simulator_reads_frequency_metadata_from_the_supplied_scan():
    scan = SimpleNamespace(bandwidth_hz=1000.0, n_acquisition=4, dwell_time_s=0.001)
    scan.get = lambda name, default=None: getattr(scan, name, default)
    simulator = CSImulator(scan)
    np.testing.assert_allclose(
        simulator.fbins, [-500.0, -250.0, 0.0, 250.0], atol=1e-12, rtol=0
    )


def test_spectrum_simulation_preserves_components_and_explicit_offsets():
    simulator = SpeckSimulator([np.array([2.0, 3.0])], np.array([0.0, 250.0]))
    simulator.simulate([0.0, 0.001], offsets=[0.0])
    np.testing.assert_allclose(
        simulator.complex_data, [5.0, 2.0 + 3.0j], atol=1e-14, rtol=0
    )


@pytest.mark.parametrize("simulator_type", [CSImulator, SpeckSimulator])
def test_isochromat_phase_and_sum_without_widgets(simulator_type):
    """A 250 Hz component rotates by pi/2 in 1 ms (1e-14 roundoff tolerance)."""
    simulator = simulator_type.__new__(simulator_type)
    simulator.fbins = np.array([0.0, 250.0])
    weights = np.array([2.0, 3.0])
    original = weights.copy()
    vectors, phase = simulator.propagate(weights, 0.001)
    np.testing.assert_allclose(vectors, [2.0, 3j], atol=1e-14, rtol=0)
    np.testing.assert_allclose(phase, [0.0, np.pi / 2], atol=1e-14, rtol=0)
    result = simulator.spectrum2phase(weights, 0.001)
    np.testing.assert_allclose(
        result, [np.sqrt(13.0), np.arctan2(3.0, 2.0)], atol=1e-14, rtol=0
    )
    np.testing.assert_array_equal(weights, original)


def test_variable_pulse_train_and_relaxation_are_analytic():
    """Longitudinal products and exponential relaxation use 1e-14 tolerance."""
    angles = np.array([10.0, 20.0, 30.0, 40.0])
    before = angles.copy()
    expected = 2.0 * np.r_[1.0, np.cumprod(np.cos(np.deg2rad(angles[:-1])))]
    mz, mxy = hyperpolarized_Mz_flipangle(2.0, 4, angles)
    np.testing.assert_allclose(mz, expected, atol=1e-14, rtol=0)
    np.testing.assert_allclose(
        mxy, expected * np.sin(np.deg2rad(angles)), atol=1e-14, rtol=0
    )
    relaxed, transverse, time = hyperpolarized_Mz_flipangle_T1(
        M0_au=2.0, T1_s=3.0, TR_s=0.1, flipangle_deg=angles, n_excitations=4, plot=False
    )
    np.testing.assert_allclose(
        relaxed, expected * np.exp(-time / 3.0), atol=1e-14, rtol=0
    )
    np.testing.assert_allclose(
        transverse, relaxed * np.sin(np.deg2rad(angles)), atol=1e-14, rtol=0
    )
    np.testing.assert_array_equal(angles, before)
