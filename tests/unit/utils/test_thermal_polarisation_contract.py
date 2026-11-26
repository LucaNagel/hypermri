"""Validate the approved thermal polarization normalization correction.

Synthetic 13C controls use the Boltzmann two-state reference from Tateishi et
al., PNAS 111 (2014), Eq. 1, doi:10.1073/pnas.1315778111. The numerical
reference separates the physical definition from the historical missing divisor.
"""

import numpy as np

from hypermri.utils.physics import (
    get_thermal_polarisation_ppm,
    get_thermal_polarisation_ratio,
    get_zeeman_energy_split,
)


def test_ppm_normalizes_the_population_difference_to_all_spins():
    ratio = get_thermal_polarisation_ratio()
    current = get_thermal_polarisation_ppm()
    normalized = np.tanh(get_zeeman_energy_split() / (2 * 1.380649e-23 * 230)) * 1e6
    old = (1 - ratio) * 1e6
    np.testing.assert_allclose(old, 15.640986447573724, rtol=1e-10, atol=1e-9)
    np.testing.assert_allclose(normalized, 7.820554384358908, rtol=1e-10, atol=1e-9)
    np.testing.assert_allclose(current, normalized, rtol=1e-10, atol=1e-9)
    np.testing.assert_allclose(old / current, 1 + ratio, rtol=1e-10, atol=1e-9)


def test_thermal_population_ratio_retains_temperature_unit_equivalence():
    np.testing.assert_allclose(
        get_thermal_polarisation_ratio(temperature=230),
        get_thermal_polarisation_ratio(temperature=-43.15, temperature_unit="C"),
        rtol=1e-12,
        atol=1e-12,
    )


def test_polarization_matches_two_level_boltzmann_reference_and_preserves_input():
    temperatures = np.array([1e-6, 1.0, 77.0, 230.0, 310.0])
    before = temperatures.copy()
    for nucleus in ["1h", "13c", "31p"]:
        for field in [0.1, 3.0, 7.0]:
            splitting = get_zeeman_energy_split(nucleus=nucleus, fieldstrength_T=field)
            reference = np.tanh(splitting / (2 * 1.380649e-23 * temperatures)) * 1e6
            observed = get_thermal_polarisation_ppm(
                nucleus=nucleus, temperature=temperatures, fieldstrength_T=field
            )
            # 1e-9 ppm covers cancellation in the retained population-ratio path.
            np.testing.assert_allclose(observed, reference, rtol=1e-10, atol=1e-9)
            assert np.all((observed >= 0) & (observed <= 1e6))
    np.testing.assert_array_equal(temperatures, before)
