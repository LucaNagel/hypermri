import pytest

from hypermri.utils.spectroscopy.calibration import (
    get_pH,
    get_pH_from_OMPD,
    get_temperature_from_frequency,
    ppm_from_temp_and_concentration,
)


def test_ppm_temperature_calibration_matches_the_declared_linear_model():
    assert ppm_from_temp_and_concentration("5mM", 0) == pytest.approx(
        12.687991248319072
    )
    assert ppm_from_temp_and_concentration("5mM", 10) == pytest.approx(
        12.687991248319072 - 0.1297
    )

    with pytest.raises(ValueError, match="Unsupported concentration"):
        ppm_from_temp_and_concentration("1mM", 20)


def test_temperature_calibration_returns_zero_at_its_crossing_and_validates_units():
    temperature, uncertainty = get_temperature_from_frequency(
        12.687991248319072, calibration_type="5mM"
    )

    assert temperature == pytest.approx(0.0)
    assert uncertainty > 0

    with pytest.raises(ValueError, match="frequency in ppm"):
        get_temperature_from_frequency(
            965.0, calibration_type="5mM", frequency_is_ppm=False
        )


def test_ompd_ph_dispatch_matches_direct_calculation():
    direct = get_pH_from_OMPD(5.0, 0.02)

    assert get_pH(5.0, 0.02, compound="OMPD") == pytest.approx(direct)
    assert direct[1] > 0

    with pytest.raises(ValueError, match="Unsupported compound"):
        get_pH(5.0, 0.02, compound="pyruvate")
